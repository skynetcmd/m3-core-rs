//! Axum HTTP server — shared between foreground mode and service mode.
//!
//! `run` takes a `ShutdownSignal` future that resolves when the caller wants
//! the server to drain (Ctrl-C in foreground; SCM SERVICE_CONTROL_STOP in
//! service mode).

#![cfg(feature = "embedded")]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, FromRequest, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use m3_dispatcher::{BreakerCfg, BreakerStatus, Dispatcher, DispatcherConfig};
use m3_embed_llamacpp::{EmbeddedBackend, Tokenizer};

use crate::config::{self, ResolvedConfig};
use crate::security;

/// Which GPU backend this binary was COMPILED with.
///
/// Reported at startup because a backend mismatch is otherwise invisible: a
/// wheel labelled `-cuda` whose server fell back to CPU presents only as
/// "embedding got slow". The model load logs the backend actually used (a WARN
/// when a GPU build lands on the CPU). Owner: `m3_embed_llamacpp`.
const fn compiled_backend() -> &'static str {
    m3_embed_llamacpp::COMPILED_BACKEND
}

#[derive(Debug, Deserialize)]
struct EmbedRequest {
    input: serde_json::Value,
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct EmbedData {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbedResponse {
    object: &'static str,
    data: Vec<EmbedData>,
    model: String,
}

/// Exit status when the listen address is already taken (EX_UNAVAILABLE).
/// Distinct from the generic 1 so a supervisor can stop retrying: a restart
/// cannot free a port another process holds. The systemd unit names it in
/// `RestartPreventExitStatus=`.
pub const EXIT_PORT_IN_USE: i32 = 69;

/// Which of the two listeners a bind failure concerns: each is set by a
/// different command, so the fix differs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Listener {
    /// `host:port`, loopback plaintext.
    Local,
    /// `remote_listen`, TLS for other machines.
    Remote,
}

/// The listen address is held by another process (often another account's
/// m3 on a shared machine).
#[derive(Debug)]
pub struct PortInUse {
    pub addr: SocketAddr,
    pub listener: Listener,
}

impl std::fmt::Display for PortInUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.listener {
            Listener::Local => write!(
                f,
                "cannot listen on {}: another program is already using this port \
                 (on a shared computer, often another account's m3). Choose a free \
                 port with `m3 config set embed.server.port <port>`",
                self.addr
            ),
            Listener::Remote => write!(
                f,
                "cannot listen on {} (remote_listen, the TLS listener for other machines): another \
                 program is already using this address and port. Choose a free one with \
                 `m3 embedder secure --listen <ip>:<port>`, or remove it with `m3 embedder secure --off`",
                self.addr
            ),
        }
    }
}

impl std::error::Error for PortInUse {}

/// Largest request body accepted: axum's own default, made explicit so its
/// 413 can be told apart from the per-text `input_too_long` one.
pub const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

struct AppState {
    dispatcher: Arc<Dispatcher<EmbeddedBackend>>,
    model_label: String,
    /// The model path, for failure logs.
    gguf: String,
    /// The same model, for `/tokenize/*`: counts and windows match what
    /// `/embedding` measures against n_ctx, and never queue behind a batch.
    tokenizer: Tokenizer,
}

/// Build the dispatcher (eager GGUF load), bind the listener, and serve until
/// `shutdown` resolves. On graceful shutdown returns Ok(()).
pub async fn run<F>(cfg: ResolvedConfig, shutdown: F) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    run_with_ready(cfg, shutdown, || {}).await
}

/// `run`, calling `on_ready` once the model is loaded and the port is bound —
/// the earliest moment the server can actually answer. A supervisor that
/// reports "running" must wait for this, not for the start request.
pub async fn run_with_ready<F, R>(cfg: ResolvedConfig, shutdown: F, on_ready: R) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
    R: FnOnce(),
{
    // n_ctx and streams are the two knobs that decide this process's footprint,
    // so both belong in the startup line. Each stream materialises its own
    // compute graph on first use, sized for a worst-case n_ctx batch, and holds
    // it for the process lifetime -- measured 2026-09-16 at ~3.85 GiB per stream
    // at n_ctx=8192 on CUDA and ~4.0 GiB on a CPU-only build, matching
    // n_ctx^2 x heads x 4 bytes. The cost is therefore LINEAR in streams and
    // QUADRATIC in n_ctx: halving n_ctx cuts it ~4x.
    //
    // It is logged rather than merely documented because the growth is lazy --
    // it appears over the first `streams` requests, which reads as a leak to
    // anyone watching RSS climb after startup.
    // A complete startup fingerprint. Every field here was needed to diagnose a
    // real incident and was absent from the log at the time:
    //   * gguf   — the server ran for months against a model under a THIRD-PARTY
    //              app's directory; nothing said so until the path was printed.
    //   * config — which file those values came from, since env > file > default
    //              and the wrong file looks identical to no file.
    //   * backend— a `-cuda` wheel silently falling back to CPU presents only as
    //              "embedding got slow"; the compiled feature is the only proof.
    //   * streams/n_ctx — the two knobs that decide the process footprint.
    //   * n_batch/n_ubatch — these are RAISED to n_ctx at the FFI, so a value set
    //              here is not necessarily the value in force.
    let config_path = config::default_config_path();
    log::info!(
        "m3-embed-server starting: host={} port={} backend={} gguf={} (from {}) config={} ({})",
        cfg.host,
        cfg.port,
        compiled_backend(),
        cfg.gguf,
        cfg.gguf_source,
        config_path.display(),
        if config_path.exists() { "present" } else { "absent: environment and defaults only" },
    );
    log::info!(
        "embed params: streams={} n_ctx={} seq_max={} n_batch={} n_ubatch={} \
         coalesce_ms={} max_batch_tokens={}",
        cfg.streams,
        cfg.n_ctx,
        cfg.seq_max,
        cfg.n_batch,
        cfg.n_ubatch,
        cfg.coalesce_ms,
        cfg.max_batch_tokens,
    );
    // Deliberately NOT a computed byte figure: the size depends on the model's
    // head count and layer shape, which this layer does not know. Printing
    // `n_ctx^2 * 16 * 4` would be exact for bge-m3 and confidently wrong for
    // anything else. State the scaling and the measured anchor instead.
    log::info!(
        "footprint: {} compute graph(s), allocated lazily on the first \
         {} request(s) and held for the process lifetime. Scales linearly with \
         streams and QUADRATICALLY with n_ctx. Anchor: bge-m3 at n_ctx=8192 \
         measured ~3.85 GiB per stream. Reduce with M3_EMBED_STREAMS or \
         M3_EMBED_CTX.",
        cfg.streams,
        cfg.streams,
    );

    // Bind BEFORE loading the model. Loading first meant a taken port cost a
    // full model load (417 MB for bge-m3) on every failed start, and the
    // supervisor's restart loop repeated it every few seconds. A connection
    // that arrives during the load waits in the listen backlog until serving.
    // Parsed as an IP, then paired with the port: "host:port" text breaks on
    // IPv6 ("::1:8082") and used to blame the port as well.
    let ip: std::net::IpAddr = cfg.host.trim().trim_start_matches('[').trim_end_matches(']').parse().map_err(|e| {
        anyhow::anyhow!(
            "invalid listen host {:?} (from {}): {e}. It must be a loopback IP address such as \
             127.0.0.1 or ::1; a hostname is not accepted",
            cfg.host,
            cfg.host_source
        )
    })?;
    let addr = SocketAddr::new(ip, cfg.port);
    // Decide transport security before binding, so a refused configuration
    // never holds the port or loads the model.
    let sec = security::plan(addr, &cfg.security)
        .map_err(|e| anyhow::Error::new(security::ConfigRefused(format!("{e:#}"))))?;
    log::info!("listeners: {}", sec.describe(addr));
    let listener = bind(addr, Listener::Local).await?;
    let remote = match sec.remote {
        Some(r) => Some((bind(r.addr, Listener::Remote).await?, r)),
        None => None,
    };

    let backend = EmbeddedBackend::with_streams_ctx_seqmax_batch(
        cfg.gguf.clone(),
        cfg.streams,
        cfg.n_ctx,
        cfg.seq_max,
        cfg.n_batch,
        cfg.n_ubatch,
    );
    let load_ctx = || {
        format!(
            "gguf={} (from {}), backend={}, n_ctx={}, streams={}",
            cfg.gguf,
            cfg.gguf_source,
            compiled_backend(),
            cfg.n_ctx,
            cfg.streams
        )
    };
    let dim = backend
        .embedding_dim()
        .map_err(|e| anyhow::anyhow!("eager GGUF load failed ({}): {e}", load_ctx()))?;
    log::info!("model loaded, embedding dim = {dim}");
    let tokenizer = backend
        .tokenizer()
        .map_err(|e| anyhow::anyhow!("tokenizer unavailable after GGUF load ({}): {e}", load_ctx()))?;

    let dcfg = DispatcherConfig {
        streams: cfg.streams,
        coalesce_window_ms: cfg.coalesce_ms,
        max_batch_tokens: cfg.max_batch_tokens,
        length_buckets: DispatcherConfig::default().length_buckets,
        circuit_breaker: BreakerCfg::default(),
    };
    let dispatcher = Arc::new(Dispatcher::new(dcfg, backend));

    let model_label = std::path::Path::new(&cfg.gguf)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let state = Arc::new(AppState {
        dispatcher,
        model_label,
        gguf: cfg.gguf.clone(),
        tokenizer,
    });

    let app = router(state);

    log::info!("listening on http://{addr}");
    let Some((remote_listener, remote)) = remote else {
        on_ready();
        axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
        log::info!("shutdown complete");
        return Ok(());
    };
    log::info!("listening on https://{} (bearer token required)", remote.addr);
    on_ready();
    // One stop signal, both listeners.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        shutdown.await;
        let _ = stop_tx.send(true);
    });
    let stopped = |mut rx: tokio::sync::watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|s| *s).await;
    };
    let remote_app = app
        .clone()
        .layer(axum::middleware::from_fn_with_state(Arc::new(remote.token_digest), security::require_bearer));
    let local = async {
        axum::serve(listener, app)
            .with_graceful_shutdown(stopped(stop_rx.clone()))
            .await
            .map_err(anyhow::Error::from)
    };
    let tls = security::serve_tls(remote_listener, remote_app, remote.tls, stopped(stop_rx.clone()));
    tokio::try_join!(local, tls)?;

    log::info!("shutdown complete");
    Ok(())
}

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/embedding", post(embed_handler))
        .route("/v1/embeddings", post(embed_handler))
        .route("/tokenize/count", post(tokenize_count_handler))
        .route("/tokenize/windows", post(tokenize_windows_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

async fn bind(addr: SocketAddr, listener: Listener) -> anyhow::Result<tokio::net::TcpListener> {
    let which = match listener {
        Listener::Local => "the local listener",
        Listener::Remote => "remote_listen",
    };
    match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => Ok(l),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            Err(anyhow::Error::new(PortInUse { addr, listener }))
        }
        Err(e) => Err(anyhow::anyhow!("cannot listen on {addr} ({which}): {e}")),
    }
}

/// `Json`, with its body-size rejection reworded. The plain "length limit
/// exceeded" 413 read like the client's own input-too-long 413, which tells
/// the caller to split a text rather than send fewer texts per request.
struct ApiJson<T>(T);

#[axum::async_trait]
impl<T, S> FromRequest<S> for ApiJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(ApiJson(v)),
            Err(rej) => Err(json_rejection(rej)),
        }
    }
}

fn json_rejection(rej: JsonRejection) -> (StatusCode, String) {
    let status = rej.status();
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        // ⚠ Must not contain "input too long" or "tokens": the Python client
        // reads those as a per-text context overflow and subdivides the text.
        return (
            status,
            format!(
                "request body exceeds this server's limit of {MAX_BODY_BYTES} bytes; send fewer texts \
                 per request (code=body_too_large limit_bytes={MAX_BODY_BYTES})"
            ),
        );
    }
    (status, rej.body_text())
}

/// One line for the breaker's state, for logs.
fn describe_breaker(b: &BreakerStatus) -> String {
    match b.retry_after {
        Some(left) if b.open => format!(
            "open after {} consecutive failures, refusing calls for another {}s",
            b.consecutive_failures,
            ceil_secs(left)
        ),
        _ if b.consecutive_failures > 0 => format!("closed, {} consecutive failure(s)", b.consecutive_failures),
        _ => "closed".into(),
    }
}

fn ceil_secs(d: std::time::Duration) -> u64 {
    d.as_secs() + u64::from(d.subsec_nanos() > 0)
}

/// The reply for a failed backend call: 503 with Retry-After while the
/// breaker refuses calls (a retry after the cooldown can succeed), else 500.
fn backend_failure_reply(msg: String, b: &BreakerStatus) -> Response {
    match b.retry_after {
        Some(left) if b.open => (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, ceil_secs(left).to_string())],
            msg,
        )
            .into_response(),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
    }
}

/// Log a failed backend call with what identifies it (the client's copy is
/// otherwise the only record) and build its reply. No input text is logged.
async fn backend_failure(state: &AppState, what: &str, inputs: usize, err: &m3_error::M3Error) -> Response {
    let msg = format!("{what} failed: {err}");
    let b = state.dispatcher.breaker_status().await;
    log::error!(
        "{msg} ({inputs} input(s); backend={} gguf={}; circuit breaker {})",
        compiled_backend(),
        state.gguf,
        describe_breaker(&b)
    );
    backend_failure_reply(msg, &b)
}

/// The JSON type name of `v`, for error messages that must not echo `v` itself.
fn json_type(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// The `input` of a request as texts: a string, or an array of strings.
/// Shared by `/embedding` and `/tokenize/count` so both reject the same shapes.
fn texts_from(input: serde_json::Value) -> Result<Vec<String>, (StatusCode, String)> {
    let texts: Vec<String> = match input {
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(arr) => {
            // Each element must be a string. A number (or an OpenAI-style token
            // array) used to become "" and be embedded as the empty string.
            let mut out = Vec::with_capacity(arr.len());
            for (i, v) in arr.into_iter().enumerate() {
                match v {
                    serde_json::Value::String(s) => out.push(s),
                    other => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            format!("input[{i}] must be a string; got {}", json_type(&other)),
                        ));
                    }
                }
            }
            out
        }
        // Name the type only: the rejected payload can be large, and echoing it
        // back serves no one.
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("input must be a string or an array of strings; got {}", json_type(&other)),
            ));
        }
    };
    Ok(texts)
}

async fn embed_handler(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<EmbedRequest>,
) -> Result<Json<EmbedResponse>, Response> {
    let texts = texts_from(req.input).map_err(IntoResponse::into_response)?;
    if texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "input is empty").into_response());
    }
    let inputs = texts.len();

    // ⚠ ONLY a length problem becomes 413. OOM, a poisoned mutex and a dead
    // worker pool must keep propagating as 500: mislabelling an infrastructure
    // failure as a client length error sends an operator off to shrink their
    // inputs while the real fault goes unreported — the same misdiagnosis
    // issue #139 caused, in the opposite direction. The match is on the TYPED
    // variant, never on message text.
    let rows = match state.dispatcher.embed_batch(texts).await {
        Ok(rows) => rows,
        Err(m3_error::M3Error::InputTooLong { tokens, n_ctx, .. }) => {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                // Message preserved verbatim so an older client parsing the
                // text keeps working; the counts are also machine-readable.
                format!(
                    "input too long: {tokens} tokens > n_ctx {n_ctx} \
                     (code=input_too_long observed_tokens={tokens} max_tokens={n_ctx})"
                ),
            )
                .into_response())
        }
        Err(other) => return Err(backend_failure(&state, "embed", inputs, &other).await),
    };

    let data: Vec<EmbedData> = rows
        .into_iter()
        .enumerate()
        .map(|(i, e)| EmbedData {
            object: "embedding",
            index: i,
            embedding: e,
        })
        .collect();

    Ok(Json(EmbedResponse {
        object: "list",
        data,
        model: state.model_label.clone(),
    }))
}

#[derive(Deserialize)]
struct CountRequest {
    input: serde_json::Value,
}

#[derive(Serialize)]
struct CountResponse {
    counts: Vec<usize>,
}

/// Exact token counts, BOS/EOS included: the numbers `/embedding` checks
/// against n_ctx. Lets a client size its chunks to the model instead of to an
/// estimate that over-counts prose ~3x.
async fn tokenize_count_handler(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<CountRequest>,
) -> Result<Json<CountResponse>, (StatusCode, String)> {
    let texts = texts_from(req.input)?;
    let inputs = texts.len();
    let tok = state.tokenizer.clone();
    let counts = tokio::task::spawn_blocking(move || tok.count_tokens(&texts)).await;
    let counts = tokenize_outcome(&state, "/tokenize/count", inputs, counts)?;
    Ok(Json(CountResponse { counts }))
}

#[derive(Deserialize)]
struct WindowsRequest {
    input: String,
    budget: usize,
    #[serde(default)]
    overlap: usize,
}

#[derive(Serialize)]
struct WindowsResponse {
    /// `[start, end]` pairs in CHARACTER offsets (Unicode scalar values).
    windows: Vec<(usize, usize)>,
}

/// Token-exact windows over one text: each at most `budget` tokens (BOS/EOS
/// included), overlapping by about `overlap`. Offsets are characters, which is
/// what a Python `str` index is.
async fn tokenize_windows_handler(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<WindowsRequest>,
) -> Result<Json<WindowsResponse>, (StatusCode, String)> {
    // The budget is validated by the tokenizer itself (inside the BOS/EOS
    // frame, or above n_ctx, is a Config error), answered 400 below.
    let tok = state.tokenizer.clone();
    let windows = tokio::task::spawn_blocking(move || tok.token_windows(&req.input, req.budget, req.overlap)).await;
    let windows = tokenize_outcome(&state, "/tokenize/windows", 1, windows)?;
    Ok(Json(WindowsResponse { windows }))
}

/// A tokenize result, with each failure logged (the 500 used to reach only
/// the client) and named: a panicked task is not a tokenizer error.
fn tokenize_outcome<T>(
    state: &AppState,
    route: &str,
    inputs: usize,
    r: Result<m3_error::Result<T>, tokio::task::JoinError>,
) -> Result<T, (StatusCode, String)> {
    let msg = match r {
        Ok(Ok(v)) => return Ok(v),
        // A caller error (e.g. a budget the model cannot honour), not a fault.
        Ok(Err(e @ m3_error::M3Error::Config(_))) => return Err((StatusCode::BAD_REQUEST, format!("{route}: {e}"))),
        Ok(Err(e)) => format!("{route} failed: {e}"),
        Err(e) if e.is_panic() => format!("{route} failed: the tokenizer task panicked ({e})"),
        Err(e) => format!("{route} failed: the tokenizer task was cancelled ({e})"),
    };
    log::error!("{msg} ({inputs} input(s); backend={} gguf={})", compiled_backend(), state.gguf);
    Err((StatusCode::INTERNAL_SERVER_ERROR, msg))
}

/// Liveness probe. Returns JSON `{"status":"ok","model":"..."}`.
///
/// ⚠ MUST BE JSON. This previously returned the bare string `"OK\n"`, which
/// silently disabled m3's own recovery path: `bin/memory/embed.py`'s
/// `_try_recover_shared_embedder()` does `json.loads()` on this body and
/// requires `status == "ok"` before it will steer traffic back here. Against a
/// bare string it raised `JSONDecodeError`, a bare `except` swallowed it, and
/// the function became structurally dead code -- the server would come back up
/// and no client would ever notice. Nothing failed loudly; the recovery simply
/// never happened (§3).
///
/// The shape matches upstream `llama-server`, which answers `{"status":"ok"}`
/// when serving and `503 {"status":"loading model"}` while loading -- i.e. the
/// contract our client was already written against.
///
/// Backward compatible: the literal substring `ok` still appears in the body,
/// so a caller doing a naive string match keeps working. `model` is included so
/// a probe can tell WHICH embedder answered without a second request -- an
/// identity check that matters when several servers can bind this port.
/// `pid` lets `start` confirm the answer came from the process the service
/// manager started, not from another account's server on the same port.
///
/// While the dispatcher's circuit breaker refuses calls this answers 503
/// `{"status":"degraded","reason":...}` (same keys plus `reason` and
/// `retry_after_s`): "ok" there sent clients back to a server that rejects
/// every embed. Clients accepting only "ok"/"loading" read it as unhealthy,
/// which is the intent.
async fn health_handler(State(state): State<Arc<AppState>>) -> Response {
    health_reply(&state.dispatcher.breaker_status().await, &state.model_label)
}

/// ⚠ `/health` needs no token, on the remote listener too: the reason gives
/// counts and the cooldown only, never the backend's error text (it can carry
/// a path). The error text is in the log and in authenticated embed replies.
fn health_reply(b: &BreakerStatus, model: &str) -> Response {
    let pid = std::process::id();
    match b.retry_after {
        Some(left) if b.open => {
            let secs = ceil_secs(left);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::RETRY_AFTER, secs.to_string())],
                Json(serde_json::json!({
                    "status": "degraded",
                    "reason": format!(
                        "embed backend circuit breaker open after {} consecutive failures; refusing \
                         embeds for another {secs}s (the cause is in the server log)",
                        b.consecutive_failures
                    ),
                    "retry_after_s": secs,
                    "model": model,
                    "pid": pid,
                })),
            )
                .into_response()
        }
        _ => Json(serde_json::json!({ "status": "ok", "model": model, "pid": pid })).into_response(),
    }
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let s = state.dispatcher.stats();
    // p50/p99 serialize as JSON `null` until at least one batch has completed.
    // Deliberately null and not 0.0: a constant zero is indistinguishable from
    // a genuinely fast server, so it reads as observability while providing
    // none. `null` says "no samples yet", which is the truth and is what a
    // scraper can correctly ignore.
    Json(serde_json::json!({
        "in_flight": s.in_flight,
        "queue_depth": s.queue_depth,
        "p50_ms": s.p50_ms,
        "p99_ms": s.p99_ms,
        "model": state.model_label,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn body_text(r: Response) -> String {
        let b = axum::body::to_bytes(r.into_body(), usize::MAX).await.unwrap();
        String::from_utf8_lossy(&b).into_owned()
    }

    fn open_breaker() -> BreakerStatus {
        BreakerStatus {
            open: true,
            consecutive_failures: 5,
            retry_after: Some(Duration::from_millis(7200)),
            last_error: Some("llama encode failed: -3 /models/secret-path.gguf".into()),
        }
    }

    #[tokio::test]
    async fn an_oversized_body_is_not_mistaken_for_an_overlong_input() {
        let req = Request::builder()
            .method("POST")
            .uri("/embedding")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(vec![b' '; MAX_BODY_BYTES + 1]))
            .unwrap();
        let Err((status, msg)) = ApiJson::<serde_json::Value>::from_request(req, &()).await else {
            panic!("an oversized body was accepted");
        };
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(msg.contains("code=body_too_large"), "{msg}");
        // The Python client subdivides a text on these; this needs fewer texts.
        assert!(!msg.contains("input too long") && !msg.contains("tokens"), "{msg}");
    }

    #[tokio::test]
    async fn an_open_breaker_is_503_with_retry_after_and_a_closed_one_500() {
        let r = backend_failure_reply("embed failed: x".into(), &open_breaker());
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(r.headers()[header::RETRY_AFTER], "8");
        let closed = BreakerStatus { consecutive_failures: 1, ..Default::default() };
        let r = backend_failure_reply("embed failed: x".into(), &closed);
        assert_eq!(r.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body_text(r).await, "embed failed: x");
        assert_eq!(describe_breaker(&closed), "closed, 1 consecutive failure(s)");
    }

    #[tokio::test]
    async fn health_reports_degraded_while_the_breaker_is_open() {
        let r = health_reply(&open_breaker(), "bge-m3.gguf");
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v: serde_json::Value = serde_json::from_str(&body_text(r).await).unwrap();
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["model"], "bge-m3.gguf");
        assert!(v["pid"].is_u64() && v["retry_after_s"] == 8, "{v}");
        let reason = v["reason"].as_str().unwrap();
        assert!(reason.contains("5 consecutive"), "{reason}");
        // Unauthenticated endpoint: the backend's error text stays in the log.
        assert!(!reason.contains("secret-path"), "{reason}");

        let r = health_reply(&BreakerStatus::default(), "bge-m3.gguf");
        assert_eq!(r.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body_text(r).await).unwrap();
        assert_eq!(v["status"], "ok");
        assert!(v.get("reason").is_none());
    }

    #[tokio::test]
    async fn a_taken_port_names_the_listener_and_its_own_fix() {
        let held = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = held.local_addr().unwrap();
        for (which, fix) in [(Listener::Local, "embed.server.port"), (Listener::Remote, "secure --listen")] {
            let e = bind(addr, which).await.unwrap_err();
            let p = e.downcast_ref::<PortInUse>().expect("PortInUse");
            assert_eq!(p.listener, which);
            assert!(e.to_string().contains(fix), "{e}");
        }
    }
}
