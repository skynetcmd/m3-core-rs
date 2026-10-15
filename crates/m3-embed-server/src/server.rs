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
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use m3_dispatcher::{BreakerCfg, Dispatcher, DispatcherConfig};
use m3_embed_llamacpp::{EmbeddedBackend, Tokenizer};

use crate::config::{self, ResolvedConfig};

/// Which GPU backend this binary was COMPILED with.
///
/// Reported at startup because a backend mismatch is otherwise invisible: a
/// wheel labelled `-cuda` whose server fell back to CPU presents only as
/// "embedding got slow". The feature flags are mutually exclusive and enforced
/// at compile time in `m3-embed-llamacpp`, so at most one arm is ever live.
const fn compiled_backend() -> &'static str {
    if cfg!(feature = "embedded-cuda") {
        "cuda"
    } else if cfg!(feature = "embedded-vulkan") {
        "vulkan"
    } else if cfg!(feature = "embedded-metal") {
        "metal"
    } else if cfg!(feature = "embedded") {
        "cpu"
    } else {
        "none (no embed backend compiled in)"
    }
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

/// The listen address is held by another process (often another account's
/// m3 on a shared machine).
#[derive(Debug)]
pub struct PortInUse {
    pub addr: SocketAddr,
}

impl std::fmt::Display for PortInUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cannot listen on {}: another program is already using this port \
             (on a shared computer, often another account's m3). Choose a free \
             port with `m3 config set embed.server.port <port>`",
            self.addr
        )
    }
}

impl std::error::Error for PortInUse {}

struct AppState {
    dispatcher: Arc<Dispatcher<EmbeddedBackend>>,
    model_label: String,
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
    log::info!(
        "m3-embed-server starting: host={} port={} backend={} gguf={} config={}",
        cfg.host,
        cfg.port,
        compiled_backend(),
        cfg.gguf,
        config::default_config_path().display(),
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
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse().map_err(|e| {
        anyhow::anyhow!("invalid listen address {}:{} (M3_EMBED_SERVER_HOST / _PORT): {e}",
                        cfg.host, cfg.port)
    })?;
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            return Err(anyhow::Error::new(PortInUse { addr }));
        }
        Err(e) => return Err(anyhow::anyhow!("cannot listen on {addr}: {e}")),
    };

    let backend = EmbeddedBackend::with_streams_ctx_seqmax_batch(
        cfg.gguf.clone(),
        cfg.streams,
        cfg.n_ctx,
        cfg.seq_max,
        cfg.n_batch,
        cfg.n_ubatch,
    );
    let dim = backend
        .embedding_dim()
        .map_err(|e| anyhow::anyhow!("eager GGUF load failed: {e}"))?;
    log::info!("model loaded, embedding dim = {dim}");
    let tokenizer = backend
        .tokenizer()
        .map_err(|e| anyhow::anyhow!("tokenizer unavailable after GGUF load: {e}"))?;

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
        tokenizer,
    });

    let app = Router::new()
        .route("/embedding", post(embed_handler))
        .route("/v1/embeddings", post(embed_handler))
        .route("/tokenize/count", post(tokenize_count_handler))
        .route("/tokenize/windows", post(tokenize_windows_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    log::info!("listening on http://{addr}");
    on_ready();

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    log::info!("shutdown complete");
    Ok(())
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
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    let texts = texts_from(req.input)?;
    if texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "input is empty".into()));
    }

    // ⚠ ONLY a length problem becomes 413. OOM, a poisoned mutex and a dead
    // worker pool must keep propagating as 500: mislabelling an infrastructure
    // failure as a client length error sends an operator off to shrink their
    // inputs while the real fault goes unreported — the same misdiagnosis
    // issue #139 caused, in the opposite direction. The match is on the TYPED
    // variant, never on message text.
    let rows = state
        .dispatcher
        .embed_batch(texts)
        .await
        .map_err(|e| match e {
            m3_error::M3Error::InputTooLong { tokens, n_ctx } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                // Message preserved verbatim so an older client parsing the
                // text keeps working; the counts are also machine-readable.
                format!(
                    "input too long: {tokens} tokens > n_ctx {n_ctx} \
                     (code=input_too_long observed_tokens={tokens} max_tokens={n_ctx})"
                ),
            ),
            other => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embed failed: {other}"),
            ),
        })?;

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
    Json(req): Json<CountRequest>,
) -> Result<Json<CountResponse>, (StatusCode, String)> {
    let texts = texts_from(req.input)?;
    let tok = state.tokenizer.clone();
    let counts = tokio::task::spawn_blocking(move || tok.count_tokens(&texts))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenize task failed: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
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
    Json(req): Json<WindowsRequest>,
) -> Result<Json<WindowsResponse>, (StatusCode, String)> {
    // A budget no window can meet (BOS + EOS alone are 2 tokens) would cut the
    // text into one window per character.
    if req.budget < 3 {
        return Err((StatusCode::BAD_REQUEST, format!("budget must be at least 3 tokens; got {}", req.budget)));
    }
    let tok = state.tokenizer.clone();
    let windows = tokio::task::spawn_blocking(move || tok.token_windows(&req.input, req.budget, req.overlap))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenize task failed: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(WindowsResponse { windows }))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": state.model_label,
        "pid": std::process::id(),
    }))
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
