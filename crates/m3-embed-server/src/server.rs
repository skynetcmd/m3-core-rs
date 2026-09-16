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
use m3_embed_llamacpp::EmbeddedBackend;

use crate::config::ResolvedConfig;

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

struct AppState {
    dispatcher: Arc<Dispatcher<EmbeddedBackend>>,
    model_label: String,
}

/// Build the dispatcher (eager GGUF load), bind the listener, and serve until
/// `shutdown` resolves. On graceful shutdown returns Ok(()).
pub async fn run<F>(cfg: ResolvedConfig, shutdown: F) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    log::info!(
        "m3-embed-server starting: host={} port={} streams={} gguf={}",
        cfg.host,
        cfg.port,
        cfg.streams,
        cfg.gguf
    );

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
    });

    let app = Router::new()
        .route("/embedding", post(embed_handler))
        .route("/v1/embeddings", post(embed_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    log::info!("shutdown complete");
    Ok(())
}

async fn embed_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    let texts: Vec<String> = match req.input {
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| v.as_str().map(String::from).unwrap_or_default())
            .collect(),
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("input must be string or [string]; got {other}"),
            ));
        }
    };
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
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": state.model_label,
    }))
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let s = state.dispatcher.stats();
    Json(serde_json::json!({
        "in_flight": s.in_flight,
        "queue_depth": s.queue_depth,
        "p50_ms": s.p50_ms,
        "p99_ms": s.p99_ms,
        "model": state.model_label,
    }))
}
