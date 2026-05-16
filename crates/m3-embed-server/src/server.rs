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

    let rows = state
        .dispatcher
        .embed_batch(texts)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("embed failed: {e}")))?;

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

async fn health_handler() -> &'static str {
    "OK\n"
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
