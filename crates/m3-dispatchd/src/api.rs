//! HTTP routes, served only over the local IPC endpoint (see `ipc`).

use std::sync::Arc;
use std::time::Duration;

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use crate::hub::{Hub, Wake};

/// Longest a single wait may block. A waiter re-waits after a timeout.
pub const MAX_WAIT: Duration = Duration::from_secs(3600);
/// Most recipients one ring may carry.
pub const MAX_RECIPIENTS: usize = 1000;
/// Longest node name a ring may carry as its origin.
pub const MAX_ORIGIN: usize = 128;

#[derive(Deserialize)]
struct RingRequest {
    recipients: Vec<String>,
    /// The node the message was committed on, when the writer names one.
    #[serde(default)]
    origin: Option<String>,
}

#[derive(Deserialize)]
struct WaitRequest {
    readers: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
}

fn default_timeout() -> f64 {
    300.0
}

pub fn router(hub: Arc<Hub>) -> Router {
    Router::new()
        .route("/ring", post(ring))
        .route("/wait", post(wait))
        .route("/status", get(status))
        .route(
            "/health",
            get(|| async {
                Json(serde_json::json!({ "status": "ok", "pid": std::process::id() }))
            }),
        )
        .with_state(hub)
}

/// An error reply, always JSON so a client parses one shape.
type ApiError = (StatusCode, Json<serde_json::Value>);

/// `Json`, with its rejections (malformed JSON, wrong field types, a missing
/// content type) answered in the same JSON shape as every other error; axum's
/// own are plain text.
struct ApiJson<T>(T);

#[axum::async_trait]
impl<T, S> FromRequest<S> for ApiJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(ApiJson(v)),
            Err(rej) => Err((rej.status(), Json(serde_json::json!({ "error": rej.body_text() })))),
        }
    }
}

fn bad(msg: String) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
}

async fn ring(
    State(hub): State<Arc<Hub>>,
    ApiJson(req): ApiJson<RingRequest>,
) -> Result<StatusCode, ApiError> {
    if req.recipients.is_empty() || req.recipients.len() > MAX_RECIPIENTS {
        return Err(bad(format!(
            "recipients: 1..={MAX_RECIPIENTS} required; got {}",
            req.recipients.len()
        )));
    }
    // An empty id rings nobody: all-empty answered 204 for a ring that never
    // happened. Mixed ones still ring the rest (a writer's partial list stays
    // useful), and the skip is said.
    let empty = req.recipients.iter().filter(|r| r.is_empty()).count();
    if empty == req.recipients.len() {
        return Err(bad(format!(
            "recipients: all {empty} are empty strings, so nothing would ring; give non-empty agent ids"
        )));
    }
    if empty > 0 {
        log::warn!(
            "ring: skipped {empty} empty recipient(s) of {}; rang the rest",
            req.recipients.len()
        );
    }
    if let Some(o) = &req.origin {
        if o.is_empty() || o.len() > MAX_ORIGIN || o.chars().any(char::is_control) {
            return Err(bad(format!(
                "origin: 1..={MAX_ORIGIN} bytes, no control characters; got {} bytes",
                o.len()
            )));
        }
    }
    hub.ring(req.recipients, req.origin.as_deref());
    Ok(StatusCode::NO_CONTENT)
}

async fn wait(
    State(hub): State<Arc<Hub>>,
    ApiJson(req): ApiJson<WaitRequest>,
) -> Result<Json<Wake>, ApiError> {
    if req.readers.iter().all(|r| r.is_empty()) {
        return Err(bad(
            "readers: at least one non-empty agent id required".into()
        ));
    }
    if !(req.timeout_s.is_finite() && req.timeout_s >= 0.0) {
        return Err(bad(format!(
            "timeout_s must be a non-negative number; got {}",
            req.timeout_s
        )));
    }
    let timeout = Duration::from_secs_f64(req.timeout_s).min(MAX_WAIT);
    Ok(Json(hub.wait(&req.readers, timeout).await))
}

async fn status(State(hub): State<Arc<Hub>>) -> Json<serde_json::Value> {
    Json(hub.status())
}
