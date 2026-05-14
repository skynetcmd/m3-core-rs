//! llama.cpp embedding backends (Phase 3b).
//!
//! Two backends, same API: `HttpBackend` (POST to llama-server) and
//! `EmbeddedBackend` (link llama-cpp-rs, zero IPC). The model math stays in
//! llama.cpp. The default build compiles only `HttpBackend`; `EmbeddedBackend`
//! is a stub unless the `embedded` cargo feature is enabled.

use m3_dispatcher::{Batch, BatchOutput, ModelBackend};
use m3_error::{M3Error, Result};
use serde::{Deserialize, Serialize};

/// Selects how a caller wants to talk to llama.cpp. Kept for config ergonomics
/// in `m3-core-py`; each variant maps to a concrete backend type below.
pub enum Backend {
    Http { url: String },
    Embedded { model_path: String },
}

/// Request body POSTed to llama-server's `/embedding` endpoint.
/// OpenAI-ish: `{ "input": ["text a", "text b"] }`.
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
}

/// One element of llama-server's `/embedding` response.
#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

/// llama-server `/embedding` response: `{ "data": [ { "embedding": [...] } ] }`.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

/// HTTP embedding backend — POSTs batches to a llama-server (or any
/// OpenAI-compatible embedding server).
pub struct HttpBackend {
    client: reqwest::Client,
    endpoint: String,
}

impl HttpBackend {
    /// `base_url` is the server root, e.g. `http://127.0.0.1:8081`. The
    /// `/embedding` path is appended.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base = base_url.into();
        let endpoint = format!("{}/embedding", base.trim_end_matches('/'));
        Self { client: reqwest::Client::new(), endpoint }
    }

    pub fn with_client(base_url: impl Into<String>, client: reqwest::Client) -> Self {
        let base = base_url.into();
        let endpoint = format!("{}/embedding", base.trim_end_matches('/'));
        Self { client, endpoint }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

impl ModelBackend for HttpBackend {
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        let body = EmbeddingRequest { input: batch.texts };
        let resp = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| M3Error::Backend(format!("embedding request failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(M3Error::Backend(format!(
                "embedding server returned {}",
                resp.status()
            )));
        }
        let parsed: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| M3Error::Backend(format!("embedding response decode failed: {e}")))?;
        let rows = parsed.data.into_iter().map(|i| i.embedding).collect();
        Ok(BatchOutput::new(rows))
    }
}

/// Embedded (in-process) backend that links llama-cpp-rs directly — zero IPC,
/// zero JSON. The real implementation is gated behind the `embedded` cargo
/// feature, which is a documented stub: it requires a locally built llama.cpp.
/// The default build constructs this struct but `run` returns an error.
pub struct EmbeddedBackend {
    #[allow(dead_code)]
    model_path: String,
}

impl EmbeddedBackend {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self { model_path: model_path.into() }
    }
}

impl ModelBackend for EmbeddedBackend {
    #[cfg(not(feature = "embedded"))]
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        Err(M3Error::Backend(
            "embedded backend not compiled — build with --features embedded".into(),
        ))
    }

    #[cfg(feature = "embedded")]
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        // TODO(embedded): link llama-cpp-rs, load self.model_path, run
        // llama_decode with mean pooling. Requires a locally built llama.cpp.
        Err(M3Error::Backend(
            "embedded backend feature enabled but llama-cpp-rs integration is a stub".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_backend_constructs_endpoint() {
        let b = HttpBackend::new("http://127.0.0.1:8081/");
        assert_eq!(b.endpoint(), "http://127.0.0.1:8081/embedding");
        let b2 = HttpBackend::new("http://localhost:8081");
        assert_eq!(b2.endpoint(), "http://localhost:8081/embedding");
    }

    #[test]
    fn request_body_serializes_to_expected_json_shape() {
        let req = EmbeddingRequest {
            input: vec!["hello".to_string(), "world".to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json, serde_json::json!({ "input": ["hello", "world"] }));
    }

    #[test]
    fn response_body_deserializes() {
        let raw = serde_json::json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] },
                { "embedding": [0.4, 0.5, 0.6] }
            ]
        });
        let parsed: EmbeddingResponse = serde_json::from_value(raw).unwrap();
        assert_eq!(parsed.data.len(), 2);
        assert_eq!(parsed.data[0].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn embedded_backend_is_stub_by_default() {
        let b = EmbeddedBackend::new("models/bge-m3-q8_0.gguf");
        let err = b.run(Batch::new(vec!["x".into()], 1)).await.unwrap_err();
        assert!(format!("{err}").contains("embedded backend not compiled"));
    }
}
