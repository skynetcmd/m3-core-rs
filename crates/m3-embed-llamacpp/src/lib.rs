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

/// Embedded (in-process) backend that links llama.cpp directly via
/// `llama-cpp-2` — zero IPC, zero JSON. Gated behind the `embedded` cargo
/// feature, which cmake-builds llama.cpp from source (CPU-only). Without the
/// feature this struct still constructs but `run` returns a "not compiled"
/// error.
///
/// Construction is cheap: `new` only stores the path. The GGUF model + the
/// llama backend are loaded lazily on the first `run` call and cached.
pub struct EmbeddedBackend {
    #[allow(dead_code)]
    model_path: String,
    #[cfg(feature = "embedded")]
    state: std::sync::OnceLock<std::sync::Arc<embedded::LoadedModel>>,
}

impl EmbeddedBackend {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            #[cfg(feature = "embedded")]
            state: std::sync::OnceLock::new(),
        }
    }
}

#[cfg(feature = "embedded")]
mod embedded {
    use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use m3_error::{M3Error, Result};
    use std::num::NonZeroU32;

    /// Backend + loaded GGUF model, cached after first use. The `LlamaBackend`
    /// must outlive every model/context, hence stored alongside.
    pub struct LoadedModel {
        backend: LlamaBackend,
        model: LlamaModel,
    }

    impl LoadedModel {
        pub fn load(model_path: &str) -> Result<Self> {
            let backend = LlamaBackend::init()
                .map_err(|e| M3Error::Backend(format!("llama backend init failed: {e}")))?;
            let model = LlamaModel::load_from_file(
                &backend,
                model_path,
                &LlamaModelParams::default(),
            )
            .map_err(|e| {
                M3Error::Backend(format!("failed to load gguf model '{model_path}': {e}"))
            })?;
            Ok(Self { backend, model })
        }

        /// Embedding dimension reported by the model.
        pub fn n_embd(&self) -> i32 {
            self.model.n_embd()
        }

        /// Tokenize + embed a batch of texts with mean pooling. One row per
        /// input text, each of length `n_embd`.
        pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let n_ctx = 8192u32;
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_embeddings(true)
                .with_pooling_type(LlamaPoolingType::Mean);
            let mut ctx = self
                .model
                .new_context(&self.backend, ctx_params)
                .map_err(|e| M3Error::Backend(format!("llama context create failed: {e}")))?;

            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                let tokens = self
                    .model
                    .str_to_token(text, AddBos::Always)
                    .map_err(|e| M3Error::Backend(format!("tokenize failed: {e}")))?;
                if tokens.len() > n_ctx as usize {
                    return Err(M3Error::Backend(format!(
                        "input too long: {} tokens > n_ctx {n_ctx}",
                        tokens.len()
                    )));
                }
                let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
                batch.add_sequence(&tokens, 0, false).map_err(|e| {
                    M3Error::Backend(format!("batch add_sequence failed: {e}"))
                })?;
                ctx.clear_kv_cache();
                ctx.decode(&mut batch)
                    .map_err(|e| M3Error::Backend(format!("llama decode failed: {e}")))?;
                let emb = ctx
                    .embeddings_seq_ith(0)
                    .map_err(|e| M3Error::Backend(format!("read embeddings failed: {e}")))?;
                out.push(emb.to_vec());
            }
            Ok(out)
        }
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
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        let model = if let Some(m) = self.state.get() {
            m.clone()
        } else {
            let loaded = std::sync::Arc::new(embedded::LoadedModel::load(&self.model_path)?);
            let _ = self.state.set(loaded.clone());
            self.state.get().cloned().unwrap_or(loaded)
        };
        let texts = batch.texts;
        let rows = tokio::task::spawn_blocking(move || model.embed(&texts))
            .await
            .map_err(|e| M3Error::Backend(format!("embedding task join failed: {e}")))??;
        Ok(BatchOutput::new(rows))
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

    #[cfg(not(feature = "embedded"))]
    #[tokio::test]
    async fn embedded_backend_is_stub_by_default() {
        let b = EmbeddedBackend::new("models/bge-m3-q8_0.gguf");
        let err = b.run(Batch::new(vec!["x".into()], 1)).await.unwrap_err();
        assert!(format!("{err}").contains("embedded backend not compiled"));
    }

    // Compile-level wiring check for the real backend. Does NOT run inference:
    // that needs a GGUF model file on disk. This only proves the type links
    // against llama-cpp-2 and that `run` returns an error (not a stub string)
    // when the model path is bogus.
    #[cfg(feature = "embedded")]
    #[tokio::test]
    async fn embedded_backend_constructs_and_links() {
        let b = EmbeddedBackend::new("does-not-exist.gguf");
        let err = b.run(Batch::new(vec!["x".into()], 1)).await.unwrap_err();
        let msg = format!("{err}");
        assert!(!msg.contains("not compiled"), "should be the real backend");
        assert!(
            msg.contains("failed to load gguf model") || msg.contains("llama backend init"),
            "unexpected error: {msg}"
        );
    }
}
