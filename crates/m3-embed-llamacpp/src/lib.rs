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
/// Construction is cheap: `new` only stores the path + stream count. The GGUF
/// model and the worker-thread context pool are loaded lazily on the first
/// `run`/`embedding_dim` call and cached.
///
/// ## Concurrency model (CHANGE 1)
///
/// llama.cpp parallelism is *one model, N contexts*. `llama-cpp-2`'s
/// `LlamaContext` is `!Send` and borrows the `LlamaModel`, so it cannot be
/// pooled in a `Vec<Mutex<LlamaContext>>` shared across tokio tasks. Instead
/// the pool is `streams` dedicated OS worker threads; each thread owns one
/// `LlamaContext` for its whole lifetime (the context never leaves the thread
/// that made it, satisfying `!Send`). `LlamaModel`/`LlamaBackend` *are*
/// `Send + Sync`, so they are shared via `Arc`. Work is dispatched over an
/// `mpsc` channel — any idle worker picks up the next batch, so `streams`
/// concurrent `run()` calls genuinely decode in parallel on `streams` cores.
pub struct EmbeddedBackend {
    #[allow(dead_code)]
    model_path: String,
    /// Size of the worker-thread context pool. Reaches this generic crate as a
    /// typed parameter — never read from an env var here (that is `m3-core-py`'s
    /// job). Defaults to 4 via `new`; use `with_streams` to override.
    #[allow(dead_code)]
    streams: usize,
    #[cfg(feature = "embedded")]
    state: std::sync::OnceLock<std::sync::Arc<embedded::ContextPool>>,
}

impl EmbeddedBackend {
    /// Construct with the default pool size (4 streams).
    pub fn new(model_path: impl Into<String>) -> Self {
        Self::with_streams(model_path, 4)
    }

    /// Construct with an explicit context-pool size. `streams` is clamped to
    /// at least 1. This is the value that should flow from
    /// `DispatcherConfig::streams` so the pool and the dispatcher's slot
    /// semaphore agree.
    pub fn with_streams(model_path: impl Into<String>, streams: usize) -> Self {
        Self {
            model_path: model_path.into(),
            streams: streams.max(1),
            #[cfg(feature = "embedded")]
            state: std::sync::OnceLock::new(),
        }
    }

    /// The configured pool size (number of worker threads / concurrent contexts).
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// Load the model (if not already cached) and return its embedding
    /// dimension. Forces the lazy load, so the first call pays the GGUF
    /// load cost. Only available with the `embedded` feature.
    #[cfg(feature = "embedded")]
    pub fn embedding_dim(&self) -> Result<i32> {
        let pool = self.pool()?;
        Ok(pool.n_embd())
    }

    /// Get-or-load the cached worker-thread `ContextPool`. Internal helper
    /// shared by `run` and `embedding_dim`.
    #[cfg(feature = "embedded")]
    fn pool(&self) -> Result<std::sync::Arc<embedded::ContextPool>> {
        if let Some(p) = self.state.get() {
            return Ok(p.clone());
        }
        let loaded = std::sync::Arc::new(embedded::ContextPool::load(
            &self.model_path,
            self.streams,
        )?);
        let _ = self.state.set(loaded.clone());
        Ok(self.state.get().cloned().unwrap_or(loaded))
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
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    /// Process-global `LlamaBackend`. `llama-cpp-2` enforces that
    /// `LlamaBackend::init()` is called **at most once per process**
    /// (`BackendAlreadyInitialized` otherwise) — so every `ContextPool` in the
    /// process must share one backend. `LlamaBackend` is `Send + Sync`, and it
    /// only needs to outlive every model/context, which a `'static` `OnceLock`
    /// guarantees.
    static BACKEND: std::sync::OnceLock<std::result::Result<LlamaBackend, String>> =
        std::sync::OnceLock::new();

    /// Get-or-init the shared backend. Returns a `&'static LlamaBackend`.
    fn shared_backend() -> Result<&'static LlamaBackend> {
        let slot = BACKEND.get_or_init(|| {
            LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))
        });
        match slot {
            Ok(b) => Ok(b),
            Err(e) => Err(M3Error::Backend(e.clone())),
        }
    }

    /// Per-context size. bge-m3 trains at 8192; one context can hold several
    /// short sequences at once (CHANGE 2 — within-batch multi-sequence decode).
    const N_CTX: u32 = 8192;
    /// Max sequences decoded together in one `LlamaBatch`. The context is
    /// created with this `n_seq_max`; a caller batch larger than this (or one
    /// that would exceed `N_CTX` total tokens) is split into chunks.
    const N_SEQ_MAX: u32 = 32;

    /// One job handed to a worker thread: texts to embed + a reply channel.
    struct Job {
        texts: Vec<String>,
        reply: Sender<Result<Vec<Vec<f32>>>>,
    }

    /// Worker-thread context pool. See `EmbeddedBackend`'s doc comment for why
    /// this is threads-with-owned-contexts rather than a `Vec<Mutex<Context>>`:
    /// `LlamaContext` is `!Send` and lifetime-bound to its `LlamaModel`, so each
    /// context must live and die on the thread that created it.
    ///
    /// `n_embd` is captured at load time so `embedding_dim()` need not round-trip
    /// through a worker.
    pub struct ContextPool {
        tx: Mutex<Option<Sender<Job>>>,
        workers: Mutex<Vec<JoinHandle<()>>>,
        n_embd: i32,
        streams: usize,
    }

    impl ContextPool {
        /// Load the GGUF model once and spin up `streams` worker threads, each
        /// with its own `LlamaContext`. Blocks until every worker has created
        /// its context (or one fails).
        pub fn load(model_path: &str, streams: usize) -> Result<Self> {
            let streams = streams.max(1);
            // Shared, process-global backend (see `BACKEND` above).
            let backend = shared_backend()?;
            let model = Arc::new(
                LlamaModel::load_from_file(backend, model_path, &LlamaModelParams::default())
                    .map_err(|e| {
                        M3Error::Backend(format!(
                            "failed to load gguf model '{model_path}': {e}"
                        ))
                    })?,
            );
            let n_embd = model.n_embd();

            let (tx, rx) = mpsc::channel::<Job>();
            let rx = Arc::new(Mutex::new(rx));

            // Each worker confirms context creation over this channel so `load`
            // surfaces a context-init failure instead of silently degrading.
            let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
            let mut workers = Vec::with_capacity(streams);
            for id in 0..streams {
                let model = model.clone();
                let rx = rx.clone();
                let ready_tx = ready_tx.clone();
                let handle = std::thread::Builder::new()
                    .name(format!("m3-embed-ctx-{id}"))
                    .spawn(move || worker_loop(id, backend, model, rx, ready_tx))
                    .map_err(|e| {
                        M3Error::Backend(format!("failed to spawn embed worker {id}: {e}"))
                    })?;
                workers.push(handle);
            }
            drop(ready_tx);

            // Collect one readiness result per worker.
            for _ in 0..streams {
                match ready_rx.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(M3Error::Backend(
                            "embed worker died before reporting readiness".into(),
                        ))
                    }
                }
            }

            Ok(Self {
                tx: Mutex::new(Some(tx)),
                workers: Mutex::new(workers),
                n_embd,
                streams,
            })
        }

        /// Embedding dimension reported by the model.
        pub fn n_embd(&self) -> i32 {
            self.n_embd
        }

        /// Number of worker threads / concurrent contexts.
        #[allow(dead_code)]
        pub fn streams(&self) -> usize {
            self.streams
        }

        /// Submit a batch to the pool and block until a worker returns it. The
        /// caller (`EmbeddedBackend::run`) wraps this in `spawn_blocking`, so
        /// blocking here is fine and lets the OS scheduler spread concurrent
        /// `run()` calls across idle workers.
        pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let (reply_tx, reply_rx) = mpsc::channel();
            {
                let guard = self
                    .tx
                    .lock()
                    .map_err(|_| M3Error::Backend("embed pool mutex poisoned".into()))?;
                let sender = guard
                    .as_ref()
                    .ok_or_else(|| M3Error::Backend("embed pool shut down".into()))?;
                sender
                    .send(Job { texts, reply: reply_tx })
                    .map_err(|_| M3Error::Backend("embed pool has no live workers".into()))?;
            }
            reply_rx
                .recv()
                .map_err(|_| M3Error::Backend("embed worker dropped the job".into()))?
        }
    }

    impl Drop for ContextPool {
        fn drop(&mut self) {
            // Drop the sender so every worker's `rx.recv()` returns `Err` and
            // the threads exit their loops, then join them.
            if let Ok(mut g) = self.tx.lock() {
                *g = None;
            }
            if let Ok(mut workers) = self.workers.lock() {
                for h in workers.drain(..) {
                    let _ = h.join();
                }
            }
        }
    }

    /// Body of one worker thread: create a context, then serve jobs until the
    /// channel closes. The `LlamaContext` is created here and never leaves this
    /// thread — that is what makes a `!Send` context safe to "pool".
    fn worker_loop(
        _id: usize,
        backend: &'static LlamaBackend,
        model: Arc<LlamaModel>,
        rx: Arc<Mutex<Receiver<Job>>>,
        ready_tx: Sender<Result<()>>,
    ) {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_batch(N_CTX)
            .with_n_seq_max(N_SEQ_MAX)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Cls);
        let mut ctx = match model.new_context(backend, ctx_params) {
            Ok(c) => {
                let _ = ready_tx.send(Ok(()));
                c
            }
            Err(e) => {
                let _ = ready_tx.send(Err(M3Error::Backend(format!(
                    "llama context create failed: {e}"
                ))));
                return;
            }
        };
        drop(ready_tx);

        loop {
            // Hold the lock only long enough to dequeue one job, so other idle
            // workers can grab the next one.
            let job = {
                let guard = match rx.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                match guard.recv() {
                    Ok(j) => j,
                    Err(_) => return, // sender dropped — pool shutting down
                }
            };
            let result = embed_on_ctx(&model, &mut ctx, &job.texts);
            let _ = job.reply.send(result);
        }
    }

    /// Tokenize + embed a batch of texts on one already-created context.
    ///
    /// CHANGE 2 — within-batch multi-sequence decode: texts are packed into one
    /// `LlamaBatch` as distinct sequences (`seq_id = 0..k`) and decoded in a
    /// single `decode` call, then `embeddings_seq_ith(seq_id)` reads each. A
    /// chunk is closed when adding the next text would exceed either `N_CTX`
    /// total tokens or `N_SEQ_MAX` sequences; the loop then decodes the chunk
    /// and starts a fresh one. So a small caller batch is one decode; a large
    /// one falls back to chunked decodes.
    ///
    /// Pooling is CLS and output is L2-normalized to match m3-memory's HTTP
    /// embed path (llama-server `--pooling cls` + OpenAI `/v1/embeddings`
    /// L2-normalizes by default). Mean pooling here produced vectors that
    /// cosine-0.74 against the stored corpus instead of ~1.0 — do not regress.
    fn embed_on_ctx(
        model: &LlamaModel,
        ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        // Tokenize everything up front; reject any single over-long input.
        let mut tokenized = Vec::with_capacity(texts.len());
        for text in texts {
            let tokens = model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| M3Error::Backend(format!("tokenize failed: {e}")))?;
            if tokens.len() > N_CTX as usize {
                return Err(M3Error::Backend(format!(
                    "input too long: {} tokens > n_ctx {N_CTX}",
                    tokens.len()
                )));
            }
            tokenized.push(tokens);
        }

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut idx = 0usize;
        while idx < tokenized.len() {
            // Greedily pack a chunk: up to N_SEQ_MAX sequences and N_CTX tokens.
            let mut chunk_end = idx;
            let mut chunk_tokens = 0usize;
            while chunk_end < tokenized.len() {
                let next = tokenized[chunk_end].len().max(1);
                let seqs = chunk_end - idx;
                if seqs >= N_SEQ_MAX as usize {
                    break;
                }
                if seqs > 0 && chunk_tokens + next > N_CTX as usize {
                    break;
                }
                chunk_tokens += next;
                chunk_end += 1;
            }

            let chunk = &tokenized[idx..chunk_end];
            let total_tokens: usize = chunk.iter().map(|t| t.len().max(1)).sum();
            let mut batch = LlamaBatch::new(total_tokens.max(1), chunk.len() as i32);
            for (seq_i, tokens) in chunk.iter().enumerate() {
                batch
                    .add_sequence(tokens, seq_i as i32, false)
                    .map_err(|e| {
                        M3Error::Backend(format!("batch add_sequence failed: {e}"))
                    })?;
            }
            // Fresh KV state per chunk; sequences within the chunk are
            // independent so one clear before the decode is enough (no
            // per-text clear thrash).
            ctx.clear_kv_cache();
            ctx.decode(&mut batch)
                .map_err(|e| M3Error::Backend(format!("llama decode failed: {e}")))?;
            for seq_i in 0..chunk.len() {
                let emb = ctx
                    .embeddings_seq_ith(seq_i as i32)
                    .map_err(|e| M3Error::Backend(format!("read embeddings failed: {e}")))?;
                out.push(l2_normalize(emb));
            }
            idx = chunk_end;
        }
        Ok(out)
    }

    /// L2-normalize a vector to unit length. A zero vector is returned
    /// unchanged (no division by zero).
    fn l2_normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
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
        let pool = self.pool()?;
        let texts = batch.texts;
        // `pool.embed` blocks on an mpsc round-trip to a worker thread; run it
        // on the blocking pool so the tokio reactor stays free. Concurrent
        // `run()` calls each block on a different idle worker, so they decode
        // in parallel — genuinely multi-stream up to `streams` contexts.
        let rows = tokio::task::spawn_blocking(move || pool.embed(texts))
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

    /// End-to-end inference against a real GGUF model. Opt-in: set
    /// `M3_TEST_GGUF` to a GGUF embedding model path. Skipped when unset so
    /// CI without a model file stays green.
    #[cfg(feature = "embedded")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn embedded_backend_runs_real_inference() {
        let Ok(model_path) = std::env::var("M3_TEST_GGUF") else {
            eprintln!("M3_TEST_GGUF unset — skipping real-inference test");
            return;
        };
        // 3-stream pool exercises the worker-thread context pool (CHANGE 1).
        let b = EmbeddedBackend::with_streams(model_path, 3);
        let out = b
            .run(Batch::new(vec!["hello world".into(), "a different sentence".into()], 4))
            .await
            .expect("embedding run should succeed");
        assert_eq!(out.rows.len(), 2, "one embedding row per input text");
        let dim = out.rows[0].len();
        assert!(dim > 0, "embedding dimension must be non-zero");
        assert_eq!(out.rows[1].len(), dim, "all rows share one dimension");
        assert!(
            out.rows[0].iter().any(|&x| x != 0.0),
            "embedding must not be all zeros"
        );
        // Parity guard: CLS pooling + L2-normalized output (norm ~= 1.0).
        for row in &out.rows {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "embedding must be L2-normalized, got norm {norm}"
            );
        }
    }

    /// Concurrency check: fire `streams` overlapping `run()` calls through one
    /// backend and confirm the worker-thread pool serves them all correctly.
    /// Opt-in via `M3_TEST_GGUF`.
    #[cfg(feature = "embedded")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn embedded_backend_concurrent_streams() {
        let Ok(model_path) = std::env::var("M3_TEST_GGUF") else {
            eprintln!("M3_TEST_GGUF unset — skipping concurrency test");
            return;
        };
        let b = std::sync::Arc::new(EmbeddedBackend::with_streams(model_path, 3));
        // Force the model+pool load once before timing the concurrent phase.
        let dim = b.embedding_dim().expect("model should load");
        let started = std::time::Instant::now();
        let mut handles = Vec::new();
        for i in 0..6 {
            let b = b.clone();
            handles.push(tokio::spawn(async move {
                b.run(Batch::new(
                    vec![format!("concurrent embedding request number {i}")],
                    8,
                ))
                .await
            }));
        }
        for h in handles {
            let out = h.await.unwrap().expect("concurrent run should succeed");
            assert_eq!(out.rows.len(), 1);
            assert_eq!(out.rows[0].len() as i32, dim);
        }
        eprintln!(
            "6 concurrent run() calls over a 3-stream pool finished in {:?}",
            started.elapsed()
        );
    }
}
