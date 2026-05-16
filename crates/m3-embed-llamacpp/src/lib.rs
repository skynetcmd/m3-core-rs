//! llama.cpp embedding backends (Phase 3b).
//!
//! Two backends, same API: `HttpBackend` (POST to llama-server) and
//! `EmbeddedBackend` (link llama-cpp-rs, zero IPC). The model math stays in
//! llama.cpp. The default build compiles only `HttpBackend`; `EmbeddedBackend`
//! is a stub unless the `embedded` cargo feature is enabled.

// Wave 9.0 — mutual exclusivity for GPU backends: at most one of
// cuda/vulkan/metal. Enforced at build time because llama-cpp-2's GPU backends
// conflict in the underlying ggml-cpu / ggml-cuda / ggml-vulkan / ggml-metal
// C build. Cargo features are additive, so without this gate a `cargo check
// --features embedded-cuda,embedded-vulkan` would happily forward both
// backend selectors to llama-cpp-sys-2 and the C toolchain would fail
// downstream with a much less obvious error.
#[cfg(any(
    all(feature = "embedded-cuda", feature = "embedded-vulkan"),
    all(feature = "embedded-cuda", feature = "embedded-metal"),
    all(feature = "embedded-vulkan", feature = "embedded-metal"),
))]
compile_error!(
    "exactly one of embedded-cuda / embedded-vulkan / embedded-metal may be enabled; \
     cargo features are additive, but llama.cpp's GPU backends are not."
);

// `embedded-metal` only produces a real Metal build on macOS — on Linux/Windows
// the upstream `llama-cpp-sys-2` build silently falls back to CPU, which is a
// foot-gun (operator gets a wheel labeled "metal" that actually serves CPU
// embeddings). Hard-fail at compile time on non-macOS targets.
#[cfg(all(feature = "embedded-metal", not(target_os = "macos")))]
compile_error!(
    "embedded-metal requires target_os=macos; on Linux/Windows use embedded \
     (CPU), embedded-cuda, or embedded-vulkan instead."
);

/// String label of the active embedded backend, exposed to Python (via
/// `m3-core-py::embed_backend_label`) so callers can observe which compute
/// backend served a request — wave 9.4 will surface this in backend stats.
/// CPU is the default `embedded` build; the GPU variants override.
#[cfg(feature = "embedded")]
pub fn active_backend() -> &'static str {
    #[cfg(feature = "embedded-cuda")]
    {
        return "cuda";
    }
    #[cfg(feature = "embedded-vulkan")]
    {
        return "vulkan";
    }
    #[cfg(feature = "embedded-metal")]
    {
        return "metal";
    }
    #[cfg(not(any(
        feature = "embedded-cuda",
        feature = "embedded-vulkan",
        feature = "embedded-metal"
    )))]
    {
        "cpu"
    }
}

use m3_dispatcher::{Batch, BatchOutput, ModelBackend};
use m3_error::{M3Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Duration;

/// Process-wide shared `reqwest::Client` used by `HttpBackend::new`. Wave-7
/// fix #14: previously each `HttpBackend::new` built a fresh
/// `reqwest::Client`, so independent backend instances couldn't reuse
/// connections to a localhost llama-server. `reqwest::Client` is internally
/// `Arc`-wrapped, so cloning out of the OnceLock is cheap and every clone
/// shares the same connection pool.
///
/// Tuning: `pool_max_idle_per_host(16)` gives headroom for an 8-stream
/// dispatcher; `tcp_keepalive(60s)` keeps long-lived localhost sockets warm.
/// HTTP/2 prior-knowledge is intentionally NOT enabled here — llama-server's
/// embedded HTTP layer (cpp-httplib) is HTTP/1.1 only, so forcing prior
/// knowledge would break every request. Callers who run against an HTTP/2
/// server can opt in via [`HttpBackend::with_config`] +
/// [`HttpBackendConfig::http2_prior_knowledge`].
static SHARED_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn shared_client() -> reqwest::Client {
    SHARED_CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .pool_max_idle_per_host(16)
                .tcp_keepalive(Duration::from_secs(60))
                .build()
                .expect("default reqwest client build failed")
        })
        .clone()
}

/// Tunables for [`HttpBackend::with_config`] (wave-7 fix #14). The
/// `HttpBackend::new` path uses a process-wide shared client with the
/// defaults below; supplying a non-default config builds a fresh client
/// dedicated to that backend instance instead.
#[derive(Debug, Clone)]
pub struct HttpBackendConfig {
    /// Max idle connections kept warm per (scheme, host, port). Default 16.
    pub pool_max_idle_per_host: usize,
    /// TCP keep-alive interval. `None` disables. Default `Some(60s)`.
    pub tcp_keepalive: Option<Duration>,
    /// Force HTTP/2 without ALPN negotiation. **Off by default** — llama-server
    /// is HTTP/1.1-only, and turning this on against an HTTP/1.1 server fails
    /// every request. Only enable when you know the server speaks HTTP/2
    /// unconditionally.
    pub http2_prior_knowledge: bool,
    /// Per-request timeout (Fix #27). Default 30s.
    pub request_timeout: Duration,
}

impl Default for HttpBackendConfig {
    fn default() -> Self {
        Self {
            pool_max_idle_per_host: 16,
            tcp_keepalive: Some(Duration::from_secs(60)),
            http2_prior_knowledge: false,
            request_timeout: Duration::from_secs(30),
        }
    }
}

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
    /// Per-request timeout (Fix #27). A wedged llama-server otherwise hangs
    /// forever — the dispatcher's circuit breaker only trips on errors, so
    /// "slow but not failed" requests never recover. Defaults to 30s.
    request_timeout: std::time::Duration,
}

impl HttpBackend {
    /// `base_url` is the server root, e.g. `http://127.0.0.1:8081`. The
    /// `/embedding` path is appended.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base = base_url.into();
        let endpoint = format!("{}/embedding", base.trim_end_matches('/'));
        // Wave-7 fix #14: share the process-wide tuned client so multiple
        // HttpBackend instances reuse connections.
        Self {
            client: shared_client(),
            endpoint,
            request_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_client(base_url: impl Into<String>, client: reqwest::Client) -> Self {
        let base = base_url.into();
        let endpoint = format!("{}/embedding", base.trim_end_matches('/'));
        Self {
            client,
            endpoint,
            request_timeout: Duration::from_secs(30),
        }
    }

    /// Build an `HttpBackend` from an explicit [`HttpBackendConfig`] (wave-7
    /// fix #14). Unlike [`HttpBackend::new`] this builds a fresh dedicated
    /// `reqwest::Client` for this backend instance — use it when the defaults
    /// (16-idle pool, 60s keep-alive, HTTP/1.1 negotiation, 30s timeout)
    /// don't fit, e.g. against a non-localhost server or one that speaks
    /// HTTP/2 unconditionally.
    pub fn with_config(base_url: impl Into<String>, config: HttpBackendConfig) -> Self {
        let base = base_url.into();
        let endpoint = format!("{}/embedding", base.trim_end_matches('/'));
        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host);
        if let Some(ka) = config.tcp_keepalive {
            builder = builder.tcp_keepalive(ka);
        }
        if config.http2_prior_knowledge {
            builder = builder.http2_prior_knowledge();
        }
        let client = builder
            .build()
            .expect("custom reqwest client build failed");
        Self {
            client,
            endpoint,
            request_timeout: config.request_timeout,
        }
    }

    /// Override the per-request timeout (Fix #27). Default: 30s.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn request_timeout(&self) -> std::time::Duration {
        self.request_timeout
    }
}

impl ModelBackend for HttpBackend {
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        let body = EmbeddingRequest { input: batch.texts };
        let resp = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .timeout(self.request_timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    M3Error::Backend(format!(
                        "embedding request timed out after {}s",
                        self.request_timeout.as_secs_f64()
                    ))
                } else {
                    M3Error::Backend(format!("embedding request failed: {e}"))
                }
            })?;
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
    /// Per-context token window. Set via `M3_EMBED_CTX` (read in `m3-core-py`).
    /// Defaults to 8192. Controls n_ctx (the KV-cache budget) on the llama.cpp context.
    #[allow(dead_code)]
    n_ctx: u32,
    /// Max sequences packed into one LlamaBatch decode call per worker.
    /// Set via `M3_EMBED_SEQ_MAX` (read in `m3-core-py`). Default: 32.
    #[allow(dead_code)]
    seq_max: u32,
    /// llama.cpp's prompt-processing batch ceiling. Decoupled from `n_ctx` in
    /// wave-3 fix #1: previously `n_batch == n_ctx == 8192` was wildly
    /// over-provisioned for typical embed batches and bloated allocator
    /// pressure. Set via `M3_EMBED_N_BATCH` (read in `m3-core-py`).
    #[allow(dead_code)]
    n_batch: u32,
    /// llama.cpp's micro-batch (the inner SIMD tile size). Smaller than
    /// `n_batch`; tuned independently. Set via `M3_EMBED_N_UBATCH`.
    #[allow(dead_code)]
    n_ubatch: u32,
    #[cfg(feature = "embedded")]
    state: std::sync::OnceLock<std::sync::Arc<embedded::ContextPool>>,
}

impl EmbeddedBackend {
    /// Construct with the default pool size (4 streams), default ctx (8192), and default seq_max (32).
    /// Default n_batch = min(n_ctx, 2048), n_ubatch = min(n_batch, 512).
    pub fn new(model_path: impl Into<String>) -> Self {
        Self::with_streams_ctx_seqmax(model_path, 4, 8192, 32)
    }

    /// Construct with an explicit context-pool size.
    /// Defaults: n_batch = min(n_ctx, 2048), n_ubatch = min(n_batch, 512).
    pub fn with_streams(model_path: impl Into<String>, streams: usize) -> Self {
        Self::with_streams_ctx_seqmax(model_path, streams, 8192, 32)
    }

    /// Construct with explicit stream count and per-context token window.
    /// Defaults: n_batch = min(n_ctx, 2048), n_ubatch = min(n_batch, 512).
    pub fn with_streams_and_ctx(model_path: impl Into<String>, streams: usize, n_ctx: u32) -> Self {
        Self::with_streams_ctx_seqmax(model_path, streams, n_ctx, 32)
    }

    /// Construct with explicit stream count, per-context token window, and max sequences per decode.
    /// `n_ctx` is the total token budget per decode call (shared across all sequences in a chunk).
    /// `seq_max` caps how many sequences are packed into one decode call per worker thread.
    /// Defaults n_batch / n_ubatch as documented on `with_streams_ctx_seqmax_batch`.
    pub fn with_streams_ctx_seqmax(model_path: impl Into<String>, streams: usize, n_ctx: u32, seq_max: u32) -> Self {
        // Wave-3 fix #1 defaults: cap n_batch at 2048, n_ubatch at 512 — these
        // are the values that bench wave 0 showed give the best
        // throughput-per-allocator-pressure tradeoff for BGE-M3 sized inputs,
        // and they also work well for short-batch cases.
        let n_batch = n_ctx.min(2048);
        let n_ubatch = n_batch.min(512);
        Self::with_streams_ctx_seqmax_batch(model_path, streams, n_ctx, seq_max, n_batch, n_ubatch)
    }

    /// Full-control constructor (wave-3 fix #1). All five tunables explicit:
    /// pool size, context budget, max sequences/decode, n_batch (prompt-process
    /// batch ceiling), and n_ubatch (micro-batch / SIMD tile). Existing
    /// constructors above are thin shims that pass sensible defaults
    /// (`n_batch = min(n_ctx, 2048)`, `n_ubatch = min(n_batch, 512)`).
    pub fn with_streams_ctx_seqmax_batch(
        model_path: impl Into<String>,
        streams: usize,
        n_ctx: u32,
        seq_max: u32,
        n_batch: u32,
        n_ubatch: u32,
    ) -> Self {
        let n_ctx = n_ctx.max(64);
        // n_batch must be >= n_ubatch and >= 1; clamp into [1, n_ctx].
        let n_batch = n_batch.max(1).min(n_ctx);
        let n_ubatch = n_ubatch.max(1).min(n_batch);
        Self {
            model_path: model_path.into(),
            streams: streams.max(1),
            n_ctx,
            seq_max: seq_max.max(1),
            n_batch,
            n_ubatch,
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

    /// Test-only: peek at the cached `Arc<ContextPool>` without forcing a
    /// load. Returns `None` until `embedding_dim`/`run` has populated the
    /// `OnceLock`. Used by the shared-pool invariant test in `tests` below
    /// (Fix #12) — comparing `Arc::as_ptr` across two handles proves they
    /// point at the same allocation, i.e. one underlying ContextPool per
    /// `PyEmbeddedEmbedder` instance.
    #[cfg(feature = "embedded")]
    #[doc(hidden)]
    pub fn _pool_handle_for_test(&self) -> Option<std::sync::Arc<embedded::ContextPool>> {
        self.state.get().cloned()
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
            self.n_ctx,
            self.seq_max,
            self.n_batch,
            self.n_ubatch,
        )?);
        let _ = self.state.set(loaded.clone());
        Ok(self.state.get().cloned().unwrap_or(loaded))
    }
}

#[cfg(feature = "embedded")]
#[doc(hidden)]
pub mod embedded {
    use crossbeam_channel::{unbounded, Receiver, Sender};
    use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use m3_error::{M3Error, Result};
    use std::num::NonZeroU32;
    use std::sync::mpsc as stdmpsc;
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
            // Wave 9.1: Blackwell/595.xx CUDA Graphs workaround — applied before
            // LlamaBackend::init so the env var is read by ggml-cuda during its
            // own init. See `maybe_disable_cuda_graphs_for_blackwell` below.
            #[cfg(feature = "embedded-cuda")]
            maybe_disable_cuda_graphs_for_blackwell();

            LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))
        });
        match slot {
            Ok(b) => Ok(b),
            Err(e) => Err(M3Error::Backend(e.clone())),
        }
    }

    /// Auto-detect NVIDIA compute capability via `nvidia-smi`. If sm_120 (Blackwell)
    /// or newer, set GGML_CUDA_DISABLE_GRAPHS=1 to work around the llama.cpp CUDA
    /// Graphs / Blackwell-595.xx pathology that drops embed latency from ~2100ms to
    /// ~8ms. Called exactly once, before LlamaBackend::init().
    ///
    /// Safe to call without nvidia-smi available: logs a warning and defaults to
    /// setting the env var (5-10% throughput cost on pre-Blackwell vs ~260× regression
    /// on Blackwell with graphs enabled — safer to disable).
    #[cfg(feature = "embedded-cuda")]
    pub(crate) fn maybe_disable_cuda_graphs_for_blackwell() {
        // Skip if the operator already set the var (any value — explicit override wins).
        if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some() {
            eprintln!("GGML_CUDA_DISABLE_GRAPHS already set, leaving it alone");
            return;
        }

        let cap = detect_cuda_compute_cap();
        match cap {
            Some(c) if c >= 12.0 => {
                eprintln!("detected CUDA compute cap {c:.1} (Blackwell+) — setting GGML_CUDA_DISABLE_GRAPHS=1");
                std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1");
            }
            Some(c) => {
                eprintln!("detected CUDA compute cap {c:.1} (<12.0) — CUDA Graphs enabled");
            }
            None => {
                eprintln!("could not detect CUDA compute cap — setting GGML_CUDA_DISABLE_GRAPHS=1 defensively");
                std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1");
            }
        }
    }

    #[cfg(feature = "embedded-cuda")]
    pub(crate) fn detect_cuda_compute_cap() -> Option<f32> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        // First GPU's compute cap if multi-GPU box.
        let first_line = stdout.lines().next()?.trim();
        first_line.parse::<f32>().ok()
    }

    /// One job handed to a worker thread: texts to embed + a reply channel.
    /// The reply channel stays on `std::sync::mpsc` because it is single-shot
    /// (one producer = the worker that picks the job, one consumer = the
    /// caller). Only the work-distribution channel needed MPMC.
    struct Job {
        texts: Vec<String>,
        reply: stdmpsc::Sender<Result<Vec<Vec<f32>>>>,
    }

    /// Worker-thread context pool. See `EmbeddedBackend`'s doc comment for why
    /// this is threads-with-owned-contexts rather than a `Vec<Mutex<Context>>`:
    /// `LlamaContext` is `!Send` and lifetime-bound to its `LlamaModel`, so each
    /// context must live and die on the thread that created it.
    ///
    /// `n_embd` is captured at load time so `embedding_dim()` need not round-trip
    /// through a worker.
    pub struct ContextPool {
        /// Wave-3 fix #4: crossbeam MPMC. `Mutex<Option<Sender<Job>>>` slot is
        /// retained so the API surface (Drop-time shutdown by taking the
        /// sender) is identical — only the underlying channel type changed.
        /// Crossbeam's Sender is Clone+Send+Sync; dropping the last sender
        /// causes worker `recv()` calls to return `Err`, which exits the loops.
        tx: Mutex<Option<Sender<Job>>>,
        workers: Mutex<Vec<JoinHandle<()>>>,
        n_embd: i32,
        streams: usize,
    }

    impl ContextPool {
        /// Load the GGUF model once and spin up `streams` worker threads, each
        /// with its own `LlamaContext`. `n_ctx` is the total token budget per
        /// decode call (shared across all sequences in one chunk). `seq_max` is
        /// the max sequences packed into one `LlamaBatch` decode per worker.
        /// `n_batch` / `n_ubatch` are the llama.cpp prompt-process batch and
        /// micro-batch ceilings (wave-3 fix #1).
        pub fn load(
            model_path: &str,
            streams: usize,
            n_ctx: u32,
            seq_max: u32,
            n_batch: u32,
            n_ubatch: u32,
        ) -> Result<Self> {
            let streams = streams.max(1);
            let n_ctx = n_ctx.max(64);
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

            // Wave-3 fix #4: crossbeam_channel::unbounded gives an MPMC pair.
            // The Receiver is `Clone + Send + Sync`, so each worker holds its
            // own clone and `recv()` is lock-free — no `Mutex<Receiver>` hot
            // lock as streams scales.
            let (tx, rx) = unbounded::<Job>();

            // Each worker confirms context creation over this channel so `load`
            // surfaces a context-init failure instead of silently degrading.
            // Kept on std mpsc — single consumer (this thread).
            let (ready_tx, ready_rx) = stdmpsc::channel::<Result<()>>();
            let mut workers = Vec::with_capacity(streams);
            for id in 0..streams {
                let model = model.clone();
                let rx = rx.clone();
                let ready_tx = ready_tx.clone();
                let handle = std::thread::Builder::new()
                    .name(format!("m3-embed-ctx-{id}"))
                    .spawn(move || {
                        worker_loop(id, backend, model, rx, ready_tx, n_ctx, seq_max, n_batch, n_ubatch)
                    })
                    .map_err(|e| {
                        M3Error::Backend(format!("failed to spawn embed worker {id}: {e}"))
                    })?;
                workers.push(handle);
            }
            drop(ready_tx);
            // Drop the original receiver — workers each own a clone. This way
            // when the ContextPool's Sender slot is taken on Drop, *all* live
            // senders are gone and worker recv()s return Err.
            drop(rx);

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
            let (reply_tx, reply_rx) = stdmpsc::channel();
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
    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        _id: usize,
        backend: &'static LlamaBackend,
        model: Arc<LlamaModel>,
        rx: Receiver<Job>,
        ready_tx: stdmpsc::Sender<Result<()>>,
        n_ctx: u32,
        seq_max: u32,
        n_batch: u32,
        n_ubatch: u32,
    ) {
        // Wave-3 fix #1: decouple n_batch / n_ubatch from n_ctx, and turn on
        // flash attention explicitly. Previously n_batch == n_ubatch == n_ctx
        // (8192 by default) which over-provisions the prompt-process tile
        // wildly for normal embed batches. Flash-attn cuts attention memory
        // bandwidth ~3x on long inputs.
        //
        // The flash-attn setter in llama-cpp-2 0.1.146 is
        // `with_flash_attention_policy(llama_cpp_sys_2::llama_flash_attn_type)`.
        // The crate's default is AUTO; we set ENABLED so the per-context log
        // line is unambiguous and any future default flip can't disable it.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_ubatch)
            .with_n_seq_max(seq_max)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Cls)
            .with_flash_attention_policy(
                llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
            );
        let mut ctx = match model.new_context(backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                let _ = ready_tx.send(Err(M3Error::Backend(format!(
                    "llama context create failed: {e}"
                ))));
                return;
            }
        };

        // Fix #2: warmup decode. The first real decode call pays the
        // kernel-cache + KV-allocator + (on GPU) shader-compile cost — easily
        // 1-5s of first-token latency. Run a single trivial decode + embedding
        // read here so the user-visible cold call only pays the GGUF-load +
        // worker-spawn cost, not the per-context first-decode cost.
        let warm_tokens = match model.str_to_token("x", AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                let _ = ready_tx.send(Err(M3Error::Backend(format!(
                    "warmup tokenize failed: {e}"
                ))));
                return;
            }
        };
        let mut warm_batch = LlamaBatch::new(warm_tokens.len().max(1), 1);
        // Match the hot path: CLS pooling needs all tokens flagged for
        // logits, and BGE-M3 is non-causal so encode() is the right entry.
        // See embed_on_ctx for the full rationale.
        if let Err(e) = warm_batch.add_sequence(&warm_tokens, 0, true) {
            let _ = ready_tx.send(Err(M3Error::Backend(format!(
                "warmup add_sequence failed: {e}"
            ))));
            return;
        }
        ctx.clear_kv_cache();
        if let Err(e) = ctx.encode(&mut warm_batch) {
            let _ = ready_tx.send(Err(M3Error::Backend(format!(
                "warmup encode failed: {e}"
            ))));
            return;
        }
        if let Err(e) = ctx.embeddings_seq_ith(0) {
            let _ = ready_tx.send(Err(M3Error::Backend(format!(
                "warmup embeddings read failed: {e}"
            ))));
            return;
        }
        // Discard warmup result; KV cache will be cleared again per real chunk.

        let _ = ready_tx.send(Ok(()));
        drop(ready_tx);

        // Wave-3 fix #3: keep one LlamaBatch per worker, sized once at the
        // worst case the worker will ever face (`n_ctx` total tokens across
        // `seq_max` sequences). `LlamaBatch::clear()` exists in 0.1.146 and is
        // cheap (just resets counters + zeroes the pos/seq tables in place),
        // so reusing the allocation across every chunk eliminates a fresh
        // `LlamaBatch::new` per decode.
        let mut batch = LlamaBatch::new(n_ctx as usize, seq_max as i32);

        // Wave-3 fix #2: hoist the L2-normalize destination buffer out of the
        // chunk loop too; reused across every sequence in every chunk.
        let mut norm_buf: Vec<f32> = Vec::with_capacity(model.n_embd() as usize);

        loop {
            // Wave-3 fix #4: lock-free recv. crossbeam's Receiver is Sync, so
            // every worker just calls recv() directly — no Mutex<Receiver>
            // contention as streams scales.
            let job = match rx.recv() {
                Ok(j) => j,
                Err(_) => return, // all senders dropped — pool shutting down
            };
            let result = embed_on_ctx(
                &model, &mut ctx, &job.texts, n_ctx, seq_max, &mut batch, &mut norm_buf,
            );
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
        n_ctx: u32,
        seq_max: u32,
        batch: &mut LlamaBatch,
        norm_buf: &mut Vec<f32>,
    ) -> Result<Vec<Vec<f32>>> {
        let n_ctx_usize = n_ctx as usize;
        let seq_max_usize = seq_max as usize;
        let n_embd_usize = model.n_embd() as usize;
        // Per-sequence capacity = total n_ctx / seq_max.
        // Single texts longer than this are allowed (they use the full n_ctx alone).
        let per_seq_capacity = n_ctx_usize.saturating_div(seq_max_usize.max(1));
        // Tokenize everything up front.
        let mut tokenized = Vec::with_capacity(texts.len());
        for text in texts {
            let tokens = model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| M3Error::Backend(format!("tokenize failed: {e}")))?;
            if tokens.len() > n_ctx_usize {
                return Err(M3Error::Backend(format!(
                    "input too long: {} tokens > n_ctx {}",
                    tokens.len(), n_ctx
                )));
            }
            tokenized.push(tokens);
        }

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut idx = 0usize;
        while idx < tokenized.len() {
            // Greedily pack a chunk: up to seq_max sequences, fitting within n_ctx tokens.
            // A single long text (> per_seq_capacity) decodes alone using the full n_ctx.
            let mut chunk_end = idx;
            let mut chunk_tokens = 0usize;
            while chunk_end < tokenized.len() {
                let next = tokenized[chunk_end].len().max(1);
                let seqs = chunk_end - idx;
                // Single-text fallback: if this text alone exceeds per_seq_capacity,
                // let it run solo (full n_ctx available for one sequence).
                if seqs == 0 && next > per_seq_capacity {
                    chunk_end += 1;
                    break;
                }
                if seqs >= seq_max_usize {
                    break;
                }
                if seqs > 0 && chunk_tokens + next > n_ctx_usize {
                    break;
                }
                chunk_tokens += next;
                chunk_end += 1;
            }

            let chunk = &tokenized[idx..chunk_end];
            // Wave-3 fix #3: reuse the worker's persistent LlamaBatch. It was
            // allocated once at worker startup at the worst-case capacity
            // (n_ctx tokens, seq_max sequences). clear() resets counters in
            // place — no per-chunk allocation.
            batch.clear();
            for (seq_i, tokens) in chunk.iter().enumerate() {
                // CLS pooling reads the embedding of token 0, but
                // `add_sequence(.., logits_all=false)` only flags the LAST
                // token as a logits output. llama.cpp detects the mismatch
                // every call and logs `init: embeddings required but some
                // input tokens were not marked as outputs -> overriding`,
                // then re-marks every token (the slow fix-up path) and
                // allocates a fresh compute buffer per call. Passing `true`
                // here marks all tokens up front — same per-call compute
                // cost as the silent fallback, but no warning, no
                // re-detection, and no allocator churn under sustained
                // concurrent load. See reports/hang_diagnosis.md.
                batch
                    .add_sequence(tokens, seq_i as i32, true)
                    .map_err(|e| {
                        M3Error::Backend(format!("batch add_sequence failed: {e}"))
                    })?;
            }
            // Fresh KV state per chunk; sequences within the chunk are
            // independent so one clear before the decode is enough (no
            // per-text clear thrash).
            ctx.clear_kv_cache();
            // BGE-M3 is non-causal BERT — `encode()` is the documented
            // entry point for embedding contexts. `decode()` works (silent
            // fallback) but logs `decode: cannot decode batches with this
            // context (calling encode() instead)` per call and pays the
            // re-route cost. See reports/hang_diagnosis.md.
            ctx.encode(batch)
                .map_err(|e| M3Error::Backend(format!("llama encode failed: {e}")))?;
            // Wave-3 fix #2: write into a hoisted reusable buffer, then
            // `mem::take` it out into the result vec. The taken Vec carries
            // its allocation away; we reserve a fresh allocation of the same
            // capacity for the next iteration. Replaces the old per-row
            // `Vec::with_capacity(n_embd)` + iter().map().collect() pattern.
            for seq_i in 0..chunk.len() {
                let emb = ctx
                    .embeddings_seq_ith(seq_i as i32)
                    .map_err(|e| M3Error::Backend(format!("read embeddings failed: {e}")))?;
                l2_normalize_into(emb, norm_buf);
                out.push(std::mem::take(norm_buf));
                norm_buf.reserve_exact(n_embd_usize);
            }
            idx = chunk_end;
        }
        Ok(out)
    }

    /// Wave-3 fix #2: in-place L2-normalize from `src` into `dst`. `dst` is
    /// cleared first, then refilled via `extend_from_slice` (one bulk memcpy
    /// reusing the existing capacity), then divided in place. Replaces the
    /// old `l2_normalize(&[f32]) -> Vec<f32>` which allocated a fresh Vec
    /// every call. Zero vector is left unchanged (no divide by zero).
    fn l2_normalize_into(src: &[f32], dst: &mut Vec<f32>) {
        dst.clear();
        dst.extend_from_slice(src);
        let norm: f32 = dst.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm != 0.0 {
            for x in dst.iter_mut() {
                *x /= norm;
            }
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

    /// Fix #27: HttpBackend must surface a timeout error rather than hang
    /// forever when the upstream server accepts but never responds. We spin
    /// up a tiny TCP listener that accept()s and then sleeps; the request
    /// times out at the configured deadline.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_backend_request_times_out() {
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept connections forever and just hold them open.
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                // Pin the socket so the OS keeps the connection open.
                tokio::spawn(async move {
                    let _keep = sock;
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                });
            }
        });

        let url = format!("http://{addr}");
        let backend = HttpBackend::new(url).with_timeout(std::time::Duration::from_millis(150));
        let started = std::time::Instant::now();
        let err = backend
            .run(Batch::new(vec!["x".into()], 1))
            .await
            .expect_err("should time out");
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "should have failed fast on timeout, took {elapsed:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("timed out"),
            "expected timeout error, got: {msg}"
        );
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

    /// Fix #12 invariant: a single `EmbeddedBackend` shared between the
    /// dispatcher (via `Dispatcher::backend()`) and the Python class is the
    /// SAME `Arc<EmbeddedBackend>`, and after a forced load both handles see
    /// the same `Arc<ContextPool>` allocation. Proves no double GGUF load.
    #[cfg(feature = "embedded")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shared_pool_invariant_one_arc_per_backend() {
        use m3_dispatcher::{Dispatcher, DispatcherConfig};
        let Ok(model_path) = std::env::var("M3_TEST_GGUF") else {
            eprintln!("M3_TEST_GGUF unset — skipping shared-pool invariant test");
            return;
        };

        let backend = EmbeddedBackend::with_streams(model_path, 2);
        let cfg = DispatcherConfig { streams: 2, ..Default::default() };
        let dispatcher = Dispatcher::new(cfg, backend);
        // Pull the dispatcher's Arc; this is what `PyEmbeddedEmbedder` keeps.
        let backend_arc = dispatcher.backend().clone();

        // Force the lazy load on one handle.
        let dim = backend_arc.embedding_dim().expect("model should load");
        assert!(dim > 0);

        // Both `embedding_dim()` and the dispatcher's internal Arc must observe
        // the same `Arc<ContextPool>` — i.e. a single underlying allocation.
        let pool_a = backend_arc
            ._pool_handle_for_test()
            .expect("pool should be loaded after embedding_dim()");
        let pool_b = dispatcher
            .backend()
            ._pool_handle_for_test()
            .expect("dispatcher's backend handle should see the same pool");
        assert!(
            std::sync::Arc::ptr_eq(&pool_a, &pool_b),
            "expected one Arc<ContextPool> shared between the dispatcher's backend and the external handle"
        );
        assert_eq!(
            std::sync::Arc::as_ptr(&pool_a) as usize,
            std::sync::Arc::as_ptr(&pool_b) as usize,
            "Arc::as_ptr must match — same ContextPool allocation"
        );
        // And the two backend Arcs themselves must be the same allocation.
        assert!(
            std::sync::Arc::ptr_eq(&backend_arc, dispatcher.backend()),
            "dispatcher.backend() must return the same Arc the caller cloned earlier"
        );
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

    /// Wave 9.1: confirm the compute-cap detector returns without panicking,
    /// whether or not `nvidia-smi` is on PATH. Builds only under embedded-cuda.
    #[cfg(all(feature = "embedded-cuda"))]
    #[test]
    fn detect_cuda_compute_cap_doesnt_panic() {
        let _ = super::embedded::detect_cuda_compute_cap();
    }

    /// Wave 9.1: second call must respect whatever the first call set. Once
    /// GGML_CUDA_DISABLE_GRAPHS is present, the helper must leave it alone.
    #[cfg(all(feature = "embedded-cuda"))]
    #[test]
    fn maybe_disable_cuda_graphs_idempotent() {
        super::embedded::maybe_disable_cuda_graphs_for_blackwell();
        let after_first = std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some();
        super::embedded::maybe_disable_cuda_graphs_for_blackwell();
        let after_second = std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some();
        assert_eq!(after_first, after_second, "second call must not unset");
    }
}
