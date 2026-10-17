//! llama.cpp embedding backends (Phase 3b).
//!
//! `EmbeddedBackend` links llama.cpp in process (llama-cpp-rs, zero IPC); the
//! model math stays in llama.cpp. It is a stub unless the `embedded` cargo
//! feature is enabled. (The HTTP client backend was removed: nothing used it,
//! and it was the core's only TLS client, on `ring`.)

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

/// The backend this build was COMPILED for: `cpu`, `cuda`, `vulkan`, `metal`,
/// or `none` without `embedded`. The single owner of that fact — m3-core-py's
/// `__build_backend__` / `embed_backend_label()` and the embed server's
/// startup line all read it. ⚠ Not what served a request: a GPU build on a
/// host without a usable device runs on the CPU. The observed label is
/// [`EmbeddedBackend::runtime_backend`]. The GPU features each imply
/// `embedded`, so they are tested first.
pub const COMPILED_BACKEND: &str = if cfg!(feature = "embedded-cuda") {
    "cuda"
} else if cfg!(feature = "embedded-vulkan") {
    "vulkan"
} else if cfg!(feature = "embedded-metal") {
    "metal"
} else if cfg!(feature = "embedded") {
    "cpu"
} else {
    "none"
};

/// [`COMPILED_BACKEND`], for callers of the older function form.
#[cfg(feature = "embedded")]
pub fn active_backend() -> &'static str {
    COMPILED_BACKEND
}

mod cuda_env;
pub mod windows;

#[cfg(feature = "embedded")]
pub use cuda_env::prepare_process_env;

use m3_dispatcher::{Batch, BatchOutput, ModelBackend};
use m3_error::{M3Error, Result};

/// Default per-context token window (BGE-M3's training context). Also the
/// `M3_EMBED_CTX` default in m3-core-py.
pub const DEFAULT_N_CTX: u32 = 8192;
/// Default sequences packed per decode (`M3_EMBED_SEQ_MAX` default).
pub const DEFAULT_SEQ_MAX: u32 = 32;
/// Pool size used by the constructors that take no stream count. ⚠ Differs
/// from the dispatcher / `M3_EMBED_STREAMS` default (8); m3-core-py and the
/// embed server always pass the stream count explicitly.
pub const DEFAULT_STREAMS: usize = 4;
/// Smallest context the pool will create; smaller requests are raised (and logged).
pub const MIN_N_CTX: u32 = 64;

/// Embedded (in-process) backend that links llama.cpp directly via
/// `llama-cpp-2` — zero IPC, zero JSON. Gated behind the `embedded` cargo
/// feature, which cmake-builds llama.cpp from source (CPU, or the GPU backend
/// an `embedded-*` feature selects). Without the feature this struct still
/// constructs but `run` returns a "not compiled" error.
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
/// `Send + Sync`, so they are shared via `Arc`. Work is dispatched over a
/// crossbeam MPMC channel — any idle worker picks up the next batch, so `streams`
/// concurrent `run()` calls genuinely decode in parallel on `streams` cores.
pub struct EmbeddedBackend {
    #[allow(dead_code)]
    model_path: String,
    /// Size of the worker-thread context pool. Reaches this generic crate as a
    /// typed parameter — never read from an env var here (that is `m3-core-py`'s
    /// job). Defaults to `DEFAULT_STREAMS` via `new`; use `with_streams` to override.
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
    /// llama.cpp's prompt-processing batch ceiling, floored at `n_ctx` (the
    /// BERT encoder needs the whole decode chunk in one ubatch). Set via
    /// `M3_EMBED_N_BATCH` (read in `m3-core-py`).
    #[allow(dead_code)]
    n_batch: u32,
    /// llama.cpp's micro-batch, floored at `n_ctx` and capped at `n_batch`.
    /// Set via `M3_EMBED_N_UBATCH`.
    #[allow(dead_code)]
    n_ubatch: u32,
    #[cfg(feature = "embedded")]
    state: std::sync::OnceLock<std::sync::Arc<embedded::ContextPool>>,
    /// Serialises the first load: without it, concurrent first calls each
    /// loaded the GGUF and spun up a whole worker pool, and all but one were
    /// thrown away.
    #[cfg(feature = "embedded")]
    load_lock: std::sync::Mutex<()>,
}

/// Tokenizes with a loaded model, independently of the backend that loaded it.
/// Cheap to clone; tokenization never queues behind an embed batch.
#[cfg(feature = "embedded")]
#[derive(Clone)]
pub struct Tokenizer {
    pool: std::sync::Arc<embedded::ContextPool>,
}

#[cfg(feature = "embedded")]
impl Tokenizer {
    /// Exact token counts, BOS/EOS included. See [`EmbeddedBackend::count_tokens`].
    pub fn count_tokens(&self, texts: &[String]) -> Result<Vec<usize>> {
        self.pool.count_tokens(texts)
    }

    /// Token-exact windows (character offsets). See [`windows::plan_windows`].
    pub fn token_windows(
        &self,
        text: &str,
        budget: usize,
        overlap: usize,
    ) -> Result<Vec<(usize, usize)>> {
        self.pool.token_windows(text, budget, overlap)
    }
}

impl EmbeddedBackend {
    /// Construct with `DEFAULT_STREAMS`, `DEFAULT_N_CTX` and `DEFAULT_SEQ_MAX`.
    /// Default n_batch = n_ubatch = n_ctx (the encoder-safe value; see `with_streams_ctx_seqmax`).
    pub fn new(model_path: impl Into<String>) -> Self {
        Self::with_streams_ctx_seqmax(model_path, DEFAULT_STREAMS, DEFAULT_N_CTX, DEFAULT_SEQ_MAX)
    }

    /// Construct with an explicit context-pool size.
    /// Defaults: n_batch = n_ubatch = n_ctx (the encoder-safe value).
    pub fn with_streams(model_path: impl Into<String>, streams: usize) -> Self {
        Self::with_streams_ctx_seqmax(model_path, streams, DEFAULT_N_CTX, DEFAULT_SEQ_MAX)
    }

    /// Construct with explicit stream count and per-context token window.
    /// Defaults: n_batch = n_ubatch = n_ctx (the encoder-safe value).
    pub fn with_streams_and_ctx(model_path: impl Into<String>, streams: usize, n_ctx: u32) -> Self {
        Self::with_streams_ctx_seqmax(model_path, streams, n_ctx, DEFAULT_SEQ_MAX)
    }

    /// Construct with explicit stream count, per-context token window, and max sequences per decode.
    /// `n_ctx` is the total token budget per decode call (shared across all sequences in a chunk).
    /// `seq_max` caps how many sequences are packed into one decode call per worker thread.
    /// Defaults n_batch = n_ubatch = n_ctx (encoder-safe; see body for why).
    pub fn with_streams_ctx_seqmax(
        model_path: impl Into<String>,
        streams: usize,
        n_ctx: u32,
        seq_max: u32,
    ) -> Self {
        // n_batch / n_ubatch default to n_ctx. llama.cpp's BERT encoder asserts
        // `n_ubatch >= n_tokens` for the WHOLE batch it is handed, and a chunk in
        // `embed_on_ctx` can pack up to `n_ctx` tokens (across `seq_max` seqs, or
        // one solo long text). An earlier default of `n_ubatch = 512`
        // therefore crashed the encoder — `GGML_ASSERT(cparams.n_ubatch >=
        // n_tokens)` — on any text/chunk over 512 tokens, which for BGE-M3
        // (8192-token trained ctx) is routine. `n_ctx` is the simplest default
        // safe for every chunk the worker can build (a tighter per-decode bound
        // exists but would couple this constructor to `embed_on_ctx`'s chunker).
        let n_batch = n_ctx;
        let n_ubatch = n_ctx;
        Self::with_streams_ctx_seqmax_batch(model_path, streams, n_ctx, seq_max, n_batch, n_ubatch)
    }

    /// Full-control constructor. All five tunables explicit:
    /// pool size, context budget, max sequences/decode, n_batch (prompt-process
    /// batch ceiling), and n_ubatch (micro-batch / SIMD tile). Existing
    /// constructors above are thin shims that pass `n_batch = n_ubatch = n_ctx`.
    ///
    /// Note: `n_batch` and `n_ubatch` are raised to a floor of `n_ctx` here.
    /// `embed_on_ctx` bounds a decode chunk at `n_ctx` tokens and the BERT
    /// encoder asserts `n_ubatch >= n_tokens` over the whole batch, so `n_ctx`
    /// is the simplest value guaranteed safe for every chunk the worker can
    /// build. A smaller value would crash at `encode()` time; rather than let
    /// that happen the constructor floors it AND logs a `log::warn!` so a
    /// deliberate small setting isn't silently discarded.
    pub fn with_streams_ctx_seqmax_batch(
        model_path: impl Into<String>,
        streams: usize,
        n_ctx: u32,
        seq_max: u32,
        n_batch: u32,
        n_ubatch: u32,
    ) -> Self {
        if n_ctx < MIN_N_CTX {
            log::warn!(
                "m3-embed: n_ctx {n_ctx} is below the minimum {MIN_N_CTX}; using {MIN_N_CTX}"
            );
        }
        let n_ctx = n_ctx.max(MIN_N_CTX);
        // Floor n_batch / n_ubatch at n_ctx (see doc comment for the encoder
        // assert). n_batch must also stay >= n_ubatch. A value above n_ctx is
        // harmless — just a larger prompt-process tile — so it passes through.
        let (n_batch, n_ubatch) = {
            let req_batch = n_batch;
            let req_ubatch = n_ubatch;
            let n_batch = n_batch.max(n_ctx);
            let n_ubatch = n_ubatch.max(n_ctx).min(n_batch);
            if req_batch < n_batch || req_ubatch < n_ubatch {
                log::warn!(
                    "m3-embed: n_batch/n_ubatch ({req_batch}/{req_ubatch}) below \
                     n_ctx ({n_ctx}); raised to {n_batch}/{n_ubatch} — the BERT \
                     encoder requires n_ubatch >= the largest decode chunk"
                );
            } else if req_ubatch > n_ubatch {
                log::warn!(
                    "m3-embed: n_ubatch {req_ubatch} exceeds n_batch {n_batch}; lowered to \
                     {n_ubatch} (llama.cpp requires n_ubatch <= n_batch)"
                );
            }
            (n_batch, n_ubatch)
        };
        if streams == 0 || seq_max == 0 {
            log::warn!("m3-embed: streams={streams} / seq_max={seq_max}: zero is raised to 1");
        }
        Self {
            model_path: model_path.into(),
            streams: streams.max(1),
            n_ctx,
            seq_max: seq_max.max(1),
            n_batch,
            n_ubatch,
            #[cfg(feature = "embedded")]
            state: std::sync::OnceLock::new(),
            #[cfg(feature = "embedded")]
            load_lock: std::sync::Mutex::new(()),
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

    /// Exact token counts for `texts`, one per input, in input order.
    ///
    /// The counts include BOS/EOS, matching what the embed path measures
    /// against `n_ctx`. Forces the GGUF load on first call (like
    /// `embedding_dim`) but performs no decode — tokenization needs only the
    /// model, so this never queues behind an in-flight embed batch.
    #[cfg(feature = "embedded")]
    pub fn count_tokens(&self, texts: &[String]) -> Result<Vec<usize>> {
        let pool = self.pool()?;
        pool.count_tokens(texts)
    }

    /// Token-exact windows over `text` (character offsets). See
    /// [`windows::plan_windows`]. Forces the GGUF load, performs no decode.
    #[cfg(feature = "embedded")]
    pub fn token_windows(
        &self,
        text: &str,
        budget: usize,
        overlap: usize,
    ) -> Result<Vec<(usize, usize)>> {
        self.pool()?.token_windows(text, budget, overlap)
    }

    /// A handle that tokenizes with this backend's model after the backend
    /// itself has been moved into a dispatcher (the embed server keeps one for
    /// its `/tokenize/*` routes). Forces the GGUF load.
    #[cfg(feature = "embedded")]
    pub fn tokenizer(&self) -> Result<Tokenizer> {
        Ok(Tokenizer { pool: self.pool()? })
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

    /// The compute backend that actually holds the model's offloaded layers,
    /// as observed after load (`cuda`, `vulkan`, `metal`, or `cpu`) — unlike
    /// [`active_backend`], which is only what the build was compiled for.
    /// Forces the GGUF load.
    #[cfg(feature = "embedded")]
    pub fn runtime_backend(&self) -> Result<String> {
        Ok(self.pool()?.runtime_backend().to_string())
    }

    /// Get-or-load the cached worker-thread `ContextPool`: exactly one load per
    /// backend, however many callers race for the first one. A failed load is
    /// not cached, so the next call retries it.
    #[cfg(feature = "embedded")]
    fn pool(&self) -> Result<std::sync::Arc<embedded::ContextPool>> {
        if let Some(p) = self.state.get() {
            return Ok(p.clone());
        }
        // A poisoned lock only means an earlier loader panicked; the OnceLock
        // is still consistent, so proceed.
        let _g = self.load_lock.lock().unwrap_or_else(|p| p.into_inner());
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
        Ok(self.state.get_or_init(|| loaded).clone())
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
    use llama_cpp_2::token::LlamaToken;
    use llama_cpp_2::{list_llama_ggml_backend_devices, LlamaBackendDeviceType};
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
            // Before init, so ggml-cuda sees GGML_CUDA_DISABLE_GRAPHS. Hosts
            // call this earlier still; it runs once (see cuda_env).
            crate::cuda_env::prepare_process_env();
            LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))
        });
        match slot {
            Ok(b) => Ok(b),
            Err(e) => Err(M3Error::Backend(e.clone())),
        }
    }

    /// The ONE tokenizer call: counting, windowing, warmup and embedding all
    /// go through it, so a count always matches what the embed path measures
    /// against `n_ctx` (BOS/EOS included — `AddBos::Always`).
    fn tokenize(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>> {
        model.str_to_token(text, AddBos::Always).map_err(|e| {
            M3Error::Backend(format!(
                "tokenize failed for a {}-byte text: {e}",
                text.len()
            ))
        })
    }

    /// Text of a panic payload (`&str` or `String`; anything else is opaque).
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
        if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "(non-string panic payload)".to_string()
        }
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
        /// Crossbeam MPMC sender, in a slot so Drop can take it: dropping the
        /// last sender makes every worker's `recv()` return `Err`, which ends
        /// its loop.
        tx: Mutex<Option<Sender<Job>>>,
        workers: Mutex<Vec<JoinHandle<()>>>,
        n_embd: i32,
        streams: usize,
        /// Per-context token window; `token_windows` validates budgets against it.
        n_ctx: u32,
        /// Observed after load: the backend of the device holding the
        /// offloaded layers, or `cpu`.
        runtime_backend: String,
        /// The loaded GGUF, retained so `count_tokens` can tokenize WITHOUT
        /// going through a worker.
        ///
        /// `str_to_token` is a `&LlamaModel` method — it needs no
        /// `LlamaContext`, no KV cache and no decode. Only the CONTEXT is
        /// `!Send` and lifetime-bound to its thread; the model itself is
        /// already shared across every worker (each gets an `Arc` clone at
        /// spawn), so holding one more clone here adds no synchronisation and
        /// keeps token counting off the embed job queue entirely. Counting
        /// therefore never waits behind an in-flight batch, which matters
        /// because callers count in order to DECIDE how to batch.
        model: Arc<LlamaModel>,
    }

    /// Which device took the model, as ggml reports it after load. Logged at
    /// info; a GPU build that ended up on the CPU is logged at WARN, because
    /// that is a 10-50x slowdown nobody asked for.
    fn observe_device(model: &LlamaModel, params: &LlamaModelParams, model_path: &str) -> String {
        let gpus: Vec<_> = list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|d| {
                matches!(
                    d.device_type,
                    LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
                )
            })
            .collect();
        let requested = params.n_gpu_layers();
        let n_layer = model.n_layer();
        // llama.cpp reads a negative or >= n_layer request as "all layers".
        let offload = if requested < 0 {
            n_layer
        } else {
            (requested as u32).min(n_layer)
        };
        let compiled = crate::active_backend();
        match gpus.first() {
            Some(d) if offload > 0 => {
                log::info!(
                    "m3-embed: loaded '{model_path}' on {} device {} ({}, {} MiB free of {} MiB); \
                     {offload} of {n_layer} layers requested for offload (n_gpu_layers={requested}; \
                     the count is inferred from the request, llama.cpp does not report it)",
                    d.backend,
                    d.name,
                    d.description,
                    d.memory_free / (1 << 20),
                    d.memory_total / (1 << 20),
                );
                d.backend.to_ascii_lowercase()
            }
            _ => {
                if compiled == "cpu" {
                    log::info!("m3-embed: loaded '{model_path}' on the CPU ({n_layer} layers)");
                } else {
                    log::warn!(
                        "m3-embed: this is a {compiled} build but '{model_path}' runs on the CPU: \
                         ggml reported {} GPU device(s) and n_gpu_layers={requested}. Check the GPU \
                         driver / runtime libraries (e.g. CUDA_PATH, the Vulkan loader); embedding \
                         will be much slower",
                        gpus.len()
                    );
                }
                "cpu".to_string()
            }
        }
    }

    impl ContextPool {
        /// Load the GGUF model once and spin up `streams` worker threads, each
        /// with its own `LlamaContext`. `n_ctx` is the total token budget per
        /// decode call (shared across all sequences in one chunk). `seq_max` is
        /// the max sequences packed into one `LlamaBatch` decode per worker.
        /// `n_batch` / `n_ubatch` are the llama.cpp prompt-process batch and
        /// micro-batch ceilings.
        pub fn load(
            model_path: &str,
            streams: usize,
            n_ctx: u32,
            seq_max: u32,
            n_batch: u32,
            n_ubatch: u32,
        ) -> Result<Self> {
            let streams = streams.max(1);
            let n_ctx = n_ctx.max(crate::MIN_N_CTX);
            if !std::path::Path::new(model_path).is_file() {
                return Err(M3Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("GGUF model '{model_path}' does not exist or is not a file"),
                )));
            }
            // Shared, process-global backend (see `BACKEND` above).
            let backend = shared_backend()?;
            let params = LlamaModelParams::default();
            let model = Arc::new(
                LlamaModel::load_from_file(backend, model_path, &params).map_err(|e| {
                    M3Error::Backend(format!("failed to load gguf model '{model_path}': {e}"))
                })?,
            );
            let n_embd = model.n_embd();
            let runtime_backend = observe_device(&model, &params, model_path);

            // MPMC: each worker holds its own Receiver clone and `recv()` is
            // lock-free — no `Mutex<Receiver>` hot lock as streams scales.
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
                        worker_loop(
                            id, backend, model, rx, ready_tx, n_ctx, seq_max, n_batch, n_ubatch,
                        )
                    })
                    .map_err(|e| {
                        M3Error::Backend(format!(
                            "failed to spawn embed worker {id} of {streams}: {e}"
                        ))
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
                    Err(_) => return Err(dead_before_ready(&mut workers, streams)),
                }
            }

            Ok(Self {
                tx: Mutex::new(Some(tx)),
                workers: Mutex::new(workers),
                n_embd,
                streams,
                n_ctx,
                runtime_backend,
                model,
            })
        }

        /// Exact token counts, one per input, in input order.
        ///
        /// Uses the same `tokenize` the embed path uses, so a count returned
        /// here is exactly what `embed_on_ctx` will measure against `n_ctx` —
        /// a caller that chunks to a budget below `n_ctx` cannot then be
        /// surprised by an overflow. An estimate cannot offer that guarantee.
        /// Counts INCLUDE the BOS/EOS frame, which occupies `n_ctx` like any
        /// other token.
        pub fn count_tokens(&self, texts: &[String]) -> Result<Vec<usize>> {
            texts
                .iter()
                .map(|t| tokenize(&self.model, t).map(|v| v.len()))
                .collect()
        }

        /// Token-exact windows over `text`: `[(start, end), ...]` in CHARACTER
        /// offsets, each holding at most `budget` tokens (BOS/EOS included, as
        /// `count_tokens` counts them) and overlapping by about `overlap`. See
        /// [`crate::windows::plan_windows`].
        ///
        /// `budget` must exceed the BOS/EOS frame (else every window would be
        /// one character) and must not exceed `n_ctx` (else a window cannot
        /// be embedded): either is a caller error, reported as such.
        pub fn token_windows(
            &self,
            text: &str,
            budget: usize,
            overlap: usize,
        ) -> Result<Vec<(usize, usize)>> {
            let frame = tokenize(&self.model, "")?.len();
            if budget <= frame || budget > self.n_ctx as usize {
                return Err(M3Error::Config(format!(
                    "token_windows budget {budget} is out of range: it must exceed the {frame}-token \
                     BOS/EOS frame and be at most n_ctx {} (M3_EMBED_CTX)",
                    self.n_ctx
                )));
            }
            // Byte offset of every character boundary, so a character slice
            // maps onto `text` without re-encoding.
            let bounds: Vec<usize> = text
                .char_indices()
                .map(|(b, _)| b)
                .chain(std::iter::once(text.len()))
                .collect();
            let n_chars = bounds.len() - 1;
            let mut count = |s: usize, e: usize| -> Result<usize> {
                tokenize(&self.model, &text[bounds[s]..bounds[e]]).map(|t| t.len())
            };
            crate::windows::plan_windows(n_chars, budget, overlap, &mut count)
        }

        /// Embedding dimension reported by the model.
        pub fn n_embd(&self) -> i32 {
            self.n_embd
        }

        /// The compute backend observed at load (`cuda`, `vulkan`, `metal`, `cpu`).
        pub fn runtime_backend(&self) -> &str {
            &self.runtime_backend
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
                    .send(Job {
                        texts,
                        reply: reply_tx,
                    })
                    .map_err(|_| {
                        M3Error::Backend(format!(
                        "embed pool has no live workers (all {} exited; see the earlier worker \
                         panic logs) — recreate the embedder",
                        self.streams
                    ))
                    })?;
            }
            // A worker that panics replies with its id and the panic message
            // before exiting (worker_loop), so this only fires if the reply
            // itself could not be sent.
            reply_rx.recv().map_err(|_| {
                M3Error::Backend("embed worker exited without replying to the job".into())
            })?
        }
    }

    /// The error for a worker that died before reporting readiness: names the
    /// worker and its panic message, found by joining the finished threads.
    fn dead_before_ready(workers: &mut Vec<JoinHandle<()>>, streams: usize) -> M3Error {
        let mut why = Vec::new();
        for (id, h) in workers.drain(..).enumerate() {
            if h.is_finished() {
                if let Err(p) = h.join() {
                    why.push(format!("worker {id} panicked: {}", panic_message(&*p)));
                }
            }
        }
        M3Error::Backend(format!(
            "an embed worker (of {streams}) died before reporting readiness: {}",
            if why.is_empty() {
                "no panic was recorded".to_string()
            } else {
                why.join("; ")
            }
        ))
    }

    impl Drop for ContextPool {
        fn drop(&mut self) {
            // Drop the sender so every worker's `rx.recv()` returns `Err` and
            // the threads exit their loops, then join them.
            if let Ok(mut g) = self.tx.lock() {
                *g = None;
            }
            if let Ok(mut workers) = self.workers.lock() {
                for (id, h) in workers.drain(..).enumerate() {
                    if let Err(p) = h.join() {
                        log::error!("m3-embed: worker {id} had panicked: {}", panic_message(&*p));
                    }
                }
            }
        }
    }

    /// Body of one worker thread: create a context, then serve jobs until the
    /// channel closes. The `LlamaContext` is created here and never leaves this
    /// thread — that is what makes a `!Send` context safe to "pool".
    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        id: usize,
        backend: &'static LlamaBackend,
        model: Arc<LlamaModel>,
        rx: Receiver<Job>,
        ready_tx: stdmpsc::Sender<Result<()>>,
        n_ctx: u32,
        seq_max: u32,
        n_batch: u32,
        n_ubatch: u32,
    ) {
        // n_batch / n_ubatch reach here already floored to >= n_ctx by the
        // constructor — the BERT encoder asserts `n_ubatch >= n_tokens` and a
        // decode chunk can be as large as n_ctx, so they cannot be smaller.
        // Flash attention is turned on explicitly: it cuts attention memory
        // bandwidth ~3x on long inputs, and an explicit setter keeps the
        // per-context log line unambiguous against any future default flip.
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
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED);
        let mut ctx = match model.new_context(backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                // The usual cause is the KV cache not fitting: every worker
                // allocates its own n_ctx-sized context, so name the knobs.
                let _ = ready_tx.send(Err(M3Error::Backend(format!(
                    "llama context create failed for worker {id} \
                     (n_ctx={n_ctx}, seq_max={seq_max}, n_batch={n_batch}, \
                     n_ubatch={n_ubatch}): {e}. If memory ran out, lower \
                     M3_EMBED_STREAMS or M3_EMBED_CTX"
                ))));
                return;
            }
        };

        // Warmup decode. The first real decode call pays the
        // kernel-cache + KV-allocator + (on GPU) shader-compile cost — easily
        // 1-5s of first-token latency. Run a single trivial decode + embedding
        // read here so the user-visible cold call only pays the GGUF-load +
        // worker-spawn cost, not the per-context first-decode cost.
        let warm_tokens = match tokenize(&model, "x") {
            Ok(t) => t,
            Err(e) => {
                let _ = ready_tx.send(Err(M3Error::Backend(format!("worker {id} warmup: {e}"))));
                return;
            }
        };
        let mut warm_batch = LlamaBatch::new(warm_tokens.len().max(1), 1);
        // Match the hot path: CLS pooling needs all tokens flagged for
        // logits, and BGE-M3 is non-causal so encode() is the right entry.
        // See embed_on_ctx for the full rationale.
        if let Err(e) = warm_batch.add_sequence(&warm_tokens, 0, true) {
            let _ = ready_tx.send(Err(M3Error::Backend(format!(
                "worker {id} warmup add_sequence failed: {e}"
            ))));
            return;
        }
        ctx.clear_kv_cache();
        if let Err(e) = ctx.encode(&mut warm_batch) {
            let _ = ready_tx.send(Err(M3Error::Backend(format!(
                "worker {id} warmup encode failed: {e}"
            ))));
            return;
        }
        if let Err(e) = ctx.embeddings_seq_ith(0) {
            let _ = ready_tx.send(Err(M3Error::Backend(format!(
                "worker {id} warmup embeddings read failed: {e}"
            ))));
            return;
        }
        // Discard warmup result; KV cache will be cleared again per real chunk.

        let _ = ready_tx.send(Ok(()));
        drop(ready_tx);

        // Keep one LlamaBatch per worker, sized once at the
        // worst case the worker will ever face (`n_ctx` total tokens across
        // `seq_max` sequences). `LlamaBatch::clear()` exists in 0.1.146 and is
        // cheap (just resets counters + zeroes the pos/seq tables in place),
        // so reusing the allocation across every chunk eliminates a fresh
        // `LlamaBatch::new` per decode.
        let mut batch = LlamaBatch::new(n_ctx as usize, seq_max as i32);

        // Hoist the L2-normalize destination buffer out of the
        // chunk loop too; reused across every sequence in every chunk.
        let mut norm_buf: Vec<f32> = Vec::with_capacity(model.n_embd() as usize);

        loop {
            // Lock-free recv: crossbeam's Receiver is Sync.
            let job = match rx.recv() {
                Ok(j) => j,
                Err(_) => return, // all senders dropped — pool shutting down
            };
            // A panic inside llama.cpp's Rust wrapper would otherwise drop the
            // reply channel and leave the caller an anonymous "dropped the
            // job". Answer with this worker's id and the panic message, then
            // exit: a context that panicked mid-decode is not trusted again.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                embed_on_ctx(
                    &model,
                    &mut ctx,
                    &job.texts,
                    n_ctx,
                    seq_max,
                    &mut batch,
                    &mut norm_buf,
                )
            }));
            match result {
                Ok(r) => {
                    let _ = job.reply.send(r);
                }
                Err(p) => {
                    let msg = panic_message(&*p);
                    log::error!(
                        "m3-embed: worker {id} panicked on a {}-text batch and exited: {msg}",
                        job.texts.len()
                    );
                    let _ = job.reply.send(Err(M3Error::Backend(format!(
                        "embed worker {id} panicked and exited: {msg}"
                    ))));
                    return;
                }
            }
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
        for (index, text) in texts.iter().enumerate() {
            let tokens = tokenize(model, text)?;
            if tokens.len() > n_ctx_usize {
                // Typed, not Backend(String): the HTTP layer maps this to 413
                // while real backend faults stay 500. Counts travel as fields
                // so no caller has to parse the message. Display still renders
                // the original wording verbatim.
                return Err(M3Error::InputTooLong {
                    tokens: tokens.len(),
                    n_ctx: n_ctx as usize,
                    index: Some(index),
                });
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
            // Reuse the worker's persistent LlamaBatch. It was
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
                // concurrent load.
                batch
                    .add_sequence(tokens, seq_i as i32, true)
                    .map_err(|e| M3Error::Backend(format!("batch add_sequence failed: {e}")))?;
            }
            // Fresh KV state per chunk; sequences within the chunk are
            // independent so one clear before the decode is enough (no
            // per-text clear thrash).
            ctx.clear_kv_cache();
            // BGE-M3 is non-causal BERT — `encode()` is the documented
            // entry point for embedding contexts. `decode()` works (silent
            // fallback) but logs `decode: cannot decode batches with this
            // context (calling encode() instead)` per call and pays the
            // re-route cost.
            ctx.encode(batch)
                .map_err(|e| M3Error::Backend(format!("llama encode failed: {e}")))?;
            // Write into a hoisted reusable buffer, then
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

    /// In-place L2-normalize from `src` into `dst`. `dst` is
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

    #[cfg(not(feature = "embedded"))]
    #[tokio::test]
    async fn embedded_backend_is_stub_by_default() {
        let b = EmbeddedBackend::new("models/bge-m3-q8_0.gguf");
        let err = b.run(Batch::new(vec!["x".into()], 1)).await.unwrap_err();
        assert!(format!("{err}").contains("embedded backend not compiled"));
    }

    /// End-to-end inference against a real GGUF model. Opt-in: set
    /// `M3_TEST_GGUF` and run with `--ignored`. Ignored (reported as such),
    /// never silently passed, when no model is available.
    #[cfg(feature = "embedded")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "needs M3_TEST_GGUF (a GGUF embedding model path)"]
    async fn embedded_backend_runs_real_inference() {
        let model_path =
            std::env::var("M3_TEST_GGUF").expect("set M3_TEST_GGUF to a GGUF embedding model");
        // 3-stream pool exercises the worker-thread context pool (CHANGE 1).
        let b = EmbeddedBackend::with_streams(model_path, 3);
        let out = b
            .run(Batch::new(
                vec!["hello world".into(), "a different sentence".into()],
                4,
            ))
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
    #[ignore = "needs M3_TEST_GGUF (a GGUF embedding model path)"]
    async fn shared_pool_invariant_one_arc_per_backend() {
        use m3_dispatcher::{Dispatcher, DispatcherConfig};
        let model_path =
            std::env::var("M3_TEST_GGUF").expect("set M3_TEST_GGUF to a GGUF embedding model");

        let backend = EmbeddedBackend::with_streams(model_path, 2);
        let cfg = DispatcherConfig {
            streams: 2,
            ..Default::default()
        };
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
    #[ignore = "needs M3_TEST_GGUF (a GGUF embedding model path)"]
    async fn embedded_backend_concurrent_streams() {
        let model_path =
            std::env::var("M3_TEST_GGUF").expect("set M3_TEST_GGUF to a GGUF embedding model");
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

    /// Real-tokenizer windows: every window fits by an exact count of its own
    /// slice, the text is covered, offsets survive multi-byte characters, and
    /// prose needs far fewer windows than the estimator's (`max(bytes/3,
    /// chars) + 2`) plan. Opt-in via `M3_TEST_GGUF`.
    #[cfg(feature = "embedded")]
    #[test]
    #[ignore = "needs M3_TEST_GGUF (a GGUF embedding model path)"]
    fn embedded_token_windows_fit_and_beat_the_estimate() {
        let model_path =
            std::env::var("M3_TEST_GGUF").expect("set M3_TEST_GGUF to a GGUF embedding model");
        let b = EmbeddedBackend::new(model_path);
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!(
                "Memory {i} records that the embedder moved to a free port. "
            ));
            if i % 97 == 0 {
                text.push_str("記憶は共有コンピュータで別のポートを使う 🚀 ");
            }
        }
        let chars: Vec<char> = text.chars().collect();
        let (budget, overlap) = (512usize, 64usize);
        // Budgets the pool cannot honour are caller errors, not degenerate plans.
        assert!(
            b.token_windows(&text, 2, 0).is_err(),
            "budget inside the BOS/EOS frame"
        );
        assert!(
            b.token_windows(&text, 1_000_000, 0).is_err(),
            "budget above n_ctx"
        );
        let w = b.token_windows(&text, budget, overlap).expect("windows");
        assert_eq!(w.first().unwrap().0, 0);
        assert_eq!(w.last().unwrap().1, chars.len());
        let slices: Vec<String> = w
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect())
            .collect();
        let counts = b.count_tokens(&slices).expect("counts");
        for (i, c) in counts.iter().enumerate() {
            assert!(
                *c <= budget,
                "window {i} {:?} has {c} tokens > {budget}",
                w[i]
            );
        }
        let estimate = |s: usize, e: usize| -> Result<usize> {
            let slice: String = chars[s..e].iter().collect();
            Ok((slice.len() / 3).max(e - s) + 2)
        };
        let est =
            crate::windows::plan_windows(chars.len(), budget, overlap, &mut { estimate }).unwrap();
        eprintln!(
            "{} chars: {} exact windows vs {} estimated",
            chars.len(),
            w.len(),
            est.len()
        );
        assert!(
            w.len() * 2 < est.len(),
            "exact {} vs estimated {}",
            w.len(),
            est.len()
        );
    }

    /// The compute-cap probe returns (a list or a cause) whether or not
    /// `nvidia-smi` is on PATH. Builds only under embedded-cuda.
    #[cfg(feature = "embedded-cuda")]
    #[test]
    fn detect_cuda_compute_caps_returns_a_result() {
        match crate::cuda_env::detect_cuda_compute_caps() {
            Ok(gpus) => assert!(!gpus.is_empty()),
            Err(cause) => assert!(!cause.is_empty()),
        }
    }

    /// Preparing the environment twice must not change what the first call set.
    #[cfg(feature = "embedded-cuda")]
    #[test]
    fn prepare_process_env_idempotent() {
        crate::prepare_process_env();
        let after_first = std::env::var_os("GGML_CUDA_DISABLE_GRAPHS");
        crate::prepare_process_env();
        assert_eq!(
            after_first,
            std::env::var_os("GGML_CUDA_DISABLE_GRAPHS"),
            "second call must not change it"
        );
    }
}
