//! PyO3 bindings for m3-core-rs.
//!
//! The only crate Python sees. This is also the ONLY crate that reads `M3_*`
//! env vars (per plan §9.6) — it translates them into the typed config structs
//! the generic crates consume. Generic crates never call `std::env::var` and
//! never depend on pyo3.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub use m3_dispatcher;
pub use m3_embed_llamacpp;
pub use m3_error;
pub use m3_graph;
pub use m3_hash;
pub use m3_ner_ort;
pub use m3_rank;
pub use m3_redact;
pub use m3_route;
pub use m3_vector;

mod config;

use m3_error::M3Error;

/// Newtype so we can own the `M3Error` -> `PyErr` conversion (orphan rule).
struct PyM3Error(M3Error);

impl From<M3Error> for PyM3Error {
    fn from(e: M3Error) -> Self {
        PyM3Error(e)
    }
}

impl From<PyM3Error> for PyErr {
    fn from(e: PyM3Error) -> PyErr {
        match e.0 {
            M3Error::VectorDimMismatch { .. } | M3Error::Config(_) => {
                PyValueError::new_err(e.0.to_string())
            }
            M3Error::DatabaseLocked | M3Error::Io(_) | M3Error::Backend(_) => {
                PyOSError::new_err(e.0.to_string())
            }
            M3Error::Parity { .. } | M3Error::Other(_) => PyValueError::new_err(e.0.to_string()),
        }
    }
}

/// `m3_error::Result` -> `PyResult`, routed through the `PyM3Error` mapping.
fn map_err<T>(r: m3_error::Result<T>) -> PyResult<T> {
    r.map_err(|e| PyM3Error::from(e).into())
}

// ---------------------------------------------------------------------------
// m3-hash
// ---------------------------------------------------------------------------

/// SHA-256 hex digest of `text` — parity with Python `hashlib.sha256().hexdigest()`.
#[pyfunction]
fn sha256_hex(text: &str) -> String {
    m3_hash::sha256_hex(text.as_bytes())
}

/// SHA-256 hex digest of raw `bytes` — byte-exact parity with
/// `hashlib.sha256(data).hexdigest()` for arbitrary (non-UTF-8) input.
#[pyfunction]
fn sha256_hex_bytes(data: &[u8]) -> String {
    m3_hash::sha256_hex(data)
}

/// Active hash provider string; surfaced in `m3:health` for FIPS drift detection.
#[pyfunction]
fn hash_provider() -> &'static str {
    m3_hash::active_provider()
}

// ---------------------------------------------------------------------------
// m3-vector
// ---------------------------------------------------------------------------

#[pyfunction]
fn cosine(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    map_err(m3_vector::cosine(&a, &b))
}

#[pyfunction]
fn cosine_batch(query: Vec<f32>, corpus: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
    map_err(m3_vector::cosine_batch(&query, &refs))
}

#[pyfunction]
fn mmr_rerank(
    query: Vec<f32>,
    candidates: Vec<Vec<f32>>,
    lambda: f32,
    k: usize,
) -> PyResult<Vec<usize>> {
    let refs: Vec<&[f32]> = candidates.iter().map(|v| v.as_slice()).collect();
    map_err(m3_vector::mmr_rerank(&query, &refs, lambda, k))
}

#[pyfunction]
fn mmr_rerank_scored(
    relevance: Vec<f32>,
    candidate_vectors: Vec<Vec<f32>>,
    lambda: f32,
    k: usize,
    force_seed_first: bool,
) -> PyResult<Vec<usize>> {
    let refs: Vec<&[f32]> = candidate_vectors.iter().map(|v| v.as_slice()).collect();
    map_err(m3_vector::mmr_rerank_scored(
        &relevance,
        &refs,
        lambda,
        k,
        force_seed_first,
    ))
}

/// Expansion-displacement guard. Takes `(score, is_expansion)` tuples in
/// current ranked order, returns the index permutation that reorders them —
/// indices, not the rows themselves, so callers can map back to their own row
/// objects even when `(score, is_expansion)` pairs collide.
#[pyfunction]
fn enforce_displacement_guard(
    items: Vec<(f32, bool)>,
    protected_ranks: usize,
    margin: f32,
) -> Vec<usize> {
    let rows: Vec<m3_vector::DisplacementRow> = items
        .iter()
        .map(|&(score, is_expansion)| m3_vector::DisplacementRow { score, is_expansion })
        .collect();
    m3_vector::displacement_permutation(&rows, protected_ranks, margin)
}

#[pyfunction]
fn blob_as_f32(blob: Vec<u8>) -> PyResult<Vec<f32>> {
    map_err(m3_vector::blob_as_f32(&blob)).map(|s| s.to_vec())
}

#[pyfunction]
fn f32_as_blob(vec: Vec<f32>) -> Vec<u8> {
    m3_vector::f32_as_blob(&vec).to_vec()
}

// ---------------------------------------------------------------------------
// m3-redact
// ---------------------------------------------------------------------------

/// Last `scrub()` call's custom_regex compile errors. Mirrors the Python
/// `chatlog_redaction.get_compile_errors()` contract. Thread-local so a
/// caller reads back its own call's errors.
use std::cell::RefCell;
thread_local! {
    static REDACTION_COMPILE_ERRORS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Build a `RedactionConfig` from the `redaction` config dict, exactly as
/// `chatlog_redaction.py`'s `scrub` reads it (keys: enabled, patterns,
/// custom_regex, redact_pii). Missing keys take Python's `.get()` defaults.
fn redaction_config_from_dict(config: &Bound<'_, PyDict>) -> PyResult<m3_redact::RedactionConfig> {
    let enabled = match config.get_item("enabled")? {
        Some(v) => v.extract::<bool>()?,
        None => false,
    };
    let patterns = match config.get_item("patterns")? {
        Some(v) => v.extract::<Vec<String>>()?,
        None => Vec::new(),
    };
    let custom_regex = match config.get_item("custom_regex")? {
        Some(v) => v.extract::<Vec<String>>()?,
        None => Vec::new(),
    };
    let redact_pii = match config.get_item("redact_pii")? {
        Some(v) => v.extract::<bool>()?,
        None => false,
    };
    Ok(m3_redact::RedactionConfig {
        enabled,
        patterns,
        custom_regex,
        redact_pii,
    })
}

/// Byte-exact port of `chatlog_redaction.scrub`. Takes the `redaction` config
/// dict, returns `(scrubbed_content, match_count, groups_fired)`. custom_regex
/// compile errors are stashed; read them with `redaction_compile_errors()`.
#[pyfunction]
fn scrub(content: &str, config: &Bound<'_, PyDict>) -> PyResult<(String, usize, Vec<String>)> {
    let cfg = redaction_config_from_dict(config)?;
    let result = m3_redact::scrub(content, &cfg);
    REDACTION_COMPILE_ERRORS.with(|e| {
        *e.borrow_mut() = result.compile_errors.clone();
    });
    Ok((result.content, result.match_count, result.groups_fired))
}

/// custom_regex compilation errors from the most recent `scrub()` call on
/// this thread. Parity with `chatlog_redaction.get_compile_errors()`.
#[pyfunction]
fn redaction_compile_errors() -> Vec<String> {
    REDACTION_COMPILE_ERRORS.with(|e| e.borrow().clone())
}

// ---------------------------------------------------------------------------
// m3-rank
// ---------------------------------------------------------------------------

/// One ranked candidate from either the FTS5 or vector result set.
/// `source` is the string `"fts"` or `"vector"`.
#[pyclass(name = "RankRow", from_py_object)]
#[derive(Clone)]
struct PyRankRow {
    #[pyo3(get, set)]
    id: String,
    #[pyo3(get, set)]
    score: f32,
    #[pyo3(get, set)]
    source: String,
}

#[pymethods]
impl PyRankRow {
    #[new]
    fn new(id: String, score: f32, source: String) -> Self {
        PyRankRow { id, score, source }
    }
    fn __repr__(&self) -> String {
        format!("RankRow(id={:?}, score={}, source={:?})", self.id, self.score, self.source)
    }
}

fn source_to_str(s: m3_rank::RankSource) -> String {
    match s {
        m3_rank::RankSource::Fts => "fts".to_string(),
        m3_rank::RankSource::Vector => "vector".to_string(),
    }
}

fn source_from_str(s: &str) -> m3_rank::RankSource {
    match s.to_ascii_lowercase().as_str() {
        "vector" => m3_rank::RankSource::Vector,
        _ => m3_rank::RankSource::Fts,
    }
}

fn rankrow_in(r: &PyRankRow) -> m3_rank::RankRow {
    m3_rank::RankRow { id: r.id.clone(), score: r.score, source: source_from_str(&r.source) }
}

fn rankrow_out(r: m3_rank::RankRow) -> PyRankRow {
    PyRankRow { id: r.id, score: r.score, source: source_to_str(r.source) }
}

/// Fuse FTS5 and vector result sets with per-source weights (default 0.5/0.5).
#[pyfunction]
#[pyo3(signature = (fts, vector, fts_weight=0.5, vector_weight=0.5))]
fn fuse(
    fts: Vec<PyRankRow>,
    vector: Vec<PyRankRow>,
    fts_weight: f32,
    vector_weight: f32,
) -> Vec<PyRankRow> {
    let fts_in: Vec<m3_rank::RankRow> = fts.iter().map(rankrow_in).collect();
    let vec_in: Vec<m3_rank::RankRow> = vector.iter().map(rankrow_in).collect();
    let weights = m3_rank::FusionWeights { fts: fts_weight, vector: vector_weight };
    m3_rank::fuse(fts_in, vec_in, weights).into_iter().map(rankrow_out).collect()
}

// ---------------------------------------------------------------------------
// m3-route
// ---------------------------------------------------------------------------

/// Signals extracted from a query, fed into the route scorer.
#[pyclass(name = "RouteSignals", skip_from_py_object)]
#[derive(Clone)]
struct PyRouteSignals {
    #[pyo3(get)]
    token_count: usize,
    #[pyo3(get)]
    has_entity_hint: bool,
    #[pyo3(get)]
    recency_cue: bool,
    #[pyo3(get)]
    intent_marker: Option<String>,
    #[pyo3(get)]
    question_form: bool,
}

#[pymethods]
impl PyRouteSignals {
    fn __repr__(&self) -> String {
        format!(
            "RouteSignals(token_count={}, has_entity_hint={}, recency_cue={}, intent_marker={:?}, question_form={})",
            self.token_count, self.has_entity_hint, self.recency_cue, self.intent_marker, self.question_form
        )
    }
}

/// The route decision: chosen branch, confidence, per-branch score breakdown.
#[pyclass(name = "RouteDecision")]
struct PyRouteDecision {
    #[pyo3(get)]
    branch: String,
    #[pyo3(get)]
    confidence: f32,
    #[pyo3(get)]
    signal_breakdown: Vec<(String, f32)>,
}

#[pymethods]
impl PyRouteDecision {
    fn __repr__(&self) -> String {
        format!("RouteDecision(branch={:?}, confidence={})", self.branch, self.confidence)
    }
}

fn signals_in(s: &PyRouteSignals) -> m3_route::RouteSignals {
    m3_route::RouteSignals {
        token_count: s.token_count,
        has_entity_hint: s.has_entity_hint,
        recency_cue: s.recency_cue,
        intent_marker: s.intent_marker.clone(),
        question_form: s.question_form,
    }
}

#[pyfunction]
fn extract_signals(query: &str) -> PyRouteSignals {
    let s = m3_route::extract_signals(query);
    PyRouteSignals {
        token_count: s.token_count,
        has_entity_hint: s.has_entity_hint,
        recency_cue: s.recency_cue,
        intent_marker: s.intent_marker,
        question_form: s.question_form,
    }
}

#[pyfunction]
fn decide_route(query: &str, signals: &PyRouteSignals) -> PyRouteDecision {
    let d = m3_route::decide_route(query, &signals_in(signals));
    PyRouteDecision {
        branch: d.branch,
        confidence: d.confidence,
        signal_breakdown: d.signal_breakdown,
    }
}

// ---------------------------------------------------------------------------
// m3-graph
// ---------------------------------------------------------------------------

/// In-memory relationship graph with multi-hop traversal.
#[pyclass(name = "GraphIndex")]
struct PyGraphIndex {
    inner: m3_graph::GraphIndex,
}

#[pymethods]
impl PyGraphIndex {
    #[new]
    fn new() -> Self {
        PyGraphIndex { inner: m3_graph::GraphIndex::new() }
    }

    fn add_node(&mut self, id: &str) {
        self.inner.add_node(id);
    }

    fn add_edge(&mut self, from: &str, to: &str, kind: &str, weight: f32) {
        self.inner.add_edge(from, to, kind, weight);
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn neighbors_within(&self, start: &str, max_hops: usize) -> Vec<(String, usize)> {
        self.inner.neighbors_within(start, max_hops)
    }

    fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        self.inner.shortest_path(from, to)
    }

    fn expand(&self, seeds: Vec<String>, max_hops: usize, limit: usize) -> Vec<String> {
        let refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        self.inner.expand(&refs, max_hops, limit)
    }
}

/// Circuit breaker: N consecutive failures trip it open for `reset_after_secs`.
#[pyclass(name = "CircuitBreaker")]
struct PyCircuitBreaker {
    inner: m3_graph::CircuitBreaker,
}

#[pymethods]
impl PyCircuitBreaker {
    #[new]
    fn new(threshold: u32, reset_after_secs: f64) -> Self {
        PyCircuitBreaker {
            inner: m3_graph::CircuitBreaker::new(
                threshold,
                std::time::Duration::from_secs_f64(reset_after_secs),
            ),
        }
    }

    /// One of `"closed"`, `"open"`, `"half_open"`.
    fn state(&self) -> &'static str {
        match self.inner.state() {
            m3_graph::BreakerState::Closed => "closed",
            m3_graph::BreakerState::Open => "open",
            m3_graph::BreakerState::HalfOpen => "half_open",
        }
    }

    fn allow_request(&mut self) -> bool {
        self.inner.allow_request()
    }

    fn record_success(&mut self) {
        self.inner.record_success();
    }

    fn record_failure(&mut self) {
        self.inner.record_failure();
    }
}

/// Exponential-backoff retry policy. `delay_for_attempt` is pure math.
#[pyclass(name = "RetryPolicy")]
struct PyRetryPolicy {
    inner: m3_graph::RetryPolicy,
}

#[pymethods]
impl PyRetryPolicy {
    #[new]
    fn new(max_attempts: u32, base_delay_secs: f64, max_delay_secs: f64) -> Self {
        PyRetryPolicy {
            inner: m3_graph::RetryPolicy::new(
                max_attempts,
                std::time::Duration::from_secs_f64(base_delay_secs),
                std::time::Duration::from_secs_f64(max_delay_secs),
            ),
        }
    }

    #[getter]
    fn max_attempts(&self) -> u32 {
        self.inner.max_attempts
    }

    /// Backoff (in seconds) before retry `attempt` (1-based).
    fn delay_for_attempt(&self, attempt: u32) -> f64 {
        self.inner.delay_for_attempt(attempt).as_secs_f64()
    }
}

// ---------------------------------------------------------------------------
// m3-ner-ort
// ---------------------------------------------------------------------------

/// A decoded NER span: a labelled, scored `[start, end)` token range.
#[pyclass(name = "Span", skip_from_py_object)]
#[derive(Clone)]
struct PySpan {
    #[pyo3(get)]
    start: usize,
    #[pyo3(get)]
    end: usize,
    #[pyo3(get)]
    label: usize,
    #[pyo3(get)]
    score: f32,
}

#[pymethods]
impl PySpan {
    fn __repr__(&self) -> String {
        format!(
            "Span(start={}, end={}, label={}, score={})",
            self.start, self.end, self.label, self.score
        )
    }
}

fn platform_from_str(p: &str) -> PyResult<m3_ner_ort::Platform> {
    Ok(match p.to_ascii_lowercase().as_str() {
        "linux_nvidia" => m3_ner_ort::Platform::LinuxNvidia,
        "linux_amd" => m3_ner_ort::Platform::LinuxAmd,
        "linux_intel_gpu" => m3_ner_ort::Platform::LinuxIntelGpu,
        "windows" => m3_ner_ort::Platform::Windows,
        "macos_apple_silicon" => m3_ner_ort::Platform::MacOsAppleSilicon,
        "cpu_only" => m3_ner_ort::Platform::CpuOnly,
        other => {
            return Err(PyValueError::new_err(format!("unknown platform: {other}")));
        }
    })
}

/// Execution-provider priority list for a platform. `platform` is one of
/// `linux_nvidia`, `linux_amd`, `linux_intel_gpu`, `windows`,
/// `macos_apple_silicon`, `cpu_only`.
#[pyfunction]
fn ep_priority(platform: &str) -> PyResult<Vec<&'static str>> {
    Ok(m3_ner_ort::ep_priority(platform_from_str(platform)?))
}

/// Decode a flattened GLiNER span-score tensor. `shape` is
/// `(max_spans, span_width, num_labels)`.
#[pyfunction]
fn decode_spans(scores: Vec<f32>, shape: (usize, usize, usize), threshold: f32) -> Vec<PySpan> {
    m3_ner_ort::decode_spans(&scores, shape, threshold)
        .into_iter()
        .map(|s| PySpan { start: s.start, end: s.end, label: s.label, score: s.score })
        .collect()
}

// ---------------------------------------------------------------------------
// m3-dispatcher + M3_* env config
// ---------------------------------------------------------------------------

/// Rough token-length estimate (~4 chars/token, minimum 1).
#[pyfunction]
fn estimate_tokens(text: &str) -> usize {
    m3_dispatcher::estimate_tokens(text)
}

/// Typed dispatcher configuration. Construction precedence is
/// kwarg > `M3_*` env var > crate default (plan §9.7). The `Dispatcher` itself
/// is async/generic and not bound — this class demonstrates the §9.6 config
/// pattern and is introspectable from Python.
#[pyclass(name = "DispatcherConfig")]
struct PyDispatcherConfig {
    inner: m3_dispatcher::DispatcherConfig,
}

#[pymethods]
impl PyDispatcherConfig {
    #[new]
    #[pyo3(signature = (streams=None, coalesce_window_ms=None, max_batch_tokens=None))]
    fn new(
        streams: Option<usize>,
        coalesce_window_ms: Option<u64>,
        max_batch_tokens: Option<usize>,
    ) -> Self {
        // env layer first, then any provided kwarg overrides it.
        let mut cfg = config::dispatcher_config_from_env();
        if let Some(s) = streams {
            cfg.streams = s;
        }
        if let Some(c) = coalesce_window_ms {
            cfg.coalesce_window_ms = c;
        }
        if let Some(m) = max_batch_tokens {
            cfg.max_batch_tokens = m;
        }
        PyDispatcherConfig { inner: cfg }
    }

    #[getter]
    fn streams(&self) -> usize {
        self.inner.streams
    }

    #[getter]
    fn coalesce_window_ms(&self) -> u64 {
        self.inner.coalesce_window_ms
    }

    #[getter]
    fn max_batch_tokens(&self) -> usize {
        self.inner.max_batch_tokens
    }

    #[getter]
    fn length_buckets(&self) -> Vec<usize> {
        self.inner.length_buckets.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DispatcherConfig(streams={}, coalesce_window_ms={}, max_batch_tokens={})",
            self.inner.streams, self.inner.coalesce_window_ms, self.inner.max_batch_tokens
        )
    }
}

/// Resolved `M3_*` config snapshot — what the env layer produced before any
/// kwarg overrides. Lets Python introspect the §9.6 env wiring.
#[pyfunction]
fn env_config_summary(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("M3_EMBED_STREAMS", config::embed_streams())?;
    d.set_item("M3_EMBED_COALESCE_MS", config::embed_coalesce_ms())?;
    d.set_item("M3_EMBED_MAX_BATCH_TOKENS", config::embed_max_batch_tokens())?;
    d.set_item("M3_HASH_PROVIDER", config::hash_provider())?;
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// m3-embed-llamacpp — embedded (in-process) llama.cpp backend
// ---------------------------------------------------------------------------
//
// Only compiled with `--features embedded`. Without the feature the
// `EmbeddedEmbedder` class is not registered at all, so `m3_core_rs.EmbeddedEmbedder`
// raises `AttributeError` in Python — a clear signal the wheel was built
// without llama.cpp.

/// In-process bge-m3 (or any GGUF) embedding model via linked llama.cpp.
/// Construction is cheap; the GGUF model loads lazily on the first
/// `embed`/`embedding_dim` call.
#[cfg(feature = "embedded")]
#[pyclass(name = "EmbeddedEmbedder")]
struct PyEmbeddedEmbedder {
    backend: m3_embed_llamacpp::EmbeddedBackend,
    runtime: tokio::runtime::Runtime,
}

#[cfg(feature = "embedded")]
#[pymethods]
impl PyEmbeddedEmbedder {
    /// `model_path` is an absolute path to a GGUF embedding model.
    /// Does not load the model — that happens lazily on first use.
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PyOSError::new_err(format!("failed to build tokio runtime: {e}"))
            })?;
        Ok(PyEmbeddedEmbedder {
            backend: m3_embed_llamacpp::EmbeddedBackend::new(model_path),
            runtime,
        })
    }

    /// Embed a batch of texts. Returns one row per input, each of length
    /// `embedding_dim()`. Blocks the calling thread: `EmbeddedBackend::run`
    /// is async but the heavy work runs on tokio's blocking pool, so a
    /// `block_on` here is fine (chosen for simplicity over pyo3-async-runtimes).
    fn embed(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let batch = m3_dispatcher::Batch::new(texts, 0);
        let out = self
            .runtime
            .block_on(<m3_embed_llamacpp::EmbeddedBackend as m3_dispatcher::ModelBackend>::run(
                &self.backend,
                batch,
            ));
        map_err(out).map(|b| b.rows)
    }

    /// Embedding dimension reported by the model. Forces the lazy GGUF
    /// load on first call (no full inference needed).
    fn embedding_dim(&self) -> PyResult<i32> {
        map_err(self.backend.embedding_dim())
    }
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

#[pymodule]
fn m3_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bridge env_logger to stderr; `RUST_LOG` controls verbosity.
    let _ = env_logger::try_init();
    log::info!("m3_core_rs initialized (hash provider: {})", m3_hash::active_provider());

    m.add_function(wrap_pyfunction!(sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_hex_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hash_provider, m)?)?;
    m.add_function(wrap_pyfunction!(cosine, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank_scored, m)?)?;
    m.add_function(wrap_pyfunction!(enforce_displacement_guard, m)?)?;
    m.add_function(wrap_pyfunction!(blob_as_f32, m)?)?;
    m.add_function(wrap_pyfunction!(f32_as_blob, m)?)?;
    m.add_function(wrap_pyfunction!(scrub, m)?)?;
    m.add_function(wrap_pyfunction!(redaction_compile_errors, m)?)?;
    m.add_function(wrap_pyfunction!(fuse, m)?)?;
    m.add_function(wrap_pyfunction!(extract_signals, m)?)?;
    m.add_function(wrap_pyfunction!(decide_route, m)?)?;
    m.add_function(wrap_pyfunction!(ep_priority, m)?)?;
    m.add_function(wrap_pyfunction!(decode_spans, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(env_config_summary, m)?)?;

    m.add_class::<PyRankRow>()?;
    m.add_class::<PyRouteSignals>()?;
    m.add_class::<PyRouteDecision>()?;
    m.add_class::<PyGraphIndex>()?;
    m.add_class::<PyCircuitBreaker>()?;
    m.add_class::<PyRetryPolicy>()?;
    m.add_class::<PySpan>()?;
    m.add_class::<PyDispatcherConfig>()?;

    // Only present when built with `--features embedded`. Absent otherwise,
    // so `m3_core_rs.EmbeddedEmbedder` raises AttributeError in Python.
    #[cfg(feature = "embedded")]
    m.add_class::<PyEmbeddedEmbedder>()?;

    Ok(())
}
