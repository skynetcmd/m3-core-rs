//! PyO3 bindings for m3-core-rs.
//!
//! The only crate Python sees. This is also the ONLY crate that reads `M3_*`
//! env vars (per plan §9.6) — it translates them into the typed config structs
//! the generic crates consume. Generic crates never call `std::env::var` and
//! never depend on pyo3.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyString};

pub use m3_dispatcher;
pub use m3_embed_llamacpp;
pub use m3_error;
pub use m3_fts;
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

/// DEPRECATED: prefer `cosine_batch_packed_flat`, which takes a single
/// contiguous bytes buffer and avoids per-row Python -> Rust copies on the
/// GIL-attached thread. This signature copies each blob into a fresh `Vec<u8>`
/// before `py.detach` and remains only for backward compatibility.
///
/// Cosine of `query` against a list of packed-blob embeddings. `dim` is the
/// embedding dimension. Each `bytes`-typed blob must be exactly `dim * 4`
/// bytes (sequence of little-endian f32s); any other length scores 0.0 for
/// that row. Releases the GIL while rayon scores in parallel.
#[pyfunction]
fn cosine_batch_packed(
    py: Python<'_>,
    query: Vec<f32>,
    blobs: Vec<Vec<u8>>,
    dim: usize,
) -> PyResult<Vec<f32>> {
    py.detach(||{
        let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
        map_err(m3_vector::cosine_batch_packed(&query, &refs, dim))
    })
}

/// Flat-bytes hot-path variant of `cosine_batch_packed`.
///
/// `blobs` is one contiguous `bytes` buffer of `n_rows * dim * 4` bytes; row
/// `i` lives at `blobs[i*dim*4 .. (i+1)*dim*4]`. The number of rows is
/// `blobs.len() / (dim * 4)` and the function errors with `ValueError` if
/// `blobs.len()` is not evenly divisible by `dim * 4`.
///
/// **Pass a `bytes` object** (not `bytearray`, not a `list`): PyO3 borrows
/// `&[u8]` directly from the underlying buffer with zero copies. The whole
/// point of this entry point is eliminating the per-row `Vec<u8>` copy that
/// the deprecated `cosine_batch_packed` performs on the GIL-attached thread.
/// Releases the GIL while rayon scores in parallel.
#[pyfunction]
fn cosine_batch_packed_flat(
    py: Python<'_>,
    query: Vec<f32>,
    blobs: &[u8],
    dim: usize,
) -> PyResult<Vec<f32>> {
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("dim * 4 overflow"))?;
    if row_bytes == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if !blobs.len().is_multiple_of(row_bytes) {
        return Err(PyValueError::new_err(format!(
            "blobs length {} is not a multiple of dim*4 = {}",
            blobs.len(),
            row_bytes,
        )));
    }
    py.detach(|| {
        let refs: Vec<&[u8]> = blobs.chunks_exact(row_bytes).collect();
        map_err(m3_vector::cosine_batch_packed(&query, &refs, dim))
    })
}

/// Max-pooled multi-anchor cosine over packed candidate blobs — the "max-similarity"
/// rerank: for each candidate, `max_j cosine(anchor_j, candidate)`. Pushes the whole
/// N-candidate × M-anchor matrix op into Rust so the Python caller never runs a
/// per-anchor FFI loop. Rayon-parallel over candidates; GIL released.
///
/// Both buffers are flat little-endian f32 bytes (same contract as
/// `cosine_batch_packed_flat`): `anchors` = `m * dim * 4` bytes (M anchor rows),
/// `blobs` = `n * dim * 4` bytes (N candidate rows). **Pass `bytes`** (zero-copy
/// borrow). Errors with `ValueError` if either length isn't a multiple of `dim*4`.
/// A candidate of the wrong length scores 0.0; with zero anchors every score is -1.0.
#[pyfunction]
fn cosine_batch_maxpool_packed(
    py: Python<'_>,
    anchors: &[u8],
    blobs: &[u8],
    dim: usize,
) -> PyResult<Vec<f32>> {
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("dim * 4 overflow"))?;
    if row_bytes == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if !anchors.len().is_multiple_of(row_bytes) {
        return Err(PyValueError::new_err(format!(
            "anchors length {} is not a multiple of dim*4 = {}",
            anchors.len(),
            row_bytes,
        )));
    }
    if !blobs.len().is_multiple_of(row_bytes) {
        return Err(PyValueError::new_err(format!(
            "blobs length {} is not a multiple of dim*4 = {}",
            blobs.len(),
            row_bytes,
        )));
    }
    py.detach(|| {
        // Cast each anchor's bytes to &[f32]; a bad cast (alignment) -> error, surfaced.
        let anchor_vecs: std::result::Result<Vec<&[f32]>, _> = anchors
            .chunks_exact(row_bytes)
            .map(bytemuck::try_cast_slice::<u8, f32>)
            .collect();
        let anchor_vecs = match anchor_vecs {
            Ok(v) => v,
            Err(e) => return Err(PyValueError::new_err(format!("anchor cast failed: {e}"))),
        };
        let cand_refs: Vec<&[u8]> = blobs.chunks_exact(row_bytes).collect();
        map_err(m3_vector::cosine_batch_maxpool_packed(&anchor_vecs, &cand_refs, dim))
    })
}

/// Vectorized hybrid scoring — body of the per-row Python scoring loop in
/// `memory_search_scored_impl`, fully rayon-parallel. Releases the GIL.
///
/// All input lists must be the same length (one entry per candidate).
/// Returns one final blended score per candidate. Caller still layers on
/// role/intent boosts and recency/temporal adjustments downstream.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hybrid_score_batch(
    py: Python<'_>,
    vector_scores: Vec<f32>,
    bm25_scores: Vec<f32>,
    content_lens: Vec<u32>,
    importances: Vec<f32>,
    title_overlaps: Vec<f32>,
    vector_weight: f32,
    importance_weight: f32,
    title_match_boost: f32,
    short_turn_threshold: u32,
) -> PyResult<Vec<f32>> {
    py.detach(||{
        map_err(m3_vector::hybrid_score_batch(
            &vector_scores,
            &bm25_scores,
            &content_lens,
            &importances,
            &title_overlaps,
            vector_weight,
            importance_weight,
            title_match_boost,
            short_turn_threshold,
        ))
    })
}

/// Rank-based linear recency bonus aligned to `valid_froms`. Items with
/// empty / None `valid_from` get 0.0. Dated items get
/// `bias * rank / (n_dated - 1)` after lex-sort. Bonus is `0.0` everywhere
/// when fewer than two dated items exist or `bias <= 0`.
///
/// Returns a list aligned 1:1 with `valid_froms`; caller adds to existing
/// hybrid scores.
#[pyfunction]
fn recency_bonus_ranks(
    py: Python<'_>,
    valid_froms: Vec<Option<String>>,
    bias: f32,
) -> Vec<f32> {
    py.detach(||m3_vector::recency_bonus_ranks(&valid_froms, bias))
}

#[pyfunction]
fn mmr_rerank(
    py: Python<'_>,
    query: Vec<f32>,
    candidates: Vec<Vec<f32>>,
    lambda: f32,
    k: usize,
) -> PyResult<Vec<usize>> {
    py.detach(|| {
        let refs: Vec<&[f32]> = candidates.iter().map(|v| v.as_slice()).collect();
        map_err(m3_vector::mmr_rerank(&query, &refs, lambda, k))
    })
}

#[pyfunction]
fn mmr_rerank_scored(
    py: Python<'_>,
    relevance: Vec<f32>,
    candidate_vectors: Vec<Vec<f32>>,
    lambda: f32,
    k: usize,
    force_seed_first: bool,
) -> PyResult<Vec<usize>> {
    py.detach(|| {
        let refs: Vec<&[f32]> = candidate_vectors.iter().map(|v| v.as_slice()).collect();
        map_err(m3_vector::mmr_rerank_scored(
            &relevance,
            &refs,
            lambda,
            k,
            force_seed_first,
        ))
    })
}

/// Flat-bytes hot-path variant of `mmr_rerank`.
///
/// `candidates` is one contiguous `bytes` buffer of `n_rows * dim * 4` bytes
/// (little-endian f32s). Number of rows is `candidates.len() / (dim * 4)` —
/// errors with `ValueError` if the length is not evenly divisible or the byte
/// slice fails to cast to `&[f32]` (alignment). Pass a `bytes` object so PyO3
/// can borrow `&[u8]` zero-copy. Releases the GIL while MMR runs.
#[pyfunction]
fn mmr_rerank_packed(
    py: Python<'_>,
    query: Vec<f32>,
    candidates: &[u8],
    dim: usize,
    lambda: f32,
    k: usize,
) -> PyResult<Vec<usize>> {
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("dim * 4 overflow"))?;
    if row_bytes == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if !candidates.len().is_multiple_of(row_bytes) {
        return Err(PyValueError::new_err(format!(
            "candidates length {} is not a multiple of dim*4 = {}",
            candidates.len(),
            row_bytes,
        )));
    }
    let floats: &[f32] = bytemuck::try_cast_slice::<u8, f32>(candidates)
        .map_err(|e| PyValueError::new_err(format!("candidates byte->f32 cast failed: {e}")))?;
    py.detach(|| {
        let refs: Vec<&[f32]> = floats.chunks_exact(dim).collect();
        map_err(m3_vector::mmr_rerank(&query, &refs, lambda, k))
    })
}

/// Flat-bytes hot-path variant of `mmr_rerank_scored`. Same byte-layout
/// contract as `mmr_rerank_packed`. Releases the GIL while MMR runs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mmr_rerank_scored_packed(
    py: Python<'_>,
    relevance: Vec<f32>,
    candidate_vectors: &[u8],
    dim: usize,
    lambda: f32,
    k: usize,
    force_seed_first: bool,
) -> PyResult<Vec<usize>> {
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("dim * 4 overflow"))?;
    if row_bytes == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if !candidate_vectors.len().is_multiple_of(row_bytes) {
        return Err(PyValueError::new_err(format!(
            "candidate_vectors length {} is not a multiple of dim*4 = {}",
            candidate_vectors.len(),
            row_bytes,
        )));
    }
    let floats: &[f32] = bytemuck::try_cast_slice::<u8, f32>(candidate_vectors)
        .map_err(|e| PyValueError::new_err(format!("candidate_vectors byte->f32 cast failed: {e}")))?;
    let n_rows = floats.len() / dim;
    if relevance.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "relevance length {} != n_rows {} (derived from candidate_vectors / dim*4)",
            relevance.len(),
            n_rows,
        )));
    }
    py.detach(|| {
        let refs: Vec<&[f32]> = floats.chunks_exact(dim).collect();
        map_err(m3_vector::mmr_rerank_scored(
            &relevance,
            &refs,
            lambda,
            k,
            force_seed_first,
        ))
    })
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

#[pyfunction]
fn token_jaccard(a: &str, b: &str) -> f32 {
    m3_vector::token_jaccard(a, b)
}

#[pyfunction]
fn token_jaccard_batch(query: &str, candidates: Vec<String>) -> Vec<f32> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    m3_vector::token_jaccard_batch(query, &refs)
}

#[pyfunction]
fn rank_hybrid(
    py: Python<'_>,
    relevance: Vec<f32>,
    contents: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    lambda: f32,
    k: usize,
) -> PyResult<Vec<usize>> {
    py.detach(|| {
        let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
        let c_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
        map_err(m3_vector::rank_hybrid(&relevance, &c_refs, &refs, lambda, k))
    })
}

#[pyfunction]
fn rank_hybrid_packed(
    py: Python<'_>,
    relevance: Vec<f32>,
    contents: Vec<String>,
    embeddings: &[u8],
    dim: usize,
    lambda: f32,
    k: usize,
) -> PyResult<Vec<usize>> {
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("dim * 4 overflow"))?;
    if row_bytes == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if !embeddings.len().is_multiple_of(row_bytes) {
        return Err(PyValueError::new_err(format!(
            "embeddings length {} is not a multiple of dim*4 = {}",
            embeddings.len(),
            row_bytes,
        )));
    }
    let floats: &[f32] = bytemuck::try_cast_slice::<u8, f32>(embeddings)
        .map_err(|e| PyValueError::new_err(format!("embeddings byte->f32 cast failed: {e}")))?;
    py.detach(|| {
        let refs: Vec<&[f32]> = floats.chunks_exact(dim).collect();
        let c_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
        map_err(m3_vector::rank_hybrid(&relevance, &c_refs, &refs, lambda, k))
    })
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
    /// Compiled-`Redactor` cache: (config, redactor). Compiling the ~17
    /// built-in regexes per call made the Rust path ~13x slower than the
    /// Python one (which caches by config hash). The config is effectively
    /// static per process, so a single last-config slot suffices — recompile
    /// only when the config actually changes.
    static REDACTOR_CACHE: RefCell<Option<(m3_redact::RedactionConfig, m3_redact::Redactor)>> =
        const { RefCell::new(None) };
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
///
/// The compiled `Redactor` is cached per-thread keyed by config — a repeated
/// call with the same config skips regex recompilation entirely.
#[pyfunction]
fn scrub<'py>(
    py: Python<'py>,
    content: Bound<'py, PyString>,
    config: &Bound<'_, PyDict>,
) -> PyResult<(Bound<'py, PyAny>, usize, Vec<String>)> {
    let cfg = redaction_config_from_dict(config)?;
    let content_str = content.to_str()?;
    let result = REDACTOR_CACHE.with(|cache| {
        let mut slot = cache.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some((cached_cfg, _)) => cached_cfg != &cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = Some((cfg.clone(), m3_redact::Redactor::new(&cfg)));
        }
        slot.as_ref().unwrap().1.apply(content_str)
    });
    REDACTION_COMPILE_ERRORS.with(|e| {
        *e.borrow_mut() = result.compile_errors.clone();
    });
    let returned_content = if result.match_count == 0 {
        content.into_any()
    } else {
        PyString::new(py, &result.content).into_any()
    };
    Ok((returned_content, result.match_count, result.groups_fired))
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
// m3-governor
// ---------------------------------------------------------------------------

/// Adaptive background-workload governor. Constructs with the user-selectable
/// thresholds and decides a pacing dict identical in shape to the Python
/// `get_governor_pacing` return — a drop-in replacement behind the
/// `M3_CORE_RS_DISABLE` fallback gate.
#[pyclass(name = "Governor")]
struct PyGovernor {
    inner: m3_governor::GovernorConfig,
}

#[pymethods]
impl PyGovernor {
    #[new]
    fn new(initial_limit: i64, limit_threshold: i64) -> Self {
        PyGovernor {
            inner: m3_governor::GovernorConfig::new(initial_limit, limit_threshold),
        }
    }

    /// Decide pacing for the given host `load` (0–100, max across CPU/RAM/GPU)
    /// and `elapsed` seconds since the last user interaction. Returns a dict
    /// with keys matching the Python truth table: always `background` and
    /// `interactive_delay`; `background_delay` is present only in the modes
    /// where Python includes it (omitted in critical/HALTED-on-load mode).
    fn decide(&self, py: Python<'_>, load: f64, elapsed: f64) -> PyResult<Py<PyDict>> {
        let p = self.inner.decide(load, elapsed);
        let d = PyDict::new(py);
        d.set_item("background", p.mode.as_str())?;
        if let Some(bg) = p.background_delay {
            d.set_item("background_delay", bg)?;
        }
        d.set_item("interactive_delay", p.interactive_delay)?;
        Ok(d.into())
    }
}

// ---------------------------------------------------------------------------
// m3-ingest
// ---------------------------------------------------------------------------

/// Recursively sweep `root`, returning one dict per entry with keys
/// `path`, `size`, `mtime`, `is_dir`. Directories whose basename is in
/// `dir_ignores` are pruned (their subtrees are skipped). `max_depth` is
/// measured from `root` (`0` = direct children only); pass a negative value
/// for unbounded. Symlinked directories are descended only when
/// `follow_symlinks` is true. Individual unreadable entries are skipped.
///
/// This is the mechanical, syscall-bound half of the Python walker; the
/// caller still applies its gitignore/glob/binary-sniff/size filters to the
/// returned list.
#[pyfunction]
#[pyo3(signature = (root, dir_ignores, max_depth=-1, follow_symlinks=false))]
fn fs_walk(
    py: Python<'_>,
    root: &str,
    dir_ignores: Vec<String>,
    max_depth: i64,
    follow_symlinks: bool,
) -> PyResult<Py<pyo3::types::PyList>> {
    let md = if max_depth < 0 {
        None
    } else {
        Some(max_depth as usize)
    };
    let entries = m3_ingest::walk_entries(root, &dir_ignores, md, follow_symlinks);
    let list = pyo3::types::PyList::empty(py);
    for e in entries {
        let d = PyDict::new(py);
        d.set_item("path", e.path)?;
        d.set_item("size", e.size)?;
        d.set_item("mtime", e.mtime)?;
        d.set_item("is_dir", e.is_dir)?;
        list.append(d)?;
    }
    Ok(list.into())
}

/// Batch-hash file contents in parallel. Returns one dict per input path (in
/// input order) with keys `path`, `sha256` (hex string or `None` on failure),
/// and `error` (message string or `None`). Byte-identical to the Python
/// `file_content_sha256` for readable files.
#[pyfunction]
fn hash_files(py: Python<'_>, paths: Vec<String>) -> PyResult<Py<pyo3::types::PyList>> {
    // Release the GIL for the parallel I/O + hashing, then marshal results.
    let results = py.detach(|| m3_ingest::hash_files(&paths));
    let list = pyo3::types::PyList::empty(py);
    for (path, res) in results {
        let d = PyDict::new(py);
        d.set_item("path", path)?;
        match res {
            Ok(hex) => {
                d.set_item("sha256", hex)?;
                d.set_item("error", py.None())?;
            }
            Err(msg) => {
                d.set_item("sha256", py.None())?;
                d.set_item("error", msg)?;
            }
        }
        list.append(d)?;
    }
    Ok(list.into())
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

#[cfg(feature = "onnx")]
#[pyclass(name = "OrtNer")]
struct PyOrtNer {
    inner: std::sync::Arc<m3_ner_ort::OrtNer>,
    runtime: tokio::runtime::Runtime,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl PyOrtNer {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, labels, platform="cpu_only", threshold=0.5))]
    fn new(
        model_path: String,
        tokenizer_path: String,
        labels: Vec<String>,
        platform: &str,
        threshold: f32,
    ) -> PyResult<Self> {
        let p = platform_from_str(platform)?;
        let inner = m3_ner_ort::OrtNer::new(
            model_path,
            tokenizer_path,
            None,
            labels,
            threshold,
            m3_ner_ort::Quant::Fp32,
            p,
        )
        .map_err(|e| PyValueError::new_err(format!("failed to load NER model: {e}")))?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|e| PyOSError::new_err(format!("failed to build tokio runtime: {e}")))?;

        Ok(PyOrtNer {
            inner: std::sync::Arc::new(inner),
            runtime,
        })
    }

    fn run(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        py.detach(|| {
            let batch = m3_dispatcher::Batch::new(texts, 0);
            use m3_dispatcher::ModelBackend;
            let out = self.runtime.block_on(inner.run(batch));
            map_err(out).map(|b| b.rows)
        })
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels().to_vec()
    }
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

// ---------------------------------------------------------------------------
// m3-fts — FTS5 query sanitization / lexical tokenization
// ---------------------------------------------------------------------------

/// Strip FTS5 operators (OR/AND/NOT/NEAR + bracket/wildcard punctuation) from
/// user input and trim. Byte-exact port of `_sanitize_fts` in fts.py.
/// `max_len` counts code points (matches Python `len()` on `str`).
#[pyfunction]
#[pyo3(signature = (query, max_len = 500))]
fn sanitize_fts(query: &str, max_len: usize) -> String {
    m3_fts::sanitize_fts(query, max_len)
}

/// Compile a raw user query into an FTS5 MATCH string, returning `(query, ok)`.
/// Byte-exact port of `_compile_fts_query` in fts.py: exact-phrase passthrough,
/// the searchable-punctuation transform mirroring the mi_fts_insert trigger, and
/// mode-dependent OR-join / wildcard branching.
#[pyfunction]
fn compile_fts_query(query: &str, mode: &str) -> (String, bool) {
    m3_fts::compile_fts_query(query, mode)
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

/// Wave 9.0 — label of the active embedded compute backend. Returns
/// `"cpu"` for the default `embedded` build, `"cuda"` / `"vulkan"` /
/// `"metal"` when the matching feature is enabled, and `"none"` when the
/// wheel was built without `embedded` at all. Wave 9.4 will surface this in
/// backend stats; for now it lets Python tests assert which backend the
/// wheel is linked against.
#[pyfunction]
fn embed_backend_label() -> &'static str {
    #[cfg(feature = "embedded")]
    {
        m3_embed_llamacpp::active_backend()
    }
    #[cfg(not(feature = "embedded"))]
    {
        "none"
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
    d.set_item("M3_EMBED_CTX", config::embed_ctx())?;
    d.set_item("M3_EMBED_SEQ_MAX", config::embed_seq_max())?;
    d.set_item("M3_EMBED_N_BATCH", config::embed_n_batch())?;
    d.set_item("M3_EMBED_N_UBATCH", config::embed_n_ubatch())?;
    d.set_item("M3_HASH_PROVIDER", config::hash_provider())?;
    Ok(d.into())
}

/// Zero-allocation, high-frequency structured log formatter.
/// Parity with Python StructuredLogger.format().
#[pyfunction]
#[pyo3(signature = (event, *args, **kwargs))]
fn format_log(
    event: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let mut out = String::with_capacity(128 + args.len() * 16 + kwargs.map_or(0, |d| d.len()) * 24);
    out.push_str(event);

    for arg in args.iter() {
        // Parity with Python `if a is None or a == "": continue`. Note this is
        // the OBJECT equalling "" (an empty str arg), NOT its stringification
        // being empty — an object whose __str__ returns "" is still appended
        // (as ""), exactly as Python's `parts.append(str(a))` would.
        if arg.is_none() || arg.eq(PyString::new(arg.py(), ""))? {
            continue;
        }
        let py_str = arg.str()?;
        let s = py_str.to_str()?;
        out.push_str(" | ");
        out.push_str(s);
    }

    if let Some(kwargs) = kwargs {
        for (k, v) in kwargs.iter() {
            if v.is_none() {
                continue;
            }
            let k_str = k.str()?;
            let key = k_str.to_str()?;
            let v_str = v.str()?;
            let val = v_str.to_str()?;
            out.push_str(" | ");
            out.push_str(key);
            out.push_str("=");
            out.push_str(val);
        }
    }

    Ok(out)
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
///
/// ## Dispatcher wiring (CHANGE 3)
///
/// `embed` no longer calls `EmbeddedBackend::run` directly — it routes through
/// `m3_dispatcher::Dispatcher<EmbeddedBackend>`, the same coalescer the HTTP
/// path uses. The stream count flows env var -> `DispatcherConfig::streams` ->
/// `EmbeddedBackend` context-pool size, so the dispatcher's slot semaphore and
/// the backend's worker-thread pool always agree (no point in a 4-context pool
/// behind an 8-slot dispatcher). For introspection / `embedding_dim` a handle
/// to the raw backend is kept alongside the dispatcher.
#[cfg(feature = "embedded")]
#[pyclass(name = "EmbeddedEmbedder")]
struct PyEmbeddedEmbedder {
    dispatcher: m3_dispatcher::Dispatcher<m3_embed_llamacpp::EmbeddedBackend>,
    backend: std::sync::Arc<m3_embed_llamacpp::EmbeddedBackend>,
    runtime: tokio::runtime::Runtime,
}

#[cfg(feature = "embedded")]
#[pymethods]
impl PyEmbeddedEmbedder {
    /// `model_path` is an absolute path to a GGUF embedding model.
    ///
    /// `warmup` (Fix #2, default `True`): when true, the constructor forces
    /// the GGUF load + worker-pool spin-up + per-context warmup decode
    /// synchronously inside `new()`. The very first real `embed()` call then
    /// only pays the per-batch cost. Pass `warmup=False` to keep the legacy
    /// lazy behavior (cold cost paid on first `embed`/`embedding_dim`).
    ///
    /// The dispatcher config is built from the `M3_*` env layer; `streams`
    /// (`M3_EMBED_STREAMS`) sizes both the dispatcher and the backend's
    /// context pool.
    #[new]
    #[pyo3(signature = (model_path, warmup=None))]
    fn new(model_path: &str, warmup: Option<bool>) -> PyResult<Self> {
        // Multi-thread runtime: the embed worker pool blocks on
        // `spawn_blocking`, and `Dispatcher::new` spawns a scheduler task.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| {
                PyOSError::new_err(format!("failed to build tokio runtime: {e}"))
            })?;

        let cfg = config::dispatcher_config_from_env();
        let streams = cfg.streams;
        let n_ctx = config::embed_ctx();
        let seq_max = config::embed_seq_max();
        let n_batch = config::embed_n_batch();
        let n_ubatch = config::embed_n_ubatch();
        // env var -> DispatcherConfig::streams -> EmbeddedBackend pool size.
        // n_ctx flows from M3_EMBED_CTX (default 8192) — total token budget per decode call.
        // seq_max flows from M3_EMBED_SEQ_MAX (default 32) — sequences packed per decode call.
        // n_batch / n_ubatch flow from M3_EMBED_N_BATCH / M3_EMBED_N_UBATCH,
        // both defaulting to M3_EMBED_CTX. The BERT encoder asserts
        // `n_ubatch >= n_tokens` over the whole decode batch (a chunk can hold
        // up to n_ctx tokens), so n_ctx is the only encoder-safe default; the
        // embed crate floors any smaller value up to n_ctx.
        //
        // Fix #12 (shape b): `Dispatcher::new` already wraps its backend in
        // `Arc<B>` internally, so we build ONE `EmbeddedBackend`, hand it to the
        // dispatcher, and pull the exact same `Arc<EmbeddedBackend>` back out
        // via `dispatcher.backend()` for `embedding_dim` / `streams`. This
        // guarantees a single underlying `Arc<ContextPool>` (and therefore a
        // single GGUF load + worker-thread pool) per `PyEmbeddedEmbedder`.
        let single_backend = m3_embed_llamacpp::EmbeddedBackend::with_streams_ctx_seqmax_batch(
            model_path, streams, n_ctx, seq_max, n_batch, n_ubatch,
        );
        // `Dispatcher::new` calls `tokio::spawn` — must run inside the runtime.
        let dispatcher = {
            let _guard = runtime.enter();
            m3_dispatcher::Dispatcher::new(cfg, single_backend)
        };
        let backend = dispatcher.backend().clone();

        let me = PyEmbeddedEmbedder {
            dispatcher,
            backend,
            runtime,
        };
        // Fix #2: synchronous warmup. `embedding_dim()` triggers the
        // ContextPool::load path, which now performs a per-worker decode +
        // embedding read before reporting ready — so by the time this returns,
        // the next embed() call lands on a fully-warm context.
        if warmup.unwrap_or(true) {
            map_err(me.backend.embedding_dim())?;
        }
        Ok(me)
    }

    /// Embed a batch of texts. Returns one row per input, each of length
    /// `embedding_dim()`. Routes through the dispatcher's `embed_batch`, which
    /// applies the slot semaphore + circuit breaker before handing the batch
    /// to the multi-stream `EmbeddedBackend`. Blocks the calling thread.
    ///
    /// Releases the GIL across the `runtime.block_on` so other Python threads
    /// can make progress while llama.cpp decodes. The `Vec<String>` extraction
    /// happens before the detach (GIL-required); inside the detach closure
    /// nothing touches Python — `dispatcher`, `runtime`, and the owned
    /// `texts: Vec<String>` are all pure Rust.
    fn embed(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let out = py.detach(|| self.runtime.block_on(self.dispatcher.embed_batch(texts)));
        map_err(out)
    }

    /// Embedding dimension reported by the model. Forces the lazy GGUF
    /// load on first call (no full inference needed).
    fn embedding_dim(&self) -> PyResult<i32> {
        map_err(self.backend.embedding_dim())
    }

    /// Configured concurrent stream count (`M3_EMBED_STREAMS`). Equal to both
    /// the dispatcher's slot count and the backend's context-pool size.
    fn streams(&self) -> usize {
        self.backend.streams()
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
    m.add_function(wrap_pyfunction!(format_log, m)?)?;
    m.add_function(wrap_pyfunction!(cosine, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_batch_packed, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_batch_packed_flat, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_batch_maxpool_packed, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(recency_bonus_ranks, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank_scored, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank_packed, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank_scored_packed, m)?)?;
    m.add_function(wrap_pyfunction!(enforce_displacement_guard, m)?)?;
    m.add_function(wrap_pyfunction!(rank_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(rank_hybrid_packed, m)?)?;
    m.add_function(wrap_pyfunction!(blob_as_f32, m)?)?;
    m.add_function(wrap_pyfunction!(f32_as_blob, m)?)?;
    m.add_function(wrap_pyfunction!(token_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(token_jaccard_batch, m)?)?;
    m.add_function(wrap_pyfunction!(scrub, m)?)?;
    m.add_function(wrap_pyfunction!(redaction_compile_errors, m)?)?;
    m.add_function(wrap_pyfunction!(fuse, m)?)?;
    m.add_function(wrap_pyfunction!(extract_signals, m)?)?;
    m.add_function(wrap_pyfunction!(decide_route, m)?)?;
    m.add_function(wrap_pyfunction!(ep_priority, m)?)?;
    m.add_function(wrap_pyfunction!(decode_spans, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(sanitize_fts, m)?)?;
    m.add_function(wrap_pyfunction!(compile_fts_query, m)?)?;
    m.add_function(wrap_pyfunction!(env_config_summary, m)?)?;
    m.add_function(wrap_pyfunction!(embed_backend_label, m)?)?;
    m.add_function(wrap_pyfunction!(fs_walk, m)?)?;
    m.add_function(wrap_pyfunction!(hash_files, m)?)?;

    m.add_class::<PyRankRow>()?;
    m.add_class::<PyRouteSignals>()?;
    m.add_class::<PyRouteDecision>()?;
    m.add_class::<PyGraphIndex>()?;
    m.add_class::<PyCircuitBreaker>()?;
    m.add_class::<PyRetryPolicy>()?;
    m.add_class::<PyGovernor>()?;
    m.add_class::<PySpan>()?;
    #[cfg(feature = "onnx")]
    m.add_class::<PyOrtNer>()?;
    m.add_class::<PyDispatcherConfig>()?;

    // Only present when built with `--features embedded`. Absent otherwise,
    // so `m3_core_rs.EmbeddedEmbedder` raises AttributeError in Python.
    #[cfg(feature = "embedded")]
    m.add_class::<PyEmbeddedEmbedder>()?;

    Ok(())
}
