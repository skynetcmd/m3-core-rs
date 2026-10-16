"""Type stubs for the m3_core_rs PyO3 extension module.

Written by hand from crates/m3-core-py/src/lib.rs. Keep in sync when the
binding surface changes. (`lambda_` below is positional-only in practice: the
Rust parameter is named `lambda`, a Python keyword.)
"""

from typing import Optional

__version__: str
__build_backend__: str  # "cpu" | "cuda" | "vulkan" | "metal" | "none" (compiled, not observed)

# --- exceptions ------------------------------------------------------------
#
# Every error the core raises is an M3Error. Each subclass ALSO subclasses the
# builtin that variant raised before the hierarchy existed, so
# `except ValueError` / `except OSError` keep working.

class M3Error(Exception): ...

class InputTooLong(M3Error, ValueError):
    tokens: int
    n_ctx: int
    index: Optional[int]  # position in the caller's batch, when known

class VectorDimMismatch(M3Error, ValueError):
    expected: int
    got: int
    index: Optional[int]  # offending row, for batch inputs

class ConfigError(M3Error, ValueError): ...
class BackendError(M3Error, OSError): ...
class IoError(M3Error, OSError): ...  # errno/strerror set when the OS reported one
class DatabaseLockedError(M3Error, OSError): ...
class ParityError(M3Error, ValueError): ...
class M3ValueError(M3Error, ValueError): ...

# --- m3-hash ---------------------------------------------------------------

def sha256_hex(text: str) -> str: ...
def sha256_hex_bytes(data: bytes) -> str: ...
def hash_provider() -> str: ...

# --- logging ---------------------------------------------------------------

def format_log(event: str, *args: object, **kwargs: object) -> str: ...

# --- m3-vector -------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float: ...
def cosine_batch(query: list[float], corpus: list[list[float]]) -> list[float]: ...
def cosine_batch_packed(query: list[float], blobs: list[bytes], dim: int) -> list[float]:
    """DEPRECATED: prefer cosine_batch_packed_flat. Wrong-length blobs score 0.0 (warned)."""
    ...
def cosine_batch_packed_flat(query: list[float], blobs: bytes, dim: int) -> list[float]: ...
def cosine_batch_maxpool_packed(anchors: bytes, blobs: bytes, dim: int) -> list[float]: ...
def hybrid_score_batch(
    vector_scores: list[float],
    bm25_scores: list[float],
    content_lens: list[int],
    importances: list[float],
    title_overlaps: list[float],
    vector_weight: float,
    importance_weight: float,
    title_match_boost: float,
    short_turn_threshold: int,
) -> list[float]: ...
def recency_bonus_ranks(valid_froms: list[Optional[str]], bias: float) -> list[float]: ...
def mmr_rerank(
    query: list[float], candidates: list[list[float]], lambda_: float, k: int
) -> list[int]: ...
def mmr_rerank_scored(
    relevance: list[float],
    candidate_vectors: list[list[float]],
    lambda_: float,
    k: int,
    force_seed_first: bool,
) -> list[int]: ...
def mmr_rerank_packed(
    query: list[float], candidates: bytes, dim: int, lambda_: float, k: int
) -> list[int]: ...
def mmr_rerank_scored_packed(
    relevance: list[float],
    candidate_vectors: bytes,
    dim: int,
    lambda_: float,
    k: int,
    force_seed_first: bool,
) -> list[int]: ...
def enforce_displacement_guard(
    items: list[tuple[float, bool]], protected_ranks: int, margin: float
) -> list[int]: ...
def rank_hybrid(
    relevance: list[float],
    contents: list[str],
    embeddings: list[list[float]],
    lambda_: float,
    k: int,
) -> list[int]: ...
def rank_hybrid_packed(
    relevance: list[float],
    contents: list[str],
    embeddings: bytes,
    dim: int,
    lambda_: float,
    k: int,
) -> list[int]: ...
def blob_as_f32(blob: bytes) -> list[float]: ...
def f32_as_blob(vec: list[float]) -> bytes: ...
def token_jaccard(a: str, b: str) -> float: ...
def token_jaccard_batch(query: str, candidates: list[str]) -> list[float]: ...

class VectorIndex:
    """Resident int8 vector index; thread-safe, searches run without the GIL."""

    dim: int
    def __init__(self, dim: int) -> None: ...
    def upsert_packed(self, keys: list[int], blobs: bytes) -> None:
        """All-or-nothing; a wrong length or NaN/inf component raises ConfigError naming the key."""
        ...
    def remove(self, keys: list[int]) -> int: ...
    def search(
        self, query: list[float], k: int, allowed: Optional[list[int]] = None
    ) -> list[tuple[int, float]]: ...
    def nbytes(self) -> int: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: int) -> bool: ...

# --- m3-redact -------------------------------------------------------------

def scrub(content: str, config: dict) -> tuple[str, int, list[str]]:
    """Byte-exact port of chatlog_redaction.scrub. `config` is the `redaction`
    sub-dict (keys: enabled, patterns, custom_regex, redact_pii). Returns
    (scrubbed_content, match_count, groups_fired)."""
    ...

def redaction_compile_errors() -> list[str]:
    """Config errors from the most recent scrub() on this thread: custom_regex
    entries that failed to compile (never echoing the pattern) and unknown
    `patterns` group names."""
    ...

# --- m3-rank ---------------------------------------------------------------

class RankRow:
    id: str
    score: float
    source: str  # "fts" | "vector" (any case); anything else raises ConfigError in fuse()
    def __init__(self, id: str, score: float, source: str) -> None: ...
    def __repr__(self) -> str: ...

def fuse(
    fts: list[RankRow],
    vector: list[RankRow],
    fts_weight: float = ...,
    vector_weight: float = ...,
) -> list[RankRow]: ...

# --- m3-route --------------------------------------------------------------

class RouteSignals:
    token_count: int
    has_entity_hint: bool
    recency_cue: bool
    intent_marker: Optional[str]
    question_form: bool
    def __repr__(self) -> str: ...

class RouteDecision:
    branch: str
    confidence: float
    signal_breakdown: list[tuple[str, float]]
    def __repr__(self) -> str: ...

def extract_signals(query: str) -> RouteSignals: ...
def decide_route(query: str, signals: RouteSignals) -> RouteDecision: ...

# --- m3-graph --------------------------------------------------------------

class GraphIndex:
    def __init__(self) -> None: ...
    def add_node(self, id: str) -> None: ...
    def add_edge(self, from_: str, to: str, kind: str, weight: float) -> None: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def neighbors_within(self, start: str, max_hops: int) -> list[tuple[str, int]]: ...
    def shortest_path(self, from_: str, to: str) -> Optional[list[str]]: ...
    def expand(self, seeds: list[str], max_hops: int, limit: int) -> list[str]: ...

class CircuitBreaker:
    def __init__(self, threshold: int, reset_after_secs: float) -> None:
        """Raises ConfigError if reset_after_secs is negative, NaN or infinite."""
        ...
    def state(self) -> str: ...  # "closed" | "open" | "half_open"
    def allow_request(self) -> bool:
        """Admits a call; past the cooldown the caller becomes the probe and must record."""
        ...
    def peek(self) -> bool:
        """Whether allow_request would admit a call now, without claiming the probe."""
        ...
    def record_success(self) -> None: ...
    def record_failure(self, message: Optional[str] = None) -> None: ...
    def last_error(self) -> Optional[str]: ...

class RetryPolicy:
    max_attempts: int
    def __init__(
        self, max_attempts: int, base_delay_secs: float, max_delay_secs: float
    ) -> None: ...
    def delay_for_attempt(self, attempt: int) -> float: ...

# --- m3-governor -----------------------------------------------------------

class Governor:
    def __init__(self, initial_limit: int, limit_threshold: int) -> None: ...
    def decide(self, load: float, elapsed: float) -> dict[str, object]: ...

# --- m3-ingest -------------------------------------------------------------

def fs_walk(
    root: str, dir_ignores: list[str], max_depth: int = -1, follow_symlinks: bool = False
) -> list[dict[str, object]]:
    """Entries with keys path, size, mtime, is_dir. Raises IoError for a missing
    or unreadable root; skipped entries below it are logged."""
    ...
def fs_walk_with_errors(
    root: str, dir_ignores: list[str], max_depth: int = -1, follow_symlinks: bool = False
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """(entries, errors); each error has keys path, error."""
    ...
def hash_files(paths: list[str]) -> list[dict[str, Optional[str]]]:
    """One dict per path (input order): path, sha256 (or None), error (or None)."""
    ...

# --- m3-ner-ort ------------------------------------------------------------

class Span:
    start: int
    end: int
    label: int
    score: float
    def __repr__(self) -> str: ...

def ep_priority(platform: str) -> list[str]: ...
def decode_spans(
    scores: list[float], shape: tuple[int, int, int], threshold: float
) -> list[Span]:
    """Raises ConfigError if len(scores) != the product of shape."""
    ...

# Present only in wheels built with the `onnx` feature.
class OrtNer:
    labels: list[str]
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        labels: list[str],
        platform: str = "cpu_only",
        threshold: float = 0.5,
    ) -> None: ...
    def run(self, texts: list[str]) -> list[list[float]]: ...

# --- m3-fts ----------------------------------------------------------------

def sanitize_fts(query: str, max_len: int = 500) -> str: ...
def compile_fts_query(query: str, mode: str) -> tuple[str, bool]: ...

# --- m3-dispatcher + M3_* env config --------------------------------------

def estimate_tokens(text: str) -> int:
    """max(len(text.encode()) // 3, len(text)) + 2; never under-counts BGE-M3."""
    ...

class DispatcherConfig:
    """Per setting: the argument > its legacy M3_* environment name (deprecated) > default.
    ctx only feeds the max_batch_tokens default (streams * ctx)."""

    streams: int
    coalesce_window_ms: int
    max_batch_tokens: int
    length_buckets: list[int]
    def __init__(
        self,
        streams: Optional[int] = ...,
        coalesce_window_ms: Optional[int] = ...,
        max_batch_tokens: Optional[int] = ...,
        ctx: Optional[int] = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...

def env_config_summary() -> dict[str, object]:
    """DEPRECATED: what an EmbeddedEmbedder built with no setting arguments would use."""
    ...
def embed_backend_label() -> str:
    """The COMPILED backend; see EmbeddedEmbedder.runtime_backend for the observed one."""
    ...

# --- m3-embed-llamacpp (embedded backend) ----------------------------------
#
# `EmbeddedEmbedder` is present in every wheel `build_wheel.py` builds (all of
# them enable `embedded`), but not in a plain `cargo`/`maturin` build without
# the feature. Guard with `hasattr(m3_core_rs, "EmbeddedEmbedder")`.

class EmbeddedEmbedder:
    """In-process GGUF embedding model via linked llama.cpp."""

    def __init__(
        self,
        model_path: str,
        warmup: Optional[bool] = None,
        *,
        streams: Optional[int] = None,
        ctx: Optional[int] = None,
        seq_max: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        coalesce_ms: Optional[int] = None,
        max_batch_tokens: Optional[int] = None,
    ) -> None:
        """warmup (default True) loads the model and warms every context now.

        The settings come from m3_config.toml: streams (embed.streams), ctx
        (embed.ctx), seq_max (embed.seq_max), n_batch (embed.n_batch), n_ubatch
        (embed.n_ubatch), coalesce_ms (embed.coalesce_ms), max_batch_tokens
        (embed.max_batch_tokens). One not given falls back to its legacy
        M3_EMBED_* environment name (deprecated, warned once), then the default.
        """
        ...
    def embed(self, texts: list[str]) -> list[list[float]]:
        """One embedding row per input text, each of length embedding_dim()."""
        ...
    def count_tokens(self, texts: list[str]) -> list[int]:
        """Exact token counts, BOS/EOS included."""
        ...
    def token_windows(self, text: str, budget: int, overlap: int) -> list[tuple[int, int]]:
        """Token-exact (start, end) str-index windows of at most `budget` tokens."""
        ...
    def embedding_dim(self) -> int:
        """Embedding dimension reported by the model (forces lazy load)."""
        ...
    def runtime_backend(self) -> str:
        """Observed backend holding the model: cuda | vulkan | metal | cpu."""
        ...
    def streams(self) -> int: ...
