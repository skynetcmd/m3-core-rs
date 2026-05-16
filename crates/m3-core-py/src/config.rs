//! The single home for `M3_*` env-var reading (plan §9.6). Generic crates take
//! typed configs; this module is the only place `std::env::var` is called.
//! Precedence is kwarg > env > default (§9.7) — kwarg handling lives in the
//! `#[pymethods]` constructors; this module supplies the env-or-default layer.

use std::env;

use m3_dispatcher::{BreakerCfg, DispatcherConfig};

/// Parse an `M3_*` env var into `T`, falling back to `default` when unset or
/// unparseable.
fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `M3_EMBED_STREAMS` — concurrent dispatcher streams.
pub fn embed_streams() -> usize {
    env_or("M3_EMBED_STREAMS", 8)
}

/// `M3_EMBED_COALESCE_MS` — coalescing-window width in milliseconds.
pub fn embed_coalesce_ms() -> u64 {
    env_or("M3_EMBED_COALESCE_MS", 3)
}

/// `M3_EMBED_MAX_BATCH_TOKENS` — per-batch token ceiling.
pub fn embed_max_batch_tokens() -> usize {
    env_or("M3_EMBED_MAX_BATCH_TOKENS", 2048)
}

/// `M3_EMBED_CTX` — total KV-cache token budget per worker context.
/// Defaults to 8192 (BGE-M3 training context). Shared across all sequences in
/// one decode call: effective per-sequence capacity = M3_EMBED_CTX / M3_EMBED_SEQ_MAX.
/// Texts longer than that per-seq limit are decoded alone (using the full budget).
/// KV memory per worker ≈ M3_EMBED_CTX × 96 KB (BGE-M3 fp16 KV estimate).
pub fn embed_ctx() -> u32 {
    env_or("M3_EMBED_CTX", 8192u32)
}

/// `M3_EMBED_SEQ_MAX` — max sequences packed into one llama.cpp decode call per
/// worker context. Higher values amortise the per-decode overhead across more
/// texts; lower values reduce KV-cache memory per context. Default: 32.
/// Each worker context's total token budget per decode is capped by `M3_EMBED_CTX`
/// regardless of this setting, so effective seqs/decode = min(seq_max, ctx/avg_tokens).
pub fn embed_seq_max() -> u32 {
    env_or("M3_EMBED_SEQ_MAX", 32u32)
}

/// `M3_EMBED_N_BATCH` — llama.cpp's prompt-process batch ceiling (wave-3 fix #1).
/// Decoupled from `n_ctx` so an over-provisioned KV budget doesn't force a
/// proportionally over-provisioned compute tile. Default: 2048 (the value that
/// gives the best throughput-per-allocator-pressure tradeoff for BGE-M3 sized
/// inputs in the wave-0 baseline). Clamped to `[1, n_ctx]` at the embed crate.
pub fn embed_n_batch() -> u32 {
    env_or("M3_EMBED_N_BATCH", 2048u32)
}

/// `M3_EMBED_N_UBATCH` — llama.cpp's micro-batch (the inner SIMD tile size,
/// wave-3 fix #1). Smaller than `n_batch`; tuned independently. Default: 512.
/// Clamped to `[1, n_batch]` at the embed crate.
pub fn embed_n_ubatch() -> u32 {
    env_or("M3_EMBED_N_UBATCH", 512u32)
}

/// `M3_HASH_PROVIDER` — preferred hash backend label.
pub fn hash_provider() -> String {
    env::var("M3_HASH_PROVIDER").unwrap_or_else(|_| "ring".to_string())
}

/// Build a `DispatcherConfig` from the env layer; `length_buckets` and the
/// circuit breaker keep their crate defaults (no env var defined for them yet).
pub fn dispatcher_config_from_env() -> DispatcherConfig {
    DispatcherConfig {
        streams: embed_streams(),
        coalesce_window_ms: embed_coalesce_ms(),
        max_batch_tokens: embed_max_batch_tokens(),
        length_buckets: DispatcherConfig::default().length_buckets,
        circuit_breaker: BreakerCfg::default(),
    }
}
