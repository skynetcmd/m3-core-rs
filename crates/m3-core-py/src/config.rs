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
