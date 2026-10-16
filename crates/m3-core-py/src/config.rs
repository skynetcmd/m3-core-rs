//! The single home for `M3_*` env-var reading (plan §9.6). Generic crates take
//! typed configs; this module is the only place `std::env::var` is called.
//! Precedence is kwarg > env > default (§9.7) — kwarg handling lives in the
//! `#[pymethods]` constructors; this module supplies the env-or-default layer.

use std::env;

use m3_dispatcher::{BreakerCfg, DispatcherConfig};

/// Parse an `M3_*` env var into `T`, falling back to `default` when unset,
/// and raising a value below `min` to `min`. Every other outcome is reported
/// with the key, the raw value and what was used instead: an unparseable
/// value, a non-UTF-8 one, or one below the minimum must not read as the
/// value the operator set.
fn env_or<T>(key: &str, default: T, min: T) -> T
where
    T: std::str::FromStr + PartialOrd + std::fmt::Display + Copy,
    T::Err: std::fmt::Display,
{
    let raw = match env::var(key) {
        Ok(raw) if !raw.is_empty() => raw,
        Ok(_) | Err(env::VarError::NotPresent) => return default,
        Err(env::VarError::NotUnicode(os)) => {
            log::warn!("ignoring {key}={os:?}: not valid UTF-8; using the default {default}");
            return default;
        }
    };
    match raw.trim().parse::<T>() {
        Ok(v) if v < min => {
            log::warn!("{key}={raw:?} is below the minimum {min}; using {min}");
            min
        }
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "ignoring {key}={raw:?}: not a valid {} ({e}); using the default {default}",
                std::any::type_name::<T>()
            );
            default
        }
    }
}

/// `M3_EMBED_STREAMS` — concurrent dispatcher streams.
pub fn embed_streams() -> usize {
    env_or("M3_EMBED_STREAMS", m3_dispatcher::DEFAULT_STREAMS, 1)
}

/// `M3_EMBED_COALESCE_MS` — coalescing-window width in milliseconds.
pub fn embed_coalesce_ms() -> u64 {
    env_or(
        "M3_EMBED_COALESCE_MS",
        m3_dispatcher::DEFAULT_COALESCE_WINDOW_MS,
        0,
    )
}

/// `M3_EMBED_MAX_BATCH_TOKENS` — the dispatcher's coalescing ceiling: how many
/// tokens' worth of separate jobs are packed into one batch before it is
/// flushed to the worker pool. Defaults to `M3_EMBED_STREAMS * M3_EMBED_CTX` so
/// a single coalescing window can fill *every* worker's context at once — the
/// earlier flat 2048 default capped a batch at roughly one context's worth,
/// under-feeding the `streams`-wide pool by a factor of `streams`. This knob
/// bounds host-side queue memory only; per-worker GPU KV-cache stays bounded by
/// `M3_EMBED_CTX` regardless. `saturating_mul` guards the (unrealistic) overflow
/// case on 32-bit targets.
pub fn embed_max_batch_tokens() -> usize {
    let default = embed_streams().saturating_mul(embed_ctx() as usize);
    env_or("M3_EMBED_MAX_BATCH_TOKENS", default, 1)
}

/// `M3_EMBED_CTX` — total KV-cache token budget per worker context.
/// Defaults to 8192 (BGE-M3 training context). Shared across all sequences in
/// one decode call: effective per-sequence capacity = M3_EMBED_CTX / M3_EMBED_SEQ_MAX.
/// Texts longer than that per-seq limit are decoded alone (using the full budget).
/// KV memory per worker ≈ M3_EMBED_CTX × 96 KB (BGE-M3 fp16 KV estimate).
pub fn embed_ctx() -> u32 {
    env_or(
        "M3_EMBED_CTX",
        m3_embed_llamacpp::DEFAULT_N_CTX,
        m3_embed_llamacpp::MIN_N_CTX,
    )
}

/// `M3_EMBED_SEQ_MAX` — max sequences packed into one llama.cpp decode call per
/// worker context. Higher values amortise the per-decode overhead across more
/// texts; lower values reduce KV-cache memory per context. Default: 32.
/// Each worker context's total token budget per decode is capped by `M3_EMBED_CTX`
/// regardless of this setting, so effective seqs/decode = min(seq_max, ctx/avg_tokens).
pub fn embed_seq_max() -> u32 {
    env_or("M3_EMBED_SEQ_MAX", m3_embed_llamacpp::DEFAULT_SEQ_MAX, 1)
}

/// `M3_EMBED_N_BATCH` — llama.cpp's prompt-process batch ceiling. Defaults to
/// `M3_EMBED_CTX` (8192). It must be >= `n_ubatch`, and `n_ubatch` must cover
/// the largest decode chunk — which `embed_on_ctx` bounds at `n_ctx` — so the
/// only encoder-safe default is `n_batch == n_ctx`. An earlier 2048 default
/// (decoupled from `n_ctx`) crashed the BERT encoder's
/// `GGML_ASSERT(n_ubatch >= n_tokens)` on routine BGE-M3 inputs. Operators may
/// still raise it above `n_ctx`; the embed crate floors any smaller value.
pub fn embed_n_batch() -> u32 {
    env_or("M3_EMBED_N_BATCH", embed_ctx(), 1)
}

/// `M3_EMBED_N_UBATCH` — llama.cpp's micro-batch (the inner SIMD tile size).
/// Defaults to `M3_EMBED_CTX` (8192). The BERT encoder asserts `n_ubatch >=
/// n_tokens` for the whole batch and a decode chunk can be up to `n_ctx`
/// tokens, so `n_ubatch` below `n_ctx` is unsafe — the embed crate floors it
/// up to `n_ctx`. The earlier 512 default was the source of the encoder crash
/// on any text/chunk over 512 tokens.
pub fn embed_n_ubatch() -> u32 {
    env_or("M3_EMBED_N_UBATCH", embed_ctx(), 1)
}

/// The ACTIVE content-hash provider. It is chosen at build time (the
/// `non-fips-perf` feature), so `M3_HASH_PROVIDER` cannot change it; echoing
/// the env var reported a provider that was not in use. A request for a
/// different one is warned about.
pub fn hash_provider() -> String {
    let active = m3_hash::active_provider();
    if let Ok(wanted) = env::var("M3_HASH_PROVIDER") {
        if !wanted.is_empty() && !active.starts_with(wanted.trim()) {
            log::warn!(
                "M3_HASH_PROVIDER={wanted:?} is ignored: the hash provider is fixed when the wheel \
                 is built, and this one uses {active:?}"
            );
        }
    }
    active.to_string()
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
