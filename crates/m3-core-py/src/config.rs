//! The in-process embedder's settings, as PARAMETERS from Python.
//!
//! m3 reads every setting from its registry (m3_config.toml and its sources)
//! and passes the values here (m3 memory `9655f30a`: settings reach the core as
//! parameters, not environment). Per setting: the parameter > the default. The
//! legacy `M3_EMBED_*` environment names were read for one release (3.10.17)
//! and are not read any more.

use m3_dispatcher::{BreakerCfg, DispatcherConfig};

/// One setting's parameter name and m3 registry key.
struct Setting {
    param: &'static str,
    key: &'static str,
}

const STREAMS: Setting = Setting {
    param: "streams",
    key: "embed.streams",
};
const COALESCE_MS: Setting = Setting {
    param: "coalesce_ms",
    key: "embed.coalesce_ms",
};
const MAX_BATCH_TOKENS: Setting = Setting {
    param: "max_batch_tokens",
    key: "embed.max_batch_tokens",
};
const CTX: Setting = Setting {
    param: "ctx",
    key: "embed.ctx",
};
const SEQ_MAX: Setting = Setting {
    param: "seq_max",
    key: "embed.seq_max",
};
const N_BATCH: Setting = Setting {
    param: "n_batch",
    key: "embed.n_batch",
};
const N_UBATCH: Setting = Setting {
    param: "n_ubatch",
    key: "embed.n_ubatch",
};

/// `given`, else `default`; a value below `min` is raised to it, and says so.
fn value<T>(s: &Setting, given: Option<T>, default: T, min: T) -> T
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    let Some(v) = given else {
        return default;
    };
    if v < min {
        log::warn!(
            "{}={v} is below the minimum {min} for {}; using {min}",
            s.param,
            s.key
        );
        return min;
    }
    v
}

/// The settings a caller may pass; `None` = not given.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmbedParams {
    pub streams: Option<usize>,
    pub ctx: Option<u32>,
    pub seq_max: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub coalesce_ms: Option<u64>,
    pub max_batch_tokens: Option<usize>,
}

/// The settings in force.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedSettings {
    pub streams: usize,
    pub ctx: u32,
    pub seq_max: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub coalesce_ms: u64,
    pub max_batch_tokens: usize,
}

/// Resolve every setting.
///
/// * `ctx` — total KV-cache token budget per worker context; default 8192 (the
///   BGE-M3 training context), shared across the sequences of one decode call.
/// * `seq_max` — sequences packed into one decode call; default 32.
/// * `n_batch` / `n_ubatch` — default `ctx`. The BERT encoder asserts
///   `n_ubatch >= n_tokens` over the whole batch and a chunk can hold `ctx`
///   tokens, so `ctx` is the only encoder-safe default (an earlier 512 / 2048
///   crashed it on routine inputs); the embed crate floors smaller values.
/// * `max_batch_tokens` — the dispatcher's coalescing ceiling; default
///   `streams * ctx`, so one window can fill every worker's context (a flat
///   2048 under-fed the pool by a factor of `streams`). Bounds host-side queue
///   memory only. `saturating_mul` guards a 32-bit overflow.
pub fn resolve(p: EmbedParams) -> EmbedSettings {
    let streams = value(&STREAMS, p.streams, m3_dispatcher::DEFAULT_STREAMS, 1);
    let ctx = value(
        &CTX,
        p.ctx,
        m3_embed_llamacpp::DEFAULT_N_CTX,
        m3_embed_llamacpp::MIN_N_CTX,
    );
    EmbedSettings {
        streams,
        ctx,
        seq_max: value(&SEQ_MAX, p.seq_max, m3_embed_llamacpp::DEFAULT_SEQ_MAX, 1),
        n_batch: value(&N_BATCH, p.n_batch, ctx, 1),
        n_ubatch: value(&N_UBATCH, p.n_ubatch, ctx, 1),
        coalesce_ms: value(
            &COALESCE_MS,
            p.coalesce_ms,
            m3_dispatcher::DEFAULT_COALESCE_WINDOW_MS,
            0,
        ),
        max_batch_tokens: value(
            &MAX_BATCH_TOKENS,
            p.max_batch_tokens,
            streams.saturating_mul(ctx as usize),
            1,
        ),
    }
}

/// The dispatcher's config from resolved settings; `length_buckets` and the
/// circuit breaker keep their crate defaults.
pub fn dispatcher_config(s: &EmbedSettings) -> DispatcherConfig {
    DispatcherConfig {
        streams: s.streams,
        coalesce_window_ms: s.coalesce_ms,
        max_batch_tokens: s.max_batch_tokens,
        length_buckets: DispatcherConfig::default().length_buckets,
        circuit_breaker: BreakerCfg::default(),
    }
}

/// The ACTIVE content-hash provider. It is chosen at build time (the
/// `non-fips-perf` feature), so it is not a setting and has no m3_config.toml
/// key. `M3_HASH_PROVIDER` was read until 3.10.17 only to warn that it was
/// ignored; it is not read any more.
pub fn hash_provider() -> String {
    m3_hash::active_provider().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A leftover legacy name changes nothing: 3.10.17 read it for one release.
    #[test]
    fn a_leftover_legacy_env_name_is_not_read() {
        std::env::set_var("M3_EMBED_STREAMS", "7");
        let s = resolve(EmbedParams::default());
        std::env::remove_var("M3_EMBED_STREAMS");
        assert_eq!(s.streams, m3_dispatcher::DEFAULT_STREAMS);
    }

    /// Parameters decide without consulting the environment, and the derived
    /// defaults follow the parameters (n_batch/n_ubatch = ctx, max = streams*ctx).
    #[test]
    fn parameters_decide_and_derived_defaults_follow_them() {
        let s = resolve(EmbedParams {
            streams: Some(3),
            ctx: Some(4096),
            ..Default::default()
        });
        assert_eq!((s.streams, s.ctx), (3, 4096));
        assert_eq!((s.n_batch, s.n_ubatch), (4096, 4096));
        assert_eq!(s.max_batch_tokens, 3 * 4096);
        let s = resolve(EmbedParams {
            streams: Some(0),
            ctx: Some(1),
            ..Default::default()
        });
        assert_eq!(s.streams, 1, "below the minimum is raised");
        assert_eq!(s.ctx, m3_embed_llamacpp::MIN_N_CTX);
    }
}
