//! The in-process embedder's settings, as PARAMETERS from Python.
//!
//! m3 reads every setting from its registry (m3_config.toml and its sources)
//! and passes the values here (m3 memory `9655f30a`: settings reach the core as
//! parameters, not environment). Per setting: the parameter > its legacy
//! `M3_*` environment name > the default. The environment name is honoured for
//! one release when the caller passes nothing, with one warning naming the
//! parameter and the m3_config.toml key that replace it (owner of that read
//! and warning: `m3_config::deprecated_env`).

use m3_dispatcher::{BreakerCfg, DispatcherConfig};

/// One setting's parameter name, legacy environment name and m3 registry key.
struct Setting {
    param: &'static str,
    env: &'static str,
    key: &'static str,
}

const STREAMS: Setting = Setting {
    param: "streams",
    env: "M3_EMBED_STREAMS",
    key: "embed.streams",
};
const COALESCE_MS: Setting = Setting {
    param: "coalesce_ms",
    env: "M3_EMBED_COALESCE_MS",
    key: "embed.coalesce_ms",
};
const MAX_BATCH_TOKENS: Setting = Setting {
    param: "max_batch_tokens",
    env: "M3_EMBED_MAX_BATCH_TOKENS",
    key: "embed.max_batch_tokens",
};
const CTX: Setting = Setting {
    param: "ctx",
    env: "M3_EMBED_CTX",
    key: "embed.ctx",
};
const SEQ_MAX: Setting = Setting {
    param: "seq_max",
    env: "M3_EMBED_SEQ_MAX",
    key: "embed.seq_max",
};
const N_BATCH: Setting = Setting {
    param: "n_batch",
    env: "M3_EMBED_N_BATCH",
    key: "embed.n_batch",
};
const N_UBATCH: Setting = Setting {
    param: "n_ubatch",
    env: "M3_EMBED_N_UBATCH",
    key: "embed.n_ubatch",
};

/// `given`, else the legacy environment value, else `default`; a value below
/// `min` is raised to it. An unparseable environment value is reported with the
/// name, the raw value and what was used instead: it must not read as the value
/// the operator set.
fn value<T>(s: &Setting, given: Option<T>, default: T, min: T) -> T
where
    T: std::str::FromStr + PartialOrd + std::fmt::Display + Copy,
    T::Err: std::fmt::Display,
{
    let (v, from) = match given {
        Some(v) => (v, format!("{}=", s.param)),
        None => match m3_config::deprecated_env(s.env, s.param, s.key) {
            None => return default,
            Some(raw) => match raw.trim().parse::<T>() {
                Ok(v) => (v, s.env.to_string()),
                Err(e) => {
                    log::warn!(
                        "ignoring {}={raw:?}: not a valid {} ({e}); using the default {default}",
                        s.env,
                        std::any::type_name::<T>()
                    );
                    return default;
                }
            },
        },
    };
    if v < min {
        log::warn!(
            "{from}{v} is below the minimum {min} for {}; using {min}",
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
/// key; `M3_HASH_PROVIDER` is read only to say that a request for another
/// provider is ignored (deprecated: no longer read after this release).
pub fn hash_provider() -> String {
    let active = m3_hash::active_provider();
    if let Ok(wanted) = std::env::var("M3_HASH_PROVIDER") {
        if !wanted.is_empty() && !active.starts_with(wanted.trim()) {
            log::warn!(
                "M3_HASH_PROVIDER={wanted:?} is ignored: the hash provider is fixed when the wheel \
                 is built, and this one uses {active:?}. The variable is deprecated and stops being \
                 read in the next release; unset it"
            );
        }
    }
    active.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
