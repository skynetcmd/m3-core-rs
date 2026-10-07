//! Configuration resolution: env vars > config.toml > built-in defaults.
//!
//! Foreground/dev mode typically uses env vars. Service mode (running under
//! LocalSystem) cannot see the operator's user env, so it falls back to the
//! TOML file written by `m3-embed-server install`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedSection {
    pub gguf: Option<String>,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub streams: Option<usize>,
    pub ctx: Option<u32>,
    pub seq_max: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub coalesce_ms: Option<u64>,
    pub max_batch_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileConfig {
    #[serde(default)]
    pub embed: EmbedSection,
}

pub const DEFAULT_PORT: u16 = 8082;
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Where the SUPERVISED service listens. launchd/systemd start it with only
/// `M3_EMBED_GGUF` in its environment, so it binds from config.toml or the
/// defaults; the calling shell's `M3_EMBED_SERVER_*` are not the service's.
#[cfg(not(windows))]
pub fn service_addr() -> (String, u16) {
    let file = load_file_config(&default_config_path()).unwrap_or_default();
    let host = file.embed.host.unwrap_or_else(|| DEFAULT_HOST.into());
    // A wildcard bind is reached on loopback.
    let host = if host == "0.0.0.0" { DEFAULT_HOST.into() } else { host };
    (host, file.embed.port.unwrap_or(DEFAULT_PORT))
}

/// Fully-resolved runtime config (after env/toml/default merge).
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub gguf: String,
    pub port: u16,
    pub host: String,
    pub streams: usize,
    pub n_ctx: u32,
    pub seq_max: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub coalesce_ms: u64,
    pub max_batch_tokens: usize,
}

/// Per-user base directory for config (`$XDG_CONFIG_HOME` or `~/.config` on
/// Linux, `~/Library/Application Support` on macOS). Falls back to the current
/// directory only if `$HOME` is somehow unset.
#[cfg(not(windows))]
fn user_config_base() -> PathBuf {
    if cfg!(target_os = "macos") {
        home()
            .map(|h| h.join("Library").join("Application Support"))
            .unwrap_or_else(|| PathBuf::from("."))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| home().map(|h| h.join(".config")))
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

/// Per-user base directory for logs/state (`$XDG_STATE_HOME` or
/// `~/.local/state` on Linux, `~/Library/Logs` on macOS).
///
/// `allow(dead_code)`: only `default_log_path` calls this, and that in turn is
/// consumed by the Windows service and the macOS launchd agent — not the Linux
/// systemd path (which logs to the journal, no file path). So a Linux build
/// legitimately leaves both unused.
#[cfg(not(windows))]
#[allow(dead_code)]
fn user_state_base() -> PathBuf {
    if cfg!(target_os = "macos") {
        home()
            .map(|h| h.join("Library").join("Logs"))
            .unwrap_or_else(|| PathBuf::from("."))
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| home().map(|h| h.join(".local").join("state")))
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

#[cfg(not(windows))]
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// Default location for the service config file. The Unix service runs as a
/// *per-user* agent (launchd / systemd --user), so config lives under the
/// user's config dir — `%PROGRAMDATA%\m3-embed-server\config.toml` on Windows,
/// `~/.config/m3-embed-server/config.toml` (Linux) or
/// `~/Library/Application Support/m3-embed-server/config.toml` (macOS).
pub fn default_config_path() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var("PROGRAMDATA")
            .unwrap_or_else(|_| "C:\\ProgramData".into());
        PathBuf::from(base).join("m3-embed-server").join("config.toml")
    }
    #[cfg(not(windows))]
    {
        user_config_base().join("m3-embed-server").join("config.toml")
    }
}

/// Default location for the service log file. Per-user on Unix to match the
/// per-user service model (no root-owned `/var/log` write).
///
/// `allow(dead_code)`: consumed by the Windows service (`service.rs`) and the
/// macOS launchd agent (`StandardOutPath`), but not the Linux systemd unit,
/// which logs to the journal — so a Linux build leaves this unused.
#[allow(dead_code)]
pub fn default_log_path() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var("PROGRAMDATA")
            .unwrap_or_else(|_| "C:\\ProgramData".into());
        PathBuf::from(base).join("m3-embed-server").join("service.log")
    }
    #[cfg(not(windows))]
    {
        user_state_base().join("m3-embed-server").join("service.log")
    }
}

pub fn load_file_config(path: &Path) -> anyhow::Result<FileConfig> {
    if !path.exists() {
        return Ok(FileConfig::default());
    }
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read config file {}: {e}", path.display()))?;
    let cfg: FileConfig = toml::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("cannot parse config file {}: {e}", path.display()))?;
    Ok(cfg)
}

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

/// A set but unparseable value is NOT the same as an unset one: say so, then
/// fall back, so a typo (`M3_EMBED_STREAMS=2x`) is not silently the default.
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    let raw = env_str(key)?;
    match raw.parse() {
        Ok(v) => Some(v),
        Err(_) => {
            log::warn!("ignoring {key}={raw:?}: not a valid value; using the configured or default value");
            None
        }
    }
}

/// Default worker-context count, chosen by the COMPILED backend.
///
/// Each stream materialises its own compute graph on first use, sized for a
/// worst-case `n_ctx` batch, and holds it for the process lifetime. Measured
/// 2026-09-16 on bge-m3 at `n_ctx=8192`: **~3.85 GiB per stream on CUDA and
/// ~4.0 GiB on a CPU-only build** — the cost is a property of the model and
/// `n_ctx`, not of the backend, so a CPU box pays exactly as much per stream as
/// a GPU box. It is linear in `streams` and quadratic in `n_ctx`.
///
/// The split is therefore about the HARDWARE the backend implies, not the
/// backend's own cost:
///
/// * **GPU builds (cuda / vulkan / metal) -> 2.** A machine with a discrete GPU
///   or Apple unified memory is very likely to have the headroom, and the second
///   context buys concurrency for bulk ingest.
/// * **CPU-only -> 1.** A CPU-only deployment is the modest-hardware case
///   almost by definition, and ~4 GiB for a second graph is a large fraction of
///   such a machine. Better to ship something that runs than something fast that
///   will not fit.
///
/// ⚠ **This trades THROUGHPUT for memory, it is not a free win.** One context
/// serves one embedding batch at a time, so at `streams = 1` concurrent callers
/// queue rather than running in parallel — most visible during bulk ingest.
/// `queue_depth` on `/metrics` is the signal: sustained non-zero means callers
/// are waiting and the host would benefit from another stream if it has ~4 GiB
/// spare.
///
/// ⚠ This is a DEFAULT, not a cap. `M3_EMBED_STREAMS` and the `[embed].streams`
/// key both still win, so a CPU box with plenty of RAM can raise it and a small
/// GPU box can lower it. The resolved value is logged at startup.
const fn default_streams() -> usize {
    if cfg!(any(
        feature = "embedded-cuda",
        feature = "embedded-vulkan",
        feature = "embedded-metal"
    )) {
        // GPU build (cuda / vulkan / metal): 2 contexts.
        2
    } else {
        // CPU-ONLY build: 1 context. This is the modest-hardware case, and a
        // second compute graph costs ~4 GiB it probably does not have.
        1
    }
}

/// Resolve config with priority: env var > file value > default.
/// Returns an error only if `gguf` is unresolved (it has no default).
pub fn resolve(file: &FileConfig) -> anyhow::Result<ResolvedConfig> {
    let (gguf, source) = match env_str("M3_EMBED_GGUF") {
        Some(g) => (g, "M3_EMBED_GGUF".to_string()),
        None => match file.embed.gguf.clone() {
            Some(g) => (g, format!("[embed].gguf in {}", default_config_path().display())),
            None => anyhow::bail!(
                "M3_EMBED_GGUF is unset and {} has no [embed].gguf — set the env \
                 var, or run `m3-embed-server install` with it set",
                default_config_path().display()
            ),
        },
    };

    if !Path::new(&gguf).exists() {
        anyhow::bail!("GGUF path does not exist: {gguf} (from {source})");
    }

    Ok(ResolvedConfig {
        gguf,
        port: env_parse("M3_EMBED_SERVER_PORT")
            .or(file.embed.port)
            .unwrap_or(DEFAULT_PORT),
        host: env_str("M3_EMBED_SERVER_HOST")
            .or_else(|| file.embed.host.clone())
            .unwrap_or_else(|| DEFAULT_HOST.into()),
        streams: env_parse("M3_EMBED_STREAMS")
            .or(file.embed.streams)
            .unwrap_or_else(default_streams),
        n_ctx: env_parse("M3_EMBED_CTX").or(file.embed.ctx).unwrap_or(8192),
        seq_max: env_parse("M3_EMBED_SEQ_MAX")
            .or(file.embed.seq_max)
            .unwrap_or(32),
        n_batch: env_parse("M3_EMBED_N_BATCH")
            .or(file.embed.n_batch)
            .unwrap_or(2048),
        n_ubatch: env_parse("M3_EMBED_N_UBATCH")
            .or(file.embed.n_ubatch)
            .unwrap_or(512),
        coalesce_ms: env_parse("M3_EMBED_COALESCE_MS")
            .or(file.embed.coalesce_ms)
            .unwrap_or(3),
        max_batch_tokens: env_parse("M3_EMBED_MAX_BATCH_TOKENS")
            .or(file.embed.max_batch_tokens)
            .unwrap_or(2048),
    })
}

/// Capture the current shell's env vars and serialize them as a starter
/// config.toml. Called by the Windows `install` subcommand so the SYSTEM-account
/// service inherits the operator's intended settings.
///
/// `allow(dead_code)`: Windows-service-only. The Unix installers
/// (`service_unix`) pin `M3_EMBED_GGUF` directly into the launchd plist /
/// systemd unit's `Environment=`, so they need no config.toml snapshot — this
/// function is legitimately unused on a Unix build.
#[allow(dead_code)]
pub fn snapshot_env_to_file() -> FileConfig {
    FileConfig {
        embed: EmbedSection {
            gguf: env_str("M3_EMBED_GGUF"),
            port: env_parse("M3_EMBED_SERVER_PORT"),
            host: env_str("M3_EMBED_SERVER_HOST"),
            streams: env_parse("M3_EMBED_STREAMS"),
            ctx: env_parse("M3_EMBED_CTX"),
            seq_max: env_parse("M3_EMBED_SEQ_MAX"),
            n_batch: env_parse("M3_EMBED_N_BATCH"),
            n_ubatch: env_parse("M3_EMBED_N_UBATCH"),
            coalesce_ms: env_parse("M3_EMBED_COALESCE_MS"),
            max_batch_tokens: env_parse("M3_EMBED_MAX_BATCH_TOKENS"),
        },
    }
}

/// `allow(dead_code)`: paired with `snapshot_env_to_file` — Windows-service-only
/// (the Unix installers pin config into the unit file instead).
#[allow(dead_code)]
pub fn write_config_file(path: &Path, cfg: &FileConfig) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = toml::to_string_pretty(cfg)?;
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The backend-dependent default must actually DIFFER by backend.
    ///
    /// A `cfg!` chain that silently always takes one arm looks correct, compiles
    /// clean, and quietly ships the wrong default to half the fleet. This pins
    /// the value for whichever backend the test binary was built with, so a
    /// mis-edited feature name fails here rather than in a user's RAM.
    #[test]
    fn default_streams_matches_the_compiled_backend() {
        let expected = if cfg!(any(
            feature = "embedded-cuda",
            feature = "embedded-vulkan",
            feature = "embedded-metal"
        )) {
            2 // GPU build: headroom assumed
        } else {
            1 // CPU-only: ~4 GiB per extra graph is too much for modest hardware
        };
        assert_eq!(
            default_streams(),
            expected,
            "default_streams() disagrees with the compiled backend"
        );
    }

    /// A CPU-only box with spare RAM must still be able to ask for more.
    /// The default is a floor for the common case, never a cap.
    #[test]
    fn file_value_overrides_the_backend_default() {
        let (file, _guard) = file_config_with_real_gguf(Some(4));
        let resolved = temp_env_without_streams(|| resolve(&file).expect("resolve"));
        assert_eq!(resolved.streams, 4, "[embed].streams must beat the default");
    }

    /// Guard the *other* direction too: with nothing set anywhere, the resolved
    /// value is the backend default rather than an accidental hardcode.
    #[test]
    fn unset_falls_back_to_the_backend_default() {
        let (file, _guard) = file_config_with_real_gguf(None);
        let resolved = temp_env_without_streams(|| resolve(&file).expect("resolve"));
        assert_eq!(resolved.streams, default_streams());
    }

    /// `resolve()` rejects a non-existent GGUF, so these tests need a real file.
    /// Returns the config plus a guard whose Drop removes the temp file — the
    /// path must stay alive for the duration of the call.
    fn file_config_with_real_gguf(streams: Option<usize>) -> (FileConfig, TempGguf) {
        let guard = TempGguf::new();
        let mut file = FileConfig::default();
        file.embed.gguf = Some(guard.path.to_string_lossy().into_owned());
        file.embed.streams = streams;
        (file, guard)
    }

    struct TempGguf {
        path: PathBuf,
    }

    impl TempGguf {
        fn new() -> Self {
            // Unique per test: the suite runs threads in one process.
            let path = std::env::temp_dir().join(format!(
                "m3-streams-test-{}-{:?}.gguf",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::fs::write(&path, b"not a real gguf; resolve() only stats it")
                .expect("write temp gguf");
            Self { path }
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Run `f` with `M3_EMBED_STREAMS` removed, restoring it afterwards.
    ///
    /// Tests share a process, so a stray env var from the developer's shell
    /// would make these pass or fail for the wrong reason.
    fn temp_env_without_streams<T>(f: impl FnOnce() -> T) -> T {
        let saved = std::env::var("M3_EMBED_STREAMS").ok();
        std::env::remove_var("M3_EMBED_STREAMS");
        let out = f();
        if let Some(v) = saved {
            std::env::set_var("M3_EMBED_STREAMS", v);
        }
        out
    }
}
