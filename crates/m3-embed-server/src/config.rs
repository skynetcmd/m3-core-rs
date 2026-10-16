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
    // Transport security (src/security.rs). File-only, written by
    // `m3-embed-server secure`: no environment variable opens or changes the
    // listener for other machines.
    pub remote_listen: Option<String>,
    pub tls_cert: Option<String>,
    pub tls_key: Option<String>,
    pub token_sha256: Option<String>,
    // m3's crypto.fips_mode / fips_strict, for a service that starts without
    // the caller's environment (captured from M3_FIPS_MODE / M3_FIPS_STRICT).
    pub fips_mode: Option<bool>,
    pub fips_strict: Option<bool>,
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
    pub security: crate::security::SecurityConfig,
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

static CONFIG_OVERRIDE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// `--config <path>`: use this file instead of the default. Foreground and
/// `secure` only (main.rs refuses it elsewhere), so a test or a second instance
/// never reads or rewrites the installed service's file.
pub fn set_config_override(path: PathBuf) {
    let _ = CONFIG_OVERRIDE.set(path);
}

/// The four keys that make up the listener for other machines.
#[derive(Debug, Default)]
pub struct SecurityKeys<'a> {
    pub remote_listen: Option<&'a str>,
    pub tls_cert: Option<&'a str>,
    pub tls_key: Option<&'a str>,
    pub token_sha256: Option<&'a str>,
}

/// Set (`Some`) or remove (`None`) the transport-security keys in `[embed]`,
/// editing the file in place so every comment and other key survives.
pub fn write_security_keys(path: &Path, keys: &SecurityKeys) -> anyhow::Result<()> {
    let raw = if path.exists() { std::fs::read_to_string(path)? } else { String::new() };
    let mut doc: toml_edit::DocumentMut = raw
        .parse()
        .map_err(|e| anyhow::anyhow!("cannot parse config file {}: {e}", path.display()))?;
    if !doc.contains_table("embed") {
        doc["embed"] = toml_edit::table();
    }
    let embed = doc["embed"]
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("[embed] in {} is not a table", path.display()))?;
    for (k, v) in [
        ("remote_listen", keys.remote_listen),
        ("tls_cert", keys.tls_cert),
        ("tls_key", keys.tls_key),
        ("token_sha256", keys.token_sha256),
    ] {
        match v {
            Some(v) => embed[k] = toml_edit::value(v),
            None => {
                embed.remove(k);
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, doc.to_string())?;
    Ok(())
}

/// Default location for the service config file. The Unix service runs as a
/// *per-user* agent (launchd / systemd --user), so config lives under the
/// user's config dir — `%PROGRAMDATA%\m3-embed-server\config.toml` on Windows,
/// `~/.config/m3-embed-server/config.toml` (Linux) or
/// `~/Library/Application Support/m3-embed-server/config.toml` (macOS).
pub fn default_config_path() -> PathBuf {
    if let Some(p) = CONFIG_OVERRIDE.get() {
        return p.clone();
    }
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
        security: security_config(&file.embed),
    })
}

/// The transport-security settings: TLS and token from the file only, the FIPS
/// flags from m3's environment variables first, then the file.
pub fn security_config(e: &EmbedSection) -> crate::security::SecurityConfig {
    crate::security::SecurityConfig {
        remote_listen: e.remote_listen.clone(),
        tls_cert: e.tls_cert.clone().map(PathBuf::from),
        tls_key: e.tls_key.clone().map(PathBuf::from),
        token_sha256: e.token_sha256.clone(),
        fips_mode: env_flag("M3_FIPS_MODE").or(e.fips_mode).unwrap_or(false),
        fips_strict: env_flag("M3_FIPS_STRICT").or(e.fips_strict).unwrap_or(false),
    }
}

/// m3's boolean spelling (1/true/yes/on, 0/false/no/off). A set value that is
/// neither is said, then treated as unset.
fn env_flag(key: &str) -> Option<bool> {
    let raw = env_str(key)?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => {
            log::warn!("ignoring {key}={raw:?}: not a boolean; using the configured or default value");
            None
        }
    }
}

/// Capture the current shell's env vars and serialize them as a starter
/// config.toml. Called by the Windows `install` subcommand so the SYSTEM-account
/// service inherits the operator's intended settings.
///
/// Windows writes it whole; the Unix installers merge it into the file
/// (`merge_env_into_file`), since their unit pins only `M3_EMBED_GGUF`.
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
            remote_listen: None,
            tls_cert: None,
            tls_key: None,
            token_sha256: None,
            fips_mode: env_flag("M3_FIPS_MODE"),
            fips_strict: env_flag("M3_FIPS_STRICT"),
        },
    }
}

/// Record the settings given to `install` in config.toml, keeping any value the
/// file already has that the environment does not set. The Unix installers pin
/// only M3_EMBED_GGUF into the unit, and a supervised service starts without the
/// caller's environment, so a port (or host, streams...) handed to `install` was
/// dropped and the service bound the default.
#[cfg_attr(windows, allow(dead_code))] // the Unix installers call it
pub fn merge_env_into_file(path: &Path) -> anyhow::Result<()> {
    let mut file = load_file_config(path).unwrap_or_default();
    let env = snapshot_env_to_file().embed;
    let e = &mut file.embed;
    e.gguf = env.gguf.or(e.gguf.take());
    e.port = env.port.or(e.port);
    e.host = env.host.or(e.host.take());
    e.streams = env.streams.or(e.streams);
    e.ctx = env.ctx.or(e.ctx);
    e.seq_max = env.seq_max.or(e.seq_max);
    e.n_batch = env.n_batch.or(e.n_batch);
    e.n_ubatch = env.n_ubatch.or(e.n_ubatch);
    e.coalesce_ms = env.coalesce_ms.or(e.coalesce_ms);
    e.max_batch_tokens = env.max_batch_tokens.or(e.max_batch_tokens);
    e.fips_mode = env.fips_mode.or(e.fips_mode);
    e.fips_strict = env.fips_strict.or(e.fips_strict);
    write_config_file(path, &file)
}

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
    /// A port given to `install` reaches config.toml, where the supervised
    /// service reads it; values already in the file and not in the env stay.
    #[test]
    fn install_records_a_given_port_and_keeps_file_values() {
        let dir = std::env::temp_dir().join(format!("m3-embed-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "[embed]\ngguf = \"/m/bge.gguf\"\nstreams = 3\n").unwrap();
        let saved: Vec<(&str, Option<String>)> = ["M3_EMBED_SERVER_PORT", "M3_EMBED_GGUF", "M3_EMBED_STREAMS"]
            .iter()
            .map(|k| (*k, std::env::var(k).ok()))
            .collect();
        for (k, _) in &saved {
            std::env::remove_var(k);
        }
        std::env::set_var("M3_EMBED_SERVER_PORT", "8083");
        merge_env_into_file(&path).unwrap();
        for (k, v) in saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        let cfg = load_file_config(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(cfg.embed.port, Some(8083));
        assert_eq!(cfg.embed.gguf.as_deref(), Some("/m/bge.gguf"));
        assert_eq!(cfg.embed.streams, Some(3));
    }

    /// `secure` edits the operator's file: comments and other keys survive,
    /// Windows paths round-trip, and `--off` removes exactly the three keys.
    #[test]
    fn security_keys_are_edited_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "[embed]\n# why one stream: measured\nstreams = 1\nport = 8082\n";
        std::fs::write(&path, original).unwrap();
        let cert = r"C:\Users\x\.m3\config\tls\cert.pem";
        let digest = "a".repeat(64);
        let keys = SecurityKeys {
            remote_listen: Some("100.100.50.63"),
            tls_cert: Some(cert),
            tls_key: Some("/k.pem"),
            token_sha256: Some(&digest),
        };
        write_security_keys(&path, &keys).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# why one stream: measured"), "{text}");
        let cfg = load_file_config(&path).unwrap();
        assert_eq!(cfg.embed.tls_cert.as_deref(), Some(cert));
        assert_eq!(cfg.embed.remote_listen.as_deref(), Some("100.100.50.63"));
        assert_eq!(cfg.embed.streams, Some(1));
        assert_eq!(cfg.embed.token_sha256.as_deref(), Some(digest.as_str()));

        write_security_keys(&path, &SecurityKeys::default()).unwrap();
        let cfg = load_file_config(&path).unwrap();
        assert!(cfg.embed.remote_listen.is_none() && cfg.embed.tls_cert.is_none());
        assert!(cfg.embed.tls_key.is_none() && cfg.embed.token_sha256.is_none());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    }

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
