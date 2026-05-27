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
    let raw = std::fs::read_to_string(path)?;
    let cfg: FileConfig = toml::from_str(&raw)?;
    Ok(cfg)
}

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// Resolve config with priority:
///   env var (M3_EMBED_GGUF)
///   > file value ([embed].gguf in config.toml)
///   > **discovery cascade** (B5: probe common BGE-M3 GGUF locations)
///   > error.
///
/// Returns an error only if all sources fail.
pub fn resolve(file: &FileConfig) -> anyhow::Result<ResolvedConfig> {
    let gguf = env_str("M3_EMBED_GGUF")
        .or_else(|| file.embed.gguf.clone())
        .or_else(discover_gguf)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "M3_EMBED_GGUF is unset, config.toml has no [embed].gguf, \
                 and no BGE-M3 GGUF was found in the standard discovery \
                 paths (LM Studio cache, ~/models, ~/.cache/m3/models). \
                 Set the env var or run `m3-embed-server install` with the \
                 env set."
            )
        })?;

    if !Path::new(&gguf).exists() {
        anyhow::bail!("GGUF path does not exist: {gguf}");
    }

    Ok(ResolvedConfig {
        gguf,
        port: env_parse("M3_EMBED_SERVER_PORT")
            .or(file.embed.port)
            .unwrap_or(8082),
        host: env_str("M3_EMBED_SERVER_HOST")
            .or_else(|| file.embed.host.clone())
            .unwrap_or_else(|| "127.0.0.1".into()),
        streams: env_parse("M3_EMBED_STREAMS")
            .or(file.embed.streams)
            .unwrap_or(2),
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

// ── B5: GGUF discovery cascade ──────────────────────────────────────────────
//
// Probed in order; first existing match wins. Each candidate is a directory
// to glob for any `*bge-m3*.gguf` or `*BGE-M3*.gguf` file. The cascade exists
// so a fresh install on a machine with LM Studio already configured "just
// works" without the operator needing to find and type the GGUF path.
//
// Order matters: LM Studio cache first (most users have BGE-M3 from m3 setup
// or other agent installs), then m3 own cache, then plain ~/models.

/// Discover a BGE-M3 GGUF in standard cache locations. Returns the first
/// matching path, or None if none of the candidate dirs contain one.
fn discover_gguf() -> Option<String> {
    for dir in discovery_candidate_dirs() {
        if let Some(p) = find_bge_m3_gguf_in(&dir) {
            return Some(p.to_string_lossy().into_owned());
        }
    }
    None
}

/// Candidate base directories, in priority order.
///
/// `pub(crate)` so the `doctor` subcommand can show users where it looked.
pub(crate) fn discovery_candidate_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();

    // 1. LM Studio model cache — the most common source on developer boxes.
    //    Both the new ".lmstudio/models" and the legacy "Library" path on
    //    macOS.
    if let Some(h) = home_dir() {
        out.push(h.join(".lmstudio").join("models"));
        #[cfg(target_os = "macos")]
        out.push(h.join("Library").join("Application Support").join("LM Studio").join("models"));
    }

    // 2. m3 own cache (populated by `fetch_sovereign_assets.py`).
    if let Some(h) = home_dir() {
        out.push(h.join(".cache").join("m3").join("models"));
        out.push(h.join(".m3-memory").join("_assets").join("embedder"));
    }

    // 3. Plain ~/models — many people drop GGUFs here.
    if let Some(h) = home_dir() {
        out.push(h.join("models"));
    }

    // 4. Windows-specific: LM Studio default model path.
    #[cfg(windows)]
    {
        if let Ok(profile) = std::env::var("USERPROFILE") {
            out.push(PathBuf::from(profile).join(".lmstudio").join("models"));
        }
    }

    out
}

/// Recursively search `dir` (up to 4 levels deep) for a file whose name
/// case-insensitively contains "bge-m3" or "bge_m3" and ends in `.gguf`.
/// Returns the first match in directory-iteration order; None if no match
/// or if `dir` doesn't exist.
fn find_bge_m3_gguf_in(dir: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, depth: usize, max_depth: usize) -> Option<PathBuf> {
        if depth > max_depth {
            return None;
        }
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            let ftype = entry.file_type().ok()?;
            if ftype.is_dir() {
                if let Some(p) = walk(&path, depth + 1, max_depth) {
                    return Some(p);
                }
            } else if ftype.is_file() {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if name.ends_with(".gguf")
                    && (name.contains("bge-m3") || name.contains("bge_m3"))
                {
                    return Some(path);
                }
            }
        }
        None
    }
    if !dir.is_dir() {
        return None;
    }
    walk(dir, 0, 4)
}

#[cfg(windows)]
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

#[cfg(not(windows))]
fn home_dir() -> Option<PathBuf> {
    home()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_returns_none_when_no_dirs_contain_gguf() {
        // Smoke: if discovery is called with no candidates having a BGE-M3
        // GGUF, it returns None (doesn't panic, doesn't error).
        // We can't easily mock home_dir without conditional compilation, so
        // we just verify the helper handles non-existent dirs gracefully.
        let p = find_bge_m3_gguf_in(Path::new("/this/path/should/not/exist"));
        assert!(p.is_none());
    }

    #[test]
    fn find_bge_m3_gguf_in_tmpdir() {
        // Walks a fresh tmpdir with a planted GGUF.
        let tmp = tempfile::tempdir().expect("create tmpdir");
        let inner = tmp.path().join("subdir");
        std::fs::create_dir(&inner).unwrap();
        let target = inner.join("bge-m3-GGUF-Q4_K_M.gguf");
        std::fs::write(&target, b"fake").unwrap();
        let found = find_bge_m3_gguf_in(tmp.path()).expect("should find planted GGUF");
        assert_eq!(found, target);
    }

    #[test]
    fn find_bge_m3_gguf_ignores_unrelated_files() {
        let tmp = tempfile::tempdir().expect("create tmpdir");
        std::fs::write(tmp.path().join("llama-3-8b.gguf"), b"fake").unwrap();
        std::fs::write(tmp.path().join("bge-m3-readme.txt"), b"fake").unwrap();
        let found = find_bge_m3_gguf_in(tmp.path());
        assert!(found.is_none(), "should not match unrelated files: {found:?}");
    }

    #[test]
    fn candidate_dirs_includes_lmstudio() {
        let dirs = discovery_candidate_dirs();
        assert!(!dirs.is_empty(), "should always produce some candidates");
        let has_lmstudio = dirs.iter().any(|d| {
            d.to_string_lossy().contains(".lmstudio")
        });
        assert!(has_lmstudio, "candidates should include .lmstudio paths: {dirs:?}");
    }
}
