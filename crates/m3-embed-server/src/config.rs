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

/// Default location for the service config file:
/// `%PROGRAMDATA%\m3-embed-server\config.toml` on Windows,
/// `/etc/m3-embed-server/config.toml` elsewhere (rarely used; foreground only).
pub fn default_config_path() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("PROGRAMDATA")
            .unwrap_or_else(|_| "C:\\ProgramData".into());
        PathBuf::from(base).join("m3-embed-server").join("config.toml")
    } else {
        PathBuf::from("/etc/m3-embed-server/config.toml")
    }
}

/// Default location for the service log file.
pub fn default_log_path() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("PROGRAMDATA")
            .unwrap_or_else(|_| "C:\\ProgramData".into());
        PathBuf::from(base).join("m3-embed-server").join("service.log")
    } else {
        PathBuf::from("/var/log/m3-embed-server.log")
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

/// Resolve config with priority: env var > file value > default.
/// Returns an error only if `gguf` is unresolved (it has no default).
pub fn resolve(file: &FileConfig) -> anyhow::Result<ResolvedConfig> {
    let gguf = env_str("M3_EMBED_GGUF")
        .or_else(|| file.embed.gguf.clone())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "M3_EMBED_GGUF is unset and config.toml has no [embed].gguf — \
                 set the env var or run `m3-embed-server install` with the env set."
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
/// config.toml. Called by the `install` subcommand so the SYSTEM-account
/// service inherits the operator's intended settings.
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

pub fn write_config_file(path: &Path, cfg: &FileConfig) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = toml::to_string_pretty(cfg)?;
    std::fs::write(path, s)?;
    Ok(())
}
