//! The server's settings, from m3's two config files.
//!
//! m3_config.toml and the secrets file it names are the server's only config
//! files (m3 memory `9655f30a`); `m3_config` locates and parses them. This
//! module says WHICH keys the server reads, their defaults, and what a
//! supervised service will see. Precedence per key, m3's:
//! `--set` > m3_config.toml > the default. A former environment name is not
//! read; one still set is logged once at startup.
//!
//! A service is started with `--config <m3_config.toml>` (`install` pins it):
//! the Windows service runs as LocalSystem, and launchd / systemd start it
//! with a bare environment, so neither can find the user's root on its own.
//!
//! The server's own `config.toml` from earlier versions is no longer read; if
//! it is still there, one warning says so ([`warn_legacy_file`]). m3 migrates
//! its values and renames it to `config.toml.migrated` (unread; kept for a
//! rollback to an older core).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use m3_config::{secrets, Config, Key, Overrides, ProcessEnv, Source};

// ── The keys the server reads (m3 registry names) ───────────────────────────
pub const GGUF: Key = Key::path("embed.gguf", &["M3_EMBED_GGUF"]);
pub const HOST: Key = Key::string(
    "embed.server.host",
    &["M3_EMBED_SERVER_HOST", "EMBED_SERVER_HOST"],
);
pub const PORT: Key = Key::int("embed.server.port", &["M3_EMBED_SERVER_PORT"], 1, 65535);
pub const CTX: Key = Key::int("embed.ctx", &["M3_EMBED_CTX"], 1, u32::MAX as i64);
pub const STREAMS: Key = Key::int("embed.server.streams", &["M3_EMBED_STREAMS"], 1, 4096);
pub const SEQ_MAX: Key = Key::int(
    "embed.server.seq_max",
    &["M3_EMBED_SEQ_MAX"],
    1,
    u32::MAX as i64,
);
pub const N_BATCH: Key = Key::int(
    "embed.server.n_batch",
    &["M3_EMBED_N_BATCH"],
    1,
    u32::MAX as i64,
);
pub const N_UBATCH: Key = Key::int(
    "embed.server.n_ubatch",
    &["M3_EMBED_N_UBATCH"],
    1,
    u32::MAX as i64,
);
pub const COALESCE_MS: Key = Key::int(
    "embed.server.coalesce_ms",
    &["M3_EMBED_COALESCE_MS"],
    0,
    i64::MAX,
);
pub const MAX_BATCH_TOKENS: Key = Key::int(
    "embed.server.max_batch_tokens",
    &["M3_EMBED_MAX_BATCH_TOKENS"],
    1,
    i64::MAX,
);
// The listener for other machines: no environment name opens or changes it.
pub const REMOTE_LISTEN: Key = Key::string("embed.server.remote_listen", &[]);
pub const TLS_CERT: Key = Key::path("embed.server.tls_cert", &[]);
pub const TLS_KEY: Key = Key::path("embed.server.tls_key", &[]);
// The FIPS keys and the tier rule belong to the security seam (m3_config::security).
pub use m3_config::security::{FIPS_MODE, FIPS_STRICT};

/// The SHA-256 of the bearer token, in the SECRETS file (not m3_config.toml: a
/// digest of a guessable token can be brute-forced offline). A `[secrets]`
/// entry name, so in m3's flat NAME style.
pub const TOKEN_SECRET: &str = "M3_EMBED_SERVER_TOKEN_SHA256";

/// Every key the server reads; `--set` accepts these (not `secrets.file`, a
/// locator). m3's registry must define each.
pub const KEYS: [Key; 16] = [
    GGUF,
    HOST,
    PORT,
    CTX,
    STREAMS,
    SEQ_MAX,
    N_BATCH,
    N_UBATCH,
    COALESCE_MS,
    MAX_BATCH_TOKENS,
    REMOTE_LISTEN,
    TLS_CERT,
    TLS_KEY,
    FIPS_MODE,
    FIPS_STRICT,
    secrets::SECRETS_FILE,
];

pub const DEFAULT_PORT: u16 = 8082;
pub const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_CTX: u32 = 8192;
const DEFAULT_SEQ_MAX: u32 = 32;
const DEFAULT_COALESCE_MS: u64 = 3;
const DEFAULT_MAX_BATCH_TOKENS: usize = 2048;

// ── The command line's global options ───────────────────────────────────────
/// `--config <file>` and `--set={...}`, given before the subcommand.
#[derive(Debug, Default, Clone)]
pub struct Args {
    pub config: Option<PathBuf>,
    pub set: Option<String>,
}

static ARGS: OnceLock<Args> = OnceLock::new();

/// Record the global options once, at startup.
pub fn set_args(args: Args) {
    let _ = ARGS.set(args);
}

fn args() -> Args {
    ARGS.get().cloned().unwrap_or_default()
}

/// Whether `--config` was given.
pub fn config_given() -> bool {
    args().config.is_some()
}

fn load_with(overrides: bool, aliases: bool) -> anyhow::Result<Config> {
    let a = args();
    let location =
        m3_config::locate(a.config.as_deref(), &ProcessEnv).map_err(anyhow::Error::msg)?;
    let set = match (&a.set, overrides) {
        (Some(text), true) => m3_config::parse_set(text, &KEYS)?,
        _ => Overrides::default(),
    };
    let alias_env: Option<Box<dyn m3_config::Env>> = if aliases {
        Some(Box::new(ProcessEnv))
    } else {
        None
    };
    Ok(Config::load(
        location,
        set,
        alias_env,
        Some(Box::new(ProcessEnv)),
    )?)
}

/// This process's settings: `--set`, then the file, then the defaults.
pub fn load() -> anyhow::Result<Config> {
    load_with(true, true)
}

/// What a supervised service will see: the file only. launchd, systemd and
/// the SCM start it with no `--set` and without this shell's environment.
pub fn load_service_view() -> anyhow::Result<Config> {
    load_with(false, false)
}

/// The m3_config.toml this invocation reads, for a message; never fails.
pub fn located_file() -> String {
    match m3_config::locate(args().config.as_deref(), &ProcessEnv) {
        Ok(l) => format!("{} (located by {})", l.file.display(), l.tier),
        Err(e) => format!("<m3_config.toml could not be located: {e}>"),
    }
}

/// What `install` pins into the service's command line: `--config <file>`,
/// after checking the service can start from that file alone. Settings come
/// only from the file (and --set), so this shell cannot give the server
/// anything the service would not see.
pub fn service_args() -> anyhow::Result<(Vec<String>, PathBuf)> {
    let svc = load_service_view()?;
    let file = svc.location().file.clone();
    resolve(&svc).map_err(|e| {
        anyhow::anyhow!(
            "refusing to install: the service could not start from {}: {e:#}",
            file.display()
        )
    })?;
    Ok((
        vec!["--config".into(), file.to_string_lossy().into_owned()],
        file,
    ))
}

// ── Resolution ──────────────────────────────────────────────────────────────
/// Fully-resolved runtime config.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub gguf: String,
    /// Where `gguf` came from, for the startup log.
    pub gguf_source: String,
    pub port: u16,
    pub host: String,
    /// Where `host` came from, so a bad value names the setting to fix.
    pub host_source: String,
    pub streams: usize,
    pub n_ctx: u32,
    pub seq_max: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub coalesce_ms: u64,
    pub max_batch_tokens: usize,
    pub security: crate::security::SecurityConfig,
    /// The m3_config.toml read, the tier that located it, and whether it existed.
    pub config_file: PathBuf,
    pub config_tier: &'static str,
    pub config_present: bool,
    /// `key <- source` for every value not at its default.
    pub sources: Vec<String>,
}

/// Default worker-context count, chosen by the COMPILED backend.
///
/// Each stream materialises its own compute graph on first use, sized for a
/// worst-case `n_ctx` batch, and holds it for the process lifetime. Measured
/// 2026-09-16 on bge-m3 at `n_ctx=8192`: **~3.85 GiB per stream on CUDA and
/// ~4.0 GiB on a CPU-only build** — the cost is a property of the model and
/// `n_ctx`, not of the backend. It is linear in `streams` and quadratic in
/// `n_ctx`.
///
/// * **GPU builds (cuda / vulkan / metal) -> 2.** A machine with a discrete GPU
///   or Apple unified memory is very likely to have the headroom.
/// * **CPU-only -> 1.** The modest-hardware case almost by definition; ~4 GiB
///   for a second graph is a large fraction of such a machine.
///
/// ⚠ This trades THROUGHPUT for memory: at `streams = 1` concurrent callers
/// queue. `queue_depth` on `/metrics` is the signal. It is a DEFAULT, not a
/// cap: `[embed.server] streams` wins. The resolved value is logged at startup.
const fn default_streams() -> usize {
    if cfg!(any(
        feature = "embedded-cuda",
        feature = "embedded-vulkan",
        feature = "embedded-metal"
    )) {
        2
    } else {
        1
    }
}

fn note(sources: &mut Vec<String>, key: &Key, from: &Source) {
    if *from != Source::Default {
        sources.push(format!("{} <- {from}", key.name));
    }
}

fn int<T: TryFrom<i64> + Copy>(
    cfg: &Config,
    key: &Key,
    default: T,
    sources: &mut Vec<String>,
) -> anyhow::Result<T> {
    let (v, from) = cfg.int_or(key, default)?;
    note(sources, key, &from);
    Ok(v)
}

/// Resolve the server's settings. Errors when a value is invalid or `embed.gguf`
/// is unresolved (it has no default).
pub fn resolve(cfg: &Config) -> anyhow::Result<ResolvedConfig> {
    let file = cfg.location().file.clone();
    let mut sources = Vec::new();
    let (gguf, from) = cfg.path(&GGUF)?.ok_or_else(|| {
        anyhow::anyhow!(
            "embed.gguf is not set: {} ({}; located by {}) has no [embed] gguf, and no --set \
             gives one. Set it: m3 config set embed.gguf <model.gguf>{}",
            file.display(),
            if cfg.file_present() {
                "present"
            } else {
                "absent"
            },
            cfg.location().tier,
            if config_given() {
                ""
            } else {
                ". A supervised service must be started with --config <your m3_config.toml> \
                 (`m3-embed-server --config <file> install` records it)"
            }
        )
    })?;
    note(&mut sources, &GGUF, &from);
    let gguf_source = from.to_string();
    let gguf = gguf.to_string_lossy().into_owned();
    if !Path::new(&gguf).exists() {
        anyhow::bail!("GGUF path does not exist: {gguf} (from {gguf_source})");
    }
    let (host, host_from) = cfg
        .string(&HOST)?
        .unwrap_or_else(|| (DEFAULT_HOST.to_string(), Source::Default));
    note(&mut sources, &HOST, &host_from);
    let port = int(cfg, &PORT, DEFAULT_PORT, &mut sources)?;
    let streams = int(cfg, &STREAMS, default_streams(), &mut sources)?;
    let n_ctx = int(cfg, &CTX, DEFAULT_CTX, &mut sources)?;
    let seq_max = int(cfg, &SEQ_MAX, DEFAULT_SEQ_MAX, &mut sources)?;
    let n_batch = int(cfg, &N_BATCH, n_ctx, &mut sources)?;
    let n_ubatch = int(cfg, &N_UBATCH, n_ctx, &mut sources)?;
    let coalesce_ms = int(cfg, &COALESCE_MS, DEFAULT_COALESCE_MS, &mut sources)?;
    let max_batch_tokens = int(
        cfg,
        &MAX_BATCH_TOKENS,
        DEFAULT_MAX_BATCH_TOKENS,
        &mut sources,
    )?;
    let security = security_config(cfg, true)?;
    for (key, from) in [
        (&FIPS_MODE, &security.fips_mode_source),
        (&FIPS_STRICT, &security.fips_strict_source),
    ] {
        note(&mut sources, key, from);
    }
    Ok(ResolvedConfig {
        gguf,
        gguf_source,
        port,
        host,
        host_source: host_from.to_string(),
        streams,
        n_ctx,
        seq_max,
        n_batch,
        n_ubatch,
        coalesce_ms,
        max_batch_tokens,
        security,
        config_file: file,
        config_tier: cfg.location().tier.as_str(),
        config_present: cfg.file_present(),
        sources,
    })
}

/// The transport-security settings. The token digest is read from the secrets
/// file only when `read_token` and a listener for other machines is
/// configured: a loopback-only server never depends on the secrets file.
pub fn security_config(
    cfg: &Config,
    read_token: bool,
) -> anyhow::Result<crate::security::SecurityConfig> {
    // The tier, as m3 Python computes it: strict implies mode.
    let fips = m3_config::security::fips_tier(cfg)?;
    let (fips_mode, fips_strict) = (
        fips.tier.is_fips(),
        fips.tier == m3_config::security::FipsTier::Strict,
    );
    let (fips_mode_source, fips_strict_source) = (fips.mode_source, fips.strict_source);
    let remote_listen = cfg.string(&REMOTE_LISTEN)?.map(|(v, _)| v);
    let (secrets_path, _) = secrets::secrets_file(cfg)?;
    let token_from = format!(
        "{TOKEN_SECRET} in the secrets file {}",
        secrets_path.display()
    );
    let token_sha256 = if read_token && remote_listen.is_some() {
        secrets::read_secret(&secrets_path, TOKEN_SECRET)?
    } else {
        None
    };
    Ok(crate::security::SecurityConfig {
        remote_listen,
        tls_cert: cfg.path(&TLS_CERT)?.map(|(p, _)| p),
        tls_key: cfg.path(&TLS_KEY)?.map(|(p, _)| p),
        token_sha256,
        fips_mode,
        fips_strict,
        config_path: cfg.location().file.clone(),
        fips_mode_from: fips_mode_source.to_string(),
        fips_strict_from: fips_strict_source.to_string(),
        fips_mode_source,
        fips_strict_source,
        token_from,
    })
}

/// Where the SUPERVISED service listens: the file's host and port, or the
/// defaults. A wildcard bind is reached on loopback.
#[cfg(not(windows))]
pub fn service_addr() -> anyhow::Result<(String, u16)> {
    let cfg = load_service_view()?;
    let host = cfg
        .string(&HOST)?
        .map(|(h, _)| h)
        .unwrap_or_else(|| DEFAULT_HOST.into());
    let host = if host == "0.0.0.0" {
        DEFAULT_HOST.into()
    } else {
        host
    };
    Ok((host, cfg.int_or(&PORT, DEFAULT_PORT)?.0))
}

// ── The retired per-server file ─────────────────────────────────────────────
/// Where earlier versions kept the server's own config.toml.
pub fn legacy_config_path() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var("PROGRAMDATA").unwrap_or_else(|_| "C:\\ProgramData".into());
        PathBuf::from(base)
            .join("m3-embed-server")
            .join("config.toml")
    }
    #[cfg(not(windows))]
    {
        user_config_base()
            .join("m3-embed-server")
            .join("config.toml")
    }
}

/// The warning for a retired config.toml still at `legacy`, or `None` when
/// there is none.
pub fn legacy_file_notice(legacy: &Path, m3_config: &Path) -> Option<String> {
    if !legacy.is_file() {
        return None;
    }
    Some(format!(
        "observed: {} exists, but m3-embed-server no longer reads it; its values are NOT in effect. \
         They belong in m3_config.toml ({}): [embed] gguf, ctx; [embed.server] host, port, streams, \
         seq_max, n_batch, n_ubatch, coalesce_ms, max_batch_tokens, remote_listen, tls_cert, tls_key; \
         [crypto] fips_mode, fips_strict; token_sha256 becomes {TOKEN_SECRET} in the secrets file. \
         m3 migrates them and renames the old file to config.toml.migrated (`m3 embedder` \
         commands); until then compare the two by hand",
        legacy.display(),
        m3_config.display()
    ))
}

/// Log [`legacy_file_notice`] once, for the file this server runs with.
pub fn warn_legacy_file(cfg: &Config) {
    if let Some(msg) = legacy_file_notice(&legacy_config_path(), &cfg.location().file) {
        log::warn!("{msg}");
    }
}

// ── OS paths (locators; the per-user service model) ─────────────────────────
/// Per-user base directory for config (`$XDG_CONFIG_HOME` or `~/.config` on
/// Linux, `~/Library/Application Support` on macOS).
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
/// `allow(dead_code)`: only `default_log_path` calls this, consumed by the
/// Windows service and the macOS launchd agent, not the Linux systemd path
/// (which logs to the journal).
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

/// Default location for the service log file. Per-user on Unix to match the
/// per-user service model.
///
/// `allow(dead_code)`: consumed by the Windows service and the macOS launchd
/// agent, but not the Linux systemd unit, which logs to the journal.
#[allow(dead_code)]
pub fn default_log_path() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var("PROGRAMDATA").unwrap_or_else(|_| "C:\\ProgramData".into());
        PathBuf::from(base)
            .join("m3-embed-server")
            .join("service.log")
    }
    #[cfg(not(windows))]
    {
        user_state_base()
            .join("m3-embed-server")
            .join("service.log")
    }
}

#[cfg(test)]
mod tests {
    //! Every root points at a temp dir and every environment is a map: no test
    //! reads ~/.m3, ~/.config/m3-embed-server or %PROGRAMDATA%.
    use super::*;
    use m3_config::{Location, MapEnv, Tier};

    struct Fixture {
        dir: tempfile::TempDir,
        gguf: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let gguf = dir.path().join("model.gguf");
            std::fs::write(&gguf, b"not a real gguf; resolve() only stats it").unwrap();
            Fixture { dir, gguf }
        }

        fn gguf_toml(&self) -> String {
            format!("{:?}", self.gguf.to_string_lossy())
        }

        /// m3_config.toml with `text`, read with `env` as the legacy names'
        /// environment and `set` as the --set table.
        fn config(&self, text: &str, env: MapEnv, set: Option<&str>) -> Config {
            let file = self.dir.path().join("m3_config.toml");
            std::fs::write(&file, text).unwrap();
            let loc = Location {
                file,
                root: self.dir.path().to_path_buf(),
                tier: Tier::Arg,
            };
            let set = set
                .map(|s| m3_config::parse_set(s, &KEYS).unwrap())
                .unwrap_or_default();
            Config::load(loc, set, Some(Box::new(env)), None).unwrap()
        }
    }

    /// The trigger: a FIPS setting made in m3_config.toml must reach the server,
    /// win over a former environment name (which is not read), and be reported
    /// as the file's.
    #[test]
    fn fips_flags_set_only_in_m3_config_take_effect_and_outrank_the_env() {
        let f = Fixture::new();
        let env = MapEnv::new()
            .with("M3_FIPS_MODE", "0")
            .with("M3_FIPS_STRICT", "false");
        let cfg = f.config(
            &format!(
                "[embed]\ngguf = {}\n[crypto]\nfips_mode = true\nfips_strict = true\n",
                f.gguf_toml()
            ),
            env,
            None,
        );
        let r = resolve(&cfg).unwrap();
        assert!(r.security.fips_mode && r.security.fips_strict);
        assert!(
            r.security
                .fips_mode_from
                .starts_with("[crypto].fips_mode in ")
                && r.security.fips_mode_from.ends_with("m3_config.toml"),
            "{}",
            r.security.fips_mode_from
        );
        assert!(
            r.security
                .fips_strict_from
                .starts_with("[crypto].fips_strict in "),
            "{}",
            r.security.fips_strict_from
        );
    }

    /// Fixed batch defaults (2048 / 512) sat below the default ctx (8192), so
    /// every start raised them at the FFI and warned although nothing was set.
    #[test]
    fn batch_sizes_follow_ctx_unless_set() {
        let f = Fixture::new();
        let r = resolve(&f.config(
            &format!("[embed]\ngguf = {}\n", f.gguf_toml()),
            MapEnv::new(),
            None,
        ))
        .unwrap();
        assert_eq!((r.n_batch, r.n_ubatch), (r.n_ctx, r.n_ctx));
        let r = resolve(&f.config(
            &format!(
                "[embed]\ngguf = {}\nctx = 4096\n[embed.server]\nn_batch = 16384\n",
                f.gguf_toml()
            ),
            MapEnv::new(),
            None,
        ))
        .unwrap();
        assert_eq!((r.n_ctx, r.n_batch, r.n_ubatch), (4096, 16384, 4096));
    }

    #[test]
    fn port_and_gguf_set_only_in_m3_config_take_effect_and_outrank_the_env() {
        let f = Fixture::new();
        let env = MapEnv::new()
            .with("M3_EMBED_SERVER_PORT", "9999")
            .with("M3_EMBED_GGUF", "/elsewhere/other.gguf");
        let cfg = f.config(
            &format!(
                "[embed]\ngguf = {}\n[embed.server]\nport = 8093\n",
                f.gguf_toml()
            ),
            env,
            None,
        );
        let r = resolve(&cfg).unwrap();
        assert_eq!(r.port, 8093);
        assert_eq!(Path::new(&r.gguf), f.gguf);
        assert!(
            r.gguf_source.starts_with("[embed].gguf in "),
            "{}",
            r.gguf_source
        );
        assert!(
            r.sources
                .iter()
                .any(|s| s.starts_with("embed.server.port <- [embed.server].port in ")),
            "{:?}",
            r.sources
        );
    }

    #[test]
    fn the_legacy_env_names_are_not_read_and_set_wins_over_all() {
        let f = Fixture::new();
        let text = format!("[embed]\ngguf = {}\n", f.gguf_toml());
        let env = MapEnv::new()
            .with("M3_EMBED_SERVER_PORT", "8094")
            .with("M3_FIPS_MODE", "1")
            .with("M3_EMBED_GGUF", "/elsewhere/other.gguf");
        let r = resolve(&f.config(&text, env.clone(), None)).unwrap();
        assert_eq!(r.port, 8082, "M3_EMBED_SERVER_PORT must not be read");
        assert!(!r.security.fips_mode, "M3_FIPS_MODE must not be read");
        assert!(
            !r.gguf_source.contains("M3_EMBED_GGUF"),
            "{}",
            r.gguf_source
        );

        let r = resolve(&f.config(
            &format!(
                "[embed]\ngguf = {}\n[embed.server]\nport = 8095\n",
                f.gguf_toml()
            ),
            env,
            Some("{embed.server.port=8096, crypto.fips_mode=false}"),
        ))
        .unwrap();
        assert_eq!(r.port, 8096);
        assert!(!r.security.fips_mode);
        assert_eq!(r.security.fips_mode_from, "--set crypto.fips_mode");
    }

    #[test]
    fn the_server_only_knobs_are_read_under_embed_server() {
        let f = Fixture::new();
        let cfg = f.config(
            &format!(
                "[embed]\ngguf = {}\nctx = 4096\n[embed.server]\nstreams = 4\nseq_max = 8\n\
                 n_batch = 4096\nn_ubatch = 1024\ncoalesce_ms = 0\nmax_batch_tokens = 9000\n",
                f.gguf_toml()
            ),
            MapEnv::new(),
            None,
        );
        let r = resolve(&cfg).unwrap();
        assert_eq!(
            (
                r.n_ctx,
                r.streams,
                r.seq_max,
                r.n_batch,
                r.n_ubatch,
                r.coalesce_ms,
                r.max_batch_tokens
            ),
            (4096, 4, 8, 4096, 1024, 0, 9000)
        );
    }

    /// The backend-dependent default must actually DIFFER by backend, and the
    /// file must still be able to raise it (a default, never a cap).
    #[test]
    fn streams_default_follows_the_backend_and_the_file_beats_it() {
        let expected = if cfg!(any(
            feature = "embedded-cuda",
            feature = "embedded-vulkan",
            feature = "embedded-metal"
        )) {
            2
        } else {
            1
        };
        assert_eq!(default_streams(), expected);
        let f = Fixture::new();
        let gguf = format!("[embed]\ngguf = {}\n", f.gguf_toml());
        assert_eq!(
            resolve(&f.config(&gguf, MapEnv::new(), None))
                .unwrap()
                .streams,
            expected
        );
        let raised = format!("{gguf}[embed.server]\nstreams = 4\n");
        assert_eq!(
            resolve(&f.config(&raised, MapEnv::new(), None))
                .unwrap()
                .streams,
            4
        );
    }

    #[test]
    fn the_token_digest_comes_from_the_secrets_file_named_by_secrets_file() {
        let f = Fixture::new();
        let secrets = f.dir.path().join("private").join("s.toml");
        std::fs::create_dir_all(secrets.parent().unwrap()).unwrap();
        let digest = "ab".repeat(32);
        std::fs::write(
            &secrets,
            format!("[secrets]\n{TOKEN_SECRET} = \"{digest}\"\n"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&secrets, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let cfg = f.config(
            &format!(
                "[embed]\ngguf = {}\n[embed.server]\nremote_listen = \"100.64.0.9\"\n\
                 [secrets]\nfile = \"private/s.toml\"\n",
                f.gguf_toml()
            ),
            MapEnv::new(),
            None,
        );
        let sec = resolve(&cfg).unwrap().security;
        assert_eq!(sec.token_sha256.as_deref(), Some(digest.as_str()));
        assert!(
            sec.token_from.contains(&secrets.display().to_string()),
            "{}",
            sec.token_from
        );
        // A digest left in m3_config.toml is not a source.
        let cfg = f.config(
            &format!(
                "[embed]\ngguf = {}\n[embed.server]\nremote_listen = \"100.64.0.9\"\n",
                f.gguf_toml()
            ),
            MapEnv::new(),
            None,
        );
        assert_eq!(resolve(&cfg).unwrap().security.token_sha256, None);
    }

    #[cfg(unix)]
    #[test]
    fn a_group_or_world_readable_secrets_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let f = Fixture::new();
        let secrets = f.dir.path().join("m3_secrets.toml");
        std::fs::write(
            &secrets,
            format!("[secrets]\n{TOKEN_SECRET} = \"{}\"\n", "ab".repeat(32)),
        )
        .unwrap();
        std::fs::set_permissions(&secrets, std::fs::Permissions::from_mode(0o644)).unwrap();
        let cfg = f.config(
            &format!(
                "[embed]\ngguf = {}\n[embed.server]\nremote_listen = \"100.64.0.9\"\n",
                f.gguf_toml()
            ),
            MapEnv::new(),
            None,
        );
        let e = format!("{:#}", resolve(&cfg).unwrap_err());
        assert!(e.contains("user-only") && e.contains("chmod 600"), "{e}");
    }

    #[test]
    fn a_loopback_only_server_does_not_read_the_secrets_file() {
        let f = Fixture::new();
        // Not TOML at all: reading it would fail.
        std::fs::write(f.dir.path().join("m3_secrets.toml"), "[[[").unwrap();
        let cfg = f.config(
            &format!("[embed]\ngguf = {}\n", f.gguf_toml()),
            MapEnv::new(),
            None,
        );
        assert!(resolve(&cfg).unwrap().security.token_sha256.is_none());
    }

    #[test]
    fn a_missing_gguf_names_the_file_and_the_fix() {
        let f = Fixture::new();
        let e = resolve(&f.config("", MapEnv::new(), None))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("embed.gguf is not set") && e.contains("m3_config.toml"),
            "{e}"
        );
        assert!(e.contains("m3 config set embed.gguf"), "{e}");
    }

    #[test]
    fn the_retired_config_toml_is_warned_about_not_read() {
        let d = tempfile::tempdir().unwrap();
        let old = d.path().join("config.toml");
        let m3 = d.path().join("m3_config.toml");
        assert!(
            legacy_file_notice(&old, &m3).is_none(),
            "no file, no warning"
        );
        // m3 keeps the migrated file as config.toml.migrated: that copy alone is not warned about.
        std::fs::write(
            d.path().join("config.toml.migrated"),
            "[embed]\nport = 9000\n",
        )
        .unwrap();
        assert!(legacy_file_notice(&old, &m3).is_none());
        std::fs::write(&old, "[embed]\nport = 9000\n").unwrap();
        let msg = legacy_file_notice(&old, &m3).unwrap();
        assert!(msg.contains(&old.display().to_string()) && msg.contains("no longer reads it"));
        assert!(msg.contains(&m3.display().to_string()) && msg.contains(TOKEN_SECRET));
        assert!(msg.contains("config.toml.migrated"), "{msg}");
    }
}
