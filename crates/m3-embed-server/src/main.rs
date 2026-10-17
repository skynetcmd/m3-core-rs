//! m3-embed-server — OpenAI-compatible CPU embedding service on port 8082.
//!
//! Modes (selected by argv[1]):
//!   (none)            Foreground / dev mode. Logs to stderr, Ctrl-C / SIGTERM
//!                     to stop. This is also what the OS supervisor runs.
//!   install           Register as an OS service: Windows Service (admin),
//!                     macOS launchd user agent, or Linux `systemd --user`
//!                     unit. Writes a starter config from the current env.
//!   uninstall         Stop + remove the OS service.
//!   start | stop      Convenience wrappers over the OS service manager.
//!   status            Print "running" / "stopped" / "not installed".
//!   run-as-service    Internal — Windows SCM invokes this when starting the
//!                     service. (Unix supervisors run foreground mode directly.)
//!
//! Config: m3's m3_config.toml and the secrets file it names (src/config.rs).
//! Per key: `--set` > m3_config.toml > the legacy environment name > default.

#[cfg(feature = "embedded")]
mod config;
// Transport security compiles without `embedded` so its tests (a real TLS
// listener included) run without building llama.cpp.
#[cfg_attr(not(feature = "embedded"), allow(dead_code))]
mod security;
#[cfg(feature = "embedded")]
mod server;
#[cfg(all(windows, feature = "embedded"))]
mod service;
// Platform-neutral unit-file templating — compiles everywhere so its render
// tests run on any CI box; consumed by service_unix on Unix. Each item is used
// by exactly one platform path (render_plist/LAUNCHD_LABEL → macOS,
// render_unit/SERVICE_NAME → Linux), so *every* single-platform build leaves
// some of them unused. `allow(dead_code)` module-wide is correct here — the
// `#[cfg(test)]` render tests exercise all of them regardless of host.
#[cfg(all(not(windows), feature = "embedded"))]
mod service_unix;
#[cfg(feature = "embedded")]
#[allow(dead_code)]
mod unit_render;

#[cfg(not(feature = "embedded"))]
fn main() -> anyhow::Result<()> {
    eprintln!(
        "m3-embed-server was built without the `embedded` feature — \
         rebuild with `--features embedded` to get a working binary."
    );
    std::process::exit(2);
}

#[cfg(feature = "embedded")]
fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let global = take_global_options(&mut args)?;
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("").to_string();
    let sub = sub.as_str();
    // A supervised service sees no --set (install pins only --config), so a
    // --set given to a service subcommand would describe a server that never runs.
    if global.set.is_some() && !matches!(sub, "" | "secure") {
        anyhow::bail!(
            "--set applies to foreground mode and `secure --check` only. A service reads its settings \
             from m3_config.toml: record them with `m3 config set <key> <value>`, then restart it"
        );
    }
    config::set_args(global);
    // Every subcommand reads config and may warn (an unparseable env value, a
    // half-configured listener); with no logger those warnings vanished. The
    // service path is the exception: it installs its own file logger, and
    // `log` takes only the first logger installed.
    if sub != "run-as-service" {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
            .ok();
    }

    match sub {
        "" => run_foreground(),
        "run-as-service" => run_as_service(),
        "install" => run_install(),
        "uninstall" => run_uninstall(),
        "start" => run_start(),
        "stop" => run_stop(),
        "status" => run_status(),
        "secure" => run_secure(&args[2..]),
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            print_help();
            std::process::exit(2);
        }
    }
}

/// Remove `--config <file>` / `--config=<file>` and `--set {...}` /
/// `--set={...}` from the front of `args` (before the subcommand), each at most
/// once. `--set` is m3's flag: one TOML table with m3_config.toml's key names.
#[cfg(feature = "embedded")]
fn take_global_options(args: &mut Vec<String>) -> anyhow::Result<config::Args> {
    let mut out = config::Args::default();
    while let Some(a) = args.get(1).cloned() {
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) if n == "--config" || n == "--set" => (n.to_string(), Some(v.to_string())),
            _ if a == "--config" || a == "--set" => (a.clone(), None),
            _ => break,
        };
        args.remove(1);
        let value = match inline {
            Some(v) => v,
            None => {
                if args.len() < 2 {
                    anyhow::bail!("{name} needs a value, e.g. {name} {}", example(&name));
                }
                args.remove(1)
            }
        };
        let given = if name == "--config" {
            out.config.is_some()
        } else {
            out.set.is_some()
        };
        if given {
            anyhow::bail!(
                "{name} was given twice; give it once{}",
                if name == "--set" {
                    ", with every setting in one table, e.g. \"--set={embed.server.port=8083, crypto.fips_mode=true}\""
                } else {
                    ""
                }
            );
        }
        if name == "--config" {
            out.config = Some(std::path::PathBuf::from(value));
        } else {
            out.set = Some(value);
        }
    }
    Ok(out)
}

#[cfg(feature = "embedded")]
fn example(flag: &str) -> &'static str {
    if flag == "--set" {
        "\"--set={embed.server.port=8083}\""
    } else {
        "<path to m3_config.toml>"
    }
}

#[cfg(feature = "embedded")]
fn print_help() {
    // The `install` line names the concrete mechanism for the host OS so the
    // operator knows up front whether elevation is needed.
    let install_line = if cfg!(windows) {
        "register as a Windows Service (Administrator required)"
    } else if cfg!(target_os = "macos") {
        "register as a launchd user agent (no sudo)"
    } else if cfg!(target_os = "linux") {
        "register as a systemd --user unit (no sudo)"
    } else {
        "register as an OS service (unsupported on this platform)"
    };
    eprintln!(
        "m3-embed-server — OpenAI-compatible CPU embed server (port 8082 by default)\n\
         \n\
         USAGE:\n  \
           m3-embed-server [--config <m3_config.toml>] [\"--set={{key=value, ...}}\"] [SUBCOMMAND]\n\
         \n\
         Settings come from m3's m3_config.toml (M3_CONFIG_ROOT > M3_MEMORY_ROOT/config > ~/.m3/config,\n\
         or --config) and the secrets file it names. --set (foreground and secure only) outranks the\n\
         file; legacy M3_* environment names rank below it. `install` records --config for the service.\n\
         \n\
         SUBCOMMANDS:\n  \
           (none)        run in foreground (dev mode); Ctrl-C to stop \
(also SIGTERM on Unix, console close/shutdown on Windows)\n  \
           install       {install_line}\n  \
           uninstall     stop and remove the OS service\n  \
           start         start the installed service\n  \
           stop          stop the installed service\n  \
           status        running / stopped / not installed\n  \
           secure        --check --listen <ip[:port]> --cert <pem> --key <pem> --token-sha256 <hex>: \
check a TLS listener with a bearer token for other machines as the server will; writes nothing \
(m3 embedder secure records it)\n  \
           help          show this message\n"
    );
}

/// `secure --check --listen ADDR --cert PEM --key PEM --token-sha256 HEX`
/// checks a listener for other machines (TLS and a bearer token) exactly as the
/// server will at startup, FIPS flags included, and writes NOTHING: the
/// settings live in m3's files, which m3 writes (`m3 embedder secure` records
/// them after this check passes). `--check` is required so a caller expecting
/// the old write-in-place behaviour fails loudly instead of recording nothing.
#[cfg(feature = "embedded")]
fn run_secure(args: &[String]) -> anyhow::Result<()> {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    let usage =
        "usage: m3-embed-server secure --check --listen <ip[:port]> --cert <pem> --key <pem> \
                 --token-sha256 <hex>";
    let cfg = config::load()?;
    let file = cfg.location().file.clone();
    let (secrets_path, _) = m3_config::secrets::secrets_file(&cfg)?;
    if args.iter().any(|a| a == "--off") {
        anyhow::bail!(
            "m3-embed-server no longer writes configuration, so `secure --off` has nothing to do. \
             Remove the listener for other machines from {}: m3 config unset embed.server.remote_listen \
             (and embed.server.tls_cert, embed.server.tls_key), then restart the server",
            file.display()
        );
    }
    if !args.iter().any(|a| a == "--check") {
        anyhow::bail!(
            "`secure` only checks settings now (m3-embed-server no longer writes {}); pass --check. \
             `m3 embedder secure` checks and records them\n{usage}",
            file.display()
        );
    }
    let (mut listen, mut cert, mut key, mut token) = (None, None, None, None);
    let mut it = args.iter().filter(|a| *a != "--check");
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--listen" => &mut listen,
            "--cert" => &mut cert,
            "--key" => &mut key,
            "--token-sha256" => &mut token,
            other => anyhow::bail!("unknown option {other:?}\n{usage}"),
        };
        *slot = Some(
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value\n{usage}"))?
                .clone(),
        );
    }
    let missing: Vec<&str> = [
        ("--listen", &listen),
        ("--cert", &cert),
        ("--key", &key),
        ("--token-sha256", &token),
    ]
    .into_iter()
    .filter(|(_, v)| v.is_none())
    .map(|(name, _)| name)
    .collect();
    let (Some(listen), Some(cert), Some(key), Some(token)) = (listen, cert, key, token) else {
        anyhow::bail!("secure is missing {}\n{usage}", missing.join(", "));
    };
    // Absolute, because the service does not start in this directory.
    let cert = std::path::absolute(&cert)?;
    let key = std::path::absolute(&key)?;
    let mut sec = config::security_config(&cfg, false)?;
    sec.remote_listen = Some(listen.trim().to_string());
    sec.tls_cert = Some(cert.clone());
    sec.tls_key = Some(key.clone());
    sec.token_sha256 = Some(token.trim().to_ascii_lowercase());
    sec.token_from = "--token-sha256".into();
    let port = cfg.int_or(&config::PORT, config::DEFAULT_PORT)?.0;
    let local = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let remote = security::plan(local, &sec)?
        .remote
        .map(|r| r.addr.to_string())
        .unwrap_or_default();
    // Checked, not recorded, and not verified: nothing listens there until the
    // settings are recorded and the server restarts.
    println!(
        "checked: remote_listen {remote} with TLS certificate {} and a bearer token passes the \
         server's startup checks. Nothing was written. Record it (m3 embedder secure does):\n  \
         m3 config set embed.server.remote_listen {}\n  \
         m3 config set embed.server.tls_cert {}\n  \
         m3 config set embed.server.tls_key {}\n  \
         the digest as {} in the secrets file {}\n\
         then restart the server and verify from another machine: curl --cacert {} https://{remote}/health",
        cert.display(),
        listen.trim(),
        cert.display(),
        key.display(),
        config::TOKEN_SECRET,
        secrets_path.display(),
        cert.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Foreground mode (works on every platform)
// ---------------------------------------------------------------------------

/// The exit status for a failure `e`: a refused configuration (an invalid
/// m3_config.toml, value or security setting) and a taken port are not cured by
/// a restart, so supervisors are told not to retry.
#[cfg(feature = "embedded")]
fn exit_code_for(e: &anyhow::Error) -> Option<i32> {
    if e.downcast_ref::<server::PortInUse>().is_some() {
        return Some(server::EXIT_PORT_IN_USE);
    }
    if e.downcast_ref::<security::ConfigRefused>().is_some()
        || e.chain()
            .any(|c| c.downcast_ref::<m3_config::ConfigError>().is_some())
    {
        return Some(security::EXIT_CONFIG_REFUSED);
    }
    None
}

#[cfg(feature = "embedded")]
fn run_foreground() -> anyhow::Result<()> {
    let result = (|| {
        let cfg = config::load()?;
        config::warn_legacy_file(&cfg);
        let resolved = config::resolve(&cfg)?;

        // Still single-threaded here: the one safe point for llama.cpp's env
        // (GGML_CUDA_DISABLE_GRAPHS on CUDA builds; a no-op otherwise).
        m3_embed_llamacpp::prepare_process_env();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async move { server::run(resolved, shutdown_signal()).await })
    })();
    if let Err(e) = &result {
        if let Some(code) = exit_code_for(e) {
            log::error!("{e:#}");
            std::process::exit(code);
        }
    }
    result
}

/// Future that resolves on the first stop signal. Ctrl-C everywhere; on Unix
/// also SIGTERM — which is what `systemctl stop` / `launchctl kill` send, so
/// the service drains gracefully instead of being hard-killed.
#[cfg(all(feature = "embedded", unix))]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            // Can't install the SIGTERM handler — fall back to ctrl-c only.
            log::warn!(
                "observed: SIGTERM handler could not be installed ({e}). \
                 effect: shutdown is driven by ctrl-c only, so a supervisor \
                 stopping this process will kill it rather than let it drain. \
                 inspect: whether the process is running under a restricted \
                 signal mask or a sandbox that blocks signal registration."
            );
            ctrl_c_or_never().await;
            log::info!("ctrl-c received, draining...");
            return;
        }
    };
    tokio::select! {
        _ = ctrl_c_or_never() => log::info!("ctrl-c received, draining..."),
        _ = term.recv()       => log::info!("SIGTERM received, draining..."),
    }
}

/// Resolves on Ctrl-C. A handler that cannot be installed is warned about and
/// never resolves: an `Err` here is not a stop request, and treating it as one
/// shut the server down the moment it started.
#[cfg(feature = "embedded")]
async fn ctrl_c_or_never() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        log::warn!(
            "the Ctrl-C handler could not be installed ({e}); Ctrl-C will not drain this process, \
             so stop it through its other stop signals or its service manager"
        );
        std::future::pending::<()>().await;
    }
}

/// Windows foreground mode. There is no SIGTERM on Windows, so the equivalent
/// stop signals are the console control events: CTRL_CLOSE (window closed),
/// CTRL_SHUTDOWN (system shutting down) and CTRL_LOGOFF. Waiting on ctrl-c
/// ALONE — as this did — meant every stop that was not a literal Ctrl-C keypress
/// hard-killed the process mid-request instead of draining, despite the usage
/// text promising "Ctrl-C / SIGTERM to stop".
///
/// The installed SERVICE path is unaffected: it drains via the SCM
/// `ServiceControl::Stop`/`Shutdown` handler in service.rs. This is the
/// foreground/dev path only.
#[cfg(all(feature = "embedded", not(unix)))]
async fn shutdown_signal() {
    use tokio::signal::windows;

    // Each listener is independent; if one cannot be installed we still honor
    // the others rather than losing graceful shutdown entirely.
    fn installed<T>(what: &str, r: std::io::Result<T>) -> Option<T> {
        r.map_err(|e| {
            log::warn!(
                "the {what} handler could not be installed ({e}); that event will end this process \
                 without draining in-flight requests"
            )
        })
        .ok()
    }
    let mut close = installed("console-close", windows::ctrl_close());
    let mut shutdown = installed("system-shutdown", windows::ctrl_shutdown());
    let mut logoff = installed("logoff", windows::ctrl_logoff());

    // Helper: a future that never resolves, standing in for a listener we could
    // not install, so `select!` still compiles with a uniform shape.
    async fn never() {
        std::future::pending::<()>().await
    }

    tokio::select! {
        _ = ctrl_c_or_never() => log::info!("ctrl-c received, draining..."),
        _ = async { match close.as_mut() { Some(s) => { s.recv().await; }, None => never().await } } =>
            log::info!("console close received, draining..."),
        _ = async { match shutdown.as_mut() { Some(s) => { s.recv().await; }, None => never().await } } =>
            log::info!("system shutdown received, draining..."),
        _ = async { match logoff.as_mut() { Some(s) => { s.recv().await; }, None => never().await } } =>
            log::info!("logoff received, draining..."),
    }
}

/// Render an error with its whole `source()` chain. `windows-service` displays
/// every Win32 failure as "IO error in winapi call" and keeps the OS error code
/// in the source, so printing the top error alone drops the one fact that
/// tells access-denied from not-installed.
#[cfg(all(windows, feature = "embedded"))]
fn with_causes(e: Box<dyn std::error::Error>) -> anyhow::Error {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        let s_txt = s.to_string();
        if !msg.contains(&s_txt) {
            msg.push_str(": ");
            msg.push_str(&s_txt);
        }
        src = s.source();
    }
    anyhow::anyhow!(msg)
}

// ---------------------------------------------------------------------------
// Subcommand routing — Windows-specific paths gated, others print a friendly
// "Windows-only" error.
// ---------------------------------------------------------------------------

#[cfg(all(windows, feature = "embedded"))]
fn run_as_service() -> anyhow::Result<()> {
    service::run_dispatcher().map_err(with_causes)
}

#[cfg(all(windows, feature = "embedded"))]
fn run_install() -> anyhow::Result<()> {
    service::install().map_err(with_causes)
}

#[cfg(all(windows, feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    service::uninstall().map_err(with_causes)
}

#[cfg(all(windows, feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    service::start().map_err(with_causes)
}

#[cfg(all(windows, feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    service::stop().map_err(with_causes)
}

#[cfg(all(windows, feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    service::status().map_err(with_causes)
}

// On Unix the OS service manager (launchd / systemd) runs the binary in
// foreground mode directly — there is no SCM-style `run-as-service` callback.
#[cfg(all(not(windows), feature = "embedded"))]
fn run_as_service() -> anyhow::Result<()> {
    anyhow::bail!(
        "`run-as-service` is a Windows-SCM-internal entry point and has no \
         meaning on this platform — the launchd/systemd unit runs foreground \
         mode (`m3-embed-server` with no subcommand) directly."
    )
}

#[cfg(all(not(windows), feature = "embedded"))]
fn run_install() -> anyhow::Result<()> {
    service_unix::install().map_err(anyhow::Error::msg)
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    service_unix::uninstall().map_err(anyhow::Error::msg)
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    service_unix::start().map_err(anyhow::Error::msg)
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    service_unix::stop().map_err(anyhow::Error::msg)
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    service_unix::status().map_err(anyhow::Error::msg)
}

#[cfg(all(test, feature = "embedded"))]
mod tests {
    use super::*;

    fn argv(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn global_options_come_off_the_front_in_either_form_and_order() {
        let mut a = argv(&[
            "exe",
            "--set={embed.server.port=8083}",
            "--config",
            "/c/m3.toml",
            "secure",
            "--check",
        ]);
        let g = take_global_options(&mut a).unwrap();
        assert_eq!(
            g.config.as_deref(),
            Some(std::path::Path::new("/c/m3.toml"))
        );
        assert_eq!(g.set.as_deref(), Some("{embed.server.port=8083}"));
        assert_eq!(a, argv(&["exe", "secure", "--check"]));

        let mut a = argv(&["exe", "--config=/c/m3.toml", "install"]);
        let g = take_global_options(&mut a).unwrap();
        assert!(g.config.is_some() && g.set.is_none());
        assert_eq!(a, argv(&["exe", "install"]));
    }

    #[test]
    fn a_repeated_or_valueless_option_is_refused() {
        let mut a = argv(&["exe", "--set", "{a=1}", "--set", "{b=2}"]);
        assert!(take_global_options(&mut a)
            .unwrap_err()
            .to_string()
            .contains("given twice"));
        let mut a = argv(&["exe", "--config"]);
        assert!(take_global_options(&mut a)
            .unwrap_err()
            .to_string()
            .contains("needs a value"));
    }
}
