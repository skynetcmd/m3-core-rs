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
//! Config priority: env var > per-user/per-system config.toml > default.

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
#[cfg(feature = "embedded")]
#[allow(dead_code)]
mod unit_render;
#[cfg(all(not(windows), feature = "embedded"))]
mod service_unix;

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
    if args.get(1).map(String::as_str) == Some("--config") {
        let Some(path) = args.get(2).cloned() else {
            anyhow::bail!("--config needs a path");
        };
        args.drain(1..3);
        // The service subcommands act on the installed service's own file.
        if !matches!(args.get(1).map(String::as_str), None | Some("secure")) {
            anyhow::bail!("--config applies to foreground mode and `secure` only");
        }
        config::set_config_override(std::path::PathBuf::from(path));
    }
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("");
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
           m3-embed-server [--config <path>] [SUBCOMMAND]\n\
         \n\
         --config uses that config.toml instead of the service's (foreground and secure only).\n\
         \n\
         SUBCOMMANDS:\n  \
           (none)        run in foreground (dev mode); Ctrl-C to stop \
(also SIGTERM on Unix, console close/shutdown on Windows)\n  \
           install       {install_line}\n  \
           uninstall     stop and remove the OS service\n  \
           start         start the installed service\n  \
           stop          stop the installed service\n  \
           status        running / stopped / not installed\n  \
           secure        --listen <ip[:port]> --cert <pem> --key <pem> --token-sha256 <hex>: \
add a TLS listener with a bearer token for other machines; --off removes it\n  \
           help          show this message\n"
    );
}

/// `secure --listen ADDR --cert PEM --key PEM --token-sha256 HEX` adds the
/// listener for other machines (TLS and a bearer token) to config.toml, after
/// checking it exactly as the server will at startup (FIPS flags included);
/// `secure --off` removes it, leaving the loopback listener alone.
/// `m3 embedder secure` drives this.
#[cfg(feature = "embedded")]
fn run_secure(args: &[String]) -> anyhow::Result<()> {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    let usage = "usage: m3-embed-server secure --listen <ip[:port]> --cert <pem> --key <pem> \
                 --token-sha256 <hex> | --off";
    let path = config::default_config_path();
    let file = config::load_file_config(&path)?;
    let write = |keys: &config::SecurityKeys| {
        config::write_security_keys(&path, keys).map_err(|e| {
            // Elevation fixes only a refusal on the default (%PROGRAMDATA%) file.
            let hint = if cfg!(windows) && !config::config_overridden() && config::is_permission_denied(&e) {
                ". The file is under %PROGRAMDATA%, writable by Administrators: run this from an elevated prompt"
            } else {
                ""
            };
            anyhow::anyhow!("{e:#}{hint}")
        })
    };
    if args.iter().any(|a| a == "--off") {
        write(&config::SecurityKeys::default())?;
        println!(
            "removed the listener for other machines from {}; the server serves this machine only. \
             Restart it to apply.",
            path.display()
        );
        return Ok(());
    }
    let (mut listen, mut cert, mut key, mut token) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--listen" => &mut listen,
            "--cert" => &mut cert,
            "--key" => &mut key,
            "--token-sha256" => &mut token,
            other => anyhow::bail!("unknown option {other:?}\n{usage}"),
        };
        *slot = Some(it.next().ok_or_else(|| anyhow::anyhow!("{a} needs a value\n{usage}"))?.clone());
    }
    let missing: Vec<&str> = [("--listen", &listen), ("--cert", &cert), ("--key", &key), ("--token-sha256", &token)]
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
    let mut sec = config::security_config(&file.embed);
    sec.remote_listen = Some(listen.trim().to_string());
    sec.tls_cert = Some(cert.clone());
    sec.tls_key = Some(key.clone());
    sec.token_sha256 = Some(token.trim().to_ascii_lowercase());
    let local = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), file.embed.port.unwrap_or(config::DEFAULT_PORT));
    let remote = security::plan(local, &sec)?.remote.map(|r| r.addr);
    write(&config::SecurityKeys {
        remote_listen: sec.remote_listen.as_deref(),
        tls_cert: Some(&cert.display().to_string()),
        tls_key: Some(&key.display().to_string()),
        token_sha256: sec.token_sha256.as_deref(),
    })?;
    // Recorded, not verified: nothing listens there until the server restarts,
    // and reachability depends on the network in between.
    let remote = remote.map(|a| a.to_string()).unwrap_or_default();
    println!(
        "recorded remote_listen {remote} (TLS certificate {}, bearer token required) in {}. After a \
         restart the server listens there for other machines. Verify from one of them: \
         curl --cacert {} https://{remote}/health",
        cert.display(),
        path.display(),
        cert.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Foreground mode (works on every platform)
// ---------------------------------------------------------------------------

#[cfg(feature = "embedded")]
fn run_foreground() -> anyhow::Result<()> {
    // Foreground reads env first, but will still pick up a config.toml if
    // present (handy for testing the service config without installing).
    // A missing file is the empty default; a broken one is an error, not that.
    let file_cfg = config::load_file_config(&config::default_config_path())?;
    let cfg = config::resolve(&file_cfg)?;

    // Still single-threaded here: the one safe point for llama.cpp's env
    // (GGML_CUDA_DISABLE_GRAPHS on CUDA builds; a no-op otherwise).
    m3_embed_llamacpp::prepare_process_env();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let result = runtime.block_on(async move {
        server::run(cfg, shutdown_signal()).await
    });
    if let Err(e) = &result {
        if e.downcast_ref::<server::PortInUse>().is_some() {
            log::error!("{e}");
            std::process::exit(server::EXIT_PORT_IN_USE);
        }
        if e.downcast_ref::<security::ConfigRefused>().is_some() {
            log::error!("{e}");
            std::process::exit(security::EXIT_CONFIG_REFUSED);
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
