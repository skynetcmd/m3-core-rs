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
#[cfg(feature = "embedded")]
mod server;
#[cfg(all(windows, feature = "embedded"))]
mod service;
// Platform-neutral unit-file templating — compiles everywhere so its render
// tests run on the (Windows) CI box; consumed by service_unix on Unix. On
// Windows the functions are exercised only by the test module, so silence the
// unused-code lint there.
#[cfg(feature = "embedded")]
#[cfg_attr(windows, allow(dead_code))]
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
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match sub {
        "" => run_foreground(),
        "run-as-service" => run_as_service(),
        "install" => run_install(),
        "uninstall" => run_uninstall(),
        "start" => run_start(),
        "stop" => run_stop(),
        "status" => run_status(),
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
           m3-embed-server [SUBCOMMAND]\n\
         \n\
         SUBCOMMANDS:\n  \
           (none)        run in foreground (dev mode); Ctrl-C / SIGTERM to stop\n  \
           install       {install_line}\n  \
           uninstall     stop and remove the OS service\n  \
           start         start the installed service\n  \
           stop          stop the installed service\n  \
           status        running / stopped / not installed\n  \
           help          show this message\n"
    );
}

// ---------------------------------------------------------------------------
// Foreground mode (works on every platform)
// ---------------------------------------------------------------------------

#[cfg(feature = "embedded")]
fn run_foreground() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init()
        .ok();

    // Foreground reads env first, but will still pick up a config.toml if
    // present (handy for testing the service config without installing).
    let file_cfg = config::load_file_config(&config::default_config_path()).unwrap_or_default();
    let cfg = config::resolve(&file_cfg)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        server::run(cfg, shutdown_signal()).await
    })
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
            log::warn!("could not install SIGTERM handler: {e}; ctrl-c only");
            let _ = tokio::signal::ctrl_c().await;
            log::info!("ctrl-c received, draining...");
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => log::info!("ctrl-c received, draining..."),
        _ = term.recv()             => log::info!("SIGTERM received, draining..."),
    }
}

#[cfg(all(feature = "embedded", not(unix)))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    log::info!("ctrl-c received, draining...");
}

// ---------------------------------------------------------------------------
// Subcommand routing — Windows-specific paths gated, others print a friendly
// "Windows-only" error.
// ---------------------------------------------------------------------------

#[cfg(all(windows, feature = "embedded"))]
fn run_as_service() -> anyhow::Result<()> {
    service::run_dispatcher().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_install() -> anyhow::Result<()> {
    service::install().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    service::uninstall().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    service::start().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    service::stop().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    service::status().map_err(|e| anyhow::anyhow!("{e}"))
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
    service_unix::install().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    service_unix::uninstall().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    service_unix::start().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    service_unix::stop().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    service_unix::status().map_err(|e| anyhow::anyhow!("{e}"))
}
