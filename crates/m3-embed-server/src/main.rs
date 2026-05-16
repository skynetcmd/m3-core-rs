//! m3-embed-server — OpenAI-compatible CPU embedding service on port 8082.
//!
//! Modes (selected by argv[1]):
//!   (none)            Foreground / dev mode. Logs to stderr, Ctrl-C to stop.
//!   install           Register as a Windows Service (admin). Writes a starter
//!                     %PROGRAMDATA%\m3-embed-server\config.toml from current env.
//!   uninstall         Stop + remove the Windows Service.
//!   start | stop      Convenience wrappers over the SCM (avoid sc.exe).
//!   status            Print "running" / "stopped" / "not installed".
//!   run-as-service    Internal — SCM invokes this when starting the service.
//!
//! On non-Windows targets, only foreground mode is supported.
//!
//! Config priority: env var > %PROGRAMDATA%\m3-embed-server\config.toml > default.

#[cfg(feature = "embedded")]
mod config;
#[cfg(feature = "embedded")]
mod server;
#[cfg(all(windows, feature = "embedded"))]
mod service;

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
    eprintln!(
        "m3-embed-server — OpenAI-compatible CPU embed server (port 8082 by default)\n\
         \n\
         USAGE:\n  \
           m3-embed-server [SUBCOMMAND]\n\
         \n\
         SUBCOMMANDS:\n  \
           (none)        run in foreground (dev mode); Ctrl-C to stop\n  \
           install       register as a Windows Service (admin required)\n  \
           uninstall     stop and remove the Windows Service (admin)\n  \
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
        server::run(cfg, async {
            let _ = tokio::signal::ctrl_c().await;
            log::info!("ctrl-c received, draining...");
        })
        .await
    })
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

#[cfg(all(not(windows), feature = "embedded"))]
fn windows_only(sub: &str) -> anyhow::Result<()> {
    anyhow::bail!("`{sub}` is Windows-only; build & run on Windows to use the service lifecycle")
}

#[cfg(all(not(windows), feature = "embedded"))]
fn run_as_service() -> anyhow::Result<()> {
    windows_only("run-as-service")
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_install() -> anyhow::Result<()> {
    windows_only("install")
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    windows_only("uninstall")
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    windows_only("start")
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    windows_only("stop")
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    windows_only("status")
}
