//! Windows Service Control Manager integration.
//!
//! Exposes:
//! - `install` / `uninstall` — register the service with SCM (admin required).
//! - `start` / `stop` — convenience wrappers around `ServiceController`.
//! - `status` — print "running" / "stopped" / "not installed".
//! - `run_dispatcher` — invoked by SCM when the service starts (via the
//!   internal `run-as-service` subcommand).
//!
//! The whole module is `#[cfg(windows)]`. On other platforms the binary
//! supports only foreground mode.

#![cfg(all(windows, feature = "embedded"))]

use std::ffi::OsString;
use std::sync::mpsc;
use std::time::Duration;

use windows_service::{
    define_windows_service,
    service::{
        ServiceAccess, ServiceControl, ServiceControlAccept, ServiceErrorControl, ServiceExitCode,
        ServiceInfo, ServiceStartType, ServiceState, ServiceStatus, ServiceType,
    },
    service_control_handler::{self, ServiceControlHandlerResult},
    service_dispatcher,
    service_manager::{ServiceManager, ServiceManagerAccess},
};

use crate::config;

pub const SERVICE_NAME: &str = "m3-embed-server";
pub const SERVICE_DISPLAY: &str = "M3 Embed Server (CPU fallback)";
pub const SERVICE_DESC: &str =
    "OpenAI-compatible CPU embed server on port 8082 (fallback for m3-memory in-process embedder).";

// SCM entry point — registered with `define_windows_service!`. SCM calls this
// (not main) when the service starts.
define_windows_service!(ffi_service_main, service_main);

fn service_main(args: Vec<OsString>) {
    if let Err(e) = run_service(args) {
        // We can't log to stderr usefully under SCM; the env_logger sink
        // (set up by run_dispatcher) will already be pointed at the log file.
        log::error!("service error: {e}");
    }
}

/// Called from `main` when argv is `run-as-service`. Hands control to SCM,
/// which will call back into `service_main` on its own thread.
pub fn run_dispatcher() -> Result<(), Box<dyn std::error::Error>> {
    // Hold the appender guard for the lifetime of this function — dropping it
    // flushes and closes the non-blocking writer. service_dispatcher::start
    // blocks until SCM signals stop, so the guard lives across the whole
    // service runtime.
    let _log_guard = init_service_logging()?;
    service_dispatcher::start(SERVICE_NAME, ffi_service_main)?;
    Ok(())
}

/// Prune `service.log.YYYY-MM-DD` files older than 14 days in `log_dir`.
/// Best-effort: errors are swallowed (we don't want pruning failures to
/// prevent service startup).
fn prune_old_logs(log_dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(log_dir) else {
        return;
    };
    let cutoff = std::time::SystemTime::now()
        .checked_sub(Duration::from_secs(14 * 24 * 60 * 60));
    let Some(cutoff) = cutoff else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue; };
        // Match rolled logs: "service.log.YYYY-MM-DD". Leave the active
        // "service.log" file alone (tracing-appender::rolling::daily writes
        // dated filenames by default, but be conservative).
        if !name.starts_with("service.log.") {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue; };
        let Ok(modified) = meta.modified() else { continue; };
        if modified < cutoff {
            let _ = std::fs::remove_file(&path);
        }
    }
}

fn init_service_logging() -> std::io::Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_path = config::default_log_path();
    let log_dir: std::path::PathBuf = log_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    std::fs::create_dir_all(&log_dir)?;

    // Drop old rolled files (>14 days) before opening today's file.
    prune_old_logs(&log_dir);

    // Daily-rotated, non-blocking writer. Produces files of the form
    // `<log_dir>/service.log.YYYY-MM-DD`.
    let file_appender = tracing_appender::rolling::daily(&log_dir, "service.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Bridge any `log::` macro calls (existing code paths use them) into the
    // tracing subscriber, then install the file-targeted subscriber.
    let _ = tracing_log::LogTracer::init();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(filter)
        .finish();
    // Ignore double-init errors so foreground -> service tests are tolerant.
    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(guard)
}

fn run_service(_args: Vec<OsString>) -> Result<(), Box<dyn std::error::Error>> {
    // Channel that SCM's stop callback uses to notify the async server.
    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();

    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                let _ = shutdown_tx.send(());
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    let status_handle = service_control_handler::register(SERVICE_NAME, event_handler)?;

    // StartPending while we eager-load the GGUF.
    status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::StartPending,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::from_secs(120),
        process_id: None,
    })?;

    // Resolve config from %PROGRAMDATA% file (env vars unlikely to be set
    // under SYSTEM, but `resolve` honors them when present).
    let file_cfg = config::load_file_config(&config::default_config_path())
        .map_err(|e| format!("failed to load config.toml: {e}"))?;
    let cfg = config::resolve(&file_cfg)
        .map_err(|e| format!("config resolve failed: {e}"))?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Bridge std::sync::mpsc -> tokio oneshot so axum's graceful_shutdown
    // future can await the SCM stop signal.
    let (tk_tx, tk_rx) = tokio::sync::oneshot::channel::<()>();
    std::thread::spawn(move || {
        let _ = shutdown_rx.recv();
        let _ = tk_tx.send(());
    });

    // Mark Running once we're about to call serve(). The first request can't
    // actually arrive until bind() completes inside run(), but SCM only
    // needs an upper bound on start latency, which `wait_hint` already gave.
    status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Running,
        controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    })?;

    let run_result = runtime.block_on(async move {
        crate::server::run(cfg, async move {
            let _ = tk_rx.await;
        })
        .await
    });

    // Always publish Stopped, even on error.
    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(if run_result.is_ok() { 0 } else { 1 }),
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    });

    run_result.map_err(|e| Box::<dyn std::error::Error>::from(e.to_string()))
}

// ---------------------------------------------------------------------------
// Operator subcommands
// ---------------------------------------------------------------------------

pub fn install() -> Result<(), Box<dyn std::error::Error>> {
    // Connect with CONNECT only. Requesting CREATE_SERVICE here needs
    // Administrator, so unelevated this call failed outright and `?` returned
    // before the idempotency check below could run — the 3.7.27 wheel shipped
    // that check as dead code and `install` still printed the opaque
    // "IO error in winapi call" against an already-registered service.
    //
    // The absence of a UAC prompt was the tell: the failure is at SCM CONNECT,
    // before any service call, so nothing ever asks to elevate. Querying an
    // existing service needs only CONNECT + QUERY_STATUS, both available to a
    // normal user, so the common "already installed" path now works unelevated
    // and we escalate to CREATE_SERVICE only when there is really something to
    // create.
    let service_manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;

    // Idempotency: `create_service` on an existing service fails with
    // ERROR_SERVICE_EXISTS (1073), which `windows-service` flattens into the
    // same opaque "IO error in winapi call" — indistinguishable from a
    // permission failure. m3 setup then declared the embedder "SKIPPED (not
    // installed)" and told the operator to re-run elevated, while the service
    // was already registered, Automatic, and serving :8082. Re-registering is a
    // no-op, so report success rather than erroring.
    if let Ok(existing) = service_manager.open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS) {
        let state = existing.query_status().map(|s| s.current_state).ok();
        println!("service already installed: {SERVICE_NAME}");
        match state {
            Some(ServiceState::Running) => println!("state: running (nothing to do)"),
            Some(_) => println!("state: stopped — start it with `m3-embed-server start`"),
            None => println!("state: unknown (could not query SCM)"),
        }
        println!("to re-register from scratch: `m3-embed-server uninstall` then `install`");
        return Ok(());
    }

    // Nothing registered — creating one genuinely needs Administrator. Reconnect
    // asking for CREATE_SERVICE and translate the access-denied case into advice
    // instead of the opaque winapi string (§3 fail loud, and loudly ACCURATE:
    // the old message could not tell "already installed" from "needs elevation",
    // which is what sent the operator chasing the wrong fix).
    let service_manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE,
    )
    .map_err(|e| {
        Box::<dyn std::error::Error>::from(format!(
            "cannot register the service: {e}\n  \
             Registering a Windows Service requires Administrator rights.\n  \
             Open an *Administrator* terminal and run: m3-embed-server install\n  \
             (The service is NOT currently registered — this is not the \
             already-installed case.)"
        ))
    })?;

    let exe = std::env::current_exe()?;
    let service_info = ServiceInfo {
        name: OsString::from(SERVICE_NAME),
        display_name: OsString::from(SERVICE_DISPLAY),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: exe,
        launch_arguments: vec![OsString::from("run-as-service")],
        dependencies: vec![],
        account_name: None, // Local System
        account_password: None,
    };

    let service =
        service_manager.create_service(&service_info, ServiceAccess::CHANGE_CONFIG | ServiceAccess::START)?;
    service.set_description(SERVICE_DESC)?;

    // Best-effort: configure SCM recovery actions. `windows-service` 0.8
    // doesn't expose `ChangeServiceConfig2`, so we shell out to sc.exe (base
    // Windows since XP). A failure here is non-fatal — the service is
    // already registered, the operator can re-run the command by hand.
    match configure_recovery_actions() {
        Ok(()) => println!("recovery actions: restart x3, 5s delay, 60s reset window"),
        Err(e) => eprintln!(
            "WARN: recovery actions not configured ({e}). Run manually:\n  \
             sc.exe failure {SERVICE_NAME} reset= 60 actions= restart/5000/restart/5000/restart/5000"
        ),
    }

    // Snapshot current env into the config file so SYSTEM-account service
    // can find the GGUF. Don't clobber an existing file.
    let cfg_path = config::default_config_path();
    if !cfg_path.exists() {
        let snapshot = config::snapshot_env_to_file();
        config::write_config_file(&cfg_path, &snapshot)?;
        println!("wrote starter config: {}", cfg_path.display());
    } else {
        println!("config already exists, not overwritten: {}", cfg_path.display());
    }

    println!("service installed: {SERVICE_NAME}");
    println!("log file:          {}", config::default_log_path().display());
    println!();
    println!("Next steps:");
    println!("  1. Edit {} to confirm [embed].gguf is set.", cfg_path.display());
    println!("  2. Start it:  m3-embed-server start    (or: sc start {SERVICE_NAME})");
    Ok(())
}

/// Configure SCM recovery actions for the service. Equivalent to:
///   sc.exe failure m3-embed-server reset= 60 actions= restart/5000/restart/5000/restart/5000
/// Note: sc.exe is *picky* — there must be a space AFTER the `=` in each
/// `key= value` pair, and the args must be passed as separate tokens (not a
/// single command-line string).
fn configure_recovery_actions() -> Result<(), String> {
    let output = std::process::Command::new("sc.exe")
        .args([
            "failure",
            SERVICE_NAME,
            "reset=",
            "60",
            "actions=",
            "restart/5000/restart/5000/restart/5000",
        ])
        .output()
        .map_err(|e| format!("failed to spawn sc.exe: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "sc.exe failure exited {}: {} {}",
            output.status,
            stderr.trim(),
            stdout.trim()
        ));
    }
    Ok(())
}

pub fn uninstall() -> Result<(), Box<dyn std::error::Error>> {
    let manager_access = ServiceManagerAccess::CONNECT;
    let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

    // Check "is it even installed?" with QUERY_STATUS FIRST. The privileged open
    // below asks for STOP|DELETE, which an unprivileged user cannot open
    // (ERROR_ACCESS_DENIED, 5) — so unelevated it failed there and the
    // ERROR_SERVICE_DOES_NOT_EXIST (1060) arm was unreachable. Uninstalling a
    // service that is already absent is the desired end state, but it reported
    // the opaque "IO error in winapi call" and exit 1 instead.
    if !service_exists(&service_manager) {
        println!("service not installed: {SERVICE_NAME}");
        return Ok(());
    }

    let service_access =
        ServiceAccess::QUERY_STATUS | ServiceAccess::STOP | ServiceAccess::DELETE;
    let service = match service_manager.open_service(SERVICE_NAME, service_access) {
        Ok(s) => s,
        Err(windows_service::Error::Winapi(e))
            if e.raw_os_error() == Some(1060) =>
        {
            println!("service not installed: {SERVICE_NAME}");
            return Ok(());
        }
        Err(e) => {
            return Err(Box::<dyn std::error::Error>::from(format!(
                "cannot remove the service: {e}\n  \
                 Removing a Windows Service requires Administrator rights.\n  \
                 Open an *Administrator* terminal and run: m3-embed-server uninstall\n  \
                 (The service IS currently registered — `m3-embed-server status` \
                 shows its state.)"
            )))
        }
    };

    // Best-effort stop.
    if let Ok(status) = service.query_status() {
        if status.current_state != ServiceState::Stopped {
            let _ = service.stop();
            for _ in 0..20 {
                std::thread::sleep(Duration::from_millis(500));
                if let Ok(s) = service.query_status() {
                    if s.current_state == ServiceState::Stopped {
                        break;
                    }
                }
            }
        }
    }

    service.delete()?;
    println!("service removed: {SERVICE_NAME}");
    println!(
        "config file left in place: {} (delete manually if desired)",
        config::default_config_path().display()
    );
    Ok(())
}

/// ERROR_SERVICE_ALREADY_RUNNING — `start` on a service that is already up.
const ERROR_SERVICE_ALREADY_RUNNING: i32 = 1056;
/// ERROR_SERVICE_NOT_ACTIVE — `stop` on a service that is already stopped.
const ERROR_SERVICE_NOT_ACTIVE: i32 = 1062;

// PRIVILEGE MAP (measured on Windows 11, unelevated, 2026-07-27):
//
//   SCM      CONNECT ............. OK        CONNECT|CREATE_SERVICE .. DENIED(5)
//   service  QUERY_STATUS ........ OK        START ................... DENIED(5)
//            QUERY_CONFIG ........ OK        STOP .................... DENIED(5)
//            ENUMERATE_DEPENDENTS  OK        CHANGE_CONFIG ........... DENIED(5)
//                                            DELETE .................. DENIED(5)
//
// Every MUTATING right needs Administrator; the read rights do not. Because
// `open_service` fails as a whole, requesting a mutating right up front means
// the call dies with ERROR_ACCESS_DENIED before any state check can run — and
// `windows-service` flattens that into "IO error in winapi call", which is
// indistinguishable from every other failure.
//
// Hence the rule these helpers exist to enforce: ANSWER "IS THERE WORK TO DO?"
// WITH READ ACCESS FIRST, AND ESCALATE ONLY WHEN THERE IS. Four subcommands
// shipped without it (install through 3.7.27; start/stop/uninstall through
// 3.7.28), each reporting a no-op as a failure.
//
// WINDOWS-ONLY BY CONSTRUCTION. This module is `cfg(windows)`; `service_unix.rs`
// serves macOS (launchd) and Linux (systemd) and does NOT have this flaw:
// `systemctl start` on a running unit and `launchctl kickstart` on a running job
// both exit 0, and both `uninstall` paths already guard on `plist.exists()` /
// `unit.exists()` before acting. The asymmetry is real, not an oversight there —
// SCM is the only one of the three that refuses to even OPEN a handle when you
// ask for a right you lack, which is what makes the check-before-escalate order
// mandatory here and merely tidy elsewhere. (Audited across all three OSes,
// 2026-07-27.)

/// True when the service is registered at all. QUERY_STATUS only, so it works
/// unelevated — unlike an open that asks for DELETE.
fn service_exists(manager: &ServiceManager) -> bool {
    manager
        .open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS)
        .is_ok()
}

/// True when the service is already in `want`. Uses QUERY_STATUS only, which an
/// unprivileged user CAN open — unlike START/STOP.
fn already_in_state(manager: &ServiceManager, want: ServiceState) -> bool {
    manager
        .open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS)
        .ok()
        .and_then(|s| s.query_status().ok())
        .map(|s| s.current_state == want)
        .unwrap_or(false)
}

pub fn start() -> Result<(), Box<dyn std::error::Error>> {
    let service_manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;

    // Check BEFORE asking for privileged access. `open_service` with START (or
    // STOP) requires Administrator and returns ERROR_ACCESS_DENIED (5)
    // unelevated — verified at the Win32 layer — so the privileged open failed
    // first and the caller got the opaque "IO error in winapi call" even when
    // the service was already running and there was nothing to do. `m3 setup`
    // surfaced that as "`m3-embed-server start` exited 1" against a service
    // that was Automatic, running, and serving :8082 (2026-07-27).
    //
    // QUERY_STATUS opens fine for a normal user, so answer the "is this already
    // done?" question with the access we HAVE, and only escalate when there is
    // real work. Same shape as the install() fix one level down.
    if already_in_state(&service_manager, ServiceState::Running) {
        println!("{SERVICE_NAME} is already running (nothing to do)");
        return Ok(());
    }

    let service = service_manager
        .open_service(SERVICE_NAME, ServiceAccess::START | ServiceAccess::QUERY_STATUS)
        .map_err(|e| {
            Box::<dyn std::error::Error>::from(format!(
                "cannot start the service: {e}\n  \
                 Starting a Windows Service requires Administrator rights.\n  \
                 Open an *Administrator* terminal and run: m3-embed-server start\n  \
                 (It is not already running — `m3-embed-server status` shows the \
                 current state.)"
            ))
        })?;

    // Belt-and-braces: it could have started between the check and here.
    match service.start::<&str>(&[]) {
        Ok(()) => println!("start signal sent to {SERVICE_NAME}"),
        Err(windows_service::Error::Winapi(e))
            if e.raw_os_error() == Some(ERROR_SERVICE_ALREADY_RUNNING) =>
        {
            println!("{SERVICE_NAME} is already running (nothing to do)");
        }
        Err(e) => return Err(Box::new(e)),
    }
    Ok(())
}

pub fn stop() -> Result<(), Box<dyn std::error::Error>> {
    let service_manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;

    // Mirror of start(): STOP access also needs Administrator, so answer
    // "already stopped?" with QUERY_STATUS before escalating.
    if already_in_state(&service_manager, ServiceState::Stopped) {
        println!("{SERVICE_NAME} is already stopped (nothing to do)");
        return Ok(());
    }

    let service = service_manager
        .open_service(SERVICE_NAME, ServiceAccess::STOP | ServiceAccess::QUERY_STATUS)
        .map_err(|e| {
            Box::<dyn std::error::Error>::from(format!(
                "cannot stop the service: {e}\n  \
                 Stopping a Windows Service requires Administrator rights.\n  \
                 Open an *Administrator* terminal and run: m3-embed-server stop\n  \
                 (It is not already stopped — `m3-embed-server status` shows the \
                 current state.)"
            ))
        })?;

    match service.stop() {
        Ok(_) => println!("stop signal sent to {SERVICE_NAME}"),
        Err(windows_service::Error::Winapi(e))
            if e.raw_os_error() == Some(ERROR_SERVICE_NOT_ACTIVE) =>
        {
            println!("{SERVICE_NAME} is already stopped (nothing to do)");
        }
        Err(e) => return Err(Box::new(e)),
    }
    Ok(())
}

pub fn status() -> Result<(), Box<dyn std::error::Error>> {
    let service_manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = match service_manager.open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS) {
        Ok(s) => s,
        Err(windows_service::Error::Winapi(e))
            if e.raw_os_error() == Some(1060) =>
        {
            println!("not installed");
            return Ok(());
        }
        Err(e) => return Err(Box::new(e)),
    };
    let st = service.query_status()?;
    let label = match st.current_state {
        ServiceState::Stopped => "stopped",
        ServiceState::StartPending => "start-pending",
        ServiceState::StopPending => "stop-pending",
        ServiceState::Running => "running",
        ServiceState::ContinuePending => "continue-pending",
        ServiceState::PausePending => "pause-pending",
        ServiceState::Paused => "paused",
    };
    println!("{label}");
    Ok(())
}
