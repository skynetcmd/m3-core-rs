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
    init_service_logging()?;
    service_dispatcher::start(SERVICE_NAME, ffi_service_main)?;
    Ok(())
}

fn init_service_logging() -> std::io::Result<()> {
    let log_path = config::default_log_path();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    // env_logger writes through std::io::Write — use a Mutex-wrapped File via
    // the `Pipe` target so concurrent log macros serialize cleanly.
    let mut builder = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    );
    builder.target(env_logger::Target::Pipe(Box::new(file)));
    // Ignore double-init errors so foreground -> service tests are tolerant.
    let _ = builder.try_init();
    Ok(())
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
    let manager_access = ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE;
    let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

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
    println!();
    println!("NOTE: recovery actions (restart on crash) are not set by this installer.");
    println!("      Configure via Services.msc -> Properties -> Recovery, or run:");
    println!("      sc failure {SERVICE_NAME} reset= 60 actions= restart/5000/restart/5000/restart/5000");
    Ok(())
}

pub fn uninstall() -> Result<(), Box<dyn std::error::Error>> {
    let manager_access = ServiceManagerAccess::CONNECT;
    let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

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
        Err(e) => return Err(Box::new(e)),
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

pub fn start() -> Result<(), Box<dyn std::error::Error>> {
    let service_manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = service_manager.open_service(
        SERVICE_NAME,
        ServiceAccess::START | ServiceAccess::QUERY_STATUS,
    )?;
    service.start::<&str>(&[])?;
    println!("start signal sent to {SERVICE_NAME}");
    Ok(())
}

pub fn stop() -> Result<(), Box<dyn std::error::Error>> {
    let service_manager =
        ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = service_manager.open_service(
        SERVICE_NAME,
        ServiceAccess::STOP | ServiceAccess::QUERY_STATUS,
    )?;
    service.stop()?;
    println!("stop signal sent to {SERVICE_NAME}");
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
