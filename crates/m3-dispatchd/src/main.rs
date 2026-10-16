//! `m3-dispatchd --endpoint PATH [--coalesce-ms N] [--keep-s N]`
//!
//! The endpoint is always given by the caller (m3's OS seam owns its default),
//! so its location is decided in one place.

use std::sync::Arc;
use std::time::Duration;

use m3_dispatchd::{api, hub::Hub, ipc};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("usage: m3-dispatchd --endpoint PATH [--coalesce-ms 250] [--keep-s 3600]");
        return Ok(());
    }
    let endpoint = arg(&args, "--endpoint")
        .ok_or_else(|| anyhow::anyhow!("--endpoint PATH is required (m3 passes it)"))?;
    let window_ms: u64 = arg(&args, "--coalesce-ms")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(250);
    let keep_s: u64 = arg(&args, "--keep-s")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(3600);
    let hub = Arc::new(Hub::new(
        Duration::from_millis(window_ms),
        Duration::from_secs(keep_s),
    ));
    let app = api::router(hub);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(async move {
        // Service managers stop with SIGTERM (systemd, launchd), a console with
        // Ctrl-C; either must reach the clean exit that removes the socket.
        let shutdown = async {
            #[cfg(unix)]
            {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("installing a SIGTERM handler");
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
            log::info!("shutting down");
        };
        ipc::serve(&endpoint, app, shutdown).await
    })
}
