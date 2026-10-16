//! `m3-dispatchd --endpoint PATH [--coalesce-ms N] [--keep-s N]`
//!
//! The endpoint is always given by the caller (m3's OS seam owns its default),
//! so its location is decided in one place.

use std::sync::Arc;
use std::time::Duration;

use m3_dispatchd::{api, hub::Hub, ipc};

const USAGE: &str = "usage: m3-dispatchd --endpoint PATH [--coalesce-ms 250] [--keep-s 3600]";

struct Args {
    endpoint: String,
    window_ms: u64,
    keep_s: u64,
}

/// Every argument is known or refused: an ignored typo (`--keep 60`) would
/// run with the default and say nothing.
fn parse(argv: &[String]) -> anyhow::Result<Args> {
    let (mut endpoint, mut window_ms, mut keep_s) = (None, 250u64, 3600u64);
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value\n{USAGE}"))
        };
        let number = |v: String| {
            v.parse::<u64>().map_err(|e| {
                anyhow::anyhow!(
                    "{a} {v:?} is not a whole number of {}: {e}",
                    if a == "--keep-s" {
                        "seconds"
                    } else {
                        "milliseconds"
                    }
                )
            })
        };
        match a.as_str() {
            "--endpoint" => endpoint = Some(val()?),
            "--coalesce-ms" => window_ms = number(val()?)?,
            "--keep-s" => keep_s = number(val()?)?,
            other => anyhow::bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    let endpoint = endpoint
        .ok_or_else(|| anyhow::anyhow!("--endpoint PATH is required (m3 passes it)\n{USAGE}"))?;
    Ok(Args {
        endpoint,
        window_ms,
        keep_s,
    })
}

/// Resolves on Ctrl-C. A handler that cannot be installed is warned about and
/// never resolves: an `Err` is not a stop request, and treating it as one shut
/// the daemon down the moment it started.
async fn ctrl_c_or_never() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        log::warn!(
            "the Ctrl-C handler could not be installed ({e}); Ctrl-C will not stop this daemon \
             cleanly, so the endpoint may be left behind"
        );
        std::future::pending::<()>().await;
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let Args {
        endpoint,
        window_ms,
        keep_s,
    } = parse(&argv)?;
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
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(mut term) => {
                        tokio::select! {
                            _ = ctrl_c_or_never() => {}
                            _ = term.recv() => {}
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "the SIGTERM handler could not be installed ({e}); a service manager's \
                             stop will kill this daemon without removing its socket"
                        );
                        ctrl_c_or_never().await;
                    }
                }
            }
            #[cfg(not(unix))]
            {
                ctrl_c_or_never().await;
            }
            log::info!("shutting down");
        };
        ipc::serve(&endpoint, app, shutdown).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn arguments_are_known_or_refused_by_name() {
        let a = parse(&v(&["--endpoint", "/s", "--keep-s", "60"])).unwrap();
        assert_eq!(
            (a.endpoint.as_str(), a.window_ms, a.keep_s),
            ("/s", 250, 60)
        );
        let e = parse(&v(&["--endpoint", "/s", "--keep", "60"]))
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("unknown argument \"--keep\""), "{e}");
        let e = parse(&v(&["--endpoint", "/s", "--coalesce-ms", "2x"]))
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("--coalesce-ms \"2x\""), "{e}");
        assert!(parse(&v(&["--keep-s", "1"]))
            .err()
            .unwrap()
            .to_string()
            .contains("--endpoint"));
        assert!(parse(&v(&["--endpoint"]))
            .err()
            .unwrap()
            .to_string()
            .contains("needs a value"));
    }
}
