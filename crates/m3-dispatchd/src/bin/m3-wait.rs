//! `m3-wait --endpoint PATH --reader ID [--reader ID ...] [--timeout S] [-- CMD ARGS...]`
//!
//! Blocks on m3-dispatchd until a reader has mail, holding a few MB instead of
//! a Python process with m3 loaded. Without CMD it prints the wake as JSON. With
//! CMD it runs CMD on each wake (the wake JSON in `M3_WAKE`, stdio inherited):
//! CMD's exit 0 ends the wait, 75 (EX_TEMPFAIL: nothing new after all) waits
//! again, and any other code is passed through. CMD is how m3 confirms and
//! stamps receipt in its store: the daemon issues no SQL.
//!
//! Exit: 0 woke, 1 error, 2 timed out, 3 no daemon (the caller polls instead).

use std::process::ExitCode;
use std::time::{Duration, Instant};

use m3_dispatchd::client::{post, ClientError};

const KEEP_WAITING: i32 = 75;
/// The daemon caps one wait at an hour (api::MAX_WAIT); longer waits re-wait.
const ONE_WAIT_S: f64 = 3600.0;

#[derive(Debug)]
struct Args {
    endpoint: String,
    readers: Vec<String>,
    timeout: Duration,
    then: Vec<String>,
}

fn usage() -> &'static str {
    "usage: m3-wait --endpoint PATH --reader ID [--reader ID ...] [--timeout SECONDS] [-- CMD ARGS...]"
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let (own, then) = match argv.iter().position(|a| a == "--") {
        Some(i) => (&argv[..i], argv[i + 1..].to_vec()),
        None => (argv, Vec::new()),
    };
    let (mut endpoint, mut readers, mut timeout) = (None, Vec::new(), 3600.0_f64);
    let mut it = own.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{a} needs a value"))
        };
        match a.as_str() {
            "--endpoint" => endpoint = Some(val()?),
            "--reader" => readers.push(val()?),
            "--timeout" => {
                timeout = val()?.parse().map_err(|e| format!("--timeout: {e}"))?;
                if !(timeout.is_finite() && timeout >= 0.0) {
                    return Err(format!(
                        "--timeout must be a non-negative number; got {timeout}"
                    ));
                }
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    readers.retain(|r| !r.is_empty());
    if readers.is_empty() {
        return Err("at least one --reader is required".into());
    }
    let endpoint = endpoint.ok_or("--endpoint PATH is required (m3 passes it)")?;
    Ok(Args {
        endpoint,
        readers,
        timeout: Duration::from_secs_f64(timeout),
        then,
    })
}

/// Run CMD for one wake; its exit code.
fn run_then(then: &[String], wake: &str) -> Result<i32, String> {
    let status = std::process::Command::new(&then[0])
        .args(&then[1..])
        .env("M3_WAKE", wake)
        .status()
        .map_err(|e| format!("cannot run {}: {e}", then[0]))?;
    // A child killed by a signal has no code; treat it as an error.
    Ok(status.code().unwrap_or(1))
}

async fn run(args: Args) -> u8 {
    let deadline = Instant::now() + args.timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            eprintln!(
                "{}",
                serde_json::json!({ "timeout": true, "readers": args.readers })
            );
            return 2;
        }
        let body = serde_json::json!({ "readers": args.readers, "timeout_s": left.as_secs_f64().min(ONE_WAIT_S) });
        let (status, reply) = match post(&args.endpoint, "/wait", &body).await {
            Ok(r) => r,
            Err(e @ ClientError::NoDaemon(_)) => {
                eprintln!("m3-wait: {e}");
                return 3;
            }
            Err(e) => {
                eprintln!("m3-wait: {e}");
                return 1;
            }
        };
        let text = String::from_utf8_lossy(&reply).into_owned();
        if status != 200 {
            eprintln!("m3-wait: m3-dispatchd answered {status}: {text}");
            return 1;
        }
        let wake: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("m3-wait: unreadable reply from m3-dispatchd ({e}): {text}");
                return 1;
            }
        };
        if wake["wake"] == "timeout" {
            continue;
        }
        if args.then.is_empty() {
            println!("{text}");
            return 0;
        }
        match run_then(&args.then, &text) {
            Ok(0) => return 0,
            Ok(KEEP_WAITING) => continue,
            Ok(code) => return u8::try_from(code).unwrap_or(1),
            Err(e) => {
                eprintln!("m3-wait: {e}");
                return 1;
            }
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv
        .iter()
        .take_while(|a| *a != "--")
        .any(|a| a == "--help" || a == "-h")
    {
        println!("{}", usage());
        return ExitCode::SUCCESS;
    }
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("m3-wait: {e}\n{}", usage());
            return ExitCode::from(1);
        }
    };
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("m3-wait: cannot start the runtime: {e}");
            return ExitCode::from(1);
        }
    };
    ExitCode::from(rt.block_on(run(args)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn arguments() {
        let a = parse(&v(&[
            "--endpoint",
            "/s",
            "--reader",
            "a",
            "--reader",
            "b@1",
            "--timeout",
            "5",
            "--",
            "py",
            "-x",
        ]))
        .unwrap();
        assert_eq!(
            (a.endpoint.as_str(), a.readers, a.timeout, a.then),
            (
                "/s",
                v(&["a", "b@1"]),
                Duration::from_secs(5),
                v(&["py", "-x"])
            )
        );
        assert!(parse(&v(&["--endpoint", "/s"]))
            .unwrap_err()
            .contains("--reader"));
        assert!(parse(&v(&["--reader", "a"]))
            .unwrap_err()
            .contains("--endpoint"));
        assert!(parse(&v(&[
            "--endpoint",
            "/s",
            "--reader",
            "a",
            "--timeout",
            "-1"
        ]))
        .is_err());
        assert!(parse(&v(&["--endpoint", "/s", "--reader", "a", "--bogus"])).is_err());
    }
}
