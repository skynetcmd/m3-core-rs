//! Unix service-manager integration — the non-Windows counterpart of
//! `service.rs`. Provides the same operator lifecycle (`install` / `uninstall`
//! / `start` / `stop` / `status`) backed by:
//!
//! - **macOS** — a per-user `launchd` agent (`~/Library/LaunchAgents/<label>.plist`),
//!   driven via `launchctl`.
//! - **Linux** — a `systemd --user` unit (`~/.config/systemd/user/<name>.service`),
//!   driven via `systemctl --user`.
//!
//! Design notes (see also the crate-level plan):
//!
//! * **User-level, not system-level.** Unlike the Windows service (Local
//!   System), these are *user* agents. That means **no `sudo`/root** to
//!   install — the embedder only serves `127.0.0.1:8082` for one user's
//!   m3-memory, so a user agent is both sufficient and lower-friction.
//! * **No service-manager API crate.** launchd and systemd have no Rust API;
//!   they are driven by writing a unit file then invoking `launchctl` /
//!   `systemctl`. This mirrors how `service.rs` already shells out to `sc.exe`
//!   for recovery actions.
//! * **`ExecStart` runs the binary with no subcommand** → foreground mode.
//!   systemd / launchd *are* the supervisor; the process just runs in the
//!   foreground and exits cleanly on SIGTERM (handled in `main::run_foreground`).
//!
//! The whole module is `#[cfg(not(windows))]`; on a Unix that is neither macOS
//! nor Linux the public fns return a clear "unsupported platform" error.

#![cfg(all(not(windows), feature = "embedded"))]

use std::path::PathBuf;

use crate::config;
// The `unit_render` items are imported per-submodule below — `render_plist` /
// `LAUNCHD_LABEL` are macOS-only, `render_unit` / `SERVICE_NAME` Linux-only —
// so a single-platform build has no unused-import warning.

/// Resolve `$HOME`, erroring clearly when unset (cron-like contexts).
fn home_dir() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| {
            "$HOME is not set — cannot locate the per-user service directory".to_string()
        })
}

/// Absolute path to this executable — what the unit file's `ExecStart` /
/// `ProgramArguments` must point at so the supervisor can re-launch it.
fn current_exe() -> Result<PathBuf, String> {
    std::env::current_exe().map_err(|e| format!("cannot resolve current executable path: {e}"))
}

/// The GGUF path the service must embed with. The unit file pins it into the
/// service environment so a login-less supervisor still finds the model.
/// Resolved here (install time) the same way `config::resolve` does at runtime.
fn resolve_gguf_for_unit() -> Result<String, String> {
    if let Some(g) = std::env::var("M3_EMBED_GGUF")
        .ok()
        .filter(|s| !s.is_empty())
    {
        return Ok(g);
    }
    // Fall back to a config.toml that an earlier run may have written.
    let path = config::default_config_path();
    let file_cfg = config::load_file_config(&path).map_err(|e| e.to_string())?;
    file_cfg.embed.gguf.clone().ok_or_else(|| {
        format!(
            "M3_EMBED_GGUF is unset and {} has no [embed].gguf — set the env var \
             before `install` so the service can find the model",
            path.display()
        )
    })
}

/// Record the settings `install` was given (port, host, streams...) in
/// config.toml, where the supervised service reads them: the unit pins only
/// the model, so a port handed to `install` used to be dropped.
#[allow(dead_code)] // each Unix target calls it from its own install
fn record_settings() -> Result<(), String> {
    let path = config::default_config_path();
    config::merge_env_into_file(&path).map_err(|e| {
        format!(
            "cannot record install settings in {}: {e:#}",
            path.display()
        )
    })
}

/// The service's address from config.toml, as an error string for this module.
fn service_addr() -> Result<(String, u16), String> {
    config::service_addr().map_err(|e| format!("cannot tell where the service listens: {e:#}"))
}

/// What a service exit status means; the server sets these (main.rs).
#[allow(dead_code)] // each Unix target calls it from its own status
fn exit_status_meaning(code: i64) -> &'static str {
    match code {
        0 => "exited cleanly",
        69 => "the listen port is taken by another program",
        78 => "the configuration was refused",
        _ => "startup failed",
    }
}

/// What `/health` said. `Ok(Some(pid))`: it answered ok and named its process
/// (3.10.14+). `Ok(None)`: an older server that answers ok without a pid.
/// `Err`: why it is not serving (refused, timed out, loading, degraded...).
/// `/health` answers ok only once the model is loaded, so this is "serving",
/// which a supervisor's "running" (the process exists) is not.
fn health(host: &str, port: u16) -> Result<Option<u32>, String> {
    use std::io::{Read, Write};
    use std::net::ToSocketAddrs;
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve {host}:{port}: {e}"))?
        .next()
        .ok_or_else(|| format!("{host}:{port} resolves to no address"))?;
    let timeout = std::time::Duration::from_secs(2);
    let mut stream = std::net::TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| format!("cannot connect to {addr}: {e}"))?;
    let _ = stream.set_read_timeout(Some(timeout));
    let req = format!("GET /health HTTP/1.0\r\nHost: {host}\r\n\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("cannot send to {addr}: {e}"))?;
    let mut response = String::new();
    if let Err(e) = stream.read_to_string(&mut response) {
        if response.is_empty() {
            return Err(format!("no answer from {addr}: {e}"));
        }
    }
    parse_health(&response)
}

/// The response half of [`health`], split out so it is testable without a socket.
fn parse_health(response: &str) -> Result<Option<u32>, String> {
    if response.is_empty() {
        return Err("connected, but no HTTP answer".into());
    }
    let status_line = response.lines().next().unwrap_or_default();
    let code = status_line.split_whitespace().nth(1).unwrap_or("?");
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .unwrap_or(response);
    let v: serde_json::Value = serde_json::from_str(body)
        .map_err(|_| format!("answered {code}, but not with m3-embed-server's JSON health body"))?;
    match v.get("status").and_then(|s| s.as_str()) {
        Some("ok") => Ok(v
            .get("pid")
            .and_then(|p| p.as_u64())
            .and_then(|p| u32::try_from(p).ok())),
        Some("degraded") => Err(format!(
            "answered {code} degraded: {}",
            v.get("reason")
                .and_then(|r| r.as_str())
                .unwrap_or("no reason given")
        )),
        Some(s) if s.starts_with("loading") => {
            Err(format!("answered {code} {s:?}: still loading the model"))
        }
        Some(s) => Err(format!("answered {code} with status {s:?}")),
        None => Err(format!("answered {code} without a status")),
    }
}

/// Whose server answered, given `/health`'s pid and the supervisor's main pid.
#[derive(Debug, PartialEq)]
enum Answer {
    /// The service manager's own process answered.
    Ours,
    /// Another process answered (`pid`): the service manager runs a different one.
    Foreign(u32),
    /// `pid` answered, and the service manager named no main process to
    /// compare (it has none, or could not be asked): probably another process.
    LikelyForeign(u32),
    /// Cannot tell: an older server without a pid.
    Unknown,
}

fn whose(health_pid: Option<u32>, main_pid: Option<u32>) -> Answer {
    match (health_pid, main_pid) {
        (Some(h), Some(m)) if h == m => Answer::Ours,
        (Some(h), Some(_)) => Answer::Foreign(h),
        (Some(h), None) => Answer::LikelyForeign(h),
        (None, _) => Answer::Unknown,
    }
}

fn foreign_message(label: &str, host: &str, port: u16, pid: u32, certain: bool) -> String {
    let who = if certain {
        format!("is answered by another process (pid {pid}), not this service")
    } else {
        format!(
            "is answered by pid {pid}, and the service manager reports no process for this \
             service (or could not be asked), so it is probably another process"
        )
    };
    format!(
        "{label}: {host}:{port} {who}; this service cannot use the port while it is taken. On a \
         shared computer this is often another account's m3. Choose a free port: \
         m3 config set embed.server.port <port>"
    )
}

/// After the supervisor reports the process running, wait for it to serve and
/// say which it is. `answered_before` is whether the address already answered
/// before this start: then an ok from a server without a pid may be another
/// process. `main_pid` asks the supervisor which process it runs now.
fn report_serving(
    label: &str,
    answered_before: bool,
    secs: u64,
    log_hint: &str,
    main_pid: &dyn Fn() -> Option<u32>,
) -> Result<(), String> {
    let (host, port) = service_addr()?;
    let mut last = String::from("not probed");
    for _ in 0..(secs * 2) {
        let pid = match health(&host, port) {
            Ok(pid) => pid,
            Err(why) => {
                last = why;
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
        };
        match whose(pid, main_pid()) {
            Answer::Ours => println!("{label}: running, serving on {host}:{port}"),
            Answer::Foreign(other) => return Err(foreign_message(label, &host, port, other, true)),
            Answer::LikelyForeign(other) => {
                return Err(foreign_message(label, &host, port, other, false))
            }
            Answer::Unknown if answered_before => {
                eprintln!(
                    "WARN: {host}:{port} was already answering before this start; another \
                     process may hold the port, so this service may not be the one serving"
                );
                println!(
                    "{label}: running; {host}:{port} answers, but it did before this start too"
                );
            }
            Answer::Unknown => println!("{label}: running, serving on {host}:{port}"),
        }
        return Ok(());
    }
    // The process exists, but nothing answered ok: not a success.
    Err(format!(
        "{label}: the process is running, but {host}:{port} did not serve within {secs}s \
         (last probe: {last}); {log_hint}"
    ))
}

/// For `status`: a note when the port answers from a process other than the
/// service's own. stdout stays the one state word m3 parses.
fn note_foreign_answer(label: &str, main_pid: Option<u32>) {
    let (host, port) = match service_addr() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{label}: {e}");
            return;
        }
    };
    if let Ok(Some(pid)) = health(&host, port) {
        match whose(Some(pid), main_pid) {
            Answer::Foreign(other) => {
                eprintln!("{}", foreign_message(label, &host, port, other, true))
            }
            Answer::LikelyForeign(other) => {
                eprintln!("{}", foreign_message(label, &host, port, other, false))
            }
            Answer::Ours | Answer::Unknown => {}
        }
    }
}

/// Whether the service's address answers right now.
fn serving_now() -> bool {
    match service_addr() {
        Ok((host, port)) => health(&host, port).is_ok(),
        Err(e) => {
            eprintln!("WARN: {e}; assuming nothing answered before this start");
            false
        }
    }
}

#[cfg(test)]
mod health_tests {
    use super::*;

    const OK_WITH_PID: &str =
        "HTTP/1.0 200 OK\r\ncontent-type: application/json\r\n\r\n{\"status\":\"ok\",\"model\":\"bge-m3.gguf\",\"pid\":4242}";

    #[test]
    fn a_new_server_names_its_pid() {
        assert_eq!(parse_health(OK_WITH_PID), Ok(Some(4242)));
    }

    #[test]
    fn an_older_server_answers_without_a_pid() {
        let r = "HTTP/1.0 200 OK\r\n\r\n{\"status\":\"ok\",\"model\":\"bge-m3.gguf\"}";
        assert_eq!(parse_health(r), Ok(None));
    }

    #[test]
    fn a_loading_degraded_or_wrong_server_is_not_ok_and_says_why() {
        let loading =
            parse_health("HTTP/1.0 503 Service Unavailable\r\n\r\n{\"status\":\"loading model\"}");
        assert!(loading.unwrap_err().contains("still loading"));
        let degraded = parse_health(
            "HTTP/1.0 503 Service Unavailable\r\n\r\n{\"status\":\"degraded\",\"reason\":\"breaker open\",\"pid\":1}",
        );
        assert!(degraded.unwrap_err().contains("degraded: breaker open"));
        assert!(parse_health("").unwrap_err().contains("no HTTP answer"));
        assert!(parse_health("HTTP/1.0 404 Not Found\r\n\r\nnope")
            .unwrap_err()
            .contains("404"));
    }

    #[test]
    fn the_answer_is_ours_only_when_the_pids_match() {
        assert_eq!(whose(Some(7), Some(7)), Answer::Ours);
        assert_eq!(whose(Some(7), Some(8)), Answer::Foreign(7));
        // No main pid: our process may be gone (it exited because the port
        // was taken) or the supervisor could not be asked; probably foreign.
        assert_eq!(whose(Some(7), None), Answer::LikelyForeign(7));
        assert_eq!(whose(None, Some(7)), Answer::Unknown);
    }
}

// ===========================================================================
// macOS — launchd user agent
// ===========================================================================
//
// Built and exercised on a real Mac (Apple Silicon, launchd gui domain); the
// release wheels for macOS are built natively there. Cross-compiling to
// *-apple-darwin still fails at the llama.cpp/ring C build, so a change here
// needs a Mac to verify.
#[cfg(target_os = "macos")]
pub mod macos {
    use super::*;
    use crate::unit_render::{render_plist, LAUNCHD_LABEL};

    fn plist_path() -> Result<PathBuf, String> {
        Ok(home_dir()?
            .join("Library")
            .join("LaunchAgents")
            .join(format!("{LAUNCHD_LABEL}.plist")))
    }

    /// `gui/<uid>` — the launchd domain for the calling user's GUI session.
    /// The uid comes from `id -u` (universally present on macOS) — avoids a
    /// `libc` dependency just for `getuid()`.
    fn gui_domain() -> Result<String, String> {
        let out = std::process::Command::new("id")
            .arg("-u")
            .output()
            .map_err(|e| format!("failed to spawn `id -u`: {e}"))?;
        if !out.status.success() {
            return Err(format!("`id -u` exited {}", out.status));
        }
        let uid = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if uid.is_empty() || !uid.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("`id -u` returned an unexpected value: {uid:?}"));
        }
        Ok(format!("gui/{uid}"))
    }

    fn service_target() -> Result<String, String> {
        Ok(format!("{}/{LAUNCHD_LABEL}", gui_domain()?))
    }

    pub fn install() -> Result<(), String> {
        let exe = current_exe()?;
        let gguf = resolve_gguf_for_unit()?;
        record_settings()?;
        let log_path = config::default_log_path();
        if let Some(dir) = log_path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create log dir {}: {e}", dir.display()))?;
        }
        let plist = plist_path()?;
        if let Some(dir) = plist.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create LaunchAgents dir {}: {e}", dir.display()))?;
        }

        let body = render_plist(&exe.to_string_lossy(), &gguf, &log_path.to_string_lossy());
        std::fs::write(&plist, body)
            .map_err(|e| format!("cannot write plist {}: {e}", plist.display()))?;

        // bootstrap loads the agent into the user's GUI domain; idempotent-ish
        // (re-bootstrap of an already-loaded label errors, so bootout first,
        // best-effort).
        let domain = gui_domain()?;
        let target = service_target()?;
        // Re-bootstrap of a loaded label errors, so unload first. Absent is
        // the expected case here, hence no report.
        if is_loaded(&target)? {
            bootout_and_wait(&target)?;
        }
        let answered_before = serving_now();
        run_launchctl(&["bootstrap", &domain, &plist.to_string_lossy()])?;

        println!("launchd agent installed: {LAUNCHD_LABEL}");
        println!("plist:    {}", plist.display());
        println!("config:   {}", config::default_config_path().display());
        println!("log file: {}", log_path.display());
        // RunAtLoad starts it now; report whether it actually came up.
        let state = wait_for("running", 10)?;
        if state == "running" {
            let hint = format!("see {}", log_path.display());
            // Registered either way; a taken port is reported, not fatal here.
            if let Err(e) = report_serving(LAUNCHD_LABEL, answered_before, 30, &hint, &main_pid) {
                eprintln!("WARN: {e}");
            }
        } else {
            println!("state:    {state}");
            eprintln!(
                "WARN: the agent did not reach running within 10s; see {}",
                log_path.display()
            );
        }
        Ok(())
    }

    pub fn uninstall() -> Result<(), String> {
        let plist = plist_path()?;
        let target = service_target()?;
        let loaded = is_loaded(&target)?;
        if !loaded && !plist.exists() {
            println!("launchd agent not installed: {LAUNCHD_LABEL}");
            return Ok(());
        }
        if loaded {
            bootout_and_wait(&target)?;
        }
        if plist.exists() {
            std::fs::remove_file(&plist)
                .map_err(|e| format!("cannot remove plist {}: {e}", plist.display()))?;
        }
        let state = observed_state()?;
        if state != "not installed" {
            return Err(format!("{LAUNCHD_LABEL} is still {state} after uninstall"));
        }
        println!("launchd agent removed: {LAUNCHD_LABEL}");
        println!(
            "config file left in place: {} (delete manually if desired)",
            config::default_config_path().display()
        );
        Ok(())
    }

    pub fn start() -> Result<(), String> {
        let plist = plist_path()?;
        if !plist.exists() {
            return Err(format!(
                "launchd agent not installed ({} is missing) — run `m3-embed-server install` first",
                plist.display()
            ));
        }
        let target = service_target()?;
        let answered_before = observed_state()? != "running" && serving_now();
        // `stop` unloads the agent, and kickstart cannot reach an unloaded
        // label; load it first (RunAtLoad then starts it).
        if !is_loaded(&target)? {
            run_launchctl(&["bootstrap", &gui_domain()?, &plist.to_string_lossy()])?;
        }
        run_launchctl(&["kickstart", &target])?;
        let state = wait_for("running", 10)?;
        if state == "running" {
            let hint = format!("see {}", config::default_log_path().display());
            report_serving(LAUNCHD_LABEL, answered_before, 30, &hint, &main_pid)
        } else {
            Err(format!(
                "{LAUNCHD_LABEL} did not reach running within 10s (state: {state}; {}); see {}",
                last_exit(),
                config::default_log_path().display()
            ))
        }
    }

    /// The job's last exit, from `launchctl print` (`last exit code = N`),
    /// with what the server's own codes mean.
    fn last_exit_code() -> Option<String> {
        let out = launchctl_raw(&["print", &service_target().ok()?]).ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
            l.trim()
                .strip_prefix("last exit code = ")
                .map(|v| v.trim().to_string())
        })
    }

    fn last_exit() -> String {
        match last_exit_code() {
            Some(c) => match c.parse::<i64>() {
                Ok(n) => format!("last exit code {n}: {}", exit_status_meaning(n)),
                Err(_) => format!("last exit code {c}"),
            },
            None => "no exit code recorded".into(),
        }
    }

    pub fn stop() -> Result<(), String> {
        let target = service_target()?;
        if !is_loaded(&target)? {
            println!("{LAUNCHD_LABEL} is already stopped (nothing to do)");
            return Ok(());
        }
        // Unload, not `kill`: the plist sets KeepAlive, so launchd restarts a
        // killed job at once. The plist stays, so `status` says stopped and
        // RunAtLoad brings it back at the next login.
        bootout_and_wait(&target)?;
        println!("{LAUNCHD_LABEL}: {}", observed_state()?);
        Ok(())
    }

    /// `bootout` returns before launchd has torn the job down (measured: the
    /// label still prints as loaded right after), so wait for it to go. A
    /// bootout that fails because the job is already unloading is success.
    fn bootout_and_wait(target: &str) -> Result<(), String> {
        let result = run_launchctl(&["bootout", target]);
        for _ in 0..40 {
            if !is_loaded(target)? {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        result?;
        Err(format!("{LAUNCHD_LABEL} is still loaded 10s after bootout"))
    }

    pub fn status() -> Result<(), String> {
        let state = observed_state()?;
        println!("{state}");
        if state == "running" {
            note_foreign_answer(LAUNCHD_LABEL, main_pid());
        } else if last_exit_code().is_some_and(|c| c != "0") {
            // A loaded job that failed. stderr: stdout stays the state word.
            eprintln!("{LAUNCHD_LABEL}: {}", last_exit());
        }
        Ok(())
    }

    /// The agent's process id, from `launchctl print` (`pid = N`); `None` when
    /// it has no running process or launchd cannot be asked.
    fn main_pid() -> Option<u32> {
        let out = launchctl_raw(&["print", &service_target().ok()?]).ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find_map(|l| l.trim().strip_prefix("pid = ")?.trim().parse().ok())
    }

    /// Whether launchd has the label loaded. `print` exits non-zero for an
    /// unknown label; failing to spawn launchctl at all is an error, not "no".
    fn is_loaded(target: &str) -> Result<bool, String> {
        Ok(launchctl_raw(&["print", target])?.status.success())
    }

    /// `running` / `stopped` / `not installed` — the words m3 parses.
    fn observed_state() -> Result<&'static str, String> {
        let out = launchctl_raw(&["print", &service_target()?])?;
        if out.status.success() {
            // `print` dumps a big dict; the `state = running` line is the
            // signal. Exact match: `state = not running` also contains "running".
            let text = String::from_utf8_lossy(&out.stdout);
            return Ok(if text.lines().any(|l| l.trim() == "state = running") {
                "running"
            } else {
                "stopped"
            });
        }
        // Not loaded is not the same as not installed: `launchctl unload` (what
        // `m3 stop` does) and our own `stop` leave the plist in place, and the
        // plist on disk is the registration.
        Ok(if plist_path()?.exists() {
            "stopped"
        } else {
            "not installed"
        })
    }

    /// Poll until the agent is `want` or `secs` pass; returns the last state.
    fn wait_for(want: &str, secs: u64) -> Result<&'static str, String> {
        let mut state = observed_state()?;
        for _ in 0..(secs * 4) {
            if state == want {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
            state = observed_state()?;
        }
        Ok(state)
    }

    fn launchctl_raw(args: &[&str]) -> Result<std::process::Output, String> {
        std::process::Command::new("launchctl")
            .args(args)
            .output()
            .map_err(|e| format!("failed to spawn launchctl: {e}"))
    }

    fn run_launchctl(args: &[&str]) -> Result<(), String> {
        let out = launchctl_output(args)?;
        if !out.is_empty() {
            print!("{out}");
        }
        Ok(())
    }

    fn launchctl_output(args: &[&str]) -> Result<String, String> {
        let output = launchctl_raw(args)?;
        if !output.status.success() {
            return Err(format!(
                "launchctl {} exited {}: {}",
                args.join(" "),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim(),
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

// ===========================================================================
// Linux — systemd --user unit
// ===========================================================================
#[cfg(target_os = "linux")]
pub mod linux {
    use super::*;
    use crate::unit_render::{render_unit, SERVICE_NAME};

    fn unit_name() -> String {
        format!("{SERVICE_NAME}.service")
    }

    fn unit_path() -> Result<PathBuf, String> {
        // Honor XDG_CONFIG_HOME; fall back to ~/.config.
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .map(Ok)
            .unwrap_or_else(|| home_dir().map(|h| h.join(".config")))?;
        Ok(base.join("systemd").join("user").join(unit_name()))
    }

    pub fn install() -> Result<(), String> {
        let exe = current_exe()?;
        let gguf = resolve_gguf_for_unit()?;
        record_settings()?;
        let unit = unit_path()?;
        if let Some(dir) = unit.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create systemd user dir {}: {e}", dir.display()))?;
        }
        let body = render_unit(&exe.to_string_lossy(), &gguf);
        std::fs::write(&unit, body)
            .map_err(|e| format!("cannot write unit {}: {e}", unit.display()))?;

        let was_running = observed_state() == "running";
        let answered_before = !was_running && serving_now();
        run_systemctl(&["daemon-reload"])?;
        // `enable --now` registers it for auto-start AND starts it immediately,
        // but leaves a unit that is already running on its previous binary
        // and environment. Re-registering is how a new core reaches it, so
        // restart a running unit (macOS gets this from bootout + bootstrap).
        run_systemctl(&["enable", "--now", &unit_name()])?;
        if was_running {
            run_systemctl(&["restart", &unit_name()])?;
        }

        println!("systemd --user unit installed: {}", unit_name());
        println!("unit:     {}", unit.display());
        println!("config:   {}", config::default_config_path().display());
        println!("log:      journalctl --user -u {SERVICE_NAME}");
        // `enable --now` returns once the start is queued; report the outcome.
        let state = wait_for("running", 10);
        if state == "running" {
            // Registered either way; a taken port is reported, not fatal here.
            if let Err(e) = report_serving(
                SERVICE_NAME,
                answered_before,
                30,
                &journal_hint(),
                &main_pid,
            ) {
                eprintln!("WARN: {e}");
            }
        } else {
            println!("state:    {state}");
            eprintln!(
                "WARN: the unit did not reach running within 10s; {}",
                journal_hint()
            );
        }
        if linger_enabled() {
            return Ok(());
        }
        println!();
        println!("Note: a `systemd --user` service stops when you log out. To keep");
        println!("the embedder running across logout / on a headless box, enable lingering:");
        println!("  loginctl enable-linger \"$USER\"");
        Ok(())
    }

    /// Whether systemd keeps this user's services running after logout. An
    /// unknown answer reads as no, so the note is shown rather than hidden.
    fn linger_enabled() -> bool {
        let user = std::env::var("USER").unwrap_or_default();
        if user.is_empty() {
            return false;
        }
        std::process::Command::new("loginctl")
            .args(["show-user", &user, "-p", "Linger", "--value"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "yes")
            .unwrap_or(false)
    }

    pub fn uninstall() -> Result<(), String> {
        let unit = unit_path()?;
        if !unit.exists() && observed_state() == "not installed" {
            println!("systemd --user unit not installed: {}", unit_name());
            return Ok(());
        }
        // `disable --now` stops it and removes the auto-start symlink. A
        // failure is reported: the result is re-checked below either way.
        if let Err(e) = run_systemctl(&["disable", "--now", &unit_name()]) {
            eprintln!("WARN: {e}");
        }
        if unit.exists() {
            std::fs::remove_file(&unit)
                .map_err(|e| format!("cannot remove unit {}: {e}", unit.display()))?;
        }
        run_systemctl(&["daemon-reload"])?;
        let state = observed_state();
        if state != "not installed" {
            return Err(format!("{} is still {state} after uninstall", unit_name()));
        }
        println!("systemd --user unit removed: {}", unit_name());
        println!(
            "config file left in place: {} (delete manually if desired)",
            config::default_config_path().display()
        );
        Ok(())
    }

    pub fn start() -> Result<(), String> {
        let answered_before = observed_state() != "running" && serving_now();
        run_systemctl(&["start", &unit_name()])?;
        // `start` returns once the main process is forked (Type=simple), so a
        // bad GGUF or a busy port still exits 0. Report the outcome.
        let state = wait_for("running", 10);
        if state == "running" {
            report_serving(
                SERVICE_NAME,
                answered_before,
                30,
                &journal_hint(),
                &main_pid,
            )
        } else {
            Err(format!(
                "{SERVICE_NAME} did not reach running within 10s (state: {state}; {}); {}",
                last_exit(),
                journal_hint()
            ))
        }
    }

    pub fn stop() -> Result<(), String> {
        // `systemctl stop` blocks until the unit has stopped.
        run_systemctl(&["stop", &unit_name()])?;
        let state = observed_state();
        if state == "running" {
            return Err(format!("{SERVICE_NAME} is still running after stop"));
        }
        println!("{SERVICE_NAME}: {state}");
        Ok(())
    }

    pub fn status() -> Result<(), String> {
        let state = observed_state_noted(true);
        println!("{state}");
        if state == "running" {
            note_foreign_answer(SERVICE_NAME, main_pid());
        }
        Ok(())
    }

    /// The unit's main process id; `None` when it has none (systemd reports 0)
    /// or systemd cannot be asked.
    fn main_pid() -> Option<u32> {
        let out = systemctl_output(&["show", "-p", "MainPID", "--value", &unit_name()]).ok()?;
        out.trim().parse::<u32>().ok().filter(|&p| p != 0)
    }

    fn observed_state() -> String {
        observed_state_noted(false)
    }

    fn journal_hint() -> String {
        format!("see `journalctl --user -u {SERVICE_NAME} -n 50`")
    }

    /// The main process's last exit, as systemd recorded it, with what the
    /// server's own codes mean.
    fn last_exit() -> String {
        let out = systemctl_output(&["show", "-p", "Result", "-p", "ExecMainStatus", &unit_name()]);
        let Ok(out) = out else {
            return "exit status unavailable".into();
        };
        let prop = |k: &str| {
            out.lines()
                .find_map(|l| l.strip_prefix(k)?.strip_prefix('='))
                .unwrap_or("?")
                .trim()
                .to_string()
        };
        let (result, status) = (prop("Result"), prop("ExecMainStatus"));
        match status.parse::<i64>() {
            Ok(code) => format!(
                "result {result}, exit status {code}: {}",
                exit_status_meaning(code)
            ),
            Err(_) => format!("result {result}, exit status {status}"),
        }
    }

    /// The state word m3 parses: `running`, `stopped`, `not installed`, or
    /// systemd's own transitional word (`activating`, `deactivating`, …).
    /// Detail an operator needs goes to stderr so stdout stays one word.
    fn observed_state_noted(report: bool) -> String {
        // `is-active` prints active/inactive/failed/… and sets the exit code;
        // `is-enabled` distinguishes "not installed" from "installed, stopped".
        let active = match systemctl_output(&["is-active", &unit_name()]) {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                // No answer (no user bus, systemctl missing) is neither
                // "stopped" nor "not installed": a unit file says it is
                // registered, not whether it runs.
                if report {
                    let unit = match unit_path() {
                        Ok(p) if p.exists() => format!("the unit file {} exists", p.display()),
                        Ok(p) => format!("there is no unit file at {}", p.display()),
                        Err(e) => e,
                    };
                    eprintln!("cannot query systemd, so the state is unknown ({unit}): {e}");
                }
                return "unknown".to_string();
            }
        };
        match active.as_str() {
            "active" => "running".to_string(),
            "inactive" | "failed" | "unknown" => {
                let enabled = systemctl_output(&["is-enabled", &unit_name()])
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                let installed = !(enabled.is_empty() || enabled == "not-found")
                    || unit_path().map(|p| p.exists()).unwrap_or(false);
                if !installed {
                    return "not installed".to_string();
                }
                if report && active == "failed" {
                    eprintln!("the unit failed ({}); {}", last_exit(), journal_hint());
                }
                "stopped".to_string()
            }
            other => other.to_string(),
        }
    }

    /// Poll until the unit is `want` or `secs` pass; returns the last state.
    fn wait_for(want: &str, secs: u64) -> String {
        let mut state = observed_state();
        for _ in 0..(secs * 4) {
            if state == want {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
            state = observed_state();
        }
        state
    }

    fn run_systemctl(args: &[&str]) -> Result<(), String> {
        let out = systemctl_output(args)?;
        if !out.trim().is_empty() {
            print!("{out}");
        }
        Ok(())
    }

    /// Run `systemctl --user <args>`. Returns stdout on success; on failure
    /// returns the stderr-bearing error string.
    fn systemctl_output(args: &[&str]) -> Result<String, String> {
        let mut full = vec!["--user"];
        full.extend_from_slice(args);
        let output = std::process::Command::new("systemctl")
            .args(&full)
            .output()
            .map_err(|e| format!("failed to spawn systemctl: {e}"))?;
        // is-active / is-enabled exit non-zero for inactive/disabled but still
        // print a meaningful word on stdout — callers that tolerate that read
        // stdout directly. run_systemctl uses this for state-changing verbs
        // where a non-zero exit is a real error.
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // For query verbs the word on stdout IS the answer — surface it.
            if !stdout.is_empty()
                && (args.first() == Some(&"is-active") || args.first() == Some(&"is-enabled"))
            {
                return Ok(stdout);
            }
            return Err(format!(
                "systemctl --user {} exited {}: {}",
                args.join(" "),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim(),
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

// ===========================================================================
// Platform dispatch — what `main.rs` calls. macOS / Linux route to the modules
// above; any other Unix returns a clear unsupported-platform error.
// ===========================================================================

// `allow(unused_macros)`: only the `#[cfg(not(any(macos, linux)))]` dispatch
// arms expand this, so a macOS or Linux build never references it.
#[allow(unused_macros)]
macro_rules! unsupported {
    ($verb:expr) => {
        Err(format!(
            "`{}` has no service integration on this platform — only Windows \
             (Service), macOS (launchd) and Linux (systemd) are supported. \
             Run `m3-embed-server` with no arguments for foreground mode.",
            $verb
        ))
    };
}

pub fn install() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    return macos::install();
    #[cfg(target_os = "linux")]
    return linux::install();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    unsupported!("install")
}

pub fn uninstall() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    return macos::uninstall();
    #[cfg(target_os = "linux")]
    return linux::uninstall();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    unsupported!("uninstall")
}

pub fn start() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    return macos::start();
    #[cfg(target_os = "linux")]
    return linux::start();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    unsupported!("start")
}

pub fn stop() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    return macos::stop();
    #[cfg(target_os = "linux")]
    return linux::stop();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    unsupported!("stop")
}

pub fn status() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    return macos::status();
    #[cfg(target_os = "linux")]
    return linux::status();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    unsupported!("status")
}
