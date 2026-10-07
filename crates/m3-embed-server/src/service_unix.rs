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
        .ok_or_else(|| "$HOME is not set — cannot locate the per-user service directory".to_string())
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
    if let Some(g) = std::env::var("M3_EMBED_GGUF").ok().filter(|s| !s.is_empty()) {
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

/// True when the service's `/health` answers ok. The server binds only after
/// the model is loaded, so this is "serving", which a supervisor's "running"
/// (the process exists) is not.
fn health_ok(host: &str, port: u16) -> bool {
    use std::io::{Read, Write};
    use std::net::ToSocketAddrs;
    let Some(addr) = (host, port).to_socket_addrs().ok().and_then(|mut a| a.next()) else {
        return false;
    };
    let timeout = std::time::Duration::from_secs(2);
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, timeout) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let req = format!("GET /health HTTP/1.0\r\nHost: {host}\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut body = String::new();
    let _ = stream.read_to_string(&mut body);
    body.contains("\"status\":\"ok\"")
}

/// After the supervisor reports the process running, wait for it to serve and
/// say which it is. `answered_before` is whether the address already answered
/// before this start: then a later ok may be another process, not this one.
fn report_serving(label: &str, answered_before: bool, secs: u64, log_hint: &str) {
    let (host, port) = config::service_addr();
    if answered_before {
        eprintln!(
            "WARN: {host}:{port} was already answering before this start; another \
             process may hold the port, so this service may not be the one serving"
        );
    }
    for _ in 0..(secs * 2) {
        if health_ok(&host, port) {
            if answered_before {
                println!("{label}: running; {host}:{port} answers, but it did before this start too");
            } else {
                println!("{label}: running, serving on {host}:{port}");
            }
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    println!(
        "{label}: running, not yet serving on {host}:{port} after {secs}s (still \
         loading the model?); {log_hint}"
    );
}

/// Whether the service's address answers right now.
fn serving_now() -> bool {
    let (host, port) = config::service_addr();
    health_ok(&host, port)
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

        let body = render_plist(
            &exe.to_string_lossy(),
            &gguf,
            &log_path.to_string_lossy(),
        );
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
            report_serving(LAUNCHD_LABEL, answered_before, 30, &hint);
        } else {
            println!("state:    {state}");
            eprintln!("WARN: the agent did not reach running within 10s; see {}", log_path.display());
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
            report_serving(LAUNCHD_LABEL, answered_before, 30, &hint);
            Ok(())
        } else {
            Err(format!(
                "{LAUNCHD_LABEL} did not reach running within 10s (state: {state}); see {}",
                config::default_log_path().display()
            ))
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
        println!("{}", observed_state()?);
        Ok(())
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
        Ok(if plist_path()?.exists() { "stopped" } else { "not installed" })
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
        let unit = unit_path()?;
        if let Some(dir) = unit.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create systemd user dir {}: {e}", dir.display()))?;
        }
        let body = render_unit(&exe.to_string_lossy(), &gguf);
        std::fs::write(&unit, body)
            .map_err(|e| format!("cannot write unit {}: {e}", unit.display()))?;

        let answered_before = observed_state() != "running" && serving_now();
        run_systemctl(&["daemon-reload"])?;
        // `enable --now` registers it for auto-start AND starts it immediately.
        run_systemctl(&["enable", "--now", &unit_name()])?;

        println!("systemd --user unit installed: {}", unit_name());
        println!("unit:     {}", unit.display());
        println!("config:   {}", config::default_config_path().display());
        println!("log:      journalctl --user -u {SERVICE_NAME}");
        // `enable --now` returns once the start is queued; report the outcome.
        let state = wait_for("running", 10);
        if state == "running" {
            report_serving(SERVICE_NAME, answered_before, 30, &journal_hint());
        } else {
            println!("state:    {state}");
            eprintln!("WARN: the unit did not reach running within 10s; {}", journal_hint());
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
            report_serving(SERVICE_NAME, answered_before, 30, &journal_hint());
            Ok(())
        } else {
            Err(format!(
                "{SERVICE_NAME} did not reach running within 10s (state: {state}); {}",
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
        println!("{}", observed_state_noted(true));
        Ok(())
    }

    fn observed_state() -> String {
        observed_state_noted(false)
    }

    fn journal_hint() -> String {
        format!("see `journalctl --user -u {SERVICE_NAME} -n 50`")
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
                // No answer (no user bus, systemctl missing) is not "not
                // installed"; fall back to the unit file and say why.
                if report {
                    eprintln!("cannot query systemd: {e}");
                }
                return if unit_path().map(|p| p.exists()).unwrap_or(false) {
                    "stopped".to_string()
                } else {
                    "not installed".to_string()
                };
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
                    eprintln!("the unit failed; {}", journal_hint());
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
