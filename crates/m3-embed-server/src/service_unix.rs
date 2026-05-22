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
    let file_cfg = config::load_file_config(&config::default_config_path()).unwrap_or_default();
    file_cfg.embed.gguf.clone().ok_or_else(|| {
        "M3_EMBED_GGUF is unset and no config.toml has [embed].gguf — \
         set the env var before `install` so the service can find the model"
            .to_string()
    })
}

// ===========================================================================
// macOS — launchd user agent
// ===========================================================================
//
// VERIFICATION STATUS: this module was written and reviewed but, as of
// 2026-05-22, NEVER COMPILED — the feature was developed on Windows and the
// Linux/systemd sibling verified on a Debian box, but no macOS host was
// available and cross-compiling to *-apple-darwin fails at the llama.cpp/ring
// C build. Before relying on the launchd path, run `cargo build/test/clippy
// -p m3-embed-server --features embedded` on a real Mac plus a launchctl
// install/status/stop/uninstall smoke test. Full checklist: m3-memory to-do
// `c5508907` ("Verify the macOS launchd path in m3-embed-server").
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
        let _ = run_launchctl(&["bootout", &target]);
        run_launchctl(&["bootstrap", &domain, &plist.to_string_lossy()])?;

        println!("launchd agent installed: {LAUNCHD_LABEL}");
        println!("plist:    {}", plist.display());
        println!("log file: {}", log_path.display());
        println!("It will start now and on every login (RunAtLoad + KeepAlive).");
        Ok(())
    }

    pub fn uninstall() -> Result<(), String> {
        let plist = plist_path()?;
        // bootout is best-effort — the agent may already be unloaded.
        if let Ok(target) = service_target() {
            let _ = run_launchctl(&["bootout", &target]);
        }
        if plist.exists() {
            std::fs::remove_file(&plist)
                .map_err(|e| format!("cannot remove plist {}: {e}", plist.display()))?;
        }
        println!("launchd agent removed: {LAUNCHD_LABEL}");
        println!(
            "config file left in place: {} (delete manually if desired)",
            config::default_config_path().display()
        );
        Ok(())
    }

    pub fn start() -> Result<(), String> {
        run_launchctl(&["kickstart", &service_target()?])?;
        println!("start signal sent to {LAUNCHD_LABEL}");
        Ok(())
    }

    pub fn stop() -> Result<(), String> {
        // `kill` sends a signal to the running job without unloading it, so a
        // later `start`/KeepAlive can bring it back.
        run_launchctl(&["kill", "SIGTERM", &service_target()?])?;
        println!("stop signal sent to {LAUNCHD_LABEL}");
        Ok(())
    }

    pub fn status() -> Result<(), String> {
        // `launchctl print` exits non-zero when the label isn't loaded.
        match launchctl_output(&["print", &service_target()?]) {
            Ok(out) => {
                // `print` dumps a big dict; the `state = running` line is the
                // signal. Absence of that line ⇒ loaded but not running.
                if out.lines().any(|l| l.trim_start().starts_with("state =") && l.contains("running")) {
                    println!("running");
                } else {
                    println!("stopped");
                }
                Ok(())
            }
            Err(_) => {
                println!("not installed");
                Ok(())
            }
        }
    }

    fn run_launchctl(args: &[&str]) -> Result<(), String> {
        let out = launchctl_output(args)?;
        if !out.is_empty() {
            print!("{out}");
        }
        Ok(())
    }

    fn launchctl_output(args: &[&str]) -> Result<String, String> {
        let output = std::process::Command::new("launchctl")
            .args(args)
            .output()
            .map_err(|e| format!("failed to spawn launchctl: {e}"))?;
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

        run_systemctl(&["daemon-reload"])?;
        // `enable --now` registers it for auto-start AND starts it immediately.
        run_systemctl(&["enable", "--now", &unit_name()])?;

        println!("systemd --user unit installed: {}", unit_name());
        println!("unit:     {}", unit.display());
        println!("log:      journalctl --user -u {SERVICE_NAME}");
        println!();
        println!("Note: a `systemd --user` service stops when you log out. To keep");
        println!("the embedder running across logout / on a headless box, enable lingering:");
        println!("  loginctl enable-linger \"$USER\"");
        Ok(())
    }

    pub fn uninstall() -> Result<(), String> {
        let unit = unit_path()?;
        // `disable --now` stops it and removes the auto-start symlink.
        // Best-effort — the unit may already be gone.
        let _ = run_systemctl(&["disable", "--now", &unit_name()]);
        if unit.exists() {
            std::fs::remove_file(&unit)
                .map_err(|e| format!("cannot remove unit {}: {e}", unit.display()))?;
        }
        let _ = run_systemctl(&["daemon-reload"]);
        println!("systemd --user unit removed: {}", unit_name());
        println!(
            "config file left in place: {} (delete manually if desired)",
            config::default_config_path().display()
        );
        Ok(())
    }

    pub fn start() -> Result<(), String> {
        run_systemctl(&["start", &unit_name()])?;
        println!("start signal sent to {SERVICE_NAME}");
        Ok(())
    }

    pub fn stop() -> Result<(), String> {
        run_systemctl(&["stop", &unit_name()])?;
        println!("stop signal sent to {SERVICE_NAME}");
        Ok(())
    }

    pub fn status() -> Result<(), String> {
        // `is-active` prints active/inactive/failed/unknown and sets exit code;
        // `is-enabled` distinguishes "not installed" from "installed, stopped".
        let active = systemctl_output(&["is-active", &unit_name()])
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        match active.as_str() {
            "active" => println!("running"),
            "" | "inactive" | "failed" | "unknown" => {
                // Distinguish stopped-but-installed from never-installed.
                let enabled = systemctl_output(&["is-enabled", &unit_name()])
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                if enabled.is_empty() || enabled == "not-found" {
                    println!("not installed");
                } else {
                    println!("stopped");
                }
            }
            other => println!("{other}"),
        }
        Ok(())
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
