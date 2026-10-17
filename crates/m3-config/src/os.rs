//! OS facts the core needs when it starts another program.
//!
//! One owner, so every spawn behaves the same. A console program started from a
//! process that has no console of its own (pythonw, a service, an agent's hook)
//! gets a NEW console, and on Windows that is a window that flashes on the
//! user's desktop. Measured 2026-10-17: `import m3_core_rs` on a CUDA wheel ran
//! `nvidia-smi` for the compute-capability probe and flashed a window on every
//! hook run.

use std::process::Command;

/// Start `cmd` without a console window on Windows (`CREATE_NO_WINDOW`); a
/// no-op elsewhere. Output capture and the exit code are unaffected.
pub fn no_console_window(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // winbase.h CREATE_NO_WINDOW: the child gets a console it can use, with
        // no window. Not DETACHED_PROCESS, which leaves it no console at all.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag must not change what the child does: it still runs, its output
    /// is captured and its exit code comes back.
    #[test]
    fn a_windowless_child_still_runs_and_reports() {
        #[cfg(windows)]
        let mut cmd = {
            let mut c = Command::new("cmd");
            c.args(["/c", "echo m3& exit 3"]);
            c
        };
        #[cfg(not(windows))]
        let mut cmd = {
            let mut c = Command::new("sh");
            c.args(["-c", "echo m3; exit 3"]);
            c
        };
        let out = no_console_window(&mut cmd).output().expect("spawn");
        assert_eq!(out.status.code(), Some(3));
        assert!(String::from_utf8_lossy(&out.stdout).contains("m3"));
    }
}
