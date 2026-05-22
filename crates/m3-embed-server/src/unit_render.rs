//! Pure rendering of OS service-unit files (launchd plist, systemd unit).
//!
//! Deliberately **platform-neutral** — no `#[cfg]`, no OS calls — so the
//! templating is unit-testable on any host (including the Windows CI box that
//! cannot otherwise compile `service_unix`). `service_unix.rs` calls these;
//! the `launchctl` / `systemctl` invocation that consumes the rendered files
//! stays in the `cfg`-gated module.

/// launchd label / systemd unit base name.
pub const SERVICE_NAME: &str = "m3-embed-server";

/// Reverse-DNS label launchd requires; also the plist file stem.
pub const LAUNCHD_LABEL: &str = "com.skynetcmd.m3-embed-server";

/// Minimal XML escaping for launchd plist text values. Paths come from the OS
/// / the operator's env, not untrusted input, but escaping `&` `<` `>`
/// defensively keeps an odd path from breaking the XML.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Render a launchd property list for a per-user agent.
///
/// `exe` — absolute path to the m3-embed-server binary (the agent's
/// `ProgramArguments`). `gguf` — model path pinned into the agent environment.
/// `log_path` — file the agent's stdout+stderr are redirected to.
pub fn render_plist(exe: &str, gguf: &str, log_path: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>M3_EMBED_GGUF</key>
        <string>{gguf}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"#,
        label = LAUNCHD_LABEL,
        exe = xml_escape(exe),
        gguf = xml_escape(gguf),
        log = xml_escape(log_path),
    )
}

/// Render a `systemd --user` unit file.
///
/// `exe` — absolute path to the binary (the unit's `ExecStart`). `gguf` —
/// model path pinned via `Environment=`.
pub fn render_unit(exe: &str, gguf: &str) -> String {
    // systemd `Environment=` values are wrapped in double quotes so a path
    // with spaces survives; escape embedded backslash / double-quote.
    let env_val = gguf.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "[Unit]\n\
         Description=M3 Embed Server (CPU fallback) — OpenAI-compatible embed server on :8082\n\
         Documentation=https://github.com/skynetcmd/m3-core-rs\n\
         After=network.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe}\n\
         Environment=\"M3_EMBED_GGUF={env_val}\"\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        exe = exe,
        env_val = env_val,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_has_label_exe_gguf_and_log() {
        let p = render_plist("/usr/local/bin/m3-embed-server", "/models/bge-m3.gguf", "/tmp/svc.log");
        assert!(p.contains(&format!("<string>{LAUNCHD_LABEL}</string>")));
        assert!(p.contains("<string>/usr/local/bin/m3-embed-server</string>"));
        // GGUF is pinned in EnvironmentVariables, keyed M3_EMBED_GGUF.
        assert!(p.contains("<key>M3_EMBED_GGUF</key>"));
        assert!(p.contains("<string>/models/bge-m3.gguf</string>"));
        // Both stdout and stderr point at the log path.
        assert_eq!(p.matches("<string>/tmp/svc.log</string>").count(), 2);
        // Auto-start + restart semantics.
        assert!(p.contains("<key>RunAtLoad</key>"));
        assert!(p.contains("<key>KeepAlive</key>"));
        // Well-formed-ish: declared plist, closed plist.
        assert!(p.starts_with("<?xml"));
        assert!(p.trim_end().ends_with("</plist>"));
    }

    #[test]
    fn plist_escapes_xml_metacharacters_in_paths() {
        let p = render_plist("/opt/a&b/m3-embed-server", "/m/x<y>.gguf", "/l/z&.log");
        assert!(p.contains("/opt/a&amp;b/m3-embed-server"));
        assert!(p.contains("/m/x&lt;y&gt;.gguf"));
        assert!(p.contains("/l/z&amp;.log"));
        // No raw unescaped metacharacter from the inputs leaked through.
        assert!(!p.contains("a&b"));
        assert!(!p.contains("x<y>"));
    }

    #[test]
    fn unit_has_execstart_env_and_install_section() {
        let u = render_unit("/usr/local/bin/m3-embed-server", "/models/bge-m3.gguf");
        assert!(u.contains("ExecStart=/usr/local/bin/m3-embed-server"));
        assert!(u.contains("Environment=\"M3_EMBED_GGUF=/models/bge-m3.gguf\""));
        assert!(u.contains("Restart=on-failure"));
        assert!(u.contains("[Install]"));
        assert!(u.contains("WantedBy=default.target"));
    }

    #[test]
    fn unit_quotes_and_escapes_gguf_with_spaces_and_backslashes() {
        let u = render_unit("/bin/m3-embed-server", r#"/m/with space/and"quote/bge.gguf"#);
        // Backslash-escaped quote inside the double-quoted Environment= value.
        assert!(u.contains(r#"Environment="M3_EMBED_GGUF=/m/with space/and\"quote/bge.gguf""#));
    }
}
