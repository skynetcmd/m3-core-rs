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
/// `exe` — absolute path to the m3-embed-server binary; `args` follow it in
/// `ProgramArguments` (`--config <m3_config.toml>`: the agent starts with a
/// bare environment, so the file is named rather than discovered).
/// `log_path` — file the agent's stdout+stderr are redirected to.
pub fn render_plist(exe: &str, args: &[String], log_path: &str) -> String {
    let mut program = format!("        <string>{}</string>", xml_escape(exe));
    for a in args {
        program.push_str(&format!("\n        <string>{}</string>", xml_escape(a)));
    }
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{program}
    </array>
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
        log = xml_escape(log_path),
    )
}

/// One `ExecStart=` word: double-quoted (a path with spaces survives), with
/// `\` and `"` escaped, and systemd's specifier `%` and variable `$` doubled
/// so a path containing them is taken literally.
fn systemd_word(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('%', "%%")
        .replace('$', "$$");
    format!("\"{escaped}\"")
}

/// Render a `systemd --user` unit file.
///
/// `exe` — absolute path to the binary; `args` follow it in `ExecStart`
/// (`--config <m3_config.toml>`: the unit starts with a bare environment).
pub fn render_unit(exe: &str, args: &[String]) -> String {
    let exec = std::iter::once(exe)
        .chain(args.iter().map(String::as_str))
        .map(systemd_word)
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "[Unit]\n\
         Description=M3 Embed Server (CPU fallback) — OpenAI-compatible embed server on :8082\n\
         Documentation=https://github.com/skynetcmd/m3-core-rs\n\
         After=network.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exec}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         RestartPreventExitStatus={port_in_use} {refused}\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        exec = exec,
        // A taken port is not cured by a restart: without this the unit
        // retried every 5 s for as long as another program held the port.
        port_in_use = crate::server::EXIT_PORT_IN_USE,
        // Nor is a refused configuration (TLS/token rules, src/security.rs).
        refused = crate::security::EXIT_CONFIG_REFUSED,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Vec<String> {
        vec![
            "--config".into(),
            "/home/u/.m3/config/m3_config.toml".into(),
        ]
    }

    #[test]
    fn plist_has_label_exe_args_and_log_and_no_environment() {
        let p = render_plist("/usr/local/bin/m3-embed-server", &args(), "/tmp/svc.log");
        assert!(p.contains(&format!("<string>{LAUNCHD_LABEL}</string>")));
        assert!(p.contains(
            "<string>/usr/local/bin/m3-embed-server</string>\n        <string>--config</string>\n        \
             <string>/home/u/.m3/config/m3_config.toml</string>"
        ));
        // Settings come from the named file, never the agent's environment.
        assert!(!p.contains("EnvironmentVariables") && !p.contains("M3_EMBED_GGUF"));
        assert_eq!(p.matches("<string>/tmp/svc.log</string>").count(), 2);
        assert!(p.contains("<key>RunAtLoad</key>"));
        assert!(p.contains("<key>KeepAlive</key>"));
        assert!(p.starts_with("<?xml"));
        assert!(p.trim_end().ends_with("</plist>"));
    }

    #[test]
    fn plist_escapes_xml_metacharacters_in_paths() {
        let p = render_plist(
            "/opt/a&b/m3-embed-server",
            &["/m/x<y>.toml".into()],
            "/l/z&.log",
        );
        assert!(p.contains("/opt/a&amp;b/m3-embed-server"));
        assert!(p.contains("/m/x&lt;y&gt;.toml"));
        assert!(p.contains("/l/z&amp;.log"));
        assert!(!p.contains("a&b") && !p.contains("x<y>"));
    }

    #[test]
    fn unit_runs_the_binary_with_the_config_file_and_no_environment() {
        let u = render_unit("/usr/local/bin/m3-embed-server", &args());
        assert!(u.contains(
            "ExecStart=\"/usr/local/bin/m3-embed-server\" \"--config\" \"/home/u/.m3/config/m3_config.toml\"\n"
        ));
        assert!(!u.contains("Environment="));
        assert!(u.contains("Restart=on-failure"));
        assert!(u.contains(&format!(
            "RestartPreventExitStatus={} {}",
            crate::server::EXIT_PORT_IN_USE,
            crate::security::EXIT_CONFIG_REFUSED
        )));
        assert!(u.contains("[Install]"));
        assert!(u.contains("WantedBy=default.target"));
    }

    #[test]
    fn unit_quotes_and_escapes_spaces_quotes_specifiers_and_variables() {
        let u = render_unit(
            "/bin/m3-embed-server",
            &[r#"/m/with space/a"q/100%/$HOME.toml"#.into()],
        );
        assert!(
            u.contains(r#""/m/with space/a\"q/100%%/$$HOME.toml""#),
            "{u}"
        );
    }
}
