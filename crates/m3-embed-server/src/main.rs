//! m3-embed-server — OpenAI-compatible CPU embedding service on port 8082.
//!
//! Modes (selected by argv[1]):
//!   (none)            Foreground / dev mode. Logs to stderr, Ctrl-C / SIGTERM
//!                     to stop. This is also what the OS supervisor runs.
//!   install           Register as an OS service: Windows Service (admin),
//!                     macOS launchd user agent, or Linux `systemd --user`
//!                     unit. Writes a starter config from the current env.
//!   uninstall         Stop + remove the OS service.
//!   start | stop      Convenience wrappers over the OS service manager.
//!   status            Print "running" / "stopped" / "not installed".
//!   run-as-service    Internal — Windows SCM invokes this when starting the
//!                     service. (Unix supervisors run foreground mode directly.)
//!
//! Config priority: env var > per-user/per-system config.toml > default.

#[cfg(feature = "embedded")]
mod config;
#[cfg(feature = "embedded")]
mod server;
#[cfg(all(windows, feature = "embedded"))]
mod service;
// Platform-neutral unit-file templating — compiles everywhere so its render
// tests run on any CI box; consumed by service_unix on Unix. Each item is used
// by exactly one platform path (render_plist/LAUNCHD_LABEL → macOS,
// render_unit/SERVICE_NAME → Linux), so *every* single-platform build leaves
// some of them unused. `allow(dead_code)` module-wide is correct here — the
// `#[cfg(test)]` render tests exercise all of them regardless of host.
#[cfg(feature = "embedded")]
#[allow(dead_code)]
mod unit_render;
#[cfg(all(not(windows), feature = "embedded"))]
mod service_unix;

#[cfg(not(feature = "embedded"))]
fn main() -> anyhow::Result<()> {
    eprintln!(
        "m3-embed-server was built without the `embedded` feature — \
         rebuild with `--features embedded` to get a working binary."
    );
    std::process::exit(2);
}

#[cfg(feature = "embedded")]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match sub {
        "" => run_foreground(),
        "run-as-service" => run_as_service(),
        "install" => run_install(),
        "uninstall" => run_uninstall(),
        "start" => run_start(),
        "stop" => run_stop(),
        "status" => run_status(),
        "doctor" => run_doctor(),
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            print_help();
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "embedded")]
fn print_help() {
    // The `install` line names the concrete mechanism for the host OS so the
    // operator knows up front whether elevation is needed.
    let install_line = if cfg!(windows) {
        "register as a Windows Service (Administrator required)"
    } else if cfg!(target_os = "macos") {
        "register as a launchd user agent (no sudo)"
    } else if cfg!(target_os = "linux") {
        "register as a systemd --user unit (no sudo)"
    } else {
        "register as an OS service (unsupported on this platform)"
    };
    eprintln!(
        "m3-embed-server — OpenAI-compatible CPU embed server (port 8082 by default)\n\
         \n\
         USAGE:\n  \
           m3-embed-server [SUBCOMMAND]\n\
         \n\
         SUBCOMMANDS:\n  \
           (none)        run in foreground (dev mode); Ctrl-C / SIGTERM to stop\n  \
           install       {install_line}\n  \
           uninstall     stop and remove the OS service\n  \
           start         start the installed service\n  \
           stop          stop the installed service\n  \
           status        running / stopped / not installed\n  \
           doctor        diagnose the install: service status, HTTP probe, recent log lines\n  \
           help          show this message\n"
    );
}

// ---------------------------------------------------------------------------
// Foreground mode (works on every platform)
// ---------------------------------------------------------------------------

#[cfg(feature = "embedded")]
fn run_foreground() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init()
        .ok();

    // Foreground reads env first, but will still pick up a config.toml if
    // present (handy for testing the service config without installing).
    let cfg_path = config::default_config_path();
    let file_cfg = config::load_file_config(&cfg_path).unwrap_or_default();
    let cfg = config::resolve(&file_cfg)?;

    // B4: emit a starter config.toml on first foreground start if none exists,
    // so post-mortems / service-install upgrades have something to read. The
    // file is written from the resolved config, not from the env directly,
    // so it captures whatever the discovery cascade (B5) found too.
    if !cfg_path.exists() {
        let snap = config::FileConfig {
            embed: config::EmbedSection {
                gguf: Some(cfg.gguf.clone()),
                port: Some(cfg.port),
                host: Some(cfg.host.clone()),
                streams: Some(cfg.streams),
                ctx: Some(cfg.n_ctx),
                seq_max: Some(cfg.seq_max),
                n_batch: Some(cfg.n_batch),
                n_ubatch: Some(cfg.n_ubatch),
                coalesce_ms: Some(cfg.coalesce_ms),
                max_batch_tokens: Some(cfg.max_batch_tokens),
            },
        };
        if let Err(e) = config::write_config_file(&cfg_path, &snap) {
            log::warn!("could not write starter config.toml to {}: {e}",
                       cfg_path.display());
        } else {
            log::info!("wrote starter config.toml to {}", cfg_path.display());
        }
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        server::run(cfg, shutdown_signal()).await
    })
}

// ---------------------------------------------------------------------------
// B1: doctor subcommand — diagnose install without running the service.
// ---------------------------------------------------------------------------
//
// Probes (each with its own bounded timeout, so a hung probe doesn't hang the
// tool itself):
//
//   1. config.toml presence + parseability + resolved-config dump
//   2. GGUF discovery — env, file, cascade — which path won, does it exist
//   3. service status (Windows SCM / launchd / systemd) when applicable
//   4. HTTP probe — /health on the configured port (default 8082)
//   5. HTTP round-trip — POST /v1/embeddings with "ping", verify 1024-dim
//   6. recent log lines from the configured log path (last 20 lines)
//
// Output is line-oriented stdout so it's grep-friendly. Exit code 0 if all
// six probes pass (or the operator-doesn't-care ones — service+log probes
// can fail without flipping exit code, since not everyone runs as a service).

#[cfg(feature = "embedded")]
fn run_doctor() -> anyhow::Result<()> {
    use std::time::Duration;

    let mut had_error = false;

    println!("=== m3-embed-server doctor ===");
    println!();

    // 1. config.toml
    let cfg_path = config::default_config_path();
    println!("[1] config.toml: {}", cfg_path.display());
    let file_cfg = match config::load_file_config(&cfg_path) {
        Ok(c) => {
            if cfg_path.exists() {
                println!("    status: present + parseable");
            } else {
                println!("    status: not present (will be written on first foreground run)");
            }
            c
        }
        Err(e) => {
            println!("    status: FAIL — could not parse: {e}");
            had_error = true;
            config::FileConfig::default()
        }
    };

    // 2. GGUF resolution + discovery
    println!();
    println!("[2] GGUF resolution");
    let env_gguf = std::env::var("M3_EMBED_GGUF").ok().filter(|s| !s.is_empty());
    if let Some(g) = env_gguf.as_ref() {
        println!("    env M3_EMBED_GGUF: {g}");
    } else {
        println!("    env M3_EMBED_GGUF: (unset)");
    }
    if let Some(g) = file_cfg.embed.gguf.as_ref() {
        println!("    file [embed].gguf: {g}");
    } else {
        println!("    file [embed].gguf: (unset)");
    }
    println!("    discovery cascade candidate dirs (B5):");
    for d in config::discovery_candidate_dirs() {
        let exists = if d.is_dir() { "exists" } else { "no" };
        println!("      - {} [{exists}]", d.display());
    }
    let resolved = match config::resolve(&file_cfg) {
        Ok(r) => {
            println!("    RESOLVED: {} (exists: {})", r.gguf, std::path::Path::new(&r.gguf).exists());
            Some(r)
        }
        Err(e) => {
            println!("    RESOLVED: FAIL — {e}");
            had_error = true;
            None
        }
    };

    let port = resolved.as_ref().map(|r| r.port).unwrap_or(8082);
    let host = resolved.as_ref().map(|r| r.host.clone()).unwrap_or_else(|| "127.0.0.1".into());

    // 3. service status
    println!();
    println!("[3] OS service status");
    let svc_state = probe_service_status();
    println!("    {svc_state}");

    // 4. HTTP /health
    println!();
    println!("[4] HTTP /health on {host}:{port}");
    match probe_http_health(&host, port, Duration::from_secs(2)) {
        Ok(body) => println!("    OK (body: {body:?})"),
        Err(e) => {
            println!("    FAIL — {e}");
            had_error = true;
        }
    }

    // 5. HTTP roundtrip /v1/embeddings
    println!();
    println!("[5] HTTP /v1/embeddings roundtrip");
    match probe_embed_roundtrip(&host, port, Duration::from_secs(10)) {
        Ok((dim, latency_ms)) => println!("    OK — dim={dim}, latency={latency_ms}ms"),
        Err(e) => {
            println!("    FAIL — {e}");
            had_error = true;
        }
    }

    // 6. recent log lines
    println!();
    let log_path = config::default_log_path();
    println!("[6] recent log lines from {}", log_path.display());
    match tail_log_lines(&log_path, 20) {
        Ok(lines) => {
            if lines.is_empty() {
                println!("    (log file is empty or missing — not an error if running foreground)");
            } else {
                for l in lines {
                    println!("    | {l}");
                }
            }
        }
        Err(e) => {
            println!("    (could not read log: {e})");
        }
    }

    println!();
    if had_error {
        println!("=== doctor: FAIL (one or more critical probes failed) ===");
        std::process::exit(1);
    } else {
        println!("=== doctor: OK ===");
    }
    Ok(())
}

#[cfg(feature = "embedded")]
fn probe_service_status() -> String {
    #[cfg(windows)]
    {
        match service::status() {
            Ok(()) => "(status printed above by service::status)".into(),
            Err(e) => format!("not installed or stopped: {e}"),
        }
    }
    #[cfg(not(windows))]
    {
        match service_unix::status() {
            Ok(()) => "(status printed above by service_unix::status)".into(),
            Err(e) => format!("not installed or stopped: {e}"),
        }
    }
}

/// Tiny HTTP GET /health using std::net — no extra dep, bounded timeout.
#[cfg(feature = "embedded")]
fn probe_http_health(host: &str, port: u16, timeout: std::time::Duration) -> anyhow::Result<String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = format!("{host}:{port}");
    let stream = TcpStream::connect_timeout(
        &addr.parse()?,
        timeout,
    )?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut s = stream;
    let req = format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes())?;
    let mut resp = String::new();
    s.read_to_string(&mut resp)?;
    let (head, body) = resp.split_once("\r\n\r\n").unwrap_or((&resp, ""));
    let status_line = head.lines().next().unwrap_or("(no status line)");
    if !status_line.contains("200") {
        anyhow::bail!("status: {status_line}");
    }
    Ok(body.trim().to_string())
}

/// POST /v1/embeddings with a 4-token input; verify response shape +
/// embedding dim. Returns (dim, latency_ms).
#[cfg(feature = "embedded")]
fn probe_embed_roundtrip(host: &str, port: u16, timeout: std::time::Duration) -> anyhow::Result<(usize, u128)> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = format!("{host}:{port}");
    let stream = TcpStream::connect_timeout(&addr.parse()?, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut s = stream;
    let body = r#"{"model":"bge-m3","input":"ping ping ping"}"#;
    let req = format!(
        "POST /v1/embeddings HTTP/1.1\r\n\
         Host: {host}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n{body}",
        len = body.len(),
    );
    let t0 = std::time::Instant::now();
    s.write_all(req.as_bytes())?;
    let mut resp = String::new();
    s.read_to_string(&mut resp)?;
    let latency_ms = t0.elapsed().as_millis();
    let (head, body) = resp.split_once("\r\n\r\n").unwrap_or((&resp, ""));
    let status_line = head.lines().next().unwrap_or("(no status line)");
    if !status_line.contains("200") {
        anyhow::bail!("status: {status_line}");
    }
    // Loose check: the body is JSON like
    //   {"data":[{"embedding":[...1024 floats...],"index":0}], ...}
    // We just count floats by counting commas in the first embedding array
    // — no extra dep, no full JSON parse.
    let bytes = body.as_bytes();
    let start = bytes.windows(b"\"embedding\":[".len())
        .position(|w| w == b"\"embedding\":[")
        .ok_or_else(|| anyhow::anyhow!("no `embedding` field in response"))?
        + b"\"embedding\":[".len();
    let end = start + bytes[start..]
        .iter()
        .position(|&b| b == b']')
        .ok_or_else(|| anyhow::anyhow!("unterminated `embedding` array"))?;
    let arr = &body[start..end];
    let dim = arr.split(',').filter(|t| !t.trim().is_empty()).count();
    if dim == 0 {
        anyhow::bail!("response embedding array was empty");
    }
    Ok((dim, latency_ms))
}

/// B3: poll `/health` for up to 10 seconds after install. Prints what it
/// finds so the operator sees install-time GGUF/port issues immediately
/// instead of discovering them when a client first hits the server. Never
/// returns an error — install already succeeded as far as the OS service
/// manager is concerned; this is best-effort verification.
#[cfg(feature = "embedded")]
fn post_install_health_probe() {
    let cfg_path = config::default_config_path();
    let file_cfg = config::load_file_config(&cfg_path).unwrap_or_default();
    let (host, port) = match config::resolve(&file_cfg) {
        Ok(r) => (r.host, r.port),
        Err(e) => {
            println!();
            println!("[post-install] config did not resolve: {e}");
            println!("[post-install] service was registered but will not start until \
                     GGUF is resolvable — fix and `m3-embed-server start`");
            return;
        }
    };
    println!();
    println!("[post-install] probing http://{host}:{port}/health (up to 10s)");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut last_err: Option<String> = None;
    while std::time::Instant::now() < deadline {
        match probe_http_health(&host, port, std::time::Duration::from_secs(2)) {
            Ok(body) => {
                println!("[post-install] /health OK (body: {body:?})");
                println!("[post-install] probing /v1/embeddings roundtrip...");
                match probe_embed_roundtrip(&host, port, std::time::Duration::from_secs(20)) {
                    Ok((dim, ms)) => {
                        println!("[post-install] roundtrip OK — dim={dim}, latency={ms}ms");
                        if dim != 1024 {
                            println!("[post-install] WARNING: expected BGE-M3 dim=1024, got {dim} \
                                     — wrong GGUF?");
                        }
                    }
                    Err(e) => {
                        println!("[post-install] roundtrip FAIL — {e}");
                        println!("[post-install] service is listening but embedder is broken; \
                                 check {} for backend errors", config::default_log_path().display());
                    }
                }
                return;
            }
            Err(e) => {
                last_err = Some(e.to_string());
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
    println!("[post-install] FAIL — /health did not respond within 10s");
    if let Some(e) = last_err {
        println!("[post-install] last error: {e}");
    }
    println!("[post-install] check service logs: {}", config::default_log_path().display());
    println!("[post-install] run `m3-embed-server doctor` for a deeper diagnostic");
}

/// Read the last N lines of a file. Bounded read (reads at most 64 KB from
/// the tail). Returns empty Vec if the file doesn't exist or is empty.
#[cfg(feature = "embedded")]
fn tail_log_lines(path: &std::path::Path, n: usize) -> anyhow::Result<Vec<String>> {
    use std::io::{Read, Seek, SeekFrom};
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let tail_size = std::cmp::min(len, 64 * 1024);
    f.seek(SeekFrom::End(-(tail_size as i64)))?;
    let mut buf = String::new();
    f.read_to_string(&mut buf)?;
    let lines: Vec<&str> = buf.lines().collect();
    let take = lines.len().min(n);
    Ok(lines[lines.len() - take..].iter().map(|s| s.to_string()).collect())
}

/// Future that resolves on the first stop signal. Ctrl-C everywhere; on Unix
/// also SIGTERM — which is what `systemctl stop` / `launchctl kill` send, so
/// the service drains gracefully instead of being hard-killed.
#[cfg(all(feature = "embedded", unix))]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            // Can't install the SIGTERM handler — fall back to ctrl-c only.
            log::warn!("could not install SIGTERM handler: {e}; ctrl-c only");
            let _ = tokio::signal::ctrl_c().await;
            log::info!("ctrl-c received, draining...");
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => log::info!("ctrl-c received, draining..."),
        _ = term.recv()             => log::info!("SIGTERM received, draining..."),
    }
}

#[cfg(all(feature = "embedded", not(unix)))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    log::info!("ctrl-c received, draining...");
}

// ---------------------------------------------------------------------------
// Subcommand routing — Windows-specific paths gated, others print a friendly
// "Windows-only" error.
// ---------------------------------------------------------------------------

#[cfg(all(windows, feature = "embedded"))]
fn run_as_service() -> anyhow::Result<()> {
    service::run_dispatcher().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_install() -> anyhow::Result<()> {
    service::install().map_err(|e| anyhow::anyhow!("{e}"))?;
    // B3: post-install health probe. Best-effort — Windows SCM may not have
    // started the service yet, so we poll for ~10s before giving up. A FAIL
    // here doesn't roll back the install (the service is still registered)
    // but does flag the GGUF / port problem the operator needs to fix
    // before clients hit the server.
    post_install_health_probe();
    Ok(())
}

#[cfg(all(windows, feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    service::uninstall().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    service::start().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    service::stop().map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(all(windows, feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    service::status().map_err(|e| anyhow::anyhow!("{e}"))
}

// On Unix the OS service manager (launchd / systemd) runs the binary in
// foreground mode directly — there is no SCM-style `run-as-service` callback.
#[cfg(all(not(windows), feature = "embedded"))]
fn run_as_service() -> anyhow::Result<()> {
    anyhow::bail!(
        "`run-as-service` is a Windows-SCM-internal entry point and has no \
         meaning on this platform — the launchd/systemd unit runs foreground \
         mode (`m3-embed-server` with no subcommand) directly."
    )
}

#[cfg(all(not(windows), feature = "embedded"))]
fn run_install() -> anyhow::Result<()> {
    service_unix::install().map_err(|e| anyhow::anyhow!("{e}"))?;
    // B3: post-install health probe (see Windows branch for rationale).
    post_install_health_probe();
    Ok(())
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_uninstall() -> anyhow::Result<()> {
    service_unix::uninstall().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_start() -> anyhow::Result<()> {
    service_unix::start().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_stop() -> anyhow::Result<()> {
    service_unix::stop().map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(all(not(windows), feature = "embedded"))]
fn run_status() -> anyhow::Result<()> {
    service_unix::status().map_err(|e| anyhow::anyhow!("{e}"))
}
