//! Transport security for the shared embed server: TLS and a bearer token.
//!
//! The server receives raw memory text and returns vectors that can be partly
//! inverted back to it, so both directions are sensitive. Two listeners:
//!
//! * `host:port`, the local listener, must be a loopback address (default
//!   127.0.0.1) and serves plaintext: only this machine can connect, the same
//!   boundary the agent doorbell relies on. Local clients, probes and
//!   installers keep using it unchanged.
//! * `remote_listen`, optional, is for other machines and always requires BOTH
//!   TLS and a bearer token; there is no plaintext form of it. A private
//!   network (Tailscale is the recommended one) decides who can reach the
//!   port; TLS and the token still apply inside it, because WireGuard's
//!   algorithms are not FIPS-approved and network reachability is not identity.
//!
//! A non-loopback `host` is refused (it would be plaintext on the network),
//! with the message pointing at `remote_listen`.
//!
//! TLS is rustls on aws-lc-rs, TLS 1.3 only, AES-GCM suites and the P-256/P-384
//! groups (approved algorithms on every build). Only the `fips` feature links
//! AWS-LC's FIPS module; under `crypto.fips_strict` the server refuses TLS from
//! any other build, since that is a fact it can check rather than attest.
//!
//! The token is stored as its SHA-256 (`token_sha256`): the config file can be
//! readable by other local users (Windows keeps it under %PROGRAMDATA%), and a
//! digest of a random 256-bit token is no use to whoever reads it. `/health`
//! needs no token, so supervisors and probes can check liveness.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aws_lc_rs::{constant_time, digest};
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use rustls::crypto::{aws_lc_rs as provider, CryptoProvider};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// Exit status when the configuration is refused (EX_CONFIG). Like a taken
/// port, a restart cannot cure it, so supervisors are told not to retry.
pub const EXIT_CONFIG_REFUSED: i32 = 78;

/// `plan` refused the configuration; the message says why and what fixes it.
#[derive(Debug)]
pub struct ConfigRefused(pub String);

impl std::fmt::Display for ConfigRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigRefused {}

/// The settings that decide transport security, as read from config.
#[derive(Debug, Clone, Default)]
pub struct SecurityConfig {
    /// The listener for other machines: `IP` or `IP:port` (the port defaults
    /// to the local listener's).
    pub remote_listen: Option<String>,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    /// Lowercase hex SHA-256 of the bearer token.
    pub token_sha256: Option<String>,
    pub fips_mode: bool,
    pub fips_strict: bool,
}

/// The listener for other machines, ready to serve.
pub struct Remote {
    pub addr: SocketAddr,
    pub tls: Arc<rustls::ServerConfig>,
    pub token_digest: [u8; 32],
}

/// What the server enforces, once the config has been checked.
pub struct Security {
    pub remote: Option<Remote>,
}

impl Security {
    /// One line for the startup log: what each listener is.
    pub fn describe(&self, local: SocketAddr) -> String {
        match &self.remote {
            None => format!("{local} plaintext (loopback only); no listener for other machines"),
            Some(r) => {
                let module = if provider_is_fips() { "aws-lc-rs FIPS module" } else { "aws-lc-rs, not the FIPS module" };
                format!("{local} plaintext (loopback only); {} TLS 1.3 ({module}) with a bearer token", r.addr)
            }
        }
    }
}

/// True when this build's TLS runs on AWS-LC's FIPS module (the `fips` feature).
pub fn provider_is_fips() -> bool {
    provider::default_provider().fips()
}

/// Check `cfg` against the local listen address and build what the server
/// enforces. Every refusal says what was observed and what fixes it.
pub fn plan(local: SocketAddr, cfg: &SecurityConfig) -> anyhow::Result<Security> {
    if !local.ip().is_loopback() {
        anyhow::bail!(
            "refusing to serve embeddings in plaintext on {}: the server receives raw memory text, and \
             any address other than loopback can be reached from other machines. Set the host back to \
             127.0.0.1; to serve other machines, add a TLS listener with a bearer token: \
             `m3 embedder secure --listen <address>` (on a private network such as Tailscale as well)",
            local.ip()
        );
    }
    let token_digest = match cfg.token_sha256.as_deref() {
        None => None,
        Some(hex) => Some(parse_digest(hex)?),
    };
    let tls_pair = match (&cfg.tls_cert, &cfg.tls_key) {
        (None, None) => None,
        (Some(cert), Some(key)) => Some((cert, key)),
        _ => anyhow::bail!(
            "tls_cert and tls_key must be set together (one is missing). Set both with \
             `m3 embedder secure`, or remove both with `m3 embedder secure --off`"
        ),
    };
    let Some(remote) = cfg.remote_listen.as_deref() else {
        if tls_pair.is_some() || token_digest.is_some() {
            log::warn!(
                "TLS or a token is configured but remote_listen is not, so only loopback is served \
                 and they are unused. Set the address with `m3 embedder secure --listen <address>`"
            );
        }
        return Ok(Security { remote: None });
    };
    let addr = parse_listen(remote, local.port())?;
    if addr == local {
        anyhow::bail!("remote_listen {addr} is the local listener's own address; choose another address or port");
    }
    let (Some((cert, key)), Some(token_digest)) = (tls_pair, token_digest) else {
        anyhow::bail!(
            "remote_listen {addr} needs both a TLS certificate and a bearer token; there is no plaintext \
             listener for other machines. Run `m3 embedder secure`"
        );
    };
    if cfg.fips_strict && !provider_is_fips() {
        anyhow::bail!(
            "crypto.fips_strict is on, but this m3-embed-server was built without AWS-LC's FIPS module, so \
             its TLS is not a validated module. Use the FIPS build of m3-embed-server, or remove \
             remote_listen to serve this machine only"
        );
    }
    if cfg.fips_mode && !provider_is_fips() {
        log::info!(
            "crypto.fips_mode: TLS here is restricted to approved algorithms but runs on aws-lc-rs without \
             its FIPS module; crypto.fips_strict would refuse it"
        );
    }
    let tls = Arc::new(server_tls_config(cert, key)?);
    Ok(Security { remote: Some(Remote { addr, tls, token_digest }) })
}

/// `IP` or `IP:port` (IPv6 as `[::1]:port`); a bare IP takes `default_port`.
pub fn parse_listen(s: &str, default_port: u16) -> anyhow::Result<SocketAddr> {
    let s = s.trim();
    if let Ok(a) = s.parse::<SocketAddr>() {
        return Ok(a);
    }
    if let Ok(ip) = s.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>() {
        return Ok(SocketAddr::new(ip, default_port));
    }
    anyhow::bail!("remote_listen {s:?} is not an IP address or IP:port (a hostname is not accepted)")
}

fn parse_digest(hex: &str) -> anyhow::Result<[u8; 32]> {
    let hex = hex.trim();
    let bad = || {
        anyhow::anyhow!(
            "token_sha256 must be 64 hexadecimal characters (the SHA-256 of the token), got {} \
             characters. Re-run `m3 embedder secure`",
            hex.len()
        )
    };
    if hex.len() != 64 {
        return Err(bad());
    }
    let mut out = [0u8; 32];
    for (i, pair) in hex.as_bytes().chunks(2).enumerate() {
        let s = std::str::from_utf8(pair).map_err(|_| bad())?;
        out[i] = u8::from_str_radix(s, 16).map_err(|_| bad())?;
    }
    Ok(out)
}

/// The SHA-256 of `token`, through aws-lc-rs (the FIPS module in a FIPS build).
pub fn token_digest(token: &[u8]) -> [u8; 32] {
    let d = digest::digest(&digest::SHA256, token);
    let mut out = [0u8; 32];
    out.copy_from_slice(d.as_ref());
    out
}

/// TLS 1.3 only, AES-GCM suites, P-256/P-384 key exchange: approved algorithms
/// whichever aws-lc-rs build this is.
fn approved_provider() -> CryptoProvider {
    let base = provider::default_provider();
    CryptoProvider {
        cipher_suites: vec![
            provider::cipher_suite::TLS13_AES_256_GCM_SHA384,
            provider::cipher_suite::TLS13_AES_128_GCM_SHA256,
        ],
        kx_groups: vec![provider::kx_group::SECP384R1, provider::kx_group::SECP256R1],
        ..base
    }
}

/// Load the certificate chain and private key, refusing a key file that other
/// users can read (Unix; the Windows writer restricts the file's ACL).
pub fn server_tls_config(cert: &Path, key: &Path) -> anyhow::Result<rustls::ServerConfig> {
    check_key_file_private(key)?;
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)
        .map_err(|e| anyhow::anyhow!("cannot read TLS certificate {}: {e}", cert.display()))?
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("cannot parse TLS certificate {}: {e}", cert.display()))?;
    if certs.is_empty() {
        anyhow::bail!("no certificate found in {} (expected PEM)", cert.display());
    }
    let key_der = PrivateKeyDer::from_pem_file(key)
        .map_err(|e| anyhow::anyhow!("cannot read TLS private key {}: {e}", key.display()))?;
    let mut tls = rustls::ServerConfig::builder_with_provider(Arc::new(approved_provider()))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| anyhow::anyhow!("TLS setup failed: {e}"))?
        .with_no_client_auth()
        .with_single_cert(certs, key_der)
        .map_err(|e| {
            anyhow::anyhow!(
                "TLS certificate {} and key {} do not form a usable pair: {e}",
                cert.display(),
                key.display()
            )
        })?;
    tls.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(tls)
}

#[cfg(unix)]
fn check_key_file_private(key: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(key)
        .map_err(|e| anyhow::anyhow!("cannot read TLS private key {}: {e}", key.display()))?;
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        anyhow::bail!(
            "TLS private key {} is readable by other users (mode {mode:03o}). Fix: chmod 600 {}",
            key.display(),
            key.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_key_file_private(_key: &Path) -> anyhow::Result<()> {
    Ok(())
}

/// Axum middleware: every route except `/health` needs `Authorization: Bearer`.
pub async fn require_bearer(State(want): State<Arc<[u8; 32]>>, req: Request, next: Next) -> Response {
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }
    let got = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| token_digest(t.trim().as_bytes()));
    match got {
        Some(d) if constant_time::verify_slices_are_equal(&d, want.as_ref()).is_ok() => next.run(req).await,
        _ => {
            log::info!("rejected {} {}: missing or wrong bearer token", req.method(), req.uri().path());
            (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                "a bearer token is required: on the client, `m3 secret set M3_EMBED_SERVER_TOKEN`",
            )
                .into_response()
        }
    }
}

/// Serve `app` over TLS on `listener` until `shutdown` resolves, then give open
/// connections up to 10 s to finish.
pub async fn serve_tls<F>(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    tls: Arc<rustls::ServerConfig>,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    // HTTP/1.1 only: the clients (m3's httpx/urllib) speak it, and llama-server
    // parity kept the plaintext path HTTP/1.1 as well.
    use hyper::server::conn::http1::Builder;
    use hyper_util::rt::TokioIo;
    use hyper_util::server::graceful::GracefulShutdown;
    use hyper_util::service::TowerToHyperService;
    use std::time::Duration;

    let acceptor = tokio_rustls::TlsAcceptor::from(tls);
    let graceful = GracefulShutdown::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (tcp, peer) = match accepted {
                    Ok(x) => x,
                    Err(e) => {
                        // EMFILE and friends: back off instead of spinning.
                        log::warn!("accept failed: {e}; retrying in 100 ms");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        continue;
                    }
                };
                let acceptor = acceptor.clone();
                let svc = TowerToHyperService::new(app.clone());
                let watcher = graceful.watcher();
                tokio::spawn(async move {
                    let stream = match tokio::time::timeout(Duration::from_secs(10), acceptor.accept(tcp)).await {
                        Ok(Ok(s)) => s,
                        Ok(Err(e)) => {
                            log::info!("TLS handshake from {peer} failed: {e}");
                            return;
                        }
                        Err(_) => {
                            log::info!("TLS handshake from {peer} timed out after 10 s");
                            return;
                        }
                    };
                    let conn = Builder::new().serve_connection(TokioIo::new(stream), svc);
                    if let Err(e) = watcher.watch(conn).await {
                        log::debug!("connection from {peer} ended: {e}");
                    }
                });
            }
            _ = &mut shutdown => break,
        }
    }
    drop(listener);
    if tokio::time::timeout(Duration::from_secs(10), graceful.shutdown()).await.is_err() {
        log::warn!("connections still open 10 s after shutdown; closing them");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    const LOOP: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
    const LOCAL: SocketAddr = SocketAddr::new(LOOP, 8082);
    const LAN: IpAddr = IpAddr::V4(Ipv4Addr::new(100, 100, 50, 63));

    fn hex(d: &[u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    struct Pem {
        _dir: tempfile::TempDir,
        cert: PathBuf,
        key: PathBuf,
        ca_der: Vec<u8>,
    }

    fn self_signed() -> Pem {
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cert = dir.path().join("cert.pem");
        let key = dir.path().join("key.pem");
        std::fs::write(&cert, ck.cert.pem()).unwrap();
        std::fs::write(&key, ck.key_pair.serialize_pem()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        Pem { ca_der: ck.cert.der().to_vec(), _dir: dir, cert, key }
    }

    fn secured(pem: &Pem, listen: &str) -> SecurityConfig {
        SecurityConfig {
            remote_listen: Some(listen.into()),
            tls_cert: Some(pem.cert.clone()),
            tls_key: Some(pem.key.clone()),
            token_sha256: Some(hex(&token_digest(b"t"))),
            ..Default::default()
        }
    }

    #[test]
    fn the_default_is_loopback_plaintext_with_no_remote_listener() {
        assert!(plan(LOCAL, &SecurityConfig::default()).unwrap().remote.is_none());
        let v6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8082);
        assert!(plan(v6, &SecurityConfig::default()).is_ok());
    }

    #[test]
    fn a_non_loopback_host_is_refused_and_pointed_at_remote_listen() {
        let pem = self_signed();
        for ip in [LAN, IpAddr::V4(Ipv4Addr::UNSPECIFIED), IpAddr::V6(Ipv6Addr::UNSPECIFIED)] {
            // Refused even with TLS and a token configured: the host listener is plaintext.
            let e = plan(SocketAddr::new(ip, 8082), &secured(&pem, "100.100.50.63")).err().unwrap().to_string();
            assert!(e.contains("plaintext") && e.contains("--listen"), "{e}");
        }
    }

    #[test]
    fn the_remote_listener_needs_tls_and_a_token() {
        let pem = self_signed();
        let full = secured(&pem, "100.100.50.63");
        let r = plan(LOCAL, &full).unwrap().remote.expect("remote listener");
        assert_eq!(r.addr, SocketAddr::new(LAN, 8082), "a bare IP takes the local port");
        let no_token = SecurityConfig { token_sha256: None, ..full.clone() };
        let no_tls = SecurityConfig { tls_cert: None, tls_key: None, ..full.clone() };
        for cfg in [no_token, no_tls] {
            let e = plan(LOCAL, &cfg).err().unwrap().to_string();
            assert!(e.contains("needs both") && e.contains("m3 embedder secure"), "{e}");
        }
        let same = secured(&pem, "127.0.0.1:8082");
        assert!(plan(LOCAL, &same).err().unwrap().to_string().contains("own address"));
        let host = secured(&pem, "embed.example.ts.net");
        assert!(plan(LOCAL, &host).err().unwrap().to_string().contains("hostname is not accepted"));
        assert_eq!(parse_listen("[::1]:9", 1).unwrap().port(), 9);
        assert_eq!(parse_listen("::1", 7).unwrap().port(), 7);
    }

    #[test]
    fn tls_without_remote_listen_serves_loopback_only() {
        let pem = self_signed();
        let cfg = SecurityConfig { remote_listen: None, ..secured(&pem, "x") };
        assert!(plan(LOCAL, &cfg).unwrap().remote.is_none());
    }

    #[test]
    fn half_a_tls_pair_and_a_malformed_digest_are_refused() {
        let pem = self_signed();
        let half = SecurityConfig { tls_cert: Some(pem.cert), ..Default::default() };
        assert!(plan(LOCAL, &half).err().unwrap().to_string().contains("set together"));
        let bad = SecurityConfig { token_sha256: Some("abc".into()), ..Default::default() };
        assert!(plan(LOCAL, &bad).err().unwrap().to_string().contains("64 hexadecimal"));
        let non_hex = SecurityConfig { token_sha256: Some("z".repeat(64)), ..Default::default() };
        assert!(plan(LOCAL, &non_hex).is_err());
    }

    /// The `fips` feature, and only it, reports AWS-LC's FIPS module: a FIPS
    /// build that silently linked the ordinary library would fail here.
    #[test]
    fn the_fips_feature_is_what_reports_fips() {
        assert_eq!(provider_is_fips(), cfg!(feature = "fips"));
    }

    #[test]
    fn fips_strict_refuses_tls_from_a_build_without_the_fips_module() {
        let pem = self_signed();
        let cfg = SecurityConfig { fips_strict: true, ..secured(&pem, "100.100.50.63") };
        let r = plan(LOCAL, &cfg);
        if provider_is_fips() {
            assert!(r.is_ok());
        } else {
            assert!(r.err().unwrap().to_string().contains("FIPS build"));
        }
        // Loopback only needs no TLS, so strict mode does not refuse it.
        assert!(plan(LOCAL, &SecurityConfig { fips_strict: true, ..Default::default() }).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn a_key_other_users_can_read_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let pem = self_signed();
        std::fs::set_permissions(&pem.key, std::fs::Permissions::from_mode(0o644)).unwrap();
        let e = server_tls_config(&pem.cert, &pem.key).err().unwrap().to_string();
        assert!(e.contains("chmod 600"), "{e}");
    }

    #[test]
    fn the_digest_matches_sha256() {
        // SHA-256("abc"), FIPS 180-2 appendix B.1.
        assert_eq!(
            hex(&token_digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// End to end over a real socket: TLS 1.3 handshake against the pinned
    /// certificate, /health open, everything else gated by the token.
    #[tokio::test]
    async fn tls_and_bearer_gate_a_real_listener() {
        use axum::routing::get;
        let pem = self_signed();
        let token = "s3cret-token";
        // A loopback remote listener on another port stands in for a tailnet address.
        let cfg = SecurityConfig {
            token_sha256: Some(hex(&token_digest(token.as_bytes()))),
            ..secured(&pem, "127.0.0.1:1")
        };
        let remote = plan(LOCAL, &cfg).unwrap().remote.unwrap();
        let app = axum::Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/embedding", get(|| async { "vec" }))
            .layer(axum::middleware::from_fn_with_state(Arc::new(remote.token_digest), require_bearer));
        let listener = tokio::net::TcpListener::bind((LOOP, 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve_tls(listener, app, remote.tls, async move {
            let _ = stop_rx.await;
        }));

        let mut roots = rustls::RootCertStore::empty();
        roots.add(CertificateDer::from(pem.ca_der.clone())).unwrap();
        let client_cfg = rustls::ClientConfig::builder_with_provider(Arc::new(provider::default_provider()))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(client_cfg));

        async fn get_status(
            connector: &tokio_rustls::TlsConnector,
            port: u16,
            path: &str,
            auth: Option<&str>,
        ) -> String {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let tcp = tokio::net::TcpStream::connect((LOOP, port)).await.unwrap();
            let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
            let mut tls = connector.connect(name, tcp).await.unwrap();
            assert_eq!(tls.get_ref().1.protocol_version(), Some(rustls::ProtocolVersion::TLSv1_3));
            let auth = auth.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
            let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n");
            tls.write_all(req.as_bytes()).await.unwrap();
            let mut buf = Vec::new();
            let _ = tls.read_to_end(&mut buf).await;
            String::from_utf8_lossy(&buf).lines().next().unwrap_or_default().to_string()
        }

        assert!(get_status(&connector, port, "/health", None).await.contains("200"));
        assert!(get_status(&connector, port, "/embedding", None).await.contains("401"));
        assert!(get_status(&connector, port, "/embedding", Some("wrong")).await.contains("401"));
        assert!(get_status(&connector, port, "/embedding", Some(token)).await.contains("200"));

        // A plaintext client gets no HTTP answer from the TLS port.
        {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut tcp = tokio::net::TcpStream::connect((LOOP, port)).await.unwrap();
            tcp.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
            let mut buf = Vec::new();
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), tcp.read_to_end(&mut buf)).await;
            assert!(!String::from_utf8_lossy(&buf).contains("200 OK"));
        }

        stop_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(15), server).await.unwrap().unwrap().unwrap();
    }
}
