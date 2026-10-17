//! Serve the router over a LOCAL endpoint only, where the OS decides who may
//! connect: no tokens, no TLS, nothing to leak, and no cryptography in this
//! process (m3 routes all cryptography through `crypto_provider`; see
//! docs/FIPS_MODULE_BOUNDARY.md).
//!
//! * Unix: a domain socket created mode 0600, and each connection's peer uid
//!   must equal ours.
//! * Windows: a named pipe whose DACL admits only the current user's SID, with
//!   remote clients rejected.

use std::future::Future;

use axum::Router;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;

async fn serve_one<S>(stream: S, app: Router)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let svc = TowerToHyperService::new(app.into_service::<Incoming>());
    if let Err(e) = hyper::server::conn::http1::Builder::new()
        .serve_connection(TokioIo::new(stream), svc)
        .await
    {
        log::debug!("connection ended: {e}");
    }
}

#[cfg(unix)]
pub async fn serve(
    endpoint: &str,
    app: Router,
    shutdown: impl Future<Output = ()>,
) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    use tokio::net::{UnixListener, UnixStream};

    let path = std::path::Path::new(endpoint);
    if path.exists() {
        // A live daemon accepts; a stale socket file from a crash refuses.
        // Anything else (EACCES: another user's socket) is not proof of
        // staleness, so the file is left alone.
        match UnixStream::connect(path).await {
            Ok(_) => anyhow::bail!(
                "another m3-dispatchd is already serving {endpoint} (it accepted a connection)"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                std::fs::remove_file(path).map_err(|e| {
                    anyhow::anyhow!("cannot remove the stale socket {endpoint} (nothing accepts on it): {e}")
                })?;
                log::info!("removed the stale socket {endpoint}: nothing was accepting connections on it");
            }
            Err(e) => anyhow::bail!(
                "{endpoint} exists, and whether a daemon serves it cannot be told ({e}); it was left \
                 in place. Check its owner and mode (ls -l {endpoint}), and remove it if no \
                 m3-dispatchd runs"
            ),
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| {
            anyhow::anyhow!(
                "cannot create the socket's directory {}: {e}",
                dir.display()
            )
        })?;
    }
    // Create with a 0177 umask so the socket is never briefly wider than 0600.
    let old = unsafe { libc::umask(0o177) };
    let bound = UnixListener::bind(path);
    unsafe { libc::umask(old) };
    let listener = bound.map_err(|e| {
        // sun_path holds 108 bytes on Linux and 104 on macOS, NUL included.
        anyhow::anyhow!(
            "cannot bind the unix socket {endpoint} ({} bytes; a socket path must stay under about \
             104): {e}",
            endpoint.len()
        )
    })?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| anyhow::anyhow!("cannot restrict {endpoint} to mode 0600: {e}"))?;
    let me = unsafe { libc::geteuid() };
    log::info!("serving on unix socket {endpoint}");
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                // One failed accept (EMFILE, ECONNABORTED) must not end the
                // daemon: back off and keep serving.
                let stream = match accepted {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        log::warn!("accept on {endpoint} failed: {e}; retrying in 100 ms");
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    }
                };
                match stream.peer_cred() {
                    Ok(cred) if cred.uid() == me => { tokio::spawn(serve_one(stream, app.clone())); }
                    Ok(cred) => log::warn!("refused a connection from uid {} (serving uid {me})", cred.uid()),
                    Err(e) => log::warn!("refused a connection whose peer could not be identified: {e}"),
                }
            }
        }
    }
    let _ = std::fs::remove_file(path);
    Ok(())
}

#[cfg(windows)]
mod win {
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;

    /// Security attributes for the pipe. Sound to send between threads: the
    /// descriptor they point at is leaked and never written after creation.
    pub struct Attrs(pub SECURITY_ATTRIBUTES);
    unsafe impl Send for Attrs {}

    /// Security attributes admitting only `sid` (full access) and nobody else.
    /// The descriptor is leaked on purpose: it must outlive every pipe instance,
    /// which is the life of the process.
    pub fn only_user(sid: &str) -> anyhow::Result<Attrs> {
        let sddl: Vec<u16> = format!("D:P(A;;GA;;;{sid})")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let mut sd = std::ptr::null_mut();
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut sd,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            anyhow::bail!(
                "building the pipe's security descriptor failed: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(Attrs(SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: sd,
            bInheritHandle: 0,
        }))
    }
}

#[cfg(windows)]
pub async fn serve(
    endpoint: &str,
    app: Router,
    shutdown: impl Future<Output = ()>,
) -> anyhow::Result<()> {
    use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};

    if !endpoint.starts_with(r"\\.\pipe\") {
        anyhow::bail!(r"a Windows endpoint must be a named pipe (\\.\pipe\NAME); got {endpoint}");
    }
    let sid = m3_config::os::current_user_sid().map_err(anyhow::Error::msg)?;
    let mut attrs = win::only_user(&sid)?;
    let create = |first: bool, attrs: &mut win::Attrs| -> std::io::Result<NamedPipeServer> {
        let mut opts = ServerOptions::new();
        opts.reject_remote_clients(true).first_pipe_instance(first);
        unsafe {
            opts.create_with_security_attributes_raw(
                endpoint,
                &mut attrs.0 as *mut _ as *mut core::ffi::c_void,
            )
        }
    };
    let mut server = create(true, &mut attrs).map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            // Inferred, not observed: access denied on the FIRST instance is
            // what an existing pipe of that name causes, ours or another user's.
            anyhow::anyhow!(
                "cannot create {endpoint}: access denied for the first pipe instance, which is what \
                 an existing pipe of that name causes; another m3-dispatchd (this user's or another's) \
                 is probably serving it ({e})"
            )
        } else {
            anyhow::anyhow!("cannot create the named pipe {endpoint}: {e}")
        }
    })?;
    log::info!("serving on named pipe {endpoint} (user {sid} only, remote clients rejected)");
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            connected = server.connect() => {
                // One failed connect (a client that vanished mid-connect) must
                // not end the daemon. The instance may be unusable after it,
                // so it is replaced either way.
                let client = match connected {
                    Ok(()) => Some(server),
                    Err(e) => {
                        log::warn!("a connection on {endpoint} failed before it was served: {e}; replacing the pipe instance");
                        None
                    }
                };
                server = next_instance(&create, &mut attrs, endpoint).await?;
                if let Some(client) = client {
                    tokio::spawn(serve_one(client, app.clone()));
                }
            }
        }
    }
    Ok(())
}

/// The next pipe instance, retried for 5 s: a transient failure is warned
/// about and survived, a lasting one ends the daemon with its cause.
#[cfg(windows)]
async fn next_instance<C>(
    create: &C,
    attrs: &mut win::Attrs,
    endpoint: &str,
) -> anyhow::Result<tokio::net::windows::named_pipe::NamedPipeServer>
where
    C: Fn(
        bool,
        &mut win::Attrs,
    ) -> std::io::Result<tokio::net::windows::named_pipe::NamedPipeServer>,
{
    const TRIES: u32 = 50;
    let mut last = None;
    for _ in 0..TRIES {
        match create(false, attrs) {
            Ok(s) => return Ok(s),
            Err(e) => {
                log::warn!(
                    "cannot create the next instance of {endpoint}: {e}; retrying in 100 ms"
                );
                last = Some(e);
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    Err(anyhow::anyhow!(
        "cannot create a new instance of the named pipe {endpoint} after {TRIES} tries over 5 s ({}); \
         m3-dispatchd stops serving",
        last.map(|e| e.to_string()).unwrap_or_default()
    ))
}
