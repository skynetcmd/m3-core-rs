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
    if let Err(e) = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await {
        log::debug!("connection ended: {e}");
    }
}

#[cfg(unix)]
pub async fn serve(endpoint: &str, app: Router, shutdown: impl Future<Output = ()>) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    use tokio::net::{UnixListener, UnixStream};

    let path = std::path::Path::new(endpoint);
    if path.exists() {
        // A live daemon answers; a stale socket file from a crash does not.
        if UnixStream::connect(path).await.is_ok() {
            anyhow::bail!("another m3-dispatchd is already serving {endpoint}");
        }
        std::fs::remove_file(path)?;
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Create with a 0177 umask so the socket is never briefly wider than 0600.
    let old = unsafe { libc::umask(0o177) };
    let bound = UnixListener::bind(path);
    unsafe { libc::umask(old) };
    let listener = bound?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    let me = unsafe { libc::geteuid() };
    log::info!("serving on unix socket {endpoint}");
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
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
    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, HANDLE};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{GetTokenInformation, TokenUser, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    /// The current user's SID as a string ("S-1-5-21-...").
    pub fn current_user_sid() -> anyhow::Result<String> {
        unsafe {
            let mut token: HANDLE = std::ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                anyhow::bail!("OpenProcessToken failed: {}", std::io::Error::last_os_error());
            }
            let mut len = 0u32;
            GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len);
            let mut buf = vec![0u8; len as usize];
            let ok = GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), len, &mut len);
            CloseHandle(token);
            if ok == 0 {
                anyhow::bail!("GetTokenInformation failed: {}", std::io::Error::last_os_error());
            }
            let user = &*(buf.as_ptr() as *const TOKEN_USER);
            let mut wide: *mut u16 = std::ptr::null_mut();
            if ConvertSidToStringSidW(user.User.Sid, &mut wide) == 0 {
                anyhow::bail!("ConvertSidToStringSidW failed: {}", std::io::Error::last_os_error());
            }
            let n = (0..).take_while(|&i| *wide.add(i) != 0).count();
            let s = String::from_utf16_lossy(std::slice::from_raw_parts(wide, n));
            LocalFree(wide.cast());
            Ok(s)
        }
    }

    /// Security attributes admitting only `sid` (full access) and nobody else.
    /// The descriptor is leaked on purpose: it must outlive every pipe instance,
    /// which is the life of the process.
    pub fn only_user(sid: &str) -> anyhow::Result<SECURITY_ATTRIBUTES> {
        let sddl: Vec<u16> = format!("D:P(A;;GA;;;{sid})").encode_utf16().chain(std::iter::once(0)).collect();
        let mut sd = std::ptr::null_mut();
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.as_ptr(), SDDL_REVISION_1, &mut sd, std::ptr::null_mut())
        };
        if ok == 0 {
            anyhow::bail!("building the pipe's security descriptor failed: {}", std::io::Error::last_os_error());
        }
        Ok(SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: sd,
            bInheritHandle: 0,
        })
    }
}

#[cfg(windows)]
pub async fn serve(endpoint: &str, app: Router, shutdown: impl Future<Output = ()>) -> anyhow::Result<()> {
    use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};

    if !endpoint.starts_with(r"\\.\pipe\") {
        anyhow::bail!(r"a Windows endpoint must be a named pipe (\\.\pipe\NAME); got {endpoint}");
    }
    let sid = win::current_user_sid()?;
    let mut attrs = win::only_user(&sid)?;
    let create = |first: bool, attrs: &mut windows_sys::Win32::Security::SECURITY_ATTRIBUTES| -> std::io::Result<NamedPipeServer> {
        let mut opts = ServerOptions::new();
        opts.reject_remote_clients(true).first_pipe_instance(first);
        unsafe { opts.create_with_security_attributes_raw(endpoint, attrs as *mut _ as *mut core::ffi::c_void) }
    };
    let mut server = create(true, &mut attrs).map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            anyhow::anyhow!("another m3-dispatchd is already serving {endpoint}")
        } else {
            anyhow::anyhow!("cannot create {endpoint}: {e}")
        }
    })?;
    log::info!("serving on named pipe {endpoint} (user {sid} only, remote clients rejected)");
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            connected = server.connect() => {
                connected?;
                let client = server;
                server = create(false, &mut attrs)?;
                tokio::spawn(serve_one(client, app.clone()));
            }
        }
    }
    Ok(())
}
