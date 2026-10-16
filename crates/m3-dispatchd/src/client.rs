//! A client for the local endpoint (see `ipc`): one HTTP/1 request per
//! connection, as m3's Python client makes them.

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::Request;
use hyper_util::rt::TokioIo;

/// Why a request failed.
#[derive(Debug)]
pub enum ClientError {
    /// Nothing is serving the endpoint: the caller falls back to polling.
    NoDaemon(String),
    /// The daemon answered badly or the exchange broke.
    Failed(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::NoDaemon(m) => write!(f, "no m3-dispatchd: {m}"),
            ClientError::Failed(m) => write!(f, "m3-dispatchd request failed: {m}"),
        }
    }
}

#[cfg(unix)]
async fn connect(endpoint: &str) -> Result<tokio::net::UnixStream, ClientError> {
    use std::io::ErrorKind;
    // Only a missing socket or a refused one means "no daemon": the caller
    // then polls. Access denied is a daemon this user may not use, which
    // polling would hide.
    tokio::net::UnixStream::connect(endpoint).await.map_err(|e| match e.kind() {
        ErrorKind::NotFound | ErrorKind::ConnectionRefused => {
            ClientError::NoDaemon(format!("{endpoint}: {e}"))
        }
        ErrorKind::PermissionDenied => ClientError::Failed(format!(
            "{endpoint}: {e}. The socket exists but this user may not connect: it is mode 0600 for \
             the user who started m3-dispatchd, so another user's daemon may hold this path"
        )),
        _ => ClientError::Failed(format!("connecting to {endpoint}: {e}")),
    })
}

#[cfg(windows)]
async fn connect(
    endpoint: &str,
) -> Result<tokio::net::windows::named_pipe::NamedPipeClient, ClientError> {
    use tokio::net::windows::named_pipe::ClientOptions;
    const ERROR_FILE_NOT_FOUND: i32 = 2;
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_PIPE_BUSY: i32 = 231;
    // Every instance busy is transient (the server creates the next instance
    // as it accepts); only a missing pipe means no daemon.
    for _ in 0..50 {
        match ClientOptions::new().open(endpoint) {
            Ok(c) => return Ok(c),
            Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY) => {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await
            }
            Err(e) if e.raw_os_error() == Some(ERROR_FILE_NOT_FOUND) => {
                return Err(ClientError::NoDaemon(format!("{endpoint}: {e}")))
            }
            Err(e) if e.raw_os_error() == Some(ERROR_ACCESS_DENIED) => {
                return Err(ClientError::Failed(format!(
                    "{endpoint}: {e}. The pipe exists but this user may not open it: its DACL admits \
                     only the user who started m3-dispatchd, so another user's daemon may hold this name"
                )))
            }
            Err(e) => return Err(ClientError::Failed(format!("opening {endpoint}: {e}"))),
        }
    }
    Err(ClientError::Failed(format!(
        "{endpoint}: every pipe instance stayed busy for 1s"
    )))
}

/// POST `body` (JSON) to `path` and return the status and the reply body.
pub async fn post(
    endpoint: &str,
    path: &str,
    body: &serde_json::Value,
) -> Result<(u16, Bytes), ClientError> {
    let stream = connect(endpoint).await?;
    let fail = |what: &str, e: &dyn std::fmt::Display| {
        ClientError::Failed(format!("{what} (POST {path} on {endpoint}): {e}"))
    };
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|e| fail("handshake", &e))?;
    tokio::spawn(async move {
        let _ = conn.await;
    });
    let req = Request::post(path)
        .header("host", "m3-dispatchd")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .map_err(|e| fail("building the request", &e))?;
    let resp = sender
        .send_request(req)
        .await
        .map_err(|e| fail("sending", &e))?;
    let status = resp.status().as_u16();
    let bytes = resp
        .into_body()
        .collect()
        .await
        .map_err(|e| fail("reading the reply", &e))?
        .to_bytes();
    Ok((status, bytes))
}
