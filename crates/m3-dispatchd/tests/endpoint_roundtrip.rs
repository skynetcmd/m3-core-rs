//! The client and the server over a real local endpoint: a ring with an origin
//! reaches a waiter, and a missing endpoint reads as "no daemon".

use std::sync::Arc;
use std::time::Duration;

use m3_dispatchd::client::{post, ClientError};
use m3_dispatchd::{api, hub::Hub, ipc};

fn endpoint(tag: &str) -> String {
    let unique = format!("m3-dispatchd-test-{tag}-{}", std::process::id());
    if cfg!(windows) {
        format!(r"\\.\pipe\{unique}")
    } else {
        std::env::temp_dir()
            .join(format!("{unique}.sock"))
            .to_string_lossy()
            .into_owned()
    }
}

#[tokio::test]
async fn a_ring_with_an_origin_reaches_a_waiter() {
    let ep = endpoint("rt");
    let hub = Arc::new(Hub::new(Duration::from_millis(20), Duration::from_secs(60)));
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = {
        let ep = ep.clone();
        tokio::spawn(async move {
            ipc::serve(&ep, api::router(hub), async {
                let _ = stop_rx.await;
            })
            .await
        })
    };
    // The server binds asynchronously; retry until it answers.
    let wait = serde_json::json!({ "readers": ["a"], "timeout_s": 0 });
    let mut first = None;
    for _ in 0..100 {
        match post(&ep, "/wait", &wait).await {
            Ok(r) => {
                first = Some(r);
                break;
            }
            Err(ClientError::NoDaemon(_)) => tokio::time::sleep(Duration::from_millis(20)).await,
            Err(e) => panic!("{e}"),
        }
    }
    let (status, body) = first.expect("the server never answered");
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["wake"],
        "resync"
    );

    let (status, _) = post(
        &ep,
        "/ring",
        &serde_json::json!({ "recipients": ["a@1"], "origin": "node-b" }),
    )
    .await
    .unwrap();
    assert_eq!(status, 204);
    let (status, body) = post(
        &ep,
        "/wait",
        &serde_json::json!({ "readers": ["a"], "timeout_s": 5 }),
    )
    .await
    .unwrap();
    assert_eq!(status, 200);
    let wake: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        wake,
        serde_json::json!({ "wake": "mail", "count": 1, "recipients": ["a@1"], "origins": ["node-b"] })
    );

    let (status, _) = post(
        &ep,
        "/ring",
        &serde_json::json!({ "recipients": ["a"], "origin": "" }),
    )
    .await
    .unwrap();
    assert_eq!(status, 400, "an empty origin is refused");

    let _ = stop_tx.send(());
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn no_endpoint_is_no_daemon() {
    let err = post(&endpoint("absent"), "/wait", &serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(matches!(err, ClientError::NoDaemon(_)), "{err}");
}
