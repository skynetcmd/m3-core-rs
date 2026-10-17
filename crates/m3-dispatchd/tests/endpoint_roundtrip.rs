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

    // Nothing to ring is refused, not answered 204.
    let (status, body) = post(&ep, "/ring", &serde_json::json!({ "recipients": [""] }))
        .await
        .unwrap();
    assert_eq!(status, 400);
    let err: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(err["error"].as_str().unwrap().contains("empty"), "{err}");

    // A body of the wrong shape is answered in the same JSON error shape.
    let (status, body) = post(&ep, "/ring", &serde_json::json!({ "recipients": 5 }))
        .await
        .unwrap();
    assert!((400..500).contains(&status), "{status}");
    let err: serde_json::Value = serde_json::from_slice(&body).expect("a JSON error body");
    assert!(err["error"].is_string(), "{err}");

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

#[tokio::test]
async fn m3_wait_exits_when_its_parent_closes_stdin() {
    let ep = endpoint("stdin");
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
    for _ in 0..100 {
        if post(
            &ep,
            "/wait",
            &serde_json::json!({ "readers": ["r"], "timeout_s": 0 }),
        )
        .await
        .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_m3-wait"))
        .args([
            "--endpoint",
            &ep,
            "--reader",
            "r",
            "--timeout",
            "60",
            "--exit-when-stdin-closes",
        ])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        child.try_wait().unwrap().is_none(),
        "m3-wait should be blocked on the wait"
    );
    drop(child.stdin.take()); // the parent "dies"
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "m3-wait outlived its parent"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(status.code(), Some(4));
    let _ = stop_tx.send(());
    server.await.unwrap().unwrap();
}
