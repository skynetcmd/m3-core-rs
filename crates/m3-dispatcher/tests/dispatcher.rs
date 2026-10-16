//! Dispatcher behavior tests using a deterministic mock backend.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use m3_dispatcher::{
    Batch, BatchOutput, BreakerCfg, Dispatcher, DispatcherConfig, ModelBackend,
};
use m3_error::{M3Error, Result};

/// Echoes a deterministic vector per text; records the size of each batch it saw.
struct MockBackend {
    batch_sizes: Arc<std::sync::Mutex<Vec<usize>>>,
    calls: Arc<AtomicUsize>,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            batch_sizes: Arc::new(std::sync::Mutex::new(Vec::new())),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ModelBackend for MockBackend {
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.batch_sizes.lock().unwrap().push(batch.texts.len());
        let rows = batch
            .texts
            .iter()
            .map(|t| vec![t.len() as f32, 1.0])
            .collect();
        Ok(BatchOutput::new(rows))
    }
}

/// Always fails — drives the circuit breaker.
struct FailBackend;
impl ModelBackend for FailBackend {
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        Err(M3Error::Backend("boom".into()))
    }
}

/// Hangs forever — used to fill in-flight slots so the queue backs up.
struct HangBackend;
impl ModelBackend for HangBackend {
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        loop {
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
    }
}

#[tokio::test]
async fn coalesces_multiple_embed_calls_into_one_batch() {
    let backend = MockBackend::new();
    let sizes = backend.batch_sizes.clone();
    let calls = backend.calls.clone();
    let cfg = DispatcherConfig {
        coalesce_window_ms: 25,
        ..Default::default()
    };
    let disp = Arc::new(Dispatcher::new(cfg, backend));

    let mut handles = Vec::new();
    for i in 0..5 {
        let d = disp.clone();
        handles.push(tokio::spawn(async move {
            d.embed(format!("text-{i}")).await
        }));
    }
    for h in handles {
        let v = h.await.unwrap().unwrap();
        assert_eq!(v.len(), 2);
    }

    // All 5 should have landed in the same coalescing window -> 1 backend call.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(sizes.lock().unwrap().as_slice(), &[5]);
}

#[tokio::test]
async fn circuit_breaker_opens_after_n_failures() {
    let cfg = DispatcherConfig {
        coalesce_window_ms: 5,
        circuit_breaker: BreakerCfg { failure_threshold: 3, open_secs: 30 },
        ..Default::default()
    };
    let disp = Dispatcher::new(cfg, FailBackend);

    // First 3 calls hit the backend and fail.
    for _ in 0..3 {
        assert!(disp.embed("x".into()).await.is_err());
    }
    // Breaker now open: subsequent call fast-fails before reaching the backend.
    let err = disp.embed("x".into()).await.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("circuit breaker open"), "got: {msg}");
    // The refusal carries the failure that opened it, once, and the cooldown.
    assert!(msg.contains("last failure: boom") && msg.contains("3 consecutive"), "got: {msg}");
    assert!(!msg.contains("backend error: backend error"), "got: {msg}");
    let st = disp.breaker_status().await;
    assert!(st.open && st.retry_after.is_some(), "{st:?}");
    assert_eq!((st.consecutive_failures, st.last_error.as_deref()), (3, Some("boom")));
}

/// Fails every batch with a client length error.
struct TooLongBackend;
impl ModelBackend for TooLongBackend {
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        Err(M3Error::InputTooLong { tokens: 9000, n_ctx: 8192, index: Some(0) })
    }
}

/// Over-long input is the caller's error: it stays typed on the coalesced path
/// (so the server answers 413, not 500) and never opens the breaker.
#[tokio::test]
async fn input_too_long_stays_typed_and_does_not_open_the_breaker() {
    let cfg = DispatcherConfig {
        coalesce_window_ms: 5,
        circuit_breaker: BreakerCfg { failure_threshold: 2, open_secs: 30 },
        ..Default::default()
    };
    let disp = Dispatcher::new(cfg, TooLongBackend);
    for _ in 0..5 {
        let e = disp.embed("x".into()).await.unwrap_err();
        assert!(matches!(e, M3Error::InputTooLong { tokens: 9000, n_ctx: 8192, .. }), "{e}");
        let e = disp.embed_batch(vec!["a".into(), "b".into()]).await.unwrap_err();
        assert!(matches!(e, M3Error::InputTooLong { .. }), "{e}");
    }
    assert!(!disp.breaker_status().await.open);
}

/// Counts how many times the backend was actually invoked.
struct CountingBackend {
    calls: Arc<AtomicUsize>,
}

impl ModelBackend for CountingBackend {
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let rows = batch.texts.iter().map(|_| vec![0.0]).collect();
        Ok(BatchOutput::new(rows))
    }
}

#[tokio::test]
async fn embed_with_deadline_in_past_short_circuits() {
    let calls = Arc::new(AtomicUsize::new(0));
    let backend = CountingBackend { calls: calls.clone() };
    let cfg = DispatcherConfig {
        coalesce_window_ms: 5,
        ..Default::default()
    };
    let disp = Dispatcher::new(cfg, backend);
    // Deadline already in the past at submission.
    let past = std::time::Instant::now() - Duration::from_secs(1);
    let err = disp
        .embed_with_deadline("hi".into(), past)
        .await
        .expect_err("past deadline must error");
    assert!(
        format!("{err}").contains("job deadline exceeded"),
        "got: {err}"
    );
    // Backend must not have been touched.
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn backpressure_rejects_when_queue_full() {
    // streams=1 -> channel cap = 4. One slot, backend hangs, so nothing drains.
    let cfg = DispatcherConfig {
        streams: 1,
        coalesce_window_ms: 5,
        ..Default::default()
    };
    let disp = Arc::new(Dispatcher::new(cfg, HangBackend));

    // Spawn many concurrent embed calls; the bounded channel (cap 4) must
    // reject some of them rather than buffering unboundedly.
    let mut handles = Vec::new();
    for i in 0..50 {
        let d = disp.clone();
        handles.push(tokio::spawn(async move {
            tokio::time::timeout(Duration::from_millis(200), d.embed(format!("t{i}")))
                .await
        }));
    }
    let mut rejected = 0;
    for h in handles {
        match h.await.unwrap() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) if format!("{e}").contains("backpressure") => {
                assert!(format!("{e}").contains("capacity 4"), "{e}");
                rejected += 1
            }
            Ok(Err(_)) => {}
            Err(_elapsed) => {}
        }
    }
    assert!(rejected > 0, "expected some jobs rejected by backpressure");
}

/// Fails with InputTooLong naming the text "LONG" by its batch index.
struct OneLongBackend;
impl ModelBackend for OneLongBackend {
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        if let Some(i) = batch.texts.iter().position(|t| t == "LONG") {
            return Err(M3Error::InputTooLong { tokens: 9000, n_ctx: 8192, index: Some(i) });
        }
        Ok(BatchOutput::new(batch.texts.iter().map(|_| vec![1.0]).collect()))
    }
}

/// One caller's over-long text must not fail the other callers coalesced
/// into the same batch.
#[tokio::test]
async fn an_over_long_text_fails_only_its_own_caller() {
    let cfg = DispatcherConfig { coalesce_window_ms: 30, ..Default::default() };
    let disp = Arc::new(Dispatcher::new(cfg, OneLongBackend));
    let mut handles = Vec::new();
    for t in ["a", "LONG", "b"] {
        let d = disp.clone();
        handles.push(tokio::spawn(async move { (t, d.embed(t.to_string()).await) }));
    }
    for h in handles {
        let (t, r) = h.await.unwrap();
        if t == "LONG" {
            let e = r.unwrap_err();
            assert!(matches!(e, M3Error::InputTooLong { index: Some(0), .. }), "{e}");
        } else {
            assert!(r.is_ok(), "{t} failed: {:?}", r.err().map(|e| e.to_string()));
        }
    }
}

/// queue_depth counts jobs waiting for a stream slot even while the scheduler
/// is busy (a try_lock reported 0 exactly then).
#[tokio::test]
async fn queue_depth_counts_waiting_jobs() {
    let cfg = DispatcherConfig { streams: 1, coalesce_window_ms: 1, max_batch_tokens: 1, ..Default::default() };
    let disp = Arc::new(Dispatcher::new(cfg, HangBackend));
    for i in 0..3 {
        let d = disp.clone();
        tokio::spawn(async move { d.embed(format!("t{i}")).await });
    }
    tokio::time::sleep(Duration::from_millis(150)).await;
    let s = disp.stats();
    assert_eq!((s.in_flight, s.queue_depth), (1, 2), "{s:?}");
}

/// A cancelled caller must not leave in_flight permanently raised.
#[tokio::test]
async fn in_flight_drops_when_the_caller_is_cancelled() {
    let disp = Dispatcher::new(DispatcherConfig::default(), HangBackend);
    let r = tokio::time::timeout(
        Duration::from_millis(50),
        disp.embed_batch(vec!["a".into(), "b".into()]),
    )
    .await;
    assert!(r.is_err(), "the hang backend must time out");
    assert_eq!(disp.stats().in_flight, 0);
}

/// Returns one row fewer than it was given.
struct ShortBackend;
impl ModelBackend for ShortBackend {
    async fn run(&self, batch: Batch) -> Result<BatchOutput> {
        Ok(BatchOutput::new(vec![vec![0.0]; batch.texts.len() - 1]))
    }
}

#[tokio::test]
async fn a_short_reply_is_an_error_with_both_counts_on_the_direct_path() {
    let disp = Dispatcher::new(DispatcherConfig::default(), ShortBackend);
    let e = disp.embed_batch(vec!["a".into(), "b".into(), "c".into()]).await.unwrap_err();
    assert!(e.to_string().contains("2 embedding rows for 3 input texts"), "{e}");
}
