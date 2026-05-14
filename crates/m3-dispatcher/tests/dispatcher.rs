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
            Ok(Err(e)) if format!("{e}").contains("backpressure") => rejected += 1,
            Ok(Err(_)) => {}
            Err(_elapsed) => {}
        }
    }
    assert!(rejected > 0, "expected some jobs rejected by backpressure");
}
