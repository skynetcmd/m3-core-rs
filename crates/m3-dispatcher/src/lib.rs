//! Generic model-backend coalescer (Phase 3b/3c).
//!
//! Owns scheduling, length bucketing, coalescing windows, and backpressure.
//! Backends (llama.cpp embed, ONNX NER) implement `ModelBackend`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use m3_error::{M3Error, Result};
use tokio::sync::{mpsc, oneshot, Mutex, Semaphore};

/// A unit of work flowing through the dispatcher: a set of texts to run
/// through the backend in one forward pass.
#[derive(Debug, Clone)]
pub struct Batch {
    pub texts: Vec<String>,
    /// Estimated total token length across all texts; drives bucket selection.
    pub token_len_hint: usize,
}

impl Batch {
    pub fn new(texts: Vec<String>, token_len_hint: usize) -> Self {
        Self { texts, token_len_hint }
    }
}

/// The result of running a `Batch` through a backend.
///
/// `rows` is one `Vec<f32>` per input text. Generic enough for both embedding
/// vectors (one dense vector per text) and NER span-score rows (a flattened
/// span-score tensor per text); the consumer interprets the shape.
#[derive(Debug, Clone)]
pub struct BatchOutput {
    pub rows: Vec<Vec<f32>>,
}

impl BatchOutput {
    pub fn new(rows: Vec<Vec<f32>>) -> Self {
        Self { rows }
    }
}

/// Backend abstraction shared by all dispatcher consumers.
pub trait ModelBackend {
    fn run(
        &self,
        batch: Batch,
    ) -> impl std::future::Future<Output = Result<BatchOutput>> + Send;
}

/// Circuit-breaker tuning.
#[derive(Debug, Clone)]
pub struct BreakerCfg {
    /// Consecutive failures before the breaker opens.
    pub failure_threshold: usize,
    /// How long the breaker stays open before a half-open probe.
    pub open_secs: u64,
}

impl Default for BreakerCfg {
    fn default() -> Self {
        Self { failure_threshold: 5, open_secs: 10 }
    }
}

/// Typed dispatcher configuration. Generic crates never read env vars;
/// `m3-core-py` builds this from `M3_*` vars.
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    pub streams: usize,
    pub coalesce_window_ms: u64,
    pub max_batch_tokens: usize,
    pub length_buckets: Vec<usize>,
    pub circuit_breaker: BreakerCfg,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            streams: 8,
            coalesce_window_ms: 3,
            max_batch_tokens: 2048,
            length_buckets: vec![64, 256, 1024, 4096],
            circuit_breaker: BreakerCfg::default(),
        }
    }
}

/// Point-in-time dispatcher metrics.
#[derive(Debug, Clone, Default)]
pub struct DispatcherStats {
    pub in_flight: usize,
    pub queue_depth: usize,
    /// TODO: latency histogram not yet wired; reported as 0.
    pub p50_ms: f64,
    /// TODO: latency histogram not yet wired; reported as 0.
    pub p99_ms: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// N consecutive failures open the breaker; after `open_secs` a single
/// half-open probe is allowed, and a success closes it again.
pub struct CircuitBreaker {
    cfg: BreakerCfg,
    state: Mutex<(BreakerState, usize, Option<Instant>)>,
}

impl CircuitBreaker {
    pub fn new(cfg: BreakerCfg) -> Self {
        Self {
            cfg,
            state: Mutex::new((BreakerState::Closed, 0, None)),
        }
    }

    /// Returns Err fast when the breaker is open and the cooldown has not elapsed.
    pub async fn check(&self) -> Result<()> {
        let mut g = self.state.lock().await;
        match g.0 {
            BreakerState::Closed | BreakerState::HalfOpen => Ok(()),
            BreakerState::Open => {
                let elapsed = g.2.map(|t| t.elapsed()).unwrap_or_default();
                if elapsed >= Duration::from_secs(self.cfg.open_secs) {
                    g.0 = BreakerState::HalfOpen;
                    Ok(())
                } else {
                    Err(M3Error::Backend("circuit breaker open".into()))
                }
            }
        }
    }

    pub async fn record_success(&self) {
        let mut g = self.state.lock().await;
        *g = (BreakerState::Closed, 0, None);
    }

    pub async fn record_failure(&self) {
        let mut g = self.state.lock().await;
        g.1 += 1;
        if g.1 >= self.cfg.failure_threshold || g.0 == BreakerState::HalfOpen {
            g.0 = BreakerState::Open;
            g.2 = Some(Instant::now());
        }
    }
}

/// One pending single-shot job: its text plus a channel to deliver the vector.
struct Job {
    text: String,
    token_len: usize,
    reply: oneshot::Sender<Result<Vec<f32>>>,
}

/// Buckets pending jobs by token length into the nearest configured bucket.
pub struct LengthBucketQueue {
    /// Sorted bucket ceilings; index i collects jobs up to `buckets[i]` tokens.
    buckets: Vec<usize>,
    pending: Vec<Vec<Job>>,
}

impl LengthBucketQueue {
    pub fn new(mut buckets: Vec<usize>) -> Self {
        if buckets.is_empty() {
            buckets.push(usize::MAX);
        }
        buckets.sort_unstable();
        let n = buckets.len();
        Self { buckets, pending: (0..n).map(|_| Vec::new()).collect() }
    }

    fn bucket_index(&self, token_len: usize) -> usize {
        self.buckets
            .iter()
            .position(|&ceil| token_len <= ceil)
            .unwrap_or(self.buckets.len() - 1)
    }

    fn push(&mut self, job: Job) {
        let idx = self.bucket_index(job.token_len);
        self.pending[idx].push(job);
    }

    fn depth(&self) -> usize {
        self.pending.iter().map(|b| b.len()).sum()
    }

    /// Drains every bucket that has work, capping each flushed batch at
    /// `max_batch_tokens`. Returns one drained batch per call (round-robin
    /// over buckets) so the scheduler can flush incrementally.
    fn drain_one(&mut self, max_batch_tokens: usize) -> Option<Vec<Job>> {
        for bucket in &mut self.pending {
            if bucket.is_empty() {
                continue;
            }
            let mut out = Vec::new();
            let mut tokens = 0usize;
            while let Some(job) = bucket.first() {
                if !out.is_empty() && tokens + job.token_len > max_batch_tokens {
                    break;
                }
                tokens += job.token_len;
                out.push(bucket.remove(0));
            }
            return Some(out);
        }
        None
    }
}

/// Generic coalescing dispatcher in front of a `ModelBackend`.
pub struct Dispatcher<B: ModelBackend> {
    #[allow(dead_code)]
    cfg: DispatcherConfig,
    backend: Arc<B>,
    breaker: Arc<CircuitBreaker>,
    queue: Arc<Mutex<LengthBucketQueue>>,
    slots: Arc<Semaphore>,
    in_flight: Arc<AtomicUsize>,
    /// Bounded channel: backpressure point for `embed`. Cap = 4 x streams.
    tx: mpsc::Sender<Job>,
}

impl<B: ModelBackend + Send + Sync + 'static> Dispatcher<B> {
    pub fn new(cfg: DispatcherConfig, backend: B) -> Self {
        let backend = Arc::new(backend);
        let breaker = Arc::new(CircuitBreaker::new(cfg.circuit_breaker.clone()));
        let queue = Arc::new(Mutex::new(LengthBucketQueue::new(cfg.length_buckets.clone())));
        let slots = Arc::new(Semaphore::new(cfg.streams.max(1)));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let cap = (cfg.streams.max(1)) * 4;
        let (tx, rx) = mpsc::channel::<Job>(cap);

        let d = Self {
            cfg: cfg.clone(),
            backend: backend.clone(),
            breaker: breaker.clone(),
            queue: queue.clone(),
            slots: slots.clone(),
            in_flight: in_flight.clone(),
            tx,
        };
        tokio::spawn(scheduler_loop(
            cfg, backend, breaker, queue, slots, in_flight, rx,
        ));
        d
    }

    /// Single-shot embed. Joins the next coalescing window. Returns an error
    /// immediately if the bounded queue is full (backpressure) or the breaker
    /// is open.
    pub async fn embed(&self, text: String) -> Result<Vec<f32>> {
        self.breaker.check().await?;
        let token_len = estimate_tokens(&text);
        let (reply, rx) = oneshot::channel();
        let job = Job { text, token_len, reply };
        self.tx
            .try_send(job)
            .map_err(|_| M3Error::Backend("dispatcher queue full (backpressure)".into()))?;
        rx.await
            .map_err(|_| M3Error::Backend("dispatcher dropped job".into()))?
    }

    /// Bulk embed. Bypasses coalescing — runs the caller's batch directly,
    /// still subject to the slot semaphore and circuit breaker.
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.breaker.check().await?;
        let _permit = self
            .slots
            .acquire()
            .await
            .map_err(|_| M3Error::Backend("dispatcher closed".into()))?;
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let token_len = texts.iter().map(|t| estimate_tokens(t)).sum();
        let res = self.backend.run(Batch::new(texts, token_len)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        match res {
            Ok(out) => {
                self.breaker.record_success().await;
                Ok(out.rows)
            }
            Err(e) => {
                self.breaker.record_failure().await;
                Err(e)
            }
        }
    }

    pub fn stats(&self) -> DispatcherStats {
        let queue_depth = self.queue.try_lock().map(|q| q.depth()).unwrap_or(0);
        DispatcherStats {
            in_flight: self.in_flight.load(Ordering::SeqCst),
            queue_depth,
            p50_ms: 0.0,
            p99_ms: 0.0,
        }
    }
}

/// Rough token-length estimate; the dispatcher only needs bucket granularity,
/// not exact tokenization. ~4 chars/token, minimum 1.
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

#[allow(clippy::too_many_arguments)]
async fn scheduler_loop<B: ModelBackend + Send + Sync + 'static>(
    cfg: DispatcherConfig,
    backend: Arc<B>,
    breaker: Arc<CircuitBreaker>,
    queue: Arc<Mutex<LengthBucketQueue>>,
    slots: Arc<Semaphore>,
    in_flight: Arc<AtomicUsize>,
    mut rx: mpsc::Receiver<Job>,
) {
    let window = Duration::from_millis(cfg.coalesce_window_ms.max(1));
    loop {
        // Block until at least one job arrives, then open a coalescing window.
        let first = match rx.recv().await {
            Some(j) => j,
            None => break,
        };
        {
            let mut q = queue.lock().await;
            q.push(first);
        }
        let deadline = tokio::time::sleep(window);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => break,
                maybe = rx.recv() => {
                    match maybe {
                        Some(j) => {
                            let mut q = queue.lock().await;
                            q.push(j);
                            if q.depth() >= cfg.max_batch_tokens {
                                break;
                            }
                        }
                        None => break,
                    }
                }
            }
        }

        // Flush every ready batch.
        loop {
            let drained = {
                let mut q = queue.lock().await;
                q.drain_one(cfg.max_batch_tokens)
            };
            let jobs = match drained {
                Some(j) if !j.is_empty() => j,
                _ => break,
            };

            if let Err(e) = breaker.check().await {
                for j in jobs {
                    let _ = j.reply.send(Err(M3Error::Backend(format!("{e}"))));
                }
                continue;
            }

            let permit = match slots.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => return,
            };
            let backend = backend.clone();
            let breaker = breaker.clone();
            let in_flight = in_flight.clone();
            tokio::spawn(async move {
                let _permit = permit;
                in_flight.fetch_add(1, Ordering::SeqCst);
                let (texts, replies): (Vec<String>, Vec<_>) =
                    jobs.into_iter().map(|j| (j.text, j.reply)).unzip();
                let token_len = texts.iter().map(|t| estimate_tokens(t)).sum();
                let res = backend.run(Batch::new(texts, token_len)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                match res {
                    Ok(out) => {
                        breaker.record_success().await;
                        if out.rows.len() == replies.len() {
                            for (reply, row) in replies.into_iter().zip(out.rows) {
                                let _ = reply.send(Ok(row));
                            }
                        } else {
                            for reply in replies {
                                let _ = reply.send(Err(M3Error::Backend(
                                    "backend returned wrong row count".into(),
                                )));
                            }
                        }
                    }
                    Err(e) => {
                        breaker.record_failure().await;
                        let msg = format!("{e}");
                        for reply in replies {
                            let _ = reply.send(Err(M3Error::Backend(msg.clone())));
                        }
                    }
                }
            });
        }
    }
}
