//! Generic model-backend coalescer (Phase 3b/3c).
//!
//! Owns scheduling, length bucketing, coalescing windows, and backpressure.
//! Backends (llama.cpp embed, ONNX NER) implement `ModelBackend`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
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

// Encoded breaker states for the lock-free fast path (Fix #28).
const BREAKER_CLOSED: u8 = 0;
const BREAKER_OPEN: u8 = 1;
const BREAKER_HALF_OPEN: u8 = 2;

/// N consecutive failures open the breaker; after `open_secs` a single
/// half-open probe is allowed, and a success closes it again.
pub struct CircuitBreaker {
    cfg: BreakerCfg,
    state: Mutex<(BreakerState, usize, Option<Instant>)>,
    /// Fix #28 — lock-free state for the Closed-fast-path.
    state_fast: AtomicU8,
    /// Fix #28 — debug-only counter incremented when `check()` returns via
    /// the lock-free path. Compiled out in release.
    #[cfg(test)]
    fast_path_hits: AtomicUsize,
}

impl CircuitBreaker {
    pub fn new(cfg: BreakerCfg) -> Self {
        Self {
            cfg,
            state: Mutex::new((BreakerState::Closed, 0, None)),
            state_fast: AtomicU8::new(BREAKER_CLOSED),
            #[cfg(test)]
            fast_path_hits: AtomicUsize::new(0),
        }
    }

    /// Returns Err fast when the breaker is open and the cooldown has not elapsed.
    ///
    /// Fix #28 — Closed-state check is lock-free. When the breaker is Closed
    /// (the overwhelmingly common case), this loads a single `AtomicU8` with
    /// Acquire ordering and returns Ok without touching the mutex. Any other
    /// state falls through to the original mutex-guarded transition logic.
    pub async fn check(&self) -> Result<()> {
        if self.state_fast.load(Ordering::Acquire) == BREAKER_CLOSED {
            #[cfg(test)]
            self.fast_path_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let mut g = self.state.lock().await;
        match g.0 {
            BreakerState::Closed | BreakerState::HalfOpen => Ok(()),
            BreakerState::Open => {
                let elapsed = g.2.map(|t| t.elapsed()).unwrap_or_default();
                if elapsed >= Duration::from_secs(self.cfg.open_secs) {
                    g.0 = BreakerState::HalfOpen;
                    self.state_fast.store(BREAKER_HALF_OPEN, Ordering::Release);
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
        self.state_fast.store(BREAKER_CLOSED, Ordering::Release);
    }

    pub async fn record_failure(&self) {
        let mut g = self.state.lock().await;
        g.1 += 1;
        if g.1 >= self.cfg.failure_threshold || g.0 == BreakerState::HalfOpen {
            g.0 = BreakerState::Open;
            g.2 = Some(Instant::now());
            self.state_fast.store(BREAKER_OPEN, Ordering::Release);
        }
    }
}

/// One pending single-shot job: its text plus a channel to deliver the vector.
///
/// `deadline` (Fix #27) lets callers tag a job with a wall-clock cutoff. The
/// scheduler drops past-deadline jobs before dispatching them to the backend,
/// surfacing `M3Error::Backend("job deadline exceeded")` to the caller. None
/// = no deadline (existing `embed()` API path).
struct Job {
    text: String,
    token_len: usize,
    reply: oneshot::Sender<Result<Vec<f32>>>,
    deadline: Option<Instant>,
}

/// Buckets pending jobs by token length into the nearest configured bucket.
pub struct LengthBucketQueue {
    /// Sorted bucket ceilings; index i collects jobs up to `buckets[i]` tokens.
    buckets: Vec<usize>,
    pending: Vec<VecDeque<Job>>,
    /// Fix #9: running total of pending tokens. Read by the scheduler without
    /// taking the queue mutex (via `total_tokens_handle()`). Updated under the
    /// mutex on push/drain so the invariant `sum(bucket.token_len) == total`
    /// holds.
    total_tokens: Arc<AtomicUsize>,
}

impl LengthBucketQueue {
    pub fn new(mut buckets: Vec<usize>) -> Self {
        if buckets.is_empty() {
            buckets.push(usize::MAX);
        }
        buckets.sort_unstable();
        let n = buckets.len();
        Self {
            buckets,
            pending: (0..n).map(|_| VecDeque::new()).collect(),
            total_tokens: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Fix #9: clone the shared atomic so the scheduler can poll queue depth
    /// (in tokens) without taking the queue mutex.
    fn total_tokens_handle(&self) -> Arc<AtomicUsize> {
        self.total_tokens.clone()
    }

    fn bucket_index(&self, token_len: usize) -> usize {
        // Fix #8: binary search for the first ceiling >= token_len. Falls back
        // to the last bucket when token_len exceeds every ceiling, matching
        // the original linear behavior.
        self.buckets
            .partition_point(|&c| c < token_len)
            .min(self.buckets.len() - 1)
    }

    fn push(&mut self, job: Job) {
        let idx = self.bucket_index(job.token_len);
        self.total_tokens.fetch_add(job.token_len, Ordering::Relaxed);
        self.pending[idx].push_back(job);
    }

    fn depth(&self) -> usize {
        self.pending.iter().map(|b| b.len()).sum()
    }

    /// Drains every bucket that has work, capping each flushed batch at
    /// `max_batch_tokens`. Returns one drained batch per call (round-robin
    /// over buckets) so the scheduler can flush incrementally.
    ///
    /// Fix #7: uses `VecDeque` internally, so the drain is a single
    /// `drain(..take_count)` call — no O(n) shift per job.
    fn drain_one(&mut self, max_batch_tokens: usize) -> Option<Vec<Job>> {
        for bucket in &mut self.pending {
            if bucket.is_empty() {
                continue;
            }
            // First pass: scan to decide how many jobs fit under the cap.
            let mut take_count = 0usize;
            let mut tokens = 0usize;
            for job in bucket.iter() {
                if take_count > 0 && tokens + job.token_len > max_batch_tokens {
                    break;
                }
                tokens += job.token_len;
                take_count += 1;
            }
            // Second pass: one drain call, one output allocation.
            let out: Vec<Job> = bucket.drain(..take_count).collect();
            self.total_tokens.fetch_sub(tokens, Ordering::Relaxed);
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
        let queue_inner = LengthBucketQueue::new(cfg.length_buckets.clone());
        let total_tokens = queue_inner.total_tokens_handle();
        let queue = Arc::new(Mutex::new(queue_inner));
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
            cfg, backend, breaker, queue, total_tokens, slots, in_flight, rx,
        ));
        d
    }

    /// Single-shot embed. Joins the next coalescing window. Returns an error
    /// immediately if the bounded queue is full (backpressure) or the breaker
    /// is open.
    pub async fn embed(&self, text: String) -> Result<Vec<f32>> {
        self.embed_inner(text, None, None).await
    }

    /// Fix #10 — Single-shot embed with a caller-supplied token count.
    ///
    /// `estimate_tokens` uses `len/4`, which is off by 3–5× for BGE-M3 on
    /// CJK / multilingual text and produces wrong bucket placement. Callers
    /// that have already tokenized (or maintain a better char→token ratio)
    /// should pass the real count here. The dispatcher uses it solely for
    /// bucket selection and the running-total cap; the backend still sees the
    /// raw text.
    pub async fn embed_with_token_count(
        &self,
        text: String,
        token_len: usize,
    ) -> Result<Vec<f32>> {
        self.embed_inner(text, None, Some(token_len)).await
    }

    /// Like `embed()` but with a wall-clock deadline (Fix #27). If the
    /// dispatcher pulls this job off the queue past `deadline`, it replies
    /// with `M3Error::Backend("job deadline exceeded")` without invoking the
    /// backend. A deadline already in the past at submission time also
    /// short-circuits before the queue. `embed_batch` does not yet honour
    /// deadlines — kept narrow on purpose.
    pub async fn embed_with_deadline(&self, text: String, deadline: Instant) -> Result<Vec<f32>> {
        if Instant::now() >= deadline {
            return Err(M3Error::Backend("job deadline exceeded".into()));
        }
        self.embed_inner(text, Some(deadline), None).await
    }

    async fn embed_inner(
        &self,
        text: String,
        deadline: Option<Instant>,
        token_len_hint: Option<usize>,
    ) -> Result<Vec<f32>> {
        self.breaker.check().await?;
        let token_len = token_len_hint.unwrap_or_else(|| estimate_tokens(&text));
        let (reply, rx) = oneshot::channel();
        let job = Job { text, token_len, reply, deadline };
        self.tx
            .try_send(job)
            .map_err(|_| M3Error::Backend("dispatcher queue full (backpressure)".into()))?;
        rx.await
            .map_err(|_| M3Error::Backend("dispatcher dropped job".into()))?
    }

    /// Bulk embed. Bypasses coalescing — runs the caller's batch directly,
    /// still subject to the slot semaphore and circuit breaker.
    ///
    /// Fix #11 — exception: a single-text "batch" is routed through the
    /// coalescer so concurrent `embed_batch([t1])` + `embed_batch([t2])`
    /// calls from different threads can merge into one backend pass instead
    /// of two. Multi-text batches still bypass to preserve the caller's
    /// explicit batching intent.
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            let mut texts = texts;
            let row = self.embed(texts.pop().unwrap()).await?;
            return Ok(vec![row]);
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

    /// Fix #10 — Bulk embed with caller-supplied token counts per text.
    ///
    /// Same bypass semantics as `embed_batch`, but the dispatcher uses the
    /// caller's per-text token counts for its internal `token_len_hint`
    /// instead of the cheap `estimate_tokens` heuristic. Required:
    /// `texts.len() == token_lens.len()`.
    pub async fn embed_batch_with_token_counts(
        &self,
        texts: Vec<String>,
        token_lens: Vec<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.len() != token_lens.len() {
            return Err(M3Error::Backend(
                "embed_batch_with_token_counts: texts.len() != token_lens.len()".into(),
            ));
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            let mut texts = texts;
            let row = self
                .embed_with_token_count(texts.pop().unwrap(), token_lens[0])
                .await?;
            return Ok(vec![row]);
        }
        self.breaker.check().await?;
        let _permit = self
            .slots
            .acquire()
            .await
            .map_err(|_| M3Error::Backend("dispatcher closed".into()))?;
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let token_len: usize = token_lens.iter().sum();
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

    /// Borrow the shared `Arc<B>` backend handle. The dispatcher already wraps
    /// the backend in an `Arc` internally; this exposes that exact same Arc so
    /// callers (e.g. `m3-core-py`'s `PyEmbeddedEmbedder`) can hold a second
    /// handle to the same instance for direct introspection — no
    /// double-instantiation of the backend.
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
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
    total_tokens: Arc<AtomicUsize>,
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
                            {
                                let mut q = queue.lock().await;
                                q.push(j);
                            }
                            // Fix #9: lock-free depth read — atomic running
                            // total maintained by push/drain.
                            if total_tokens.load(Ordering::Relaxed) >= cfg.max_batch_tokens {
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

            // Fix #27: drop past-deadline jobs before dispatching.
            let now = Instant::now();
            let (jobs, expired): (Vec<Job>, Vec<Job>) = jobs
                .into_iter()
                .partition(|j| j.deadline.is_none_or(|d| now < d));
            for j in expired {
                let _ = j
                    .reply
                    .send(Err(M3Error::Backend("job deadline exceeded".into())));
            }
            if jobs.is_empty() {
                continue;
            }

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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    fn mk_job(token_len: usize) -> Job {
        let (reply, _rx) = oneshot::channel();
        Job { text: String::new(), token_len, reply, deadline: None }
    }

    /// Fix #8 — bucket_index boundaries.
    #[test]
    fn bucket_index_boundaries() {
        let q = LengthBucketQueue::new(vec![64, 256, 1024, 4096]);
        // token_len = 0 -> first bucket
        assert_eq!(q.bucket_index(0), 0);
        // exactly a ceiling -> that bucket
        assert_eq!(q.bucket_index(64), 0);
        assert_eq!(q.bucket_index(256), 1);
        assert_eq!(q.bucket_index(1024), 2);
        assert_eq!(q.bucket_index(4096), 3);
        // one over a ceiling -> next bucket
        assert_eq!(q.bucket_index(65), 1);
        assert_eq!(q.bucket_index(257), 2);
        assert_eq!(q.bucket_index(1025), 3);
        // larger than max -> last bucket (saturating)
        assert_eq!(q.bucket_index(1_000_000), 3);
    }

    /// Fix #7 — FIFO drain respects token cap and uses VecDeque internally.
    #[test]
    fn drain_one_fifo_and_token_cap() {
        // Single-bucket queue so all jobs share one VecDeque and we exercise
        // the FIFO + cap path directly.
        let mut q = LengthBucketQueue::new(vec![usize::MAX]);
        let lens = [50usize, 100, 80, 120, 60, 90, 70, 110, 40, 130];
        for &l in &lens {
            q.push(mk_job(l));
        }
        // Fix #9 — running total invariant.
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), lens.iter().sum::<usize>());

        // Drain with cap = 300. Greedy FIFO: 50+100+80 = 230; next is 120 ->
        // 350 > 300 so stop. First job is always taken even if it exceeds cap
        // (matches original behavior: `out.is_empty()` guard).
        let batch = q.drain_one(300).expect("non-empty batch");
        let drained_lens: Vec<usize> = batch.iter().map(|j| j.token_len).collect();
        assert_eq!(drained_lens, vec![50, 100, 80], "FIFO + token cap");
        let remaining: usize = lens.iter().sum::<usize>() - 230;
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), remaining);

        // Subsequent drain continues from where we left off (FIFO preserved).
        let batch2 = q.drain_one(300).expect("non-empty batch");
        let drained2: Vec<usize> = batch2.iter().map(|j| j.token_len).collect();
        // 120+60+90 = 270, next is 70 -> 340 > 300 -> stop.
        assert_eq!(drained2, vec![120, 60, 90]);
    }

    /// Fix #28 — closed breaker `check()` takes the lock-free path; an open
    /// breaker falls through to the mutex.
    #[tokio::test]
    async fn breaker_check_uses_lock_free_path_when_closed() {
        let cb = CircuitBreaker::new(BreakerCfg { failure_threshold: 2, open_secs: 60 });
        // Closed: 1000 checks should all hit the fast path.
        for _ in 0..1000 {
            cb.check().await.unwrap();
        }
        assert_eq!(cb.fast_path_hits.load(Ordering::Relaxed), 1000);

        // Trip the breaker. record_failure() must publish Open to state_fast.
        cb.record_failure().await;
        cb.record_failure().await;
        // Now the fast path should NOT be taken and check() should return Err.
        let before = cb.fast_path_hits.load(Ordering::Relaxed);
        assert!(cb.check().await.is_err());
        assert_eq!(
            cb.fast_path_hits.load(Ordering::Relaxed),
            before,
            "open-state check must not take fast path"
        );

        // record_success() republishes Closed -> fast path resumes.
        cb.record_success().await;
        cb.check().await.unwrap();
        assert_eq!(cb.fast_path_hits.load(Ordering::Relaxed), before + 1);
    }

    /// Fix #9 — running total stays consistent across mixed push/drain.
    #[test]
    fn total_tokens_running_invariant() {
        let mut q = LengthBucketQueue::new(vec![64, 256, 1024]);
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), 0);
        q.push(mk_job(30));
        q.push(mk_job(200));
        q.push(mk_job(500));
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), 730);
        let _ = q.drain_one(usize::MAX);
        // The first non-empty bucket (idx 0, len 30) was fully drained.
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), 700);
        let _ = q.drain_one(usize::MAX);
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), 500);
        let _ = q.drain_one(usize::MAX);
        assert_eq!(q.total_tokens.load(Ordering::Relaxed), 0);
        assert!(q.drain_one(usize::MAX).is_none());
    }
}
