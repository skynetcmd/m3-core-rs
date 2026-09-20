//! Generic model-backend coalescer (Phase 3b/3c).
//!
//! Owns scheduling, length bucketing, coalescing windows, and backpressure.
//! Backends (llama.cpp embed, ONNX NER) implement `ModelBackend`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
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
///
/// `p50_ms`/`p99_ms` are computed over the most recent
/// [`LATENCY_WINDOW`] completed batches (see [`LatencyRing`]). They are
/// `None` until at least one batch has completed — deliberately NOT `0.0`,
/// because a constant zero is indistinguishable from a genuinely fast server
/// and reads as observability while providing none.
#[derive(Debug, Clone, Default)]
pub struct DispatcherStats {
    pub in_flight: usize,
    pub queue_depth: usize,
    /// p50 over the recent window; `None` when no batch has completed yet.
    pub p50_ms: Option<f64>,
    /// p99 over the recent window; `None` when no batch has completed yet.
    pub p99_ms: Option<f64>,
}

/// Number of recent batch latencies retained for percentiles.
///
/// A fixed ring, not an unbounded histogram: percentiles over a RECENT window
/// are what an operator watching a live server needs, and a bounded ring costs
/// a fixed 8 KiB with no allocation on the hot path. 1024 samples covers
/// several minutes at realistic embed rates while still reacting within
/// seconds when latency moves.
pub const LATENCY_WINDOW: usize = 1024;

/// Lock-free ring of recent batch latencies, in microseconds.
///
/// Written on every completed batch from the dispatcher's hot path and read by
/// [`Dispatcher::stats`], which must never block a caller that is only asking
/// for metrics. Hence atomics rather than a `Mutex<Vec<_>>`: a metrics endpoint
/// must not be able to contend with — or stall — request serving.
///
/// Microseconds (`u64`) rather than `f64` so samples can live in atomics at
/// all; the conversion to milliseconds happens once, at read time.
#[derive(Debug)]
struct LatencyRing {
    slots: Vec<AtomicU64>,
    /// Total samples ever recorded. Doubles as the write cursor (modulo len)
    /// and as the "have we seen anything yet" flag.
    count: AtomicUsize,
}

impl LatencyRing {
    fn new(n: usize) -> Self {
        Self {
            slots: (0..n.max(1)).map(|_| AtomicU64::new(0)).collect(),
            count: AtomicUsize::new(0),
        }
    }

    /// Record one completed batch. Wait-free: a single fetch_add plus a store.
    fn record(&self, d: Duration) {
        let idx = self.count.fetch_add(1, Ordering::Relaxed) % self.slots.len();
        // saturating: a pathological multi-thousand-second batch must clamp,
        // never wrap into a small number that would silently flatter p99.
        let us = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
        self.slots[idx].store(us, Ordering::Relaxed);
    }

    /// (p50, p99) in milliseconds over the recent window, or `None` if empty.
    ///
    /// Snapshots only the slots actually written, so the first reads after
    /// startup are not diluted by unwritten zeros — that dilution is exactly
    /// how a percentile ends up reporting a reassuring number it has not
    /// earned.
    fn percentiles(&self) -> Option<(f64, f64)> {
        let n = self.count.load(Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        let filled = n.min(self.slots.len());
        let mut v: Vec<u64> = self.slots[..filled]
            .iter()
            .map(|s| s.load(Ordering::Relaxed))
            .collect();
        v.sort_unstable();
        Some((pct(&v, 0.50), pct(&v, 0.99)))
    }
}

/// Nearest-rank percentile of a sorted microsecond slice, returned as ms.
///
/// Nearest-rank (not interpolated): with a window this small an interpolated
/// p99 invents a value between two real samples, and for a tail metric the
/// honest answer is an observation that actually happened.
fn pct(sorted_us: &[u64], q: f64) -> f64 {
    debug_assert!(!sorted_us.is_empty());
    let rank = (q * sorted_us.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted_us.len() - 1);
    sorted_us[idx] as f64 / 1000.0
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
    /// Recent batch latencies backing `stats().p50_ms` / `p99_ms`.
    latency: Arc<LatencyRing>,
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
        let latency = Arc::new(LatencyRing::new(LATENCY_WINDOW));
        let cap = (cfg.streams.max(1)) * 4;
        let (tx, rx) = mpsc::channel::<Job>(cap);

        let d = Self {
            cfg: cfg.clone(),
            backend: backend.clone(),
            breaker: breaker.clone(),
            queue: queue.clone(),
            slots: slots.clone(),
            in_flight: in_flight.clone(),
            latency: latency.clone(),
            tx,
        };
        tokio::spawn(scheduler_loop(
            cfg, backend, breaker, queue, total_tokens, slots, in_flight, latency, rx,
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
        // Time the backend call only — the queue/semaphore wait before this
        // point is already visible as queue_depth, and folding it in here
        // would make a backlog look like a slow model.
        let t0 = Instant::now();
        let res = self.backend.run(Batch::new(texts, token_len)).await;
        self.latency.record(t0.elapsed());
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
        // See the sibling site above: backend call only, not the queue wait.
        let t0 = Instant::now();
        let res = self.backend.run(Batch::new(texts, token_len)).await;
        self.latency.record(t0.elapsed());
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
        let (p50, p99) = match self.latency.percentiles() {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };
        DispatcherStats {
            in_flight: self.in_flight.load(Ordering::SeqCst),
            queue_depth,
            p50_ms: p50,
            p99_ms: p99,
        }
    }
}

/// Special tokens BGE-M3 wraps every sequence in (BOS + EOS; `AddBos::Always`
/// in m3-embed-llamacpp). Measured exactly 2, independent of input length —
/// even the empty string encodes to 2, not 0. They occupy n_ctx like any other
/// token, so an estimate that omits them under-counts by exactly this much.
pub const SPECIAL_TOKENS: usize = 2;

/// Conservative token-length estimate that never UNDER-counts (BGE-M3).
///
/// `max(bytes/3, chars) + SPECIAL_TOKENS`.
///
/// The previous form was `(text.len() / 4).max(1)` — bytes/4, which is the
/// ENGLISH ratio. Measured against the real BGE-M3 tokenizer, density spans 4x
/// across content one corpus routinely mixes:
///
/// | content        | chars/token | bytes/token |
/// |----------------|-------------|-------------|
/// | English prose  | 4.18        | 4.18        |
/// | Python code    | 2.96        | 3.15        |
/// | logs / traces  | 2.26        | 2.26        |
/// | Chinese        | 1.66        | 4.88        |
/// | JSON           | 1.63        | 1.63        |
/// | UUID lists     | 1.61        | 1.61        |
/// | base64         | 1.00        | 1.00        |
///
/// So bytes/4 under-counted by up to 4x, which mis-placed jobs in the
/// dispatcher's `LengthBucketQueue` and mis-fed its running-total cap: a batch
/// of JSON or base64 was scheduled as though it were a quarter of its true
/// size. (The `embed_with_token_count` escape hatch below exists precisely
/// because someone already diagnosed this — see its doc comment — but nothing
/// upstream could supply a real count.)
///
/// The `chars` term is load-bearing: BGE-M3 uses SentencePiece, which cannot
/// emit more CONTENT tokens than there are characters, so `chars` is a hard
/// upper bound for ANY input rather than for sampled ones. Formulas measured
/// and REJECTED: `max(bytes/4, chars/1.1)` (worst ratio 0.91),
/// `max(bytes/3, chars/1.05)` (0.95), `max(bytes/3, chars)` without the special
/// tokens (0.999 — under-counts base64 by exactly 2).
///
/// MUST stay byte-identical to `estimate_tokens` in
/// `m3-memory/bin/memory/tokens.py`; the Python fallback and this path are two
/// implementations of one contract, and `tests/test_token_budget.py` plus the
/// Rust tests below pin the same cases on both sides.
///
/// ⚠ The `chars` ceiling is a property of SentencePiece, NOT of tokenizers in
/// general — a byte-level BPE model can exceed 1 token/char. A model swap must
/// revisit this bound.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return SPECIAL_TOKENS;
    }
    // `text.len()` is BYTES in Rust; `chars().count()` is the character count.
    let by_bytes = text.len() / 3;
    let by_chars = text.chars().count();
    by_bytes.max(by_chars) + SPECIAL_TOKENS
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
    latency: Arc<LatencyRing>,
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
            let latency = latency.clone();
            tokio::spawn(async move {
                let _permit = permit;
                in_flight.fetch_add(1, Ordering::SeqCst);
                let (texts, replies): (Vec<String>, Vec<_>) =
                    jobs.into_iter().map(|j| (j.text, j.reply)).unzip();
                let token_len = texts.iter().map(|t| estimate_tokens(t)).sum();
                // This is the COALESCED path — the one that normally serves
                // traffic. Missing it here would leave p50/p99 reflecting only
                // the two direct-submit sites, i.e. almost nothing.
                let t0 = Instant::now();
                let res = backend.run(Batch::new(texts, token_len)).await;
                latency.record(t0.elapsed());
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

    // ── estimate_tokens: the contract is "never UNDER-count" ─────────────────
    // Mirrors m3-memory/tests/test_token_budget.py. Both sides are two
    // implementations of ONE contract; drift here silently re-opens the n_ctx
    // overflow these tests exist to stop.

    #[test]
    fn estimate_never_below_chars_plus_frame() {
        // The load-bearing bound: SentencePiece cannot emit more CONTENT tokens
        // than there are characters, plus the BOS/EOS frame.
        let cases = [
            String::new(),
            "a".to_string(),
            "ab".repeat(100),
            "\u{0}".repeat(50),
            "\u{1F389}".repeat(40),
            "\u{6570}\u{636E}\u{5E93}".repeat(30),
        ];
        for s in &cases {
            assert!(
                estimate_tokens(s) >= s.chars().count() + SPECIAL_TOKENS,
                "estimate dropped below chars+frame for a {}-char input",
                s.chars().count()
            );
        }
    }

    #[test]
    fn empty_costs_the_frame_not_zero() {
        // BGE-M3 emits BOS+EOS even for empty input; reporting 0 would be a lie
        // a caller might budget against.
        assert_eq!(estimate_tokens(""), SPECIAL_TOKENS);
    }

    #[test]
    fn byte_term_dominates_for_multibyte_text() {
        // 4-byte astral chars: bytes/3 (133) beats chars (100).
        let s = "\u{1F389}".repeat(100);
        assert_eq!(estimate_tokens(&s), 400 / 3 + SPECIAL_TOKENS);
    }

    #[test]
    fn cjk_is_not_under_counted() {
        // The regression that mattered. 3-byte CJK under the old bytes/4 form
        // scored 0.75 tokens/char; the truth is ~0.6 chars/token, so the old
        // form mis-placed these jobs in the length-bucket queue.
        let s = "\u{6570}\u{636E}\u{5E93}\u{8FDE}\u{63A5}".repeat(100);
        let old_form = s.len() / 4; // the previous implementation
        assert!(
            estimate_tokens(&s) > old_form,
            "new estimate must exceed the old bytes/4 form for CJK"
        );
        assert!(estimate_tokens(&s) >= s.chars().count());
    }

    #[test]
    fn monotonic_in_length() {
        let base = "abc".repeat(50);
        let mut prev = 0usize;
        for n in 0..10 {
            let e = estimate_tokens(&base.repeat(n));
            assert!(e >= prev, "estimate went down as the input grew");
            prev = e;
        }
    }

    #[test]
    fn matches_the_python_seam_formula() {
        // Byte-for-byte parity with memory/tokens.py estimate_tokens().
        let cases = [
            "hello world",
            "\u{6570}\u{636E}\u{5E93}\u{8FDE}\u{63A5}\u{5931}\u{8D25}",
            "\u{1F389}\u{1F525}",
            "",
            "a",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ];
        for s in cases {
            let expected = if s.is_empty() {
                SPECIAL_TOKENS
            } else {
                std::cmp::max(s.len() / 3, s.chars().count()) + SPECIAL_TOKENS
            };
            assert_eq!(
                estimate_tokens(s),
                expected,
                "drift from the Python seam for {s:?}"
            );
        }
    }

    // ── latency histogram ────────────────────────────────────────────────
    //
    // These pin the behaviour that the hardcoded `p50_ms: 0.0` could not have:
    // each of them fails against a constant zero. §12c — a guard that cannot
    // demonstrate a catch is indistinguishable from one that is blind.

    struct SleepBackend(Duration);

    impl ModelBackend for SleepBackend {
        async fn run(&self, batch: Batch) -> Result<BatchOutput> {
            tokio::time::sleep(self.0).await;
            Ok(BatchOutput::new(vec![vec![0.0; 4]; batch.texts.len()]))
        }
    }

    #[test]
    fn latency_is_none_before_any_batch_completes() {
        // The honest answer to "what is p99?" with no samples is "I do not
        // know" -- not 0.0, which reads as a fast server.
        let ring = LatencyRing::new(8);
        assert!(ring.percentiles().is_none());
    }

    #[test]
    fn latency_reports_real_percentiles() {
        let ring = LatencyRing::new(128);
        // 98 fast + 2 slow. The tail must be BIGGER than 1% of the window for
        // p99 to be required to see it: at exactly 1-in-100, nearest-rank p99
        // is the 99th of 100 samples and the single outlier sits above it, so
        // returning the fast value is CORRECT, not a miss. The first draft of
        // this test asserted otherwise and failed against a correct
        // implementation — kept here so it is not "fixed" back.
        for _ in 0..98 {
            ring.record(Duration::from_millis(10));
        }
        ring.record(Duration::from_millis(500));
        ring.record(Duration::from_millis(500));
        let (p50, p99) = ring.percentiles().expect("samples recorded");
        assert!((p50 - 10.0).abs() < 1.0, "p50 was {p50}");
        assert!(p99 >= 500.0, "p99 must surface the tail, was {p99}");
    }

    #[test]
    fn nearest_rank_percentile_is_an_observed_sample() {
        // Pins the definition: p_q is the ceil(q*n)-th smallest, so every
        // reported value is a latency that actually occurred rather than an
        // interpolation between two of them.
        let us: Vec<u64> = (1..=100).map(|i| i * 1000).collect(); // 1..100 ms
        assert_eq!(pct(&us, 0.50), 50.0);
        assert_eq!(pct(&us, 0.99), 99.0);
        // Degenerate inputs must not panic or index out of bounds.
        assert_eq!(pct(&[7_000], 0.50), 7.0);
        assert_eq!(pct(&[7_000], 0.99), 7.0);
    }

    #[test]
    fn latency_ring_wraps_and_forgets_old_samples() {
        // A recent-window metric must TRACK, not average over all time: once
        // the slow era scrolls out, the numbers must come back down.
        let ring = LatencyRing::new(4);
        for _ in 0..4 {
            ring.record(Duration::from_millis(900));
        }
        let (p50_slow, _) = ring.percentiles().unwrap();
        assert!(p50_slow >= 900.0, "was {p50_slow}");
        for _ in 0..4 {
            ring.record(Duration::from_millis(5));
        }
        let (p50_fast, p99_fast) = ring.percentiles().unwrap();
        assert!(p50_fast < 50.0, "window did not forget: p50 {p50_fast}");
        assert!(p99_fast < 50.0, "window did not forget: p99 {p99_fast}");
    }

    #[test]
    fn latency_percentiles_ignore_unwritten_slots() {
        // Only the filled prefix is read. Counting the zeroed remainder would
        // dilute every percentile toward 0 for the first LATENCY_WINDOW
        // batches -- precisely the reassuring-but-unearned number this work
        // exists to remove.
        let ring = LatencyRing::new(1024);
        ring.record(Duration::from_millis(100));
        let (p50, p99) = ring.percentiles().unwrap();
        assert!((p50 - 100.0).abs() < 1.0, "p50 diluted by empty slots: {p50}");
        assert!((p99 - 100.0).abs() < 1.0, "p99 diluted by empty slots: {p99}");
    }

    #[tokio::test]
    async fn dispatcher_stats_report_measured_latency_end_to_end() {
        // The whole point: drive a real Dispatcher and confirm stats() carries
        // a number that came from the backend actually taking time.
        let d = Dispatcher::new(
            DispatcherConfig::default(),
            SleepBackend(Duration::from_millis(40)),
        );
        assert!(d.stats().p50_ms.is_none(), "no samples before any work");

        d.embed("hello".to_string()).await.expect("embed");

        let s = d.stats();
        let p50 = s.p50_ms.expect("a completed batch must produce a sample");
        assert!(p50 >= 30.0, "p50 {p50} did not reflect a 40ms backend");
        assert!(p50 < 5_000.0, "p50 {p50} implausible");
    }
}
