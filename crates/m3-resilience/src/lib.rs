//! Resilience primitives shared across the workspace: a circuit breaker and a
//! retry policy. This crate owns the breaker's state machine; the async
//! dispatcher and the Python binding wrap it rather than re-implement it.
//!
//! Every time-dependent method has an `_at(now)` form so the transitions can be
//! tested without sleeping; the plain forms pass `Instant::now()`.

use std::time::{Duration, Instant};

/// Circuit breaker state, as a caller would meet it at a given instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Requests flow normally.
    Closed,
    /// Tripped: requests are refused until the cooldown elapses.
    Open,
    /// The cooldown has elapsed, or a half-open probe is outstanding: at most
    /// one request at a time is let through to test the backend.
    HalfOpen,
}

/// The answer to "may a call go through now?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Closed, or the cooldown has elapsed (an admitting check made this
    /// caller the half-open probe).
    Allowed,
    /// Open inside the cooldown.
    Open {
        /// Cooldown left before a probe is allowed.
        retry_after: Duration,
    },
    /// A half-open probe is outstanding and has not reported yet.
    ProbeOutstanding {
        /// Time left before that probe is presumed lost and the next caller
        /// probes instead.
        retry_after: Duration,
    },
}

impl Admission {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Admission::Allowed)
    }
}

/// What the breaker is doing, for a health probe or an error reply.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BreakerStatus {
    /// Calls are being refused: open (or probing) and the cooldown has not
    /// elapsed.
    pub open: bool,
    /// Failures since the last success.
    pub consecutive_failures: usize,
    /// Cooldown left before a half-open probe is allowed; `None` unless `open`.
    pub retry_after: Option<Duration>,
    /// The most recent failure message recorded since the last success.
    pub last_error: Option<String>,
}

/// The recorded phase. `Open` and `HalfOpen` carry the instant of the
/// transition: opened at, or probe admitted at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Closed,
    Open(Instant),
    HalfOpen(Instant),
}

/// `threshold` consecutive failures open the breaker for `cooldown`. After the
/// cooldown exactly one half-open probe is admitted; its success closes the
/// breaker, its failure re-opens it for a fresh cooldown. A probe that never
/// reports (its caller was cancelled, or it ended in an error the caller chose
/// not to count) is presumed lost after another `cooldown`, and the next caller
/// probes. Without that presumption one lost probe refuses every call forever.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    threshold: usize,
    cooldown: Duration,
    consecutive_failures: usize,
    phase: Phase,
    last_error: Option<String>,
}

impl CircuitBreaker {
    /// A `threshold` of 0 is treated as 1: the first failure opens.
    pub fn new(threshold: u32, cooldown: Duration) -> Self {
        CircuitBreaker {
            threshold: (threshold as usize).max(1),
            cooldown,
            consecutive_failures: 0,
            phase: Phase::Closed,
            last_error: None,
        }
    }

    pub fn cooldown(&self) -> Duration {
        self.cooldown
    }

    pub fn consecutive_failures(&self) -> usize {
        self.consecutive_failures
    }

    /// The most recent failure message recorded since the last success.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// The recorded phase with no allowance for elapsed time: `Open` stays
    /// `Open` past its cooldown until a check admits a probe. For a cached
    /// fast-path flag, which must change only when a transition is recorded.
    pub fn recorded_state(&self) -> BreakerState {
        match self.phase {
            Phase::Closed => BreakerState::Closed,
            Phase::Open(_) => BreakerState::Open,
            Phase::HalfOpen(_) => BreakerState::HalfOpen,
        }
    }

    /// Current state, accounting for time elapsed since the last transition.
    pub fn state(&self) -> BreakerState {
        self.state_at(Instant::now())
    }

    pub fn state_at(&self, now: Instant) -> BreakerState {
        match self.gate(now) {
            None => BreakerState::Closed,
            Some((Phase::Open(_), Some(_))) => BreakerState::Open,
            Some(_) => BreakerState::HalfOpen,
        }
    }

    /// Whether a call should be attempted now. Past the cooldown this caller
    /// becomes the half-open probe, so it MUST report through
    /// `record_success`/`record_failure`; use `peek` to ask without claiming.
    pub fn allow_request(&mut self) -> bool {
        self.admit_at(Instant::now()).is_allowed()
    }

    /// The admitting check, with the reason when refused.
    pub fn admit(&mut self) -> Admission {
        self.admit_at(Instant::now())
    }

    pub fn admit_at(&mut self, now: Instant) -> Admission {
        let a = self.check_at(now);
        if a.is_allowed() && !matches!(self.phase, Phase::Closed) {
            self.phase = Phase::HalfOpen(now);
        }
        a
    }

    /// Whether `allow_request` would admit a call now, without claiming the
    /// half-open probe.
    pub fn peek(&self) -> bool {
        self.check_at(Instant::now()).is_allowed()
    }

    /// The non-admitting check, with the reason when refused.
    pub fn check(&self) -> Admission {
        self.check_at(Instant::now())
    }

    pub fn check_at(&self, now: Instant) -> Admission {
        match self.gate(now) {
            None | Some((_, None)) => Admission::Allowed,
            Some((Phase::HalfOpen(_), Some(retry_after))) => {
                Admission::ProbeOutstanding { retry_after }
            }
            Some((_, Some(retry_after))) => Admission::Open { retry_after },
        }
    }

    /// `None` when closed; otherwise the phase and the cooldown left, `None`
    /// once it has elapsed. The one place elapsed time is judged.
    fn gate(&self, now: Instant) -> Option<(Phase, Option<Duration>)> {
        let since = match self.phase {
            Phase::Closed => return None,
            Phase::Open(t) | Phase::HalfOpen(t) => t,
        };
        let left = self
            .cooldown
            .checked_sub(now.saturating_duration_since(since))
            .filter(|d| !d.is_zero());
        Some((self.phase, left))
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.phase = Phase::Closed;
        self.last_error = None;
    }

    /// Count a failure. The last recorded message is kept.
    pub fn record_failure(&mut self) {
        self.record_failure_at(Instant::now());
    }

    pub fn record_failure_at(&mut self, now: Instant) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures >= self.threshold || matches!(self.phase, Phase::HalfOpen(_)) {
            self.phase = Phase::Open(now);
        }
    }

    /// Count a failure and keep `message` for refusals and `status`.
    pub fn record_failure_with(&mut self, message: impl Into<String>) {
        self.record_failure_with_at(message, Instant::now());
    }

    pub fn record_failure_with_at(&mut self, message: impl Into<String>, now: Instant) {
        self.last_error = Some(message.into());
        self.record_failure_at(now);
    }

    /// The breaker as a caller would meet it now. Open past its cooldown reads
    /// as not open: the next call is let through as the half-open probe. While
    /// a probe is outstanding other calls are refused, so that reads as open.
    pub fn status(&self) -> BreakerStatus {
        self.status_at(Instant::now())
    }

    pub fn status_at(&self, now: Instant) -> BreakerStatus {
        let retry_after = self.gate(now).and_then(|(_, left)| left);
        BreakerStatus {
            open: retry_after.is_some(),
            consecutive_failures: self.consecutive_failures,
            retry_after,
            last_error: self.last_error.clone(),
        }
    }
}

/// Whole seconds, rounded up, for a refusal message or a `Retry-After` header:
/// truncating would tell a caller "0s" while the breaker still refuses.
pub fn ceil_secs(d: Duration) -> u64 {
    d.as_secs() + u64::from(d.subsec_nanos() > 0)
}

/// Exponential-backoff retry policy. `delay_for_attempt` is pure math; the
/// caller owns the sleeping and the attempt loop.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32, base_delay: Duration, max_delay: Duration) -> Self {
        RetryPolicy {
            max_attempts,
            base_delay,
            max_delay,
        }
    }

    /// Backoff before retry `attempt` (1-based): `base * 2^(attempt-1)`, clamped
    /// to `max_delay`. Attempt 0 or 1 yields the base delay.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let shift = attempt.saturating_sub(1).min(31);
        let factor = 1u32 << shift;
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }
}
