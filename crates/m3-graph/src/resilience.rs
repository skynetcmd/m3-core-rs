//! Reusable Phase 5 resilience primitives: a circuit breaker and a retry
//! policy. These are deliberately generic and standalone — other crates
//! (dispatcher, embed backends, route) can reuse them, and they may later
//! extract into their own crate.

use std::time::{Duration, Instant};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Requests flow normally.
    Closed,
    /// Tripped — requests are rejected until the reset window elapses.
    Open,
    /// Reset window elapsed — a single probe request is allowed through.
    HalfOpen,
}

/// N consecutive failures trip the breaker open for `reset_after`; the first
/// request after that window is a half-open probe that either closes the
/// breaker (on success) or re-opens it (on failure).
pub struct CircuitBreaker {
    threshold: u32,
    reset_after: Duration,
    consecutive_failures: u32,
    opened_at: Option<Instant>,
    probing: bool,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_after: Duration) -> Self {
        CircuitBreaker {
            threshold: threshold.max(1),
            reset_after,
            consecutive_failures: 0,
            opened_at: None,
            probing: false,
        }
    }

    /// Current state, accounting for elapsed time since the breaker opened.
    pub fn state(&self) -> BreakerState {
        match self.opened_at {
            None => BreakerState::Closed,
            Some(t) => {
                if self.probing || t.elapsed() >= self.reset_after {
                    BreakerState::HalfOpen
                } else {
                    BreakerState::Open
                }
            }
        }
    }

    /// Whether a request should be attempted now. Transitions Open → HalfOpen
    /// when the reset window has elapsed, admitting exactly one probe.
    pub fn allow_request(&mut self) -> bool {
        match self.opened_at {
            None => true,
            Some(t) => {
                if self.probing {
                    false
                } else if t.elapsed() >= self.reset_after {
                    self.probing = true;
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.opened_at = None;
        self.probing = false;
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.probing || self.consecutive_failures >= self.threshold {
            self.opened_at = Some(Instant::now());
            self.probing = false;
        }
    }
}

/// Exponential-backoff retry policy. `delay_for_attempt` is pure math — the
/// caller owns the actual sleeping and attempt loop.
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
        let factor = 1u64 << shift;
        let delay = self.base_delay.saturating_mul(factor as u32);
        delay.min(self.max_delay)
    }
}
