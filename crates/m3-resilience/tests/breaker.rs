//! The breaker's state machine, driven with injected instants so no test
//! sleeps or depends on scheduling.

use std::time::{Duration, Instant};

use m3_resilience::{
    ceil_secs, Admission, BreakerState, BreakerStatus, CircuitBreaker, RetryPolicy,
};

const COOLDOWN: Duration = Duration::from_secs(10);

fn secs(s: u64) -> Duration {
    Duration::from_secs(s)
}

/// A breaker opened at the returned instant by `threshold` failures.
fn opened(threshold: u32) -> (CircuitBreaker, Instant) {
    let t0 = Instant::now();
    let mut b = CircuitBreaker::new(threshold, COOLDOWN);
    for _ in 0..threshold {
        b.record_failure_at(t0);
    }
    (b, t0)
}

#[test]
fn closed_admits_until_the_threshold() {
    let t0 = Instant::now();
    let mut b = CircuitBreaker::new(3, COOLDOWN);
    b.record_failure_at(t0);
    b.record_failure_at(t0);
    assert_eq!(b.state_at(t0), BreakerState::Closed);
    assert_eq!(b.admit_at(t0), Admission::Allowed);
    assert_eq!(b.consecutive_failures(), 2);
    b.record_failure_at(t0);
    assert_eq!(b.state_at(t0), BreakerState::Open);
}

#[test]
fn a_zero_threshold_opens_on_the_first_failure() {
    let t0 = Instant::now();
    let mut z = CircuitBreaker::new(0, COOLDOWN);
    z.record_failure_at(t0);
    assert_eq!(z.state_at(t0), BreakerState::Open);
}

#[test]
fn open_refuses_with_the_cooldown_left() {
    let (mut b, t0) = opened(3);
    assert_eq!(
        b.admit_at(t0 + secs(4)),
        Admission::Open {
            retry_after: secs(6)
        }
    );
    assert_eq!(
        b.check_at(t0 + secs(4)),
        Admission::Open {
            retry_after: secs(6)
        }
    );
    assert_eq!(b.recorded_state(), BreakerState::Open);
}

#[test]
fn a_success_closes_and_resets_the_count() {
    let (mut b, t0) = opened(2);
    b.record_success();
    assert_eq!(b.state_at(t0), BreakerState::Closed);
    assert_eq!(b.consecutive_failures(), 0);
    b.record_failure_at(t0);
    assert_eq!(
        b.state_at(t0),
        BreakerState::Closed,
        "the count restarted at zero"
    );
}

/// Ported from the dispatcher: half-open admits ONE probe, and its result
/// decides.
#[test]
fn half_open_admits_a_single_probe() {
    let (mut b, t0) = opened(1);
    assert_eq!(b.state_at(t0 + COOLDOWN), BreakerState::HalfOpen);
    assert_eq!(b.admit_at(t0 + COOLDOWN), Admission::Allowed, "the probe");
    assert_eq!(
        b.admit_at(t0 + secs(11)),
        Admission::ProbeOutstanding {
            retry_after: secs(9)
        },
        "a second caller waits for the probe"
    );
    assert_eq!(b.state_at(t0 + secs(11)), BreakerState::HalfOpen);
    b.record_success();
    assert_eq!(b.admit_at(t0 + secs(11)), Admission::Allowed);
    assert_eq!(
        b.admit_at(t0 + secs(11)),
        Admission::Allowed,
        "closed: no probe limit"
    );
}

/// Ported from the dispatcher: a failed probe re-opens for a fresh cooldown.
#[test]
fn a_failed_probe_reopens_for_a_fresh_cooldown() {
    let (mut b, t0) = opened(3);
    assert!(b.admit_at(t0 + COOLDOWN).is_allowed());
    b.record_failure_at(t0 + secs(12));
    assert_eq!(b.state_at(t0 + secs(13)), BreakerState::Open);
    assert_eq!(
        b.check_at(t0 + secs(13)),
        Admission::Open {
            retry_after: secs(9)
        }
    );
    assert!(
        b.admit_at(t0 + secs(22)).is_allowed(),
        "the next probe after the new cooldown"
    );
}

/// A probe whose caller never reports must not refuse every call forever: it is
/// presumed lost after another cooldown and the next caller probes.
#[test]
fn a_probe_that_never_reports_is_presumed_lost_after_another_cooldown() {
    let (mut b, t0) = opened(1);
    assert!(
        b.admit_at(t0 + COOLDOWN).is_allowed(),
        "the probe, which never reports"
    );
    assert_eq!(
        b.admit_at(t0 + secs(19)),
        Admission::ProbeOutstanding {
            retry_after: secs(1)
        }
    );
    assert!(
        b.admit_at(t0 + secs(20)).is_allowed(),
        "presumed lost: this caller probes"
    );
    assert_eq!(
        b.admit_at(t0 + secs(21)),
        Admission::ProbeOutstanding {
            retry_after: secs(9)
        },
        "and only this one"
    );
}

/// `peek`/`check` answer "would a call be allowed" without claiming the probe.
#[test]
fn peek_never_claims_the_probe() {
    let (mut b, t0) = opened(1);
    for _ in 0..3 {
        assert_eq!(b.check_at(t0 + COOLDOWN), Admission::Allowed);
    }
    assert_eq!(
        b.recorded_state(),
        BreakerState::Open,
        "checking recorded nothing"
    );
    assert!(
        b.admit_at(t0 + COOLDOWN).is_allowed(),
        "the probe is still there to claim"
    );
    assert_eq!(
        b.check_at(t0 + COOLDOWN),
        Admission::ProbeOutstanding {
            retry_after: COOLDOWN
        }
    );
}

#[test]
fn recorded_state_moves_only_on_a_transition() {
    let (mut b, t0) = opened(1);
    assert_eq!(b.state_at(t0 + secs(60)), BreakerState::HalfOpen);
    assert_eq!(b.recorded_state(), BreakerState::Open);
    b.admit_at(t0 + secs(60));
    assert_eq!(b.recorded_state(), BreakerState::HalfOpen);
    b.record_failure_at(t0 + secs(60));
    assert_eq!(b.recorded_state(), BreakerState::Open);
    b.record_success();
    assert_eq!(b.recorded_state(), BreakerState::Closed);
}

#[test]
fn status_reports_refusal_count_cooldown_and_last_error() {
    let t0 = Instant::now();
    let mut b = CircuitBreaker::new(2, COOLDOWN);
    assert_eq!(b.status_at(t0), BreakerStatus::default());
    b.record_failure_with_at("connection refused", t0);
    let st = b.status_at(t0);
    assert!(!st.open, "below the threshold calls still flow");
    assert_eq!(st.consecutive_failures, 1);
    assert_eq!(st.last_error.as_deref(), Some("connection refused"));

    b.record_failure_with_at("timed out", t0);
    assert_eq!(
        b.status_at(t0 + secs(3)),
        BreakerStatus {
            open: true,
            consecutive_failures: 2,
            retry_after: Some(secs(7)),
            last_error: Some("timed out".to_string()),
        }
    );
    assert!(
        !b.status_at(t0 + COOLDOWN).open,
        "past the cooldown the next call probes"
    );
    b.admit_at(t0 + COOLDOWN);
    assert!(
        b.status_at(t0 + secs(11)).open,
        "a probe outstanding refuses others"
    );

    b.record_failure_at(t0 + secs(11));
    assert_eq!(
        b.last_error(),
        Some("timed out"),
        "a message-less failure keeps the last message"
    );
    b.record_success();
    assert_eq!(b.status_at(t0 + secs(11)), BreakerStatus::default());
}

#[test]
fn time_before_the_transition_does_not_panic() {
    // An instant earlier than the recorded one (a caller's stale `now`) reads as
    // zero elapsed, not a panic or a wrapped duration.
    let t0 = Instant::now();
    let mut b = CircuitBreaker::new(1, COOLDOWN);
    b.record_failure_at(t0 + secs(5));
    assert_eq!(
        b.check_at(t0),
        Admission::Open {
            retry_after: COOLDOWN
        }
    );
}

#[test]
fn retry_backoff_doubles_and_clamps() {
    let p = RetryPolicy::new(5, Duration::from_millis(100), Duration::from_secs(2));
    assert_eq!(p.delay_for_attempt(0), Duration::from_millis(100));
    assert_eq!(p.delay_for_attempt(1), Duration::from_millis(100));
    assert_eq!(p.delay_for_attempt(4), Duration::from_millis(800));
    assert_eq!(p.delay_for_attempt(6), Duration::from_secs(2));
    assert_eq!(p.delay_for_attempt(u32::MAX), Duration::from_secs(2));
}

#[test]
fn ceil_secs_rounds_up_and_never_reports_zero_for_a_remainder() {
    assert_eq!(ceil_secs(Duration::ZERO), 0);
    assert_eq!(ceil_secs(Duration::from_nanos(1)), 1);
    assert_eq!(ceil_secs(secs(60)), 60);
    assert_eq!(ceil_secs(Duration::from_millis(59_001)), 60);
}
