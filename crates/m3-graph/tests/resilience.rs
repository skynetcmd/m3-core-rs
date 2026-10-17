use std::time::Duration;

use m3_graph::{BreakerState, CircuitBreaker, RetryPolicy};

// Wide enough that two consecutive calls cannot straddle a window boundary on a
// loaded runner: a probe outstanding for a whole window is presumed lost, so a
// tight window would let a slow "second caller" through.
const RESET: Duration = Duration::from_millis(200);
const PAST_RESET: Duration = Duration::from_millis(250);

#[test]
fn breaker_closed_to_open_after_threshold() {
    let mut b = CircuitBreaker::new(3, Duration::from_millis(50));
    assert_eq!(b.state(), BreakerState::Closed);
    b.record_failure();
    b.record_failure();
    assert_eq!(b.state(), BreakerState::Closed);
    b.record_failure();
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.allow_request());
}

#[test]
fn breaker_open_to_half_open_after_reset() {
    let mut b = CircuitBreaker::new(1, RESET);
    b.record_failure();
    assert_eq!(b.state(), BreakerState::Open);
    std::thread::sleep(PAST_RESET);
    assert_eq!(b.state(), BreakerState::HalfOpen);
    assert!(b.allow_request()); // probe admitted
    assert!(!b.allow_request()); // only one probe
}

#[test]
fn breaker_half_open_to_closed_on_success() {
    let mut b = CircuitBreaker::new(1, RESET);
    b.record_failure();
    std::thread::sleep(PAST_RESET);
    assert!(b.allow_request());
    b.record_success();
    assert_eq!(b.state(), BreakerState::Closed);
    assert!(b.allow_request());
}

#[test]
fn breaker_half_open_to_open_on_failure() {
    let mut b = CircuitBreaker::new(1, RESET);
    b.record_failure();
    std::thread::sleep(PAST_RESET);
    assert!(b.allow_request());
    b.record_failure();
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.allow_request());
}

#[test]
fn breaker_presumes_a_lost_probe_after_another_reset_window() {
    let mut b = CircuitBreaker::new(1, RESET);
    b.record_failure();
    std::thread::sleep(PAST_RESET);
    assert!(b.allow_request()); // the probe, which never reports
    assert!(!b.allow_request());
    std::thread::sleep(PAST_RESET);
    assert!(
        b.allow_request(),
        "a lost probe must not refuse every call forever"
    );
    assert!(!b.allow_request()); // the new probe is the only one
}

#[test]
fn breaker_peek_never_claims_the_probe() {
    let mut b = CircuitBreaker::new(1, RESET);
    b.record_failure();
    assert!(!b.peek());
    std::thread::sleep(PAST_RESET);
    assert!(b.peek());
    assert!(b.peek());
    assert!(b.allow_request(), "peeking left the probe to claim");
    assert!(!b.peek());
}

#[test]
fn retry_backoff_math() {
    let p = RetryPolicy::new(5, Duration::from_millis(100), Duration::from_secs(2));
    assert_eq!(p.delay_for_attempt(1), Duration::from_millis(100));
    assert_eq!(p.delay_for_attempt(2), Duration::from_millis(200));
    assert_eq!(p.delay_for_attempt(3), Duration::from_millis(400));
    assert_eq!(p.delay_for_attempt(4), Duration::from_millis(800));
    // attempt 5 -> 1600ms, still under cap
    assert_eq!(p.delay_for_attempt(5), Duration::from_millis(1600));
}

#[test]
fn retry_backoff_clamped_to_max() {
    let p = RetryPolicy::new(20, Duration::from_millis(100), Duration::from_secs(1));
    assert_eq!(p.delay_for_attempt(10), Duration::from_secs(1));
    assert_eq!(p.delay_for_attempt(31), Duration::from_secs(1));
}
