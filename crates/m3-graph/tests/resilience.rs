use std::time::Duration;

use m3_graph::{BreakerState, CircuitBreaker, RetryPolicy};

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
    let mut b = CircuitBreaker::new(1, Duration::from_millis(20));
    b.record_failure();
    assert_eq!(b.state(), BreakerState::Open);
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(b.state(), BreakerState::HalfOpen);
    assert!(b.allow_request()); // probe admitted
    assert!(!b.allow_request()); // only one probe
}

#[test]
fn breaker_half_open_to_closed_on_success() {
    let mut b = CircuitBreaker::new(1, Duration::from_millis(20));
    b.record_failure();
    std::thread::sleep(Duration::from_millis(30));
    assert!(b.allow_request());
    b.record_success();
    assert_eq!(b.state(), BreakerState::Closed);
    assert!(b.allow_request());
}

#[test]
fn breaker_half_open_to_open_on_failure() {
    let mut b = CircuitBreaker::new(1, Duration::from_millis(20));
    b.record_failure();
    std::thread::sleep(Duration::from_millis(30));
    assert!(b.allow_request());
    b.record_failure();
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.allow_request());
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
