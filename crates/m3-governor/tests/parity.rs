//! Table-driven parity tests for the governor pacing ladder.
//!
//! Each row encodes `(initial, limit, load, elapsed) -> (mode, bg_delay,
//! interactive_delay)` and mirrors the Python `get_governor_pacing` truth table
//! in `bin/m3_sdk.py`. If these diverge, the native and fallback paths would
//! return different dicts — the one thing the oxidation must never do.

use m3_governor::{GovernorConfig, PacingMode};

struct Case {
    initial: i64,
    limit: i64,
    load: f64,
    elapsed: f64,
    mode: PacingMode,
    bg: Option<f64>,
    interactive: f64,
}

#[test]
fn pacing_ladder_parity() {
    let cases = [
        // Critical: load >= limit (limit != 100), regardless of elapsed.
        Case {
            initial: 85,
            limit: 95,
            load: 95.0,
            elapsed: 0.0,
            mode: PacingMode::Halted,
            bg: None,
            interactive: 30.0,
        },
        Case {
            initial: 85,
            limit: 95,
            load: 99.9,
            elapsed: 999.0,
            mode: PacingMode::Halted,
            bg: None,
            interactive: 30.0,
        },
        // Throttled: initial <= load < limit.
        Case {
            initial: 85,
            limit: 95,
            load: 85.0,
            elapsed: 0.0,
            mode: PacingMode::Throttled,
            bg: Some(10.0),
            interactive: 0.0,
        },
        Case {
            initial: 85,
            limit: 95,
            load: 94.9,
            elapsed: 999.0,
            mode: PacingMode::Throttled,
            bg: Some(10.0),
            interactive: 0.0,
        },
        // Normal, recent interaction (< 30s) -> Halted.
        Case {
            initial: 85,
            limit: 95,
            load: 10.0,
            elapsed: 0.0,
            mode: PacingMode::Halted,
            bg: None,
            interactive: 0.0,
        },
        Case {
            initial: 85,
            limit: 95,
            load: 10.0,
            elapsed: 29.9,
            mode: PacingMode::Halted,
            bg: None,
            interactive: 0.0,
        },
        // Normal, tapering window [30, 60).
        Case {
            initial: 85,
            limit: 95,
            load: 10.0,
            elapsed: 30.0,
            mode: PacingMode::Tapered,
            bg: Some(5.0),
            interactive: 0.0,
        },
        Case {
            initial: 85,
            limit: 95,
            load: 10.0,
            elapsed: 59.9,
            mode: PacingMode::Tapered,
            bg: Some(5.0),
            interactive: 0.0,
        },
        // Normal, idle (>= 60s) -> Continuous.
        Case {
            initial: 85,
            limit: 95,
            load: 0.0,
            elapsed: 60.0,
            mode: PacingMode::Continuous,
            bg: Some(0.1),
            interactive: 0.0,
        },
        // limit == 100 DISABLES critical mode: even load 100 falls through to throttled.
        Case {
            initial: 85,
            limit: 100,
            load: 100.0,
            elapsed: 0.0,
            mode: PacingMode::Throttled,
            bg: Some(10.0),
            interactive: 0.0,
        },
    ];

    for (i, c) in cases.iter().enumerate() {
        let cfg = GovernorConfig::new(c.initial, c.limit);
        let p = cfg.decide(c.load, c.elapsed);
        assert_eq!(p.mode, c.mode, "case {i}: mode");
        assert_eq!(p.background_delay, c.bg, "case {i}: background_delay");
        assert_eq!(
            p.interactive_delay, c.interactive,
            "case {i}: interactive_delay"
        );
    }
}

#[test]
fn clamp_initial_below_floor() {
    // Python: min(99, max(10, x)) -> floor 10.
    let cfg = GovernorConfig::new(0, 95);
    assert_eq!(cfg.initial_limit, 10);
}

#[test]
fn clamp_initial_above_ceiling() {
    let cfg = GovernorConfig::new(250, 100);
    assert_eq!(cfg.initial_limit, 99);
}

#[test]
fn clamp_limit_range() {
    // Python: min(100, max(20, x)).
    assert_eq!(GovernorConfig::new(50, 5).limit_threshold, 20);
    assert_eq!(GovernorConfig::new(50, 250).limit_threshold, 100);
}

#[test]
fn sanity_fix_initial_ge_limit() {
    // initial >= limit && limit != 100 -> initial = limit - 5.
    let cfg = GovernorConfig::new(95, 90);
    assert_eq!(cfg.initial_limit, 85);
    assert_eq!(cfg.limit_threshold, 90);
}

#[test]
fn sanity_fix_skipped_when_limit_is_100() {
    // limit == 100 exempts the sanity fix (matches the Python `!= 100` guard).
    let cfg = GovernorConfig::new(99, 100);
    assert_eq!(cfg.initial_limit, 99);
    assert_eq!(cfg.limit_threshold, 100);
}
