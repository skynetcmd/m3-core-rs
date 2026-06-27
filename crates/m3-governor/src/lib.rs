//! Adaptive background-workload governor — pure pacing-decision logic.
//!
//! This is the Rust source-of-truth for the pacing ladder that decides how
//! aggressively background daemons may run given current host load and how
//! long it has been since the last user interaction. It mirrors, byte-for-byte
//! in behavior, the Python `get_governor_pacing` in `bin/m3_sdk.py` so the two
//! can be swapped via the `M3_CORE_RS_DISABLE` fallback gate.
//!
//! No I/O, no allocation in the hot path, no clock access — `decide` is a pure
//! function of `(load, elapsed_since_interaction)`. The caller supplies both,
//! which keeps this crate testable and deterministic.

/// The four background-pacing modes, ordered most → least restrictive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// Background work stops entirely.
    Halted,
    /// Host under heavy load — background work runs with a long delay.
    Throttled,
    /// Recent interaction — background work ramps back up with a short delay.
    Tapered,
    /// Idle host — background work runs continuously.
    Continuous,
}

impl PacingMode {
    /// The wire string used by the Python dict (`background` key).
    pub fn as_str(&self) -> &'static str {
        match self {
            PacingMode::Halted => "HALTED",
            PacingMode::Throttled => "THROTTLED",
            PacingMode::Tapered => "TAPERED",
            PacingMode::Continuous => "CONTINUOUS",
        }
    }
}

/// A pacing decision. `background_delay` is `None` in modes where the Python
/// dict omits the `background_delay` key (critical mode), so the binding can
/// reproduce the exact dict shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pacing {
    pub mode: PacingMode,
    pub background_delay: Option<f64>,
    pub interactive_delay: f64,
}

/// User-selectable governor thresholds, clamped to the same ranges the Python
/// module-level globals use.
#[derive(Debug, Clone, Copy)]
pub struct GovernorConfig {
    pub initial_limit: u32,
    pub limit_threshold: u32,
}

impl GovernorConfig {
    /// Build a config applying the EXACT clamps and sanity fix from
    /// `bin/m3_sdk.py`:
    ///   - `initial_limit` clamped to `[10, 99]`
    ///   - `limit_threshold` clamped to `[20, 100]`
    ///   - if `initial >= limit` and `limit != 100`, set `initial = limit - 5`
    pub fn new(initial_limit: i64, limit_threshold: i64) -> Self {
        let mut initial = initial_limit.clamp(10, 99) as u32;
        let limit = limit_threshold.clamp(20, 100) as u32;
        if initial >= limit && limit != 100 {
            initial = limit - 5;
        }
        GovernorConfig {
            initial_limit: initial,
            limit_threshold: limit,
        }
    }

    /// Decide the pacing for the given host `load` (0–100, the max across CPU /
    /// RAM / GPU) and `elapsed` seconds since the last user interaction.
    ///
    /// Branch ladder (identical to the Python truth table):
    ///
    /// 1. Critical: `limit != 100 && load >= limit` → Halted, interactive 30s.
    /// 2. Throttled: `load >= initial` → Throttled, background 10s.
    /// 3. Normal: `elapsed < 30` → Halted (interactive 0); `elapsed < 60` →
    ///    Tapered, background 5s; else → Continuous, background 0.1s.
    pub fn decide(&self, load: f64, elapsed: f64) -> Pacing {
        // 1. Critical mode.
        if self.limit_threshold != 100 && load >= self.limit_threshold as f64 {
            return Pacing {
                mode: PacingMode::Halted,
                background_delay: None,
                interactive_delay: 30.0,
            };
        }
        // 2. Throttled mode.
        if load >= self.initial_limit as f64 {
            return Pacing {
                mode: PacingMode::Throttled,
                background_delay: Some(10.0),
                interactive_delay: 0.0,
            };
        }
        // 3. Normal mode.
        if elapsed < 30.0 {
            Pacing {
                mode: PacingMode::Halted,
                background_delay: None,
                interactive_delay: 0.0,
            }
        } else if elapsed < 60.0 {
            Pacing {
                mode: PacingMode::Tapered,
                background_delay: Some(5.0),
                interactive_delay: 0.0,
            }
        } else {
            Pacing {
                mode: PacingMode::Continuous,
                background_delay: Some(0.1),
                interactive_delay: 0.0,
            }
        }
    }
}
