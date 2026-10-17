//! Re-export of the shared resilience primitives, which `m3-resilience` owns.
//! Kept so `m3_graph::resilience::*` and `m3_graph::CircuitBreaker` still
//! resolve.

pub use m3_resilience::{Admission, BreakerState, BreakerStatus, CircuitBreaker, RetryPolicy};
