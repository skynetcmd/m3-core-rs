//! m3-dispatchd: a per-user doorbell for m3 agent mail.
//!
//! Writers ring after committing a message; waiters wait for their readers and
//! are woken once per burst, one outstanding wake at a time. The daemon issues
//! no SQL (messages, claims and acks stay in m3's store, under the dialect seam)
//! and performs no cryptography (local IPC only; see `ipc`). Design:
//! m3-memory docs/ROADMAP_DISPATCHER.md, "Implementation".

pub mod addressing;
pub mod api;
pub mod hub;
pub mod ipc;
