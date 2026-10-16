//! The doorbell: rings in, paced and coalesced wakes out.
//!
//! A writer that committed a message RINGS with its recipients. A waiter WAITS
//! for its readers. The hub keeps a short log of rings with sequence numbers and
//! one cursor per waiter (keyed by its reader set), which gives:
//!
//! * **one outstanding wake per waiter**: a wake advances the cursor to the
//!   newest ring it covered, and nothing more is delivered until the waiter
//!   waits again (after its agent has handled the batch);
//! * **nothing missed between waits**: rings arriving while no waiter is
//!   connected stay in the log and are delivered at the next wait;
//! * **one wake per burst**: once a matching ring is seen, the hub waits until
//!   the coalescing window after the first one has passed, so rings within the
//!   window arrive as one wake with a count.
//!
//! State is volatile by design. A reader set the hub has not seen (a new waiter,
//! or any waiter after a daemon restart) gets `Resync`: read the inbox once,
//! because rings before this moment are unknown. Messages themselves live in
//! m3's store; the hub only says when to look.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tokio::sync::Notify;
use tokio::time::Instant;

use crate::addressing::sees;

struct Ring {
    seq: u64,
    recipient: String,
    /// The node that rang, when the writer says (a multi-node relay names the
    /// remote node, so a waiter knows the message may need a sync to read).
    origin: Option<String>,
    at: Instant,
}

#[derive(Default)]
struct State {
    seq: u64,
    log: VecDeque<Ring>,
    cursors: HashMap<String, u64>,
    waiting: HashMap<String, usize>,
    wakes: HashMap<String, u64>,
}

/// What a wait returns.
#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "wake", rename_all = "lowercase")]
pub enum Wake {
    /// New mail for these readers: how many rings, for which recipients, and
    /// from which nodes (empty when no ring named one).
    Mail {
        count: usize,
        recipients: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        origins: Vec<String>,
    },
    /// The hub has no history for these readers: read the inbox once.
    Resync,
    /// Nothing arrived before the timeout.
    Timeout,
}

pub struct Hub {
    state: Mutex<State>,
    notify: Notify,
    window: Duration,
    keep: Duration,
    max_log: usize,
}

impl Hub {
    /// `window`: the coalescing window. `keep`: how long rings stay in the log
    /// for a waiter that is not connected.
    pub fn new(window: Duration, keep: Duration) -> Self {
        Self {
            state: Mutex::new(State::default()),
            notify: Notify::new(),
            window,
            keep,
            max_log: 100_000,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic while holding the lock leaves only counters behind; keep serving.
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record that messages for `recipients` were committed, on `origin` when given.
    pub fn ring<I: IntoIterator<Item = String>>(
        &self,
        recipients: I,
        origin: Option<&str>,
    ) -> usize {
        let now = Instant::now();
        let mut n = 0;
        {
            let mut st = self.lock();
            for r in recipients {
                if r.is_empty() {
                    continue;
                }
                st.seq += 1;
                let seq = st.seq;
                st.log.push_back(Ring {
                    seq,
                    recipient: r,
                    origin: origin.map(str::to_owned),
                    at: now,
                });
                n += 1;
            }
            while st.log.len() > self.max_log
                || st
                    .log
                    .front()
                    .is_some_and(|r| now.duration_since(r.at) > self.keep)
            {
                st.log.pop_front();
            }
        }
        if n > 0 {
            self.notify.notify_waiters();
        }
        n
    }

    fn key(readers: &[String]) -> String {
        readers
            .iter()
            .filter(|r| !r.is_empty())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Wait up to `timeout` for mail any of `readers` would see.
    pub async fn wait(&self, readers: &[String], timeout: Duration) -> Wake {
        let key = Self::key(readers);
        if key.is_empty() {
            return Wake::Timeout;
        }
        let deadline = Instant::now() + timeout;
        {
            let mut st = self.lock();
            if !st.cursors.contains_key(&key) {
                let seq = st.seq;
                st.cursors.insert(key, seq);
                return Wake::Resync;
            }
            *st.waiting.entry(key.clone()).or_default() += 1;
        }
        let out = self.wait_known(&key, readers, deadline).await;
        let mut st = self.lock();
        if let Some(w) = st.waiting.get_mut(&key) {
            *w = w.saturating_sub(1);
        }
        out
    }

    fn first_match(&self, key: &str, readers: &[String]) -> Option<Instant> {
        let st = self.lock();
        let cursor = st.cursors.get(key).copied().unwrap_or(0);
        st.log
            .iter()
            .find(|r| r.seq > cursor && readers.iter().any(|rd| sees(rd, &r.recipient)))
            .map(|r| r.at)
    }

    async fn wait_known(&self, key: &str, readers: &[String], deadline: Instant) -> Wake {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(first_at) = self.first_match(key, readers) {
                let ready = first_at + self.window;
                if Instant::now() < ready {
                    tokio::time::sleep_until(ready.min(deadline)).await;
                }
                return self.take(key, readers);
            }
            if Instant::now() >= deadline {
                return Wake::Timeout;
            }
            tokio::select! {
                _ = &mut notified => {}
                _ = tokio::time::sleep_until(deadline) => return Wake::Timeout,
            }
        }
    }

    fn take(&self, key: &str, readers: &[String]) -> Wake {
        let mut st = self.lock();
        let cursor = st.cursors.get(key).copied().unwrap_or(0);
        let mut count = 0;
        let mut last = cursor;
        let mut recipients = BTreeSet::new();
        let mut origins = BTreeSet::new();
        for r in st
            .log
            .iter()
            .filter(|r| r.seq > cursor && readers.iter().any(|rd| sees(rd, &r.recipient)))
        {
            count += 1;
            last = r.seq;
            recipients.insert(r.recipient.clone());
            origins.extend(r.origin.clone());
        }
        if count == 0 {
            return Wake::Timeout;
        }
        st.cursors.insert(key.to_string(), last);
        *st.wakes.entry(key.to_string()).or_default() += 1;
        Wake::Mail {
            count,
            recipients: recipients.into_iter().collect(),
            origins: origins.into_iter().collect(),
        }
    }

    /// The cross-agent view: per reader set, rings waiting and since when,
    /// whether a waiter is connected, wakes delivered.
    pub fn status(&self) -> serde_json::Value {
        let st = self.lock();
        let now = Instant::now();
        let mut readers = serde_json::Map::new();
        for (key, cursor) in &st.cursors {
            let set: Vec<&str> = key.split('\n').collect();
            let pending: Vec<&Ring> = st
                .log
                .iter()
                .filter(|r| r.seq > *cursor && set.iter().any(|rd| sees(rd, &r.recipient)))
                .collect();
            readers.insert(
                key.replace('\n', ","),
                serde_json::json!({
                    "pending": pending.len(),
                    "oldest_pending_ms": pending.first().map(|r| now.duration_since(r.at).as_millis() as u64),
                    "waiting": st.waiting.get(key).copied().unwrap_or(0),
                    "wakes": st.wakes.get(key).copied().unwrap_or(0),
                }),
            );
        }
        serde_json::json!({ "rings_logged": st.log.len(), "last_seq": st.seq, "readers": readers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn hub() -> Arc<Hub> {
        Arc::new(Hub::new(
            Duration::from_millis(250),
            Duration::from_secs(3600),
        ))
    }

    fn r(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test(start_paused = true)]
    async fn a_new_reader_set_resyncs_first() {
        let h = hub();
        assert_eq!(
            h.wait(&r(&["a"]), Duration::from_secs(5)).await,
            Wake::Resync
        );
        assert_eq!(
            h.wait(&r(&["a"]), Duration::from_secs(5)).await,
            Wake::Timeout
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_burst_is_one_wake() {
        let h = hub();
        h.wait(&r(&["a"]), Duration::ZERO).await; // register
        let h2 = h.clone();
        let waiter =
            tokio::spawn(async move { h2.wait(&r(&["a"]), Duration::from_secs(60)).await });
        tokio::task::yield_now().await;
        for _ in 0..50 {
            h.ring(r(&["a@1"]), None);
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(
            waiter.await.unwrap(),
            Wake::Mail {
                count: 50,
                recipients: r(&["a@1"]),
                origins: vec![]
            }
        );
    }

    #[tokio::test(start_paused = true)]
    async fn rings_while_away_are_delivered_at_the_next_wait() {
        let h = hub();
        h.wait(&r(&["a@1"]), Duration::ZERO).await;
        h.ring(r(&["a@1", "b"]), None);
        assert_eq!(
            h.wait(&r(&["a@1"]), Duration::from_secs(5)).await,
            Wake::Mail {
                count: 1,
                recipients: r(&["a@1"]),
                origins: vec![]
            }
        );
        assert_eq!(
            h.wait(&r(&["a@1"]), Duration::from_secs(1)).await,
            Wake::Timeout,
            "one wake per batch"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn only_readers_that_see_the_recipient_wake() {
        let h = hub();
        for set in [&["a"][..], &["a@1"], &["a@2"], &["b"]] {
            h.wait(&r(set), Duration::ZERO).await;
        }
        h.ring(r(&["a@1"]), None);
        assert!(
            matches!(
                h.wait(&r(&["a"]), Duration::from_secs(1)).await,
                Wake::Mail { .. }
            ),
            "bare type fans out"
        );
        assert!(matches!(
            h.wait(&r(&["a@1"]), Duration::from_secs(1)).await,
            Wake::Mail { .. }
        ));
        assert_eq!(
            h.wait(&r(&["a@2"]), Duration::from_secs(1)).await,
            Wake::Timeout
        );
        assert_eq!(
            h.wait(&r(&["b"]), Duration::from_secs(1)).await,
            Wake::Timeout
        );
    }

    #[tokio::test(start_paused = true)]
    async fn status_shows_pending_and_waiters() {
        let h = hub();
        h.wait(&r(&["a"]), Duration::ZERO).await;
        h.ring(r(&["a@9", "a@9"]), None);
        let s = h.status();
        assert_eq!(s["readers"]["a"]["pending"], 2);
        assert_eq!(s["readers"]["a"]["waiting"], 0);
    }

    #[tokio::test(start_paused = true)]
    async fn old_rings_age_out_of_the_log() {
        let h = Arc::new(Hub::new(Duration::from_millis(10), Duration::from_secs(60)));
        h.wait(&r(&["a"]), Duration::ZERO).await;
        h.ring(r(&["a"]), None);
        tokio::time::advance(Duration::from_secs(120)).await;
        h.ring(r(&["z"]), None); // a later ring trims the log
        assert_eq!(
            h.wait(&r(&["a"]), Duration::from_secs(1)).await,
            Wake::Timeout
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_wake_names_the_nodes_that_rang() {
        let h = hub();
        h.wait(&r(&["a"]), Duration::ZERO).await;
        h.ring(r(&["a@1"]), Some("node-b"));
        h.ring(r(&["a@2"]), None);
        h.ring(r(&["a@1"]), Some("node-b"));
        assert_eq!(
            h.wait(&r(&["a"]), Duration::from_secs(1)).await,
            Wake::Mail {
                count: 3,
                recipients: r(&["a@1", "a@2"]),
                origins: r(&["node-b"])
            }
        );
    }
}
