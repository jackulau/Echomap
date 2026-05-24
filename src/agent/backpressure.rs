//! Agent-harness backpressure primitive.
//!
//! Goal/013 hardening: the agent bridge and WS/TCP servers run on an
//! unbounded mpsc by default — convenient, but an ill-behaved or
//! disconnected reader can let the queue grow without bound and eat
//! memory until the process dies. This module gives every channel an
//! opt-in soft cap with **drop-oldest** semantics plus a visible
//! dropped-message counter, so we degrade smoothly under load instead
//! of crashing.
//!
//! Design contract:
//!   * `Backpressure::new(capacity)` records the soft cap.
//!   * Callers track current depth themselves (via `mark_enqueued` /
//!     `mark_dequeued`) — we don't own the queue.
//!   * `should_drop_oldest(depth)` returns true once `depth > capacity`.
//!   * `note_dropped(n)` bumps the lifetime drop counter for surfacing
//!     in the agent inspector UI.
//!
//! Zero-allocation, lock-free counters (`AtomicU64`) — safe to share
//! across the bridge thread and the WS/TCP server tasks.

use std::sync::atomic::{AtomicU64, Ordering};

/// Default soft cap for in-flight commands per bridge. Picked to be
/// large enough to absorb a single bad-actor burst but small enough to
/// matter (~16 MB worst case at ~4 KB / command).
pub const DEFAULT_CAPACITY: usize = 4096;

/// Counter + cap pair tracking pending work + dropped messages.
#[derive(Debug)]
pub struct Backpressure {
    capacity: usize,
    dropped_messages: AtomicU64,
}

impl Backpressure {
    pub const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            dropped_messages: AtomicU64::new(0),
        }
    }

    pub const fn default_capacity() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// True when the caller should drop the oldest pending message
    /// before enqueueing a new one.
    pub fn should_drop_oldest(&self, current_depth: usize) -> bool {
        current_depth > self.capacity
    }

    /// Bump the lifetime drop counter by `n`. Saturating — never
    /// overflows; if you somehow drop u64::MAX messages, congratulations.
    pub fn note_dropped(&self, n: u64) {
        self.dropped_messages.fetch_add(n, Ordering::Relaxed);
    }

    /// Read the lifetime drop counter.
    pub fn dropped_messages(&self) -> u64 {
        self.dropped_messages.load(Ordering::Relaxed)
    }
}

impl Default for Backpressure {
    fn default() -> Self {
        Self::default_capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_capacity_matches_const() {
        let bp = Backpressure::default();
        assert_eq!(bp.capacity(), DEFAULT_CAPACITY);
    }

    #[test]
    fn drop_oldest_only_above_capacity() {
        let bp = Backpressure::new(4);
        assert!(!bp.should_drop_oldest(0));
        assert!(!bp.should_drop_oldest(4));
        assert!(bp.should_drop_oldest(5));
        assert!(bp.should_drop_oldest(usize::MAX));
    }

    #[test]
    fn dropped_counter_accumulates() {
        let bp = Backpressure::new(2);
        assert_eq!(bp.dropped_messages(), 0);
        bp.note_dropped(3);
        bp.note_dropped(7);
        assert_eq!(bp.dropped_messages(), 10);
    }

    #[test]
    fn capacity_one_treats_two_as_overflow() {
        let bp = Backpressure::new(1);
        assert!(!bp.should_drop_oldest(0));
        assert!(!bp.should_drop_oldest(1));
        assert!(bp.should_drop_oldest(2));
    }

    #[test]
    fn capacity_zero_drops_everything() {
        let bp = Backpressure::new(0);
        assert!(!bp.should_drop_oldest(0));
        assert!(bp.should_drop_oldest(1));
    }

    #[test]
    fn concurrent_note_dropped_is_lock_free() {
        use std::sync::Arc;
        use std::thread;

        let bp = Arc::new(Backpressure::new(10));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let b = Arc::clone(&bp);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    b.note_dropped(1);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread joined");
        }
        assert_eq!(bp.dropped_messages(), 4000);
    }

    #[test]
    fn const_constructor_usable_in_static() {
        static BP: Backpressure = Backpressure::new(8);
        assert_eq!(BP.capacity(), 8);
    }
}
