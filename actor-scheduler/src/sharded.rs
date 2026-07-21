//! Sharded SPSC inbox — multiple producers, each with a dedicated SPSC channel.
//!
//! Instead of N producers contending on one MPSC channel, each producer gets
//! its own SPSC ring buffer to the consumer. The consumer drains all shards
//! with round-robin fairness to prevent any single producer from starving others.
//!
//! # Shuffle-shard analogy
//!
//! Like shuffle sharding in the k8s API server, each producer is isolated:
//! a noisy producer can fill its own shard but cannot affect other producers'
//! ability to deliver messages. The topology is fixed at initialization time.
//!
//! # Lifecycle
//!
//! ```text
//! 1. InboxBuilder::new(capacity)       — create builder
//! 2. builder.add_producer()            — returns SpscSender, can call N times
//! 3. builder.build()                   — seals the registry, returns ShardedInbox
//! 4. inbox.drain(limit, handler)       — consumer polls all shards
//! ```
//!
//! No producers can be added after `build()`. This is the "register at init"
//! constraint — you must know all producers before the scheduler starts.

use crate::HandlerResult;
pub use crate::error::DrainStatus;
use crate::spsc::{self, SpscReceiver, SpscSender, TryRecvError};

/// Builder for a sharded inbox. Add producers, then seal with `build()`.
pub struct InboxBuilder<T> {
    receivers: Vec<SpscReceiver<T>>,
    capacity: usize,
}

impl<T> InboxBuilder<T> {
    /// Create a new builder. Each producer's SPSC channel will have
    /// at least `capacity` slots (rounded up to next power of 2).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            receivers: Vec::new(),
            capacity,
        }
    }

    /// Register a new producer. Returns the sender end of a dedicated SPSC channel.
    ///
    /// Call this once per producer during initialization, before `build()`.
    pub fn add_producer(&mut self) -> SpscSender<T> {
        let (tx, rx) = spsc::spsc_channel(self.capacity);
        self.receivers.push(rx);
        tx
    }

    /// Seal the registry and return the sharded inbox.
    ///
    /// No more producers can be added after this call.
    /// Panics if no producers were registered.
    #[must_use]
    pub fn build(self) -> ShardedInbox<T> {
        assert!(
            !self.receivers.is_empty(),
            "ShardedInbox requires at least one producer"
        );
        ShardedInbox {
            shards: self.receivers,
            round_robin: 0,
        }
    }
}

/// Consumer-side sharded inbox. Holds N SPSC receivers and drains them fairly.
pub struct ShardedInbox<T> {
    shards: Vec<SpscReceiver<T>>,
    /// Starting index for round-robin. Rotated after each drain cycle
    /// to prevent the first shard from always getting priority.
    round_robin: usize,
}

impl<T> ShardedInbox<T> {
    /// Drain messages from all shards up to a total `limit`.
    ///
    /// Uses round-robin across shards: each shard gets drained up to
    /// `per_shard` messages (limit / num_shards, minimum 1), then we
    /// rotate the starting shard for fairness.
    ///
    /// A `limit` of 0 is clamped to 1: every drain must be able to make
    /// progress. The scheduler's exit condition (all lanes observed
    /// disconnected) and its wake loop (`More` means come back) both rely
    /// on shards actually being polled — a zero-budget drain can prove
    /// neither, so it would either strand queued messages or spin forever.
    ///
    /// Returns:
    /// - `Ok(DrainStatus::Empty)` — all shards empty
    /// - `Ok(DrainStatus::More)` — hit limit, more messages may exist
    /// - `Ok(DrainStatus::Disconnected)` — all producers dropped
    /// - `Err(HandlerError)` — handler failed
    pub fn drain(
        &mut self,
        limit: usize,
        mut handler: impl FnMut(T) -> HandlerResult,
    ) -> Result<DrainStatus, crate::HandlerError> {
        let limit = limit.max(1);
        let n = self.shards.len();
        let per_shard = (limit / n).max(1);
        let mut total = 0usize;
        let mut all_empty = true;
        let mut all_disconnected = true;

        for i in 0..n {
            let idx = (self.round_robin + i) % n;
            let shard = &mut self.shards[idx];
            let mut shard_count = 0;

            loop {
                if total >= limit || shard_count >= per_shard {
                    // Hit per-shard or total limit — there might be more.
                    // This shard was NOT observed disconnected: we stopped
                    // before polling it (again), so it must not count toward
                    // all_disconnected. Otherwise a drain that exhausts its
                    // limit early (or a limit of 0) reports Disconnected with
                    // live producers, and the scheduler shuts the actor down.
                    all_empty = false;
                    all_disconnected = false;
                    break;
                }

                match shard.try_recv() {
                    Ok(msg) => {
                        handler(msg)?;
                        shard_count += 1;
                        total += 1;
                        all_disconnected = false;
                    }
                    Err(TryRecvError::Empty) => {
                        all_disconnected = false;
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        break;
                    }
                }
            }
        }

        // Rotate starting shard for next drain
        self.round_robin = (self.round_robin + 1) % n;

        match (all_disconnected, total >= limit, all_empty) {
            (true, _, _) => Ok(DrainStatus::Disconnected),
            (false, false, true) => Ok(DrainStatus::Empty),
            _ => Ok(DrainStatus::More),
        }
    }

    /// Number of registered shards (producers).
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HandlerError;

    #[test]
    fn basic_sharded_drain() {
        let mut builder = InboxBuilder::new(16);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        tx1.try_send(1).unwrap();
        tx1.try_send(2).unwrap();
        tx2.try_send(10).unwrap();
        tx2.try_send(20).unwrap();

        let mut received = Vec::new();
        let status = inbox
            .drain(100, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        assert_eq!(status, DrainStatus::Empty);
        assert_eq!(received.len(), 4);
        // All messages received (order depends on round-robin start)
        received.sort();
        assert_eq!(received, vec![1, 2, 10, 20]);
    }

    #[test]
    fn burst_limit_respected() {
        let mut builder = InboxBuilder::new(64);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0..50 {
            tx.try_send(i).unwrap();
        }

        let mut count = 0;
        let status = inbox
            .drain(10, |_msg| {
                count += 1;
                Ok(())
            })
            .unwrap();

        assert_eq!(count, 10);
        assert_eq!(status, DrainStatus::More);
    }

    #[test]
    fn round_robin_fairness() {
        let mut builder = InboxBuilder::new(64);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        // Producer 1 floods, producer 2 sends one
        for i in 0..50 {
            tx1.try_send(i).unwrap();
        }
        tx2.try_send(100).unwrap();

        // With limit=4 and 2 shards, per_shard=2
        let mut received = Vec::new();
        inbox
            .drain(4, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        // Producer 2's message should appear (not starved by producer 1)
        assert!(
            received.contains(&100),
            "Producer 2 was starved! Got: {:?}",
            received
        );
    }

    #[test]
    fn all_producers_disconnect() {
        let mut builder = InboxBuilder::new(16);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        drop(tx1);
        drop(tx2);

        let status = inbox.drain(100, |_: u32| Ok(())).unwrap();
        assert_eq!(status, DrainStatus::Disconnected);
    }

    #[test]
    fn drain_buffered_after_disconnect() {
        let mut builder = InboxBuilder::new(16);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        tx.try_send(42).unwrap();
        drop(tx);

        let mut received = Vec::new();
        inbox
            .drain(100, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        assert_eq!(received, vec![42]);
        // First drain gets the message; the shard reports Disconnected but
        // we still got data, so it's not all-disconnected yet
        // Second drain should show disconnected
        let status2 = inbox.drain(100, |_: u32| Ok(())).unwrap();
        assert_eq!(status2, DrainStatus::Disconnected);
    }

    #[test]
    fn handler_error_propagates() {
        let mut builder = InboxBuilder::new(16);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        let result = inbox.drain(100, |msg: u32| {
            if msg == 1 {
                Err(HandlerError::fatal("boom"))
            } else {
                Ok(())
            }
        });

        assert!(result.is_err());
    }

    // Regression: a zero-budget drain used to skip every shard while leaving
    // all_disconnected vacuously true, reporting Disconnected with live
    // producers — and the scheduler treats Disconnected-on-all-lanes as
    // "actor is done", silently dropping queued messages. The limit is now
    // clamped to 1 so a drain always makes progress and only reports what it
    // actually observed.
    #[test]
    fn zero_limit_drain_makes_progress_and_reports_honestly() {
        let mut builder = InboxBuilder::new(8);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        tx.try_send(42u32).unwrap();

        let mut got = Vec::new();
        let status = inbox
            .drain(0, |msg: u32| {
                got.push(msg);
                Ok(())
            })
            .unwrap();
        assert_ne!(
            status,
            DrainStatus::Disconnected,
            "producer is alive — reporting Disconnected is unsound"
        );
        assert_eq!(got, vec![42], "clamped drain should deliver the message");

        // Once the producer drops and the queue is empty, Disconnected is the
        // honest answer even at limit 0.
        drop(tx);
        let status = inbox.drain(0, |_msg: u32| Ok(())).unwrap();
        assert_eq!(status, DrainStatus::Disconnected);
    }

    #[test]
    #[should_panic(expected = "at least one producer")]
    fn panics_on_empty_build() {
        let builder = InboxBuilder::<u32>::new(16);
        let _inbox = builder.build();
    }

    // Kills: replace shard_count -> usize with 0 (line 159)
    // Kills: replace shard_count -> usize with 1 (line 159)
    #[test]
    fn shard_count_returns_number_of_registered_producers() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let _tx1 = builder.add_producer();
        let _tx2 = builder.add_producer();
        let _tx3 = builder.add_producer();
        let inbox = builder.build();

        assert_eq!(
            inbox.shard_count(),
            3,
            "Should have 3 shards for 3 producers"
        );
        assert_ne!(inbox.shard_count(), 0);
        assert_ne!(inbox.shard_count(), 1);
    }

    #[test]
    fn shard_count_is_one_for_single_producer() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let _tx = builder.add_producer();
        let inbox = builder.build();
        assert_eq!(inbox.shard_count(), 1);
    }

    // Kills: replace || with && in condition on line 109 (total >= limit || shard_count >= per_shard)
    // With &&: both conditions must be true to stop, so per-shard limit is effectively ignored
    // unless total limit is also reached.
    #[test]
    fn per_shard_limit_enforced_independently_of_total() {
        // 2 shards, limit=4, per_shard=2. Each shard should drain at most 2 messages.
        // Shard 1 has 10, Shard 2 has 10. Without per-shard limit, one shard could drain 4.
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0u32..10 {
            tx1.try_send(i).unwrap();
            tx2.try_send(i + 100).unwrap();
        }

        let mut from_shard1 = 0usize;
        let mut from_shard2 = 0usize;
        inbox
            .drain(4, |msg: u32| {
                if msg < 100 {
                    from_shard1 += 1;
                } else {
                    from_shard2 += 1;
                }
                Ok(())
            })
            .unwrap();

        // Each shard should contribute at most per_shard = 4/2 = 2 messages
        assert!(
            from_shard1 <= 2,
            "Shard 1 drained {} messages, expected <= 2",
            from_shard1
        );
        assert!(
            from_shard2 <= 2,
            "Shard 2 drained {} messages, expected <= 2",
            from_shard2
        );
        assert_eq!(
            from_shard1 + from_shard2,
            4,
            "Total should equal limit of 4"
        );
    }
}
