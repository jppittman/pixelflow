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

        if all_disconnected {
            Ok(DrainStatus::Disconnected)
        } else if total >= limit || !all_empty {
            // Either we hit the total limit outright, or per-shard caps
            // stopped us early with unproven shards remaining — both cases
            // report More.
            Ok(DrainStatus::More)
        } else {
            Ok(DrainStatus::Empty)
        }
    }

    /// Number of registered shards (producers).
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Take one message, round-robin across shards. The single-message counterpart to
    /// [`Self::drain`], for a [`Node`](crate::mealy::Node) — which takes at most one input per
    /// [`poll`](crate::mealy::Node::poll) — to sit on the same multi-producer lanes an
    /// [`ActorHandle`](crate::ActorHandle) already feeds, instead of needing a second inbox type.
    ///
    /// `Disconnected` only when every shard is: same all-or-nothing rule `drain` uses, so a
    /// lane with one live producer and the rest gone still reports `Empty`, not halted.
    fn take_one(&mut self) -> Result<T, TryRecvError> {
        let n = self.shards.len();
        let mut all_disconnected = true;

        for i in 0..n {
            let idx = (self.round_robin + i) % n;
            match self.shards[idx].try_recv() {
                Ok(msg) => {
                    self.round_robin = (idx + 1) % n;
                    return Ok(msg);
                }
                Err(TryRecvError::Empty) => all_disconnected = false,
                Err(TryRecvError::Disconnected) => {}
            }
        }

        // Rotate even on a miss, for the same fairness reason `drain` rotates every call.
        self.round_robin = (self.round_robin + 1) % n;

        if all_disconnected {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }
}

impl<T> crate::mealy::Inbox for ShardedInbox<T> {
    type Item = T;

    fn take(&mut self) -> Result<T, TryRecvError> {
        self.take_one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HandlerError;

    #[test]
    fn drain_delivers_every_message_from_every_producer() {
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
    fn drain_stops_at_the_limit_and_reports_more() {
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
    fn drain_does_not_starve_a_quiet_producer_behind_a_flooding_one() {
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
    fn drain_reports_disconnected_once_every_producer_is_dropped() {
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
    fn drain_delivers_buffered_messages_after_the_producer_is_dropped() {
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
    fn drain_returns_the_error_the_handler_failed_with() {
        let mut builder = InboxBuilder::new(16);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        let result = inbox.drain(100, |msg: u32| {
            if msg == 1 {
                Err(HandlerError::new("boom"))
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
    fn build_panics_when_no_producer_was_added() {
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

    // Kills: replace || with &&, and delete !, on `total >= limit || !all_empty` (drain's
    // status-selection guard). Every other test in this module uses a `limit` evenly
    // divisible by the shard count, so `total` lands exactly on `limit` right as the last
    // shard also hits its per-shard cap — the two guard conditions become true together and
    // neither mutant is distinguishable. An indivisible limit forces per-shard caps to stop
    // every shard *below* the total limit, so `!all_empty` alone must carry the guard.
    #[test]
    fn drain_reports_more_when_per_shard_caps_leave_total_below_limit() {
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0u32..10 {
            tx1.try_send(i).unwrap();
            tx2.try_send(i + 100).unwrap();
        }

        // limit=5, 2 shards -> per_shard = 2, so at most 4 messages come out even though
        // the total limit of 5 was never reached.
        let mut received = Vec::new();
        let status = inbox
            .drain(5, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        assert_eq!(
            received.len(),
            4,
            "capped by per-shard limits, not the total"
        );
        assert_eq!(
            status,
            DrainStatus::More,
            "per-shard caps stopped early — both shards still have queued messages"
        );
    }

    // Kills: replace `total += 1` with `total *= 1` (drain's total-message counter). With
    // shard_count divisible into limit evenly, `total`'s accumulated value coincides with
    // shard_count's own per-shard cap and the mutant is equivalent — that's true for every
    // other test in this module. It stops being equivalent once shard count exceeds limit:
    // `per_shard = (limit / n).max(1)` floors to 1, so with `total` stuck at 0 the loop's
    // `total >= limit` early-exit never fires and every shard independently delivers its
    // one `per_shard`-capped message — n messages, not the requested limit.
    #[test]
    fn drain_total_stops_the_scan_once_the_limit_is_hit_even_with_more_shards_than_limit() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let tx0 = builder.add_producer();
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let tx3 = builder.add_producer();
        let mut inbox = builder.build();

        tx0.try_send(0).unwrap();
        tx1.try_send(1).unwrap();
        tx2.try_send(2).unwrap();
        tx3.try_send(3).unwrap();

        // 4 shards, limit=2: per_shard = (2/4).max(1) = 1, so nothing but `total` itself can
        // stop the scan before it reaches the third and fourth shards.
        let mut received = Vec::new();
        inbox
            .drain(2, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        assert_eq!(
            received.len(),
            2,
            "total must cut the scan off at the requested limit, not let every \
             per-shard-capped shard through: got {received:?}"
        );
    }

    // Kills: replace % with /, and replace + with *, in the round_robin rotation at the end
    // of `drain` (`self.round_robin = (self.round_robin + 1) % n`). Both mutants leave
    // `round_robin` stuck at 0 forever (since it starts at 0), which no count-based
    // assertion can see — item *order* across successive calls is the only observable
    // signal, so this drains one message per shard per call and checks which shard answers
    // first on the second call.
    #[test]
    fn drain_round_robin_start_rotates_across_calls() {
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx0 = builder.add_producer();
        let tx1 = builder.add_producer();
        let mut inbox = builder.build();

        tx0.try_send(100).unwrap();
        tx0.try_send(101).unwrap();
        tx1.try_send(200).unwrap();
        tx1.try_send(201).unwrap();

        let mut first = Vec::new();
        inbox
            .drain(2, |msg| {
                first.push(msg);
                Ok(())
            })
            .unwrap();

        let mut second = Vec::new();
        inbox
            .drain(2, |msg| {
                second.push(msg);
                Ok(())
            })
            .unwrap();

        // The contract is rotation, not a specific starting shard: which
        // producer a call started from shows up in its first message's
        // hundreds digit (100s = producer 0, 200s = producer 1), so assert
        // the two calls started from different shards rather than pinning
        // either to shard 0.
        assert_ne!(
            first[0] / 100,
            second[0] / 100,
            "the second call must start from a different shard than the first: \
             {first:?} then {second:?}"
        );
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

    // `take_one` (the `Inbox::take` impl used by `mealy::Node`'s data lane) had no test
    // coverage at all before this: every arithmetic op in its round-robin search and
    // rotation survived mutation. Reached through the public `Inbox` trait rather than
    // calling the private `take_one` directly.
    use crate::mealy::Inbox;

    // Kills: replace % with /, and replace + with *, in the search index
    // `(self.round_robin + i) % n`. Both mutants collapse the search to always recheck
    // index 0, so a call that should skip an empty leading shard and find a later one
    // instead reports Empty.
    #[test]
    fn take_skips_empty_shards_via_wrapping_search() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let _tx0 = builder.add_producer();
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        // Shard 0 (round_robin's starting point) is empty; shards 1 and 2 have data.
        tx1.try_send(20).unwrap();
        tx2.try_send(30).unwrap();

        assert_eq!(
            inbox.take(),
            Ok(20),
            "search must wrap past the empty shard 0"
        );
    }

    // Kills: replace % with /, replace + with *, on the post-hit rotation
    // (`self.round_robin = (idx + 1) % n`), and the equivalent pair on the post-miss
    // rotation (`self.round_robin = (self.round_robin + 1) % n`). Exercises a full lap
    // across 3 shards, a refill with one shard left empty (proving the search wraps from
    // the rotated start rather than always from 0), and a total-miss round-trip (proving
    // rotation still advances when nothing was found).
    #[test]
    fn take_round_robins_across_shards_and_wraps() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let tx0 = builder.add_producer();
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        tx0.try_send(10).unwrap();
        tx1.try_send(20).unwrap();
        tx2.try_send(30).unwrap();

        // One full lap must visit every shard exactly once — in whichever order
        // round_robin happens to start (the public contract promises fairness and
        // rotation, not that shard 0 answers first).
        let lap = [
            inbox.take().unwrap(),
            inbox.take().unwrap(),
            inbox.take().unwrap(),
        ];
        let mut sorted_lap = lap;
        sorted_lap.sort_unstable();
        assert_eq!(
            sorted_lap,
            [10, 20, 30],
            "one full lap must deliver every shard's message exactly once"
        );

        // Kills the post-hit rotation's `(idx + 1) % n` -> `(idx + 1) / n`: refilling only
        // some shards lets the search's own wraparound (which tries every index regardless
        // of where round_robin starts) mask an error in round_robin's exact value — with a
        // single live shard left, `take()` finds it either way. Refilling *every* shard means
        // the very first index the search tries is guaranteed to hit immediately, so which
        // value comes back first directly reveals round_robin's post-hit value. The shard
        // that answered *last* in the lap above (identified by `lap[2]`'s producer tag, not
        // an assumed absolute index) determines the correct next shard under `%`; `/`
        // predicts a different one for every possible last-shard value.
        let idx_of = |v: u32| (v / 10 - 1) as usize;
        let producers = [&tx0, &tx1, &tx2];
        let refills = [11u32, 21, 31];
        for (i, p) in producers.iter().enumerate() {
            p.try_send(refills[i]).unwrap();
        }
        let expected_next = refills[(idx_of(lap[2]) + 1) % 3];
        assert_eq!(
            inbox.take(),
            Ok(expected_next),
            "rotation must continue from the shard after the one that answered last, \
             not from an arithmetically wrong index"
        );

        let mut rest = [inbox.take().unwrap(), inbox.take().unwrap()];
        rest.sort_unstable();
        let mut expected_rest: Vec<u32> = refills
            .iter()
            .copied()
            .filter(|&v| v != expected_next)
            .collect();
        expected_rest.sort_unstable();
        assert_eq!(
            rest,
            expected_rest.as_slice(),
            "the remaining two shards, in either order"
        );
    }

    // Kills: replace + with * on the post-miss rotation line, isolated from any lane search
    // mutation by using a total miss (every shard empty) as the rotation trigger, then
    // proving which shard answers first afterward.
    #[test]
    fn take_rotates_the_start_index_even_on_a_total_miss() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let tx0 = builder.add_producer();
        let tx1 = builder.add_producer();
        let mut inbox = builder.build();

        assert_eq!(
            inbox.take(),
            Err(TryRecvError::Empty),
            "both shards start empty"
        );

        tx0.try_send(100).unwrap();
        tx1.try_send(200).unwrap();
        assert_eq!(
            inbox.take(),
            Ok(200),
            "the miss still rotated the start index past shard 0"
        );
    }
}
