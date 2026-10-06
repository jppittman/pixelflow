//! `ActorScheduler`'s own priority and FIFO ordering contract.
//!
//! `dedicated_thread.rs`'s `priority_holds_on_a_dedicated_thread_fed_by_ordinary_handles`
//! pins Control > Management > Data for the `mealy` module's `Transducer`/`Node`/`Host`
//! substrate. `ActorScheduler<D, C, M>` (`lib.rs`'s `handle_wake`) is a separate
//! implementation of the same priority contract — drained via `ShardedInbox`, not `mealy` —
//! and it is the type `core-term` actually drives directly through `Actor`/`ActorBuilder`
//! (see `core-term/tests/actor_roundtrip_tests.rs`'s `pty_writer_*` tests). Its own priority
//! order, and FIFO order within each lane, need their own direct proof rather than borrowing
//! the mealy-substrate one.

use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};

/// Records the lane and payload of every message handled, in the order `handle_wake` called
/// out to the actor.
struct OrderRecorder {
    log: Vec<String>,
}

impl Actor<u32, u32, u32> for OrderRecorder {
    fn handle_data(&mut self, msg: u32) -> HandlerResult {
        self.log.push(format!("D{msg}"));
        Ok(())
    }
    fn handle_control(&mut self, msg: u32) -> HandlerResult {
        self.log.push(format!("C{msg}"));
        Ok(())
    }
    fn handle_management(&mut self, msg: u32) -> HandlerResult {
        self.log.push(format!("M{msg}"));
        Ok(())
    }
    fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Every message across all three lanes is queued before the scheduler ever wakes to drain
/// them — priority is a property of a drain pass choosing among messages already pending, not
/// of arrival order (`handle_wake`'s own doc, and see
/// `a_control_message_enqueued_mid_drain_is_not_seen_until_the_drain_ends` in `lib.rs` for the
/// mid-drain case this deliberately does not exercise). `run` drains to completion once every
/// handle is dropped, so no sleep-based synchronization is needed.
#[test]
fn the_control_lane_drains_before_management_which_drains_before_data() {
    let (tx, mut rx) = ActorScheduler::<u32, u32, u32>::new(100, 100);

    for i in 0..3 {
        tx.send(Message::Data(i)).unwrap();
    }
    for i in 0..3 {
        tx.send(Message::Management(i)).unwrap();
    }
    for i in 0..3 {
        tx.send(Message::Control(i)).unwrap();
    }
    drop(tx);

    let mut actor = OrderRecorder { log: Vec::new() };
    rx.run(&mut actor);

    assert_eq!(
        actor.log,
        vec!["C0", "C1", "C2", "M0", "M1", "M2", "D0", "D1", "D2"],
        "control drains before management before data, and each lane stays FIFO within itself"
    );
}
