//! Comprehensive tests for the actor model in pixelflow-runtime.
//!
//! These tests verify the core contracts of the actor scheduler:
//! - Priority ordering (Control > Management > Data)
//! - Burst limiting for Data and Management lanes
//! - Backpressure behavior for bounded channels
//! - Channel disconnection handling
//! - ActorStatus behavior
//! - Message ordering guarantees (FIFO within lanes)
//! - Thread safety and concurrent access
//!
//! Following STYLE.md: tests focus on public API contracts, not implementation details.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Simple actor that logs all received messages with timestamps for ordering verification.
struct OrderingActor {
    log: Arc<Mutex<Vec<(String, std::time::Instant)>>>,
}

impl Actor<String, String, String> for OrderingActor {
    fn handle_data(&mut self, msg: String) -> HandlerResult {
        self.log
            .lock()
            .unwrap()
            .push((format!("D:{}", msg), std::time::Instant::now()));
        Ok(())
    }

    fn handle_control(&mut self, msg: String) -> HandlerResult {
        self.log
            .lock()
            .unwrap()
            .push((format!("C:{}", msg), std::time::Instant::now()));
        Ok(())
    }

    fn handle_management(&mut self, msg: String) -> HandlerResult {
        self.log
            .lock()
            .unwrap()
            .push((format!("M:{}", msg), std::time::Instant::now()));
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Actor that counts messages per lane.
struct CountingActor {
    data_count: AtomicUsize,
    control_count: AtomicUsize,
    management_count: AtomicUsize,
}

impl CountingActor {
    fn new() -> Self {
        Self {
            data_count: AtomicUsize::new(0),
            control_count: AtomicUsize::new(0),
            management_count: AtomicUsize::new(0),
        }
    }

    fn data_count(&self) -> usize {
        self.data_count.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    fn control_count(&self) -> usize {
        self.control_count.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    fn management_count(&self) -> usize {
        self.management_count.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    fn total(&self) -> usize {
        self.data_count() + self.control_count() + self.management_count()
    }
}

impl Actor<i32, i32, i32> for CountingActor {
    fn handle_data(&mut self, _msg: i32) -> HandlerResult {
        self.data_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn handle_control(&mut self, _msg: i32) -> HandlerResult {
        self.control_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn handle_management(&mut self, _msg: i32) -> HandlerResult {
        self.management_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Actor that can block processing to simulate slow handlers.
struct SlowActor {
    delay: Duration,
    processed: Arc<Mutex<Vec<String>>>,
}

impl Actor<String, String, String> for SlowActor {
    fn handle_data(&mut self, msg: String) -> HandlerResult {
        thread::sleep(self.delay);
        self.processed.lock().unwrap().push(format!("D:{}", msg));
        Ok(())
    }

    fn handle_control(&mut self, msg: String) -> HandlerResult {
        thread::sleep(self.delay);
        self.processed.lock().unwrap().push(format!("C:{}", msg));
        Ok(())
    }

    fn handle_management(&mut self, msg: String) -> HandlerResult {
        thread::sleep(self.delay);
        self.processed.lock().unwrap().push(format!("M:{}", msg));
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Actor that tracks SystemStatus hints from the scheduler.
struct ParkTrackingActor {
    park_hints: Arc<Mutex<Vec<SystemStatus>>>,
    return_status: ActorStatus,
}

impl Actor<(), (), ()> for ParkTrackingActor {
    fn handle_data(&mut self, _msg: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _msg: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _msg: ()) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        self.park_hints.lock().unwrap().push(status);
        Ok(self.return_status)
    }
}

// ============================================================================
// Priority Ordering Tests
// ============================================================================

#[test]
fn control_messages_processed_before_earlier_data_messages() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let handle = thread::spawn(move || {
        let mut actor = OrderingActor { log: log_clone };
        rx.run(&mut actor);
    });

    // Send data first, then control
    tx.send(Message::Data("first".to_string())).unwrap();
    tx.send(Message::Data("second".to_string())).unwrap();
    tx.send(Message::Control("priority".to_string())).unwrap();
    tx.send(Message::Data("third".to_string())).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let messages = log.lock().unwrap();
    let ctrl_idx = messages.iter().position(|(s, _)| s.starts_with("C:"));
    let first_data_idx = messages
        .iter()
        .position(|(s, _)| s == "D:first")
        .unwrap_or(usize::MAX);

    // Control should be processed before any data that arrived before it
    assert!(
        ctrl_idx.unwrap() < first_data_idx,
        "Control message should be processed before earlier data messages. Got: {:?}",
        messages.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );
}

#[test]
fn management_messages_processed_before_data_messages() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let handle = thread::spawn(move || {
        let mut actor = OrderingActor { log: log_clone };
        rx.run(&mut actor);
    });

    // Send data, then management
    tx.send(Message::Data("data1".to_string())).unwrap();
    tx.send(Message::Management("mgmt".to_string())).unwrap();
    tx.send(Message::Data("data2".to_string())).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let messages = log.lock().unwrap();
    let mgmt_idx = messages.iter().position(|(s, _)| s.starts_with("M:"));
    let data1_idx = messages
        .iter()
        .position(|(s, _)| s == "D:data1")
        .unwrap_or(usize::MAX);

    assert!(
        mgmt_idx.unwrap() < data1_idx,
        "Management should be processed before earlier data. Got: {:?}",
        messages.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );
}

#[test]
fn control_processed_before_management() {
    // This test verifies that when control and management messages are both
    // queued, control is processed first. We check which one triggers processing
    // FIRST by having the actor track the first message of each type it sees.
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let control_first = Arc::new(AtomicBool::new(false));
    let management_first = Arc::new(AtomicBool::new(false));
    let control_first_clone = control_first.clone();
    let management_first_clone = management_first.clone();

    let handle = thread::spawn(move || {
        struct FirstWinsActor {
            control_first: Arc<AtomicBool>,
            management_first: Arc<AtomicBool>,
            control_seen: bool,
            management_seen: bool,
        }
        impl Actor<(), (), ()> for FirstWinsActor {
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                if !self.control_seen && !self.management_seen {
                    self.control_first.store(true, Ordering::SeqCst);
                }
                self.control_seen = true;
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                if !self.control_seen && !self.management_seen {
                    self.management_first.store(true, Ordering::SeqCst);
                }
                self.management_seen = true;
                Ok(())
            }
            fn handle_data(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        let mut actor = FirstWinsActor {
            control_first: control_first_clone,
            management_first: management_first_clone,
            control_seen: false,
            management_seen: false,
        };
        rx.run(&mut actor);
    });

    // Send both control and management rapidly to ensure they're queued before processing
    tx.send(Message::Management(())).unwrap();
    tx.send(Message::Control(())).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    assert!(
        control_first.load(Ordering::SeqCst),
        "Control message should be processed before management message"
    );
}

#[test]
fn fifo_ordering_within_same_lane() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let handle = thread::spawn(move || {
        let mut actor = OrderingActor { log: log_clone };
        rx.run(&mut actor);
    });

    // Send multiple data messages
    for i in 0..10 {
        tx.send(Message::Data(format!("{}", i))).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let messages = log.lock().unwrap();
    let data_msgs: Vec<_> = messages
        .iter()
        .filter(|(s, _)| s.starts_with("D:"))
        .map(|(s, _)| s.strip_prefix("D:").unwrap().parse::<i32>().unwrap())
        .collect();

    // Verify FIFO ordering
    for i in 0..data_msgs.len() - 1 {
        assert!(
            data_msgs[i] < data_msgs[i + 1],
            "FIFO violated: {} should come before {}",
            data_msgs[i],
            data_msgs[i + 1]
        );
    }
}

// ============================================================================
// Robustness Tests
// ============================================================================

#[test]
fn mixed_priority_messages_all_delivered() {
    // Verify that mixing different priority messages doesn't cause loss or corruption.
    // This prevents bugs where priority switching could skip messages.
    let (tx, mut rx) = ActorScheduler::new(100, 1000);
    let counts = Arc::new((
        AtomicUsize::new(0), // data
        AtomicUsize::new(0), // control
        AtomicUsize::new(0), // management
    ));
    let counts_clone = counts.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<(AtomicUsize, AtomicUsize, AtomicUsize)>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0 .0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                self.0 .1.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                self.0 .2.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(counts_clone));
    });

    // Interleave different priority messages
    for i in 0..100 {
        match i % 3 {
            0 => tx.send(Message::Data(i)).unwrap(),
            1 => tx.send(Message::Control(i)).unwrap(),
            _ => tx.send(Message::Management(i)).unwrap(),
        }
    }

    // Small delay to ensure messages start processing
    thread::sleep(Duration::from_millis(10));
    drop(tx);
    handle.join().unwrap();

    // All messages must be delivered (33 or 34 of each type)
    let (data, ctrl, mgmt) = (
        counts.0.load(Ordering::SeqCst),
        counts.1.load(Ordering::SeqCst),
        counts.2.load(Ordering::SeqCst),
    );
    let total = data + ctrl + mgmt;

    assert_eq!(
        total, 100,
        "All 100 messages must be processed. Got data={}, ctrl={}, mgmt={}",
        data, ctrl, mgmt
    );
}

#[test]
fn no_starvation_with_continuous_high_priority() {
    // Verify that lower priority messages eventually get processed even when
    // high priority messages keep arriving. This prevents starvation bugs.
    let (tx, mut rx) = ActorScheduler::new(50, 500);
    let data_processed = Arc::new(AtomicBool::new(false));
    let data_clone = data_processed.clone();

    let handle = thread::spawn(move || {
        struct Tracker {
            data_processed: Arc<AtomicBool>,
        }
        impl Actor<(), (), ()> for Tracker {
            fn handle_data(&mut self, _: ()) -> HandlerResult {
                self.data_processed.store(true, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                // High priority, but shouldn't starve data
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Tracker {
            data_processed: data_clone,
        });
    });

    // Send one data message
    tx.send(Message::Data(())).unwrap();

    // Flood with control messages
    for _ in 0..100 {
        tx.send(Message::Control(())).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Data message must have been processed despite control flood
    assert!(
        data_processed.load(Ordering::SeqCst),
        "Data message must not be starved by control messages"
    );
}

#[test]
fn management_burst_limit_prevents_starvation() {
    // Management has a burst limit (128), but this doesn't mean control
    // will be checked "in the middle" of processing queued messages.
    // It means processing is limited per wake cycle, allowing the scheduler
    // to loop and check higher priority lanes.
    //
    // This test verifies that all messages are eventually processed
    // and the system doesn't deadlock with lots of management messages.

    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<String, String, String> for Counter {
            fn handle_data(&mut self, _: String) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: String) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_management(&mut self, _: String) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Flood management lane
    for i in 0..300 {
        tx.send(Message::Management(format!("M{}", i))).unwrap();
    }
    // Add some control messages
    for i in 0..10 {
        tx.send(Message::Control(format!("C{}", i))).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // All 310 messages should be processed
    assert_eq!(
        count.load(Ordering::SeqCst),
        310,
        "All messages should be processed despite burst limiting"
    );
}

// ============================================================================
// Backpressure Tests
// ============================================================================

#[test]
fn data_lane_blocks_when_buffer_full() {
    // Very small buffer to trigger backpressure
    let mut builder = ActorBuilder::<String, String, String>::new(2, None);
    let tx = builder.add_producer();
    let tx_clone = builder.add_producer();
    let mut rx = builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());
    let processed = Arc::new(Mutex::new(Vec::new()));
    let processed_clone = processed.clone();
    let send_complete = Arc::new(AtomicBool::new(false));
    let send_complete_clone = send_complete.clone();

    // Start actor thread
    let handle = thread::spawn(move || {
        // Slow actor to ensure buffer fills
        let mut actor = SlowActor {
            delay: Duration::from_millis(50),
            processed: processed_clone,
        };
        rx.run(&mut actor);
    });

    // Sender thread
    let sender = thread::spawn(move || {
        for i in 0..5 {
            tx_clone.send(Message::Data(format!("{}", i))).unwrap();
        }
        send_complete_clone.store(true, Ordering::SeqCst);
    });

    // Wait a bit - sender should block on backpressure
    thread::sleep(Duration::from_millis(30));

    // Eventually everything completes
    sender.join().unwrap();
    drop(tx);
    handle.join().unwrap();

    assert!(send_complete.load(Ordering::SeqCst));
    assert_eq!(processed.lock().unwrap().len(), 5);
}

#[test]
fn multiple_senders_all_messages_delivered() {
    let num_senders = 10;
    let msgs_per_sender = 100;

    let mut builder = ActorBuilder::<i32, i32, i32>::new(1000, None);
    // Create one producer per sender thread, plus one "original" to drop at the end
    let sender_handles: Vec<_> = (0..num_senders).map(|_| builder.add_producer()).collect();
    let tx = builder.add_producer();
    let mut rx = builder.build_with_burst(100, actor_scheduler::ShutdownMode::default());

    let actor = Arc::new(CountingActor::new());

    // Clone actor reference for handler
    let actor_for_handler = Arc::clone(&actor);

    let handle = thread::spawn(move || {
        // Create a wrapper that uses the Arc
        struct ArcCountingActor(Arc<CountingActor>);
        impl Actor<i32, i32, i32> for ArcCountingActor {
            fn handle_data(&mut self, _msg: i32) -> HandlerResult {
                self.0.data_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _msg: i32) -> HandlerResult {
                self.0.control_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_management(&mut self, _msg: i32) -> HandlerResult {
                self.0.management_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        let mut wrapper = ArcCountingActor(actor_for_handler);
        rx.run(&mut wrapper);
    });

    let barrier = Arc::new(Barrier::new(num_senders));

    let senders: Vec<_> = sender_handles
        .into_iter()
        .map(|tx| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..msgs_per_sender {
                    tx.send(Message::Data(i as i32)).unwrap();
                }
            })
        })
        .collect();

    for s in senders {
        s.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        actor.data_count(),
        num_senders * msgs_per_sender,
        "All messages from all senders should be delivered"
    );
}

// ============================================================================
// Channel Disconnection Tests
// ============================================================================

#[test]
fn actor_run_exits_when_all_senders_dropped() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);
    let exited = Arc::new(AtomicBool::new(false));
    let exited_clone = exited.clone();

    let handle = thread::spawn(move || {
        struct NoopActor;
        impl Actor<(), (), ()> for NoopActor {
            fn handle_data(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut NoopActor);
        exited_clone.store(true, Ordering::SeqCst);
    });

    // Ensure actor is running
    tx.send(Message::Control(())).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert!(
        !exited.load(Ordering::SeqCst),
        "Actor should still be running"
    );

    // Drop sender
    drop(tx);
    handle.join().unwrap();
    assert!(exited.load(Ordering::SeqCst), "Actor should have exited");
}

#[test]
fn cloned_handle_works_after_original_dropped() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
    let tx = builder.add_producer();
    let tx2 = builder.add_producer();
    let mut rx = builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());

    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct CounterActor(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for CounterActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut CounterActor(count_clone));
    });

    // Send from original
    tx.send(Message::Data(1)).unwrap();
    // Drop original
    drop(tx);

    // Clone should still work
    thread::sleep(Duration::from_millis(20));
    tx2.send(Message::Data(2)).unwrap();

    thread::sleep(Duration::from_millis(20));
    drop(tx2);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 2);
}

// ============================================================================
// ActorStatus Tests
// ============================================================================

#[test]
fn park_hint_wait_when_queues_empty() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let hints = Arc::new(Mutex::new(Vec::new()));
    let hints_clone = hints.clone();

    let handle = thread::spawn(move || {
        let mut actor = ParkTrackingActor {
            park_hints: hints_clone,
            return_status: ActorStatus::Idle,
        };
        rx.run(&mut actor);
    });

    // Send one message to wake actor
    tx.send(Message::Data(())).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let hints = hints.lock().unwrap();
    // After processing the single message, queues are empty -> Idle SystemStatus
    assert!(
        hints.contains(&SystemStatus::Idle),
        "Should have Idle SystemStatus when queues empty. Hints: {:?}",
        hints
    );
}

#[test]
fn park_hint_poll_when_burst_limit_hit() {
    // Use tiny burst limits
    let (tx, mut rx) = ActorScheduler::new(2, 100);
    let hints = Arc::new(Mutex::new(Vec::new()));
    let hints_clone = hints.clone();

    let handle = thread::spawn(move || {
        let mut actor = ParkTrackingActor {
            park_hints: hints_clone,
            return_status: ActorStatus::Idle,
        };
        rx.run(&mut actor);
    });

    // Send more than burst limit
    for _ in 0..10 {
        tx.send(Message::Data(())).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let hints = hints.lock().unwrap();
    // Should have Busy SystemStatus when burst limit is hit (more work available)
    assert!(
        hints.contains(&SystemStatus::Busy),
        "Should have Busy SystemStatus when burst limit hit. Hints: {:?}",
        hints
    );
}

#[test]
fn actor_can_override_park_hint_to_poll() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let hints = Arc::new(Mutex::new(Vec::new()));
    let hints_clone = hints.clone();

    let handle = thread::spawn(move || {
        let mut actor = ParkTrackingActor {
            park_hints: hints_clone,
            return_status: ActorStatus::Busy,
        };
        rx.run(&mut actor);
    });

    tx.send(Message::Data(())).unwrap();

    // Give it time to loop with Poll
    thread::sleep(Duration::from_millis(30));
    drop(tx);
    handle.join().unwrap();

    let hints = hints.lock().unwrap();
    // Actor returned Poll, so scheduler should keep working
    // This means park() gets called multiple times
    assert!(
        hints.len() > 1,
        "Actor overriding to Poll should cause multiple park() calls. Got: {}",
        hints.len()
    );
}

// ============================================================================
// Message Type Tests
// ============================================================================

#[test]
fn message_enum_variants_distinguishable() {
    let data: Message<i32, String, bool> = Message::Data(42);
    let ctrl: Message<i32, String, bool> = Message::Control("test".to_string());
    let mgmt: Message<i32, String, bool> = Message::Management(true);

    assert!(matches!(data, Message::Data(42)));
    assert!(matches!(ctrl, Message::Control(ref s) if s == "test"));
    assert!(matches!(mgmt, Message::Management(true)));
}

#[test]
fn different_message_types_per_lane() {
    use std::collections::HashMap;

    // Verify that each lane can have completely different types
    let (tx, mut rx) =
        ActorScheduler::<Vec<u8>, HashMap<String, i32>, std::time::Duration>::new(10, 100);

    let received = Arc::new(Mutex::new((false, false, false)));
    let received_clone = received.clone();

    let handle = thread::spawn(move || {
        struct TypedActor(Arc<Mutex<(bool, bool, bool)>>);
        impl Actor<Vec<u8>, HashMap<String, i32>, std::time::Duration> for TypedActor {
            fn handle_data(&mut self, _: Vec<u8>) -> HandlerResult {
                self.0.lock().unwrap().0 = true;
                Ok(())
            }
            fn handle_control(&mut self, _: HashMap<String, i32>) -> HandlerResult {
                self.0.lock().unwrap().1 = true;
                Ok(())
            }
            fn handle_management(&mut self, _: std::time::Duration) -> HandlerResult {
                self.0.lock().unwrap().2 = true;
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut TypedActor(received_clone));
    });

    tx.send(Message::Data(vec![1, 2, 3])).unwrap();
    tx.send(Message::Control(HashMap::new())).unwrap();
    tx.send(Message::Management(Duration::from_secs(1)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let r = received.lock().unwrap();
    assert!(r.0 && r.1 && r.2, "All three lane types should work");
}

// ============================================================================
// Handle Cloning Tests
// ============================================================================

#[test]
fn handle_clone_is_independent() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
    let tx = builder.add_producer();
    let tx2 = builder.add_producer();
    let tx3 = builder.add_producer();
    let mut rx = builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());

    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Send from all handles
    tx.send(Message::Data(1)).unwrap();
    tx2.send(Message::Data(2)).unwrap();
    tx3.send(Message::Data(3)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    drop(tx2);
    drop(tx3);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[test]
fn handle_debug_impl_works() {
    let (tx, _rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
    let debug_str = format!("{:?}", tx);
    assert!(
        debug_str.contains("ActorHandle"),
        "Debug should include type name"
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn high_throughput_single_sender() {
    let (tx, mut rx) = ActorScheduler::new(1000, 10000);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    let num_messages = 50_000;
    for i in 0..num_messages {
        tx.send(Message::Data(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(200));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        count.load(Ordering::SeqCst),
        num_messages as usize,
        "All messages should be processed"
    );
}

#[test]
fn concurrent_senders_stress_test() {
    let num_senders = 20;
    let msgs_per_sender = 1000;

    let mut builder = ActorBuilder::<i32, i32, i32>::new(1000, None);
    let sender_handles: Vec<_> = (0..num_senders).map(|_| builder.add_producer()).collect();
    let tx = builder.add_producer();
    let mut rx = builder.build_with_burst(100, actor_scheduler::ShutdownMode::default());

    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    let barrier = Arc::new(Barrier::new(num_senders));

    let senders: Vec<_> = sender_handles
        .into_iter()
        .enumerate()
        .map(|(sender_id, tx)| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..msgs_per_sender {
                    let msg_type = i % 3;
                    match msg_type {
                        0 => tx.send(Message::Data(sender_id as i32)).unwrap(),
                        1 => tx.send(Message::Control(sender_id as i32)).unwrap(),
                        _ => tx.send(Message::Management(sender_id as i32)).unwrap(),
                    }
                }
            })
        })
        .collect();

    for s in senders {
        s.join().unwrap();
    }

    thread::sleep(Duration::from_millis(200));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        count.load(Ordering::SeqCst),
        num_senders * msgs_per_sender,
        "All messages from all senders should be processed"
    );
}

#[test]
fn priority_maintained_when_both_lanes_have_messages() {
    // This test verifies that when BOTH control and data are queued,
    // control is processed first. It does NOT test that control sent LATER
    // interrupts data processing - that's not how the scheduler works.
    //
    // The scheduler checks priority at the start of each batch, so if
    // data is sent first and control second, some data may be processed
    // before control arrives. This is expected.

    let (tx, mut rx) = ActorScheduler::new(50, 500);
    let first_message_type = Arc::new(Mutex::new(None::<&'static str>));

    let first_clone = first_message_type.clone();

    let handle = thread::spawn(move || {
        struct FirstChecker {
            first: Arc<Mutex<Option<&'static str>>>,
        }
        impl Actor<i32, i32, i32> for FirstChecker {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                let mut first = self.first.lock().unwrap();
                if first.is_none() {
                    *first = Some("data");
                }
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                let mut first = self.first.lock().unwrap();
                if first.is_none() {
                    *first = Some("control");
                }
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut FirstChecker { first: first_clone });
    });

    // Send both control and data before scheduler processes anything
    // by sending them quickly
    tx.send(Message::Data(1)).unwrap();
    tx.send(Message::Control(1)).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Control should be processed first since it has higher priority
    let first = first_message_type.lock().unwrap();
    assert_eq!(
        *first,
        Some("control"),
        "Control should be processed before data when both are queued"
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn empty_message_types_work() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);

    let handle = thread::spawn(move || {
        struct UnitActor;
        impl Actor<(), (), ()> for UnitActor {
            fn handle_data(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut UnitActor);
    });

    tx.send(Message::Data(())).unwrap();
    tx.send(Message::Control(())).unwrap();
    tx.send(Message::Management(())).unwrap();

    thread::sleep(Duration::from_millis(20));
    drop(tx);
    handle.join().unwrap();
}

#[test]
fn zero_size_type_messages() {
    #[derive(Clone, Copy)]
    struct Zst;

    let (tx, mut rx) = ActorScheduler::<Zst, Zst, Zst>::new(10, 100);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct ZstActor(Arc<AtomicUsize>);
        impl Actor<Zst, Zst, Zst> for ZstActor {
            fn handle_data(&mut self, _: Zst) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: Zst) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_management(&mut self, _: Zst) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut ZstActor(count_clone));
    });

    for _ in 0..100 {
        tx.send(Message::Data(Zst)).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 100);
}

#[test]
fn large_message_type_works() {
    // Test with a large message type
    #[derive(Clone)]
    struct LargeMessage {
        #[allow(dead_code)]
        data: [u8; 4096],
    }

    let (tx, mut rx) = ActorScheduler::<LargeMessage, (), ()>::new(10, 50);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct LargeActor(Arc<AtomicUsize>);
        impl Actor<LargeMessage, (), ()> for LargeActor {
            fn handle_data(&mut self, _: LargeMessage) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut LargeActor(count_clone));
    });

    for _ in 0..20 {
        tx.send(Message::Data(LargeMessage { data: [0u8; 4096] }))
            .unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 20);
}

#[test]
fn immediate_shutdown_no_messages() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);

    let handle = thread::spawn(move || {
        struct NoopActor;
        impl Actor<(), (), ()> for NoopActor {
            fn handle_data(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut NoopActor);
    });

    // Drop immediately without sending anything
    drop(tx);

    // Should complete quickly
    let result = handle.join();
    assert!(result.is_ok(), "Actor should exit cleanly with no messages");
}

// ============================================================================
// Scheduler Configuration Tests
// ============================================================================

#[test]
fn custom_burst_and_buffer_sizes() {
    // Very small burst, very small buffer
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(1, 1);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    for i in 0..10 {
        tx.send(Message::Data(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 10);
}

#[test]
fn large_burst_and_buffer_sizes() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10000, 100000);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Fill the buffer
    for i in 0..50000 {
        tx.send(Message::Data(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(200));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 50000);
}
