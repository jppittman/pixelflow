//! Stress tests for actor-scheduler concurrency edge cases
//!
//! These tests verify the scheduler behaves correctly under heavy load,
//! concurrent access, and various edge conditions.
//!
//! With SPSC-sharded channels, each producer has its own dedicated ring buffer.
//! Multi-producer tests use ActorBuilder to create separate handles.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    ShutdownMode, SystemStatus,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

// ============================================================================
// Test Handler Implementations
// ============================================================================

struct CountingHandler {
    data_count: Arc<AtomicUsize>,
    ctrl_count: Arc<AtomicUsize>,
    mgmt_count: Arc<AtomicUsize>,
}

impl Actor<u64, u64, u64> for CountingHandler {
    fn handle_data(&mut self, _msg: u64) -> HandlerResult {
        self.data_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn handle_control(&mut self, _msg: u64) -> HandlerResult {
        self.ctrl_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn handle_management(&mut self, _msg: u64) -> HandlerResult {
        self.mgmt_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

struct SlowHandler {
    delay: Duration,
    processed: Arc<AtomicUsize>,
}

impl Actor<u64, u64, u64> for SlowHandler {
    fn handle_data(&mut self, _msg: u64) -> HandlerResult {
        thread::sleep(self.delay);
        self.processed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn handle_control(&mut self, _msg: u64) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _msg: u64) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

struct NoOpHandler;

impl Actor<(), (), ()> for NoOpHandler {
    fn handle_data(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ============================================================================
// High Contention Tests (SPSC: each sender has dedicated channels)
// ============================================================================

#[test]
fn high_contention_all_messages_delivered() {
    const NUM_SENDERS: usize = 10;
    const MESSAGES_PER_SENDER: usize = 100;

    let mut builder = ActorBuilder::<u64, u64, u64>::new(1024, None);
    let senders: Vec<_> = (0..NUM_SENDERS).map(|_| builder.add_producer()).collect();
    let mut rx = builder.build_with_burst(1024, ShutdownMode::default());

    let data_count = Arc::new(AtomicUsize::new(0));
    let ctrl_count = Arc::new(AtomicUsize::new(0));
    let mgmt_count = Arc::new(AtomicUsize::new(0));

    let handler = CountingHandler {
        data_count: data_count.clone(),
        ctrl_count: ctrl_count.clone(),
        mgmt_count: mgmt_count.clone(),
    };

    // Spawn receiver
    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    // Spawn multiple senders — each with its own SPSC channel
    let mut sender_handles = Vec::new();
    for tx in senders {
        let handle = thread::spawn(move || {
            for i in 0..MESSAGES_PER_SENDER {
                match i % 3 {
                    0 => tx.send(Message::Data(i as u64)).unwrap(),
                    1 => tx.send(Message::Control(i as u64)).unwrap(),
                    _ => tx.send(Message::Management(i as u64)).unwrap(),
                }
            }
        });
        sender_handles.push(handle);
    }

    // Wait for all senders
    for handle in sender_handles {
        handle.join().unwrap();
    }

    // Give time for processing, then all handles are dropped → scheduler exits
    thread::sleep(Duration::from_millis(100));
    receiver_handle.join().unwrap();

    // Verify all messages were delivered
    let total = data_count.load(Ordering::SeqCst)
        + ctrl_count.load(Ordering::SeqCst)
        + mgmt_count.load(Ordering::SeqCst);

    assert_eq!(
        total,
        NUM_SENDERS * MESSAGES_PER_SENDER,
        "All messages should be delivered under high contention"
    );
}

#[test]
fn high_contention_fairness() {
    const NUM_SENDERS: usize = 5;
    const MESSAGES_PER_SENDER: usize = 50;

    let mut builder = ActorBuilder::<u64, u64, u64>::new(100, None);
    let senders: Vec<_> = (0..NUM_SENDERS).map(|_| builder.add_producer()).collect();
    let mut rx = builder.build_with_burst(10, ShutdownMode::default());

    let ctrl_count = Arc::new(AtomicUsize::new(0));

    let handler = CountingHandler {
        data_count: Arc::new(AtomicUsize::new(0)),
        ctrl_count: ctrl_count.clone(),
        mgmt_count: Arc::new(AtomicUsize::new(0)),
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    let mut sender_handles = Vec::new();
    for tx in senders {
        let handle = thread::spawn(move || {
            for i in 0..MESSAGES_PER_SENDER {
                tx.send(Message::Control(i as u64)).unwrap();
            }
        });
        sender_handles.push(handle);
    }

    for handle in sender_handles {
        handle.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    receiver_handle.join().unwrap();

    assert_eq!(
        ctrl_count.load(Ordering::SeqCst),
        NUM_SENDERS * MESSAGES_PER_SENDER,
        "All control messages should be delivered"
    );
}

// ============================================================================
// Rapid Channel Creation/Destruction Tests
// ============================================================================





// ============================================================================
// Backpressure Tests
// ============================================================================

#[test]
fn backpressure_with_slow_consumer() {
    let (tx, mut rx) = ActorScheduler::new(10, 10); // Small buffer
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

    let handler = SlowHandler {
        delay: Duration::from_millis(1),
        processed: processed_clone,
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    // Send many messages — spin-yield backpressure when buffer full
    let sender_handle = thread::spawn(move || {
        for i in 0..50 {
            tx.send(Message::Data(i)).unwrap();
        }
    });

    sender_handle.join().unwrap();
    thread::sleep(Duration::from_millis(100));
    receiver_handle.join().unwrap();

    assert_eq!(
        processed.load(Ordering::SeqCst),
        50,
        "All messages should eventually be processed"
    );
}

// ============================================================================
// Wake Handler Tests
// ============================================================================

#[test]
fn custom_wake_handler_is_called() {
    use actor_scheduler::WakeHandler;

    struct TestWakeHandler {
        called: Arc<AtomicBool>,
    }

    impl WakeHandler for TestWakeHandler {
        fn wake(&self) {
            self.called.store(true, Ordering::SeqCst);
        }
    }

    let called = Arc::new(AtomicBool::new(false));
    let wake_handler = Arc::new(TestWakeHandler {
        called: called.clone(),
    });

    let (tx, _rx) =
        ActorScheduler::<u64, u64, u64>::new_with_wake_handler(10, 100, Some(wake_handler));

    tx.send(Message::Data(42)).unwrap();

    assert!(
        called.load(Ordering::SeqCst),
        "Wake handler should be called on message send"
    );
}

// ============================================================================
// Burst Limit Tests
// ============================================================================

#[test]
fn burst_limit_prevents_data_starvation() {
    let (tx, mut rx) = ActorScheduler::new(2, 1000); // Burst limit of 2
    let data_processed = Arc::new(AtomicUsize::new(0));
    let mgmt_processed = Arc::new(AtomicUsize::new(0));

    struct BurstHandler {
        data_processed: Arc<AtomicUsize>,
        mgmt_processed: Arc<AtomicUsize>,
    }

    impl Actor<u64, u64, u64> for BurstHandler {
        fn handle_data(&mut self, _: u64) -> HandlerResult {
            self.data_processed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: u64) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: u64) -> HandlerResult {
            self.mgmt_processed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let handler = BurstHandler {
        data_processed: data_processed.clone(),
        mgmt_processed: mgmt_processed.clone(),
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    // Send many data messages and some management
    for i in 0..100 {
        tx.send(Message::Data(i)).unwrap();
    }
    for i in 0..10 {
        tx.send(Message::Management(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver_handle.join().unwrap();

    // Both should be fully processed
    assert_eq!(data_processed.load(Ordering::SeqCst), 100);
    assert_eq!(mgmt_processed.load(Ordering::SeqCst), 10);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
fn handle_is_send() {
    // ActorHandle is Send (can be moved to another thread) but NOT Sync
    // (Cell<usize> in SpscSender prevents shared references across threads).
    // This is correct: SPSC = one producer per channel.
    fn assert_send<T: Send>() {}
    assert_send::<actor_scheduler::ActorHandle<u64, u64, u64>>();
}

#[test]
fn multi_producer_send() {
    // Multiple producers each with their own SPSC channel
    let mut builder = ActorBuilder::<u64, u64, u64>::new(1000, None);
    let senders: Vec<_> = (0..10).map(|_| builder.add_producer()).collect();
    let mut rx = builder.build_with_burst(10, ShutdownMode::default());

    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    struct SimpleHandler {
        count: Arc<AtomicUsize>,
    }

    impl Actor<u64, u64, u64> for SimpleHandler {
        fn handle_data(&mut self, _: u64) -> HandlerResult {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: u64) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: u64) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver_handle = thread::spawn(move || {
        let mut h = SimpleHandler { count: count_clone };
        rx.run(&mut h);
    });

    // Each thread gets its own handle (no cloning, no contention)
    let mut handles = Vec::new();
    for tx in senders {
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                tx.send(Message::Data(i)).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    receiver_handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 1000);
}

// ============================================================================
// Empty Message Type Tests
// ============================================================================



// ============================================================================
// Large Message Tests
// ============================================================================

#[test]
fn large_messages_work() {
    struct LargeMessage {
        data: [u8; 4096],
    }

    struct LargeHandler {
        received: Arc<AtomicUsize>,
    }

    impl Actor<LargeMessage, (), ()> for LargeHandler {
        fn handle_data(&mut self, msg: LargeMessage) -> HandlerResult {
            // Verify data integrity
            assert!(msg.data.iter().all(|&b| b == 42));
            self.received.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let received = Arc::new(AtomicUsize::new(0));

    let handler = LargeHandler {
        received: received.clone(),
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    for _ in 0..10 {
        tx.send(Message::Data(LargeMessage { data: [42; 4096] }))
            .unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    receiver_handle.join().unwrap();

    assert_eq!(received.load(Ordering::SeqCst), 10);
}
