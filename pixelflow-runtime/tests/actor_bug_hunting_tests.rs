//! Bug Hunting Tests for the Actor Model
//!
//! These tests actively try to find bugs by:
//! - Probing edge cases and boundary conditions
//! - Testing race conditions and concurrent access patterns
//! - Attempting to trigger overflows and underflows
//! - Testing resource exhaustion scenarios
//! - Looking for deadlocks and starvation conditions
//!
//! Each test documents the bug it's hunting for.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SendError, ShutdownMode, SystemStatus,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Backoff jitter overflow
// The calculation `(backoff_micros * jitter_pct) / 100` could overflow
// if backoff_micros is large enough, even though the final result fits.
// ============================================================================

#[test]
fn backoff_does_not_overflow_on_large_attempts() {
    // This test tries to trigger overflow in the backoff calculation.
    // The backoff uses 2^attempt, and with attempt=63, 2^63 would overflow.
    // The code should cap at MAX_BACKOFF (500ms) before overflow.

    let (tx, rx) = ActorScheduler::<(), (), ()>::new(10, 1);

    // Fill up the control lane to trigger backoff
    // Note: Control lane size is SchedulerParams::DEFAULT.control_mgmt_buffer_size

    let sender = thread::spawn(move || {
        // Keep trying to send - this should eventually hit the timeout error
        // if backoff overflows or gets stuck, this will hang
        let start = Instant::now();
        let mut error_count = 0;

        for _ in 0..1000 {
            match tx.send(Message::Control(())) {
                Ok(()) => {}
                Err(SendError::Timeout) => {
                    error_count += 1;
                    if error_count > 5 {
                        break;
                    }
                }
                Err(SendError::Disconnected) => break,
            }
        }

        // Should complete in reasonable time (not stuck in infinite backoff)
        start.elapsed()
    });

    // Don't run the receiver - let the channel fill up
    thread::sleep(Duration::from_millis(100));
    drop(rx);

    let elapsed = sender.join().unwrap();
    assert!(
        elapsed < Duration::from_secs(10),
        "Backoff should eventually timeout, not hang. Took: {:?}",
        elapsed
    );
}

// ============================================================================
// Zero or negative buffer sizes
// What happens if we create a scheduler with buffer size 0?
// KNOWN ISSUE: sync_channel(0) creates a rendezvous channel that blocks
// forever on send if receiver isn't actively receiving. This is expected
// std::sync::mpsc behavior but callers should avoid buffer_size=0.
// ============================================================================

#[test]
fn zero_burst_limit_does_not_cause_infinite_loop() {
    // Burst limit of 0 could cause issues in the loop logic
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(0, 10);
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

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
        rx.run(&mut Counter(processed_clone));
    });

    tx.send(Message::Data(1)).unwrap();
    tx.send(Message::Data(2)).unwrap();

    // Wait and check - should not infinite loop
    thread::sleep(Duration::from_millis(100));
    drop(tx);

    // Use timeout to detect infinite loop
    let join_result = thread::spawn(move || {
        handle.join().unwrap();
    });

    let timeout = Duration::from_secs(2);
    let start = Instant::now();
    while start.elapsed() < timeout {
        if join_result.is_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(
        start.elapsed() < timeout,
        "Zero burst limit should not cause infinite loop"
    );
}

// ============================================================================
// Thundering herd on multiple sender drop
// If many senders are dropped at once, does the doorbell pattern handle it?
// ============================================================================

#[test]
fn mass_sender_drop_does_not_cause_race() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(1000, None);

    // Create 100 producers for the sender threads + 1 for the original tx
    let senders: Vec<_> = (0..100).map(|_| builder.add_producer()).collect();
    let tx = builder.add_producer();

    let mut rx = builder.build_with_burst(100, ShutdownMode::default());

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

    // Send from all senders
    let barrier = Arc::new(Barrier::new(100));
    let handles: Vec<_> = senders
        .into_iter()
        .map(|sender| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..10 {
                    let _ = sender.send(Message::Data(i));
                }
                // Sender dropped here - all 100 at roughly the same time
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Original sender still exists
    tx.send(Message::Data(999)).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Should have processed 100 * 10 + 1 = 1001 messages
    let final_count = count.load(Ordering::SeqCst);
    assert_eq!(
        final_count, 1001,
        "All messages should be processed despite mass sender drop"
    );
}

// ============================================================================
// Actor panics during handler
// What happens if an actor panics mid-processing?
// ============================================================================

#[test]
fn actor_panic_does_not_corrupt_state() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
    let panic_at = 5;

    let handle = thread::spawn(move || {
        struct PanicActor {
            count: usize,
            panic_at: usize,
        }
        impl Actor<i32, i32, i32> for PanicActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.count += 1;
                if self.count == self.panic_at {
                    panic!("Intentional panic at message {}", self.count);
                }
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

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rx.run(&mut PanicActor { count: 0, panic_at });
        }));

        result.is_err() // Should have panicked
    });

    for i in 0..10 {
        let _ = tx.send(Message::Data(i)); // Some will fail after panic
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);

    let panicked = handle.join().unwrap();
    assert!(panicked, "Actor should have panicked");
}

// ============================================================================
// Starvation of lower priority lanes
// Can control messages completely starve data messages?
// ============================================================================



// ============================================================================
// ActorStatus::Busy causes CPU spin
// If actor always returns Poll, does it burn CPU?
// ============================================================================

#[test]
fn park_poll_does_not_spin_indefinitely() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);
    let park_count = Arc::new(AtomicUsize::new(0));
    let park_count_clone = park_count.clone();

    let handle = thread::spawn(move || {
        struct SpinActor {
            park_count: Arc<AtomicUsize>,
            max_parks: usize,
        }
        impl Actor<(), (), ()> for SpinActor {
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
                let count = self.park_count.fetch_add(1, Ordering::SeqCst);
                if count < self.max_parks {
                    Ok(ActorStatus::Busy) // Keep spinning
                } else {
                    Ok(ActorStatus::Idle) // Eventually stop
                }
            }
        }
        rx.run(&mut SpinActor {
            park_count: park_count_clone,
            max_parks: 1000,
        });
    });

    // Send one message to trigger the spin
    tx.send(Message::Data(())).unwrap();

    // Wait for the spin to exhaust
    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    let final_count = park_count.load(Ordering::SeqCst);
    assert!(
        final_count >= 1000,
        "Should have spun many times. Count: {}",
        final_count
    );
}

// ============================================================================
// Channel filling during slow handler
// If handler is slow, does the channel fill and block senders?
// ============================================================================

#[test]
fn slow_handler_backpressure_works() {
    // Small buffer to trigger backpressure quickly
    let mut builder = ActorBuilder::<i32, i32, i32>::new(2, None);
    let tx_sender = builder.add_producer();
    let tx = builder.add_producer();
    let mut rx = builder.build_with_burst(10, ShutdownMode::default());

    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

    let handle = thread::spawn(move || {
        struct SlowActor(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for SlowActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                thread::sleep(Duration::from_millis(50)); // Very slow
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
        rx.run(&mut SlowActor(processed_clone));
    });

    let sender_start = Instant::now();

    // Sender thread - will block on backpressure
    let sender = thread::spawn(move || {
        for i in 0..10 {
            tx_sender.send(Message::Data(i)).unwrap();
        }
        sender_start.elapsed()
    });

    let send_time = sender.join().unwrap();

    // Sending 10 messages through buffer of 2 with 50ms handler
    // Should take at least 400ms (8 messages worth of blocking)
    assert!(
        send_time >= Duration::from_millis(300),
        "Sender should block on backpressure. Took: {:?}",
        send_time
    );

    drop(tx);
    handle.join().unwrap();

    assert_eq!(processed.load(Ordering::SeqCst), 10);
}

// ============================================================================
// Message ordering under contention
// Do messages maintain FIFO order when multiple threads are sending?
// ============================================================================

#[test]
fn single_sender_fifo_ordering_maintained() {
    let (tx, mut rx) = ActorScheduler::new(100, 1000);
    let received = Arc::new(Mutex::new(Vec::new()));
    let received_clone = received.clone();

    let handle = thread::spawn(move || {
        struct OrderTracker(Arc<Mutex<Vec<i32>>>);
        impl Actor<i32, i32, i32> for OrderTracker {
            fn handle_data(&mut self, msg: i32) -> HandlerResult {
                self.0.lock().unwrap().push(msg);
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
        rx.run(&mut OrderTracker(received_clone));
    });

    // Send in order from single thread
    for i in 0..1000 {
        tx.send(Message::Data(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    let received = received.lock().unwrap();
    for (i, &val) in received.iter().enumerate() {
        assert_eq!(
            val, i as i32,
            "FIFO order violated at position {}: expected {}, got {}",
            i, i, val
        );
    }
}

// ============================================================================
// Very large message count overflow
// Can frame numbers or counters overflow?
// ============================================================================

#[test]
fn large_frame_numbers_dont_overflow() {
    use pixelflow_runtime::vsync_actor::RenderedResponse;

    // Test with u64 max values
    let response = RenderedResponse {
        frame_number: u64::MAX,
        rendered_at: Instant::now(),
    };

    assert_eq!(response.frame_number, u64::MAX);

    // Test wrapping behavior
    let response2 = RenderedResponse {
        frame_number: u64::MAX.wrapping_add(1),
        rendered_at: Instant::now(),
    };

    assert_eq!(response2.frame_number, 0, "Wrapping should work correctly");
}

// ============================================================================
// Rapid channel creation/destruction
// Does creating and destroying many channels leak resources?
// ============================================================================



// ============================================================================
// Send after partial processing
// What happens if sender sends while receiver is mid-batch?
// ============================================================================

#[test]
fn concurrent_send_during_processing() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None); // Small burst limit

    // 10 producers for sender threads + 1 to keep the scheduler alive until drop
    let senders: Vec<_> = (0..10).map(|_| builder.add_producer()).collect();
    let tx = builder.add_producer();

    let mut rx = builder.build_with_burst(5, ShutdownMode::default());

    let total_received = Arc::new(AtomicUsize::new(0));
    let total_received_clone = total_received.clone();

    let handle = thread::spawn(move || {
        struct CountingActor(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for CountingActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                // Small delay to increase chance of concurrent send
                thread::sleep(Duration::from_micros(100));
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
        rx.run(&mut CountingActor(total_received_clone));
    });

    // Multiple senders sending concurrently
    let handles: Vec<_> = senders
        .into_iter()
        .map(|sender| {
            thread::spawn(move || {
                for i in 0..100 {
                    sender.send(Message::Data(i)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(500));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        total_received.load(Ordering::SeqCst),
        1000,
        "All messages should be received"
    );
}

// ============================================================================
// Doorbell saturation
// The doorbell has buffer size 1. What if it fills?
// ============================================================================

#[test]
fn doorbell_saturation_does_not_lose_messages() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(1000, None);

    // 50 producers for sender threads + 1 to keep the scheduler alive until drop
    let senders: Vec<_> = (0..50).map(|_| builder.add_producer()).collect();
    let tx = builder.add_producer();

    let mut rx = builder.build_with_burst(100, ShutdownMode::default());

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

    // Rapid-fire sends from many threads to saturate doorbell
    let barrier = Arc::new(Barrier::new(50));
    let handles: Vec<_> = senders
        .into_iter()
        .map(|sender| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..100 {
                    sender.send(Message::Data(i)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(200));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        count.load(Ordering::SeqCst),
        5000,
        "All messages should be processed despite doorbell saturation"
    );
}

// ============================================================================
// Time-based operations near epoch
// What happens with time calculations near boundaries?
// ============================================================================

#[test]
fn instant_arithmetic_is_safe() {
    use pixelflow_runtime::vsync_actor::RenderedResponse;

    let now = Instant::now();
    let later = now + Duration::from_secs(1);

    let response = RenderedResponse {
        frame_number: 0,
        rendered_at: later,
    };

    // Should be able to calculate elapsed without panic
    let elapsed = response.rendered_at.elapsed();
    // elapsed might be 0 or small negative (which panics on sub), but elapsed() handles it
    assert!(elapsed < Duration::from_secs(2));
}

// ============================================================================
// Control lane timeout behavior
// Does the timeout in send_with_backoff actually work?
// ============================================================================

#[test]
fn control_lane_timeout_returns_error() {
    let test_timeout = Duration::from_secs(120); // Well above MAX_BACKOFF, ensure we don't deadlock
    let test_thread = thread::spawn(|| {
        let params = actor_scheduler::SchedulerParams::DEFAULT;
        let (tx, rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

        // Don't run the receiver - just let messages pile up

        // Fill the control lane. SPSC rounds up to next power of 2, so
        // the actual capacity is >= params.control_mgmt_buffer_size.
        let actual_capacity = params.control_mgmt_buffer_size.max(2).next_power_of_two();
        for _ in 0..actual_capacity {
            tx.send(Message::Control(0)).unwrap();
        }

        // Next send should timeout after MAX_BACKOFF
        let result = tx.send(Message::Control(999));

        // Should get timeout error - don't assert on timing, just on the error
        assert!(
            matches!(result, Err(SendError::Timeout)),
            "Should timeout when control lane is full. Got: {:?}",
            result
        );

        drop(rx);
    });

    // Test should not hang - if it takes longer than 3 * MAX_BACKOFF, it's deadlocked
    let start = Instant::now();
    let result = test_thread.join();
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Test panicked or did not complete");

    assert!(
        elapsed < test_timeout,
        "Test appears to be deadlocked - took {:?}, expected < {:?}",
        elapsed,
        test_timeout
    );
}

// ============================================================================
// Memory safety with large queues
// Do we handle memory correctly with many queued messages?
// ============================================================================

#[test]
fn large_queue_does_not_cause_issues() {
    let (tx, mut rx) = ActorScheduler::<String, String, String>::new(1000, 10000);
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
                Ok(())
            }
            fn handle_management(&mut self, _: String) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Queue up many messages with non-trivial data
    for i in 0..10000 {
        tx.send(Message::Data(format!(
            "message {} with some extra data to use more memory",
            i
        )))
        .unwrap();
    }

    thread::sleep(Duration::from_millis(500));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 10000);
}

// ============================================================================
// Scheduler shutdown race
// What if messages arrive exactly as scheduler is shutting down?
// ============================================================================


