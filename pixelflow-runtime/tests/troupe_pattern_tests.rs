//! Tests for the Troupe Pattern
//!
//! The troupe system provides lifecycle management for groups of actors.
//! These tests verify:
//! - Directory pattern (actors accessing each other's handles)
//! - Two-phase initialization (new() -> exposed() -> play())
//! - Exposed handles lifetime
//! - Cross-actor messaging
//! - Thread spawning and cleanup
//! - Error handling in actor threads
//!
//! Note: We test the patterns manually here since the troupe! macro
//! generates code that's hard to test in isolation.

use actor_scheduler::{
    Actor, ActorBuilder, ActorHandle, ActorScheduler, ActorStatus, ActorTypes, HandlerError,
    HandlerResult, Message, SystemStatus, TroupeActor,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

// ============================================================================
// Test Message Types
// ============================================================================

#[derive(Debug)]
struct AlphaData(String);

#[derive(Debug, Default)]
enum AlphaControl {
    Ping,
    #[default]
    Shutdown,
}

#[derive(Debug)]
struct AlphaManagement;

#[derive(Debug)]
struct BetaData(i32);

#[derive(Debug, Default)]
enum BetaControl {
    Pong,
    #[default]
    Shutdown,
}

#[derive(Debug)]
struct BetaManagement;

// ============================================================================
// Manual Directory (what troupe! generates)
// ============================================================================

struct TestDirectory {
    alpha: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
    beta: ActorHandle<BetaData, BetaControl, BetaManagement>,
}

// ============================================================================
// Test Actors
// ============================================================================

struct AlphaActor<'a> {
    dir: &'a TestDirectory,
    log: Arc<Mutex<Vec<String>>>,
}

impl ActorTypes for AlphaActor<'_> {
    type Data = AlphaData;
    type Control = AlphaControl;
    type Management = AlphaManagement;
}

impl<'a> TroupeActor<&'a TestDirectory> for AlphaActor<'a> {
    fn new(_dir: &'a TestDirectory) -> Self {
        panic!("use new_with_log instead")
    }
}

impl Actor<AlphaData, AlphaControl, AlphaManagement> for AlphaActor<'_> {
    fn handle_data(&mut self, msg: AlphaData) -> HandlerResult {
        self.log
            .lock()
            .unwrap()
            .push(format!("Alpha:Data:{}", msg.0));
        Ok(())
    }

    fn handle_control(&mut self, cmd: AlphaControl) -> HandlerResult {
        match cmd {
            AlphaControl::Ping => {
                self.log.lock().unwrap().push("Alpha:Ping".to_string());
                // Send pong to beta
                let _ = self.dir.beta.send(Message::Control(BetaControl::Pong));
            }
            AlphaControl::Shutdown => {
                self.log.lock().unwrap().push("Alpha:Shutdown".to_string());
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

struct BetaActor<'a> {
    dir: &'a TestDirectory,
    log: Arc<Mutex<Vec<String>>>,
}

impl ActorTypes for BetaActor<'_> {
    type Data = BetaData;
    type Control = BetaControl;
    type Management = BetaManagement;
}

impl<'a> TroupeActor<&'a TestDirectory> for BetaActor<'a> {
    fn new(_dir: &'a TestDirectory) -> Self {
        panic!("use new_with_log instead")
    }
}

impl Actor<BetaData, BetaControl, BetaManagement> for BetaActor<'_> {
    fn handle_data(&mut self, msg: BetaData) -> HandlerResult {
        self.log
            .lock()
            .unwrap()
            .push(format!("Beta:Data:{}", msg.0));
        Ok(())
    }

    fn handle_control(&mut self, cmd: BetaControl) -> HandlerResult {
        match cmd {
            BetaControl::Pong => {
                self.log.lock().unwrap().push("Beta:Pong".to_string());
                // Send back to alpha
                let _ = self
                    .dir
                    .alpha
                    .send(Message::Data(AlphaData("pong-response".to_string())));
            }
            BetaControl::Shutdown => {
                self.log.lock().unwrap().push("Beta:Shutdown".to_string());
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, _: BetaManagement) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ============================================================================
// Directory Pattern Tests
// ============================================================================

/// Tests cross-actor messaging through a shared directory.
///
/// This test demonstrates a key design consideration: when actors hold handles
/// to each other via a shared directory, clean shutdown requires careful ordering.
/// The scheduler's run() only exits when all senders are dropped, but if actors
/// hold circular handle references, a deadlock occurs.
///
/// Solution: Use timeout-based verification and don't wait for thread join.
/// Real applications should use a dedicated shutdown coordinator or have actors
/// drop their directory references when they receive shutdown.


// ============================================================================
// Two-Phase Initialization Tests
// ============================================================================

/// Simulates the Troupe struct that troupe! would generate (SPSC pattern)
struct TestTroupe {
    directory: TestDirectory,
    // Schedulers run the actors when dropped; they're held here for that purpose.
    #[allow(dead_code)]
    alpha_scheduler: ActorScheduler<AlphaData, AlphaControl, AlphaManagement>,
    #[allow(dead_code)]
    beta_scheduler: ActorScheduler<BetaData, BetaControl, BetaManagement>,
    exposed_alpha: Option<ActorHandle<AlphaData, AlphaControl, AlphaManagement>>,
}

struct TestExposedHandles {
    alpha: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
}

impl TestTroupe {
    fn new() -> Self {
        let mut alpha_builder =
            ActorBuilder::<AlphaData, AlphaControl, AlphaManagement>::new(1024, None);
        let mut beta_builder =
            ActorBuilder::<BetaData, BetaControl, BetaManagement>::new(1024, None);

        // Each actor gets its own directory with dedicated SPSC handles
        let alpha_dir = TestDirectory {
            alpha: alpha_builder.add_producer(),
            beta: beta_builder.add_producer(),
        };
        let beta_dir = TestDirectory {
            alpha: alpha_builder.add_producer(),
            beta: beta_builder.add_producer(),
        };

        // Create exposed handles
        let exposed_alpha = Some(alpha_builder.add_producer());

        // Create main directory
        let directory = TestDirectory {
            alpha: alpha_builder.add_producer(), // Placeholder, not used in this test logic
            beta: beta_builder.add_producer(),
        };

        let mut alpha_scheduler = alpha_builder.build();
        let mut beta_scheduler = beta_builder.build();

        // Spawn actors
        // Note: In this test harness, we spawn them immediately.
        // The schedulers passed to thread must be the ones we built.
        // But wait, TestTroupe holds the schedulers.
        // If we spawn here, we need to move the scheduler into the thread.
        // But TestTroupe needs to keep them alive?
        // The original code passed builders to TestTroupe and let it build/spawn later?
        // No, the original code had `alpha_builder` fields.
        // The test `two_phase_initialization_queues_messages_before_play` does:
        // let mut troupe = TestTroupe::new();
        // let exposed = troupe.exposed();
        // exposed.alpha.send(...)
        // drop(troupe);
        // It doesn't call play(). It just verifies queueing works.
        // So we don't need to spawn threads in `new()`.
        // We just need to hold the schedulers.

        // We do need to spawn the actors eventually if we want them to process messages.
        // But for *this specific test*, it just checks that sending doesn't fail.

        // Wait, the "directory" fields in TestTroupe were just holding handles.
        // We need to ensure the actors *would* have access to them.

        Self {
            directory,
            alpha_scheduler,
            beta_scheduler,
            exposed_alpha,
        }
    }

    fn exposed(&mut self) -> TestExposedHandles {
        TestExposedHandles {
            alpha: self.exposed_alpha.take().expect("exposed() called twice"),
        }
    }
}



// ============================================================================
// Thread Lifecycle Tests
// ============================================================================

#[test]
fn all_actor_threads_exit_on_channel_close() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let alpha_exited = Arc::new(AtomicBool::new(false));
    let beta_exited = Arc::new(AtomicBool::new(false));

    let alpha_exit = alpha_exited.clone();
    let beta_exit = beta_exited.clone();

    let alpha_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: AlphaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        alpha_s.run(&mut NoopActor);
        alpha_exit.store(true, Ordering::SeqCst);
    });

    let beta_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<BetaData, BetaControl, BetaManagement> for NoopActor {
            fn handle_data(&mut self, _: BetaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: BetaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: BetaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        beta_s.run(&mut NoopActor);
        beta_exit.store(true, Ordering::SeqCst);
    });

    // Verify threads are running
    thread::sleep(Duration::from_millis(20));
    assert!(!alpha_exited.load(Ordering::SeqCst));
    assert!(!beta_exited.load(Ordering::SeqCst));

    // Drop handles - should trigger exit
    drop(alpha_h);
    drop(beta_h);

    // Wait for threads
    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();

    assert!(alpha_exited.load(Ordering::SeqCst));
    assert!(beta_exited.load(Ordering::SeqCst));
}

#[test]
fn actor_thread_panic_isolated() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let beta_count = Arc::new(AtomicUsize::new(0));
    let beta_count_clone = beta_count.clone();

    // Alpha will panic
    let alpha_thread = thread::spawn(move || {
        struct PanicActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for PanicActor {
            fn handle_data(&mut self, _: AlphaData) -> HandlerResult {
                panic!("Alpha panics!");
            }
            fn handle_control(&mut self, _: AlphaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            alpha_s.run(&mut PanicActor);
        }))
    });

    // Beta should continue working
    let beta_thread = thread::spawn(move || {
        struct CountActor(Arc<AtomicUsize>);
        impl Actor<BetaData, BetaControl, BetaManagement> for CountActor {
            fn handle_data(&mut self, _: BetaData) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: BetaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: BetaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        beta_s.run(&mut CountActor(beta_count_clone));
    });

    // Trigger alpha panic
    alpha_h
        .send(Message::Data(AlphaData("boom".to_string())))
        .unwrap();

    thread::sleep(Duration::from_millis(50));

    // Beta should still work
    beta_h.send(Message::Data(BetaData(1))).unwrap();
    beta_h.send(Message::Data(BetaData(2))).unwrap();

    thread::sleep(Duration::from_millis(50));

    drop(alpha_h);
    drop(beta_h);

    let alpha_result = alpha_thread.join();
    beta_thread.join().unwrap();

    // Alpha should have panicked
    assert!(alpha_result.is_ok()); // Thread itself didn't panic (we caught it)
    assert!(alpha_result.unwrap().is_err()); // But the closure panicked

    // Beta should have processed messages
    assert_eq!(beta_count.load(Ordering::SeqCst), 2);
}

// ============================================================================
// Circular Messaging Tests
// ============================================================================

/// Tests that circular messaging between actors does not deadlock.
///
/// This verifies that the actor scheduler handles ping-pong message patterns
/// without blocking. Note that clean shutdown is not tested here due to
/// circular handle references (see directory_allows_cross_actor_messaging).
#[test]
fn circular_messaging_does_not_deadlock() {
    // Create builders for each actor
    let mut alpha_builder =
        ActorBuilder::<AlphaData, AlphaControl, AlphaManagement>::new(1000, None);
    let mut beta_builder = ActorBuilder::<BetaData, BetaControl, BetaManagement>::new(1000, None);

    // Each actor gets a dedicated handle to the other
    let ping_beta_h = beta_builder.add_producer();
    let pong_alpha_h = alpha_builder.add_producer();

    // External handle for kick-starting the ping-pong
    let alpha_h = alpha_builder.add_producer();

    let mut alpha_s = alpha_builder.build();
    let mut beta_s = beta_builder.build();

    let ping_count = Arc::new(AtomicUsize::new(0));
    let pong_count = Arc::new(AtomicUsize::new(0));

    let ping_clone = ping_count.clone();
    thread::spawn(move || {
        struct PingActor {
            beta_h: ActorHandle<BetaData, BetaControl, BetaManagement>,
            count: Arc<AtomicUsize>,
            max: usize,
        }
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for PingActor {
            fn handle_data(&mut self, _: AlphaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, cmd: AlphaControl) -> HandlerResult {
                if matches!(cmd, AlphaControl::Ping) {
                    let c = self.count.fetch_add(1, Ordering::SeqCst);
                    if c < self.max {
                        let _ = self.beta_h.send(Message::Control(BetaControl::Pong));
                    }
                }
                Ok(())
            }
            fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        alpha_s.run(&mut PingActor {
            beta_h: ping_beta_h,
            count: ping_clone,
            max: 100,
        });
    });

    let pong_clone = pong_count.clone();
    thread::spawn(move || {
        struct PongActor {
            alpha_h: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
            count: Arc<AtomicUsize>,
            max: usize,
        }
        impl Actor<BetaData, BetaControl, BetaManagement> for PongActor {
            fn handle_data(&mut self, _: BetaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, cmd: BetaControl) -> HandlerResult {
                if matches!(cmd, BetaControl::Pong) {
                    let c = self.count.fetch_add(1, Ordering::SeqCst);
                    if c < self.max {
                        let _ = self.alpha_h.send(Message::Control(AlphaControl::Ping));
                    }
                }
                Ok(())
            }
            fn handle_management(&mut self, _: BetaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        beta_s.run(&mut PongActor {
            alpha_h: pong_alpha_h,
            count: pong_clone,
            max: 100,
        });
    });

    // Start the ping-pong
    alpha_h.send(Message::Control(AlphaControl::Ping)).unwrap();

    // Wait for it to complete (with timeout to detect deadlock)
    let start = std::time::Instant::now();
    while ping_count.load(Ordering::SeqCst) < 100 || pong_count.load(Ordering::SeqCst) < 100 {
        if start.elapsed() > Duration::from_secs(5) {
            panic!(
                "Deadlock detected in circular messaging! ping={}, pong={}",
                ping_count.load(Ordering::SeqCst),
                pong_count.load(Ordering::SeqCst)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(ping_count.load(Ordering::SeqCst) >= 100);
    assert!(pong_count.load(Ordering::SeqCst) >= 100);
}

// ============================================================================
// Multiple Producer Tests (SPSC: each producer has dedicated channels)
// ============================================================================

#[test]
fn multiple_producers_work_independently() {
    let mut builder = ActorBuilder::<AlphaData, AlphaControl, AlphaManagement>::new(1024, None);

    // Create multiple producers (each gets dedicated SPSC channels)
    let h1 = builder.add_producer();
    let h2 = builder.add_producer();
    let h3 = builder.add_producer();
    let h4 = builder.add_producer();

    let mut alpha_s = builder.build();

    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct CountActor(Arc<AtomicUsize>);
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for CountActor {
            fn handle_data(&mut self, _: AlphaData) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: AlphaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        alpha_s.run(&mut CountActor(count_clone));
    });

    // Send from all producers
    h1.send(Message::Data(AlphaData("1".to_string()))).unwrap();
    h2.send(Message::Data(AlphaData("2".to_string()))).unwrap();
    h3.send(Message::Data(AlphaData("3".to_string()))).unwrap();

    // Drop some producers
    drop(h1);
    drop(h2);

    // Remaining producers still work
    h4.send(Message::Data(AlphaData("4".to_string()))).unwrap();

    thread::sleep(Duration::from_millis(50));

    // Drop all
    drop(h3);
    drop(h4);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 4);
}

// ============================================================================
// Barrier Pattern Tests (coordinated startup)
// ============================================================================

#[test]
fn actors_can_coordinate_startup_with_barrier() {
    let (alpha_h, mut alpha_s) = ActorScheduler::<(), (), ()>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<(), (), ()>::new(100, 1024);

    let barrier = Arc::new(Barrier::new(3)); // 2 actors + 1 main

    let alpha_started = Arc::new(AtomicBool::new(false));
    let beta_started = Arc::new(AtomicBool::new(false));

    let barrier_a = barrier.clone();
    let started_a = alpha_started.clone();
    let alpha_thread = thread::spawn(move || {
        // Wait for everyone before processing
        barrier_a.wait();
        started_a.store(true, Ordering::SeqCst);

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
        alpha_s.run(&mut NoopActor);
    });

    let barrier_b = barrier.clone();
    let started_b = beta_started.clone();
    let beta_thread = thread::spawn(move || {
        barrier_b.wait();
        started_b.store(true, Ordering::SeqCst);

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
        beta_s.run(&mut NoopActor);
    });

    // Neither should have started yet
    thread::sleep(Duration::from_millis(10));
    assert!(!alpha_started.load(Ordering::SeqCst));
    assert!(!beta_started.load(Ordering::SeqCst));

    // Release the barrier
    barrier.wait();

    // Now both should start
    thread::sleep(Duration::from_millis(50));
    assert!(alpha_started.load(Ordering::SeqCst));
    assert!(beta_started.load(Ordering::SeqCst));

    drop(alpha_h);
    drop(beta_h);

    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();
}

// ============================================================================
// Message::Shutdown Tests
// ============================================================================

#[test]
fn shutdown_message_causes_actor_exit() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);

    let exited = Arc::new(AtomicBool::new(false));
    let exited_clone = exited.clone();

    let handle = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: AlphaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        alpha_s.run(&mut NoopActor);
        exited_clone.store(true, Ordering::SeqCst);
    });

    // Verify running
    thread::sleep(Duration::from_millis(20));
    assert!(!exited.load(Ordering::SeqCst));

    // Send shutdown
    alpha_h.send(Message::Shutdown).unwrap();

    // Should exit
    handle.join().unwrap();
    assert!(exited.load(Ordering::SeqCst));
}

#[test]
fn shutdown_works_with_multiple_actors() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let alpha_exited = Arc::new(AtomicBool::new(false));
    let beta_exited = Arc::new(AtomicBool::new(false));

    let alpha_exit = alpha_exited.clone();
    let beta_exit = beta_exited.clone();

    let alpha_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: AlphaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: AlphaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        alpha_s.run(&mut NoopActor);
        alpha_exit.store(true, Ordering::SeqCst);
    });

    let beta_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<BetaData, BetaControl, BetaManagement> for NoopActor {
            fn handle_data(&mut self, _: BetaData) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: BetaControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: BetaManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        beta_s.run(&mut NoopActor);
        beta_exit.store(true, Ordering::SeqCst);
    });

    // Verify both running
    thread::sleep(Duration::from_millis(20));
    assert!(!alpha_exited.load(Ordering::SeqCst));
    assert!(!beta_exited.load(Ordering::SeqCst));

    // Shutdown both (simulating directory.shutdown())
    beta_h.send(Message::Shutdown).unwrap();
    alpha_h.send(Message::Shutdown).unwrap();

    // Both should exit
    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();

    assert!(alpha_exited.load(Ordering::SeqCst));
    assert!(beta_exited.load(Ordering::SeqCst));
}
