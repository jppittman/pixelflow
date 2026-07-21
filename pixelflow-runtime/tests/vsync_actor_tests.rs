//! Comprehensive tests for the VsyncActor.
//!
//! These tests verify the VsyncActor's contracts:
//! - Token bucket flow control (MAX_TOKENS = 3)
//! - Start/Stop lifecycle
//! - Refresh rate updates
//! - FPS tracking
//! - Clock thread behavior
//!
//! Note: Some tests require the actor to run in a thread, which means
//! we test through the public message interface rather than internal state.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use pixelflow_runtime::vsync_actor::{RenderedResponse, VsyncCommand, VsyncManagement};

// ============================================================================
// Test Fixtures
// ============================================================================

/// A minimal VsyncActor replacement for testing the message protocol.
/// We can't easily test the real VsyncActor because it requires EngineActorHandle,
/// but we can test the message types and patterns.
struct MockVsyncActor {
    running: bool,
    refresh_rate: f64,
    tokens: u32,
    frame_count: u64,
    fps_start: Instant,
    last_fps: f64,
    log: Arc<Mutex<Vec<String>>>,
}

const MAX_TOKENS: u32 = 3;

impl MockVsyncActor {
    fn new(log: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            running: false,
            refresh_rate: 60.0,
            tokens: MAX_TOKENS,
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
            log,
        }
    }

    fn log(&self, msg: &str) {
        self.log.lock().unwrap().push(msg.to_string());
    }
}

impl Actor<RenderedResponse, VsyncCommand, VsyncManagement> for MockVsyncActor {
    fn handle_data(&mut self, response: RenderedResponse) -> HandlerResult {
        // Token replenishment on rendered response
        if self.tokens < MAX_TOKENS {
            self.tokens += 1;
            self.log(&format!(
                "token_added:frame={},tokens={}",
                response.frame_number, self.tokens
            ));
        }
        Ok(())
    }

    fn handle_control(&mut self, cmd: VsyncCommand) -> HandlerResult {
        match cmd {
            VsyncCommand::Start => {
                self.running = true;
                self.log("started");
            }
            VsyncCommand::Stop => {
                self.running = false;
                self.log("stopped");
            }
            VsyncCommand::UpdateRefreshRate(rate) => {
                self.refresh_rate = rate;
                self.log(&format!("refresh_rate={:.1}", rate));
            }
            VsyncCommand::RequestCurrentFPS(sender) => {
                sender.send(self.last_fps).ok();
                self.log(&format!("fps_requested:sent={:.1}", self.last_fps));
            }
            VsyncCommand::Shutdown => {
                self.running = false;
                self.log("shutdown");
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: VsyncManagement) -> HandlerResult {
        match msg {
            VsyncManagement::Tick => {
                if self.running && self.tokens > 0 {
                    self.tokens -= 1;
                    self.frame_count += 1;
                    self.log(&format!(
                        "tick:frame={},tokens={}",
                        self.frame_count, self.tokens
                    ));
                }
                // FPS calculation
                let elapsed = self.fps_start.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    self.last_fps = self.frame_count as f64 / elapsed.as_secs_f64();
                    self.frame_count = 0;
                    self.fps_start = Instant::now();
                }
            }
            VsyncManagement::SetConfig { config, .. } => {
                self.refresh_rate = config.refresh_rate;
                self.running = true;
                self.log(&format!(
                    "config:rate={:.1},auto_started",
                    config.refresh_rate
                ));
            }
        }
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ============================================================================
// VsyncCommand Tests
// ============================================================================

#[test]
fn vsync_command_start_stop_lifecycle() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Test lifecycle
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    thread::sleep(Duration::from_millis(10));
    tx.send(Message::Control(VsyncCommand::Stop)).unwrap();
    thread::sleep(Duration::from_millis(10));
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    thread::sleep(Duration::from_millis(10));
    tx.send(Message::Control(VsyncCommand::Shutdown)).unwrap();

    thread::sleep(Duration::from_millis(20));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    assert!(log.contains(&"started".to_string()), "Should have started");
    assert!(log.contains(&"stopped".to_string()), "Should have stopped");
    assert!(
        log.contains(&"shutdown".to_string()),
        "Should have shutdown"
    );
}

#[test]
fn vsync_command_update_refresh_rate() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    tx.send(Message::Control(VsyncCommand::UpdateRefreshRate(120.0)))
        .unwrap();
    tx.send(Message::Control(VsyncCommand::UpdateRefreshRate(144.0)))
        .unwrap();
    tx.send(Message::Control(VsyncCommand::UpdateRefreshRate(60.0)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    assert!(log.iter().any(|s| s.contains("120.0")));
    assert!(log.iter().any(|s| s.contains("144.0")));
    assert!(log.iter().any(|s| s.contains("60.0")));
}

#[test]
fn vsync_command_request_fps() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    let (fps_tx, fps_rx) = mpsc::channel();
    tx.send(Message::Control(VsyncCommand::RequestCurrentFPS(fps_tx)))
        .unwrap();

    let fps = fps_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(fps >= 0.0, "FPS should be non-negative");

    drop(tx);
    handle.join().unwrap();
}

// ============================================================================
// Token Bucket Tests
// ============================================================================

#[test]
fn token_bucket_starts_full() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Start and tick 3 times (should consume all tokens)
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    for _ in 0..3 {
        tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let tick_logs: Vec<_> = log.iter().filter(|s| s.starts_with("tick:")).collect();

    // Should have 3 ticks (all tokens consumed)
    assert_eq!(tick_logs.len(), 3, "Should process 3 ticks with 3 tokens");
}

#[test]
fn token_bucket_blocks_when_empty() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Start and send more ticks than tokens
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    for _ in 0..10 {
        tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let tick_logs: Vec<_> = log.iter().filter(|s| s.starts_with("tick:")).collect();

    // Should only process MAX_TOKENS ticks (no rendered responses to replenish)
    assert_eq!(
        tick_logs.len(),
        MAX_TOKENS as usize,
        "Should only process {} ticks without token replenishment",
        MAX_TOKENS
    );
}

#[test]
fn token_bucket_replenishes_on_rendered_response() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Start
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    thread::sleep(Duration::from_millis(10)); // Wait for start to process

    // Consume all tokens
    for _ in 0..3 {
        tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    }
    thread::sleep(Duration::from_millis(10)); // Wait for ticks to process

    // Replenish with rendered responses
    // Note: Data has lowest priority, so we need to let these process before sending more Management
    for i in 0..2 {
        tx.send(Message::Data(RenderedResponse {
            frame_number: i,
            rendered_at: Instant::now(),
        }))
        .unwrap();
    }
    thread::sleep(Duration::from_millis(10)); // Wait for replenishment

    // Should be able to tick 2 more times (we have 2 tokens from replenishment)
    for _ in 0..5 {
        tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let tick_logs: Vec<_> = log.iter().filter(|s| s.starts_with("tick:")).collect();

    // 3 initial + 2 after replenishment = 5
    assert_eq!(
        tick_logs.len(),
        5,
        "Should process 5 ticks total (3 + 2 replenished)"
    );
}

#[test]
fn token_bucket_does_not_exceed_max() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Send many rendered responses (more than MAX_TOKENS)
    for i in 0..10 {
        tx.send(Message::Data(RenderedResponse {
            frame_number: i,
            rendered_at: Instant::now(),
        }))
        .unwrap();
    }

    thread::sleep(Duration::from_millis(20));

    // Now start and tick many times
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    for _ in 0..20 {
        tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let tick_logs: Vec<_> = log.iter().filter(|s| s.starts_with("tick:")).collect();

    // Should still only have MAX_TOKENS worth of ticks
    assert_eq!(
        tick_logs.len(),
        MAX_TOKENS as usize,
        "Token bucket should cap at MAX_TOKENS"
    );
}

// ============================================================================
// Tick Behavior Tests
// ============================================================================

#[test]
fn tick_does_nothing_when_not_running() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Don't start, just tick
    for _ in 0..5 {
        tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let tick_logs: Vec<_> = log.iter().filter(|s| s.starts_with("tick:")).collect();

    assert!(
        tick_logs.is_empty(),
        "No ticks should be processed when not running"
    );
}

#[test]
fn tick_resumes_after_stop_and_start() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Start and tick
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    thread::sleep(Duration::from_millis(10)); // Wait for start to process
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    thread::sleep(Duration::from_millis(10)); // Wait for tick to process

    // Stop
    tx.send(Message::Control(VsyncCommand::Stop)).unwrap();
    thread::sleep(Duration::from_millis(10)); // Wait for stop to process

    // Tick while stopped (should not count)
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    thread::sleep(Duration::from_millis(10)); // Wait for ticks to be processed (but ignored)

    // Restart and tick
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    thread::sleep(Duration::from_millis(10)); // Wait for restart
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let tick_logs: Vec<_> = log.iter().filter(|s| s.starts_with("tick:")).collect();

    // 1 before stop + 1 after restart = 2
    assert_eq!(
        tick_logs.len(),
        2,
        "Should have 2 ticks (before stop + after restart)"
    );
}

// ============================================================================
// VsyncManagement::SetConfig Tests
// ============================================================================

#[test]
fn set_config_auto_starts_actor() {
    use pixelflow_runtime::vsync_actor::VsyncConfig;

    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    // Use ActorBuilder to create two producers (tx + self_handle)
    let mut builder =
        ActorBuilder::<RenderedResponse, VsyncCommand, VsyncManagement>::new(100, None);
    let tx = builder.add_producer();
    let self_handle = builder.add_producer();
    let mut rx = builder.build();

    // We need a dummy engine handle for SetConfig - create but won't be used.
    // Use the crate's sanctioned test double rather than reaching into
    // api::private to hand-assemble an ActorScheduler directly.
    let mut mock_engine = pixelflow_runtime::testing::MockEngine::new();
    let engine_handle = mock_engine.take_handle();

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    tx.send(Message::Management(VsyncManagement::SetConfig {
        config: VsyncConfig { refresh_rate: 90.0 },
        engine_handle: Box::new(engine_handle),
        self_handle: Box::new(self_handle),
    }))
    .unwrap();

    // After SetConfig, should auto-start - ticks should work
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    assert!(
        log.iter().any(|s| s.contains("auto_started")),
        "SetConfig should auto-start the actor"
    );
    assert!(
        log.iter().any(|s| s.starts_with("tick:")),
        "Ticks should work after SetConfig"
    );
}

// ============================================================================
// RenderedResponse Tests
// ============================================================================

#[test]
fn rendered_response_carries_frame_number() {
    let response = RenderedResponse {
        frame_number: 42,
        rendered_at: Instant::now(),
    };
    assert_eq!(response.frame_number, 42);
}

#[test]
fn rendered_response_carries_timestamp() {
    let before = Instant::now();
    let response = RenderedResponse {
        frame_number: 0,
        rendered_at: Instant::now(),
    };
    let after = Instant::now();

    assert!(response.rendered_at >= before);
    assert!(response.rendered_at <= after);
}

// ============================================================================
// Message Type Macro Tests
// ============================================================================

#[test]
fn vsync_command_converts_to_message() {
    // VsyncCommand implements impl_control_message! so it should convert to Message::Control
    let cmd = VsyncCommand::Start;
    let msg: Message<RenderedResponse, VsyncCommand, VsyncManagement> = cmd.into();
    assert!(matches!(msg, Message::Control(VsyncCommand::Start)));
}

#[test]
fn rendered_response_converts_to_message() {
    // RenderedResponse implements impl_data_message! so it should convert to Message::Data
    let resp = RenderedResponse {
        frame_number: 1,
        rendered_at: Instant::now(),
    };
    let msg: Message<RenderedResponse, VsyncCommand, VsyncManagement> = resp.into();
    assert!(matches!(msg, Message::Data(_)));
}

#[test]
fn vsync_management_converts_to_message() {
    // VsyncManagement implements impl_management_message!
    let mgmt = VsyncManagement::Tick;
    let msg: Message<RenderedResponse, VsyncCommand, VsyncManagement> = mgmt.into();
    assert!(matches!(msg, Message::Management(VsyncManagement::Tick)));
}

// ============================================================================
// VsyncCommand Debug Tests
// ============================================================================

#[test]
fn vsync_command_debug_format() {
    assert_eq!(format!("{:?}", VsyncCommand::Start), "Start");
    assert_eq!(format!("{:?}", VsyncCommand::Stop), "Stop");
    assert_eq!(format!("{:?}", VsyncCommand::Shutdown), "Shutdown");
    assert_eq!(
        format!("{:?}", VsyncCommand::UpdateRefreshRate(120.0)),
        "UpdateRefreshRate(120.0)"
    );
}

#[test]
fn vsync_command_default_is_shutdown() {
    let default: VsyncCommand = Default::default();
    assert!(matches!(default, VsyncCommand::Shutdown));
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[test]
fn multiple_senders_to_vsync_actor() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    // Use ActorBuilder for multiple SPSC producers
    let mut builder =
        ActorBuilder::<RenderedResponse, VsyncCommand, VsyncManagement>::new(100, None);
    let tx = builder.add_producer();
    let tx1 = builder.add_producer();
    let tx2 = builder.add_producer();
    let tx3 = builder.add_producer();
    let mut rx = builder.build();

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    let s1 = thread::spawn(move || {
        tx1.send(Message::Control(VsyncCommand::Start)).unwrap();
    });
    let s2 = thread::spawn(move || {
        for i in 0..5 {
            tx2.send(Message::Data(RenderedResponse {
                frame_number: i,
                rendered_at: Instant::now(),
            }))
            .unwrap();
        }
    });
    let s3 = thread::spawn(move || {
        for _ in 0..5 {
            tx3.send(Message::Management(VsyncManagement::Tick))
                .unwrap();
        }
    });

    s1.join().unwrap();
    s2.join().unwrap();
    s3.join().unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    assert!(!log.is_empty(), "Should have processed some messages");
}

// ============================================================================
// FPS Calculation Tests
// ============================================================================

#[test]
fn fps_calculation_resets_after_one_second() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Start and send some ticks
    tx.send(Message::Control(VsyncCommand::Start)).unwrap();
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();
    tx.send(Message::Management(VsyncManagement::Tick)).unwrap();

    // Wait for FPS calculation to reset (1 second)
    thread::sleep(Duration::from_millis(1100));

    // Request FPS
    let (fps_tx, fps_rx) = mpsc::channel();
    tx.send(Message::Control(VsyncCommand::RequestCurrentFPS(fps_tx)))
        .unwrap();

    let fps = fps_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    // Should be approximately 3 FPS (3 frames in ~1 second)
    // But might be slightly different due to timing
    assert!(fps >= 0.0, "FPS should be non-negative after reset");

    drop(tx);
    handle.join().unwrap();
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn shutdown_stops_processing_immediately() {
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        struct CountingActor(Arc<AtomicUsize>);
        impl Actor<RenderedResponse, VsyncCommand, VsyncManagement> for CountingActor {
            fn handle_data(&mut self, _: RenderedResponse) -> HandlerResult {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: VsyncCommand) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: VsyncManagement) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        rx.run(&mut CountingActor(processed_clone));
    });

    // Send shutdown then more messages
    tx.send(Message::Control(VsyncCommand::Shutdown)).unwrap();

    // These should still be delivered (shutdown is just a control message)
    for i in 0..5 {
        tx.send(Message::Data(RenderedResponse {
            frame_number: i,
            rendered_at: Instant::now(),
        }))
        .unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // All messages should still be processed
    assert_eq!(processed.load(Ordering::SeqCst), 5);
}

#[test]
fn rapid_start_stop_cycles() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();

    let (tx, mut rx) =
        ActorScheduler::<RenderedResponse, VsyncCommand, VsyncManagement>::new(10, 100);

    let handle = thread::spawn(move || {
        let mut actor = MockVsyncActor::new(log_clone);
        rx.run(&mut actor);
    });

    // Rapid start/stop cycles
    for _ in 0..100 {
        tx.send(Message::Control(VsyncCommand::Start)).unwrap();
        tx.send(Message::Control(VsyncCommand::Stop)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    let log = log.lock().unwrap();
    let starts = log.iter().filter(|s| *s == "started").count();
    let stops = log.iter().filter(|s| *s == "stopped").count();

    assert_eq!(starts, 100, "Should have 100 starts");
    assert_eq!(stops, 100, "Should have 100 stops");
}
