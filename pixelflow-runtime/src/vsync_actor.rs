//! VSync Actor - Separate thread that generates vsync timing signals.
//!
//! The VSync actor runs in its own thread and can be controlled via message passing.
//! It sends periodic vsync signals that the engine uses for frame timing.
//!
//! # clock thread
//! To avoid scheduling starvation, the VSync timing is driven by a dedicated
//! clock thread that sends explicit `Tick` messages to the actor. This ensures
//! the actor wakes up reliably regardless of other system load, without relying
//! on blocking `park` calls that could stall the actor scheduler.

use actor_scheduler::{
    Actor, ActorBuilder, ActorHandle, HandlerError, HandlerResult, SystemStatus,
};
use log::info;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

/// Global VSync token bucket - controls frame request rate.
/// VSync decrements before sending ticks, engine/app increments when manifold ready.
/// This is separate from FPS measurement (which counts actual rasterization).
static VSYNC_TOKEN_BUCKET: AtomicU32 = AtomicU32::new(MAX_TOKENS);

/// Default refresh rate in Hz.
const DEFAULT_REFRESH_RATE: f64 = 60.0;

/// Try to consume a VSync token. Returns true if token was available.
#[inline]
pub(crate) fn try_consume_vsync_token() -> bool {
    VSYNC_TOKEN_BUCKET
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |tokens| {
            if tokens > 0 {
                Some(tokens - 1)
            } else {
                None
            }
        })
        .is_ok()
}

/// Return a VSync token to the bucket (up to MAX_TOKENS).
#[inline]
pub(crate) fn return_vsync_token() {
    let prev = VSYNC_TOKEN_BUCKET.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |tokens| {
        if tokens < MAX_TOKENS {
            Some(tokens + 1)
        } else {
            None
        }
    });

    // Rate-limit warning to 1% of calls to avoid log spam
    if prev.is_err() {
        static WARN_COUNTER: AtomicU32 = AtomicU32::new(0);
        if WARN_COUNTER
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(100)
        {
            log::warn!("VSync token bucket already at max capacity");
        }
    }
}

/// Get current token count (for debugging).
#[inline]
pub(crate) fn vsync_token_count() -> u32 {
    VSYNC_TOKEN_BUCKET.load(Ordering::Relaxed)
}

/// Configuration for VsyncActor
#[derive(Debug, Clone)]
pub struct VsyncConfig {
    pub refresh_rate: f64,
}

impl Default for VsyncConfig {
    fn default() -> Self {
        Self {
            refresh_rate: DEFAULT_REFRESH_RATE,
        }
    }
}

/// Messages TO the VSync actor (commands) - Control lane
#[derive(Debug, Default)]
pub enum VsyncCommand {
    /// Start sending vsync signals
    Start,
    /// Stop sending vsync signals (pause)
    Stop,
    /// Update refresh rate (for VRR displays)
    UpdateRefreshRate(f64),
    /// Request current FPS stats
    RequestCurrentFPS(Sender<f64>),
    /// Shutdown the actor
    #[default]
    Shutdown,
}
actor_scheduler::impl_control_message!(VsyncCommand);

/// Response from engine after rendering a frame - Data lane
#[derive(Debug, Clone, Copy)]
pub struct RenderedResponse {
    /// Frame number that was rendered
    pub frame_number: u64,
    /// When the frame was rendered
    pub rendered_at: Instant,
}
actor_scheduler::impl_data_message!(RenderedResponse);

/// Management messages
#[derive(Debug)]
pub enum VsyncManagement {
    /// Internal clock tick - wakes the actor to check vsync timing
    Tick,
    /// Configure the vsync actor (set refresh rate, engine handle, etc.)
    SetConfig {
        config: VsyncConfig,
        engine_handle: Box<crate::api::private::EngineActorHandle>,
        self_handle: Box<ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>>,
    },
}
actor_scheduler::impl_management_message!(VsyncManagement);

/// Internal commands sent to the clock thread
#[derive(Debug)]
enum ClockCommand {
    /// Update the tick interval
    SetInterval(Duration),
    /// Stop the clock thread
    Stop,
}

/// VSync actor - generates periodic vsync timing signals.
pub struct VsyncActor {
    engine_handle: Option<crate::api::private::EngineActorHandle>,

    // VSync state
    refresh_rate: f64,
    interval: Duration,
    running: bool,
    next_vsync: Instant,

    // FPS tracking (actual rasterization rate, not token rate)
    frame_count: u64,
    fps_start: Instant,
    last_fps: f64,

    // Control for the clock thread
    clock_control: Option<Sender<ClockCommand>>,
}

const MAX_TOKENS: u32 = 100;

impl VsyncActor {
    /// Helper to spawn the clock thread.
    fn spawn_clock_thread(
        interval: Duration,
        self_handle: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    ) -> Sender<ClockCommand> {
        let (clock_tx, clock_rx) = std::sync::mpsc::channel();

        thread::Builder::new()
            .name("vsync-clock".to_string())
            .spawn(move || {
                let mut current_interval = interval;
                loop {
                    match clock_rx.recv_timeout(current_interval) {
                        Ok(ClockCommand::Stop) => break,
                        Ok(ClockCommand::SetInterval(d)) => current_interval = d,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            // Time to tick
                            if self_handle.send(VsyncManagement::Tick).is_err() {
                                // Actor is gone
                                break;
                            }
                        }
                        Err(_) => break, // Channel disconnected
                    }
                }
            })
            .expect("Failed to spawn vsync clock thread");

        clock_tx
    }

    /// Create empty VsyncActor for troupe pattern - configured via SetConfig management message.
    #[must_use]
    pub fn new_empty() -> Self {
        Self {
            engine_handle: None,
            refresh_rate: DEFAULT_REFRESH_RATE,
            interval: Duration::from_secs_f64(1.0 / DEFAULT_REFRESH_RATE),
            running: false,
            next_vsync: Instant::now(),
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
            clock_control: None,
        }
    }

    /// Create a new VsyncActor. Takes the handle to itself (for the clock thread).
    pub fn new(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle,
        self_handle: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    ) -> Self {
        let interval = Duration::from_secs_f64(1.0 / refresh_rate);

        info!(
            "VsyncActor: Started with refresh rate {:.2} Hz ({:.2}ms interval), token bucket max: {}",
            refresh_rate,
            interval.as_secs_f64() * 1000.0,
            MAX_TOKENS
        );

        // Spawn the clock thread — self_handle is a dedicated SPSC channel
        let clock_tx = Self::spawn_clock_thread(interval, self_handle);

        Self {
            engine_handle: Some(engine_handle),
            refresh_rate,
            interval,
            running: false,
            next_vsync: Instant::now(),
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
            clock_control: Some(clock_tx),
        }
    }

    /// Spawn VSync actor in a new thread.
    ///
    /// Returns an ActorHandle that can be used to send commands and responses.
    pub fn spawn(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle,
    ) -> ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement> {
        let mut builder =
            ActorBuilder::<RenderedResponse, VsyncCommand, VsyncManagement>::new(1024, None);
        let handle = builder.add_producer(); // For the caller
        let clock_handle = builder.add_producer(); // For the clock thread (self-handle)
        let mut scheduler = builder.build();

        thread::Builder::new()
            .name("vsync-actor".to_string())
            .spawn(move || {
                let mut actor = VsyncActor::new(refresh_rate, engine_handle, clock_handle);
                scheduler.run(&mut actor);
            })
            .expect("Failed to spawn vsync actor thread");

        handle
    }

    fn send_vsync(&mut self) {
        let Some(ref engine_handle) = self.engine_handle else {
            return; // Not configured yet
        };

        let now = Instant::now();
        let timestamp = now;
        let target_timestamp = now + self.interval;

        use crate::api::private::EngineData;
        use actor_scheduler::Message;

        if engine_handle
            .send(Message::Data(EngineData::VSync {
                timestamp,
                target_timestamp,
                refresh_interval: self.interval,
            }))
            .is_ok()
        {
            // Calculate next vsync (no cumulative drift)
            self.next_vsync = timestamp + self.interval;
        } else {
            // Engine dropped the receiver
            info!("VsyncActor: Engine disconnected");
        }
    }

    fn update_fps(&mut self) {
        let elapsed = self.fps_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.last_fps = self.frame_count as f64 / elapsed.as_secs_f64();
            info!("VsyncActor: Current FPS: {:.2}", self.last_fps);
            self.frame_count = 0;
            self.fps_start = Instant::now();
        }
    }

    fn handle_tick(&mut self) {
        if !self.running {
            return;
        }

        let now = Instant::now();

        // If it's time for next vsync (or close enough/past due), check token bucket and send
        if now >= self.next_vsync {
            if try_consume_vsync_token() {
                log::trace!(
                    "VsyncActor: Token consumed, {} remaining",
                    vsync_token_count()
                );
                self.send_vsync();
            } else {
                // No tokens available - backpressure engaged
                log::trace!("VsyncActor: No tokens available, skipping vsync");
            }
        }
    }
}

impl Actor<RenderedResponse, VsyncCommand, VsyncManagement> for VsyncActor {
    fn handle_data(&mut self, response: RenderedResponse) -> HandlerResult {
        // Count actual rendered frames for accurate FPS measurement
        // (Token management is now handled via atomic bucket)
        self.frame_count += 1;
        log::trace!("VsyncActor: Frame {} rendered", response.frame_number);
        self.update_fps();
        Ok(())
    }

    fn handle_control(&mut self, cmd: VsyncCommand) -> HandlerResult {
        match cmd {
            VsyncCommand::Start => {
                self.running = true;
                self.next_vsync = Instant::now(); // Reset timing
                info!("VsyncActor: Started");
            }
            VsyncCommand::Stop => {
                self.running = false;
                info!("VsyncActor: Stopped");
            }
            VsyncCommand::UpdateRefreshRate(new_rate) => {
                self.refresh_rate = new_rate;
                self.interval = Duration::from_secs_f64(1.0 / self.refresh_rate);
                info!(
                    "VsyncActor: Updated refresh rate to {:.2} Hz ({:.2}ms interval)",
                    self.refresh_rate,
                    self.interval.as_secs_f64() * 1000.0
                );

                // Update clock thread
                if let Some(ref tx) = self.clock_control {
                    tx.send(ClockCommand::SetInterval(self.interval))
                        .expect("Failed to update clock thread interval");
                }
            }
            VsyncCommand::RequestCurrentFPS(sender) => {
                info!("VsyncActor: FPS requested - {:.2} fps", self.last_fps);
                if let Err(e) = sender.send(self.last_fps) {
                    log::warn!("VsyncActor: Failed to send FPS response: {:?}", e);
                }
            }
            VsyncCommand::Shutdown => {
                info!("VsyncActor: Shutting down");
                if let Some(ref tx) = self.clock_control {
                    tx.send(ClockCommand::Stop)
                        .expect("Failed to stop clock thread on shutdown");
                }
                // Scheduler will exit when all senders are dropped
                // We should probably drop our own handles if we held any that loop back?
                // But we don't hold loopback handles in struct, only for clock thread.
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: VsyncManagement) -> HandlerResult {
        match msg {
            VsyncManagement::Tick => self.handle_tick(),
            VsyncManagement::SetConfig {
                config,
                engine_handle,
                self_handle,
            } => {
                // Configure the vsync actor (called via Management after construction)
                self.engine_handle = Some(*engine_handle);
                self.refresh_rate = config.refresh_rate;
                self.interval = Duration::from_secs_f64(1.0 / config.refresh_rate);

                info!("VsyncActor: Configured with {:.2} Hz", config.refresh_rate);

                // Spawn clock thread
                let clock_tx = Self::spawn_clock_thread(self.interval, *self_handle);
                self.clock_control = Some(clock_tx);

                // Auto-start after configuration (clock thread is ready now)
                // This avoids priority inversion where Start (Control) runs before SetConfig (Management)
                self.running = true;
                self.next_vsync = Instant::now();
                info!("VsyncActor: Auto-started after configuration");
            }
        }
        Ok(())
    }

    fn park(
        &mut self,
        _status: SystemStatus,
    ) -> Result<actor_scheduler::ActorStatus, HandlerError> {
        // VSync is driven by internal clock thread sending messages
        Ok(actor_scheduler::ActorStatus::Idle)
    }
}

// ActorTypes impl for VsyncActor
impl actor_scheduler::ActorTypes for VsyncActor {
    type Data = RenderedResponse;
    type Control = VsyncCommand;
    type Management = VsyncManagement;
}

// TroupeActor impl for VsyncActor — ignores directory, configured via SetConfig message
impl<Dir> actor_scheduler::TroupeActor<Dir> for VsyncActor {
    fn new(_dir: Dir) -> Self {
        Self::new_empty()
    }
}
