//! Priority Channel - A multi-priority message passing system
//!
//! This crate provides a message scheduler with three priority levels:
//! - **Control**: Highest priority, burst-limited to prevent starvation
//! - **Management**: Medium priority, burst-limited
//! - **Data**: Lowest priority, burst-limited with backpressure
//!
//! # Architecture
//!
//! The scheduler uses a "doorbell" pattern where:
//! 1. The receiver blocks on the Control channel
//! 2. Data messages send a Wake signal to unblock the receiver
//! 3. Priority processing drains Control → Management → Data
//!
//! # Troupe System
//!
//! The troupe system provides lifecycle management for groups of actors.
//! Troupes can nest - a child troupe's `play()` can run inside a parent's spawned thread.
//!
//! ## Basic Usage
//!
//! ```ignore
//! troupe! {
//!     engine: EngineActor [expose],    // handle exposed to parent
//!     vsync: VsyncActor,               // internal only
//!     display: DisplayActor [main],    // runs on calling thread
//! }
//!
//! // Simple: create and run in one step
//! run().expect("troupe failed");
//! ```
//!
//! ## Two-Phase Initialization (for nesting)
//!
//! ```ignore
//! // Phase 1: Create child troupe (no threads yet)
//! let child = Troupe::new();
//!
//! // Phase 2: Parent grabs exposed handles
//! let child_engine = child.exposed().engine;
//!
//! // Phase 3: Spawn child troupe as an actor in parent
//! s.spawn(|| child.play());
//!
//! // Parent can now send to child_engine
//! ```
//!
//! ## Nesting Architecture
//!
//! ```text
//! RootTroupe.play()                          <- main thread (GUI)
//! ├── spawn thread -> ActorA.run()
//! ├── spawn thread -> ChildTroupe.play()    <- blocks, owns scoped threads
//! │   ├── spawn thread -> ChildActorX.run()
//! │   └── ChildActorY.run() [child's main]
//! └── RootMainActor.run() [root's main]     <- GUI actor, on main thread
//! ```
//!
//! # Example (Basic Scheduler)
//!
//! ```rust
//! use actor_scheduler::{ActorScheduler, Message, SchedulerHandler, ActorStatus, SystemStatus, HandlerResult, HandlerError};
//!
//! struct MyHandler;
//!
//! impl SchedulerHandler<String, String, String> for MyHandler {
//!     fn handle_data(&mut self, msg: String) -> HandlerResult {
//!         println!("Data: {}", msg);
//!         Ok(())
//!     }
//!     fn handle_control(&mut self, msg: String) -> HandlerResult {
//!         println!("Control: {}", msg);
//!         Ok(())
//!     }
//!     fn handle_management(&mut self, msg: String) -> HandlerResult {
//!         println!("Management: {}", msg);
//!         Ok(())
//!     }
//!     fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> { Ok(ActorStatus::Idle) }
//! }
//!
//! let (tx, mut rx) = ActorScheduler::<String, String, String>::new(10, 100);
//!
//! // Spawn receiver thread
//! std::thread::spawn(move || {
//!     let mut handler = MyHandler;
//!     rx.run(&mut handler);
//! });
//!
//! // Send messages from any thread
//! tx.send(Message::Data("low priority data".to_string())).unwrap();
//! tx.send(Message::Control("high priority control".to_string())).unwrap();
//! ```

mod error;
pub mod kubelet;
mod lifecycle;
mod params;
pub mod registry;
pub mod service;
pub mod sharded;
pub mod spsc;

use error::DrainStatus;
pub use error::{HandlerError, HandlerResult, SendError};
pub use kubelet::{Kubelet, KubeletBuilder, SpawnedPod, spawn_managed};
pub use lifecycle::{PodPhase, RestartPolicy};
pub use params::SchedulerParams;
pub use registry::{PodGone, PodSlot};
pub use service::{ServiceError, ServiceHandle};

// Re-export macros from the proc-macro crate
pub use actor_scheduler_macros::{actor_impl, troupe};

use sharded::{InboxBuilder, ShardedInbox};
use spsc::SpscSender;
use std::sync::{
    Arc,
    mpsc::{self, Receiver, SyncSender},
};
use std::time::Duration;

/// The types of messages supported by the scheduler.
///
/// Messages are organized into three priority lanes, with different guarantees and semantics.
///
/// # Message Lanes
///
/// | Lane | Priority | Throughput | Blocking | Use Case |
/// |------|----------|-----------|----------|----------|
/// | **Data** (D) | Lowest | High | Yes (backpressure) | Continuous, high-volume data |
/// | **Control** (C) | High | Medium | Unlimited | Time-critical state changes |
/// | **Management** (M) | Medium | Low | Unlimited | Lifecycle, configuration |
///
/// ## Data Lane (D)
///
/// **Purpose**: High-throughput, low-latency data messages.
///
/// **Contract**:
/// - **Sender**: Sends data continuously; may block on full buffer
/// - **Receiver**: Drains after Control and Management, subject to burst limiting
/// - **Guarantee**: Best-effort delivery; may drop if buffer overflows
/// - **Ordering**: FIFO within lane
///
/// **Example**: Frame data, sensor readings, streaming events
///
/// ## Control Lane (C)
///
/// **Purpose**: Time-critical control messages that need immediate attention.
///
/// **Contract**:
/// - **Sender**: Retries with exponential backoff if buffer full (prevents slow-loris attacks)
/// - **Receiver**: Drains before Management and Data, with burst limit to prevent starvation
/// - **Guarantee**: Best-effort priority delivery (bounded buffer with backoff)
/// - **Ordering**: FIFO within lane, typically processed before Data/Management
///
/// **Example**: User input (keypresses, mouse), window resize, close requests
///
/// ## Management Lane (M)
///
/// **Purpose**: Configuration and lifecycle messages.
///
/// **Contract**:
/// - **Sender**: Retries with exponential backoff if buffer full
/// - **Receiver**: Drains between Control and Data, with burst limiting
/// - **Guarantee**: Best-effort delivery (bounded buffer with backoff)
/// - **Ordering**: FIFO within lane
///
/// **Example**: Configuration changes, resource allocation, subscription/unsubscription
///
/// # Scheduling Strategy
///
/// The scheduler drains messages in priority order with burst limits:
///
/// ```text
/// Loop:
///   1. Drain Control messages (capped at burst limit)
///   2. Drain Management messages (capped at burst limit)
///   3. Drain Control messages again (priority recheck)
///   4. Drain Data messages (capped at burst limit)
///   5. Call park() - let actor/OS do other work
///   6. Repeat
/// ```
///
/// This provides best-effort priority with starvation protection:
/// - Control messages typically process before Data/Management
/// - All lanes are burst-limited to prevent monopolization
/// - No cross-lane ordering guarantees - only best-effort priority
/// - Protection against slow-loris attacks (poorly-behaved senders can't drown channels)
///
/// Configurable shutdown behavior per actor via `ShutdownMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShutdownMode {
    /// Exit immediately, drop all pending messages (default, current behavior)
    #[default]
    Immediate,

    /// Drain control+management lanes, drop data
    /// Use for actors where control/management cleanup is critical
    DrainControl,

    /// Process all pending messages before exit (with timeout fallback)
    /// Use for actors that must process all messages (e.g., logging, persistence)
    DrainAll { timeout: std::time::Duration },
}

#[derive(Debug)]
pub enum Message<D, C, M> {
    /// A data message (lowest priority, high throughput).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - May block if buffer is full (backpressure)
    /// - Should not send Control/Management equivalent if Data suffices
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_data()`
    /// - Processing is deferred behind Control and Management
    /// - May be burst-limited (batches processed per iteration)
    ///
    /// # Example
    ///
    /// ```ignore
    /// tx.send(Message::Data(PixelData { x: 100, y: 50, color: red }))?;
    /// // May block if the 10-message buffer is full
    /// ```
    Data(D),

    /// A control message (highest priority, time-critical).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - Retries with exponential backoff if buffer is full
    /// - Use for messages that need priority (user input, resize events)
    /// - Backoff prevents poorly-behaved senders from monopolizing the channel
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_control()`
    /// - Best-effort priority processing before Data/Management messages
    /// - Draining is burst-limited to prevent starvation of other lanes
    ///
    /// # Example
    ///
    /// ```ignore
    /// // User clicked the close button - this should be processed with priority
    /// tx.send(Message::Control(CloseRequested))?;
    /// // Retries with backoff if buffer full
    /// ```
    ///
    /// # Backpressure
    ///
    /// Control messages use a bounded buffer with exponential backoff on retry.
    /// If senders overwhelm the receiver, they will experience increasing delays.
    /// This prevents poorly-behaved senders from monopolizing the control channel.
    Control(C),

    /// A management message (medium priority, configuration/lifecycle).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - Retries with exponential backoff if buffer is full
    /// - Use for lifecycle and configuration (create, destroy, configure)
    /// - Lower priority than Control but higher than Data
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_management()`
    /// - Best-effort delivery (bounded buffer with backoff)
    /// - Typically processed after Control but before Data messages
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Allocate a new resource - this doesn't need to be super-urgent
    /// // but it's more important than continuous data stream
    /// tx.send(Message::Management(AllocateBuffer { size: 1024 }))?;
    /// ```
    Management(M),

    /// Shutdown signal.
    ///
    /// # Contract
    ///
    /// **Sender**: Signals that the actor should shut down cleanly.
    ///
    /// **Receiver**: The scheduler handles this directly—the actor never sees it.
    /// When the scheduler receives `Shutdown`, it exits the run loop immediately
    /// and `rx.run()` returns.
    ///
    /// # Implementation Details
    ///
    /// - This is never delivered to the actor's `handle_*` methods
    /// - It's a special signal interpreted by the scheduler itself
    /// - Useful for graceful shutdown of the actor system
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Shut down the actor
    /// tx.send(Message::Shutdown)?;
    /// // rx.run() will exit and return
    /// ```
    Shutdown,
}

/// Implement From for a Control message type.
#[macro_export]
macro_rules! impl_control_message {
    ($ty:ty) => {
        impl<D, M> From<$ty> for $crate::Message<D, $ty, M> {
            fn from(msg: $ty) -> Self {
                $crate::Message::Control(msg)
            }
        }
    };
}

/// Implement From for a Data message type.
#[macro_export]
macro_rules! impl_data_message {
    ($ty:ty) => {
        impl<C, M> From<$ty> for $crate::Message<$ty, C, M> {
            fn from(msg: $ty) -> Self {
                $crate::Message::Data(msg)
            }
        }
    };
}

/// Implement From for a Management message type.
#[macro_export]
macro_rules! impl_management_message {
    ($ty:ty) => {
        impl<D, C> From<$ty> for $crate::Message<D, C, $ty> {
            fn from(msg: $ty) -> Self {
                $crate::Message::Management(msg)
            }
        }
    };
}

/// Actor status returned from park() to hint the scheduler about blocking behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorStatus {
    Idle, // Actor has no unfinished work. Scheduler can block. (0% CPU)
    Busy, // Actor has unfinished work (yielding). Scheduler should poll.
}

/// Status provided to the actor's park method indicating the state of the scheduler's queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemStatus {
    Idle, // Scheduler queues are empty
    Busy, // Scheduler queues have more work (burst limit reached)
}

/// The Actor trait - implement this to define your actor's behavior.
///
/// Actors process messages from three priority lanes:
/// - **Data** (D): High-throughput data messages
/// - **Control** (C): Time-critical control messages
/// - **Management** (M): Lifecycle and configuration messages
pub trait Actor<D, C, M> {
    /// Handle a data message.
    ///
    /// Returns `Ok(())` on success, or a `HandlerError` on failure.
    /// - `HandlerError::Temporary`: Scheduler logs and continues
    /// - `HandlerError::Fatal`: Scheduler initiates shutdown
    fn handle_data(&mut self, msg: D) -> HandlerResult;

    /// Handle a control message.
    ///
    /// Returns `Ok(())` on success, or a `HandlerError` on failure.
    /// - `HandlerError::Temporary`: Scheduler logs and continues
    /// - `HandlerError::Fatal`: Scheduler initiates shutdown
    fn handle_control(&mut self, msg: C) -> HandlerResult;

    /// Handle a management message.
    ///
    /// Returns `Ok(())` on success, or a `HandlerError` on failure.
    /// - `HandlerError::Temporary`: Scheduler logs and continues
    /// - `HandlerError::Fatal`: Scheduler initiates shutdown
    fn handle_management(&mut self, msg: M) -> HandlerResult;

    /// The "Hook" where the Actor creates the bridge to the OS.
    /// Called when the scheduler has drained available messages (or hit burst limits).
    ///
    /// Returns actor status: Busy if yielding with unfinished work, Idle if done.
    /// Can return `HandlerError::Fatal` to trigger shutdown.
    fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError>;
}

/// Legacy alias for backward compatibility
#[deprecated(since = "0.2.0", note = "Use `Actor` instead")]
pub use Actor as SchedulerHandler;

/// Defines the message types for an actor managed by the troupe! macro.
///
/// This trait is separate from `TroupeActor` to allow extracting type information
/// without lifetime parameters, which is necessary for the `troupe!` macro to
/// generate struct field types.
///
/// # Example
///
/// ```ignore
/// impl ActorTypes for MyActor {
///     type Data = MyData;
///     type Control = MyControl;
///     type Management = MyManagement;
/// }
/// ```
pub trait ActorTypes {
    /// The data message type for this actor.
    type Data: Send + 'static;
    /// The control message type for this actor.
    type Control: Send + 'static;
    /// The management message type for this actor.
    type Management: Send + 'static;
}

/// The TroupeActor trait for actors managed by the troupe! macro.
///
/// Unlike the basic `Actor` trait, `TroupeActor` is parameterized over a Directory
/// type, enabling type-safe access to other actors in the group. The `#[actor_impl]`
/// macro generates the impl for this trait.
///
/// # Example
///
/// ```ignore
/// pub struct EngineActor<'a> {
///     dir: &'a Directory,
/// }
///
/// impl ActorTypes for EngineActor<'_> {
///     type Data = EngineData;
///     type Control = EngineControl;
///     type Management = EngineManagement;
/// }
///
/// impl<'a> TroupeActor<'a, Directory> for EngineActor<'a> {
///     fn new(dir: &'a Directory) -> Self { Self { dir } }
/// }
///
/// impl Actor<EngineData, EngineControl, EngineManagement> for EngineActor<'_> {
///     fn handle_data(&mut self, msg: EngineData) { }
///     fn handle_control(&mut self, msg: EngineControl) { }
///     fn handle_management(&mut self, msg: EngineManagement) { }
///     fn park(&mut self, status: SystemStatus) -> ActorStatus { ActorStatus::Idle }
/// }
/// ```
pub trait TroupeActor<Dir>:
    Sized
    + ActorTypes
    + Actor<
        <Self as ActorTypes>::Data,
        <Self as ActorTypes>::Control,
        <Self as ActorTypes>::Management,
    >
{
    /// Create a new actor from its directory of handles.
    ///
    /// With SPSC channels, each actor OWNS its directory instance.
    /// The directory contains dedicated SPSC handles to every other actor.
    fn new(dir: Dir) -> Self;
}

/// Create a new actor with one producer handle.
///
/// Convenience function for single-producer actors. For multi-producer setups
/// (e.g., troupe actors where multiple peers send to the same target), use
/// [`ActorBuilder`] directly.
///
/// # Arguments
/// * `data_buffer_size` - Size of bounded data buffer
/// * `wake_handler` - Optional wake handler for platform event loops
#[must_use]
pub fn create_actor<D, C, M>(
    data_buffer_size: usize,
    wake_handler: Option<Arc<dyn WakeHandler>>,
) -> (ActorHandle<D, C, M>, ActorScheduler<D, C, M>) {
    ActorScheduler::new_with_wake_handler(
        SchedulerParams::DEFAULT.default_data_burst_limit,
        data_buffer_size,
        wake_handler,
    )
}

/// Builder for multi-producer actor channels.
///
/// Each call to [`add_producer`](ActorBuilder::add_producer) creates a dedicated
/// SPSC channel per lane. Call [`build`](ActorBuilder::build) to seal the registry
/// and get the [`ActorScheduler`].
///
/// # Lifecycle
///
/// ```text
/// 1. ActorBuilder::new(buffer_size, waker)
/// 2. builder.add_producer()  → ActorHandle  (repeat N times)
/// 3. builder.build()         → ActorScheduler (seals registry)
/// ```
///
/// # Example
///
/// ```ignore
/// let mut builder = ActorBuilder::<Data, Control, Mgmt>::new(1024, None);
/// let handle_a = builder.add_producer();  // Actor A's dedicated channels
/// let handle_b = builder.add_producer();  // Actor B's dedicated channels
/// let mut scheduler = builder.build();    // Seals — no more producers
/// ```
pub struct ActorBuilder<D, C, M> {
    tx_doorbell: SyncSender<System>,
    rx_doorbell: Option<Receiver<System>>,
    data_inbox: InboxBuilder<D>,
    control_inbox: InboxBuilder<C>,
    mgmt_inbox: InboxBuilder<M>,
    wake_handler: Option<Arc<dyn WakeHandler>>,
    params: SchedulerParams,
}

impl<D, C, M> ActorBuilder<D, C, M> {
    /// Create a new builder with default scheduler parameters.
    ///
    /// # Arguments
    /// * `data_buffer_size` - Per-producer SPSC buffer size for the data lane
    /// * `wake_handler` - Optional platform wake handler (e.g., macOS Cocoa waker)
    #[must_use]
    pub fn new(data_buffer_size: usize, wake_handler: Option<Arc<dyn WakeHandler>>) -> Self {
        Self::new_with_params(data_buffer_size, wake_handler, SchedulerParams::DEFAULT)
    }

    /// Create a new builder with explicit tuning parameters.
    #[must_use]
    pub fn new_with_params(
        data_buffer_size: usize,
        wake_handler: Option<Arc<dyn WakeHandler>>,
        params: SchedulerParams,
    ) -> Self {
        assert!(
            data_buffer_size > 0,
            "data_buffer_size must be >= 1, got {}",
            data_buffer_size
        );
        params.validate();

        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);

        Self {
            tx_doorbell,
            rx_doorbell: Some(rx_doorbell),
            data_inbox: InboxBuilder::new(data_buffer_size),
            control_inbox: InboxBuilder::new(params.control_mgmt_buffer_size),
            mgmt_inbox: InboxBuilder::new(params.control_mgmt_buffer_size),
            wake_handler,
            params,
        }
    }

    /// Register a new producer. Returns a unique [`ActorHandle`] with dedicated
    /// SPSC channels to this actor's three priority lanes.
    ///
    /// Call this once per producer during initialization, before [`build`](Self::build).
    pub fn add_producer(&mut self) -> ActorHandle<D, C, M> {
        ActorHandle {
            tx_doorbell: self.tx_doorbell.clone(),
            tx_data: self.data_inbox.add_producer(),
            tx_control: self.control_inbox.add_producer(),
            tx_mgmt: self.mgmt_inbox.add_producer(),
            wake_handler: self.wake_handler.clone(),
            params: self.params,
        }
    }

    /// Seal the registry and return the scheduler.
    ///
    /// Uses default burst limits from [`SchedulerParams`].
    /// No more producers can be added after this call.
    #[must_use]
    pub fn build(self) -> ActorScheduler<D, C, M> {
        let burst = self.params.default_data_burst_limit;
        self.build_with_burst(burst, ShutdownMode::default())
    }

    /// Seal the registry with explicit burst limit and shutdown mode.
    #[must_use]
    pub fn build_with_burst(
        self,
        data_burst_limit: usize,
        shutdown_mode: ShutdownMode,
    ) -> ActorScheduler<D, C, M> {
        ActorScheduler {
            rx_doorbell: self.rx_doorbell.expect("ActorBuilder::build called twice"),
            rx_data: self.data_inbox.build(),
            rx_control: self.control_inbox.build(),
            rx_mgmt: self.mgmt_inbox.build(),
            data_burst_limit,
            management_burst_limit: self.params.management_burst_limit(),
            control_burst_limit: self.params.control_burst_limit(),
            shutdown_mode,
        }
    }
}

/// Trait for waking a blocked actor scheduler.
///
/// Implement this trait for platform-specific wake mechanisms (e.g., NSEvent on macOS).
/// When messages are sent, the wake handler is called to ensure the scheduler
/// processes them immediately, even if blocked on a platform event loop.
pub trait WakeHandler: Send + Sync {
    /// Wake the scheduler from a blocked state.
    ///
    /// Called automatically when Data/Management/Control messages are sent.
    /// Platform implementations might send events to wake up event loops,
    /// while the default implementation sends a Wake message through the control channel.
    fn wake(&self);
}

/// Fibonacci hash constant for jitter calculation.
const JITTER_HASH_CONSTANT: u64 = 0x9e3779b97f4a7c15;

/// Calculate exponential backoff with jitter.
///
/// Uses a simple exponential backoff strategy with added jitter to prevent
/// thundering herd problems when multiple actors wake simultaneously.
fn backoff_with_jitter(attempt: u32, params: &SchedulerParams) -> Result<Duration, SendError> {
    let base_micros = params.min_backoff.as_micros() as u64;
    let max_micros = params.max_backoff.as_micros() as u64;

    let multiplier = 2u64.saturating_pow(attempt);
    let backoff_micros = base_micros.saturating_mul(multiplier);
    if backoff_micros > max_micros {
        return Err(SendError::Timeout);
    }

    // Add jitter: random value between [min_pct%, (min_pct+range_pct)%] of backoff
    // Use wall clock time for actual randomness (prevents thundering herd)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));

    // Mix nanoseconds with attempt number for better distribution across threads
    let hash = (now.as_nanos() as u64 ^ (attempt as u64).wrapping_mul(0x517cc1b727220a95))
        .wrapping_mul(JITTER_HASH_CONSTANT);

    let jitter_pct = params.jitter_min_pct + (hash % params.jitter_range_pct);
    let jittered_micros = (backoff_micros * jitter_pct) / 100;

    Ok(Duration::from_micros(jittered_micros))
}

/// System messages - combines wake and shutdown into one channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum System {
    /// Wake the scheduler to process messages
    Wake,
    /// Shutdown the scheduler
    Shutdown,
}

/// A unified sender handle that routes messages to the scheduler with priority lanes.
///
/// Each handle owns dedicated SPSC channels (one per lane) to the target actor.
/// Not `Clone` — use [`ActorBuilder::add_producer`] to create additional handles.
/// This eliminates all send-side contention: each producer gets its own wait-free path.
pub struct ActorHandle<D, C, M> {
    // Doorbell channel (buffer: 1) - wake and shutdown signals (MPSC, shared)
    tx_doorbell: SyncSender<System>,
    // Each lane is a dedicated SPSC channel (one producer per handle)
    tx_data: SpscSender<D>,
    tx_control: SpscSender<C>,
    tx_mgmt: SpscSender<M>,
    // Optional custom wake handler for platform-specific wake mechanisms
    wake_handler: Option<Arc<dyn WakeHandler>>,
    // Tunable parameters for backoff/retry behavior
    params: SchedulerParams,
}

// Manual Debug implementation - wake_handler is opaque (trait object)
impl<D, C, M> std::fmt::Debug for ActorHandle<D, C, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorHandle")
            .field("has_wake_handler", &self.wake_handler.is_some())
            .finish_non_exhaustive()
    }
}

/// Send with retry and exponential backoff + jitter for fairness.
///
/// Backoff strategy:
/// 1. Spin (immediate retry) for first `params.spin_attempts`
/// 2. Yield (cooperative) for next `params.yield_attempts`
/// 3. Sleep (blocking) with exponential backoff for remaining attempts
///
/// Used for control and management lanes to prevent thundering herd when
/// multiple senders compete for buffer space.
fn send_with_backoff<T>(
    tx: &SpscSender<T>,
    mut msg: T,
    params: &SchedulerParams,
) -> Result<(), SendError> {
    let mut attempt = 0u32;
    loop {
        match tx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(spsc::TrySendError::Full(returned_msg)) => {
                // Restore message for retry
                msg = returned_msg;

                // Backoff strategy: spin → yield → sleep
                if attempt < params.spin_attempts {
                    // Phase 1: Spin (immediate retry, hot loop)
                    // No sleep/yield - just retry immediately
                } else if attempt < params.spin_attempts + params.yield_attempts {
                    // Phase 2: Yield (cooperative, let other threads run)
                    std::thread::yield_now();
                } else {
                    // Phase 3: Sleep (exponential backoff with jitter)
                    #[cfg(debug_assertions)]
                    if attempt.is_multiple_of(10) {
                        eprintln!(
                            "[ActorScheduler] Priority channel full, backing off (attempt {})",
                            attempt
                        );
                    }

                    let sleep_attempt = attempt - (params.spin_attempts + params.yield_attempts);
                    let backoff = backoff_with_jitter(sleep_attempt, params)?;
                    std::thread::sleep(backoff);
                }

                attempt = attempt.saturating_add(1);
            }
            Err(spsc::TrySendError::Disconnected(_)) => {
                return Err(SendError::Disconnected);
            }
        }
    }
}

impl<D, C, M> ActorHandle<D, C, M> {
    /// Sends a message to the appropriate priority lane and wakes the scheduler.
    ///
    /// Accepts any type that implements `IntoMessage` for this handle's message types.
    /// Use the `impl_control_message!`, `impl_data_message!`, or `impl_management_message!`
    /// macros to mark your message types.
    ///
    /// # Blocking Behavior
    /// - `Data`: Blocking send (backpressure when buffer full)
    /// - `Control`: Retry with exponential backoff + jitter for fairness
    /// - `Management`: Retry with exponential backoff + jitter for fairness
    ///
    /// Backoff on control/management prevents thundering herd when multiple
    /// senders compete for these lanes.
    ///
    /// # Errors
    /// Returns `Err` only if the receiver has been dropped.
    pub fn send<T: Into<Message<D, C, M>>>(&self, msg: T) -> Result<(), SendError> {
        let msg = msg.into();
        self.send_message(msg)
    }

    fn send_message(&self, msg: Message<D, C, M>) -> Result<(), SendError> {
        match msg {
            Message::Data(mut d) => {
                // Data lane: spin-yield until space available (backpressure)
                loop {
                    match self.tx_data.try_send(d) {
                        Ok(()) => break,
                        Err(spsc::TrySendError::Full(returned_d)) => {
                            d = returned_d;
                            std::thread::yield_now();
                        }
                        Err(spsc::TrySendError::Disconnected(_)) => {
                            return Err(SendError::Disconnected);
                        }
                    }
                }
                self.wake();
            }
            Message::Control(ctrl_msg) => {
                // Control lane: retry with backoff for fairness
                send_with_backoff(&self.tx_control, ctrl_msg, &self.params)?;
                self.wake();
            }
            Message::Management(m) => {
                // Management lane: retry with backoff for fairness
                send_with_backoff(&self.tx_mgmt, m, &self.params)?;
                self.wake();
            }
            Message::Shutdown => {
                // Shutdown: blocking send to guarantee delivery
                // Actor must be running before calling this (doorbell will be drained)
                self.tx_doorbell.send(System::Shutdown)?;

                // Also call custom wake handler if present
                if let Some(waker) = &self.wake_handler {
                    waker.wake();
                }
            }
        };
        Ok(())
    }

    /// Wake the scheduler to process messages.
    ///
    /// If a custom wake handler is configured, calls it first to wake the platform
    /// event loop (e.g., sending NSEvent on macOS). Then sends a doorbell signal to
    /// unblock ActorScheduler.run().
    ///
    /// Doorbell uses try_send (drops if full) - safe because one pending wake is sufficient.
    fn wake(&self) {
        if let Some(waker) = &self.wake_handler {
            waker.wake();
        }
        match self.tx_doorbell.try_send(System::Wake) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                // Doorbell is bounded(1) - if full, a wake is already pending
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                panic!("Doorbell receiver disconnected - scheduler dropped unexpectedly");
            }
        }
    }
}

/// The receiver side that implements the priority scheduling logic.
///
/// Internally uses [`ShardedInbox`] per lane: each registered producer has
/// a dedicated SPSC ring buffer, and the scheduler drains all shards with
/// round-robin fairness. The MPSC doorbell channel is kept for wake/shutdown signals.
pub struct ActorScheduler<D, C, M> {
    rx_doorbell: Receiver<System>, // Wake and shutdown signals (MPSC)
    rx_data: ShardedInbox<D>,
    rx_control: ShardedInbox<C>,
    rx_mgmt: ShardedInbox<M>,
    data_burst_limit: usize,
    management_burst_limit: usize,
    control_burst_limit: usize,
    shutdown_mode: ShutdownMode,
}

/// System status after processing messages
enum SchedulerLoopStatus {
    /// More work available, keep polling
    Working,
    /// Queues drained, can block
    Idle,
}

impl<D, C, M> ActorScheduler<D, C, M> {
    /// Drain control and management channels without limit, ignoring data.
    ///
    /// Used for `ShutdownMode::DrainControl` to process critical cleanup messages
    /// while dropping lower-priority data messages.
    fn drain_control_and_management<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        // Drain control completely
        while let DrainStatus::More = self
            .rx_control
            .drain(usize::MAX, |msg| actor.handle_control(msg))?
        {}

        // Drain management completely
        while let DrainStatus::More = self
            .rx_mgmt
            .drain(usize::MAX, |msg| actor.handle_management(msg))?
        {}

        Ok(())
    }

    /// Drain all channels (control, management, data) with timeout fallback.
    ///
    /// Used for `ShutdownMode::DrainAll` to process all pending messages before shutdown.
    /// If the timeout is exceeded, remaining messages are dropped.
    fn drain_all_with_timeout<A>(
        &mut self,
        actor: &mut A,
        timeout: std::time::Duration,
    ) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        use std::time::Instant;

        let deadline = Instant::now() + timeout;
        let batch_size = 10;

        loop {
            let control_status = self
                .rx_control
                .drain(batch_size, |msg| actor.handle_control(msg))?;
            if Instant::now() >= deadline {
                return Ok(());
            }

            let mgmt_status = self
                .rx_mgmt
                .drain(batch_size, |msg| actor.handle_management(msg))?;
            if Instant::now() >= deadline {
                return Ok(());
            }

            let data_status = self
                .rx_data
                .drain(batch_size, |msg| actor.handle_data(msg))?;
            if Instant::now() >= deadline {
                return Ok(());
            }

            // Done when all channels are empty or disconnected
            let all_done = !matches!(control_status, DrainStatus::More)
                && !matches!(mgmt_status, DrainStatus::More)
                && !matches!(data_status, DrainStatus::More);

            if all_done {
                return Ok(());
            }
        }
    }

    /// Process messages from all priority lanes, return status.
    ///
    /// Returns:
    /// - `Ok(Some(status))` - Processed messages, continue with given status
    /// - `Ok(None)` - All channels disconnected, normal shutdown
    /// - `Err(HandlerError)` - Handler failed
    #[inline]
    fn handle_wake<A>(&mut self, actor: &mut A) -> Result<Option<SchedulerLoopStatus>, HandlerError>
    where
        A: Actor<D, C, M>,
    {
        // Drain Control → Mgmt → Control → Data
        // Control budget is split evenly between the two control runs to prevent double priority
        let half_control = self.control_burst_limit / 2;

        let control1 = self
            .rx_control
            .drain(half_control, |msg| actor.handle_control(msg))?;

        let mgmt = self.rx_mgmt.drain(self.management_burst_limit, |msg| {
            actor.handle_management(msg)
        })?;

        let control2 = self
            .rx_control
            .drain(half_control, |msg| actor.handle_control(msg))?;

        let data = self
            .rx_data
            .drain(self.data_burst_limit, |msg| actor.handle_data(msg))?;

        // All disconnected = normal shutdown
        if matches!(
            (&control1, &mgmt, &control2, &data),
            (
                DrainStatus::Disconnected,
                DrainStatus::Disconnected,
                DrainStatus::Disconnected,
                DrainStatus::Disconnected
            )
        ) {
            return Ok(None);
        }

        // Any channel hit burst limit = more work available
        let more_work = matches!(control1, DrainStatus::More)
            || matches!(mgmt, DrainStatus::More)
            || matches!(control2, DrainStatus::More)
            || matches!(data, DrainStatus::More);

        let system_status = if more_work {
            SystemStatus::Busy
        } else {
            SystemStatus::Idle
        };

        let returned_hint = actor.park(system_status)?;

        let status = if more_work || returned_hint == ActorStatus::Busy {
            SchedulerLoopStatus::Working
        } else {
            SchedulerLoopStatus::Idle
        };

        Ok(Some(status))
    }

    #[cold]
    fn handle_shutdown<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        match self.shutdown_mode {
            ShutdownMode::Immediate => Ok(()),
            ShutdownMode::DrainControl => self.drain_control_and_management(actor),
            ShutdownMode::DrainAll { timeout } => self.drain_all_with_timeout(actor, timeout),
        }
    }

    /// Create a new scheduler with a single producer.
    ///
    /// Convenience method for the common case of one sender. Returns
    /// `(handle, scheduler)`. For multiple producers, use [`ActorBuilder`].
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold).
    ///
    /// # Panics
    /// Panics if `data_buffer_size` is 0.
    #[must_use]
    pub fn new(data_burst_limit: usize, data_buffer_size: usize) -> (ActorHandle<D, C, M>, Self) {
        Self::new_with_params(data_burst_limit, data_buffer_size, SchedulerParams::DEFAULT)
    }

    /// Create a new scheduler with explicit tuning parameters and a single producer.
    #[must_use]
    pub fn new_with_params(
        data_burst_limit: usize,
        data_buffer_size: usize,
        params: SchedulerParams,
    ) -> (ActorHandle<D, C, M>, Self) {
        let mut builder = ActorBuilder::new_with_params(data_buffer_size, None, params);
        let handle = builder.add_producer();
        let scheduler = builder.build_with_burst(data_burst_limit, ShutdownMode::default());
        (handle, scheduler)
    }

    /// Create a new scheduler with a custom wake handler and a single producer.
    #[must_use]
    pub fn new_with_wake_handler(
        data_burst_limit: usize,
        data_buffer_size: usize,
        wake_handler: Option<Arc<dyn WakeHandler>>,
    ) -> (ActorHandle<D, C, M>, Self) {
        let mut builder = ActorBuilder::new(data_buffer_size, wake_handler);
        let handle = builder.add_producer();
        let scheduler = builder.build_with_burst(data_burst_limit, ShutdownMode::default());
        (handle, scheduler)
    }

    /// Create a new scheduler with configurable shutdown behavior and a single producer.
    #[must_use]
    pub fn new_with_shutdown_mode(
        data_burst_limit: usize,
        data_buffer_size: usize,
        shutdown_mode: ShutdownMode,
    ) -> (ActorHandle<D, C, M>, Self) {
        let mut builder = ActorBuilder::new(data_buffer_size, None);
        let handle = builder.add_producer();
        let scheduler = builder.build_with_burst(data_burst_limit, shutdown_mode);
        (handle, scheduler)
    }

    /// The main scheduler loop.
    ///
    /// Blocks on the doorbell channel. Drains priority lanes in order:
    /// Shutdown > Control > Management > Data.
    ///
    /// Returns a [`PodPhase`] describing why the scheduler exited, so a
    /// supervisor can decide whether to restart the pod:
    ///
    /// | Exit reason | Returned phase |
    /// |-------------|----------------|
    /// | `Message::Shutdown` received | `PodPhase::Completed` |
    /// | All sender handles dropped | `PodPhase::Completed` |
    /// | `HandlerError::Recoverable` | `PodPhase::Failed(msg)` |
    /// | `HandlerError::Fatal` | panics — never returns |
    ///
    /// The return value is intentionally not `#[must_use]` so existing call
    /// sites that don't supervise actors don't need to change. Supervisors
    /// should inspect it via [`RestartPolicy::should_restart`].
    pub fn run<A>(&mut self, actor: &mut A) -> PodPhase
    where
        A: Actor<D, C, M>,
    {
        match self.run_inner(actor) {
            Ok(()) => PodPhase::Completed,
            Err(HandlerError::Recoverable(msg)) => PodPhase::Failed(msg),
            Err(HandlerError::Fatal(msg)) => panic!("Actor fatal error: {msg}"),
        }
    }

    /// Single non-blocking drain cycle for cooperative scheduling.
    ///
    /// Intended for actors running on a shared Kubelet thread rather than a
    /// dedicated OS thread. The Kubelet calls `poll_once()` on each cooperative
    /// pod in round-robin during its controller loop.
    ///
    /// Unlike [`run`], `poll_once()` never blocks:
    /// - If the doorbell is empty, it still attempts one drain pass (the actor
    ///   may have work from a previous `Working` state).
    /// - Returns `Some(phase)` when the pod should stop; `None` to keep polling.
    ///
    /// # Caller responsibility
    ///
    /// The Kubelet must continue calling `poll_once()` after a `Disconnected`
    /// doorbell until `Some` is returned — buffered SPSC messages need draining.
    pub fn poll_once<A>(&mut self, actor: &mut A) -> Option<PodPhase>
    where
        A: Actor<D, C, M>,
    {
        use std::sync::mpsc::TryRecvError;

        let signal = self.rx_doorbell.try_recv();

        match signal {
            Ok(System::Shutdown) => {
                let phase = match self.handle_shutdown(actor) {
                    Ok(()) => PodPhase::Completed,
                    Err(HandlerError::Recoverable(msg)) => PodPhase::Failed(msg),
                    Err(HandlerError::Fatal(msg)) => panic!("Actor fatal error: {msg}"),
                };
                Some(phase)
            }

            Ok(System::Wake) | Err(TryRecvError::Empty) => {
                match self.handle_wake(actor) {
                    Ok(Some(_)) => None,                   // still running
                    Ok(None) => Some(PodPhase::Completed), // all disconnected
                    Err(HandlerError::Recoverable(msg)) => Some(PodPhase::Failed(msg)),
                    Err(HandlerError::Fatal(msg)) => panic!("Actor fatal error: {msg}"),
                }
            }

            Err(TryRecvError::Disconnected) => {
                // All handles dropped — drain one batch, report done when empty
                match self.handle_wake(actor) {
                    Ok(Some(_)) => None, // more buffered work; caller polls again
                    Ok(None) => Some(PodPhase::Completed),
                    Err(HandlerError::Recoverable(msg)) => Some(PodPhase::Failed(msg)),
                    Err(HandlerError::Fatal(msg)) => panic!("Actor fatal error: {msg}"),
                }
            }
        }
    }

    fn run_inner<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        use std::sync::mpsc::TryRecvError;

        let mut working = false;

        loop {
            let signal = if working {
                self.rx_doorbell.try_recv()
            } else {
                self.rx_doorbell
                    .recv()
                    .map_err(|_| TryRecvError::Disconnected)
            };

            match signal {
                Ok(System::Shutdown) => {
                    self.handle_shutdown(actor)?;
                    return Ok(());
                }
                Ok(System::Wake) | Err(TryRecvError::Empty) => {
                    match self.handle_wake(actor)? {
                        Some(status) => {
                            working = matches!(status, SchedulerLoopStatus::Working);
                        }
                        None => return Ok(()), // All channels disconnected
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    // Doorbell disconnected — all handles dropped.
                    // SPSC shards may still have buffered messages.
                    // Drain until all shards report Disconnected.
                    loop {
                        match self.handle_wake(actor)? {
                            Some(_) => {} // keep draining
                            None => return Ok(()),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    struct TestHandler {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl SchedulerHandler<String, String, String> for TestHandler {
        fn handle_data(&mut self, msg: String) -> HandlerResult {
            self.log.lock().unwrap().push(format!("Data: {}", msg));
            Ok(())
        }
        fn handle_control(&mut self, msg: String) -> HandlerResult {
            self.log.lock().unwrap().push(format!("Ctrl: {}", msg));
            Ok(())
        }
        fn handle_management(&mut self, msg: String) -> HandlerResult {
            self.log.lock().unwrap().push(format!("Mgmt: {}", msg));
            Ok(())
        }

        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    #[test]
    fn verify_data_lane_backpressure_contract() {
        let (tx, mut rx) = ActorScheduler::new(2, 1);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        // Send from this thread — the 3rd message may spin-yield on backpressure
        let send_thread = thread::spawn(move || {
            tx.send(Message::Data("1".to_string())).unwrap();
            tx.send(Message::Data("2".to_string())).unwrap();
            tx.send(Message::Data("3".to_string())).unwrap();
        });

        send_thread.join().unwrap();
        thread::sleep(Duration::from_millis(100));
        let messages = log.lock().unwrap();
        assert_eq!(messages.len(), 3, "All messages should be processed");
    }

    #[test]
    fn verify_actor_trait_contract() {
        struct CountingHandler {
            data_count: usize,
            ctrl_count: usize,
            mgmt_count: usize,
        }

        impl SchedulerHandler<i32, String, bool> for CountingHandler {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.data_count += 1;
                Ok(())
            }
            fn handle_control(&mut self, _: String) -> HandlerResult {
                self.ctrl_count += 1;
                Ok(())
            }
            fn handle_management(&mut self, _: bool) -> HandlerResult {
                self.mgmt_count += 1;
                Ok(())
            }
            fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        let (tx, mut rx) = ActorScheduler::new(10, 100);

        let handle = thread::spawn(move || {
            let mut handler = CountingHandler {
                data_count: 0,
                ctrl_count: 0,
                mgmt_count: 0,
            };
            rx.run(&mut handler);
            handler
        });

        tx.send(Message::Data(1)).unwrap();
        tx.send(Message::Data(2)).unwrap();
        tx.send(Message::Control("test".to_string())).unwrap();
        tx.send(Message::Management(true)).unwrap();

        thread::sleep(Duration::from_millis(50));
        drop(tx);

        let actor = handle.join().unwrap();
        assert_eq!(actor.data_count, 2);
        assert_eq!(actor.ctrl_count, 1);
        assert_eq!(actor.mgmt_count, 1);
    }

    #[test]
    fn shutdown_message_exits_scheduler_immediately() {
        use std::sync::atomic::{AtomicBool, Ordering};

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
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Idle)
                }
            }
            rx.run(&mut NoopActor);
            exited_clone.store(true, Ordering::SeqCst);
        });

        // Verify running
        thread::sleep(Duration::from_millis(20));
        assert!(!exited.load(Ordering::SeqCst), "should still be running");

        // Send shutdown
        tx.send(Message::Shutdown).unwrap();

        // Should exit quickly
        handle.join().unwrap();
        assert!(exited.load(Ordering::SeqCst), "should have exited");
    }
}

#[cfg(test)]
mod poll_once_tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    struct CountActor {
        data: usize,
        ctrl: usize,
    }

    impl Actor<i32, i32, i32> for CountActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data += 1;
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            self.ctrl += 1;
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    #[test]
    fn poll_once_returns_none_when_still_running() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        // No messages yet — doorbell empty, poll_once drains nothing, returns None
        let result = rx.poll_once(&mut actor);
        assert_eq!(result, None);
        drop(tx);
    }

    #[test]
    fn poll_once_drains_messages_and_returns_none() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(100, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        tx.send(Message::Control(1)).unwrap();
        tx.send(Message::Data(2)).unwrap();

        // Give messages time to arrive
        thread::sleep(Duration::from_millis(5));

        let result = rx.poll_once(&mut actor);
        assert_eq!(result, None); // still connected
        assert!(actor.ctrl >= 1, "control message should have been drained");

        drop(tx);
    }

    #[test]
    fn poll_once_returns_completed_after_all_handles_dropped() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        drop(tx);

        // Keep polling until Completed
        let phase = loop {
            if let Some(p) = rx.poll_once(&mut actor) {
                break p;
            }
        };
        assert_eq!(phase, PodPhase::Completed);
    }

    #[test]
    fn poll_once_returns_failed_on_recoverable_error() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

        struct FailOnData;
        impl Actor<i32, i32, i32> for FailOnData {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                Err(HandlerError::recoverable("injected failure"))
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        tx.send(Message::Data(1)).unwrap();
        thread::sleep(Duration::from_millis(5));

        let mut actor = FailOnData;
        let phase = loop {
            if let Some(p) = rx.poll_once(&mut actor) {
                break p;
            }
        };

        assert!(phase.is_failed());
        drop(tx);
    }

    #[test]
    fn poll_once_returns_completed_on_shutdown_message() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        tx.send(Message::Shutdown).unwrap();
        thread::sleep(Duration::from_millis(5));

        let phase = loop {
            if let Some(p) = rx.poll_once(&mut actor) {
                break p;
            }
        };
        assert_eq!(phase, PodPhase::Completed);
    }
}

// Tests targeting missed mutations in backoff_with_jitter and send_with_backoff.
// These functions are private so tests live in the same module.
#[cfg(test)]
mod backoff_unit_tests {
    use super::*;
    use std::time::Duration;

    fn params_with_bounds(min_us: u64, max_us: u64) -> SchedulerParams {
        SchedulerParams {
            min_backoff: Duration::from_micros(min_us),
            max_backoff: Duration::from_micros(max_us),
            jitter_min_pct: 50,
            jitter_range_pct: 49,
            ..SchedulerParams::DEFAULT
        }
    }

    // Kills: replace > with == (would only timeout at exact max, not above)
    // Kills: replace > with < (would timeout when still under max)
    #[test]
    fn backoff_returns_timeout_when_over_max() {
        // min=100us, max=1000us. attempt=4: backoff = 100 * 2^4 = 1600 > 1000 → Timeout
        let params = params_with_bounds(100, 1000);
        assert!(
            matches!(backoff_with_jitter(4, &params), Err(SendError::Timeout)),
            "Should return Timeout when backoff_micros > max_micros"
        );
    }

    // Kills: replace > with >= (>= fires at equality; > should NOT fire at equality)
    #[test]
    fn backoff_returns_ok_at_exact_max() {
        // min=max=100us. attempt=0: backoff = 100 * 1 = 100, max = 100.
        // `100 > 100` is false → should return Ok, not Timeout.
        let params = params_with_bounds(100, 100);
        assert!(
            backoff_with_jitter(0, &params).is_ok(),
            "backoff == max should NOT trigger timeout (> not >=)"
        );
    }

    #[test]
    fn backoff_returns_ok_when_under_max() {
        // min=100us, max=10000us. attempt=0: backoff=100 < 10000 → Ok
        let params = params_with_bounds(100, 10_000);
        assert!(
            backoff_with_jitter(0, &params).is_ok(),
            "Should return Ok when backoff_micros < max_micros"
        );
    }

    // Kills arithmetic mutations on lines 645-646:
    //   replace + with - (jitter_min + hash%range)
    //   replace % with / or + (hash % jitter_range_pct)
    //   replace * with + or / (backoff * jitter_pct)
    //   replace / with % or * (result / 100)
    //
    // Strategy: verify output duration is in [backoff*jitter_min/100, backoff*(jitter_min+range-1)/100]
    #[test]
    fn backoff_duration_within_jitter_bounds() {
        // backoff = 10000us, jitter 50-98% → duration in [5000, 9800] us
        let params = params_with_bounds(10_000, 1_000_000);
        let backoff_us = 10_000u64;
        let min_expected = Duration::from_micros(backoff_us * 50 / 100);
        let max_expected = Duration::from_micros(backoff_us * 98 / 100);

        // Run multiple times to exercise varying hash values
        for _ in 0..20 {
            let dur = backoff_with_jitter(0, &params).unwrap();
            assert!(
                dur >= min_expected,
                "Duration {}us below minimum {}us",
                dur.as_micros(),
                min_expected.as_micros()
            );
            assert!(
                dur <= max_expected,
                "Duration {}us above maximum {}us",
                dur.as_micros(),
                max_expected.as_micros()
            );
        }
    }

    // Kills: replace backoff_with_jitter with Ok(Default::default())
    #[test]
    fn backoff_duration_nonzero_for_nonzero_backoff() {
        let params = params_with_bounds(1000, 1_000_000);
        let dur = backoff_with_jitter(0, &params).unwrap();
        assert!(
            dur.as_micros() >= 500,
            "Duration should be at least 50% of 1000us"
        );
    }

    // Kills: send_with_backoff arithmetic/comparison mutations via observable behavior
    #[test]
    fn send_with_backoff_succeeds_on_empty_channel() {
        let (tx, _rx) = spsc::spsc_channel::<u32>(4);
        let params = SchedulerParams::DEFAULT;
        assert!(send_with_backoff(&tx, 42u32, &params).is_ok());
    }

    #[test]
    fn send_with_backoff_returns_disconnected_when_receiver_dropped() {
        let (tx, rx) = spsc::spsc_channel::<u32>(4);
        drop(rx);
        let params = SchedulerParams::DEFAULT;
        assert!(
            matches!(
                send_with_backoff(&tx, 42, &params),
                Err(SendError::Disconnected)
            ),
            "Should return Disconnected when receiver has dropped"
        );
    }

    // Kills: comparison mutations on attempt thresholds (< vs == vs <=)
    // With correct code: spin phase attempts 0..spin_attempts, then times out via backoff.
    // With wrong comparisons: phase transitions differ, but timeout still fires since
    // we use instant-timeout params (max_backoff barely above min_backoff).
    #[test]
    fn send_with_backoff_returns_timeout_on_permanently_full_channel() {
        // Channel of capacity 2, fill it. Use minimal backoff so timeout fires on attempt 1.
        let (tx, _rx) = spsc::spsc_channel::<u32>(2);
        tx.try_send(1u32).unwrap();
        tx.try_send(2u32).unwrap();

        let params = SchedulerParams {
            spin_attempts: 0,
            yield_attempts: 0,
            min_backoff: Duration::from_micros(1),
            max_backoff: Duration::from_micros(1),
            jitter_min_pct: 50,
            jitter_range_pct: 49,
            ..SchedulerParams::DEFAULT
        };
        // attempt=0 → sleep phase, sleep_attempt=0, backoff=1*1=1, max=1, 1>1 false → sleep
        // attempt=1 → sleep phase, sleep_attempt=1, backoff=1*2=2, max=1, 2>1 → Timeout
        assert!(
            matches!(send_with_backoff(&tx, 3, &params), Err(SendError::Timeout)),
            "Should return Timeout when channel permanently full"
        );
    }
}

// Tests targeting missed mutations in drain_all_with_timeout.
#[cfg(test)]
mod drain_all_targeted_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    struct FastCountActor {
        data: Arc<AtomicUsize>,
    }

    impl Actor<i32, (), ()> for FastCountActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data.fetch_add(1, Ordering::Relaxed);
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

    // Kills: replace >= with < in drain_all_with_timeout (line 910)
    // With <: `if Instant::now() < deadline` fires immediately → returns after first batch of 10
    // Kills: replace && with || in all_done (lines 915-917)
    // With ||: after first batch, control is Disconnected (non-More) → all_done=true → exits early
    //
    // Strategy: queue 30 data-only messages, queue Shutdown BEFORE scheduler starts.
    // Correct code: processes all 30. Mutated code: processes only ~10 (first batch).
    #[test]
    fn drain_all_processes_multiple_batches_of_data() {
        let (tx, mut rx) = ActorScheduler::new_with_shutdown_mode(
            100,
            1000,
            ShutdownMode::DrainAll {
                timeout: Duration::from_secs(10), // Far future — should not fire
            },
        );

        let data_count = Arc::new(AtomicUsize::new(0));

        // Queue Shutdown BEFORE starting scheduler thread, so scheduler sees it immediately.
        tx.send(Message::Shutdown).unwrap();

        // Queue 30 data messages into the SPSC buffer (scheduler not running yet).
        for i in 0..30i32 {
            tx.send(Message::Data(i)).unwrap();
        }

        // Now start the scheduler. It sees Shutdown first → calls drain_all_with_timeout.
        // drain_all must process all 30 data messages before returning.
        let data_clone = data_count.clone();
        let handle = std::thread::spawn(move || {
            let mut actor = FastCountActor { data: data_clone };
            rx.run(&mut actor);
        });

        handle.join().unwrap();
        assert_eq!(
            data_count.load(Ordering::Relaxed),
            30,
            "drain_all_with_timeout must process all 30 queued data messages"
        );
    }

    // Kills: replace >= with < in drain_all_with_timeout when control messages present.
    // Also exercises the path where control/mgmt have messages alongside data.
    #[test]
    fn drain_all_processes_all_lanes_before_timeout() {
        let (tx, mut rx) = ActorScheduler::new_with_shutdown_mode(
            100,
            1000,
            ShutdownMode::DrainAll {
                timeout: Duration::from_secs(10),
            },
        );

        let data_count = Arc::new(AtomicUsize::new(0));

        // Send shutdown first, then queue messages
        tx.send(Message::Shutdown).unwrap();
        for i in 0..20i32 {
            tx.send(Message::Data(i)).unwrap();
        }
        for _ in 0..15 {
            tx.send(Message::Control(())).unwrap();
        }
        for _ in 0..15 {
            tx.send(Message::Management(())).unwrap();
        }

        let data_clone = data_count.clone();
        let ctrl_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        struct FullCountActor {
            data: Arc<AtomicUsize>,
            ctrl: Arc<AtomicUsize>,
            mgmt: Arc<AtomicUsize>,
        }
        impl Actor<i32, (), ()> for FullCountActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.data.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                self.ctrl.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                self.mgmt.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        let ctrl_clone = ctrl_count.clone();
        let mgmt_clone = mgmt_count.clone();
        let handle = std::thread::spawn(move || {
            let mut actor = FullCountActor {
                data: data_clone,
                ctrl: ctrl_clone,
                mgmt: mgmt_clone,
            };
            rx.run(&mut actor);
        });

        handle.join().unwrap();
        assert_eq!(data_count.load(Ordering::Relaxed), 20);
        assert_eq!(ctrl_count.load(Ordering::Relaxed), 15);
        assert_eq!(mgmt_count.load(Ordering::Relaxed), 15);
    }

    // Kills: replace drain_control_and_management body with Ok(()) (line 861)
    // With body replaced: control/mgmt messages queued at shutdown time are dropped.
    #[test]
    fn drain_control_processes_queued_control_and_mgmt_on_shutdown() {
        let (tx, mut rx) =
            ActorScheduler::new_with_shutdown_mode(100, 1000, ShutdownMode::DrainControl);

        // Queue Shutdown BEFORE scheduler starts, then queue control/mgmt messages.
        // Scheduler will see Shutdown immediately → calls drain_control_and_management.
        tx.send(Message::Shutdown).unwrap();
        for _ in 0..25 {
            tx.send(Message::Control(())).unwrap();
        }
        for _ in 0..25 {
            tx.send(Message::Management(())).unwrap();
        }

        let ctrl_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        struct CmActor {
            ctrl: Arc<AtomicUsize>,
            mgmt: Arc<AtomicUsize>,
        }
        impl Actor<(), (), ()> for CmActor {
            fn handle_data(&mut self, _: ()) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: ()) -> HandlerResult {
                self.ctrl.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn handle_management(&mut self, _: ()) -> HandlerResult {
                self.mgmt.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        let ctrl_clone = ctrl_count.clone();
        let mgmt_clone = mgmt_count.clone();
        let handle = std::thread::spawn(move || {
            let mut actor = CmActor {
                ctrl: ctrl_clone,
                mgmt: mgmt_clone,
            };
            rx.run(&mut actor);
        });

        handle.join().unwrap();
        assert_eq!(
            ctrl_count.load(Ordering::Relaxed),
            25,
            "DrainControl must drain all 25 control messages"
        );
        assert_eq!(
            mgmt_count.load(Ordering::Relaxed),
            25,
            "DrainControl must drain all 25 management messages"
        );
    }
}

#[cfg(test)]
mod troupe_tests {
    #![allow(dead_code)] // Test module - structs demonstrate pattern but may not all be constructed

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // === Message types ===

    pub struct EngineData;
    #[derive(Default)]
    pub enum EngineControl {
        Tick,
        #[default]
        Shutdown,
    }
    pub struct EngineManagement;

    pub struct DisplayData;
    #[derive(Default)]
    pub enum DisplayControl {
        Render,
        #[default]
        Shutdown,
    }
    pub struct DisplayManagement;

    // === Actors ===

    pub struct EngineActor<'a> {
        dir: &'a Directory,
        tick_count: &'a AtomicUsize,
    }

    impl Actor<EngineData, EngineControl, EngineManagement> for EngineActor<'_> {
        fn handle_data(&mut self, _msg: EngineData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, msg: EngineControl) -> HandlerResult {
            match msg {
                EngineControl::Tick => {
                    self.tick_count.fetch_add(1, Ordering::SeqCst);
                    self.dir
                        .display
                        .send(Message::Control(DisplayControl::Render))
                        .expect("Failed to send render command to display actor");
                }
                EngineControl::Shutdown => {}
            }
            Ok(())
        }
        fn handle_management(&mut self, _msg: EngineManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl ActorTypes for EngineActor<'_> {
        type Data = EngineData;
        type Control = EngineControl;
        type Management = EngineManagement;
    }

    impl<'a> TroupeActor<&'a Directory> for EngineActor<'a> {
        fn new(_dir: &'a Directory) -> Self {
            panic!("use new_with_counter instead")
        }
    }

    pub struct DisplayActor<'a> {
        dir: &'a Directory,
        render_count: &'a AtomicUsize,
        shutdown_after: usize,
    }

    impl Actor<DisplayData, DisplayControl, DisplayManagement> for DisplayActor<'_> {
        fn handle_data(&mut self, _msg: DisplayData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, msg: DisplayControl) -> HandlerResult {
            match msg {
                DisplayControl::Render => {
                    let count = self.render_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if count >= self.shutdown_after {
                        self.dir
                            .engine
                            .send(Message::Control(EngineControl::Shutdown))
                            .expect("Failed to send shutdown to engine");
                    }
                }
                DisplayControl::Shutdown => {}
            }
            Ok(())
        }
        fn handle_management(&mut self, _msg: DisplayManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl ActorTypes for DisplayActor<'_> {
        type Data = DisplayData;
        type Control = DisplayControl;
        type Management = DisplayManagement;
    }

    impl<'a> TroupeActor<&'a Directory> for DisplayActor<'a> {
        fn new(_dir: &'a Directory) -> Self {
            panic!("use new_with_counter instead")
        }
    }

    // === Per-actor Directory (what troupe! generates with SPSC) ===
    // Each actor gets its OWN Directory with dedicated SPSC handles.

    pub struct Directory {
        pub engine: ActorHandle<EngineData, EngineControl, EngineManagement>,
        pub display: ActorHandle<DisplayData, DisplayControl, DisplayManagement>,
    }

    /// Test the SPSC-based directory pattern: each actor gets its own Directory
    /// with dedicated SPSC handles to every other actor.
    #[test]
    fn test_troupe_directory_pattern() {
        // Create builders for each actor
        let mut engine_builder =
            ActorBuilder::<EngineData, EngineControl, EngineManagement>::new(1024, None);
        let mut display_builder =
            ActorBuilder::<DisplayData, DisplayControl, DisplayManagement>::new(1024, None);

        // Each "producer" (actor + external caller) gets dedicated SPSC handles
        // Directory for the test caller:
        let test_dir = Directory {
            engine: engine_builder.add_producer(),
            display: display_builder.add_producer(),
        };

        // An additional producer handle (e.g. for exposed handles)
        let extra_engine_handle = engine_builder.add_producer();

        // Build schedulers (seals builders — no more producers after this)
        let _engine_s = engine_builder.build();
        let _display_s = display_builder.build();

        // Verify cross-actor messaging works via directory
        test_dir
            .display
            .send(Message::Control(DisplayControl::Render))
            .unwrap();
        test_dir
            .engine
            .send(Message::Control(EngineControl::Tick))
            .unwrap();

        // Multiple handles are independent (each is a separate SPSC channel)
        extra_engine_handle
            .send(Message::Control(EngineControl::Tick))
            .unwrap();
    }

    /// Adversarial test: Malicious control sender trying to starve data lane
    /// Uses CONTINUOUS flooding to ensure burst limiting works during active attack.
    /// With SPSC, each producer has its own channel — no send-side contention.
    #[test]
    fn adversarial_control_flood_vs_data() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, (), ()>::new(100, None);
        let tx_flood = builder.add_producer();
        let tx_data = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

        let receiver_handle = thread::spawn(move || {
            struct TestActor {
                control_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, (), ()> for TestActor {
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Malicious control sender: CONTINUOUS flood via dedicated SPSC
        let stop_flag = stop_flooding.clone();
        let control_sender = thread::spawn(move || {
            let mut sent = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                if tx_flood.send(Message::Control(())).is_ok() {
                    sent += 1;
                }
            }
            sent
        });

        thread::sleep(Duration::from_millis(20));

        // Well-behaved data sender via its own dedicated SPSC
        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                let _ = tx_data.send(Message::Data(i));
            }
        });

        data_sender.join().unwrap();
        thread::sleep(Duration::from_millis(50));

        stop_flooding.store(true, Ordering::Relaxed);
        let control_sent = control_sender.join().unwrap();

        // All handles dropped → scheduler exits
        thread::sleep(Duration::from_millis(50));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Control flood vs data - Control sent: {}, processed: {}, Data processed: {}/100",
            control_sent, control_count, data_count
        );

        assert!(
            data_count > 0,
            "Data lane was completely starved during continuous control flood"
        );
        assert!(
            data_count > 50,
            "Burst limiting too weak - only {}/100 data processed during flood",
            data_count
        );
    }

    /// Adversarial test: Multiple bad actors teaming up to flood control.
    /// With SPSC, each flooder has its own channel — the scheduler still needs
    /// burst limiting to prevent consumer-side starvation.
    #[test]
    fn adversarial_multiple_control_flooders() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, (), ()>::new(100, None);
        // 5 flood producers + 1 data producer
        let flood_handles: Vec<_> = (0..5).map(|_| builder.add_producer()).collect();
        let tx_data = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

        let receiver_handle = thread::spawn(move || {
            struct TestActor {
                control_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, (), ()> for TestActor {
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Each flooder has its own SPSC channel
        let mut control_threads = vec![];
        for tx_flood in flood_handles {
            let stop_flag = stop_flooding.clone();
            let handle = thread::spawn(move || {
                let mut sent = 0;
                while !stop_flag.load(Ordering::Relaxed) {
                    if tx_flood.send(Message::Control(())).is_ok() {
                        sent += 1;
                    }
                }
                sent
            });
            control_threads.push(handle);
        }

        thread::sleep(Duration::from_millis(20));

        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                let _ = tx_data.send(Message::Data(i));
            }
        });

        data_sender.join().unwrap();
        thread::sleep(Duration::from_millis(50));

        stop_flooding.store(true, Ordering::Relaxed);
        let mut total_control_sent = 0;
        for handle in control_threads {
            total_control_sent += handle.join().unwrap();
        }

        thread::sleep(Duration::from_millis(50));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Multiple attackers - Control sent: {}, processed: {}, Data: {}/100",
            total_control_sent, control_count, data_count
        );

        assert!(
            data_count > 0,
            "Data lane completely starved by coordinated control attack"
        );
        assert!(
            data_count > 50,
            "Burst limiting too weak against coordinated attack - only {}/100 data processed",
            data_count
        );
    }

    /// Adversarial test: Continuous control flood with concurrent data
    #[test]
    fn adversarial_continuous_control_flood() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, (), ()>::new(100, None);
        let tx_flood = builder.add_producer();
        let tx_data = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

        let receiver_handle = thread::spawn(move || {
            struct TestActor {
                control_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, (), ()> for TestActor {
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        let stop_flag = stop_flooding.clone();
        let control_flooder = thread::spawn(move || {
            let mut sent = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                if tx_flood.send(Message::Control(())).is_ok() {
                    sent += 1;
                }
            }
            sent
        });

        thread::sleep(Duration::from_millis(50));

        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                let _ = tx_data.send(Message::Data(i));
            }
        });

        data_sender.join().unwrap();
        thread::sleep(Duration::from_millis(100));

        stop_flooding.store(true, Ordering::Relaxed);
        let control_sent = control_flooder.join().unwrap();

        thread::sleep(Duration::from_millis(50));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Continuous flood - Control sent: {}, processed: {}, Data processed: {}/100",
            control_sent, control_count, data_count
        );

        assert!(
            data_count > 0,
            "Burst limiting FAILED - data was starved during continuous control flood"
        );
        assert!(
            data_count > 50,
            "Burst limiting is too weak - only {}/100 data messages processed",
            data_count
        );
    }

    /// Adversarial test: Slow receiver with multiple aggressive senders.
    /// Each sender has its own SPSC channels — backoff is per-producer.
    #[test]
    fn adversarial_slow_receiver_resilience() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, i32, i32>::new(10, None);
        let senders: Vec<_> = (0..3).map(|_| builder.add_producer()).collect();
        let mut rx = builder.build_with_burst(10, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let mgmt_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));

        let cp = control_processed.clone();
        let mp = mgmt_processed.clone();
        let dp = data_processed.clone();

        let receiver_handle = thread::spawn(move || {
            struct SlowActor {
                control_count: Arc<AtomicUsize>,
                mgmt_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, i32, i32> for SlowActor {
                fn handle_control(&mut self, _: i32) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(2));
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(2));
                    Ok(())
                }
                fn handle_management(&mut self, _: i32) -> HandlerResult {
                    self.mgmt_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(2));
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = SlowActor {
                control_count: cp,
                mgmt_count: mp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Each sender has its own SPSC channels
        let mut sender_handles = vec![];
        for (sender_id, tx) in senders.into_iter().enumerate() {
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let msg_val = (sender_id * 1000 + i) as i32;
                    let _ = tx.send(Message::Control(msg_val));
                    let _ = tx.send(Message::Management(msg_val));
                }
            });
            sender_handles.push(handle);
        }

        for handle in sender_handles {
            handle.join().unwrap();
        }

        thread::sleep(Duration::from_millis(1000));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let mgmt_count = mgmt_processed.load(Ordering::Relaxed);

        println!(
            "Slow receiver resilience - Control: {}, Mgmt: {}, Data: {}",
            control_count,
            mgmt_count,
            data_processed.load(Ordering::Relaxed)
        );

        assert_eq!(
            control_count, 300,
            "Backoff should allow all control messages through"
        );
        assert_eq!(
            mgmt_count, 300,
            "Backoff should allow all management messages through"
        );
    }
}

/// Test module for troupe nesting pattern (SPSC-based)
#[cfg(test)]
mod troupe_nesting_tests {
    #![allow(dead_code)]

    use super::*;

    // === Simple actors for nesting test ===

    pub struct WorkerData(pub String);
    #[derive(Default)]
    pub enum WorkerControl {
        Process,
        #[default]
        Shutdown,
    }
    pub struct WorkerManagement;

    /// Worker actor that just receives work items
    pub struct WorkerActor<'a> {
        _dir: &'a WorkerDirectory,
    }

    impl Actor<WorkerData, WorkerControl, WorkerManagement> for WorkerActor<'_> {
        fn handle_data(&mut self, _msg: WorkerData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _msg: WorkerControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: WorkerManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl ActorTypes for WorkerActor<'_> {
        type Data = WorkerData;
        type Control = WorkerControl;
        type Management = WorkerManagement;
    }

    impl<'a> TroupeActor<&'a WorkerDirectory> for WorkerActor<'a> {
        fn new(_dir: &'a WorkerDirectory) -> Self {
            panic!("test only")
        }
    }

    // Manual directory for worker troupe (per-actor owned)
    pub struct WorkerDirectory {
        pub worker: ActorHandle<WorkerData, WorkerControl, WorkerManagement>,
    }

    // Manual ExposedHandles for worker troupe
    pub struct WorkerExposedHandles {
        pub worker: ActorHandle<WorkerData, WorkerControl, WorkerManagement>,
    }

    // Manual Troupe struct for worker - stores builder (not scheduler) until play()
    pub struct WorkerTroupe {
        // Builder stays alive until play() so exposed() can add producers
        worker_builder: ActorBuilder<WorkerData, WorkerControl, WorkerManagement>,
        // Pre-created directory for the worker actor itself
        pub worker_dir: WorkerDirectory,
    }

    impl WorkerTroupe {
        pub fn new() -> Self {
            let mut builder =
                ActorBuilder::<WorkerData, WorkerControl, WorkerManagement>::new(1024, None);

            // Worker's own handle to itself (self-loop)
            let worker_dir = WorkerDirectory {
                worker: builder.add_producer(),
            };

            Self {
                worker_builder: builder,
                worker_dir,
            }
        }

        /// Create exposed handles by adding new producers to the builder.
        /// Must be called before play() since play() consumes the builder.
        pub fn exposed(&mut self) -> WorkerExposedHandles {
            WorkerExposedHandles {
                worker: self.worker_builder.add_producer(),
            }
        }
    }

    /// Test the two-phase Troupe pattern: new() → exposed() → play()
    #[test]
    fn test_troupe_two_phase_pattern() {
        // Phase 1: Create child troupe (no threads yet)
        let mut child = WorkerTroupe::new();

        // Phase 2: Parent grabs exposed handles (each call creates new SPSC channels)
        let exposed = child.exposed();

        // Parent can now send to child even before child.play()
        exposed
            .worker
            .send(Message::Control(WorkerControl::Process))
            .unwrap();
        exposed
            .worker
            .send(Message::Data(WorkerData("hello".to_string())))
            .unwrap();

        // Multiple exposed() calls create independent handles
        let exposed2 = child.exposed();
        exposed2
            .worker
            .send(Message::Control(WorkerControl::Process))
            .unwrap();

        // Note: We don't call play() here since that would block.
        // The test verifies the two-phase construction pattern works.
    }

    /// Test that ExposedHandles can outlive the Troupe struct
    #[test]
    fn test_exposed_handles_outlive_troupe_struct() {
        let exposed = {
            let mut child = WorkerTroupe::new();
            child.exposed() // ExposedHandles escapes
        };
        // Troupe struct dropped, but handles still valid (SPSC channels still open
        // until both sides drop). Builder was not consumed by build(), so receiver
        // side is also dropped — handles become disconnected.

        // Just verify the type works
        let _: WorkerExposedHandles = exposed;
    }
}

#[cfg(test)]
mod shutdown_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    struct CountingActor {
        data_count: Arc<AtomicUsize>,
        control_count: Arc<AtomicUsize>,
        mgmt_count: Arc<AtomicUsize>,
    }

    impl Actor<i32, (), ()> for CountingActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn handle_control(&mut self, _: ()) -> HandlerResult {
            self.control_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn handle_management(&mut self, _: ()) -> HandlerResult {
            self.mgmt_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            match status {
                SystemStatus::Idle => Ok(ActorStatus::Idle),
                SystemStatus::Busy => Ok(ActorStatus::Busy),
            }
        }
    }

    #[test]
    fn test_shutdown_immediate_exits_quickly_under_flood() {
        let (tx, mut rx) =
            ActorScheduler::new_with_shutdown_mode(100, 100, ShutdownMode::Immediate);

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        let actor_handle = thread::spawn(move || {
            let mut actor = CountingActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Flood with data messages
        for i in 0..1000 {
            let _ = tx.send(Message::Data(i));
        }

        // Give time for messages to queue
        thread::sleep(Duration::from_millis(10));

        // Shutdown should return quickly even with backlog
        let shutdown_start = std::time::Instant::now();
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();
        let shutdown_duration = shutdown_start.elapsed();

        // Should shutdown within 100ms (fast, not waiting for drain)
        assert!(
            shutdown_duration < Duration::from_millis(100),
            "Immediate shutdown should exit quickly, took {:?}",
            shutdown_duration
        );
    }

    #[test]
    fn test_shutdown_drain_control_processes_control_and_mgmt() {
        let (tx, mut rx) =
            ActorScheduler::new_with_shutdown_mode(100, 100, ShutdownMode::DrainControl);

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        let actor_handle = thread::spawn(move || {
            let mut actor = CountingActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Send messages
        for i in 0..50 {
            tx.send(Message::Data(i)).unwrap();
        }
        for _ in 0..50 {
            tx.send(Message::Control(())).unwrap();
        }
        for _ in 0..50 {
            tx.send(Message::Management(())).unwrap();
        }

        // Give time for some to queue
        thread::sleep(Duration::from_millis(10));

        // Shutdown - should drain control+mgmt
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();

        // All control+mgmt should be processed, data may be dropped
        let control = control_count.load(Ordering::Relaxed);
        let mgmt = mgmt_count.load(Ordering::Relaxed);
        let data = data_count.load(Ordering::Relaxed);

        assert_eq!(control, 50, "All control messages should be processed");
        assert_eq!(mgmt, 50, "All management messages should be processed");
        // Data might be partially processed or dropped
        assert!(data <= 50, "Data messages may be dropped");
    }

    #[test]
    fn test_shutdown_drain_all_processes_everything() {
        let (tx, mut rx) = ActorScheduler::new_with_shutdown_mode(
            100,
            100,
            ShutdownMode::DrainAll {
                timeout: Duration::from_secs(1),
            },
        );

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        let actor_handle = thread::spawn(move || {
            let mut actor = CountingActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Send 100 of each type
        for i in 0..100 {
            tx.send(Message::Data(i)).unwrap();
            tx.send(Message::Control(())).unwrap();
            tx.send(Message::Management(())).unwrap();
        }

        // Give time for messages to queue
        thread::sleep(Duration::from_millis(50));

        // Shutdown - should drain all
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();

        // All messages should be processed
        assert_eq!(data_count.load(Ordering::Relaxed), 100);
        assert_eq!(control_count.load(Ordering::Relaxed), 100);
        assert_eq!(mgmt_count.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_shutdown_drain_all_timeout_fallback() {
        let (tx, mut rx) = ActorScheduler::new_with_shutdown_mode(
            10,   // Small burst limit to check shutdown frequently
            1000, // Large buffer to avoid blocking sends
            ShutdownMode::DrainAll {
                timeout: Duration::from_millis(50), // Short timeout
            },
        );

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        // Slow actor that sleeps on each message
        struct SlowActor {
            data_count: Arc<AtomicUsize>,
            control_count: Arc<AtomicUsize>,
            mgmt_count: Arc<AtomicUsize>,
        }

        impl Actor<i32, (), ()> for SlowActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                thread::sleep(Duration::from_millis(1)); // Slow!
                self.data_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn handle_control(&mut self, _: ()) -> HandlerResult {
                self.control_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn handle_management(&mut self, _: ()) -> HandlerResult {
                self.mgmt_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                match status {
                    SystemStatus::Idle => Ok(ActorStatus::Idle),
                    SystemStatus::Busy => Ok(ActorStatus::Busy),
                }
            }
        }

        let actor_handle = thread::spawn(move || {
            let mut actor = SlowActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Send 200 data messages (would take 200ms to process fully)
        for i in 0..200 {
            tx.send(Message::Data(i)).unwrap();
        }

        // Give actor time to start processing but not finish
        thread::sleep(Duration::from_millis(5));

        // Shutdown with 20ms timeout - should timeout before processing all 200
        let shutdown_start = std::time::Instant::now();
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();
        let shutdown_duration = shutdown_start.elapsed();

        // Shutdown should respect timeout (~50ms + overhead for normal run loop batch)
        assert!(
            shutdown_duration < Duration::from_millis(150),
            "Timeout should limit shutdown duration, took {:?}",
            shutdown_duration
        );

        // Should have processed SOME but definitely not all 200
        let processed = data_count.load(Ordering::Relaxed);
        assert!(
            processed < 200,
            "Timeout should prevent processing all messages, processed {}",
            processed
        );
        assert!(
            processed > 10,
            "Should process at least some messages before timeout"
        );
    }
}
