//! Message types for rasterizer actor communication.
//!
//! The rasterizer actor uses a **bootstrap pattern** for initialization:
//!
//! 1. Call `RasterizerActor::spawn_with_setup(num_threads)` to get a `SetupHandle`
//! 2. Send your response channel via `setup_handle.register(response_tx)`
//! 3. Receive the full `RasterizerHandle` back - now you can send render requests
//!
//! This pattern enforces at the **type level** that you cannot send render requests
//! without first registering where responses should go.
//!
//! ## Priority Lanes (after bootstrap)
//!
//! - **Data**: Frame rendering requests (backpressure when full)
//! - **Control**: Shutdown and high-priority commands
//! - **Management**: Configuration updates (thread count, etc.)

use crate::render::frame::Frame;
use crate::render::Pixel;
use actor_scheduler::ActorHandle;
use pixelflow_core::{Discrete, Manifold};
use std::sync::mpsc::{self, Sender, SyncSender};
use std::sync::Arc;
use std::time::Duration;

/// Frame rendering request (Data lane - high throughput, backpressure).
///
/// The Data lane is designed for high-volume work items and will block
/// senders when the buffer is full, providing natural backpressure.
pub struct RenderRequest<P: Pixel> {
    /// The color manifold to render.
    pub manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>,
    /// The frame buffer to render into.
    pub frame: Frame<P>,
}

/// Completed frame rendering response.
pub struct RenderResponse<P: Pixel> {
    /// The rendered frame.
    pub frame: Frame<P>,
    /// Time taken to render the frame.
    pub render_time: Duration,
}

/// Control messages (Control lane - highest priority, sleep-based fairness).
///
/// Control messages are processed before Management and Data messages.
/// The Control lane uses sleep-based backoff to ensure fairness and prevent
/// starvation of other message types.
///
/// To shut down the scheduler, use `Message::Shutdown` directly, not a control message.
#[derive(Debug, Clone, Copy)]
pub enum RasterControl {
    /// Pause rendering (stop processing Data messages).
    Pause,
    /// Resume rendering.
    Resume,
}

/// Management messages (Management lane - medium priority, configuration).
///
/// Management messages are processed after Control but before Data.
/// These are used for configuration changes that should be applied promptly
/// but don't need to interrupt ongoing work.
pub enum RasterManagement {
    /// Update the number of rendering threads.
    SetThreadCount(usize),
    /// Query current configuration (sends response via channel).
    GetConfig {
        response_tx: std::sync::mpsc::Sender<RasterConfig>,
    },
}

/// Current rasterizer configuration.
#[derive(Debug, Clone)]
pub struct RasterConfig {
    /// Number of threads used for work-stealing parallelism.
    pub num_threads: usize,
    /// Whether rendering is paused.
    pub paused: bool,
}

// ============================================================================
// Bootstrap Types - Type-level enforcement of initialization order
// ============================================================================

/// Setup message sent during bootstrap to register the response channel.
///
/// This message is sent through a dedicated setup channel, separate from
/// the actor's normal message lanes. The rasterizer blocks on this channel
/// before entering its main run loop.
pub struct RasterSetup<P: Pixel> {
    /// Channel where completed frames will be sent.
    pub response_tx: Sender<RenderResponse<P>>,
    /// Channel to send back the full actor handle.
    pub(crate) reply_tx: SyncSender<RasterizerHandle<P>>,
}

/// Handle returned after successful bootstrap - now you can send render requests.
///
/// This is the full actor handle that allows sending Data, Control, and Management
/// messages. You can only obtain this by completing the bootstrap handshake.
pub type RasterizerHandle<P> = ActorHandle<RenderRequest<P>, RasterControl, RasterManagement>;

/// Handle for initial setup - can ONLY register the response channel.
///
/// This is a capability-restricted handle. The only thing you can do with it
/// is call `register()` to complete the bootstrap handshake and receive
/// the full `RasterizerHandle`.
pub struct RasterizerSetupHandle<P: Pixel> {
    setup_tx: SyncSender<RasterSetup<P>>,
}

impl<P: Pixel> RasterizerSetupHandle<P> {
    /// Create a new setup handle with the given channel.
    pub(crate) fn new(setup_tx: SyncSender<RasterSetup<P>>) -> Self {
        Self { setup_tx }
    }

    /// Complete the bootstrap handshake by registering the response channel.
    ///
    /// This method:
    /// 1. Sends your response channel to the rasterizer
    /// 2. Waits for the rasterizer to send back its full actor handle
    /// 3. Returns the handle, allowing you to send render requests
    ///
    /// # Panics
    ///
    /// Panics if the rasterizer thread has died before completing setup.
    #[must_use]
    pub fn register(self, response_tx: Sender<RenderResponse<P>>) -> RasterizerHandle<P> {
        // Create reply channel for this handshake
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);

        // Build the setup message
        let setup = RasterSetup {
            response_tx,
            reply_tx,
        };

        // Send setup - blocks if channel full (shouldn't happen with buffer=1)
        self.setup_tx
            .send(setup)
            .expect("Rasterizer thread died before setup");

        // Wait for the full handle
        reply_rx
            .recv()
            .expect("Rasterizer thread died during setup")
    }
}

// Implement message traits for actor-scheduler integration
actor_scheduler::impl_control_message!(RasterControl);
