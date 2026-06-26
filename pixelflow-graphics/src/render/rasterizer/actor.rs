//! Rasterizer actor for asynchronous frame rendering.
//!
//! The `RasterizerActor` provides a message-based interface for parallel frame
//! rendering using the actor-scheduler's three-lane priority system:
//!
//! - **Data Lane**: Frame rendering requests with natural backpressure
//! - **Management Lane**: Thread count updates and configuration queries
//! - **Control Lane**: Shutdown and pause/resume commands
//!
//! # Bootstrap Pattern
//!
//! The rasterizer uses a **bootstrap handshake** to ensure you can't send
//! render requests before registering where responses should go:
//!
//! ```ignore
//! use pixelflow_graphics::render::rasterizer::RasterizerActor;
//! use std::sync::mpsc;
//!
//! // Step 1: Spawn with setup handle
//! let (setup_handle, join_handle) = RasterizerActor::spawn_with_setup(4);
//!
//! // Step 2: Create your response channel
//! let (response_tx, response_rx) = mpsc::channel();
//!
//! // Step 3: Register and get full handle - NOW you can send render requests
//! let rasterizer = setup_handle.register(response_tx);
//!
//! // Step 4: Send render requests
//! rasterizer.send(Message::Data(my_render_request)).unwrap();
//!
//! // Step 5: Receive responses
//! let response = response_rx.recv().unwrap();
//! ```

use super::messages::{
    RasterConfig, RasterControl, RasterManagement, RasterSetup, RasterizerSetupHandle,
    RenderRequest, RenderResponse,
};
use super::rasterize;
use crate::render::Pixel;
use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, ActorTypes, HandlerError, HandlerResult, SystemStatus,
};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Rasterizer actor for parallel frame rendering.
///
/// This actor manages a pool of worker threads for rendering frames via
/// work-stealing parallelism. It processes rendering requests asynchronously
/// while allowing dynamic reconfiguration of thread count.
///
/// Use [`spawn_with_setup`](Self::spawn_with_setup) to create and start the actor.
pub struct RasterizerActor<P: Pixel> {
    /// Number of threads for work-stealing parallelism.
    num_threads: usize,
    /// Whether rendering is currently paused.
    paused: bool,
    /// Channel to send completed frames back. Set during bootstrap.
    response_tx: Sender<RenderResponse<P>>,
}

impl<P: Pixel + Send + 'static> ActorTypes for RasterizerActor<P> {
    type Data = RenderRequest<P>;
    type Control = RasterControl;
    type Management = RasterManagement;
}

impl<P: Pixel + Send + 'static> RasterizerActor<P> {
    /// Spawn the rasterizer actor with a bootstrap handshake.
    ///
    /// This is the **primary way** to create a rasterizer. It spawns the actor
    /// thread and returns a `SetupHandle` that you must use to register your
    /// response channel before sending any render requests.
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of worker threads for parallel rendering.
    ///   Use 1 for single-threaded, or `std::thread::available_parallelism()`
    ///   for utilizing all CPU cores.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `RasterizerSetupHandle` - Use this to register your response channel
    /// - `JoinHandle` - The thread handle for the rasterizer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (setup_handle, _thread) = RasterizerActor::spawn_with_setup(4);
    /// let (response_tx, response_rx) = std::sync::mpsc::channel();
    /// let rasterizer = setup_handle.register(response_tx);
    /// // Now you can send render requests via `rasterizer`
    /// ```
    #[must_use]
    pub fn spawn_with_setup(num_threads: usize) -> (RasterizerSetupHandle<P>, JoinHandle<()>) {
        // Create the setup channel (buffer=1, only one setup message ever)
        let (setup_tx, setup_rx) = mpsc::sync_channel::<RasterSetup<P>>(1);

        // Spawn the actor thread
        let join_handle = thread::spawn(move || {
            // PHASE 1: Wait for setup message (blocks until register() is called)
            let setup = setup_rx
                .recv()
                .expect("Setup handle dropped without calling register()");

            // Extract response channel from setup
            let response_tx = setup.response_tx;
            let reply_tx = setup.reply_tx;

            // PHASE 2: Create the actor scheduler
            let (handle, mut scheduler) =
                ActorScheduler::<RenderRequest<P>, RasterControl, RasterManagement>::new(64, 16);

            // Send the full handle back to the caller
            reply_tx
                .send(handle)
                .expect("Setup caller dropped reply channel");

            // PHASE 3: Create actor and run
            let mut actor = RasterizerActor {
                num_threads: num_threads.max(1),
                paused: false,
                response_tx,
            };

            log::info!("RasterizerActor started with {} threads", actor.num_threads);

            scheduler.run(&mut actor);
        });

        // Return the setup handle
        let setup_handle = RasterizerSetupHandle::new(setup_tx);
        (setup_handle, join_handle)
    }

    /// Get current configuration.
    fn config(&self) -> RasterConfig {
        RasterConfig {
            num_threads: self.num_threads,
            paused: self.paused,
        }
    }
}

impl<P: Pixel + Send> Actor<RenderRequest<P>, RasterControl, RasterManagement>
    for RasterizerActor<P>
{
    fn handle_data(&mut self, request: RenderRequest<P>) -> HandlerResult {
        // Skip rendering if paused
        if self.paused {
            log::debug!("Rasterizer paused, dropping render request");
            return Ok(());
        }

        let RenderRequest {
            manifold,
            mut frame,
        } = request;

        // Render the frame
        let start = Instant::now();
        rasterize(&manifold, &mut frame, self.num_threads);
        let render_time = start.elapsed();

        log::trace!(
            "Rendered {}x{} frame in {:?} ({} threads)",
            frame.width,
            frame.height,
            render_time,
            self.num_threads
        );

        // Send response back - receiver may be dropped if display was shutdown
        let response = RenderResponse { frame, render_time };
        match self.response_tx.send(response) {
            Ok(()) => Ok(()),
            Err(_) => {
                // Response receiver dropped is expected during shutdown
                log::debug!("Render response receiver dropped");
                Ok(())
            }
        }
    }

    fn handle_control(&mut self, ctrl: RasterControl) -> HandlerResult {
        match ctrl {
            RasterControl::Pause => {
                log::info!("Rasterizer paused");
                self.paused = true;
            }
            RasterControl::Resume => {
                log::info!("Rasterizer resumed");
                self.paused = false;
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, mgmt: RasterManagement) -> HandlerResult {
        match mgmt {
            RasterManagement::SetThreadCount(count) => {
                let new_count = count.max(1);
                log::info!(
                    "Rasterizer thread count updated: {} -> {}",
                    self.num_threads,
                    new_count
                );
                self.num_threads = new_count;
            }
            RasterManagement::GetConfig { response_tx } => {
                // Receiver may be dropped if requester cancelled, that's fine
                response_tx.send(self.config()).ok();
            }
        }
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // No external work to do during park, just wait for messages
        Ok(ActorStatus::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::frame::Frame;
    use crate::render::Color;
    use actor_scheduler::Message;
    use std::sync::{mpsc, Arc};

    #[test]
    fn test_rasterizer_actor_basic() {
        // Step 1: Spawn with setup handle
        let (setup_handle, actor_thread) = RasterizerActor::<Rgba8>::spawn_with_setup(1);

        // Step 2: Create response channel and register
        let (response_tx, response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Create a render request (no response_tx field anymore!)
        let frame = Frame::new(64, 64);
        let red = Color::Rgb(255, 0, 0);

        let request = RenderRequest {
            manifold: Arc::new(red),
            frame,
        };

        // Send render request
        handle
            .send(Message::Data(request))
            .expect("Failed to send render request");

        // Wait for response (comes through our registered channel)
        let response = response_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("Failed to receive response");

        // Verify frame was rendered
        assert_eq!(response.frame.width, 64);
        assert_eq!(response.frame.height, 64);
        assert!(response.render_time.as_nanos() > 0);

        // Shutdown
        handle
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn test_rasterizer_actor_thread_count_update() {
        // Spawn with setup
        let (setup_handle, actor_thread) = RasterizerActor::<Rgba8>::spawn_with_setup(2);

        // Register response channel
        let (response_tx, _response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Update thread count
        handle
            .send(Message::Management(RasterManagement::SetThreadCount(4)))
            .expect("Failed to send SetThreadCount");

        // Query config
        let (config_tx, config_rx) = mpsc::channel();
        handle
            .send(Message::Management(RasterManagement::GetConfig {
                response_tx: config_tx,
            }))
            .expect("Failed to send GetConfig");

        let config = config_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("Failed to receive config");

        assert_eq!(config.num_threads, 4);
        assert!(!config.paused);

        // Shutdown
        handle
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn test_rasterizer_actor_pause_resume() {
        // Spawn with setup
        let (setup_handle, actor_thread) = RasterizerActor::<Rgba8>::spawn_with_setup(1);

        // Register response channel
        let (response_tx, response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Pause rendering
        handle
            .send(Message::Control(RasterControl::Pause))
            .expect("Failed to send Pause");

        // Send a render request (should be dropped because paused)
        let frame = Frame::new(32, 32);
        let blue = Color::Rgb(0, 0, 255);

        let request = RenderRequest {
            manifold: Arc::new(blue),
            frame,
        };

        handle
            .send(Message::Data(request))
            .expect("Failed to send render request");

        // Should timeout because rendering is paused
        assert!(response_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err());

        // Resume
        handle
            .send(Message::Control(RasterControl::Resume))
            .expect("Failed to send Resume");

        // Shutdown
        handle
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }
}
