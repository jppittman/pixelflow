#![cfg(use_web_display)]

//! Web DisplayDriver implementation using OffscreenCanvas and SharedArrayBuffer.
//!
//! Driver struct is cmd_tx - trivially Clone.
//! run() reads Configure, creates canvas resources, runs event loop.

pub mod ipc;

use crate::api::private::{
    DriverCommand, EngineActorHandle as EngineSender, EngineControl, EngineData,
};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::error::RuntimeError;
use actor_scheduler::Message;
use ipc::SharedRingBuffer;
use js_sys::SharedArrayBuffer;
use log::error;
use pixelflow_render::color::Rgba;
use pixelflow_render::Frame;
use std::cell::RefCell;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{ImageData, OffscreenCanvas, OffscreenCanvasRenderingContext2d};

// Thread-local storage for web resources passed from JS
thread_local! {
    static RESOURCES: RefCell<Option<(OffscreenCanvas, SharedArrayBuffer, f64)>> = RefCell::new(None);
}

/// Initialize web resources. Must be called from JS before creating driver.
pub fn init_resources(canvas: OffscreenCanvas, sab: SharedArrayBuffer, scale_factor: f64) {
    RESOURCES.with(|r| *r.borrow_mut() = Some((canvas, sab, scale_factor)));
}

// --- Run State (only original driver has this) ---
struct RunState {
    cmd_rx: Receiver<DriverCommand<Rgba>>,
    engine_tx: EngineSender,
}

// --- Display Driver ---

/// Web display driver using OffscreenCanvas.
///
/// Clone to get a handle for sending commands. Only the original can run().
pub struct WebDisplayDriver {
    cmd_tx: SyncSender<DriverCommand<Rgba>>,
    /// Only present on original, None on clones
    run_state: Option<RunState>,
}

impl Clone for WebDisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            run_state: None, // Clones can't run
        }
    }
}

impl DisplayDriver for WebDisplayDriver {
    type Pixel = Rgba;

    fn new(engine_tx: EngineSender) -> Result<Self, RuntimeError> {
        let (cmd_tx, cmd_rx) = sync_channel(16);

        Ok(Self {
            cmd_tx,
            run_state: Some(RunState { cmd_rx, engine_tx }),
        })
    }

    fn send(&self, cmd: DriverCommand<Rgba>) -> Result<(), RuntimeError> {
        self.cmd_tx
            .send(cmd)
            .map_err(|_| RuntimeError::DriverChannelDisconnected)?;
        Ok(())
    }

    fn run(&self) -> Result<(), RuntimeError> {
        let run_state = self
            .run_state
            .as_ref()
            .ok_or_else(|| RuntimeError::DriverCloneError)?;

        run_event_loop(&run_state.cmd_rx, &run_state.engine_tx)
    }
}

// --- Event Loop ---

fn run_event_loop(
    cmd_rx: &Receiver<DriverCommand<Rgba>>,
    engine_tx: &EngineSender,
) -> Result<(), RuntimeError> {
    // 1. Read CreateWindow command first
    let (window_id, width_px, height_px) = match cmd_rx
        .recv()
        .map_err(|_| RuntimeError::DriverChannelDisconnected)?
    {
        DriverCommand::CreateWindow {
            id, width, height, ..
        } => (id, width, height),
        other => return Err(RuntimeError::UnexpectedCommand(format!("{:?}", other))),
    };

    // Get web resources from thread-local storage
    let (canvas, sab, scale_factor) = RESOURCES.with(|r| {
        r.borrow_mut()
            .take()
            .ok_or_else(|| RuntimeError::WebResourcesNotInitialized)
    })?;

    let context = canvas
        .get_context("2d")
        .map_err(|_| RuntimeError::WebContextError)?
        .ok_or_else(|| RuntimeError::WebContextNull)?
        .dyn_into::<OffscreenCanvasRenderingContext2d>()
        .map_err(|_| RuntimeError::WebContextCastError)?;

    let ipc = SharedRingBuffer::new(&sab);

    // Resize canvas
    canvas.set_width(width_px);
    canvas.set_height(height_px);

    // Send initial resize (web scale_factor typically comes from devicePixelRatio)
    let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::WindowCreated {
        id: window_id,
        width_px,
        height_px,
        scale: scale_factor as f32, // Use provided scale factor
    }));

    // 2. Create state and run event loop
    let mut state = WebState {
        context,
        ipc,
        width_px,
        height_px,
    };

    state.event_loop(cmd_rx, engine_tx)
}

// --- Web State (only exists during run) ---

struct WebState {
    context: OffscreenCanvasRenderingContext2d,
    ipc: SharedRingBuffer,
    width_px: u32,
    height_px: u32,
}

impl WebState {
    fn event_loop(
        &mut self,
        cmd_rx: &Receiver<DriverCommand<Rgba>>,
        engine_tx: &EngineSender,
    ) -> Result<(), RuntimeError> {
        loop {
            // 1. Poll IPC events (from main thread via SharedArrayBuffer)
            match self.ipc.blocking_read_timeout(16) {
                Ok(Some(evt)) => {
                    if matches!(evt, DisplayEvent::CloseRequested { .. }) {
                        return Ok(());
                    }
                    let _ = engine_tx.send(EngineCommand::DisplayEvent(evt));
                }
                Ok(None) => {
                    // Timeout, no events
                }
                Err(e) => {
                    // Log error but continue
                    web_sys::console::error_1(&format!("IPC read error: {}", e).into());
                }
            }

            // 2. Process commands from engine
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    DriverCommand::CreateWindow { .. } => {
                        // Already created, ignore
                    }
                    DriverCommand::DestroyWindow { .. } => {
                        return Ok(());
                    }
                    DriverCommand::Shutdown => {
                        return Ok(());
                    }
                    DriverCommand::Present { frame, .. } => {
                        // CRITICAL: Always return frame to avoid deadlock, even on error
                        let (returned_frame, result) = self.handle_present(frame);
                        if let Err(e) = result {
                            error!("Web: Present failed: {:?}", e);
                        }
                        let _ = engine_tx
                            .send(Message::Data(EngineData::PresentComplete(returned_frame)));
                    }
                    DriverCommand::SetTitle { .. } => {
                        // Not supported in worker context
                    }
                    DriverCommand::SetSize { .. } => {
                        // Not supported in worker context
                    }
                    DriverCommand::CopyToClipboard(_) => {
                        // Not supported in worker context
                    }
                    DriverCommand::RequestPaste => {
                        // Not supported in worker context
                    }
                    DriverCommand::Bell => {
                        // Not supported in worker context
                    }
                    DriverCommand::SetCursorIcon { .. } => {
                        // Not supported in worker context
                    }
                }
            }
        }
    }

    fn handle_present(&mut self, frame: Frame<Rgba>) -> (Frame<Rgba>, Result<(), RuntimeError>) {
        let data = frame.as_bytes();
        let result =
            ImageData::new_with_u8_clamped_array_and_sh(Clamped(data), frame.width, frame.height)
                .map_err(|e| RuntimeError::WebImageDataError(format!("{:?}", e)))
                .and_then(|image_data| {
                    self.context
                        .put_image_data(&image_data, 0.0, 0.0)
                        .map_err(|e| RuntimeError::WebPutImageError(format!("{:?}", e)))
                });

        (frame, result)
    }
}
