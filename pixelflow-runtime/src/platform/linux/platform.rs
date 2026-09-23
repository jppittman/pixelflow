//! Linux platform implementation.
//!
//! Bridge to X11DisplayDriver using the new PlatformOps trait.

use crate::display::messages::{
    DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Surface, WindowId,
};
use crate::display::ops::{DriverOut, PlatformOps};
use crate::error::RuntimeError;
use crate::platform::linux::window::X11Window;
use crate::platform::waker::X11Waker;
use actor_scheduler::{ActorStatus, SystemStatus};
use log::{error, info};
use pixelflow_graphics::render::color::Bgra8;
use std::mem;
use std::sync::OnceLock;
use x11::xlib;

use super::events;

/// Linux platform pixel type (BGRA for X11).
pub type LinuxPixel = Bgra8;

/// Shared X11Waker instance.
///
/// The waker is created by the troupe and stored here so that LinuxOps
/// can use the same instance. This ensures that when the troupe calls
/// wake() on message send, it wakes the same waker that has been
/// initialized with the X11 display/window via set_target().
pub(super) static SHARED_WAKER: OnceLock<X11Waker> = OnceLock::new();

/// Set the shared X11Waker for the Linux platform.
///
/// Called by the troupe before creating LinuxOps.
pub fn set_shared_waker(waker: X11Waker) {
    SHARED_WAKER.set(waker).ok();
}

/// Linux platform operations - direct X11 implementation.
pub struct LinuxOps {
    waker: X11Waker,
    window: Option<X11Window>,
}

impl LinuxOps {
    /// Create new Linux platform ops.
    ///
    /// Uses the shared waker set by `set_shared_waker()`, or creates a new
    /// one if not set (for backwards compatibility in tests).
    ///
    /// Takes no engine handle: outbound events are returned via `DriverOut`, so these ops can
    /// be driven with no engine running at all. `PlatformActor` performs the sends.
    pub fn new() -> Result<Self, RuntimeError> {
        let waker = SHARED_WAKER.get().cloned().unwrap_or_else(X11Waker::new);
        Ok(Self {
            waker,
            window: None,
        })
    }
}

impl PlatformOps for LinuxOps {
    fn handle_data(
        &mut self,
        data: DisplayData,
        out: &mut DriverOut,
    ) -> Result<(), actor_scheduler::HandlerError> {
        if let Some(x11_window) = &mut self.window {
            match data {
                DisplayData::Present { mut window } => {
                    let (returned_frame, result) = x11_window.present(window.frame);
                    if let Err(e) = result {
                        error!("X11: Present failed: {:?}", e);
                    }
                    window.frame = returned_frame;
                    out.blitted(window);
                }
            }
        }
        Ok(())
    }

    fn handle_control(
        &mut self,
        ctrl: DisplayControl,
        _out: &mut DriverOut,
    ) -> Result<(), actor_scheduler::HandlerError> {
        if let Some(window) = &mut self.window {
            match ctrl {
                DisplayControl::SetTitle { title, .. } => {
                    window.set_title(&title);
                }
                DisplayControl::SetSize { width, height, .. } => {
                    window.set_size(width, height);
                }
                DisplayControl::Copy { text } => {
                    window.copy_to_clipboard(&text);
                }
                DisplayControl::RequestPaste => {
                    window.request_paste();
                }
                DisplayControl::SetCursor { cursor, .. } => {
                    window.set_cursor(cursor);
                }
                DisplayControl::Bell => {
                    window.bell();
                }
                DisplayControl::ShowWindow { .. } => window.show(),
                DisplayControl::HideWindow { .. } => window.hide(),
                // X11 retains nothing to repaint: every present uploads a whole frame, so
                // the next present is the redraw.
                DisplayControl::RequestRedraw { .. } => {}
            }
        }
        Ok(())
    }

    fn handle_management(
        &mut self,
        mgmt: DisplayMgmt,
        out: &mut DriverOut,
    ) -> Result<(), actor_scheduler::HandlerError> {
        match mgmt {
            DisplayMgmt::Create { settings } => {
                info!(
                    "X11: Creating window '{}' {}x{}",
                    settings.title, settings.width, settings.height
                );
                match X11Window::new(&settings, &self.waker) {
                    Ok(window) => {
                        // Geometry only — the driver allocates the buffer from this. X11 samples
                        // 1:1, so the lattice and the point extent are the same numbers.
                        out.event(DisplayEvent::WindowCreated {
                            surface: Surface {
                                id: WindowId(window.window),
                                width_px: window.width,
                                height_px: window.height,
                                frame_width: window.width,
                                frame_height: window.height,
                                scale: window.scale_factor,
                            },
                        });
                        self.window = Some(window);
                    }
                    Err(e) => {
                        error!("Failed to create X11 window: {}", e);
                    }
                }
            }
            DisplayMgmt::Destroy { .. } => {
                // Drop window to close it
                self.window = None;
            }
            // Answered by `PlatformActor` from its keeper; the ops hold no buffer.
            DisplayMgmt::RequestWindow => {}
        }
        Ok(())
    }

    fn handle_os(
        &mut self,
        status: SystemStatus,
        out: &mut DriverOut,
    ) -> Result<ActorStatus, actor_scheduler::HandlerError> {
        if let Some(window) = &mut self.window {
            let window_id = WindowId(window.window);

            // Poll for X11 events
            // If Busy, check pending without blocking.
            // If Idle, block on XNextEvent (waker will interrupt).
            let block = matches!(status, SystemStatus::Idle);

            unsafe {
                let has_event = if block {
                    true // XNextEvent blocks
                } else {
                    xlib::XPending(window.display) > 0
                };

                if has_event {
                    let mut event: xlib::XEvent = mem::zeroed();
                    xlib::XNextEvent(window.display, &mut event);

                    if let Some(display_event) = events::map_event(&event, window, window_id) {
                        if matches!(display_event, DisplayEvent::CloseRequested { .. }) {
                            info!("X11: CloseRequested");
                        }
                        out.event(display_event);
                    }

                    // Drain remaining pending events non-blocking
                    while xlib::XPending(window.display) > 0 {
                        xlib::XNextEvent(window.display, &mut event);
                        if let Some(display_event) = events::map_event(&event, window, window_id) {
                            out.event(display_event);
                        }
                    }
                    return Ok(ActorStatus::Busy);
                }
            }
        }
        Ok(ActorStatus::Idle)
    }
}
