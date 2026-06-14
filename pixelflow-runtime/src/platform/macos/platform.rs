use crate::api::private::{EngineActorHandle, EngineData, WindowId};
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Window};
use crate::display::ops::PlatformOps;
use crate::error::RuntimeError;
use crate::platform::macos::cocoa::{self, event_type, NSApplication, NSPasteboard};
use crate::platform::macos::events;
use crate::platform::macos::sys;
use crate::platform::macos::window::MacWindow;
use crate::platform::PlatformPixel;
use actor_scheduler::{ActorStatus, HandlerError, HandlerResult, Message, SystemStatus};
use pixelflow_graphics::render::Frame;

use std::collections::HashMap;

const NS_APPLICATION_ACTIVATION_POLICY_REGULAR: isize = 0;

/// The macOS Platform Actor.
/// Manages NSApplication, NSWindows, and Event Loop.
pub struct MetalOps {
    app: NSApplication,
    windows: HashMap<WindowId, MacWindow>,
    // Mapping from NSWindow pointer to WindowId for event routing
    // Note: NSWindow is a wrapper around Id, so we cast Id to usize or wrap generic
    window_map: HashMap<usize, WindowId>,
    // Handle to send events back to the engine
    event_tx: EngineActorHandle,
}

unsafe impl Send for MetalOps {}

impl MetalOps {
    pub fn new(event_tx: EngineActorHandle) -> Result<Self, RuntimeError> {
        // Initialize Cocoa Application
        let app = unsafe {
            // Pool:
            let cls_pool = sys::class(b"NSAutoreleasePool\0");
            let _pool: sys::Id = sys::send(
                sys::send(cls_pool, sys::sel(b"alloc\0")),
                sys::sel(b"init\0"),
            );

            let app = NSApplication::shared();

            // Use wrapper methods
            app.set_activation_policy(NS_APPLICATION_ACTIVATION_POLICY_REGULAR);

            app.finish_launching();
            app.activate_ignoring_other_apps(true);

            app
        };

        Ok(Self {
            app,
            windows: HashMap::new(),
            window_map: HashMap::new(),
            event_tx,
        })
    }
}

impl PlatformOps for MetalOps {
    fn handle_data(&mut self, msg: DisplayData) -> HandlerResult {
        match msg {
            DisplayData::Present { mut window } => {
                log::trace!("MetalOps: Presenting frame for window {:?}", window.id);
                // Present to the native window, get back the frame
                if let Some(win) = self.windows.get_mut(&window.id) {
                    // Present returns the frame after blitting
                    window.frame = win.present(window.frame);
                } else {
                    log::warn!(
                        "MetalOps: Window {:?} not found, returning window without presenting",
                        window.id
                    );
                }
                // Return the window to the engine for reuse
                self.event_tx
                    .send(Message::Data(EngineData::PresentComplete(window)))
                    .expect("Failed to send PresentComplete to engine");
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, msg: DisplayControl) -> HandlerResult {
        match msg {
            DisplayControl::SetTitle { id, title } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_title(&title);
                }
            }
            DisplayControl::SetSize { id, width, height } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_size(width, height);
                }
            }
            DisplayControl::SetCursor { id, cursor } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_cursor(cursor);
                }
            }
            DisplayControl::SetVisible { id, visible } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_visible(visible);
                }
            }
            DisplayControl::RequestRedraw { id } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.request_redraw();
                }
            }
            DisplayControl::Bell => {
                // NSBeep()
            }
            DisplayControl::Copy { text } => {
                let pb = NSPasteboard::general();
                pb.clear_contents();
                pb.set_string(&text);
            }
            DisplayControl::RequestPaste => {
                // Implementation pending
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: DisplayMgmt) -> HandlerResult {
        match msg {
            DisplayMgmt::Create { settings } => {
                match MacWindow::new(settings) {
                    Ok(win) => {
                        let ptr = win.window.0;
                        let width = win.current_width;
                        let height = win.current_height;
                        let scale = win.scale_factor();

                        // Generate window ID from pointer (like Linux does)
                        let id = WindowId(ptr as u64);

                        self.windows.insert(id, win);
                        self.window_map.insert(ptr as usize, id);

                        // Create Window with initial frame buffer
                        let window = Window {
                            id,
                            frame: Frame::<PlatformPixel>::new(width, height),
                            width_px: width,
                            height_px: height,
                            scale,
                        };

                        // Emit WindowCreated event so Engine knows initial size
                        self.event_tx
                            .send(Message::Data(EngineData::FromDriver(
                                DisplayEvent::WindowCreated { window },
                            )))
                            .expect("Failed to send WindowCreated event to engine");
                    }
                    Err(e) => {
                        eprintln!("Failed to create window: {}", e);
                    }
                }
            }
            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.set_visible(false);
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it
                    self.window_map.remove(&(win.window.0 as usize));
                }
            }
        }
        Ok(())
    }

    fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // Logic for event loop interaction
        // The CocoaWaker posts an NSEvent when messages arrive, so distantFuture is safe.
        unsafe {
            let until_date = match status {
                SystemStatus::Idle => {
                    // Block until an event arrives (waker will post NSEvent when messages come)
                    let cls = sys::class(b"NSDate\0");
                    sys::send(cls, sys::sel(b"distantFuture\0"))
                }
                SystemStatus::Busy => {
                    // Immediate return
                    let cls = sys::class(b"NSDate\0");
                    sys::send(cls, sys::sel(b"distantPast\0"))
                }
            };

            let mode = cocoa::make_nsstring("kCFRunLoopDefaultMode");

            // Poll for window resize
            for (id, mac_window) in self.windows.iter_mut() {
                if let Some((width, height)) = mac_window.poll_resize() {
                    // Create new Window with resized frame buffer
                    let window = Window {
                        id: *id,
                        frame: Frame::<PlatformPixel>::new(width, height),
                        width_px: width,
                        height_px: height,
                        scale: mac_window.scale_factor(),
                    };
                    self.event_tx
                        .send(Message::Data(EngineData::FromDriver(
                            DisplayEvent::Resized { window },
                        )))
                        .expect("Failed to send Resized event to engine");
                }
            }

            // Use wrapper for next_event
            let event = self.app.next_event(
                u64::MAX,
                until_date,
                mode,
                true, // dequeue
            );

            // Release mode string
            sys::send::<()>(mode, sys::sel(b"release\0"));

            if !event.is_null() {
                let ty = event.type_();
                match ty {
                    event_type::APP_KIT_DEFINED
                    | event_type::SYSTEM_DEFINED
                    | event_type::APPLICATION_DEFINED => {
                        log::trace!("MetalOps: Received Internal/WakeUp event");
                    }
                    _ => {
                        let ns_win = event.window();
                        if !ns_win.0.is_null() {
                            if let Some(wid) = self.window_map.get(&(ns_win.0 as usize)) {
                                // We have the window ID.
                                // Get window height from self.windows if needed for coordinate flip.
                                let height = if let Some(w) = self.windows.get(wid) {
                                    w.size().1 as f64
                                } else {
                                    0.0
                                };

                                if let Some(ev) = events::map_event(event, height) {
                                    // Dispatch ev!
                                    // We send it to the Engine via event_tx
                                    self.event_tx
                                        .send(Message::Data(EngineData::FromDriver(ev)))
                                        .expect("Failed to send display event to engine");
                                }
                            }
                        }
                    }
                }

                // Only route spatial events through Cocoa's responder chain.
                // Keyboard/scroll events are fully owned by us - routing them
                // through sendEvent: causes the system "bop" sound.
                if event_type::is_spatial(ty) {
                    self.app.send_event(event);
                }
            }
        }
        // Always return Idle since we block inside next_event when needed
        Ok(ActorStatus::Idle)
    }
}

impl Drop for MetalOps {
    fn drop(&mut self) {
        unsafe {
            // [NSApp terminate:nil]
            sys::send_1::<(), sys::Id>(self.app.0, sys::sel(b"terminate:\0"), std::ptr::null_mut());
        }
    }
}
