use crate::api::private::WindowId;
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Surface};
use crate::display::ops::{DriverOut, PlatformOps};
use crate::error::RuntimeError;
use crate::input::Selection;
use crate::platform::macos::cocoa::{self, event_type, NSApplication, NSPasteboard};
use crate::platform::macos::events;
use crate::platform::macos::sys;
use crate::platform::macos::window::MacWindow;
use actor_scheduler::{ActorStatus, HandlerError, HandlerResult, SystemStatus};

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
}

unsafe impl Send for MetalOps {}

impl MetalOps {
    pub fn new() -> Result<Self, RuntimeError> {
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
        })
    }
}

impl PlatformOps for MetalOps {
    fn handle_data(&mut self, msg: DisplayData, out: &mut DriverOut) -> HandlerResult {
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
                out.blitted(window);
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, msg: DisplayControl, out: &mut DriverOut) -> HandlerResult {
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
            DisplayControl::ShowWindow { id } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.show();
                }
            }
            DisplayControl::HideWindow { id } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.hide();
                }
            }
            DisplayControl::RequestRedraw { id } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.request_redraw();
                }
            }
            DisplayControl::Bell => cocoa::beep(),
            DisplayControl::Copy {
                selection: Selection::Clipboard,
                text,
            } => {
                let pb = NSPasteboard::general();
                pb.clear_contents();
                pb.set_string(&text);
            }
            // No primary selection on macOS; highlighting must not clobber the
            // pasteboard.
            DisplayControl::Copy {
                selection: Selection::Primary,
                ..
            } => {}
            DisplayControl::ToggleFullscreen { id } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.toggle_fullscreen();
                }
            }
            // macOS has no primary selection; both read the pasteboard.
            DisplayControl::RequestPaste { .. } => {
                if let Some(text) = NSPasteboard::general().string() {
                    out.event(DisplayEvent::PasteData { text });
                }
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: DisplayMgmt, out: &mut DriverOut) -> HandlerResult {
        match msg {
            DisplayMgmt::Create { settings } => {
                match MacWindow::new(settings) {
                    Ok(win) => {
                        let ptr = win.window.0;
                        let width = win.current_width;
                        let height = win.current_height;
                        let scale = win.scale_factor();
                        // Frame is device-pixel sized (the sample lattice);
                        // width_px/height_px stay in points for layout/input.
                        let (px_w, px_h) = win.pixel_size();

                        // Generate window ID from pointer (like Linux does)
                        let id = WindowId(ptr as u64);

                        self.windows.insert(id, win);
                        self.window_map.insert(ptr as usize, id);

                        // Geometry only — the driver allocates the buffer from this. The two
                        // extents differ on Retina: layout is in points, the lattice in pixels.
                        out.event(DisplayEvent::WindowCreated {
                            surface: Surface {
                                id,
                                width_px: width,
                                height_px: height,
                                frame_width: px_w,
                                frame_height: px_h,
                                scale,
                            },
                        });
                    }
                    Err(e) => {
                        eprintln!("Failed to create window: {}", e);
                    }
                }
            }
            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.hide();
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it
                    self.window_map.remove(&(win.window.0 as usize));
                }
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
    ) -> Result<ActorStatus, HandlerError> {
        // Logic for event loop interaction
        // The CocoaWaker posts an NSEvent when messages arrive, so distantFuture is safe.
        unsafe {
            // Only the FIRST wait may block (when the scheduler says Idle).
            // Everything already queued is then drained without blocking:
            // a handle_os pass must empty the NSEvent queue, or wake events pile
            // up ahead of real input and a KeyDown can sit behind an
            // unbounded backlog (the "can't Ctrl-C out of `yes`" wedge).
            // This mirrors LinuxOps::handle_os's `while XPending > 0` drain.
            let first_until_date: sys::Id = match status {
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
            let distant_past: sys::Id = {
                let cls = sys::class(b"NSDate\0");
                sys::send(cls, sys::sel(b"distantPast\0"))
            };

            let mode = cocoa::make_nsstring("kCFRunLoopDefaultMode");

            // Poll for window resize
            for (id, mac_window) in self.windows.iter_mut() {
                if let Some((width, height)) = mac_window.poll_resize() {
                    let (px_w, px_h) = mac_window.pixel_size();
                    out.event(DisplayEvent::Resized {
                        surface: Surface {
                            id: *id,
                            width_px: width,
                            height_px: height,
                            frame_width: px_w,
                            frame_height: px_h,
                            scale: mac_window.scale_factor(),
                        },
                    });
                }
            }

            let mut until_date = first_until_date;
            let mut processed_any = false;

            loop {
                let event = self.app.next_event(
                    u64::MAX,
                    until_date,
                    mode,
                    true, // dequeue
                );
                // Subsequent iterations never block.
                until_date = distant_past;

                if event.is_null() {
                    break;
                }
                processed_any = true;

                let ty = event.type_();
                match ty {
                    event_type::APPLICATION_DEFINED => {
                        // A CocoaWaker wake token was just dequeued; allow the
                        // next wake() to post a fresh one.
                        crate::platform::waker::consume_wake_token();
                        log::trace!("MetalOps: Received WakeUp event");
                    }
                    event_type::APP_KIT_DEFINED | event_type::SYSTEM_DEFINED => {
                        log::trace!("MetalOps: Received internal AppKit/system event");
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
                                    out.event(ev);
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

            // Release mode string
            sys::send::<()>(mode, sys::sel(b"release\0"));

            // Report windows the user closed during event routing (the click
            // on the close button runs windowWillClose: synchronously inside
            // sendEvent: above). CloseRequested triggers the engine's full
            // shutdown cascade — without this the process outlives its last
            // window as an invisible zombie.
            for ptr in crate::platform::macos::window::drain_closed_windows() {
                if let Some(id) = self.window_map.remove(&ptr) {
                    self.windows.remove(&id);
                    processed_any = true;
                    out.event(DisplayEvent::CloseRequested { id });
                }
            }

            // Report backing-scale changes (window dragged to a display with
            // different DPI). Drained after closes: a window that closed this
            // pass is already out of window_map, so its scale event is dropped.
            //
            // A scale change keeps the point-space bounds but changes the
            // sample lattice, so a circulating buffer has the wrong pixel
            // density. Emit Resized — which is what makes the driver allocate
            // at the new density and retire the old buffer — then ScaleChanged
            // so the app can re-bake density-keyed resources.
            for (ptr, scale) in crate::platform::macos::window::drain_scale_changes() {
                if let Some(id) = self.window_map.get(&ptr).copied() {
                    let win = self
                        .windows
                        .get(&id)
                        .expect("window_map entry without a matching window");
                    let (px_w, px_h) = win.pixel_size();
                    processed_any = true;
                    out.event(DisplayEvent::Resized {
                        surface: Surface {
                            id,
                            width_px: win.current_width,
                            height_px: win.current_height,
                            frame_width: px_w,
                            frame_height: px_h,
                            scale,
                        },
                    });
                    out.event(DisplayEvent::ScaleChanged { id, scale });
                }
            }

            // Busy after real work so the scheduler re-drains its lanes before
            // blocking on the doorbell (a dequeued wake event usually means
            // messages are waiting).
            Ok(if processed_any {
                ActorStatus::Busy
            } else {
                ActorStatus::Idle
            })
        }
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
