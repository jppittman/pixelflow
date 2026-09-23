//! `EngineCore` — the pure decision logic behind `EngineHandler` (`engine_troupe.rs`).
//!
//! Mirrors the `VsyncCore`/`RenderCore`/`CoordinatorCore` split: a `step_*` call takes a message
//! and **returns** what to emit, so the app/vsync/driver relays left once rendering moved to its
//! own node (`coordinator_node.rs`, step 5c of
//! `docs/designs/pixelflow-runtime-engine-mesh-migration.md`) are table-testable with no
//! scheduler, no handles, and no threads in the loop. `EngineHandler` shrinks to the thin
//! adapter that owns the real channels and flushes the returned word.
//!
//! # What left
//!
//! The render coordinator (`RenderCoordinator`, `frame_number`, and the `renderer`/
//! `driver_data`/`vsync_data` ports that used to carry its decisions) is gone from here
//! entirely — it runs as its own `Node` now, on the same green-host thread as vsync. What
//! remains is a single relay port, `coordinator`: every input that used to drive `self.render`
//! directly now just describes *which* `CoordinatorData` to forward, and the coordinator decides
//! the rest.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::{
    AppData, AppManagement, EngineEvent, EngineEventControl, EngineEventData,
    EngineEventManagement, WindowId,
};
use crate::coordinator_node::CoordinatorData;
use crate::display::messages::{DisplayControl, DisplayEvent, DisplayMgmt};
use crate::input::MouseButton;
use crate::vsync_actor::VsyncCommand;
use actor_scheduler::mealy::Transducer;
use actor_scheduler::HandlerError;

/// One optional slot per downstream edge. Default (all `None`, `quit: false`) is the silent
/// step — most inputs move state without telling any peer.
#[derive(Default)]
pub(crate) struct EngineOut {
    pub(crate) app: Option<EngineEvent>,
    pub(crate) driver_control: Option<DisplayControl>,
    pub(crate) driver_mgmt: Option<DisplayMgmt>,
    pub(crate) vsync_control: Option<VsyncCommand>,
    /// → the render coordinator's data lane (`coordinator_node.rs`). Replaces the old
    /// `renderer`/`driver_data`/`vsync_data` ports outright: this type relays a decision, the
    /// coordinator makes it and owns the ports those decisions used to ride.
    pub(crate) coordinator: Option<CoordinatorData>,
    /// Run the shutdown cascade: the green host (vsync + the coordinator node it also runs),
    /// the renderer forwarder, app-drop, driver, self.
    pub(crate) quit: bool,
}

/// Pure engine mediator: the app/driver/vsync relay decisions, with no actor handle or channel
/// in it. No self-port — retries of a dropped `RequestWindow` are the coordinator's own concern
/// now, not this type's.
pub(crate) struct EngineCore;

impl EngineCore {
    pub(crate) fn new() -> Self {
        Self
    }

    fn step_app_data(&mut self, app_data: AppData, out: &mut EngineOut) {
        match app_data {
            AppData::RenderSurface(scene) => {
                log::debug!("Engine: Received RenderSurface from app");
                // The app has provided its compute graph, so permit VSync to request another
                // frame without waiting for rasterization to finish.
                out.vsync_control = Some(VsyncCommand::ReturnToken);
                out.coordinator = Some(CoordinatorData::Submit(scene));
            }
            AppData::Skipped => {
                // App says nothing to render - return token anyway
                out.vsync_control = Some(VsyncCommand::ReturnToken);
            }
        }
    }

    fn step_driver_event(&mut self, event: DisplayEvent, out: &mut EngineOut) {
        match event {
            // Both window-lifecycle events are pure relays: the driver has already built the
            // buffer for the new geometry by the time this arrives, so there is nothing here to
            // take delivery of, stamp, or retire. What is left is telling the app its size, and
            // nudging the coordinator in case a buffer can now exist to draw into.
            DisplayEvent::WindowCreated { surface } => {
                log::debug!(
                    "Relaying WindowCreated: id={}, {}x{}, scale={}",
                    surface.id.0,
                    surface.width_px,
                    surface.height_px,
                    surface.scale
                );

                out.coordinator = Some(CoordinatorData::Advance);
                out.app = Some(EngineEvent::Control(EngineEventControl::WindowCreated {
                    id: surface.id,
                    width_px: surface.width_px,
                    height_px: surface.height_px,
                    scale: surface.scale,
                }));
            }
            DisplayEvent::Resized { surface } => {
                log::debug!(
                    "Relaying Resized: id={}, {}x{}",
                    surface.id.0,
                    surface.width_px,
                    surface.height_px
                );

                out.coordinator = Some(CoordinatorData::Advance);
                out.app = Some(EngineEvent::Control(EngineEventControl::Resized {
                    id: surface.id,
                    width_px: surface.width_px,
                    height_px: surface.height_px,
                }));
            }
            DisplayEvent::Key {
                symbol,
                modifiers,
                text,
                ..
            } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::KeyDown {
                    key: symbol,
                    mods: modifiers,
                    text,
                }));
            }
            DisplayEvent::MouseButtonPress { button, x, y, .. } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::MouseClick {
                    x: x as u32,
                    y: y as u32,
                    button: convert_mouse_button(button),
                }));
            }
            DisplayEvent::MouseButtonRelease { button, x, y, .. } => {
                out.app = Some(EngineEvent::Management(
                    EngineEventManagement::MouseRelease {
                        x: x as u32,
                        y: y as u32,
                        button: convert_mouse_button(button),
                    },
                ));
            }
            DisplayEvent::MouseMove {
                x, y, modifiers, ..
            } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::MouseMove {
                    x: x as u32,
                    y: y as u32,
                    mods: modifiers,
                }));
            }
            DisplayEvent::MouseScroll {
                dx,
                dy,
                x,
                y,
                modifiers,
                ..
            } => {
                out.app = Some(EngineEvent::Management(
                    EngineEventManagement::MouseScroll {
                        x: x as u32,
                        y: y as u32,
                        dx,
                        dy,
                        mods: modifiers,
                    },
                ));
            }
            DisplayEvent::CloseRequested { .. } => {
                log::debug!("Close requested");
                // Notify the app, then the shell drops the handle as part of the shutdown
                // cascade `quit` triggers — cleanup goes in the app's Drop impl.
                out.app = Some(EngineEvent::Control(EngineEventControl::CloseRequested));
                out.quit = true;
            }
            DisplayEvent::FocusGained { .. } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::FocusGained));
            }
            DisplayEvent::FocusLost { .. } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::FocusLost));
            }
            DisplayEvent::PasteData { text } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::Paste(text)));
            }
            DisplayEvent::ScaleChanged { id, scale } => {
                log::debug!("Relaying ScaleChanged: id={}, scale={}", id.0, scale);
                out.app = Some(EngineEvent::Control(EngineEventControl::ScaleChanged {
                    id,
                    scale,
                }));
            }
        }
    }
}

impl Transducer for EngineCore {
    type Control = EngineControl;
    type Management = AppManagement;
    type Data = EngineData;
    type Out = EngineOut;

    fn step_data(&mut self, data: EngineData) -> Result<EngineOut, HandlerError> {
        let mut out = EngineOut::default();
        match data {
            EngineData::FromApp(app_data) => self.step_app_data(app_data, &mut out),
            EngineData::FromDriver(event) => self.step_driver_event(event, &mut out),
            EngineData::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => {
                // ALWAYS request frame from app (app builds compute graphs fast). Token bucket
                // is managed atomically by VSync.
                out.app = Some(EngineEvent::Data(EngineEventData::RequestFrame {
                    timestamp,
                    target_timestamp,
                    refresh_interval,
                }));

                // The tick is also what retries a request the coordinator dropped in transit.
                out.coordinator = Some(CoordinatorData::Advance);
            }
            EngineData::WindowGranted(window) => {
                out.coordinator = Some(CoordinatorData::Granted(window));
            }
        }
        Ok(out)
    }

    fn step_control(&mut self, ctrl: EngineControl) -> Result<EngineOut, HandlerError> {
        let mut out = EngineOut::default();
        match ctrl {
            EngineControl::Quit => out.quit = true,
            EngineControl::UpdateRefreshRate(rr) => {
                out.vsync_control = Some(VsyncCommand::UpdateRefreshRate(rr));
            }
            // Handle-carrying: the shell intercepts this before the core ever sees it, since a
            // pure core has no field to hold the handle in. Only the type split arriving with
            // the port/wiring phase removes this arm.
            EngineControl::GreenReady(..) => {
                unreachable!("handled by the engine shell")
            }
        }
        Ok(out)
    }

    fn step_management(&mut self, mgmt: AppManagement) -> Result<EngineOut, HandlerError> {
        let mut out = EngineOut::default();
        match mgmt {
            // Handle-carrying / bootstrap messages: the shell intercepts these before the core
            // ever sees them, since a pure core has no field to hold a handle or an `Arc<dyn
            // Application>` in. Only the type split arriving with the port/wiring phase removes
            // these arms.
            AppManagement::Configure(_) => {
                unreachable!("handled by the engine shell")
            }
            AppManagement::RegisterApp(_) => {
                unreachable!("handled by the engine shell")
            }
            AppManagement::SetTitle(title) => {
                out.driver_control = Some(DisplayControl::SetTitle {
                    id: WindowId::PRIMARY,
                    title,
                });
            }
            AppManagement::ResizeRequest(width, height) => {
                out.driver_control = Some(DisplayControl::SetSize {
                    id: WindowId::PRIMARY,
                    width,
                    height,
                });
            }
            AppManagement::CopyToClipboard(text) => {
                out.driver_control = Some(DisplayControl::Copy { text });
            }
            AppManagement::RequestPaste => {
                out.driver_control = Some(DisplayControl::RequestPaste);
            }
            AppManagement::Bell => {
                out.driver_control = Some(DisplayControl::Bell);
            }
            AppManagement::SetCursorIcon(icon) => {
                out.driver_control = Some(DisplayControl::SetCursor {
                    id: WindowId::PRIMARY,
                    cursor: icon,
                });
            }
            AppManagement::CreateWindow(descriptor) => {
                // Engine assigns the window ID (for now, just use PRIMARY for single window)
                let id = WindowId::PRIMARY;
                log::info!(
                    "Relaying CreateWindow request: assigning id={}, {}x{} \"{}\"",
                    id.0,
                    descriptor.width,
                    descriptor.height,
                    descriptor.title
                );
                out.driver_mgmt = Some(DisplayMgmt::Create {
                    settings: descriptor,
                });
            }
            AppManagement::Quit => out.quit = true,
        }
        Ok(out)
    }
}

/// Convert raw mouse button code to MouseButton enum
fn convert_mouse_button(button: u8) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        _ => MouseButton::Other(button),
    }
}

#[cfg(test)]
mod tests {
    //! `EngineCore` in isolation — no handles, no threads, no scheduler. Mirrors
    //! `vsync_actor.rs`'s `mod tests` for `VsyncCore` and `coordinator_node.rs`'s `mod tests`
    //! for `CoordinatorCore`.
    //!
    //! What used to live here and doesn't any more: everything about the render protocol itself
    //! (request/render/present, resize races, `holds_buffer`) — that is `CoordinatorCore`'s
    //! contract now, pinned in `coordinator_node.rs`. What is left is the relay: does the right
    //! input produce the right `CoordinatorData` on the `coordinator` port.

    use super::*;
    use crate::display::messages::{Generation, Surface, Window, WindowMeta};
    use std::time::{Duration, Instant};

    /// The lattice the fixture scene is compiled for. These tests never bake
    /// it — they assert which port fired, not what reached a buffer — so it
    /// only has to be a legal frame; the windows they grant are this size.
    const FIXTURE_FRAME: [u32; 2] = [100, 100];

    /// A constant black scene — these tests care about which port fires, not what is drawn.
    fn manifold() -> pixelflow_graphics::render::scene::Scene {
        crate::testing::black_scene(FIXTURE_FRAME)
    }

    fn surface(width_px: u32, height_px: u32) -> Surface {
        Surface {
            id: WindowId(1),
            width_px,
            height_px,
            frame_width: width_px,
            frame_height: height_px,
            scale: 1.0,
        }
    }

    fn window(meta_size: (u32, u32)) -> Window {
        Window::rejoin(
            pixelflow_graphics::render::Frame::new(meta_size.0, meta_size.1),
            WindowMeta {
                id: WindowId(1),
                width_px: meta_size.0,
                height_px: meta_size.1,
                scale: 1.0,
                generation: Generation::NONE,
            },
        )
    }

    #[test]
    fn a_vsync_tick_requests_a_frame_from_the_app_and_nudges_the_coordinator() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::VSync {
                timestamp: Instant::now(),
                target_timestamp: Instant::now(),
                refresh_interval: Duration::from_millis(16),
            })
            .unwrap();

        assert!(
            matches!(
                out.app,
                Some(EngineEvent::Data(EngineEventData::RequestFrame { .. }))
            ),
            "every tick must ask the app for a frame"
        );
        assert!(
            matches!(out.coordinator, Some(CoordinatorData::Advance)),
            "every tick must also nudge the coordinator, in case a request was lost in transit"
        );
    }

    #[test]
    fn render_surface_returns_the_vsync_token_and_submits_to_the_coordinator() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::FromApp(AppData::RenderSurface(manifold())))
            .unwrap();

        assert!(
            matches!(out.vsync_control, Some(VsyncCommand::ReturnToken)),
            "submitting a scene must release the token so vsync can ask again"
        );
        assert!(
            matches!(out.coordinator, Some(CoordinatorData::Submit(_))),
            "a new scene must be forwarded to the coordinator"
        );
    }

    #[test]
    fn an_app_bell_rings_the_driver_bell() {
        let mut core = EngineCore::new();
        let out = core.step_management(AppManagement::Bell).unwrap();

        assert!(matches!(out.driver_control, Some(DisplayControl::Bell)));
    }

    #[test]
    fn skipped_frame_returns_the_token_without_touching_the_coordinator() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::FromApp(AppData::Skipped))
            .unwrap();

        assert!(matches!(out.vsync_control, Some(VsyncCommand::ReturnToken)));
        assert!(
            out.coordinator.is_none(),
            "nothing to draw, so the coordinator has nothing to hear about"
        );
    }

    #[test]
    fn window_granted_relays_to_the_coordinator() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::WindowGranted(window((100, 100))))
            .unwrap();

        assert!(
            matches!(out.coordinator, Some(CoordinatorData::Granted(_))),
            "a granted window must be forwarded to the coordinator"
        );
    }

    #[test]
    fn window_created_and_resized_nudge_the_coordinator_to_advance() {
        let mut core = EngineCore::new();

        let out = core
            .step_data(EngineData::FromDriver(DisplayEvent::WindowCreated {
                surface: surface(100, 100),
            }))
            .unwrap();
        assert!(matches!(out.coordinator, Some(CoordinatorData::Advance)));

        let out = core
            .step_data(EngineData::FromDriver(DisplayEvent::Resized {
                surface: surface(200, 200),
            }))
            .unwrap();
        assert!(matches!(out.coordinator, Some(CoordinatorData::Advance)));
    }

    #[test]
    fn quit_control_sets_quit() {
        let mut core = EngineCore::new();
        let out = core.step_control(EngineControl::Quit).unwrap();
        assert!(
            out.quit,
            "EngineControl::Quit must trigger the shutdown cascade"
        );
    }

    #[test]
    fn set_title_emits_driver_control() {
        let mut core = EngineCore::new();
        let out = core
            .step_management(AppManagement::SetTitle("hello".into()))
            .unwrap();
        assert!(
            matches!(out.driver_control, Some(DisplayControl::SetTitle { .. })),
            "SetTitle must relay to the driver's control port"
        );
    }

    #[test]
    fn close_requested_sets_quit_and_notifies_the_app() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::FromDriver(DisplayEvent::CloseRequested {
                id: WindowId::PRIMARY,
            }))
            .unwrap();

        assert!(
            out.quit,
            "a close request must trigger the shutdown cascade"
        );
        assert!(
            matches!(
                out.app,
                Some(EngineEvent::Control(EngineEventControl::CloseRequested))
            ),
            "the app must be told the window is closing"
        );
    }
}
