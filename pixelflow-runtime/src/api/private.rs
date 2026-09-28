use actor_scheduler::{ActorHandle, ActorScheduler};
use pixelflow_graphics::render::frame::Frame;
use std::sync::Arc;

use crate::api::public::CursorIcon;
use crate::pixel::PlatformPixel;
// use crate::input::MouseButton;

// Re-export WindowId from public API for backward compatibility
pub use crate::api::public::WindowId;

use crate::display::messages::{DisplayEvent, Window};

/// Commands sent to the Display Driver.
#[derive(Debug)]
pub enum DriverCommand {
    CreateWindow {
        id: WindowId,
        width: u32,
        height: u32,
        title: String,
    },
    DestroyWindow {
        id: WindowId,
    },
    Shutdown,
    Present {
        id: WindowId,
        frame: Frame<PlatformPixel>,
    },
    SetTitle {
        id: WindowId,
        title: String,
    },
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },
    CopyToClipboard(String),
    RequestPaste,
    Bell,
    SetCursorIcon {
        icon: CursorIcon,
    },
}

// Engine data message (high throughput, frame timing)
pub enum EngineData {
    FromDriver(DisplayEvent),
    FromApp(crate::api::public::AppData),
    VSync {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
    /// The driver granted the window buffer in answer to a `DisplayMgmt::RequestWindow`.
    ///
    /// Replaces `PresentComplete`. The driver is the buffer's resting owner now, so `Present`
    /// *is* the return and there is nothing to acknowledge; what remains is the outbound half —
    /// the driver handing the buffer out to be drawn into.
    WindowGranted(Window),
}

impl std::fmt::Debug for EngineData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FromDriver(event) => f.debug_tuple("FromDriver").field(event).finish(),
            Self::FromApp(data) => f.debug_tuple("FromApp").field(data).finish(),
            Self::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => f
                .debug_struct("VSync")
                .field("timestamp", timestamp)
                .field("target_timestamp", target_timestamp)
                .field("refresh_interval", refresh_interval)
                .finish(),
            Self::WindowGranted(window) => f.debug_tuple("WindowGranted").field(window).finish(),
        }
    }
}

impl From<crate::api::public::AppData> for EngineData {
    fn from(data: crate::api::public::AppData) -> Self {
        EngineData::FromApp(data)
    }
}

impl From<DisplayEvent>
    for actor_scheduler::Message<EngineData, EngineControl, crate::api::public::AppManagement>
{
    fn from(evt: DisplayEvent) -> Self {
        actor_scheduler::Message::Data(EngineData::FromDriver(evt))
    }
}

/// The green-host bootstrap bundle, handed to the engine in one message by
/// `EngineControl::GreenReady`.
///
/// Carries what the engine needs once vsync *and* the render coordinator (`coordinator_node.rs`,
/// step 5c of the mesh migration doc) are both up on the one green-host thread: the vsync
/// control edge, the coordinator's data-lane sender, and two handles kept only for the shutdown
/// cascade. Named fields rather than a tuple because `host` and `forwarder` share a type
/// (`ActorHandle<Infallible, Infallible, Infallible>`) and a positional tuple would let them be
/// silently transposed.
///
/// The renderer's own live handle is deliberately **not** here. `RendererHandle` is not
/// `Clone`, and its lanes are single-producer SPSC, so it cannot be shared between the
/// coordinator (which needs it continuously, for every render) and the engine (which would only
/// ever touch it once, at shutdown) without breaking that invariant. The coordinator keeps sole
/// ownership; the renderer's own scheduler halts gracefully once that handle drops with the
/// green host — the same disconnect-triggered shutdown the forwarder already relies on one hop
/// downstream (`engine_troupe.rs`'s `RasterizerForwarder`).
#[derive(Debug)]
pub struct GreenReadyBundle {
    pub(crate) vsync_control: actor_scheduler::GreenSender<crate::vsync_actor::VsyncCommand>,
    pub(crate) coordinator: actor_scheduler::GreenSender<crate::coordinator_node::CoordinatorData>,
    pub(crate) host: actor_scheduler::ActorHandle<
        std::convert::Infallible,
        std::convert::Infallible,
        std::convert::Infallible,
    >,
    pub(crate) forwarder: actor_scheduler::ActorHandle<
        std::convert::Infallible,
        std::convert::Infallible,
        std::convert::Infallible,
    >,
}

// Engine control message (low frequency, configuration/lifecycle)
#[derive(Debug, Default)]
pub enum EngineControl {
    UpdateRefreshRate(f64),
    /// The green host is up, hosting both vsync and the render coordinator. There is exactly
    /// one send site (`Troupe::with_config`) and one delivery (the shell intercept below), so a
    /// single message beats several that would otherwise need to race each other into place
    /// first. Boxed: the bundle is far larger than every other variant, and `Quit` is the hot
    /// one.
    GreenReady(Box<GreenReadyBundle>),
    #[default]
    Quit,
}

// For now, let's assume channel.rs was supposed to define them but the refactor moved them here.
// But if I define them here, I need to make sure channel.rs imports them.
// The error was "file not found for module api".
// So I am restoring the file.

pub const DISPLAY_EVENT_BUFFER_SIZE: usize = 256;
pub const DISPLAY_EVENT_BURST_LIMIT: usize = 32;

pub type EngineActorHandle =
    ActorHandle<EngineData, EngineControl, crate::api::public::AppManagement>;
pub type EngineActorScheduler =
    ActorScheduler<EngineData, EngineControl, crate::api::public::AppManagement>;

#[must_use]
pub fn create_engine_actor(
    wake_handler: Option<Arc<dyn actor_scheduler::WakeHandler>>,
) -> (EngineActorHandle, EngineActorScheduler) {
    actor_scheduler::ActorScheduler::new_with_wake_handler(
        DISPLAY_EVENT_BURST_LIMIT,
        DISPLAY_EVENT_BUFFER_SIZE,
        wake_handler,
    )
}
