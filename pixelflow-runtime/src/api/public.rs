use crate::input::{KeySymbol, Modifiers, MouseButton, Selection};
// use pixelflow_render::Frame;

/// Window ID wrapper for identifying windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub u64);

impl WindowId {
    /// Primary window ID (for single-window applications).
    pub const PRIMARY: Self = Self(0);
}

/// Events sent from the Engine to the Application.
///
/// The Engine emits events to communicate state changes and requests to the application.
/// Events are organized by priority: **Control** (most critical), **Management** (medium),
/// **Data** (high-frequency, lowest priority).
///
/// # Event Contract
///
/// When the Engine sends an event, it guarantees:
/// - **Timeliness**: Events are delivered as soon as feasible after the triggering action
/// - **Accuracy**: Event data accurately reflects the OS event or engine state
/// - **Ordering**: Events within a priority lane maintain causal order
///
/// The Application should handle events promptly to keep the rendering loop responsive.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Control events (critical state changes, highest priority).
    ///
    /// These indicate important changes to the window or application state
    /// that may require immediate response.
    Control(EngineEventControl),

    /// Management events (input and user interactions, medium priority).
    ///
    /// These carry information about user input (keyboard, mouse) and clipboard.
    Management(EngineEventManagement),

    /// Data events (frame requests, lowest priority).
    ///
    /// These signal that it's time to render a frame.
    Data(EngineEventData),
}

/// Control events from the Engine (window state changes).
#[derive(Debug, Clone)]
pub enum EngineEventControl {
    /// Window was created by the driver.
    ///
    /// # Contract
    ///
    /// **Engine**: Relays WindowCreated event from driver to app.
    ///
    /// **Application**: Window is now ready for rendering. Should start VSync and begin sending frames.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier for future references
    /// - `width_px`: Width in physical pixels
    /// - `height_px`: Height in physical pixels
    /// - `scale`: DPI scale factor
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },

    /// Window has been resized.
    ///
    /// # Contract
    ///
    /// **Engine**: Relays resize event from driver to app.
    ///
    /// **Application**: Should update its render target size and send new frames.
    /// The application may receive multiple `Resize` events before rendering a frame.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `width_px`: Width in physical pixels
    /// - `height_px`: Height in physical pixels
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },

    /// User requested to close the window.
    ///
    /// # Contract
    ///
    /// **Engine**: The user clicked the close button or pressed Alt+F4.
    ///
    /// **Application**: Should clean up and shut down gracefully. Not receiving
    /// this event doesn't mean the window is still open—the OS may force-close it.
    ///
    /// # Note
    ///
    /// This is a request, not a command. The application can ignore it (for unsaved
    /// changes confirmation), but the OS may force-close the window anyway.
    CloseRequested,

    /// DPI scale factor changed.
    ///
    /// # Contract
    ///
    /// **Engine**: Relays scale change event from driver to app.
    ///
    /// **Application**: May need to rerender at new resolution or adjust font sizes.
    /// The scale factor affects how logical pixels map to physical pixels.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `scale`: Scale factor (e.g., 1.0 = 96 DPI, 2.0 = 192 DPI on high-DPI displays)
    ScaleChanged { id: WindowId, scale: f64 },
}

/// Management events from the Engine (input and interactions).
#[derive(Debug, Clone)]
pub enum EngineEventManagement {
    /// A key was pressed.
    ///
    /// # Contract
    ///
    /// **Engine**: User pressed a keyboard key; Engine provides symbol and modifiers.
    ///
    /// **Application**: Should process the keystroke and update state.
    /// If text input is needed, the `text` field contains the character(s) if available.
    ///
    /// # Arguments
    ///
    /// - `key`: Key symbol (arrow, letter, function key, etc.)
    /// - `mods`: Modifier keys (Shift, Ctrl, Alt, etc.)
    /// - `text`: Composed text character (Some for printable characters, None for control keys)
    KeyDown {
        key: KeySymbol,
        mods: Modifiers,
        text: Option<String>,
    },

    /// Mouse button pressed.
    ///
    /// # Contract
    ///
    /// **Engine**: User clicked a mouse button; coordinates are in logical pixels.
    ///
    /// **Application**: Should process the click (e.g., select text, open menu).
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: Click position in logical pixels
    /// - `button`: Which button was clicked (left=0, right=1, middle=2, etc.)
    MouseClick { x: u32, y: u32, button: MouseButton },

    /// Mouse button released.
    ///
    /// # Contract
    ///
    /// **Engine**: User released a mouse button; coordinates match button location.
    ///
    /// **Application**: Should complete any click action started by the corresponding press.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: Release position in logical pixels
    /// - `button`: Which button was released
    MouseRelease { x: u32, y: u32, button: MouseButton },

    /// Mouse moved.
    ///
    /// # Contract
    ///
    /// **Engine**: Mouse pointer moved to a new position.
    ///
    /// **Application**: May update cursor icon, highlight selections, or track position.
    /// High-frequency event; application should handle efficiently.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: New position in logical pixels
    /// - `mods`: Current modifier key state
    MouseMove { x: u32, y: u32, mods: Modifiers },

    /// Mouse wheel or trackpad scrolled.
    ///
    /// # Contract
    ///
    /// **Engine**: User scrolled; delta is in logical units.
    ///
    /// **Application**: Should scroll content in the specified direction.
    /// Sign convention: Positive = scroll down/right, Negative = scroll up/left.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: Cursor position at time of scroll (logical pixels)
    /// - `dx`, `dy`: Scroll delta (touchpad may report fine-grained values)
    /// - `mods`: Modifier key state (Shift, Ctrl for alternate scroll behavior)
    MouseScroll {
        x: u32,
        y: u32,
        dx: f32,
        dy: f32,
        mods: Modifiers,
    },

    /// Window gained focus.
    ///
    /// # Contract
    ///
    /// **Engine**: Window is now the active (foreground) window.
    ///
    /// **Application**: May resume animations, activate input handlers, etc.
    FocusGained,

    /// Window lost focus.
    ///
    /// # Contract
    ///
    /// **Engine**: Window is no longer the active window.
    ///
    /// **Application**: May pause animations, suspend input processing, etc.
    FocusLost,

    /// The content of a selection, answering `AppManagement::RequestPaste`.
    ///
    /// # Contract
    ///
    /// **Engine**: Sends exactly one per `RequestPaste`, naming the selection
    /// that was read. The text is empty when the selection is empty or
    /// unavailable. Answers for the same selection arrive in request order;
    /// answers for different selections may not.
    ///
    /// **Application**: Decides what the text is for — typically a paste.
    Paste { selection: Selection, text: String },
}

/// Data events from the Engine (frame synchronization).
///
/// These are high-frequency events that drive the render loop.
#[derive(Debug, Clone)]
pub enum EngineEventData {
    /// Request a new frame.
    ///
    /// # Contract
    ///
    /// **Engine**: It's time to render a new frame; provides timing information.
    ///
    /// **Application**: Should render and send a frame via `AppData::RenderSurface`.
    ///
    /// # Arguments
    ///
    /// - `timestamp`: Actual time this event was emitted (can be used for delta-time)
    /// - `target_timestamp`: Ideal time the frame should display on screen
    /// - `refresh_interval`: Monitor refresh period (e.g., 16.67ms for 60Hz)
    ///
    /// # Example Use
    ///
    /// Use `target_timestamp` to animate content that should be time-accurate.
    /// Write that timestamp into the compiled scene's block
    /// (`PackedManifold::bind_with`) rather than recompiling it per frame.
    RequestFrame {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
}

/// Commands sent from the Application to the Engine (frame rendering).
///
/// Application sends these to respond to frame requests. These are the output
/// of the application's render loop—typically sent in response to `EngineEvent::Data(RequestFrame)`.
///
/// # Message Contract
///
/// When the application sends a frame, it establishes a contract:
/// - **Precondition**: Application received a `RequestFrame` event
/// - **Action**: Engine buffers the scene and renders it to the window
/// - **Postcondition**: Pixels are on screen at the next VSync
/// - **Blocking**: May block if buffer is full, but high-priority (won't be dropped)
///
pub enum AppData {
    /// Render a [`Scene`](pixelflow_graphics::render::scene::Scene) to the
    /// window.
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Provides the scene — four channel kernels
    /// compiled into one program over the frame's lattice, with the pixel
    /// pack inside the kernel.
    ///
    /// **Receiver** (Engine): Renders the scene into the frame buffer and
    /// presents it. The scene is rendered fresh for each submission, so it
    /// can be animated or interactive.
    ///
    /// # Coordinate space
    ///
    /// A scene is DEVICE-PIXEL space by construction — its kernels were
    /// compiled against a frame's own lattice, so an author working in points
    /// precomposed the embedding with `Kernel::at` before compiling, and the
    /// engine applies no transform of its own.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_graphics::render::scene::{constant_platform_scene, Scene};
    ///
    /// // A solid colour at the frame's size.
    /// tx.send(Message::Data(AppData::RenderSurface(
    ///     constant_platform_scene([1.0, 0.0, 0.0, 1.0], [width, height]),
    /// )))?;
    ///
    /// // Any other scene is the same shape: a compiled program, bound.
    /// tx.send(Message::Data(AppData::RenderSurface(Scene::Packed(frame))))?;
    /// ```
    ///
    /// # Performance Notes
    ///
    /// One internal-loop JIT call per stripe, pack included, stripes pulled
    /// by the render workers. Compiling a scene is what costs — do it on
    /// resize, not per frame; a value that changes every frame is an
    /// argument of the program (a `Uniform`, written into its
    /// `UniformBlock`), not a constant.
    RenderSurface(pixelflow_graphics::render::scene::Scene),

    /// Skip this frame (no rendering needed).
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Indicates that the frame hasn't changed and doesn't need rendering.
    ///
    /// **Receiver** (Engine): Reuses the previous frame or pauses rendering.
    /// Useful for reducing power consumption when nothing is animating.
    ///
    /// # Example Use
    ///
    /// Terminal emulator receives `RequestFrame` but has no new content to render.
    /// Instead of rendering the same pixels again, it sends `Skipped`.
    ///
    /// # Note
    ///
    /// Not all platforms support frame skipping. The engine may ignore this
    /// and present a blank/black frame instead. Use only when you know the
    /// previous frame is still correct.
    Skipped,
}

impl std::fmt::Debug for AppData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RenderSurface(_) => f.debug_tuple("RenderSurface").finish(),
            Self::Skipped => f.debug_tuple("Skipped").finish(),
        }
    }
}

/// Application trait that defines the logic.
pub trait Application {
    fn send(&self, event: EngineEvent) -> Result<(), crate::error::RuntimeError>;
}

impl Application
    for actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>
{
    fn send(&self, event: EngineEvent) -> Result<(), crate::error::RuntimeError> {
        let msg = match event {
            EngineEvent::Control(ctrl) => actor_scheduler::Message::Control(ctrl),
            EngineEvent::Management(mgmt) => actor_scheduler::Message::Management(mgmt),
            EngineEvent::Data(data) => actor_scheduler::Message::Data(data),
        };
        self.send(msg)
            .map_err(|e| crate::error::RuntimeError::EventSendError(e.to_string()))
    }
}

/// Application management commands (change title, etc.)
pub enum AppManagement {
    /// Configure the engine with initial settings (sent on startup).
    Configure(crate::config::EngineConfig),
    /// Register the application handle so the engine can send events back to the app.
    RegisterApp(std::sync::Arc<dyn Application + Send + Sync>),
    /// Request window creation.
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Requests a new window with the specified settings.
    ///
    /// **Receiver** (Engine): Relays the request to the driver. Driver will create the window,
    /// assign an ID, and respond with WindowCreated event containing the assigned ID.
    CreateWindow(WindowDescriptor),
    SetTitle(String),
    ResizeRequest(u32, u32),
    /// Put text in a selection, for other applications to paste.
    Copy {
        selection: Selection,
        text: String,
    },
    /// Read a selection; the text arrives as `EngineEventManagement::Paste`.
    RequestPaste(Selection),
    /// Ring the bell: the platform's alert sound or equivalent.
    Bell,
    /// Enter full screen, or leave it.
    ToggleFullscreen,
    SetCursorIcon(CursorIcon),
    Quit,
}

impl std::fmt::Debug for AppManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppManagement::Configure(config) => f.debug_tuple("Configure").field(config).finish(),
            AppManagement::RegisterApp(_) => f.debug_tuple("RegisterApp").field(&"<app>").finish(),
            AppManagement::CreateWindow(descriptor) => {
                f.debug_tuple("CreateWindow").field(descriptor).finish()
            }
            AppManagement::SetTitle(s) => f.debug_tuple("SetTitle").field(s).finish(),
            AppManagement::ResizeRequest(w, h) => {
                f.debug_tuple("ResizeRequest").field(&(w, h)).finish()
            }
            AppManagement::Copy { selection, text } => f
                .debug_struct("Copy")
                .field("selection", selection)
                .field("text", text)
                .finish(),
            AppManagement::RequestPaste(selection) => {
                f.debug_tuple("RequestPaste").field(selection).finish()
            }
            AppManagement::Bell => f.write_str("Bell"),
            AppManagement::ToggleFullscreen => f.write_str("ToggleFullscreen"),
            AppManagement::SetCursorIcon(icon) => {
                f.debug_tuple("SetCursorIcon").field(icon).finish()
            }
            AppManagement::Quit => f.write_str("Quit"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
}

/// Descriptor for creating a new window.
#[derive(Debug, Clone)]
pub struct WindowDescriptor {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub resizable: bool,
}

impl Default for WindowDescriptor {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "PixelFlow".into(),
            resizable: true,
        }
    }
}

/// Unregistered engine handle - can ONLY register an application.
///
/// This handle enforces correct initialization: you must call `register()`
/// before you can send frames or management commands. This makes improper
/// initialization inexpressible at the type level.
pub struct UnregisteredEngineHandle {
    inner: actor_scheduler::ActorHandle<
        crate::api::private::EngineData,
        crate::api::private::EngineControl,
        AppManagement,
    >,
}

impl UnregisteredEngineHandle {
    /// Create from raw actor handle (internal use only)
    pub(crate) fn new(
        handle: actor_scheduler::ActorHandle<
            crate::api::private::EngineData,
            crate::api::private::EngineControl,
            AppManagement,
        >,
    ) -> Self {
        Self { inner: handle }
    }

    /// Register application and create window.
    ///
    /// This atomically sends both RegisterApp and CreateWindow messages,
    /// ensuring the engine knows about your app before creating the window.
    ///
    /// Returns a full `EngineHandle` that can send frames and commands.
    ///
    /// The app will receive `WindowCreated` event via its control channel
    /// when the window is ready.
    pub fn register(
        self,
        app: std::sync::Arc<dyn Application + Send + Sync>,
        window: WindowDescriptor,
    ) -> Result<EngineHandle, crate::error::RuntimeError> {
        use actor_scheduler::Message;

        // Send RegisterApp first
        self.inner
            .send(Message::Management(AppManagement::RegisterApp(app)))
            .map_err(|e| {
                crate::error::RuntimeError::InitError(format!("Failed to register app: {}", e))
            })?;

        // Then send CreateWindow
        self.inner
            .send(Message::Management(AppManagement::CreateWindow(window)))
            .map_err(|e| {
                crate::error::RuntimeError::InitError(format!("Failed to create window: {}", e))
            })?;

        // Return full handle
        Ok(EngineHandle { inner: self.inner })
    }
}

/// Registered engine handle - full access to engine API.
///
/// This handle is returned by `UnregisteredEngineHandle::register()` and
/// provides all engine functionality (sending frames, management commands).
pub struct EngineHandle {
    inner: actor_scheduler::ActorHandle<
        crate::api::private::EngineData,
        crate::api::private::EngineControl,
        AppManagement,
    >,
}

impl EngineHandle {
    /// Test-only constructor for creating EngineHandle directly.
    ///
    /// This bypasses the registration API for testing purposes only.
    #[doc(hidden)]
    pub fn new_for_test(
        inner: actor_scheduler::ActorHandle<
            crate::api::private::EngineData,
            crate::api::private::EngineControl,
            AppManagement,
        >,
    ) -> Self {
        Self { inner }
    }

    /// Send a message to the engine (delegates to inner ActorHandle).
    ///
    /// This allows EngineHandle to be used anywhere ActorHandle is expected.
    pub fn send(
        &self,
        msg: actor_scheduler::Message<
            crate::api::private::EngineData,
            crate::api::private::EngineControl,
            AppManagement,
        >,
    ) -> Result<(), actor_scheduler::SendError> {
        self.inner.send(msg)
    }
}
