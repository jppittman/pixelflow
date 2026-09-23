pub use crate::api::private::WindowId;
use crate::api::public::{CursorIcon, WindowDescriptor};
use crate::input::{KeySymbol, Modifiers, Selection};
use crate::pixel::PlatformPixel;
use pixelflow_graphics::render::Frame;

/// A frame buffer together with everything needed to identify it.
///
/// The driver allocates this and is its resting owner; it is lent out to be drawn into and
/// handed straight back. It moves by value the whole way round, so exactly one actor holds it
/// at any moment and "how many buffers are in flight?" is a property of the type rather than of
/// a counter someone has to maintain.
#[derive(Debug)]
pub struct Window {
    /// Platform-assigned window identifier.
    pub id: WindowId,
    /// Current frame buffer for rendering.
    pub frame: Frame<PlatformPixel>,
    /// Width in physical pixels.
    pub width_px: u32,
    /// Height in physical pixels.
    pub height_px: u32,
    /// Display scale factor (e.g., 2.0 for Retina).
    pub scale: f64,
    /// Which buffer this is. Carried by the buffer rather than tracked alongside it: a returned
    /// buffer has to be able to answer "am I still the one you want?" on its own, and a queue of
    /// generations kept in send order can only answer that while nothing is ever discarded.
    pub generation: Generation,
}

impl Window {
    /// Take the buffer out to be drawn into, keeping what is needed to put it back.
    ///
    /// Rendering has to split this apart — the renderer wants a `Frame` and knows nothing
    /// about windows — and the two halves have to meet again afterwards. Naming both directions
    /// is what stops the metadata from being stashed in the coordinator in between, which is the
    /// field `pending_render` used to be (§7.1 of the migration doc).
    #[must_use]
    pub fn tear(self) -> (Frame<PlatformPixel>, WindowMeta) {
        let Window {
            id,
            frame,
            width_px,
            height_px,
            scale,
            generation,
        } = self;
        (
            frame,
            WindowMeta {
                id,
                width_px,
                height_px,
                scale,
                generation,
            },
        )
    }

    /// The inverse of [`tear`](Self::tear).
    #[must_use]
    pub fn rejoin(frame: Frame<PlatformPixel>, meta: WindowMeta) -> Self {
        Self {
            id: meta.id,
            frame,
            width_px: meta.width_px,
            height_px: meta.height_px,
            scale: meta.scale,
            generation: meta.generation,
        }
    }
}

/// The geometry the platform reported for a window.
///
/// Two extents, deliberately both named: the scene is authored in **point space**, and the
/// buffer is the platform's **sample lattice**, which is denser on a HiDPI display. They are
/// equal on X11 and on a 1:1 Mac, which is exactly what makes conflating them a bug that only
/// appears on hardware the author may not have. Carrying both is what lets the driver allocate
/// the right buffer and the coordinator derive the point→pixel warp from the buffer it is
/// handed, rather than from a scale factor tracked somewhere on the side.
#[derive(Debug, Clone, Copy)]
pub struct Surface {
    pub id: WindowId,
    /// Logical width in points — what the app lays out in.
    pub width_px: u32,
    /// Logical height in points.
    pub height_px: u32,
    /// Frame-buffer width in device pixels.
    pub frame_width: u32,
    /// Frame-buffer height in device pixels.
    pub frame_height: u32,
    /// Display scale factor (e.g. 2.0 for Retina).
    pub scale: f64,
}

/// Which buffer a [`Window`] is, among the ones the driver has handed over.
///
/// A resize does not modify a window in place — the driver allocates a whole new correctly-sized
/// one and sends it while the previous buffer is still circulating through the renderer or the
/// compositor. So more than one can be in flight, and "is this one still current?" is not
/// answerable from the buffer itself: dimensions repeat (resize out and back), and "is a render
/// outstanding?" only says *some* buffer is busy, not which. It is a generation, so it is
/// carried as one — stamped by the driver when it allocates the buffer, and returned untouched
/// however far round the loop the buffer travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Generation(u64);

impl Generation {
    /// Before the driver has produced a window at all. No buffer ever carries this, which is
    /// what makes it the right `Default`: a keeper that has allocated nothing matches nothing.
    pub const NONE: Self = Self(0);

    /// The generation of the next buffer to arrive.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The half of a [`Window`] that isn't the frame buffer — see [`Window::tear`].
///
/// It travels with the render request as `RenderRequest::meta` and comes back untouched in
/// `RenderResponse::meta`, which is what lets the `Window` be reassembled at completion instead
/// of stashed somewhere in the meantime.
#[derive(Debug, Clone, Copy)]
pub struct WindowMeta {
    pub id: WindowId,
    pub width_px: u32,
    pub height_px: u32,
    pub scale: f64,
    /// Which buffer this render was aimed at, so a completion arriving after a resize can be
    /// recognised as aimed at a buffer that no longer exists.
    pub generation: Generation,
}

/// Data messages for the display driver (high throughput, high priority).
///
/// Data messages are used for continuous, high-frequency operations like frame presentation.
/// These bypass burst limiting and always drain first.
///
/// # Message Contract
///
/// When sending a data message, you establish a contract:
/// - **Precondition**: The window identified by `id` must exist
/// - **Action**: The display driver will render the frame to the window
/// - **Postcondition**: The pixels are visible on screen (subject to VSync)
/// - **Blocking**: May block if buffer is full (backpressure)
///
/// # Data Message Types
///
/// - **`Present`**: Render a frame to a window immediately
#[derive(Debug)]
pub enum DisplayData {
    /// Present a window's frame buffer to the screen.
    ///
    /// # Contract
    ///
    /// **Sender**: Provides a `Window` with rendered pixels in its frame buffer.
    ///
    /// **Receiver**: Promises to:
    /// 1. Blit the frame to the native window identified by `window.id`
    /// 2. Return the `Window` via `PresentComplete` for reuse
    /// 3. NOT block the sender (use backpressure if buffer full)
    ///
    /// # Arguments
    ///
    /// - `window`: The window with its rendered frame buffer
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Render into window.frame...
    /// tx.send(Message::Data(DisplayData::Present { window }))?;
    /// // Window is returned via PresentComplete after blitting
    /// ```
    Present { window: Window },
}

/// Control messages for the display driver (configuration and lifecycle).
///
/// Control messages are for state changes and non-rendering operations.
/// These are lower priority than data messages but higher than management.
///
/// # Message Contract
///
/// When sending a control message, you establish a contract:
/// - **Precondition**: Window must exist (where applicable)
/// - **Action**: The display driver executes the requested control operation
/// - **Postcondition**: Window state is updated; operation may not be visible until next frame
/// - **Queueing**: Burst-limited (subject to flow control, may queue)
///
/// # Control Message Types
#[derive(Debug, Clone)]
pub enum DisplayControl {
    /// Set the window title.
    ///
    /// # Contract
    ///
    /// **Sender**: Provides a new title string.
    ///
    /// **Receiver**: Sets the window's title bar text. Platform-dependent behavior:
    /// - macOS: Updates the main window title
    /// - Linux (X11): Updates `_NET_WM_NAME`
    /// - Fallback: Best effort, may be ignored on some platforms
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `title`: New title string (UTF-8)
    ///
    /// # Example
    ///
    /// ```ignore
    /// tx.send(Message::Control(DisplayControl::SetTitle {
    ///     id: window_id,
    ///     title: "My Application - Updated".to_string(),
    /// }))?;
    /// ```
    SetTitle { id: WindowId, title: String },

    /// Resize the window.
    ///
    /// # Contract
    ///
    /// **Sender**: Provides new width and height in logical pixels.
    ///
    /// **Receiver**: Requests the OS to resize the window. The actual resize may:
    /// - Complete immediately or queue for event processing
    /// - Trigger a `Resized` event when it completes
    /// - Be rejected or adjusted by the OS window manager
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `width`: New width in logical pixels
    /// - `height`: New height in logical pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// tx.send(Message::Control(DisplayControl::SetSize {
    ///     id: window_id,
    ///     width: 1920,
    ///     height: 1080,
    /// }))?;
    /// ```
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },

    /// Change the mouse cursor icon.
    ///
    /// # Contract
    ///
    /// **Sender**: Specifies a cursor icon.
    ///
    /// **Receiver**: Sets the cursor icon for the window. Effective immediately when
    /// the mouse is over the window.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `cursor`: Cursor icon (arrow, text, etc.)
    SetCursor { id: WindowId, cursor: CursorIcon },

    /// Show the window.
    ///
    /// # Contract
    ///
    /// **Sender**: Requests to make the window visible.
    ///
    /// **Receiver**: Shows the window. Window remains in the taskbar/app switcher.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    ShowWindow { id: WindowId },

    /// Hide the window.
    ///
    /// # Contract
    ///
    /// **Sender**: Requests to hide the window.
    ///
    /// **Receiver**: Hides the window. Window remains in the taskbar/app switcher.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    HideWindow { id: WindowId },

    /// Request an immediate redraw.
    ///
    /// # Contract
    ///
    /// **Sender**: Requests redraw at next opportunity.
    ///
    /// **Receiver**: Marks the window as needing redraw. May batch with other redraws
    /// or wait for VSync, depending on platform.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    ///
    /// # Example
    ///
    /// Use when the application has updated internal state and needs to redraw
    /// without waiting for the next frame request.
    RequestRedraw { id: WindowId },

    /// Ring the bell (system beep).
    ///
    /// # Contract
    ///
    /// **Sender**: Requests an audible notification.
    ///
    /// **Receiver**: Emits a system beep or plays the system alert sound.
    /// Platform-dependent (some desktops may show a visual indicator instead).
    ///
    /// # Note
    ///
    /// This is used by terminal emulators to signal important events (bell in ANSI).
    Bell,

    /// Copy text to the clipboard.
    ///
    /// # Contract
    ///
    /// **Sender**: Provides text to copy and the selection it goes to.
    ///
    /// **Receiver**: Stores the text in that selection, overwriting what it
    /// held. Other applications can read it via standard paste. A platform
    /// with no primary selection ignores a copy to `Primary`, rather than
    /// overwriting its clipboard every time text is highlighted.
    ///
    /// # Arguments
    ///
    /// - `selection`: Which selection to set
    /// - `text`: UTF-8 text to copy
    Copy { selection: Selection, text: String },

    /// Request clipboard paste.
    ///
    /// # Contract
    ///
    /// **Sender**: Requests the content of a selection.
    ///
    /// **Receiver**: Reads that selection and emits a `DisplayEvent::PasteData`
    /// with the content. If it is empty or unavailable, may emit no event. A
    /// platform with no primary selection reads its clipboard for `Primary`.
    ///
    /// # Example Use
    ///
    /// Terminal emulator receives Ctrl+V, sends `RequestPaste`, and receives the
    /// pasted text via `DisplayEvent::PasteData`.
    RequestPaste { selection: Selection },

    /// Enter full screen, or leave it.
    ToggleFullscreen { id: WindowId },
}

/// Management messages for the display driver (lifecycle operations).
///
/// Management messages are for window creation and destruction—operations that
/// change the set of windows in existence. These have strict ordering requirements.
///
/// # Message Contract
///
/// When sending a management message, you establish a contract:
/// - **Precondition**: Must not create duplicate windows; must destroy existing windows
/// - **Action**: The display driver executes the lifecycle operation
/// - **Postcondition**: Windows are created or destroyed; subsequent messages can use the new state
/// - **Ordering**: Messages must respect window lifecycle (can't present before create)
///
/// # Management Message Types
#[derive(Debug, Clone)]
pub enum DisplayMgmt {
    /// Create a new window.
    ///
    /// # Contract
    ///
    /// **Sender**: Provides a unique window ID and configuration.
    ///
    /// **Receiver**: Creates a new window with the specified properties and emits
    /// a `DisplayEvent::WindowCreated` when the window is ready. Until then, other
    /// messages for that window ID will fail or queue.
    ///
    /// # Arguments
    ///
    /// - `id`: Unique window identifier. Must not already exist.
    /// - `settings`: Window configuration (size, title, etc.)
    ///
    /// # Invariants
    ///
    /// - Window IDs must be globally unique across the application lifetime
    /// - Creating two windows with the same ID is an error (undefined behavior)
    /// - The sender should wait for `WindowCreated` event before sending other messages for this ID
    ///
    /// # Example
    ///
    /// ```ignore
    /// tx.send(Message::Management(DisplayMgmt::Create {
    ///     settings: WindowDescriptor {
    ///         width: 800,
    ///         height: 600,
    ///         title: "My App".to_string(),
    ///     },
    /// }))?;
    /// // ... wait for DisplayEvent::WindowCreated ...
    /// ```
    Create { settings: WindowDescriptor },

    /// Ask for the window buffer, if it is free.
    ///
    /// # Contract
    ///
    /// **Sender**: Has something to draw and is not already holding the buffer.
    ///
    /// **Receiver**: Hands the buffer over if it is resting, and otherwise does nothing at all.
    /// There is deliberately no refusal reply and no queued request: the driver cannot hold a
    /// request without also deciding *when* to reconsider it, which is scheduling policy that
    /// belongs to whoever is drawing. An unanswered request is retried by the next tick.
    ///
    /// Carries nothing, which is why it is safe on a droppable edge — a lost request costs one
    /// frame, where a lost buffer would destroy the only one there is.
    RequestWindow,

    /// Destroy an existing window.
    ///
    /// # Contract
    ///
    /// **Sender**: Specifies a window ID to close.
    ///
    /// **Receiver**: Closes the window. After this, any message for that window ID is invalid.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier. Must have been created.
    ///
    /// # Invariants
    ///
    /// - The window must have been created with `Create`
    /// - Destroying a non-existent window is an error
    /// - After destruction, using the same ID is invalid until recreated
    ///
    /// # Example
    ///
    /// ```ignore
    /// tx.send(Message::Management(DisplayMgmt::Destroy { id: window_id }))?;
    /// ```
    Destroy { id: WindowId },
}

/// Events emitted by the display driver
#[derive(Debug)]
pub enum DisplayEvent {
    /// A new window was created by the platform.
    ///
    /// Geometry only. No frame buffer travels on this edge: the driver allocates and keeps the
    /// buffer, and hands it out on [`DisplayMgmt::RequestWindow`]. What is left is what the
    /// event always actually was — the platform reporting the size it chose.
    WindowCreated {
        surface: Surface,
    },
    /// Window was resized (by user or programmatically).
    ///
    /// Geometry only, as for `WindowCreated`. Nothing downstream has to act on this to keep
    /// rendering correct: the driver has already replaced its buffer by the time this is
    /// observed, and a renderer holding the old one finds out when it hands it back.
    Resized {
        surface: Surface,
    },
    ScaleChanged {
        id: WindowId,
        scale: f64,
    },
    Key {
        id: WindowId,
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },
    MouseButtonPress {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseButtonRelease {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseMove {
        id: WindowId,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseScroll {
        id: WindowId,
        dx: f32,
        dy: f32,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    FocusGained {
        id: WindowId,
    },
    FocusLost {
        id: WindowId,
    },
    PasteData {
        text: String,
    },
    CloseRequested {
        id: WindowId,
    },
}
