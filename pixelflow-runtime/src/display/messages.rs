pub use crate::api::private::WindowId;
use crate::api::public::{CursorIcon, WindowDescriptor};
use crate::input::{KeySymbol, Modifiers};
use crate::pixel::PlatformPixel;
use pixelflow_graphics::render::Frame;

/// A window with its associated frame buffer.
///
/// This struct bundles the window identity with its rendering surface,
/// ensuring the driver always has full context when presenting frames.
/// The platform assigns the window ID (e.g., from the native window handle),
/// and the frame buffer is owned by the window for the render loop.
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

    /// Show or hide the window.
    ///
    /// # Contract
    ///
    /// **Sender**: Specifies visibility state.
    ///
    /// **Receiver**: Shows or hides the window. Window remains in the taskbar/app switcher.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `visible`: `true` to show, `false` to hide
    SetVisible { id: WindowId, visible: bool },

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
    /// **Sender**: Provides text to copy.
    ///
    /// **Receiver**: Stores the text in the system clipboard. Overwrites previous
    /// clipboard content. Other applications can read it via standard paste.
    ///
    /// # Arguments
    ///
    /// - `text`: UTF-8 text to copy
    Copy { text: String },

    /// Request clipboard paste.
    ///
    /// # Contract
    ///
    /// **Sender**: Requests the current clipboard content.
    ///
    /// **Receiver**: Reads the system clipboard and emits a `DisplayEvent::PasteData`
    /// with the content. If clipboard is empty or unavailable, may emit no event.
    ///
    /// # Example Use
    ///
    /// Terminal emulator receives Ctrl+V, sends `RequestPaste`, and receives the
    /// pasted text via `DisplayEvent::PasteData`.
    RequestPaste,
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

    /// Destroy an existing window.
    ///
    /// # Contract
    ///
    /// **Sender**: Specifies a window ID to close.
    ///
    /// **Receiver**: Closes the window and emits a `DisplayEvent::WindowDestroyed` event.
    /// After this event, any messages for that window ID are invalid.
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
    /// // ... wait for DisplayEvent::WindowDestroyed ...
    /// ```
    Destroy { id: WindowId },
}

/// Events emitted by the display driver
#[derive(Debug)]
pub enum DisplayEvent {
    /// A new window was created by the platform.
    ///
    /// The `Window` contains the platform-assigned ID and initial frame buffer.
    WindowCreated {
        window: Window,
    },
    WindowDestroyed {
        id: WindowId,
    },
    /// Window was resized (by user or programmatically).
    ///
    /// The `Window` contains an appropriately-sized frame buffer.
    Resized {
        window: Window,
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
    ClipboardDataRequested,
    CloseRequested {
        id: WindowId,
    },
}
