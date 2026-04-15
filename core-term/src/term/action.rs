// src/term/action.rs

//! Actions and events for the terminal emulator module.
//!
//! The terminal emulator processes three types of events:
//!
//! 1. **UserInputAction**: Input from the user (keyboard, mouse, clipboard)
//! 2. **ControlEvent**: Control signals from the system (resize, PTY ready, etc.)
//! 3. **EmulatorAction**: Output that the emulator signals to the application
//!
//! # Event Flow
//!
//! ```text
//! User Input / OS Events
//!       ↓
//! [UserInputAction] or [ControlEvent]
//!       ↓
//! TerminalEmulator::process()
//!       ↓
//! [EmulatorAction] (output)
//!       ↓
//! Orchestrator (execute: write PTY, render, set title, etc.)
//! ```
//!
//! # Contract Model
//!
//! Each action type establishes a contract:
//! - **Precondition**: What state must be true before sending
//! - **Action**: What the emulator does with the action
//! - **Postcondition**: What state changes result
//! - **Invariants**: Constraints that must always hold

use crate::keys::{KeySymbol, Modifiers};
use crate::term::cursor_visibility::CursorVisibility;
use serde::{Deserialize, Serialize};

// --- User Input Actions ---

/// Represents user-initiated actions that serve as input to the terminal emulator.
///
/// These are high-level semantic actions derived from raw OS events:
/// - Raw input (keyboard, mouse) is translated to UserInputAction
/// - The emulator processes these and produces EmulatorAction(s)
/// - The orchestrator executes the resulting actions
///
/// # Contract
///
/// **Precondition**: User performed an action (key press, mouse click, etc.)
///
/// **Emulator**: Interprets the action in the context of terminal state
/// - Updates cursor, selection, scroll position
/// - Generates write-to-PTY actions for user input
/// - May queue render requests for visual feedback
///
/// **Postcondition**: Terminal state is updated; EmulatorAction(s) are produced
///
/// This corresponds to `EmulatorInput::User(UserInputAction)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserInputAction {
    /// Keyboard input from the user.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed a key on the keyboard.
    ///
    /// **Emulator**:
    /// - Generates ASCII/UTF-8 sequences for the key and modifiers
    /// - Writes to PTY (generates `WritePty` action)
    /// - May handle special keys (arrows, escape, etc.) with ANSI sequences
    ///
    /// **Postcondition**: PTY receives the user input; cursor may advance
    ///
    /// # Fields
    ///
    /// - `symbol`: The key that was pressed (arrow, letter, function key, etc.)
    /// - `modifiers`: Shift, Ctrl, Alt, Meta key states
    /// - `text`: Composed text character from IME (Some for printable, None for control keys)
    KeyInput {
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<std::borrow::Cow<'static, str>>>,
    },

    /// Initiate copy of selected text.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed Ctrl+Shift+C or other copy command.
    ///
    /// **Emulator**:
    /// - Retrieves the current text selection
    /// - Generates `CopyToClipboard` action
    ///
    /// **Postcondition**: Selected text is in the system clipboard
    InitiateCopy,

    /// Window gained focus.
    ///
    /// # Contract
    ///
    /// **Precondition**: Window became the active/foreground window.
    ///
    /// **Emulator**: May resume animations or pause input buffering.
    FocusGained,

    /// Window lost focus.
    ///
    /// # Contract
    ///
    /// **Precondition**: Window is no longer the active window.
    ///
    /// **Emulator**: May pause animations to save CPU when hidden.
    FocusLost,

    /// Text to paste from the clipboard.
    ///
    /// # Contract
    ///
    /// **Precondition**: Orchestrator fetched text from system clipboard via `RequestClipboardContent`.
    ///
    /// **Emulator**:
    /// - Interprets as user input (writes to PTY)
    /// - May apply bracketed-paste mode (ANSI sequence wrapper) for safety
    ///
    /// **Postcondition**: PTY receives the pasted text
    ///
    /// # Note
    ///
    /// The orchestrator should request clipboard content separately via
    /// `EmulatorAction::RequestClipboardContent` and feed back via this action.
    PasteText(String),

    /// Start a text selection.
    ///
    /// # Contract
    ///
    /// **Precondition**: User clicked the mouse (or began selection).
    ///
    /// **Emulator**:
    /// - Converts pixel coordinates to cell coordinates using Layout
    /// - Records selection start position
    /// - Clears any previous selection
    /// - May queue a `RequestRedraw` to show selection visually
    ///
    /// **Postcondition**: Selection start is set; selection is active
    ///
    /// # Fields
    ///
    /// - `x_px`, `y_px`: Mouse position in logical pixels
    ///
    /// # Coordinate System
    ///
    /// Coordinates are in logical pixels (after DPI scaling). The emulator
    /// converts to cell coordinates (grid position) using its Layout.
    StartSelection { x_px: u16, y_px: u16 },

    /// Extend an ongoing selection.
    ///
    /// # Contract
    ///
    /// **Precondition**: Selection is active (started by `StartSelection`).
    ///
    /// **Emulator**:
    /// - Converts pixel coordinates to cell coordinates
    /// - Updates selection end position
    /// - Queues `RequestRedraw` to show updated selection
    ///
    /// **Postcondition**: Selection is extended to the new position
    ///
    /// # Fields
    ///
    /// - `x_px`, `y_px`: Mouse position in logical pixels (e.g., mouse drag)
    ExtendSelection { x_px: u16, y_px: u16 },

    /// Finalize and potentially clear a selection.
    ///
    /// # Contract
    ///
    /// **Precondition**: User released mouse button or completed selection gesture.
    ///
    /// **Emulator**:
    /// - If selection is non-empty: finalizes it (text is now selectable for copy)
    /// - If selection is empty (click without drag): clears any previous selection
    /// - Queues `RequestRedraw` to refresh selection appearance
    ///
    /// **Postcondition**: Selection is finalized or cleared
    ///
    /// # Note
    ///
    /// The finalized selection text can be copied via `InitiateCopy`.
    ApplySelectionClear,

    /// Request clipboard paste from the orchestrator.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed Ctrl+V or other paste command.
    ///
    /// **Emulator**:
    /// - Generates `RequestClipboardContent` action
    /// - Waits for orchestrator to fetch and return via `PasteText`
    ///
    /// **Postcondition**: Orchestrator will fetch clipboard and send back `PasteText`
    ///
    /// # Note
    ///
    /// This is asynchronous: the emulator sends the request, and the orchestrator
    /// will send back a `PasteText` action when the clipboard content is available.
    RequestClipboardPaste,

    /// Request primary selection paste (X11 specific).
    ///
    /// # Contract
    ///
    /// **Precondition**: User middle-clicked or used primary-paste command (X11).
    ///
    /// **Emulator**: Generates `RequestClipboardContent` action with primary selection hint.
    ///
    /// **Postcondition**: Orchestrator will fetch primary selection and send back `PasteText`
    ///
    /// # Platform Notes
    ///
    /// - X11: Primary selection is the text currently selected (highlighted)
    /// - macOS/Windows: Falls back to clipboard or is unsupported
    RequestPrimaryPaste,

    /// Request application quit.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed Ctrl+D (EOF), Alt+F4, or other quit command.
    ///
    /// **Emulator**: Generates `Quit` action.
    ///
    /// **Postcondition**: Orchestrator will shut down the application
    RequestQuit,

    /// Request zoom in.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed Ctrl+Plus or equivalent zoom command.
    ///
    /// **Emulator**: Queues a render update; font size is increased by application.
    ///
    /// **Postcondition**: Terminal display will be re-rendered at larger size
    RequestZoomIn,

    /// Request zoom out.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed Ctrl+Minus or equivalent zoom command.
    ///
    /// **Emulator**: Queues a render update; font size is decreased.
    RequestZoomOut,

    /// Reset zoom to default level.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed Ctrl+0 or equivalent reset command.
    ///
    /// **Emulator**: Resets font size to default; queues render update.
    RequestZoomReset,

    /// Toggle fullscreen mode.
    ///
    /// # Contract
    ///
    /// **Precondition**: User pressed F11 or equivalent fullscreen command.
    ///
    /// **Emulator**: Signals application to toggle fullscreen mode.
    RequestToggleFullscreen,

    /// Scroll up one line.
    RequestScrollLineUp,

    /// Scroll down one line.
    RequestScrollLineDown,

    /// Scroll up one page.
    RequestScrollPageUp,

    /// Scroll down one page.
    RequestScrollPageDown,

    /// Scroll to the top of the history.
    RequestScrollToTop,

    /// Scroll to the bottom (live input).
    RequestScrollToBottom,
}

// --- Emulator Control Events ---

/// Represents internal control events for the terminal emulator.
///
/// These are system-level signals that trigger internal state changes
/// (not user input). They include render requests, resize notifications,
/// and PTY data availability signals.
///
/// # Contract
///
/// **Sender**: Orchestrator (after receiving OS events or VSync ticks)
///
/// **Emulator**: Processes the control event and may queue EmulatorAction(s)
/// - Updates internal state (grid size, layout, etc.)
/// - Queues rendering work if needed
///
/// **Postcondition**: Emulator state is updated; orchestrator may need to render
#[derive(Debug)]
pub enum ControlEvent {
    /// Request to generate a snapshot for rendering.
    ///
    /// # Contract
    ///
    /// **Sender**: Orchestrator (from VSync, or after user input requiring immediate redraw)
    ///
    /// **Emulator**:
    /// - Generates a snapshot of the current terminal state (grid, styles, cursor)
    /// - Returns the snapshot (manifold) to be rendered
    /// - Efficient: snapshot is cached if the terminal state hasn't changed
    ///
    /// **Postcondition**: Snapshot is available for rendering to display
    ///
    /// # Frequency
    ///
    /// Typically sent at VSync rate (60 Hz on most systems).
    /// May be sent more frequently if user input triggers immediate visual feedback.
    RequestSnapshot,

    /// Terminal display area has been resized.
    ///
    /// # Contract
    ///
    /// **Precondition**: Window was resized (sent by orchestrator after OS resize event).
    ///
    /// **Emulator**:
    /// 1. Calculates new terminal grid (cols, rows) from pixel dimensions using Layout
    /// 2. Reflows or truncates existing content to fit new grid
    /// 3. Updates cursor position if it's now out of bounds
    /// 4. Queues `RequestRedraw` and potentially `ResizePty` actions
    ///
    /// **Postcondition**: Terminal grid is resized; PTY may be resized to match
    ///
    /// # Fields
    ///
    /// - `width_px`: New width in logical pixels
    /// - `height_px`: New height in logical pixels
    ///
    /// # Invariants
    ///
    /// - The new grid size is derived deterministically from pixel dimensions
    /// - Existing content is preserved (within new bounds) or truncated
    /// - Cursor is kept within valid bounds
    Resize { width_px: u16, height_px: u16 },

    /// PTY has data ready to be read.
    ///
    /// # Contract
    ///
    /// **Precondition**: The PTY queue has received data from the shell process.
    ///
    /// **Emulator**:
    /// - Acts as a "doorbell" signal to wake the orchestrator
    /// - Signals that the emulator is ready to process PTY data
    /// - Allows batching of PTY reads for efficiency
    ///
    /// **Postcondition**: Orchestrator will read and process PTY data
    ///
    /// # Design Notes
    ///
    /// This is a **low-priority signal** used for flow control. The PTY I/O thread
    /// signals when data is available, and the main orchestrator thread processes
    /// it asynchronously. This prevents the PTY from blocking the UI.
    ///
    /// Follows a "doorbell" pattern:
    /// - PTY thread enqueues data and signals `PtyDataReady`
    /// - Orchestrator reads the enqueued data when convenient
    /// - Prevents busy-waiting or blocking the main event loop
    PtyDataReady,
}

// --- Emulator Actions (Signaled to Orchestrator) ---

/// Actions that the terminal emulator signals to the orchestrator.
///
/// After processing user input or control events, the emulator generates
/// a sequence of `EmulatorAction`(s) that the orchestrator must execute.
/// These actions represent changes to the outside world (PTY, window, clipboard, etc.)
///
/// # Execution Model
///
/// The orchestrator **must execute actions in order** to maintain causal consistency:
///
/// ```text
/// UserInputAction / ControlEvent
///       ↓
/// TerminalEmulator::process()
///       ↓
/// [EmulatorAction, EmulatorAction, ...]
///       ↓
/// Orchestrator executes each action in sequence
/// ```
///
/// # Contract
///
/// **Emulator** sends actions as a side effect of processing input.
///
/// **Orchestrator** must:
/// 1. Execute each action in the order sent
/// 2. Handle asynchronous operations (e.g., RequestClipboardContent may take time)
/// 3. Report results back to the emulator (e.g., pasted text via UserInputAction::PasteText)
///
/// **Postcondition**: The outside world is updated; emulator awaits next input
#[derive(Debug, Clone, PartialEq)]
pub enum EmulatorAction {
    /// Write bytes to the pseudo-terminal.
    ///
    /// # Contract
    ///
    /// **Emulator**: Generates this action when user input should be sent to the shell.
    ///
    /// **Orchestrator**:
    /// 1. Writes the bytes to the PTY
    /// 2. May batch multiple writes for efficiency
    /// 3. Handles PTY-full (backpressure) gracefully
    ///
    /// **Postcondition**: Shell receives the input
    ///
    /// # Examples
    ///
    /// - User pressed 'a' → `WritePty(b"a")`
    /// - User pressed arrow-up → `WritePty(b"\x1b[A")` (ANSI sequence)
    /// - User pasted "hello\nworld" → `WritePty(b"hello\nworld")`
    ///
    /// # Performance
    ///
    /// The emulator may batch multiple characters into a single WritePty action.
    WritePty(Vec<u8>),

    /// Set the terminal window's title.
    ///
    /// # Contract
    ///
    /// **Emulator**: Generated in response to ANSI escape sequence (OSC 0 or OSC 2).
    ///
    /// **Orchestrator**:
    /// 1. Updates the window title bar
    /// 2. May be no-op if the window doesn't support titles
    ///
    /// **Postcondition**: Window title is updated
    ///
    /// # Example
    ///
    /// - Shell sends: `\x1b]0;My Project\x07` (OSC 0)
    /// - Emulator generates: `SetTitle("My Project")`
    SetTitle(String),

    /// Ring the terminal bell (audible or visual alert).
    ///
    /// # Contract
    ///
    /// **Emulator**: Generated in response to BEL character (0x07) or ESC BEL escape sequence.
    ///
    /// **Orchestrator**:
    /// 1. Plays a system beep or bell sound
    /// 2. May show a visual indicator (flash, notification)
    /// 3. May be silent/no-op if disabled by user preferences
    ///
    /// **Postcondition**: Alert is signaled to the user
    ///
    /// # Usage
    ///
    /// Used by shell programs to signal completion (e.g., `echo -e '\a'`).
    RingBell,

    /// Request a redraw of the terminal display.
    ///
    /// # Contract
    ///
    /// **Emulator**: Queues this whenever terminal state changes visually.
    ///
    /// **Orchestrator**:
    /// 1. Marks the display as needing redraw
    /// 2. May queue a new snapshot request
    /// 3. Typically batches multiple redraws into one render pass
    ///
    /// **Postcondition**: Next frame will reflect the new terminal state
    ///
    /// # Examples
    ///
    /// - User input affects cursor position → RequestRedraw
    /// - PTY data changes cell contents → RequestRedraw
    /// - Selection is modified → RequestRedraw
    RequestRedraw,

    /// Set the visibility of the cursor.
    ///
    /// # Contract
    ///
    /// **Emulator**: Generated in response to ANSI cursor visibility sequences.
    ///
    /// **Orchestrator**:
    /// 1. Updates the cursor visibility state
    /// 2. May control a native OS cursor or a rendered cursor
    /// 3. Affects the next redraw
    ///
    /// **Postcondition**: Cursor visibility is updated
    ///
    /// # Examples
    ///
    /// - ANSI: `\x1b[?25h` (show cursor) → `SetCursorVisibility(Visible)`
    /// - ANSI: `\x1b[?25l` (hide cursor) → `SetCursorVisibility(Hidden)`
    SetCursorVisibility(CursorVisibility),

    /// Copy text to the system clipboard.
    ///
    /// # Contract
    ///
    /// **Emulator**: Generated when:
    /// 1. User selects text and presses copy command (Ctrl+Shift+C)
    /// 2. ANSI escape sequence requests clipboard write (OSC 52)
    ///
    /// **Orchestrator**:
    /// 1. Stores the text in the system clipboard
    /// 2. May update primary selection (X11)
    /// 3. Other applications can now paste this text
    ///
    /// **Postcondition**: Clipboard contains the text
    ///
    /// # Example
    ///
    /// User selects "$ hello world" and presses Ctrl+Shift+C
    /// → `CopyToClipboard("$ hello world")`
    CopyToClipboard(String),

    /// Request the orchestrator to fetch clipboard content.
    ///
    /// # Contract
    ///
    /// **Emulator**: Requests clipboard content asynchronously.
    ///
    /// **Orchestrator**:
    /// 1. Reads the system clipboard
    /// 2. Asynchronously sends the content back via `UserInputAction::PasteText`
    /// 3. May time out or return empty if clipboard is unavailable
    ///
    /// **Postcondition**: Emulator receives the clipboard content via PasteText action
    ///
    /// # Flow
    ///
    /// 1. Emulator sends: `RequestClipboardContent`
    /// 2. Orchestrator fetches clipboard asynchronously
    /// 3. Orchestrator sends: `UserInputAction::PasteText(content)`
    /// 4. Emulator processes the pasted text
    ///
    /// # Note
    ///
    /// This is asynchronous—the emulator doesn't block waiting for the response.
    RequestClipboardContent,

    /// Resize the pseudo-terminal to match the terminal grid.
    ///
    /// # Contract
    ///
    /// **Emulator**: Generated when the terminal grid is resized.
    ///
    /// **Orchestrator**:
    /// 1. Calls `ioctl(PTY_FD, TIOCSWINSZ, ...)` with new size
    /// 2. Shell receives SIGWINCH and updates its layout
    /// 3. Full-screen programs (vim, less, etc.) adapt to new size
    ///
    /// **Postcondition**: PTY size matches terminal grid; SIGWINCH is sent to shell
    ///
    /// # Fields
    ///
    /// - `cols`: Number of columns in the terminal grid
    /// - `rows`: Number of rows in the terminal grid
    ///
    /// # Timing
    ///
    /// Typically sent immediately after a resize, before RequestRedraw.
    ResizePty { cols: u16, rows: u16 },

    /// Request the application to quit.
    ///
    /// # Contract
    ///
    /// **Emulator**: Generated when:
    /// 1. User pressed Ctrl+D (EOF) in an interactive shell
    /// 2. Shell exited cleanly
    /// 3. User explicitly requested quit (Ctrl+D, RequestQuit)
    ///
    /// **Orchestrator**:
    /// 1. Initiates graceful shutdown
    /// 2. May prompt user to save work
    /// 3. Closes the PTY and terminates the shell
    /// 4. Exits the application
    ///
    /// **Postcondition**: Application shuts down
    Quit,
}
