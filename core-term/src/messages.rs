//! Message types for App thread ↔ Engine proxy communication.
//!
//! The terminal application runs in a worker thread and communicates with the
//! rendering engine (running in the proxy thread) via messages. This module defines
//! the protocol between these threads.
//!
//! # Architecture
//!
//! ```text
//! Engine Thread (GUI/Display)
//!      ↓
//!  [EngineEvent] → Proxy → [AppEvent] → Worker (Terminal App)
//!      ↑
//!  [Render] ← Proxy ← [Snapshot] ← Worker
//! ```

use crate::ansi::commands::AnsiCommand;
use pixelflow_runtime::{EngineEvent, EngineEventData};

/// Messages sent from proxy (engine thread) to worker (app thread).
///
/// # Event Contract
///
/// **Sender** (Proxy/Engine):
/// - Translates raw `EngineEvent` into application-level `AppEvent`
/// - Sends events in the order they occurred
/// - Sends `Shutdown` when the engine is closing
///
/// **Receiver** (App/Worker):
/// - Processes each event and updates terminal state
/// - May queue a render snapshot in response
/// - Gracefully shuts down when `Shutdown` is received
///
/// # Ordering
///
/// Events are processed in causal order (if event A causes event B on the OS,
/// A is sent before B). However, the app should not assume strict ordering
/// across independent event sources.
#[derive(Debug)]
pub enum AppEvent {
    /// Display event from engine (keyboard, mouse, resize, focus, etc.).
    ///
    /// # Contract
    ///
    /// **Proxy**: Forwards an `EngineEvent` from the rendering engine.
    ///
    /// **App**: Should interpret and respond to the event:
    /// - **KeyDown**: Insert characters into PTY, update terminal state
    /// - **MouseClick/Release**: Initiate/finalize selection
    /// - **MouseMove**: Update selection if dragging
    /// - **Resize**: Recalculate terminal grid, notify PTY of new size
    /// - **FocusGained/Lost**: May pause/resume rendering
    /// - **Paste**: Insert text from clipboard into PTY
    /// - **CloseRequested**: Prepare for shutdown
    ///
    /// # Example
    ///
    /// ```ignore
    /// match app_event {
    ///     AppEvent::Engine(EngineEvent::Control(Resize(w, h))) => {
    ///         terminal.resize(w, h);
    ///         request_render_snapshot();
    ///     },
    ///     AppEvent::Engine(EngineEvent::Management(KeyDown { ... })) => {
    ///         terminal.handle_input(...);
    ///     },
    ///     AppEvent::Shutdown => {
    ///         terminal.cleanup();
    ///         exit();
    ///     },
    /// }
    /// ```
    Engine(EngineEvent),

    /// Shutdown signal.
    ///
    /// # Contract
    ///
    /// **Proxy**: The engine is shutting down cleanly (window closed, app exiting).
    ///
    /// **App**: Should:
    /// 1. Save any application state (scroll history, etc.)
    /// 2. Gracefully terminate the PTY connection
    /// 3. Clean up resources
    /// 4. Exit the worker thread (or signal exit to main thread)
    ///
    /// # Note
    ///
    /// This is a signal to shut down gracefully. The proxy may force-close
    /// if the worker doesn't respond in a reasonable time.
    Shutdown,
}

/// Render request from proxy to worker.
///
/// # Contract
///
/// **Sender** (Proxy):
/// - Requests a render snapshot for the given size
/// - May be sent multiple times if window is resized
/// - Typically sent in response to VSync signals (60 Hz on most systems)
///
/// **Receiver** (App):
/// - Should render the terminal state for the given viewport size
/// - Should return a `Snapshot` via the response channel
/// - Should not block rendering if PTY data is slow (render cached state)
///
/// # Fields
///
/// - `width_px`: Width in logical pixels (may include scale factor)
/// - `height_px`: Height in logical pixels
///
/// # Rendering Semantics
///
/// The app should:
/// 1. Calculate terminal grid (cols, rows) from pixel dimensions
/// 2. Render terminal cells (glyphs + colors) as a manifold
/// 3. Materialize the manifold to pixels
/// 4. Return the snapshot
///
/// The manifold is cached between frames to avoid re-evaluation
/// if the size hasn't changed.
#[derive(Debug, Clone, Copy)]
pub struct RenderRequest {
    /// Width in logical pixels (before DPI scaling)
    pub width_px: u32,
    /// Height in logical pixels (before DPI scaling)
    pub height_px: u32,
}

/// Unified data message for TerminalApp (combines Engine and PTY streams).
#[derive(Debug, Clone)]
pub enum TerminalData {
    /// Data event from the engine (frame request).
    Engine(EngineEventData),
    /// Data from the PTY (ANSI commands).
    Pty(Vec<AnsiCommand>),
    /// The PTY child process exited (EOF on the master FD).
    ChildExited,
}
