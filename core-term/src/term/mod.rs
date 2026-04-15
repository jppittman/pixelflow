// myterm/src/term/mod.rs

//! This module defines the core terminal emulator logic.
//! It acts as a state machine processing inputs and producing actions.

// Sub-modules - existing and new
pub mod cursor;
pub mod cursor_visibility;
pub mod screen;
pub mod unicode;

pub mod action;
pub mod charset;
mod emulator;
pub mod layout;
pub mod modes;
pub mod snapshot; // Add this line to declare the module

// Re-export items for easier use by other modules and within this module
pub use action::{ControlEvent, EmulatorAction, UserInputAction};
pub use charset::{map_to_dec_line_drawing, CharacterSet};
pub use emulator::mouse::{MouseEventKind, MouseEncodingParams};
pub use emulator::TerminalEmulator;
pub use layout::Layout;
pub use snapshot::{
    CursorRenderState, CursorShape, Point, Selection, SelectionMode, SnapshotLine, TerminalSnapshot,
};

// Crate-level imports (adjust paths based on where items are moved)
use crate::ansi::commands::AnsiCommand;

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;

/// Minimum dimension for the terminal grid (width or height) to prevent division by zero or invalid states.
pub const MIN_GRID_DIMENSION: usize = 1;

/// Inputs that the terminal emulator processes.
///
/// This enum encapsulates the different kinds of data or events
/// that the `TerminalEmulator` can receive and act upon. It serves as the
/// primary "instruction set" for the terminal's internal state machine.
#[derive(Debug)]
pub enum EmulatorInput {
    /// An ANSI command parsed from the output of the attached PTY.
    Ansi(AnsiCommand),

    /// A user-initiated event, such as a keypress or mouse action.
    User(UserInputAction),

    /// An internal control event, such as a resize notification from the orchestrator.
    Control(ControlEvent),

    /// A single raw character.
    ///
    /// This is typically used for direct character input that doesn't involve
    /// full ANSI processing.
    RawChar(char),
}

/// Defines the essential public interface for a terminal emulator.
///
/// This trait abstracts the core functionality of a terminal emulator, allowing
/// components like the `AppOrchestrator` and `Renderer` to interact with it
/// without being tied to a specific implementation. It handles state updates
/// from inputs and provides snapshots of its state for rendering.
pub trait TerminalInterface {
    /// Interprets an `EmulatorInput`, updates the terminal's state accordingly,
    /// and returns an optional `EmulatorAction` for external handling.
    ///
    /// # Arguments
    ///
    /// * `input` - The `EmulatorInput` to be processed.
    ///
    /// # Returns
    ///
    /// An `Option<EmulatorAction>` if the input triggers an action that the
    /// caller (e.g., the `AppOrchestrator`) needs to handle, such as writing
    /// output to the PTY. Returns `None` if the input is fully handled internally.
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction>;

    /// Creates a `TerminalSnapshot` of the terminal's current visible state.
    ///
    /// This method provides all necessary information for the `Renderer` to draw the
    /// terminal screen, including cell data, dirty flags, cursor state, and selection.
    ///
    /// It returns `None` if no snapshot is generated, for example, if the terminal
    /// is not in a renderable state.
    fn get_render_snapshot(&mut self) -> Option<TerminalSnapshot>;
}

impl TerminalInterface for TerminalEmulator {
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        self.interpret_input(input)
    }

    fn get_render_snapshot(&mut self) -> Option<TerminalSnapshot> {
        self.get_render_snapshot()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod core_tests;
