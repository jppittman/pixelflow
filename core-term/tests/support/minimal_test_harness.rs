//! Minimal test harness for testing ANSI → Terminal Grid
//!
//! This harness allows you to inject text and ANSI commands directly into a
//! terminal emulator and inspect the grid state.

use core_term::ansi::commands::AnsiCommand;
use core_term::config::Config;
use core_term::term::{EmulatorInput, TerminalEmulator, TerminalSnapshot};

/// Minimal test harness for ANSI→Grid testing
pub struct MinimalTestHarness {
    pub emulator: TerminalEmulator,
    // Stored for completeness; individual tests access emulator directly.
    #[allow(dead_code)]
    pub config: Config,
}

impl MinimalTestHarness {
    /// Create a new test harness with default 80x24 terminal
    pub fn new() -> Self {
        Self::with_dimensions(80, 24)
    }

    /// Create a test harness with custom dimensions
    pub fn with_dimensions(cols: u16, rows: u16) -> Self {
        let config = Config::default();
        let emulator = TerminalEmulator::new(cols as usize, rows as usize);

        Self { emulator, config }
    }

    /// Inject an ANSI command into the terminal emulator
    pub fn inject_ansi(&mut self, cmd: AnsiCommand) {
        let input = EmulatorInput::Ansi(cmd);
        let _actions = self.emulator.interpret_input(input);
        // Ignore actions (they would be PTY writes or redraws)
    }

    /// Print PTY text into the terminal emulator
    pub fn print_text(&mut self, text: &str) {
        self.emulator.print_text(text);
    }

    /// Get the grid snapshot for inspection
    pub fn get_snapshot(&mut self) -> Option<TerminalSnapshot> {
        self.emulator.get_render_snapshot()
    }
}

impl MinimalTestHarness {
    /// Compute a simple checksum of the grid state
    /// This allows detecting if the grid has actually changed
    pub fn compute_grid_checksum(&mut self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let snapshot = match self.get_snapshot() {
            Some(s) => s,
            None => return 0,
        };

        let mut hasher = DefaultHasher::new();

        // Hash all visible characters
        for line in &snapshot.lines {
            for cell in line.cells.iter() {
                cell.display_char().hash(&mut hasher);
            }
        }

        hasher.finish()
    }
}
