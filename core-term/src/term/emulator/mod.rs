// src/term/emulator/mod.rs

//! Core terminal emulation logic and state management.
//!
//! This module defines the `TerminalEmulator` struct, which is the heart of the
//! terminal processing. It manages screen state, cursor, character sets, modes,
//! and handles input (both from PTY and user) to update the terminal state
//! and generate actions for the orchestrator.

// Crate-level imports
use crate::{
    glyph::Attributes, // Ensure Glyph and Attributes are imported
    term::{
        action::{
            EmulatorAction,
            // MouseButton, // Unused
            // MouseEventType, // Unused
            // UserInputAction, // Unused
        },
        charset::CharacterSet,
        cursor::{self, CursorController, ScreenContext}, // Import cursor module for its CursorShape
        layout::Layout,
        modes::DecPrivateModes,
        screen::Screen,
        snapshot::{
            CursorRenderState, CursorShape, Point, SelectionMode, SnapshotLine, TerminalSnapshot,
        },
        EmulatorInput,
    },
};

// Logging (optional, but good practice if used)
use log::debug;

mod ansi_handler;
mod char_processor;
mod cursor_handler;
mod input_handler;
mod key_translator;
mod methods;
mod mode_handler;
pub(crate) mod mouse;
mod osc_handler;
mod screen_ops;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FocusState {
    Focused,
    Unfocused,
}

/// The core terminal emulator.
#[derive(Clone)]
pub struct TerminalEmulator {
    pub(super) screen: Screen,
    pub(super) focus_state: FocusState,
    pub(super) cursor_controller: CursorController,
    pub(super) dec_modes: DecPrivateModes,
    pub(super) active_charsets: [CharacterSet; 4],
    pub(super) active_charset_g_level: usize,
    pub(super) cursor_wrap_next: bool,
    /// Layout manager - handles coordinate transformations and geometry
    pub(super) layout: Layout,
    /// Viewport offset for scrollback navigation.
    /// 0 means viewing the live screen, >0 means scrolled back into history.
    viewport_offset: usize,
    /// The `(scrollback_generation, viewport_offset)` pair the previous
    /// snapshot rendered. The visible surface is a pure function of the
    /// grids, the scrollback, and this pair — so when the pair is unchanged,
    /// only the grids' own per-line dirt remains, and a scrolled-back idle
    /// terminal reads as clean instead of redrawing on every timer tick.
    /// When it changes, every line's screen position changed with it, so the
    /// whole snapshot is dirty (this includes returning to offset 0, where no
    /// scrollback-sourced line exists to carry the signal).
    last_snapshot_view: Option<(u64, usize)>,
    /// The `(x, y, shape)` of the cursor the previous snapshot rendered
    /// (`None` when it rendered no cursor). CUF/CUB/CUP and style changes
    /// mutate the cursor controller without touching any line, so line dirt
    /// alone would let the damage gate skip a frame whose cursor moved —
    /// freezing the visible cursor at its old cell.
    last_cursor_mark: Option<(usize, usize, crate::term::snapshot::CursorShape)>,
}

impl TerminalEmulator {
    /// Creates a new `TerminalEmulator`.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let initial_attributes = Attributes::default(); // SGR Reset attributes
                                                        // Screen::new now gets scrollback_limit from CONFIG
        let mut screen = Screen::new(width, height);
        // Ensure the screen's default_attributes are initialized correctly.
        // This is crucial for clearing operations.
        // Note: Screen::new now also initializes default_attributes from CONFIG.
        // This line might be redundant or ensure a specific SGR reset state if different.
        // For now, keeping it to ensure SGR reset state is applied if it differs from CONFIG's default.
        screen.default_attributes = initial_attributes;

        // Create layout manager
        let layout = Layout::new(width, height);

        TerminalEmulator {
            screen,
            cursor_controller: CursorController::new(initial_attributes),
            dec_modes: DecPrivateModes::default(),
            active_charsets: [
                CharacterSet::Ascii, // G0
                CharacterSet::Ascii, // G1
                CharacterSet::Ascii, // G2
                CharacterSet::Ascii, // G3
            ],
            focus_state: FocusState::Focused,
            active_charset_g_level: 0, // Default to G0
            cursor_wrap_next: false,
            layout,
            viewport_offset: 0,
            last_snapshot_view: None,
            last_cursor_mark: None,
        }
    }

    /// Helper to create the current `ScreenContext` for `CursorController`.
    pub(super) fn current_screen_context(&self) -> ScreenContext {
        ScreenContext {
            width: self.screen.width,
            height: self.screen.height,
            scroll_top: self.screen.scroll_top(),
            scroll_bot: self.screen.scroll_bot(),
            origin_mode_active: self.dec_modes.origin_mode,
        }
    }

    pub(super) fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    /// Resizes the terminal display grid.
    pub(super) fn resize(&mut self, cols: usize, rows: usize) {
        self.cursor_wrap_next = false;
        // Screen::resize now gets scrollback_limit from CONFIG
        self.screen.resize(cols, rows);
        // Update layout with new dimensions
        self.layout.resize(cols, rows);
        let (log_x, log_y) = self.cursor_controller.logical_pos();
        self.cursor_controller
            .move_to_logical(log_x, log_y, &self.current_screen_context());
        debug!(
            "Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty by screen.resize().",
            cols, rows
        );
    }

    /// Interprets a given `EmulatorInput` and updates terminal state.
    /// Returns an `Option<EmulatorAction>` if the input results in an action
    /// that needs to be handled externally (e.g., writing to PTY).
    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        match input {
            EmulatorInput::Ansi(command) => {
                // Reset viewport to live screen when receiving PTY output
                self.viewport_offset = 0;
                // Delegate to ANSI command handler
                ansi_handler::process_ansi_command(self, command)
            }
            EmulatorInput::User(action) => {
                // Delegate to user input action handler
                input_handler::process_user_input_action(self, action)
            }
            EmulatorInput::Control(event) => {
                // Delegate to control event handler
                input_handler::process_control_event(self, event)
            }
            EmulatorInput::RawChar(ch) => {
                // Reset viewport to live screen when receiving PTY output
                self.viewport_offset = 0;
                // Delegate to raw character processor
                self.print_char(ch);
                None
            }
        }
    }

    /// Prints a run of PTY text, exactly as one `AnsiCommand::Print` per
    /// character would.
    pub fn print_text(&mut self, run: &str) {
        // Reset viewport to live screen when receiving PTY output
        self.viewport_offset = 0;
        self.print_str(run);
    }

    /// Creates a fresh snapshot of the terminal's current visible state.
    /// Returns None if synchronized_output is active (skip frame).
    ///
    /// Uses Copy-on-Write: clones Arc references to row data (cheap).
    /// The terminal can continue mutating via Arc::make_mut while
    /// the snapshot holds immutable references to the old data.
    ///
    /// When viewport_offset > 0, displays lines from scrollback history.
    pub fn get_render_snapshot(&mut self) -> Option<TerminalSnapshot> {
        if self.dec_modes.synchronized_output {
            return None;
        }

        let (width, height) = (self.screen.width, self.screen.height);
        let active_grid = self.screen.active_grid();
        let scrollback_len = self.screen.scrollback.len();

        // Clamp viewport_offset to available scrollback
        let effective_offset = self.viewport_offset.min(scrollback_len);

        // Did the view itself move — scroll position changed, or scrollback
        // shifted underneath a held position? (The offset is anchored to the
        // scrollback's end, so a push while scrolled back changes what every
        // scrollback-sourced row shows without changing the offset.)
        let view = (self.screen.scrollback_generation, effective_offset);
        let view_changed = self.last_snapshot_view != Some(view);
        self.last_snapshot_view = Some(view);

        // Build lines by cloning Arc references (cheap ref count bump)
        // When scrolled back, top lines come from scrollback, rest from active grid
        let mut lines: Vec<SnapshotLine> = (0..height)
            .map(|y_idx| {
                if y_idx < effective_offset {
                    // This line comes from scrollback
                    // scrollback is ordered oldest first, so we need to index from the end
                    let scrollback_idx = scrollback_len - effective_offset + y_idx;
                    // Scrollback content is immutable once written: a line is
                    // dirty exactly when the view moved, never merely because
                    // it is visible — the latter kept an idle scrolled-back
                    // terminal rendering on every frame request.
                    SnapshotLine::from_arc(
                        self.screen.scrollback[scrollback_idx].clone(),
                        view_changed.into(),
                    )
                } else {
                    // This line comes from active grid
                    let grid_idx = y_idx - effective_offset;
                    if grid_idx < active_grid.len() {
                        let is_dirty =
                            view_changed || self.screen.dirty.get(grid_idx).is_none_or(|&d| d != 0);
                        SnapshotLine::from_arc(active_grid[grid_idx].clone(), is_dirty.into())
                    } else {
                        // Beyond active grid, return empty line
                        SnapshotLine::from_arc(
                            std::sync::Arc::new(vec![
                                crate::glyph::Glyph::Single(
                                    crate::glyph::ContentCell::default_space()
                                );
                                width
                            ]),
                            true.into(),
                        )
                    }
                }
            })
            .collect();

        // Build cursor state - only show cursor when viewing live screen (no scroll offset)
        let cursor_state = if self.dec_modes.text_cursor_enable_mode && effective_offset == 0 {
            let (cursor_x, cursor_y) = self
                .cursor_controller
                .physical_screen_pos(&self.current_screen_context());

            let (cell_char_underneath, cell_attributes_underneath) =
                if cursor_y < height && cursor_x < width {
                    match &active_grid[cursor_y][cursor_x] {
                        crate::glyph::Glyph::Single(cell)
                        | crate::glyph::Glyph::WidePrimary(cell) => (cell.c, cell.attr),
                        crate::glyph::Glyph::WideSpacer => {
                            (crate::glyph::WIDE_CHAR_PLACEHOLDER, Attributes::default())
                        }
                    }
                } else {
                    (' ', Attributes::default())
                };

            let mapped_shape = match self.cursor_controller.cursor.shape {
                cursor::CursorShape::BlinkingBlock | cursor::CursorShape::SteadyBlock => {
                    CursorShape::Block
                }
                cursor::CursorShape::BlinkingUnderline | cursor::CursorShape::SteadyUnderline => {
                    CursorShape::Underline
                }
                cursor::CursorShape::BlinkingBar | cursor::CursorShape::SteadyBar => {
                    CursorShape::Bar
                }
            };

            Some(CursorRenderState {
                x: cursor_x,
                y: cursor_y,
                shape: mapped_shape,
                cell_char_underneath,
                cell_attributes_underneath,
            })
        } else {
            None
        };

        self.screen.mark_all_clean();

        // A cursor that moved or restyled dirties the row it left and the
        // row it entered, whether or not any cell changed. The char/attrs
        // underneath are excluded from the comparison on purpose: they only
        // change with a cell edit, which dirties its line already.
        let cursor_mark = cursor_state.as_ref().map(|c| (c.x, c.y, c.shape));
        if cursor_mark != self.last_cursor_mark {
            for (_, row, _) in [cursor_mark, self.last_cursor_mark].into_iter().flatten() {
                if let Some(line) = lines.get_mut(row) {
                    line.is_dirty = true;
                }
            }
            self.last_cursor_mark = cursor_mark;
        }

        Some(TerminalSnapshot {
            dimensions: (width, height),
            lines,
            cursor_state,
            selection: self.screen.selection,
            cell_width_px: self.layout.cell_width_px,
            cell_height_px: self.layout.cell_height_px,
        })
    }

    /// The cell under a point in the window, in logical points. The cell size
    /// is the emulator's — zoom changes it — so hit-testing asks here.
    #[must_use]
    pub fn cell_at(&self, x_px: u32, y_px: u32) -> (usize, usize) {
        let cell_px = |size: usize| u32::try_from(size.max(1)).unwrap_or(u32::MAX);
        (
            (x_px / cell_px(self.layout.cell_width_px)) as usize,
            (y_px / cell_px(self.layout.cell_height_px)) as usize,
        )
    }

    // --- Scrollback Navigation Methods ---

    /// Scroll the viewport by the given number of lines.
    /// Positive values scroll up (into history), negative scroll down (toward live).
    /// Returns true if the viewport actually changed.
    pub fn scroll_viewport(&mut self, lines: i32) -> bool {
        let old_offset = self.viewport_offset;
        let max_offset = self.screen.scrollback.len();

        let distance = lines.unsigned_abs() as usize;
        self.viewport_offset = match lines > 0 {
            // Scroll up into history
            true => self
                .viewport_offset
                .saturating_add(distance)
                .min(max_offset),
            // Scroll down toward live screen
            false => self.viewport_offset.saturating_sub(distance),
        };

        // Mark all lines dirty if viewport changed
        if self.viewport_offset != old_offset {
            self.screen.mark_all_dirty();
            true
        } else {
            false
        }
    }

    /// Reset viewport to show the live screen (scroll to bottom).
    /// Returns true if the viewport was changed.
    pub fn reset_viewport(&mut self) -> bool {
        if self.viewport_offset > 0 {
            self.viewport_offset = 0;
            self.screen.mark_all_dirty();
            true
        } else {
            false
        }
    }

    /// Returns the current viewport offset (0 = live screen).
    #[must_use]
    pub fn viewport_offset(&self) -> usize {
        self.viewport_offset
    }

    /// Returns the maximum scrollback available.
    #[must_use]
    pub fn scrollback_len(&self) -> usize {
        self.screen.scrollback.len()
    }

    // --- Selection Handling Methods ---

    pub fn start_selection(&mut self, point: Point, mode: SelectionMode) {
        self.screen.start_selection(point, mode);
        debug!(
            "Selection started at ({}, {}) with mode {:?}",
            point.x, point.y, mode
        );
    }

    pub fn extend_selection(&mut self, point: Point) {
        self.screen.update_selection(point);
        debug!("Selection extended to ({}, {})", point.x, point.y);
    }

    pub fn apply_selection_clear(&mut self) {
        if self.screen.selection.is_active {
            if let Some(range) = self.screen.selection.range {
                if range.start == range.end {
                    self.screen.clear_selection();
                    debug!("Selection applied (was a click): cleared.");
                } else {
                    self.screen.selection.is_active = false;
                    debug!(
                        "Selection applied (was a drag): finalized at range {:?}.",
                        range
                    );
                }
            } else {
                self.screen.selection.is_active = false;
                debug!("Selection applied (no range): cleared active state.");
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.screen.clear_selection();
        debug!("Selection cleared.");
    }

    #[must_use]
    pub fn get_selected_text(&self) -> Option<String> {
        self.screen.get_selected_text()
    }

    /// Encode a mouse event as terminal escape sequence bytes, respecting the
    /// currently active mouse tracking and encoding modes.
    ///
    /// Returns `None` if no mouse tracking mode is active or the event kind
    /// is not reported by the current mode.
    ///
    /// Coordinates `col` and `row` are 0-based cell positions.
    #[must_use]
    pub fn encode_mouse_event(&self, params: mouse::MouseEncodingParams) -> Option<Vec<u8>> {
        mouse::encode_mouse_event(&self.dec_modes, params)
    }

    /// Returns true if any mouse tracking mode is active.
    #[must_use]
    pub fn is_mouse_tracking_active(&self) -> bool {
        self.dec_modes.mouse_x10_mode
            || self.dec_modes.mouse_vt200_mode
            || self.dec_modes.mouse_button_event_mode
            || self.dec_modes.mouse_any_event_mode
    }

    /// Returns true if the any-event mouse mode (1003) is active,
    /// which reports all mouse motion regardless of button state.
    #[must_use]
    pub fn reports_all_motion(&self) -> bool {
        self.dec_modes.mouse_any_event_mode
    }

    /// Returns true if button-event or any-event mouse mode is active,
    /// which report mouse motion while a button is held.
    #[must_use]
    pub fn reports_button_motion(&self) -> bool {
        self.dec_modes.mouse_button_event_mode || self.dec_modes.mouse_any_event_mode
    }
}
