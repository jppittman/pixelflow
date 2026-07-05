// src/term/emulator/screen_ops.rs

use super::TerminalEmulator;
// For default_attributes
use crate::term::modes::EraseMode; // EraseMode is used by erase_in_display, erase_in_line
use crate::term::screen::ScrollHistory;
use std::cmp::min; // For erase_chars

use log::warn;

impl TerminalEmulator {
    pub(super) fn reverse_index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        match current_physical_y {
            y if y == screen_ctx.scroll_top => {
                self.screen.scroll_down(1);
            }
            y if y > 0 => {
                self.cursor_controller.move_up(1);
            }
            _ => {}
        }
        if current_physical_y < self.screen.height {
            self.screen.mark_line_dirty(current_physical_y);
        }
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y && new_physical_y < self.screen.height {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }

    pub(super) fn erase_in_display(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => {
                self.screen
                    .clear_line_segment(cy_phys, cx_phys, screen_ctx.width);
                for y in (cy_phys + 1)..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::ToStart => {
                for y in 0..cy_phys {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
                self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1);
            }
            EraseMode::All => {
                for y in 0..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::Scrollback => {
                self.screen.scrollback.clear();
                // CSI 3J should also clear the screen like CSI 2J
                for y in 0..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::Unknown => warn!("Unknown ED mode used."),
        }
        if mode != EraseMode::Scrollback {
            self.screen.mark_all_dirty();
        }
    }

    pub(super) fn erase_in_line(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => self
                .screen
                .clear_line_segment(cy_phys, cx_phys, screen_ctx.width),
            EraseMode::ToStart => self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1),
            EraseMode::All => self.screen.clear_line_segment(cy_phys, 0, screen_ctx.width),
            EraseMode::Scrollback => {
                warn!("EraseMode::Scrollback is not applicable to EraseInLine (EL).")
            }
            EraseMode::Unknown => warn!("Unknown EL mode used."),
        }
    }

    pub(super) fn erase_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
    }

    pub(super) fn insert_blank_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
    }

    pub(super) fn delete_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
    }

    pub(super) fn insert_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_down(n);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);

            for y_dirty in cy_phys..=original_scroll_bottom {
                if y_dirty < self.screen.height {
                    self.screen.mark_line_dirty(y_dirty);
                }
            }
        }
    }

    pub(super) fn delete_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_up(n, ScrollHistory::Discard);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            for y_dirty in cy_phys..=original_scroll_bottom {
                if y_dirty < self.screen.height {
                    self.screen.mark_line_dirty(y_dirty);
                }
            }
        }
    }

    pub(super) fn scroll_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_up(n, ScrollHistory::Discard);
    }

    pub(super) fn scroll_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_down(n);
    }
    /// Handles the screen operations for moving the cursor down one line,
    /// typically as part of a Line Feed (LF/`\n`) or similar control sequence.
    ///
    /// This function is responsible for:
    /// - Resetting any pending auto-wrap (`cursor_wrap_next`).
    /// - Determining if a scroll is necessary based on the cursor's position,
    ///   the active scrolling region (DECSTBM), and Origin Mode (DECOM).
    /// - Performing the scroll by calling `self.screen.scroll_up_serial(1)` if needed.
    /// - Moving the cursor down one logical line if no scroll occurs and space is available.
    /// - Marking affected screen lines as dirty for rendering.
    ///
    /// The behavior aims to be compatible with `st.c`, especially regarding how
    /// scrolling regions are handled when Origin Mode is off.
    pub(super) fn move_down_one_line_and_dirty(&mut self) {
        self.cursor_wrap_next = false; // Always reset pending wrap on vertical movement

        // Get the current screen context, which includes dimensions, scroll region, and origin mode.
        let screen_ctx = self.current_screen_context();

        // Get the cursor's current logical and physical positions.
        // current_logical_y is relative to scroll_top if origin mode is active.
        // current_physical_y is the absolute row on the screen.
        let (_current_logical_x, current_logical_y) = self.cursor_controller.logical_pos();
        let (_current_physical_x, current_physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        let mut scrolled_this_op = false;

        if screen_ctx.origin_mode_active {
            // --- Origin Mode IS Active (DECOM ON, CSI ?6h) ---
            // Cursor Y is relative to scroll_top (0-indexed within the scrolling region).
            // Physical Y = scroll_top + logical_y.
            // Scrolling occurs if the physical cursor is at the bottom of the scrolling region (scroll_bot).
            if current_physical_y == screen_ctx.scroll_bot {
                self.screen.scroll_up(1, ScrollHistory::Save); // Scrolls region [scroll_top, scroll_bot]
                scrolled_this_op = true;
                // Cursor's logical_y remains at (scroll_bot - scroll_top), effectively staying on the
                // new bottom line of the region (which is now blank).
                log::trace!(
                    "move_down_one_line (origin_mode ON): Scrolled region [{},{}] due to cursor at scroll_bot ({})",
                    screen_ctx.scroll_top, screen_ctx.scroll_bot, current_physical_y
                );
            } else {
                // Not at scroll_bot, try to move logical_y down if there's space within the logical region.
                // max_logical_y_in_logical_region is the last valid logical_y index within the scrolling region.
                let max_logical_y_in_logical_region =
                    screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top);
                if current_logical_y < max_logical_y_in_logical_region {
                    self.cursor_controller.move_down(1, &screen_ctx);
                }
                // If current_logical_y == max_logical_y_in_logical_region, cursor is already at the logical bottom
                // of the region, so no further downward movement within the region is possible without scrolling.
            }
        } else {
            // --- Origin Mode IS OFF (DECOM OFF, CSI ?6l) ---
            // Cursor logical_y is the same as physical_y.
            // Scrolling should occur if the cursor is at the bottom of the active scrolling region (scroll_bot).
            // This aligns with st.c's tnewline behavior: `if (y == term.bot) tscrollup(term.top, 1);`
            match current_physical_y {
                y if y == screen_ctx.scroll_bot => {
                    self.screen.scroll_up(1, ScrollHistory::Save); // scroll_up uses screen.scroll_top and screen.scroll_bot
                    scrolled_this_op = true;
                    log::trace!(
                        "move_down_one_line (origin_mode OFF): Scrolled region [{},{}] due to cursor at scroll_bot ({})",
                        screen_ctx.scroll_top, screen_ctx.scroll_bot, current_physical_y
                    );
                    // Cursor logical_y (and physical_y) effectively stays on this line (screen_ctx.scroll_bot),
                    // which is now blanked due to the scroll. The subsequent carriage_return (if part of LF handling)
                    // will move the cursor to column 0 of this line.
                }
                y if y < screen_ctx.height.saturating_sub(1) => {
                    // Not at scroll_bot, and also not at the very last physical line of the screen, so simply move down.
                    self.cursor_controller.move_down(1, &screen_ctx);
                }
                _ => {}
            }
            // If current_physical_y == screen_ctx.height.saturating_sub(1) AND it was NOT screen_ctx.scroll_bot
            // (i.e., cursor is on physical last line, but this line is *below* a smaller active scroll region),
            // then no scroll of the active region happens due to this logic, and no further move_down occurs
            // because it's already at the physical bottom. This matches st.c's behavior where an LF on
            // the actual screen bottom, when the cursor is below term.bot (the STBM bottom margin),
            // results in no scroll and the cursor effectively attempts to move off-screen (clamped by tmoveto).
        }

        // Mark lines dirty for rendering.
        // The Screen::scroll_up_serial method already marks all lines within its scrolled region as dirty.
        let (_final_physical_x, final_physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Mark the original line the cursor was on as dirty.
        if current_physical_y < screen_ctx.height {
            self.screen.mark_line_dirty(current_physical_y);
            log::trace!(
                "move_down_one_line: Marked old line {} dirty. Cursor moved to new line {}. Scrolled: {}.",
                current_physical_y, final_physical_y, scrolled_this_op
            );
        }

        // Mark the new line the cursor is on as dirty if it changed OR if we scrolled
        // (as the line content changes because it's a new blank line or the cursor moved to it).
        if final_physical_y < screen_ctx.height {
            // Mark new line if cursor physically moved or if a scroll happened (even if cursor y didn't change relative to screen)
            if scrolled_this_op || final_physical_y != current_physical_y {
                self.screen.mark_line_dirty(final_physical_y);
                log::trace!(
                    "move_down_one_line: Marked new line {} dirty.",
                    final_physical_y
                );
            }
        }
    }
}
