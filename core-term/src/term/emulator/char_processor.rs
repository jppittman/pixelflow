// src/term/emulator/char_processor.rs

use super::TerminalEmulator;
use crate::{
    glyph::{ContentCell, Glyph},
    term::{
        charset::{map_to_dec_line_drawing, CharacterSet}, // For map_char_to_active_charset
        unicode::get_char_display_width,
    },
};
use log::{trace, warn};

impl TerminalEmulator {
    /// Attaches a combining (zero-width) character to the most recently written cell.
    ///
    /// The cursor is already past the base character, so the target cell is at
    /// cursor position − 1 on the current row, or the last column of the previous
    /// row when the cursor is at column 0.
    fn attach_combining_char(&mut self, combining: char) {
        let screen_ctx = self.current_screen_context();
        let (cursor_x, cursor_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Determine the position of the base character cell.
        let (base_x, base_y) = match (cursor_x > 0, cursor_y > 0) {
            (true, _) => (cursor_x - 1, cursor_y),
            (false, true) => (screen_ctx.width.saturating_sub(1), cursor_y - 1),
            (false, false) => {
                // Cursor is at (0, 0) — no previous cell to attach to.
                trace!(
                    "attach_combining_char: no previous cell at origin, discarding '{}'",
                    combining
                );
                return;
            }
        };

        if base_y >= self.screen.height || base_x >= self.screen.width {
            return;
        }

        let updated = match self.screen.active_grid()[base_y][base_x] {
            Glyph::Single(mut cc) => {
                cc.combining = Some(combining);
                Glyph::Single(cc)
            }
            Glyph::WidePrimary(mut cc) => {
                cc.combining = Some(combining);
                Glyph::WidePrimary(cc)
            }
            Glyph::WideSpacer => {
                // For a wide-char spacer, walk back one more column to reach WidePrimary.
                if base_x > 0 {
                    if let Glyph::WidePrimary(mut cc) =
                        self.screen.active_grid()[base_y][base_x - 1]
                    {
                        cc.combining = Some(combining);
                        self.screen
                            .set_glyph(base_x - 1, base_y, Glyph::WidePrimary(cc));
                        self.screen.mark_line_dirty(base_y);
                    }
                }
                return;
            }
        };

        self.screen.set_glyph(base_x, base_y, updated);
        self.screen.mark_line_dirty(base_y);
    }

    /// Maps a character to its equivalent in the currently active G0/G1/G2/G3 character set.
    // This is a helper for print_char, so it can be private to this impl block.
    #[inline]
    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => {
                if ch == '#' {
                    '£'
                } else {
                    ch
                }
            }
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
        }
    }

    /// Prints a single character to the terminal at the current cursor position.
    /// Handles character width, line wrapping, and updates cursor position.
    // Called from ansi_handler.rs, so pub(super) or pub.
    // pub(super) is fine as ansi_handler is a sibling module.
    pub(super) fn print_char(&mut self, ch: char) {
        if ch == '\n' {
            self.carriage_return();
            self.move_down_one_line_and_dirty();
            return;
        }

        // Map character to the active G0/G1/G2/G3 character set.
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print);

        // Zero-width characters (combining marks, diacritics) do not advance the cursor.
        // Attach them to the base character in the most recently written cell.
        if char_width == 0 {
            trace!(
                "print_char: combining char '{}' — attaching to previous cell",
                ch_to_print
            );
            self.attach_combining_char(ch_to_print);
            return;
        }

        let mut screen_ctx = self.current_screen_context();

        // Handle line wrap if cursor_wrap_next was set by the previous character.
        // This flag indicates that the cursor is at the end of the line and the next
        // character should wrap to the beginning of the next line.
        if self.cursor_wrap_next {
            self.carriage_return();
            self.move_down_one_line_and_dirty();
            // move_down_one_line_and_dirty also resets self.cursor_wrap_next to false.
            screen_ctx = self.current_screen_context();
        }

        // Get current physical cursor position for placing the glyph.
        // This position is now correctly at the start of the line if a wrap just occurred.
        let (mut physical_x, mut physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Check if the character (considering its width) would exceed the line width
        // from the current physical_x. This handles cases where the character is wider
        // than the remaining space on the line, even if cursor_wrap_next was false initially.
        if physical_x + char_width > screen_ctx.width {
            // If a wide char (width 2) is at the very last column (e.g. col 79 of 80), it can't fit.
            // Standard behavior: print a space in the last cell, then wrap.
            if char_width == 2 && physical_x == screen_ctx.width.saturating_sub(1) {
                let fill_glyph = Glyph::Single(ContentCell::default_space());
                if physical_y < self.screen.height {
                    // Bounds check
                    self.screen.set_glyph(physical_x, physical_y, fill_glyph);
                    self.screen.mark_line_dirty(physical_y);
                }
            }

            // Perform wrap: CR then effectively LF.
            self.carriage_return();
            self.move_down_one_line_and_dirty();
            screen_ctx = self.current_screen_context();
            // Get new physical cursor position after this wrap.
            (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        }
        // Place the character glyph on the screen.
        let glyph_attrs = self.cursor_controller.attributes();
        if physical_y < self.screen.height {
            // Ensure y is within bounds before writing

            // Before setting the new glyph, check if the old glyph was a WIDE_CHAR_PRIMARY.
            // If so, and the new char is not wide or is different, clear the old spacer.
            if physical_y < screen_ctx.height && physical_x < screen_ctx.width {
                // Bounds check for old_glyph_at_pos
                let old_glyph_at_pos = self.screen.active_grid()[physical_y][physical_x];
                if matches!(old_glyph_at_pos, Glyph::WidePrimary(_)) {
                    // If the new char isn't wide, or if it is but we're overwriting,
                    // we must clear the old spacer.
                    // We check old_glyph_at_pos.c by accessing the ContentCell inside WidePrimary variant if needed,
                    // but since we are changing to new Glyph enums, direct comparison of ch_to_print might be different.
                    // The condition `char_width != 2` should cover cases where new char is not wide.
                    // If new char is wide, `Glyph::WidePrimary(ContentCell { c: ch_to_print, .. })` will overwrite.
                    // So, we only need to clear the spacer if the old one was WidePrimary and new is not (or different, implicitly handled by overwrite).
                    let clear_spacer = match old_glyph_at_pos {
                        Glyph::WidePrimary(cell) => char_width != 2 || cell.c != ch_to_print,
                        _ => false, // Not a WidePrimary, so no spacer to clear based on this logic.
                    };

                    if clear_spacer && physical_x + 1 < screen_ctx.width {
                        // Using default_space() for simplicity, assuming default attributes are desired.
                        // If specific default_attrs are needed, they should be defined as per original code.
                        let default_glyph = Glyph::Single(ContentCell::default_space());
                        self.screen
                            .set_glyph(physical_x + 1, physical_y, default_glyph);
                        // Line will be marked dirty anyway by the new char.
                    }
                }
            }

            // If it's a wide character, place a placeholder and set flags.
            if char_width == 2 {
                self.screen.set_glyph(
                    physical_x,
                    physical_y,
                    Glyph::WidePrimary(ContentCell {
                        c: ch_to_print,
                        attr: glyph_attrs,
                        combining: None,
                    }),
                );

                if physical_x + 1 < screen_ctx.width {
                    self.screen
                        .set_glyph(physical_x + 1, physical_y, Glyph::WideSpacer);
                    // Line is already marked dirty from the primary character.
                } else {
                    // This case implies a wide char was printed at the exact last column.
                    // The WidePrimary is set, but no spacer is placed.
                    // The cursor advancement logic below will handle cursor_wrap_next.
                    trace!(
                        "Wide char placeholder for '{}' at ({},{}) could not be placed as it's at the edge of screen (width {}). Only primary part written.",
                        ch_to_print, physical_x, physical_y, screen_ctx.width
                    );
                }
            } else {
                // char_width == 1
                self.screen.set_glyph(
                    physical_x,
                    physical_y,
                    Glyph::Single(ContentCell {
                        c: ch_to_print,
                        attr: glyph_attrs,
                        combining: None,
                    }),
                );
            }
            self.screen.mark_line_dirty(physical_y); // Mark line dirty via screen method.
        } else {
            warn!(
                "print_char: Attempted to print at physical_y {} out of bounds (height {})",
                physical_y, self.screen.height
            );
        }

        // Advance the logical cursor position by the character's width.
        // self.cursor_controller.move_right uses the current logical position and advances it.
        // The logical position should be correct after any wrapping.
        self.cursor_controller.move_right(char_width, &screen_ctx);

        // Check if the new logical cursor position requires a wrap on the *next* character.
        let (final_logical_x, _) = self.cursor_controller.logical_pos();
        // Set cursor_wrap_next if the cursor is exactly at or beyond the width AND autowrap is on.
        self.cursor_wrap_next = final_logical_x >= screen_ctx.width && self.dec_modes.autowrap_mode;
        if self.cursor_wrap_next {
            // This logging might be too verbose for normal operation
            // Consider using trace level or removing if not essential for debugging.
            log::trace!("cursor_wrap_next set to true. final_logical_x: {}, screen_ctx.width: {}, autowrap: {}",
                      final_logical_x, screen_ctx.width, self.dec_modes.autowrap_mode);
        }
    }
}
