// src/term/emulator/methods.rs

// Note: This file is part of the `emulator` module.
// `TerminalEmulator` struct is defined in `src/term/emulator/mod.rs`
use super::TerminalEmulator; // Bring the struct into scope from the parent module
use crate::{
    glyph::Attributes,
    term::{
        action::EmulatorAction,
        cursor_visibility::CursorVisibility,
        modes::{DecPrivateModes, EraseMode},
        screen::{ScrollHistory, TabClearMode},
        DEFAULT_TAB_INTERVAL,
    },
};

// Crate-level imports for items outside src/term/
impl TerminalEmulator {
    pub(super) fn line_feed(&mut self) {
        log::trace!("perform_line_feed called in ansi_handler");
        self.move_down_one_line_and_dirty(); // Call as method
        if self.dec_modes.linefeed_newline_mode {
            // Check LNM mode
            self.carriage_return(); // Call as method
        }
    }

    pub(super) fn reset(&mut self) -> Option<EmulatorAction> {
        if self.screen.alt_screen_active {
            self.screen.exit_alt_screen();
        }
        let default_attrs = Attributes::default();
        self.cursor_controller.reset();
        self.screen.default_attributes = default_attrs;
        self.erase_in_display(EraseMode::All); // Call as method on emulator
        self.dec_modes = DecPrivateModes::default();
        self.screen.origin_mode = self.dec_modes.origin_mode;
        let (_, h) = self.dimensions();
        self.screen.set_scrolling_region(1, h);
        self.active_charsets = [crate::term::charset::CharacterSet::Ascii; 4];
        self.active_charset_g_level = 0;
        self.screen.clear_tabstops(0, TabClearMode::All);
        let (w, _) = self.dimensions();
        for i in (DEFAULT_TAB_INTERVAL as usize..w).step_by(DEFAULT_TAB_INTERVAL as usize) {
            self.screen.set_tabstop(i);
        }
        self.cursor_wrap_next = false;
        if self.dec_modes.text_cursor_enable_mode {
            return Some(EmulatorAction::SetCursorVisibility(
                CursorVisibility::Visible,
            ));
        }
        None
    }

    pub(super) fn carriage_return(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.carriage_return();
    }

    pub(super) fn save_cursor(&mut self) {
        self.cursor_controller.save_state();
    }

    pub(super) fn restore_cursor(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .restore_state(&self.current_screen_context(), Attributes::default());
        self.screen.default_attributes = self.cursor_controller.attributes();
    }

    pub(super) fn index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        match current_physical_y {
            y if y == screen_ctx.scroll_bot => {
                self.screen.scroll_up(1, ScrollHistory::Save);
            }
            y if y < screen_ctx.height.saturating_sub(1) => {
                self.cursor_controller.move_down(1, &screen_ctx);
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
}
