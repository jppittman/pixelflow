// src/term/emulator/mode_handler.rs

use super::TerminalEmulator;
use crate::term::screen::AltScreenClear;
use crate::{
    ansi::commands::Attribute,
    glyph::{AttrFlags, Attributes},
    term::{
        action::EmulatorAction,
        charset::CharacterSet,
        cursor_visibility::CursorVisibility,
        modes::{DecModeConstant, Mode, ModeAction, StandardModeConstant},
    },
};
use log::{trace, warn};

impl TerminalEmulator {
    pub(super) fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
            trace!("Switched to G{} character set mapping.", g_level);
        } else {
            warn!("Attempted to set invalid G-level: {}", g_level);
        }
    }

    pub(super) fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else {
            warn!("Invalid G-set index for designate charset: {}", g_set_index);
        }
    }

    pub(super) fn handle_sgr_attributes(&mut self, attributes_vec: Vec<Attribute>) {
        let mut current_attrs = self.cursor_controller.attributes();
        for attr_cmd in attributes_vec {
            match attr_cmd {
                Attribute::Reset => current_attrs = Attributes::default(),
                Attribute::Bold => current_attrs.flags.insert(AttrFlags::BOLD),
                Attribute::Faint => current_attrs.flags.insert(AttrFlags::FAINT),
                Attribute::Italic => current_attrs.flags.insert(AttrFlags::ITALIC),
                Attribute::Underline => current_attrs.flags.insert(AttrFlags::UNDERLINE),
                Attribute::BlinkSlow | Attribute::BlinkRapid => {
                    current_attrs.flags.insert(AttrFlags::BLINK)
                }
                Attribute::Reverse => current_attrs.flags.insert(AttrFlags::REVERSE),
                Attribute::Conceal => current_attrs.flags.insert(AttrFlags::HIDDEN),
                Attribute::Strikethrough => current_attrs.flags.insert(AttrFlags::STRIKETHROUGH),
                Attribute::UnderlineDouble => current_attrs.flags.insert(AttrFlags::UNDERLINE),
                Attribute::NoBold => {
                    current_attrs.flags.remove(AttrFlags::BOLD);
                    current_attrs.flags.remove(AttrFlags::FAINT);
                }
                Attribute::NoItalic => current_attrs.flags.remove(AttrFlags::ITALIC),
                Attribute::NoUnderline => current_attrs.flags.remove(AttrFlags::UNDERLINE),
                Attribute::NoBlink => current_attrs.flags.remove(AttrFlags::BLINK),
                Attribute::NoReverse => current_attrs.flags.remove(AttrFlags::REVERSE),
                Attribute::NoConceal => current_attrs.flags.remove(AttrFlags::HIDDEN),
                Attribute::NoStrikethrough => current_attrs.flags.remove(AttrFlags::STRIKETHROUGH),
                Attribute::Foreground(color) => {
                    current_attrs.fg = color;
                }
                Attribute::Background(color) => {
                    current_attrs.bg = color;
                }
                Attribute::Overlined => warn!("SGR Overlined not yet visually supported."),
                Attribute::NoOverlined => warn!("SGR NoOverlined not yet visually supported."),
                Attribute::UnderlineColor(color) => {
                    warn!("SGR UnderlineColor not yet fully supported: {:?}", color)
                }
            }
        }
        self.cursor_controller.set_attributes(current_attrs);
        self.screen.default_attributes = current_attrs;
    }

    pub(super) fn handle_set_mode(
        &mut self,
        mode_type: Mode,
        action: ModeAction,
    ) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false;
        let mut action_to_return = None;
        let enable = action == ModeAction::Enable;

        match mode_type {
            Mode::DecPrivate(mode_num) => {
                trace!("Setting DEC Private Mode {} to {:?}", mode_num, action);
                match DecModeConstant::from_u16(mode_num) {
                    Some(DecModeConstant::CursorKeys) => {
                        self.dec_modes.cursor_keys_app_mode = enable
                    }
                    Some(DecModeConstant::Origin) => {
                        self.dec_modes.origin_mode = enable;
                        self.screen.origin_mode = enable;
                        self.cursor_controller.move_to_logical(
                            0,
                            0,
                            &self.current_screen_context(),
                        );
                    }
                    Some(DecModeConstant::TextCursorEnable) => {
                        self.dec_modes.text_cursor_enable_mode = enable;
                        let visibility = CursorVisibility::from(enable);
                        match visibility {
                            CursorVisibility::Visible => self.cursor_controller.show(),
                            CursorVisibility::Hidden => self.cursor_controller.hide(),
                        }
                        action_to_return = Some(EmulatorAction::SetCursorVisibility(visibility));
                    }
                    Some(DecModeConstant::AltScreenBufferClear)
                    | Some(DecModeConstant::AltScreenBufferSaveRestore) => {
                        if !self.dec_modes.allow_alt_screen {
                            // Assuming this flag exists in DecPrivateModes
                            warn!(
                                "Alternate screen disabled by configuration, ignoring mode {}.",
                                mode_num
                            );
                            return None;
                        }
                        let clear_mode = if mode_num == DecModeConstant::AltScreenBufferClear as u16
                            || mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16
                        {
                            AltScreenClear::Clear
                        } else {
                            AltScreenClear::Preserve
                        };

                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                // Assuming this flag exists
                                if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                    self.save_cursor();
                                }
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes();
                                self.screen.enter_alt_screen(clear_mode);
                                self.dec_modes.using_alt_screen = true;
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                self.screen.mark_all_dirty();
                                action_to_return = Some(EmulatorAction::RequestRedraw);
                            }
                        } else if self.dec_modes.using_alt_screen {
                            self.screen.exit_alt_screen();
                            self.dec_modes.using_alt_screen = false;
                            if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                self.restore_cursor();
                            } else {
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes();
                            }
                            self.screen.mark_all_dirty();
                            action_to_return = Some(EmulatorAction::RequestRedraw);
                        }
                    }
                    Some(DecModeConstant::SaveRestoreCursor) => {
                        if enable {
                            self.save_cursor();
                        } else {
                            self.restore_cursor();
                        }
                    }
                    Some(DecModeConstant::BracketedPaste) => {
                        self.dec_modes.bracketed_paste_mode = enable
                    }
                    Some(DecModeConstant::FocusEvent) => self.dec_modes.focus_event_mode = enable,
                    Some(DecModeConstant::MouseX10) => self.dec_modes.mouse_x10_mode = enable,
                    Some(DecModeConstant::MouseVt200) => self.dec_modes.mouse_vt200_mode = enable,
                    Some(DecModeConstant::MouseVt200Highlight) => {
                        self.dec_modes.mouse_vt200_highlight_mode = enable
                    }
                    Some(DecModeConstant::MouseButtonEvent) => {
                        self.dec_modes.mouse_button_event_mode = enable
                    }
                    Some(DecModeConstant::MouseAnyEvent) => {
                        self.dec_modes.mouse_any_event_mode = enable
                    }
                    Some(DecModeConstant::MouseUtf8) => self.dec_modes.mouse_utf8_mode = enable,
                    Some(DecModeConstant::MouseSgr) => self.dec_modes.mouse_sgr_mode = enable,
                    Some(DecModeConstant::MouseUrxvt) => {
                        warn!(
                            "DEC Private Mode {} (MouseUrxvt) set to {} - not fully implemented.",
                            mode_num, enable
                        );
                    }
                    Some(DecModeConstant::MousePixelPosition) => {
                        warn!(
                            "DEC Private Mode {} (MousePixelPosition) set to {} - not fully implemented.",
                            mode_num, enable
                        );
                    }
                    Some(DecModeConstant::Att610CursorBlink) => {
                        self.dec_modes.cursor_blink_mode = enable;
                        warn!(
                            "DEC Private Mode 12 (ATT610 Cursor Blink) set to {}. Visual blink not implemented.",
                            enable
                        );
                    }
                    Some(DecModeConstant::SynchronizedOutput) => {
                        self.dec_modes.synchronized_output = enable;
                    }
                    Some(DecModeConstant::AutoWrapMode) => {
                        self.dec_modes.autowrap_mode = enable;
                        // Re-evaluate cursor_wrap_next based on new autowrap_mode state
                        let (logical_x, _) = self.cursor_controller.logical_pos();
                        let screen_ctx = self.current_screen_context();
                        self.cursor_wrap_next =
                            logical_x >= screen_ctx.width && self.dec_modes.autowrap_mode;
                        trace!("DEC Private Mode 7 (DECAWM Autowrap) set to {}, cursor_wrap_next is now {}", enable, self.cursor_wrap_next);
                    }
                    Some(DecModeConstant::Unknown7727) => {
                        warn!(
                            "DEC Private Mode 7727 set to {} - behavior undefined.",
                            enable
                        );
                    }
                    None => {
                        warn!(
                            "Unknown DEC private mode {} to set/reset: {}",
                            mode_num, enable
                        );
                    }
                }
            }
            Mode::Standard(mode_num) => match StandardModeConstant::from_u16(mode_num) {
                Some(StandardModeConstant::InsertMode) => {
                    self.dec_modes.insert_mode = enable;
                }
                Some(StandardModeConstant::LinefeedNewlineMode) => {
                    self.dec_modes.linefeed_newline_mode = enable;
                }
                None => {
                    warn!(
                        "Standard mode {} set/reset to {} - not fully implemented yet.",
                        mode_num, enable
                    );
                }
            },
        }
        action_to_return
    }
}
