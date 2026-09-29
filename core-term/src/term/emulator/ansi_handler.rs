// src/term/emulator/ansi_handler.rs

use super::TerminalEmulator;
use crate::{
    ansi::commands::{
        AnsiCommand, C0Control, CsiCommand, EscCommand, DA1_RESPONSE_VT102, DSR_DEFAULT,
        DSR_REPORT_CURSOR_POSITION, DSR_RESPONSE_OK, DSR_STATUS_OK,
    },
    term::{
        action::EmulatorAction,
        charset::{CharacterSet, G0, G1, G2, G3},
        cursor::CursorShape,
        emulator::FocusState,
        modes::{EraseMode, Mode, ModeAction},
        screen::TabClearMode,
    },
};

use log::{debug, warn};

/// Processes a single ANSI command, modifying the emulator state.
///
/// Returns an `Option<EmulatorAction>` if the command requires an external action (e.g., writing to PTY).
pub(super) fn process_ansi_command(
    emulator: &mut TerminalEmulator,
    command: AnsiCommand,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;

    match command {
        AnsiCommand::C0Control(c0) => handle_c0_control(emulator, c0),
        AnsiCommand::Esc(esc_cmd) => handle_esc_command(emulator, esc_cmd),
        AnsiCommand::Csi(csi) => handle_csi_command(emulator, csi),
        AnsiCommand::Osc(data) => emulator.handle_osc(data),
        _ => {
            debug!(
                "Unhandled ANSI command type in TerminalEmulator: {:?}",
                command
            );
            None
        }
    }
}

fn handle_c0_control(emulator: &mut TerminalEmulator, c0: C0Control) -> Option<EmulatorAction> {
    match c0 {
        C0Control::BS => {
            emulator.backspace();
            None
        }
        C0Control::HT => {
            emulator.horizontal_tab();
            None
        }
        C0Control::LF | C0Control::VT | C0Control::FF => {
            emulator.line_feed();
            None
        }
        C0Control::CR => {
            emulator.carriage_return();
            None
        }
        C0Control::SO => {
            emulator.set_g_level(G1);
            None
        }
        C0Control::SI => {
            emulator.set_g_level(G0);
            None
        }
        C0Control::BEL => Some(EmulatorAction::RingBell),
        _ => {
            debug!("Unhandled C0 control: {:?}", c0);
            None
        }
    }
}

fn handle_esc_command(
    emulator: &mut TerminalEmulator,
    esc_cmd: EscCommand,
) -> Option<EmulatorAction> {
    match esc_cmd {
        EscCommand::SetTabStop => {
            let (cursor_x, _) = emulator.cursor_controller.logical_pos();
            emulator.screen.set_tabstop(cursor_x);
            None
        }
        EscCommand::Index => {
            emulator.index();
            None
        }
        EscCommand::NextLine => {
            emulator.carriage_return();
            emulator.line_feed();
            None
        }
        EscCommand::ReverseIndex => {
            emulator.reverse_index();
            None
        }
        EscCommand::SaveCursor => {
            emulator.save_cursor();
            None
        }
        EscCommand::RestoreCursor => {
            emulator.restore_cursor();
            None
        }
        EscCommand::SelectCharacterSet(intermediate_char, final_char) => {
            let g_idx = match intermediate_char {
                '(' => G0,
                ')' => G1,
                '*' => G2,
                '+' => G3,
                _ => {
                    warn!(
                        "Unsupported G-set designator intermediate: {}",
                        intermediate_char
                    );
                    G0
                }
            };
            emulator.designate_character_set(g_idx, CharacterSet::from_char(final_char));
            None
        }
        EscCommand::ResetToInitialState => {
            emulator.reset();
            None
        }
        _ => {
            debug!("Unhandled Esc command: {:?}", esc_cmd);
            None
        }
    }
}

fn handle_csi_command(emulator: &mut TerminalEmulator, csi: CsiCommand) -> Option<EmulatorAction> {
    match csi {
        CsiCommand::CursorUp(n) => {
            emulator.cursor_up(n.max(1) as usize);
            None
        }
        CsiCommand::CursorDown(n) => {
            emulator.cursor_down(n.max(1) as usize);
            None
        }
        CsiCommand::CursorForward(n) => {
            emulator.cursor_forward(n.max(1) as usize);
            None
        }
        CsiCommand::CursorBackward(n) => {
            emulator.cursor_backward(n.max(1) as usize);
            None
        }
        CsiCommand::CursorNextLine(n) => {
            emulator.cursor_down(n.max(1) as usize);
            emulator.carriage_return();
            None
        }
        CsiCommand::CursorPrevLine(n) => {
            emulator.cursor_up(n.max(1) as usize);
            emulator.carriage_return();
            None
        }
        CsiCommand::CursorCharacterAbsolute(n) => {
            emulator.cursor_to_column(n.saturating_sub(1) as usize);
            None
        }
        CsiCommand::CursorPosition(r, c) => {
            emulator.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize);
            None
        }
        CsiCommand::LinePositionAbsolute(n) => {
            emulator.cursor_to_row(n.saturating_sub(1) as usize);
            None
        }
        CsiCommand::EraseInDisplay(mode_val) => {
            emulator.erase_in_display(EraseMode::from(mode_val));
            None
        }
        CsiCommand::EraseInLine(mode_val) => {
            emulator.erase_in_line(EraseMode::from(mode_val));
            None
        }
        CsiCommand::InsertCharacter(n) => {
            emulator.insert_blank_chars(n.max(1) as usize);
            None
        }
        CsiCommand::DeleteCharacter(n) => {
            emulator.delete_chars(n.max(1) as usize);
            None
        }
        CsiCommand::InsertLine(n) => {
            emulator.insert_lines(n.max(1) as usize);
            None
        }
        CsiCommand::DeleteLine(n) => {
            emulator.delete_lines(n.max(1) as usize);
            None
        }
        CsiCommand::SetGraphicsRendition(attrs_vec) => {
            emulator.handle_sgr_attributes(attrs_vec);
            None
        }
        CsiCommand::SetMode(mode_num) => {
            emulator.handle_set_mode(Mode::Standard(mode_num), ModeAction::Enable)
        }
        CsiCommand::ResetMode(mode_num) => {
            emulator.handle_set_mode(Mode::Standard(mode_num), ModeAction::Disable)
        }
        CsiCommand::SetModePrivate(mode_num) => {
            emulator.handle_set_mode(Mode::DecPrivate(mode_num), ModeAction::Enable)
        }
        CsiCommand::ResetModePrivate(mode_num) => {
            emulator.handle_set_mode(Mode::DecPrivate(mode_num), ModeAction::Disable)
        }
        CsiCommand::DeviceStatusReport(dsr_param) => match dsr_param {
            DSR_DEFAULT | DSR_REPORT_CURSOR_POSITION => {
                let screen_ctx = emulator.current_screen_context();
                let (abs_x, abs_y) = emulator.cursor_controller.physical_screen_pos(&screen_ctx);
                // Bolt Optimization: Avoid String allocation by writing directly to Vec
                use std::io::Write;
                let mut response = Vec::with_capacity(16);
                drop(write!(&mut response, "\x1B[{};{}R", abs_y + 1, abs_x + 1));
                Some(EmulatorAction::WritePty(response))
            }
            DSR_STATUS_OK => Some(EmulatorAction::WritePty(DSR_RESPONSE_OK.to_vec())),
            _ => {
                warn!("Unhandled DSR parameter: {}", dsr_param);
                None
            }
        },
        CsiCommand::EraseCharacter(n) => {
            emulator.erase_chars(n.max(1) as usize);
            None
        }
        CsiCommand::ScrollUp(n) => {
            emulator.scroll_up(n.max(1) as usize);
            None
        }
        CsiCommand::ScrollDown(n) => {
            emulator.scroll_down(n.max(1) as usize);
            None
        }
        CsiCommand::SaveCursor | CsiCommand::SaveCursorAnsi => {
            emulator.save_cursor();
            None
        }
        CsiCommand::RestoreCursor | CsiCommand::RestoreCursorAnsi => {
            emulator.restore_cursor();
            None
        }
        CsiCommand::ClearTabStops(mode_val) => {
            let (cursor_x, _) = emulator.cursor_controller.logical_pos();
            emulator
                .screen
                .clear_tabstops(cursor_x, TabClearMode::from(mode_val));
            None
        }
        CsiCommand::SetScrollingRegion { top, bottom } => {
            emulator
                .screen
                .set_scrolling_region(top as usize, bottom as usize);
            emulator
                .cursor_controller
                .move_to_logical(0, 0, &emulator.current_screen_context());
            None
        }
        CsiCommand::SetCursorStyle { shape } => {
            let focus_state = emulator.focus_state;
            let cursor_shape = CursorShape::from_decscusr_code(shape);
            match focus_state {
                FocusState::Focused => emulator.cursor_controller.cursor.shape = cursor_shape,
                FocusState::Unfocused => {
                    emulator.cursor_controller.cursor.unfocused_shape = cursor_shape
                }
            }
            debug!("Set cursor style to shape: {}", shape);
            None
        }
        CsiCommand::WindowManipulation { ps1, ps2, ps3 } => {
            emulator.handle_window_manipulation(ps1, ps2, ps3)
        }
        CsiCommand::PrimaryDeviceAttributes => {
            Some(EmulatorAction::WritePty(DA1_RESPONSE_VT102.to_vec()))
        }
        CsiCommand::SetTabStop => {
            let (cursor_x, _) = emulator.cursor_controller.logical_pos();
            emulator.screen.set_tabstop(cursor_x);
            None
        }
        CsiCommand::Reset => {
            emulator.reset();
            None
        }
        CsiCommand::Unsupported(intermediates, final_byte_opt) => {
            warn!(
                "TerminalEmulator received CsiCommand::Unsupported: intermediates={:?}, final={:?}. This is usually an error from the parser.",
                intermediates, final_byte_opt
            );
            None
        }
    }
}
