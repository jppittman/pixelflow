// src/term/emulator/input_handler.rs

use super::{key_translator, FocusState, TerminalEmulator};
use crate::term::{
    action::{EmulatorAction, UserInputAction},
    snapshot::{Point, SelectionMode},
    ControlEvent, MIN_GRID_DIMENSION,
};
use log::{debug, trace};

const BRACKETED_PASTE_START: &[u8] = b"\x1b[200~";
const BRACKETED_PASTE_END: &[u8] = b"\x1b[201~";

struct KeyInput {
    symbol: pixelflow_runtime::input::KeySymbol,
    modifiers: pixelflow_runtime::input::Modifiers,
    text: Option<std::borrow::Cow<'static, str>>,
}

pub(super) fn process_user_input_action(
    emulator: &mut TerminalEmulator,
    action: UserInputAction,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;

    match action {
        UserInputAction::FocusLost => {
            emulator.focus_state = FocusState::Unfocused;
            None
        }
        UserInputAction::FocusGained => {
            emulator.focus_state = FocusState::Focused;
            None
        }
        UserInputAction::KeyInput {
            symbol,
            modifiers,
            text,
        } => handle_key_input(
            emulator,
            KeyInput {
                symbol,
                modifiers,
                text,
            },
        ),
        UserInputAction::StartSelection { x_px, y_px } => {
            handle_start_selection(emulator, x_px, y_px)
        }
        UserInputAction::ExtendSelection { x_px, y_px } => {
            handle_extend_selection(emulator, x_px, y_px)
        }
        UserInputAction::ApplySelectionClear => {
            emulator.apply_selection_clear();
            Some(EmulatorAction::RequestRedraw)
        }
        UserInputAction::RequestClipboardPaste => {
            debug!(
                "UserInputAction: RequestClipboardPaste received. Requesting clipboard content."
            );
            Some(EmulatorAction::RequestClipboardContent)
        }
        UserInputAction::RequestPrimaryPaste => {
            debug!("UserInputAction: RequestPrimaryPaste received. (Currently not fully implemented, forwarding to RequestClipboardContent)");
            Some(EmulatorAction::RequestClipboardContent)
        }
        UserInputAction::InitiateCopy => handle_initiate_copy(emulator),
        UserInputAction::PasteText(text_to_paste) => handle_paste_text(emulator, &text_to_paste),
        UserInputAction::RequestQuit => Some(EmulatorAction::Quit),
        // Add catch-all for other UserInputAction variants to satisfy exhaustiveness
        _ => {
            log::debug!(
                "Unhandled UserInputAction variant in input_handler: {:?}",
                action
            );
            None
        }
    }
}

fn handle_key_input(emulator: &mut TerminalEmulator, input: KeyInput) -> Option<EmulatorAction> {
    let bytes_to_send = key_translator::translate_key_input(
        input.symbol,
        input.modifiers,
        input.text,
        &emulator.dec_modes,
    );
    if !bytes_to_send.is_empty() {
        Some(EmulatorAction::WritePty(bytes_to_send))
    } else {
        None
    }
}

fn handle_start_selection(
    emulator: &mut TerminalEmulator,
    x_px: u16,
    y_px: u16,
) -> Option<EmulatorAction> {
    if let Some((col, row)) = emulator.layout.pixels_to_cells(x_px, y_px) {
        emulator.start_selection(Point { x: col, y: row }, SelectionMode::Cell);
        Some(EmulatorAction::RequestRedraw)
    } else {
        None
    }
}

fn handle_extend_selection(
    emulator: &mut TerminalEmulator,
    x_px: u16,
    y_px: u16,
) -> Option<EmulatorAction> {
    if let Some((col, row)) = emulator.layout.pixels_to_cells(x_px, y_px) {
        emulator.extend_selection(Point { x: col, y: row });
        Some(EmulatorAction::RequestRedraw)
    } else {
        None
    }
}

fn handle_initiate_copy(emulator: &mut TerminalEmulator) -> Option<EmulatorAction> {
    if let Some(text) = emulator.get_selected_text() {
        if !text.is_empty() {
            return Some(EmulatorAction::CopyToClipboard(text));
        }
    }
    debug!("UserInputAction: InitiateCopy called but no text selected or selection empty.");
    None
}

/// Handles text paste operations, respecting bracketed paste mode.
fn handle_paste_text(
    emulator: &mut TerminalEmulator,
    text_to_paste: &str,
) -> Option<EmulatorAction> {
    if emulator.dec_modes.bracketed_paste_mode {
        log::debug!("InputHandler: Bracketed paste mode ON. Wrapping and sending to PTY.");
        let text_bytes = text_to_paste.as_bytes();
        let capacity = BRACKETED_PASTE_START.len() + text_bytes.len() + BRACKETED_PASTE_END.len();
        let mut pasted_bytes = Vec::with_capacity(capacity);
        pasted_bytes.extend_from_slice(BRACKETED_PASTE_START);
        pasted_bytes.extend_from_slice(text_bytes);
        pasted_bytes.extend_from_slice(BRACKETED_PASTE_END);
        Some(EmulatorAction::WritePty(pasted_bytes))
    } else {
        log::debug!("InputHandler: Bracketed paste mode OFF. Calling emulator.paste_text.");
        for char_val in text_to_paste.chars() {
            emulator.print_char(char_val);
        }
        Some(EmulatorAction::RequestRedraw)
    }
}

pub(super) fn process_control_event(
    emulator: &mut TerminalEmulator,
    event: ControlEvent,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;
    match event {
        ControlEvent::RequestSnapshot => {
            trace!("TerminalEmulator: RequestSnapshot event received.");
            None
        }
        ControlEvent::Resize {
            width_px,
            height_px,
        } => {
            // width_px and height_px are in logical pixels (engine handles scaling)
            // Calculate cols/rows using the emulator's Layout
            let cols = ((width_px as f64 / emulator.layout.cell_width_px.max(1) as f64) as usize)
                .max(MIN_GRID_DIMENSION);
            let rows = ((height_px as f64 / emulator.layout.cell_height_px.max(1) as f64) as usize)
                .max(MIN_GRID_DIMENSION);

            trace!(
                "TerminalEmulator: ControlEvent::Resize to {}x{} cells ({}x{} logical px)",
                cols,
                rows,
                width_px,
                height_px
            );
            emulator.resize(cols, rows);

            // Signal orchestrator to resize the PTY so shell receives SIGWINCH
            Some(EmulatorAction::ResizePty {
                cols: cols as u16,
                rows: rows as u16,
            })
        }
        ControlEvent::PtyDataReady => {
            // Orchestrator wake-up signal, ignored by emulator
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::AnsiCommand;
    use crate::term::emulator::TerminalEmulator;
    use crate::term::EmulatorInput;

    fn create_test_emu_for_input() -> TerminalEmulator {
        TerminalEmulator::new(80, 24)
    }

    #[test]
    fn it_should_wrap_pasted_text_in_bracketed_paste_sequences_when_bracketed_paste_mode_is_on() {
        let mut emu = create_test_emu_for_input();
        // Enable bracketed paste mode via the public message-passing surface (CSI ? 2004 h).
        use crate::ansi::commands::CsiCommand;
        use crate::term::modes::DecModeConstant;
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetModePrivate(DecModeConstant::BracketedPaste as u16),
        )));

        let text_to_paste = "Hello\nWorld".to_string();
        let action = UserInputAction::PasteText(text_to_paste.clone());

        let result = emu.interpret_input(EmulatorInput::User(action));

        let expected_bytes = format!("\x1b[200~{}\x1b[201~", text_to_paste).into_bytes();
        assert_eq!(result, Some(EmulatorAction::WritePty(expected_bytes)));
    }

    #[test]
    fn it_should_print_pasted_text_directly_to_the_grid_when_bracketed_paste_mode_is_off() {
        let mut emu = create_test_emu_for_input();
        assert!(!emu.dec_modes.bracketed_paste_mode);

        let text_to_paste = "Hello\nWorld".to_string();
        let action = UserInputAction::PasteText(text_to_paste.clone());

        let result = emu.interpret_input(EmulatorInput::User(action));
        assert_eq!(result, Some(EmulatorAction::RequestRedraw));

        let snapshot_option = emu.get_render_snapshot();
        let snapshot = snapshot_option.as_ref().expect("Snapshot was None");

        match snapshot.lines[0].cells[0] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'H')
            }
            _ => panic!("Expected H at [0][0]"),
        }
        match snapshot.lines[0].cells[1] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'e')
            }
            _ => panic!("Expected e at [0][1]"),
        }
        match snapshot.lines[0].cells[2] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'l')
            }
            _ => panic!("Expected l at [0][2]"),
        }
        match snapshot.lines[0].cells[3] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'l')
            }
            _ => panic!("Expected l at [0][3]"),
        }
        match snapshot.lines[0].cells[4] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'o')
            }
            _ => panic!("Expected o at [0][4]"),
        }
        match snapshot.lines[1].cells[0] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'W')
            }
            _ => panic!("Expected W at [1][0]"),
        }
    }

    #[test]
    fn it_should_set_focus_state_to_unfocused_when_it_receives_focus_lost() {
        let mut emu = create_test_emu_for_input();
        let result = emu.interpret_input(EmulatorInput::User(UserInputAction::FocusLost));
        assert_eq!(result, None);
        assert!(matches!(emu.focus_state, FocusState::Unfocused));
    }

    #[test]
    fn it_should_set_focus_state_to_focused_when_it_receives_focus_gained() {
        let mut emu = create_test_emu_for_input();
        emu.interpret_input(EmulatorInput::User(UserInputAction::FocusLost));

        let result = emu.interpret_input(EmulatorInput::User(UserInputAction::FocusGained));

        assert_eq!(result, None);
        assert!(matches!(emu.focus_state, FocusState::Focused));
    }

    #[test]
    fn it_should_request_clipboard_content_when_it_receives_request_clipboard_paste() {
        let mut emu = create_test_emu_for_input();
        let result =
            emu.interpret_input(EmulatorInput::User(UserInputAction::RequestClipboardPaste));
        assert_eq!(result, Some(EmulatorAction::RequestClipboardContent));
    }

    #[test]
    fn it_should_request_clipboard_content_when_it_receives_request_primary_paste() {
        let mut emu = create_test_emu_for_input();
        let result = emu.interpret_input(EmulatorInput::User(UserInputAction::RequestPrimaryPaste));
        assert_eq!(result, Some(EmulatorAction::RequestClipboardContent));
    }

    #[test]
    fn it_should_return_quit_when_it_receives_request_quit() {
        let mut emu = create_test_emu_for_input();
        let result = emu.interpret_input(EmulatorInput::User(UserInputAction::RequestQuit));
        assert_eq!(result, Some(EmulatorAction::Quit));
    }

    #[test]
    fn control_event_resize_returns_resize_pty_action() {
        let mut emu = create_test_emu_for_input();

        // Default cell size is 10x16 (from config)
        // Resize to 1000x800 -> 100x50 cells
        let resize_event = ControlEvent::Resize {
            width_px: 1000,
            height_px: 800,
        };

        let result = emu.interpret_input(EmulatorInput::Control(resize_event));

        // Should return ResizePty action with calculated dimensions
        assert_eq!(
            result,
            Some(EmulatorAction::ResizePty {
                cols: 100,
                rows: 50
            }),
            "Resize control event should return ResizePty action"
        );

        // Verify emulator was also resized
        let snapshot = emu.get_render_snapshot().expect("Snapshot");
        assert_eq!(
            snapshot.dimensions,
            (100, 50),
            "Emulator dimensions should match"
        );
    }

    #[test]
    fn it_should_clamp_resize_dimensions_to_the_minimum_grid_size() {
        let mut emu = create_test_emu_for_input();

        // Very small resize (should clamp to MIN_GRID_DIMENSION)
        let resize_event = ControlEvent::Resize {
            width_px: 1,
            height_px: 1,
        };

        let result = emu.interpret_input(EmulatorInput::Control(resize_event));

        // Should clamp to minimum dimensions
        match result {
            Some(EmulatorAction::ResizePty { cols, rows }) => {
                assert!(
                    cols >= MIN_GRID_DIMENSION as u16,
                    "Cols {} should be >= MIN_GRID_DIMENSION {}",
                    cols,
                    MIN_GRID_DIMENSION
                );
                assert!(
                    rows >= MIN_GRID_DIMENSION as u16,
                    "Rows {} should be >= MIN_GRID_DIMENSION {}",
                    rows,
                    MIN_GRID_DIMENSION
                );
            }
            other => panic!("Expected ResizePty action, got {:?}", other),
        }
    }

    #[test]
    fn control_event_request_snapshot_returns_none() {
        let mut emu = create_test_emu_for_input();

        let result = emu.interpret_input(EmulatorInput::Control(ControlEvent::RequestSnapshot));

        assert_eq!(result, None, "RequestSnapshot should return None");
    }

    #[test]
    fn control_event_pty_data_ready_returns_none() {
        let mut emu = create_test_emu_for_input();

        let result = emu.interpret_input(EmulatorInput::Control(ControlEvent::PtyDataReady));

        assert_eq!(result, None, "PtyDataReady should return None");
    }
}
