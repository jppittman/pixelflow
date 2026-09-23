// src/term/emulator/input_handler.rs

use super::{key_translator, FocusState, TerminalEmulator};
use crate::term::{
    action::{EmulatorAction, UserInputAction},
    layout::Zoom,
    snapshot::{Point, SelectionMode},
    ControlEvent, MIN_GRID_DIMENSION,
};
use log::{debug, trace};

const BRACKETED_PASTE_START: &[u8] = b"\x1b[200~";
const BRACKETED_PASTE_END: &[u8] = b"\x1b[201~";
/// Focus reports (DEC mode 1004): the window gained or lost focus.
const FOCUS_IN_REPORT: &[u8] = b"\x1b[I";
const FOCUS_OUT_REPORT: &[u8] = b"\x1b[O";

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
            report_focus(emulator, FOCUS_OUT_REPORT)
        }
        UserInputAction::FocusGained => {
            emulator.focus_state = FocusState::Focused;
            report_focus(emulator, FOCUS_IN_REPORT)
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
        UserInputAction::RequestZoomIn => zoom(emulator, Zoom::In),
        UserInputAction::RequestZoomOut => zoom(emulator, Zoom::Out),
        UserInputAction::RequestZoomReset => zoom(emulator, Zoom::Reset),
        UserInputAction::RequestScrollLineUp => scroll(emulator, 1),
        UserInputAction::RequestScrollLineDown => scroll(emulator, -1),
        UserInputAction::RequestScrollPageUp => scroll(emulator, page(emulator)),
        UserInputAction::RequestScrollPageDown => scroll(emulator, -page(emulator)),
        UserInputAction::RequestScrollToTop => scroll(emulator, i32::MAX),
        UserInputAction::RequestScrollToBottom => scroll(emulator, i32::MIN),
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

/// Resizes the grid to fill a window of the given logical size at the
/// current cell size, and has the PTY follow so the program hears SIGWINCH.
fn fit_to_window(emulator: &mut TerminalEmulator, width_px: u16, height_px: u16) -> EmulatorAction {
    let (cols, rows) = emulator.layout.grid_for_window(width_px, height_px);
    let cols = cols.max(MIN_GRID_DIMENSION);
    let rows = rows.max(MIN_GRID_DIMENSION);
    trace!(
        "TerminalEmulator: fitting {}x{} cells to {}x{} logical px",
        cols,
        rows,
        width_px,
        height_px
    );
    emulator.resize(cols, rows);
    EmulatorAction::ResizePty {
        cols: cols as u16,
        rows: rows as u16,
    }
}

/// Changes the cell size and refits the grid to the window at the new size.
fn zoom(emulator: &mut TerminalEmulator, change: Zoom) -> Option<EmulatorAction> {
    if !emulator.layout.zoom(change) {
        return None;
    }
    match emulator.layout.window_px() {
        Some((width_px, height_px)) => Some(fit_to_window(emulator, width_px, height_px)),
        // No window yet: nothing to refit; the first resize will use the new size.
        None => Some(EmulatorAction::RequestRedraw),
    }
}

/// Tells the program about a focus change, if it asked to hear about them.
fn report_focus(emulator: &TerminalEmulator, report: &[u8]) -> Option<EmulatorAction> {
    match emulator.dec_modes.focus_event_mode {
        true => Some(EmulatorAction::WritePty(report.to_vec())),
        false => None,
    }
}

/// Moves the scrollback viewport; positive is into history.
fn scroll(emulator: &mut TerminalEmulator, lines: i32) -> Option<EmulatorAction> {
    match emulator.scroll_viewport(lines) {
        true => Some(EmulatorAction::RequestRedraw),
        false => None,
    }
}

/// One screenful of lines.
fn page(emulator: &TerminalEmulator) -> i32 {
    let (_, rows) = emulator.dimensions();
    i32::try_from(rows).unwrap_or(i32::MAX)
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

/// Sends pasted text to the program, bracketed when it asked for that
/// (DEC mode 2004) so it can tell a paste from typing.
fn handle_paste_text(
    emulator: &mut TerminalEmulator,
    text_to_paste: &str,
) -> Option<EmulatorAction> {
    let text_bytes = text_to_paste.as_bytes();
    if !emulator.dec_modes.bracketed_paste_mode {
        return Some(EmulatorAction::WritePty(text_bytes.to_vec()));
    }
    let capacity = BRACKETED_PASTE_START.len() + text_bytes.len() + BRACKETED_PASTE_END.len();
    let mut pasted_bytes = Vec::with_capacity(capacity);
    pasted_bytes.extend_from_slice(BRACKETED_PASTE_START);
    pasted_bytes.extend_from_slice(text_bytes);
    pasted_bytes.extend_from_slice(BRACKETED_PASTE_END);
    Some(EmulatorAction::WritePty(pasted_bytes))
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
            emulator.layout.set_window_px(width_px, height_px);
            Some(fit_to_window(emulator, width_px, height_px))
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
    fn it_should_send_pasted_text_to_the_pty_unwrapped_when_bracketed_paste_mode_is_off() {
        let mut emu = create_test_emu_for_input();

        let result = emu.interpret_input(EmulatorInput::User(UserInputAction::PasteText(
            "Hello\nWorld".to_string(),
        )));

        assert_eq!(
            result,
            Some(EmulatorAction::WritePty(b"Hello\nWorld".to_vec()))
        );
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

    fn enable_mode(emu: &mut TerminalEmulator, mode: crate::term::modes::DecModeConstant) {
        use crate::ansi::commands::CsiCommand;
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetModePrivate(mode as u16),
        )));
    }

    #[test]
    fn focus_changes_are_reported_only_when_the_program_asked_for_them() {
        let mut emu = create_test_emu_for_input();
        let focus_in = || EmulatorInput::User(UserInputAction::FocusGained);
        let focus_out = || EmulatorInput::User(UserInputAction::FocusLost);

        assert_eq!(emu.interpret_input(focus_out()), None);

        enable_mode(&mut emu, crate::term::modes::DecModeConstant::FocusEvent);
        assert_eq!(
            emu.interpret_input(focus_out()),
            Some(EmulatorAction::WritePty(b"\x1b[O".to_vec()))
        );
        assert_eq!(
            emu.interpret_input(focus_in()),
            Some(EmulatorAction::WritePty(b"\x1b[I".to_vec()))
        );
    }

    #[test]
    fn scroll_actions_move_the_viewport_through_history_and_back() {
        let mut emu = TerminalEmulator::new(10, 3);
        // Ten lines through a three-row screen leaves seven in history.
        for _ in 0..10 {
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
                crate::ansi::commands::C0Control::LF,
            )));
        }
        let scroll =
            |emu: &mut TerminalEmulator, action| emu.interpret_input(EmulatorInput::User(action));
        let redraw = Some(EmulatorAction::RequestRedraw);

        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollToBottom),
            None
        );
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollLineUp),
            redraw
        );
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollPageUp),
            redraw
        );
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollToTop),
            redraw
        );
        assert_eq!(scroll(&mut emu, UserInputAction::RequestScrollToTop), None);
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollPageDown),
            redraw
        );
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollLineDown),
            redraw
        );
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollToBottom),
            redraw
        );
        assert_eq!(
            scroll(&mut emu, UserInputAction::RequestScrollLineDown),
            None
        );
    }

    #[test]
    fn zoom_scales_the_cells_and_refits_the_grid_to_the_window() {
        let mut emu = create_test_emu_for_input();
        let resize = EmulatorInput::Control(ControlEvent::Resize {
            width_px: 800,
            height_px: 480,
        });
        let before = emu.interpret_input(resize);
        let cell = |emu: &mut TerminalEmulator| {
            let snapshot = emu.get_render_snapshot().expect("snapshot");
            (snapshot.cell_width_px, snapshot.cell_height_px)
        };
        let base = cell(&mut emu);

        let zoomed = emu.interpret_input(EmulatorInput::User(UserInputAction::RequestZoomIn));
        let (width, height) = cell(&mut emu);
        assert!(width > base.0 && height > base.1, "zoom in grows the cell");
        assert_eq!(
            zoomed,
            Some(EmulatorAction::ResizePty {
                cols: (800 / width) as u16,
                rows: (480 / height) as u16,
            }),
            "the grid refits the same window at the new size"
        );

        let reset = emu.interpret_input(EmulatorInput::User(UserInputAction::RequestZoomReset));
        assert_eq!(cell(&mut emu), base);
        assert_eq!(reset, before, "reset is the grid the window had before");
    }

    #[test]
    fn zoom_stops_at_its_limits() {
        let mut emu = create_test_emu_for_input();
        let zoom_out = || EmulatorInput::User(UserInputAction::RequestZoomOut);
        let steps = std::iter::repeat_with(|| emu.interpret_input(zoom_out()))
            .take(100)
            .take_while(Option::is_some)
            .count();
        assert!(steps < 100, "zooming out stops changing the cell size");
        assert_eq!(emu.interpret_input(zoom_out()), None);
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
