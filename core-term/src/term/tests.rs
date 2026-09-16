// src/term/tests.rs

use crate::ansi::commands::{Attribute, C0Control, CsiCommand};
use crate::color::{Color, NamedColor};
use crate::glyph::{Attributes, Glyph};
use crate::keys::{KeySymbol, Modifiers};
// use crate::term::action::{MouseButton, MouseEventType}; // Not used directly in this file anymore
use crate::term::{
    modes::{DecModeConstant, StandardModeConstant}, // For DECTCEM test
    AnsiCommand,
    ControlEvent,
    EmulatorAction,
    EmulatorInput,
    MouseEncodingParams,
    MouseEventKind,
    // Imports for new tests:
    Point,
    Selection,
    SelectionMode,
    TerminalEmulator,
    TerminalSnapshot,
    UserInputAction,
};
use pixelflow_runtime::input::MouseButton; // For mouse input

// Default scrollback for tests, can be adjusted.
// const TEST_SCROLLBACK_LIMIT: usize = 100;

fn create_test_emulator(cols: usize, rows: usize) -> TerminalEmulator {
    TerminalEmulator::new(cols, rows)
}

/// Helper to create a ControlEvent::Resize with logical pixel dimensions based on cols/rows.
/// Uses default cell dimensions from CONFIG (10x16 px).
fn resize_event(cols: usize, rows: usize) -> ControlEvent {
    ControlEvent::Resize {
        width_px: (cols * 10) as u16,
        height_px: (rows * 16) as u16,
    }
}

/// Helper to create a UserInputAction::StartSelection with logical pixel coordinates from cell coords.
/// Uses default cell dimensions from CONFIG (10x16 px).
fn start_selection_at(col: usize, row: usize) -> UserInputAction {
    UserInputAction::StartSelection {
        x_px: (col * 10) as u16,
        y_px: (row * 16) as u16,
    }
}

/// Helper to create a UserInputAction::ExtendSelection with logical pixel coordinates from cell coords.
/// Uses default cell dimensions from CONFIG (10x16 px).
fn extend_selection_to(col: usize, row: usize) -> UserInputAction {
    UserInputAction::ExtendSelection {
        x_px: (col * 10) as u16,
        y_px: (row * 16) as u16,
    }
}

// Helper to get a Glyph from the snapshot.
fn get_glyph_from_snapshot(snapshot: &TerminalSnapshot, row: usize, col: usize) -> Option<Glyph> {
    if row < snapshot.dimensions.1 && col < snapshot.dimensions.0 {
        snapshot
            .lines
            .get(row)
            .and_then(|line| line.cells.get(col).cloned())
    } else {
        None
    }
}

// asserts screen content and cursor position
#[allow(clippy::panic_in_result_fn)] // Allow panic in this test helper
fn assert_screen_state(
    snapshot: &TerminalSnapshot,
    expected_screen: &[&str],
    expected_cursor_pos: Option<(usize, usize)>, // (row, col) for physical cursor
) {
    assert_eq!(
        snapshot.dimensions.1, // rows
        expected_screen.len(),
        "Snapshot row count mismatch. Expected {}, got {}. Snapshot lines: {:?}",
        expected_screen.len(),
        snapshot.dimensions.1,
        snapshot.lines
    );
    if !expected_screen.is_empty() {
        // Ensure the first expected line (if any) isn't wider than the snapshot's column count.
        // This helps catch issues where the test itself might define an impossible expected screen.
        assert!(
            snapshot.dimensions.0
                >= expected_screen[0]
                    .chars()
                    .map(|c| crate::term::unicode::get_char_display_width(c).max(1))
                    .sum::<usize>(),
            "Snapshot col count ({}) is less than the character-width-aware width of the first expected row ({}). Expected screen: {:?}",
            snapshot.dimensions.0,
            expected_screen[0]
                .chars()
                .map(|c| crate::term::unicode::get_char_display_width(c).max(1))
                .sum::<usize>(),
            expected_screen[0]
        );
    }

    for r in 0..snapshot.dimensions.1 {
        let expected_row_str = expected_screen.get(r).unwrap_or_else(|| {
            panic!("Expected screen data missing for row {}", r);
        });
        let mut s_col = 0; // current column in the snapshot being checked
        let mut expected_chars_iter = expected_row_str.chars().peekable();

        while let Some(expected_char) = expected_chars_iter.next() {
            if s_col >= snapshot.dimensions.0 {
                // This condition means we've run out of snapshot columns to check,
                // but there are still expected characters. This indicates a mismatch
                // if the expected string (considering char widths) is wider than the terminal.
                let remaining_expected: String = expected_chars_iter.collect();
                panic!(
                    "Snapshot row {} (len {}) is shorter than expected string '{}'. Expected char '{}' (and potentially '{}') at snapshot col {} would exceed width.",
                    r,
                    snapshot.dimensions.0,
                    expected_row_str,
                    expected_char,
                    remaining_expected,
                    s_col
                );
            }

            let glyph_wrapper = get_glyph_from_snapshot(snapshot, r, s_col).unwrap_or_else(|| {
                panic!(
                    "Glyph ({}, {}) not found in snapshot. Expected char: '{}'",
                    r, s_col, expected_char
                )
            });

            let (cell_char, _cell_attrs) = match glyph_wrapper {
                // cell_attrs prefixed with _
                Glyph::Single(cell) => (cell.c, cell.attr),
                Glyph::WidePrimary(cell) => (cell.c, cell.attr),
                Glyph::WideSpacer => (crate::glyph::WIDE_CHAR_PLACEHOLDER, Attributes::default()),
            };

            assert_eq!(
                cell_char,
                expected_char,
                "Char mismatch at (row {}, snapshot_col {}). Expected '{}', got '{}'. Full expected row: '{}', Full actual row: '{:?}'",
                r,
                s_col,
                expected_char,
                cell_char,
                expected_row_str,
                snapshot.lines.get(r).map(|l| &l.cells)
            );

            let char_width = crate::term::unicode::get_char_display_width(expected_char).max(1);

            // If it's a wide char, check the spacer cell
            if char_width == 2 {
                assert!(
                    matches!(glyph_wrapper, Glyph::WidePrimary(_)),
                    "Expected WidePrimary for char '{}' at ({}, {})",
                    expected_char,
                    r,
                    s_col
                );
                if s_col + 1 < snapshot.dimensions.0 {
                    let spacer_glyph_wrapper = get_glyph_from_snapshot(snapshot, r, s_col + 1)
                        .unwrap_or_else(|| {
                            panic!(
                                "Wide char spacer glyph ({}, {}) not found. Primary char: '{}'",
                                r,
                                s_col + 1,
                                expected_char
                            )
                        });
                    assert!(
                        matches!(spacer_glyph_wrapper, Glyph::WideSpacer),
                        "Expected WideSpacer for char '{}' at ({}, {})",
                        expected_char,
                        r,
                        s_col + 1
                    );
                }
            }
            s_col += char_width;
        }

        // Check that remaining cells in the row are spaces (default fill)
        for c_fill in s_col..snapshot.dimensions.0 {
            let glyph_wrapper = get_glyph_from_snapshot(snapshot, r, c_fill)
                .unwrap_or_else(|| panic!("Glyph ({}, {}) not found for fill check", r, c_fill));
            let cell_char = match glyph_wrapper {
                Glyph::Single(cell) => cell.c,
                Glyph::WidePrimary(cell) => cell.c, // Should not happen for fill (wide chars imply non-space)
                Glyph::WideSpacer => crate::glyph::WIDE_CHAR_PLACEHOLDER, // Should not happen for fill
            };
            assert_eq!(
                cell_char, ' ',
                "Expected empty char ' ' for fill at ({}, {}), got '{}'",
                r, c_fill, cell_char
            );
        }
    }

    match expected_cursor_pos {
        Some((r_expected, c_expected)) => {
            let cursor_state = snapshot.cursor_state.as_ref().unwrap_or_else(|| {
                    panic!(
                        "Expected cursor to be visible, but cursor_state is None. Expected pos: ({},{})",
                        r_expected, c_expected
                    );
                });
            assert_eq!(cursor_state.y, r_expected, "Cursor row mismatch");
            assert_eq!(cursor_state.x, c_expected, "Cursor col mismatch");
        }
        None => {
            assert!(
                snapshot.cursor_state.is_none(),
                "Expected cursor to be hidden, but cursor_state is Some"
            );
        }
    }
}

#[test]
fn it_should_print_characters_left_to_right_and_advance_the_cursor() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // Cursor is at (0,1) *after* printing 'A' at (0,0)
    assert_screen_state(&snapshot, &["A         "], Some((0, 1)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot_b = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_b, &["AB        "], Some((0, 2)));
}

#[test]
fn it_should_move_to_column_zero_of_the_next_line_when_lnm_is_set_and_lf_is_received() {
    let mut term = create_test_emulator(10, 2);
    // Enable Linefeed/Newline Mode (LNM)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetMode(
        StandardModeConstant::LinefeedNewlineMode as u16,
    ))));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // With LNM ON, LF moves to next line AND performs carriage return. 'B' prints at (1,0), cursor moves to (1,1)
    // Testing behavior: cursor should be at column 0 after LF (that's what LNM does)
    assert_screen_state(&snapshot, &["A         ", "B         "], Some((1, 1)));
}

#[test]
fn it_should_return_cursor_to_column_zero_and_overwrite_on_carriage_return() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // "ABC", CR -> (0,0), "D" prints at (0,0) over 'A', cursor moves to (0,1)
    assert_screen_state(&snapshot, &["DBC       "], Some((0, 1)));
}

// --- Helpers for Selection Integration Tests ---
// MouseEventType is not defined in term::action, using a placeholder for now
fn send_mouse_input(
    emu: &mut TerminalEmulator,
    action: UserInputAction,
    _button: MouseButton, // Marked as unused for now
) -> Option<EmulatorAction> {
    emu.interpret_input(EmulatorInput::User(action))
}

fn fill_emulator_screen(emu: &mut TerminalEmulator, text_lines: Vec<String>) {
    for (r, line) in text_lines.iter().enumerate() {
        if r > 0 {
            // If not the first line, and the line isn't empty (to avoid CR LF for empty lines if that's not desired)
            // However, fill_emulator_screen is usually used to set up a known state,
            // so CR+LF is generally the safer bet to ensure cursor is at start of next line.
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
        }
        for char_val in line.chars() {
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(char_val)));
        }
    }
}

// --- End of Helpers for Selection Integration Tests ---

// --- Basic Selection Flow Tests ---
#[test]
fn mouse_press_starts_selection() {
    let mut emu = create_test_emulator(10, 5);
    let action = send_mouse_input(&mut emu, start_selection_at(1, 1), MouseButton::Left);

    let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
    assert!(
        snapshot.selection.is_active,
        "Selection should be active after left press."
    );
    assert_eq!(
        snapshot.selection.range.map(|r| r.start),
        Some(Point { x: 1, y: 1 }),
        "Selection start point mismatch."
    );
    assert_eq!(
        snapshot.selection.range.map(|r| r.end),
        Some(Point { x: 1, y: 1 }),
        "Selection end point should be same as start initially."
    );
    assert_eq!(
        snapshot.selection.mode,
        SelectionMode::Cell,
        "Default selection mode should be Cell."
    );
    assert_eq!(
        action,
        Some(EmulatorAction::RequestRedraw),
        "Action should be RequestRedraw."
    );
}

#[test]
fn mouse_drag_updates_selection() {
    let mut emu = create_test_emulator(10, 5);
    send_mouse_input(&mut emu, start_selection_at(1, 1), MouseButton::Left);
    let action = send_mouse_input(&mut emu, extend_selection_to(5, 2), MouseButton::Left);

    let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
    assert!(
        snapshot.selection.is_active,
        "Selection should remain active during drag."
    );
    assert_eq!(
        snapshot.selection.range.map(|r| r.start),
        Some(Point { x: 1, y: 1 }),
        "Selection start point should not change during drag."
    );
    assert_eq!(
        snapshot.selection.range.map(|r| r.end),
        Some(Point { x: 5, y: 2 }),
        "Selection end point should update to drag position."
    );
    assert_eq!(
        action,
        Some(EmulatorAction::RequestRedraw),
        "Action should be RequestRedraw."
    );
}

#[test]
fn mouse_release_ends_selection_activity() {
    let mut emu = create_test_emulator(10, 5);
    send_mouse_input(&mut emu, start_selection_at(1, 1), MouseButton::Left);
    send_mouse_input(&mut emu, extend_selection_to(5, 2), MouseButton::Left);
    let action = send_mouse_input(
        &mut emu,
        UserInputAction::ApplySelectionClear,
        MouseButton::Left,
    );

    let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
    assert!(
        // ApplySelectionClear now clears the selection if start == end, otherwise just deactivates
        !snapshot.selection.is_active || snapshot.selection.range.is_none(),
        "Selection should be inactive or cleared after release."
    );
    if snapshot.selection.range.is_some() {
        // If not cleared (was a drag)
        assert_eq!(
            snapshot.selection.range.map(|r| r.start),
            Some(Point { x: 1, y: 1 }),
            "Selection start point should be retained."
        );
        assert_eq!(
            snapshot.selection.range.map(|r| r.end),
            Some(Point { x: 5, y: 2 }),
            "Selection end point should be retained."
        );
    }
    assert_eq!(
        action,
        Some(EmulatorAction::RequestRedraw),
        "Action should be RequestRedraw."
    );
}

// --- End of Basic Selection Flow Tests ---

// --- Copy Action Tests ---
#[test]
fn initiate_copy_returns_none_when_no_selection_is_active() {
    let mut emu = create_test_emulator(10, 5);
    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    assert_eq!(action, None, "Should return None if no selection exists.");
}

#[test]
fn initiate_copy_returns_the_selected_text_as_a_copy_to_clipboard_action() {
    let mut emu = create_test_emulator(10, 2);
    fill_emulator_screen(&mut emu, vec!["Hello".to_string(), "World".to_string()]);

    send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
    send_mouse_input(&mut emu, extend_selection_to(4, 0), MouseButton::Left);
    send_mouse_input(
        &mut emu,
        UserInputAction::ApplySelectionClear,
        MouseButton::Left,
    );

    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    assert_eq!(
        action,
        Some(EmulatorAction::CopyToClipboard("Hello".to_string())),
        "Selected text mismatch."
    );
}

#[test]
fn initiate_copy_joins_block_selected_rows_with_newlines() {
    let mut emu = create_test_emulator(3, 3);
    fill_emulator_screen(
        &mut emu,
        vec!["ABC".to_string(), "DEF".to_string(), "GHI".to_string()],
    );

    // Use the send API to create a selection
    send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
    send_mouse_input(&mut emu, extend_selection_to(1, 1), MouseButton::Left);
    send_mouse_input(
        &mut emu,
        UserInputAction::ApplySelectionClear,
        MouseButton::Left,
    );

    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    assert_eq!(
        action,
        Some(EmulatorAction::CopyToClipboard("ABC\nDE".to_string())), // Adjusted expected output
        "Block selected text mismatch."
    );
}
// --- End of Copy Action Tests ---

// --- Selection Clearing Tests ---
#[test]
fn new_mouse_press_clears_old_selection() {
    let mut emu = create_test_emulator(10, 5);

    send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
    send_mouse_input(&mut emu, extend_selection_to(2, 0), MouseButton::Left);
    send_mouse_input(
        &mut emu,
        UserInputAction::ApplySelectionClear,
        MouseButton::Left,
    );

    let snapshot_old = emu.get_render_snapshot().expect("Snapshot was None");
    let old_selection_end = snapshot_old.selection.range.map(|r| r.end);
    assert_eq!(
        old_selection_end,
        Some(Point { x: 2, y: 0 }),
        "Pre-condition: First selection should be (0,0) to (2,0)"
    );
    assert!(!snapshot_old.selection.is_active);

    let action = send_mouse_input(&mut emu, start_selection_at(1, 1), MouseButton::Left);

    let snapshot_new = emu.get_render_snapshot().expect("Snapshot was None");
    assert!(
        snapshot_new.selection.is_active,
        "New selection should be active."
    );
    assert_eq!(
        snapshot_new.selection.range.map(|r| r.start),
        Some(Point { x: 1, y: 1 }),
        "New selection start point mismatch."
    );
    assert_eq!(
        snapshot_new.selection.range.map(|r| r.end),
        Some(Point { x: 1, y: 1 }),
        "New selection end point should be same as new start."
    );
    assert_ne!(
        snapshot_new.selection.range.map(|r| r.end),
        old_selection_end,
        "New selection should differ from old one."
    );
    assert_eq!(action, Some(EmulatorAction::RequestRedraw));
}
// --- End of Selection Clearing Tests ---

// --- Selection Interaction with Scrolling Test ---
#[test]
fn selection_coordinates_adjust_on_scroll() {
    let mut emu = create_test_emulator(10, 3);
    fill_emulator_screen(
        &mut emu,
        vec![
            "Line0".to_string(),
            "Line1".to_string(),
            "Line2".to_string(),
        ],
    );

    send_mouse_input(&mut emu, start_selection_at(0, 1), MouseButton::Left);
    send_mouse_input(&mut emu, extend_selection_to(4, 1), MouseButton::Left);
    send_mouse_input(
        &mut emu,
        UserInputAction::ApplySelectionClear,
        MouseButton::Left,
    );

    let snapshot_before = emu.get_render_snapshot().expect("Snapshot was None");
    assert_eq!(
        snapshot_before.selection.range.map(|r| r.start),
        Some(Point { x: 0, y: 1 })
    );
    assert_eq!(
        snapshot_before.selection.range.map(|r| r.end),
        Some(Point { x: 4, y: 1 })
    );
    assert!(!snapshot_before.selection.is_active);

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 1),
    )));
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    let snapshot_after = emu.get_render_snapshot().expect("Snapshot was None");
    assert_eq!(
        snapshot_after.selection.range.map(|r| r.start),
        Some(Point { x: 0, y: 1 }),
        "Selection start Y should not change due to scroll."
    );
    assert_eq!(
        snapshot_after.selection.range.map(|r| r.end),
        Some(Point { x: 4, y: 1 }),
        "Selection end Y should not change due to scroll."
    );

    let selected_text_after_scroll = emu.get_selected_text();
    assert_eq!(
        selected_text_after_scroll,
        Some("Line2".to_string()),
        "Selected text should be from the new line 1 (old Line2)."
    );
}
// --- End of Selection Interaction with Scrolling Test ---

// --- Selection with Alternate Screen Test ---
#[test]
fn selection_is_cleared_when_returning_from_the_alt_screen_to_the_primary_screen() {
    let mut emu = create_test_emulator(10, 3);
    fill_emulator_screen(
        &mut emu,
        vec!["Primary1".to_string(), "Primary2".to_string()],
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));
    fill_emulator_screen(&mut emu, vec!["Alt1".to_string(), "Alt2".to_string()]);

    send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
    send_mouse_input(&mut emu, extend_selection_to(3, 0), MouseButton::Left);
    send_mouse_input(
        &mut emu,
        UserInputAction::ApplySelectionClear,
        MouseButton::Left,
    );

    // Verify selection on alt screen by checking selected text
    assert_eq!(
        emu.get_selected_text(),
        Some("Alt1".to_string()),
        "Selection on alt screen incorrect."
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));

    // Verify we're back on primary screen and selection is cleared
    let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
    assert_eq!(
        snapshot.selection,
        Selection::default(),
        "Selection should be cleared after exiting alt screen."
    );
    assert_eq!(
        emu.get_selected_text(),
        None,
        "No selection should be active/present on primary screen after exiting alt."
    );

    // Verify we're showing primary screen content
    match snapshot.lines[0].cells[0] {
        Glyph::Single(cell) | Glyph::WidePrimary(cell) => assert_eq!(cell.c, 'P'),
        other => panic!("Expected char P, got {:?}", other),
    }
}
// --- End of Selection with Alternate Screen Test ---

#[test]
fn it_should_preserve_content_and_cursor_when_resizing_to_a_larger_grid() {
    let mut term = create_test_emulator(5, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('3')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('4')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('5')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E')));

    term.interpret_input(EmulatorInput::Control(resize_event(10, 4)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot,
        &["12345     ", "ABCDE     ", "          ", "          "],
        Some((1, 5)),
    );
}

#[test]
fn it_should_truncate_content_when_resizing_to_a_smaller_grid() {
    let mut term = create_test_emulator(5, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('H')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('e')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('o')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('W')));

    term.interpret_input(EmulatorInput::Control(resize_event(3, 1)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["Hel"], Some((0, 1)));
}

#[test]
fn it_should_write_a_printable_keypress_to_the_pty_and_echo_it_to_the_grid() {
    let mut term = create_test_emulator(5, 1);
    let key_input = UserInputAction::KeyInput {
        symbol: KeySymbol::Char('x'),
        modifiers: Modifiers::SHIFT,
        text: Some(std::borrow::Cow::Borrowed("X")),
    };
    let action = term.interpret_input(EmulatorInput::User(key_input));
    assert_eq!(
        action,
        Some(EmulatorAction::WritePty("X".to_string().into_bytes()))
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["X    "], Some((0, 1)));
}

#[test]
fn it_should_write_the_arrow_up_escape_sequence_to_the_pty_without_moving_the_cursor() {
    let mut term = create_test_emulator(5, 1);
    let key_input = UserInputAction::KeyInput {
        symbol: KeySymbol::Up,
        modifiers: Modifiers::empty(),
        text: None,
    };
    let action = term.interpret_input(EmulatorInput::User(key_input));

    let expected_pty_output = "\x1b[A".to_string().into_bytes();
    assert_eq!(action, Some(EmulatorAction::WritePty(expected_pty_output)));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["     "], Some((0, 0)));
}

// --- PS1 Multi-line Prompt Tests ---

#[test]
fn ps1_multiline_prompt_at_bottom_causes_scroll() {
    let mut term = create_test_emulator(5, 3);

    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('>')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["BBBBB", "P1>  ", "$    "], Some((2, 2)));
}

#[test]
fn ps1_multiline_prompt_ends_on_last_line_no_scroll_by_prompt() {
    let mut term = create_test_emulator(5, 3);

    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["AAAAA", "L1   ", "$    "], Some((2, 2)));
}

#[test]
fn ps1_multiline_prompt_last_line_fills_screen_then_input() {
    let mut term = create_test_emulator(3, 2);

    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E')));

    let snapshot_after_prompt = term.get_render_snapshot().expect("Snapshot was None");
    // After filling line with 3 chars (CDE), cursor should be at rightmost position
    assert_screen_state(&snapshot_after_prompt, &["B  ", "CDE"], Some((1, 2)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));

    let snapshot_after_input = term.get_render_snapshot().expect("Snapshot was None");
    // After wrap, 'X' should appear on next line, screen should scroll
    assert_screen_state(&snapshot_after_input, &["CDE", "X  "], Some((1, 1)));
}

#[test]
fn ps1_prompt_causes_multiple_scrolls() {
    let mut term = create_test_emulator(3, 2);

    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["L2 ", "$  "], Some((1, 2)));
}

#[test]
fn ps1_prompt_with_internal_wrapping_and_scrolling() {
    let mut term = create_test_emulator(3, 2);
    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('o')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('n')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('g')));
    // This LF is after 'g' which is the last char on the line and causes a wrap.
    // The wrap itself moves to the next line, column 0.
    // So, only an LF is needed here, not CR+LF, if the intent is just to move down.
    // However, if the prompt logic implies "end of this line of prompt, start next",
    // then CR+LF might be what the original test author would do if LNM was false.
    // Let's assume the intent is to mimic typical shell prompt behavior where each
    // "line" of the prompt ends with a newline that positions for the next segment.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["ong", "L2 "], Some((1, 2)));
}

#[test]
fn ps1_multiline_exact_fill_then_scroll_on_final_lf() {
    let mut term = create_test_emulator(3, 2);

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["P2 ", "   "], Some((1, 0)));
}

#[test]
fn ps1_multiline_with_sgr_at_bottom_scrolls() {
    let mut term = create_test_emulator(5, 2);

    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Red,
        ))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Green,
        ))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["P1   ", "$    "], Some((1, 2)));

    let glyph_p_wrapper = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    let glyph_1_wrapper = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
    let glyph_dollar_wrapper = get_glyph_from_snapshot(&snapshot, 1, 0).unwrap();
    let glyph_space_after_dollar_wrapper = get_glyph_from_snapshot(&snapshot, 1, 1).unwrap();
    let glyph_final_cursor_cell_wrapper = get_glyph_from_snapshot(&snapshot, 1, 2).unwrap();

    match glyph_p_wrapper {
        Glyph::Single(cell) | Glyph::WidePrimary(cell) => {
            assert_eq!(cell.attr.fg, Color::Named(NamedColor::Red))
        }
        other => panic!("Expected Single or WidePrimary for P, got {:?}", other),
    }
    match glyph_1_wrapper {
        Glyph::Single(cell) | Glyph::WidePrimary(cell) => {
            assert_eq!(cell.attr.fg, Color::Named(NamedColor::Red))
        }
        other => panic!("Expected Single or WidePrimary for 1, got {:?}", other),
    }
    match glyph_dollar_wrapper {
        Glyph::Single(cell) | Glyph::WidePrimary(cell) => {
            assert_eq!(cell.attr.fg, Color::Named(NamedColor::Green))
        }
        other => panic!("Expected Single or WidePrimary for $, got {:?}", other),
    }
    match glyph_space_after_dollar_wrapper {
        Glyph::Single(cell) | Glyph::WidePrimary(cell) => {
            assert_eq!(cell.attr.fg, Color::Named(NamedColor::Green))
        }
        other => panic!(
            "Expected Single or WidePrimary for space after $, got {:?}",
            other
        ),
    }
    match glyph_final_cursor_cell_wrapper {
        Glyph::Single(cell) | Glyph::WidePrimary(cell) => assert_eq!(
            cell.attr.fg,
            Attributes::default().fg,
            "Cursor cell attributes should be reset"
        ),
        other => panic!(
            "Expected Single or WidePrimary for final cursor cell, got {:?}",
            other
        ),
    }
}

#[test]
fn lf_scrolls_the_partial_scrolling_region_when_the_cursor_is_at_its_bottom_with_origin_mode_off() {
    let cols = 10;
    let rows = 5;
    let mut emu = create_test_emulator(cols, rows);
    // Disable Linefeed/Newline Mode (LNM) - testing that LF doesn't do CR
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetMode(StandardModeConstant::LinefeedNewlineMode as u16),
    )));

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::Origin as u16),
    )));

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 2, bottom: 4 },
    )));

    let snap_after_stbm = emu.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snap_after_stbm,
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((0, 0)),
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    for _ in 0..5 {
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    }

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    )));
    for _ in 0..5 {
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    }
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 1),
    )));
    for _ in 0..5 {
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Y')));
    }
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(4, 1),
    )));
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z')));
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z')));

    let snapshot_before_lf = emu.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_before_lf,
        &[
            "XXXXX     ",
            "XXXXX     ",
            "YYYYY     ",
            "ZZ        ",
            "          ",
        ],
        Some((3, 2)),
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    let snapshot_after_lf = emu.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_after_lf,
        &[
            "XXXXX     ",
            "YYYYY     ",
            "ZZ        ",
            "          ",
            "          ",
        ],
        Some((3, 2)),
    );
}
#[test]
fn it_should_respond_to_primary_device_attributes_query_with_csi_question_mark_6c() {
    let mut term = create_test_emulator(80, 24);

    let input_da = EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::PrimaryDeviceAttributes));

    let action = term.interpret_input(input_da);

    let expected_response_bytes = b"\x1b[?6c".to_vec();
    assert_eq!(
        action,
        Some(EmulatorAction::WritePty(expected_response_bytes)),
        "Terminal should respond to Primary Device Attributes query (CSI c)"
    );
}

// --- Tests for Selection Logic in TerminalEmulator ---
#[cfg(test)]
mod selection_logic_tests {
    use super::*;
    use crate::term::snapshot::SelectionRange;

    #[test]
    fn it_should_activate_selection_at_the_point_the_start_action_targets() {
        let mut emu = create_test_emulator(10, 5);
        let point = Point { x: 2, y: 1 };

        // Use the send API to start selection
        emu.interpret_input(EmulatorInput::User(start_selection_at(2, 1)));

        let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
        assert_eq!(
            snapshot.selection.range,
            Some(SelectionRange {
                start: point,
                end: point
            })
        );
        assert!(snapshot.selection.is_active);
        assert_eq!(snapshot.selection.mode, SelectionMode::Cell);
    }

    #[test]
    fn extend_selection_only_updates_the_range_after_a_selection_has_started() {
        let mut emu = create_test_emulator(10, 5);
        let start_point = Point { x: 2, y: 1 };
        let extend_point = Point { x: 5, y: 2 };

        // Try extending selection when none exists - should have no effect
        emu.interpret_input(EmulatorInput::User(extend_selection_to(5, 2)));
        let snapshot1 = emu.get_render_snapshot().expect("Snapshot was None");
        assert_eq!(snapshot1.selection.range, None);
        assert!(!snapshot1.selection.is_active);

        // Start selection and then extend it
        emu.interpret_input(EmulatorInput::User(start_selection_at(2, 1)));
        emu.interpret_input(EmulatorInput::User(extend_selection_to(5, 2)));
        let snapshot2 = emu.get_render_snapshot().expect("Snapshot was None");
        assert_eq!(
            snapshot2.selection.range,
            Some(SelectionRange {
                start: start_point,
                end: extend_point
            })
        );
        assert!(snapshot2.selection.is_active);
    }

    #[test]
    fn apply_selection_clear_discards_a_click_but_finalizes_a_drag() {
        let mut emu = create_test_emulator(10, 5);
        let point1 = Point { x: 2, y: 1 };
        let point2 = Point { x: 5, y: 1 };

        // Test click (no drag) - selection should be cleared
        emu.interpret_input(EmulatorInput::User(start_selection_at(2, 1)));
        let snapshot1 = emu.get_render_snapshot().expect("Snapshot was None");
        assert!(snapshot1.selection.is_active);

        emu.interpret_input(EmulatorInput::User(UserInputAction::ApplySelectionClear));
        let snapshot2 = emu.get_render_snapshot().expect("Snapshot was None");
        assert_eq!(
            snapshot2.selection.range, None,
            "Selection should be cleared on click"
        );
        assert!(
            !snapshot2.selection.is_active,
            "Selection should be inactive after click"
        );

        // Test drag - selection should be maintained but deactivated
        emu.interpret_input(EmulatorInput::User(start_selection_at(2, 1)));
        emu.interpret_input(EmulatorInput::User(extend_selection_to(5, 1)));
        let snapshot3 = emu.get_render_snapshot().expect("Snapshot was None");
        assert!(snapshot3.selection.is_active);

        emu.interpret_input(EmulatorInput::User(UserInputAction::ApplySelectionClear));
        let snapshot4 = emu.get_render_snapshot().expect("Snapshot was None");
        assert_eq!(
            snapshot4.selection.range,
            Some(SelectionRange {
                start: point1,
                end: point2
            }),
            "Selection range should be maintained after drag"
        );
        assert!(
            !snapshot4.selection.is_active,
            "Selection should be inactive after drag"
        );
    }

    #[test]
    fn clear_selection_removes_a_finalized_drag_range() {
        let mut emu = create_test_emulator(10, 5);

        // Create and deactivate a selection
        emu.interpret_input(EmulatorInput::User(start_selection_at(2, 1)));
        emu.interpret_input(EmulatorInput::User(extend_selection_to(5, 1)));
        emu.interpret_input(EmulatorInput::User(UserInputAction::ApplySelectionClear));

        let snapshot1 = emu.get_render_snapshot().expect("Snapshot was None");
        assert!(snapshot1.selection.range.is_some());
        assert!(!snapshot1.selection.is_active);

        // Clear selection using public API
        emu.clear_selection();
        let snapshot2 = emu.get_render_snapshot().expect("Snapshot was None");
        assert_eq!(snapshot2.selection.range, None);
        assert!(!snapshot2.selection.is_active);
    }
}

#[cfg(test)]
mod get_selected_text_tests {
    use super::*;

    #[test]
    fn get_selected_text_returns_none_when_no_selection_is_active() {
        let emu = create_test_emulator(10, 5);
        assert_eq!(emu.get_selected_text(), None);
    }

    #[test]
    fn get_selected_text_returns_a_substring_of_a_single_line() {
        let mut emu = create_test_emulator(10, 5);
        fill_emulator_screen(&mut emu, vec!["Hello World".to_string()]);

        // Create selection from (0,0) to (4,0)
        send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(4, 0), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("Hello".to_string()));
    }

    #[test]
    fn get_selected_text_includes_trailing_spaces_within_the_selection() {
        let mut emu = create_test_emulator(10, 1);
        fill_emulator_screen(&mut emu, vec!["Hi   ".to_string()]);

        send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(4, 0), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("Hi   ".to_string()));
    }

    #[test]
    fn get_selected_text_joins_a_partial_multi_line_selection_with_a_newline() {
        let mut emu = create_test_emulator(10, 5);
        fill_emulator_screen(
            &mut emu,
            vec![
                "First line".to_string(),
                "Second line".to_string(),
                "Third line".to_string(),
            ],
        );

        send_mouse_input(&mut emu, start_selection_at(2, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(3, 1), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("rst line\nSeco".to_string()));
    }

    #[test]
    fn get_selected_text_handles_both_single_and_multi_line_full_selections() {
        let mut emu = create_test_emulator(10, 3);
        fill_emulator_screen(
            &mut emu,
            vec![
                "Line One".to_string(),
                "Line Two".to_string(),
                "Line Three".to_string(),
            ],
        );

        // Test single line selection
        send_mouse_input(&mut emu, start_selection_at(0, 1), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(7, 1), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );
        assert_eq!(emu.get_selected_text(), Some("Line Two".to_string()));

        // Test multi-line selection
        send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(7, 1), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );
        assert_eq!(
            emu.get_selected_text(),
            Some("Line One\nLine Two".to_string())
        );
    }

    #[test]
    fn get_selected_text_spans_a_line_boundary_correctly() {
        let mut emu = create_test_emulator(10, 2);
        fill_emulator_screen(
            &mut emu,
            vec!["0123456789".to_string(), "abcdefghij".to_string()],
        );

        send_mouse_input(&mut emu, start_selection_at(7, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(2, 1), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("789\nabc".to_string()));
    }

    #[test]
    fn get_selected_text_renders_untouched_cells_as_spaces() {
        let mut emu = create_test_emulator(5, 1);
        // Create sparse content: "A   E" by printing A, moving cursor to col 4, then printing E
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::CursorPosition(1, 5),
        )));
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E')));

        send_mouse_input(&mut emu, start_selection_at(0, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(4, 0), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("A   E".to_string()));
    }

    #[test]
    fn get_selected_text_pads_a_selection_extending_past_the_line_content() {
        let mut emu = create_test_emulator(10, 1);
        fill_emulator_screen(&mut emu, vec!["Test".to_string()]);

        send_mouse_input(&mut emu, start_selection_at(1, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(7, 0), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("est    ".to_string()));
    }

    #[test]
    fn get_selected_text_normalizes_a_selection_dragged_backwards() {
        let mut emu = create_test_emulator(10, 5);
        fill_emulator_screen(&mut emu, vec!["Hello World".to_string()]);

        // Select with reversed points (end before start)
        send_mouse_input(&mut emu, start_selection_at(4, 0), MouseButton::Left);
        send_mouse_input(&mut emu, extend_selection_to(0, 0), MouseButton::Left);
        send_mouse_input(
            &mut emu,
            UserInputAction::ApplySelectionClear,
            MouseButton::Left,
        );

        assert_eq!(emu.get_selected_text(), Some("Hello".to_string()));
    }
}

#[cfg(test)]
mod paste_text_tests {
    use super::*;

    #[test]
    fn paste_text_bracketed_off_simple() {
        let mut emu = create_test_emulator(20, 1);
        // Bracketed paste mode is off by default - just verify behavior

        let text_to_paste = "Pasted text.";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
        let expected_cursor_x = text_to_paste.chars().count();
        assert_screen_state(
            &snapshot,
            &["Pasted text.        "],
            Some((0, expected_cursor_x)),
        );
    }

    #[test]
    fn paste_text_bracketed_off_with_newline() {
        let mut emu = create_test_emulator(20, 2);
        // Bracketed paste mode is off by default - just verify behavior

        let text_to_paste = "Line1\nLine2";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
        let expected_screen = ["Line1               ", "Line2               "];
        let expected_cursor_x = "Line2".chars().count();
        assert_screen_state(&snapshot, &expected_screen, Some((1, expected_cursor_x)));
    }

    #[test]
    fn paste_text_bracketed_off_causes_wrap() {
        let mut emu = create_test_emulator(5, 2);
        // Bracketed paste mode is off by default - just verify wrapping behavior

        let text_to_paste = "HelloWorld";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
        let expected_screen = [
            "Hello", // This line should be exactly 5 chars
            "World", // This line should be exactly 5 chars
        ];
        assert_screen_state(&snapshot, &expected_screen, Some((1, 4))); // Expected cursor position is (1, 4) due to wrap
    }

    #[test]
    fn paste_text_bracketed_on_logs_warning_processes_chars() {
        let mut emu = create_test_emulator(20, 1);
        // Enable bracketed paste mode
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetModePrivate(DecModeConstant::BracketedPaste as u16),
        )));

        let text_to_paste = "Pasted";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot().expect("Snapshot was None");
        let expected_cursor_x = text_to_paste.chars().count();
        assert_screen_state(
            &snapshot,
            &["Pasted              "],
            Some((0, expected_cursor_x)),
        );
    }

    #[test]
    fn ansi_resize_sets_terminal_dimensions() {
        let mut emu = create_test_emulator(80, 24);
        assert_eq!(
            emu.get_render_snapshot()
                .expect("Snapshot was None")
                .dimensions,
            (80, 24)
        );

        let resize_command = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 8,
            ps2: Some(30),
            ps3: Some(100),
        });
        emu.interpret_input(EmulatorInput::Ansi(resize_command));

        assert_eq!(
            emu.get_render_snapshot()
                .expect("Snapshot was None")
                .dimensions,
            (100, 30),
            "Terminal should resize to 100 cols x 30 rows"
        );
    }

    #[test]
    fn ansi_resize_updates_snapshot_dimensions() {
        let mut emu = create_test_emulator(80, 24);

        let resize_command = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 8,
            ps2: Some(20),
            ps3: Some(60),
        });
        emu.interpret_input(EmulatorInput::Ansi(resize_command));

        let snapshot = emu.get_render_snapshot().expect("Snapshot should exist");
        assert_eq!(
            snapshot.dimensions,
            (60, 20),
            "Snapshot dimensions should reflect the resize"
        );
    }

    #[test]
    fn ansi_resize_with_zero_rows_ignored() {
        let mut emu = create_test_emulator(80, 24);

        let resize_command = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 8,
            ps2: Some(0),
            ps3: Some(100),
        });
        emu.interpret_input(EmulatorInput::Ansi(resize_command));

        assert_eq!(
            emu.get_render_snapshot()
                .expect("Snapshot was None")
                .dimensions,
            (80, 24),
            "Terminal should ignore resize with zero rows"
        );
    }

    #[test]
    fn ansi_resize_with_zero_cols_ignored() {
        let mut emu = create_test_emulator(80, 24);

        let resize_command = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 8,
            ps2: Some(30),
            ps3: Some(0),
        });
        emu.interpret_input(EmulatorInput::Ansi(resize_command));

        assert_eq!(
            emu.get_render_snapshot()
                .expect("Snapshot was None")
                .dimensions,
            (80, 24),
            "Terminal should ignore resize with zero cols"
        );
    }

    #[test]
    fn ansi_resize_with_missing_params_ignored() {
        let mut emu = create_test_emulator(80, 24);

        let resize_command = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 8,
            ps2: None,
            ps3: Some(100),
        });
        emu.interpret_input(EmulatorInput::Ansi(resize_command));

        assert_eq!(
            emu.get_render_snapshot()
                .expect("Snapshot was None")
                .dimensions,
            (80, 24),
            "Terminal should ignore resize with missing rows parameter"
        );

        let resize_command2 = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 8,
            ps2: Some(30),
            ps3: None,
        });
        emu.interpret_input(EmulatorInput::Ansi(resize_command2));

        assert_eq!(
            emu.get_render_snapshot()
                .expect("Snapshot was None")
                .dimensions,
            (80, 24),
            "Terminal should ignore resize with missing cols parameter"
        );
    }

    #[test]
    fn it_should_report_current_terminal_dimensions_for_window_manipulation_ps_18() {
        let mut emu = create_test_emulator(80, 24);

        let report_command = AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 18,
            ps2: None,
            ps3: None,
        });
        let action = emu.interpret_input(EmulatorInput::Ansi(report_command));

        assert_eq!(
            action,
            Some(EmulatorAction::WritePty(b"\x1b[8;24;80t".to_vec())),
            "Should report current terminal dimensions"
        );
    }
}

// ============================================================================
// Damage tracking across the scrollback viewport (PR #974 review)
// ============================================================================

/// Fill enough lines to populate scrollback, scroll back, and let the
/// terminal go idle: the SECOND snapshot at an unchanged view must be fully
/// clean, or the app's damage gate re-renders on every timer tick for as
/// long as the user stays scrolled back.
#[test]
fn scrolled_back_idle_snapshot_is_clean() {
    let mut term = create_test_emulator(4, 2);
    for ch in "a\r\nb\r\nc\r\nd\r\ne".chars() {
        match ch {
            '\r' => term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
                crate::ansi::commands::C0Control::CR,
            ))),
            '\n' => term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
                crate::ansi::commands::C0Control::LF,
            ))),
            c => term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(c))),
        };
    }
    assert!(term.scroll_viewport(1), "scrollback should be available");

    let first = term.get_render_snapshot().expect("snapshot");
    assert!(
        first.lines.iter().any(|l| l.is_dirty),
        "the view moved: the first scrolled-back snapshot must be dirty"
    );

    let second = term.get_render_snapshot().expect("snapshot");
    assert!(
        !second.lines.iter().any(|l| l.is_dirty),
        "idle at an unchanged view: nothing may be dirty, else the damage \
         gate never skips while scrolled back"
    );
}

/// Returning to the live screen is a view change even though no
/// scrollback-sourced line remains to carry the signal — every line's screen
/// position changed, so the snapshot must be dirty.
#[test]
fn returning_to_live_screen_marks_dirty() {
    let mut term = create_test_emulator(4, 2);
    for _ in 0..4 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
            crate::ansi::commands::C0Control::LF,
        )));
    }
    assert!(term.scroll_viewport(1));
    let _ = term.get_render_snapshot().expect("snapshot");
    assert!(term.scroll_viewport(-1), "scroll back to live");

    let snapshot = term.get_render_snapshot().expect("snapshot");
    assert!(
        snapshot.lines.iter().any(|l| l.is_dirty),
        "offset n -> 0 changed every visible line"
    );
}

/// A scrollback push while scrolled back shifts what a held offset shows
/// (the offset anchors to the scrollback's END), so new output must dirty
/// the scrolled-back view.
#[test]
fn scrollback_growth_under_held_offset_marks_dirty() {
    let mut term = create_test_emulator(4, 2);
    for _ in 0..4 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
            crate::ansi::commands::C0Control::LF,
        )));
    }
    assert!(term.scroll_viewport(1));
    let _ = term.get_render_snapshot().expect("snapshot");

    // New output scrolls another line into scrollback underneath the view.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
        crate::ansi::commands::C0Control::LF,
    )));

    let snapshot = term.get_render_snapshot().expect("snapshot");
    assert!(
        snapshot.lines.iter().any(|l| l.is_dirty),
        "the held offset now shows different history"
    );
}

/// A cursor-only change (no cell mutated) must still dirty the snapshot:
/// CUF/CUB/CUP and style changes move the cursor without touching a line,
/// and a skipped frame would freeze the visible cursor at its old cell.
#[test]
fn cursor_only_movement_marks_its_rows_dirty() {
    let mut term = create_test_emulator(4, 2);
    let _ = term.get_render_snapshot().expect("snapshot");
    let clean = term.get_render_snapshot().expect("snapshot");
    assert!(
        !clean.lines.iter().any(|l| l.is_dirty),
        "precondition: an idle terminal snapshots clean"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(1),
    )));
    let moved = term.get_render_snapshot().expect("snapshot");
    assert!(
        moved.lines.iter().any(|l| l.is_dirty),
        "the cursor moved: the frame must redraw"
    );

    let idle = term.get_render_snapshot().expect("snapshot");
    assert!(
        !idle.lines.iter().any(|l| l.is_dirty),
        "cursor at rest: clean again"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorDown(1),
    )));
    let rowed = term.get_render_snapshot().expect("snapshot");
    assert_eq!(
        rowed.lines.iter().filter(|l| l.is_dirty).count(),
        2,
        "a row change dirties the row left and the row entered"
    );
}

// White-box test of `get_render_snapshot`'s documented per-line dirty
// contract, not a claim any consumer reads per-row state — no production
// code checks `is_dirty` per row today, only whole-snapshot (`any()`).
// Row-specific on purpose: rows inside the region are independently dirty
// from their own content change regardless of `view_changed`, so only rows
// outside it (with no other dirty reason) isolate `view_changed`'s own
// contribution to the OR.
#[test]
fn scroll_region_push_dirties_rows_outside_the_region() {
    let mut term = create_test_emulator(4, 6);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 1, bottom: 3 },
    )));
    let _ = term.get_render_snapshot().expect("snapshot");

    // Cursor starts at the region's top row; three linefeeds walk it to the
    // region's bottom and then scroll the region, pushing row 0 into
    // scrollback and bumping the generation. Rows 3-5 sit outside the
    // region and are never touched by the scroll — nothing except
    // `view_changed` can dirty them.
    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(
            crate::ansi::commands::C0Control::LF,
        )));
    }

    let after_scroll = term.get_render_snapshot().expect("snapshot");
    assert!(
        after_scroll.lines[3..6].iter().all(|l| l.is_dirty),
        "the view changed (scrollback generation bumped): `get_render_snapshot`'s \
         documented contract dirties every row, including ones the scroll itself \
         never touched"
    );

    let idle = term.get_render_snapshot().expect("snapshot");
    assert!(
        !idle.lines[3..6].iter().any(|l| l.is_dirty),
        "the view is unchanged and those rows were never touched: clean"
    );
}

// --- scroll_viewport / reset_viewport / scrollback_len ---

#[test]
fn scroll_viewport_with_no_scrollback_does_not_move_and_returns_false() {
    let mut term = create_test_emulator(5, 3);
    assert!(!term.scroll_viewport(1));
    assert_eq!(term.viewport_offset(), 0);
}

#[test]
fn scroll_viewport_moves_by_the_requested_amount_and_returns_true() {
    let mut term = create_test_emulator(5, 3);
    for _ in 0..6 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    }
    assert!(term.scrollback_len() >= 2);
    assert!(term.scroll_viewport(1));
    assert_eq!(term.viewport_offset(), 1);
    assert!(term.scroll_viewport(-1));
    assert_eq!(term.viewport_offset(), 0);
}

#[test]
fn scroll_viewport_negative_decrements_the_offset_by_the_given_amount() {
    let mut term = create_test_emulator(5, 3);
    for _ in 0..12 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    }
    assert!(term.scrollback_len() >= 4);
    term.scroll_viewport(4);
    assert_eq!(term.viewport_offset(), 4);
    // Regression check: negating a negative `lines` via plain `as usize` (instead of
    // `-lines`) wraps to a huge value, and `saturating_sub` on that always floors to 0 —
    // indistinguishable from a correct decrement-by-1 only when the offset started at or
    // below the requested amount. Starting well above 1 isolates the actual magnitude.
    term.scroll_viewport(-1);
    assert_eq!(term.viewport_offset(), 3);
}

#[test]
fn scroll_viewport_clamps_to_the_available_scrollback() {
    let mut term = create_test_emulator(5, 3);
    for _ in 0..6 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    }
    let available = term.scrollback_len();
    term.scroll_viewport(1000);
    assert_eq!(term.viewport_offset(), available);
}

#[test]
fn scrolled_back_view_shows_scrollback_lines_in_chronological_order() {
    let mut term = create_test_emulator(10, 3);
    fill_emulator_screen(
        &mut term,
        vec![
            "line0".to_string(),
            "line1".to_string(),
            "line2".to_string(),
            "line3".to_string(),
            "line4".to_string(),
            "line5".to_string(),
        ],
    );
    assert_eq!(term.scrollback_len(), 3);
    term.scroll_viewport(3);
    let snapshot = term.get_render_snapshot().expect("snapshot");
    let digit_at =
        |row: usize| get_glyph_from_snapshot(&snapshot, row, 4).map(|g| g.display_char());
    assert_eq!(digit_at(0), Some('0'));
    assert_eq!(digit_at(1), Some('1'));
    assert_eq!(digit_at(2), Some('2'));
}

#[test]
fn reset_viewport_returns_to_live_and_reports_whether_it_moved() {
    let mut term = create_test_emulator(5, 3);
    for _ in 0..6 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    }
    term.scroll_viewport(1);
    assert!(term.reset_viewport());
    assert_eq!(term.viewport_offset(), 0);
    assert!(!term.reset_viewport());
}

// --- mouse tracking mode queries and event encoding ---

#[test]
fn is_mouse_tracking_active_reflects_whether_any_mouse_mode_is_set() {
    let mut term = create_test_emulator(10, 5);
    assert!(!term.is_mouse_tracking_active());
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::MouseVt200 as u16),
    )));
    assert!(term.is_mouse_tracking_active());
}

#[test]
fn is_mouse_tracking_active_is_true_when_only_the_any_event_mode_is_set() {
    let mut term = create_test_emulator(10, 5);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::MouseAnyEvent as u16),
    )));
    assert!(term.is_mouse_tracking_active());
}

#[test]
fn reports_all_motion_is_true_only_under_the_any_event_mouse_mode() {
    let mut term = create_test_emulator(10, 5);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::MouseButtonEvent as u16),
    )));
    assert!(!term.reports_all_motion());
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::MouseAnyEvent as u16),
    )));
    assert!(term.reports_all_motion());
}

#[test]
fn reports_button_motion_is_true_under_button_event_or_any_event_mode() {
    let mut term = create_test_emulator(10, 5);
    assert!(!term.reports_button_motion());
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::MouseButtonEvent as u16),
    )));
    assert!(term.reports_button_motion());
}

#[test]
fn encode_mouse_event_returns_none_when_no_tracking_mode_is_active() {
    let term = create_test_emulator(10, 5);
    let bytes = term.encode_mouse_event(MouseEncodingParams {
        button: MouseButton::Left,
        col: 1,
        row: 1,
        kind: MouseEventKind::Press,
    });
    assert_eq!(bytes, None);
}

#[test]
fn encode_mouse_event_returns_bytes_once_a_tracking_mode_is_active() {
    let mut term = create_test_emulator(10, 5);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::MouseVt200 as u16),
    )));
    let bytes = term.encode_mouse_event(MouseEncodingParams {
        button: MouseButton::Left,
        col: 1,
        row: 1,
        kind: MouseEventKind::Press,
    });
    assert!(bytes.is_some());
}
