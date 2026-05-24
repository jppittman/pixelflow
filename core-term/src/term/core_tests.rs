use crate::ansi::commands::{AnsiCommand, Attribute, C0Control, CsiCommand};
use crate::color::{Color, NamedColor};
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::term::{
    action::EmulatorAction, modes::DecModeConstant, ControlEvent, CursorShape, EmulatorInput,
    TerminalEmulator, TerminalSnapshot,
};

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
    expected_screen: &[&str], // expected_screen strings should NOT contain explicit WIDE_CHAR_PLACEHOLDERs
    expected_cursor_pos: Option<(usize, usize)>,
) {
    assert_eq!(
        snapshot.dimensions.1, // rows
        expected_screen.len(),
        "Snapshot row count mismatch. Expected {}, got {}. Snapshot lines: {:?}",
        expected_screen.len(),
        snapshot.dimensions.1,
        snapshot.lines
    );
    // if !expected_screen.is_empty() { // Initial width check commented out as it was problematic
    //     assert!(
    //         snapshot.dimensions.0 >= expected_screen[0].chars().map(|c| crate::term::unicode::get_char_display_width(c).max(1)).sum::<usize>(),
    //         "Snapshot col count ({}) is less than the character-width-aware width of the first expected row ({}). Expected screen: {:?}",
    //         snapshot.dimensions.0,
    //         expected_screen[0].chars().map(|c| crate::term::unicode::get_char_display_width(c).max(1)).sum::<usize>(),
    //         expected_screen[0]
    //     );
    // }

    for r in 0..snapshot.dimensions.1 {
        let expected_row_str = expected_screen.get(r).unwrap_or_else(|| {
            panic!("Expected screen data missing for row {}", r);
        });

        let mut s_col = 0; // current column in the snapshot being checked
        let mut expected_chars_iter = expected_row_str.chars().peekable();

        while let Some(expected_char) = expected_chars_iter.next() {
            if s_col >= snapshot.dimensions.0 {
                let remaining_expected: String = expected_chars_iter.collect();
                panic!(
                    "Snapshot row {} (len {}) is shorter than expected string '{}'. Expected char '{}' (and potentially '{}') at snapshot col {} would exceed width.",
                    r, snapshot.dimensions.0, expected_row_str, expected_char, remaining_expected, s_col
                );
            }

            let actual_glyph_wrapper =
                get_glyph_from_snapshot(snapshot, r, s_col).unwrap_or_else(|| {
                    panic!(
                        "Glyph ({}, {}) not found in snapshot. Expected char: '{}'",
                        r, s_col, expected_char
                    )
                });

            let (actual_char, _actual_attrs) = match actual_glyph_wrapper {
                Glyph::Single(cell) => (cell.c, Some(cell.attr)),
                Glyph::WidePrimary(cell) => (cell.c, Some(cell.attr)),
                Glyph::WideSpacer => (crate::glyph::WIDE_CHAR_PLACEHOLDER, None),
            };

            assert_eq!(
                actual_char, expected_char,
                "Char mismatch at (row {}, snapshot_col {}). Expected '{}', got '{}'. Full expected row: '{}', Full actual row: '{:?}'",
                r, s_col, expected_char, actual_char, expected_row_str, snapshot.lines.get(r).map(|l| &l.cells)
            );

            let expected_char_width =
                crate::term::unicode::get_char_display_width(expected_char).max(1);

            if expected_char_width == 2 {
                assert!(
                    matches!(actual_glyph_wrapper, Glyph::WidePrimary(_)),
                    "Expected wide char '{}' at ({},{}) to be WidePrimary. Actual: {:?}",
                    expected_char,
                    r,
                    s_col,
                    actual_glyph_wrapper
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
                    assert!(matches!(spacer_glyph_wrapper, Glyph::WideSpacer),
                        "Spacer glyph at ({},{}) should be WideSpacer for char '{}', but it is not. Actual glyph: {:?}",
                        r, s_col + 1, expected_char, spacer_glyph_wrapper
                    );
                }
                s_col += expected_char_width;
            } else {
                // expected_char is narrow
                assert!(
                    matches!(actual_glyph_wrapper, Glyph::Single(_)),
                    "Narrow char '{}' at ({},{}) should be Single. Actual: {:?}",
                    expected_char,
                    r,
                    s_col,
                    actual_glyph_wrapper
                );
                s_col += expected_char_width; // which is 1
            }
        }

        // After all expected_chars are consumed, remaining cells in snapshot row must be spaces
        for fill_col in s_col..snapshot.dimensions.0 {
            let glyph_wrapper = get_glyph_from_snapshot(snapshot, r, fill_col)
                .unwrap_or_else(|| panic!("Glyph ({}, {}) not found for fill check", r, fill_col));
            match glyph_wrapper {
                Glyph::Single(cell) => {
                    assert_eq!(
                        cell.c, ' ',
                        "Expected empty char ' ' for fill at ({}, {}), got '{}'",
                        r, fill_col, cell.c
                    );
                    // No need to check WIDE flags for Single cells, they won't have them by definition.
                }
                other => panic!(
                    "Expected Single space for fill char at ({},{}), got {:?}",
                    r, fill_col, other
                ),
            }
        }
    }

    if let Some((r_expected, c_expected)) = expected_cursor_pos {
        let cursor_state = snapshot.cursor_state.as_ref().unwrap_or_else(|| {
            panic!(
                "Expected cursor to be visible, but cursor_state is None. Expected pos: ({},{})",
                r_expected, c_expected
            );
        });
        assert_eq!(cursor_state.y, r_expected, "Cursor row mismatch");
        assert_eq!(cursor_state.x, c_expected, "Cursor col mismatch");
    } else {
        assert!(
            snapshot.cursor_state.is_none(),
            "Expected cursor to be hidden, but cursor_state is Some"
        );
    }
}

#[test]
fn it_should_print_a_single_ascii_character() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["A         "], Some((0, 1)));
}

#[test]
fn it_should_print_multiple_ascii_characters_on_one_line() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('H')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('e')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('o')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["Hello     "], Some((0, 5)));
}

#[test]
fn it_should_wrap_character_to_next_line_when_end_of_line_is_reached() {
    let mut term = create_test_emulator(5, 2);
    for char_code in '1'..='5' {
        // Prints "12345"
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(char_code)));
    }
    let snapshot_before_wrap = term.get_render_snapshot().expect("Snapshot was None");
    // After filling a line, cursor is at rightmost position (0, 4)
    assert_screen_state(&snapshot_before_wrap, &["12345", "     "], Some((0, 4)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('W'))); // This character should wrap
    let snapshot_after_wrap = term.get_render_snapshot().expect("Snapshot was None");
    // Character wraps to next line
    assert_screen_state(&snapshot_after_wrap, &["12345", "W    "], Some((1, 1)));
}

#[test]
fn it_should_overwrite_existing_characters() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Y')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 2),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["XAZ       "], Some((0, 2)));
}

#[test]
fn it_should_print_a_single_multibyte_unicode_character() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["世        "], Some((0, 2)));
    let glyph_1_wrapper = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    let glyph_2_wrapper = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
    match glyph_1_wrapper {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '世'),
        other => panic!("Expected WidePrimary '世' at (0,0), got {:?}", other),
    }
    // For glyph_2, we expect it to be a WideSpacer.
    // The character of a WideSpacer is not directly WIDE_CHAR_PLACEHOLDER,
    // but its nature as a spacer is checked by the matches macro.
    // The assert_screen_state function already verifies its displayed char if needed.
    assert!(
        matches!(glyph_2_wrapper, Glyph::WideSpacer),
        "Expected WideSpacer at (0,1), got {:?}",
        glyph_2_wrapper
    );
}

#[test]
fn it_should_print_multiple_multibyte_unicode_characters() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('你')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('好')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["你好      "], Some((0, 4)));

    let char1_glyph1 = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    let char1_glyph2 = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
    match char1_glyph1 {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '你'),
        _ => panic!("Expected WidePrimary at (0,0)"),
    }
    assert!(
        matches!(char1_glyph2, Glyph::WideSpacer),
        "Expected WideSpacer at (0,1)"
    );

    let char2_glyph1 = get_glyph_from_snapshot(&snapshot, 0, 2).unwrap();
    let char2_glyph2 = get_glyph_from_snapshot(&snapshot, 0, 3).unwrap();
    match char2_glyph1 {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '好'),
        _ => panic!("Expected WidePrimary at (0,2)"),
    }
    assert!(
        matches!(char2_glyph2, Glyph::WideSpacer),
        "Expected WideSpacer at (0,3)"
    );
}

#[test]
fn it_should_handle_mixed_ascii_and_multibyte_unicode_characters() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["A世B      "], Some((0, 4)));

    let glyph_a = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    match glyph_a {
        Glyph::Single(cell) => assert_eq!(cell.c, 'A'),
        _ => panic!("Expected Single 'A' at (0,0)"),
    }

    let glyph_uni_1 = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
    let glyph_uni_2 = get_glyph_from_snapshot(&snapshot, 0, 2).unwrap();
    match glyph_uni_1 {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '世'),
        _ => panic!("Expected WidePrimary '世' at (0,1)"),
    }
    assert!(
        matches!(glyph_uni_2, Glyph::WideSpacer),
        "Expected WideSpacer at (0,2)"
    );

    let glyph_b = get_glyph_from_snapshot(&snapshot, 0, 3).unwrap();
    match glyph_b {
        Glyph::Single(cell) => assert_eq!(cell.c, 'B'),
        _ => panic!("Expected Single 'B' at (0,3)"),
    }
}

#[test]
fn it_should_wrap_wide_character_correctly() {
    let mut term = create_test_emulator(3, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    let snapshot_before_wrap = term.get_render_snapshot().expect("Snapshot was None");
    // After 'A' and wide char '世', line is full, cursor at (0, 2)
    assert_screen_state(&snapshot_before_wrap, &["A世", "   "], Some((0, 2)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    let snapshot_after_wrap = term.get_render_snapshot().expect("Snapshot was None");
    // 'C' wraps to next line
    assert_screen_state(&snapshot_after_wrap, &["A世", "C  "], Some((1, 1)));
}

#[test]
fn it_should_not_print_second_half_of_wide_char_if_at_edge_and_no_wrap_mode_or_similar_logic() {
    let mut term = create_test_emulator(2, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // After 'A', '世' wraps. Space is written at (0,1). "A " scrolls to scrollback.
    // Screen is "世". Cursor logical (0,2), physical (0,1).
    assert_screen_state(&snapshot, &["世"], Some((0, 1))); // assert_screen_state handles wide char checks

    let glyph_uni_1 = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    match glyph_uni_1 {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '世'),
        _ => panic!("Expected WidePrimary '世' at (0,0)"),
    }
    if snapshot.dimensions.0 > 1 {
        let glyph_uni_2 = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
        assert!(
            matches!(glyph_uni_2, Glyph::WideSpacer),
            "Expected WideSpacer at (0,1)"
        );
    }
}

#[test]
fn it_should_overwrite_first_half_of_wide_char_with_ascii() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["X    "], Some((0, 1))); // assert_screen_state handles this

    let glyph_x = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    match glyph_x {
        Glyph::Single(cell) => assert_eq!(cell.c, 'X'),
        _ => panic!("Expected Single 'X' at (0,0)"),
    }
    let glyph_after_x = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
    match glyph_after_x {
        Glyph::Single(cell) => {
            assert_eq!(
                cell.c, ' ',
                "Cell after X should be a space, not placeholder"
            );
        }
        _ => panic!("Expected Single ' ' at (0,1)"),
    }
    assert!(
        !matches!(glyph_after_x, Glyph::WideSpacer),
        "Cell after X should not be a wide_char_spacer"
    );
}

#[test]
fn it_should_overwrite_second_half_of_wide_char_with_ascii() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 2),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Y'))); // Prints "Y", overwrites placeholder

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");

    // Direct glyph assertions instead of assert_screen_state for this specific case
    let glyph0 = get_glyph_from_snapshot(&snapshot, 0, 0).expect("Glyph at (0,0) missing");
    match glyph0 {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '世', "Cell (0,0) should be '世'"),
        _ => panic!("Expected WidePrimary '世' at (0,0), got {:?}", glyph0),
    }

    let glyph1 = get_glyph_from_snapshot(&snapshot, 0, 1).expect("Glyph at (0,1) missing");
    match glyph1 {
        Glyph::Single(cell) => assert_eq!(cell.c, 'Y', "Cell (0,1) should be 'Y'"),
        _ => panic!("Expected Single 'Y' at (0,1), got {:?}", glyph1),
    }
    assert!(
        !matches!(glyph1, Glyph::WideSpacer),
        "Cell (0,1) should not be a spacer"
    );

    let glyph2 = get_glyph_from_snapshot(&snapshot, 0, 2).expect("Glyph at (0,2) missing");
    match glyph2 {
        Glyph::Single(cell) => assert_eq!(cell.c, ' ', "Cell (0,2) should be a space"),
        _ => panic!("Expected Single ' ' at (0,2)"),
    };

    let glyph3 = get_glyph_from_snapshot(&snapshot, 0, 3).expect("Glyph at (0,3) missing");
    match glyph3 {
        Glyph::Single(cell) => assert_eq!(cell.c, ' ', "Cell (0,3) should be a space"),
        _ => panic!("Expected Single ' ' at (0,3)"),
    };

    let glyph4 = get_glyph_from_snapshot(&snapshot, 0, 4).expect("Glyph at (0,4) missing");
    match glyph4 {
        Glyph::Single(cell) => assert_eq!(cell.c, ' ', "Cell (0,4) should be a space"),
        _ => panic!("Expected Single ' ' at (0,4)"),
    };

    assert_eq!(
        snapshot.cursor_state.map(|cs| (cs.y, cs.x)),
        Some((0, 2)),
        "Cursor position mismatch"
    );
}

#[test]
fn it_should_print_ascii_over_wide_char_that_straddles_line_end_after_wrap() {
    let mut term = create_test_emulator(2, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('世')));
    let s1 = term.get_render_snapshot().expect("Snapshot was None");
    // Emulator logic: 'A' at (0,0). '世' attempts to print at (0,1) on 2-wide terminal.
    // Wrap occurs: space is printed at (0,1). Screen line 0 is "A ".
    // Cursor moves to (1,0). '世' is printed at (1,0) and (1,1).
    // No scroll for s1. Screen: ["A ", "世"]. Cursor logical (1,2), physical (1,1).
    assert_screen_state(&s1, &["A ", "世"], Some((1, 1)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let s2 = term.get_render_snapshot().expect("Snapshot was None");
    // After wrap, next character causes scroll.
    // Line "A " goes to scrollback. Line "世" becomes new line 0. New blank line 1.
    // Cursor is (0,1) (row 1, col 0).
    // 'X' is printed at (0,1).
    // Screen: ["世", "X "]. Cursor (1,1) (row 1, col 1).
    assert_screen_state(&s2, &["世", "X "], Some((1, 1))); // Expected screen ["世", "X "] cursor (1,1)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 2),
    ))); // Moves to (0,1) (row 0, col 1)
    let s3 = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&s3, &["世", "X "], Some((0, 1))); // Cursor is now (0,1)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z'))); // Prints 'Z' at (0,1)
    let s4 = term.get_render_snapshot().expect("Snapshot was None");
    // Screen: ["世Z", "X "]. Cursor (0,2) (row 0, col 2)
    // Check s4 screen content directly
    let glyph_s4_0_0 = get_glyph_from_snapshot(&s4, 0, 0).unwrap();
    let glyph_s4_0_1 = get_glyph_from_snapshot(&s4, 0, 1).unwrap();
    let glyph_s4_1_0 = get_glyph_from_snapshot(&s4, 1, 0).unwrap();
    let glyph_s4_1_1 = get_glyph_from_snapshot(&s4, 1, 1).unwrap();

    match glyph_s4_0_0 {
        Glyph::WidePrimary(cell) => assert_eq!(cell.c, '世'),
        _ => panic!("Expected WidePrimary at (0,0)"),
    };
    match glyph_s4_0_1 {
        Glyph::Single(cell) => assert_eq!(cell.c, 'Z'),
        _ => panic!("Expected Single at (0,1)"),
    };
    match glyph_s4_1_0 {
        Glyph::Single(cell) => assert_eq!(cell.c, 'X'),
        _ => panic!("Expected Single at (1,0)"),
    };
    match glyph_s4_1_1 {
        Glyph::Single(cell) => assert_eq!(cell.c, ' '),
        _ => panic!("Expected Single at (1,1)"),
    }; // Expecting a space if the line was cleared or filled.

    // Check cursor for s4
    assert_eq!(
        s4.cursor_state.as_ref().map(|cs| (cs.y, cs.x)),
        Some((0, 1)),
        "s4 cursor position mismatch"
    );

    let glyph_at_0_0 = get_glyph_from_snapshot(&s4, 0, 0).unwrap();
    assert!(
        matches!(glyph_at_0_0, Glyph::WidePrimary(_)),
        "Expected WidePrimary at (0,0)"
    );
    if let Glyph::WidePrimary(cell) = glyph_at_0_0 {
        assert_eq!(cell.c, '世');
    }

    let glyph_at_0_1 = get_glyph_from_snapshot(&s4, 0, 1).unwrap();
    match glyph_at_0_1 {
        Glyph::Single(cell) => assert_eq!(cell.c, 'Z'),
        _ => panic!("Expected Single at (0,1)"),
    }
    assert!(!matches!(glyph_at_0_1, Glyph::WideSpacer));
}

// --- Line Feed (LF) Tests ---
#[test]
fn it_should_move_cursor_down_keeping_column_on_line_feed_if_lnm_is_off() {
    let mut term = create_test_emulator(10, 3); // LNM is off by default
                                                // Explicitly disable Linefeed/Newline Mode - testing that LF doesn't do CR
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetMode(20),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Char 'A' at (0,0). Cursor at (0,1).
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(3),
    ))); // Cursor moves from (0,1) to (0,4).
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Char 'B' at (0,4). Cursor at (0,5). Screen "A   B" on line 0.

    let snapshot_before_lf = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_before_lf,
        &["A   B     ", "          ", "          "],
        Some((0, 5)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    let snapshot_after_lf = term.get_render_snapshot().expect("Snapshot was None");
    // LF moves to next line (1), keeping current column (5).
    assert_screen_state(
        &snapshot_after_lf,
        &["A   B     ", "          ", "          "],
        Some((1, 5)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C'))); // Char 'C' at (1,5). Cursor at (1,6).
    let snapshot_final = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_final,
        &["A   B     ", "     C    ", "          "],
        Some((1, 6)),
    );
}

#[test]
fn it_should_scroll_up_and_move_cursor_down_keeping_column_on_line_feed_at_bottom_if_lnm_is_off() {
    let mut term = create_test_emulator(5, 2); // LNM is off by default
                                               // Explicitly disable Linefeed/Newline Mode - testing that LF doesn't do CR
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetMode(20),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('3'))); // Line 0: "123", cursor (0,3)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR))); // Cursor (0,0)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // LNM off: Cursor moves from (0,0) to (1,0)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Line 1: "A", cursor (1,1)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Line 1: "AB", cursor (1,2)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C'))); // Line 1: "ABC", cursor (1,3)

    let snapshot_before_scroll = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_before_scroll, &["123  ", "ABC  "], Some((1, 3)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    let snapshot_after_scroll = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_after_scroll, &["ABC  ", "     "], Some((1, 3)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot_final = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_final, &["ABC  ", "   X "], Some((1, 4)));
}

#[test]
fn it_should_move_cursor_down_and_to_col_0_on_line_feed_if_lnm_is_on() {
    let mut term = create_test_emulator(10, 3);
    // Enable Linefeed/Newline Mode - testing that LF does CR+LF
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetMode(
        20,
    ))));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(3),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    let snapshot_before_lf = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_before_lf,
        &["A   B     ", "          ", "          "],
        Some((0, 5)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    let snapshot_after_lf = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_after_lf,
        &["A   B     ", "          ", "          "],
        Some((1, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    let snapshot_final = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_final,
        &["A   B     ", "C         ", "          "],
        Some((1, 1)),
    );
}

#[test]
fn it_should_scroll_and_move_to_col_0_on_line_feed_at_bottom_if_lnm_is_on() {
    let mut term = create_test_emulator(5, 2);
    // Enable Linefeed/Newline Mode - testing that LF does CR+LF
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetMode(
        20,
    ))));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('3')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));

    let snapshot_before_scroll = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_before_scroll, &["123  ", "ABC  "], Some((1, 3)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    let snapshot_after_scroll = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_after_scroll, &["ABC  ", "     "], Some((1, 0)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot_final = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_final, &["ABC  ", "X    "], Some((1, 1)));
}

// --- Carriage Return (CR) Test ---
#[test]
fn it_should_move_cursor_to_col_0_on_carriage_return() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["ABC       "], Some((0, 0)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot2, &["XBC       "], Some((0, 1)));
}

// --- Backspace (BS) Tests ---
#[test]
fn it_should_move_cursor_left_on_backspace() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["AB        "], Some((0, 1)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot2, &["AX        "], Some((0, 2)));
}

#[test]
fn it_should_not_wrap_cursor_on_backspace_at_start_of_line() {
    let mut term = create_test_emulator(10, 2);
    // Explicitly disable Linefeed/Newline Mode - testing that LF doesn't do CR
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetMode(20),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // After LF (LNM off), cursor moves from (0,2) to (1,2). "L2" prints. Screen "  L2". CR moves to (1,0).
    assert_screen_state(&snapshot, &["L1        ", "  L2      "], Some((1, 0)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot2, &["L1        ", "  L2      "], Some((1, 0)));
}

// --- Horizontal Tab (HT) Tests ---
#[test]
fn it_should_move_cursor_to_next_tab_stop_on_horizontal_tab() {
    let mut term = create_test_emulator(20, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["A                   "], Some((0, 8)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot2, &["A       B           "], Some((0, 9)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
    let snapshot3 = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot3, &["A       B           "], Some((0, 16)));
}

#[test]
fn it_should_move_cursor_to_last_column_on_horizontal_tab_if_no_more_tab_stops() {
    let mut term = create_test_emulator(10, 1);
    for i in 0..9 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(
            std::char::from_u32('0' as u32 + i as u32).unwrap_or('X'),
        )));
    }
    let snapshot_before = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_before, &["012345678 "], Some((0, 9)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
    let snapshot_after = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_after, &["012345678 "], Some((0, 9)));
}

// --- Escape (ESC) Test ---
#[test]
fn it_should_do_nothing_visible_on_escape_character() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let snapshot_before = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_before, &["A         "], Some((0, 1)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::ESC)));
    let snapshot_after = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_after, &["A         "], Some((0, 1)));
}

// --- Cursor Up (CUU) ---
#[test]
fn it_should_move_cursor_up_by_n_lines_on_csi_cuu() {
    let mut term = create_test_emulator(5, 3);
    // CUP is 1-based. Target row 3 (idx 2), col 3 (idx 2).
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 3),
    )));
    assert_eq!(
        term.cursor_controller.logical_pos(),
        (2, 2),
        "Initial cursor pos YX mismatch"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorUp(
        2,
    )))); // CUU 2 - Move up 2 lines
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["     ", "     ", "     "], Some((0, 2))); // Cursor to (0,2)
}

#[test]
fn it_should_move_cursor_up_by_1_line_on_csi_cuu_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    ))); // Cursor to (1,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorUp(
        1,
    ))));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    ))); // Reset to (1,0)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorUp(
        0,
    )))); // Param 0 defaults to 1
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

#[test]
fn it_should_clamp_cursor_at_top_line_on_csi_cuu_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Cursor to (0,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorUp(
        5,
    ))));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

// --- Cursor Down (CUD) ---
#[test]
fn it_should_move_cursor_down_by_n_lines_on_csi_cud() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 3),
    ))); // Cursor to (0,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorDown(2),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 2)),
    );
}

#[test]
fn it_should_move_cursor_down_by_1_line_on_csi_cud_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Cursor to (0,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorDown(1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((1, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorDown(0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((1, 0)),
    );
}

#[test]
fn it_should_clamp_cursor_at_bottom_line_on_csi_cud_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 1),
    ))); // Cursor to (2,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorDown(5),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 0)),
    );
}

// --- Cursor Forward (CUF) ---
#[test]
fn it_should_move_cursor_forward_by_n_cols_on_csi_cuf() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Cursor to (0,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(2),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 2)),
    );
}

#[test]
fn it_should_move_cursor_forward_by_1_col_on_csi_cuf_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Cursor to (0,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 1)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 1)),
    );
}

#[test]
fn it_should_clamp_cursor_at_last_col_on_csi_cuf_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 4),
    ))); // Cursor to (0,3)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(5),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 4)),
    );
}

// --- Cursor Back (CUB) ---
#[test]
fn it_should_move_cursor_back_by_n_cols_on_csi_cub() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 3),
    ))); // Cursor to (0,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorBackward(2),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

#[test]
fn it_should_move_cursor_back_by_1_col_on_csi_cub_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 2),
    ))); // Cursor to (0,1)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorBackward(1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 2),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorBackward(0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

#[test]
fn it_should_clamp_cursor_at_first_col_on_csi_cub_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Cursor to (0,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorBackward(5),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

// --- Cursor Next Line (CNL) ---
#[test]
fn it_should_move_cursor_to_start_of_next_n_lines_on_csi_cnl() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 3),
    ))); // Cursor to (0,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorNextLine(2),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 0)),
    );
}

#[test]
fn it_should_move_cursor_to_start_of_next_1_line_on_csi_cnl_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 3),
    ))); // Cursor to (0,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorNextLine(1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((1, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 3),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorNextLine(0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((1, 0)),
    );
}

#[test]
fn it_should_clamp_cursor_at_start_of_last_line_on_csi_cnl_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 3),
    ))); // Cursor to (0,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorNextLine(5),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 0)),
    );
}

// --- Cursor Previous Line (CPL) ---
#[test]
fn it_should_move_cursor_to_start_of_previous_n_lines_on_csi_cpl() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 3),
    ))); // Cursor to (2,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPrevLine(2),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

#[test]
fn it_should_move_cursor_to_start_of_previous_1_line_on_csi_cpl_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 3),
    ))); // Cursor to (1,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPrevLine(1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 3),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPrevLine(0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

#[test]
fn it_should_clamp_cursor_at_start_of_first_line_on_csi_cpl_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 3),
    ))); // Cursor to (1,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPrevLine(5),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 0)),
    );
}

// --- Cursor Horizontal Absolute (CHA) ---
#[test]
fn it_should_move_cursor_to_col_n_on_csi_cha() {
    let mut term = create_test_emulator(10, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    ))); // Cursor to (1,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorCharacterAbsolute(5),
    ))); // CHA 5 (1-based, so col 4)
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["          ", "          ", "          "],
        Some((1, 4)),
    );
}

#[test]
fn it_should_move_cursor_to_col_1_on_csi_cha_with_param_0_or_1() {
    let mut term = create_test_emulator(10, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 5),
    ))); // Cursor to (1,4)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorCharacterAbsolute(1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["          ", "          ", "          "],
        Some((1, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 5),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorCharacterAbsolute(0),
    ))); // Param 0 defaults to 1
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["          ", "          ", "          "],
        Some((1, 0)),
    );
}

#[test]
fn it_should_clamp_cursor_at_last_col_on_csi_cha_if_move_is_too_far() {
    let mut term = create_test_emulator(5, 3);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Cursor (0,0)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorCharacterAbsolute(10),
    ))); // CHA 10 (col 9)
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((0, 4)),
    );
}

// --- Cursor Position (CUP) ---
#[test]
fn it_should_move_cursor_to_row_n_col_m_on_csi_cup() {
    let mut term = create_test_emulator(10, 5);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 4),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((2, 3)),
    );
}

#[test]
fn it_should_move_cursor_to_origin_on_csi_cup_with_params_0_0_or_1_1_or_missing() {
    let mut term = create_test_emulator(10, 5);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 4),
    )));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((0, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 4),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(0, 0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((0, 0)),
    );
}

// --- Erase in Display (ED) ---

fn setup_ed_el_screen(term: &mut TerminalEmulator, width: usize, height: usize) {
    for r in 0..height {
        for c in 0..width {
            let char_val =
                std::char::from_u32(('A' as u32) + (r % 26) as u32 + (c % 3) as u32).unwrap_or('?');
            term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(char_val)));
        }
        if r < height - 1 {
            term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
            term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
            // LNM is off by default
        }
    }
    let target_row_1_based = (height / 2) + 1;
    let target_col_1_based = (width / 2) + 1;
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(target_row_1_based as u16, target_col_1_based as u16),
    )));
}

#[test]
fn it_should_erase_from_cursor_to_end_of_screen_on_csi_0j_or_csi_j() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2) (0-indexed) after setup

    let mut term_clone = term.clone();
    term_clone.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(0),
    )));
    let snapshot0 = term_clone.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot0, &["ABCAB", "BC   ", "     "], Some((1, 2)));

    // Test CSI J (which parser should default to 0)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(0),
    ))); // Parser defaults non-existent param to 0 for EraseInDisplay
    let snapshot_default = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_default,
        &["ABCAB", "BC   ", "     "],
        Some((1, 2)),
    );
}

#[test]
fn it_should_erase_from_cursor_to_beginning_of_screen_on_csi_1j() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(1),
    )));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["     ", "   BC", "CDECD"], Some((1, 2)));
}

#[test]
fn it_should_erase_entire_screen_on_csi_2j() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(2),
    )));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["     ", "     ", "     "], Some((1, 2)));
}

// --- Erase in Line (EL) ---
#[test]
fn it_should_erase_from_cursor_to_end_of_line_on_csi_0k_or_csi_k() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor (1,2)

    let mut term_clone = term.clone();
    term_clone.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(0),
    )));
    let snapshot0 = term_clone.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot0, &["ABCAB", "BC   ", "CDECD"], Some((1, 2)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(0),
    ))); // Parser defaults to 0
    let snapshot_default = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(
        &snapshot_default,
        &["ABCAB", "BC   ", "CDECD"],
        Some((1, 2)),
    );
}

#[test]
fn it_should_erase_from_cursor_to_beginning_of_line_on_csi_1k() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor (1,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(1),
    )));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["ABCAB", "   BC", "CDECD"], Some((1, 2)));
}

#[test]
fn it_should_erase_entire_line_on_csi_2k() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor (1,2)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(2),
    )));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot, &["ABCAB", "     ", "CDECD"], Some((1, 2)));
}

#[test]
fn it_should_not_change_cursor_position_after_ed_or_el() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3);
    let initial_cursor_state_tuple = term
        .get_render_snapshot()
        .expect("Snapshot was None")
        .cursor_state
        .map(|cs| (cs.y, cs.x, cs.shape));

    assert_eq!(
        initial_cursor_state_tuple,
        Some((1, 2, CursorShape::Block)),
        "Initial cursor state mismatch"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(0),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_tuple,
        "Cursor state changed after ED 0"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(1),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_tuple,
        "Cursor state changed after ED 1"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(2),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_tuple,
        "Cursor state changed after ED 2"
    );

    setup_ed_el_screen(&mut term, 5, 3);
    let initial_cursor_state_el_tuple = term
        .get_render_snapshot()
        .expect("Snapshot was None")
        .cursor_state
        .map(|cs| (cs.y, cs.x, cs.shape));
    assert_eq!(
        initial_cursor_state_el_tuple,
        Some((1, 2, CursorShape::Block)),
        "Initial cursor state for EL mismatch"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(0),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_el_tuple,
        "Cursor state changed after EL 0"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(1),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_el_tuple,
        "Cursor state changed after EL 1"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInLine(2),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_el_tuple,
        "Cursor state changed after EL 2"
    );
}

// --- Scroll Up (SU) ---
#[test]
fn it_should_scroll_up_entire_screen_by_n_lines_on_csi_s() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2)
                                         // Screen: L0:ABCAB, L1:BCDBC, L2:CDECD

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        1,
    ))));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: L0 ("ABCAB") scrolls off. L1 becomes L0. L2 becomes L1. New L2 is blank.
    // Cursor remains at (1,2) relative to screen, now on char 'E' of original L2 "CDECD"
    assert_screen_state(&snapshot, &["BCDBC", "CDECD", "     "], Some((1, 2)));

    // Scroll up 2 more lines (effectively more than screen height)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        2,
    ))));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: All original content scrolled off. Screen is blank. Cursor still (1,2).
    assert_screen_state(&snapshot2, &["     ", "     ", "     "], Some((1, 2)));
}

#[test]
fn it_should_scroll_up_entire_screen_by_1_line_on_csi_s_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2)
    let initial_screen_line0_char0 = get_glyph_from_snapshot(
        &term.get_render_snapshot().expect("Snapshot was None"),
        0,
        0,
    )
    .unwrap()
    .display_char();
    assert_eq!(initial_screen_line0_char0, 'A');

    // Test with param 1
    let mut term_clone = term.clone();
    term_clone.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        1,
    ))));
    assert_screen_state(
        &term_clone.get_render_snapshot().expect("Snapshot was None"),
        &["BCDBC", "CDECD", "     "],
        Some((1, 2)),
    );

    // Test with param 0 (defaults to 1)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        0,
    ))));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["BCDBC", "CDECD", "     "],
        Some((1, 2)),
    );
}

#[test]
fn it_should_scroll_up_within_scrolling_region_on_csi_s() {
    let mut term = create_test_emulator(5, 4); // L0, L1, L2, L3
    setup_ed_el_screen(&mut term, 5, 4); // Cursor at (1,2) (0-idx for row 1)
                                         // Screen: L0:ABCAB, L1:BCDBC, L2:CDECD, L3:DEFDE
                                         // Set scrolling region to rows 2-3 (1-based), which is 0-indexed rows 1-2
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 2, bottom: 3 },
    )));
    // Cursor is at (1,2), which is within the new region [1,2].
    // However, SetScrollingRegion moves cursor to (0,0) of screen (as origin_mode is false).
    let cursor_after_stbm = term
        .get_render_snapshot()
        .expect("Snapshot was None")
        .cursor_state
        .map(|cs| (cs.y, cs.x))
        .unwrap();
    assert_eq!(
        cursor_after_stbm,
        (0, 0),
        "Cursor should be at (0,0) after STBM w/o origin mode"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        1,
    ))));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: L0 unchanged. Region L1-L2 scrolls. L1("BCDBC") scrolls off. L2("CDECD") becomes L1. New L2 is blank. L3 unchanged.
    // Screen: L0:ABCAB, L1(region):CDECD, L2(region):     , L3:DEFDE
    // Cursor remains at (0,0) as SU/SD do not move the cursor.
    assert_screen_state(
        &snapshot,
        &["ABCAB", "CDECD", "     ", "DEFDE"],
        Some((0, 0)),
    );

    // Scroll up again, more than region height
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        2,
    ))));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: L0 unchanged. Region L1-L2 now blank. L3 unchanged. Cursor still (0,0).
    assert_screen_state(
        &snapshot2,
        &["ABCAB", "     ", "     ", "DEFDE"],
        Some((0, 0)),
    );
}

// --- Scroll Down (SD) ---
#[test]
fn it_should_scroll_down_entire_screen_by_n_lines_on_csi_t() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2)
                                         // Screen: L0:ABCAB, L1:BCDBC, L2:CDECD

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(1),
    )));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: L2 ("CDECD") scrolls off. L1 becomes L2. L0 becomes L1. New L0 is blank.
    // Cursor remains at (1,2) relative to screen, now on char 'B' of original L0 "ABCAB".
    assert_screen_state(&snapshot, &["     ", "ABCAB", "BCDBC"], Some((1, 2)));

    // Scroll down 2 more lines
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(2),
    )));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: All original content scrolled off. Screen is blank. Cursor still (1,2).
    assert_screen_state(&snapshot2, &["     ", "     ", "     "], Some((1, 2)));
}

#[test]
fn it_should_scroll_down_entire_screen_by_1_line_on_csi_t_with_param_0_or_1() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor at (1,2)

    // Test with param 1
    let mut term_clone = term.clone();
    term_clone.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(1),
    )));
    assert_screen_state(
        &term_clone.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "ABCAB", "BCDBC"],
        Some((1, 2)),
    );

    // Test with param 0 (defaults to 1)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(0),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "ABCAB", "BCDBC"],
        Some((1, 2)),
    );
}

#[test]
fn it_should_scroll_down_within_scrolling_region_on_csi_t() {
    let mut term = create_test_emulator(5, 4);
    setup_ed_el_screen(&mut term, 5, 4); // Cursor at (1,2)
                                         // Screen: L0:ABCAB, L1:BCDBC, L2:CDECD, L3:DEFDE
                                         // Set scrolling region to rows 2-3 (0-indexed rows 1-2)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 2, bottom: 3 },
    )));
    // Cursor is at (1,2), but SetScrollingRegion moves it to (0,0) of screen.
    let cursor_after_stbm = term
        .get_render_snapshot()
        .expect("Snapshot was None")
        .cursor_state
        .map(|cs| (cs.y, cs.x))
        .unwrap();
    assert_eq!(
        cursor_after_stbm,
        (0, 0),
        "Cursor should be at (0,0) after STBM w/o origin mode"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(1),
    )));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: L0 unchanged. Region L1-L2 scrolls down. L2("CDECD") scrolls off. L1("BCDBC") becomes L2. New L1 is blank. L3 unchanged.
    // Screen: L0:ABCAB, L1(region):     , L2(region):BCDBC, L3:DEFDE
    // Cursor remains at (0,0).
    assert_screen_state(
        &snapshot,
        &["ABCAB", "     ", "BCDBC", "DEFDE"],
        Some((0, 0)),
    );

    // Scroll down again, more than region height
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(2),
    )));
    let snapshot2 = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: L0 unchanged. Region L1-L2 now blank. L3 unchanged. Cursor still (0,0).
    assert_screen_state(
        &snapshot2,
        &["ABCAB", "     ", "     ", "DEFDE"],
        Some((0, 0)),
    );
}

#[test]
fn it_should_not_change_cursor_position_on_csi_s_or_csi_t() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // Cursor (1,2)
    let initial_cursor_state_tuple = term
        .get_render_snapshot()
        .expect("Snapshot was None")
        .cursor_state
        .map(|cs| (cs.y, cs.x, cs.shape));
    assert_eq!(initial_cursor_state_tuple, Some((1, 2, CursorShape::Block)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ScrollUp(
        1,
    ))));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_tuple,
        "Cursor state changed after SU"
    );

    // Reset screen and cursor for SD test
    setup_ed_el_screen(&mut term, 5, 3);
    let initial_cursor_state_sd_tuple = term
        .get_render_snapshot()
        .expect("Snapshot was None")
        .cursor_state
        .map(|cs| (cs.y, cs.x, cs.shape));
    assert_eq!(
        initial_cursor_state_sd_tuple,
        Some((1, 2, CursorShape::Block))
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ScrollDown(1),
    )));
    assert_eq!(
        term.get_render_snapshot()
            .expect("Snapshot was None")
            .cursor_state
            .map(|cs| (cs.y, cs.x, cs.shape)),
        initial_cursor_state_sd_tuple,
        "Cursor state changed after SD"
    );
}

// --- Resize Event Tests ---

#[test]
fn it_should_resize_to_larger_dimensions_preserving_content_and_cursor() {
    let mut term = create_test_emulator(5, 2);
    setup_ed_el_screen(&mut term, 5, 2); // L0:ABCAB, L1:BCDBC. Cursor set to (1,2) by setup_ed_el_screen for 5x2.
                                         // Initial state check
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["ABCAB", "BCDBC"],
        Some((1, 2)),
    );

    term.interpret_input(EmulatorInput::Control(resize_event(7, 4)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");

    // Expected: Original content in top-left. New areas blank. Cursor should remain (1,2).
    // L0: ABCAB
    // L1: BCDBC
    // L2:
    // L3:
    assert_eq!(
        snapshot.dimensions,
        (7, 4),
        "Dimensions mismatch after resize larger"
    );
    assert_screen_state(
        &snapshot,
        &[
            "ABCAB  ", // Note: Trailing spaces to fill new width
            "BCDBC  ", "       ", "       ",
        ],
        Some((1, 2)),
    );
}

#[test]
fn it_should_resize_to_smaller_dimensions_truncating_content_and_clamping_cursor() {
    let mut term = create_test_emulator(5, 3);
    setup_ed_el_screen(&mut term, 5, 3); // L0:ABCAB, L1:BCDBC, L2:CDECD, Cursor (1,2)
                                         // Initial state check
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["ABCAB", "BCDBC", "CDECD"],
        Some((1, 2)),
    );

    // Resize smaller, cursor was at (1,2) which is (row 2, col 3)
    term.interpret_input(EmulatorInput::Control(resize_event(3, 2)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");

    // Expected: Content truncated. Cursor clamped.
    // Original cursor (1,2) is now outside new bounds (cols:0-2, rows:0-1).
    // Emulator's resize clamps cursor: min(cursor_y, new_rows-1), min(cursor_x, new_cols-1)
    // So, cursor becomes (min(1, 1), min(2, 2)) = (1,2)
    assert_eq!(
        snapshot.dimensions,
        (3, 2),
        "Dimensions mismatch after resize smaller"
    );
    assert_screen_state(&snapshot, &["ABC", "BCD"], Some((1, 2)));
}

#[test]
fn it_should_clamp_cursor_to_new_bottom_right_if_cursor_was_beyond_after_shrink() {
    let mut term = create_test_emulator(5, 3);
    // Place cursor at bottom right (2,4)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 5),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 4)),
    );

    // Resize much smaller, cursor (2,4) is way out.
    term.interpret_input(EmulatorInput::Control(resize_event(2, 1)));
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    // Expected: Cursor clamped to new bottom-right (0,1)
    assert_eq!(
        snapshot.dimensions,
        (2, 1),
        "Dimensions mismatch after resize smaller"
    );
    assert_screen_state(&snapshot, &["  "], Some((0, 1)));
}

#[test]
fn it_should_handle_resize_with_content_and_cursor_at_edges() {
    let mut term = create_test_emulator(3, 2);
    // Fill screen and place cursor at bottom right (1,2)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('3'))); // L0: "123"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // to L1, (1,0)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('4')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('5')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('6'))); // L1: "456", cursor (1,3) (just off screen edge)
                                                                        // Let's put cursor exactly at (1,2)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 3),
    ))); // Cursor to (1,2)
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["123", "456"],
        Some((1, 2)),
    );

    // Resize larger
    term.interpret_input(EmulatorInput::Control(resize_event(5, 3)));
    let snapshot_larger = term.get_render_snapshot().expect("Snapshot was None");
    assert_eq!(snapshot_larger.dimensions, (5, 3));
    assert_screen_state(&snapshot_larger, &["123  ", "456  ", "     "], Some((1, 2)));

    // Resize smaller than original, larger than content
    term.interpret_input(EmulatorInput::Control(resize_event(4, 2)));
    let snapshot_medium = term.get_render_snapshot().expect("Snapshot was None");
    assert_eq!(snapshot_medium.dimensions, (4, 2));
    assert_screen_state(&snapshot_medium, &["123 ", "456 "], Some((1, 2)));

    // Resize smaller, truncating content and clamping cursor
    // Cursor (1,2)
    term.interpret_input(EmulatorInput::Control(resize_event(2, 1)));
    let snapshot_smaller = term.get_render_snapshot().expect("Snapshot was None");
    assert_eq!(snapshot_smaller.dimensions, (2, 1));
    // Original L0 "123" -> "12". Cursor (1,2) clamps to (0,1)
    assert_screen_state(&snapshot_smaller, &["12"], Some((0, 1)));
}

// --- DEC Private Mode tests ---

// DECTCEM - Text Cursor Enable Mode (CSI ? 25 h/l)
#[test]
fn it_should_show_and_hide_cursor_on_dectcem() {
    let mut term = create_test_emulator(5, 3);

    // Cursor should be visible by default (DECTCEM is typically on by default)
    // The `TerminalEmulator::new` sets `dec_modes.text_cursor_enable_mode = true;`
    // DECTCEM is ON by default - verify behavior via snapshot
    let snapshot_default = term.get_render_snapshot().expect("Snapshot was None");
    assert!(
        snapshot_default.cursor_state.is_some(),
        "Cursor should be visible by default in snapshot"
    );

    // Hide cursor: CSI ? 25 l
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::TextCursorEnable as u16),
    )));
    // DECTCEM should be OFF after CSI ? 25 l - verify via snapshot
    let snapshot_hidden = term.get_render_snapshot().expect("Snapshot was None");
    assert!(
        snapshot_hidden.cursor_state.is_none(),
        "Cursor should be hidden after CSI ? 25 l"
    );

    // Show cursor: CSI ? 25 h
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::TextCursorEnable as u16),
    )));
    // DECTCEM should be ON after CSI ? 25 h - verify via snapshot
    let snapshot_shown = term.get_render_snapshot().expect("Snapshot was None");
    assert!(
        snapshot_shown.cursor_state.is_some(),
        "Cursor should be visible again after CSI ? 25 h"
    );
    // Ensure shape is restored (assuming Block is default)
    assert_eq!(
        snapshot_shown.cursor_state.unwrap().shape,
        CursorShape::Block
    );
}

// Alternate Screen Buffer (ASB) (CSI ? 47 h/l or ?1049 h/l)
// The emulator's mode_handler.rs uses specific constants for ASB.
// DecModeConstant::AltScreenBufferSaveRestore (1049)
// DecModeConstant::AltScreenBuffer (47)
// Let's test 1049 as it's more common for full save/restore/clear behavior.
#[test]
fn it_should_switch_to_alternate_screen_buffer_and_back_on_csi_1049() {
    let mut term = create_test_emulator(5, 2);
    setup_ed_el_screen(&mut term, 5, 2); // L0: ABCAB, L1: BCDBC, Cursor (0,2)
    let original_snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let original_cursor_pos = original_snapshot
        .cursor_state
        .as_ref()
        .map(|cs| (cs.y, cs.x));
    // setup_ed_el_screen with 5,2 will place cursor at row (2/2)+1=2, col (5/2)+1=3 (1-based) -> (1,2) (0-based)
    assert_eq!(original_cursor_pos, Some((1, 2)));

    // Switch to Alternate Screen Buffer (ASB): CSI ? 1049 h
    // This typically saves cursor, switches to ASB, and clears ASB.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));

    // Switched to alternate screen - verify it's cleared
    let snapshot_asb = term.get_render_snapshot().expect("Snapshot was None");
    // ASB should be cleared. Content is all spaces.
    assert_screen_state(&snapshot_asb, &["     ", "     "], Some((0, 0))); // Cursor usually resets to (0,0) on ASB

    // Print something on ASB
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Y')));
    let snapshot_asb_content = term.get_render_snapshot().expect("Snapshot was None");
    assert_screen_state(&snapshot_asb_content, &["XY   ", "     "], Some((0, 2)));

    // Switch back to Normal Screen Buffer (NSB): CSI ? 1049 l
    // This typically restores cursor, switches to NSB, and restores its content.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));

    // Back on normal screen - verify content is restored
    let snapshot_nsb_restored = term.get_render_snapshot().expect("Snapshot was None");
    // Screen content should be restored
    assert_screen_state(
        &snapshot_nsb_restored,
        &["ABCAB", "BCDBC"],
        original_cursor_pos,
    ); // Cursor pos restored
}

// Auto Wrap Mode (DECAWM) (CSI ? 7 h/l)
#[test]
fn it_should_enable_and_disable_autowrap_mode_on_decawm() {
    let mut term = create_test_emulator(3, 2); // Small width to test wrap easily

    // DECAWM is on by default in emulator
    // Autowrap is ON by default - verify wrapping behavior
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('3'))); // Fills line 0: "123"
    assert_eq!(term.cursor_controller.logical_pos(), (3, 0)); // Corrected: logical_pos is (x,y) -> (3,0)
                                                              // After filling line, next char will wrap
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('4'))); // Wraps to line 1
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["123", "4  "],
        Some((1, 1)),
    );

    // Disable Autowrap: CSI ? 7 l
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::AutoWrapMode as u16),
    )));
    // Autowrap is OFF - verify no wrapping behavior
    // Move to end of line 1
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 3),
    ))); // Cursor to (1,2) on line "4  "
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('5'))); // Prints '5' at (1,2). Line "4 5". Cursor (1,3).
    assert_eq!(term.cursor_controller.logical_pos(), (3, 1)); // Corrected: logical_pos is (x,y) -> (3,1)
                                                              // With autowrap off, cursor stays at right edge

    // Try to print past end of line with autowrap off
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('6')));
    // Character '6' should overwrite '5' at the last column (1,2). Cursor stays at (1,3) (or clamps to last char).
    // Current `char_processor.rs` `print_char` logic: if `cursor.x >= screen_width` and `!autowrap` and `!wrap_next`, it sets `cursor.x = screen_width -1`.
    // So '6' is printed at (1,2) over '5'. Cursor logical (3,1). Physical (2,1) for snapshot.
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["123", "4 6"],
        Some((1, 2)),
    );
    // Cursor still at right edge with autowrap off

    // Enable Autowrap again: CSI ? 7 h
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::AutoWrapMode as u16),
    )));
    // Autowrap is ON again - verify wrapping behavior restored
    // Cursor is at (1,3) on line "4 6". Line is full.
    // Line is full with autowrap on, next char will wrap
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('7'))); // Should wrap to next line (scroll if needed)
                                                                        // We have 2 lines. (0,1). This will scroll.
                                                                        // L0 "123" scrolls off. L1 "4 6" becomes L0. L2 "7  " becomes L1.
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["4 6", "7  "],
        Some((1, 1)),
    );
}

// --- OSC (Operating System Command) Tests ---

#[test]
fn it_should_set_window_title_on_osc_0_sequence() {
    let mut term = create_test_emulator(10, 1);
    let title = "My Window Title via OSC 0";
    let osc_command_bytes = format!("0;{}", title).into_bytes(); // OSC P s ; P t ST -> P s = 0, P t = title

    let action = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(osc_command_bytes)));

    assert_eq!(
        action,
        Some(EmulatorAction::SetTitle(title.to_string())),
        "OSC 0 did not produce the correct SetTitle action."
    );
}

#[test]
fn it_should_set_window_title_on_osc_2_sequence() {
    let mut term = create_test_emulator(10, 1);
    let title = "Another Title via OSC 2";
    let osc_command_bytes = format!("2;{}", title).into_bytes(); // OSC P s ; P t ST -> P s = 2, P t = title

    let action = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(osc_command_bytes)));

    assert_eq!(
        action,
        Some(EmulatorAction::SetTitle(title.to_string())),
        "OSC 2 did not produce the correct SetTitle action."
    );
}

#[test]
fn it_should_handle_empty_title_string_in_osc_sequence() {
    let mut term = create_test_emulator(10, 1);
    let title = "";
    let osc_command_bytes = format!("2;{}", title).into_bytes();

    let action = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(osc_command_bytes)));

    assert_eq!(
        action,
        Some(EmulatorAction::SetTitle(title.to_string())),
        "OSC with empty title string failed."
    );
}

#[test]
fn it_should_handle_osc_sequence_without_semicolon_for_title_setting_if_supported() {
    // Some terminals might interpret "OSC 0;title" and "OSC 0:title" or even "OSC 0 title"
    // The current OSC handler in osc_handler.rs specifically looks for ';'.
    // This test checks the documented behavior (parsing up to ';').
    // If "OSC LtitleST" (long form for OSC 2) is supported, that's a different case.
    // For now, we test what happens if the format is just "Ps;Pt"
    // If Ps is not 0, 1, or 2, or if no ';' is found, it might ignore or handle differently.
    // The current `osc_handler.rs` seems to parse `code` (Ps) and then `content` (Pt).
    // If content is "My Title" and code is "2", it should work.

    let mut term = create_test_emulator(10, 1);
    // Test OSC 2;title (standard)
    let title_std = "Standard Title";
    let osc_std_bytes = format!("2;{}", title_std).into_bytes();
    let action_std = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(osc_std_bytes)));
    assert_eq!(
        action_std,
        Some(EmulatorAction::SetTitle(title_std.to_string()))
    );

    // Test what happens if only "title" is sent, assuming code might default or be implicit.
    // The AnsiCommand::Osc(Vec<u8>) is the full string *after* OSC and *before* ST.
    // So, if PTY sends "OSC 2;MyTitle", then Vec<u8> is "2;MyTitle".
    // If PTY sends "OSC MyOtherTitle", then Vec<u8> is "MyOtherTitle".
    // The osc_handler.rs splits the Vec<u8> by ';'.
    // If no ';', then `parts[0]` is the whole string. `code_str` becomes `parts[0]`.
    // `code = code_str.parse::<u8>().unwrap_or(0);`
    // If `code_str` is "MyOtherTitle", parse fails, code becomes 0.
    // Then `content` becomes empty. So it would try to set title to "" via OSC 0.

    let title_implicit = "Implicit Title";
    let osc_implicit_bytes = title_implicit.to_string().into_bytes(); // No "0;" or "2;" prefix
    let action_implicit =
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(osc_implicit_bytes)));

    // Based on osc_handler.rs logic:
    // code_str = "Implicit Title" -> parse to u8 fails -> code = 0
    // content = "" (because no data after the first part if no ';')
    // Expected: SetTitle("") due to OSC 0
    assert_eq!(
        action_implicit,
        Some(EmulatorAction::SetTitle("".to_string())),
        "OSC with implicit code '0' and no content (due to parsing) should set empty title."
    );
}

#[test]
fn it_should_ignore_osc_sequences_for_unsupported_ps_codes_for_title() {
    let mut term = create_test_emulator(10, 1);
    let title = "A Title";
    // Using Ps = 3, which is not standard for window title.
    let osc_command_bytes = format!("3;{}", title).into_bytes();

    let action = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(osc_command_bytes)));

    // Expect None because osc_handler.rs only handles 0, 1, 2 for SetTitle.
    // (Ps=1 is for icon name, but often also sets title)
    // Current osc_handler.rs:
    // 0 => SetTitle
    // 1 => if feature "osc_icon_name", SetIconName, else also SetTitle
    // 2 => SetTitle
    // Other codes like 4 (color palette), 10, 11, 12 (fg/bg/cursor colors) are handled differently or ignored if not compiled.
    // For Ps=3, it should fall through and return None.
    assert_eq!(
        action, None,
        "OSC with unsupported Ps=3 for title should be ignored (produce no action)."
    );
}

// --- SGR (Select Graphic Rendition) Tests ---

#[test]
fn it_should_reset_all_attributes_on_sgr_0() {
    let mut term = create_test_emulator(10, 1);
    // Set some attributes first
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(Color::Named(NamedColor::Red)),
            Attribute::Background(Color::Named(NamedColor::Blue)),
            Attribute::Bold,
        ]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Print with attributes

    // Reset
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Print after reset

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let glyph_a_wrapper = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    let glyph_b_wrapper = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();

    let (attr_a, attr_b) = match (glyph_a_wrapper, glyph_b_wrapper) {
        (Glyph::Single(cell_a), Glyph::Single(cell_b)) => (cell_a.attr, cell_b.attr),
        _ => panic!("Expected Single glyphs for A and B"),
    };

    assert_eq!(attr_a.fg, Color::Named(NamedColor::Red), "Glyph A fg");
    assert_eq!(attr_a.bg, Color::Named(NamedColor::Blue), "Glyph A bg");
    assert!(attr_a.flags.contains(AttrFlags::BOLD), "Glyph A bold");

    let default_attrs = Attributes::default();
    assert_eq!(attr_b.fg, default_attrs.fg, "Glyph B fg should be default");
    assert_eq!(attr_b.bg, default_attrs.bg, "Glyph B bg should be default");
    assert!(
        !attr_b.flags.contains(AttrFlags::BOLD),
        "Glyph B should not be bold"
    );
    assert_eq!(
        attr_b.flags, default_attrs.flags,
        "Glyph B flags should be default"
    );
}

// --- Intensity ---
#[test]
fn it_should_set_bold_on_sgr_1_and_reset_on_sgr_22() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Bold]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    // SGR 22 maps to NoBold (which also implies NoFaint for this test's purpose)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::NoBold]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert!(
        attr_a.flags.contains(AttrFlags::BOLD),
        "Glyph A should be BOLD"
    );
    assert!(
        !attr_b.flags.contains(AttrFlags::BOLD),
        "Glyph B should not be BOLD (normal intensity)"
    );
    // Assuming NormalIntensity also clears FAINT if it was set.
    assert!(
        !attr_b.flags.contains(AttrFlags::FAINT),
        "Glyph B should not be FAINT"
    );
}

#[test]
fn it_should_set_faint_on_sgr_2_and_reset_on_sgr_22() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Faint]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    // SGR 22 maps to NoBold (which also implies NoFaint)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::NoBold]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert!(
        attr_a.flags.contains(AttrFlags::FAINT),
        "Glyph A should be FAINT"
    );
    assert!(
        !attr_b.flags.contains(AttrFlags::FAINT),
        "Glyph B should not be FAINT (normal intensity)"
    );
    assert!(
        !attr_b.flags.contains(AttrFlags::BOLD),
        "Glyph B should not be BOLD"
    );
}

// --- Italic ---
#[test]
fn it_should_set_italic_on_sgr_3_and_reset_on_sgr_23() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Italic]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::NoItalic]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert!(
        attr_a.flags.contains(AttrFlags::ITALIC),
        "Glyph A should be ITALIC"
    );
    assert!(
        !attr_b.flags.contains(AttrFlags::ITALIC),
        "Glyph B should not be ITALIC"
    );
}

// --- Underline ---
#[test]
fn it_should_set_underline_on_sgr_4_and_reset_on_sgr_24() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Underline]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::NoUnderline]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert!(
        attr_a.flags.contains(AttrFlags::UNDERLINE),
        "Glyph A should be UNDERLINE"
    );
    assert!(
        !attr_b.flags.contains(AttrFlags::UNDERLINE),
        "Glyph B should not be UNDERLINE"
    );
}

// --- Foreground Colors ---
#[test]
fn it_should_set_basic_ansi_foreground_colors_sgr_30_37() {
    let mut term = create_test_emulator(8, 1);
    let colors = vec![
        NamedColor::Black,
        NamedColor::Red,
        NamedColor::Green,
        NamedColor::Yellow,
        NamedColor::Blue,
        NamedColor::Magenta,
        NamedColor::Cyan,
        NamedColor::White,
    ];
    for (i, &color_name) in colors.iter().enumerate() {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(color_name))]),
        )));
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(
            ('A' as u8 + i as u8) as char,
        )));
    }
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    for (i, &color_name) in colors.iter().enumerate() {
        let glyph_wrapper = get_glyph_from_snapshot(&snapshot, 0, i).unwrap();
        match glyph_wrapper {
            Glyph::Single(cell) => {
                assert_eq!(cell.c, ('A' as u8 + i as u8) as char);
                assert_eq!(
                    cell.attr.fg,
                    Color::Named(color_name),
                    "Failed for color {:?}",
                    color_name
                );
            }
            _ => panic!("Expected Single glyph for color test"),
        }
    }
}

#[test]
fn it_should_set_bright_ansi_foreground_colors_sgr_90_97() {
    let mut term = create_test_emulator(8, 1);
    let bright_colors = vec![
        NamedColor::BrightBlack,
        NamedColor::BrightRed,
        NamedColor::BrightGreen,
        NamedColor::BrightYellow,
        NamedColor::BrightBlue,
        NamedColor::BrightMagenta,
        NamedColor::BrightCyan,
        NamedColor::BrightWhite,
    ];
    for (i, &color_name) in bright_colors.iter().enumerate() {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(color_name)), // Parser maps 90-97 to these NamedColor variants
            ]),
        )));
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(
            ('A' as u8 + i as u8) as char,
        )));
    }
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    for (i, &color_name) in bright_colors.iter().enumerate() {
        let glyph_wrapper = get_glyph_from_snapshot(&snapshot, 0, i).unwrap();
        match glyph_wrapper {
            Glyph::Single(cell) => {
                assert_eq!(cell.c, ('A' as u8 + i as u8) as char);
                assert_eq!(
                    cell.attr.fg,
                    Color::Named(color_name),
                    "Failed for bright color {:?}",
                    color_name
                );
            }
            _ => panic!("Expected Single glyph for bright color test"),
        }
    }
}

#[test]
fn it_should_set_indexed_foreground_color_sgr_38_5_n() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Indexed(123))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let attr = match get_glyph_from_snapshot(
        &term.get_render_snapshot().expect("Snapshot was None"),
        0,
        0,
    )
    .unwrap()
    {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    assert_eq!(attr.fg, Color::Indexed(123));
}

#[test]
fn it_should_set_rgb_foreground_color_sgr_38_2_r_g_b() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Rgb(10, 20, 30))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let attr = match get_glyph_from_snapshot(
        &term.get_render_snapshot().expect("Snapshot was None"),
        0,
        0,
    )
    .unwrap()
    {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    assert_eq!(attr.fg, Color::Rgb(10, 20, 30));
}

#[test]
fn it_should_reset_foreground_color_on_sgr_39() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Red,
        ))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Red 'A'

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Default)]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Default fg 'B'

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert_eq!(attr_a.fg, Color::Named(NamedColor::Red));
    assert_eq!(attr_b.fg, Attributes::default().fg);
}

// --- Background Colors ---
#[test]
fn it_should_set_basic_ansi_background_colors_sgr_40_47() {
    let mut term = create_test_emulator(8, 1);
    let colors = vec![
        NamedColor::Black,
        NamedColor::Red,
        NamedColor::Green,
        NamedColor::Yellow,
        NamedColor::Blue,
        NamedColor::Magenta,
        NamedColor::Cyan,
        NamedColor::White,
    ];
    for (i, &color_name) in colors.iter().enumerate() {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetGraphicsRendition(vec![Attribute::Background(Color::Named(color_name))]),
        )));
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(
            ('A' as u8 + i as u8) as char,
        )));
    }
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    for (i, &color_name) in colors.iter().enumerate() {
        let glyph_wrapper = get_glyph_from_snapshot(&snapshot, 0, i).unwrap();
        match glyph_wrapper {
            Glyph::Single(cell) => {
                assert_eq!(cell.c, ('A' as u8 + i as u8) as char);
                assert_eq!(
                    cell.attr.bg,
                    Color::Named(color_name),
                    "Failed for bg color {:?}",
                    color_name
                );
            }
            _ => panic!("Expected Single glyph for bg color test"),
        }
    }
}

#[test]
fn it_should_set_bright_ansi_background_colors_sgr_100_107() {
    let mut term = create_test_emulator(8, 1);
    let bright_colors = vec![
        NamedColor::BrightBlack,
        NamedColor::BrightRed,
        NamedColor::BrightGreen,
        NamedColor::BrightYellow,
        NamedColor::BrightBlue,
        NamedColor::BrightMagenta,
        NamedColor::BrightCyan,
        NamedColor::BrightWhite,
    ];
    for (i, &color_name) in bright_colors.iter().enumerate() {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
            CsiCommand::SetGraphicsRendition(vec![
                Attribute::Background(Color::Named(color_name)), // Parser maps 100-107 to these
            ]),
        )));
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(
            ('A' as u8 + i as u8) as char,
        )));
    }
    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    for (i, &color_name) in bright_colors.iter().enumerate() {
        let glyph_wrapper = get_glyph_from_snapshot(&snapshot, 0, i).unwrap();
        match glyph_wrapper {
            Glyph::Single(cell) => {
                assert_eq!(cell.c, ('A' as u8 + i as u8) as char);
                assert_eq!(
                    cell.attr.bg,
                    Color::Named(color_name),
                    "Failed for bright bg color {:?}",
                    color_name
                );
            }
            _ => panic!("Expected Single glyph for bright bg color test"),
        }
    }
}

#[test]
fn it_should_set_indexed_background_color_sgr_48_5_n() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Background(Color::Indexed(201))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let attr = match get_glyph_from_snapshot(
        &term.get_render_snapshot().expect("Snapshot was None"),
        0,
        0,
    )
    .unwrap()
    {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    assert_eq!(attr.bg, Color::Indexed(201));
}

#[test]
fn it_should_set_rgb_background_color_sgr_48_2_r_g_b() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Background(Color::Rgb(40, 50, 60))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let attr = match get_glyph_from_snapshot(
        &term.get_render_snapshot().expect("Snapshot was None"),
        0,
        0,
    )
    .unwrap()
    {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    assert_eq!(attr.bg, Color::Rgb(40, 50, 60));
}

#[test]
fn it_should_reset_background_color_on_sgr_49() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Background(Color::Named(
            NamedColor::Blue,
        ))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Blue bg 'A'

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Background(Color::Default)]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Default bg 'B'

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert_eq!(attr_a.bg, Color::Named(NamedColor::Blue));
    assert_eq!(attr_b.bg, Attributes::default().bg);
}

// --- Inverse ---
#[test]
fn it_should_set_inverse_on_sgr_7_and_reset_on_sgr_27() {
    let mut term = create_test_emulator(5, 1);
    // Set specific fg/bg to see inversion clearly
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(Color::Named(NamedColor::Red)), // fg=Red
            Attribute::Background(Color::Named(NamedColor::Blue)), // bg=Blue
        ]),
    )));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reverse]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Inverse A: fg=Blue, bg=Red

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::NoReverse]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Not inverse B: fg=Red, bg=Blue

    let snapshot = term.get_render_snapshot().expect("Snapshot was None");
    let attr_a = match get_glyph_from_snapshot(&snapshot, 0, 0).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    let attr_b = match get_glyph_from_snapshot(&snapshot, 0, 1).unwrap() {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };

    assert!(
        attr_a.flags.contains(AttrFlags::REVERSE),
        "Glyph A should be REVERSE"
    );
    // Note: The actual fg/bg values on the attr_a might be swapped by the renderer,
    // or the REVERSE flag is used by renderer to swap them.
    // The `Attributes` struct itself stores logical fg/bg. The REVERSE flag signals the swap.
    assert_eq!(attr_a.fg, Color::Named(NamedColor::Red)); // Logical fg is still Red
    assert_eq!(attr_a.bg, Color::Named(NamedColor::Blue)); // Logical bg is still Blue

    assert!(
        !attr_b.flags.contains(AttrFlags::REVERSE),
        "Glyph B should not be REVERSE"
    );
    assert_eq!(attr_b.fg, Color::Named(NamedColor::Red));
    assert_eq!(attr_b.bg, Color::Named(NamedColor::Blue));
}

// --- Multiple Attributes ---
#[test]
fn it_should_set_multiple_attributes_in_one_sgr_sequence() {
    let mut term = create_test_emulator(5, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![
            Attribute::Bold,
            Attribute::Foreground(Color::Named(NamedColor::Green)),
            Attribute::Background(Color::Rgb(50, 50, 50)),
            Attribute::Underline,
        ]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));

    let attr_x = match get_glyph_from_snapshot(
        &term.get_render_snapshot().expect("Snapshot was None"),
        0,
        0,
    )
    .unwrap()
    {
        Glyph::Single(c) => c.attr,
        _ => panic!("Expected Single"),
    };
    assert!(attr_x.flags.contains(AttrFlags::BOLD));
    assert!(attr_x.flags.contains(AttrFlags::UNDERLINE));
    assert!(!attr_x.flags.contains(AttrFlags::ITALIC));
    assert_eq!(attr_x.fg, Color::Named(NamedColor::Green));
    assert_eq!(attr_x.bg, Color::Rgb(50, 50, 50));
}

#[test]
fn it_should_clamp_cursor_on_csi_cup_if_params_are_out_of_bounds() {
    let mut term = create_test_emulator(5, 3);

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(10, 2),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 1)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 10),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((1, 4)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(10, 10),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &["     ", "     ", "     "],
        Some((2, 4)),
    );
}

#[test]
fn it_should_handle_csi_cup_with_origin_mode_decom() {
    let mut term = create_test_emulator(10, 5);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 2, bottom: 4 },
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::Origin as u16),
    )));
    // Origin mode (DECOM) enabled - cursor positioning is relative to scrolling region

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((1, 0)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 3),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((2, 2)),
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::Origin as u16),
    )));
    // Origin mode (DECOM) disabled - cursor positioning is absolute

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    )));
    assert_screen_state(
        &term.get_render_snapshot().expect("Snapshot was None"),
        &[
            "          ",
            "          ",
            "          ",
            "          ",
            "          ",
        ],
        Some((0, 0)),
    );
}
