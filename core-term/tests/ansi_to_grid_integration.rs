//! Integration tests: ANSI commands → Terminal grid
//!
//! These tests inject text and ANSI commands directly into the terminal emulator
//! and verify that the grid state is updated correctly.

mod support;

use core_term::ansi::commands::{AnsiCommand, C0Control, CsiCommand};
use support::minimal_test_harness::MinimalTestHarness;

#[test]
fn it_should_place_a_printed_character_at_the_cursor_and_advance_the_cursor() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print a single character
    harness.print_text("a");

    // VERIFY: Grid has the character at (0, 0)
    let snapshot = harness
        .get_snapshot()
        .expect("Should have terminal snapshot");

    assert_eq!(
        snapshot.lines[0].cells[0].display_char(),
        'a',
        "Character 'a' should be at position (0, 0)"
    );

    // Cursor should have advanced
    if let Some(cursor) = &snapshot.cursor_state {
        assert_eq!(cursor.x, 1);
        assert_eq!(cursor.y, 0);
    }
}

#[test]
fn it_should_print_a_sequence_of_characters_left_to_right() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print multiple characters
    harness.print_text("Hello");

    // VERIFY: Grid has "Hello"
    let snapshot = harness.get_snapshot().unwrap();

    let text: String = snapshot.lines[0]
        .cells
        .iter()
        .take(5)
        .map(|cell| cell.display_char())
        .collect();

    assert_eq!(text, "Hello", "Grid should contain 'Hello'");
}

#[test]
fn it_should_advance_to_the_next_grid_row_on_line_feed() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print, newline, print again
    harness.print_text("A");
    harness.inject_ansi(AnsiCommand::C0Control(C0Control::LF));
    harness.print_text("B");

    // VERIFY: 'A' on row 0, 'B' on row 1
    let snapshot = harness.get_snapshot().unwrap();

    assert_eq!(snapshot.lines[0].cells[0].display_char(), 'A');
    assert_eq!(snapshot.lines[1].cells[0].display_char(), 'B');
}

#[test]
fn it_should_move_the_cursor_to_the_position_specified_by_cup() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Move cursor to (5, 10), then print
    harness.inject_ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(5, 10)));
    harness.print_text("X");

    // VERIFY: 'X' at position (9, 4) [0-indexed]
    let snapshot = harness.get_snapshot().unwrap();

    // CursorPosition is 1-indexed, grid is 0-indexed
    assert_eq!(snapshot.lines[4].cells[9].display_char(), 'X');
}

#[test]
fn it_should_wrap_to_a_new_grid_line_on_each_lf_cr_pair() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print text with newlines
    let text = "Line1\nLine2\nLine3";
    for ch in text.chars() {
        if ch == '\n' {
            harness.inject_ansi(AnsiCommand::C0Control(C0Control::LF));
            harness.inject_ansi(AnsiCommand::C0Control(C0Control::CR));
        } else {
            harness.print_text(ch.encode_utf8(&mut [0; 4]));
        }
    }

    // VERIFY: Three lines of text
    let snapshot = harness.get_snapshot().unwrap();

    let line1: String = snapshot.lines[0]
        .cells
        .iter()
        .take(5)
        .map(|c| c.display_char())
        .collect();

    let line2: String = snapshot.lines[1]
        .cells
        .iter()
        .take(5)
        .map(|c| c.display_char())
        .collect();

    let line3: String = snapshot.lines[2]
        .cells
        .iter()
        .take(5)
        .map(|c| c.display_char())
        .collect();

    assert_eq!(line1, "Line1");
    assert_eq!(line2, "Line2");
    assert_eq!(line3, "Line3");
}

// =============================================================================
// Grid Checksum Tests - Verify grid state actually changes
// =============================================================================

#[test]
fn it_should_change_the_grid_checksum_after_each_of_several_prints() {
    let mut harness = MinimalTestHarness::new();

    // Get initial checksum (empty grid)
    let checksum1 = harness.compute_grid_checksum();

    // Print a character
    harness.print_text("a");
    let checksum2 = harness.compute_grid_checksum();

    // Checksums should be DIFFERENT
    assert_ne!(
        checksum1, checksum2,
        "Grid checksum should change after printing 'a'"
    );

    // Print another character
    harness.print_text("b");
    let checksum3 = harness.compute_grid_checksum();

    // Checksum should change again
    assert_ne!(
        checksum2, checksum3,
        "Grid checksum should change after printing 'b'"
    );

    // All three checksums should be unique
    assert_ne!(checksum1, checksum3);
}

#[test]
fn it_should_report_the_same_checksum_when_the_grid_is_unchanged() {
    let mut harness = MinimalTestHarness::new();

    // Print a character
    harness.print_text("x");

    // Get checksum twice without changes
    let checksum1 = harness.compute_grid_checksum();
    let checksum2 = harness.compute_grid_checksum();

    // Should be the same
    assert_eq!(
        checksum1, checksum2,
        "Grid checksum should be stable when no changes made"
    );
}

#[test]
fn it_should_change_the_grid_checksum_after_each_character_printed() {
    let mut harness = MinimalTestHarness::new();

    let mut checksums = Vec::new();

    // Print "Hello" one character at a time, checking checksum after each
    for ch in "Hello".chars() {
        harness.print_text(ch.encode_utf8(&mut [0; 4]));
        checksums.push(harness.compute_grid_checksum());
    }

    // All checksums should be unique
    for i in 0..checksums.len() {
        for j in (i + 1)..checksums.len() {
            assert_ne!(
                checksums[i], checksums[j],
                "Checksum after char {} should differ from char {}",
                i, j
            );
        }
    }
}
