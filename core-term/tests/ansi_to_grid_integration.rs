//! Integration tests: ANSI commands → Terminal grid
//!
//! These tests inject ANSI commands directly into the terminal emulator
//! and verify that the grid state is updated correctly.

mod support;

use core_term::ansi::commands::{AnsiCommand, C0Control, CsiCommand};
use support::minimal_test_harness::MinimalTestHarness;

#[test]
fn test_single_character_print() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print a single character
    harness.inject_ansi(AnsiCommand::Print('a'));

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
fn test_multiple_characters() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print multiple characters
    harness.inject_ansi_batch(vec![
        AnsiCommand::Print('H'),
        AnsiCommand::Print('e'),
        AnsiCommand::Print('l'),
        AnsiCommand::Print('l'),
        AnsiCommand::Print('o'),
    ]);

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
fn test_newline_advances_row() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print, newline, print again
    harness.inject_ansi_batch(vec![
        AnsiCommand::Print('A'),
        AnsiCommand::C0Control(C0Control::LF),
        AnsiCommand::Print('B'),
    ]);

    // VERIFY: 'A' on row 0, 'B' on row 1
    let snapshot = harness.get_snapshot().unwrap();

    assert_eq!(snapshot.lines[0].cells[0].display_char(), 'A');
    assert_eq!(snapshot.lines[1].cells[0].display_char(), 'B');
}

#[test]
fn test_cursor_position_command() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Move cursor to (5, 10), then print
    harness.inject_ansi_batch(vec![
        AnsiCommand::Csi(CsiCommand::CursorPosition(5, 10)),
        AnsiCommand::Print('X'),
    ]);

    // VERIFY: 'X' at position (9, 4) [0-indexed]
    let snapshot = harness.get_snapshot().unwrap();

    // CursorPosition is 1-indexed, grid is 0-indexed
    assert_eq!(snapshot.lines[4].cells[9].display_char(), 'X');
}

#[test]
fn test_multiline_text() {
    let mut harness = MinimalTestHarness::new();

    // TEST: Print text with newlines
    let text = "Line1\nLine2\nLine3";
    for ch in text.chars() {
        if ch == '\n' {
            harness.inject_ansi(AnsiCommand::C0Control(C0Control::LF));
            harness.inject_ansi(AnsiCommand::C0Control(C0Control::CR));
        } else {
            harness.inject_ansi(AnsiCommand::Print(ch));
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
fn test_grid_checksum_changes_on_input() {
    let mut harness = MinimalTestHarness::new();

    // Get initial checksum (empty grid)
    let checksum1 = harness.compute_grid_checksum();

    // Print a character
    harness.inject_ansi(AnsiCommand::Print('a'));
    let checksum2 = harness.compute_grid_checksum();

    // Checksums should be DIFFERENT
    assert_ne!(
        checksum1, checksum2,
        "Grid checksum should change after printing 'a'"
    );

    // Print another character
    harness.inject_ansi(AnsiCommand::Print('b'));
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
fn test_grid_checksum_stable_without_changes() {
    let mut harness = MinimalTestHarness::new();

    // Print a character
    harness.inject_ansi(AnsiCommand::Print('x'));

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
fn test_multiple_characters_change_checksum() {
    let mut harness = MinimalTestHarness::new();

    let mut checksums = Vec::new();

    // Print "Hello" one character at a time, checking checksum after each
    for ch in "Hello".chars() {
        harness.inject_ansi(AnsiCommand::Print(ch));
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

// =============================================================================
// Bug Reproduction: Grid changes but render not triggered
// =============================================================================

#[test]
fn test_bug_grid_changes_without_render_trigger() {
    // This test documents the ACTUAL BUG:
    // Grid state changes when ANSI commands are processed,
    // but send_frame() is never called, so the display doesn't update.

    let mut harness = MinimalTestHarness::new();

    // Simulate PTY output being processed in park()
    // In the real app, park() calls:
    //   self.emulator.interpret_input(EmulatorInput::Ansi(cmd))
    // But it NEVER calls send_frame() afterward!

    harness.inject_ansi(AnsiCommand::Print('a'));
    let checksum_after_a = harness.compute_grid_checksum();

    harness.inject_ansi(AnsiCommand::Print('b'));
    let checksum_after_b = harness.compute_grid_checksum();

    // Grid DOES change (this passes)
    assert_ne!(
        checksum_after_a, checksum_after_b,
        "BUG CONFIRMED: Grid changes but no render triggered"
    );

    // But in the real app, park() doesn't call send_frame(),
    // so the manifold is never rebuilt and the frame checksum stays the same!

    // THE FIX: park() should call self.send_frame() after processing PTY output
}
