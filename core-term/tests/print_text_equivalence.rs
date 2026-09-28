//! `TerminalEmulator::print_text` must leave the terminal exactly as feeding
//! it the same text one character at a time does.
//!
//! Every scenario is applied twice from the same PTY bytes — once with whole
//! text runs, once character by character — and the rendered
//! snapshots must match. Each scenario ends with a probe character, so a
//! difference in pending-wrap state shows up as a difference on screen.

use core_term::ansi::{AnsiCommand, AnsiParser, AnsiProcessor, AnsiSink};
use core_term::term::{EmulatorInput, TerminalEmulator};

const COLS: usize = 10;
const ROWS: usize = 4;
const PROBE: &[u8] = b"Z";

/// Applies text as whole runs, the way the app does.
struct Runs<'a>(&'a mut TerminalEmulator);

impl AnsiSink for Runs<'_> {
    fn text(&mut self, run: &str) {
        self.0.print_text(run);
    }

    fn command(&mut self, command: AnsiCommand) {
        drop(self.0.interpret_input(EmulatorInput::Ansi(command)));
    }
}

/// Applies text one character per `print_text` call.
struct Chars<'a>(&'a mut TerminalEmulator);

impl AnsiSink for Chars<'_> {
    fn text(&mut self, run: &str) {
        for c in run.chars() {
            self.0.print_text(c.encode_utf8(&mut [0; 4]));
        }
    }

    fn command(&mut self, command: AnsiCommand) {
        drop(self.0.interpret_input(EmulatorInput::Ansi(command)));
    }
}

fn assert_equivalent(name: &str, bytes: &[u8]) {
    let input = [bytes, PROBE].concat();

    let mut by_runs = TerminalEmulator::new(COLS, ROWS);
    AnsiProcessor::new()
        .process_bytes(&input)
        .drain_into(&mut Runs(&mut by_runs));

    let mut by_chars = TerminalEmulator::new(COLS, ROWS);
    AnsiProcessor::new()
        .process_bytes(&input)
        .drain_into(&mut Chars(&mut by_chars));

    assert_eq!(
        by_runs.get_render_snapshot(),
        by_chars.get_render_snapshot(),
        "scenario `{name}` diverged"
    );
}

#[test]
fn short_text() {
    assert_equivalent("short", b"hello");
}

#[test]
fn text_that_exactly_fills_a_line() {
    assert_equivalent("exact width", b"0123456789");
}

#[test]
fn text_that_wraps_and_scrolls() {
    assert_equivalent(
        "wrap and scroll",
        b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ",
    );
}

#[test]
fn text_starting_mid_line() {
    assert_equivalent("mid line", b"\x1b[2;8Habcdefghij");
}

#[test]
fn text_with_autowrap_off() {
    assert_equivalent("autowrap off", b"\x1b[?7labcdefghijklmnopqrstuvwxyz");
}

#[test]
fn text_overwriting_wide_characters() {
    // Runs ending on a wide character's primary cell, on its spacer, and past it.
    assert_equivalent("end on primary", "世界世界世\ra".as_bytes());
    assert_equivalent("end on spacer", "世界世界世\rab".as_bytes());
    assert_equivalent("end past pair", "世界世界世\rabc".as_bytes());
}

#[test]
fn text_mixed_with_wide_and_combining_characters() {
    assert_equivalent("mixed", "ab世cde\u{301}fgh界ij".as_bytes());
}

#[test]
fn text_under_a_non_identity_charset() {
    assert_equivalent("dec line drawing", b"\x1b(0lqqk\x1b(Bmx");
    assert_equivalent("uk national", b"\x1b(A#1#2\x1b(B#3");
}

#[test]
fn text_inside_a_scroll_region_with_origin_mode() {
    assert_equivalent(
        "origin mode",
        b"\x1b[2;3r\x1b[?6habcdefghijklmnopqrstuvwxyz0123456789",
    );
}

#[test]
fn text_with_attributes() {
    assert_equivalent("sgr", b"\x1b[1;31mred\x1b[0mplain\x1b[7minverse");
}

#[test]
fn text_on_the_alternate_screen() {
    assert_equivalent("alt screen", b"main\x1b[?1049halternate text here");
}
