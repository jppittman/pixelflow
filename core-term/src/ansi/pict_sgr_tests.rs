// src/ansi/pict_sgr_tests.rs

//! Worked example for the PICT-style generator in [`super::pict`]: model SGR
//! sequences as independent factors, generate a pairwise covering array, and
//! check the parser against a reference oracle for every generated case.
//!
//! SGR is an ideal target — a single `ESC[...m` carries many independent,
//! order-preserving parameters (intensity, italic, underline, blink,
//! foreground, background, ...). The number of combinations is astronomical;
//! pairwise coverage exercises every 2-way interaction with a handful of cases.

use super::pict::pairwise;
use super::{
    commands::{AnsiCommand, Attribute, CsiCommand},
    AnsiParser, AnsiProcessor,
};
use crate::color::{Color, NamedColor};
use test_log::test;

/// One selectable value of a factor: the SGR fragment it emits (if any) and the
/// attributes a correct parser must produce for it, in order.
struct Level {
    /// SGR text this level contributes, e.g. `"38;5;5"`. `None` means "absent".
    fragment: Option<&'static str>,
    /// Reference output: what a correct parser yields for this fragment alone.
    expected: Vec<Attribute>,
}

fn lvl(fragment: Option<&'static str>, expected: &[Attribute]) -> Level {
    Level {
        fragment,
        expected: expected.to_vec(),
    }
}

/// The SGR factor model. Each inner slice is one factor and its levels.
fn sgr_factors() -> Vec<Vec<Level>> {
    use Attribute::*;
    let red = Color::Named(NamedColor::Red);
    let bright_red = Color::Named(NamedColor::BrightRed);
    vec![
        // Intensity
        vec![
            lvl(None, &[]),
            lvl(Some("1"), &[Bold]),
            lvl(Some("2"), &[Faint]),
            lvl(Some("22"), &[NoBold]),
        ],
        // Italic
        vec![
            lvl(None, &[]),
            lvl(Some("3"), &[Italic]),
            lvl(Some("23"), &[NoItalic]),
        ],
        // Underline
        vec![
            lvl(None, &[]),
            lvl(Some("4"), &[Underline]),
            lvl(Some("21"), &[UnderlineDouble]),
            lvl(Some("24"), &[NoUnderline]),
        ],
        // Blink
        vec![
            lvl(None, &[]),
            lvl(Some("5"), &[BlinkSlow]),
            lvl(Some("25"), &[NoBlink]),
        ],
        // Foreground (basic / bright / default / 256 / truecolor)
        vec![
            lvl(None, &[]),
            lvl(Some("31"), &[Foreground(red)]),
            lvl(Some("91"), &[Foreground(bright_red)]),
            lvl(Some("39"), &[Foreground(Color::Default)]),
            lvl(Some("38;5;5"), &[Foreground(Color::Indexed(5))]),
            lvl(Some("38;2;10;20;30"), &[Foreground(Color::Rgb(10, 20, 30))]),
        ],
        // Background
        vec![
            lvl(None, &[]),
            lvl(Some("41"), &[Background(red)]),
            lvl(Some("49"), &[Background(Color::Default)]),
            lvl(Some("48;5;7"), &[Background(Color::Indexed(7))]),
        ],
        // Miscellaneous, including an SGR code the emulator does not implement.
        vec![
            lvl(None, &[]),
            lvl(Some("7"), &[Reverse]),
            lvl(Some("9"), &[Strikethrough]),
            lvl(Some("53"), &[Overlined]),
            // 73 = "superscript" (mintty/ECMA-48:2024). Unimplemented here; a
            // correct parser must *ignore* an unknown SGR code, never reset.
            lvl(Some("73"), &[]),
        ],
    ]
}

fn process(bytes: &[u8]) -> Vec<AnsiCommand> {
    AnsiProcessor::new().process_bytes(bytes)
}

/// Build the `ESC[...m` byte sequence for a chosen row, and its oracle output.
fn build_case(factors: &[Vec<Level>], row: &[usize]) -> (Vec<u8>, Vec<Attribute>) {
    let mut fragments: Vec<&str> = Vec::new();
    let mut expected: Vec<Attribute> = Vec::new();
    for (factor, &level_idx) in factors.iter().zip(row) {
        let level = &factor[level_idx];
        let Some(frag) = level.fragment else {
            continue;
        };
        fragments.push(frag);
        expected.extend_from_slice(&level.expected);
    }

    let mut seq = b"\x1b[".to_vec();
    seq.extend_from_slice(fragments.join(";").as_bytes());
    seq.push(b'm');

    // Oracle: an empty parameter list (`ESC[m`) is the spec-defined full reset.
    // Any non-empty list of *recognized* attributes yields exactly those
    // attributes; a list of only-unrecognized codes must be a no-op (`[]`).
    let oracle = if fragments.is_empty() {
        vec![Attribute::Reset]
    } else {
        expected
    };
    (seq, oracle)
}

/// Extract the SGR attribute list a sequence parses to, or `None` if the parse
/// did not yield a single SGR command.
fn parsed_attrs(seq: &[u8]) -> Option<Vec<Attribute>> {
    let commands = process(seq);
    match commands.as_slice() {
        [AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(attrs))] => Some(attrs.clone()),
        _ => None,
    }
}

#[test]
fn pict_sgr_pairwise_matches_oracle() {
    let factors = sgr_factors();
    let level_counts: Vec<usize> = factors.iter().map(Vec::len).collect();
    let rows = pairwise(&level_counts);

    let mut failures: Vec<String> = Vec::new();
    for row in &rows {
        let (seq, oracle) = build_case(&factors, row);
        let actual = parsed_attrs(&seq);
        let seq_str = String::from_utf8_lossy(&seq).replace('\x1b', "ESC");
        match actual {
            Some(attrs) if attrs == oracle => {}
            Some(attrs) => failures.push(format!(
                "  {seq_str:<28} expected {oracle:?}\n{:>32}got      {attrs:?}",
                ""
            )),
            None => failures.push(format!(
                "  {seq_str:<28} expected {oracle:?}\n{:>32}got      <not a single SGR command>",
                ""
            )),
        }
    }

    assert!(
        failures.is_empty(),
        "PICT pairwise SGR testing found {} of {} generated cases mismatched:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n"),
    );
}

/// Focused regression for the defect the pairwise sweep surfaces: an SGR
/// sequence whose codes are all *unrecognized* must be ignored, not turned into
/// a full attribute reset.
#[test]
fn unknown_only_sgr_is_ignored_not_reset() {
    // `ESC[73m` — a single unimplemented code. Correct behavior: no-op.
    assert_eq!(
        parsed_attrs(b"\x1b[73m"),
        Some(vec![]),
        "an unknown SGR code must be ignored, but it was rewritten into a reset",
    );
}

/// Sibling to the above that exposes the *inconsistency*: the same unknown code
/// is silently dropped when it shares the sequence with a recognized attribute,
/// so `ESC[73m` and `ESC[1;73m` disagree on how `73` is handled.
#[test]
fn unknown_sgr_is_dropped_when_mixed_with_known() {
    assert_eq!(
        parsed_attrs(b"\x1b[1;73m"),
        Some(vec![Attribute::Bold]),
        "unknown code alongside a known one should drop to just the known attribute",
    );
}
