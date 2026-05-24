// src/ansi/tests.rs

//! Tests for the ANSI parser and lexer integration.

// Use the AnsiProcessor, which combines lexer and parser
// Corrected imports using commands submodule path
use super::{
    commands::{AnsiCommand, Attribute, C0Control, CsiCommand, EscCommand},
    AnsiParser, AnsiProcessor,
};
use crate::color::{Color, NamedColor};
use test_log::test; // Ensure test_log is a dev-dependency for log capturing in tests

// Helper function to process bytes and get commands
fn process_bytes(bytes: &[u8]) -> Vec<AnsiCommand> {
    let mut processor = AnsiProcessor::new();
    processor.process_bytes(bytes)
}

#[test]
fn it_should_process_a_simple_printable_string() {
    let bytes = b"Hello, world!";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::Print('H'),
            AnsiCommand::Print('e'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print(','),
            AnsiCommand::Print(' '),
            AnsiCommand::Print('w'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print('r'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('d'),
            AnsiCommand::Print('!'),
        ]
    );
}

#[test]
fn it_should_process_c0_bel() {
    let bytes = b"\x07"; // BEL
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
}

#[test]
fn it_should_process_csi_h_as_cup_1_1() {
    let bytes = b"\x1B[H"; // CSI H -> CUP (1, 1)
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]
    );
}

#[test]
fn it_should_process_csi_sgr_reset() {
    let bytes = b"\x1B[0m";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Reset
        ]))]
    );
}

#[test]
fn it_should_process_csi_sgr_set_foreground() {
    let bytes = b"\x1B[34m";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(Color::Named(NamedColor::Blue))
        ]))]
    );
}

#[test]
fn it_should_process_dec_private_mode_reset_12_att610_cursor_blink() {
    let bytes = b"\x1b[?12l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(12))],
        "Expected ResetModePrivate(12) for CSI ?12l"
    );
}

#[test]
fn it_should_process_dec_private_mode_set_25_text_cursor_enable() {
    let bytes = b"\x1b[?25h";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))],
        "Expected SetModePrivate(25) for CSI ?25h"
    );
}

#[test]
fn it_should_process_dec_private_mode_reset_25_text_cursor_enable() {
    let bytes = b"\x1b[?25l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(25))],
        "Expected ResetModePrivate(25) for CSI ?25l"
    );
}

#[test]
fn it_should_process_various_dec_private_mouse_modes() {
    let modes_to_test = vec![
        (1000, b"\x1b[?1000h", b"\x1b[?1000l", "XTERM_MOUSE_CLICK"),
        (
            1002,
            b"\x1b[?1002h",
            b"\x1b[?1002l",
            "XTERM_MOUSE_BTN_MOTION",
        ),
        (
            1003,
            b"\x1b[?1003h",
            b"\x1b[?1003l",
            "XTERM_MOUSE_ANY_MOTION",
        ),
        (1005, b"\x1b[?1005h", b"\x1b[?1005l", "XTERM_MOUSE_UTF8"),
        (1006, b"\x1b[?1006h", b"\x1b[?1006l", "XTERM_MOUSE_SGR"),
    ];
    for (mode_num, set_seq, reset_seq, _name) in modes_to_test {
        let set_commands = process_bytes(set_seq);
        assert_eq!(
            set_commands,
            vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(mode_num))],
            "Expected SetModePrivate({}) for {:?}",
            mode_num,
            String::from_utf8_lossy(set_seq)
        );
        let reset_commands = process_bytes(reset_seq);
        assert_eq!(
            reset_commands,
            vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(mode_num))],
            "Expected ResetModePrivate({}) for {:?}",
            mode_num,
            String::from_utf8_lossy(reset_seq)
        );
    }
}

#[test]
fn it_should_process_dec_private_mode_bracketed_paste_2004() {
    let bytes_set = b"\x1b[?2004h";
    let commands_set = process_bytes(bytes_set);
    assert_eq!(
        commands_set,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(2004))],
        "Expected SetModePrivate(2004) for CSI ?2004h"
    );

    let bytes_reset = b"\x1b[?2004l";
    let commands_reset = process_bytes(bytes_reset);
    assert_eq!(
        commands_reset,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(2004))],
        "Expected ResetModePrivate(2004) for CSI ?2004l"
    );
}

#[test]
fn it_should_process_dec_private_mode_focus_event_1004() {
    let bytes_set = b"\x1b[?1004h";
    let commands_set = process_bytes(bytes_set);
    assert_eq!(
        commands_set,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(1004))],
        "Expected SetModePrivate(1004) for CSI ?1004h"
    );

    let bytes_reset = b"\x1b[?1004l";
    let commands_reset = process_bytes(bytes_reset);
    assert_eq!(
        commands_reset,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(1004))],
        "Expected ResetModePrivate(1004) for CSI ?1004l"
    );
}

#[test]
fn it_should_process_dec_private_mode_uncommon_7727() {
    let bytes = b"\x1b[?7727l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(7727))],
        "Expected ResetModePrivate(7727) for CSI ?7727l"
    );
}

#[test]
fn it_should_process_csi_set_cursor_style_decscusr() {
    let bytes_steady_block = b"\x1b[2 q";
    let commands_steady_block = process_bytes(bytes_steady_block);
    assert_eq!(
        commands_steady_block,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 2 })],
        "Expected SetCursorStyle for CSI 2 SP q"
    );

    let bytes_default_cursor = b"\x1b[0 q";
    let commands_default_cursor = process_bytes(bytes_default_cursor);
    assert_eq!(
        commands_default_cursor,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 0 })],
        "Expected SetCursorStyle for CSI 0 SP q"
    );

    let bytes_blink_underline = b"\x1b[3 q";
    let commands_blink_underline = process_bytes(bytes_blink_underline);
    assert_eq!(
        commands_blink_underline,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 3 })],
        "Expected SetCursorStyle for CSI 3 SP q"
    );
}

#[test]
fn it_should_process_csi_window_manipulation_t() {
    let bytes_23_0_0_t = b"\x1b[23;0;0t";
    let commands_23_0_0_t = process_bytes(bytes_23_0_0_t);
    assert_eq!(
        commands_23_0_0_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 23,
            ps2: Some(0),
            ps3: Some(0)
        })],
        "Expected WindowManipulation for CSI 23;0;0t"
    );

    let bytes_18_t = b"\x1b[18t";
    let commands_18_t = process_bytes(bytes_18_t);
    assert_eq!(
        commands_18_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 18,
            ps2: None,
            ps3: None
        })],
        "Expected WindowManipulation for CSI 18t"
    );

    let bytes_14_t = b"\x1b[14t";
    let commands_14_t = process_bytes(bytes_14_t);
    assert_eq!(
        commands_14_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 14,
            ps2: None,
            ps3: None
        })],
        "Expected WindowManipulation for CSI 14t"
    );
}

#[test]
fn it_should_process_csi_sequence_fragmented_across_param_bytes() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B[1");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC [ 1)");
    let commands_frag2 = processor.process_bytes(b";2H");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 2))],
        "After fragment 2 (;2H)"
    );
}

#[test]
fn it_should_process_csi_sequence_fragmented_across_intermediate_bytes() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B[?");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC [ ?)");
    let commands_frag2 = processor.process_bytes(b"25h");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))],
        "After fragment 2 (25h)"
    );
}

#[test]
fn it_should_process_csi_sequence_fragmented_after_esc() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC)");
    let commands_frag2 = processor.process_bytes(b"[1A");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::CursorUp(1))],
        "After fragment 2 ([1A)"
    );
}

#[test]
fn it_should_process_string_interspersed_with_fragmented_csi() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"Hello ");
    assert_eq!(
        commands_frag1,
        vec![
            AnsiCommand::Print('H'),
            AnsiCommand::Print('e'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print(' '),
        ],
        "After fragment 1 (Hello )"
    );
    let commands_frag2 = processor.process_bytes(b"\x1B[31");
    assert_eq!(commands_frag2, vec![], "After fragment 2 (ESC [ 31)");
    let commands_frag3 = processor.process_bytes(b"m World");
    assert_eq!(
        commands_frag3,
        vec![
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Red))
            ])),
            AnsiCommand::Print(' '),
            AnsiCommand::Print('W'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print('r'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('d'),
        ],
        "After fragment 3 (m World)"
    );
}

#[test]
fn it_should_handle_fragmented_utf8_input_with_intermediate_finalization() {
    // This test demonstrates how AnsiProcessor (which calls lexer.finalize()
    // within its process_bytes) handles UTF-8 fragments delivered in separate calls.
    let mut processor_refined = AnsiProcessor::new();
    assert_eq!(
        processor_refined.process_bytes(b"A"),
        vec![AnsiCommand::Print('A')],
        "Refined Frag 0: Print 'A'"
    );
    // \xE4 is start of '你'. Since it's an incomplete sequence when process_bytes finishes, finalize() converts it.
    assert_eq!(
        processor_refined.process_bytes(b"\xE4"),
        vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER)],
        "Refined Frag 1: Incomplete UTF-8 (E4) yields replacement char"
    );
    // \xBD is now treated as a new byte. It's an invalid UTF-8 start. finalize() converts it.
    assert_eq!(
        processor_refined.process_bytes(b"\xBD"),
        vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER)],
        "Refined Frag 2: Invalid UTF-8 start (BD) yields replacement char"
    );
    // \xA0 is also an invalid UTF-8 start. finalize() converts it.
    assert_eq!(
        processor_refined.process_bytes(b"\xA0"),
        vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER)],
        "Refined Frag 3: Invalid UTF-8 start (A0) yields replacement char"
    );
    assert_eq!(
        processor_refined.process_bytes(b"B"),
        vec![AnsiCommand::Print('B')],
        "Refined Frag 4: Print 'B'"
    );

    // For contrast, show how a complete multi-byte char is processed in one call
    let mut processor_complete = AnsiProcessor::new();
    assert_eq!(
        processor_complete.process_bytes(b"\xE4\xBD\xA0"),
        vec![AnsiCommand::Print('你')],
        "Complete '你' in one call"
    );
}

#[test]
fn it_should_complete_csi_if_final_byte_arrives_after_params() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B[31");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC [ 31)");
    let commands_frag2 = processor.process_bytes(b"A");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::CursorUp(31))],
        "After fragment 2 (A)"
    );
    let commands_frag3 = processor.process_bytes(b"BC");
    assert_eq!(
        commands_frag3,
        vec![AnsiCommand::Print('B'), AnsiCommand::Print('C')],
        "After fragment 3 (BC)"
    );
}

#[test]
fn it_should_complete_osc_if_terminator_arrives_after_string_fragment() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B]0;Ti");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC ] 0 ; Ti)");
    let commands_frag2 = processor.process_bytes(b"tle\x07");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Osc(b"0;Title".to_vec())],
        "After fragment 2 (tle BEL)"
    );
}

#[test]
fn it_should_complete_dcs_if_terminator_arrives_after_string_fragment() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1BPSt");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC P St)");
    let commands_frag2 = processor.process_bytes(b"uff\x1B\\");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Dcs(b"Stuff".to_vec())],
        "After fragment 2 (uff ESC \\)"
    );
}

// --- String Sequence Tests ---

#[test]
fn it_should_process_osc_string_terminated_by_bel() {
    let bytes = b"\x1B]0;Set Title\x07";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Osc(b"0;Set Title".to_vec())]);
}

#[test]
fn it_should_process_osc_string_terminated_by_st() {
    let bytes = b"\x1B]2;Another Title\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Osc(b"2;Another Title".to_vec())]
    );
}

#[test]
fn it_should_process_dcs_string_terminated_by_st() {
    let bytes = b"\x1BP1;1$rText\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Dcs(b"1;1$rText".to_vec())]);
}

#[test]
fn it_should_process_pm_string_terminated_by_st() {
    let bytes = b"\x1B^Privacy Message\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Pm(b"Privacy Message".to_vec())]);
}

#[test]
fn it_should_process_apc_string_terminated_by_st() {
    let bytes = b"\x1B_Application Command\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Apc(b"Application Command".to_vec())]
    );
}

// --- Edge Case / Error Tests ---

#[test]
fn it_should_process_empty_input_as_no_commands() {
    let bytes = b"";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn it_should_buffer_incomplete_csi_sequence() {
    let bytes = b"\x1B[1;2"; // Incomplete CSI
    let commands = process_bytes(bytes);
    // AnsiProcessor calls finalize, which might clear incomplete CSI state
    // or the parser might hold it. If it holds, empty is correct.
    // If finalize clears it without error, empty is also correct.
    assert!(
        commands.is_empty(),
        "Incomplete CSI should not produce commands yet, or should be cleared by finalize if unrecoverable by next process_bytes call"
    );
}

#[test]
fn it_should_process_csi_with_invalid_final_byte_as_error() {
    let bytes = b"\x1B[31a"; // 'a' is not a valid CSI final byte
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Error(b'a')]);
}

#[test]
fn it_should_buffer_incomplete_osc_sequence() {
    let bytes = b"\x1B]0;Title"; // Incomplete OSC
    let commands = process_bytes(bytes);
    assert!(
        commands.is_empty(),
        "Incomplete OSC should not produce commands yet"
    );
}

#[test]
fn it_should_buffer_incomplete_dcs_sequence() {
    let bytes = b"\x1BPStuff"; // Incomplete DCS
    let commands = process_bytes(bytes);
    assert!(
        commands.is_empty(),
        "Incomplete DCS should not produce commands yet"
    );
}

#[test]
fn it_should_terminate_osc_on_bel_and_process_subsequent_chars() {
    let bytes = b"\x1B]0;String\x08with\x07BEL"; // BEL (0x07) terminates OSC
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::Osc(b"0;String\x08with".to_vec()),
            AnsiCommand::Print('B'),
            AnsiCommand::Print('E'),
            AnsiCommand::Print('L'),
        ]
    );
}

#[test]
fn it_should_abort_osc_on_esc_and_process_subsequent_commands() {
    let bytes = b"\x1B]0;String\x1B\x07BEL"; // ESC (0x1B) aborts OSC
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::C0Control(C0Control::ESC),
            AnsiCommand::C0Control(C0Control::BEL),
            AnsiCommand::Print('B'),
            AnsiCommand::Print('E'),
            AnsiCommand::Print('L'),
        ]
    );
}

#[test]
fn it_should_include_c0_controls_within_dcs_data() {
    let bytes = b"\x1BPString\x08with\x0BC0\x1B\\"; // C0 controls are part of DCS data
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Dcs(b"String\x08with\x0BC0".to_vec())]
    );
}

#[test]
fn it_should_abort_dcs_on_esc_and_process_subsequent_st() {
    let bytes = b"\x1BPString\x1B\x1B\\"; // First ESC aborts DCS, second ESC + \ is ST
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::C0Control(C0Control::ESC),
            AnsiCommand::StringTerminator,
        ]
    );
}

#[test]
fn it_should_not_process_st_in_ground_state() {
    let bytes_esc_st = b"\x1B\\"; // ST (ESC \)
    let commands_esc_st = process_bytes(bytes_esc_st);
    assert_eq!(commands_esc_st, vec![AnsiCommand::StringTerminator]);

    let bytes_c1_st = b"\x9C"; // ST (C1 version)
    let commands_c1_st = process_bytes(bytes_c1_st);
    assert_eq!(
        commands_c1_st,
        vec![AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER)],
        "standalone C1 ST (0x9C) should print replacment"
    );
}

#[test]
fn it_should_abort_csi_on_esc_and_process_subsequent_csi() {
    let bytes = b"\x1B[1;2\x1B[3m"; // ESC aborts first CSI, second CSI is processed
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Italic
        ]))]
    );
}

#[cfg(test)]
mod unicode_wide_tests {
    use crate::ansi::{AnsiCommand, AnsiParser as AnsiParserTrait, AnsiProcessor}; // Use AnsiParser trait if needed, AnsiProcessor for instantiation
    use std::char; // For char::REPLACEMENT_CHARACTER

    // Import C0Control and EscCommand if they are used in expected AnsiCommand variants
    use crate::ansi::commands::{C0Control, EscCommand};
    use test_log::test;

    // Define byte constants for clarity in tests
    const NUL: u8 = 0x00;
    const ETX: u8 = 0x03;
    const ESC: u8 = 0x1B;
    const BEL_BYTE: u8 = 0x07;
    const C1_PAD_BYTE: u8 = 0x80; // PAD
    const IND_C1_BYTE: u8 = 0x84; // IND
    const ST_C1_BYTE: u8 = 0x9C; // String Terminator

    const CHAR_A_BYTE: u8 = 0x41; // 'A'
    const CHAR_B_BYTE: u8 = 0x42; // 'B'
    const CHAR_C_BYTE: u8 = 0x63; // 'c' (used in RIS)

    // Helper function, assuming AnsiProcessor is the public API to test
    fn process_bytes_unicode(bytes: &[u8]) -> Vec<AnsiCommand> {
        let mut processor = AnsiProcessor::new();
        processor.process_bytes(bytes)
    }

    #[test]
    fn it_should_handle_esc_c_ris_after_interrupted_utf8() {
        let bytes = &[0xE2, ESC, CHAR_C_BYTE]; // Incomplete '€', then ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState), // Assuming from_esc maps 'c' to this
            ],
            "ESC c (RIS) should be processed after UTF-8 interruption"
        );
    }

    #[test]
    fn it_should_handle_c0_bel_and_char_after_interrupted_utf8() {
        let bytes = &[0xF0, BEL_BYTE, CHAR_A_BYTE]; // Incomplete '😀', then BEL, then 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::BEL),
                AnsiCommand::Print('A'),
            ],
            "BEL and char should be processed after UTF-8 interruption"
        );
    }

    #[test]
    fn it_should_decode_valid_utf8_containing_c1_byte_value_for_ind_and_then_char() {
        let bytes = &[0xE2, 0x82, IND_C1_BYTE, CHAR_B_BYTE]; // E2 82 84 is '₄' (U+2084)
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Print('₄'), AnsiCommand::Print('B'),],
            "0xE2 0x82 0x84 should decode to '₄', not be interrupted by 0x84 as C1 IND"
        );
    }

    #[test]
    fn it_should_handle_complex_interruptions_and_valid_chars_in_sequence() {
        let bytes = &[
            0xE2,        // Start of '€'
            ESC,         // ESC (interrupts '€')
            CHAR_A_BYTE, // 'A' (becomes part of ESC A or Print('A') depending on parser)
            0xF0,
            0x9F,     // Start of '😀'
            BEL_BYTE, // BEL (interrupts '😀')
            0xC2,
            0xA2,        // '¢' (complete char)
            ESC,         // ESC
            CHAR_C_BYTE, // 'c' (RIS)
        ];
        let commands = process_bytes_unicode(bytes);

        let mut expected_commands = vec![
            AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For interrupted 0xE2
        ];
        // Check how ESC 'A' is handled (assuming it's not a defined sequence, might print 'A')
        if AnsiCommand::from_esc('A').is_none() {
            expected_commands.push(AnsiCommand::Print('A'));
        } else {
            expected_commands.push(AnsiCommand::from_esc('A').unwrap());
        }
        expected_commands.extend(vec![
            AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For interrupted 0xF0, 0x9F
            AnsiCommand::C0Control(C0Control::BEL),
            AnsiCommand::Print('¢'),
            AnsiCommand::Esc(EscCommand::ResetToInitialState),
        ]);
        assert_eq!(commands, expected_commands);
    }

    #[test]
    fn it_should_process_bel_correctly() {
        let bytes = &[BEL_BYTE];
        let commands = process_bytes_unicode(bytes);
        assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
    }

    #[test]
    fn it_should_process_esc_c_ris_correctly() {
        let bytes = &[ESC, CHAR_C_BYTE];
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Esc(EscCommand::ResetToInitialState)]
        );
    }

    #[test]
    fn it_should_handle_c0_nul_and_print_after_interrupted_utf8() {
        let bytes = &[0xE2, 0x82, NUL, CHAR_A_BYTE]; // Partial '€', NUL, 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::NUL),
                AnsiCommand::Print('A'),
            ]
        );
    }

    #[test]
    fn it_should_handle_c0_etx_and_esc_sequence_after_interrupted_utf8() {
        let bytes = &[0xF0, 0x9F, ETX, ESC, b'D']; // Partial '😀', ETX, ESC D (IND)
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::ETX),
                AnsiCommand::Esc(EscCommand::Index), // Assuming from_esc maps 'D' to Index
            ]
        );
    }

    #[test]
    fn it_should_decode_valid_utf8_containing_c1_byte_value_for_pad_and_then_char() {
        let bytes = &[0xC2, C1_PAD_BYTE, CHAR_A_BYTE]; // C2 80 is U+0080 (PAD control char)
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print('\u{0080}'), // Valid UTF-8 for C1 PAD
                AnsiCommand::Print('A'),
            ]
        );
    }

    #[test]
    fn it_should_handle_c1_st_and_char_after_interrupted_4_byte_utf8() {
        let bytes = &[0xF0, 0x9F, ST_C1_BYTE, CHAR_A_BYTE]; // Partial '😀', C1 ST (0x9C), 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                // 0x9C (ST_C1_BYTE) is now ignored by process_byte_as_new_token
                // after the UTF-8 sequence F0 9F 9C fails and 9C is reprocessed.
                AnsiCommand::Print('A'),
            ],
            "C1 ST (0x9C) after failed UTF-8 should be ignored, then 'A' printed"
        );
    }

    #[test]
    fn it_should_handle_esc_ris_after_3_byte_utf8_interrupted_at_2nd_byte() {
        let bytes = &[0xE2, 0x82, ESC, CHAR_C_BYTE]; // Partial '€', ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState),
            ]
        );
    }

    #[test]
    fn it_should_handle_esc_ris_after_4_byte_utf8_interrupted_at_1st_byte() {
        let bytes = &[0xF0, ESC, CHAR_C_BYTE]; // Partial '😀', ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState),
            ]
        );
    }

    #[test]
    fn it_should_handle_esc_ris_after_4_byte_utf8_interrupted_at_3rd_byte() {
        let bytes = &[0xF0, 0x9F, 0x98, ESC, CHAR_C_BYTE]; // Partial '😀', ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState),
            ]
        );
    }

    #[test]
    fn it_should_handle_double_utf8_interruption_by_esc_then_c0() {
        let bytes = &[0xE2, ESC, b'M', 0xF0, 0x9F, BEL_BYTE]; // Partial '€', ESC M (RI), Partial '😀', BEL
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ReverseIndex), // Assuming from_esc maps 'M' to RI
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::BEL),
            ]
        );
    }

    #[test]
    fn it_should_handle_invalid_utf8_continuation_followed_by_chars() {
        let bytes = &[0xE2, 0x41, 0x42]; // 0xE2 (start €), 'A' (invalid cont.), 'B'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For 0xE2 + 0x41 attempt
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
            ]
        );
    }

    #[test]
    fn it_should_handle_overlong_utf8_sequence_c1_af_as_replacement_chars() {
        let bytes = &[0xC1, 0xAF]; // 0xC1 is invalid start, 0xAF is invalid start
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
            ]
        );
    }

    #[test]
    fn it_should_replace_incomplete_3_of_4_byte_utf8_at_stream_end() {
        let bytes = &[0xF0, 0x9F, 0x98]; // Incomplete '😀'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER),]
        );
    }

    #[test]
    fn it_should_handle_del_interrupting_utf8_then_char() {
        let bytes = &[0xE2, 0x7F, CHAR_A_BYTE]; // Partial '€', DEL (0x7F), 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::DEL),
                AnsiCommand::Print('A'),
            ]
        );
    }

    #[test]
    fn it_should_handle_standalone_c1_nel_between_chars_as_replacement() {
        let bytes = &[0x41, 0x84, 0x42]; // A, NEL (C1), B
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER),
                AnsiCommand::Print('B'),
            ],
            "C1 NEL (0x84) should be ignored between A and B"
        );
    }

    #[test]
    fn it_should_handle_c1_like_byte_as_part_of_invalid_utf8_sequence() {
        let bytes = &[0xE2, 0x84, 0x41]; // Partial UTF-8 'E2', then 0x84 (C1 NEL), then 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For the invalid E2 84 41 sequence

                AnsiCommand::Print('A'),
            ],
            "0x84, when consumed by Utf8Decoder as part of an invalid sequence, should lead to REPLACEMENT_CHARACTER for the sequence, then 'A'"
        );
    }

    #[test]
    fn it_should_handle_c1_like_byte_in_invalid_4_byte_utf8_sequence() {
        let bytes = &[0xF0, 0x9F, 0x84, 0x41]; // Partial UTF-8, 0x84 (C1 NEL), 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For the invalid F0 9F 84 41 sequence
                AnsiCommand::Print('A'),
            ],
            "0x84, when consumed by Utf8Decoder as part of an invalid 4-byte sequence, should lead to REPLACEMENT_CHARACTER, then 'A'"
        );
    }

    #[test]
    fn it_should_correctly_decode_euro_sign_followed_by_char() {
        let bytes = &[0xE2, 0x82, 0xAC, 0x41]; // € then A
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Print('€'), AnsiCommand::Print('A'),],
            "Euro sign (E2 82 AC) should decode correctly, followed by A"
        );
    }

    #[test]
    fn it_should_replace_standalone_c1_ind_after_valid_utf8() {
        let bytes = &[0xE2, 0x82, 0xAC, 0x85, 0x41]; // € , IND (C1), A
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print('€'),
                AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER),
                AnsiCommand::Print('A'),
            ],
            "C1 IND (0x85) should be ignored after €"
        );
    }

    #[test]
    fn it_should_ignore_sequence_of_standalone_c1_controls() {
        let bytes = &[0x41, 0x84, 0x85, 0x42]; // A, NEL (C1), IND (C1), B
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER),
                AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER),
                AnsiCommand::Print('B'),
            ],
            "Sequence of C1 controls (0x84, 0x85) should be ignored"
        );
    }
}

#[test]
fn it_should_handle_esc_k_screen_title_sequence() {
    let bytes = b"\x1Bkls\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Apc(b"ls".to_vec())],
        "ESC k (screen title sequence) should consume title text and not print it"
    );
}

#[test]
fn it_should_handle_esc_k_with_empty_title() {
    let bytes = b"\x1Bk\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Apc(b"".to_vec())],
        "ESC k with empty title should produce empty Apc command"
    );
}

#[test]
fn it_should_handle_esc_k_with_longer_title() {
    let bytes = b"\x1Bkvim ~/.bashrc\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Apc(b"vim ~/.bashrc".to_vec())],
        "ESC k with longer title should consume all text until ST"
    );
}

#[test]
fn it_should_handle_text_before_and_after_esc_k_sequence() {
    let bytes = b"Before\x1Bkls\x1B\\After";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::Print('B'),
            AnsiCommand::Print('e'),
            AnsiCommand::Print('f'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print('r'),
            AnsiCommand::Print('e'),
            AnsiCommand::Apc(b"ls".to_vec()),
            AnsiCommand::Print('A'),
            AnsiCommand::Print('f'),
            AnsiCommand::Print('t'),
            AnsiCommand::Print('e'),
            AnsiCommand::Print('r'),
        ],
        "Text before and after ESC k sequence should print correctly, title should be consumed"
    );
}

// --- Character Set Designation Tests (ESC ( ) * + with final byte) ---

#[test]
fn it_should_process_esc_open_paren_b_usascii_charset() {
    // ESC ( B - Select US ASCII as G0 character set
    let bytes = b"\x1B(B";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('(', 'B'))],
        "ESC ( B should select US ASCII charset"
    );
}

#[test]
fn it_should_process_esc_open_paren_0_dec_graphics_charset() {
    // ESC ( 0 - Select DEC Special Character and Line Drawing Set as G0
    let bytes = b"\x1B(0";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('(', '0'))],
        "ESC ( 0 should select DEC Special Graphics charset"
    );
}

#[test]
fn it_should_process_esc_close_paren_a_uk_charset() {
    // ESC ) A - Select UK as G1 character set
    let bytes = b"\x1B)A";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet(')', 'A'))],
        "ESC ) A should select UK charset as G1"
    );
}

#[test]
fn it_should_process_esc_star_with_dec_supplemental() {
    // ESC * < - Select DEC Supplemental as G2
    let bytes = b"\x1B*<";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('*', '<'))],
        "ESC * < should select DEC Supplemental charset as G2"
    );
}

#[test]
fn it_should_process_esc_plus_with_dec_technical() {
    // ESC + > - Select DEC Technical as G3
    let bytes = b"\x1B+>";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('+', '>'))],
        "ESC + > should select DEC Technical charset as G3"
    );
}

#[test]
fn it_should_process_charset_designator_boundary_low() {
    // ESC ( 0 - '0' is 0x30, the lowest valid charset designator
    let bytes = b"\x1B(0";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('(', '0'))],
        "ESC ( 0 (0x30) should be valid - lowest boundary"
    );
}

#[test]
fn it_should_process_charset_designator_boundary_high() {
    // ESC ( ~ - '~' is 0x7E, the highest valid charset designator
    let bytes = b"\x1B(~";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('(', '~'))],
        "ESC ( ~ (0x7E) should be valid - highest boundary"
    );
}

#[test]
fn it_should_process_charset_special_designators() {
    // Test special characters that are valid charset designators
    let test_cases = [
        (':', "colon"),
        (';', "semicolon"),
        ('=', "equals (Swiss charset)"),
        ('?', "question mark"),
        ('@', "at sign"),
        ('[', "open bracket"),
        (']', "close bracket"),
        ('^', "caret"),
        ('_', "underscore"),
        ('`', "backtick"),
        ('{', "open brace"),
        ('|', "pipe"),
        ('}', "close brace"),
    ];

    for (designator, name) in test_cases {
        let bytes = format!("\x1B({}", designator);
        let commands = process_bytes(bytes.as_bytes());
        assert_eq!(
            commands,
            vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet(
                '(', designator
            ))],
            "ESC ( {} ({}) should be a valid charset designator",
            designator,
            name
        );
    }
}

#[test]
fn it_should_reject_charset_designator_below_valid_range() {
    // ESC ( / - '/' is 0x2F, below the valid range (0x30-0x7E)
    let bytes = b"\x1B(/";
    let commands = process_bytes(bytes);
    // Invalid charset designator should not produce a SelectCharacterSet command
    assert!(
        !commands
            .iter()
            .any(|c| matches!(c, AnsiCommand::Esc(EscCommand::SelectCharacterSet(_, _)))),
        "ESC ( / (0x2F) should be rejected - below valid range"
    );
}

#[test]
fn it_should_reject_charset_designator_above_valid_range() {
    // ESC ( DEL - DEL is 0x7F, above the valid range (0x30-0x7E)
    let bytes = b"\x1B(\x7F";
    let commands = process_bytes(bytes);
    // Invalid charset designator should not produce a SelectCharacterSet command
    assert!(
        !commands
            .iter()
            .any(|c| matches!(c, AnsiCommand::Esc(EscCommand::SelectCharacterSet(_, _)))),
        "ESC ( DEL (0x7F) should be rejected - above valid range"
    );
}

#[test]
fn it_should_reject_space_as_charset_designator() {
    // ESC ( SP - Space is 0x20, below the valid range
    let bytes = b"\x1B( ";
    let commands = process_bytes(bytes);
    assert!(
        !commands
            .iter()
            .any(|c| matches!(c, AnsiCommand::Esc(EscCommand::SelectCharacterSet(_, _)))),
        "ESC ( SP (0x20) should be rejected - not a valid charset designator"
    );
}

// ---------------------------------------------------------------------------
// Mutation tests
//
// Each test here is designed to kill a specific class of mutation that might
// otherwise survive the test suite above. The comments name the mutation.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod mutation_tests {
    use super::{
        super::commands::{AnsiCommand, Attribute, CsiCommand, EscCommand},
        process_bytes,
    };
    use crate::color::{Color, NamedColor};
    use test_log::test;

    // -----------------------------------------------------------------------
    // SGR basic attributes
    // Mutations: swap SGR constants, wrong Attribute variant, missing arms
    // -----------------------------------------------------------------------

    #[test]
    fn sgr_bold_is_1() {
        let cmds = process_bytes(b"\x1b[1m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Bold
            ]))]
        );
    }

    #[test]
    fn sgr_faint_is_2() {
        let cmds = process_bytes(b"\x1b[2m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Faint
            ]))]
        );
    }

    #[test]
    fn sgr_italic_is_3() {
        let cmds = process_bytes(b"\x1b[3m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Italic
            ]))]
        );
    }

    #[test]
    fn sgr_underline_is_4() {
        let cmds = process_bytes(b"\x1b[4m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Underline
            ]))]
        );
    }

    #[test]
    fn sgr_blink_slow_is_5() {
        let cmds = process_bytes(b"\x1b[5m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::BlinkSlow
            ]))]
        );
    }

    #[test]
    fn sgr_blink_rapid_is_6() {
        let cmds = process_bytes(b"\x1b[6m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::BlinkRapid
            ]))]
        );
    }

    #[test]
    fn sgr_reverse_is_7() {
        let cmds = process_bytes(b"\x1b[7m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Reverse
            ]))]
        );
    }

    #[test]
    fn sgr_conceal_is_8() {
        let cmds = process_bytes(b"\x1b[8m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Conceal
            ]))]
        );
    }

    #[test]
    fn sgr_strikethrough_is_9() {
        let cmds = process_bytes(b"\x1b[9m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Strikethrough
            ]))]
        );
    }

    #[test]
    fn sgr_underline_double_is_21() {
        let cmds = process_bytes(b"\x1b[21m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::UnderlineDouble
            ]))]
        );
    }

    #[test]
    fn sgr_normal_intensity_is_22() {
        let cmds = process_bytes(b"\x1b[22m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoBold
            ]))]
        );
    }

    #[test]
    fn sgr_no_italic_is_23() {
        let cmds = process_bytes(b"\x1b[23m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoItalic
            ]))]
        );
    }

    #[test]
    fn sgr_no_underline_is_24() {
        let cmds = process_bytes(b"\x1b[24m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoUnderline
            ]))]
        );
    }

    #[test]
    fn sgr_no_blink_is_25() {
        let cmds = process_bytes(b"\x1b[25m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoBlink
            ]))]
        );
    }

    #[test]
    fn sgr_no_reverse_is_27() {
        let cmds = process_bytes(b"\x1b[27m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoReverse
            ]))]
        );
    }

    #[test]
    fn sgr_no_conceal_is_28() {
        let cmds = process_bytes(b"\x1b[28m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoConceal
            ]))]
        );
    }

    #[test]
    fn sgr_no_strikethrough_is_29() {
        let cmds = process_bytes(b"\x1b[29m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoStrikethrough
            ]))]
        );
    }

    #[test]
    fn sgr_overlined_is_53() {
        let cmds = process_bytes(b"\x1b[53m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Overlined
            ]))]
        );
    }

    #[test]
    fn sgr_no_overlined_is_55() {
        let cmds = process_bytes(b"\x1b[55m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoOverlined
            ]))]
        );
    }

    // -----------------------------------------------------------------------
    // SGR foreground color offset arithmetic (param - SGR_FG_BLACK)
    // Mutations: wrong base constant (29 instead of 30), wrong NamedColor variant
    // -----------------------------------------------------------------------

    #[test]
    fn sgr_fg_all_eight_normal_colors() {
        // Tests every entry in map_basic_code_to_color for Normal intensity.
        // A mutation that shifts the base constant by ±1 would fail on Black or White.
        let cases: &[(u8, NamedColor)] = &[
            (30, NamedColor::Black),
            (31, NamedColor::Red),
            (32, NamedColor::Green),
            (33, NamedColor::Yellow),
            (34, NamedColor::Blue),
            (35, NamedColor::Magenta),
            (36, NamedColor::Cyan),
            (37, NamedColor::White),
        ];
        for &(code, expected_color) in cases {
            let input = format!("\x1b[{}m", code);
            let cmds = process_bytes(input.as_bytes());
            assert_eq!(
                cmds,
                vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                    Attribute::Foreground(Color::Named(expected_color))
                ]))],
                "SGR {} should produce Foreground({:?})",
                code,
                expected_color
            );
        }
    }

    #[test]
    fn sgr_fg_default_is_39() {
        // Mutation: wrong constant (38 vs 39 would change to extended-color path)
        let cmds = process_bytes(b"\x1b[39m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Default)
            ]))]
        );
    }

    // -----------------------------------------------------------------------
    // SGR background color offset arithmetic (param - SGR_BG_BLACK)
    // Mutations: wrong base constant (39 instead of 40), wrong variant
    // -----------------------------------------------------------------------

    #[test]
    fn sgr_bg_all_eight_normal_colors() {
        let cases: &[(u8, NamedColor)] = &[
            (40, NamedColor::Black),
            (41, NamedColor::Red),
            (42, NamedColor::Green),
            (43, NamedColor::Yellow),
            (44, NamedColor::Blue),
            (45, NamedColor::Magenta),
            (46, NamedColor::Cyan),
            (47, NamedColor::White),
        ];
        for &(code, expected_color) in cases {
            let input = format!("\x1b[{}m", code);
            let cmds = process_bytes(input.as_bytes());
            assert_eq!(
                cmds,
                vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                    Attribute::Background(Color::Named(expected_color))
                ]))],
                "SGR {} should produce Background({:?})",
                code,
                expected_color
            );
        }
    }

    #[test]
    fn sgr_bg_default_is_49() {
        let cmds = process_bytes(b"\x1b[49m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Background(Color::Default)
            ]))]
        );
    }

    // -----------------------------------------------------------------------
    // SGR bright foreground colors (90-97)
    // Mutations: wrong base (89 vs 90), wrong intensity branch
    // -----------------------------------------------------------------------

    #[test]
    fn sgr_fg_all_eight_bright_colors() {
        let cases: &[(u8, NamedColor)] = &[
            (90, NamedColor::BrightBlack),
            (91, NamedColor::BrightRed),
            (92, NamedColor::BrightGreen),
            (93, NamedColor::BrightYellow),
            (94, NamedColor::BrightBlue),
            (95, NamedColor::BrightMagenta),
            (96, NamedColor::BrightCyan),
            (97, NamedColor::BrightWhite),
        ];
        for &(code, expected_color) in cases {
            let input = format!("\x1b[{}m", code);
            let cmds = process_bytes(input.as_bytes());
            assert_eq!(
                cmds,
                vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                    Attribute::Foreground(Color::Named(expected_color))
                ]))],
                "SGR {} should produce Foreground({:?})",
                code,
                expected_color
            );
        }
    }

    // -----------------------------------------------------------------------
    // SGR bright background colors (100-107)
    // -----------------------------------------------------------------------

    #[test]
    fn sgr_bg_all_eight_bright_colors() {
        let cases: &[(u8, NamedColor)] = &[
            (100, NamedColor::BrightBlack),
            (101, NamedColor::BrightRed),
            (102, NamedColor::BrightGreen),
            (103, NamedColor::BrightYellow),
            (104, NamedColor::BrightBlue),
            (105, NamedColor::BrightMagenta),
            (106, NamedColor::BrightCyan),
            (107, NamedColor::BrightWhite),
        ];
        for &(code, expected_color) in cases {
            let input = format!("\x1b[{}m", code);
            let cmds = process_bytes(input.as_bytes());
            assert_eq!(
                cmds,
                vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                    Attribute::Background(Color::Named(expected_color))
                ]))],
                "SGR {} should produce Background({:?})",
                code,
                expected_color
            );
        }
    }

    // -----------------------------------------------------------------------
    // SGR 256-color and RGB truecolor
    // Mutations: wrong mode sub-parameter (4 vs 5, 1 vs 2), Fg vs Bg confusion
    // -----------------------------------------------------------------------

    #[test]
    fn sgr_fg_256_color_index_min() {
        // 38;5;0 -> Foreground(Indexed(0))
        let cmds = process_bytes(b"\x1b[38;5;0m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Indexed(0))
            ]))]
        );
    }

    #[test]
    fn sgr_fg_256_color_index_max() {
        // 38;5;255 -> Foreground(Indexed(255))
        let cmds = process_bytes(b"\x1b[38;5;255m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Indexed(255))
            ]))]
        );
    }

    #[test]
    fn sgr_bg_256_color() {
        // 48;5;100 -> Background(Indexed(100))
        // Mutation: mixing up 38 and 48 would swap Fg/Bg
        let cmds = process_bytes(b"\x1b[48;5;100m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Background(Color::Indexed(100))
            ]))]
        );
    }

    #[test]
    fn sgr_fg_rgb_truecolor() {
        // 38;2;255;128;0 -> Foreground(Rgb(255, 128, 0))
        let cmds = process_bytes(b"\x1b[38;2;255;128;0m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Rgb(255, 128, 0))
            ]))]
        );
    }

    #[test]
    fn sgr_bg_rgb_truecolor() {
        // 48;2;10;20;30 -> Background(Rgb(10, 20, 30))
        let cmds = process_bytes(b"\x1b[48;2;10;20;30m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Background(Color::Rgb(10, 20, 30))
            ]))]
        );
    }

    #[test]
    fn sgr_rgb_channel_order_is_r_g_b_not_b_g_r() {
        // Explicitly verify R,G,B order - a mutation swapping two channel reads would survive
        // a test using equal values. Using distinct non-symmetric values.
        let cmds = process_bytes(b"\x1b[38;2;1;2;3m");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Rgb(1, 2, 3))
            ]))]
        );
    }

    // -----------------------------------------------------------------------
    // CSI cursor movement: default parameter must be 1
    // Mutations: param_or_1 returns 0 instead of 1, or uses param_or(0) default
    // -----------------------------------------------------------------------

    #[test]
    fn csi_cursor_up_no_param_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[A");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorUp(1))]
        );
    }

    #[test]
    fn csi_cursor_up_param_zero_is_coerced_to_1() {
        // param_or_1 applies max(1), so 0 -> 1
        let cmds = process_bytes(b"\x1b[0A");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorUp(1))]
        );
    }

    #[test]
    fn csi_cursor_up_explicit_5() {
        let cmds = process_bytes(b"\x1b[5A");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorUp(5))]
        );
    }

    #[test]
    fn csi_cursor_down_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[B");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorDown(1))]
        );
    }

    #[test]
    fn csi_cursor_forward_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[C");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorForward(1))]
        );
    }

    #[test]
    fn csi_cursor_backward_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[D");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorBackward(1))]
        );
    }

    #[test]
    fn csi_cursor_next_line_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[E");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(1))]
        );
    }

    #[test]
    fn csi_cursor_prev_line_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[F");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPrevLine(1))]
        );
    }

    #[test]
    fn csi_cursor_char_absolute_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[G");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(1))]
        );
    }

    // -----------------------------------------------------------------------
    // CSI cursor position: both row and col default to 1
    // -----------------------------------------------------------------------

    #[test]
    fn csi_cup_both_params_present() {
        let cmds = process_bytes(b"\x1b[5;10H");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(5, 10))]
        );
    }

    #[test]
    fn csi_cup_row_only_col_defaults_to_1() {
        let cmds = process_bytes(b"\x1b[3H");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(3, 1))]
        );
    }

    #[test]
    fn csi_cup_no_params_both_default_to_1() {
        // ESC [ H with no params = home (1,1); tested above but repeat here for clarity
        let cmds = process_bytes(b"\x1b[H");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]
        );
    }

    // -----------------------------------------------------------------------
    // CSI erase: EraseInDisplay and EraseInLine default to 0, not 1
    // Mutation: using param_or_1 instead of param_or(0,0) would give 1
    // -----------------------------------------------------------------------

    #[test]
    fn csi_erase_in_display_no_param_defaults_to_0() {
        let cmds = process_bytes(b"\x1b[J");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(0))]
        );
    }

    #[test]
    fn csi_erase_in_display_explicit_2() {
        let cmds = process_bytes(b"\x1b[2J");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))]
        );
    }

    #[test]
    fn csi_erase_in_line_no_param_defaults_to_0() {
        let cmds = process_bytes(b"\x1b[K");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::EraseInLine(0))]
        );
    }

    #[test]
    fn csi_erase_in_line_explicit_1() {
        let cmds = process_bytes(b"\x1b[1K");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::EraseInLine(1))]
        );
    }

    // -----------------------------------------------------------------------
    // ESC command dispatch: every ESC final-char mapping
    // Mutations: wrong char in from_esc match arm, swapped EscCommand variants
    // -----------------------------------------------------------------------

    #[test]
    fn esc_d_is_index() {
        let cmds = process_bytes(b"\x1bD");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::Index)]);
    }

    #[test]
    fn esc_e_is_next_line() {
        let cmds = process_bytes(b"\x1bE");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::NextLine)]);
    }

    #[test]
    fn esc_h_is_set_tab_stop() {
        let cmds = process_bytes(b"\x1bH");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::SetTabStop)]);
    }

    #[test]
    fn esc_m_is_reverse_index() {
        let cmds = process_bytes(b"\x1bM");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::ReverseIndex)]);
    }

    #[test]
    fn esc_7_is_save_cursor() {
        let cmds = process_bytes(b"\x1b7");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::SaveCursor)]);
    }

    #[test]
    fn esc_8_is_restore_cursor() {
        let cmds = process_bytes(b"\x1b8");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::RestoreCursor)]);
    }

    #[test]
    fn esc_c_is_reset_to_initial_state() {
        let cmds = process_bytes(b"\x1bc");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Esc(EscCommand::ResetToInitialState)]
        );
    }

    #[test]
    fn esc_n_is_single_shift_2() {
        let cmds = process_bytes(b"\x1bN");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::SingleShift2)]);
    }

    #[test]
    fn esc_o_is_single_shift_3() {
        let cmds = process_bytes(b"\x1bO");
        assert_eq!(cmds, vec![AnsiCommand::Esc(EscCommand::SingleShift3)]);
    }

    // -----------------------------------------------------------------------
    // CSI MAX_PARAMS boundary enforcement
    // Mutation: `<` to `<=` in add_param would allow 17 params (the 17th
    //           would be added and the 16th dropped or the limit off by one).
    // -----------------------------------------------------------------------

    #[test]
    fn csi_accepts_exactly_16_params() {
        // SGR with 16 parameters: 1;2;0;0;... (14 zeros)
        // All 16 must be collected and processed.
        let cmds = process_bytes(b"\x1b[1;2;0;0;0;0;0;0;0;0;0;0;0;0;0;0m");
        if let Some(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(attrs))) = cmds.first() {
            // Checking len kills mutations that reduce MAX_PARAMS below 16.
            assert_eq!(attrs.len(), 16, "all 16 params must be retained");
            assert_eq!(attrs[0], Attribute::Bold, "first of 16 params should be Bold");
            assert_eq!(
                attrs[1],
                Attribute::Faint,
                "second of 16 params should be Faint"
            );
        } else {
            panic!("expected SetGraphicsRendition, got {:?}", cmds);
        }
    }

    #[test]
    fn csi_silently_drops_17th_param() {
        // 16 zeros then ;7 (Reverse). The 17th param (7) must be ignored.
        // We send: ESC [ 0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;7m
        //          that's 16 zeros + 1 extra -> the 7 (Reverse) is the 17th and dropped.
        let cmds =
            process_bytes(b"\x1b[0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;7m");
        if let Some(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(attrs))) = cmds.first() {
            // Every kept attribute should be Reset (0); Reverse (7) must NOT appear.
            assert!(
                !attrs.contains(&Attribute::Reverse),
                "17th param (Reverse) should have been dropped; got attrs: {:?}",
                attrs
            );
        } else {
            panic!("expected SetGraphicsRendition, got {:?}", cmds);
        }
    }

    // -----------------------------------------------------------------------
    // UTF-8 boundary: exact invalid/valid byte transitions
    // Mutations: off-by-one on UTF8_2_BYTE_MIN (0xC2), UTF8_4_BYTE_MAX (0xF4),
    //            continuation range (0x80..=0xBF)
    // -----------------------------------------------------------------------

    #[test]
    fn utf8_0xc1_is_invalid_start_produces_replacement() {
        // 0xC1 is just below the valid 2-byte start range (0xC2)
        let cmds = process_bytes(&[0xC1, 0x80]);
        // Both bytes should produce replacement characters, not a valid char
        assert!(
            cmds.iter()
                .all(|c| c == &AnsiCommand::Print(char::REPLACEMENT_CHARACTER)),
            "0xC1 must be an invalid start; got {:?}",
            cmds
        );
    }

    #[test]
    fn utf8_0xc2_is_valid_2_byte_start() {
        // 0xC2 0x80 = U+0080 (PAD) - lowest valid 2-byte sequence
        let cmds = process_bytes(&[0xC2, 0x80]);
        assert_eq!(
            cmds,
            vec![AnsiCommand::Print('\u{0080}')],
            "0xC2 0x80 must decode to U+0080"
        );
    }

    #[test]
    fn utf8_0xf4_0x8f_0xbf_0xbf_is_valid() {
        // 0xF4 0x8F 0xBF 0xBF = U+10FFFF (highest valid code point)
        let cmds = process_bytes(&[0xF4, 0x8F, 0xBF, 0xBF]);
        assert_eq!(
            cmds,
            vec![AnsiCommand::Print('\u{10FFFF}')],
            "0xF4 0x8F 0xBF 0xBF must decode to U+10FFFF"
        );
    }

    #[test]
    fn utf8_0xf5_is_invalid_start_produces_replacement() {
        // 0xF5 is just above the valid 4-byte start range (0xF4)
        let cmds = process_bytes(&[0xF5, 0x80, 0x80, 0x80]);
        assert!(
            cmds.iter()
                .any(|c| c == &AnsiCommand::Print(char::REPLACEMENT_CHARACTER)),
            "0xF5 must be invalid; got {:?}",
            cmds
        );
    }

    #[test]
    fn utf8_continuation_0x7f_is_not_valid_continuation() {
        // 0xE2 starts a 3-byte sequence; 0x7F is DEL (not a continuation byte 0x80..=0xBF)
        // Should abort the UTF-8 sequence and emit replacement + DEL control
        let cmds = process_bytes(&[0xE2, 0x7F]);
        assert!(
            cmds.iter()
                .any(|c| c == &AnsiCommand::Print(char::REPLACEMENT_CHARACTER)),
            "0x7F after 3-byte start must not be a valid continuation; got {:?}",
            cmds
        );
    }

    #[test]
    fn utf8_continuation_0xc0_is_not_valid_continuation() {
        // 0xC0 is above the continuation range (0x80..=0xBF)
        // Should abort the UTF-8 sequence
        let cmds = process_bytes(&[0xE2, 0xC0]);
        assert!(
            cmds.iter()
                .any(|c| c == &AnsiCommand::Print(char::REPLACEMENT_CHARACTER)),
            "0xC0 after 3-byte start must not be a valid continuation; got {:?}",
            cmds
        );
    }

    #[test]
    fn utf8_continuation_0xbf_is_valid_continuation() {
        // 0xDF 0xBF = U+07FF - uses 0xBF which is the highest continuation byte
        let cmds = process_bytes(&[0xDF, 0xBF]);
        assert_eq!(
            cmds,
            vec![AnsiCommand::Print('\u{07FF}')],
            "0xDF 0xBF must decode to U+07FF"
        );
    }

    // -----------------------------------------------------------------------
    // CAN / SUB cancel string sequences
    // Mutations: wrong C0 byte constant in the match (0x17 instead of 0x18)
    // -----------------------------------------------------------------------

    #[test]
    fn osc_cancelled_by_can() {
        // CAN (0x18) inside an OSC should discard the string and return to ground
        let cmds = process_bytes(b"\x1b]0;title\x18rest");
        // No Osc command should appear; "rest" prints normally
        assert!(
            !cmds.iter().any(|c| matches!(c, AnsiCommand::Osc(_))),
            "CAN must cancel OSC; got {:?}",
            cmds
        );
        assert!(
            cmds.iter().any(|c| c == &AnsiCommand::Print('r')),
            "text after CAN must print; got {:?}",
            cmds
        );
    }

    #[test]
    fn osc_cancelled_by_sub() {
        // SUB (0x1A) inside an OSC should also discard it
        let cmds = process_bytes(b"\x1b]0;title\x1Arest");
        assert!(
            !cmds.iter().any(|c| matches!(c, AnsiCommand::Osc(_))),
            "SUB must cancel OSC; got {:?}",
            cmds
        );
    }

    #[test]
    fn dcs_cancelled_by_can() {
        let cmds = process_bytes(b"\x1bPdata\x18rest");
        assert!(
            !cmds.iter().any(|c| matches!(c, AnsiCommand::Dcs(_))),
            "CAN must cancel DCS; got {:?}",
            cmds
        );
    }

    // -----------------------------------------------------------------------
    // CSI param overflow: u16 saturation at MAX
    // Mutation: removing checked_mul/checked_add would panic or wrap on overflow
    // -----------------------------------------------------------------------

    #[test]
    fn csi_param_overflow_saturates_not_panics() {
        // A number larger than u16::MAX (65535) should saturate to u16::MAX, not panic
        let cmds = process_bytes(b"\x1b[99999A");
        // Should produce CursorUp with some value (saturated), not panic
        assert_eq!(cmds.len(), 1, "should produce exactly one command");
        assert!(
            matches!(cmds[0], AnsiCommand::Csi(CsiCommand::CursorUp(_))),
            "should still produce CursorUp; got {:?}",
            cmds
        );
        if let AnsiCommand::Csi(CsiCommand::CursorUp(n)) = cmds[0] {
            assert_eq!(n, u16::MAX, "overflow must saturate to u16::MAX, got {}", n);
        }
    }

    // -----------------------------------------------------------------------
    // CSI clear-tab-stop parameter values
    // Mutation: wrong W-param mapping (0->SetTabStop, 2->ClearTabStops(0),
    //           5->ClearTabStops(3) - each value is distinct)
    // -----------------------------------------------------------------------

    #[test]
    fn csi_ctc_0_is_set_tab_stop() {
        let cmds = process_bytes(b"\x1b[0W");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetTabStop)]
        );
    }

    #[test]
    fn csi_ctc_2_clears_current_tab_stop() {
        let cmds = process_bytes(b"\x1b[2W");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::ClearTabStops(0))]
        );
    }

    #[test]
    fn csi_ctc_5_clears_all_tab_stops() {
        let cmds = process_bytes(b"\x1b[5W");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::ClearTabStops(3))]
        );
    }

    // -----------------------------------------------------------------------
    // CSI scroll up/down distinction
    // Mutations: confusing S (scroll up) with T (scroll down)
    // -----------------------------------------------------------------------

    #[test]
    fn csi_s_uppercase_is_scroll_up() {
        let cmds = process_bytes(b"\x1b[3S");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::ScrollUp(3))]
        );
    }

    #[test]
    fn csi_t_uppercase_is_scroll_down() {
        let cmds = process_bytes(b"\x1b[3T");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::ScrollDown(3))]
        );
    }

    // -----------------------------------------------------------------------
    // CSI delete / insert / erase character distinction
    // Mutations: wrong command for P, @, X
    // -----------------------------------------------------------------------

    #[test]
    fn csi_at_is_insert_character() {
        let cmds = process_bytes(b"\x1b[2@");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::InsertCharacter(2))]
        );
    }

    #[test]
    fn csi_p_uppercase_is_delete_character() {
        let cmds = process_bytes(b"\x1b[2P");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::DeleteCharacter(2))]
        );
    }

    #[test]
    fn csi_x_uppercase_is_erase_character() {
        let cmds = process_bytes(b"\x1b[2X");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::EraseCharacter(2))]
        );
    }

    // -----------------------------------------------------------------------
    // CSI insert / delete line distinction (L vs M)
    // -----------------------------------------------------------------------

    #[test]
    fn csi_l_uppercase_is_insert_line() {
        let cmds = process_bytes(b"\x1b[4L");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::InsertLine(4))]
        );
    }

    #[test]
    fn csi_m_uppercase_is_delete_line() {
        let cmds = process_bytes(b"\x1b[4M");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::DeleteLine(4))]
        );
    }

    // -----------------------------------------------------------------------
    // DECSTBM scrolling region: top/bottom defaults and values
    // Mutations: swapped row/col, wrong default (1 vs 0)
    // -----------------------------------------------------------------------

    #[test]
    fn csi_decstbm_explicit_top_and_bottom() {
        let cmds = process_bytes(b"\x1b[5;24r");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetScrollingRegion {
                top: 5,
                bottom: 24
            })]
        );
    }

    #[test]
    fn csi_decstbm_top_defaults_to_1() {
        // No params: top=1, bottom=0 (convention for "last line")
        let cmds = process_bytes(b"\x1b[r");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::SetScrollingRegion {
                top: 1,
                bottom: 0
            })]
        );
    }

    // -----------------------------------------------------------------------
    // CSI final bytes with no tests elsewhere
    // Mutations: typo in final-byte match arm, wrong CsiCommand variant
    // -----------------------------------------------------------------------

    #[test]
    fn csi_f_is_cup_alias_for_h() {
        // 'f' is an alias for 'H' (CursorPosition / HVP).
        // A mutation removing the `| (false, b"", b'f')` arm would produce Error.
        let cmds = process_bytes(b"\x1b[5;10f");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(5, 10))]
        );
    }

    #[test]
    fn csi_d_is_vpa_column_always_1() {
        // 'd' = VPA (Vertical Position Absolute). Column is hardcoded to 1.
        // Mutation: changing the hardcoded 1 to 0, or wiring column to param.
        let cmds = process_bytes(b"\x1b[7d");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(7, 1))]
        );
    }

    #[test]
    fn csi_d_default_row_is_1() {
        // No param: row defaults to 1 (param_or_1), column always 1.
        let cmds = process_bytes(b"\x1b[d");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]
        );
    }

    #[test]
    fn csi_c_is_primary_device_attributes() {
        // 'c' with no params triggers PrimaryDeviceAttributes.
        // Mutation: confusing 'c' with other single-char finals.
        let cmds = process_bytes(b"\x1b[c");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::PrimaryDeviceAttributes)]
        );
    }

    #[test]
    fn csi_n_is_device_status_report() {
        // 'n' with param 6 = DSR cursor position request.
        // Mutation: confusing 'n' with 'm' (SGR) or wrong default.
        let cmds = process_bytes(b"\x1b[6n");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::DeviceStatusReport(6))]
        );
    }

    #[test]
    fn csi_n_default_param_is_0() {
        // No param: DSR(0).
        let cmds = process_bytes(b"\x1b[n");
        assert_eq!(
            cmds,
            vec![AnsiCommand::Csi(CsiCommand::DeviceStatusReport(0))]
        );
    }

    #[test]
    fn csi_s_is_save_cursor() {
        // 's' = save cursor (ANSI). Mutation: confusing 's' with 'S' (ScrollUp).
        let cmds = process_bytes(b"\x1b[s");
        assert_eq!(cmds, vec![AnsiCommand::Csi(CsiCommand::SaveCursor)]);
    }

    #[test]
    fn csi_u_is_restore_cursor() {
        // 'u' = restore cursor (ANSI). Mutation: confusing 'u' with 'U'.
        let cmds = process_bytes(b"\x1b[u");
        assert_eq!(cmds, vec![AnsiCommand::Csi(CsiCommand::RestoreCursor)]);
    }
}
