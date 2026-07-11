// src/ansi/lexer.rs

//! ANSI escape sequence lexer.
//! Converts a byte stream into `AnsiToken`s, processing byte by byte,
//! handling UTF-8 decoding and state across calls.

use log::{trace, warn};
use std::{mem, str};

/// Unicode replacement character (U+FFFD).
/// Used when encountering invalid UTF-8 sequences.
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

/// Represents the outcome of a single byte being processed by the Utf8Decoder.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Utf8DecodeResult {
    /// Successfully decoded a valid Unicode character.
    Decoded(char),
    /// The byte sequence was invalid. Decoder is reset.
    InvalidSequence,
    /// Current byte was validly consumed/buffered; more bytes needed.
    NeedsMoreBytes,
}

// --- Constants for Control Code Ranges ---
const DEL_BYTE: u8 = 0x7F;
const ESC_BYTE: u8 = 0x1B;

// --- Constants for Unicode boundaries ---
const UNICODE_MAX_CODE_POINT: u32 = 0x10FFFF;
const UNICODE_SURROGATE_START: u32 = 0xD800;
const UNICODE_SURROGATE_END: u32 = 0xDFFF;

// --- Constants for UTF-8 byte classification (used by Utf8Decoder) ---
const UTF8_ASCII_MAX: u8 = 0x7F;
const UTF8_CONT_MIN: u8 = 0x80; // Start of continuation byte range
const UTF8_CONT_MAX: u8 = 0xBF; // End of continuation byte range
const UTF8_2_BYTE_MIN: u8 = 0xC2; // Excludes overlong 0xC0, 0xC1
const UTF8_2_BYTE_MAX: u8 = 0xDF;
const UTF8_3_BYTE_MIN: u8 = 0xE0;
const UTF8_3_BYTE_MAX: u8 = 0xEF;
const UTF8_4_BYTE_MIN: u8 = 0xF0;
const UTF8_4_BYTE_MAX: u8 = 0xF4; // Max valid start for 4-byte sequence (RFC 3629)

// Invalid ranges for pattern matching
// 0x80..=0xC1 are invalid start bytes
const UTF8_INVALID_START_MIN: u8 = 0x80;
const UTF8_INVALID_START_MAX: u8 = 0xC1;
// 0xF5..=0xFF are invalid start bytes
const UTF8_INVALID_LATE_MIN: u8 = 0xF5;
const UTF8_INVALID_LATE_MAX: u8 = 0xFF;

/// Represents a single token identified by the lexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnsiToken {
    /// A printable character, decoded from UTF-8.
    Print(char),
    /// A C0 control code (0x00 - 0x1F, plus DEL 0x7F).
    C0Control(u8),
}

/// Internal state machine for decoding UTF-8 byte streams incrementally.
#[derive(Debug, Clone, Default)]
struct Utf8Decoder {
    buffer: [u8; 4],
    len: usize,
    expected: usize,
}

impl Utf8Decoder {
    /// Resets the decoder state.
    #[inline]
    fn reset(&mut self) {
        self.len = 0;
        self.expected = 0;
    }

    /// Decodes a single byte.
    fn decode(&mut self, byte: u8) -> Utf8DecodeResult {
        if self.len == 0 {
            return self.decode_first_byte(byte);
        }
        self.decode_continuation_byte(byte)
    }

    #[inline]
    fn decode_first_byte(&mut self, byte: u8) -> Utf8DecodeResult {
        match byte {
            0x00..=UTF8_ASCII_MAX => Utf8DecodeResult::Decoded(byte as char),
            UTF8_2_BYTE_MIN..=UTF8_2_BYTE_MAX => {
                self.expected = 2;
                self.buffer[0] = byte;
                self.len = 1;
                Utf8DecodeResult::NeedsMoreBytes
            }
            UTF8_3_BYTE_MIN..=UTF8_3_BYTE_MAX => {
                self.expected = 3;
                self.buffer[0] = byte;
                self.len = 1;
                Utf8DecodeResult::NeedsMoreBytes
            }
            UTF8_4_BYTE_MIN..=UTF8_4_BYTE_MAX => {
                self.expected = 4;
                self.buffer[0] = byte;
                self.len = 1;
                Utf8DecodeResult::NeedsMoreBytes
            }
            // Catches invalid start bytes: 0x80-0xC1 (continuation / overlong C0/C1) and 0xF5-0xFF
            UTF8_INVALID_START_MIN..=UTF8_INVALID_START_MAX
            | UTF8_INVALID_LATE_MIN..=UTF8_INVALID_LATE_MAX => {
                warn!("invalid utf8 sequence byte: {:X?}", byte);
                self.reset();
                Utf8DecodeResult::InvalidSequence
            }
        }
    }

    #[inline]
    fn decode_continuation_byte(&mut self, byte: u8) -> Utf8DecodeResult {
        if !(UTF8_CONT_MIN..=UTF8_CONT_MAX).contains(&byte) {
            // Current `byte` is not a valid UTF-8 continuation.
            // The previously buffered sequence is now considered invalid.
            self.reset();
            return Utf8DecodeResult::InvalidSequence;
        }

        self.buffer[self.len] = byte;
        self.len += 1;

        if self.len != self.expected {
            return Utf8DecodeResult::NeedsMoreBytes;
        }

        let b = self.buffer;
        // Sequence is now notionally complete, try to make a char.
        // This also implicitly handles overlong sequences if `str::from_utf8` is strict.
        let char_str_result = str::from_utf8(&b[0..self.len]);
        self.reset();

        match char_str_result {
            Ok(s) => {
                // `from_utf8` guarantees `s` is valid UTF-8.
                // A multi-byte UTF-8 sequence will produce exactly one char.
                if let Some(c) = s.chars().next() {
                    let cp = c as u32;
                    // Final check for Unicode constraints (surrogates, max codepoint).
                    if cp <= UNICODE_MAX_CODE_POINT
                        && !(UNICODE_SURROGATE_START..=UNICODE_SURROGATE_END).contains(&cp)
                    {
                        Utf8DecodeResult::Decoded(c)
                    } else {
                        Utf8DecodeResult::InvalidSequence // Valid UTF-8 but invalid Unicode scalar value
                    }
                } else {
                    // Should be impossible for non-empty slice that `from_utf8` said was Ok.
                    Utf8DecodeResult::InvalidSequence
                }
            }
            Err(_) => Utf8DecodeResult::InvalidSequence, // Malformed UTF-8 byte sequence.
        }
    }
}

/// Lexer that processes a stream of bytes into `AnsiToken`s.
#[derive(Debug, Clone, Default)]
pub struct AnsiLexer {
    tokens: Vec<AnsiToken>,
    utf8_decoder: Utf8Decoder,
}

impl AnsiLexer {
    /// Creates a new `AnsiLexer`.
    pub fn new() -> Self {
        AnsiLexer::default()
    }

    /// Determines if a byte is a C0 (excluding NUL, HT, LF, CR, etc. if meant as data),
    /// ESC, or DEL that should unambiguously interrupt any ongoing UTF-8 sequence.
    /// C1 codes are *not* checked here because their byte values can be valid UTF-8 continuations;
    /// they are handled by the Utf8Decoder returning InvalidSequence if they break a sequence.
    #[inline]
    fn is_unambiguous_interrupting_control(byte: u8) -> bool {
        byte == ESC_BYTE ||
        // Consider which C0s truly interrupt. For now, all except common formatting.
        // Some tests might expect NUL or other C0s to also interrupt.
        // This list should match C0s that are *never* valid data mid-UTF-8.
        match byte {
            // Explicitly allow common formatting C0s to pass through to the decoder
            // if they are not part of a multi-byte sequence (handled by decoder).
            // However, if UTF-8 is in progress, ANY C0 is an interruption.
            0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1A | 0x1C..=0x1F | DEL_BYTE => true,
            _ => false,
        }
    }

    /// Determines if a byte is any C0, ESC, or DEL control code.
    /// Used when processing a byte from a ground state (no active UTF-8 sequence).
    #[inline]
    fn is_control_code(byte: u8) -> bool {
        // C0 Part 1: 0x00..=0x1A
        // C0 Part 2: 0x1C..=0x1F
        // DEL: 0x7F
        // ESC: 0x1B
        matches!(byte, 0x00..=0x1A | 0x1C..=0x1F | DEL_BYTE | ESC_BYTE)
    }

    fn process_byte_as_new_token(&mut self, byte: u8) {
        // This function is called when utf8_decoder.len == 0.
        // It decides if 'byte' is a control code or starts a new UTF-8 sequence.
        if Self::is_control_code(byte) {
            self.tokens.push(AnsiToken::C0Control(byte));
            return;
        }
        // Not a control code, so attempt to process as UTF-8 start.
        // Utf8Decoder is fresh (len == 0).
        match self.utf8_decoder.decode(byte) {
            Utf8DecodeResult::Decoded(c) => self.tokens.push(AnsiToken::Print(c)),
            Utf8DecodeResult::NeedsMoreBytes => { /* Byte buffered, wait for more */ }
            Utf8DecodeResult::InvalidSequence => {
                // This means 'byte' itself was an invalid UTF-8 start (e.g., 0xC0, 0xF5).
                warn!("invalid utf8 byte: {:X?}", byte);
                self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
            }
        }
    }

    /// Processes a single byte and updates the lexer state.
    ///
    /// # Parameters
    /// * `byte` - The byte to process.
    pub fn process_byte(&mut self, byte: u8) {
        if self.utf8_decoder.len > 0 {
            // --- Currently building a multi-byte UTF-8 char ---
            // Check for unambiguous interruptions (ESC, most C0s)
            if Self::is_unambiguous_interrupting_control(byte) {
                warn!("encountered control byte: {:X?} mid utf8 stream", byte);
                self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER)); // For the aborted UTF-8
                self.utf8_decoder.reset();
                self.process_byte_as_new_token(byte); // Process the interrupting C0/ESC
                return;
            }

            // Let the Utf8Decoder try to process it. If it's not a valid continuation for
            // the current sequence, Utf8Decoder will return InvalidSequence.
            match self.utf8_decoder.decode(byte) {
                Utf8DecodeResult::Decoded(c) => {
                    self.tokens.push(AnsiToken::Print(c));
                    // Decoder has reset.
                }
                Utf8DecodeResult::InvalidSequence => {
                    // `byte` (which could be a C1, or non-control like 'A')
                    // was not a valid continuation for what was in the buffer.
                    // Utf8Decoder has reset.
                    self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER)); // For the broken sequence
                                                                               // Now, reprocess `byte` from a ground state.
                                                                               // process_byte_as_new_token will correctly identify it if it's C1, C0, ESC, or data.
                    self.process_byte_as_new_token(byte);
                }
                Utf8DecodeResult::NeedsMoreBytes => {
                    // Valid continuation, byte buffered. Wait for more.
                }
            }
        } else {
            // --- Not currently building a multi-byte char (utf8_decoder.len == 0) ---
            self.process_byte_as_new_token(byte);
        }
    }

    /// Consumes and returns all accumulated tokens.
    pub fn take_tokens(&mut self) -> Vec<AnsiToken> {
        trace!("taking {:?} tokens from lexer", self.tokens);
        mem::take(&mut self.tokens)
    }

    /// Finalizes any incomplete UTF-8 sequence, e.g., at end of stream.
    /// This is called by the AnsiProcessor after processing a chunk of bytes.
    pub fn finalize(&mut self) {
        if self.utf8_decoder.len > 0 {
            self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
            self.utf8_decoder.reset();
        }
    }
}
