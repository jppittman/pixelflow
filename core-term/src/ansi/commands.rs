// src/ansi/commands.rs

//! Defines the `AnsiCommand` enum representing parsed ANSI escape sequences
//! and related helper enums/structs.

// Import the canonical Color and NamedColor from the crate's color module
use crate::color::{Color, NamedColor};
use log::warn;
use std::fmt;
use std::iter::Peekable;
use std::slice::Iter;

// --- SGR Parameter Constants ---
// These constants represent the numeric parameters used in Select Graphic Rendition (SGR) sequences.
// Using constants improves readability and maintainability over magic numbers.

// Basic Attributes
pub const SGR_RESET: u16 = 0;
pub const SGR_BOLD: u16 = 1;
pub const SGR_FAINT: u16 = 2; // Also known as dim
pub const SGR_ITALIC: u16 = 3;
pub const SGR_UNDERLINE: u16 = 4;
pub const SGR_BLINK_SLOW: u16 = 5;
pub const SGR_BLINK_RAPID: u16 = 6; // Often treated same as slow blink
pub const SGR_REVERSE: u16 = 7; // Inverse video
pub const SGR_CONCEAL: u16 = 8; // Hidden
pub const SGR_STRIKETHROUGH: u16 = 9; // Crossed-out

// Reset Specific Attributes
pub const SGR_NORMAL_INTENSITY: u16 = 22; // Neither bold nor faint
pub const SGR_NO_ITALIC: u16 = 23;
pub const SGR_NO_UNDERLINE: u16 = 24; // Turns off single and double underline
pub const SGR_NO_BLINK: u16 = 25;
pub const SGR_NO_REVERSE: u16 = 27;
pub const SGR_NO_CONCEAL: u16 = 28;
pub const SGR_NO_STRIKETHROUGH: u16 = 29;

// Foreground Colors (30-37)
pub const SGR_FG_BLACK: u16 = 30;
pub const SGR_FG_RED: u16 = 31;
pub const SGR_FG_GREEN: u16 = 32;
pub const SGR_FG_YELLOW: u16 = 33;
pub const SGR_FG_BLUE: u16 = 34;
pub const SGR_FG_MAGENTA: u16 = 35;
pub const SGR_FG_CYAN: u16 = 36;
pub const SGR_FG_WHITE: u16 = 37;
pub const SGR_FG_DEFAULT: u16 = 39;

// Background Colors (40-47)
pub const SGR_BG_BLACK: u16 = 40;
pub const SGR_BG_RED: u16 = 41;
pub const SGR_BG_GREEN: u16 = 42;
pub const SGR_BG_YELLOW: u16 = 43;
pub const SGR_BG_BLUE: u16 = 44;
pub const SGR_BG_MAGENTA: u16 = 45;
pub const SGR_BG_CYAN: u16 = 46;
pub const SGR_BG_WHITE: u16 = 47;
pub const SGR_BG_DEFAULT: u16 = 49;

// Bright Foreground Colors (90-97)
pub const SGR_FG_BRIGHT_BLACK: u16 = 90;
pub const SGR_FG_BRIGHT_RED: u16 = 91;
pub const SGR_FG_BRIGHT_GREEN: u16 = 92;
pub const SGR_FG_BRIGHT_YELLOW: u16 = 93;
pub const SGR_FG_BRIGHT_BLUE: u16 = 94;
pub const SGR_FG_BRIGHT_MAGENTA: u16 = 95;
pub const SGR_FG_BRIGHT_CYAN: u16 = 96;
pub const SGR_FG_BRIGHT_WHITE: u16 = 97;

// Bright Background Colors (100-107)
pub const SGR_BG_BRIGHT_BLACK: u16 = 100;
pub const SGR_BG_BRIGHT_RED: u16 = 101;
pub const SGR_BG_BRIGHT_GREEN: u16 = 102;
pub const SGR_BG_BRIGHT_YELLOW: u16 = 103;
pub const SGR_BG_BRIGHT_BLUE: u16 = 104;
pub const SGR_BG_BRIGHT_MAGENTA: u16 = 105;
pub const SGR_BG_BRIGHT_CYAN: u16 = 106;
pub const SGR_BG_BRIGHT_WHITE: u16 = 107;

// Extended Colors (introduced by '38' for FG, '48' for BG)
pub const SGR_EXTENDED_COLOR_FG: u16 = 38;
pub const SGR_EXTENDED_COLOR_BG: u16 = 48;
/// SGR sub-parameter: Indicates the next parameter is a 256-color palette index.
pub const SGR_EXT_MODE_256_INDEX: u16 = 5;
/// SGR sub-parameter: Indicates the next three parameters are R, G, B true color values.
pub const SGR_EXT_MODE_RGB_TRUECOLOR: u16 = 2;

// Other SGR attributes
pub const SGR_UNDERLINE_DOUBLE: u16 = 21;
pub const SGR_OVERLINED: u16 = 53;
/// SGR parameter to turn off overline (according to ECMA-48).
pub const SGR_NO_OVERLINED: u16 = 55;

pub const SGR_UNDERLINE_COLOR_SET: u16 = 58; // Followed by extended color params
pub const SGR_UNDERLINE_COLOR_DEFAULT: u16 = 59;

// --- DSR (Device Status Report) Constants ---
pub const DSR_DEFAULT: u16 = 0;
pub const DSR_STATUS_OK: u16 = 5;
pub const DSR_REPORT_CURSOR_POSITION: u16 = 6;

// --- Device Attribute Responses ---
/// Response to Primary Device Attributes (DA1) request (CSI c), identifying as a VT102-compatible terminal.
pub const DA1_RESPONSE_VT102: &[u8] = b"\x1b[?6c";

/// Response to Device Status Report (DSR) status request (CSI 5 n), indicating status OK.
pub const DSR_RESPONSE_OK: &[u8] = b"\x1b[0n";

// --- Color Definitions ---
// The local `Color` enum has been removed. We now use `crate::color::Color`.

/// Represents the intensity of a basic ANSI color (normal or bright).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorIntensity {
    Normal,
    Bright,
}

// --- SGR Attributes ---
/// Represents a single Select Graphic Rendition (SGR) attribute.
/// Now uses `crate::color::Color` for its color fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attribute {
    /// Reset all attributes to default.
    Reset,
    /// Bold text.
    Bold,
    /// Faint (dim) text.
    Faint,
    /// Italic text.
    Italic,
    /// Underlined text.
    Underline,
    /// Slow blink.
    BlinkSlow,
    /// Rapid blink.
    BlinkRapid,
    /// Inverse video.
    Reverse,
    /// Hidden text.
    Conceal,
    /// Strikethrough text.
    Strikethrough,
    /// Double underline.
    UnderlineDouble,
    /// Turn off bold/faint (normal intensity).
    NoBold,
    /// Turn off italic.
    NoItalic,
    /// Turn off underline.
    NoUnderline,
    /// Turn off blink.
    NoBlink,
    /// Turn off inverse video.
    NoReverse,
    /// Turn off hidden text.
    NoConceal,
    /// Turn off strikethrough.
    NoStrikethrough,
    /// Set foreground color.
    Foreground(Color),
    /// Set background color.
    Background(Color),
    /// Overlined text.
    Overlined,
    /// Turn off overline.
    NoOverlined,
    /// Set underline color.
    UnderlineColor(Color),
}

// --- C0 Control Enum ---
/// Represents C0 control characters (0x00-0x1F and 0x7F).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum C0Control {
    NUL = 0x00,
    SOH = 0x01,
    STX = 0x02,
    ETX = 0x03,
    EOT = 0x04,
    ENQ = 0x05,
    ACK = 0x06,
    BEL = 0x07,
    BS = 0x08,
    HT = 0x09,
    LF = 0x0A,
    VT = 0x0B,
    FF = 0x0C,
    CR = 0x0D,
    SO = 0x0E,
    SI = 0x0F,
    DLE = 0x10,
    DC1 = 0x11,
    DC2 = 0x12,
    DC3 = 0x13,
    DC4 = 0x14,
    NAK = 0x15,
    SYN = 0x16,
    ETB = 0x17,
    CAN = 0x18,
    EM = 0x19,
    SUB = 0x1A,
    ESC = 0x1B,
    FS = 0x1C,
    GS = 0x1D,
    RS = 0x1E,
    US = 0x1F,
    DEL = 0x7F,
}

impl C0Control {
    /// Creates a `C0Control` from a byte if it's a valid C0 code.
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        if (byte <= 0x1F && byte != 0x1B/* ESC is handled separately by parser state */)
            || byte == 0x7F
        {
            Some(unsafe { std::mem::transmute::<u8, C0Control>(byte) })
        } else {
            None
        }
    }
}

// --- CSI Command Enum ---
/// Represents Control Sequence Introducer (CSI) commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsiCommand {
    /// Clear tab stops.
    ClearTabStops(u16),
    /// Move cursor backward by `n`.
    CursorBackward(u16),
    /// Move cursor to column `n`.
    CursorCharacterAbsolute(u16),
    /// Move cursor down by `n`.
    CursorDown(u16),
    /// Move cursor forward by `n`.
    CursorForward(u16),
    /// Move cursor to beginning of line `n` lines down.
    CursorNextLine(u16),
    /// Move cursor to `(row, col)`. Parameters are 1-based.
    CursorPosition(u16, u16),
    /// Move cursor to beginning of line `n` lines up.
    CursorPrevLine(u16),
    /// Move cursor up by `n`.
    CursorUp(u16),
    /// Delete `n` characters.
    DeleteCharacter(u16),
    /// Delete `n` lines.
    DeleteLine(u16),
    /// Request device status report.
    DeviceStatusReport(u16),
    /// Erase `n` characters.
    EraseCharacter(u16),
    /// Erase in display (mode `n`).
    EraseInDisplay(u16),
    /// Erase in line (mode `n`).
    EraseInLine(u16),
    /// Insert `n` characters (shifting existing text).
    InsertCharacter(u16),
    /// Insert `n` lines.
    InsertLine(u16),
    /// Request primary device attributes.
    PrimaryDeviceAttributes,
    /// Soft reset.
    Reset,
    /// Reset standard mode `n`.
    ResetMode(u16),
    /// Reset private mode `n` (DEC).
    ResetModePrivate(u16),
    /// Restore cursor position (DEC).
    RestoreCursor,
    /// Restore cursor position (ANSI).
    RestoreCursorAnsi,
    /// Save cursor position (DEC).
    SaveCursor,
    /// Save cursor position (ANSI).
    SaveCursorAnsi,
    /// Scroll down by `n` lines.
    ScrollDown(u16),
    /// Scroll up by `n` lines.
    ScrollUp(u16),
    /// Set graphics rendition (SGR).
    SetGraphicsRendition(Vec<Attribute>),
    /// Set standard mode `n`.
    SetMode(u16),
    /// Set private mode `n` (DEC).
    SetModePrivate(u16),
    /// Set tab stop at current column.
    SetTabStop,
    /// Set scrolling region (top, bottom).
    SetScrollingRegion {
        /// Top line (1-based).
        top: u16,
        /// Bottom line (1-based).
        bottom: u16,
    },
    /// Set cursor style (DECSCUSR).
    SetCursorStyle {
        /// Shape parameter.
        shape: u16,
    },
    /// Window manipulation (dtterm).
    WindowManipulation {
        /// First parameter.
        ps1: u16,
        /// Optional second parameter.
        ps2: Option<u16>,
        /// Optional third parameter.
        ps3: Option<u16>,
    },
    /// Unsupported CSI sequence.
    Unsupported(Vec<u8>, Option<u8>),
}

// --- ESC Command Enum ---
/// Represents Escape (ESC) sequences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscCommand {
    /// Set tab stop at current column.
    SetTabStop,
    /// Index (move cursor down).
    Index,
    /// Next line.
    NextLine,
    /// Reverse index (move cursor up).
    ReverseIndex,
    /// Save cursor position (DEC).
    SaveCursor,
    /// Restore cursor position (DEC).
    RestoreCursor,
    /// Full reset (RIS).
    ResetToInitialState,
    /// Select character set.
    SelectCharacterSet(char, char),
    /// Single Shift 2.
    SingleShift2,
    /// Single Shift 3.
    SingleShift3,
}

// --- Main AnsiCommand Enum ---
/// Represents a parsed ANSI escape command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnsiCommand {
    /// A printable character.
    Print(char),
    /// A C0 control code (e.g., CR, LF).
    C0Control(C0Control),
    /// A C1 control code.
    C1Control(u8),
    /// A CSI command.
    Csi(CsiCommand),
    /// An ESC command.
    Esc(EscCommand),
    /// An Operating System Command (OSC).
    Osc(Vec<u8>),
    /// Device Control String (DCS).
    Dcs(Vec<u8>),
    /// Privacy Message (PM).
    Pm(Vec<u8>),
    /// Application Program Command (APC).
    Apc(Vec<u8>),
    /// String Terminator (ST).
    StringTerminator,
    /// Ignored byte.
    Ignore(u8),
    /// Error byte.
    Error(u8),
}

impl AnsiCommand {
    /// Parses a C0 control code.
    pub(crate) fn from_c0(byte: u8) -> Option<Self> {
        C0Control::from_byte(byte).map(AnsiCommand::C0Control)
    }

    /*
    pub(crate) fn from_c1(byte: u8) -> Option<Self> {
        match byte {
            0x84 => Some(AnsiCommand::Esc(EscCommand::Index)),
            0x85 => Some(AnsiCommand::Esc(EscCommand::NextLine)),
            0x88 => Some(AnsiCommand::Esc(EscCommand::SetTabStop)),
            0x8D => Some(AnsiCommand::Esc(EscCommand::ReverseIndex)),
            0x8E => Some(AnsiCommand::Esc(EscCommand::SingleShift2)),
            0x8F => Some(AnsiCommand::Esc(EscCommand::SingleShift3)),
            0x9C => Some(AnsiCommand::StringTerminator),
            0x90 | 0x9B | 0x9D | 0x9E | 0x9F => None,
            _ => Some(AnsiCommand::C1Control(byte)),
        }
    }
    */

    /// Parses a generic escape sequence from its final character.
    pub(crate) fn from_esc(final_char: char) -> Option<Self> {
        match final_char {
            'D' => Some(AnsiCommand::Esc(EscCommand::Index)),
            'E' => Some(AnsiCommand::Esc(EscCommand::NextLine)),
            'H' => Some(AnsiCommand::Esc(EscCommand::SetTabStop)),
            'M' => Some(AnsiCommand::Esc(EscCommand::ReverseIndex)),
            '7' => Some(AnsiCommand::Esc(EscCommand::SaveCursor)),
            '8' => Some(AnsiCommand::Esc(EscCommand::RestoreCursor)),
            'c' => Some(AnsiCommand::Esc(EscCommand::ResetToInitialState)),
            'N' => Some(AnsiCommand::Esc(EscCommand::SingleShift2)),
            'O' => Some(AnsiCommand::Esc(EscCommand::SingleShift3)),
            _ => None,
        }
    }

    /// Checks if a character is a valid charset designator final byte.
    ///
    /// According to ECMA-35/ISO 2022, valid final bytes for character set designation
    /// are in the range 0x30-0x7E (ASCII '0' through '~'). This includes:
    /// - Private-use designators: 0x30-0x3F ('0'-'9', ':', ';', '<', '=', '>', '?')
    /// - Standard registered sets: 0x40-0x7E ('@'-'~')
    ///
    /// Common charset designators include:
    /// - '0' - DEC Special Character and Line Drawing Set
    /// - 'A' - UK (United Kingdom)
    /// - 'B' - USASCII
    /// - '<' - DEC Supplemental
    /// - '>' - DEC Technical
    fn is_valid_charset_designator(c: char) -> bool {
        matches!(c, '0'..='~')
    }

    /// Parses an escape sequence with an intermediate character.
    pub(crate) fn from_esc_intermediate(intermediate: char, final_char: char) -> Option<Self> {
        if ['(', ')', '*', '+'].contains(&intermediate) {
            if Self::is_valid_charset_designator(final_char) {
                Some(AnsiCommand::Esc(EscCommand::SelectCharacterSet(
                    intermediate,
                    final_char,
                )))
            } else {
                warn!(
                    "Invalid charset designator '{}' (0x{:02X}) for ESC {} sequence. \
                     Valid range is 0x30-0x7E ('0'-'~').",
                    final_char, final_char as u32, intermediate
                );
                None
            }
        } else {
            None
        }
    }

    /// Parses SGR parameters into a list of `Attribute`s.
    fn parse_sgr(params: Vec<u16>) -> Vec<Attribute> {
        if params.is_empty() {
            return vec![Attribute::Reset];
        }
        let mut attrs = Vec::with_capacity(params.len());
        let mut iter = params.iter().peekable();
        while let Some(&param) = iter.next() {
            match param {
                SGR_RESET => attrs.push(Attribute::Reset),
                SGR_BOLD => attrs.push(Attribute::Bold),
                SGR_FAINT => attrs.push(Attribute::Faint),
                SGR_ITALIC => attrs.push(Attribute::Italic),
                SGR_UNDERLINE => attrs.push(Attribute::Underline),
                SGR_BLINK_SLOW => attrs.push(Attribute::BlinkSlow),
                SGR_BLINK_RAPID => attrs.push(Attribute::BlinkRapid),
                SGR_REVERSE => attrs.push(Attribute::Reverse),
                SGR_CONCEAL => attrs.push(Attribute::Conceal),
                SGR_STRIKETHROUGH => attrs.push(Attribute::Strikethrough),
                SGR_UNDERLINE_DOUBLE => attrs.push(Attribute::UnderlineDouble),
                SGR_NORMAL_INTENSITY => attrs.push(Attribute::NoBold),
                SGR_NO_ITALIC => attrs.push(Attribute::NoItalic),
                SGR_NO_UNDERLINE => attrs.push(Attribute::NoUnderline),
                SGR_NO_BLINK => attrs.push(Attribute::NoBlink),
                SGR_NO_REVERSE => attrs.push(Attribute::NoReverse),
                SGR_NO_CONCEAL => attrs.push(Attribute::NoConceal),
                SGR_NO_STRIKETHROUGH => attrs.push(Attribute::NoStrikethrough),
                SGR_FG_BLACK..=SGR_FG_WHITE => attrs.push(Attribute::Foreground(
                    Self::map_basic_code_to_color(param - SGR_FG_BLACK, ColorIntensity::Normal),
                )),
                SGR_FG_DEFAULT => attrs.push(Attribute::Foreground(Color::Default)),
                SGR_BG_BLACK..=SGR_BG_WHITE => attrs.push(Attribute::Background(
                    Self::map_basic_code_to_color(param - SGR_BG_BLACK, ColorIntensity::Normal),
                )),
                SGR_BG_DEFAULT => attrs.push(Attribute::Background(Color::Default)),
                SGR_OVERLINED => attrs.push(Attribute::Overlined),
                SGR_NO_OVERLINED => attrs.push(Attribute::NoOverlined),
                SGR_UNDERLINE_COLOR_SET => {
                    if let Some(color) = Self::parse_extended_color(&mut iter) {
                        attrs.push(Attribute::UnderlineColor(color));
                    }
                }
                SGR_UNDERLINE_COLOR_DEFAULT => {
                    attrs.push(Attribute::UnderlineColor(Color::Default))
                }
                SGR_FG_BRIGHT_BLACK..=SGR_FG_BRIGHT_WHITE => {
                    attrs.push(Attribute::Foreground(Self::map_basic_code_to_color(
                        param - SGR_FG_BRIGHT_BLACK,
                        ColorIntensity::Bright,
                    )))
                }
                SGR_BG_BRIGHT_BLACK..=SGR_BG_BRIGHT_WHITE => {
                    attrs.push(Attribute::Background(Self::map_basic_code_to_color(
                        param - SGR_BG_BRIGHT_BLACK,
                        ColorIntensity::Bright,
                    )))
                }
                SGR_EXTENDED_COLOR_FG => {
                    if let Some(color) = Self::parse_extended_color(&mut iter) {
                        attrs.push(Attribute::Foreground(color));
                    }
                }
                SGR_EXTENDED_COLOR_BG => {
                    if let Some(color) = Self::parse_extended_color(&mut iter) {
                        attrs.push(Attribute::Background(color));
                    }
                }
                _ => {
                    warn!("Unknown SGR parameter: {}", param);
                }
            }
        }
        // Consolidate multiple Resets or Reset followed by nothing else into a single Reset.
        if attrs.is_empty() || (attrs.len() == 1 && attrs[0] == Attribute::Reset) {
            attrs.clear(); // Ensure it's clean if it was just a single reset
            attrs.push(Attribute::Reset);
        } else if attrs.iter().all(|&a| a == Attribute::Reset) && attrs.len() > 1 {
            attrs.clear();
            attrs.push(Attribute::Reset);
        }
        attrs
    }

    /// Maps a basic color code (0-7) and intensity to `crate::color::Color`.
    fn map_basic_code_to_color(code: u16, intensity: ColorIntensity) -> Color {
        let named_color = match (intensity, code) {
            (ColorIntensity::Normal, 0) => NamedColor::Black,
            (ColorIntensity::Normal, 1) => NamedColor::Red,
            (ColorIntensity::Normal, 2) => NamedColor::Green,
            (ColorIntensity::Normal, 3) => NamedColor::Yellow,
            (ColorIntensity::Normal, 4) => NamedColor::Blue,
            (ColorIntensity::Normal, 5) => NamedColor::Magenta,
            (ColorIntensity::Normal, 6) => NamedColor::Cyan,
            (ColorIntensity::Normal, 7) => NamedColor::White,
            (ColorIntensity::Bright, 0) => NamedColor::BrightBlack,
            (ColorIntensity::Bright, 1) => NamedColor::BrightRed,
            (ColorIntensity::Bright, 2) => NamedColor::BrightGreen,
            (ColorIntensity::Bright, 3) => NamedColor::BrightYellow,
            (ColorIntensity::Bright, 4) => NamedColor::BrightBlue,
            (ColorIntensity::Bright, 5) => NamedColor::BrightMagenta,
            (ColorIntensity::Bright, 6) => NamedColor::BrightCyan,
            (ColorIntensity::Bright, 7) => NamedColor::BrightWhite,
            _ => {
                warn!(
                    "Invalid basic color code: {}, intensity: {:?}",
                    code, intensity
                );
                return Color::Default; // Fallback
            }
        };
        Color::Named(named_color)
    }

    /// Parses an extended color sequence (256-color or RGB) from SGR parameters.
    /// Returns `Option<crate::color::Color>`.
    fn parse_extended_color(iter: &mut Peekable<Iter<u16>>) -> Option<Color> {
        match iter.next() {
            Some(&SGR_EXT_MODE_256_INDEX) => iter.next().and_then(|&idx_param| {
                if idx_param <= u8::MAX as u16 {
                    Some(Color::Indexed(idx_param as u8))
                } else {
                    warn!("Invalid 256-color index: {}", idx_param);
                    None
                }
            }),
            Some(&SGR_EXT_MODE_RGB_TRUECOLOR) => {
                let r = iter.next().map(|&v| v as u8);
                let g = iter.next().map(|&v| v as u8);
                let b = iter.next().map(|&v| v as u8);
                match (r, g, b) {
                    (Some(r_val), Some(g_val), Some(b_val)) => {
                        Some(Color::Rgb(r_val, g_val, b_val))
                    }
                    _ => {
                        warn!("Incomplete RGB color sequence");
                        None
                    }
                }
            }
            Some(other) => {
                warn!("Unsupported extended color mode specifier: {}", other);
                None
            }
            None => {
                warn!("Missing parameters for extended color mode");
                None
            }
        }
    }

    pub(crate) fn from_csi(
        params: Vec<u16>,
        intermediates: Vec<u8>,
        is_private: bool,
        final_byte: u8,
    ) -> Option<Self> {
        let param_or = |idx: usize, default: u16| params.get(idx).copied().unwrap_or(default);
        let param_or_1 = |idx: usize| param_or(idx, 1).max(1);

        match (is_private, intermediates.as_slice(), final_byte) {
            (false, b" ", b'q') => Some(AnsiCommand::Csi(CsiCommand::SetCursorStyle {
                // DECSCUSR
                shape: param_or(0, 1), // Default shape 1 (blinking block) or 0 (default)
            })),
            // Handle 't' for WindowManipulation, checking intermediates for safety
            (_, _, b't') if intermediates.is_empty() || intermediates == b" " => {
                // Ensure 't' is not part of a more complex sequence like DECSTUI
                let ps1 = param_or(0, 0);
                let ps2 = params.get(1).copied();
                let ps3 = params.get(2).copied();
                Some(AnsiCommand::Csi(CsiCommand::WindowManipulation {
                    ps1,
                    ps2,
                    ps3,
                }))
            }
            (true, _, b'h') => Some(AnsiCommand::Csi(CsiCommand::SetModePrivate(param_or(0, 0)))),
            (false, b"", b'h') => Some(AnsiCommand::Csi(CsiCommand::SetMode(param_or(0, 0)))),
            (true, _, b'l') => Some(AnsiCommand::Csi(CsiCommand::ResetModePrivate(param_or(
                0, 0,
            )))),
            (false, b"", b'l') => Some(AnsiCommand::Csi(CsiCommand::ResetMode(param_or(0, 0)))),
            (false, b"", b'A') => Some(AnsiCommand::Csi(CsiCommand::CursorUp(param_or_1(0)))),
            (false, b"", b'B') => Some(AnsiCommand::Csi(CsiCommand::CursorDown(param_or_1(0)))),
            (false, b"", b'C') => Some(AnsiCommand::Csi(CsiCommand::CursorForward(param_or_1(0)))),
            (false, b"", b'D') => Some(AnsiCommand::Csi(CsiCommand::CursorBackward(param_or_1(0)))),
            (false, b"", b'E') => Some(AnsiCommand::Csi(CsiCommand::CursorNextLine(param_or_1(0)))),
            (false, b"", b'F') => Some(AnsiCommand::Csi(CsiCommand::CursorPrevLine(param_or_1(0)))),
            (false, b"", b'G') => Some(AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(
                param_or_1(0),
            ))),
            (false, b"", b'H') | (false, b"", b'f') => {
                let row = param_or_1(0);
                let col = param_or_1(1);
                Some(AnsiCommand::Csi(CsiCommand::CursorPosition(row, col)))
            }
            (false, b"", b'd') => Some(AnsiCommand::Csi(CsiCommand::CursorPosition(
                // VPA
                param_or_1(0), // Row
                1,             // Column is implicitly 1
            ))),
            (false, b"", b'J') => {
                Some(AnsiCommand::Csi(CsiCommand::EraseInDisplay(param_or(0, 0))))
            }
            // In AnsiCommand::from_csi, add a case for 'c':
            (false, b"", b'c') => Some(AnsiCommand::Csi(CsiCommand::PrimaryDeviceAttributes)),
            (false, b"", b'K') => Some(AnsiCommand::Csi(CsiCommand::EraseInLine(param_or(0, 0)))),
            (false, b"", b'X') => Some(AnsiCommand::Csi(CsiCommand::EraseCharacter(param_or_1(0)))),
            (false, b"", b'@') => {
                Some(AnsiCommand::Csi(CsiCommand::InsertCharacter(param_or_1(0))))
            }
            (false, b"", b'L') => Some(AnsiCommand::Csi(CsiCommand::InsertLine(param_or_1(0)))),
            (false, b"", b'P') => {
                Some(AnsiCommand::Csi(CsiCommand::DeleteCharacter(param_or_1(0))))
            }
            (false, b"", b'M') => Some(AnsiCommand::Csi(CsiCommand::DeleteLine(param_or_1(0)))),
            (false, b"", b'S') => Some(AnsiCommand::Csi(CsiCommand::ScrollUp(param_or_1(0)))),
            // Ensure 'T' for ScrollDown is only matched if no intermediates, to avoid conflict with other 'T' sequences
            (false, b"", b'T') => Some(AnsiCommand::Csi(CsiCommand::ScrollDown(param_or_1(0)))),
            (false, b"", b'g') => Some(AnsiCommand::Csi(CsiCommand::ClearTabStops(param_or(0, 0)))),
            (false, b"", b'W') => match param_or(0, 0) {
                0 => Some(AnsiCommand::Csi(CsiCommand::SetTabStop)),
                2 => Some(AnsiCommand::Csi(CsiCommand::ClearTabStops(0))), // Clear current
                5 => Some(AnsiCommand::Csi(CsiCommand::ClearTabStops(3))), // Clear all
                _ => {
                    warn!("Unsupported CTC parameter: {:?}", params.first());
                    Some(AnsiCommand::Csi(CsiCommand::Unsupported(
                        intermediates,
                        Some(final_byte),
                    )))
                }
            },
            (false, b"", b'm') => Some(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(
                Self::parse_sgr(params),
            ))),
            (false, b"", b'n') => Some(AnsiCommand::Csi(CsiCommand::DeviceStatusReport(param_or(
                0, 0,
            )))),
            (false, b"", b's') => Some(AnsiCommand::Csi(CsiCommand::SaveCursor)), // Typically DECSC
            (false, b"", b'u') => Some(AnsiCommand::Csi(CsiCommand::RestoreCursor)), // Typically DECRC
            (false, b"", b'r') => {
                // DECSTBM
                let top = param_or(0, 1); // Default top is 1
                let bottom = param_or(1, 0); // Default bottom is 0 (often means last line)
                Some(AnsiCommand::Csi(CsiCommand::SetScrollingRegion {
                    top,
                    bottom,
                }))
            }
            _ => {
                warn!(
                    "Unsupported or unhandled CSI sequence in from_csi: Private={}, Intermediates={:?}, Final={}({}) Params={:?}",
                    is_private, intermediates, final_byte as char, final_byte, params
                );
                // Return an error/unsupported command representation
                Some(AnsiCommand::Csi(CsiCommand::Unsupported(
                    intermediates,
                    Some(final_byte),
                )))
            }
        }
    }
}

impl fmt::Display for C0Control {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            C0Control::NUL => write!(f, "NUL"),
            C0Control::SOH => write!(f, "SOH"),
            C0Control::STX => write!(f, "STX"),
            C0Control::ETX => write!(f, "ETX"),
            C0Control::EOT => write!(f, "EOT"),
            C0Control::ENQ => write!(f, "ENQ"),
            C0Control::ACK => write!(f, "ACK"),
            C0Control::BEL => write!(f, "BEL"),
            C0Control::BS => write!(f, "BS"),
            C0Control::HT => write!(f, "HT"),
            C0Control::LF => write!(f, "LF"),
            C0Control::VT => write!(f, "VT"),
            C0Control::FF => write!(f, "FF"),
            C0Control::CR => write!(f, "CR"),
            C0Control::SO => write!(f, "SO"),
            C0Control::SI => write!(f, "SI"),
            C0Control::DLE => write!(f, "DLE"),
            C0Control::DC1 => write!(f, "DC1"),
            C0Control::DC2 => write!(f, "DC2"),
            C0Control::DC3 => write!(f, "DC3"),
            C0Control::DC4 => write!(f, "DC4"),
            C0Control::NAK => write!(f, "NAK"),
            C0Control::SYN => write!(f, "SYN"),
            C0Control::ETB => write!(f, "ETB"),
            C0Control::CAN => write!(f, "CAN"),
            C0Control::EM => write!(f, "EM"),
            C0Control::SUB => write!(f, "SUB"),
            C0Control::ESC => write!(f, "ESC"),
            C0Control::FS => write!(f, "FS"),
            C0Control::GS => write!(f, "GS"),
            C0Control::RS => write!(f, "RS"),
            C0Control::US => write!(f, "US"),
            C0Control::DEL => write!(f, "DEL"),
        }
    }
}
