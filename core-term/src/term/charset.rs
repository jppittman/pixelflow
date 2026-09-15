// src/term/charset.rs

//! Defines character set related enums and mapping functions for the terminal emulator.

use log::warn; // For logging warnings, if any future logic needs it.

/// G0 character set index (0).
pub const G0: usize = 0;
/// G1 character set index (1).
pub const G1: usize = 1;
/// G2 character set index (2).
pub const G2: usize = 2;
/// G3 character set index (3).
pub const G3: usize = 3;

/// Represents the G0, G1, G2, G3 character sets that can be designated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterSet {
    /// Standard US-ASCII.
    Ascii,
    /// UK National character set.
    UkNational,
    /// DEC Special Graphics character set (line drawing).
    DecLineDrawing,
    // Other character sets like DEC Supplemental, Portuguese, etc., could be added here.
}

impl CharacterSet {
    /// Creates a `CharacterSet` from its character designator (e.g., 'B' for ASCII).
    ///
    /// # Arguments
    /// * `ch` - The final character of the ESC ( G, ESC ) G, etc. sequence.
    ///
    /// Returns the corresponding `CharacterSet`, defaulting to `Ascii` for unsupported designators.
    #[must_use]
    pub fn from_char(ch: char) -> Self {
        match ch {
            'B' => CharacterSet::Ascii, // Designates G0, G1, G2, or G3 to US-ASCII
            'A' => CharacterSet::UkNational, // Designates G0, G1, G2, or G3 to UK National
            '0' => CharacterSet::DecLineDrawing, // Designates G0, G1, G2, or G3 to DEC Special Graphics
            // '1' => CharacterSet::DecAlternateRom, // DEC Alternate Character ROM Standard Character Set
            // '2' => CharacterSet::DecAlternateRomSpecial, // DEC Alternate Character ROM Special Graphics
            // Add other character set designators as needed, e.g.:
            // 'K' => CharacterSet::German,
            // '<' => CharacterSet::DecSupplemental, // (DEC Multinational)
            // '>' => CharacterSet::DecTechnical,
            // '%' => CharacterSet::DecSupplementaryGraphic, // Portuguese
            // ... and many others from different standards ...
            _ => {
                warn!(
                    "Unsupported character set designator: '{}'. Defaulting to ASCII.",
                    ch
                );
                CharacterSet::Ascii
            }
        }
    }
}

/// Maps a character to its DEC Special Graphics and Line Drawing equivalent.
///
/// This function takes a character (typically from the ASCII range `0x20` to `0x7E`)
/// and returns the corresponding line drawing or special graphics character if a mapping
/// exists in the DEC Special Graphics character set. If no specific mapping is found,
/// the original character is returned.
///
/// The mappings are based on common implementations like that seen in `st` or `xterm`.
///
/// # Arguments
/// * `ch` - The input character to map.
///
/// # Returns
/// The mapped DEC Special Graphics character, or the original character if no mapping exists.
#[must_use]
pub fn map_to_dec_line_drawing(ch: char) -> char {
    match ch {
        // Common DEC Special Graphics Mappings
        '`' => '◆',        // Diamond (often 0x60)
        'a' => '▒',        // Checkerboard/Shade (often 0x61)
        'b' => '\u{2409}', // HT symbol (often 0x62)
        'c' => '\u{240C}', // FF symbol (often 0x63)
        'd' => '\u{240D}', // CR symbol (often 0x64)
        'e' => '\u{240A}', // LF symbol (often 0x65)
        'f' => '°',        // Degree sign (often 0x66)
        'g' => '±',        // Plus/Minus sign (often 0x67)
        'h' => '\u{2424}', // NL symbol (often 0x68)
        'i' => '\u{240B}', // VT symbol (often 0x69)
        'j' => '┘',        // Lower right corner (often 0x6A)
        'k' => '┐',        // Upper right corner (often 0x6B)
        'l' => '┌',        // Upper left corner (often 0x6C)
        'm' => '└',        // Lower left corner (often 0x6D)
        'n' => '┼',        // Crossing lines/Plus (often 0x6E)
        'o' => '─',        // Horizontal line (scan 1) (often 0x6F)
        'p' => '─',        // Horizontal line (scan 3) (often 0x70)
        'q' => '─',        // Horizontal line (scan 5 - often same as 'o') (often 0x71)
        'r' => '─',        // Horizontal line (scan 7) (often 0x72)
        's' => '─',        // Horizontal line (scan 9) (often 0x73)
        't' => '├',        // Tee pointing right (often 0x74)
        'u' => '┤',        // Tee pointing left (often 0x75)
        'v' => '┴',        // Tee pointing up (often 0x76)
        'w' => '┬',        // Tee pointing down (often 0x77)
        'x' => '│',        // Vertical line (often 0x78)
        'y' => '≤',        // Less than or equal to (often 0x79)
        'z' => '≥',        // Greater than or equal to (often 0x7A)
        '{' => 'π',        // Greek pi (often 0x7B)
        '|' => '≠',        // Not equal to (often 0x7C)
        '}' => '£',        // Pound sterling (often 0x7D)
        '~' => '·',        // Centered dot/Bullet (often 0x7E)
        // Some terminals might map other characters like '_' to a horizontal line as well.
        // For '_' -> ' ', use ' ' directly if it's meant to be blank under line drawing.
        // If '_' should be a line, map it to '─'.
        // '_' => ' ', // Or '─' if it's a line
        _ => ch, // Default: return the character itself if no mapping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_should_map_the_ascii_designator_to_the_ascii_character_set_on_from_char() {
        assert_eq!(CharacterSet::from_char('B'), CharacterSet::Ascii);
    }

    #[test]
    fn it_should_map_the_uk_national_designator_to_the_uk_national_character_set_on_from_char() {
        assert_eq!(CharacterSet::from_char('A'), CharacterSet::UkNational);
    }

    #[test]
    fn it_should_map_the_dec_special_graphics_designator_to_dec_line_drawing_on_from_char() {
        assert_eq!(CharacterSet::from_char('0'), CharacterSet::DecLineDrawing);
    }

    #[test]
    fn it_should_default_to_ascii_for_an_unrecognized_designator_on_from_char() {
        assert_eq!(CharacterSet::from_char('Z'), CharacterSet::Ascii);
    }

    #[test]
    fn it_should_map_every_dec_special_graphics_designator_to_its_documented_glyph() {
        let expected_mappings: &[(char, char)] = &[
            ('`', '◆'),
            ('a', '▒'),
            ('b', '\u{2409}'),
            ('c', '\u{240C}'),
            ('d', '\u{240D}'),
            ('e', '\u{240A}'),
            ('f', '°'),
            ('g', '±'),
            ('h', '\u{2424}'),
            ('i', '\u{240B}'),
            ('j', '┘'),
            ('k', '┐'),
            ('l', '┌'),
            ('m', '└'),
            ('n', '┼'),
            ('o', '─'),
            ('p', '─'),
            ('q', '─'),
            ('r', '─'),
            ('s', '─'),
            ('t', '├'),
            ('u', '┤'),
            ('v', '┴'),
            ('w', '┬'),
            ('x', '│'),
            ('y', '≤'),
            ('z', '≥'),
            ('{', 'π'),
            ('|', '≠'),
            ('}', '£'),
            ('~', '·'),
        ];
        for &(input, expected) in expected_mappings {
            assert_eq!(
                map_to_dec_line_drawing(input),
                expected,
                "mapping for '{}' was wrong",
                input
            );
        }
    }

    #[test]
    fn it_should_pass_through_characters_with_no_dec_special_graphics_mapping() {
        assert_eq!(map_to_dec_line_drawing('Z'), 'Z');
        assert_eq!(map_to_dec_line_drawing(' '), ' ');
    }
}
