//! This module handles Unicode character width determination using the system's `wcwidth`
//! function via FFI, and locale initialization via a lazily initialized static controller
//! using `std::sync::OnceLock`.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::OnceLock;

// --- FFI Declarations ---
extern "C" {
    fn wcwidth(wc: c_uint) -> c_int;
    fn setlocale(category: c_int, locale: *const c_char) -> *mut c_char;
}

const LC_CTYPE: c_int = 0; // Common value for LC_CTYPE

/// Internal struct to manage the C locale initialization.
#[derive(Debug)]
struct LocaleInitializer;

impl LocaleInitializer {
    /// Performs the one-time C locale initialization for `LC_CTYPE`.
    fn new() -> Self {
        // SAFETY: Calling C's setlocale function.
        unsafe {
            // Try UTF-8 locales in order of preference
            // Start with explicit UTF-8 locales since environment may be unset
            for locale in &["C.utf8", "C.UTF-8", "en_US.UTF-8", "en_US.utf8", ""] {
                let locale_cstr = CString::new(*locale).unwrap();
                if !setlocale(LC_CTYPE, locale_cstr.as_ptr()).is_null() {
                    // Verify it actually works for CJK by testing a known wide char
                    if wcwidth('世' as c_uint) == 2 {
                        return LocaleInitializer;
                    }
                }
            }
        }
        LocaleInitializer
    }

    /// Internal method to calculate character display width.
    fn char_display_width_internal(&self, c: char) -> usize {
        // Explicitly handle zero-width characters first.
        if c == '\u{200D}' || c == '\u{200B}' {
            return 0;
        }

        let wc = c as c_uint;
        // SAFETY: Calling C's wcwidth.
        let width_from_c = unsafe { wcwidth(wc) };

        match width_from_c {
            -1 => 0, // Non-printable
            0 => 0,  // Zero-width (combining marks, etc.)
            1 => 1,
            2 => 2,
            _ => 1, // Unexpected value, default to 1
        }
    }
}

static GLOBAL_LOCALE_INITIALIZER: OnceLock<LocaleInitializer> = OnceLock::new();

/// Public function to get the display width of a character.
///
/// # Returns
/// * `0` for non-printing characters or characters that do not advance the cursor.
/// * `1` for standard-width printable characters.
/// * `2` for characters that typically occupy two terminal cells.
pub fn get_char_display_width(c: char) -> usize {
    let controller = GLOBAL_LOCALE_INITIALIZER.get_or_init(LocaleInitializer::new);
    controller.char_display_width_internal(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_char_width() {
        assert_eq!(get_char_display_width('A'), 1, "Width of 'A' should be 1");
        assert_eq!(get_char_display_width(' '), 1, "Width of space should be 1");
        assert_eq!(get_char_display_width('~'), 1, "Width of '~' should be 1");
    }

    #[test]
    fn test_box_drawing_char_widths() {
        assert_eq!(
            get_char_display_width('─'),
            1,
            "Width of U+2500 BOX DRAWINGS LIGHT HORIZONTAL"
        );
        assert_eq!(
            get_char_display_width('│'),
            1,
            "Width of U+2502 BOX DRAWINGS LIGHT VERTICAL"
        );
        assert_eq!(
            get_char_display_width('┌'),
            1,
            "Width of U+250C BOX DRAWINGS LIGHT DOWN AND RIGHT"
        );
        assert_eq!(
            get_char_display_width('┐'),
            1,
            "Width of U+2510 BOX DRAWINGS LIGHT DOWN AND LEFT"
        );
        assert_eq!(
            get_char_display_width('└'),
            1,
            "Width of U+2514 BOX DRAWINGS LIGHT UP AND RIGHT"
        );
        assert_eq!(
            get_char_display_width('┘'),
            1,
            "Width of U+2518 BOX DRAWINGS LIGHT UP AND LEFT"
        );
        assert_eq!(
            get_char_display_width('├'),
            1,
            "Width of U+251C BOX DRAWINGS LIGHT VERTICAL AND RIGHT"
        );
        assert_eq!(
            get_char_display_width('┤'),
            1,
            "Width of U+2524 BOX DRAWINGS LIGHT VERTICAL AND LEFT"
        );
        assert_eq!(
            get_char_display_width('┬'),
            1,
            "Width of U+252C BOX DRAWINGS LIGHT DOWN AND HORIZONTAL"
        );
        assert_eq!(
            get_char_display_width('┴'),
            1,
            "Width of U+2534 BOX DRAWINGS LIGHT UP AND HORIZONTAL"
        );
        assert_eq!(
            get_char_display_width('┼'),
            1,
            "Width of U+253C BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL"
        );
        // Add more box characters if you suspect issues with specific ones
    }

    #[test]
    fn test_cjk_wide_char_widths() {
        assert_eq!(
            get_char_display_width('世'),
            2,
            "Width of '世' (U+4E16) should be 2"
        );
        assert_eq!(
            get_char_display_width('界'),
            2,
            "Width of '界' (U+754C) should be 2"
        );
        assert_eq!(
            get_char_display_width('你'),
            2,
            "Width of '你' (U+4F60) should be 2"
        );
        assert_eq!(
            get_char_display_width('好'),
            2,
            "Width of '好' (U+597D) should be 2"
        );
    }

    #[test]
    fn test_control_char_widths() {
        assert_eq!(
            get_char_display_width('\u{0000}'),
            0,
            "Width of NUL (U+0000) should be 0"
        );
        assert_eq!(
            get_char_display_width('\u{0007}'),
            0,
            "Width of BEL (U+0007) should be 0"
        );
        assert_eq!(
            get_char_display_width('\u{001B}'),
            0,
            "Width of ESC (U+001B) should be 0"
        );
        // Check a C1 control character (e.g., IND - Index U+0084)
        // Rust's char::is_control should cover C1 too.
        assert_eq!(
            get_char_display_width('\u{0084}'),
            0,
            "Width of IND (U+0084) should be 0"
        );
    }

    #[test]
    fn test_zero_width_chars() {
        assert_eq!(
            get_char_display_width('\u{200D}'),
            0,
            "Width of ZWJ (U+200D) should be 0"
        );
        assert_eq!(
            get_char_display_width('\u{0301}'),
            0,
            "Width of Combining Acute Accent (U+0301) should be 0"
        );
    }

    #[test]
    fn test_locale_initializer_called() {
        // Ensure OnceLock mechanism is engaged
        let _ = get_char_display_width(' ');
        assert!(
            GLOBAL_LOCALE_INITIALIZER.get().is_some(),
            "LocaleInitializer should be initialized after first call"
        );
    }
}
