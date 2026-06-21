// src/term/emulator/key_translator.rs

use crate::{
    keys::{KeySymbol, Modifiers},
    term::modes::DecPrivateModes,
};

#[rustfmt::skip]
pub(super) fn translate_key_input(
    symbol: KeySymbol,
    modifiers: Modifiers,
    text: Option<std::borrow::Cow<'_, str>>,
    dec_modes: &DecPrivateModes,
) -> Vec<u8> {
    // Pre-allocate buffer for key sequence (most are < 16 bytes)
    // to avoid reallocation on push
    let mut bytes_to_send: Vec<u8> = Vec::with_capacity(16);

    // Handle Alt modifier by prepending ESC
    if modifiers.contains(Modifiers::ALT) {
        bytes_to_send.push(0x1B); // ESC
    }

    // Handle Ctrl modifier for specific keys
    if modifiers.contains(Modifiers::CONTROL) {
        match symbol {
            KeySymbol::Char(c) if c.is_ascii_alphabetic() => {
                bytes_to_send.push((c.to_ascii_lowercase() as u8) - b'a' + 1);
                return bytes_to_send;
            }
            KeySymbol::Char('[') => {
                bytes_to_send.push(0x1b); // Ctrl+[ is ESC
                return bytes_to_send;
            }
            KeySymbol::Char('\\') => {
                bytes_to_send.push(0x1c); // Ctrl+\ is FS
                return bytes_to_send;
            }
            KeySymbol::Char(']') => {
                bytes_to_send.push(0x1d); // Ctrl+] is GS
                return bytes_to_send;
            }
            KeySymbol::Char('^') => {
                bytes_to_send.push(0x1e); // Ctrl+^ is RS
                return bytes_to_send;
            }
            KeySymbol::Char('_') => {
                bytes_to_send.push(0x1f); // Ctrl+_ is US
                return bytes_to_send;
            }
            KeySymbol::Char(' ') => {
                bytes_to_send.push(0x00); // Ctrl+Space is NUL
                return bytes_to_send;
            }
            KeySymbol::Char('?') => {
                bytes_to_send.push(0x7f); // Ctrl+? is DEL
                return bytes_to_send;
            }
            _ => {}
        }
    }

    // If text is provided, and we haven't already handled a Ctrl combination, use it.
    // However, ignore macOS private use area characters (U+F700-U+F8FF) which are
    // placeholders for special keys like arrow keys - we need to use KeySymbol instead.
    if let Some(txt_val) = &text {
        if !txt_val.is_empty() {
            // Check if the text contains private use area characters
            let has_private_use = txt_val.chars().any(|c| {
                matches!(c, '\u{E000}'..='\u{F8FF}')
            });

            if !has_private_use {
                bytes_to_send.extend(txt_val.as_bytes());
                return bytes_to_send;
            }
        }
    }

    // If no text or empty text, generate sequence from KeySymbol
    match symbol {
        KeySymbol::Enter | KeySymbol::KeypadEnter => bytes_to_send.push(b'\r'),
        KeySymbol::Backspace => bytes_to_send.push(0x08),
        KeySymbol::Tab => {
            if modifiers.contains(Modifiers::SHIFT) {
                bytes_to_send.extend_from_slice(b"\x1b[Z");
            } else {
                bytes_to_send.push(b'\t');
            }
        }
        KeySymbol::Escape => bytes_to_send.push(0x1B),

        KeySymbol::Up => bytes_to_send.extend_from_slice(
            if dec_modes.cursor_keys_app_mode { b"\x1bOA" } else { b"\x1b[A" }
        ),
        KeySymbol::Down => bytes_to_send.extend_from_slice(
            if dec_modes.cursor_keys_app_mode { b"\x1bOB" } else { b"\x1b[B" }
        ),
        KeySymbol::Right => bytes_to_send.extend_from_slice(
            if dec_modes.cursor_keys_app_mode { b"\x1bOC" } else { b"\x1b[C" }
        ),
        KeySymbol::Left => bytes_to_send.extend_from_slice(
            if dec_modes.cursor_keys_app_mode { b"\x1bOD" } else { b"\x1b[D" }
        ),

        KeySymbol::Home => bytes_to_send.extend_from_slice(b"\x1b[1~"),
        KeySymbol::End => bytes_to_send.extend_from_slice(b"\x1b[4~"),
        KeySymbol::PageUp => bytes_to_send.extend_from_slice(b"\x1b[5~"),
        KeySymbol::PageDown => bytes_to_send.extend_from_slice(b"\x1b[6~"),
        KeySymbol::Insert => bytes_to_send.extend_from_slice(b"\x1b[2~"),
        KeySymbol::Delete => bytes_to_send.extend_from_slice(b"\x1b[3~"),

        KeySymbol::F1 => bytes_to_send.extend_from_slice(b"\x1bOP"),
        KeySymbol::F2 => bytes_to_send.extend_from_slice(b"\x1bOQ"),
        KeySymbol::F3 => bytes_to_send.extend_from_slice(b"\x1bOR"),
        KeySymbol::F4 => bytes_to_send.extend_from_slice(b"\x1bOS"),
        KeySymbol::F5 => bytes_to_send.extend_from_slice(b"\x1b[15~"),
        KeySymbol::F6 => bytes_to_send.extend_from_slice(b"\x1b[17~"),
        KeySymbol::F7 => bytes_to_send.extend_from_slice(b"\x1b[18~"),
        KeySymbol::F8 => bytes_to_send.extend_from_slice(b"\x1b[19~"),
        KeySymbol::F9 => bytes_to_send.extend_from_slice(b"\x1b[20~"),
        KeySymbol::F10 => bytes_to_send.extend_from_slice(b"\x1b[21~"),
        KeySymbol::F11 => bytes_to_send.extend_from_slice(b"\x1b[23~"),
        KeySymbol::F12 => bytes_to_send.extend_from_slice(b"\x1b[24~"),

        KeySymbol::KeypadPlus => bytes_to_send.push(b'+'),
        KeySymbol::KeypadMinus => bytes_to_send.push(b'-'),
        KeySymbol::KeypadMultiply => bytes_to_send.push(b'*'),
        KeySymbol::KeypadDivide => bytes_to_send.push(b'/'),
        KeySymbol::KeypadDecimal => bytes_to_send.push(b'.'),
        KeySymbol::Keypad0 => bytes_to_send.push(b'0'),
        KeySymbol::Keypad1 => bytes_to_send.push(b'1'),
        KeySymbol::Keypad2 => bytes_to_send.push(b'2'),
        KeySymbol::Keypad3 => bytes_to_send.push(b'3'),
        KeySymbol::Keypad4 => bytes_to_send.push(b'4'),
        KeySymbol::Keypad5 => bytes_to_send.push(b'5'),
        KeySymbol::Keypad6 => bytes_to_send.push(b'6'),
        KeySymbol::Keypad7 => bytes_to_send.push(b'7'),
        KeySymbol::Keypad8 => bytes_to_send.push(b'8'),
        KeySymbol::Keypad9 => bytes_to_send.push(b'9'),

        KeySymbol::Char(c) => {
            let mut buf = [0; 4];
            bytes_to_send.extend(c.encode_utf8(&mut buf).as_bytes());
        }
        _ => {
            log::trace!(
                "Unhandled KeySymbol (with no text): {:?}, Modifiers: {:?}",
                symbol,
                modifiers
            );
        }
    }

    bytes_to_send
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{KeySymbol, Modifiers};
    use crate::term::modes::DecPrivateModes;

    #[test]
    fn simple_chars_works() {
        let modes = DecPrivateModes::default();
        assert_eq!(
            translate_key_input(
                KeySymbol::Char('a'),
                Modifiers::empty(),
                Some(std::borrow::Cow::Borrowed("a")),
                &modes
            ),
            vec![b'a']
        );
        assert_eq!(
            translate_key_input(KeySymbol::Enter, Modifiers::empty(), None, &modes),
            vec![b'\r']
        );
    }

    #[test]
    fn ctrl_chars_works() {
        let modes = DecPrivateModes::default();
        // Test Ctrl+c
        assert_eq!(
            translate_key_input(KeySymbol::Char('c'), Modifiers::CONTROL, None, &modes),
            vec![0x03]
        );
        // Test Ctrl+Space
        assert_eq!(
            translate_key_input(KeySymbol::Char(' '), Modifiers::CONTROL, None, &modes),
            vec![0x00]
        );
    }

    #[test]
    fn alt_chars_works() {
        let modes = DecPrivateModes::default();
        assert_eq!(
            translate_key_input(KeySymbol::Char('a'), Modifiers::ALT, None, &modes),
            vec![0x1b, b'a']
        );
    }

    #[test]
    fn arrow_keys_normal_mode_works() {
        let modes = DecPrivateModes {
            cursor_keys_app_mode: false,
            ..Default::default()
        };
        assert_eq!(
            translate_key_input(KeySymbol::Up, Modifiers::empty(), None, &modes),
            b"\x1b[A".to_vec()
        );
        assert_eq!(
            translate_key_input(KeySymbol::Down, Modifiers::empty(), None, &modes),
            b"\x1b[B".to_vec()
        );
        assert_eq!(
            translate_key_input(KeySymbol::Right, Modifiers::empty(), None, &modes),
            b"\x1b[C".to_vec()
        );
        assert_eq!(
            translate_key_input(KeySymbol::Left, Modifiers::empty(), None, &modes),
            b"\x1b[D".to_vec()
        );
    }

    #[test]
    fn arrow_keys_app_mode_works() {
        let modes = DecPrivateModes {
            cursor_keys_app_mode: true,
            ..Default::default()
        };
        assert_eq!(
            translate_key_input(KeySymbol::Up, Modifiers::empty(), None, &modes),
            b"\x1bOA".to_vec()
        );
        assert_eq!(
            translate_key_input(KeySymbol::Down, Modifiers::empty(), None, &modes),
            b"\x1bOB".to_vec()
        );
        assert_eq!(
            translate_key_input(KeySymbol::Right, Modifiers::empty(), None, &modes),
            b"\x1bOC".to_vec()
        );
        assert_eq!(
            translate_key_input(KeySymbol::Left, Modifiers::empty(), None, &modes),
            b"\x1bOD".to_vec()
        );
    }

    #[test]
    fn shift_tab_works() {
        let modes = DecPrivateModes::default();
        assert_eq!(
            translate_key_input(KeySymbol::Tab, Modifiers::SHIFT, None, &modes),
            b"\x1b[Z".to_vec()
        );
    }

    #[test]
    fn macos_private_use_area_characters_ignored_for_arrow_keys() {
        // macOS sends U+F700-F703 for arrow keys in the `characters` field
        // These should be ignored and the KeySymbol should be used instead
        let modes = DecPrivateModes::default();

        // Up arrow with macOS private use placeholder
        assert_eq!(
            translate_key_input(
                KeySymbol::Up,
                Modifiers::empty(),
                Some(std::borrow::Cow::Borrowed("\u{F700}")), // macOS sends this for up arrow
                &modes
            ),
            b"\x1b[A".to_vec()
        );

        // Down arrow
        assert_eq!(
            translate_key_input(
                KeySymbol::Down,
                Modifiers::empty(),
                Some(std::borrow::Cow::Borrowed("\u{F701}")), // macOS sends this for down arrow
                &modes
            ),
            b"\x1b[B".to_vec()
        );

        // Right arrow
        assert_eq!(
            translate_key_input(
                KeySymbol::Right,
                Modifiers::empty(),
                Some(std::borrow::Cow::Borrowed("\u{F703}")), // macOS sends this for right arrow
                &modes
            ),
            b"\x1b[C".to_vec()
        );

        // Left arrow
        assert_eq!(
            translate_key_input(
                KeySymbol::Left,
                Modifiers::empty(),
                Some(std::borrow::Cow::Borrowed("\u{F702}")), // macOS sends this for left arrow
                &modes
            ),
            b"\x1b[D".to_vec()
        );
    }
}
