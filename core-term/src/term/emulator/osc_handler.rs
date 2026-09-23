// src/term/emulator/osc_handler.rs

use super::TerminalEmulator;
use crate::term::action::{EmulatorAction, Selection};
use log::debug;

/// OSC 52: manipulate selection data.
const OSC_SELECTION: u32 = 52;
/// An OSC 52 payload asking for the selection's contents back.
const OSC_SELECTION_QUERY: &str = "?";

impl TerminalEmulator {
    pub(super) fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let osc_str = String::from_utf8_lossy(&data);
        // PERFORMANCE: Avoid heap allocation from splitn().collect::<Vec<_>>() by using split_once().
        // This makes OSC parsing faster and avoids unnecessary memory allocations.
        let (ps_str, content_str) = match osc_str.split_once(';') {
            Some((ps, content)) => (ps, content),
            None => {
                // No semicolon found, treat the whole string as Ps, content is empty
                debug!(
                    "OSC sequence without semicolon: '{}'. Interpreting Ps='{}', Pt=''",
                    osc_str, osc_str
                );
                (osc_str.as_ref(), "")
            }
        };

        // Attempt to parse Ps, default to 0 if parsing fails (e.g., "Implicit Title")
        // Using u32::MAX as a sentinel for unhandled 'ps' codes later is fine,
        // but for the default when parsing "text" as 'ps', '0' is more appropriate
        // as per the test's expectation for implicit title setting.
        let ps = ps_str.parse::<u32>().unwrap_or(0);

        match ps {
            0 | 2 => {
                // OSC Set Icon Name (0) or Set Window Title (2)
                // For Ps=0 where ps_str was unparseable (like "Implicit Title"),
                // content_str will be "" as set above.
                // For Ps=0 where ps_str was "0", content_str will be from parts[1] or "".
                Some(EmulatorAction::SetTitle(content_str.to_string()))
            }
            OSC_SELECTION => set_selection(content_str),
            _ => {
                debug!(
                    "Unhandled OSC command code: Ps={}, Pt='{}'",
                    ps, content_str
                );
                None
            }
        }
    }
}

/// OSC 52 `Pc;Pd`: put base64-encoded `Pd` in the selections `Pc` names.
///
/// `c` (or no target at all, which xterm reads as its configured default) is
/// the clipboard; `p` alone is the primary selection. A `Pd` of `?` asks for
/// the selection's contents to be written back to the program, which would
/// let anything that can print to the terminal read the user's clipboard, so
/// it is refused.
fn set_selection(content: &str) -> Option<EmulatorAction> {
    let (targets, payload) = content.split_once(';')?;
    if payload == OSC_SELECTION_QUERY {
        debug!("OSC 52: refusing a selection query");
        return None;
    }
    let selection = match (
        targets.is_empty() || targets.contains('c'),
        targets.contains('p'),
    ) {
        (false, true) => Selection::Primary,
        _ => Selection::Clipboard,
    };
    let text = String::from_utf8(decode_base64(payload)?).ok()?;
    Some(EmulatorAction::Copy { selection, text })
}

/// Decodes standard base64 (RFC 4648 §4), padding optional. `None` for any
/// character outside the alphabet.
fn decode_base64(encoded: &str) -> Option<Vec<u8>> {
    let sextet = |c: u8| -> Option<u32> {
        Some(u32::from(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        }))
    };
    let digits = encoded.trim_end_matches('=').as_bytes();
    let mut decoded = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut bits = 0u32;
        for &digit in chunk {
            bits = (bits << 6) | sextet(digit)?;
        }
        // A chunk of n digits carries 6n bits, the top 8(n-1) of them data.
        let data_bytes = chunk.len().saturating_sub(1);
        bits <<= 6 * (4 - chunk.len());
        decoded.extend_from_slice(&bits.to_be_bytes()[1..=data_bytes]);
    }
    Some(decoded)
}

#[cfg(test)]
mod tests {
    use crate::ansi::commands::AnsiCommand;
    use crate::term::action::{EmulatorAction, Selection};
    use crate::term::{EmulatorInput, TerminalEmulator};

    fn osc(payload: &str) -> Option<EmulatorAction> {
        TerminalEmulator::new(80, 24).interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(
            payload.as_bytes().to_vec(),
        )))
    }

    fn copy(selection: Selection, text: &str) -> Option<EmulatorAction> {
        Some(EmulatorAction::Copy {
            selection,
            text: text.to_string(),
        })
    }

    #[test]
    fn osc_52_puts_decoded_text_in_the_named_selection() {
        assert_eq!(osc("52;c;aGVsbG8="), copy(Selection::Clipboard, "hello"));
        assert_eq!(osc("52;;aGVsbG8"), copy(Selection::Clipboard, "hello"));
        assert_eq!(osc("52;p;aMOpbGxv"), copy(Selection::Primary, "héllo"));
        assert_eq!(osc("52;c;"), copy(Selection::Clipboard, ""));
    }

    #[test]
    fn osc_52_refuses_to_read_the_selection_back_and_ignores_garbage() {
        assert_eq!(osc("52;c;?"), None);
        assert_eq!(osc("52;c;not base64!"), None);
        assert_eq!(osc("52;c;/w=="), None, "0xFF is not UTF-8");
    }
}
