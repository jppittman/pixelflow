// src/term/emulator/osc_handler.rs

use super::TerminalEmulator;
use crate::term::action::{EmulatorAction, Selection, SelectionReport};
use crate::term::base64;
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
/// the selection's contents to be written back to the program; whether that
/// is answered is the app's call (`behavior.allow_clipboard_read`).
fn set_selection(content: &str) -> Option<EmulatorAction> {
    let (targets, payload) = content.split_once(';')?;
    let selection = match (
        targets.is_empty() || targets.contains('c'),
        targets.contains('p'),
    ) {
        (false, true) => Selection::Primary,
        _ => Selection::Clipboard,
    };
    if payload == OSC_SELECTION_QUERY {
        return Some(EmulatorAction::ReportSelection {
            selection,
            report: SelectionReport::new(targets),
        });
    }
    let text = String::from_utf8(base64::decode(payload)?).ok()?;
    Some(EmulatorAction::Copy { selection, text })
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
    fn osc_52_query_asks_for_the_named_selection_and_echoes_its_target() {
        let Some(EmulatorAction::ReportSelection { selection, report }) = osc("52;c;?") else {
            panic!("a query must ask for the selection");
        };
        assert_eq!(selection, Selection::Clipboard);
        assert_eq!(report.reply("hi"), b"\x1b]52;c;aGk=\x1b\\".to_vec());

        let Some(EmulatorAction::ReportSelection { selection, report }) = osc("52;p;?") else {
            panic!("a query must ask for the selection");
        };
        assert_eq!(selection, Selection::Primary);
        assert_eq!(report.reply(""), b"\x1b]52;p;\x1b\\".to_vec());
    }

    #[test]
    fn osc_52_ignores_garbage() {
        assert_eq!(osc("52;c;not base64!"), None);
        assert_eq!(osc("52;c;/w=="), None, "0xFF is not UTF-8");
    }
}
