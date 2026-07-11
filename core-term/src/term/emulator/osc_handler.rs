// src/term/emulator/osc_handler.rs

use super::TerminalEmulator;
use crate::term::action::EmulatorAction;
use log::debug;

impl TerminalEmulator {
    pub(super) fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let osc_str = String::from_utf8_lossy(&data);
        // PERFORMANCE: Use `split_once` instead of `.splitn(2, ';').collect::<Vec<_>>()`
        // to avoid allocating an intermediate heap vector for extracting two string segments.
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
