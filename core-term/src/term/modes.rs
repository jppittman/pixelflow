// src/term/modes.rs

//! Defines various mode-related enums and structs used by the terminal emulator.
//! This includes modes for erasing content, DEC private modes, and general
//! mode types for sequences like SM/RM.

use log::warn; // For logging warnings on unknown mode values

/// Defines the modes for erase operations (ED - Erase in Display, EL - Erase in Line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub(super) enum EraseMode {
    /// Erase from the active position to the end of the screen/line.
    ToEnd = 0,
    /// Erase from the start of the screen/line to the active position.
    ToStart = 1,
    /// Erase the entire screen/line.
    All = 2,
    /// Erase the scrollback buffer (for ED only).
    Scrollback = 3,
    /// Represents an unknown or unsupported erase mode.
    Unknown = u16::MAX,
}

impl From<u16> for EraseMode {
    fn from(value: u16) -> Self {
        match value {
            0 => EraseMode::ToEnd,
            1 => EraseMode::ToStart,
            2 => EraseMode::All,
            3 => EraseMode::Scrollback,
            _ => {
                warn!("Unknown erase mode value: {}", value);
                EraseMode::Unknown
            }
        }
    }
}

/// Defines specific DEC private mode numbers.
/// These are used in CSI ? Pm h (set) / l (reset) sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub(super) enum DecModeConstant {
    /// Application Cursor Keys (DECCKM). Affects sequences sent by cursor keys.
    CursorKeys = 1,
    /// Origin Mode (DECOM). Changes how cursor coordinates are interpreted relative to scrolling margins.
    Origin = 6,
    /// Autowrap Mode (DECAWM). Controls whether cursor wraps to next line at EOL.
    AutoWrapMode = 7, // DECAWM
    /// Text Cursor Enable Mode (DECTCEM). Controls visibility of the text cursor.
    TextCursorEnable = 25,

    // Mouse modes
    /// X10 Mouse Reporting (Compatibility). Sends basic click information.
    MouseX10 = 9,
    /// VT200 Mouse Reporting. Sends click information with button and modifiers.
    MouseVt200 = 1000,
    /// VT200 Highlight Mouse Tracking. Reports mouse movement while a button is held (rarely used).
    MouseVt200Highlight = 1001,
    /// Button-Event Mouse Tracking. Reports clicks and mouse movement while any button is pressed.
    MouseButtonEvent = 1002,
    /// Any-Event Mouse Tracking. Reports all mouse movements, regardless of button state, plus clicks.
    MouseAnyEvent = 1003,
    /// Send FocusIn/FocusOut events as escape sequences.
    FocusEvent = 1004,
    /// UTF-8 Mouse Coordinate Encoding. Mouse coordinates are sent as UTF-8.
    MouseUtf8 = 1005,
    /// SGR Extended Mouse Coordinate Encoding. Uses SGR-like sequences for mouse reports.
    MouseSgr = 1006,
    /// urxvt Extended Mouse Mode. An alternative extended mouse protocol.
    MouseUrxvt = 1015,
    /// SGR Pixel Position Mouse Mode. Reports mouse position in pixels using SGR sequences.
    MousePixelPosition = 1016,

    // Screen modes
    /// Use Alternate Screen Buffer and clear it on switch (like xterm's 1047).
    AltScreenBufferClear = 1047,
    /// Save/Restore cursor state (DECSC/DECRC variant, often used with 1049).
    SaveRestoreCursor = 1048,
    /// Use Alternate Screen Buffer, save/restore cursor, and clear buffer on switch (like xterm's 1049).
    AltScreenBufferSaveRestore = 1049,

    /// Bracketed Paste Mode. Pasted text is bracketed by special sequences.
    BracketedPaste = 2004,

    /// While the mode is active, the terminal may batch rendering commands instead of drawing them immediately as they are received.
    /// The application then sends all of its updates for a single "frame."
    /// Finally, the application disables the mode with CSI ? 2026 l, which signals the terminal to process the batched commands and display the completed frame all at once.
    SynchronizedOutput = 2026,

    // Other modes sometimes encountered
    /// ATT610: Controls cursor blinking (l=stop, h=start).
    Att610CursorBlink = 12,
    /// A mode number seen in logs, behavior may be specific or undefined.
    Unknown7727 = 7727,
}

impl DecModeConstant {
    /// Converts a `u16` value to an `Option<DecModeConstant>`.
    /// Returns `None` if the value does not correspond to a known constant.
    pub fn from_u16(value: u16) -> Option<Self> {
        // Made pub as it's useful for the calling module
        match value {
            1 => Some(DecModeConstant::CursorKeys),
            6 => Some(DecModeConstant::Origin),
            7 => Some(DecModeConstant::AutoWrapMode), // DECAWM
            9 => Some(DecModeConstant::MouseX10),
            12 => Some(DecModeConstant::Att610CursorBlink),
            25 => Some(DecModeConstant::TextCursorEnable),
            1000 => Some(DecModeConstant::MouseVt200),
            1001 => Some(DecModeConstant::MouseVt200Highlight),
            1002 => Some(DecModeConstant::MouseButtonEvent),
            1003 => Some(DecModeConstant::MouseAnyEvent),
            1004 => Some(DecModeConstant::FocusEvent),
            1005 => Some(DecModeConstant::MouseUtf8),
            1006 => Some(DecModeConstant::MouseSgr),
            1015 => Some(DecModeConstant::MouseUrxvt),
            1016 => Some(DecModeConstant::MousePixelPosition),
            1047 => Some(DecModeConstant::AltScreenBufferClear),
            1048 => Some(DecModeConstant::SaveRestoreCursor),
            1049 => Some(DecModeConstant::AltScreenBufferSaveRestore),
            2004 => Some(DecModeConstant::BracketedPaste),
            2026 => Some(DecModeConstant::SynchronizedOutput),
            7727 => Some(DecModeConstant::Unknown7727),
            _ => None,
        }
    }
}

/// Represents the state of various DEC private modes toggled by sequences like DECSET/DECRST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DecPrivateModes {
    /// Origin Mode (DECOM - `?6h`/`?6l`).
    /// If true, cursor addressing is relative to the scrolling region's top-left.
    pub origin_mode: bool,
    /// Application Cursor Keys Mode (DECCKM - `?1h`/`?1l`).
    /// If true, cursor keys send application-specific sequences.
    pub cursor_keys_app_mode: bool,
    pub allow_alt_screen: bool,
    /// Indicates if the alternate screen buffer is currently active (due to modes like 1047 or 1049).
    pub using_alt_screen: bool,
    // Note: Cursor visibility (DECTCEM - ?25) is managed by `CursorController` but influenced by these modes.
    /// Bracketed Paste Mode (`?2004h`/`?2004l`).
    /// If true, pasted text is enclosed in `\x1b[200~` and `\x1b[201~`.
    pub bracketed_paste_mode: bool,
    /// Focus Event Reporting Mode (`?1004h`/`?1004l`).
    /// If true, the terminal sends CSI I for focus-in and CSI O for focus-out.
    pub focus_event_mode: bool,

    // Mouse modes - these flags track the enabled state of various mouse reporting protocols.
    /// X10 Mouse Reporting Mode (`?9h`/`?9l`).
    pub mouse_x10_mode: bool,
    /// VT200 Mouse Reporting Mode (`?1000h`/`?1000l`). Reports button presses.
    pub mouse_vt200_mode: bool,
    /// VT200 Highlight Mouse Reporting Mode (`?1001h`/`?1001l`). Reports motion with button held.
    pub mouse_vt200_highlight_mode: bool,
    /// Button-Event Mouse Reporting Mode (`?1002h`/`?1002l`). Reports motion with button held & releases.
    pub mouse_button_event_mode: bool,
    /// Any-Event Mouse Reporting Mode (`?1003h`/`?1003l`). Reports all mouse motion.
    pub mouse_any_event_mode: bool,
    /// UTF-8 Mouse Coordinate Encoding Mode (`?1005h`/`?1005l`). Coordinates sent as UTF-8.
    pub mouse_utf8_mode: bool,
    /// SGR Extended Mouse Coordinate Encoding Mode (`?1006h`/`?1006l`). Uses SGR-like sequences.
    pub mouse_sgr_mode: bool,
    // Note: mouse_urxvt_mode (?1015) and mouse_pixel_position_mode (?1016) would be added here
    // if they were being fully implemented with state flags.
    pub insert_mode: bool,
    pub linefeed_newline_mode: bool,
    pub text_cursor_enable_mode: bool,
    pub cursor_blink_mode: bool,
    pub(super) synchronized_output: bool,
    pub autowrap_mode: bool,
}

impl Default for DecPrivateModes {
    fn default() -> Self {
        DecPrivateModes {
            origin_mode: false,
            cursor_keys_app_mode: false,
            using_alt_screen: false,
            bracketed_paste_mode: false,
            focus_event_mode: false,
            mouse_x10_mode: false,
            mouse_vt200_mode: false,
            mouse_vt200_highlight_mode: false,
            mouse_button_event_mode: false,
            mouse_any_event_mode: false,
            mouse_utf8_mode: false,
            mouse_sgr_mode: false,
            text_cursor_enable_mode: true, // Default: cursor visible
            allow_alt_screen: true,        // Default: allow alt screen
            cursor_blink_mode: true,       // Default: blinking enabled (visuals TBD)
            insert_mode: false,            // Default: replace mode
            linefeed_newline_mode: true,   // Default: LF acts as LF+CR (Newline Mode)
            autowrap_mode: true,           // Default: Autowrap ON
            synchronized_output: false,
        }
    }
}

/// Represents the action to take on a mode (set or reset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeAction {
    /// Enable/set the mode.
    Enable,
    /// Disable/reset the mode.
    Disable,
}

/// Defines specific standard mode numbers used in SM/RM sequences.
/// These modes control general terminal behavior (as opposed to DEC private modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum StandardModeConstant {
    /// Insert Mode (IRM). When enabled, characters are inserted at the cursor instead of overwriting.
    InsertMode = 4,
    /// Linefeed/Newline Mode (LNM). When enabled, LF also moves cursor to the next line.
    LinefeedNewlineMode = 20,
}

impl StandardModeConstant {
    /// Converts a `u16` value to an `Option<StandardModeConstant>`.
    /// Returns `None` if the value does not correspond to a known constant.
    #[must_use]
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            4 => Some(StandardModeConstant::InsertMode),
            20 => Some(StandardModeConstant::LinefeedNewlineMode),
            _ => None,
        }
    }
}

/// Represents the type of mode being set or reset by SM/RM sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// A DEC private mode (parameter introduced by `?`).
    DecPrivate(u16),
    /// A standard ANSI mode.
    Standard(u16),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_should_map_known_values_to_their_erase_mode_on_from_u16() {
        let expected: &[(u16, EraseMode)] = &[
            (0, EraseMode::ToEnd),
            (1, EraseMode::ToStart),
            (2, EraseMode::All),
            (3, EraseMode::Scrollback),
        ];
        for &(value, mode) in expected {
            assert_eq!(EraseMode::from(value), mode, "value {} mapped wrong", value);
        }
    }

    #[test]
    fn it_should_map_an_unrecognized_value_to_unknown_on_erase_mode_from_u16() {
        assert_eq!(EraseMode::from(42), EraseMode::Unknown);
    }

    #[test]
    fn it_should_map_every_known_value_to_its_dec_mode_constant_on_from_u16() {
        let expected: &[(u16, DecModeConstant)] = &[
            (1, DecModeConstant::CursorKeys),
            (6, DecModeConstant::Origin),
            (7, DecModeConstant::AutoWrapMode),
            (9, DecModeConstant::MouseX10),
            (12, DecModeConstant::Att610CursorBlink),
            (25, DecModeConstant::TextCursorEnable),
            (1000, DecModeConstant::MouseVt200),
            (1001, DecModeConstant::MouseVt200Highlight),
            (1002, DecModeConstant::MouseButtonEvent),
            (1003, DecModeConstant::MouseAnyEvent),
            (1004, DecModeConstant::FocusEvent),
            (1005, DecModeConstant::MouseUtf8),
            (1006, DecModeConstant::MouseSgr),
            (1015, DecModeConstant::MouseUrxvt),
            (1016, DecModeConstant::MousePixelPosition),
            (1047, DecModeConstant::AltScreenBufferClear),
            (1048, DecModeConstant::SaveRestoreCursor),
            (1049, DecModeConstant::AltScreenBufferSaveRestore),
            (2004, DecModeConstant::BracketedPaste),
            (2026, DecModeConstant::SynchronizedOutput),
            (7727, DecModeConstant::Unknown7727),
        ];
        for &(value, constant) in expected {
            assert_eq!(
                DecModeConstant::from_u16(value),
                Some(constant),
                "value {} mapped wrong",
                value
            );
        }
    }

    #[test]
    fn it_should_return_none_for_an_unrecognized_value_on_dec_mode_constant_from_u16() {
        assert_eq!(DecModeConstant::from_u16(9999), None);
    }

    #[test]
    fn it_should_map_every_known_value_to_its_standard_mode_constant_on_from_u16() {
        assert_eq!(
            StandardModeConstant::from_u16(4),
            Some(StandardModeConstant::InsertMode)
        );
        assert_eq!(
            StandardModeConstant::from_u16(20),
            Some(StandardModeConstant::LinefeedNewlineMode)
        );
    }

    #[test]
    fn it_should_return_none_for_an_unrecognized_value_on_standard_mode_constant_from_u16() {
        assert_eq!(StandardModeConstant::from_u16(9999), None);
    }
}
