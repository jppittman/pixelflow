use bitflags::bitflags;
use serde::{Deserialize, Serialize};

bitflags! {
    /// Represents a keyboard modifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
    pub struct Modifiers: u8 {
        const SHIFT = 1 << 0;
        const CONTROL = 1 << 1;
        const ALT = 1 << 2; // Also known as Option on macOS
        const SUPER = 1 << 3; // Also known as Windows key or Command key
        const CAPS_LOCK = 1 << 4;
        const NUM_LOCK = 1 << 5;
    }
}

/// Represents a key symbol.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub enum KeySymbol {
    // Alphanumeric keys
    Char(char),

    // Function keys
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,

    // Modifier keys
    Shift,
    Control,
    Alt,
    Super,
    CapsLock,
    NumLock,

    // Navigation keys
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Insert,
    Delete,

    // Other common keys
    Enter,
    Backspace,
    Tab,
    Escape,
    PrintScreen,
    ScrollLock,
    Pause,

    // Keypad keys
    Keypad0,
    Keypad1,
    Keypad2,
    Keypad3,
    Keypad4,
    Keypad5,
    Keypad6,
    Keypad7,
    Keypad8,
    Keypad9,
    KeypadEnter,
    KeypadPlus,
    KeypadMinus,
    KeypadMultiply,
    KeypadDivide,
    KeypadDecimal,
    KeypadEquals,

    Menu,

    #[default]
    Unknown,
}

impl KeySymbol {
    #[must_use]
    pub fn is_modifier(&self) -> bool {
        matches!(
            self,
            KeySymbol::Shift
                | KeySymbol::Control
                | KeySymbol::Alt
                | KeySymbol::Super
                | KeySymbol::CapsLock
                | KeySymbol::NumLock
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    ScrollUp,
    ScrollDown,
    Other(u8),
}

/// Which of the system's selections a paste reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Selection {
    /// The clipboard: what an explicit copy put there.
    Clipboard,
    /// The primary selection: whatever text was last highlighted, anywhere
    /// (X11). A platform without one reads its clipboard instead.
    Primary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
}
