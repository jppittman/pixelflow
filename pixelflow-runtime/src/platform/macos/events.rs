//! Event Mapper
//!
//! Pure functions to map `NSEvent` to `DisplayEvent`.

use crate::api::private::WindowId;
use crate::display::messages::DisplayEvent;
use crate::input::{KeySymbol, Modifiers};
use crate::platform::macos::cocoa::{event_type, NSEvent};

/// Maps an `NSEvent` to a `DisplayEvent`, if applicable.
/// `window_height` is needed to flip Y coordinates (macOS origin is bottom-left).
#[must_use]
pub fn map_event(event: NSEvent, window_height: f64) -> Option<DisplayEvent> {
    let ty = event.type_();

    match ty {
        event_type::KEY_DOWN => {
            let code = event.key_code();
            let text = event.characters();
            let mods = map_modifiers(event.modifier_flags());

            // Pressed is always true for KEY_DOWN.
            Some(DisplayEvent::Key {
                id: WindowId::PRIMARY,
                symbol: map_key(code),
                modifiers: mods,
                text: Some(text),
            })
        }
        event_type::LEFT_MOUSE_DOWN | event_type::RIGHT_MOUSE_DOWN => {
            let pos = event.location_in_window();
            let button = match ty {
                event_type::LEFT_MOUSE_DOWN => 1,
                event_type::RIGHT_MOUSE_DOWN => 3,
                _ => 1,
            };
            Some(DisplayEvent::MouseButtonPress {
                id: WindowId::PRIMARY,
                button,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        event_type::LEFT_MOUSE_UP | event_type::RIGHT_MOUSE_UP => {
            let pos = event.location_in_window();
            let button = match ty {
                event_type::LEFT_MOUSE_UP => 1,
                event_type::RIGHT_MOUSE_UP => 3,
                _ => 1,
            };
            Some(DisplayEvent::MouseButtonRelease {
                id: WindowId::PRIMARY,
                button,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        event_type::MOUSE_MOVED
        | event_type::LEFT_MOUSE_DRAGGED
        | event_type::RIGHT_MOUSE_DRAGGED => {
            let pos = event.location_in_window();
            Some(DisplayEvent::MouseMove {
                id: WindowId::PRIMARY,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        event_type::SCROLL_WHEEL => {
            let pos = event.location_in_window();
            let dx = event.scrolling_delta_x() as f32;
            let dy = event.scrolling_delta_y() as f32;
            Some(DisplayEvent::MouseScroll {
                id: WindowId::PRIMARY,
                dx,
                dy,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        _ => None,
    }
}

fn map_modifiers(flags: u64) -> Modifiers {
    let mut m = Modifiers::empty();
    // NSEventModifierFlags constants
    const NS_EVENT_MODIFIER_FLAG_SHIFT: u64 = 1 << 17;
    const NS_EVENT_MODIFIER_FLAG_CONTROL: u64 = 1 << 18;
    const NS_EVENT_MODIFIER_FLAG_OPTION: u64 = 1 << 19;
    const NS_EVENT_MODIFIER_FLAG_COMMAND: u64 = 1 << 20;

    if flags & NS_EVENT_MODIFIER_FLAG_SHIFT != 0 {
        m |= Modifiers::SHIFT;
    }
    if flags & NS_EVENT_MODIFIER_FLAG_CONTROL != 0 {
        m |= Modifiers::CONTROL;
    }
    if flags & NS_EVENT_MODIFIER_FLAG_OPTION != 0 {
        m |= Modifiers::ALT;
    }
    if flags & NS_EVENT_MODIFIER_FLAG_COMMAND != 0 {
        m |= Modifiers::SUPER;
    }
    m
}

fn map_key(code: u16) -> KeySymbol {
    match code {
        0x00 => KeySymbol::Char('a'),
        0x01 => KeySymbol::Char('s'),
        0x02 => KeySymbol::Char('d'),
        0x03 => KeySymbol::Char('f'),
        0x04 => KeySymbol::Char('h'),
        0x05 => KeySymbol::Char('g'),
        0x06 => KeySymbol::Char('z'),
        0x07 => KeySymbol::Char('x'),
        0x08 => KeySymbol::Char('c'),
        0x09 => KeySymbol::Char('v'),
        0x0B => KeySymbol::Char('b'),
        0x0C => KeySymbol::Char('q'),
        0x0D => KeySymbol::Char('w'),
        0x0E => KeySymbol::Char('e'),
        0x0F => KeySymbol::Char('r'),
        0x10 => KeySymbol::Char('y'),
        0x11 => KeySymbol::Char('t'),
        0x12 => KeySymbol::Char('1'),
        0x13 => KeySymbol::Char('2'),
        0x14 => KeySymbol::Char('3'),
        0x15 => KeySymbol::Char('4'),
        0x16 => KeySymbol::Char('6'),
        0x17 => KeySymbol::Char('5'),
        0x18 => KeySymbol::Char('='), // Equal?
        0x19 => KeySymbol::Char('9'),
        0x1A => KeySymbol::Char('7'),
        0x1B => KeySymbol::Char('-'),
        0x1C => KeySymbol::Char('8'),
        0x1D => KeySymbol::Char('0'),
        0x1E => KeySymbol::Char(']'),
        0x1F => KeySymbol::Char('o'),
        0x20 => KeySymbol::Char('u'),
        0x21 => KeySymbol::Char('['),
        0x22 => KeySymbol::Char('i'),
        0x23 => KeySymbol::Char('p'),
        0x24 => KeySymbol::Enter,
        0x25 => KeySymbol::Char('l'),
        0x26 => KeySymbol::Char('j'),
        0x27 => KeySymbol::Char('\''),
        0x28 => KeySymbol::Char('k'),
        0x29 => KeySymbol::Char(';'),
        0x2A => KeySymbol::Char('\\'),
        0x2B => KeySymbol::Char(','),
        0x2C => KeySymbol::Char('/'),
        0x2D => KeySymbol::Char('n'),
        0x2E => KeySymbol::Char('m'),
        0x2F => KeySymbol::Char('.'),
        0x30 => KeySymbol::Tab,
        0x31 => KeySymbol::Char(' '), // Space
        0x32 => KeySymbol::Char('`'),
        0x33 => KeySymbol::Backspace,
        0x35 => KeySymbol::Escape,
        // Named keys, by their kVK_ virtual key codes (HIToolbox/Events.h).
        0x7B => KeySymbol::Left,
        0x7C => KeySymbol::Right,
        0x7D => KeySymbol::Down,
        0x7E => KeySymbol::Up,
        0x73 => KeySymbol::Home,
        0x77 => KeySymbol::End,
        0x74 => KeySymbol::PageUp,
        0x79 => KeySymbol::PageDown,
        0x72 => KeySymbol::Insert, // kVK_Help: the Insert position on PC keyboards
        0x75 => KeySymbol::Delete, // kVK_ForwardDelete
        0x7A => KeySymbol::F1,
        0x78 => KeySymbol::F2,
        0x63 => KeySymbol::F3,
        0x76 => KeySymbol::F4,
        0x60 => KeySymbol::F5,
        0x61 => KeySymbol::F6,
        0x62 => KeySymbol::F7,
        0x64 => KeySymbol::F8,
        0x65 => KeySymbol::F9,
        0x6D => KeySymbol::F10,
        0x67 => KeySymbol::F11,
        0x6F => KeySymbol::F12,
        0x69 => KeySymbol::F13,
        0x6B => KeySymbol::F14,
        0x71 => KeySymbol::F15,
        0x6A => KeySymbol::F16,
        0x40 => KeySymbol::F17,
        0x4F => KeySymbol::F18,
        0x50 => KeySymbol::F19,
        0x5A => KeySymbol::F20,
        0x52 => KeySymbol::Keypad0,
        0x53 => KeySymbol::Keypad1,
        0x54 => KeySymbol::Keypad2,
        0x55 => KeySymbol::Keypad3,
        0x56 => KeySymbol::Keypad4,
        0x57 => KeySymbol::Keypad5,
        0x58 => KeySymbol::Keypad6,
        0x59 => KeySymbol::Keypad7,
        0x5B => KeySymbol::Keypad8,
        0x5C => KeySymbol::Keypad9,
        0x41 => KeySymbol::KeypadDecimal,
        0x43 => KeySymbol::KeypadMultiply,
        0x45 => KeySymbol::KeypadPlus,
        0x4B => KeySymbol::KeypadDivide,
        0x4C => KeySymbol::KeypadEnter,
        0x4E => KeySymbol::KeypadMinus,
        0x51 => KeySymbol::KeypadEquals,
        0x47 => KeySymbol::NumLock, // kVK_ANSI_KeypadClear
        0x38 | 0x3C => KeySymbol::Shift,
        0x3B | 0x3E => KeySymbol::Control,
        0x3A | 0x3D => KeySymbol::Alt,
        0x37 | 0x36 => KeySymbol::Super,
        0x39 => KeySymbol::CapsLock,
        0x6E => KeySymbol::Menu, // kVK_ContextualMenu
        _ => KeySymbol::Unknown,
    }
}
