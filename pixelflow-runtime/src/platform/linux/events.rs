//! Event mapping for X11 -> DisplayEvent.

use crate::display::messages::{DisplayEvent, Surface, WindowId};
use crate::input::{KeySymbol, Modifiers};
use std::ffi::c_int;
use std::ptr;
use x11::{keysym, xlib};

pub fn map_event(
    event: &xlib::XEvent,
    window: &mut super::window::X11Window,
    window_id: WindowId,
) -> Option<DisplayEvent> {
    unsafe {
        match event.type_ {
            xlib::ClientMessage => {
                let client = event.client_message;
                if client.data.as_longs()[0] as xlib::Atom == window.wm_delete_window {
                    return Some(DisplayEvent::CloseRequested { id: window_id });
                }
                None
            }
            xlib::SelectionRequest => {
                handle_selection_request(event, window);
                None
            }
            xlib::SelectionNotify => handle_selection_notify(event, window),
            xlib::KeyPress => {
                let key_event = event.key;
                let mut keysym = 0;
                let mut buffer = [0u8; 32];
                let count = xlib::XLookupString(
                    &key_event as *const _ as *mut _,
                    buffer.as_mut_ptr() as *mut i8,
                    buffer.len() as c_int,
                    &mut keysym,
                    ptr::null_mut(),
                );
                let text = if count > 0 {
                    Some(String::from_utf8_lossy(&buffer[..count as usize]).to_string())
                } else {
                    None
                };
                let modifiers = extract_modifiers(key_event.state);
                let symbol = xkeysym_to_keysymbol(keysym, text.as_deref().unwrap_or(""));
                Some(DisplayEvent::Key {
                    id: window_id,
                    symbol,
                    modifiers,
                    text,
                })
            }
            xlib::ConfigureNotify => {
                let conf = event.configure;
                // Only emit Resized if size actually changed
                if conf.width as u32 != window.width || conf.height as u32 != window.height {
                    window.width = conf.width as u32;
                    window.height = conf.height as u32;
                    Some(DisplayEvent::Resized {
                        surface: Surface {
                            id: window_id,
                            width_px: window.width,
                            height_px: window.height,
                            frame_width: window.width,
                            frame_height: window.height,
                            scale: window.scale_factor,
                        },
                    })
                } else {
                    None
                }
            }
            xlib::FocusIn => Some(DisplayEvent::FocusGained { id: window_id }),
            xlib::FocusOut => Some(DisplayEvent::FocusLost { id: window_id }),
            xlib::ButtonPress => handle_button_press(event.button, window_id, event.button.state),
            xlib::ButtonRelease => {
                handle_button_release(event.button, window_id, event.button.state)
            }
            xlib::MotionNotify => {
                let e = event.motion;
                let modifiers = extract_modifiers(e.state);
                Some(DisplayEvent::MouseMove {
                    id: window_id,
                    x: e.x,
                    y: e.y,
                    modifiers,
                })
            }
            _ => None,
        }
    }
}

unsafe fn handle_selection_request(event: &xlib::XEvent, window: &mut super::window::X11Window) {
    let req = event.selection_request;
    let mut response: xlib::XSelectionEvent = std::mem::zeroed();
    response.type_ = xlib::SelectionNotify;
    response.requestor = req.requestor;
    response.selection = req.selection;
    response.target = req.target;
    response.time = req.time;
    response.property = req.property;

    let offered = window.selection_data(req.selection);
    match req.target {
        _ if offered.is_none() => {
            response.property = 0; // Not a selection this window owns
        }
        t if t == window.atoms.targets => {
            let targets = [
                window.atoms.targets,
                window.atoms.utf8_string,
                window.atoms.text,
                window.atoms.xa_string,
            ];
            xlib::XChangeProperty(
                window.display,
                req.requestor,
                req.property,
                xlib::XA_ATOM,
                32,
                xlib::PropModeReplace,
                targets.as_ptr() as *const u8,
                targets.len() as i32,
            );
        }
        t if t == window.atoms.utf8_string
            || t == window.atoms.text
            || t == window.atoms.xa_string =>
        {
            let data = offered.unwrap_or_default().as_bytes();
            xlib::XChangeProperty(
                window.display,
                req.requestor,
                req.property,
                req.target,
                8,
                xlib::PropModeReplace,
                data.as_ptr(),
                data.len() as i32,
            );
        }
        _ => {
            response.property = 0; // Reject
        }
    }
    xlib::XSendEvent(
        window.display,
        req.requestor,
        xlib::False,
        0,
        &mut response as *mut _ as *mut xlib::XEvent,
    );
    xlib::XFlush(window.display);
}

unsafe fn handle_selection_notify(
    event: &xlib::XEvent,
    window: &mut super::window::X11Window,
) -> Option<DisplayEvent> {
    let sel = event.selection;
    if sel.property == 0 {
        return None;
    }

    let mut type_ret = 0;
    let mut format_ret = 0;
    let mut nitems = 0;
    let mut bytes_after = 0;
    let mut prop_ret: *mut u8 = ptr::null_mut();

    xlib::XGetWindowProperty(
        window.display,
        sel.requestor,
        sel.property,
        0,
        i64::MAX / 4,
        xlib::True,
        xlib::AnyPropertyType as u64,
        &mut type_ret,
        &mut format_ret,
        &mut nitems,
        &mut bytes_after,
        &mut prop_ret,
    );

    if !prop_ret.is_null() {
        let data = std::slice::from_raw_parts(prop_ret, nitems as usize);
        let text = String::from_utf8_lossy(data).to_string();
        xlib::XFree(prop_ret as *mut std::ffi::c_void);
        return Some(DisplayEvent::PasteData { text });
    }
    None
}

fn handle_button_press(e: xlib::XButtonEvent, id: WindowId, state: u32) -> Option<DisplayEvent> {
    let modifiers = extract_modifiers(state);
    match e.button {
        4 => Some(DisplayEvent::MouseScroll {
            id,
            dx: 0.0,
            dy: 1.0,
            x: e.x,
            y: e.y,
            modifiers,
        }),
        5 => Some(DisplayEvent::MouseScroll {
            id,
            dx: 0.0,
            dy: -1.0,
            x: e.x,
            y: e.y,
            modifiers,
        }),
        6 => Some(DisplayEvent::MouseScroll {
            id,
            dx: -1.0,
            dy: 0.0,
            x: e.x,
            y: e.y,
            modifiers,
        }),
        7 => Some(DisplayEvent::MouseScroll {
            id,
            dx: 1.0,
            dy: 0.0,
            x: e.x,
            y: e.y,
            modifiers,
        }),
        _ => Some(DisplayEvent::MouseButtonPress {
            id,
            button: e.button as u8,
            x: e.x,
            y: e.y,
            modifiers,
        }),
    }
}

fn handle_button_release(e: xlib::XButtonEvent, id: WindowId, state: u32) -> Option<DisplayEvent> {
    if e.button >= 4 && e.button <= 7 {
        return None;
    }
    let modifiers = extract_modifiers(state);
    Some(DisplayEvent::MouseButtonRelease {
        id,
        button: e.button as u8,
        x: e.x,
        y: e.y,
        modifiers,
    })
}

fn extract_modifiers(state: u32) -> Modifiers {
    let mut modifiers = Modifiers::empty();
    if (state & xlib::ShiftMask) != 0 {
        modifiers.insert(Modifiers::SHIFT);
    }
    if (state & xlib::ControlMask) != 0 {
        modifiers.insert(Modifiers::CONTROL);
    }
    if (state & xlib::Mod1Mask) != 0 {
        modifiers.insert(Modifiers::ALT);
    }
    if (state & xlib::Mod4Mask) != 0 {
        modifiers.insert(Modifiers::SUPER);
    }
    modifiers
}

/// Keysyms at or above this carry a Unicode code point in their low bits.
const UNICODE_KEYSYM_BASE: u32 = 0x0100_0000;

/// The key a keysym names.
///
/// The keysym is the key; `text` is only what it typed. Reading the text
/// first reported Enter as `'\r'` and Ctrl+C as U+0003 — characters, where
/// every other platform reports keys — so named keys are looked up by keysym,
/// a Latin-1 or Unicode keysym names its character, and the typed text is
/// consulted only for the legacy keysyms that do neither.
fn xkeysym_to_keysymbol(keysym_val: xlib::KeySym, text: &str) -> KeySymbol {
    let keysym_val = keysym_val as u32;
    if let Some(named) = named_key(keysym_val) {
        return named;
    }
    let from_keysym = match keysym_val {
        0x20..=0x7e | 0xa0..=0xff => char::from_u32(keysym_val),
        UNICODE_KEYSYM_BASE.. => char::from_u32(keysym_val - UNICODE_KEYSYM_BASE),
        _ => None,
    };
    if let Some(c) = from_keysym {
        return KeySymbol::Char(c);
    }
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if c != '\u{FFFD}' => KeySymbol::Char(c),
        _ => KeySymbol::Unknown,
    }
}

/// Keys that are not characters. The keypad's navigation keysyms (Num Lock
/// off) are the keys they act as.
fn named_key(keysym_val: u32) -> Option<KeySymbol> {
    Some(match keysym_val {
        keysym::XK_Return => KeySymbol::Enter,
        keysym::XK_BackSpace => KeySymbol::Backspace,
        keysym::XK_Tab | keysym::XK_ISO_Left_Tab => KeySymbol::Tab,
        keysym::XK_Escape => KeySymbol::Escape,
        keysym::XK_Left | keysym::XK_KP_Left => KeySymbol::Left,
        keysym::XK_Right | keysym::XK_KP_Right => KeySymbol::Right,
        keysym::XK_Up | keysym::XK_KP_Up => KeySymbol::Up,
        keysym::XK_Down | keysym::XK_KP_Down => KeySymbol::Down,
        keysym::XK_Page_Up | keysym::XK_KP_Page_Up => KeySymbol::PageUp,
        keysym::XK_Page_Down | keysym::XK_KP_Page_Down => KeySymbol::PageDown,
        keysym::XK_Home | keysym::XK_KP_Home => KeySymbol::Home,
        keysym::XK_End | keysym::XK_KP_End => KeySymbol::End,
        keysym::XK_Insert | keysym::XK_KP_Insert => KeySymbol::Insert,
        keysym::XK_Delete | keysym::XK_KP_Delete => KeySymbol::Delete,
        keysym::XK_F1 => KeySymbol::F1,
        keysym::XK_F2 => KeySymbol::F2,
        keysym::XK_F3 => KeySymbol::F3,
        keysym::XK_F4 => KeySymbol::F4,
        keysym::XK_F5 => KeySymbol::F5,
        keysym::XK_F6 => KeySymbol::F6,
        keysym::XK_F7 => KeySymbol::F7,
        keysym::XK_F8 => KeySymbol::F8,
        keysym::XK_F9 => KeySymbol::F9,
        keysym::XK_F10 => KeySymbol::F10,
        keysym::XK_F11 => KeySymbol::F11,
        keysym::XK_F12 => KeySymbol::F12,
        keysym::XK_F13 => KeySymbol::F13,
        keysym::XK_F14 => KeySymbol::F14,
        keysym::XK_F15 => KeySymbol::F15,
        keysym::XK_F16 => KeySymbol::F16,
        keysym::XK_F17 => KeySymbol::F17,
        keysym::XK_F18 => KeySymbol::F18,
        keysym::XK_F19 => KeySymbol::F19,
        keysym::XK_F20 => KeySymbol::F20,
        keysym::XK_F21 => KeySymbol::F21,
        keysym::XK_F22 => KeySymbol::F22,
        keysym::XK_F23 => KeySymbol::F23,
        keysym::XK_F24 => KeySymbol::F24,
        keysym::XK_KP_0 => KeySymbol::Keypad0,
        keysym::XK_KP_1 => KeySymbol::Keypad1,
        keysym::XK_KP_2 => KeySymbol::Keypad2,
        keysym::XK_KP_3 => KeySymbol::Keypad3,
        keysym::XK_KP_4 => KeySymbol::Keypad4,
        keysym::XK_KP_5 => KeySymbol::Keypad5,
        keysym::XK_KP_6 => KeySymbol::Keypad6,
        keysym::XK_KP_7 => KeySymbol::Keypad7,
        keysym::XK_KP_8 => KeySymbol::Keypad8,
        keysym::XK_KP_9 => KeySymbol::Keypad9,
        keysym::XK_KP_Enter => KeySymbol::KeypadEnter,
        keysym::XK_KP_Add => KeySymbol::KeypadPlus,
        keysym::XK_KP_Subtract => KeySymbol::KeypadMinus,
        keysym::XK_KP_Multiply => KeySymbol::KeypadMultiply,
        keysym::XK_KP_Divide => KeySymbol::KeypadDivide,
        keysym::XK_KP_Decimal => KeySymbol::KeypadDecimal,
        keysym::XK_KP_Equal => KeySymbol::KeypadEquals,
        keysym::XK_Shift_L | keysym::XK_Shift_R => KeySymbol::Shift,
        keysym::XK_Control_L | keysym::XK_Control_R => KeySymbol::Control,
        keysym::XK_Alt_L | keysym::XK_Alt_R | keysym::XK_Meta_L | keysym::XK_Meta_R => {
            KeySymbol::Alt
        }
        keysym::XK_Super_L | keysym::XK_Super_R => KeySymbol::Super,
        keysym::XK_Caps_Lock => KeySymbol::CapsLock,
        keysym::XK_Num_Lock => KeySymbol::NumLock,
        keysym::XK_Print => KeySymbol::PrintScreen,
        keysym::XK_Scroll_Lock => KeySymbol::ScrollLock,
        keysym::XK_Pause => KeySymbol::Pause,
        keysym::XK_Menu => KeySymbol::Menu,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(keysym_val: u32, text: &str) -> KeySymbol {
        xkeysym_to_keysymbol(xlib::KeySym::from(keysym_val), text)
    }

    #[test]
    fn named_keys_are_keys_not_the_text_they_type() {
        assert_eq!(key(keysym::XK_Return, "\r"), KeySymbol::Enter);
        assert_eq!(key(keysym::XK_Tab, "\t"), KeySymbol::Tab);
        assert_eq!(key(keysym::XK_BackSpace, "\u{8}"), KeySymbol::Backspace);
        assert_eq!(key(keysym::XK_Escape, "\u{1b}"), KeySymbol::Escape);
        assert_eq!(key(keysym::XK_Delete, "\u{7f}"), KeySymbol::Delete);
        assert_eq!(key(keysym::XK_Page_Up, ""), KeySymbol::PageUp);
        assert_eq!(key(keysym::XK_F12, ""), KeySymbol::F12);
    }

    #[test]
    fn a_control_chord_names_the_key_held() {
        // Ctrl+C types U+0003; the key is still `c`.
        assert_eq!(key(keysym::XK_c, "\u{3}"), KeySymbol::Char('c'));
        assert_eq!(key(keysym::XK_minus, "\u{1f}"), KeySymbol::Char('-'));
    }

    #[test]
    fn the_keypad_is_navigation_with_num_lock_off_and_digits_with_it_on() {
        assert_eq!(key(keysym::XK_KP_Home, ""), KeySymbol::Home);
        assert_eq!(key(keysym::XK_KP_7, "7"), KeySymbol::Keypad7);
    }

    #[test]
    fn unicode_keysyms_name_their_character_and_legacy_ones_fall_back_to_text() {
        assert_eq!(key(UNICODE_KEYSYM_BASE + 0x20ac, "€"), KeySymbol::Char('€'));
        // XK_Cyrillic_a: a legacy keysym, neither Latin-1 nor Unicode-coded.
        assert_eq!(key(keysym::XK_Cyrillic_a, "а"), KeySymbol::Char('а'));
        assert_eq!(key(keysym::XK_Cyrillic_a, ""), KeySymbol::Unknown);
    }
}
