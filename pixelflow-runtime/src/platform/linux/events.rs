//! Event mapping for X11 -> DisplayEvent.

use crate::display::messages::{DisplayEvent, WindowId};
use crate::input::{KeySymbol, Modifiers};
use crate::pixel::PlatformPixel;
use pixelflow_graphics::render::Frame;
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
                    let frame = Frame::<PlatformPixel>::new(window.width, window.height);
                    let win = crate::display::messages::Window {
                        id: window_id,
                        frame,
                        width_px: window.width,
                        height_px: window.height,
                        scale: window.scale_factor,
                    };
                    Some(DisplayEvent::Resized { window: win })
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

    match req.target {
        target if target == window.atoms.targets => {
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
        target if target == window.atoms.utf8_string
            || target == window.atoms.text
            || target == window.atoms.xa_string =>
        {
            let data = window.clipboard_data.as_bytes();
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

fn xkeysym_to_keysymbol(keysym_val: xlib::KeySym, text: &str) -> KeySymbol {
    if !text.is_empty() {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() == 1 && chars[0] != '\u{FFFD}' {
            return KeySymbol::Char(chars[0]);
        }
    }
    match keysym_val as u32 {
        keysym::XK_Return => KeySymbol::Enter,
        keysym::XK_BackSpace => KeySymbol::Backspace,
        keysym::XK_Tab => KeySymbol::Tab,
        keysym::XK_Escape => KeySymbol::Escape,
        keysym::XK_Left => KeySymbol::Left,
        keysym::XK_Right => KeySymbol::Right,
        keysym::XK_Up => KeySymbol::Up,
        keysym::XK_Down => KeySymbol::Down,
        // Add more keys as needed
        _ => KeySymbol::Unknown,
    }
}
