//! Cocoa Wrapper Layer (The Clean Room)
//!
//! This module defines safe, Rust-friendly wrappers around the raw Objective-C runtime.
//! No strings in driver logic. No unsafe blocks in driver logic.

use super::sys::{self, Id, BOOL, NO, YES};
use std::ffi::c_void;

// --- Helpers ---

/// Convert Rust string to NSString id.
/// Note: This returns an autoreleased object usually, or we manage it?
/// For simplicity, we create a new NSString that we must release if we own it,
/// or we just use it as a temporary argument.
/// Since `objc_msgSend` arguments are just pointers, we can return the Id.
pub fn make_nsstring(s: &str) -> Id {
    unsafe {
        let cls = sys::class(b"NSString\0");
        let alloc: Id = sys::send(cls, sys::sel(b"alloc\0"));
        // initWithUTF8String:
        let s_ptr = std::ffi::CString::new(s).unwrap();
        sys::send_1(alloc, sys::sel(b"initWithUTF8String:\0"), s_ptr.as_ptr())
    }
}

pub fn nsstring_to_string(ns_str: Id) -> String {
    unsafe {
        let utf8: *const i8 = sys::send(ns_str, sys::sel(b"UTF8String\0"));
        if utf8.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(utf8)
                .to_string_lossy()
                .into_owned()
        }
    }
}

// --- Structs ---

/// Wrapper for NSPasteboard
#[derive(Copy, Clone)]
pub struct NSPasteboard(pub Id);

impl NSPasteboard {
    pub fn general() -> Self {
        unsafe {
            let cls = sys::class(b"NSPasteboard\0");
            let ptr: Id = sys::send(cls, sys::sel(b"generalPasteboard\0"));
            Self(ptr)
        }
    }

    pub fn clear_contents(&self) {
        unsafe {
            sys::send::<()>(self.0, sys::sel(b"clearContents\0"));
        }
    }

    pub fn set_string(&self, s: &str) {
        unsafe {
            let ns_str = make_nsstring(s);
            // setString:forType: (NSPasteboardTypeString)
            // We need the string constant for NSPasteboardTypeString.
            // It's usually "public.utf8-plain-text" or similar in modern macOS.
            // Or just use the old "NSStringPboardType".
            // Let's use string literal.
            let type_str = make_nsstring("public.utf8-plain-text");

            sys::send_2::<(), Id, Id>(self.0, sys::sel(b"setString:forType:\0"), ns_str, type_str);

            // Release temp strings
            sys::send::<()>(ns_str, sys::sel(b"release\0"));
            sys::send::<()>(type_str, sys::sel(b"release\0"));
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NSPoint {
    pub x: f64,
    pub y: f64,
}

impl NSPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NSSize {
    pub width: f64,
    pub height: f64,
}

impl NSSize {
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NSRect {
    pub origin: NSPoint,
    pub size: NSSize,
}

impl NSRect {
    pub fn new(origin: NSPoint, size: NSSize) -> Self {
        Self { origin, size }
    }
}

// --- Wrappers ---

/// Wrapper for NSApplication
#[derive(Copy, Clone)]
pub struct NSApplication(pub Id);

impl NSApplication {
    pub fn shared() -> Self {
        unsafe {
            let cls = sys::class(b"NSApplication\0");
            let ptr: Id = sys::send(cls, sys::sel(b"sharedApplication\0"));
            Self(ptr)
        }
    }

    pub fn set_activation_policy(&self, policy: isize) {
        unsafe {
            sys::send_1::<(), isize>(self.0, sys::sel(b"setActivationPolicy:\0"), policy);
        }
    }

    pub fn activate_ignoring_other_apps(&self, ignore: bool) {
        unsafe {
            let val = if ignore { YES } else { NO };
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\0"), val);
        }
    }

    pub fn finish_launching(&self) {
        unsafe {
            sys::send::<()>(self.0, sys::sel(b"finishLaunching\0"));
        }
    }

    pub fn send_event(&self, event: NSEvent) {
        unsafe {
            sys::send_1::<(), Id>(self.0, sys::sel(b"sendEvent:\0"), event.0);
        }
    }

    // nextEventMatchingMask:untilDate:inMode:dequeue:
    pub fn next_event(&self, mask: u64, date: Id, mode: Id, dequeue: bool) -> NSEvent {
        unsafe {
            let d = if dequeue { YES } else { NO };
            let ptr: Id = sys::send_4(
                self.0,
                sys::sel(b"nextEventMatchingMask:untilDate:inMode:dequeue:\0"),
                mask,
                date,
                mode,
                d,
            );
            NSEvent(ptr)
        }
    }
}

/// Wrapper for NSWindow
#[derive(Copy, Clone)]
pub struct NSWindow(pub Id);

impl NSWindow {
    pub fn alloc() -> Self {
        unsafe {
            let cls = sys::class(b"NSWindow\0");
            let ptr: Id = sys::send(cls, sys::sel(b"alloc\0"));
            Self(ptr)
        }
    }

    pub fn init_with_content_rect(
        &self,
        rect: NSRect,
        style_mask: u64,
        backing: u64,
        defer: bool,
    ) -> Self {
        unsafe {
            let d = if defer { YES } else { NO };
            let ptr: Id = sys::send_4(
                self.0,
                sys::sel(b"initWithContentRect:styleMask:backing:defer:\0"),
                rect,
                style_mask,
                backing,
                d,
            );
            Self(ptr)
        }
    }

    pub fn set_title(&self, title: &str) {
        unsafe {
            let ns_str = make_nsstring(title);
            sys::send_1::<(), Id>(self.0, sys::sel(b"setTitle:\0"), ns_str);
            // Verify release needed for temporary string?
            // In init we alloced, so we own (+1). setTitle likely copies or retains.
            // So we should release ns_str.
            sys::send::<()>(ns_str, sys::sel(b"release\0"));
        }
    }

    pub fn set_content_view(&self, view: NSView) {
        unsafe {
            sys::send_1::<(), Id>(self.0, sys::sel(b"setContentView:\0"), view.0);
        }
    }

    pub fn make_key_and_order_front(&self) {
        unsafe {
            sys::send_1::<(), *mut c_void>(
                self.0,
                sys::sel(b"makeKeyAndOrderFront:\0"),
                std::ptr::null_mut(),
            );
        }
    }

    pub fn content_view(&self) -> NSView {
        unsafe {
            let ptr: Id = sys::send(self.0, sys::sel(b"contentView\0"));
            NSView(ptr)
        }
    }

    pub fn close(&self) {
        unsafe {
            sys::send::<()>(self.0, sys::sel(b"close\0"));
        }
    }
}

/// Wrapper for NSView
#[derive(Copy, Clone)]
pub struct NSView(pub Id);

impl NSView {
    pub fn alloc() -> Self {
        unsafe {
            let cls = sys::class(b"NSView\0");
            Self(sys::send(cls, sys::sel(b"alloc\0")))
        }
    }

    pub fn init_with_frame(&self, frame: NSRect) -> Self {
        unsafe {
            let ptr: Id = sys::send_1(self.0, sys::sel(b"initWithFrame:\0"), frame);
            Self(ptr)
        }
    }

    pub fn set_wants_layer(&self, wants: bool) {
        unsafe {
            let val = if wants { YES } else { NO };
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\0"), val);
        }
    }

    pub fn layer(&self) -> Id {
        unsafe { sys::send(self.0, sys::sel(b"layer\0")) }
    }
}

/// Wrapper for NSEvent
#[derive(Copy, Clone)]
pub struct NSEvent(pub Id); // Id can be null

impl NSEvent {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn type_(&self) -> u64 {
        unsafe { sys::send(self.0, sys::sel(b"type\0")) }
    }

    pub fn window(&self) -> NSWindow {
        unsafe { NSWindow(sys::send(self.0, sys::sel(b"window\0"))) }
    }

    pub fn location_in_window(&self) -> NSPoint {
        unsafe { sys::send(self.0, sys::sel(b"locationInWindow\0")) }
    }

    pub fn modifier_flags(&self) -> u64 {
        unsafe { sys::send(self.0, sys::sel(b"modifierFlags\0")) }
    }

    // KeyDown specifics
    pub fn characters(&self) -> String {
        unsafe {
            let ns_str: Id = sys::send(self.0, sys::sel(b"characters\0"));
            nsstring_to_string(ns_str)
        }
    }

    pub fn key_code(&self) -> u16 {
        unsafe { sys::send(self.0, sys::sel(b"keyCode\0")) }
    }

    // Scroll wheel specifics
    pub fn scrolling_delta_x(&self) -> f64 {
        unsafe { sys::send(self.0, sys::sel(b"scrollingDeltaX\0")) }
    }

    pub fn scrolling_delta_y(&self) -> f64 {
        unsafe { sys::send(self.0, sys::sel(b"scrollingDeltaY\0")) }
    }
}

// --- Constants ---
pub mod event_type {
    pub const LEFT_MOUSE_DOWN: u64 = 1;
    pub const LEFT_MOUSE_UP: u64 = 2;
    pub const RIGHT_MOUSE_DOWN: u64 = 3;
    pub const RIGHT_MOUSE_UP: u64 = 4;
    pub const MOUSE_MOVED: u64 = 5;
    pub const LEFT_MOUSE_DRAGGED: u64 = 6;
    pub const RIGHT_MOUSE_DRAGGED: u64 = 7;
    pub const KEY_DOWN: u64 = 10;
    pub const SCROLL_WHEEL: u64 = 22;
    pub const APP_KIT_DEFINED: u64 = 13;
    pub const SYSTEM_DEFINED: u64 = 14;
    pub const APPLICATION_DEFINED: u64 = 15;

    /// Returns true if this event type is spatial (has window chrome component).
    /// Spatial events need to go through sendEvent: for window management.
    /// Pure input events (keyboard, scroll) are fully handled by us.
    pub fn is_spatial(ty: u64) -> bool {
        matches!(
            ty,
            LEFT_MOUSE_DOWN
                | LEFT_MOUSE_UP
                | RIGHT_MOUSE_DOWN
                | RIGHT_MOUSE_UP
                | MOUSE_MOVED
                | LEFT_MOUSE_DRAGGED
                | RIGHT_MOUSE_DRAGGED
        )
    }
}
