import re

file_path = "pixelflow-runtime/src/platform/macos/cocoa.rs"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Make sure we don't duplicate enums!
if "EventDequeue" not in content:
    enums = """
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EventDequeue {
    Dequeue,
    Peek,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowDeferral {
    Defer,
    Immediate,
}
"""
    content = content.replace("use std::ffi::c_void;\n", "use std::ffi::c_void;\n" + enums)

old_activate = """    pub fn activate_ignoring_other_apps(&self, ignore: bool) {
        unsafe {
            let val = if ignore { YES } else { NO };
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), val);
        }
    }"""
new_activate = """    pub fn activate_ignoring_other_apps(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), YES);
        }
    }

    pub fn activate_observing_other_apps(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), NO);
        }
    }"""
content = content.replace(old_activate, new_activate)

old_next = """    pub fn next_event(&self, mask: u64, date: Id, mode: Id, dequeue: bool) -> NSEvent {
        unsafe {
            let d = if dequeue { YES } else { NO };"""
new_next = """    pub fn next_event(&self, mask: u64, date: Id, mode: Id, dequeue: EventDequeue) -> NSEvent {
        unsafe {
            let d = if dequeue == EventDequeue::Dequeue { YES } else { NO };"""
content = content.replace(old_next, new_next)

old_init = """    pub fn init_with_content_rect(
        &self,
        rect: NSRect,
        style_mask: u64,
        backing: u64,
        defer: bool,
    ) -> Self {
        unsafe {
            let d = if defer { YES } else { NO };"""
new_init = """    pub fn init_with_content_rect(
        &self,
        rect: NSRect,
        style_mask: u64,
        backing: u64,
        defer: WindowDeferral,
    ) -> Self {
        unsafe {
            let d = if defer == WindowDeferral::Defer { YES } else { NO };"""
content = content.replace(old_init, new_init)

old_layer = """    pub fn set_wants_layer(&self, wants: bool) {
        unsafe {
            let val = if wants { YES } else { NO };
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), val);
        }
    }"""
new_layer = """    pub fn enable_layer(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), YES);
        }
    }

    pub fn disable_layer(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), NO);
        }
    }"""
content = content.replace(old_layer, new_layer)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
