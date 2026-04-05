import re

file_path = "pixelflow-runtime/src/platform/macos/cocoa.rs"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add enums at the top
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

# 2. Split activate_ignoring_other_apps
activate_old = r'pub fn activate_ignoring_other_apps\(&self, ignore: bool\) \{[\s\S]*?\}'

activate_new = """pub fn activate_ignoring_other_apps(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), YES);
        }
    }

    pub fn activate_observing_other_apps(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), NO);
        }
    }"""
content = re.sub(activate_old, activate_new, content, count=1)

# 3. Modify next_event
next_old = r'pub fn next_event\(&self, mask: u64, date: Id, mode: Id, dequeue: bool\) -> NSEvent \{[\s\S]*?let d = if dequeue \{ YES \} else \{ NO \};'
next_new = """pub fn next_event(&self, mask: u64, date: Id, mode: Id, dequeue: EventDequeue) -> NSEvent {
        unsafe {
            let d = if dequeue == EventDequeue::Dequeue { YES } else { NO };"""
content = re.sub(next_old, next_new, content, count=1)

# 4. Modify init_with_content_rect
init_old = r'pub fn init_with_content_rect\([\s\S]*?defer: bool,[\s\S]*?\) -> Self \{[\s\S]*?let d = if defer \{ YES \} else \{ NO \};'
init_new = """pub fn init_with_content_rect(
        &self,
        rect: NSRect,
        style_mask: u64,
        backing: u64,
        defer: WindowDeferral,
    ) -> Self {
        unsafe {
            let d = if defer == WindowDeferral::Defer { YES } else { NO };"""
content = re.sub(init_old, init_new, content, count=1)

# 5. Split set_wants_layer
wants_layer_old = r'pub fn set_wants_layer\(&self, wants: bool\) \{[\s\S]*?\}'
wants_layer_new = """pub fn enable_layer(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), YES);
        }
    }

    pub fn disable_layer(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), NO);
        }
    }"""
content = re.sub(wants_layer_old, wants_layer_new, content, count=1)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
