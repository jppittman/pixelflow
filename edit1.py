import re

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "r") as f:
    content = f.read()

# Add the enum before NSWindow struct
enum_def = """
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowDeferral {
    Defer,
    Immediate,
}

/// Wrapper for NSWindow
"""
content = content.replace("/// Wrapper for NSWindow\n", enum_def)

# Modify init_with_content_rect
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

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "w") as f:
    f.write(content)
