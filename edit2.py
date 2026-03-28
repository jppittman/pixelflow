import re

with open("pixelflow-runtime/src/platform/macos/window.rs", "r") as f:
    content = f.read()

# Add import for WindowDeferral
content = content.replace("use crate::platform::macos::cocoa::{NSPoint, NSRect, NSSize, NSView, NSWindow};", "use crate::platform::macos::cocoa::{NSPoint, NSRect, NSSize, NSView, NSWindow, WindowDeferral};")

# Change false to WindowDeferral::Immediate
old_call = "let window = NSWindow::alloc().init_with_content_rect(rect, style_mask, backing, false);"
new_call = "let window = NSWindow::alloc().init_with_content_rect(rect, style_mask, backing, WindowDeferral::Immediate);"
content = content.replace(old_call, new_call)

with open("pixelflow-runtime/src/platform/macos/window.rs", "w") as f:
    f.write(content)
