import re

with open("pixelflow-runtime/src/platform/macos/platform.rs", "r") as f:
    content = f.read()

# Add import for EventDequeue
content = content.replace("use crate::platform::macos::cocoa::{NSApplication, NSEvent, NSRect};", "use crate::platform::macos::cocoa::{NSApplication, NSEvent, NSRect, EventDequeue};")
content = content.replace("use crate::platform::macos::cocoa::{", "use crate::platform::macos::cocoa::{EventDequeue, ")

old_call = """            let event = self.app.next_event(
                u64::MAX,
                until_date,
                mode,
                true, // dequeue
            );"""

new_call = """            let event = self.app.next_event(
                u64::MAX,
                until_date,
                mode,
                EventDequeue::Dequeue,
            );"""

content = content.replace(old_call, new_call)

with open("pixelflow-runtime/src/platform/macos/platform.rs", "w") as f:
    f.write(content)
