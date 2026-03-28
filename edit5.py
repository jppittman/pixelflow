import re

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "r") as f:
    content = f.read()

enum_def = """
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EventDequeue {
    Dequeue,
    Peek,
}

/// Wrapper for NSEvent
"""
content = content.replace("/// Wrapper for NSEvent\n", enum_def)

old_next = """    // nextEventMatchingMask:untilDate:inMode:dequeue:
    pub fn next_event(&self, mask: u64, date: Id, mode: Id, dequeue: bool) -> NSEvent {
        unsafe {
            let d = if dequeue { YES } else { NO };"""

new_next = """    // nextEventMatchingMask:untilDate:inMode:dequeue:
    pub fn next_event(&self, mask: u64, date: Id, mode: Id, dequeue: EventDequeue) -> NSEvent {
        unsafe {
            let d = if dequeue == EventDequeue::Dequeue { YES } else { NO };"""

content = content.replace(old_next, new_next)

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "w") as f:
    f.write(content)
