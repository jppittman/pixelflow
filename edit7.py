import re

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "r") as f:
    content = f.read()

old_func = """    pub fn set_wants_layer(&self, wants: bool) {
        unsafe {
            let val = if wants { YES } else { NO };
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), val);
        }
    }"""

new_func = """    pub fn enable_layer(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), YES);
        }
    }

    pub fn disable_layer(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"setWantsLayer:\\0"), NO);
        }
    }"""

content = content.replace(old_func, new_func)

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "w") as f:
    f.write(content)
