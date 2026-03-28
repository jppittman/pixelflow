import re

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "r") as f:
    content = f.read()

old_func = """    pub fn activate_ignoring_other_apps(&self, ignore: bool) {
        unsafe {
            let val = if ignore { YES } else { NO };
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), val);
        }
    }"""

new_func = """    pub fn activate_ignoring_other_apps(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), YES);
        }
    }

    pub fn activate_respecting_other_apps(&self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.0, sys::sel(b"activateIgnoringOtherApps:\\0"), NO);
        }
    }"""

content = content.replace(old_func, new_func)

with open("pixelflow-runtime/src/platform/macos/cocoa.rs", "w") as f:
    f.write(content)
