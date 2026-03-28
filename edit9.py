import re

with open("pixelflow-runtime/src/platform/macos/window.rs", "r") as f:
    content = f.read()

old_func = """    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.window.make_key_and_order_front();
        } else {
            unsafe {
                sys::send::<()>(self.window.0, sys::sel(b"orderOut:\\0"));
            }
        }
    }"""

new_func = """    pub fn show(&mut self) {
        self.window.make_key_and_order_front();
    }

    pub fn hide(&mut self) {
        unsafe {
            sys::send::<()>(self.window.0, sys::sel(b"orderOut:\\0"));
        }
    }"""

content = content.replace(old_func, new_func)

with open("pixelflow-runtime/src/platform/macos/window.rs", "w") as f:
    f.write(content)
