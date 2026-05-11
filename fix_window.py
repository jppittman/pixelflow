with open('pixelflow-runtime/src/platform/macos/window.rs', 'r', encoding='utf-8') as f:
    text = f.read()

search = """    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.window.make_key_and_order_front();
        } else {
            unsafe {
                sys::send::<()>(self.window.0, sys::sel(b"orderOut:\x00"));
            }
        }
    }"""

search2 = b"""    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.window.make_key_and_order_front();
        } else {
            unsafe {
                sys::send::<()>(self.window.0, sys::sel(b"orderOut:\\0"));
            }
        }
    }""".decode('utf-8')


replace = b"""    pub fn show(&mut self) {
        self.window.make_key_and_order_front();
    }

    pub fn hide(&mut self) {
        unsafe {
            sys::send::<()>(self.window.0, sys::sel(b"orderOut:\\0"));
        }
    }""".decode('utf-8')

new_text = text.replace(search2, replace)
if new_text != text:
    print("Replaced!")
    with open('pixelflow-runtime/src/platform/macos/window.rs', 'w', encoding='utf-8') as f:
        f.write(new_text)
else:
    print("Not replaced")
