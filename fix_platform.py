with open('pixelflow-runtime/src/platform/macos/platform.rs', 'r', encoding='utf-8') as f:
    text = f.read()

search1 = """            DisplayControl::SetVisible { id, visible } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_visible(visible);
                }
            }"""

replace1 = """            DisplayControl::SetVisible { id, visible } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    if visible {
                        win.show();
                    } else {
                        win.hide();
                    }
                }
            }"""

search2 = """            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.set_visible(false);
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it
                    self.window_map.remove(&(win.window.0 as usize));
                }
            }"""

replace2 = """            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.hide();
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it
                    self.window_map.remove(&(win.window.0 as usize));
                }
            }"""

new_text = text.replace(search1, replace1).replace(search2, replace2)
if new_text != text:
    print("Replaced!")
    with open('pixelflow-runtime/src/platform/macos/platform.rs', 'w', encoding='utf-8') as f:
        f.write(new_text)
else:
    print("Not replaced")
