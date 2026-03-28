import re

with open("pixelflow-runtime/src/platform/macos/platform.rs", "r") as f:
    content = f.read()

old1 = """            DisplayControl::SetVisible { id, visible } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_visible(visible);
                }
            }"""

new1 = """            DisplayControl::SetVisible { id, visible } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    if visible {
                        win.show();
                    } else {
                        win.hide();
                    }
                }
            }"""

old2 = """            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.set_visible(false);
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it"""

new2 = """            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.hide();
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it"""

content = content.replace(old1, new1)
content = content.replace(old2, new2)

with open("pixelflow-runtime/src/platform/macos/platform.rs", "w") as f:
    f.write(content)
