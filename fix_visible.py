with open("pixelflow-runtime/src/platform/macos/window.rs", "r") as f:
    text = f.read()

print("Found set_visible in window.rs:", text.find("pub fn set_visible(&mut self, visible: bool)"))
