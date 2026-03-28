import re

with open("pixelflow-runtime/src/platform/macos/window.rs", "r") as f:
    content = f.read()

content = content.replace("view.set_wants_layer(true);", "view.enable_layer();")

with open("pixelflow-runtime/src/platform/macos/window.rs", "w") as f:
    f.write(content)
