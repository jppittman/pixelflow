import re

with open("pixelflow-runtime/src/platform/macos/platform.rs", "r") as f:
    content = f.read()

content = content.replace("app.activate_ignoring_other_apps(true);", "app.activate_ignoring_other_apps();")

with open("pixelflow-runtime/src/platform/macos/platform.rs", "w") as f:
    f.write(content)
