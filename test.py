import re

file_path = "pixelflow-runtime/src/platform/macos/cocoa.rs"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

print("Contains activate?", "activate_ignoring_other_apps" in content)
