import os
import re

filepath = "core-term/src/surface/manifold.rs"
with open(filepath, 'r') as f:
    content = f.read()

# We need to change `fn test_color_manifold` to `fn verify_color_manifold`
content = content.replace("fn test_color_manifold()", "fn verify_color_manifold()")

with open(filepath, 'w') as f:
    f.write(content)
