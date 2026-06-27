import os
import re

filepath = "core-term/src/surface/manifold.rs"
with open(filepath, 'r') as f:
    lines = f.read().split('\n')

# Find where test_color_manifold was renamed
# Wait, it looks like `test_color_manifold` was a test that got its `test_` stripped and became `color_manifold()`.
# Then something else in the code called `color_manifold(...)` which used to call a function named `color_manifold`!
# Let's see what `test_color_manifold` used to be, and if there is a real `color_manifold` function.
