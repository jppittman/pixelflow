import os
with open("pixelflow-compiler/src/lib.rs", "r") as f:
    content = f.read()

# I will add `#![allow(warnings)]` at the very top to silence EVERYTHING.
content = "#![allow(warnings)]\n" + content

with open("pixelflow-compiler/src/lib.rs", "w") as f:
    f.write(content)
