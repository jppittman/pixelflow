import re

with open("pixelflow-compiler/src/optimize.rs", "r", encoding="utf-8") as f:
    opt = f.read()

# Revert the wrong optimize_via_egraph fix and just let `optimize_with_egraph` be renamed back? Wait, the compiler couldn't find `optimize_with_egraph` because it was never defined! Let's check `git log pixelflow-compiler/src/optimize.rs` to see what function was there before.
