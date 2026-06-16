with open("pixelflow-search/tests/prod_kernel_jit.rs", "r") as f:
    content = f.read()

# Let's print out the values to see what we're getting.
content = content.replace(
    "        assert!(",
    "        eprintln!(\"original at ({}, {}): got {}, want {}\", x, y, got_orig, want);\n        eprintln!(\"optimized at ({}, {}): got {}, want {}\", x, y, got_opt, want);\n        assert!("
)

with open("pixelflow-search/tests/prod_kernel_jit.rs", "w") as f:
    f.write(content)
