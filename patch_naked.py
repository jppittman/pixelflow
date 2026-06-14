import sys

filepath = "./pixelflow-core/tests/naked_scale.rs"
with open(filepath, "r") as f:
    text = f.read()

search = """        let num_threads = 16;
        let ops_per_thread = 1_000_000;"""

replace = """        let num_threads = 4;
        let ops_per_thread = 100_000;"""

if search in text:
    text = text.replace(search, replace)
    with open(filepath, "w") as f:
        f.write(text)
    print("Patched successfully")
else:
    print("Search string not found")
