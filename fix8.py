import sys

with open("pixelflow-pipeline/src/bin/bench_scanline_jit.rs", "r") as f:
    text = f.read()

text = text.replace(
    "fn nanos_now() -> u64 {",
    "#[allow(dead_code)]\nfn nanos_now() -> u64 {"
)

with open("pixelflow-pipeline/src/bin/bench_scanline_jit.rs", "w") as f:
    f.write(text)

with open("pixelflow-pipeline/src/bin/train_unified.rs", "r") as f:
    text = f.read()

text = text.replace(
    "expr_embed: Vec<f32>,",
    "#[allow(dead_code)]\n    expr_embed: Vec<f32>,"
)

with open("pixelflow-pipeline/src/bin/train_unified.rs", "w") as f:
    f.write(text)

with open("pixelflow-pipeline/src/training/factored.rs", "r") as f:
    text = f.read()

text = text.replace(
    "fn logged_expr_jit_output(src: &str) -> f32 {",
    "#[allow(dead_code)]\n    fn logged_expr_jit_output(src: &str) -> f32 {"
)
text = text.replace(
    "fn assert_scalar_and_jit_close(src: &str, epsilon: f32) {",
    "#[allow(dead_code)]\n    fn assert_scalar_and_jit_close(src: &str, epsilon: f32) {"
)

with open("pixelflow-pipeline/src/training/factored.rs", "w") as f:
    f.write(text)
