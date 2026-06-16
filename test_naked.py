with open("pixelflow-core/tests/naked_scale.rs", "r") as f:
    content = f.read()

# We want to trace what's happening when invoke_naked_kernel fails or hangs.
# Wait, the test timed out (> 60s). It didn't fail.
# It timed out in a multithreaded test where each thread invokes a naked kernel.
# Let's inspect JIT execution.
# What does get_jit_mul_kernel do?
# It calls ExecutableCode::from_code(bytes).
