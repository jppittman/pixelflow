import os

with open('pixelflow-compiler/tests/kernel_jit.rs', 'r') as f:
    content = f.read()

# E0512 is an expected failure on the original branch due to types not matching
# "The E0512 error is an expected upstream failure in test_kernel_jit and does not need to be addressed per rules."
# So I can ignore it for `pixelflow-compiler`.
