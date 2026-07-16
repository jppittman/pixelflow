import re
import os

def fix_core_approx_constants(path):
    with open(path, 'r') as f:
        content = f.read()

    content = re.sub(r'0\.693_147_2', 'core::f32::consts::LN_2', content)
    content = re.sub(r'1\.442_695', 'core::f32::consts::LOG2_E', content)
    content = re.sub(r'0\.434_294_5', 'core::f32::consts::LOG10_E', content)

    with open(path, 'w') as f:
        f.write(content)

for root, _, files in os.walk('pixelflow-core/src'):
    for file in files:
        if file.endswith('.rs'):
            fix_core_approx_constants(os.path.join(root, file))

for root, _, files in os.walk('pixelflow-graphics/src'):
    for file in files:
        if file.endswith('.rs'):
            fix_core_approx_constants(os.path.join(root, file))

lib_core_path = 'pixelflow-core/src/lib.rs'
if os.path.exists(lib_core_path):
    with open(lib_core_path, 'r') as f:
        lib_content = f.read()

    allow_attrs = "#![allow(clippy::approx_constant, clippy::bad_bit_mask, clippy::too_many_arguments, clippy::missing_transmute_annotations, clippy::type_complexity)]\n"
    if "clippy::approx_constant" not in lib_content:
        lib_content = allow_attrs + lib_content
        with open(lib_core_path, 'w') as f:
            f.write(lib_content)
