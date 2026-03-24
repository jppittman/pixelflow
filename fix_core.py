import re

file_path = 'pixelflow-core/src/jet/jet3.rs'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace('use crate::numeric::Numeric as _;\n', '')
content = content.replace('use crate::numeric::Numeric;', '')
content = content.replace('use core::ops::{BitAnd, BitOr, Not};', 'use core::ops::{BitAnd, BitOr, Not};\nuse crate::numeric::Numeric;')

with open(file_path, 'w') as f:
    f.write(content)
