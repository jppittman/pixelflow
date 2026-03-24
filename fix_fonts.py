import re

file_path = 'pixelflow-graphics/src/fonts/ttf_curve_analytical.rs'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace(
    'let in_t = V.clone().ge(0.0) & V.clone().le(1.0);',
    'let in_t = t.clone().ge(0.0) & t.clone().le(1.0);'
)

content = content.replace(
    'let x_int = V.clone() * t.clone() * ax + t.clone() * bx + cx;',
    'let x_int = t.clone() * t.clone() * ax.clone() + t.clone() * bx.clone() + cx.clone();'
)

content = content.replace(
    'let x_int = t.clone() * V.clone() * ax + t.clone() * bx + cx;',
    'let x_int = t.clone() * t.clone() * ax.clone() + t.clone() * bx.clone() + cx.clone();'
)

content = content.replace(
    'let x_int = t.clone() * t.clone() * ax + V.clone() * bx + cx;',
    'let x_int = t.clone() * t.clone() * ax.clone() + t.clone() * bx.clone() + cx.clone();'
)

content = content.replace(
    'let x_plus = t_plus.clone() * t_plus.clone() * ax.clone() + t_plus.clone() * bx.clone() + cx.clone();',
    'let x_plus = t_plus.clone() * t_plus.clone() * ax.clone() + t_plus.clone() * bx.clone() + cx.clone();'
)

content = content.replace(
    'let x_minus = t_minus.clone() * t_minus.clone() * ax + t_minus.clone() * bx + cx;',
    'let x_minus = t_minus.clone() * t_minus.clone() * ax.clone() + t_minus.clone() * bx.clone() + cx.clone();'
)

content = content.replace(
    'let dy_minus = t_minus.clone() * (2.0 * ay) + by;',
    'let dy_minus = t_minus.clone() * (2.0 * ay.clone()) + by.clone();'
)

with open(file_path, 'w') as f:
    f.write(content)
