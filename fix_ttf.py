import re

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'r') as f:
    content = f.read()

# Fix t_minus assignments
content = content.replace("let t_minus = sqrt_disc * -inv_2a + neg_b_2a;", "let t_minus = sqrt_disc * -inv_2a.clone() + neg_b_2a.clone();")

# Fix x_minus assignment
content = content.replace("let x_minus = t_minus.clone() * t_minus.clone() * ax + t_minus.clone() * bx + cx;", "let x_minus = t_minus.clone() * t_minus.clone() * ax.clone() + t_minus.clone() * bx.clone() + cx.clone();")

# Fix dy_minus assignment
content = content.replace("let dy_minus = t_minus.clone() * (2.0 * ay) + by;", "let dy_minus = t_minus.clone() * (2.0 * ay.clone()) + by.clone();")

with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'w') as f:
    f.write(content)
