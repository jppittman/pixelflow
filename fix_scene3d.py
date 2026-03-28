import re

with open('pixelflow-graphics/src/scene3d.rs', 'r') as f:
    content = f.read()

# Fix valid_t and valid_deriv usages
content = content.replace("valid_t & valid_deriv", "valid_t.clone() & valid_deriv.clone()")

# Fix hx, hy, hz usages in ColorSurface and ColorSurfaceJet
content = content.replace("let mat_val = material.at(hx, hy, hz, W);", "let mat_val = material.at(hx.clone(), hy.clone(), hz.clone(), W);")

with open('pixelflow-graphics/src/scene3d.rs', 'w') as f:
    f.write(content)
