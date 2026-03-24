import re

file_path = 'pixelflow-graphics/src/scene3d.rs'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace(
    'let valid_t = (V(t) > 0.0) & (V(t) < t_max);',
    'let valid_t = (V(t.clone()) > 0.0) & (V(t.clone()) < t_max);'
)

content = content.replace(
    'let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);',
    'let deriv_mag_sq = DX(t.clone()) * DX(t.clone()) + DY(t.clone()) * DY(t.clone()) + DZ(t.clone()) * DZ(t.clone());'
)

content = content.replace(
    'let mask = valid_t & valid_deriv;',
    'let mask = valid_t.clone() & valid_deriv.clone();'
)

content = content.replace(
    'let hx = X * t;',
    'let hx = X * t.clone();'
)

content = content.replace(
    'let hy = Y * t;',
    'let hy = Y * t.clone();'
)

content = content.replace(
    'let hz = Z * t;',
    'let hz = Z * t.clone();'
)

content = content.replace(
    'let mat_val = material.at(hx, hy, hz, W);',
    'let mat_val = material.at(hx.clone(), hy.clone(), hz.clone(), W);'
)

content = content.replace(
    'valid_t & valid_deriv',
    'valid_t.clone() & valid_deriv.clone()'
)

content = content.replace(
    'self.inner.eval(r_x, r_y, r_z, w)',
    'self.inner.eval((r_x, r_y, r_z, w))'
)

content = content.replace(
    'let t_max = 1000000.0;\n    let deriv_max = 10000.0;\n    let valid_t = (V(t.clone()) > 0.0) & (V(t.clone()) < t_max);\n    let deriv_mag_sq = DX(t.clone()) * DX(t.clone()) + DY(t.clone()) * DY(t.clone()) + DZ(t.clone()) * DZ(t.clone());\n    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);\n    valid_t.clone() & valid_deriv.clone()',
    'let t_max = 1000000.0;\n    let deriv_max = 10000.0;\n    let valid_t = V(t.clone()).gt(0.0) & V(t.clone()).lt(t_max);\n    let deriv_mag_sq = DX(t.clone()) * DX(t.clone()) + DY(t.clone()) * DY(t.clone()) + DZ(t.clone()) * DZ(t.clone());\n    let valid_deriv = deriv_mag_sq.lt(deriv_max * deriv_max);\n    valid_t & valid_deriv'
)

content = content.replace(
    'let t_max = 1000000.0;\n    let deriv_max = 10000.0;\n    let valid_t = (V(t.clone()) > 0.0) & (V(t.clone()) < t_max);\n    let deriv_mag_sq = DX(t.clone()) * DX(t.clone()) + DY(t.clone()) * DY(t.clone()) + DZ(t.clone()) * DZ(t.clone());\n    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);\n    let mask = valid_t.clone() & valid_deriv.clone();',
    'let t_max = 1000000.0;\n    let deriv_max = 10000.0;\n    let valid_t = V(t.clone()).gt(0.0) & V(t.clone()).lt(t_max);\n    let deriv_mag_sq = DX(t.clone()) * DX(t.clone()) + DY(t.clone()) * DY(t.clone()) + DZ(t.clone()) * DZ(t.clone());\n    let valid_deriv = deriv_mag_sq.lt(deriv_max * deriv_max);\n    let mask = valid_t & valid_deriv;'
)

content = content.replace(
    'let valid_t = (V(t.clone()) > 0.0) & (V(t.clone()) < t_max);',
    'let valid_t = V(t.clone()).gt(0.0) & V(t.clone()).lt(t_max);'
)

content = content.replace(
    'let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);',
    'let valid_deriv = deriv_mag_sq.lt(deriv_max * deriv_max);'
)

with open(file_path, 'w') as f:
    f.write(content)
