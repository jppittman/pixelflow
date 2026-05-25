import sys

def replace_in_file(filepath, search, replace):
    with open(filepath, 'r') as f:
        content = f.read()

    if search not in content:
        print(f"Error: Could not find search string in {filepath}")
        print("Expected:", repr(search))
        sys.exit(1)

    new_content = content.replace(search, replace)

    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Successfully updated {filepath}")

# GeometryMask return type Field
search_1 = """kernel!(pub struct GeometryMask = |geometry: kernel| Jet3 -> Field {"""
replace_1 = """kernel!(pub struct GeometryMask = |geometry: kernel| Jet3 -> Jet3 {"""
replace_in_file('pixelflow-graphics/src/scene3d.rs', search_1, replace_1)

search_2 = """    type Mask: ManifoldCompat<Jet3, Output = Field>;"""
replace_2 = """    type Mask: ManifoldCompat<Jet3, Output = Jet3>;"""
replace_in_file('pixelflow-graphics/src/scene3d.rs', search_2, replace_2)

# eval argument count
search_3 = """        self.inner.eval(r_x, r_y, r_z, w)"""
replace_3 = """        self.inner.eval((r_x, r_y, r_z, w))"""
replace_in_file('pixelflow-graphics/src/scene3d.rs', search_3, replace_3)

# Fix undefined variables in scene3d.rs kernel macros
# V, DX, DY, DZ are macros or methods that shadow/break the AST transformer when defining standard rust vars, so we need to inline their outputs or use valid ast nodes.
search_4 = """    let t_max = 1000000.0;
    let deriv_max = 10000.0;
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);
    let mask = valid_t & valid_deriv;

    // 3. Hit point: P = ray * t (always computed; Select short-circuits if mask is all-false)
    let hx = X * t;
    let hy = Y * t;
    let hz = Z * t;

    // 4. Evaluate Material at Hit Point
    // Material takes spatial coordinates (X,Y,Z,W), but we want to evaluate it at P=(hx,hy,hz).
    // The `at` method applies a coordinate transform BEFORE evaluating the manifold.
    let mat_val = material.at(hx, hy, hz, W);"""

replace_4 = """    let t_max = 1000000.0;
    let deriv_max = 10000.0;
    let valid_t_v = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv_v = deriv_mag_sq < (deriv_max * deriv_max);
    let mask_v = valid_t_v & valid_deriv_v;

    // 3. Hit point: P = ray * t (always computed; Select short-circuits if mask is all-false)
    let hx_v = X * t;
    let hy_v = Y * t;
    let hz_v = Z * t;

    // 4. Evaluate Material at Hit Point
    // Material takes spatial coordinates (X,Y,Z,W), but we want to evaluate it at P=(hx,hy,hz).
    // The `at` method applies a coordinate transform BEFORE evaluating the manifold.
    let mat_val = material.at(hx_v, hy_v, hz_v, W);"""

replace_in_file('pixelflow-graphics/src/scene3d.rs', search_4, replace_4)

search_5 = """    // 5. Select Material or Background
    mask.select(mat_val, bg_val)
});"""
replace_5 = """    // 5. Select Material or Background
    mask_v.select(mat_val, bg_val)
});"""
replace_in_file('pixelflow-graphics/src/scene3d.rs', search_5, replace_5)

# Wait, search_4 and 5 applies twice in scene3d.rs (Lines ~360 and ~390)
