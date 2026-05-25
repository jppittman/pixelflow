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

search = """    let t_max = 1000000.0;
    let deriv_max = 10000.0;
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);
    let mask = valid_t & valid_deriv;

    // 3. Hit point: P = ray * t (always computed; Select short-circuits if mask is all-false)
    let hx = X * t;
    let hy = Y * t;
    let hz = Z * t;

    // 4. Sample material at hit point, background at ray direction
    let mat_val = material.at(hx, hy, hz, W);
    let bg_val = background;

    // 5. Select based on hit validity (short-circuit avoids evaluating unused branch)
    mask.select(mat_val, bg_val)"""

replace = """    let t_max = 1000000.0;
    let deriv_max = 10000.0;
    let valid_t_v = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv_v = deriv_mag_sq < (deriv_max * deriv_max);
    let mask_v = valid_t_v & valid_deriv_v;

    // 3. Hit point: P = ray * t (always computed; Select short-circuits if mask is all-false)
    let hx_v = X * t;
    let hy_v = Y * t;
    let hz_v = Z * t;

    // 4. Sample material at hit point, background at ray direction
    let mat_val = material.at(hx_v, hy_v, hz_v, W);
    let bg_val = background;

    // 5. Select based on hit validity (short-circuit avoids evaluating unused branch)
    mask_v.select(mat_val, bg_val)"""

replace_in_file('pixelflow-graphics/src/scene3d.rs', search, replace)

search_2 = """    // Valid if: t > 0, t < max, derivatives reasonable
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);

    valid_t & valid_deriv"""

replace_2 = """    // Valid if: t > 0, t < max, derivatives reasonable
    let valid_t_v = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv_v = deriv_mag_sq < (deriv_max * deriv_max);

    valid_t_v & valid_deriv_v"""

replace_in_file('pixelflow-graphics/src/scene3d.rs', search_2, replace_2)
