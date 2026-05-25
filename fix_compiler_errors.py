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

# 1. ttf_curve_analytical.rs `t`, `t_plus`, `t_minus`
search_1 = """            let k = kernel!(|ax: f32, bx: f32, cx: f32, by: f32, cy: f32| {
                let t = (Y - cy) / by;
                let in_t = t.clone().ge(0.0) & t.clone().le(1.0);

                // x-coordinate at intersection
                let x_int = t.clone() * t.clone() * ax + t.clone() * bx + cx;"""

replace_1 = """            let k = kernel!(|ax: f32, bx: f32, cx: f32, by: f32, cy: f32| {
                let t_val = (Y - cy) / by;
                let in_t = t_val.clone().ge(0.0) & t_val.clone().le(1.0);

                // x-coordinate at intersection
                let x_int = t_val.clone() * t_val.clone() * ax + t_val.clone() * bx + cx;"""

replace_in_file('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', search_1, replace_1)

search_2 = """            // Two roots: t = (-by +/- sqrt(disc)) / (2*ay)
            let t_plus = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();
            let t_minus = sqrt_disc * -inv_2a + neg_b_2a;

            // X-coordinates at intersection points
            let x_plus = t_plus.clone() * t_plus.clone() * ax.clone() + t_plus.clone() * bx.clone() + cx.clone();
            let x_minus = t_minus.clone() * t_minus.clone() * ax + t_minus.clone() * bx + cx;

            // Tangent dy/dt at each root for winding direction
            let dy_plus = t_plus.clone() * (2.0 * ay.clone()) + by.clone();
            let dy_minus = t_minus.clone() * (2.0 * ay) + by;

            // Step: 1.0 if crossing is to the left of or at X
            let crossed_plus = (X >= x_plus).select(1.0, 0.0);
            let crossed_minus = (X >= x_minus).select(1.0, 0.0);

            // Validity: only count roots with t in [0, 1]
            let valid_plus = t_plus.clone().ge(0.0) & t_plus.clone().le(1.0);
            let valid_minus = t_minus.clone().ge(0.0) & t_minus.clone().le(1.0);"""

replace_2 = """            // Two roots: t = (-by +/- sqrt(disc)) / (2*ay)
            let t_p = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();
            let t_m = sqrt_disc * -inv_2a + neg_b_2a;

            // X-coordinates at intersection points
            let x_plus = t_p.clone() * t_p.clone() * ax.clone() + t_p.clone() * bx.clone() + cx.clone();
            let x_minus = t_m.clone() * t_m.clone() * ax + t_m.clone() * bx + cx;

            // Tangent dy/dt at each root for winding direction
            let dy_plus = t_p.clone() * (2.0 * ay.clone()) + by.clone();
            let dy_minus = t_m.clone() * (2.0 * ay) + by;

            // Step: 1.0 if crossing is to the left of or at X
            let crossed_plus = (X >= x_plus).select(1.0, 0.0);
            let crossed_minus = (X >= x_minus).select(1.0, 0.0);

            // Validity: only count roots with t in [0, 1]
            let valid_plus = t_p.clone().ge(0.0) & t_p.clone().le(1.0);
            let valid_minus = t_m.clone().ge(0.0) & t_m.clone().le(1.0);"""

replace_in_file('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', search_2, replace_2)


# 2. scene3d.rs
# valid_t, valid_deriv, hx, hy, hz, etc. in `scene3d.rs` kernel! macros.
# Let's read the macro contents and rename variables properly.
