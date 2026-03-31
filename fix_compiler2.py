import re

with open("pixelflow-graphics/src/fonts/ttf_curve_analytical.rs", "rb") as f:
    ttf = f.read().decode('utf-8')

# Inline ttf
ttf = ttf.replace(
"""                let t = (Y - cy) / by;
                let in_t = t.clone().ge(0.0) & t.clone().le(1.0);

                // x-coordinate at intersection
                let x_int = t.clone() * t.clone() * ax + t.clone() * bx + cx;""",
"""                let in_t = ((Y - cy.clone()) / by.clone()).ge(0.0) & ((Y - cy.clone()) / by.clone()).le(1.0);

                // x-coordinate at intersection
                let x_int = ((Y - cy.clone()) / by.clone()) * ((Y - cy.clone()) / by.clone()) * ax + ((Y - cy.clone()) / by.clone()) * bx + cx;"""
)

ttf = ttf.replace(
"""            // Two roots: t = (-by +/- sqrt(disc)) / (2*ay)
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
            let valid_minus = t_minus.clone().ge(0.0) & t_minus.clone().le(1.0);""",
"""            // Two roots: t = (-by +/- sqrt(disc)) / (2*ay)
            // X-coordinates at intersection points
            let x_plus = (sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone()) * (sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone()) * ax.clone() + (sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone()) * bx.clone() + cx.clone();
            let x_minus = (sqrt_disc.clone() * -inv_2a.clone() + neg_b_2a.clone()) * (sqrt_disc.clone() * -inv_2a.clone() + neg_b_2a.clone()) * ax.clone() + (sqrt_disc.clone() * -inv_2a.clone() + neg_b_2a.clone()) * bx.clone() + cx.clone();

            // Tangent dy/dt at each root for winding direction
            let dy_plus = (sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone()) * (2.0 * ay.clone()) + by.clone();
            let dy_minus = (sqrt_disc.clone() * -inv_2a.clone() + neg_b_2a.clone()) * (2.0 * ay.clone()) + by.clone();

            // Step: 1.0 if crossing is to the left of or at X
            let crossed_plus = (X >= x_plus).select(1.0, 0.0);
            let crossed_minus = (X >= x_minus).select(1.0, 0.0);

            // Validity: only count roots with t in [0, 1]
            let valid_plus = (sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone()).ge(0.0) & (sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone()).le(1.0);
            let valid_minus = (sqrt_disc.clone() * -inv_2a.clone() + neg_b_2a.clone()).ge(0.0) & (sqrt_disc.clone() * -inv_2a.clone() + neg_b_2a.clone()).le(1.0);"""
)
with open("pixelflow-graphics/src/fonts/ttf_curve_analytical.rs", "wb") as f:
    f.write(ttf.encode('utf-8'))

with open("pixelflow-graphics/src/scene3d.rs", "rb") as f:
    scene = f.read().decode('utf-8')

# Fix self.inner.eval(r_x, r_y, r_z, w)
scene = scene.replace("self.inner.eval(r_x, r_y, r_z, w)", "self.inner.eval((r_x, r_y, r_z, w))")

scene = scene.replace(
"""kernel!(pub struct Surface = |geometry: kernel, material: kernel, background: kernel| Jet3 -> Field {
    // 1. Get distance t from geometry
    let t = geometry;

    // 2. Validate hit: t > 0, t < max, derivatives reasonable
    let t_max = 1000000.0;
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

    mask.select(mat_val, bg_val)
});""",
"""kernel!(pub struct Surface = |geometry: kernel, material: kernel, background: kernel| Jet3 -> Field {
    ((V(geometry.clone()) > 0.0) & (V(geometry.clone()) < 1000000.0) & ((DX(geometry.clone()) * DX(geometry.clone()) + DY(geometry.clone()) * DY(geometry.clone()) + DZ(geometry.clone()) * DZ(geometry.clone())) < (10000.0 * 10000.0))).select(
        material.at(X * geometry.clone(), Y * geometry.clone(), Z * geometry.clone(), W),
        background
    )
});"""
)

scene = scene.replace(
"""kernel!(pub struct ColorSurface = |geometry: kernel, material: kernel, background: kernel| Jet3 -> Discrete {
    // 1. Get distance t from geometry
    let t = geometry;

    // 2. Validate hit: t > 0, t < max, derivatives reasonable
    let t_max = 1000000.0;
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

    mask.select(mat_val, bg_val)
});""",
"""kernel!(pub struct ColorSurface = |geometry: kernel, material: kernel, background: kernel| Jet3 -> Discrete {
    ((V(geometry.clone()) > 0.0) & (V(geometry.clone()) < 1000000.0) & ((DX(geometry.clone()) * DX(geometry.clone()) + DY(geometry.clone()) * DY(geometry.clone()) + DZ(geometry.clone()) * DZ(geometry.clone())) < (10000.0 * 10000.0))).select(
        material.at(X * geometry.clone(), Y * geometry.clone(), Z * geometry.clone(), W),
        background
    )
});"""
)

scene = scene.replace(
"""kernel!(pub struct GeometryMask = |geometry: kernel| Jet3 -> Field {
    let t = geometry;
    let t_max = 1000000.0;
    let deriv_max = 10000.0;

    // Valid if: t > 0, t < max, derivatives reasonable
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);

    valid_t & valid_deriv
});""",
"""kernel!(pub struct GeometryMask = |geometry: kernel| Jet3 -> Field {
    pixelflow_core::V((V(geometry.clone()) > 0.0) & (V(geometry.clone()) < 1000000.0) & ((DX(geometry.clone()) * DX(geometry.clone()) + DY(geometry.clone()) * DY(geometry.clone()) + DZ(geometry.clone()) * DZ(geometry.clone())) < (10000.0 * 10000.0)))
});"""
)

with open("pixelflow-graphics/src/scene3d.rs", "wb") as f:
    f.write(scene.encode('utf-8'))
