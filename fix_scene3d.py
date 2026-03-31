import re

with open("pixelflow-graphics/src/scene3d.rs", "rb") as f:
    scene = f.read().decode('utf-8')

scene = scene.replace("self.inner.eval(r_x, r_y, r_z, w)", "self.inner.eval((r_x, r_y, r_z, w))")

# Note we must retain the type `Jet3` in closures where needed, or just let Rust infer.
# The error in scene3d.rs was missing valid_t, valid_deriv, etc.
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
    ((V(geometry.clone()) > 0.0) & (V(geometry.clone()) < 1000000.0)) & ((DX(geometry.clone()) * DX(geometry.clone()) + DY(geometry.clone()) * DY(geometry.clone()) + DZ(geometry.clone()) * DZ(geometry.clone())) < (10000.0 * 10000.0))
});"""
)

with open("pixelflow-graphics/src/scene3d.rs", "wb") as f:
    f.write(scene.encode('utf-8'))
