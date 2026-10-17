//! The macro's public-behaviour contracts: a `kernel!` body, baked, must agree
//! with scalar `f32` ground truth.
//!
//! Ground truth is scalar `f32` computed here in Rust, deliberately not an
//! oracle that shares a definition with the thing it checks: that cannot see a
//! shared-definition bug (CLAUDE.md, "Precision is on the table; range is
//! not"). A kernel is baked over a one-point lattice, which is the only way a
//! kernel becomes a number.
//!
//! Transcendentals are deliberately absent: their accuracy is pinned against
//! libm per op in `pixelflow-codegen/tests/transcendental_jit.rs` and their
//! *range* in `pixelflow-codegen/tests/trig_range_jit.rs`. What these check is
//! that the front end, the e-graph and the backend carry an expression's
//! meaning from source to buffer.

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

/// Tabulate over a one-point lattice and read the value back.
fn bake(k: &Kernel, p: (f32, f32, f32, f32)) -> f32 {
    Lattice::eval_at(k, p.0, p.1)
}

/// `(x, y, z, w)` — the first two are coordinates, the last two are the values
/// the parameters `z` and `w` take.
const SAMPLES: &[(f32, f32, f32, f32)] = &[
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0, 4.0),
    (-1.5, 0.5, 2.25, -3.0),
    (3.0, 4.0, 0.0, 1.0),
    (0.7, 1.3, -0.9, 2.1),
];

/// Wide enough for `Recip`/`Rsqrt`-class estimates and one-versus-two-rounding
/// `MulAdd`, which the optimizer is free to introduce; a semantics-breaking
/// rewrite misses by orders of magnitude more.
fn check(name: &str, got: f32, want: f32) {
    let diff = (got - want).abs();
    let tol = 1e-3 + 1e-3 * want.abs();
    assert!(
        diff <= tol,
        "{name}: got={got} want={want} diff={diff} tol={tol}"
    );
}

#[test]
fn arithmetic_matches_scalar_truth() {
    let build = kernel!(|z: f32, w: f32| (X - Y) * z + X / (w + 10.0));
    for &p in SAMPLES {
        let want = (p.0 - p.1) * p.2 + p.0 / (p.3 + 10.0);
        check("arith", bake(&build(p.2, p.3), p), want);
    }
}

#[test]
fn params_match_scalar_truth() {
    let (cx, cy, r) = (0.25_f32, -0.75_f32, 1.5_f32);
    let k = kernel!(|cx: f32, cy: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    })(cx, cy, r);
    for &p in SAMPLES {
        let want = ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt() - r;
        check("circle_sdf", bake(&k, p), want);
    }
}

#[test]
fn piecewise_matches_scalar_truth() {
    let build = kernel!(|z: f32| (X * Y).max(z).min(10.0));
    for &p in SAMPLES {
        let want = (p.0 * p.1).max(p.2).min(10.0);
        check("minmax", bake(&build(p.2), p), want);
    }

    let build = kernel!(|z: f32, w: f32| (X.lt(Y)).select(z, w));
    for &p in SAMPLES {
        let want = if p.0 < p.1 { p.2 } else { p.3 };
        check("select", bake(&build(p.2, p.3), p), want);
    }
}

#[test]
fn abs_recip_matches_scalar_truth() {
    let build = kernel!(|z: f32| (X - Y).abs() + (z + 5.0).recip());
    for &p in SAMPLES {
        let want = (p.0 - p.1).abs() + 1.0 / (p.2 + 5.0);
        check("abs_recip", bake(&build(p.2), p), want);
    }
}

/// A projection is the *symbolic* derivative of the expression it wraps: a
/// `Dwrt` chain that the runtime tier resolves when the kernel is baked — not a
/// finite difference, and not zero.
#[test]
fn projections_are_the_symbolic_derivative() {
    let k = kernel!(|| DX(X * X));
    for &p in SAMPLES {
        check("dx_of_x_squared", bake(&k, p), 2.0 * p.0);
    }

    let k = kernel!(|| DY(X * Y * Y));
    for &p in SAMPLES {
        check("dy_of_x_y_squared", bake(&k, p), 2.0 * p.0 * p.1);
    }
}

/// `V(e)` is `e`: every expression is already a value.
#[test]
fn v_of_an_expression_is_the_expression() {
    let build = kernel!(|z: f32| V(X * Y + z));
    for &p in SAMPLES {
        check("v_identity", bake(&build(p.2), p), p.0 * p.1 + p.2);
    }
}

/// `∂/∂x √(x²+y²) = x/r` and `∂/∂y = y/r`, sampled away from the singularity
/// at `r = 0`.
#[test]
fn dx_and_dy_of_a_distance_field_are_x_over_r_and_y_over_r() {
    const AWAY_FROM_ORIGIN: &[(f32, f32, f32, f32)] = &[
        (3.0, 4.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (-2.0, 5.0, 0.0, 0.0),
        (0.5, -0.25, 0.0, 0.0),
        (10.0, -10.0, 0.0, 0.0),
    ];
    let dx = kernel!(|| DX((X * X + Y * Y).sqrt()));
    let dy = kernel!(|| DY((X * X + Y * Y).sqrt()));
    for &p in AWAY_FROM_ORIGIN {
        let r = (p.0 * p.0 + p.1 * p.1).sqrt();
        check("dx_dist", bake(&dx, p), p.0 / r);
        check("dy_dist", bake(&dy, p), p.1 / r);
    }
}

/// `d²/dx² x³ = 6x`, by nested `Dwrt`.
#[test]
fn dxx_of_a_cubic_is_six_x() {
    let k = kernel!(|| DXX(X * X * X));
    for &p in SAMPLES {
        check("dxx_cubic", bake(&k, p), 6.0 * p.0);
    }
}

/// The font-shaped coverage ramp over a line, minus the Y-range mask, which
/// carries no derivative content: with `d = x − ((y−y0)k + x0)` its gradient
/// is `√(1+k²)`, so the coverage is a clamped ramp of `d` over that gradient.
#[test]
fn the_font_coverage_ramp_matches_its_closed_form() {
    let (x0, y0, k, dir, min_grad) = (0.3_f32, -0.2_f32, 0.7_f32, -1.0_f32, 0.001_f32);
    let coverage = kernel!(
        |x0: f32, y0: f32, dx_over_dy: f32, dir: f32, min_grad: f32| {
            let d = X - ((Y - y0) * dx_over_dy + x0);
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            let ramp = (V(d) / (grad + V(min_grad)) + V(0.5))
                .max(V(0.0))
                .min(V(1.0));
            ramp * V(dir)
        }
    )(x0, y0, k, dir, min_grad);

    // Both sides of the ramp, its interior, and points far into each
    // saturated region.
    for (x, y) in [
        (0.0_f32, 0.0_f32),
        (0.3, -0.2),
        (0.31, -0.2),
        (0.29, -0.2),
        (1.5, 0.5),
        (-1.5, 0.5),
        (0.8, 1.0),
        (-0.4, -1.0),
        (2.0, -2.0),
    ] {
        let d = x - ((y - y0) * k + x0);
        let grad = (1.0 + k * k).sqrt();
        let want = (d / (grad + min_grad) + 0.5).clamp(0.0, 1.0) * dir;
        check("font_coverage", bake(&coverage, (x, y, 0.0, 0.0)), want);
    }
}
