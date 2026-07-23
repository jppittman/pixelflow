//! P4 acceptance: IR-carrying kernels compose by arena splicing.
//!
//! A `kernel_jit!` kernel with manifold-typed params is a builder over other
//! IR-carrying kernels: at construction it splices each argument's fragment
//! into its own arena (`HasIr`), substitutes the reserved slot variables, and
//! JIT-compiles ONE fused kernel. Derivative projections applied to a
//! manifold param differentiate the *composed* expression — the calculus
//! resolves after splicing, in the runtime `lower_dwrt` tier.

use pixelflow_compiler::kernel_jit;
use pixelflow_core::{Field, Manifold};

type F4 = (Field, Field, Field, Field);

fn lane0(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

fn eval(m: &impl Manifold<F4, Output = Field>, x: f32, y: f32) -> f32 {
    lane0(m.eval((
        Field::from(x),
        Field::from(y),
        Field::from(0.0),
        Field::from(0.0),
    )))
}

fn check(name: &str, got: f32, want: f32) {
    let diff = (got - want).abs();
    let tol = 1e-3 + 1e-3 * want.abs();
    assert!(
        diff <= tol,
        "{name}: got={got} want={want} diff={diff} tol={tol}"
    );
}

/// The font architecture end-to-end on the arena backend: a generic
/// gradient-normalized AA ramp composed over a circle SDF. The gradient of a
/// (unit-speed) SDF is 1, so coverage = clamp(d/(1+mg) + 0.5, 0, 1).
#[test]
fn aa_ramp_over_circle_sdf() {
    let circle = kernel_jit!(|cx: f32, cy: f32, r: f32| {
        ((X - cx) * (X - cx) + (Y - cy) * (Y - cy)).sqrt() - r
    })(0.25, -0.5, 1.0);

    let coverage = kernel_jit!(|sdf: kernel, min_grad: f32| {
        let grad = (DX(sdf) * DX(sdf) + DY(sdf) * DY(sdf)).sqrt();
        (V(sdf) / (grad + min_grad) + 0.5).max(0.0).min(1.0)
    })(circle, 0.001);

    for (x, y) in [
        (0.25f32, 0.5f32), // on the ramp (d = 0)
        (0.35, 0.55),
        (1.6, -0.5), // fully outside
        (0.25, -0.5 + 0.1), // deep inside
        (-1.0, 0.4),
    ] {
        let d = ((x - 0.25).powi(2) + (y + 0.5).powi(2)).sqrt() - 1.0;
        let want = (d / (1.0 + 0.001) + 0.5).clamp(0.0, 1.0);
        check("aa_over_circle", eval(&coverage, x, y), want);
    }
}

/// Composition chains: the output of one composed kernel is itself
/// IR-carrying and splices into the next host.
#[test]
fn composition_nests() {
    let plane = kernel_jit!(|k: f32| X * k - Y)(2.0);

    let doubled = kernel_jit!(|inner: kernel, s: f32| V(inner) * s)(plane, 3.0);

    let shifted = kernel_jit!(|inner: kernel, c: f32| V(inner) + c)(doubled, 10.0);

    for (x, y) in [(1.0f32, 0.5f32), (-2.0, 4.0), (0.0, 0.0)] {
        let want = (x * 2.0 - y) * 3.0 + 10.0;
        check("nested", eval(&shifted, x, y), want);
    }
}

/// Derivatives see through the whole spliced chain, not just one layer:
/// DX of a composed-and-scaled SDF is the scaled derivative.
#[test]
fn derivative_of_nested_composition() {
    let dist = kernel_jit!(|| (X * X + Y * Y).sqrt());

    let scaled = kernel_jit!(|inner: kernel, s: f32| V(inner) * s)(dist, 5.0);

    let ddx = kernel_jit!(|f: kernel| DX(f))(scaled);

    for (x, y) in [(3.0f32, 4.0f32), (1.0, 1.0), (-2.0, 5.0)] {
        let want = 5.0 * x / (x * x + y * y).sqrt();
        check("d_nested", eval(&ddx, x, y), want);
    }
}

/// A shared manifold param used at several sites splices once per site but
/// evaluates consistently (the fused arena keeps each site's fragment DAG).
#[test]
fn manifold_param_used_multiple_times() {
    let f = kernel_jit!(|| X * Y);

    let combined = kernel_jit!(|g: kernel| V(g) * V(g) + V(g))(f);

    for (x, y) in [(2.0f32, 3.0f32), (-1.0, 4.0)] {
        let v = x * y;
        check("multi_use", eval(&combined, x, y), v * v + v);
    }
}
