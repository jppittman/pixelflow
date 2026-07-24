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

/// `.at()` samples a manifold param at warped coordinates — each site gets
/// its own warped splice. Central difference of f = x²·y is 2xy exactly
/// (the quadratic's second differences cancel).
#[test]
fn at_sites_warp_coordinates_per_site() {
    let f = kernel_jit!(|| X * X * Y);

    let central_dx = kernel_jit!(|tex: kernel| {
        (tex.at(X + 1.0, Y, Z, W) - tex.at(X - 1.0, Y, Z, W)) * 0.5
    })(f);

    for (x, y) in [(2.0f32, 3.0f32), (-1.5, 0.5), (0.0, 4.0)] {
        check("central_dx", eval(&central_dx, x, y), 2.0 * x * y);
    }
}

/// Bare references and `.at()` sites of the same param coexist: the bare use
/// shares one fragment, each site gets its own warp.
#[test]
fn bare_and_at_sites_mix() {
    let f = kernel_jit!(|| X + Y * 10.0);

    let m = kernel_jit!(|g: kernel| V(g) + g.at(Y, X, Z, W))(f);

    for (x, y) in [(1.0f32, 2.0f32), (-3.0, 0.5)] {
        let want = (x + y * 10.0) + (y + x * 10.0);
        check("bare_plus_at", eval(&m, x, y), want);
    }
}


/// Named `kernel!` structs are spliceable leaves: the combinator ZST stays
/// the direct-eval path, but `HasIr` lets a fused JIT root absorb it — its
/// manifold fields (themselves `HasIr`) splice recursively and its scalar
/// fields bake. This is the P4 answer to "named structs can't own JIT
/// memory": they don't need to.
#[test]
fn named_struct_splices_into_jit_host() {
    use pixelflow_compiler::kernel;

    kernel!(
        struct Offset = |m: kernel, dx: f32| { m + dx }
    );

    let base = kernel_jit!(|| X * Y);
    let offset = Offset { m: base, dx: 7.0 };

    let host = kernel_jit!(|f: kernel| V(f) * 2.0)(offset);

    for (x, y) in [(2.0f32, 3.0f32), (-1.0, 4.0)] {
        check("named_splice", eval(&host, x, y), (x * y + 7.0) * 2.0);
    }
}
