//! P3 routing parity: `kernel!` must produce the same observable results
//! whether it emits the combinator backend (default) or routes eligible
//! bodies through the arena/JIT backend (`feature = "arena-backend"`).
//!
//! Every assertion here is against scalar f32 ground truth and must hold
//! under BOTH configurations — run this suite twice:
//!
//! ```sh
//! cargo test -p pixelflow-compiler --test kernel_routing_parity
//! cargo test -p pixelflow-compiler --test kernel_routing_parity --features arena-backend
//! ```
//!
//! Transcendentals are deliberately absent: the combinator backend's `sin` is
//! known-inaccurate (see the header of `jit_parity.rs`), which is a parity bug
//! in the to-be-retired backend, tracked until that backend dies.

use pixelflow_compiler::kernel;
use pixelflow_core::{Field, Manifold};

type F4 = (Field, Field, Field, Field);

fn lane0(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

fn eval(m: &impl Manifold<F4, Output = Field>, p: (f32, f32, f32, f32)) -> f32 {
    lane0(m.eval((
        Field::from(p.0),
        Field::from(p.1),
        Field::from(p.2),
        Field::from(p.3),
    )))
}

const SAMPLES: &[(f32, f32, f32, f32)] = &[
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0, 4.0),
    (-1.5, 0.5, 2.25, -3.0),
    (3.0, 4.0, 0.0, 1.0),
    (0.7, 1.3, -0.9, 2.1),
];

// Wide enough for the combinator backend's approximate-rsqrt `sqrt`
// (~5e-4 relative; the JIT emits exact hardware sqrt — its numerics are the
// canonical ones per the plan). Tightens when the combinator emitter dies.
fn check(name: &str, got: f32, want: f32) {
    let diff = (got - want).abs();
    let tol = 1e-3 + 1e-3 * want.abs();
    assert!(
        diff <= tol,
        "{name}: got={got} want={want} diff={diff} tol={tol}"
    );
}

#[test]
fn routed_arithmetic_matches_truth() {
    let m = kernel!(|| (X - Y) * Z + X / (W + 10.0))();
    for &p in SAMPLES {
        check(
            "arith",
            eval(&m, p),
            (p.0 - p.1) * p.2 + p.0 / (p.3 + 10.0),
        );
    }
}

#[test]
fn routed_params_match_truth() {
    let (cx, cy, r) = (0.25_f32, -0.75_f32, 1.5_f32);
    let m = kernel!(|cx: f32, cy: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    })(cx, cy, r);
    for &p in SAMPLES {
        let want = ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt() - r;
        check("circle_sdf", eval(&m, p), want);
    }
}

#[test]
fn routed_piecewise_matches_truth() {
    let m = kernel!(|| (X * Y).max(Z).min(10.0))();
    for &p in SAMPLES {
        check("minmax", eval(&m, p), (p.0 * p.1).max(p.2).min(10.0));
    }

    let m = kernel!(|| (X.lt(Y)).select(Z, W))();
    for &p in SAMPLES {
        let want = if p.0 < p.1 { p.2 } else { p.3 };
        check("select", eval(&m, p), want);
    }
}

#[test]
fn routed_abs_recip_matches_truth() {
    let m = kernel!(|| (X - Y).abs() + (Z + 5.0).recip())();
    for &p in SAMPLES {
        check("abs_recip", eval(&m, p), (p.0 - p.1).abs() + 1.0 / (p.2 + 5.0));
    }
}

/// Derivative projections must NOT be transparently routed: over a `Field`
/// domain the combinator backend's `DX` is 0 (no derivative information —
/// the fonts' load-bearing "hard step" case), and routing to the arena
/// backend would change that to the symbolic derivative. This must yield 0
/// under BOTH configurations; kernels wanting symbolic derivatives opt in
/// via `kernel_jit!`.
#[test]
fn projection_kernels_stay_on_field_domain_semantics() {
    let m = kernel!(|| DX(X * X))();
    for &p in SAMPLES {
        check("dx_over_field_is_zero", eval(&m, p), 0.0);
    }
}
