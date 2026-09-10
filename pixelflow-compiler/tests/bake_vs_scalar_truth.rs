//! The macro's five public-behaviour contracts: a `kernel!` body, baked, must
//! agree with scalar `f32` ground truth.
//!
//! There is one tier now. This file used to be `kernel_routing_parity`, and
//! each contract checked the same body twice — once through the LLVM tier's
//! combinator ZST evaluated by this test's own loop, once through the arena
//! tier's `Lattice::bake` — against ground truth, because the two tiers no
//! longer shared a value. The combinator tier is gone
//! (docs/plans/2026-09-06-kernel-with-a-lattice.md, S4b-2), so what is left is
//! the half that was ever the point: **truth**, not agreement between two
//! implementations of it.
//!
//! Ground truth is scalar `f32` computed here in Rust, deliberately *not* the
//! IR's own `eval_scalar` oracle: an oracle that shares a definition with the
//! thing it checks cannot see a shared-definition bug (CLAUDE.md, "Precision
//! is on the table; range is not"). A kernel is baked over a one-point
//! lattice, which is the only way a kernel becomes a number.
//!
//! Transcendentals are deliberately absent: their accuracy is pinned against
//! libm per op in `pixelflow-codegen/tests/transcendental_jit.rs`. Their
//! *range* was pinned in `pixelflow-ir/tests/trig_range.rs`, which asserted
//! it through the scalar interpreter and was deleted with it — that check
//! needs rebuilding on the JIT (CLAUDE.md says so where it states the rule). What these five check is
//! that the front end, the e-graph and the backend carry an expression's
//! meaning from source to buffer.

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

/// Tabulate over a one-point lattice and read the value back.
fn bake(k: &Kernel, p: (f32, f32, f32, f32)) -> f32 {
    Lattice::eval_at(k, p.0, p.1)
}

/// `(x, y, z, w)` — the first two are coordinates, the last two are scalars
/// the builder folds in. They were the Z and W coordinates until a lattice
/// became two axes.
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

/// A projection is the *symbolic* derivative of the expression it wraps,
/// resolved by the e-graph at expansion when it can and by codegen otherwise
/// — not a finite difference, and not zero. This was the contract that used
/// to say the two tiers disagreed here on purpose (over a `Field` domain the
/// combinator backend's `DX` was 0, which is what the fonts' "hard step" case
/// relied on); with one tier there is one answer, and it is the calculus.
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
