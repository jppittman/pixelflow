//! A macro-built derivative composed under a coordinate warp.
//!
//! `Kernel::dx` is the symbolic derivative with respect to the *sampling*
//! coordinate, so a kernel warped by [`Kernel::at`] must differentiate through
//! the warp — that is the chain rule, and it is the whole reason the font
//! antialiasing ramp is one *screen* pixel wide at any glyph scale (CLAUDE.md,
//! "Antialiasing: symbolic derivatives"; `fonts/ttf.rs`'s module doc).
//!
//! It gets that for free, by leaving `Dwrt` alone. `at` warps a kernel by
//! substituting into its `Var` leaves, so a surviving `Dwrt(f, x)` has the
//! substitution reach its *operand*: `Dwrt(x², 0)` warped by `x ↦ 2x` is
//! `Dwrt(4x², 0)`, which differentiates to `8x`. Nothing had to know about
//! composition.
//!
//! Until 2026-09-08 the macros resolved `DX` at expansion time, and a later
//! warp then substituted into the *already-resolved* derivative — into `2x`,
//! where there was no operand left for the chain rule to reach:
//!
//! ```text
//!   d/dx (x²)                 = 2x         → 6 at x=3   (correct, unwarped)
//!   d/dx ((2x)²) = d/dx (4x²) = 8x         → 24 at x=3  (chain rule)
//!   what we computed: (2·u) with u := 2x   → 12 at x=3  (wrong)
//! ```
//!
//! The production glyph kernels were correct across that whole period only by
//! coincidence: all three carry a `&` mask, which made the e-graph decline the
//! whole arena, so their `Dwrt` nodes survived expansion and were resolved
//! after the warp. Removing the mask would have silently changed their values.
//!
//! This file is also where the two tiers' *equivalence* is now asserted. It
//! used to be checked inside the compiler's arena bridge by running the
//! expansion-time
//! differentiator and the runtime one on the same arena and comparing — two
//! implementations of one calculus, holding each other honest through private
//! functions. There is one implementation now, so the surviving claim is the
//! one a caller can observe: a derivative built by the macro denotes the same
//! function as the same derivative composed from `Kernel` values.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};

/// Warp `k`'s sampling coordinate by `x -> 2x`, leaving `y` alone.
fn warped_by_double_x(k: &Kernel) -> Kernel {
    k.at(&Kernel::x().mul(&Kernel::constant(2.0)), &Kernel::y())
}

/// `d/dx (x²) = 2x`. Unwarped, both macros agree with the truth — which is
/// what makes the warped cases below a statement about the warp.
#[test]
fn an_unwarped_derivative_is_correct_in_both_macros() {
    const TRUTH_AT_3: f32 = 6.0;

    assert_eq!(
        Lattice::eval_at(&kernel!(|| DX(X * X)), 3.0, 0.0),
        TRUTH_AT_3
    );
    assert_eq!(
        Lattice::eval_at(&kernel_raw!(|| DX(X * X)), 3.0, 0.0),
        TRUTH_AT_3
    );
}

/// The chain rule through [`Kernel::at`], in `kernel!`.
#[test]
fn a_warped_derivative_follows_the_warp_in_kernel() {
    const TRUTH_AT_3: f32 = 24.0; // d/dx (2x)² = 8x

    assert_eq!(
        Lattice::eval_at(&warped_by_double_x(&kernel!(|| DX(X * X))), 3.0, 0.0),
        TRUTH_AT_3
    );
}

/// Same, in `kernel_raw!` — the e-graph tier is not what makes this work, so
/// the claim must hold with it switched off.
#[test]
fn a_warped_derivative_follows_the_warp_in_kernel_raw() {
    const TRUTH_AT_3: f32 = 24.0;

    assert_eq!(
        Lattice::eval_at(&warped_by_double_x(&kernel_raw!(|| DX(X * X))), 3.0, 0.0),
        TRUTH_AT_3
    );
}

/// The same derivative built by composing `Kernel` values. This is the
/// reference the macro cases above are measured against: it never resolved a
/// derivative early, so it was correct throughout, and the macros agreeing
/// with it *is* the tier-equivalence claim.
#[test]
fn a_composed_derivative_does_follow_the_warp() {
    const TRUTH_AT_3: f32 = 24.0;

    let x_squared = Kernel::x().mul(&Kernel::x());
    let got = Lattice::eval_at(&warped_by_double_x(&x_squared).dx(), 3.0, 0.0);

    assert_eq!(
        got, TRUTH_AT_3,
        "composing Kernel values must differentiate through the warp"
    );
}

/// A differentiand whose derivative is not a constant and not a monomial:
/// `d/dx √(x² + y²) = x / √(x² + y²)`, the gradient the glyph ramps are built
/// from. Checked against the composed form at several points rather than
/// against a hand-computed number, so the assertion is the two tiers agreeing
/// and not this test's own arithmetic.
#[test]
fn a_macro_derivative_agrees_with_the_composed_one_pointwise() {
    let macro_built = kernel!(|| DX((X * X + Y * Y).sqrt()));

    let x = Kernel::x();
    let y = Kernel::y();
    let composed = x.mul(&x).add(&y.mul(&y)).sqrt().dx();

    for &(px, py) in &[(3.0, 4.0), (1.0, 2.0), (-2.5, 0.5), (0.25, -3.0)] {
        let want = Lattice::eval_at(&composed, px, py);
        let got = Lattice::eval_at(&macro_built, px, py);
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at ({px}, {py}): kernel! gave {got}, composed Kernel values gave {want}"
        );
    }
}

/// The same, with a scalar parameter inside the differentiand — the case the
/// deleted `param_derivative_matches_runtime_tier` covered, which exercised a
/// `Param` leaf travelling through differentiation. `d/dx (p·√(x²+y²))`.
#[test]
fn a_macro_derivative_with_a_param_agrees_with_the_composed_one() {
    let build = kernel!(|p: f32| DX(p * (X * X + Y * Y).sqrt()));
    let macro_built = build(3.5);

    let x = Kernel::x();
    let y = Kernel::y();
    let composed = Kernel::constant(3.5)
        .mul(&x.mul(&x).add(&y.mul(&y)).sqrt())
        .dx();

    for &(px, py) in &[(3.0, 4.0), (1.0, 2.0), (-2.5, 0.5)] {
        let want = Lattice::eval_at(&composed, px, py);
        let got = Lattice::eval_at(&macro_built, px, py);
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at ({px}, {py}): kernel! gave {got}, composed Kernel values gave {want}"
        );
    }
}
