//! `kernel!` (optimized) agrees with `kernel_raw!` (the `Identity` optimizer),
//! and both agree with a closed-form oracle.
//!
//! This is the as-if contract of docs/process/2026-10-13-tests-check-the-as-if-contract.md
//! applied across optimization: the e-graph may fuse, reassociate and pick
//! equivalent forms, and this file asserts nothing about which form it picked.
//! It asserts only that the value a caller gets is the value the expression
//! means, within the documented accuracy of each op. Every lattice point of a
//! 64×8 frame is compared, so a rewrite that is wrong at one sample cannot hide
//! behind a lucky one.
//!
//! Inputs are chosen away from singularities and from cancellation: no
//! division near zero, no `sin` near its domain edge, no subtraction of nearly
//! equal terms. Where a comparison sits on a lattice point (`X` against
//! `8·Y`), the threshold is offset by ½ so that no sample is a tie.
//!
//! Oracle: f64 closed forms over `i`, `j` (the lattice coordinates as exact
//! integers), with f32 constants widened exactly, so the only difference from
//! the compiled value is the compiled arithmetic itself.
//!
//! Tolerances, and where each comes from:
//! - Arithmetic is `4·ε·|result|` (`four_ulps`). Each rounding moves a value by
//!   at most `ε/2` of its own magnitude and every partial value here is at most
//!   the result's scale, so the four-term sum (seven roundings) is inside it,
//!   and so is an FMA fusion (one rounding fewer).
//! - `div` may be lowered to `a · recip(b)` (`MulRecip` in
//!   `pixelflow-search/src/math/algebra.rs`), and `Recip` is a hardware estimate:
//!   `rcpps` is ~12 bits (CLAUDE.md, "Floating point at the edges"), whose
//!   documented maximum relative error is 1.5·2⁻¹², so the bound is `2⁻¹¹`.
//! - `sin` is `TRIG_DOMAIN`'s documented worst case, 1.5e-6 absolute
//!   (`pixelflow-ir/src/passes.rs`, `expand_sin_phase`), times four.
//! - `exp` is `exp2`'s documented polynomial truncation, 1.6e-4 relative on
//!   `[0, 1)` (`EXP2_POLY` in `pixelflow-ir/src/passes.rs`), times two.
//! - `min`/`max` select an operand, and the operands here are exact in f32, so
//!   the tolerance is zero.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};

const WIDTH: usize = 64;
const HEIGHT: usize = 8;

/// `sin` over `TRIG_DOMAIN`: the documented worst case, 1.5e-6 absolute, times four.
const SIN_TOL: f64 = 6.0e-6;

/// `exp` relative accuracy: `exp2`'s documented truncation error, 1.6e-4, times two.
const EXP_REL_TOL: f64 = 3.2e-4;

/// `Recip` estimate bound: `rcpps`'s 1.5·2⁻¹², rounded up to a power of two.
const RECIP_REL_TOL: f64 = 1.0 / 2048.0;

/// Four ulps of a result of this magnitude. `ε·|v|` is at least one ulp of `v`
/// and at most two, so this is four ulps or more.
fn four_ulps(magnitude: f64) -> f64 {
    4.0 * f64::from(f32::EPSILON) * magnitude.abs()
}

/// Tabulate a kernel over the whole frame, row-major, one value per lattice point.
fn lanes(k: &Kernel) -> Vec<f32> {
    Lattice::frame(WIDTH, HEIGHT).bake(k).into_buffer()
}

/// Compare every lattice point of the optimized and raw forms against `oracle`,
/// and against each other. `oracle(i, j)` returns the closed-form value at the
/// lattice point `(X, Y) = (i, j)` and the tolerance that value is owed.
fn check(name: &str, optimized: &Kernel, raw: &Kernel, oracle: impl Fn(f64, f64) -> (f64, f64)) {
    let opt = lanes(optimized);
    let raw = lanes(raw);
    assert_eq!(opt.len(), WIDTH * HEIGHT, "{name}: optimized frame size");
    assert_eq!(raw.len(), WIDTH * HEIGHT, "{name}: raw frame size");
    for j in 0..HEIGHT {
        for i in 0..WIDTH {
            let lane = j * WIDTH + i;
            let (want, tol) = oracle(i as f64, j as f64);
            let o = f64::from(opt[lane]);
            let r = f64::from(raw[lane]);
            assert!(
                (o - want).abs() <= tol,
                "{name} optimized at ({i},{j}): got {o}, want {want}, tol {tol}"
            );
            assert!(
                (r - want).abs() <= tol,
                "{name} raw at ({i},{j}): got {r}, want {want}, tol {tol}"
            );
            assert!(
                (o - r).abs() <= 2.0 * tol,
                "{name} optimized vs raw at ({i},{j}): {o} vs {r}, tol {tol}"
            );
        }
    }
}

#[test]
fn add_sub_mul_agrees() {
    check(
        "add_sub_mul",
        &kernel!(|| (X + 0.5) * 0.25 - Y * 0.125),
        &kernel_raw!(|| (X + 0.5) * 0.25 - Y * 0.125),
        |x, y| {
            let want = (x + 0.5) * 0.25 - y * 0.125;
            (want, four_ulps((x + 0.5) * 0.25 + y * 0.125))
        },
    );
}

/// Two products into a sum: the shape an FMA fusion may rewrite.
#[test]
fn mul_add_agrees() {
    let a = f64::from(0.37f32);
    let b = f64::from(0.21f32);
    check(
        "mul_add",
        &kernel!(|| X * 0.37 + Y * 0.21 + 1.5),
        &kernel_raw!(|| X * 0.37 + Y * 0.21 + 1.5),
        move |x, y| {
            let want = x * a + y * b + 1.5;
            (want, four_ulps(want))
        },
    );
}

/// The denominator is at least 1, so no sample is near the singularity. The
/// optimizer may compute this as `a · recip(b)`, so the tolerance is the
/// estimate's bound, not a few ulps.
#[test]
fn div_agrees() {
    check(
        "div",
        &kernel!(|| (X + 1.0) / (Y + 1.0)),
        &kernel_raw!(|| (X + 1.0) / (Y + 1.0)),
        |x, y| {
            let want = (x + 1.0) / (y + 1.0);
            (want, RECIP_REL_TOL * want.abs())
        },
    );
}

/// The argument is at least 1, so the square root is of a positive value.
#[test]
fn sqrt_agrees() {
    check(
        "sqrt",
        &kernel!(|| (X + 1.0).sqrt()),
        &kernel_raw!(|| (X + 1.0).sqrt()),
        |x, _| {
            let want = (x + 1.0).sqrt();
            (want, four_ulps(want))
        },
    );
}

/// `X` against `7·Y + ½` never ties, since `X` is an integer and the threshold
/// is a half-integer. Both operands are exact in f32, so min and max select
/// exactly.
#[test]
fn min_max_agrees() {
    check(
        "min_max",
        &kernel!(|| X.min(Y * 7.0 + 0.5).max(3.0)),
        &kernel_raw!(|| X.min(Y * 7.0 + 0.5).max(3.0)),
        |x, y| {
            let want = x.min(y * 7.0 + 0.5).max(3.0);
            (want, 0.0)
        },
    );
}

/// The comparison threshold is `8·Y + ½`, so no sample is a tie. Both arms run
/// on a mixed batch; the select discards the inactive one.
#[test]
fn select_on_comparison_agrees() {
    let a = f64::from(0.37f32);
    let b = f64::from(0.21f32);
    check(
        "select",
        &kernel!(|| X.lt(Y * 8.0 + 0.5).select(X * 0.37, Y * 0.21 - 4.0)),
        &kernel_raw!(|| X.lt(Y * 8.0 + 0.5).select(X * 0.37, Y * 0.21 - 4.0)),
        move |x, y| {
            let want = if x < y * 8.0 + 0.5 {
                x * a
            } else {
                y * b - 4.0
            };
            (want, four_ulps(want))
        },
    );
}

/// `X - 4·Y - ½` is a half-integer, so its magnitude is at least ½ and the
/// result is never a cancellation.
#[test]
fn abs_and_neg_agree() {
    check(
        "abs_neg",
        &kernel!(|| (X - Y * 4.0 - 0.5).abs() + (-Y)),
        &kernel_raw!(|| (X - Y * 4.0 - 0.5).abs() + (-Y)),
        |x, y| {
            let want = (x - y * 4.0 - 0.5).abs() - y;
            (want, four_ulps((x - y * 4.0 - 0.5).abs() + y))
        },
    );
}

/// The argument `(X − 32)·0.1` spans `[−3.2, 3.1]`, both signs, well inside
/// `TRIG_DOMAIN`.
#[test]
fn sin_agrees_within_domain() {
    let c = f64::from(0.1f32);
    check(
        "sin",
        &kernel!(|| ((X - 32.0) * 0.1).sin() + Y * 0.25),
        &kernel_raw!(|| ((X - 32.0) * 0.1).sin() + Y * 0.25),
        move |x, y| {
            let want = ((x - 32.0) * c).sin() + y * 0.25;
            (want, SIN_TOL + four_ulps(1.0 + y * 0.25))
        },
    );
}

/// The exponent `Y·0.5 − X·0.05` spans `[−3.15, 3.5]`, so the result is in
/// `[0.04, 34]`: far from `exp`'s saturation at ±126.
#[test]
fn exp_agrees_within_domain() {
    let c = f64::from(0.05f32);
    check(
        "exp",
        &kernel!(|| (Y * 0.5 - X * 0.05).exp()),
        &kernel_raw!(|| (Y * 0.5 - X * 0.05).exp()),
        move |x, y| {
            let want = (y * 0.5 - x * c).exp();
            (want, EXP_REL_TOL * want)
        },
    );
}

/// A fold over four terms, each a sum of products: `Σ_{i<4} (X + i)·Y = Y·(4X + 6)`.
#[test]
fn short_sum_over_agrees() {
    check(
        "sum_over",
        &kernel!(|| (0..4).map(|i| (X + i as f32) * Y).sum::<f32>()),
        &kernel_raw!(|| (0..4).map(|i| (X + i as f32) * Y).sum::<f32>()),
        |x, y| {
            let want = y * (4.0 * x + 6.0);
            (want, four_ulps(want))
        },
    );
}

kernel! {
    /// Σ_{i<n} (X + i)·Y, the count per call.
    pub fn counted(n: usize) -> f32 {
        (0..n).map(|i| (X + i as f32) * Y).sum::<f32>()
    }
}

kernel_raw! {
    /// The same, as written.
    pub fn counted_raw(n: usize) -> f32 {
        (0..n).map(|i| (X + i as f32) * Y).sum::<f32>()
    }
}

/// A per-call count: `Σ_{i<n} (X + i)·Y = Y·(nX + n(n−1)/2)`, the program
/// saturated at expansion and the one as written. The lattice's coordinates
/// are integers, so every value here is exact.
#[test]
fn a_per_call_count_agrees() {
    for n in [0_usize, 1, 5] {
        check(
            &format!("counted({n})"),
            &counted(n),
            &counted_raw(n),
            |x, y| {
                let n = n as f64;
                let want = y * (n * x + n * (n - 1.0) / 2.0);
                (want, four_ulps(want))
            },
        );
    }
}
