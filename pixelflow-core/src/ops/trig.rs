//! # Chebyshev-Based Trigonometric Functions
//!
//! SIMD-vectorized sin, cos, atan2 using Chebyshev polynomial approximations.
//! All operations vectorize across all SIMD lanes simultaneously, replacing
//! per-lane libm scalar calls with parallel polynomial evaluation.
//!
//! **Domain**: `sin`/`cos`/`tan` are defined for `|x| < TRIG_DOMAIN` (2²⁰) and
//! return NaN outside it. Past that the argument reduction can no longer
//! resolve which point of the period an f32 is naming, and a plausible-looking
//! wrong answer is worse than none — see `expand_sin_phase` in
//! `pixelflow-ir`'s `passes`, which is where this function is defined and from
//! which the constants here are imported.
//!
//! **Accuracy**: max absolute error ≈ 1.5e-6 (sin, cos) over that domain, and
//! ≈ 8.7e-5 (atan2), not full float32 precision.
//! Sufficient for graphics/harmonics but not for exact angle round-tripping.
//! **Speed**: 5-10x faster than per-lane libm on SIMD backends.
//!
//! # Implementation Note
//!
//! This module builds AST graphs using operators, enabling automatic FMA fusion
//! and other optimizations. The graph is evaluated at the return boundary.

use crate::Field;
use crate::{Manifold, ManifoldExt};
// The reduction split, polynomials, and domain limit are defined once, in the
// IR expansion every backend lowers through, and used from here so the
// combinator tier and the JIT cannot disagree about what `sin` is.
use pixelflow_ir::passes::{ATAN_MINIMAX, SIN_CHEB, TAU_HI, TAU_LO, TAU_MID, TRIG_DOMAIN};

/// The standard 4D Field domain.
type Field4 = (Field, Field, Field, Field);

/// Evaluate a manifold graph to Field.
/// Since Field is a constant manifold, coordinates don't matter.
#[inline(always)]
fn eval<M: Manifold<Field4, Output = Field>>(m: M) -> Field {
    let zero = Field::from(0.0);
    m.eval((zero, zero, zero, zero))
}

// ============================================================================
// Constants (Computed at Compile Time)
// ============================================================================

const PI: f32 = core::f32::consts::PI;
const TWO_PI: f32 = core::f32::consts::TAU;
const PI_2: f32 = core::f32::consts::FRAC_PI_2;

/// Precomputed: 1 / π (computed at compile time)
const fn inv_pi() -> f32 {
    1.0 / PI
}

/// Precomputed: 1 / 2π (computed at compile time)
const fn inv_two_pi() -> f32 {
    1.0 / TWO_PI
}

const PI_INV: f32 = inv_pi();
const TWO_PI_INV: f32 = inv_two_pi();

/// Range reduction: map `x + phase` to `[-π, π]`.
///
/// `x' = x − 2π·round((x + phase)/2π) + phase`, with `2π` split into three
/// pieces so each `k·piece` is exact. A single-f32 `2π` rounds `k·2π` to a
/// multiple of `ulp(x)`, so `x'` leaves `[-π, π]` once `|x|` is large and the
/// polynomial below is then evaluated outside the interval it was fit on —
/// which is how this returned `|sin| > 1` past `x ≈ 1.4e7` and `inf` past
/// `x ≈ 2.6e13`. Valid while `|x| < TRIG_DOMAIN`; callers guard.
///
/// The phase is added *after* reduction: `cos` is `sin(x + π/2)`, and at the
/// top of the domain `ulp(x)` is 0.0625, so shifting `x` up front would lose
/// most of π/2 before reduction ever ran.
#[inline(always)]
fn range_reduce_pi(x: Field, phase: f32) -> Field {
    let arg = if phase == 0.0 {
        x
    } else {
        eval(x + Field::from(phase))
    };
    let k = eval((arg * Field::from(TWO_PI_INV) + Field::from(0.5)).floor());

    let r = eval(x - k * Field::from(TAU_HI));
    let r = eval(r - k * Field::from(TAU_MID));
    let r = eval(r - k * Field::from(TAU_LO));
    if phase == 0.0 {
        r
    } else {
        eval(r + Field::from(phase))
    }
}

/// Chebyshev approximation for `sin(x + phase)`, over `|x| < TRIG_DOMAIN`.
///
/// Shares [`SIN_CHEB`] and the reduction constants with the IR expansion
/// (`expand_sin_phase`), which is the definition of this function that the JIT
/// and the `eval_scalar` oracle both run — see there for why the domain ends
/// where it does and why outside it the answer is NaN rather than a clamp.
#[inline(always)]
fn cheby_sin_phase(x: Field, phase: f32) -> Field {
    let r = range_reduce_pi(x, phase);

    // Normalize to [-1, 1] for the polynomial basis. Reduction error can push
    // |t| to ~1.03 at the domain edge; SIN_CHEB holds |p| ≤ 1 out to |t| = 1.3.
    let t = r * Field::from(PI_INV);
    let t2 = eval(t.clone() * t.clone());

    // Horner from the highest degree down; AST building enables FMA fusion.
    let mut p = Field::from(SIN_CHEB[SIN_CHEB.len() - 1]);
    for &c in SIN_CHEB.iter().rev().skip(1) {
        p = eval(p * t2 + Field::from(c));
    }
    let s = eval(p * t);

    // Outside the domain the reduction can no longer resolve the phase, so
    // there is no answer to give. Guarded on the unshifted argument so sin and
    // cos agree about where the answer stops existing; NaN inputs fail the
    // ordered compare and propagate.
    let in_domain = x.abs().lt(Field::from(TRIG_DOMAIN));
    eval(in_domain.select(s, Field::from(f32::NAN)))
}

/// Chebyshev approximation for sin(x), over `|x| < TRIG_DOMAIN`; NaN outside.
#[inline(always)]
pub(crate) fn cheby_sin(x: Field) -> Field {
    cheby_sin_phase(x, 0.0)
}

/// Chebyshev approximation for cos(x), over `|x| < TRIG_DOMAIN`; NaN outside.
#[inline(always)]
pub(crate) fn cheby_cos(x: Field) -> Field {
    cheby_sin_phase(x, PI_2)
}

/// `tan(x) = sin(x)/cos(x)`, over `|x| < TRIG_DOMAIN`; NaN outside.
///
/// Derived from the two functions above rather than from a separate
/// approximation, so `tan` is exactly the ratio of this tier's own `sin` and
/// `cos` and inherits their domain.
#[inline(always)]
pub(crate) fn cheby_tan(x: Field) -> Field {
    eval(cheby_sin(x) / cheby_cos(x))
}

/// `atan(x) = atan2(x, 1)`.
#[inline(always)]
pub(crate) fn cheby_atan(x: Field) -> Field {
    cheby_atan2(x, Field::from(1.0))
}

/// `√(1 − x²)`, NaN outside `[-1, 1]` rather than 0.
///
/// `Field::sqrt` is `sqrt_fast`, an rsqrt+Newton form that ends in
/// `select(x <= 0, 0, …)` — it has to, because `rsqrt(0)` is ∞ and `0·∞` is
/// NaN, so returning 0 is what makes `sqrt(0)` right. The cost is that it also
/// turns a *negative* radicand into 0, which for `asin`/`acos` would silently
/// convert an out-of-domain argument into `atan2(x, 0)` = ±π/2: a plausible
/// answer where the IR expansion and the JIT (which lower `OpKind::Sqrt` to a
/// real hardware sqrt) both return NaN. Hence the explicit guard, mirroring the
/// one in [`cheby_sin_phase`].
#[inline(always)]
fn sqrt_one_minus_sq(x: Field) -> Field {
    let s = eval((Field::from(1.0) - x * x).sqrt());
    let in_domain = x.abs().le(Field::from(1.0));
    eval(in_domain.select(s, Field::from(f32::NAN)))
}

/// `asin(x) = atan2(x, √(1 − x²))`. NaN outside `[-1, 1]`, where the square
/// root has no real value — the domain error is loud rather than clamped.
#[inline(always)]
pub(crate) fn cheby_asin(x: Field) -> Field {
    cheby_atan2(x, sqrt_one_minus_sq(x))
}

/// `acos(x) = atan2(√(1 − x²), x)`. NaN outside `[-1, 1]`.
#[inline(always)]
pub(crate) fn cheby_acos(x: Field) -> Field {
    cheby_atan2(sqrt_one_minus_sq(x), x)
}

/// Chebyshev approximation for atan2(y, x).
///
/// Computes atan2 using a minimax polynomial approximation of atan on the
/// normalized ratio. Handles all quadrants via arctangent identity and sign
/// corrections. Max absolute error of the underlying atan fit ≈ 8.7e-5.
#[inline(always)]
pub(crate) fn cheby_atan2(y: Field, x: Field) -> Field {
    // Compute the ratio and absolute value for range reduction
    // Eval here because we use r_abs multiple times in different subexpressions
    let r = eval(y / x);
    let r_abs = r.abs();

    // Horner's method for the atan(t) minimax polynomial, t ∈ [0, 1], on the
    // coefficients shared with the IR expansion.
    let atan_poly = |t: Field| -> Field {
        let t2 = eval(t * t);
        let mut p = Field::from(ATAN_MINIMAX[ATAN_MINIMAX.len() - 1]);
        for &c in ATAN_MINIMAX.iter().rev().skip(1) {
            p = eval(p * t2 + Field::from(c));
        }
        eval(p * t)
    };

    // For |r| > 1, use identity: atan(r) = π/2 - atan(1/r). This needs the
    // polynomial evaluated at the *reciprocal*, not `atan(|r|)` rescaled by
    // 1/|r| (which is a different, incorrect quantity).
    let atan_small = atan_poly(r_abs);
    let atan_large = eval(Field::from(PI_2) - atan_poly(eval(Field::from(1.0) / r_abs)));
    let mask_large = r_abs.gt(Field::from(1.0));
    let atan_val = mask_large.select(atan_large, atan_small);

    // Sign of y. `y.abs() / y` would give NaN at y == 0 (0/0); the
    // magnitude-free comparison below is exact and NaN-free there, and
    // atan(0) == 0 makes the sign choice irrelevant at that point anyway.
    let sign_y = y
        .lt(Field::from(0.0))
        .select(Field::from(-1.0), Field::from(1.0));
    let atan_signed = atan_val * sign_y.clone();

    // Apply sign of x (quadrant correction).
    // For x < 0: atan(y/x) = -sign(y)*atan_val (dividing by negative x flips
    // the ratio's sign on top of sign_y), and atan2 adds sign(y)*π on that
    // branch, so the correct combination is `correction - atan_signed`, not
    // `atan_signed - correction`.
    let mask_neg_x = x.lt(Field::from(0.0));
    let correction = Field::from(PI) * sign_y;
    let result = mask_neg_x.select(correction - atan_signed.clone(), atan_signed);

    eval(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate a manifold expression through the public `.sin()`/`.cos()`/
    /// `.atan2()` combinators and extract lane 0 for scalar comparison.
    fn eval_scalar<M: Manifold<Field4, Output = Field>>(m: M) -> f32 {
        let mut buf = [0.0f32; crate::PARALLELISM];
        eval(m).store(&mut buf);
        buf[0]
    }

    /// The approximation's own error budget, per the module's documented
    /// accuracy. Every comparison against an external reference allows this.
    const APPROX_TOL: f32 = 1e-5;

    /// `TWO_PI - 2π`: the phase one winding of the f32 constant loses against
    /// the real period.
    const WINDING_PHASE_ERROR: f32 = 1.748_455_5e-7;

    /// The distance from `x` to the next representable f32 above it.
    fn ulp(x: f32) -> f32 {
        let bits = x.abs().to_bits();
        f32::from_bits(bits + 1) - f32::from_bits(bits)
    }

    /// How far apart `sin`/`cos` of `x` and of `x + windings*TWO_PI` may land
    /// without the range reduction being at fault.
    ///
    /// The shifted angle is not `windings` exact periods from `x`: `TWO_PI` is
    /// an f32, so each winding sheds `WINDING_PHASE_ERROR` of phase, and the
    /// sum then rounds to f32. Both are errors in the *test's* construction of
    /// the argument rather than in the reduction under test, and `|d/dx| ≤ 1`
    /// passes them straight into the value — so the budget carries them
    /// alongside the approximation error each of the two evaluations has.
    fn periodicity_tolerance(shifted: f32, windings: i32) -> f32 {
        let accumulated_phase = (windings.unsigned_abs() as f32) * WINDING_PHASE_ERROR;
        let rounding = ulp(shifted) * 0.5;
        2.0 * APPROX_TOL + accumulated_phase + rounding
    }

    const PRINCIPAL_RANGE_ANGLES: &[f32] = &[
        0.0,
        PI / 6.0,
        PI / 4.0,
        PI / 3.0,
        PI / 2.0,
        2.0,
        -1.5,
        -PI / 2.0,
        PI - 0.1,
        -PI + 0.1,
    ];

    #[test]
    fn sin_matches_std_in_principal_range() {
        for &x in PRINCIPAL_RANGE_ANGLES {
            let got = eval_scalar(Field::from(x).sin());
            let want = x.sin();
            assert!(
                (got - want).abs() < APPROX_TOL,
                "sin({x}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn cos_matches_std_in_principal_range() {
        for &x in PRINCIPAL_RANGE_ANGLES {
            let got = eval_scalar(Field::from(x).cos());
            let want = x.cos();
            assert!(
                (got - want).abs() < APPROX_TOL,
                "cos({x}) = {got}, want {want}"
            );
        }
    }

    // Range reduction must map any angle to an equivalent one in [-π, π]
    // before the Chebyshev approximation runs. These checks exercise the
    // `round(x / 2π)` shift itself (inv_two_pi, the +0.5/floor rounding,
    // and the final `x - k*2π` subtraction) by walking many periods away
    // from a principal-range angle and requiring the same result.
    #[test]
    fn sin_is_periodic_across_many_windings() {
        for &x in &[0.3f32, 1.7, -2.1, 0.0, PI / 2.0] {
            let base = eval_scalar(Field::from(x).sin());
            for &k in &[-37i32, -5, -2, -1, 1, 2, 5, 37] {
                let shifted = x + (k as f32) * TWO_PI;
                let got = eval_scalar(Field::from(shifted).sin());
                let tol = periodicity_tolerance(shifted, k);
                assert!(
                    (got - base).abs() < tol,
                    "sin({x} + {k}*2π) = {got}, want {base} +/- {tol} (periodicity)"
                );
            }
        }
    }

    #[test]
    fn cos_is_periodic_across_many_windings() {
        for &x in &[0.3f32, 1.7, -2.1, 0.0, PI / 2.0] {
            let base = eval_scalar(Field::from(x).cos());
            for &k in &[-37i32, -5, -2, -1, 1, 2, 5, 37] {
                let shifted = x + (k as f32) * TWO_PI;
                let got = eval_scalar(Field::from(shifted).cos());
                let tol = periodicity_tolerance(shifted, k);
                assert!(
                    (got - base).abs() < tol,
                    "cos({x} + {k}*2π) = {got}, want {base} +/- {tol} (periodicity)"
                );
            }
        }
    }

    #[test]
    fn sin_and_cos_match_std_for_large_angles() {
        for &x in &[100.0f32, -250.0, 1000.5, -3.7e3] {
            let got_sin = eval_scalar(Field::from(x).sin());
            let got_cos = eval_scalar(Field::from(x).cos());
            assert!(
                (got_sin - x.sin()).abs() < APPROX_TOL,
                "sin({x}) = {got_sin}, want {}",
                x.sin()
            );
            assert!(
                (got_cos - x.cos()).abs() < APPROX_TOL,
                "cos({x}) = {got_cos}, want {}",
                x.cos()
            );
        }
    }

    #[test]
    fn atan2_axis_aligned_and_quadrant_values() {
        let cases: &[(f32, f32)] = &[
            (0.0, 1.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (-1.0, 0.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
        ];
        for &(y, x) in cases {
            let got = eval_scalar(Field::from(y).atan2(Field::from(x)));
            let want = y.atan2(x);
            assert!(
                (got - want).abs() < 1e-3,
                "atan2({y}, {x}) = {got}, want {want}"
            );
        }
    }

    // |y/x| > 1 triggers the `mask_large` branch (atan(r) = π/2 - atan(1/r)).
    #[test]
    fn atan2_large_ratio_branch_matches_std() {
        let cases: &[(f32, f32)] = &[(2.0, 1.0), (-2.0, 1.0), (2.0, -1.0), (-2.0, -1.0)];
        for &(y, x) in cases {
            let got = eval_scalar(Field::from(y).atan2(Field::from(x)));
            let want = y.atan2(x);
            assert!(
                (got - want).abs() < 1e-3,
                "atan2({y}, {x}) = {got}, want {want}"
            );
        }
    }

    /// The `Field` tier must agree with the IR expansion and the JIT about
    /// *where the answer stops existing*, not just about values in range.
    ///
    /// This is the regression test for a real defect: `Field::sqrt` is
    /// `sqrt_fast`, which selects 0 for a non-positive radicand, so
    /// `asin`/`acos` built naively on it turned an out-of-domain argument into
    /// `atan2(x, 0)` = ±π/2 — a plausible-looking wrong answer where the other
    /// two tiers return NaN. Silent, in-range, and invisible to any test that
    /// only checks `[-π/2, π/2]`.
    #[test]
    fn asin_acos_are_nan_outside_their_domain() {
        for &x in &[1.0001f32, -1.0001, 1.5, 2.0, -2.0, 1e9, -1e9, f32::INFINITY] {
            let s = eval_scalar(Field::from(x).asin());
            let c = eval_scalar(Field::from(x).acos());
            assert!(s.is_nan(), "asin({x}) = {s}, want NaN outside [-1, 1]");
            assert!(c.is_nan(), "acos({x}) = {c}, want NaN outside [-1, 1]");
        }
        for &x in &[1.0f32, -1.0, 0.9999, 0.0, 0.5] {
            let s = eval_scalar(Field::from(x).asin());
            assert!(
                !s.is_nan() && (s - x.asin()).abs() < 1e-3,
                "asin({x}) = {s}, want {}",
                x.asin()
            );
        }
    }

    /// `sin`/`cos` are NaN past `TRIG_DOMAIN` in this tier too, and bounded
    /// everywhere inside it.
    #[test]
    fn sin_cos_respect_the_domain_limit() {
        for &x in &[
            TRIG_DOMAIN,
            TRIG_DOMAIN * 1.5,
            1e7,
            1e30,
            f32::INFINITY,
            f32::NAN,
        ] {
            assert!(eval_scalar(Field::from(x).sin()).is_nan(), "sin({x:e})");
            assert!(eval_scalar(Field::from(x).cos()).is_nan(), "cos({x:e})");
        }
        let mut x = 1.0f32;
        while x < TRIG_DOMAIN {
            let s = eval_scalar(Field::from(x).sin());
            let c = eval_scalar(Field::from(x).cos());
            assert!(s.abs() <= 1.0, "sin({x:e}) = {s} outside [-1, 1]");
            assert!(c.abs() <= 1.0, "cos({x:e}) = {c} outside [-1, 1]");
            assert!(
                (s - x.sin()).abs() < APPROX_TOL,
                "sin({x:e}) = {s}, want {}",
                x.sin()
            );
            x *= 1.37;
        }
    }
}
