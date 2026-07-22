//! # Chebyshev-Based Trigonometric Functions
//!
//! SIMD-vectorized sin, cos, atan2 using Chebyshev polynomial approximations.
//! All operations vectorize across all SIMD lanes simultaneously, replacing
//! per-lane libm scalar calls with parallel polynomial evaluation.
//!
//! **Accuracy**: max absolute error ≈ 2.6e-4 (sin), 1.4e-3 (cos), 8.7e-5
//! (atan2) — a 7-term minimax polynomial fit, not full float32 precision.
//! Sufficient for graphics/harmonics but not for exact angle round-tripping.
//! **Speed**: 5-10x faster than per-lane libm on SIMD backends.
//!
//! # Implementation Note
//!
//! This module builds AST graphs using operators, enabling automatic FMA fusion
//! and other optimizations. The graph is evaluated at the return boundary.

use crate::Field;
use crate::{Manifold, ManifoldExt};

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

/// Range reduction: Map angle x to [-π, π].
///
/// Uses division and modulo on the SIMD vector.
/// Formula: x' = x - 2π * round(x / 2π)
#[inline(always)]
fn range_reduce_pi(x: Field) -> Field {
    // Compute k = round(x / 2π) using floor(x + 0.5)
    // The 0.5 must be added BEFORE floor, not after
    let k = eval((x * Field::from(TWO_PI_INV) + Field::from(0.5)).floor());

    // x' = x - 2π * k
    eval(x - k * Field::from(TWO_PI))
}

/// Chebyshev approximation for sin(x) on [-π, π].
///
/// Uses 7-term odd polynomial with Horner's method, minimax-fit against
/// `sin(π·t)` for `t = x/π ∈ [-1, 1]`. Max absolute error ≈ 2.6e-4.
#[inline(always)]
pub(crate) fn cheby_sin(x: Field) -> Field {
    let x = range_reduce_pi(x);

    // Normalize to [-1, 1] for the minimax polynomial basis
    let t = x * Field::from(PI_INV);

    // Minimax coefficients for sin(π·t) on t ∈ [-1,1]
    const C1: f32 = 3.139_275_7_f32;
    const C3: f32 = -5.136_387_f32;
    const C5: f32 = 2.434_668_3_f32;
    const C7: f32 = -0.437_801_8_f32;

    // Horner's method: accumulate from highest degree down
    // p(t) = C1*t + C3*t^3 + C5*t^5 + C7*t^7
    // Rewrite as: ((C7*t^2 + C5)*t^2 + C3)*t^2 + C1)*t
    // AST building enables FMA fusion
    let t2 = t.clone() * t.clone();
    let result =
        (((Field::from(C7) * t2.clone() + Field::from(C5)) * t2.clone() + Field::from(C3)) * t2
            + Field::from(C1))
            * t;

    eval(result)
}

/// Chebyshev approximation for cos(x) on [-π, π].
///
/// Uses 7-term even polynomial with Horner's method, minimax-fit against
/// `cos(π·t)` for `t = x/π ∈ [-1, 1]`. Max absolute error ≈ 1.4e-3.
#[inline(always)]
pub(crate) fn cheby_cos(x: Field) -> Field {
    let x = range_reduce_pi(x);

    // Normalize to [-1, 1] for the minimax polynomial basis
    let t = x * Field::from(PI_INV);

    // Minimax coefficients for cos(π·t) on t ∈ [-1,1]
    const C0: f32 = 0.998_564_9_f32;
    const C2: f32 = -4.888_198_6_f32;
    const C4: f32 = 3.819_262_7_f32;
    const C6: f32 = -0.930_980_2_f32;

    // Horner's method for even polynomial
    // p(t) = C0 + C2*t^2 + C4*t^4 + C6*t^6
    // Rewrite as: ((C6*t^2 + C4)*t^2 + C2)*t^2 + C0
    // AST building enables FMA fusion
    let t2 = t.clone() * t;
    let result = ((Field::from(C6) * t2.clone() + Field::from(C4)) * t2.clone() + Field::from(C2))
        * t2
        + Field::from(C0);

    eval(result)
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

    // Minimax coefficients for atan(t) on t ∈ [0, 1]
    const C1: f32 = 0.999_268_04_f32;
    const C3: f32 = -0.321_431_33_f32;
    const C5: f32 = 0.146_614_41_f32;
    const C7: f32 = -0.039_132_48_f32;

    // Horner's method for the atan(t) minimax polynomial, t ∈ [0, 1].
    let atan_poly = |t: Field| -> Field {
        let t2 = t * t;
        eval(
            (((Field::from(C7) * t2.clone() + Field::from(C5)) * t2.clone() + Field::from(C3))
                * t2
                + Field::from(C1))
                * t,
        )
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
                (got - want).abs() < 1e-3,
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
                (got - want).abs() < 2e-3,
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
                assert!(
                    (got - base).abs() < 1e-2,
                    "sin({x} + {k}*2π) = {got}, want {base} (periodicity)"
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
                assert!(
                    (got - base).abs() < 1e-2,
                    "cos({x} + {k}*2π) = {got}, want {base} (periodicity)"
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
                (got_sin - x.sin()).abs() < 5e-3,
                "sin({x}) = {got_sin}, want {}",
                x.sin()
            );
            assert!(
                (got_cos - x.cos()).abs() < 3e-3,
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
}
