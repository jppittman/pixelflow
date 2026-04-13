//! Compound operations built from primitives.
//!
//! These are transcendental functions (sin, cos, exp, log, etc.) implemented
//! as sequences of primitive operations. They work identically across all
//! backends because they only use `Primitives` trait methods.
//!
//! For JIT: these can be "inlined" by emitting the primitive sequence.

use super::primitives::Primitives;
use core::f32::consts::{FRAC_PI_2, PI, TAU};

/// Compound operations derived from primitives.
///
/// This trait has a blanket impl for all `Primitives`, so backends only
/// need to implement `Primitives`.
pub trait Compounds: Primitives {
    // =========================================================================
    // Exponential / Logarithm
    // =========================================================================

    /// Base-2 exponential: 2^x.
    ///
    /// Uses the identity: 2^x = 2^floor(x) * 2^frac(x)
    /// - Integer part: bit manipulation
    /// - Fractional part: polynomial approximation
    #[inline(always)]
    fn exp2(self) -> Self {
        // Clamp to avoid overflow/underflow
        let x = self.max(Self::splat(-126.0)).min(Self::splat(126.0));

        // Split into integer and fractional parts
        let xi = x.floor();
        let xf = x - xi;

        // Polynomial approximation for 2^xf on [0, 1)
        // Coefficients from minimax fit
        let c0 = Self::splat(1.0);
        let c1 = Self::splat(0.6931471805599453); // ln(2)
        let c2 = Self::splat(0.24022650695910071);
        let c3 = Self::splat(0.05550410866482157);
        let c4 = Self::splat(0.009618129107628477);
        let c5 = Self::splat(0.0013333558146428443);

        // Horner's method
        let p = c5.mul_add(xf, c4);
        let p = p.mul_add(xf, c3);
        let p = p.mul_add(xf, c2);
        let p = p.mul_add(xf, c1);
        let p = p.mul_add(xf, c0);

        // Multiply by 2^xi via exponent manipulation
        // 2^xi = reinterpret((xi + 127) << 23) as float
        let bias = Self::splat(127.0);
        let shift = Self::splat(8388608.0); // 2^23
        let exp_bits = (xi + bias) * shift;

        // Combine: p * 2^xi
        // This is a bit hacky - we're using the float multiply to combine
        // In a proper impl, we'd do integer bit manipulation
        p * Self::from_bits(0x3F800000) * exp_bits.shr_bits(0) // TODO: proper bit manip
    }

    /// Base-2 logarithm: log2(x).
    ///
    /// Uses the identity: log2(x) = exponent(x) + log2(mantissa)
    #[inline(always)]
    fn log2(self) -> Self {
        // Extract exponent: floor(log2(x)) via bit manipulation
        let bits = self.shr_bits(0); // reinterpret as bits
        let exp_bits = bits.shr_bits(23);
        let exp = exp_bits.bits_to_f32() - Self::splat(127.0);

        // Extract mantissa and normalize to [1, 2)
        let mantissa_bits = Self::from_bits(0x3F800000); // 1.0
        let mantissa_mask = Self::from_bits(0x007FFFFF);
        // This is simplified - proper impl needs AND/OR bit ops

        // Polynomial for log2(1+x) on [0, 1)
        let x = self * self.recip_approx() - Self::splat(1.0); // normalize to [0, 1)

        let c1 = Self::splat(1.4426950408889634); // 1/ln(2)
        let c2 = Self::splat(-0.7213475204444817);
        let c3 = Self::splat(0.4808983469629878);
        let c4 = Self::splat(-0.3606737602222408);

        let p = c4.mul_add(x, c3);
        let p = p.mul_add(x, c2);
        let p = p.mul_add(x, c1);
        let log_mantissa = p * x;

        exp + log_mantissa
    }

    /// Natural exponential: e^x.
    #[inline(always)]
    fn exp(self) -> Self {
        const LOG2_E: f32 = 1.4426950408889634;
        (self * Self::splat(LOG2_E)).exp2()
    }

    /// Natural logarithm: ln(x).
    #[inline(always)]
    fn ln(self) -> Self {
        const LN_2: f32 = 0.6931471805599453;
        self.log2() * Self::splat(LN_2)
    }

    /// Base-10 logarithm.
    #[inline(always)]
    fn log10(self) -> Self {
        const LOG10_2: f32 = 0.30102999566398120;
        self.log2() * Self::splat(LOG10_2)
    }

    /// Power: self^exp.
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        (exp * self.log2()).exp2()
    }

    // =========================================================================
    // Trigonometric
    // =========================================================================

    /// Sine using Chebyshev polynomial.
    #[inline(always)]
    fn sin(self) -> Self {
        // Range reduction to [-π, π]
        let two_pi_inv = Self::splat(1.0 / TAU);
        let k = (self * two_pi_inv + Self::splat(0.5)).floor();
        let x = self - k * Self::splat(TAU);

        // Normalize to [-1, 1] by dividing by π
        let t = x * Self::splat(1.0 / PI);

        // Chebyshev polynomial for sin(π*t)
        // sin(π*t) ≈ π*t * (1 - t²*(c3 + t²*(c5 + t²*c7)))
        let t2 = t * t;

        let c1 = Self::splat(3.14159265358979); // π
        let c3 = Self::splat(-5.16771278004997);
        let c5 = Self::splat(2.55016403987734);
        let c7 = Self::splat(-0.599264528932149);

        let p = c7.mul_add(t2, c5);
        let p = p.mul_add(t2, c3);
        let p = p.mul_add(t2, c1);

        t * p
    }

    /// Cosine using Chebyshev polynomial.
    #[inline(always)]
    fn cos(self) -> Self {
        // cos(x) = sin(x + π/2)
        (self + Self::splat(FRAC_PI_2)).sin()
    }

    /// Tangent.
    #[inline(always)]
    fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    /// Four-quadrant arctangent: atan2(y, x).
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        let y = self;
        let pi = Self::splat(PI);
        let half_pi = Self::splat(FRAC_PI_2);

        // Handle quadrants
        let abs_x = x.abs();
        let abs_y = y.abs();

        // Compute atan(y/x) or atan(x/y) depending on which is smaller
        let swap = abs_y.cmp_gt(abs_x);
        let ratio = Self::select(swap, x * y.recip_approx(), y * x.recip_approx());

        // Polynomial for atan on [-1, 1]
        let r2 = ratio * ratio;
        let c1 = Self::splat(1.0);
        let c3 = Self::splat(-0.333333333);
        let c5 = Self::splat(0.2);
        let c7 = Self::splat(-0.142857142);

        let p = c7.mul_add(r2, c5);
        let p = p.mul_add(r2, c3);
        let p = p.mul_add(r2, c1);
        let atan_small = ratio * p;

        // Adjust for swap: if swapped, result is π/2 - atan
        let atan_val = Self::select(
            swap,
            Self::select(ratio.cmp_ge(Self::splat(0.0)), half_pi, -half_pi) - atan_small,
            atan_small,
        );

        // Adjust for quadrant
        let x_neg = x.cmp_lt(Self::splat(0.0));
        let y_neg = y.cmp_lt(Self::splat(0.0));

        let adjust = Self::select(y_neg, -pi, pi);
        Self::select(x_neg, atan_val + adjust, atan_val)
    }

    /// Arctangent.
    #[inline(always)]
    fn atan(self) -> Self {
        self.atan2(Self::splat(1.0))
    }

    /// Arcsine.
    #[inline(always)]
    fn asin(self) -> Self {
        let one = Self::splat(1.0);
        let x2 = self * self;
        let sqrt_term = (one - x2).sqrt();
        self.atan2(sqrt_term)
    }

    /// Arccosine.
    #[inline(always)]
    fn acos(self) -> Self {
        let one = Self::splat(1.0);
        let x2 = self * self;
        let sqrt_term = (one - x2).sqrt();
        sqrt_term.atan2(self)
    }

    // =========================================================================
    // Derived convenience functions
    // =========================================================================

    /// Ceiling.
    #[inline(always)]
    fn ceil(self) -> Self {
        -(-self).floor()
    }

    /// Round to nearest.
    #[inline(always)]
    fn round(self) -> Self {
        (self + Self::splat(0.5)).floor()
    }

    /// Fractional part.
    #[inline(always)]
    fn fract(self) -> Self {
        self - self.floor()
    }

    /// Clamp to range.
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Hypotenuse: sqrt(x² + y²).
    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        (self * self + y * y).sqrt()
    }

    /// Multiply by reciprocal sqrt: self / sqrt(other).
    #[inline(always)]
    fn mul_rsqrt(self, other: Self) -> Self {
        self * other.rsqrt_approx()
    }
}

// Blanket impl: anything that implements Primitives gets Compounds for free
impl<T: Primitives> Compounds for T {}
