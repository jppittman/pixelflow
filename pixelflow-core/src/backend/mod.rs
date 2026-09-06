//! The SIMD backend: what a `Field` *is*.
//!
//! `SimdOps` is the one trait here. Its **required** methods are the per-ISA
//! primitives — each backend supplies its own instruction — and its
//! **provided** methods are the uniform expansions built on them. That
//! required-vs-provided split is this tier's legal/lowered partition, and it is
//! deliberately the only place it is written down for this tier.
//!
//! This lived in `pixelflow-ir` until 2026-08-02, on loan through a blanket
//! re-export. It is not IR and not codegen — it is the representation `Field`
//! is built on, so it belongs beside `Field`. The JIT never calls it: compiled
//! kernels emit their own instructions. It backs the combinator evaluation path
//! that `kernel!` still produces by default, and it is the layer the AOT work
//! intends to eventually *generate* from kernel source rather than hand-write.

use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Shl, Shr, Sub};

/// A backend provides the SIMD implementation for a specific platform.
pub trait Backend: 'static + Copy + Clone + Send + Sync + Debug {
    /// Number of lanes in the SIMD vector.
    const LANES: usize;

    /// The SIMD vector type for f32.
    type F32: SimdOps;

    /// The SIMD vector type for u32 (for packed pixels).
    type U32: SimdU32Ops;
}

/// Operations on native mask types.
pub trait MaskOps:
    Copy
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Not<Output = Self>
{
    /// Check if any lane is true (non-zero).
    fn any(self) -> bool;

    /// Check if all lanes are true (non-zero).
    fn all(self) -> bool;
}

/// All SIMD operations for f32.
pub trait SimdOps:
    Copy
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Not<Output = Self>
{
    /// Native mask type for this SIMD width.
    type Mask: MaskOps;

    /// Number of lanes.
    const LANES: usize;

    /// Splat a scalar across all lanes.
    fn splat(val: f32) -> Self;

    /// Create sequential values [start, start+1, ...].
    fn sequential(start: f32) -> Self;

    /// Store to a slice.
    fn store(&self, out: &mut [f32]);

    /// Less than comparison (returns native mask).
    fn cmp_lt(self, rhs: Self) -> Self::Mask;
    /// Less than or equal comparison (returns native mask).
    fn cmp_le(self, rhs: Self) -> Self::Mask;
    /// Greater than comparison (returns native mask).
    fn cmp_gt(self, rhs: Self) -> Self::Mask;
    /// Greater than or equal comparison (returns native mask).
    fn cmp_ge(self, rhs: Self) -> Self::Mask;

    /// Square root.
    fn simd_sqrt(self) -> Self;
    /// Absolute value.
    fn simd_abs(self) -> Self;
    /// Element-wise minimum.
    fn simd_min(self, rhs: Self) -> Self;
    /// Element-wise maximum.
    fn simd_max(self, rhs: Self) -> Self;

    /// Conditional select using native mask.
    fn simd_select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    /// Load from a slice.
    fn from_slice(slice: &[f32]) -> Self;

    /// Gather: load from slice at indices specified by self.
    fn gather(slice: &[f32], indices: Self) -> Self;

    /// Floor (round toward negative infinity).
    fn simd_floor(self) -> Self;

    /// Fused multiply-add: (self * b) + c
    fn mul_add(self, b: Self, c: Self) -> Self;

    /// Masked add using native mask: self + (mask ? val : 0)
    fn add_masked(self, val: Self, mask: Self::Mask) -> Self;

    /// Approximate reciprocal (1/x).
    fn recip(self) -> Self;

    /// Approximate reciprocal square root (1/sqrt(x)).
    fn simd_rsqrt(self) -> Self;

    // =========================================================================
    // Mask Conversion
    // =========================================================================

    /// Convert native mask to float representation.
    fn mask_to_float(mask: Self::Mask) -> Self;

    /// Convert float representation to native mask.
    fn float_to_mask(self) -> Self::Mask;

    // =========================================================================
    // Bit Manipulation (for transcendentals)
    // =========================================================================

    /// Splat u32 bit pattern as float (BITCAST).
    fn from_u32_bits(bits: u32) -> Self;

    /// Shift bits right treating as u32.
    fn shr_u32(self, n: u32) -> Self;

    /// Interpret bits as i32, convert to f32.
    fn i32_to_f32(self) -> Self;

    /// Base-2 logarithm.
    fn log2(self) -> Self;

    /// Base-2 exponential.
    fn exp2(self) -> Self;

    /// Natural exponential.
    #[inline(always)]
    fn exp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        (self * Self::splat(LOG2_E)).exp2()
    }

    // Trigonometry is deliberately absent: it lives in `pixelflow-ir`'s
    // `passes::expand_sin_phase`, once, and `ops::trig` is the combinator-tier
    // spelling of that same function. A per-width intrinsics copy here would be
    // a second definition with its own range reduction, free to drift from the
    // one every backend and the `eval_scalar` oracle actually lower through.

    // =========================================================================
    // Additional Logarithms
    // =========================================================================

    /// Natural logarithm.
    /// ln(x) = log2(x) * ln(2)
    #[inline(always)]
    fn ln(self) -> Self {
        const LN_2: f32 = core::f32::consts::LN_2;
        self.log2() * Self::splat(LN_2)
    }

    /// Base-10 logarithm.
    /// log10(x) = log2(x) * log10(2)
    #[inline(always)]
    fn log10(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        self.log2() * Self::splat(LOG10_2)
    }

    // =========================================================================
    // Power Functions
    // =========================================================================

    /// Power: self^exp.
    /// Uses identity: x^y = 2^(y * log2(x)) for positive x.
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        (exp * self.log2()).exp2()
    }

    /// Hypotenuse: sqrt(x² + y²).
    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        (self * self + y * y).simd_sqrt()
    }

    /// Multiply by reciprocal square root: self * rsqrt(other) = self / sqrt(other).
    /// This is more efficient than dividing by sqrt on most SIMD architectures.
    #[inline(always)]
    fn mul_rsqrt(self, other: Self) -> Self {
        self * other.simd_rsqrt()
    }

    // =========================================================================
    // Additional Rounding Functions
    // =========================================================================

    /// Ceiling (round toward positive infinity).
    #[inline(always)]
    fn ceil(self) -> Self {
        -(-self).simd_floor()
    }

    /// Round to nearest integer.
    #[inline(always)]
    fn round(self) -> Self {
        (self + Self::splat(0.5)).simd_floor()
    }

    /// Fractional part: x - floor(x).
    #[inline(always)]
    fn fract(self) -> Self {
        self - self.simd_floor()
    }

    // =========================================================================
    // Additional Comparisons
    // =========================================================================

    /// Equality comparison (returns native mask).
    fn cmp_eq(self, rhs: Self) -> Self::Mask;

    /// Inequality comparison (returns native mask).
    fn cmp_ne(self, rhs: Self) -> Self::Mask;

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// Clamp value to range [lo, hi].
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }
}

/// SIMD operations for u32 (packed pixels).
pub trait SimdU32Ops:
    Copy
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
{
    /// Number of lanes.
    const LANES: usize;

    /// Splat a scalar across all lanes.
    fn splat(val: u32) -> Self;

    /// Store to a slice.
    fn store(&self, out: &mut [u32]);

    /// Convert from f32 SIMD (clamp, scale by 255, truncate).
    fn from_f32_scaled<F: SimdOps>(f: F) -> Self;
}

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub mod fastmath;
