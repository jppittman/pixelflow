//! Primitive SIMD operations - 1:1 (or near 1:1) with hardware instructions.
//!
//! These are the atomic building blocks. Each method maps to a single instruction
//! or a fixed small sequence (e.g., rsqrt estimate + Newton-Raphson refinement).
//!
//! Compound operations (sin, cos, exp, log) are built from these primitives
//! in the `compounds` module.

use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Sub};

/// Mask operations - predicates for SIMD lanes.
pub trait MaskPrimitives:
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
    /// Any lane true?
    fn any(self) -> bool;

    /// All lanes true?
    fn all(self) -> bool;
}

/// Primitive SIMD operations for f32 vectors.
///
/// These map directly to hardware instructions. Each method is either:
/// - A single instruction (add, mul, sqrt, min, max, floor, etc.)
/// - A fixed 2-4 instruction sequence (rsqrt with Newton-Raphson, recip with refinement)
///
/// Compound operations (transcendentals) are NOT here - see `Compounds` trait.
pub trait Primitives:
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
    type Mask: MaskPrimitives;

    /// Number of lanes.
    const LANES: usize;

    // =========================================================================
    // Broadcast / Load / Store
    // =========================================================================

    /// Splat scalar to all lanes.
    /// ARM: `vdupq_n_f32`, x86: `_mm_set1_ps`
    fn splat(val: f32) -> Self;

    /// Sequential values [start, start+1, ...].
    fn sequential(start: f32) -> Self;

    /// Load from slice.
    /// ARM: `vld1q_f32`, x86: `_mm_loadu_ps`
    fn load(slice: &[f32]) -> Self;

    /// Store to slice.
    /// ARM: `vst1q_f32`, x86: `_mm_storeu_ps`
    fn store(&self, out: &mut [f32]);

    // =========================================================================
    // Arithmetic (single instruction each)
    // =========================================================================

    /// Square root.
    /// ARM: `vsqrtq_f32`, x86: `_mm_sqrt_ps`
    fn sqrt(self) -> Self;

    /// Absolute value.
    /// ARM: `vabsq_f32`, x86: AND with sign mask
    fn abs(self) -> Self;

    /// Element-wise minimum.
    /// ARM: `vminq_f32`, x86: `_mm_min_ps`
    fn min(self, rhs: Self) -> Self;

    /// Element-wise maximum.
    /// ARM: `vmaxq_f32`, x86: `_mm_max_ps`
    fn max(self, rhs: Self) -> Self;

    /// Floor (round toward -∞).
    /// ARM: `vrndmq_f32`, x86: `_mm_floor_ps` (SSE4.1)
    fn floor(self) -> Self;

    /// Fused multiply-add: self * b + c.
    /// ARM: `vfmaq_f32`, x86: `_mm_fmadd_ps` (FMA3)
    fn mul_add(self, b: Self, c: Self) -> Self;

    // =========================================================================
    // Approximate operations (estimate + optional refinement)
    // =========================================================================

    /// Approximate reciprocal (1/x).
    /// ARM: `vrecpeq_f32` + `vrecpsq_f32` refinement (~2-3 instructions)
    /// x86: `_mm_rcp_ps` + optional Newton-Raphson
    fn recip_approx(self) -> Self;

    /// Approximate reciprocal square root (1/sqrt(x)).
    /// ARM: `vrsqrteq_f32` + `vrsqrtsq_f32` refinement (~3-4 instructions)
    /// x86: `_mm_rsqrt_ps` + optional Newton-Raphson
    fn rsqrt_approx(self) -> Self;

    // =========================================================================
    // Comparisons (return native mask)
    // =========================================================================

    /// Less than.
    fn cmp_lt(self, rhs: Self) -> Self::Mask;

    /// Less than or equal.
    fn cmp_le(self, rhs: Self) -> Self::Mask;

    /// Greater than.
    fn cmp_gt(self, rhs: Self) -> Self::Mask;

    /// Greater than or equal.
    fn cmp_ge(self, rhs: Self) -> Self::Mask;

    /// Equal.
    fn cmp_eq(self, rhs: Self) -> Self::Mask;

    /// Not equal.
    fn cmp_ne(self, rhs: Self) -> Self::Mask;

    // =========================================================================
    // Selection / Blending
    // =========================================================================

    /// Conditional select: mask ? if_true : if_false.
    /// ARM: `vbslq_f32`, x86: `_mm_blendv_ps` (SSE4.1)
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    // =========================================================================
    // Bit manipulation (for transcendental implementations)
    // =========================================================================

    /// Splat u32 bit pattern as float (bitcast, not conversion).
    fn from_bits(bits: u32) -> Self;

    /// Reinterpret as u32, shift right, reinterpret back.
    fn shr_bits(self, n: u32) -> Self;

    /// Reinterpret bits as i32, convert to f32.
    fn bits_to_f32(self) -> Self;

    // =========================================================================
    // Mask conversion
    // =========================================================================

    /// Convert mask to float representation (all-1s or all-0s per lane).
    fn mask_to_float(mask: Self::Mask) -> Self;

    /// Convert float to mask (non-zero → true).
    fn float_to_mask(self) -> Self::Mask;

    // =========================================================================
    // Gather (may be emulated on some platforms)
    // =========================================================================

    /// Gather: load from slice at indices.
    /// Native on AVX2+, emulated on NEON/SSE.
    fn gather(slice: &[f32], indices: Self) -> Self;
}
