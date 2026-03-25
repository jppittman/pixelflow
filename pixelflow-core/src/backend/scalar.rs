//! Scalar fallback backend for non-SIMD platforms.

use super::{Backend, MaskOps, SimdOps, SimdU32Ops};
use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Shl, Shr, Sub};

/// Scalar fallback backend (1 lane - no SIMD).
#[derive(Copy, Clone, Debug, Default)]
pub struct Scalar;

impl Backend for Scalar {
    const LANES: usize = 1;
    type F32 = ScalarF32;
    type U32 = ScalarU32;
}

// ============================================================================
// MaskScalar - 1-lane mask for scalar backend
// ============================================================================

/// Scalar mask (1-lane, just a bool).
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct MaskScalar(bool);

impl MaskOps for MaskScalar {
    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0
    }
}

impl BitAnd for MaskScalar {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 && rhs.0)
    }
}

impl BitOr for MaskScalar {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 || rhs.0)
    }
}

impl Not for MaskScalar {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

/// Scalar f32 wrapper that implements all required ops.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct ScalarF32(f32);

// ============================================================================
// SimdOps for ScalarF32
// ============================================================================

impl SimdOps for ScalarF32 {
    type Mask = MaskScalar;
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(val)
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self(start)
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        out[0] = self.0;
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> MaskScalar {
        MaskScalar(self.0 < rhs.0)
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> MaskScalar {
        MaskScalar(self.0 <= rhs.0)
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> MaskScalar {
        MaskScalar(self.0 > rhs.0)
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> MaskScalar {
        MaskScalar(self.0 >= rhs.0)
    }

    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        Self(libm::sqrtf(self.0))
    }

    #[inline(always)]
    fn simd_abs(self) -> Self {
        Self(libm::fabsf(self.0))
    }

    #[inline(always)]
    fn simd_min(self, rhs: Self) -> Self {
        Self(if self.0 < rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline(always)]
    fn simd_max(self, rhs: Self) -> Self {
        Self(if self.0 > rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline(always)]
    fn simd_select(mask: MaskScalar, if_true: Self, if_false: Self) -> Self {
        Self(if mask.0 { if_true.0 } else { if_false.0 })
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        Self(slice[0])
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        let idx = (libm::floorf(indices.0) as isize).clamp(0, slice.len() as isize - 1) as usize;
        Self(slice[idx])
    }

    #[inline(always)]
    fn simd_floor(self) -> Self {
        Self(libm::floorf(self.0))
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // Use libm's fmaf for correct single-rounding FMA
        Self(libm::fmaf(self.0, b.0, c.0))
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: MaskScalar) -> Self {
        Self(if mask.0 { self.0 + val.0 } else { self.0 })
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self(1.0 / self.0)
    }

    #[inline(always)]
    fn simd_rsqrt(self) -> Self {
        Self(1.0 / libm::sqrtf(self.0))
    }

    #[inline(always)]
    fn mask_to_float(mask: MaskScalar) -> Self {
        // Convert bool mask to float representation
        Self(if mask.0 { f32::from_bits(!0u32) } else { 0.0 })
    }

    #[inline(always)]
    fn float_to_mask(self) -> MaskScalar {
        // Convert float representation to bool mask
        MaskScalar(self.0.to_bits() != 0)
    }

    #[inline(always)]
    fn from_u32_bits(bits: u32) -> Self {
        Self(f32::from_bits(bits))
    }

    #[inline(always)]
    fn shr_u32(self, n: u32) -> Self {
        Self(f32::from_bits(self.0.to_bits() >> n))
    }

    #[inline(always)]
    fn i32_to_f32(self) -> Self {
        Self((self.0.to_bits() as i32) as f32)
    }

    #[inline(always)]
    fn log2(self) -> Self {
        Self(libm::log2f(self.0))
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        Self(libm::exp2f(self.0))
    }
}

// ============================================================================
// Operator Implementations for ScalarF32
// ============================================================================

impl Add for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl BitAnd for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(f32::from_bits(self.0.to_bits() & rhs.0.to_bits()))
    }
}

impl BitOr for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(f32::from_bits(self.0.to_bits() | rhs.0.to_bits()))
    }
}

impl Not for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(f32::from_bits(!self.0.to_bits()))
    }
}

impl core::ops::Neg for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

// ============================================================================
// ScalarU32 - Scalar u32 for Discrete fallback
// ============================================================================

/// Scalar u32 wrapper for packed RGBA pixel.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct ScalarU32(u32);

impl SimdU32Ops for ScalarU32 {
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        Self(val)
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        out[0] = self.0;
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        Self::default()
    }
}

impl BitAnd for ScalarU32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for ScalarU32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl Not for ScalarU32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl Shl<u32> for ScalarU32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        Self(self.0 << rhs)
    }
}

impl Shr<u32> for ScalarU32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        Self(self.0 >> rhs)
    }
}

impl ScalarU32 {
    /// Pack 4 f32 values (RGBA in 0-1) into a packed u32 pixel.
    #[inline(always)]
    pub fn pack_rgba(r: ScalarF32, g: ScalarF32, b: ScalarF32, a: ScalarF32) -> Self {
        let r_u8 = (r.0.clamp(0.0, 1.0) * 255.0) as u32;
        let g_u8 = (g.0.clamp(0.0, 1.0) * 255.0) as u32;
        let b_u8 = (b.0.clamp(0.0, 1.0) * 255.0) as u32;
        let a_u8 = (a.0.clamp(0.0, 1.0) * 255.0) as u32;
        Self(r_u8 | (g_u8 << 8) | (b_u8 << 16) | (a_u8 << 24))
    }
}
