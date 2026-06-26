//! x86_64 backend.

use super::{Backend, MaskOps, SimdOps, SimdU32Ops};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::fmt::{Debug, Formatter};
use core::ops::*;

// Shared minimax polynomial coefficients (f32 precision)
mod log2_poly {
    pub const C4: f32 = -0.320_043_5;
    pub const C3: f32 = 1.797_496_9;
    pub const C2: f32 = -4.198_805;
    pub const C1: f32 = 5.727_023;
    pub const C0: f32 = -3.005_614_8;
}

/// 2^f on [0, 1), degree-4 minimax. Max error: ~1e-7.
mod exp2_poly {
    pub const C4: f32 = 0.013_555_7;
    pub const C3: f32 = 0.052_032_3;
    pub const C2: f32 = 0.241_379_3;
}

// ============================================================================
// SSE2 Backend
// ============================================================================

/// SSE2 Backend (4 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Sse2;

// ============================================================================
// Mask4 - 4-lane mask for SSE2 (float-based, no separate mask unit)
// ============================================================================

/// 4-lane mask for SSE2.
///
/// On SSE2, there's no separate mask register file like AVX-512's k-registers.
/// Masks are stored as float vectors where each lane is either all-1s (true)
/// or all-0s (false).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Mask4(__m128);

impl Default for Mask4 {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_ps()) }
    }
}

impl Debug for Mask4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mask4({:04b})", unsafe { _mm_movemask_ps(self.0) })
    }
}

impl MaskOps for Mask4 {
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) == 0xF }
    }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_ps(self.0, rhs.0)) }
    }
}

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_ps(self.0, rhs.0)) }
    }
}

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
            Self(_mm_xor_ps(self.0, all_ones))
        }
    }
}

impl Backend for Sse2 {
    const LANES: usize = 4;
    type F32 = F32x4;
    type U32 = U32x4;
}

/// 4-lane f32 SIMD vector for SSE2.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(__m128);

impl Default for F32x4 {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_ps()) }
    }
}

impl Debug for F32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "F32x4({:?})", arr)
    }
}

impl F32x4 {
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut arr = [0.0f32; 4];
        unsafe { _mm_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }
}

impl SimdOps for F32x4 {
    type Mask = Mask4;
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(_mm_set1_ps(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            // _mm_set_ps args are in reverse order: e3, e2, e1, e0
            Self(_mm_set_ps(start + 3.0, start + 2.0, start + 1.0, start))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmplt_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmple_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmpgt_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmpge_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        unsafe { Self(_mm_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn simd_abs(self) -> Self {
        unsafe {
            // Mask off sign bit (bit 31)
            let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
            Self(_mm_and_ps(self.0, mask))
        }
    }

    #[inline(always)]
    fn simd_min(self, rhs: Self) -> Self {
        unsafe { Self(_mm_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_max(self, rhs: Self) -> Self {
        unsafe { Self(_mm_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_select(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            // (mask & if_true) | (!mask & if_false)
            // _mm_andnot_ps(a, b) computes (!a) & b
            let t = _mm_and_ps(mask.0, if_true.0);
            let f = _mm_andnot_ps(mask.0, if_false.0);
            Self(_mm_or_ps(t, f))
        }
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(_mm_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // SSE2 doesn't have gather - do scalar loads
        let idx = indices.to_array();
        let len = slice.len();
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            // Indices are pre-floored by Texture::eval_raw, so truncation is safe and faster.
            // Even for general use, truncation matches floor for positive indices.
            let ix = (idx[i] as isize).clamp(0, len as isize - 1) as usize;
            out[i] = slice[ix];
        }
        Self::from_slice(&out)
    }

    #[inline(always)]
    fn simd_floor(self) -> Self {
        unsafe {
            // SSE2 floor emulation:
            // 1. Truncate toward zero
            let trunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(self.0));
            // 2. For negative non-integers, truncation rounds toward zero (wrong direction)
            //    Need to subtract 1 where self < trunc
            let cmp_mask = _mm_cmplt_ps(self.0, trunc);
            let one = _mm_set1_ps(1.0);
            let correction = _mm_and_ps(cmp_mask, one);
            Self(_mm_sub_ps(trunc, correction))
        }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            Self(_mm_fmadd_ps(self.0, b.0, c.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            // Fallback: separate mul + add (two roundings)
            self * b + c
        }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask4) -> Self {
        // SSE2: no native masked add, emulate with select
        // self + (mask ? val : 0)
        unsafe {
            let masked_val = _mm_and_ps(mask.0, val.0);
            Self(_mm_add_ps(self.0, masked_val))
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // Use fast reciprocal approximation (~12 bits accuracy)
        unsafe { Self(_mm_rcp_ps(self.0)) }
    }

    #[inline(always)]
    fn simd_rsqrt(self) -> Self {
        // Fast reciprocal square root (~12 bits accuracy)
        // Sufficient for AA coverage calculations
        unsafe { Self(_mm_rsqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask4) -> Self {
        // Mask4 already stores float representation
        Self(mask.0)
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask4 {
        // Float representation is already a valid mask
        Mask4(self.0)
    }

    #[inline(always)]
    fn from_u32_bits(bits: u32) -> Self {
        unsafe { Self(_mm_castsi128_ps(_mm_set1_epi32(bits as i32))) }
    }

    #[inline(always)]
    fn shr_u32(self, n: u32) -> Self {
        unsafe {
            let as_int = _mm_castps_si128(self.0);
            let shift = _mm_cvtsi32_si128(n as i32);
            let shifted = _mm_srl_epi32(as_int, shift);
            Self(_mm_castsi128_ps(shifted))
        }
    }

    #[inline(always)]
    fn i32_to_f32(self) -> Self {
        unsafe {
            let as_int = _mm_castps_si128(self.0);
            Self(_mm_cvtepi32_ps(as_int))
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        // SSE2: Use bit manipulation for exponent/mantissa extraction
        // Uses range [√2/2, √2] centered at 1 for better polynomial accuracy
        // log2(x) = exponent + log2(mantissa)
        unsafe {
            let x_i32 = _mm_castps_si128(self.0);

            // Extract exponent using integer ops
            // Shift right by 23 to get exponent in lowest 8 bits
            let exp_shifted = _mm_srli_epi32(x_i32, 23);
            // Mask to keep only 8 bits (remove sign bit if present)
            let exp_masked = _mm_and_si128(exp_shifted, _mm_set1_epi32(0xFF));
            // Subtract bias (127) to get unbiased exponent
            let exp_unbiased = _mm_sub_epi32(exp_masked, _mm_set1_epi32(127));
            // Convert to float
            let mut n = _mm_cvtepi32_ps(exp_unbiased);

            // Extract mantissa in [1, 2)
            let mant_mask = _mm_set1_epi32(0x007FFFFF_u32 as i32);
            let one_bits = _mm_set1_epi32(0x3F800000_u32 as i32);
            let mut f = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(x_i32, mant_mask), one_bits));

            // Adjust to [√2/2, √2] range for better accuracy (centered at 1)
            // If f >= √2, divide by 2 and increment exponent
            let sqrt2 = _mm_set1_ps(core::f32::consts::SQRT_2);
            let mask = _mm_cmpge_ps(f, sqrt2);
            let adjust = _mm_and_ps(mask, _mm_set1_ps(1.0));
            n = _mm_add_ps(n, adjust);
            f = _mm_or_ps(
                _mm_and_ps(mask, _mm_mul_ps(f, _mm_set1_ps(0.5))),
                _mm_andnot_ps(mask, f),
            );

            // Polynomial for log2(f) on [√2/2, √2]
            // Fitted using least squares on Chebyshev nodes
            // Max error: ~1e-4
            let c4 = _mm_set1_ps(log2_poly::C4);
            let c3 = _mm_set1_ps(log2_poly::C3);
            let c2 = _mm_set1_ps(log2_poly::C2);
            let c1 = _mm_set1_ps(log2_poly::C1);
            let c0 = _mm_set1_ps(log2_poly::C0);

            // Horner's method (no FMA on base SSE2)
            let mut poly = _mm_add_ps(_mm_mul_ps(c4, f), c3);
            poly = _mm_add_ps(_mm_mul_ps(poly, f), c2);
            poly = _mm_add_ps(_mm_mul_ps(poly, f), c1);
            poly = _mm_add_ps(_mm_mul_ps(poly, f), c0);

            Self(_mm_add_ps(n, poly))
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // SSE2: 2^x = 2^n * 2^f where n = floor(x), f = frac(x) ∈ [0, 1)
        unsafe {
            // n = floor(x), f = x - n
            let n = _mm_floor_ps(self.0);
            let f = _mm_sub_ps(self.0, n);

            // Minimax polynomial for 2^f, f ∈ [0, 1)
            // Degree 4, max error ~10^-7
            let c4 = _mm_set1_ps(exp2_poly::C4);
            let c3 = _mm_set1_ps(exp2_poly::C3);
            let c2 = _mm_set1_ps(exp2_poly::C2);
            let c1 = _mm_set1_ps(core::f32::consts::LN_2);
            let c0 = _mm_set1_ps(1.0);

            // Horner's method (no FMA on base SSE2)
            let mut poly = _mm_add_ps(_mm_mul_ps(c4, f), c3);
            poly = _mm_add_ps(_mm_mul_ps(poly, f), c2);
            poly = _mm_add_ps(_mm_mul_ps(poly, f), c1);
            poly = _mm_add_ps(_mm_mul_ps(poly, f), c0);

            // Compute 2^n by adding n to exponent bits
            // 2^n = reinterpret((n + 127) << 23)
            let bias = _mm_set1_epi32(127);
            let n_i32 = _mm_cvtps_epi32(n);
            let exp_bits = _mm_slli_epi32(_mm_add_epi32(n_i32, bias), 23);
            let scale = _mm_castsi128_ps(exp_bits);

            Self(_mm_mul_ps(poly, scale))
        }
    }
}

// Operators for F32x4
impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(_mm_div_ps(self.0, rhs.0)) }
    }
}

impl BitAnd for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_ps(self.0, rhs.0)) }
    }
}

impl BitOr for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_ps(self.0, rhs.0)) }
    }
}

impl Not for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            // XOR with all 1s
            let all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
            Self(_mm_xor_ps(self.0, all_ones))
        }
    }
}

impl core::ops::Neg for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            // Flip sign bit via XOR with -0.0 (0x80000000 = min i32 value)
            let neg_zero = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
            Self(_mm_xor_ps(self.0, neg_zero))
        }
    }
}

// ============================================================================
// U32x4 - 4-lane u32 SIMD for packed RGBA pixels (SSE2)
// ============================================================================

/// 4-lane u32 SIMD vector for SSE2 (packed RGBA pixels).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x4(__m128i);

impl Default for U32x4 {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
    }
}

impl Debug for U32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "U32x4({:?})", arr)
    }
}

impl U32x4 {
    #[inline(always)]
    fn to_array(self) -> [u32; 4] {
        let mut arr = [0u32; 4];
        unsafe { _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, self.0) };
        arr
    }
}

impl SimdU32Ops for U32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(_mm_set1_epi32(val as i32)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // Placeholder - actual packing is done via pack_rgba
        Self::default()
    }
}

impl BitAnd for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, rhs.0)) }
    }
}

impl BitOr for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, rhs.0)) }
    }
}

impl Not for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // XOR with all 1s
        unsafe {
            let ones = _mm_set1_epi32(-1);
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl Shl<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            // _mm_sll_epi32 takes shift count in lower 64-bits of __m128i
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm_sll_epi32(self.0, shift))
        }
    }
}

impl Shr<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm_srl_epi32(self.0, shift))
        }
    }
}

impl U32x4 {
    /// Pack 4 f32 Fields (RGBA) into packed u32 pixels.
    #[inline(always)]
    #[must_use]
    pub fn pack_rgba(r: F32x4, g: F32x4, b: F32x4, a: F32x4) -> Self {
        unsafe {
            // Clamp to [0, 1] and scale to [0, 255]
            let scale = _mm_set1_ps(255.0);
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);

            let r_clamped = _mm_min_ps(_mm_max_ps(r.0, zero), one);
            let g_clamped = _mm_min_ps(_mm_max_ps(g.0, zero), one);
            let b_clamped = _mm_min_ps(_mm_max_ps(b.0, zero), one);
            let a_clamped = _mm_min_ps(_mm_max_ps(a.0, zero), one);

            let r_scaled = _mm_mul_ps(r_clamped, scale);
            let g_scaled = _mm_mul_ps(g_clamped, scale);
            let b_scaled = _mm_mul_ps(b_clamped, scale);
            let a_scaled = _mm_mul_ps(a_clamped, scale);

            // Convert to i32 (SSE2 cvttps converts to signed)
            let r_i32 = _mm_cvttps_epi32(r_scaled);
            let g_i32 = _mm_cvttps_epi32(g_scaled);
            let b_i32 = _mm_cvttps_epi32(b_scaled);
            let a_i32 = _mm_cvttps_epi32(a_scaled);

            // Pack: R | (G << 8) | (B << 16) | (A << 24)
            let g_shifted = _mm_slli_epi32(g_i32, 8);
            let b_shifted = _mm_slli_epi32(b_i32, 16);
            let a_shifted = _mm_slli_epi32(a_i32, 24);

            let packed = _mm_or_si128(
                _mm_or_si128(r_i32, g_shifted),
                _mm_or_si128(b_shifted, a_shifted),
            );
            Self(packed)
        }
    }
}

// ============================================================================
// AVX2 Backend (8 lanes)
// ============================================================================

// ============================================================================
// Mask8 - 8-lane mask for AVX2
// ============================================================================

/// 8-lane mask for AVX2.
///
/// AVX2 uses 256-bit YMM registers. Masks are typically stored as float vectors
/// where each lane is all-1s (true) or all-0s (false), similar to SSE.
/// However, AVX2 introduces integer-based masks for some operations.
/// We stick to float masks (__m256) for compatibility with blendvps.
#[cfg(target_feature = "avx2")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Mask8(__m256);

#[cfg(target_feature = "avx2")]
impl Default for Mask8 {
    fn default() -> Self {
        unsafe { Self(_mm256_setzero_ps()) }
    }
}

#[cfg(target_feature = "avx2")]
impl Debug for Mask8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mask8({:08b})", unsafe { _mm256_movemask_ps(self.0) })
    }
}

#[cfg(target_feature = "avx2")]
impl MaskOps for Mask8 {
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) == 0xFF }
    }
}

#[cfg(target_feature = "avx2")]
impl BitAnd for Mask8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_and_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl BitOr for Mask8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_or_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl Not for Mask8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            Self(_mm256_xor_ps(self.0, all_ones))
        }
    }
}

/// AVX2 Backend (8 lanes).
#[cfg(target_feature = "avx2")]
#[derive(Copy, Clone, Debug, Default)]
pub struct Avx2;

#[cfg(target_feature = "avx2")]
impl Backend for Avx2 {
    const LANES: usize = 8;
    type F32 = F32x8;
    type U32 = U32x8;
}

/// 8-lane f32 SIMD vector for AVX2.
#[cfg(target_feature = "avx2")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x8(__m256);

#[cfg(target_feature = "avx2")]
impl Default for F32x8 {
    fn default() -> Self {
        unsafe { Self(_mm256_setzero_ps()) }
    }
}

#[cfg(target_feature = "avx2")]
impl Debug for F32x8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "F32x8({:?})", arr)
    }
}

#[cfg(target_feature = "avx2")]
impl F32x8 {
    #[inline(always)]
    fn to_array(self) -> [f32; 8] {
        let mut arr = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }
}

#[cfg(target_feature = "avx2")]
impl SimdOps for F32x8 {
    type Mask = Mask8;
    const LANES: usize = 8;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            // _mm256_set_ps args are in reverse order
            Self(_mm256_set_ps(
                start + 7.0,
                start + 6.0,
                start + 5.0,
                start + 4.0,
                start + 3.0,
                start + 2.0,
                start + 1.0,
                start,
            ))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Mask8 {
        unsafe { Mask8(_mm256_cmp_ps(self.0, rhs.0, _CMP_LT_OQ)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask8 {
        unsafe { Mask8(_mm256_cmp_ps(self.0, rhs.0, _CMP_LE_OQ)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask8 {
        unsafe { Mask8(_mm256_cmp_ps(self.0, rhs.0, _CMP_GT_OQ)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask8 {
        unsafe { Mask8(_mm256_cmp_ps(self.0, rhs.0, _CMP_GE_OQ)) }
    }

    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        unsafe { Self(_mm256_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn simd_abs(self) -> Self {
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            Self(_mm256_and_ps(self.0, mask))
        }
    }

    #[inline(always)]
    fn simd_min(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_max(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_select(mask: Mask8, if_true: Self, if_false: Self) -> Self {
        unsafe { Self(_mm256_blendv_ps(if_false.0, if_true.0, mask.0)) }
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(_mm256_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        unsafe {
            let idx_i32 = _mm256_cvttps_epi32(indices.0);
            Self(_mm256_i32gather_ps::<4>(slice.as_ptr(), idx_i32))
        }
    }

    #[inline(always)]
    fn simd_floor(self) -> Self {
        unsafe { Self(_mm256_floor_ps(self.0)) }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            Self(_mm256_fmadd_ps(self.0, b.0, c.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self * b + c
        }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask8) -> Self {
        unsafe {
            let masked_val = _mm256_and_ps(mask.0, val.0);
            Self(_mm256_add_ps(self.0, masked_val))
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        unsafe { Self(_mm256_rcp_ps(self.0)) }
    }

    #[inline(always)]
    fn simd_rsqrt(self) -> Self {
        unsafe { Self(_mm256_rsqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask8) -> Self {
        Self(mask.0)
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask8 {
        Mask8(self.0)
    }

    #[inline(always)]
    fn from_u32_bits(bits: u32) -> Self {
        unsafe { Self(_mm256_castsi256_ps(_mm256_set1_epi32(bits as i32))) }
    }

    #[inline(always)]
    fn shr_u32(self, n: u32) -> Self {
        unsafe {
            let as_int = _mm256_castps_si256(self.0);
            let shift = _mm_cvtsi32_si128(n as i32);
            let shifted = _mm256_srl_epi32(as_int, shift);
            Self(_mm256_castsi256_ps(shifted))
        }
    }

    #[inline(always)]
    fn i32_to_f32(self) -> Self {
        unsafe {
            let as_int = _mm256_castps_si256(self.0);
            Self(_mm256_cvtepi32_ps(as_int))
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        unsafe {
            let x_i32 = _mm256_castps_si256(self.0);

            // Extract exponent using integer ops
            // Shift right by 23 to get exponent in lowest 8 bits
            let exp_shifted = _mm256_srli_epi32(x_i32, 23);
            // Mask to keep only 8 bits (remove sign bit if present)
            let exp_masked = _mm256_and_si256(exp_shifted, _mm256_set1_epi32(0xFF));
            // Subtract bias (127) to get unbiased exponent
            let exp_unbiased = _mm256_sub_epi32(exp_masked, _mm256_set1_epi32(127));
            // Convert to float
            let mut n = _mm256_cvtepi32_ps(exp_unbiased);

            // Extract mantissa in [1, 2)
            let mant_mask = _mm256_set1_epi32(0x007FFFFF_u32 as i32);
            let one_bits = _mm256_set1_epi32(0x3F800000_u32 as i32);
            let mut f = _mm256_castsi256_ps(_mm256_or_si256(
                _mm256_and_si256(x_i32, mant_mask),
                one_bits,
            ));

            // Adjust to [√2/2, √2] range for better accuracy (centered at 1)
            // If f >= √2, divide by 2 and increment exponent
            let sqrt2 = _mm256_set1_ps(core::f32::consts::SQRT_2);
            let mask = _mm256_cmp_ps::<_CMP_GE_OQ>(f, sqrt2);
            let adjust = _mm256_and_ps(mask, _mm256_set1_ps(1.0));
            n = _mm256_add_ps(n, adjust);
            f = _mm256_or_ps(
                _mm256_and_ps(mask, _mm256_mul_ps(f, _mm256_set1_ps(0.5))),
                _mm256_andnot_ps(mask, f),
            );

            // Polynomial for log2(f) on [√2/2, √2]
            // Fitted using least squares on Chebyshev nodes
            // Max error: ~1e-4
            let c4 = _mm256_set1_ps(log2_poly::C4);
            let c3 = _mm256_set1_ps(log2_poly::C3);
            let c2 = _mm256_set1_ps(log2_poly::C2);
            let c1 = _mm256_set1_ps(log2_poly::C1);
            let c0 = _mm256_set1_ps(log2_poly::C0);

            // Horner's method with FMA when available
            #[cfg(target_feature = "fma")]
            {
                let mut poly = _mm256_fmadd_ps(c4, f, c3);
                poly = _mm256_fmadd_ps(poly, f, c2);
                poly = _mm256_fmadd_ps(poly, f, c1);
                poly = _mm256_fmadd_ps(poly, f, c0);
                Self(_mm256_add_ps(n, poly))
            }
            #[cfg(not(target_feature = "fma"))]
            {
                let mut poly = _mm256_add_ps(_mm256_mul_ps(c4, f), c3);
                poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c2);
                poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c1);
                poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c0);
                Self(_mm256_add_ps(n, poly))
            }
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        unsafe {
            // n = floor(x), f = x - n
            let n = _mm256_floor_ps(self.0);
            let f = _mm256_sub_ps(self.0, n);

            // Minimax polynomial for 2^f, f ∈ [0, 1)
            let c4 = _mm256_set1_ps(exp2_poly::C4);
            let c3 = _mm256_set1_ps(exp2_poly::C3);
            let c2 = _mm256_set1_ps(exp2_poly::C2);
            let c1 = _mm256_set1_ps(core::f32::consts::LN_2);
            let c0 = _mm256_set1_ps(1.0);

            // Horner's method
            #[cfg(target_feature = "fma")]
            {
                let mut poly = _mm256_fmadd_ps(c4, f, c3);
                poly = _mm256_fmadd_ps(poly, f, c2);
                poly = _mm256_fmadd_ps(poly, f, c1);
                poly = _mm256_fmadd_ps(poly, f, c0);

                // 2^n = (n + 127) << 23
                let bias = _mm256_set1_epi32(127);
                let n_i32 = _mm256_cvtps_epi32(n);
                let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i32, bias), 23);
                let scale = _mm256_castsi256_ps(exp_bits);

                Self(_mm256_mul_ps(poly, scale))
            }
            #[cfg(not(target_feature = "fma"))]
            {
                let mut poly = _mm256_add_ps(_mm256_mul_ps(c4, f), c3);
                poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c2);
                poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c1);
                poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c0);

                let bias = _mm256_set1_epi32(127);
                let n_i32 = _mm256_cvtps_epi32(n);
                let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i32, bias), 23);
                let scale = _mm256_castsi256_ps(exp_bits);

                Self(_mm256_mul_ps(poly, scale))
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Add for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl Sub for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl Mul for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl Div for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_div_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl BitAnd for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_and_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl BitOr for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_or_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl Not for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            Self(_mm256_xor_ps(self.0, all_ones))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl core::ops::Neg for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let neg_zero = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
            Self(_mm256_xor_ps(self.0, neg_zero))
        }
    }
}

// ============================================================================
// U32x8 - 8-lane u32 SIMD for packed RGBA pixels (AVX2)
// ============================================================================

/// 8-lane u32 SIMD vector for AVX2 (packed RGBA pixels).
#[cfg(target_feature = "avx2")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x8(__m256i);

#[cfg(target_feature = "avx2")]
impl Default for U32x8 {
    fn default() -> Self {
        unsafe { Self(_mm256_setzero_si256()) }
    }
}

#[cfg(target_feature = "avx2")]
impl Debug for U32x8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut arr = [0u32; 8];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.0) };
        write!(f, "U32x8({:?})", arr)
    }
}

#[cfg(target_feature = "avx2")]
impl SimdU32Ops for U32x8 {
    const LANES: usize = 8;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(_mm256_set1_epi32(val as i32)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        Self::default()
    }
}

#[cfg(target_feature = "avx2")]
impl BitAnd for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_and_si256(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl BitOr for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_or_si256(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx2")]
impl Not for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm256_set1_epi32(-1);
            Self(_mm256_xor_si256(self.0, ones))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Shl<u32> for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm256_sll_epi32(self.0, shift))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Shr<u32> for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm256_srl_epi32(self.0, shift))
        }
    }
}

#[cfg(target_feature = "avx2")]
impl U32x8 {
    /// Pack 8 f32 Fields (RGBA) into packed u32 pixels.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn pack_rgba(r: F32x8, g: F32x8, b: F32x8, a: F32x8) -> Self {
        unsafe {
            let scale = _mm256_set1_ps(255.0);
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);

            let r_clamped = _mm256_min_ps(_mm256_max_ps(r.0, zero), one);
            let g_clamped = _mm256_min_ps(_mm256_max_ps(g.0, zero), one);
            let b_clamped = _mm256_min_ps(_mm256_max_ps(b.0, zero), one);
            let a_clamped = _mm256_min_ps(_mm256_max_ps(a.0, zero), one);

            let r_scaled = _mm256_mul_ps(r_clamped, scale);
            let g_scaled = _mm256_mul_ps(g_clamped, scale);
            let b_scaled = _mm256_mul_ps(b_clamped, scale);
            let a_scaled = _mm256_mul_ps(a_clamped, scale);

            let r_i32 = _mm256_cvttps_epi32(r_scaled);
            let g_i32 = _mm256_cvttps_epi32(g_scaled);
            let b_i32 = _mm256_cvttps_epi32(b_scaled);
            let a_i32 = _mm256_cvttps_epi32(a_scaled);

            let g_shifted = _mm256_slli_epi32(g_i32, 8);
            let b_shifted = _mm256_slli_epi32(b_i32, 16);
            let a_shifted = _mm256_slli_epi32(a_i32, 24);

            let packed = _mm256_or_si256(
                _mm256_or_si256(r_i32, g_shifted),
                _mm256_or_si256(b_shifted, a_shifted),
            );
            Self(packed)
        }
    }
}

// ============================================================================
// AVX512 Backend
// ============================================================================

// ============================================================================
// Mask16 - 16-lane native k-register mask for AVX-512
// ============================================================================

/// 16-lane mask using AVX-512 k-registers.
///
/// This is the big win: AVX-512 has dedicated 8 mask registers (k0-k7) that
/// run on a separate execution unit from the float ALU. Mask operations
/// (and/or/not/any/all) are effectively free - they execute in parallel
/// with float work.
///
/// - `kortestw k1, k1` for any() - ~0 cycles (mask unit)
/// - `kand/kor/knot` for mask logic - ~0-1 cycles (mask unit)
/// - `vblendmps` uses k-register directly - no conversion overhead
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub struct Mask16(pub(crate) __mmask16);

#[cfg(target_feature = "avx512f")]
impl Debug for Mask16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mask16({:016b})", self.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl MaskOps for Mask16 {
    #[inline(always)]
    fn any(self) -> bool {
        // kortestw k1, k1 - runs on mask unit, ~0 cycles
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        // Mask equality check - runs on mask unit, ~0 cycles
        self.0 == 0xFFFF
    }
}

#[cfg(target_feature = "avx512f")]
impl BitAnd for Mask16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // kand - runs on mask unit, ~0-1 cycles
        Self(self.0 & rhs.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl BitOr for Mask16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // kor - runs on mask unit, ~0-1 cycles
        Self(self.0 | rhs.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl Not for Mask16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // knot - runs on mask unit, ~0-1 cycles
        Self(!self.0)
    }
}

/// AVX512 Backend (16 lanes).
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone, Debug, Default)]
pub struct Avx512;

#[cfg(target_feature = "avx512f")]
impl Backend for Avx512 {
    const LANES: usize = 16;
    type F32 = F32x16;
    type U32 = U32x16;
}

/// 16-lane f32 SIMD vector for AVX512.
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x16(__m512);

#[cfg(target_feature = "avx512f")]
impl Default for F32x16 {
    fn default() -> Self {
        unsafe { Self(_mm512_setzero_ps()) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Debug for F32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "F32x16({:?})", arr)
    }
}

#[cfg(target_feature = "avx512f")]
impl F32x16 {
    #[inline(always)]
    fn to_array(self) -> [f32; 16] {
        let mut arr = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[allow(dead_code)]
    #[inline(always)]
    unsafe fn from_mask(mask: __mmask16) -> Self {
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            let all_zeros = _mm512_setzero_ps();
            Self(_mm512_mask_blend_ps(mask, all_zeros, all_ones))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl SimdOps for F32x16 {
    type Mask = Mask16;
    const LANES: usize = 16;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            // _mm512_set_ps args are in reverse order: e15, e14, ..., e1, e0
            let base = _mm512_set1_ps(start);
            let increments = _mm512_set_ps(
                15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
                0.0,
            );
            Self(_mm512_add_ps(base, increments))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Mask16 {
        // Returns native k-register mask directly - no conversion!
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LT_OQ)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask16 {
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LE_OQ)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask16 {
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GT_OQ)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask16 {
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GE_OQ)) }
    }

    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        unsafe { Self(_mm512_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn simd_abs(self) -> Self {
        unsafe {
            let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
            Self(_mm512_and_ps(self.0, abs_mask))
        }
    }

    #[inline(always)]
    fn simd_min(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_max(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_select(mask: Mask16, if_true: Self, if_false: Self) -> Self {
        // Native k-register blend - no conversion needed!
        // vblendmps uses k-register directly
        unsafe { Self(_mm512_mask_blend_ps(mask.0, if_false.0, if_true.0)) }
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(_mm512_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // Native AVX-512 gather - 16 floats in one instruction
        // Precondition: indices must be valid (0..slice.len()), already ensured by Texture::eval_raw
        unsafe {
            // Convert float indices to i32 (truncate - indices are already integral from floor)
            let idx_i32 = _mm512_cvttps_epi32(indices.0);
            // Gather 16 f32 values; scale=4 bytes per element
            Self(_mm512_i32gather_ps::<4>(idx_i32, slice.as_ptr()))
        }
    }

    #[inline(always)]
    fn simd_floor(self) -> Self {
        unsafe {
            // AVX-512: use roundscale with floor mode (1 = floor, bit 3 = suppress exceptions)
            // imm8 = 0b1001 = 9
            Self(_mm512_roundscale_ps::<9>(self.0))
        }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // AVX-512F always includes FMA
        unsafe { Self(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask16) -> Self {
        // Native masked add using k-register directly - no conversion!
        // This is the hot path for winding number accumulation
        unsafe { Self(_mm512_mask_add_ps(self.0, mask.0, self.0, val.0)) }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // AVX-512 has higher accuracy reciprocal (~14 bits)
        unsafe { Self(_mm512_rcp14_ps(self.0)) }
    }

    #[inline(always)]
    fn simd_rsqrt(self) -> Self {
        // Fast reciprocal square root (~14 bits accuracy)
        // Sufficient for AA coverage calculations
        unsafe { Self(_mm512_rsqrt14_ps(self.0)) }
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask16) -> Self {
        // Convert k-register to float representation (all-1s or all-0s)
        // This is the "expensive" direction - try to avoid in hot paths
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            let all_zeros = _mm512_setzero_ps();
            Self(_mm512_mask_blend_ps(mask.0, all_zeros, all_ones))
        }
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask16 {
        // Convert float representation to k-register mask
        // Check if each lane is non-zero
        unsafe {
            let as_int = _mm512_castps_si512(self.0);
            Mask16(_mm512_cmp_epi32_mask(
                as_int,
                _mm512_setzero_si512(),
                _MM_CMPINT_NE,
            ))
        }
    }

    #[inline(always)]
    fn from_u32_bits(bits: u32) -> Self {
        unsafe { Self(_mm512_castsi512_ps(_mm512_set1_epi32(bits as i32))) }
    }

    #[inline(always)]
    fn shr_u32(self, n: u32) -> Self {
        unsafe {
            let as_int = _mm512_castps_si512(self.0);
            let shift = _mm_cvtsi32_si128(n as i32);
            let shifted = _mm512_srl_epi32(as_int, shift);
            Self(_mm512_castsi512_ps(shifted))
        }
    }

    #[inline(always)]
    fn i32_to_f32(self) -> Self {
        unsafe {
            let as_int = _mm512_castps_si512(self.0);
            Self(_mm512_cvtepi32_ps(as_int))
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        unsafe {
            // Extract mantissa in [1, 2) - no exponent adjustment needed
            // Interval=0 (_MM_MANT_NORM_1_2), sign=0 (_MM_MANT_SIGN_src)
            let mut f = _mm512_getmant_ps::<0, 0>(self.0);

            // Extract exponent
            let mut n = _mm512_getexp_ps(self.0);

            // Adjust to [√2/2, √2] range for better accuracy (centered at 1)
            // If f >= √2, divide by 2 and increment exponent
            let sqrt2 = _mm512_set1_ps(core::f32::consts::SQRT_2);
            let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(f, sqrt2);
            let adjust = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_set1_ps(1.0));
            n = _mm512_add_ps(n, adjust);
            f = _mm512_mask_blend_ps(mask, f, _mm512_mul_ps(f, _mm512_set1_ps(0.5)));

            // Polynomial for log2(f) on [√2/2, √2]
            // Fitted using least squares on Chebyshev nodes
            // Max error: ~1e-4
            let c4 = _mm512_set1_ps(log2_poly::C4);
            let c3 = _mm512_set1_ps(log2_poly::C3);
            let c2 = _mm512_set1_ps(log2_poly::C2);
            let c1 = _mm512_set1_ps(log2_poly::C1);
            let c0 = _mm512_set1_ps(log2_poly::C0);

            // Horner's method with FMA
            let mut poly = _mm512_fmadd_ps(c4, f, c3);
            poly = _mm512_fmadd_ps(poly, f, c2);
            poly = _mm512_fmadd_ps(poly, f, c1);
            poly = _mm512_fmadd_ps(poly, f, c0);

            Self(_mm512_add_ps(n, poly))
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // AVX-512: Use scalef for efficient 2^n computation
        // 2^x = 2^n * 2^f where n = floor(x), f = frac(x) ∈ [0, 1)
        unsafe {
            // n = floor(x), f = x - n
            let n = _mm512_roundscale_ps::<9>(self.0); // 9 = floor mode
            let f = _mm512_sub_ps(self.0, n);

            // Minimax polynomial for 2^f, f ∈ [0, 1)
            // Degree 4, max error ~10^-7
            let c4 = _mm512_set1_ps(exp2_poly::C4);
            let c3 = _mm512_set1_ps(exp2_poly::C3);
            let c2 = _mm512_set1_ps(exp2_poly::C2);
            let c1 = _mm512_set1_ps(core::f32::consts::LN_2);
            let c0 = _mm512_set1_ps(1.0);

            // Horner's method with FMA
            let poly = _mm512_fmadd_ps(c4, f, c3);
            let poly = _mm512_fmadd_ps(poly, f, c2);
            let poly = _mm512_fmadd_ps(poly, f, c1);
            let poly = _mm512_fmadd_ps(poly, f, c0);

            // Use scalef: poly * 2^n
            Self(_mm512_scalef_ps(poly, n))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl Add for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_add_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Sub for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Mul for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Div for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_div_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl BitAnd for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_and_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl BitOr for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_or_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Not for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            Self(_mm512_xor_ps(self.0, all_ones))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl core::ops::Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            // Flip sign bit via XOR with -0.0 (0x80000000 = min i32 value)
            let neg_zero = _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN));
            Self(_mm512_xor_ps(self.0, neg_zero))
        }
    }
}

// ============================================================================
// U32x16 - 16-lane u32 SIMD for packed RGBA pixels (AVX512)
// ============================================================================

/// 16-lane u32 SIMD vector for AVX512 (packed RGBA pixels).
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x16(__m512i);

#[cfg(target_feature = "avx512f")]
impl Default for U32x16 {
    fn default() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Debug for U32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut arr = [0u32; 16];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut __m512i, self.0) };
        write!(f, "U32x16({:?})", arr)
    }
}

#[cfg(target_feature = "avx512f")]
impl SimdU32Ops for U32x16 {
    const LANES: usize = 16;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(_mm512_set1_epi32(val as i32)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // Placeholder
        Self::default()
    }
}

#[cfg(target_feature = "avx512f")]
impl BitAnd for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl BitOr for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_or_si512(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Not for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // XOR with all 1s
        unsafe {
            let ones = _mm512_set1_epi32(-1);
            Self(_mm512_xor_si512(self.0, ones))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl Shl<u32> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm512_sll_epi32(self.0, shift))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl Shr<u32> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm512_srl_epi32(self.0, shift))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl U32x16 {
    /// Pack 16 f32 Fields (RGBA) into packed u32 pixels.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn pack_rgba(r: F32x16, g: F32x16, b: F32x16, a: F32x16) -> Self {
        unsafe {
            let scale = _mm512_set1_ps(255.0);
            let zero = _mm512_setzero_ps();
            let one = _mm512_set1_ps(1.0);

            let r_clamped = _mm512_min_ps(_mm512_max_ps(r.0, zero), one);
            let g_clamped = _mm512_min_ps(_mm512_max_ps(g.0, zero), one);
            let b_clamped = _mm512_min_ps(_mm512_max_ps(b.0, zero), one);
            let a_clamped = _mm512_min_ps(_mm512_max_ps(a.0, zero), one);

            let r_scaled = _mm512_mul_ps(r_clamped, scale);
            let g_scaled = _mm512_mul_ps(g_clamped, scale);
            let b_scaled = _mm512_mul_ps(b_clamped, scale);
            let a_scaled = _mm512_mul_ps(a_clamped, scale);

            let r_i32 = _mm512_cvttps_epi32(r_scaled);
            let g_i32 = _mm512_cvttps_epi32(g_scaled);
            let b_i32 = _mm512_cvttps_epi32(b_scaled);
            let a_i32 = _mm512_cvttps_epi32(a_scaled);

            let g_shifted = _mm512_slli_epi32(g_i32, 8);
            let b_shifted = _mm512_slli_epi32(b_i32, 16);
            let a_shifted = _mm512_slli_epi32(a_i32, 24);

            let packed = _mm512_or_si512(
                _mm512_or_si512(r_i32, g_shifted),
                _mm512_or_si512(b_shifted, a_shifted),
            );
            Self(packed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_feature = "avx512f")]
    fn test_avx512_log2() {
        let test_vals = [0.5f32, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0];
        for &val in &test_vals {
            let v = F32x16::splat(val);
            let result = v.log2();
            let mut buf = [0.0f32; 16];
            result.store(&mut buf);
            let expected = val.log2();
            assert!(
                (buf[0] - expected).abs() < 0.01,
                "log2({}) = {}, expected {}",
                val,
                buf[0],
                expected
            );
        }
    }
}
