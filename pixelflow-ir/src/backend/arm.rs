//! ARM NEON backend (4 lanes for f32).

use super::{Backend, MaskOps, SimdOps, SimdU32Ops};
use core::arch::aarch64::*;
use core::fmt::{Debug, Formatter};
use core::ops::*;

/// NEON Backend (4 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Neon;

impl Backend for Neon {
    const LANES: usize = 4;
    type F32 = F32x4;
    type U32 = U32x4;
}

// ============================================================================
// Mask4 - 4-lane mask for NEON (integer-based)
// ============================================================================

/// 4-lane mask for ARM NEON.
///
/// NEON doesn't have dedicated mask registers like AVX-512's k-registers.
/// Masks are stored as u32 vectors where each lane is either all-1s (0xFFFFFFFF)
/// or all-0s (0x00000000).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Mask4(uint32x4_t);

impl Default for Mask4 {
    fn default() -> Self {
        unsafe { Self(vdupq_n_u32(0)) }
    }
}

impl Debug for Mask4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        let bits = (if arr[0] != 0 { 1 } else { 0 })
            | (if arr[1] != 0 { 2 } else { 0 })
            | (if arr[2] != 0 { 4 } else { 0 })
            | (if arr[3] != 0 { 8 } else { 0 });
        write!(f, "Mask4({:04b})", bits)
    }
}

impl Mask4 {
    #[inline(always)]
    fn to_array(self) -> [u32; 4] {
        let mut arr = [0u32; 4];
        unsafe { vst1q_u32(arr.as_mut_ptr(), self.0) };
        arr
    }
}

impl MaskOps for Mask4 {
    #[inline(always)]
    fn any(self) -> bool {
        // Use inline asm to prevent LLVM from transforming vmax into movemask pattern.
        // LLVM's optimizer converts vmaxvq_u32 into a slower 6-instruction sequence.
        // Inline asm forces the optimal 2-instruction umaxv+fmov sequence.
        unsafe {
            let max_val: u32;
            core::arch::asm!(
                "umaxv {s:s}, {v:v}.4s",
                "fmov {w:w}, {s:s}",
                v = in(vreg) self.0,
                s = lateout(vreg) _,
                w = lateout(reg) max_val,
                options(pure, nomem, nostack),
            );
            max_val != 0
        }
    }

    #[inline(always)]
    fn all(self) -> bool {
        // Use inline asm to prevent LLVM from transforming vmin into movemask pattern.
        // LLVM's optimizer converts vminvq_u32 into a slower 6-instruction sequence.
        // Inline asm forces the optimal 2-instruction uminv+fmov sequence.
        unsafe {
            let min_val: u32;
            core::arch::asm!(
                "uminv {s:s}, {v:v}.4s",
                "fmov {w:w}, {s:s}",
                v = in(vreg) self.0,
                s = lateout(vreg) _,
                w = lateout(reg) min_val,
                options(pure, nomem, nostack),
            );
            min_val != 0
        }
    }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(vandq_u32(self.0, rhs.0)) }
    }
}

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(vorrq_u32(self.0, rhs.0)) }
    }
}

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(vmvnq_u32(self.0)) }
    }
}

/// 4-lane f32 SIMD vector for ARM NEON.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(float32x4_t);

impl Default for F32x4 {
    fn default() -> Self {
        unsafe { Self(vdupq_n_f32(0.0)) }
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
        unsafe { vst1q_f32(arr.as_mut_ptr(), self.0) };
        arr
    }
}

// ============================================================================
// SimdOps Implementation
// ============================================================================

impl SimdOps for F32x4 {
    type Mask = Mask4;
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(vdupq_n_f32(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            let arr = [start, start + 1.0, start + 2.0, start + 3.0];
            Self(vld1q_f32(arr.as_ptr()))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        unsafe { vst1q_f32(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(vcltq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(vcleq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(vcgtq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(vcgeq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        unsafe { Self(vsqrtq_f32(self.0)) }
    }

    #[inline(always)]
    fn simd_abs(self) -> Self {
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn simd_min(self, rhs: Self) -> Self {
        unsafe { Self(vminq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_max(self, rhs: Self) -> Self {
        unsafe { Self(vmaxq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn simd_select(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            let result = vbslq_f32(mask.0, if_true.0, if_false.0);
            Self(result)
        }
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(vld1q_f32(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // NEON doesn't have gather - do scalar loads
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
        unsafe { Self(vrndmq_f32(self.0)) }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // ARM NEON FMA: vfmaq_f32(c, a, b) computes a*b + c
        unsafe { Self(vfmaq_f32(c.0, self.0, b.0)) }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask4) -> Self {
        // NEON: no native masked add, emulate with select
        // self + (mask ? val : 0)
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let masked_val = vbslq_f32(mask.0, val.0, zero);
            Self(vaddq_f32(self.0, masked_val))
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // NEON approximate reciprocal (~8 bits accuracy)
        // One Newton-Raphson iteration improves to ~16 bits
        unsafe {
            let est = vrecpeq_f32(self.0);
            // Newton-Raphson: est = est * (2 - x * est)
            let refined = vmulq_f32(est, vrecpsq_f32(self.0, est));
            Self(refined)
        }
    }

    #[inline(always)]
    fn simd_rsqrt(self) -> Self {
        // NEON approximate reciprocal square root (~8 bits accuracy)
        // One Newton-Raphson iteration improves to ~16 bits
        unsafe {
            let est = vrsqrteq_f32(self.0);
            // Newton-Raphson: est = est * (3 - x * est^2) / 2
            // vrsqrtsq_f32(x, y) computes (3 - x * y) / 2
            let est_sq = vmulq_f32(est, est);
            let refined = vmulq_f32(est, vrsqrtsq_f32(self.0, est_sq));
            Self(refined)
        }
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask4) -> Self {
        // Convert u32 mask to float representation
        unsafe { Self(vreinterpretq_f32_u32(mask.0)) }
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask4 {
        // Convert float representation to u32 mask
        unsafe { Mask4(vreinterpretq_u32_f32(self.0)) }
    }

    #[inline(always)]
    fn from_u32_bits(bits: u32) -> Self {
        unsafe { Self(vreinterpretq_f32_u32(vdupq_n_u32(bits))) }
    }

    #[inline(always)]
    fn shr_u32(self, n: u32) -> Self {
        unsafe {
            let as_int = vreinterpretq_u32_f32(self.0);
            // NEON uses negative shift for right shift
            let shift = vdupq_n_s32(-(n as i32));
            let shifted = vshlq_u32(as_int, shift);
            Self(vreinterpretq_f32_u32(shifted))
        }
    }

    #[inline(always)]
    fn i32_to_f32(self) -> Self {
        unsafe {
            let as_int = vreinterpretq_s32_f32(self.0);
            Self(vcvtq_f32_s32(as_int))
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        // NEON: Use bit manipulation for exponent/mantissa extraction
        // Uses range [√2/2, √2] centered at 1 for better polynomial accuracy
        // log2(x) = exponent + log2(mantissa)
        unsafe {
            let x_u32 = vreinterpretq_u32_f32(self.0);

            // Extract exponent: (bits >> 23) - 127
            let exp_bits = vshrq_n_u32::<23>(x_u32);
            let bias = vdupq_n_s32(127);
            let mut n = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(exp_bits), bias));

            // Extract mantissa in [1, 2): (bits & 0x007FFFFF) | 0x3F800000
            let mant_mask = vdupq_n_u32(0x007FFFFF);
            let one_bits = vdupq_n_u32(0x3F800000);
            let mut f = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(x_u32, mant_mask), one_bits));

            // Adjust to [√2/2, √2] range for better accuracy (centered at 1)
            // If f >= √2, divide by 2 and increment exponent
            let sqrt2 = vdupq_n_f32(1.4142135624);
            let mask = vcgeq_f32(f, sqrt2);
            let adjust = vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0)));
            n = vaddq_f32(n, vreinterpretq_f32_u32(adjust));
            f = vbslq_f32(mask, vmulq_f32(f, vdupq_n_f32(0.5)), f);

            // Polynomial for log2(f) on [√2/2, √2]
            // Fitted using least squares on Chebyshev nodes
            // Max error: ~1e-4
            let c4 = vdupq_n_f32(-0.3200435159);
            let c3 = vdupq_n_f32(1.7974969154);
            let c2 = vdupq_n_f32(-4.1988046176);
            let c1 = vdupq_n_f32(5.7270231695);
            let c0 = vdupq_n_f32(-3.0056146714);

            // Horner's method using NEON FMA: vfmaq_f32(c, a, b) = a*b + c
            let poly = vfmaq_f32(c3, c4, f);
            let poly = vfmaq_f32(c2, poly, f);
            let poly = vfmaq_f32(c1, poly, f);
            let poly = vfmaq_f32(c0, poly, f);

            Self(vaddq_f32(n, poly))
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // NEON: 2^x = 2^n * 2^f where n = floor(x), f = frac(x) ∈ [0, 1)
        // Use polynomial approximation for 2^f
        unsafe {
            // n = floor(x), f = x - n
            let n = vrndmq_f32(self.0); // floor
            let f = vsubq_f32(self.0, n);

            // Minimax polynomial for 2^f, f ∈ [0, 1)
            // Degree 4, max error ~10^-7
            let c4 = vdupq_n_f32(0.0135557);
            let c3 = vdupq_n_f32(0.0520323);
            let c2 = vdupq_n_f32(0.2413793);
            let c1 = vdupq_n_f32(0.6931472);
            let c0 = vdupq_n_f32(1.0);

            // Horner's method
            let poly = vfmaq_f32(c3, c4, f);
            let poly = vfmaq_f32(c2, poly, f);
            let poly = vfmaq_f32(c1, poly, f);
            let poly = vfmaq_f32(c0, poly, f);

            // Compute 2^n by adding n to exponent bits
            // 2^n = reinterpret((n + 127) << 23)
            let bias = vdupq_n_s32(127);
            let n_i32 = vcvtq_s32_f32(n);
            let exp_bits = vshlq_n_s32::<23>(vaddq_s32(n_i32, bias));
            let scale = vreinterpretq_f32_s32(exp_bits);

            Self(vmulq_f32(poly, scale))
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        unsafe {
            const PI: f32 = core::f32::consts::PI;
            const TWO_PI: f32 = core::f32::consts::TAU;
            const TWO_PI_INV: f32 = 1.0 / TWO_PI;
            const PI_INV: f32 = 1.0 / PI;

            // Range reduce to [-π, π]
            let k = vrndmq_f32(vaddq_f32(
                vmulq_f32(self.0, vdupq_n_f32(TWO_PI_INV)),
                vdupq_n_f32(0.5),
            ));
            let x = vsubq_f32(self.0, vmulq_f32(k, vdupq_n_f32(TWO_PI)));
            let t = vmulq_f32(x, vdupq_n_f32(PI_INV));

            let c1 = vdupq_n_f32(1.6719970703125);
            let c3 = vdupq_n_f32(-0.645963541666667);
            let c5 = vdupq_n_f32(0.079689450);
            let c7 = vdupq_n_f32(-0.0046817541);

            let t2 = vmulq_f32(t, t);
            let poly = vfmaq_f32(c5, c7, t2);
            let poly = vfmaq_f32(c3, poly, t2);
            let poly = vfmaq_f32(c1, poly, t2);
            Self(vmulq_f32(poly, t))
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        unsafe {
            const PI: f32 = core::f32::consts::PI;
            const TWO_PI: f32 = core::f32::consts::TAU;
            const TWO_PI_INV: f32 = 1.0 / TWO_PI;
            const PI_INV: f32 = 1.0 / PI;

            let k = vrndmq_f32(vaddq_f32(
                vmulq_f32(self.0, vdupq_n_f32(TWO_PI_INV)),
                vdupq_n_f32(0.5),
            ));
            let x = vsubq_f32(self.0, vmulq_f32(k, vdupq_n_f32(TWO_PI)));
            let t = vmulq_f32(x, vdupq_n_f32(PI_INV));

            let c0 = vdupq_n_f32(1.5707963267948966);
            let c2 = vdupq_n_f32(-2.467401341);
            let c4 = vdupq_n_f32(0.609469381);
            let c6 = vdupq_n_f32(-0.038854038);

            let t2 = vmulq_f32(t, t);
            let poly = vfmaq_f32(c4, c6, t2);
            let poly = vfmaq_f32(c2, poly, t2);
            Self(vfmaq_f32(c0, poly, t2))
        }
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        unsafe {
            const PI: f32 = core::f32::consts::PI;
            const PI_2: f32 = core::f32::consts::FRAC_PI_2;

            let y = self.0;
            let x_val = x.0;

            let r = vdivq_f32(y, x_val);
            let r_abs = vabsq_f32(r);

            let c1 = vdupq_n_f32(0.999999999);
            let c3 = vdupq_n_f32(-0.333333333);
            let c5 = vdupq_n_f32(0.2);
            let c7 = vdupq_n_f32(-0.142857143);

            let t = r_abs;
            let t2 = vmulq_f32(t, t);
            let poly = vfmaq_f32(c5, c7, t2);
            let poly = vfmaq_f32(c3, poly, t2);
            let poly = vfmaq_f32(c1, poly, t2);
            let atan_approx = vmulq_f32(poly, t);

            let one = vdupq_n_f32(1.0);
            let mask_large = vcgtq_f32(r_abs, one);
            let recip_r = vdivq_f32(one, r_abs);
            let atan_large = vsubq_f32(vdupq_n_f32(PI_2), vmulq_f32(recip_r, atan_approx));
            let atan_val = vbslq_f32(mask_large, atan_large, atan_approx);

            let y_abs = vabsq_f32(y);
            let sign_y = vdivq_f32(y_abs, y);
            let atan_signed = vmulq_f32(atan_val, sign_y);

            let zero = vdupq_n_f32(0.0);
            let mask_neg_x = vcltq_f32(x_val, zero);
            let correction = vmulq_f32(vdupq_n_f32(PI), sign_y);
            Self(vbslq_f32(
                mask_neg_x,
                vsubq_f32(atan_signed, correction),
                atan_signed,
            ))
        }
    }

    #[inline(always)]
    fn cmp_eq(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(vceqq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_ne(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(vmvnq_u32(vceqq_f32(self.0, rhs.0))) }
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }
}

impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }
}

impl Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }
}

impl Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(vdivq_f32(self.0, rhs.0)) }
    }
}

impl BitAnd for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0);
            let b = vreinterpretq_u32_f32(rhs.0);
            Self(vreinterpretq_f32_u32(vandq_u32(a, b)))
        }
    }
}

impl BitOr for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0);
            let b = vreinterpretq_u32_f32(rhs.0);
            Self(vreinterpretq_f32_u32(vorrq_u32(a, b)))
        }
    }
}

impl Not for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0);
            Self(vreinterpretq_f32_u32(vmvnq_u32(a)))
        }
    }
}

impl core::ops::Neg for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            // Flip sign bit via XOR with -0.0 (0x80000000 in each lane)
            let neg_zero_u32 = vdupq_n_u32(0x80000000u32);
            let self_u32 = vreinterpretq_u32_f32(self.0);
            Self(vreinterpretq_f32_u32(veorq_u32(self_u32, neg_zero_u32)))
        }
    }
}

// ============================================================================
// U32x4 - 4-lane u32 SIMD for packed RGBA pixels
// ============================================================================

/// 4-lane u32 SIMD vector for ARM NEON (packed RGBA pixels).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x4(uint32x4_t);

impl Default for U32x4 {
    fn default() -> Self {
        unsafe { Self(vdupq_n_u32(0)) }
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
        unsafe { vst1q_u32(arr.as_mut_ptr(), self.0) };
        arr
    }
}

impl SimdU32Ops for U32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(vdupq_n_u32(val)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        unsafe { vst1q_u32(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // This is a placeholder - actual impl needs F to be F32x4
        // We'll handle this differently
        Self::default()
    }
}

impl BitAnd for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(vandq_u32(self.0, rhs.0)) }
    }
}

impl BitOr for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(vorrq_u32(self.0, rhs.0)) }
    }
}

impl Not for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(vmvnq_u32(self.0)) }
    }
}

impl Shl<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            let shift = vdupq_n_s32(rhs as i32);
            Self(vshlq_u32(self.0, shift))
        }
    }
}

impl Shr<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = vdupq_n_s32(-(rhs as i32));
            Self(vshlq_u32(self.0, shift))
        }
    }
}

impl U32x4 {
    /// Pack 4 f32 Fields (RGBA) into packed u32 pixels.
    #[inline(always)]
    pub fn pack_rgba(r: F32x4, g: F32x4, b: F32x4, a: F32x4) -> Self {
        unsafe {
            // Clamp to [0, 1] and scale to [0, 255]
            let scale = vdupq_n_f32(255.0);
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);

            let r_clamped = vminq_f32(vmaxq_f32(r.0, zero), one);
            let g_clamped = vminq_f32(vmaxq_f32(g.0, zero), one);
            let b_clamped = vminq_f32(vmaxq_f32(b.0, zero), one);
            let a_clamped = vminq_f32(vmaxq_f32(a.0, zero), one);

            let r_scaled = vmulq_f32(r_clamped, scale);
            let g_scaled = vmulq_f32(g_clamped, scale);
            let b_scaled = vmulq_f32(b_clamped, scale);
            let a_scaled = vmulq_f32(a_clamped, scale);

            // Convert to u32
            let r_u32 = vcvtq_u32_f32(r_scaled);
            let g_u32 = vcvtq_u32_f32(g_scaled);
            let b_u32 = vcvtq_u32_f32(b_scaled);
            let a_u32 = vcvtq_u32_f32(a_scaled);

            // Pack: R | (G << 8) | (B << 16) | (A << 24)
            let g_shifted = vshlq_n_u32(g_u32, 8);
            let b_shifted = vshlq_n_u32(b_u32, 16);
            let a_shifted = vshlq_n_u32(a_u32, 24);

            let packed = vorrq_u32(vorrq_u32(r_u32, g_shifted), vorrq_u32(b_shifted, a_shifted));
            Self(packed)
        }
    }
}
