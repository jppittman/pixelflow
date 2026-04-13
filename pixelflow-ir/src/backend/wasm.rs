//! WebAssembly SIMD128 backend (4 lanes for f32).

use super::{Backend, MaskOps, SimdOps, SimdU32Ops};
use core::arch::wasm32::*;
use core::fmt::{Debug, Formatter};
use core::ops::*;

/// WebAssembly SIMD128 Backend (4 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Wasm;

/// 4-lane mask for WASM SIMD128.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Mask4(v128);

/// 4-lane f32 SIMD vector for WASM SIMD128.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(v128);

/// 4-lane u32 SIMD vector for WASM SIMD128.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x4(v128);

impl Backend for Wasm {
    const LANES: usize = 4;
    type F32 = F32x4;
    type U32 = U32x4;
}

// ============================================================================
// Mask4 - 4-lane mask for WASM SIMD128
// ============================================================================

impl Default for Mask4 {
    #[inline(always)]
    fn default() -> Self {
        Self(u32x4(0, 0, 0, 0))
    }
}

impl Debug for Mask4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr: [u32; 4] = unsafe { core::mem::transmute(self.0) };
        let bits = (if arr[0] != 0 { 1 } else { 0 })
            | (if arr[1] != 0 { 2 } else { 0 })
            | (if arr[2] != 0 { 4 } else { 0 })
            | (if arr[3] != 0 { 8 } else { 0 });
        write!(f, "Mask4({:04b})", bits)
    }
}

impl MaskOps for Mask4 {
    #[inline(always)]
    fn any(self) -> bool {
        v128_any_true(self.0)
    }

    #[inline(always)]
    fn all(self) -> bool {
        // v128_all_true is not available, check lane-wise
        u32x4_all_true(self.0)
    }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(v128_not(self.0))
    }
}

// ============================================================================
// F32x4 - 4-lane f32 for WASM SIMD128
// ============================================================================

impl Default for F32x4 {
    #[inline(always)]
    fn default() -> Self {
        Self(f32x4_splat(0.0))
    }
}

impl Debug for F32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr: [f32; 4] = unsafe { core::mem::transmute(self.0) };
        write!(f, "F32x4({:?})", arr)
    }
}

impl F32x4 {
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut arr = [0.0f32; 4];
        unsafe { v128_store(arr.as_mut_ptr() as *mut v128, self.0) };
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
        Self(f32x4_splat(val))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self(f32x4(start, start + 1.0, start + 2.0, start + 3.0))
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_lt(self.0, rhs.0))
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_le(self.0, rhs.0))
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_gt(self.0, rhs.0))
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_ge(self.0, rhs.0))
    }

    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        Self(f32x4_sqrt(self.0))
    }

    #[inline(always)]
    fn simd_abs(self) -> Self {
        Self(f32x4_abs(self.0))
    }

    #[inline(always)]
    fn simd_min(self, rhs: Self) -> Self {
        Self(f32x4_min(self.0, rhs.0))
    }

    #[inline(always)]
    fn simd_max(self, rhs: Self) -> Self {
        Self(f32x4_max(self.0, rhs.0))
    }

    #[inline(always)]
    fn simd_select(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(v128_load(slice.as_ptr() as *const v128)) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // WASM doesn't have gather - do scalar loads
        let idx = indices.to_array();
        let len = slice.len();
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let ix = (idx[i] as isize).clamp(0, len as isize - 1) as usize;
            out[i] = slice[ix];
        }
        Self(f32x4(out[0], out[1], out[2], out[3]))
    }

    #[inline(always)]
    fn simd_floor(self) -> Self {
        Self(f32x4_floor(self.0))
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // WASM has no native FMA
        Self(f32x4_add(f32x4_mul(self.0, b.0), c.0))
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask4) -> Self {
        // Emulate with select: self + (mask ? val : 0)
        let zero = f32x4_splat(0.0);
        let masked_val = v128_bitselect(val.0, zero, mask.0);
        Self(f32x4_add(self.0, masked_val))
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // WASM has no reciprocal estimate
        Self(f32x4_div(f32x4_splat(1.0), self.0))
    }

    #[inline(always)]
    fn simd_rsqrt(self) -> Self {
        // WASM has no rsqrt estimate
        let sqrt = f32x4_sqrt(self.0);
        Self(f32x4_div(f32x4_splat(1.0), sqrt))
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask4) -> Self {
        // Bitcast mask to float
        Self(mask.0)
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask4 {
        // Bitcast float to mask
        Mask4(self.0)
    }

    #[inline(always)]
    fn from_u32_bits(bits: u32) -> Self {
        Self(u32x4_splat(bits))
    }

    #[inline(always)]
    fn shr_u32(self, n: u32) -> Self {
        Self(u32x4_shr(self.0, n))
    }

    #[inline(always)]
    fn i32_to_f32(self) -> Self {
        Self(f32x4_convert_i32x4(self.0))
    }

    #[inline(always)]
    fn log2(self) -> Self {
        // Uses range [√2/2, √2] centered at 1 for better polynomial accuracy
        // log2(x) = exponent + log2(mantissa)
        let x_u32 = self.0;

        // Extract exponent: (bits >> 23) - 127
        let exp_bits = u32x4_shr(x_u32, 23);
        let bias = i32x4_splat(127);
        let mut n = f32x4_convert_i32x4(i32x4_sub(exp_bits, bias));

        // Extract mantissa in [1, 2): (bits & 0x007FFFFF) | 0x3F800000
        let mant_mask = u32x4_splat(0x007FFFFF);
        let one_bits = u32x4_splat(0x3F800000);
        let mut f = v128_or(v128_and(x_u32, mant_mask), one_bits);

        // Adjust to [√2/2, √2] range for better accuracy (centered at 1)
        // If f >= √2, divide by 2 and increment exponent
        let sqrt2 = f32x4_splat(1.4142135624);
        let mask = f32x4_ge(f, sqrt2);
        let adjust = v128_and(mask, f32x4_splat(1.0));
        n = f32x4_add(n, adjust);
        f = v128_or(
            v128_and(mask, f32x4_mul(f, f32x4_splat(0.5))),
            v128_andnot(mask, f),
        );

        // Polynomial for log2(f) on [√2/2, √2]
        // Fitted using least squares on Chebyshev nodes
        // Max error: ~1e-4
        let c4 = f32x4_splat(-0.3200435159);
        let c3 = f32x4_splat(1.7974969154);
        let c2 = f32x4_splat(-4.1988046176);
        let c1 = f32x4_splat(5.7270231695);
        let c0 = f32x4_splat(-3.0056146714);

        // Horner's method (no FMA)
        let poly = f32x4_add(f32x4_mul(c4, f), c3);
        let poly = f32x4_add(f32x4_mul(poly, f), c2);
        let poly = f32x4_add(f32x4_mul(poly, f), c1);
        let poly = f32x4_add(f32x4_mul(poly, f), c0);

        Self(f32x4_add(n, poly))
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // 2^x = 2^n * 2^f where n = floor(x), f = frac(x) ∈ [0, 1)
        let n = f32x4_floor(self.0);
        let f = f32x4_sub(self.0, n);

        // Minimax polynomial for 2^f, f ∈ [0, 1)
        let c4 = f32x4_splat(0.0135557);
        let c3 = f32x4_splat(0.0520323);
        let c2 = f32x4_splat(0.2413793);
        let c1 = f32x4_splat(0.6931472);
        let c0 = f32x4_splat(1.0);

        // Horner's method
        let poly = f32x4_add(f32x4_mul(c4, f), c3);
        let poly = f32x4_add(f32x4_mul(poly, f), c2);
        let poly = f32x4_add(f32x4_mul(poly, f), c1);
        let poly = f32x4_add(f32x4_mul(poly, f), c0);

        // Compute 2^n by adding n to exponent bits
        // 2^n = reinterpret((n + 127) << 23)
        let bias = i32x4_splat(127);
        let n_i32 = i32x4_trunc_sat_f32x4(n);
        let exp_bits = i32x4_shl(i32x4_add(n_i32, bias), 23);
        let scale = exp_bits;

        Self(f32x4_mul(poly, scale))
    }

    #[inline(always)]
    fn sin(self) -> Self {
        const PI: f32 = core::f32::consts::PI;
        const TWO_PI: f32 = core::f32::consts::TAU;
        const TWO_PI_INV: f32 = 1.0 / TWO_PI;
        const PI_INV: f32 = 1.0 / PI;

        // Range reduce to [-π, π]
        let k = f32x4_floor(f32x4_add(
            f32x4_mul(self.0, f32x4_splat(TWO_PI_INV)),
            f32x4_splat(0.5),
        ));
        let x = f32x4_sub(self.0, f32x4_mul(k, f32x4_splat(TWO_PI)));
        let t = f32x4_mul(x, f32x4_splat(PI_INV));

        let c1 = f32x4_splat(1.6719970703125);
        let c3 = f32x4_splat(-0.645963541666667);
        let c5 = f32x4_splat(0.079689450);
        let c7 = f32x4_splat(-0.0046817541);

        let t2 = f32x4_mul(t, t);
        let mut poly = f32x4_add(f32x4_mul(c7, t2), c5);
        poly = f32x4_add(f32x4_mul(poly, t2), c3);
        poly = f32x4_add(f32x4_mul(poly, t2), c1);
        Self(f32x4_mul(poly, t))
    }

    #[inline(always)]
    fn cos(self) -> Self {
        const PI: f32 = core::f32::consts::PI;
        const TWO_PI: f32 = core::f32::consts::TAU;
        const TWO_PI_INV: f32 = 1.0 / TWO_PI;
        const PI_INV: f32 = 1.0 / PI;

        let k = f32x4_floor(f32x4_add(
            f32x4_mul(self.0, f32x4_splat(TWO_PI_INV)),
            f32x4_splat(0.5),
        ));
        let x = f32x4_sub(self.0, f32x4_mul(k, f32x4_splat(TWO_PI)));
        let t = f32x4_mul(x, f32x4_splat(PI_INV));

        let c0 = f32x4_splat(1.5707963267948966);
        let c2 = f32x4_splat(-2.467401341);
        let c4 = f32x4_splat(0.609469381);
        let c6 = f32x4_splat(-0.038854038);

        let t2 = f32x4_mul(t, t);
        let mut poly = f32x4_add(f32x4_mul(c6, t2), c4);
        poly = f32x4_add(f32x4_mul(poly, t2), c2);
        Self(f32x4_add(f32x4_mul(poly, t2), c0))
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        const PI: f32 = core::f32::consts::PI;
        const PI_2: f32 = core::f32::consts::FRAC_PI_2;

        let y = self.0;
        let x_val = x.0;

        let r = f32x4_div(y, x_val);
        let r_abs = f32x4_abs(r);

        let c1 = f32x4_splat(0.999999999);
        let c3 = f32x4_splat(-0.333333333);
        let c5 = f32x4_splat(0.2);
        let c7 = f32x4_splat(-0.142857143);

        let t = r_abs;
        let t2 = f32x4_mul(t, t);
        let mut poly = f32x4_add(f32x4_mul(c7, t2), c5);
        poly = f32x4_add(f32x4_mul(poly, t2), c3);
        poly = f32x4_add(f32x4_mul(poly, t2), c1);
        let atan_approx = f32x4_mul(poly, t);

        let one = f32x4_splat(1.0);
        let mask_large = f32x4_gt(r_abs, one);
        let recip_r = f32x4_div(one, r_abs);
        let atan_large = f32x4_sub(f32x4_splat(PI_2), f32x4_mul(recip_r, atan_approx));
        let atan_val = v128_bitselect(atan_large, atan_approx, mask_large);

        let y_abs = f32x4_abs(y);
        let sign_y = f32x4_div(y_abs, y);
        let atan_signed = f32x4_mul(atan_val, sign_y);

        let zero = f32x4_splat(0.0);
        let mask_neg_x = f32x4_lt(x_val, zero);
        let correction = f32x4_mul(f32x4_splat(PI), sign_y);
        Self(v128_bitselect(
            f32x4_sub(atan_signed, correction),
            atan_signed,
            mask_neg_x,
        ))
    }

    #[inline(always)]
    fn cmp_eq(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_eq(self.0, rhs.0))
    }

    #[inline(always)]
    fn cmp_ne(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_ne(self.0, rhs.0))
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(f32x4_add(self.0, rhs.0))
    }
}

impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }
}

impl Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(f32x4_mul(self.0, rhs.0))
    }
}

impl Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(f32x4_div(self.0, rhs.0))
    }
}

impl BitAnd for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl BitOr for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl Not for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(v128_not(self.0))
    }
}

impl Neg for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(f32x4_neg(self.0))
    }
}

// ============================================================================
// U32x4 - 4-lane u32 SIMD for packed RGBA pixels
// ============================================================================

impl Default for U32x4 {
    #[inline(always)]
    fn default() -> Self {
        Self(u32x4_splat(0))
    }
}

impl Debug for U32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr: [u32; 4] = unsafe { core::mem::transmute(self.0) };
        write!(f, "U32x4({:?})", arr)
    }
}

impl SimdU32Ops for U32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        Self(u32x4_splat(val))
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // This is a placeholder
        Self::default()
    }
}

impl BitAnd for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl BitOr for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl Not for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(v128_not(self.0))
    }
}

impl Shl<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        Self(u32x4_shl(self.0, rhs))
    }
}

impl Shr<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        Self(u32x4_shr(self.0, rhs))
    }
}

impl U32x4 {
    /// Pack 4 f32 Fields (RGBA) into packed u32 pixels.
    #[inline(always)]
    pub fn pack_rgba(r: F32x4, g: F32x4, b: F32x4, a: F32x4) -> Self {
        // Clamp to [0, 1] and scale to [0, 255]
        let scale = f32x4_splat(255.0);
        let zero = f32x4_splat(0.0);
        let one = f32x4_splat(1.0);

        let r_clamped = f32x4_min(f32x4_max(r.0, zero), one);
        let g_clamped = f32x4_min(f32x4_max(g.0, zero), one);
        let b_clamped = f32x4_min(f32x4_max(b.0, zero), one);
        let a_clamped = f32x4_min(f32x4_max(a.0, zero), one);

        let r_scaled = f32x4_mul(r_clamped, scale);
        let g_scaled = f32x4_mul(g_clamped, scale);
        let b_scaled = f32x4_mul(b_clamped, scale);
        let a_scaled = f32x4_mul(a_clamped, scale);

        // Convert to u32
        let r_u32 = u32x4_trunc_sat_f32x4(r_scaled);
        let g_u32 = u32x4_trunc_sat_f32x4(g_scaled);
        let b_u32 = u32x4_trunc_sat_f32x4(b_scaled);
        let a_u32 = u32x4_trunc_sat_f32x4(a_scaled);

        // Pack: R | (G << 8) | (B << 16) | (A << 24)
        let g_shifted = u32x4_shl(g_u32, 8);
        let b_shifted = u32x4_shl(b_u32, 16);
        let a_shifted = u32x4_shl(a_u32, 24);

        let packed = v128_or(v128_or(r_u32, g_shifted), v128_or(b_shifted, a_shifted));
        Self(packed)
    }
}
