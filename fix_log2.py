import os

file_path = "pixelflow-core/src/backend/x86.rs"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Make sure all blocks use range [1, 2)
# Revert to the original polynomial coefficients
replacement = """mod log2_poly {
    pub const C4: f32 = -0.360674;
    pub const C3: f32 = 1.9237;
    pub const C2: f32 = -4.3282;
    pub const C1: f32 = 5.7708;
    pub const C0: f32 = -3.0056;
}"""

search = """mod log2_poly {
    pub const C4: f32 = -0.320_043_5;
    pub const C3: f32 = 1.797_496_9;
    pub const C2: f32 = -4.198_805;
    pub const C1: f32 = 5.727_023;
    pub const C0: f32 = -3.005_614_8;
}"""

if search in text:
    text = text.replace(search, replacement)
    print("Reverted to original polynomial coefficients")


replacement_fn = """    fn log2(self) -> Self {
        // SSE2: Use bit manipulation for exponent/mantissa extraction
        // Uses range [1, 2)
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
            let n = _mm_cvtepi32_ps(exp_unbiased);

            // Extract mantissa in [1, 2)
            let mant_mask = _mm_set1_epi32(0x007FFFFF_u32 as i32);
            let one_bits = _mm_set1_epi32(0x3F800000_u32 as i32);
            let f = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(x_i32, mant_mask), one_bits));

            // Polynomial for log2(f) on [1, 2)
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
    }"""

search_fn = """    fn log2(self) -> Self {
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
    }"""

if search_fn in text:
    text = text.replace(search_fn, replacement_fn)
    print("Reverted SSE2 log2 to simple [1, 2) logic")


replacement_fn_avx2 = """    fn log2(self) -> Self {
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
            let n = _mm256_cvtepi32_ps(exp_unbiased);

            // Extract mantissa in [1, 2)
            let mant_mask = _mm256_set1_epi32(0x007FFFFF_u32 as i32);
            let one_bits = _mm256_set1_epi32(0x3F800000_u32 as i32);
            let f = _mm256_castsi256_ps(_mm256_or_si256(
                _mm256_and_si256(x_i32, mant_mask),
                one_bits,
            ));

            // Polynomial for log2(f) on [1, 2)
            // Fitted using least squares on Chebyshev nodes
            // Max error: ~1e-4
            let c4 = _mm256_set1_ps(log2_poly::C4);
            let c3 = _mm256_set1_ps(log2_poly::C3);
            let c2 = _mm256_set1_ps(log2_poly::C2);
            let c1 = _mm256_set1_ps(log2_poly::C1);
            let c0 = _mm256_set1_ps(log2_poly::C0);

            // Horner's method with FMA
            let mut poly = _mm256_fmadd_ps(c4, f, c3);
            poly = _mm256_fmadd_ps(poly, f, c2);
            poly = _mm256_fmadd_ps(poly, f, c1);
            poly = _mm256_fmadd_ps(poly, f, c0);

            // Return exponent + polynomial
            Self(_mm256_add_ps(n, poly))
        }
    }"""

search_fn_avx2 = """    fn log2(self) -> Self {
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

            // Horner's method with FMA
            let mut poly = _mm256_fmadd_ps(c4, f, c3);
            poly = _mm256_fmadd_ps(poly, f, c2);
            poly = _mm256_fmadd_ps(poly, f, c1);
            poly = _mm256_fmadd_ps(poly, f, c0);

            // Return exponent + polynomial
            Self(_mm256_add_ps(n, poly))
        }
    }"""

if search_fn_avx2 in text:
    text = text.replace(search_fn_avx2, replacement_fn_avx2)
    print("Reverted AVX2 log2 to simple [1, 2) logic")

replacement_fn_avx512 = """    fn log2(self) -> Self {
        unsafe {
            // Extract mantissa in [1, 2) - no exponent adjustment needed
            // Interval=0 (_MM_MANT_NORM_1_2), sign=0 (_MM_MANT_SIGN_src)
            let f = _mm512_getmant_ps::<0, 0>(self.0);

            // Extract exponent
            let n = _mm512_getexp_ps(self.0);

            // Polynomial for log2(f) on [1, 2)
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

            // Return exponent + polynomial
            Self(_mm512_add_ps(n, poly))
        }
    }"""

search_fn_avx512 = """    fn log2(self) -> Self {
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

            // Return exponent + polynomial
            Self(_mm512_add_ps(n, poly))
        }
    }"""

if search_fn_avx512 in text:
    text = text.replace(search_fn_avx512, replacement_fn_avx512)
    print("Reverted AVX512 log2 to simple [1, 2) logic")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)
