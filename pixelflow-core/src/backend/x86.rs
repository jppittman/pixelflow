//! x86_64 lane types: `Field`'s two constructors, one per ISA level.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::fmt::{Debug, Formatter};

// ============================================================================
// SSE2 (4 lanes) — selected when neither AVX2 nor AVX-512F is enabled.
// ============================================================================

/// 4-lane f32 SIMD vector for SSE2.
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct F32x4(__m128);

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
impl Debug for F32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "F32x4({:?})", self.to_array())
    }
}

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
impl F32x4 {
    pub(crate) const LANES: usize = 4;

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut arr = [0.0f32; 4];
        unsafe { _mm_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub(crate) fn splat(val: f32) -> Self {
        unsafe { Self(_mm_set1_ps(val)) }
    }

}

// ============================================================================
// AVX2 (8 lanes) — selected when AVX2 is enabled but not AVX-512F.
// ============================================================================

/// 8-lane f32 SIMD vector for AVX2.
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct F32x8(__m256);

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
impl Debug for F32x8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "F32x8({:?})", self.to_array())
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
impl F32x8 {
    pub(crate) const LANES: usize = 8;

    #[inline(always)]
    fn to_array(self) -> [f32; 8] {
        let mut arr = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub(crate) fn splat(val: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(val)) }
    }

}

// ============================================================================
// AVX-512 (16 lanes)
// ============================================================================

/// 16-lane f32 SIMD vector for AVX-512.
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct F32x16(__m512);

#[cfg(target_feature = "avx512f")]
impl Debug for F32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "F32x16({:?})", self.to_array())
    }
}

#[cfg(target_feature = "avx512f")]
impl F32x16 {
    pub(crate) const LANES: usize = 16;

    #[inline(always)]
    fn to_array(self) -> [f32; 16] {
        let mut arr = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub(crate) fn splat(val: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(val)) }
    }

}
