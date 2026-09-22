//! ARM NEON lane type: `Field`'s two constructors, at NEON's one width.

use core::arch::aarch64::*;
use core::fmt::{Debug, Formatter};

/// 4-lane f32 SIMD vector for ARM NEON.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct F32x4(float32x4_t);

impl Debug for F32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "F32x4({:?})", self.to_array())
    }
}

impl F32x4 {
    pub(crate) const LANES: usize = 4;

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut arr = [0.0f32; 4];
        unsafe { vst1q_f32(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub(crate) fn splat(val: f32) -> Self {
        unsafe { Self(vdupq_n_f32(val)) }
    }
}
