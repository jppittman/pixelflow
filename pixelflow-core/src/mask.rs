//! # Native Mask Type
//!
//! `Mask` wraps the native SIMD mask type for the current platform.
//! This is conceptually `Field<bool>` - a SIMD batch of boolean values.
//!
//! ## Benefits
//!
//! Using native masks instead of float-encoded masks:
//! - **AVX-512**: Uses k-registers which run on separate execution unit (free)
//! - **No conversion overhead**: No `mask_to_float` / `float_to_mask` round-trips
//! - **Type safety**: Can't accidentally use a float where a mask is expected
//!
//! ## Usage
//!
//! ```ignore
//! // Native mask operations
//! let mask = x.lt_mask(y);  // Returns Mask, not Field
//! let result = Mask::select(mask, if_true, if_false);
//!
//! // Convert to/from Field when needed
//! let field_mask = mask.to_field();
//! let back = Mask::from_field(field_mask);
//! ```

use crate::Field;
use crate::backend::{MaskOps, SimdOps};
use crate::storage::NativeMaskStorage;
use core::ops::{BitAnd, BitOr, Not};

// Re-import NativeSimd for conversions
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
type NativeSimd = crate::backend::x86::F32x16;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
type NativeSimd = crate::backend::x86::F32x8;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
type NativeSimd = crate::backend::x86::F32x4;

#[cfg(target_arch = "aarch64")]
type NativeSimd = crate::backend::arm::F32x4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeSimd = crate::backend::scalar::ScalarF32;

/// A SIMD batch of boolean values using native mask storage.
///
/// This is conceptually `Field<bool>`. On AVX-512, this uses the dedicated
/// k-register file which runs on a separate execution unit, making mask
/// operations essentially free (parallel with float ALU).
///
/// ## When to Use
///
/// Use `Mask` when you need to:
/// - Store comparison results without conversion overhead
/// - Perform many boolean operations (AND, OR, NOT)
/// - Select between values based on conditions
///
/// Use `Field` (with float-encoded masks) when you need to:
/// - Mix mask and float operations seamlessly
/// - Use existing `Field`-based APIs
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Mask(pub(crate) NativeMaskStorage);

impl Mask {
    /// Create a mask with all lanes true.
    #[inline(always)]
    #[must_use]
    pub fn all_true() -> Self {
        Self(!NativeMaskStorage::default())
    }

    /// Create a mask with all lanes false.
    #[inline(always)]
    #[must_use]
    pub fn all_false() -> Self {
        Self(NativeMaskStorage::default())
    }

    /// Check if any lane is true.
    #[inline(always)]
    #[must_use]
    pub fn any(&self) -> bool {
        self.0.any()
    }

    /// Check if all lanes are true.
    #[inline(always)]
    #[must_use]
    pub fn all(&self) -> bool {
        self.0.all()
    }

    /// Check if no lanes are true.
    #[inline(always)]
    #[must_use]
    pub fn none(&self) -> bool {
        !self.0.any()
    }

    /// Convert to Field (float-encoded mask).
    ///
    /// Each lane becomes either all-1s bits (NaN) or all-0s bits (0.0).
    #[inline(always)]
    #[must_use]
    pub fn to_field(self) -> Field {
        Field(NativeSimd::mask_to_float(self.0))
    }

    /// Convert from Field (float-encoded mask).
    ///
    /// Non-zero lanes become true, zero lanes become false.
    #[inline(always)]
    #[must_use]
    pub fn from_field(field: Field) -> Self {
        Self(field.0.float_to_mask())
    }

    /// Branchless select: returns `if_true` where mask is set, `if_false` elsewhere.
    #[inline(always)]
    #[must_use]
    pub fn select(self, if_true: Field, if_false: Field) -> Field {
        Field(NativeSimd::simd_select(self.0, if_true.0, if_false.0))
    }

    /// Branchless select with early-exit optimization.
    ///
    /// If all lanes are true, returns `if_true` without blending.
    /// If no lanes are true, returns `if_false` without blending.
    #[inline(always)]
    #[must_use]
    pub fn select_opt(self, if_true: Field, if_false: Field) -> Field {
        if self.all() {
            return if_true;
        }
        if self.none() {
            return if_false;
        }
        self.select(if_true, if_false)
    }
}

// ============================================================================
// Boolean Operations
// ============================================================================

impl BitAnd for Mask {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for Mask {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl Not for Mask {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl From<Mask> for Field {
    #[inline(always)]
    fn from(mask: Mask) -> Self {
        mask.to_field()
    }
}

impl From<Field> for Mask {
    #[inline(always)]
    fn from(field: Field) -> Self {
        Self::from_field(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Field is a SIMD type; extract lane 0 to check scalar results in tests.
    fn lane0(f: Field) -> f32 {
        unsafe { *(&f as *const Field as *const f32) }
    }

    #[test]
    fn all_true_is_true_everywhere() {
        let m = Mask::all_true();
        assert!(m.all());
        assert!(m.any());
        assert!(!m.none());
    }

    #[test]
    fn all_false_is_false_everywhere() {
        let m = Mask::all_false();
        assert!(!m.all());
        assert!(!m.any());
        assert!(m.none());
    }

    #[test]
    fn not_inverts_all_true_and_all_false() {
        assert!((!Mask::all_true()).none());
        assert!((!Mask::all_false()).all());
    }

    #[test]
    fn bitand_truth_table() {
        assert!((Mask::all_true() & Mask::all_true()).all());
        assert!((Mask::all_true() & Mask::all_false()).none());
        assert!((Mask::all_false() & Mask::all_true()).none());
        assert!((Mask::all_false() & Mask::all_false()).none());
    }

    #[test]
    fn bitor_truth_table() {
        assert!((Mask::all_true() | Mask::all_true()).all());
        assert!((Mask::all_true() | Mask::all_false()).all());
        assert!((Mask::all_false() | Mask::all_true()).all());
        assert!((Mask::all_false() | Mask::all_false()).none());
    }

    #[test]
    fn select_picks_if_true_when_mask_all_true() {
        let result = Mask::all_true().select(Field::from(1.0), Field::from(2.0));
        assert_eq!(lane0(result), 1.0);
    }

    #[test]
    fn select_picks_if_false_when_mask_all_false() {
        let result = Mask::all_false().select(Field::from(1.0), Field::from(2.0));
        assert_eq!(lane0(result), 2.0);
    }

    #[test]
    fn select_opt_matches_select_on_uniform_masks() {
        assert_eq!(
            lane0(Mask::all_true().select_opt(Field::from(1.0), Field::from(2.0))),
            1.0
        );
        assert_eq!(
            lane0(Mask::all_false().select_opt(Field::from(1.0), Field::from(2.0))),
            2.0
        );
    }

    #[test]
    fn to_field_from_field_roundtrip() {
        assert!(Mask::from_field(Mask::all_true().to_field()).all());
        assert!(Mask::from_field(Mask::all_false().to_field()).none());
        assert!(Mask::from(Mask::all_true().to_field()).all());
        assert!(Mask::from(Mask::all_false().to_field()).none());
    }

    #[test]
    fn from_field_into_mask_conversion() {
        let field: Field = Mask::all_true().into();
        let mask: Mask = field.into();
        assert!(mask.all());
    }
}
