//! # Algebra Trait — Unified Type System for Field<A>
//!
//! This module defines the algebraic structure that any type must implement
//! to be carried by `Field`. The key insight is that SIMD-ification is orthogonal
//! to the algebra: `f32`, `bool`, `u32`, and `Dual<N>` are all algebras that can
//! be SIMD-batched.
//!
//! ## Design
//!
//! - **`Algebra`**: Core ring operations + comparisons + selection
//! - **`Transcendental`**: Extended operations for differentiable algebras
//! - **`Mask`**: Associated type for comparison results
//!
//! ## Type Mapping
//!
//! | Type | Mask | Description |
//! |------|------|-------------|
//! | `f32` | `bool` | Scalar floats |
//! | `bool` | `bool` | Boolean masks |
//! | `u32` | `bool` | Packed pixels |
//! | `Dual<N, A>` | `A::Mask` | Autodiff (mask from base) |

/// An algebraic structure that can be carried by Field.
///
/// This trait defines the minimal operations needed for a type to participate
/// in the PixelFlow computation model. The SIMD-ification happens at the Field
/// level, orthogonal to the algebra itself.
///
/// # Associated Types
///
/// - `Mask`: The type returned by comparisons. For `Dual<N, A>`, this is `A::Mask`,
///   not `Dual<N, A::Mask>`, because comparisons only compare values, not derivatives.
///
/// # Laws
///
/// Implementations should satisfy standard ring axioms:
/// - Addition is associative and commutative with `zero()` as identity
/// - Multiplication is associative with `one()` as identity
/// - Multiplication distributes over addition
/// - `neg(a)` is the additive inverse of `a`
pub trait Algebra: Sized + Copy + Send + Sync + 'static {
    /// The type returned by comparison operations.
    ///
    /// For scalar types like `f32`, this is `bool`.
    /// For dual numbers `Dual<N, A>`, this is `A::Mask` (comparisons ignore derivatives).
    type Mask: Algebra;

    /// The additive identity (zero element).
    fn zero() -> Self;

    /// The multiplicative identity (one element).
    fn one() -> Self;

    // ========================================================================
    // Ring Operations
    // ========================================================================

    /// Addition: `self + rhs`
    fn add(self, rhs: Self) -> Self;

    /// Subtraction: `self - rhs`
    fn sub(self, rhs: Self) -> Self;

    /// Multiplication: `self * rhs`
    fn mul(self, rhs: Self) -> Self;

    /// Additive inverse: `-self`
    fn neg(self) -> Self;

    // ========================================================================
    // Comparison → Mask
    // ========================================================================

    /// Less than: `self < rhs`
    fn lt(self, rhs: Self) -> Self::Mask;

    /// Less than or equal: `self <= rhs`
    fn le(self, rhs: Self) -> Self::Mask;

    /// Greater than: `self > rhs`
    fn gt(self, rhs: Self) -> Self::Mask;

    /// Greater than or equal: `self >= rhs`
    fn ge(self, rhs: Self) -> Self::Mask;

    /// Equality: `self == rhs`
    fn eq(self, rhs: Self) -> Self::Mask;

    /// Inequality: `self != rhs`
    fn ne(self, rhs: Self) -> Self::Mask;

    // ========================================================================
    // Selection (Branchless)
    // ========================================================================

    /// Branchless conditional select.
    ///
    /// Returns `if_true` where `mask` is true, `if_false` elsewhere.
    /// This is the fundamental branching primitive for SIMD computation.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    // ========================================================================
    // Division (Optional, with default)
    // ========================================================================

    /// Division: `self / rhs`
    ///
    /// Default implementation uses multiplication by reciprocal.
    /// Types with native division can override.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self
    where
        Self: Transcendental,
    {
        self.mul(rhs.recip())
    }
}

/// Extended operations for differentiable algebras.
///
/// These operations are meaningful for continuous numeric types that support
/// calculus-like operations. Boolean and integer algebras don't implement this.
///
/// # Chain Rule
///
/// For dual numbers, these operations must propagate derivatives correctly:
/// - `sqrt(f)' = f' / (2 * sqrt(f))`
/// - `sin(f)' = cos(f) * f'`
/// - `exp(f)' = exp(f) * f'`
/// - etc.
pub trait Transcendental: Algebra {
    /// Square root: `√self`
    fn sqrt(self) -> Self;

    /// Absolute value: `|self|`
    fn abs(self) -> Self;

    /// Reciprocal: `1 / self`
    fn recip(self) -> Self;

    /// Reciprocal square root: `1 / √self`
    ///
    /// This is often faster than `sqrt` followed by `recip` due to
    /// dedicated SIMD instructions (rsqrt).
    fn rsqrt(self) -> Self;

    /// Sine: `sin(self)`
    fn sin(self) -> Self;

    /// Cosine: `cos(self)`
    fn cos(self) -> Self;

    /// Two-argument arctangent: `atan2(self, x)`
    ///
    /// Returns the angle in radians between the positive x-axis and the
    /// point `(x, self)`.
    fn atan2(self, x: Self) -> Self;

    /// Natural exponential: `e^self`
    fn exp(self) -> Self;

    /// Natural logarithm: `ln(self)`
    fn ln(self) -> Self;

    /// Base-2 exponential: `2^self`
    fn exp2(self) -> Self;

    /// Base-2 logarithm: `log2(self)`
    fn log2(self) -> Self;

    /// Power: `self^exp`
    fn pow(self, exp: Self) -> Self;

    /// Floor: round toward negative infinity
    fn floor(self) -> Self;

    /// Ceiling: round toward positive infinity
    #[inline(always)]
    fn ceil(self) -> Self {
        self.neg().floor().neg()
    }

    /// Minimum: `min(self, rhs)`
    fn min(self, rhs: Self) -> Self;

    /// Maximum: `max(self, rhs)`
    fn max(self, rhs: Self) -> Self;

    /// Clamp: `clamp(self, lo, hi)`
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Fused multiply-add: `self * a + b`
    ///
    /// May use hardware FMA for better precision and performance.
    fn mul_add(self, a: Self, b: Self) -> Self;
}

// ============================================================================
// Implementations for Primitive Types
// ============================================================================

impl Algebra for f32 {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn one() -> Self {
        1.0
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> bool {
        self < rhs
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> bool {
        self <= rhs
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> bool {
        self > rhs
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> bool {
        self >= rhs
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> bool {
        self == rhs
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> bool {
        self != rhs
    }

    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
}

// NOTE: Transcendental for f32 is only implemented on scalar fallback platforms
// where libm is available. On x86_64/aarch64, transcendental operations happen
// through SIMD Field types, not scalar f32.
//
// This is intentional: the Algebra trait provides ring operations that work
// everywhere, while Transcendental is for the SIMD-accelerated compute path.

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Transcendental for f32 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        1.0 / libm::sqrtf(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        libm::sinf(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        libm::cosf(self)
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        libm::atan2f(self, x)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        libm::expf(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        libm::logf(self)
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        libm::exp2f(self)
    }

    #[inline(always)]
    fn log2(self) -> Self {
        libm::log2f(self)
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        libm::powf(self, exp)
    }

    #[inline(always)]
    fn floor(self) -> Self {
        libm::floorf(self)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        if self < rhs { self } else { rhs }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        if self > rhs { self } else { rhs }
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
}

impl Algebra for bool {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self {
        false
    }

    #[inline(always)]
    fn one() -> Self {
        true
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self | rhs // OR for boolean addition
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self & !rhs // AND NOT for boolean subtraction
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self & rhs // AND for boolean multiplication
    }

    #[inline(always)]
    fn neg(self) -> Self {
        !self
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> bool {
        !self & rhs // false < true
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> bool {
        !self | rhs
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> bool {
        self & !rhs
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> bool {
        self | !rhs
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> bool {
        self == rhs
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> bool {
        self != rhs
    }

    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}

impl Algebra for u32 {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }

    #[inline(always)]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> bool {
        self < rhs
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> bool {
        self <= rhs
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> bool {
        self > rhs
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> bool {
        self >= rhs
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> bool {
        self == rhs
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> bool {
        self != rhs
    }

    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_algebra() {
        // Ring operations
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(2.0f32.add(3.0), 5.0);
        assert_eq!(5.0f32.sub(3.0), 2.0);
        assert_eq!(2.0f32.mul(3.0), 6.0);
        assert_eq!(5.0f32.neg(), -5.0);

        // Comparisons
        assert!(2.0f32.lt(3.0));
        assert!(!3.0f32.lt(2.0));
        assert!(2.0f32.le(2.0));
        assert!(3.0f32.gt(2.0));
        assert!(2.0f32.ge(2.0));

        // Select
        assert_eq!(f32::select(true, 1.0, 2.0), 1.0);
        assert_eq!(f32::select(false, 1.0, 2.0), 2.0);
    }

    // Transcendental tests only run on scalar fallback platforms
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_f32_transcendental() {
        let epsilon = 1e-6;

        assert!((4.0f32.sqrt() - 2.0).abs() < epsilon);
        assert_eq!((-3.0f32).abs(), 3.0);
        assert!((2.0f32.recip() - 0.5).abs() < epsilon);

        // Trig
        assert!(0.0f32.sin().abs() < epsilon);
        assert!((0.0f32.cos() - 1.0).abs() < epsilon);

        // Exp/log
        assert!((1.0f32.exp() - core::f32::consts::E).abs() < epsilon);
        assert!(core::f32::consts::E.ln().sub(1.0).abs() < epsilon);

        // Min/max
        assert_eq!(2.0f32.min(3.0), 2.0);
        assert_eq!(2.0f32.max(3.0), 3.0);
    }

    #[test]
    fn test_bool_algebra() {
        assert_eq!(bool::zero(), false);
        assert_eq!(bool::one(), true);

        // Boolean ring (OR/AND)
        assert_eq!(false.add(false), false);
        assert_eq!(false.add(true), true);
        assert_eq!(true.add(false), true);
        assert_eq!(true.add(true), true);

        assert_eq!(false.mul(false), false);
        assert_eq!(false.mul(true), false);
        assert_eq!(true.mul(true), true);

        assert_eq!(true.neg(), false);
        assert_eq!(false.neg(), true);
    }

    #[test]
    fn test_u32_algebra() {
        assert_eq!(u32::zero(), 0);
        assert_eq!(u32::one(), 1);
        assert_eq!(2u32.add(3), 5);
        assert_eq!(5u32.sub(3), 2);
        assert_eq!(2u32.mul(3), 6);

        // Wrapping arithmetic
        assert_eq!(0u32.sub(1), u32::MAX);
    }
}
