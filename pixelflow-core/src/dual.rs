//! # Dual Numbers for Automatic Differentiation
//!
//! This module provides `Dual<N, A>`, a generic dual number type that carries
//! a value and N partial derivatives. This unifies `Jet2`, `Jet3`, and other
//! autodiff types under a single parameterized implementation.
//!
//! ## Type Aliases
//!
//! For ergonomics, type aliases are provided:
//! - `Dual1<A>` = `Dual<1, A>` — 1D autodiff (ray marching)
//! - `Dual2<A>` = `Dual<2, A>` — 2D autodiff (antialiasing)
//! - `Dual3<A>` = `Dual<3, A>` — 3D autodiff (surface normals)
//!
//! ## Chain Rule
//!
//! All operations automatically propagate derivatives:
//! - `(f + g)' = f' + g'`
//! - `(f * g)' = f' * g + f * g'` (product rule)
//! - `(f / g)' = (f' * g - f * g') / g²` (quotient rule)
//! - `sqrt(f)' = f' / (2 * sqrt(f))`
//! - etc.
//!
//! ## Mask Type
//!
//! Comparisons return `A::Mask`, not `Dual<N, A::Mask>`. This is because
//! comparisons only compare values, ignoring derivatives. The mask type
//! comes from the base algebra.

use crate::algebra::{Algebra, Transcendental};

/// A dual number with N partial derivatives over base algebra A.
///
/// Dual numbers are the foundation of forward-mode automatic differentiation.
/// A `Dual<N, A>` represents a value `v` along with its partial derivatives
/// `∂v/∂x₀, ∂v/∂x₁, ..., ∂v/∂xₙ₋₁`.
///
/// # Type Parameters
///
/// - `N`: Number of partial derivatives (compile-time constant)
/// - `A`: Base algebra (defaults to `f32`)
///
/// # Example
///
/// ```ignore
/// // Create a 2D dual number seeded for x-differentiation
/// let x = Dual2::var::<0>(3.0);  // x = 3, ∂x/∂x = 1, ∂x/∂y = 0
/// let y = Dual2::var::<1>(4.0);  // y = 4, ∂y/∂x = 0, ∂y/∂y = 1
///
/// let r = (x * x + y * y).sqrt();
/// // r.val = 5.0
/// // r.partials[0] = 0.6 (∂r/∂x = x/r)
/// // r.partials[1] = 0.8 (∂r/∂y = y/r)
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Dual<const N: usize, A: Algebra = f32> {
    /// The function value
    pub val: A,
    /// Partial derivatives [∂f/∂x₀, ∂f/∂x₁, ..., ∂f/∂xₙ₋₁]
    pub partials: [A; N],
}

/// Type alias for 1D dual numbers (ray marching, path derivatives)
pub type Dual1<A = f32> = Dual<1, A>;

/// Type alias for 2D dual numbers (antialiasing, 2D gradients)
pub type Dual2<A = f32> = Dual<2, A>;

/// Type alias for 3D dual numbers (surface normals, 3D gradients)
pub type Dual3<A = f32> = Dual<3, A>;

impl<const N: usize, A: Algebra> Dual<N, A> {
    /// Create a constant dual number (value with zero derivatives).
    #[inline(always)]
    pub fn constant(val: A) -> Self {
        Self {
            val,
            partials: [A::zero(); N],
        }
    }

    /// Create a dual number seeded for the I-th variable.
    ///
    /// This creates a dual number where the value is `val` and the I-th
    /// partial derivative is 1, all others are 0.
    ///
    /// # Panics
    ///
    /// Compile-time error if `I >= N`.
    #[inline(always)]
    pub fn var<const I: usize>(val: A) -> Self
    where
        [(); N]: Sized,
    {
        // Static assertion that I < N
        const { assert!(I < N, "Variable index out of bounds") };

        let mut partials = [A::zero(); N];
        partials[I] = A::one();
        Self { val, partials }
    }

    /// Create a dual number with explicit partials.
    #[inline(always)]
    pub fn new(val: A, partials: [A; N]) -> Self {
        Self { val, partials }
    }

    /// Map a function over the partials array.
    #[allow(dead_code)]
    #[inline(always)]
    fn map_partials<F>(self, f: F) -> Self
    where
        F: Fn(A) -> A,
    {
        Self {
            val: self.val,
            partials: core::array::from_fn(|i| f(self.partials[i])),
        }
    }

    /// Zip two partials arrays with a binary function.
    #[inline(always)]
    fn zip_partials<F>(self, other: Self, f: F) -> [A; N]
    where
        F: Fn(A, A) -> A,
    {
        core::array::from_fn(|i| f(self.partials[i], other.partials[i]))
    }
}

// Convenience constructors for Dual2 (most common case)
impl<A: Algebra> Dual<2, A> {
    /// Create a Dual2 seeded for X differentiation.
    #[inline(always)]
    pub fn x(val: A) -> Self {
        Self::var::<0>(val)
    }

    /// Create a Dual2 seeded for Y differentiation.
    #[inline(always)]
    pub fn y(val: A) -> Self {
        Self::var::<1>(val)
    }

    /// Get the X partial derivative.
    #[inline(always)]
    pub fn dx(&self) -> A {
        self.partials[0]
    }

    /// Get the Y partial derivative.
    #[inline(always)]
    pub fn dy(&self) -> A {
        self.partials[1]
    }
}

// Convenience constructors for Dual3
impl<A: Algebra> Dual<3, A> {
    /// Create a Dual3 seeded for X differentiation.
    #[inline(always)]
    pub fn x(val: A) -> Self {
        Self::var::<0>(val)
    }

    /// Create a Dual3 seeded for Y differentiation.
    #[inline(always)]
    pub fn y(val: A) -> Self {
        Self::var::<1>(val)
    }

    /// Create a Dual3 seeded for Z differentiation.
    #[inline(always)]
    pub fn z(val: A) -> Self {
        Self::var::<2>(val)
    }

    /// Get the X partial derivative.
    #[inline(always)]
    pub fn dx(&self) -> A {
        self.partials[0]
    }

    /// Get the Y partial derivative.
    #[inline(always)]
    pub fn dy(&self) -> A {
        self.partials[1]
    }

    /// Get the Z partial derivative.
    #[inline(always)]
    pub fn dz(&self) -> A {
        self.partials[2]
    }

    /// Compute the normalized gradient (unit normal vector).
    ///
    /// Returns `(dx, dy, dz) / ||(dx, dy, dz)||`
    #[inline(always)]
    pub fn normal(&self) -> (A, A, A)
    where
        A: Transcendental,
    {
        let dx = self.partials[0];
        let dy = self.partials[1];
        let dz = self.partials[2];
        let len_sq = dx.mul(dx).add(dy.mul(dy)).add(dz.mul(dz));
        let inv_len = len_sq.rsqrt();
        (dx.mul(inv_len), dy.mul(inv_len), dz.mul(inv_len))
    }
}

// ============================================================================
// Algebra Implementation
// ============================================================================

impl<const N: usize, A: Algebra> Algebra for Dual<N, A> {
    /// Comparisons return the base algebra's mask, not a dual mask.
    /// This is because comparisons only compare values, ignoring derivatives.
    type Mask = A::Mask;

    #[inline(always)]
    fn zero() -> Self {
        Self::constant(A::zero())
    }

    #[inline(always)]
    fn one() -> Self {
        Self::constant(A::one())
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // (f + g)' = f' + g'
        Self {
            val: self.val.add(rhs.val),
            partials: self.zip_partials(rhs, A::add),
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // (f - g)' = f' - g'
        Self {
            val: self.val.sub(rhs.val),
            partials: self.zip_partials(rhs, A::sub),
        }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        Self {
            val: self.val.mul(rhs.val),
            partials: core::array::from_fn(|i| {
                self.partials[i]
                    .mul(rhs.val)
                    .add(self.val.mul(rhs.partials[i]))
            }),
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        // (-f)' = -f'
        Self {
            val: self.val.neg(),
            partials: core::array::from_fn(|i| self.partials[i].neg()),
        }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        // Comparisons only compare values
        self.val.lt(rhs.val)
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self::Mask {
        self.val.le(rhs.val)
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        self.val.gt(rhs.val)
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self::Mask {
        self.val.ge(rhs.val)
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        self.val.eq(rhs.val)
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self::Mask {
        self.val.ne(rhs.val)
    }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        // Select applies to both value and all partials
        Self {
            val: A::select(mask, if_true.val, if_false.val),
            partials: core::array::from_fn(|i| {
                A::select(mask, if_true.partials[i], if_false.partials[i])
            }),
        }
    }
}

// ============================================================================
// Transcendental Implementation
// ============================================================================

// Transcendental for Dual<N, A> requires A: Transcendental.
// On x86_64/aarch64, f32 doesn't implement Transcendental (operations go through SIMD).
// This impl is primarily for:
// 1. Scalar fallback platforms where f32: Transcendental
// 2. Future use with Field as the base type
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl<const N: usize, A: Transcendental> Transcendental for Dual<N, A> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Chain rule: (√f)' = f' / (2√f) = f' * rsqrt(f) / 2
        let sqrt_val = self.val.sqrt();
        let half_rsqrt = self.val.rsqrt().mul(A::one().div(A::one().add(A::one())));
        Self {
            val: sqrt_val,
            partials: core::array::from_fn(|i| self.partials[i].mul(half_rsqrt)),
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        let sign = self.val.div(self.val.abs());
        Self {
            val: self.val.abs(),
            partials: core::array::from_fn(|i| self.partials[i].mul(sign)),
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // (1/f)' = -f' / f²
        let inv = self.val.recip();
        let neg_inv_sq = inv.mul(inv).neg();
        Self {
            val: inv,
            partials: core::array::from_fn(|i| self.partials[i].mul(neg_inv_sq)),
        }
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // (1/√f)' = -f' / (2 * f^(3/2)) = -f' * rsqrt(f)³ / 2
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val.mul(rsqrt_val).mul(rsqrt_val);
        let scale = rsqrt_cubed.neg().mul(A::one().div(A::one().add(A::one())));
        Self {
            val: rsqrt_val,
            partials: core::array::from_fn(|i| self.partials[i].mul(scale)),
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        // (sin f)' = cos(f) * f'
        let cos_val = self.val.cos();
        Self {
            val: self.val.sin(),
            partials: core::array::from_fn(|i| self.partials[i].mul(cos_val)),
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        // (cos f)' = -sin(f) * f'
        let neg_sin = self.val.sin().neg();
        Self {
            val: self.val.cos(),
            partials: core::array::from_fn(|i| self.partials[i].mul(neg_sin)),
        }
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        // atan2(y, x) derivatives:
        // ∂/∂y = x / (x² + y²)
        // ∂/∂x = -y / (x² + y²)
        let r_sq = self.val.mul(self.val).add(x.val.mul(x.val));
        let inv_r_sq = r_sq.recip();
        let dy_coeff = x.val.mul(inv_r_sq);
        let dx_coeff = self.val.neg().mul(inv_r_sq);
        Self {
            val: self.val.atan2(x.val),
            partials: core::array::from_fn(|i| {
                self.partials[i]
                    .mul(dy_coeff)
                    .add(x.partials[i].mul(dx_coeff))
            }),
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        // (e^f)' = e^f * f'
        let exp_val = self.val.exp();
        Self {
            val: exp_val,
            partials: core::array::from_fn(|i| self.partials[i].mul(exp_val)),
        }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        // (ln f)' = f' / f
        let inv_val = self.val.recip();
        Self {
            val: self.val.ln(),
            partials: core::array::from_fn(|i| self.partials[i].mul(inv_val)),
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // (2^f)' = 2^f * ln(2) * f'
        let exp2_val = self.val.exp2();
        // Compute ln(2) in the algebra A
        let two = A::one().add(A::one());
        let ln_2 = two.ln();
        let coeff = exp2_val.mul(ln_2);
        Self {
            val: exp2_val,
            partials: core::array::from_fn(|i| self.partials[i].mul(coeff)),
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        // (log2 f)' = f' / (f * ln(2))
        let two = A::one().add(A::one());
        let ln_2 = two.ln();
        let coeff = self.val.mul(ln_2).recip();
        Self {
            val: self.val.log2(),
            partials: core::array::from_fn(|i| self.partials[i].mul(coeff)),
        }
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        let val = self.val.pow(exp.val);
        let ln_base = self.val.ln();
        let inv_base = self.val.recip();
        Self {
            val,
            partials: core::array::from_fn(|i| {
                val.mul(
                    exp.partials[i]
                        .mul(ln_base)
                        .add(exp.val.mul(self.partials[i]).mul(inv_base)),
                )
            }),
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        // Floor is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // min(f, g)' = f' if f < g, g' otherwise
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            partials: core::array::from_fn(|i| A::select(mask, self.partials[i], rhs.partials[i])),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // max(f, g)' = f' if f > g, g' otherwise
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            partials: core::array::from_fn(|i| A::select(mask, self.partials[i], rhs.partials[i])),
        }
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // (self * a + b)' = self' * a + self * a' + b'
        // This is just mul followed by add
        self.mul(a).add(b)
    }
}

// ============================================================================
// Default Implementation
// ============================================================================

impl<const N: usize, A: Algebra> Default for Dual<N, A> {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// From Implementations
// ============================================================================

impl<const N: usize, A: Algebra> From<A> for Dual<N, A> {
    #[inline(always)]
    fn from(val: A) -> Self {
        Self::constant(val)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual2_arithmetic() {
        let x = Dual2::<f32>::x(3.0);
        let y = Dual2::<f32>::y(4.0);

        // x + y
        let sum = x.add(y);
        assert_eq!(sum.val, 7.0);
        assert_eq!(sum.dx(), 1.0);
        assert_eq!(sum.dy(), 1.0);

        // x * y
        let prod = x.mul(y);
        assert_eq!(prod.val, 12.0);
        assert_eq!(prod.dx(), 4.0); // ∂(xy)/∂x = y
        assert_eq!(prod.dy(), 3.0); // ∂(xy)/∂y = x
    }

    // Transcendental tests only run on scalar fallback platforms
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_dual2_sqrt() {
        let x = Dual2::<f32>::x(3.0);
        let y = Dual2::<f32>::y(4.0);

        // r = sqrt(x² + y²) = sqrt(9 + 16) = 5
        let r_sq = x.mul(x).add(y.mul(y));
        let r = r_sq.sqrt();

        let epsilon = 1e-5;
        assert!((r.val - 5.0).abs() < epsilon);
        // ∂r/∂x = x/r = 3/5 = 0.6
        assert!((r.dx() - 0.6).abs() < epsilon);
        // ∂r/∂y = y/r = 4/5 = 0.8
        assert!((r.dy() - 0.8).abs() < epsilon);
    }

    // Transcendental tests only run on scalar fallback platforms
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_dual3_normal() {
        // Sphere SDF: f(x,y,z) = sqrt(x² + y² + z²) - 1
        // At point (1, 0, 0), gradient is (1, 0, 0)
        let x = Dual3::<f32>::x(1.0);
        let y = Dual3::<f32>::y(0.0);
        let z = Dual3::<f32>::z(0.0);

        let r_sq = x.mul(x).add(y.mul(y)).add(z.mul(z));
        let sdf = r_sq.sqrt().sub(Dual3::constant(1.0));

        let (nx, ny, nz) = sdf.normal();
        let epsilon = 1e-5;
        assert!((nx - 1.0).abs() < epsilon);
        assert!(ny.abs() < epsilon);
        assert!(nz.abs() < epsilon);
    }

    #[test]
    fn test_dual_select() {
        let a = Dual2::<f32>::x(1.0);
        let b = Dual2::<f32>::y(2.0);

        let selected_true = Dual2::select(true, a, b);
        assert_eq!(selected_true.val, 1.0);
        assert_eq!(selected_true.dx(), 1.0);
        assert_eq!(selected_true.dy(), 0.0);

        let selected_false = Dual2::select(false, a, b);
        assert_eq!(selected_false.val, 2.0);
        assert_eq!(selected_false.dx(), 0.0);
        assert_eq!(selected_false.dy(), 1.0);
    }

    #[test]
    fn test_dual_comparison() {
        let x = Dual2::<f32>::x(3.0);
        let y = Dual2::<f32>::y(4.0);

        // Comparisons return bool (the mask type), not Dual
        assert!(x.lt(y));
        assert!(!x.gt(y));
        assert!(x.le(y));
        assert!(y.ge(x));
    }

    #[test]
    fn test_dual_chain_rule_mul() {
        // Test: f(x) = x², f'(x) = 2x
        // At x = 3: x² = 9, 2x = 6
        let x = Dual1::<f32>::var::<0>(3.0);
        let x_sq = x.mul(x);

        let epsilon = 1e-5;
        assert!((x_sq.val - 9.0).abs() < epsilon);
        assert!((x_sq.partials[0] - 6.0).abs() < epsilon);
    }

    // Transcendental chain rule test only runs on scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_dual_chain_rule_sin() {
        // Test: f(x) = sin(x), f'(x) = cos(x)
        // At x = 0: sin(0) = 0, cos(0) = 1
        let x = Dual1::<f32>::var::<0>(0.0);
        let sin_x = x.sin();

        let epsilon = 1e-5;
        assert!(sin_x.val.abs() < epsilon);
        assert!((sin_x.partials[0] - 1.0).abs() < epsilon);
    }
}
