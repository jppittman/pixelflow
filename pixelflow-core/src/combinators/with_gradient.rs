//! # WithGradient Combinator
//!
//! Explicit AD functor that lifts Field coordinates to Jet internally.
//!
//! This combinator makes automatic differentiation a first-class, explicit operation.
//! Users can work with Field-domain inputs while getting Jet-domain outputs containing
//! both values and gradients.
//!
//! ## Problem
//!
//! AD is implicit—users must know to use Jet domain. There's no separation between
//! "compute value" and "compute value + gradient".
//!
//! ## Solution
//!
//! `WithGradient<M, DIM>` lifts Field coordinates to Jet internally:
//! - Field-domain input, Jet-domain output (cleaner API)
//! - Composable with other combinators
//! - Zero runtime overhead (monomorphizes away)
//!
//! ## Example
//!
//! ```ignore
//! use pixelflow_core::{WithGradient, X, Y, Manifold, Field, ManifoldExt};
//!
//! // Define distance from origin (fully polymorphic over computational type)
//! let distance = (X * X + Y * Y).sqrt();
//!
//! // Wrap with gradient computation for 2D
//! let with_grad = WithGradient::<_, 2>::new(distance);
//!
//! // Evaluate with Field coordinates, get Jet2 output with gradients!
//! let result = with_grad.eval((
//!     Field::from(3.0),
//!     Field::from(4.0),
//!     Field::from(0.0),
//!     Field::from(0.0),
//! ));
//! // result.val ≈ 5.0 (distance from origin)
//! // result.dx ≈ 0.6 (∂/∂x = x/r)
//! // result.dy ≈ 0.8 (∂/∂y = y/r)
//! ```
//!
//! ## Note on Polymorphism
//!
//! The inner manifold `M` must be polymorphic over the computational type.
//! This means it must work with both `Field` and `Jet` domains. Expressions
//! using only coordinate variables (X, Y, Z, W) and their combinations via
//! operators are automatically polymorphic.
//!
//! **Important:** Using raw `f32` literals (e.g., `1.0f32`) in the expression
//! will break polymorphism because `f32` always returns `Field`, not `Jet2`.
//! For expressions with constants, use the `kernel!` macro which handles
//! constant lifting properly.

use crate::jet::{Jet2, Jet3};
use crate::{Field, Manifold};

/// Type alias for 4D Field domain.
type Field4 = (Field, Field, Field, Field);

/// Type alias for 4D Jet2 domain.
type Jet2_4 = (Jet2, Jet2, Jet2, Jet2);

/// Type alias for 4D Jet3 domain.
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

/// A combinator that lifts Field coordinates to Jet domain for automatic differentiation.
///
/// `WithGradient<M, DIM>` wraps a manifold that operates on Jet domain and provides
/// a Field-domain interface. The lifting is done internally, making AD explicit
/// and compositional.
///
/// # Type Parameters
///
/// - `M`: The inner manifold that operates on Jet domain
/// - `DIM`: The number of dimensions to differentiate (2 or 3)
///
/// # Semantics
///
/// For DIM=2:
/// - Input: `(x: Field, y: Field, z: Field, w: Field)`
/// - Internally: Lifts to `(Jet2::x(x), Jet2::y(y), Jet2::constant(z), Jet2::constant(w))`
/// - Output: `Jet2` containing value and 2D gradients
///
/// For DIM=3:
/// - Input: `(x: Field, y: Field, z: Field, w: Field)`
/// - Internally: Lifts to `(Jet3::x(x), Jet3::y(y), Jet3::z(z), Jet3::constant(w))`
/// - Output: `Jet3` containing value and 3D gradients
#[derive(Clone, Copy, Debug, Default)]
pub struct WithGradient<M, const DIM: usize>(pub M);

impl<M, const DIM: usize> WithGradient<M, DIM> {
    /// Create a new WithGradient combinator.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let distance = (X * X + Y * Y).sqrt();
    /// let with_grad = WithGradient::<_, 2>::new(distance);
    /// ```
    #[inline(always)]
    pub fn new(inner: M) -> Self {
        Self(inner)
    }

    /// Get a reference to the inner manifold.
    #[inline(always)]
    pub fn inner(&self) -> &M {
        &self.0
    }

    /// Consume self and return the inner manifold.
    #[inline(always)]
    pub fn into_inner(self) -> M {
        self.0
    }
}

// ============================================================================
// 2D Gradient (DIM = 2)
// ============================================================================

impl<M> Manifold<Field4> for WithGradient<M, 2>
where
    M: Manifold<Jet2_4, Output = Jet2>,
{
    type Output = Jet2;

    /// Evaluate the manifold with automatic 2D gradient computation.
    ///
    /// Lifts Field coordinates to Jet2 domain, evaluates the inner manifold,
    /// and returns the result with value and 2D partial derivatives.
    #[inline(always)]
    fn eval(&self, p: Field4) -> Self::Output {
        let (x, y, z, w) = p;
        // Lift Field coordinates to Jet2:
        // - x is seeded for ∂/∂x (dx=1, dy=0)
        // - y is seeded for ∂/∂y (dx=0, dy=1)
        // - z, w are constants (no derivatives)
        let jp: Jet2_4 = (Jet2::x(x), Jet2::y(y), Jet2::constant(z), Jet2::constant(w));
        self.0.eval(jp)
    }
}

// ============================================================================
// 3D Gradient (DIM = 3)
// ============================================================================

impl<M> Manifold<Field4> for WithGradient<M, 3>
where
    M: Manifold<Jet3_4, Output = Jet3>,
{
    type Output = Jet3;

    /// Evaluate the manifold with automatic 3D gradient computation.
    ///
    /// Lifts Field coordinates to Jet3 domain, evaluates the inner manifold,
    /// and returns the result with value and 3D partial derivatives.
    #[inline(always)]
    fn eval(&self, p: Field4) -> Self::Output {
        let (x, y, z, w) = p;
        // Lift Field coordinates to Jet3:
        // - x is seeded for ∂/∂x
        // - y is seeded for ∂/∂y
        // - z is seeded for ∂/∂z
        // - w is a constant (no derivatives)
        let jp: Jet3_4 = (Jet3::x(x), Jet3::y(y), Jet3::z(z), Jet3::constant(w));
        self.0.eval(jp)
    }
}

// ============================================================================
// Convenience Type Aliases
// ============================================================================

/// 2D gradient wrapper (shorthand for `WithGradient<M, 2>`).
///
/// Use this when you want automatic 2D gradients (∂/∂x, ∂/∂y).
pub type WithGradient2D<M> = WithGradient<M, 2>;

/// 3D gradient wrapper (shorthand for `WithGradient<M, 3>`).
///
/// Use this when you want automatic 3D gradients (∂/∂x, ∂/∂y, ∂/∂z).
pub type WithGradient3D<M> = WithGradient<M, 3>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext::ManifoldExt;
    use crate::{X, Y, Z};

    // Helper to get the first lane from a Field
    fn first_lane(f: Field) -> f32 {
        let mut buf = [0.0f32; crate::PARALLELISM];
        f.store(&mut buf);
        buf[0]
    }

    #[test]
    fn test_with_gradient_2d_distance() {
        // Distance from origin: sqrt(x² + y²)
        // At (3, 4): value = 5
        // Gradient: (x/r, y/r) = (3/5, 4/5) = (0.6, 0.8)
        //
        // Note: We use pure variable expressions (no f32 literals) because
        // f32 literals return Field, not Jet2. The manifold must be fully
        // polymorphic over the computational type.
        //
        // Tolerance is 1e-4 due to fast rsqrt approximation used in sqrt.
        let distance = (X * X + Y * Y).sqrt();
        let with_grad = WithGradient::<_, 2>::new(distance);

        let result = with_grad.eval((
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
            Field::from(0.0),
        ));

        let val = first_lane(result.val);
        let dx = first_lane(result.dx);
        let dy = first_lane(result.dy);

        assert!((val - 5.0).abs() < 5e-4, "value should be 5.0, got {}", val);
        assert!((dx - 0.6).abs() < 5e-4, "dx should be 0.6, got {}", dx);
        assert!((dy - 0.8).abs() < 5e-4, "dy should be 0.8, got {}", dy);
    }

    #[test]
    fn test_with_gradient_2d_quadratic() {
        // Quadratic: x² + y²
        // At (2, 3): value = 4 + 9 = 13
        // Gradient: (2x, 2y) = (4, 6)
        let quadratic = X * X + Y * Y;
        let with_grad = WithGradient2D::new(quadratic);

        let result = with_grad.eval((
            Field::from(2.0),
            Field::from(3.0),
            Field::from(0.0),
            Field::from(0.0),
        ));

        let val = first_lane(result.val);
        let dx = first_lane(result.dx);
        let dy = first_lane(result.dy);

        assert!(
            (val - 13.0).abs() < 1e-5,
            "value should be 13.0, got {}",
            val
        );
        assert!((dx - 4.0).abs() < 1e-5, "dx should be 4.0, got {}", dx);
        assert!((dy - 6.0).abs() < 1e-5, "dy should be 6.0, got {}", dy);
    }

    #[test]
    fn test_with_gradient_2d_product() {
        // Product: x * y
        // At (3, 5): value = 15
        // Gradient: (y, x) = (5, 3)
        let product = X * Y;
        let with_grad = WithGradient2D::new(product);

        let result = with_grad.eval((
            Field::from(3.0),
            Field::from(5.0),
            Field::from(0.0),
            Field::from(0.0),
        ));

        let val = first_lane(result.val);
        let dx = first_lane(result.dx);
        let dy = first_lane(result.dy);

        assert!(
            (val - 15.0).abs() < 1e-5,
            "value should be 15.0, got {}",
            val
        );
        assert!((dx - 5.0).abs() < 1e-5, "dx should be y=5.0, got {}", dx);
        assert!((dy - 3.0).abs() < 1e-5, "dy should be x=3.0, got {}", dy);
    }

    #[test]
    fn test_with_gradient_3d_distance() {
        // 3D distance from origin: sqrt(x² + y² + z²)
        // At (1, 2, 2): r = 3, value = 3
        // Gradient: (x/r, y/r, z/r) = (1/3, 2/3, 2/3)
        //
        // Tolerance is 1e-4 due to fast rsqrt approximation used in sqrt.
        let distance = (X * X + Y * Y + Z * Z).sqrt();
        let with_grad = WithGradient3D::new(distance);

        let result = with_grad.eval((
            Field::from(1.0),
            Field::from(2.0),
            Field::from(2.0),
            Field::from(0.0),
        ));

        let val = first_lane(result.val);
        let dx = first_lane(result.dx);
        let dy = first_lane(result.dy);
        let dz = first_lane(result.dz);

        assert!((val - 3.0).abs() < 1e-3, "value should be 3.0, got {}", val);
        assert!(
            (dx - 1.0 / 3.0).abs() < 1e-3,
            "dx should be 1/3, got {}",
            dx
        );
        assert!(
            (dy - 2.0 / 3.0).abs() < 1e-3,
            "dy should be 2/3, got {}",
            dy
        );
        assert!(
            (dz - 2.0 / 3.0).abs() < 1e-3,
            "dz should be 2/3, got {}",
            dz
        );
    }

    #[test]
    fn test_with_gradient_3d_product() {
        // 3D product: x * y * z
        // At (2, 3, 4): value = 24
        // Gradient: (yz, xz, xy) = (12, 8, 6)
        let product = X * Y * Z;
        let with_grad = WithGradient3D::new(product);

        let result = with_grad.eval((
            Field::from(2.0),
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
        ));

        let val = first_lane(result.val);
        let dx = first_lane(result.dx);
        let dy = first_lane(result.dy);
        let dz = first_lane(result.dz);

        assert!(
            (val - 24.0).abs() < 1e-5,
            "value should be 24.0, got {}",
            val
        );
        assert!((dx - 12.0).abs() < 1e-5, "dx should be yz=12.0, got {}", dx);
        assert!((dy - 8.0).abs() < 1e-5, "dy should be xz=8.0, got {}", dy);
        assert!((dz - 6.0).abs() < 1e-5, "dz should be xy=6.0, got {}", dz);
    }
}
