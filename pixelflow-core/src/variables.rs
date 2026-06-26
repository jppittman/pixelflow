//! # Coordinate Variables
//!
//! The base coordinate manifolds: X, Y, Z, W.
//! Also defines the `Axis` enum for 4D topology.
//!
//! ## Domain-Generic Design
//!
//! Coordinate variables use the `Spatial` trait to access coordinates from
//! any domain type. This allows them to work with:
//!
//! - 2D domains: `(I, I)` - X and Y available, Z and W return zero
//! - 3D domains: `(I, I, I)` - X, Y, Z available, W returns zero
//! - 4D domains: `(I, I, I, I)` - all coordinates available
//! - Let-extended domains: `LetExtended<V, Rest>` - coordinates pass through
//!
//! ## Example
//!
//! ```ignore
//! use pixelflow_core::{X, Y, Manifold, Field};
//!
//! // X and Y work on any Spatial domain
//! let circle = (X * X + Y * Y).sqrt();
//!
//! // Evaluate on 2D domain
//! let val = circle.eval((Field::from(3.0), Field::from(4.0)));
//! // val = 5.0
//! ```

use crate::Manifold;
use crate::domain::Spatial;

/// The explicit 4D axes of the manifold topology.
/// Used for indexing `Vector` outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// The first dimension (X / Red).
    X,
    /// The second dimension (Y / Green).
    Y,
    /// The third dimension (Z / Blue).
    Z,
    /// The fourth dimension (W / Alpha).
    W,
}

// Coordinate Manifolds (Variables)

/// The X coordinate (Index 0).
#[derive(Clone, Copy, Debug, Default)]
pub struct X;

/// The Y coordinate (Index 1).
#[derive(Clone, Copy, Debug, Default)]
pub struct Y;

/// The Z coordinate (Index 2).
#[derive(Clone, Copy, Debug, Default)]
pub struct Z;

/// The W coordinate (Index 3).
#[derive(Clone, Copy, Debug, Default)]
pub struct W;

/// A marker trait for types that represent a static Axis.
pub trait Dimension {
    /// The axis this type represents.
    const AXIS: Axis;
}

impl Dimension for X {
    const AXIS: Axis = Axis::X;
}
impl Dimension for Y {
    const AXIS: Axis = Axis::Y;
}
impl Dimension for Z {
    const AXIS: Axis = Axis::Z;
}
impl Dimension for W {
    const AXIS: Axis = Axis::W;
}

// ============================================================================
// ManifoldExpr implementations for Coordinate Variables
// ============================================================================

impl crate::ManifoldExpr for X {}
impl crate::ManifoldExpr for Y {}
impl crate::ManifoldExpr for Z {}
impl crate::ManifoldExpr for W {}

// ============================================================================
// Manifold Implementations for Coordinate Variables
// ============================================================================

// X reads the x coordinate from any Spatial domain
impl<P> Manifold<P> for X
where
    P: Spatial + Send + Sync,
    P::Coord: Copy + Send + Sync,
{
    type Output = P::Coord;
    #[inline(always)]
    fn eval(&self, p: P) -> P::Coord {
        p.x()
    }
}

// Y reads the y coordinate from any Spatial domain
impl<P> Manifold<P> for Y
where
    P: Spatial + Send + Sync,
    P::Coord: Copy + Send + Sync,
{
    type Output = P::Coord;
    #[inline(always)]
    fn eval(&self, p: P) -> P::Coord {
        p.y()
    }
}

// Z reads the z coordinate from any Spatial domain
// For 2D domains, returns zero per GLSL conventions
impl<P> Manifold<P> for Z
where
    P: Spatial + Send + Sync,
    P::Coord: Copy + Send + Sync,
{
    type Output = P::Coord;
    #[inline(always)]
    fn eval(&self, p: P) -> P::Coord {
        p.z()
    }
}

// W reads the w coordinate from any Spatial domain
// For 2D/3D domains, returns zero per GLSL conventions
impl<P> Manifold<P> for W
where
    P: Spatial + Send + Sync,
    P::Coord: Copy + Send + Sync,
{
    type Output = P::Coord;
    #[inline(always)]
    fn eval(&self, p: P) -> P::Coord {
        p.w()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn test_x_on_2d() {
        let domain = (Field::from(3.0), Field::from(4.0));
        let result = X.eval(domain);
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 3.0);
    }

    #[test]
    fn test_y_on_2d() {
        let domain = (Field::from(3.0), Field::from(4.0));
        let result = Y.eval(domain);
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 4.0);
    }

    #[test]
    fn test_z_on_2d_is_zero() {
        let domain = (Field::from(3.0), Field::from(4.0));
        let result = Z.eval(domain);
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 0.0); // Zero-padded
    }

    #[test]
    fn test_on_4d() {
        let domain = (
            Field::from(1.0),
            Field::from(2.0),
            Field::from(3.0),
            Field::from(4.0),
        );
        let mut buf = [0.0f32; crate::PARALLELISM];

        X.eval(domain).store(&mut buf);
        assert_eq!(buf[0], 1.0);

        Y.eval(domain).store(&mut buf);
        assert_eq!(buf[0], 2.0);

        Z.eval(domain).store(&mut buf);
        assert_eq!(buf[0], 3.0);

        W.eval(domain).store(&mut buf);
        assert_eq!(buf[0], 4.0);
    }
}
