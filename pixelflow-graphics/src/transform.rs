//! Coordinate transformations for manifolds.
//!
//! Provides composable coordinate warping using the `At` combinator.

use pixelflow_core::ops::{Div, Sub};
use pixelflow_core::{At, Field, Manifold, X, Y, Z, W};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

// =============================================================================
// Scale transformation
// =============================================================================

/// Alias for the scaled manifold type.
pub type Scaled<M> = At<Div<X, f32>, Div<Y, f32>, Z, W, M>;

/// Uniform scaling of the manifold domain.
///
/// Effectively scales the object size by `factor`.
/// Internally, coordinates are divided by `factor`.
///
/// # Example
/// ```ignore
/// let circle = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
/// let big_circle = scale(circle, 2.0);  // radius 2 circle
/// ```
pub fn scale<M>(inner: M, factor: f32) -> Scaled<M>
where
    M: Manifold<Field4, Output = Field>,
{
    At {
        inner,
        x: X / factor,
        y: Y / factor,
        z: Z,
        w: W,
    }
}

// =============================================================================
// Translate transformation
// =============================================================================

/// Alias for the translated manifold type.
pub type Translated<M> = At<Sub<X, f32>, Sub<Y, f32>, Z, W, M>;

/// Translation of the manifold domain.
///
/// Shifts the object by `(dx, dy)`.
/// Internally, coordinates are subtracted by the offset.
///
/// # Example
/// ```ignore
/// let circle = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
/// let moved = translate(circle, 10.0, 5.0);  // circle centered at (10, 5)
/// ```
pub fn translate<M>(inner: M, dx: f32, dy: f32) -> Translated<M>
where
    M: Manifold<Field4, Output = Field>,
{
    At {
        inner,
        x: X - dx,
        y: Y - dy,
        z: Z,
        w: W,
    }
}

// =============================================================================
// Legacy types (for backwards compatibility)
// =============================================================================

/// Uniform scaling transformation struct (legacy).
///
/// Prefer using `scale()` function which returns a composable `At` combinator.
#[derive(Clone, Debug)]
pub struct Scale<M> {
    pub manifold: M,
    pub factor: f32,
}

/// Translation transformation struct (legacy).
///
/// Prefer using `translate()` function which returns a composable `At` combinator.
#[derive(Clone, Debug)]
pub struct Translate<M> {
    pub manifold: M,
    pub offset: [f32; 2],
}

// Manifold implementations for legacy types use At directly

impl<M> Manifold<Field4> for Scale<M>
where
    M: Manifold<Field4, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        // Create At combinator with coordinate expressions
        let at = At {
            inner: &self.manifold,
            x: X / self.factor,
            y: Y / self.factor,
            z: Z,
            w: W,
        };
        at.eval(p)
    }
}

impl<M> Manifold<Field4> for Translate<M>
where
    M: Manifold<Field4, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        // Create At combinator with coordinate expressions
        let at = At {
            inner: &self.manifold,
            x: X - self.offset[0],
            y: Y - self.offset[1],
            z: Z,
            w: W,
        };
        at.eval(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::Field;











    #[test]
    fn scale_is_clone() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let cloned = scaled.clone();
        assert_eq!(cloned.factor, 2.0);
    }








}
