//! Coordinate transformations for manifolds.
//!
//! Provides composable coordinate warping using the `At` combinator.

use pixelflow_core::ops::{Div, Sub};
use pixelflow_core::{At, Field, Manifold, W, X, Y, Z};

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
    fn scale_function_works() {
        let scaled = scale(X, 2.0);
        let zero = Field::from(0.0);
        let two = Field::from(2.0);

        // At x=2, scaled should give x/2 = 1
        let result = scaled.eval((two, zero, zero, zero));
        let _ = result;
    }

    #[test]
    fn translate_function_works() {
        let translated = translate(X, 1.0, 0.0);
        let zero = Field::from(0.0);
        let two = Field::from(2.0);

        // At x=2, translated should give x-1 = 1
        let result = translated.eval((two, zero, zero, zero));
        let _ = result;
    }

    #[test]
    fn scale_and_translate_compose() {
        let scaled = scale(X, 2.0);
        let composed = translate(scaled, 1.0, 0.0);

        let zero = Field::from(0.0);
        let four = Field::from(4.0);

        let result = composed.eval((four, zero, zero, zero));
        let _ = result;
    }

    #[test]
    fn scale_creation_and_eval() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        let _ = scaled.eval((one, zero, zero, zero));
    }

    #[test]
    fn scale_with_various_factors() {
        for factor in [0.5, 1.0, 2.0, 10.0] {
            let scaled = Scale {
                manifold: X,
                factor,
            };
            let zero = Field::from(0.0);
            let one = Field::from(1.0);
            let _ = scaled.eval((one, one, zero, zero));
        }
    }

    #[test]
    fn scale_is_clone() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let cloned = scaled.clone();
        assert_eq!(cloned.factor, 2.0);
    }

    #[test]
    fn translate_creation_and_eval() {
        let translated = Translate {
            manifold: X,
            offset: [1.0, 2.0],
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        let _ = translated.eval((one, one, zero, zero));
    }

    #[test]
    fn translate_with_various_offsets() {
        for offset in [[0.0, 0.0], [1.0, 1.0], [-5.0, 5.0], [100.0, -100.0]] {
            let translated = Translate {
                manifold: X,
                offset,
            };
            let zero = Field::from(0.0);
            let one = Field::from(1.0);
            let _ = translated.eval((one, one, zero, zero));
        }
    }

    #[test]
    fn struct_scale_and_translate_compose() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let composed = Translate {
            manifold: scaled,
            offset: [1.0, 1.0],
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        let _ = composed.eval((one, one, zero, zero));
    }

    #[test]
    fn struct_translate_and_scale_compose() {
        let translated = Translate {
            manifold: Y,
            offset: [1.0, 2.0],
        };
        let composed = Scale {
            manifold: translated,
            factor: 0.5,
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        let _ = composed.eval((one, one, zero, zero));
    }
}
