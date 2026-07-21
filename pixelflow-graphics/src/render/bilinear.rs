//! Bilinear interpolation combinator.
//!
//! [`Bilinear`] is the smooth companion to `DiscreteManifold`'s
//! nearest-neighbor indexing: where the discrete manifold snaps a continuous
//! coordinate to the containing lattice cell, `Bilinear` samples the four
//! surrounding integer-grid points and blends them with the fractional
//! coordinate weights.
//!
//! # Coordinate convention
//!
//! `Bilinear` works in **integer-grid space**: the wrapped manifold is
//! sampled at integer (x, y) coordinates. A query at `(i + fx, j + fy)` with
//! `fx, fy ∈ [0, 1)` blends the samples at `(i, j)`, `(i+1, j)`, `(i, j+1)`,
//! and `(i+1, j+1)`. Consequences:
//!
//! - At exact integer coordinates the wrapped manifold's value is returned
//!   untouched (the fractional weights are zero).
//! - A manifold that is affine in x and y is reproduced exactly everywhere.
//! - No half-pixel convention is baked in here. Callers that store samples
//!   at texel *centers* (the rasterizer's `i + 0.5` convention) must shift
//!   coordinates by −0.5 before sampling — see `fonts::cache::CachedGlyph`.
//!
//! Z and W pass through unchanged; interpolation is 2D over X/Y only.

use pixelflow_compiler::kernel;
use pixelflow_core::Field;

// The 4-tap bilinear kernel. `tex` is a manifold parameter: any
// `Manifold<Field4, Output = Field>` (e.g. `DiscreteManifold`) can be wrapped.
// The taps are lazy `.at()` re-evaluations of the wrapped manifold at offset
// integer coordinates. Tap expressions capture `&self.tex` and are not Copy;
// the optimizer's extracted form may reference a tap more than once, which
// codegen handles by cloning tap-bound locals at each use.
kernel!(pub struct Bilinear = |tex: kernel| Field -> Field {
    let x0 = X.floor();
    let y0 = Y.floor();
    let fx = X - x0;
    let fy = Y - y0;

    let c00 = tex.at(x0, y0, Z, W);
    let c10 = tex.at(x0 + 1.0, y0, Z, W);
    let c01 = tex.at(x0, y0 + 1.0, Z, W);
    let c11 = tex.at(x0 + 1.0, y0 + 1.0, Z, W);

    c00 * ((1.0 - fx) * (1.0 - fy))
        + c10 * (fx * (1.0 - fy))
        + c01 * ((1.0 - fx) * fy)
        + c11 * (fx * fy)
});

#[cfg(test)]
mod tests {
    use super::Bilinear;
    use pixelflow_core::{DiscreteManifold, Lattice, Manifold};

    /// Evaluate a manifold at a single point through public lattice API.
    fn sample<M>(m: &M, x: f32, y: f32) -> f32
    where
        M: Manifold<
            (
                pixelflow_core::Field,
                pixelflow_core::Field,
                pixelflow_core::Field,
                pixelflow_core::Field,
            ),
            Output = pixelflow_core::Field,
        >,
    {
        Lattice::point(x, y, 0.0, 0.0).collapse(m).into_buffer()[0]
    }

    #[test]
    fn constant_field_is_identity() {
        // A 3x3 constant grid: bilinear must return the constant everywhere.
        let tex = DiscreteManifold::new(vec![0.75; 9], 3, 3);
        let bilerp = Bilinear::new(tex);

        for &(x, y) in &[(0.0, 0.0), (0.5, 0.5), (1.25, 0.75), (1.9, 1.1)] {
            let v = sample(&bilerp, x, y);
            assert!(
                (v - 0.75).abs() < 1e-6,
                "constant field not reproduced at ({x}, {y}): got {v}"
            );
        }
    }

    #[test]
    fn linear_gradient_reproduced_exactly() {
        // Buffer holding f(x, y) = x + 2y sampled at integer coords, 4x4.
        let mut buf = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                buf.push(x as f32 + 2.0 * y as f32);
            }
        }
        let bilerp = Bilinear::new(DiscreteManifold::new(buf, 4, 4));

        // Bilinear interpolation of an affine function is exact.
        for &(x, y) in &[(0.5, 0.5), (1.25, 2.75), (0.1, 0.9), (2.5, 1.0)] {
            let expect = x + 2.0 * y;
            let v = sample(&bilerp, x, y);
            assert!(
                (v - expect).abs() < 1e-5,
                "gradient not reproduced at ({x}, {y}): got {v}, want {expect}"
            );
        }
    }

    #[test]
    fn exact_at_grid_points_blended_between() {
        // 2x2 checker: (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=0.
        let tex = DiscreteManifold::new(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
        let bilerp = Bilinear::new(tex);

        // At integer grid points: exact stored values, no smoothing.
        assert!((sample(&bilerp, 0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample(&bilerp, 1.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((sample(&bilerp, 0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((sample(&bilerp, 1.0, 1.0) - 0.0).abs() < 1e-6);

        // Midpoint of an edge: average of its two endpoints.
        assert!((sample(&bilerp, 0.5, 0.0) - 0.5).abs() < 1e-6);
        // Cell center: average of all four corners.
        assert!((sample(&bilerp, 0.5, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn no_nearest_jumps_across_cell_boundaries() {
        // Step buffer: left column 0, right column 1. Nearest-neighbor would
        // jump 0 -> 1 crossing x = 1; bilinear must ramp smoothly.
        let tex = DiscreteManifold::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2);
        let bilerp = Bilinear::new(tex);

        let step = 0.125;
        let mut prev = sample(&bilerp, 0.0, 0.5);
        let mut x = step;
        while x <= 2.0 {
            let v = sample(&bilerp, x, 0.5);
            let jump = (v - prev).abs();
            assert!(
                jump < 0.25,
                "non-smooth jump {jump} at x = {x} (bilinear should ramp, not step)"
            );
            prev = v;
            x += step;
        }
        // And the ramp actually reaches the step's top value.
        assert!((sample(&bilerp, 2.0, 0.5) - 1.0).abs() < 1e-6);
    }
}
