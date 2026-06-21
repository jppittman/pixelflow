//! Bicubic Bezier Patch as a Manifold.
//!
//! A patch is a function from (u, v) parameter space to 3D position.
//! Pull-based: sample the surface at any (u,v), get position and derivatives.

use pixelflow_core::jet::Jet2H;
use pixelflow_core::{Field, Manifold};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// A bicubic Bezier patch defined by 16 control points.
///
/// Implements `Manifold<Field>` where:
/// - Input (x, y): parametric coordinates (u, v) ∈ [0,1]²
/// - Output: z-coordinate of surface (height field)
#[derive(Clone, Copy, Debug)]
pub struct BezierPatch {
    /// Control points P[v][u] - 4x4 grid, each point is [x, y, z]
    pub points: [[[f32; 3]; 4]; 4],
}

impl BezierPatch {
    /// Create a patch from 16 control points.
    #[must_use]
    pub fn new(points: [[[f32; 3]; 4]; 4]) -> Self {
        Self { points }
    }

    /// Create a flat patch in XY plane at z=0.
    #[must_use]
    pub fn flat(size: f32) -> Self {
        let mut points = [[[0.0f32; 3]; 4]; 4];
        for (v, row) in points.iter_mut().enumerate() {
            for (u, point) in row.iter_mut().enumerate() {
                *point = [(u as f32 / 3.0) * size, (v as f32 / 3.0) * size, 0.0];
            }
        }
        Self { points }
    }

    /// Create a curved paraboloid patch.
    #[must_use]
    pub fn paraboloid(size: f32, height: f32) -> Self {
        let mut points = [[[0.0f32; 3]; 4]; 4];
        for (v, row) in points.iter_mut().enumerate() {
            for (u, point) in row.iter_mut().enumerate() {
                let nu = u as f32 / 3.0 - 0.5;
                let nv = v as f32 / 3.0 - 0.5;
                *point = [
                    nu * size,
                    nv * size,
                    height * (1.0 - 4.0 * (nu * nu + nv * nv)),
                ];
            }
        }
        Self { points }
    }

    /// Evaluate at (u, v) with full derivatives via Jet2H.
    ///
    /// Returns [px, py, pz] where each component carries:
    /// - val: position
    /// - dx, dy: first partials (tangent vectors)
    /// - dxx, dxy, dyy: second partials (curvature)
    #[inline]
    #[must_use]
    pub fn eval(&self, u: Jet2H, v: Jet2H) -> [Jet2H; 3] {
        // Bernstein basis (cubic)
        let one = Jet2H::constant(Field::from(1.0));
        let three = Jet2H::constant(Field::from(3.0));

        let u1 = one - u;
        let bu = [
            u1 * u1 * u1,
            three * u * u1 * u1,
            three * u * u * u1,
            u * u * u,
        ];

        let v1 = one - v;
        let bv = [
            v1 * v1 * v1,
            three * v * v1 * v1,
            three * v * v * v1,
            v * v * v,
        ];

        // P(u,v) = Σᵢⱼ Bᵢ(u) Bⱼ(v) Pᵢⱼ
        let zero = Jet2H::constant(Field::from(0.0));
        let mut p = [zero, zero, zero];

        for (row, bvj) in self.points.iter().zip(bv.iter()) {
            for (point, bui) in row.iter().zip(bu.iter()) {
                let w = *bui * *bvj;
                let [cx, cy, cz] = *point;
                p[0] = p[0] + w * Jet2H::constant(Field::from(cx));
                p[1] = p[1] + w * Jet2H::constant(Field::from(cy));
                p[2] = p[2] + w * Jet2H::constant(Field::from(cz));
            }
        }
        p
    }
}

/// Height field: (x, y) → z
impl Manifold<Field4> for BezierPatch {
    type Output = Field;

    #[inline]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, _z, _w) = p;
        let u = Jet2H::x(x);
        let v = Jet2H::y(y);
        self.eval(u, v)[2].val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::ManifoldCompat;

    #[test]
    fn flat_patch_should_succeed_when_invoked() {
        let patch = BezierPatch::flat(1.0);
        let z = Field::from(0.0);
        let result = patch.eval_raw(Field::from(0.5), Field::from(0.5), z, z);
        assert!(result.abs().lt(Field::from(1e-4)).all());
    }

    #[test]
    fn derivatives_should_succeed_when_invoked() {
        let patch = BezierPatch::paraboloid(2.0, 1.0);
        let u = Jet2H::x(Field::from(0.5));
        let v = Jet2H::y(Field::from(0.5));
        let p = patch.eval(u, v);

        // At center of symmetric paraboloid, tangents should be ~horizontal
        // (dx and dy of z should be near zero)
        assert!(p[2].dx.abs().lt(Field::from(0.1)).all());
        assert!(p[2].dy.abs().lt(Field::from(0.1)).all());
    }
}
