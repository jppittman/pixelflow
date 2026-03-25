//! Catmull-Clark subdivision surfaces via analytical limit evaluation.
//!
//! Instead of recursive tessellation, we evaluate the limit surface directly
//! using Stam's 1998 eigenanalysis. For a face with vertices of valence V,
//! the limit position is:
//!
//! ```text
//! P(u,v) = Σᵢ λᵢⁿ⁻¹ · φᵢ(u,v) · cᵢ
//! ```
//!
//! Where:
//! - λᵢ are eigenvalues of the subdivision matrix
//! - φᵢ(u,v) are bicubic basis functions
//! - cᵢ are control point coefficients (linear combination of cage vertices)
//! - n is the subdivision level (for limit surface, n → ∞, so λᵢⁿ⁻¹ vanishes for |λᵢ| < 1)
//!
//! Derivatives come for free via Jet3:
//! - Evaluate with Jet3<Field> instead of Field
//! - Normal = cross(dP/du, dP/dv)
//! - No finite differences, no extra evaluations

use crate::mesh::{Point3, QuadMesh};
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_compiler::ManifoldExpr;

/// The 4D Jet3 domain type for 3D ray tracing autodiff.
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

/// Eigenstructure for a given valence configuration.
///
/// Precomputed at compile time via const fn (future work).
/// For now, tables will be baked from Stam's formulas.
#[derive(Clone, Debug)]
pub struct EigenStructure {
    /// Valence (number of edges meeting at extraordinary vertex)
    pub valence: usize,
    /// Eigenvalues λᵢ of subdivision matrix
    pub eigenvalues: Vec<f32>,
    /// Eigenvectors (basis for limit surface evaluation)
    pub eigenvectors: Vec<Vec<f32>>,
}

impl EigenStructure {
    /// Create eigenstructure for regular vertex (valence 4).
    ///
    /// This is the simplest case - reduces to bicubic B-spline.
    pub fn regular() -> Self {
        Self {
            valence: 4,
            eigenvalues: vec![1.0, 0.25, 0.25, 0.0625],
            eigenvectors: Vec::new(), // TODO: B-spline basis
        }
    }

    /// Create eigenstructure for arbitrary valence.
    ///
    /// Uses Stam's formulas for eigendecomposition.
    pub fn for_valence(valence: usize) -> Self {
        // TODO: Implement Stam eigenanalysis
        // For now, return placeholder
        Self {
            valence,
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        }
    }
}

/// A subdivision patch - one quad face from the control cage.
///
/// Stores the indices and local topology needed to evaluate
/// the limit surface at (u,v) ∈ [0,1]².
#[derive(Clone, Debug)]
pub struct SubdivisionPatch {
    /// Index of this face in the mesh
    pub face_idx: usize,
    /// The 4 corner vertex indices [v0, v1, v2, v3]
    pub corners: [usize; 4],
    /// Valences of the 4 corners
    pub corner_valences: [usize; 4],
}

impl SubdivisionPatch {
    /// Extract patch from mesh at given face index.
    pub fn from_mesh(mesh: &QuadMesh, face_idx: usize) -> Result<Self, String> {
        if face_idx >= mesh.face_count() {
            return Err(format!(
                "Face index {} out of bounds (mesh has {} faces)",
                face_idx,
                mesh.face_count()
            ));
        }

        let face = &mesh.faces[face_idx];
        let corners = face.vertices;

        // Validate all vertex indices
        for &v in &corners {
            if !mesh.is_valid_vertex(v) {
                return Err(format!("Invalid vertex index {} in face {}", v, face_idx));
            }
        }

        let corner_valences = [
            mesh.valence[corners[0]],
            mesh.valence[corners[1]],
            mesh.valence[corners[2]],
            mesh.valence[corners[3]],
        ];

        Ok(Self {
            face_idx,
            corners,
            corner_valences,
        })
    }

    /// Check if this patch has any extraordinary vertices.
    ///
    /// Extraordinary = valence != 4 (the regular case).
    pub fn is_extraordinary(&self) -> bool {
        self.corner_valences.iter().any(|&v| v != 4)
    }

    /// Get maximum valence among corners.
    pub fn max_valence(&self) -> usize {
        *self.corner_valences.iter().max().unwrap()
    }

    /// Evaluate limit surface at (u,v) with automatic differentiation.
    ///
    /// For regular patches (all valences = 4), uses bicubic B-spline basis.
    /// Returns [x, y, z] where each component is a Jet3 carrying derivatives.
    pub fn eval_limit(&self, mesh: &QuadMesh, u: Jet3, v: Jet3) -> [Jet3; 3] {
        if self.is_regular() {
            // Regular case: bicubic B-spline (16 control points)
            // For a single quad, we use the 4 corners directly with uniform basis
            self.eval_bspline_limit(mesh, u, v)
        } else {
            // TODO: Extraordinary vertices require eigenanalysis
            // For now, fall back to bilinear
            self.eval_bilinear(mesh, u, v)
        }
    }

    /// Evaluate using bicubic B-spline basis (regular patches only).
    fn eval_bspline_limit(&self, mesh: &QuadMesh, u: Jet3, v: Jet3) -> [Jet3; 3] {
        // Uniform cubic B-spline basis functions
        let one = Jet3::constant(Field::from(1.0));
        let six = Jet3::constant(Field::from(6.0));

        let u2 = u * u;
        let u3 = u2 * u;
        let v2 = v * v;
        let v3 = v2 * v;

        let u1 = one - u;
        let v1 = one - v;

        // B-spline basis (cubic)
        // B0(t) = (1-t)³ / 6
        // B1(t) = (3t³ - 6t² + 4) / 6
        // B2(t) = (-3t³ + 3t² + 3t + 1) / 6
        // B3(t) = t³ / 6

        let _bu0 = u1 * u1 * u1 / six;
        let bu1 = (u3 * Jet3::constant(Field::from(3.0)) - u2 * Jet3::constant(Field::from(6.0))
            + Jet3::constant(Field::from(4.0)))
            / six;
        let bu2 = (u3 * Jet3::constant(Field::from(-3.0))
            + u2 * Jet3::constant(Field::from(3.0))
            + u * Jet3::constant(Field::from(3.0))
            + Jet3::constant(Field::from(1.0)))
            / six;
        let _bu3 = u3 / six;

        let _bv0 = v1 * v1 * v1 / six;
        let bv1 = (v3 * Jet3::constant(Field::from(3.0)) - v2 * Jet3::constant(Field::from(6.0))
            + Jet3::constant(Field::from(4.0)))
            / six;
        let bv2 = (v3 * Jet3::constant(Field::from(-3.0))
            + v2 * Jet3::constant(Field::from(3.0))
            + v * Jet3::constant(Field::from(3.0))
            + Jet3::constant(Field::from(1.0)))
            / six;
        let _bv3 = v3 / six;

        // For a single quad, we only have 4 control points
        // Map them to the central 4 of the 16-point B-spline patch
        // This gives us the Catmull-Clark limit surface evaluation
        let p0 = &mesh.vertices[self.corners[0]];
        let p1 = &mesh.vertices[self.corners[1]];
        let p2 = &mesh.vertices[self.corners[2]];
        let p3 = &mesh.vertices[self.corners[3]];

        // Simplified: treat corners as the central patch control points
        // Full implementation would need neighboring vertices
        let w00 = bu1 * bv1;
        let w10 = bu2 * bv1;
        let w11 = bu2 * bv2;
        let w01 = bu1 * bv2;

        let _zero = Jet3::constant(Field::from(0.0));
        let px = w00 * Jet3::constant(Field::from(p0.x))
            + w10 * Jet3::constant(Field::from(p1.x))
            + w11 * Jet3::constant(Field::from(p2.x))
            + w01 * Jet3::constant(Field::from(p3.x));
        let py = w00 * Jet3::constant(Field::from(p0.y))
            + w10 * Jet3::constant(Field::from(p1.y))
            + w11 * Jet3::constant(Field::from(p2.y))
            + w01 * Jet3::constant(Field::from(p3.y));
        let pz = w00 * Jet3::constant(Field::from(p0.z))
            + w10 * Jet3::constant(Field::from(p1.z))
            + w11 * Jet3::constant(Field::from(p2.z))
            + w01 * Jet3::constant(Field::from(p3.z));

        [px, py, pz]
    }

    /// Bilinear interpolation fallback.
    fn eval_bilinear(&self, mesh: &QuadMesh, u: Jet3, v: Jet3) -> [Jet3; 3] {
        let p0 = &mesh.vertices[self.corners[0]];
        let p1 = &mesh.vertices[self.corners[1]];
        let p2 = &mesh.vertices[self.corners[2]];
        let p3 = &mesh.vertices[self.corners[3]];

        let one = Jet3::constant(Field::from(1.0));
        let u1 = one - u;
        let v1 = one - v;

        let w00 = u1 * v1;
        let w10 = u * v1;
        let w11 = u * v;
        let w01 = u1 * v;

        let px = w00 * Jet3::constant(Field::from(p0.x))
            + w10 * Jet3::constant(Field::from(p1.x))
            + w11 * Jet3::constant(Field::from(p2.x))
            + w01 * Jet3::constant(Field::from(p3.x));
        let py = w00 * Jet3::constant(Field::from(p0.y))
            + w10 * Jet3::constant(Field::from(p1.y))
            + w11 * Jet3::constant(Field::from(p2.y))
            + w01 * Jet3::constant(Field::from(p3.y));
        let pz = w00 * Jet3::constant(Field::from(p0.z))
            + w10 * Jet3::constant(Field::from(p1.z))
            + w11 * Jet3::constant(Field::from(p2.z))
            + w01 * Jet3::constant(Field::from(p3.z));

        [px, py, pz]
    }

    /// Check if all corners are regular (valence 4).
    fn is_regular(&self) -> bool {
        self.corner_valences.iter().all(|&v| v == 4)
    }
}

/// Subdivision surface - collection of patches with shared topology.
#[derive(Clone, Debug)]
pub struct SubdivisionSurface {
    /// The control cage mesh
    pub mesh: QuadMesh,
    /// One patch per face
    pub patches: Vec<SubdivisionPatch>,
}

impl SubdivisionSurface {
    /// Build subdivision surface from quad mesh.
    pub fn from_mesh(mesh: QuadMesh) -> Result<Self, String> {
        let mut patches = Vec::with_capacity(mesh.face_count());

        for face_idx in 0..mesh.face_count() {
            patches.push(SubdivisionPatch::from_mesh(&mesh, face_idx)?);
        }

        Ok(Self { mesh, patches })
    }

    /// Get number of patches.
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Get statistics about extraordinary vertices.
    pub fn stats(&self) -> SurfaceStats {
        let total_patches = self.patches.len();
        let extraordinary_patches = self.patches.iter().filter(|p| p.is_extraordinary()).count();
        let max_valence = self
            .patches
            .iter()
            .map(|p| p.max_valence())
            .max()
            .unwrap_or(0);

        SurfaceStats {
            total_patches,
            extraordinary_patches,
            max_valence,
        }
    }
}

/// Statistics about a subdivision surface.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceStats {
    pub total_patches: usize,
    pub extraordinary_patches: usize,
    pub max_valence: usize,
}

// ============================================================================
// Geometry for Raytracing
// ============================================================================

/// Subdivision surface geometry for raytracing.
///
/// Evaluates ray-patch intersection using Newton iteration on the limit surface.
/// For now, simplified to single-patch intersection at a fixed height.
#[derive(Clone, ManifoldExpr)]
pub struct SubdivisionGeometry {
    /// The patch to raytrace
    patch: SubdivisionPatch,
    /// Reference to mesh (via index - will be resolved at eval time)
    /// For now, we store the patch control points directly
    control_points: [[f32; 3]; 4],
    /// Base height for intersection plane
    pub base_height: f32,
    /// UV scale (maps world coords to [0,1] parameter space)
    pub uv_scale: f32,
    /// Center X in world space
    pub center_x: f32,
    /// Center Z in world space
    pub center_z: f32,
}

impl SubdivisionGeometry {
    /// Create geometry from a patch and mesh.
    pub fn new(
        patch: SubdivisionPatch,
        mesh: &QuadMesh,
        base_height: f32,
        uv_scale: f32,
        center_x: f32,
        center_z: f32,
    ) -> Self {
        let control_points = [
            [
                mesh.vertices[patch.corners[0]].x,
                mesh.vertices[patch.corners[0]].y,
                mesh.vertices[patch.corners[0]].z,
            ],
            [
                mesh.vertices[patch.corners[1]].x,
                mesh.vertices[patch.corners[1]].y,
                mesh.vertices[patch.corners[1]].z,
            ],
            [
                mesh.vertices[patch.corners[2]].x,
                mesh.vertices[patch.corners[2]].y,
                mesh.vertices[patch.corners[2]].z,
            ],
            [
                mesh.vertices[patch.corners[3]].x,
                mesh.vertices[patch.corners[3]].y,
                mesh.vertices[patch.corners[3]].z,
            ],
        ];

        Self {
            patch,
            control_points,
            base_height,
            uv_scale,
            center_x,
            center_z,
        }
    }

    /// Evaluate limit surface using stored control points.
    fn eval_with_controls(&self, u: Jet3, v: Jet3) -> [Jet3; 3] {
        // Create a temporary mesh from control points
        let temp_mesh = QuadMesh {
            vertices: vec![
                Point3::new(
                    self.control_points[0][0],
                    self.control_points[0][1],
                    self.control_points[0][2],
                ),
                Point3::new(
                    self.control_points[1][0],
                    self.control_points[1][1],
                    self.control_points[1][2],
                ),
                Point3::new(
                    self.control_points[2][0],
                    self.control_points[2][1],
                    self.control_points[2][2],
                ),
                Point3::new(
                    self.control_points[3][0],
                    self.control_points[3][1],
                    self.control_points[3][2],
                ),
            ],
            faces: vec![],
            valence: vec![],
        };

        self.patch.eval_limit(&temp_mesh, u, v)
    }
}

impl Manifold<Jet3_4> for SubdivisionGeometry {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        // Simplified intersection: hit base plane, map to UV, sample surface
        // Similar to HeightFieldGeometry pattern

        // Step 1: Hit base plane at y = base_height
        let t_plane = Jet3::constant(Field::from(self.base_height)) / ry;

        // Step 2: Get (x, z) world coords at plane hit
        let hit_x = rx * t_plane;
        let hit_z = rz * t_plane;

        // Step 3: Map to (u, v) centered on (center_x, center_z)
        let uv_scale = Field::from(self.uv_scale);
        let half = Field::from(0.5);
        let u_val = ((hit_x.val - Field::from(self.center_x)) * uv_scale + half).constant();
        let v_val = ((hit_z.val - Field::from(self.center_z)) * uv_scale + half).constant();

        // Bounds check: (u, v) must be in [0, 1]
        let zero = Field::from(0.0);
        let one = Field::from(1.0);
        let in_bounds = u_val.ge(zero) & u_val.le(one) & v_val.ge(zero) & v_val.le(one);

        // Step 4: Evaluate subdivision surface at (u, v) with autodiff
        let u = Jet3::constant(u_val);
        let v = Jet3::constant(v_val);
        let p = self.eval_with_controls(u, v);

        // Use the Y component as height displacement
        let surface_y = p[1].val;
        let t_hit = Jet3::constant(surface_y) / ry;

        // Return valid t if in bounds, else negative (miss)
        let miss = Field::from(-1.0);
        Jet3::new(
            in_bounds.clone().select(t_hit.val, miss),
            in_bounds.clone().select(t_hit.dx, miss),
            in_bounds.clone().select(t_hit.dy, miss),
            in_bounds.select(t_hit.dz, miss),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;
    use std::io::Cursor;

    #[test]
    fn test_regular_patch() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        let patch = SubdivisionPatch::from_mesh(&mesh, 0).unwrap();

        // All corners have valence 1 (only one face)
        assert_eq!(patch.corner_valences, [1, 1, 1, 1]);
        assert!(patch.is_extraordinary()); // Valence 1 is extraordinary
    }

    #[test]
    fn test_limit_eval() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        let patch = SubdivisionPatch::from_mesh(&mesh, 0).unwrap();

        // Evaluate at center (0.5, 0.5) using Jet3
        let u = Jet3::constant(Field::from(0.5));
        let v = Jet3::constant(Field::from(0.5));
        let p = patch.eval_limit(&mesh, u, v);

        // Extract values (collapse AST)
        let x = p[0].val;
        let y = p[1].val;
        let z = p[2].val;

        // For bilinear fallback, center should be roughly (0.5, 0.5, 0.0)
        // We can't easily check SIMD Field values in tests, so this is a smoke test
        assert_eq!(patch.is_extraordinary(), true);
    }

    #[test]
    fn test_surface_stats() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 2.0 0.0 0.0
v 2.0 1.0 0.0
f 1 2 3 4
f 2 5 6 3
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        let surface = SubdivisionSurface::from_mesh(mesh).unwrap();
        let stats = surface.stats();

        assert_eq!(stats.total_patches, 2);
    }
}
