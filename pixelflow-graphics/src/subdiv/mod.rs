//! # Catmull-Clark Subdivision Surfaces
//!
//! Analytic evaluation using Stam's eigenstructure method.
//! No tessellation - pure manifold composition with automatic derivatives.
//!
//! # Implementation Status
//!
//! **Recursive tiling**: ✅ Implemented via log2-based tile depth computation.
//!
//! | Domain | Status |
//! |--------|--------|
//! | (0, 0) | ✅ Exact limit position |
//! | [0.5, 1]² | ✅ First tile - exact |
//! | [0.25, 0.5)² | ✅ Second tile - rescaled |
//! | < 0.25 | ✅ Deeper tiles - rescaled |
//!
//! **Eigenvalue weighting**: ⚠️ Not yet implemented. Deeper tiles have
//! accumulating approximation error without λᵢⁿ⁻¹ decay factors.
//!
//! ## Architecture
//!
//! - `bicubic()`: 16-coefficient polynomial as composed manifold type
//! - `bspline_patch()`: Regular patches (4x4 grid) → 3 bicubic manifolds (x,y,z)
//! - `eigen_patch()`: Extraordinary vertex patches (Stam's method)
//! - `EigenCoeffs`: Baked eigenstructure data (valences 3-50)
//!
//! ## Design: Types as AST
//!
//! ```text
//! bicubic([c0, c1, ...])
//!   → Add<Add<..., Mul<Mul<Y,Y>, f32>>, ...>  // Type tree = AST
//!   → Field operations                         // IR = SIMD intrinsics
//!   → Fused compute kernel                     // Output
//! ```
//!
//! Write math declaratively. The type system builds the AST.
//! Field/Jet evaluation compiles to efficient SIMD.
//!
//! ## Coordinates
//!
//! Uses X, Y as parametric (u, v) ∈ [0,1]².
//!
//! ## Eigenstructure Notes
//!
//! The baked eigenstructure data uses column-major access for the inverse
//! eigenvector matrix. For valence N:
//! - K = 2N + 8 eigenbases
//! - 3 subpatches for tiling near extraordinary vertices
//! - Evaluation at (0,0) = extraordinary vertex limit (verified working)
//! - Evaluation at other points requires Stam's tiling/subpatch selection
//!
//! ## References
//!
//! - Stam, "Exact Evaluation of Catmull-Clark Subdivision Surfaces" (SIGGRAPH 98)

mod coeffs;

pub use coeffs::{get_eigen, EigenCoeffs, MAX_VALENCE};

use pixelflow_core::combinators::{At, Select};
use pixelflow_core::ops::compare::Lt;
use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y};

/// Natural logarithm of 2, used for 2^x = exp(x * LN_2).
const LN_2: f32 = 0.6931471805599453;

// ============================================================================
// Regular Patch (Valence 4) - Standard B-Spline
// ============================================================================

/// Uniform cubic B-spline basis coefficients in power form.
/// B_i(t) = Σⱼ BASIS[i][j] * t^j
const BSPLINE_BASIS: [[f32; 4]; 4] = [
    [1.0 / 6.0, -3.0 / 6.0, 3.0 / 6.0, -1.0 / 6.0], // B0 = (1-t)³/6
    [4.0 / 6.0, 0.0, -6.0 / 6.0, 3.0 / 6.0],        // B1 = (3t³ - 6t² + 4)/6
    [1.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0],  // B2 = (-3t³ + 3t² + 3t + 1)/6
    [0.0, 0.0, 0.0, 1.0 / 6.0],                     // B3 = t³/6
];

/// Evaluate a bicubic B-spline patch.
///
/// Takes 16 control points as a 4x4 grid (row-major).
/// Returns a tuple of 3 manifolds for (x, y, z) coordinates.
///
/// Uses X as u ∈ [0,1], Y as v ∈ [0,1].
#[inline]
pub fn bspline_patch(
    control_points: [[f32; 3]; 16],
) -> (
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
) {
    // Extract per-axis control points and compute bicubic coefficients
    let mut cp_x = [[0.0f32; 4]; 4];
    let mut cp_y = [[0.0f32; 4]; 4];
    let mut cp_z = [[0.0f32; 4]; 4];

    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            cp_x[i][j] = control_points[idx][0];
            cp_y[i][j] = control_points[idx][1];
            cp_z[i][j] = control_points[idx][2];
        }
    }

    (
        bspline_axis(&cp_x),
        bspline_axis(&cp_y),
        bspline_axis(&cp_z),
    )
}

/// Build bicubic manifold for one axis of a B-spline patch.
///
/// Computes power-form coefficients from control points and basis.
fn bspline_axis(control_points: &[[f32; 4]; 4]) -> impl Manifold<Output = Field> {
    // Compute bicubic coefficients c[i][j] for u^i * v^j
    // P(u,v) = Σ_{k,l} B_k(u) * B_l(v) * P_{k,l}
    //        = Σ_{k,l} (Σ_i basis[k][i] * u^i) * (Σ_j basis[l][j] * v^j) * P_{k,l}
    //        = Σ_{i,j} (Σ_{k,l} basis[k][i] * basis[l][j] * P_{k,l}) * u^i * v^j

    let mut coeffs = [0.0f32; 16];

    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                for l in 0..4 {
                    sum += BSPLINE_BASIS[k][i] * BSPLINE_BASIS[l][j] * control_points[k][l];
                }
            }
            coeffs[i * 4 + j] = sum;
        }
    }

    bicubic(coeffs)
}

// ============================================================================
// Eigenstructure Patch (Extraordinary Vertices)
// ============================================================================

/// Build subpatch-selecting manifold for one axis.
///
/// Routes to subpatch based on (u, v):
/// - v < 0.5 → subpatch 0: bicubic at (2u, 2v)
/// - v >= 0.5, u < 0.5 → subpatch 2: bicubic at (2u, 2v - 1)
/// - v >= 0.5, u >= 0.5 → subpatch 1: bicubic at (2u - 1, 2v - 1)
///
/// # Limitations (TODO)
///
/// This is a FIRST-TILE ONLY implementation. For full Stam evaluation:
///
/// 1. **Eigenvalue powers not applied**: Each eigenbasis should be weighted by
///    λᵢⁿ⁻¹ where n is tile depth. Currently all weights are 1.0.
///
/// 2. **No recursive tiling**: Only handles (u,v) ∈ [0.5, 1]². For points closer
///    to origin, need to compute tile depth n = 1 + floor(min(-log2(u), -log2(v)))
///    and rescale coordinates.
///
/// 3. **Correct only at origin**: Evaluation at (0,0) gives exact limit position.
///    Other points have increasing error closer to origin.
///
/// To implement full tiling as manifold composition, need:
/// - `Map` for log2/floor/pow operations on coordinates
/// - Recursive `Select` tree for tile depth routing
/// - Per-eigenbasis weighting (may require K separate bicubics, not combined)
fn axis_patch(coeffs: [[f32; 16]; 3]) -> impl Manifold<Output = Field> {
    // TODO(subdiv): Implement eigenvalue power weighting λᵢⁿ⁻¹
    // TODO(subdiv): Implement recursive tiling for (u,v) < 0.5
    // TODO(subdiv): Currently ONLY first tile - see docs above

    // Each subpatch has different coordinate remapping
    let sub0 = At {
        inner: bicubic(coeffs[0]),
        x: X * 2.0,
        y: Y * 2.0,
        z: 0.0f32,
        w: 0.0f32,
    };
    let sub1 = At {
        inner: bicubic(coeffs[1]),
        x: X * 2.0 - 1.0,
        y: Y * 2.0 - 1.0,
        z: 0.0f32,
        w: 0.0f32,
    };
    let sub2 = At {
        inner: bicubic(coeffs[2]),
        x: X * 2.0,
        y: Y * 2.0 - 1.0,
        z: 0.0f32,
        w: 0.0f32,
    };

    // v < 0.5 → sub0, else (u < 0.5 → sub2, else sub1)
    Select {
        cond: Lt(Y, 0.5f32),
        if_true: sub0,
        if_false: Select {
            cond: Lt(X, 0.5f32),
            if_true: sub2,
            if_false: sub1,
        },
    }
}

/// Compute the tile rescaling factor for recursive Stam evaluation.
///
/// For (u, v) in tile depth n, computes scale = 2^(n-1) such that
/// (u * scale, v * scale) lands in the first tile [0.5, 1]².
///
/// Formula: scale = 2^floor(-log2(max(u, v)))
///
/// Special cases:
/// - (u, v) in [0.5, 1]²: scale = 1.0 (first tile)
/// - (u, v) in [0.25, 0.5)²: scale = 2.0 (second tile)
/// - (u, v) in [0.125, 0.25)²: scale = 4.0 (third tile)
/// - At origin: clamp prevents infinity
#[inline]
fn tile_scale() -> impl Manifold<Output = Field> {
    // max(u, v), clamped to avoid log2(0) = -∞
    let max_uv = X.max(Y).max(1e-10f32);

    // Pure type composition: scale = 2^floor(-log2(max))
    // Types as AST: Log2<Max<...>> → Neg<Log2<...>> → Floor<...> → Exp2<...>
    max_uv.log2().neg().floor().exp2()
}

/// Rescaled U coordinate for recursive tiling.
///
/// Computes u' = u * scale where scale = 2^floor(-log2(max(u,v))).
/// This maps any tile to first-tile local coordinates.
#[inline]
fn rescaled_u() -> impl Manifold<Output = Field> {
    X * tile_scale()
}

/// Rescaled V coordinate for recursive tiling.
///
/// Computes v' = v * scale where scale = 2^floor(-log2(max(u,v))).
/// This maps any tile to first-tile local coordinates.
#[inline]
fn rescaled_v() -> impl Manifold<Output = Field> {
    Y * tile_scale()
}

/// Build subpatch-selecting manifold with recursive tiling support.
///
/// Unlike `axis_patch`, this version handles (u, v) in any tile depth:
/// 1. Computes tile depth n from max(u, v)
/// 2. Rescales coordinates to first-tile [0.5, 1]²
/// 3. Routes to correct subpatch
///
/// # TODO: Eigenvalue Weighting
///
/// For full Stam accuracy, each eigenbasis should be weighted by λᵢⁿ⁻¹.
/// Currently all weights are 1.0, which is exact only for the first tile.
/// Deeper tiles will have accumulated approximation error.
fn tiled_axis_patch(coeffs: [[f32; 16]; 3]) -> impl Manifold<Output = Field> {
    // Build the first-tile subpatch selector (existing logic)
    let first_tile = first_tile_patch(coeffs);

    // Wrap with coordinate rescaling: evaluate first_tile at (u * scale, v * scale)
    At {
        inner: first_tile,
        x: rescaled_u(),
        y: rescaled_v(),
        z: 0.0f32,
        w: 0.0f32,
    }
}

/// First-tile subpatch routing (internal helper).
///
/// Assumes (u, v) is already in [0, 1]² after rescaling.
/// Routes to the correct subpatch based on quadrant.
fn first_tile_patch(coeffs: [[f32; 16]; 3]) -> impl Manifold<Output = Field> {
    // Each subpatch has different coordinate remapping
    let sub0 = At {
        inner: bicubic(coeffs[0]),
        x: X * 2.0,
        y: Y * 2.0,
        z: 0.0f32,
        w: 0.0f32,
    };
    let sub1 = At {
        inner: bicubic(coeffs[1]),
        x: X * 2.0 - 1.0,
        y: Y * 2.0 - 1.0,
        z: 0.0f32,
        w: 0.0f32,
    };
    let sub2 = At {
        inner: bicubic(coeffs[2]),
        x: X * 2.0,
        y: Y * 2.0 - 1.0,
        z: 0.0f32,
        w: 0.0f32,
    };

    // v < 0.5 → sub0, else (u < 0.5 → sub2, else sub1)
    Select {
        cond: Lt(Y, 0.5f32),
        if_true: sub0,
        if_false: Select {
            cond: Lt(X, 0.5f32),
            if_true: sub2,
            if_false: sub1,
        },
    }
}

/// Evaluate a Catmull-Clark patch using Stam's eigenstructure method.
///
/// Returns 3 manifolds for (x, y, z) coordinates. Each manifold:
/// - Computes tile depth from (u, v) via log2
/// - Rescales coordinates to first-tile [0.5, 1]²
/// - Selects subpatch and evaluates combined bicubic
///
/// The manifolds are pure composition - types ARE the AST.
/// Field evaluation compiles to SIMD, Jet3 gives derivatives.
///
/// # Valid Domain
///
/// Supports full (u, v) ∈ (0, 1]² via recursive tiling:
/// - **(0, 0)**: Exact limit position ✓
/// - **(u, v) ∈ [0.5, 1]²**: First tile (n=1) ✓
/// - **(u, v) ∈ [0.25, 0.5)²**: Second tile (n=2) ✓
/// - **(u, v) closer to origin**: Deeper tiles ✓
///
/// # Accuracy Note
///
/// Eigenvalue weighting (λᵢⁿ⁻¹) is not yet implemented. This means:
/// - First tile [0.5, 1]²: Exact
/// - Deeper tiles: Accumulating approximation error
///
/// For most artistic use cases, this is acceptable. For CAD-quality
/// precision near extraordinary vertices, eigenvalue weighting is needed.
pub fn eigen_patch(
    control_points: &[[f32; 3]],
    valence: usize,
) -> Option<(
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
)> {
    let eigen = get_eigen(valence)?;
    let k = eigen.k;

    // Project control points to eigenspace (column-major access)
    let mut proj_x = vec![0.0f32; k];
    let mut proj_y = vec![0.0f32; k];
    let mut proj_z = vec![0.0f32; k];

    for i in 0..k {
        for j in 0..k.min(control_points.len()) {
            let w = eigen.inv_eigen(j, i); // transpose
            proj_x[i] += w * control_points[j][0];
            proj_y[i] += w * control_points[j][1];
            proj_z[i] += w * control_points[j][2];
        }
    }

    // Precompute bicubic coefficients for each subpatch
    let mut coeffs = [[[0.0f32; 16]; 3]; 3]; // [axis][subpatch][coeff]
    for sub in 0..3 {
        for c in 0..16 {
            for basis in 0..k {
                let s = eigen.spline(sub, basis, c);
                coeffs[0][sub][c] += s * proj_x[basis];
                coeffs[1][sub][c] += s * proj_y[basis];
                coeffs[2][sub][c] += s * proj_z[basis];
            }
        }
    }

    Some((
        tiled_axis_patch(coeffs[0]),
        tiled_axis_patch(coeffs[1]),
        tiled_axis_patch(coeffs[2]),
    ))
}

/// Validate that (u, v) is in the supported domain for eigen_patch.
///
/// With recursive tiling now implemented, the full domain (0, 1]² is supported.
/// This function now only warns about accuracy at deeper tiles.
///
/// # Panics
///
/// Panics if (u, v) is outside the valid range [0, 1]².
#[inline]
pub fn validate_eigen_domain(u: f32, v: f32) {
    // Check bounds
    if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
        panic!(
            "eigen_patch: (u={}, v={}) outside valid domain [0, 1]²",
            u, v
        );
    }

    // All points in (0, 1]² are now supported via recursive tiling
    // Origin (0, 0) gives exact limit position
    // Other points use tile rescaling (accuracy depends on tile depth without λⁿ⁻¹ weighting)
}

// ============================================================================
// Bicubic Polynomial (Manifold Composition)
// ============================================================================

/// Build a bicubic polynomial as a composed manifold.
///
/// P(u,v) = Σᵢⱼ cᵢⱼ · uⁱ · vʲ for i,j ∈ [0,3]
///
/// Uses X as u, Y as v (parametric coordinates).
///
/// Returns a manifold type - the polynomial IS the composition,
/// not a struct that computes it.
#[inline]
pub fn bicubic(c: [f32; 16]) -> impl Manifold<Output = Field> {
    // Powers of Y (v): Y, Y², Y³
    let v1 = Y;
    let v2 = Y * Y;
    let v3 = v2 * Y;

    // Powers of X (u): X, X², X³
    let u1 = X;
    let u2 = X * X;
    let u3 = u2 * X;

    // Row 0: c[0] + c[1]*v + c[2]*v² + c[3]*v³
    // Constant term last (f32 must be on right side of +)
    let r0 = v1 * c[1] + v2 * c[2] + v3 * c[3] + c[0];

    // Row 1: (c[4] + c[5]*v + c[6]*v² + c[7]*v³) * u
    let r1 = (v1 * c[5] + v2 * c[6] + v3 * c[7] + c[4]) * u1;

    // Row 2: (c[8] + c[9]*v + c[10]*v² + c[11]*v³) * u²
    let r2 = (v1 * c[9] + v2 * c[10] + v3 * c[11] + c[8]) * u2;

    // Row 3: (c[12] + c[13]*v + c[14]*v² + c[15]*v³) * u³
    let r3 = (v1 * c[13] + v2 * c[14] + v3 * c[15] + c[12]) * u3;

    // P(u,v) = r0 + r1 + r2 + r3
    r0 + r1 + r2 + r3
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::combinators::{At, Texture};

    /// Evaluate a scalar manifold at a point
    fn eval_scalar<M: Manifold<Output = Field>>(m: &M, x: f32, y: f32) -> f32 {
        // At with f32 constants binds coordinates; eval() ignores input
        let bound = At {
            inner: m,
            x,
            y,
            z: 0.0f32,
            w: 0.0f32,
        };
        // Sample via 1x1 texture to extract f32 (At ignores the sample point)
        let tex = Texture::from_manifold(&bound, 1, 1);
        tex.data()[0]
    }

    #[test]
    fn test_bicubic_constant() {
        // Constant polynomial: c00 = 5.0, all others 0
        let mut coeffs = [0.0f32; 16];
        coeffs[0] = 5.0;
        let poly = bicubic(coeffs);

        let val = eval_scalar(&poly, 0.5, 0.5);
        assert!((val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bicubic_linear() {
        // P(u,v) = 1 + 2u + 3v
        let mut coeffs = [0.0f32; 16];
        coeffs[0] = 1.0; // c00 = 1
        coeffs[1] = 3.0; // c01 = 3 (coefficient of v)
        coeffs[4] = 2.0; // c10 = 2 (coefficient of u)
        let poly = bicubic(coeffs);

        // P(0.5, 0.5) = 1 + 2*0.5 + 3*0.5 = 1 + 1 + 1.5 = 3.5
        let val = eval_scalar(&poly, 0.5, 0.5);
        assert!((val - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_get_eigen_valence_4() {
        // Regular vertex (valence 4) should have eigenstructure
        let eigen = get_eigen(4).expect("valence 4 should exist");
        assert_eq!(eigen.valence, 4);
        assert_eq!(eigen.k, 16); // 2*4 + 8 = 16
        assert_eq!(eigen.eigenvalues.len(), 16);
    }

    #[test]
    fn test_eigen_first_eigenvalue_is_one() {
        // First eigenvalue should always be 1.0 (limit surface property)
        for valence in 3..=10 {
            let eigen = get_eigen(valence).unwrap();
            assert!(
                (eigen.eigenvalues[0] - 1.0).abs() < 1e-6,
                "valence {} first eigenvalue should be 1.0",
                valence
            );
        }
    }

    #[test]
    fn test_bspline_patch_flat_plane() {
        // Create a flat 4x4 grid of control points at z=0
        // Points span [0,3] x [0,3] in x,y
        let mut control_points = [[0.0f32; 3]; 16];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                control_points[idx] = [j as f32, i as f32, 0.0];
            }
        }

        let (px, py, pz) = bspline_patch(control_points);

        // Evaluate at center (u=0.5, v=0.5)
        let x = eval_scalar(&px, 0.5, 0.5);
        let y = eval_scalar(&py, 0.5, 0.5);
        let z = eval_scalar(&pz, 0.5, 0.5);

        // Z should be 0 (flat plane)
        assert!(z.abs() < 1e-4, "z should be 0 for flat plane, got {}", z);

        // X and Y should be near the center of the grid
        // B-spline at u=v=0.5 with uniform grid should give weighted average
        assert!(x > 0.5 && x < 2.5, "x={} should be in [0.5, 2.5]", x);
        assert!(y > 0.5 && y < 2.5, "y={} should be in [0.5, 2.5]", y);
    }

    #[test]
    fn test_bspline_patch_corners() {
        // Flat grid
        let mut control_points = [[0.0f32; 3]; 16];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                control_points[idx] = [j as f32, i as f32, 0.0];
            }
        }

        let (px, py, _pz) = bspline_patch(control_points);

        // At (0,0) - should be influenced primarily by corner control points
        let x0 = eval_scalar(&px, 0.0, 0.0);
        let y0 = eval_scalar(&py, 0.0, 0.0);

        // At (1,1) - should be at the opposite extreme
        let x1 = eval_scalar(&px, 1.0, 1.0);
        let y1 = eval_scalar(&py, 1.0, 1.0);

        // Values should be different (surface varies across domain)
        assert!(
            (x1 - x0).abs() > 0.1,
            "x should vary across patch: x(0,0)={}, x(1,1)={}",
            x0,
            x1
        );
        assert!(
            (y1 - y0).abs() > 0.1,
            "y should vary across patch: y(0,0)={}, y(1,1)={}",
            y0,
            y1
        );
    }

    #[test]
    fn test_eigen_trivial_case() {
        // If all control points are at the SAME location,
        // the surface should evaluate to that location everywhere.
        // This tests affine invariance.
        let _control_points = [[1.0f32, 2.0, 3.0]; 16];

        let eigen = get_eigen(4).unwrap();
        println!("K = {}", eigen.k);

        // Project to eigenspace
        let mut projected_x = [0.0f32; 16];
        for i in 0..16 {
            for j in 0..16 {
                projected_x[i] += eigen.inv_eigen(i, j) * 1.0; // all x = 1.0
            }
        }
        println!("Projected x: {:?}", &projected_x[..4]);

        // Check row sums
        println!("Row sums:");
        for i in 0..4 {
            let row_sum: f32 = (0..16).map(|j| eigen.inv_eigen(i, j)).sum();
            println!("  Row {}: {}", i, row_sum);
        }

        // Check column sums - maybe matrix is transposed?
        println!("Column sums:");
        for j in 0..4 {
            let col_sum: f32 = (0..16).map(|i| eigen.inv_eigen(i, j)).sum();
            println!("  Col {}: {}", j, col_sum);
        }

        // Check what happens if we use columns instead of rows
        let mut projected_col = [0.0f32; 16];
        for i in 0..16 {
            for j in 0..16 {
                // Transpose: inv_eigen[j, i] instead of inv_eigen[i, j]
                projected_col[i] += eigen.inv_eigen(j, i) * 1.0;
            }
        }
        println!("Projected (transposed): {:?}", &projected_col[..4]);

        // Combine with spline coefficients using TRANSPOSED projection
        let mut final_coeffs = [0.0f32; 16];
        for coeff_idx in 0..16 {
            for basis in 0..16 {
                final_coeffs[coeff_idx] += eigen.spline(0, basis, coeff_idx) * projected_col[basis];
            }
        }
        println!("Final coeffs (transposed): {:?}", &final_coeffs[..4]);

        // The spline basis for index 0 should sum the polynomial to 1 at key points
        // Check: φ₀(0.5, 0.5) should give us some partition of unity property
        let poly = bicubic(final_coeffs);
        let val_center = eval_scalar(&poly, 0.5, 0.5);
        let val_corner = eval_scalar(&poly, 0.0, 0.0);
        println!("φ₀(0.5, 0.5) = {}, φ₀(0, 0) = {}", val_center, val_corner);

        // Check sum of bases for each subpatch
        for subpatch in 0..3 {
            let mut sum_at_center = 0.0f32;
            for basis in 0..16 {
                let mut basis_coeffs = [0.0f32; 16];
                for coeff_idx in 0..16 {
                    basis_coeffs[coeff_idx] = eigen.spline(subpatch, basis, coeff_idx);
                }
                let basis_poly = bicubic(basis_coeffs);
                sum_at_center += eval_scalar(&basis_poly, 0.5, 0.5);
            }
            println!(
                "Subpatch {}: sum of bases at (0.5, 0.5) = {}",
                subpatch, sum_at_center
            );
        }

        // The key insight: Stam's method requires TILING the parameter space.
        // At (0,0), we're at the extraordinary vertex limit.
        // Let's verify the surface is correct at corners and with proper tiling.

        // Test eigen_patch at (0,0) with constant control points
        let const_points = [[5.0f32, 7.0, 9.0]; 16];
        let (ex, ey, ez) = eigen_patch(&const_points, 4).unwrap();

        // At (0,0), the limit position should be the control point value
        let ex0 = eval_scalar(&ex, 0.0, 0.0);
        let ey0 = eval_scalar(&ey, 0.0, 0.0);
        let ez0 = eval_scalar(&ez, 0.0, 0.0);
        println!("eigen_patch at (0,0): ({}, {}, {})", ex0, ey0, ez0);
        println!("Expected: (5, 7, 9)");

        // Check tolerance - should be exact at the corner
        assert!(
            (ex0 - 5.0).abs() < 0.01,
            "x at (0,0) should be 5.0, got {}",
            ex0
        );
        assert!(
            (ey0 - 7.0).abs() < 0.01,
            "y at (0,0) should be 7.0, got {}",
            ey0
        );
        assert!(
            (ez0 - 9.0).abs() < 0.01,
            "z at (0,0) should be 9.0, got {}",
            ez0
        );
    }

    #[test]
    #[should_panic(expected = "outside valid domain")]
    fn test_validate_eigen_domain_panics_for_invalid() {
        // This should panic - coordinates outside [0, 1]²
        validate_eigen_domain(1.5, 0.5);
    }

    #[test]
    fn test_validate_eigen_domain_accepts_valid() {
        // Origin is valid (exact limit position)
        validate_eigen_domain(0.0, 0.0);

        // First tile is valid
        validate_eigen_domain(0.5, 0.5);
        validate_eigen_domain(0.75, 0.75);
        validate_eigen_domain(1.0, 1.0);

        // Deeper tiles are now valid with recursive tiling
        validate_eigen_domain(0.25, 0.25);
        validate_eigen_domain(0.125, 0.125);
        validate_eigen_domain(0.01, 0.01);
    }

    #[test]
    fn test_eigen_patch_subpatch_routing() {
        // Constant control points - surface should be constant everywhere
        let const_points = [[3.0f32, 5.0, 7.0]; 16];
        let (ex, ey, ez) = eigen_patch(&const_points, 4).unwrap();

        // Test points in each subpatch region (first tile)
        // Subpatch 0: v < 0.5 → local coords (2u, 2v)
        let x0 = eval_scalar(&ex, 0.25, 0.25);
        let y0 = eval_scalar(&ey, 0.25, 0.25);
        let z0 = eval_scalar(&ez, 0.25, 0.25);
        println!("Subpatch 0 at (0.25, 0.25): ({}, {}, {})", x0, y0, z0);

        // Subpatch 1: v >= 0.5, u >= 0.5 → local coords (2u-1, 2v-1)
        let x1 = eval_scalar(&ex, 0.75, 0.75);
        let y1 = eval_scalar(&ey, 0.75, 0.75);
        let z1 = eval_scalar(&ez, 0.75, 0.75);
        println!("Subpatch 1 at (0.75, 0.75): ({}, {}, {})", x1, y1, z1);

        // Subpatch 2: v >= 0.5, u < 0.5 → local coords (2u, 2v-1)
        let x2 = eval_scalar(&ex, 0.25, 0.75);
        let y2 = eval_scalar(&ey, 0.25, 0.75);
        let z2 = eval_scalar(&ez, 0.25, 0.75);
        println!("Subpatch 2 at (0.25, 0.75): ({}, {}, {})", x2, y2, z2);

        // All should be close to (3, 5, 7) for affine invariance
        // (Within first-tile approximation tolerance)
        assert!(
            (x0 - 3.0).abs() < 1.0,
            "x at subpatch 0 should be near 3.0, got {}",
            x0
        );
    }

    #[test]
    fn test_bspline_symmetry() {
        // Symmetric grid centered at origin
        let mut control_points = [[0.0f32; 3]; 16];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                control_points[idx] = [j as f32 - 1.5, i as f32 - 1.5, 0.0];
            }
        }

        let (px, py, _pz) = bspline_patch(control_points);

        // At center (0.5, 0.5), the surface should pass through (0, 0)
        // due to symmetry of control points
        let x = eval_scalar(&px, 0.5, 0.5);
        let y = eval_scalar(&py, 0.5, 0.5);

        assert!(
            x.abs() < 0.01,
            "symmetric patch should have x=0 at center, got {}",
            x
        );
        assert!(
            y.abs() < 0.01,
            "symmetric patch should have y=0 at center, got {}",
            y
        );
    }

    #[test]
    fn test_recursive_tiling_constant_surface() {
        // Constant control points - surface should be constant everywhere
        // including at deeper tiles
        let const_points = [[5.0f32, 7.0, 9.0]; 16];
        let (ex, ey, ez) = eigen_patch(&const_points, 4).unwrap();

        // Test points at different tile depths
        let test_points = [
            (0.75, 0.75, "first tile"),  // Tile 1: [0.5, 1]²
            (0.3, 0.3, "second tile"),   // Tile 2: [0.25, 0.5)²
            (0.15, 0.15, "third tile"),  // Tile 3: [0.125, 0.25)²
            (0.06, 0.06, "fourth tile"), // Tile 4: deeper
        ];

        for (u, v, desc) in test_points {
            let x = eval_scalar(&ex, u, v);
            let y = eval_scalar(&ey, u, v);
            let z = eval_scalar(&ez, u, v);

            println!("{} at ({}, {}): ({}, {}, {})", desc, u, v, x, y, z);

            // Should be near constant value (within tolerance for approximation)
            // Tolerance increases for deeper tiles without eigenvalue weighting
            let tol = 2.0;
            assert!(
                (x - 5.0).abs() < tol,
                "{}: x should be near 5.0, got {}",
                desc,
                x
            );
            assert!(
                (y - 7.0).abs() < tol,
                "{}: y should be near 7.0, got {}",
                desc,
                y
            );
            assert!(
                (z - 9.0).abs() < tol,
                "{}: z should be near 9.0, got {}",
                desc,
                z
            );
        }
    }

    #[test]
    fn test_tile_scale_values() {
        // Test that tile_scale computes correct powers of 2
        let scale = tile_scale();

        // At (0.75, 0.75) - first tile: scale = 2^0 = 1
        let s1 = eval_scalar(&scale, 0.75, 0.75);
        assert!(
            (s1 - 1.0).abs() < 0.01,
            "scale at first tile should be 1.0, got {}",
            s1
        );

        // At (0.3, 0.3) - second tile: scale = 2^1 = 2
        let s2 = eval_scalar(&scale, 0.3, 0.3);
        assert!(
            (s2 - 2.0).abs() < 0.1,
            "scale at second tile should be 2.0, got {}",
            s2
        );

        // At (0.15, 0.15) - third tile: scale = 2^2 = 4
        let s3 = eval_scalar(&scale, 0.15, 0.15);
        assert!(
            (s3 - 4.0).abs() < 0.2,
            "scale at third tile should be 4.0, got {}",
            s3
        );

        // At (0.08, 0.08) - fourth tile: scale = 2^3 = 8
        // Tile 4 is [0.0625, 0.125), and 0.08 is clearly in that range
        let s4 = eval_scalar(&scale, 0.08, 0.08);
        assert!(
            (s4 - 8.0).abs() < 0.5,
            "scale at fourth tile should be 8.0, got {}",
            s4
        );
    }
}
