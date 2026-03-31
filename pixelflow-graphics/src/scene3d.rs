//! Three-Layer Pull-Based Architecture:
//! 1. Geometry: Returns `t` (Jet3)
//! 2. Surface: Warps `P = ray * t` (Creates tangent frame via Chain Rule)
//! 3. Material: Reconstructs Normal from `P` derivatives
//!
//! ## Architecture
//!
//! The "mullet" approach for full-color rendering:
//! - **Front (serious)**: Geometry computed ONCE per pixel via Jet3
//! - **Back (party)**: Colors flow as opaque `Discrete` (packed RGBA)
//!
//! This gives 3x speedup vs running geometry 3x (once per R,G,B channel).
//!
//! No iteration. Nesting is occlusion.

use pixelflow_core::jet::Jet3;
use pixelflow_core::*;
use pixelflow_compiler::{kernel, ManifoldExpr};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// The 4D Jet3 domain type for 3D ray tracing autodiff.
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

/// The 4D PathJet domain type for recursive ray tracing.
type PathJet4 = (PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>);

// ============================================================================
// LIFT: Field manifold → Jet3 manifold (explicit conversion)
// ============================================================================

/// Lifts a Field-based manifold to work with Jet3 inputs.
///
/// Uses `From<Jet3> for Field` to project jet coordinates to values,
/// discarding derivatives. Use this for constant-valued manifolds
/// (like Color) that don't need derivative information.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct Lift<M>(pub M);

impl<M: ManifoldCompat<Field> + Send + Sync> Manifold<Jet3_4> for Lift<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, p: Jet3_4) -> Self::Output {
        let (x, y, z, w) = p;
        self.0.eval_raw(x.into(), y.into(), z.into(), w.into())
    }
}

// ============================================================================
// HELPER: Lift Field mask to Jet3 manifold for Select conditions
// ============================================================================

/// Wraps a Field mask to implement Manifold<Jet3> for use as a Select condition.
/// This is needed because Select<C, T, F> for Jet3 requires C: ManifoldCompat<Jet3, Output = Jet3>.
#[derive(Clone, Copy)]
struct FieldMask(Field);

impl Manifold<Jet3_4> for FieldMask {
    type Output = Jet3;

    #[inline]
    fn eval(&self, _p: Jet3_4) -> Jet3 {
        // Convert Field mask to Jet3 with zero derivatives
        Jet3::constant(self.0)
    }
}

// ============================================================================
// ROOT: ScreenToDir
// ============================================================================

/// Converts screen coordinates to ray direction jets.
///
/// **CRITICAL**: This must seed the derivatives correctly.
/// - Screen X changes by 1.0 per pixel (dx=1, dy=0)
/// - Screen Y changes by 1.0 per pixel (dx=0, dy=1)
/// - Direction is normalized(Screen X, Screen Y, 1.0)
///
/// The Chain Rule propagates these derivatives into the ray direction,
/// allowing Materials to know "how the ray changes" across the pixel.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct ScreenToDir<M> {
    pub inner: M,
}

impl<M: ManifoldCompat<Jet3, Output = Field>> Manifold<Field4> for ScreenToDir<M> {
    type Output = Field;

    #[inline]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, _z, w) = p;
        // 1. Seed Jets from Screen Coords
        // x: varies with screen x (dx=1, dy=0, dz=0)
        let sx = Jet3::x(x);
        // y: varies with screen y (dx=0, dy=1, dz=0)
        let sy = Jet3::y(y);
        // z: constant 1.0 (pinhole focal length, no derivatives)
        let sz = Jet3::constant(Field::from(1.0));

        // 2. Normalize to get Ray Direction
        // The Jet math automatically computes d(dir)/dx and d(dir)/dy
        let len_sq = sx * sx + sy * sy + sz * sz;
        let len = len_sq.sqrt();

        let dx = sx / len;
        let dy = sy / len;
        let dz = sz / len;

        // 3. Pass pure direction Jets to the scene
        self.inner.eval_raw(dx, dy, dz, Jet3::constant(w))
    }
}

/// Converts screen coordinates to ray direction jets, outputting Discrete.
///
/// Same as ScreenToDir but for color (Discrete) pipelines.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct ColorScreenToDir<M> {
    pub inner: M,
}

impl<M: ManifoldCompat<Jet3, Output = Discrete>> Manifold<Field4> for ColorScreenToDir<M> {
    type Output = Discrete;

    #[inline]
    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, _z, w) = p;
        let sx = Jet3::x(x);
        let sy = Jet3::y(y);
        let sz = Jet3::constant(Field::from(1.0));

        let len_sq = sx * sx + sy * sy + sz * sz;
        let len = len_sq.sqrt();

        let dx = sx / len;
        let dy = sy / len;
        let dz = sz / len;

        self.inner.eval_raw(dx, dy, dz, Jet3::constant(w))
    }
}

// ============================================================================
// LAYER 1: Geometry (Returns t)
// ============================================================================

/// Unit sphere geometry kernel.
///
/// Computes t = 1/|ray| such that |t * ray| = 1.
/// Returns Jet3 with value and partial derivatives.
#[derive(Clone, Copy, Default, ManifoldExpr)]
pub struct UnitSphere;

impl Manifold<Jet3_4> for UnitSphere {
    type Output = Jet3;

    #[inline(always)]
    fn eval(&self, (x, y, z, _w): Jet3_4) -> Jet3 {
        let one = Jet3::constant(Field::from(1.0));
        let len_sq = x * x + y * y + z * z;
        one / len_sq.sqrt()
    }
}

/// Unit sphere centered at origin.
/// Solves |t * ray| = 1  =>  t = 1 / |ray|
#[inline]
pub fn unit_sphere() -> UnitSphere {
    UnitSphere
}

/// Horizontal plane geometry kernel.
#[derive(Copy, Clone, ManifoldExpr)]
pub struct PlaneKernel {
    h: f32,
}

impl Manifold<Jet3_4> for PlaneKernel {
    type Output = Jet3;
    #[inline(always)]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let h = Jet3::from(Field::from(self.h));
        h / p.1 // h / Y
    }
}

/// Horizontal plane at y = height.
/// Solves P.y = height => t * ry = height => t = height / ry
#[allow(dead_code)]
pub fn plane(height: f32) -> PlaneKernel {
    PlaneKernel { h: height }
}

// ============================================================================
// PATHJET GEOMETRY: Spheres with arbitrary ray origins
// ============================================================================

/// Sphere for PathJet rays (supports arbitrary ray origins).
///
/// Unlike the Jet3_4 sphere which assumes rays from camera at origin,
/// this handles rays with explicit origin and direction components.
///
/// Ray equation: P(t) = O + t*D
/// Sphere: |P - C|² = r²
/// Solution: t = -(oc·D) - sqrt((oc·D)² - (|oc|² - r²))
///   where oc = O - C (vector from center to ray origin)
#[derive(Clone, Copy, ManifoldExpr)]
pub struct PathJetSphere {
    pub center: (f32, f32, f32),
    pub radius: f32,
}

impl PathJetSphere {
    pub fn new(center: (f32, f32, f32), radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Manifold<PathJet4> for PathJetSphere {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: PathJet4) -> Jet3 {
        let (x, y, z, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let eps = Jet3::constant(Field::from(0.0001));

        // oc = O - C (origin minus center)
        let oc_x = x.val - cx;
        let oc_y = y.val - cy;
        let oc_z = z.val - cz;

        // Direction (assume normalized or will normalize later)
        let dx = x.dir;
        let dy = y.dir;
        let dz = z.dir;

        // oc·D (dot product)
        let oc_dot_d = oc_x * dx + oc_y * dy + oc_z * dz;

        // |oc|²
        let oc_sq = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z;

        // discriminant = (oc·D)² - (|oc|² - r²)
        let discriminant = oc_dot_d * oc_dot_d - (oc_sq - r_sq);

        // t = -(oc·D) - sqrt(discriminant + epsilon)
        let zero = Jet3::constant(Field::from(0.0));
        let neg_oc_dot_d = zero - oc_dot_d;
        neg_oc_dot_d - (discriminant + eps).sqrt()
    }
}

/// Also implement Jet3_4 for PathJetSphere (backwards compatibility).
/// When used with Jet3_4, assumes origin = 0 (camera at origin).
impl Manifold<Jet3_4> for PathJetSphere {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let eps = Jet3::constant(Field::from(0.0001));

        // For origin at 0: oc = -C
        // oc·D = -C·D = -(D·C)
        // So -(oc·D) = D·C
        let d_dot_c = rx * cx + ry * cy + rz * cz;

        // |oc|² = |C|²
        let c_sq = cx * cx + cy * cy + cz * cz;

        // discriminant = (D·C)² - (|C|² - r²)
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        // t = (D·C) - sqrt(discriminant + epsilon)
        d_dot_c - (discriminant + eps).sqrt()
    }
}

/// Height field geometry: z = base_height + scale * f(x, y)
///
/// Single-step intersection: hit base plane, sample height, adjust t.
/// No iteration - just one evaluation of the height manifold.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct HeightFieldGeometry<H> {
    pub height_field: H,
    pub base_height: f32,
    pub scale: f32,
    pub uv_scale: f32, // Maps world coords to (u, v) parameter space
    pub center_x: f32, // World x offset (patch centered here)
    pub center_z: f32, // World z offset (patch centered here)
}

impl<H: ManifoldCompat<Field, Output = Field>> Manifold<Jet3_4> for HeightFieldGeometry<H> {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        // Step 1: Hit base plane at y = base_height
        let t_plane = Jet3::constant(Field::from(self.base_height)) / ry;

        // Step 2: Get (x, z) world coords at plane hit (note: scene uses y-up)
        let hit_x = rx * t_plane;
        let hit_z = rz * t_plane;

        // Step 3: Map to (u, v) centered on (center_x, center_z)
        let uv_scale = Field::from(self.uv_scale);
        let half = Field::from(0.5);
        let u = ((hit_x.val - Field::from(self.center_x)) * uv_scale + half).constant();
        let v = ((hit_z.val - Field::from(self.center_z)) * uv_scale + half).constant();

        // Bounds check: (u, v) must be in [0, 1]
        let zero = Field::from(0.0);
        let one = Field::from(1.0);
        let in_bounds = u.ge(zero) & u.le(one) & v.ge(zero) & v.le(one);

        // Sample height field
        let h = self.height_field.eval_raw(u, v, zero, zero);

        // Step 4: Adjust t for height displacement
        let effective_height =
            (Field::from(self.base_height) + Field::from(self.scale) * h).constant();
        let t_hit = Jet3::constant(effective_height) / ry;

        // Return valid t if in bounds, else negative (miss)
        let miss = Field::from(-1.0);
        Jet3::new(
            in_bounds.select(t_hit.val, miss),
            in_bounds.select(t_hit.dx, miss),
            in_bounds.select(t_hit.dy, miss),
            in_bounds.select(t_hit.dz, miss),
        )
    }
}

// ============================================================================
// LAYER 2: Surface (The Warp)
// ============================================================================

/// The Glue. Combines Geometry, Material, and Background.
///
/// Performs **The Warp**: `P = ray * t`.
/// Because `t` carries derivatives from Layer 1, and `ray` carries derivatives
/// from Root, `P` automatically contains the Surface Tangent Frame via the Chain Rule.
///
/// Evaluates geometry to get t, computes hit point P = ray * t, then selects
/// between material (at P) and background based on hit validity.
kernel!(pub struct Surface = |geometry: kernel, material: kernel, background: kernel| Jet3 -> Field {
    // 1. Get distance t from geometry
    let t = geometry;

    // 2. Validate hit: t > 0, t < max, derivatives reasonable
    let t_max = 1000000.0;
    let deriv_max = 10000.0;
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);
    let mask = valid_t & valid_deriv;

    // 3. Hit point: P = ray * t (always computed; Select short-circuits if mask is all-false)
    let hx = X * t;
    let hy = Y * t;
    let hz = Z * t;

    // 4. Sample material at hit point, background at ray direction
    let mat_val = material.at(hx, hy, hz, W);
    let bg_val = background;

    // 5. Select based on hit validity (short-circuit avoids evaluating unused branch)
    mask.select(mat_val, bg_val)
});

/// Color Surface: geometry + material + background, outputs Discrete.
kernel!(pub struct ColorSurface = |geometry: kernel, material: kernel, background: kernel| Jet3 -> Discrete {
    // 1. Get distance t from geometry
    let t = geometry;

    // 2. Validate hit: t > 0, t < max, derivatives reasonable
    let t_max = 1000000.0;
    let deriv_max = 10000.0;
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);
    let mask = valid_t & valid_deriv;

    // 3. Hit point: P = ray * t (always computed; Select short-circuits if mask is all-false)
    let hx = X * t;
    let hy = Y * t;
    let hz = Z * t;

    // 4. Sample material at hit point, background at ray direction
    let mat_val = material.at(hx, hy, hz, W);
    let bg_val = background;

    // 5. Select based on hit validity (short-circuit avoids evaluating unused branch)
    mask.select(mat_val, bg_val)
});

// ... SCENE COMPOSITION ...

// ============================================================================
// SCENE COMPOSITION: Union via priority order
// ============================================================================

/// A Scene separates hit detection (mask) from appearance (color).
///
/// This enables Union composition: check S1 first, if miss check S2.
/// "First hit in scene graph wins" - not distance-based, but priority-based.
pub trait Scene {
    /// Mask manifold: evaluates to positive where ray hits this scene.
    type Mask: ManifoldCompat<Jet3, Output = Field>;
    /// Color manifold: evaluates to the color at the hit point.
    type Color: ManifoldCompat<Jet3, Output = Discrete>;

    fn mask(&self) -> Self::Mask;
    fn color(&self) -> Self::Color;
}

/// Union of two scenes: first hit wins.
///
/// Evaluates S1's mask first. If hit, use S1's color.
/// Otherwise, evaluate S2's mask. If hit, use S2's color.
/// Otherwise, use background.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct Union<S1, S2, B> {
    pub first: S1,
    pub second: S2,
    pub background: B,
}

impl<S1, S2, B> Manifold<Jet3_4> for Union<S1, S2, B>
where
    S1: Scene + Send + Sync,
    S2: Scene + Send + Sync,
    B: ManifoldCompat<Jet3, Output = Discrete>,
{
    type Output = Discrete;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Discrete {
        let (rx, ry, rz, w) = p;
        // First hit wins: nested Select
        // Select(S1.mask, S1.color, Select(S2.mask, S2.color, background))
        let m1 = self.first.mask();
        let c1 = self.first.color();
        let m2 = self.second.mask();
        let c2 = self.second.color();

        let mask1 = m1.eval_raw(rx, ry, rz, w);
        let mask2 = m2.eval_raw(rx, ry, rz, w);

        // Inner select: S2 vs background
        let color2 = c2.eval_raw(rx, ry, rz, w);
        let bg_color = self.background.eval_raw(rx, ry, rz, w);
        let inner = Discrete::select(mask2, color2, bg_color);

        // Outer select: S1 vs inner
        let color1 = c1.eval_raw(rx, ry, rz, w);
        Discrete::select(mask1, color1, inner)
    }
}

/// Simple scene wrapper: geometry + material with explicit mask exposure.
///
/// Unlike ColorSurface which hides the mask, SceneObject exposes it
/// for use in Union composition.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct SceneObject<G, M> {
    pub geometry: G,
    pub material: M,
}

/// Color manifold for material evaluation at hit point.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct GeometryColor<G, M> {
    geometry: G,
    material: M,
}

impl<G, M> Manifold<Jet3_4> for GeometryColor<G, M>
where
    G: ManifoldCompat<Jet3, Output = Jet3>,
    M: ManifoldCompat<Jet3, Output = Discrete>,
{
    type Output = Discrete;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Discrete {
        let (rx, ry, rz, w) = p;
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        // Compute hit point: P = ray * t
        let hx = rx * t;
        let hy = ry * t;
        let hz = rz * t;

        // Evaluate material at hit point
        self.material.eval_raw(hx, hy, hz, w)
    }
}

/// Mask manifold for geometry hit detection.
kernel!(pub struct GeometryMask = |geometry: kernel| Jet3 -> Field {
    let t = geometry;
    let t_max = 1000000.0;
    let deriv_max = 10000.0;

    // Valid if: t > 0, t < max, derivatives reasonable
    let valid_t = (V(t) > 0.0) & (V(t) < t_max);
    let deriv_mag_sq = DX(t) * DX(t) + DY(t) * DY(t) + DZ(t) * DZ(t);
    let valid_deriv = deriv_mag_sq < (deriv_max * deriv_max);

    valid_t & valid_deriv
});

impl<G, M> Scene for SceneObject<G, M>
where
    G: ManifoldCompat<Jet3, Output = Jet3> + ManifoldExpr + Clone + Copy,
    M: ManifoldCompat<Jet3, Output = Discrete> + ManifoldExpr + Clone + Copy,
{
    type Mask = GeometryMask<G>;
    type Color = GeometryColor<G, M>;

    fn mask(&self) -> Self::Mask {
        GeometryMask {
            geometry: self.geometry,
        }
    }

    fn color(&self) -> Self::Color {
        GeometryColor {
            geometry: self.geometry,
            material: self.material,
        }
    }
}

// ============================================================================
// LAYER 3: Materials
// ============================================================================

/// Reflect: The Crown Jewel.
/// Reconstructs surface normal from the Tangent Frame implied by the Jet derivatives.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct Reflect<M> {
    pub inner: M,
}

impl<M: ManifoldCompat<Jet3, Output = Field>> Manifold<Jet3_4> for Reflect<M> {
    type Output = Field;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Field {
        let (x, y, z, w) = p;
        // The input (x, y, z) is the hit point P with derivatives dP/dscreen.
        // We need to compute the reflected direction R with derivatives dR/dscreen.
        //
        // For a sphere, the normal N = normalize(P - center).
        // Since center is constant, N = normalize(P) (for unit sphere at origin style).
        // Actually for our warp, P = t * ray_dir, and we want N pointing outward.
        //
        // Key insight: The normal IS the normalized hit point direction for a sphere
        // centered at origin. For SphereAt, we'd need (P - center), but our P already
        // encodes the surface position.
        //
        // For a general surface, N comes from the tangent cross product.
        // But the tangent vectors Tu, Tv ARE the derivatives dP/dx, dP/dy.
        // So N = normalize(Tu × Tv) where Tu = (x.dx, y.dx, z.dx), Tv = (x.dy, y.dy, z.dy).

        let p_len_sq = x * x + y * y + z * z;
        let p_len = p_len_sq.sqrt();
        let one = Jet3::constant(Field::from(1.0));
        let inv_p_len = one / p_len;

        // Extract tangent vectors (as scalars)
        let tu = (x.dx, y.dx, z.dx);
        let tv = (x.dy, y.dy, z.dy);

        // Cross product Tv × Tu for outward normal
        // Build expression tree, evaluate at Jet3 struct construction boundaries
        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Build the normalized normal as a single expression tree
        // All operations stay as AST nodes until final evaluation
        let fzero = Field::from(0.0);
        let n_len_sq = cross_x.clone() * cross_x.clone()
            + cross_y.clone() * cross_y.clone()
            + cross_z.clone() * cross_z.clone();
        let inv_n_len = n_len_sq.max(Field::from(1e-10)).sqrt().rsqrt();

        // Normal components - evaluate at Jet3 construction boundary
        let nx = (cross_x * inv_n_len.clone()).constant();
        let ny = (cross_y * inv_n_len.clone()).constant();
        let nz = (cross_z * inv_n_len).constant();

        // Incident direction (normalized hit point direction from origin)
        let d_x = (x.val * inv_p_len.val).constant();
        let d_y = (y.val * inv_p_len.val).constant();
        let d_z = (z.val * inv_p_len.val).constant();

        // D·N (cosine of incidence angle) for curvature scaling
        let d_dot_n_scalar = (d_x * nx + d_y * ny + d_z * nz).constant();

        // Curvature-aware scaling: reflection magnifies angular spread
        // Scale = 2 / |cos(incidence)|, clamped to avoid infinity
        let cos_incidence = d_dot_n_scalar.abs().max(Field::from(0.1));
        let curvature_scale = (Field::from(2.0) / cos_incidence).constant();

        let n_jet_x = Jet3 {
            val: nx,
            dx: (x.dx * curvature_scale).constant(),
            dy: (x.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: (y.dx * curvature_scale).constant(),
            dy: (y.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: (z.dx * curvature_scale).constant(),
            dy: (z.dy * curvature_scale).constant(),
            dz: fzero,
        };

        // D as Jets (normalized P)
        let d_jet_x = x * inv_p_len;
        let d_jet_y = y * inv_p_len;
        let d_jet_z = z * inv_p_len;

        // Householder Reflection: R = D - 2(D·N)N
        let d_dot_n = d_jet_x * n_jet_x + d_jet_y * n_jet_y + d_jet_z * n_jet_z;
        let two = Jet3::constant(Field::from(2.0));
        let k = two * d_dot_n;

        let r_x = d_jet_x - k * n_jet_x;
        let r_y = d_jet_y - k * n_jet_y;
        let r_z = d_jet_z - k * n_jet_z;

        // Recurse with curved reflected rays
        self.inner.eval(r_x, r_y, r_z, w)
    }
}

/// Color Reflect: Householder reflection, wraps Discrete material.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct ColorReflect<M> {
    pub inner: M,
}

impl<M: ManifoldCompat<Jet3, Output = Discrete>> Manifold<Jet3_4> for ColorReflect<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval(&self, p: Jet3_4) -> Discrete {
        let (x, y, z, w) = p;
        let p_len_sq = x * x + y * y + z * z;
        let p_len = p_len_sq.sqrt();
        let one = Jet3::constant(Field::from(1.0));
        let inv_p_len = one / p_len;

        let tu = (x.dx, y.dx, z.dx);
        let tv = (x.dy, y.dy, z.dy);

        // Cross product Tv × Tu for outward normal - build expression tree
        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Build normalized normal as expression tree, evaluate at boundaries
        let fzero = Field::from(0.0);
        let n_len_sq = cross_x.clone() * cross_x.clone()
            + cross_y.clone() * cross_y.clone()
            + cross_z.clone() * cross_z.clone();
        let inv_n_len = n_len_sq.max(Field::from(1e-10)).sqrt().rsqrt();

        // Normal components - evaluate at Jet3 construction boundary
        let nx = (cross_x * inv_n_len.clone()).constant();
        let ny = (cross_y * inv_n_len.clone()).constant();
        let nz = (cross_z * inv_n_len).constant();

        // Incident direction (normalized hit point direction from origin)
        let d_x = (x.val * inv_p_len.val).constant();
        let d_y = (y.val * inv_p_len.val).constant();
        let d_z = (z.val * inv_p_len.val).constant();

        // D·N (cosine of incidence angle) for curvature scaling
        let d_dot_n_scalar = (d_x * nx + d_y * ny + d_z * nz).constant();

        // Curvature-aware scaling: reflection magnifies angular spread
        // Scale = 2 / |cos(incidence)|, clamped to avoid infinity
        let cos_incidence = d_dot_n_scalar.abs().max(Field::from(0.1));
        let curvature_scale = (Field::from(2.0) / cos_incidence).constant();

        let n_jet_x = Jet3 {
            val: nx,
            dx: (x.dx * curvature_scale).constant(),
            dy: (x.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: (y.dx * curvature_scale).constant(),
            dy: (y.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: (z.dx * curvature_scale).constant(),
            dy: (z.dy * curvature_scale).constant(),
            dz: fzero,
        };

        let d_jet_x = x * inv_p_len;
        let d_jet_y = y * inv_p_len;
        let d_jet_z = z * inv_p_len;

        let d_dot_n = d_jet_x * n_jet_x + d_jet_y * n_jet_y + d_jet_z * n_jet_z;
        let two = Jet3::constant(Field::from(2.0));
        let k = two * d_dot_n;

        let r_x = d_jet_x - k * n_jet_x;
        let r_y = d_jet_y - k * n_jet_y;
        let r_z = d_jet_z - k * n_jet_z;

        self.inner.eval_raw(r_x, r_y, r_z, w)
    }
}

// ============================================================================
// PATHJET-BASED REFLECTION: Generic reflection for any surface
// ============================================================================

use pixelflow_core::jet::PathJet;

/// Generalized Reflect for PathJet coordinates.
///
/// This implementation works with ANY surface - it extracts the normal from
/// the hit point's tangent frame (derivatives) and reflects the ray direction.
///
/// ## Input Contract
///
/// The PathJet coordinates encode:
/// - `val`: The hit point P on a surface (Jet3 with screen derivatives)
/// - `dir`: The incoming ray direction D (Jet3)
///
/// The hit point's derivatives (`val.dx`, `val.dy`) form the tangent frame,
/// from which we compute the surface normal via cross product.
///
/// ## Output
///
/// Creates a reflected ray:
/// - `val`: Same hit point (new ray origin)
/// - `dir`: Reflected direction R = D - 2(D·N)N
impl<M> Manifold<PathJet4> for Reflect<M>
where
    M: ManifoldCompat<PathJet<Jet3>, Output = Field>,
{
    type Output = Field;

    #[inline]
    fn eval(&self, p: PathJet4) -> Field {
        let (x, y, z, w) = p;
        // ====================================================================
        // 1. Extract surface normal from hit point's tangent frame
        // ====================================================================

        // Hit point P = (x.val, y.val, z.val) with screen derivatives
        // Tangent vectors: Tu = dP/dscreen_x, Tv = dP/dscreen_y
        let tu = (x.val.dx, y.val.dx, z.val.dx);
        let tv = (x.val.dy, y.val.dy, z.val.dy);

        // Normal = Tv × Tu (cross product, ordering for outward normal)
        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Normalize - build expression tree, clone for reuse
        let n_len_sq = cross_x.clone() * cross_x.clone()
            + cross_y.clone() * cross_y.clone()
            + cross_z.clone() * cross_z.clone();
        let inv_n_len = n_len_sq.max(Field::from(1e-10)).sqrt().rsqrt();

        let nx = (cross_x * inv_n_len.clone()).constant();
        let ny = (cross_y * inv_n_len.clone()).constant();
        let nz = (cross_z * inv_n_len).constant();

        // ====================================================================
        // 2. Get incoming direction D (from PathJet.dir)
        // ====================================================================

        // Normalize the incoming direction
        let d_len_sq = x.dir * x.dir + y.dir * y.dir + z.dir * z.dir;
        let inv_d_len = Jet3::constant(Field::from(1.0)) / d_len_sq.sqrt();

        let dx = x.dir * inv_d_len;
        let dy = y.dir * inv_d_len;
        let dz = z.dir * inv_d_len;

        // ====================================================================
        // 3. Curvature-aware derivative scaling for AA
        // ====================================================================

        // Compute D·N for curvature scaling
        let d_dot_n_scalar = (dx.val * nx + dy.val * ny + dz.val * nz).constant();
        let cos_incidence = d_dot_n_scalar.abs().max(Field::from(0.1));
        let curvature_scale = (Field::from(2.0) / cos_incidence).constant();

        // Normal as Jet3 with scaled derivatives
        let fzero = Field::from(0.0);
        let n_jet_x = Jet3 {
            val: nx,
            dx: (x.val.dx * curvature_scale).constant(),
            dy: (x.val.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: (y.val.dx * curvature_scale).constant(),
            dy: (y.val.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: (z.val.dx * curvature_scale).constant(),
            dy: (z.val.dy * curvature_scale).constant(),
            dz: fzero,
        };

        // ====================================================================
        // 4. Householder reflection: R = D - 2(D·N)N
        // ====================================================================

        let d_dot_n = dx * n_jet_x + dy * n_jet_y + dz * n_jet_z;
        let two = Jet3::constant(Field::from(2.0));
        let k = two * d_dot_n;

        let r_x = dx - k * n_jet_x;
        let r_y = dy - k * n_jet_y;
        let r_z = dz - k * n_jet_z;

        // ====================================================================
        // 5. Create reflected ray and recurse
        // ====================================================================

        // New ray: origin = hit point, direction = reflected
        let reflected_x = PathJet {
            val: x.val,
            dir: r_x,
        };
        let reflected_y = PathJet {
            val: y.val,
            dir: r_y,
        };
        let reflected_z = PathJet {
            val: z.val,
            dir: r_z,
        };
        let reflected_w = PathJet {
            val: w.val,
            dir: w.dir,
        };

        self.inner
            .eval_raw(reflected_x, reflected_y, reflected_z, reflected_w)
    }
}

/// Generalized ColorReflect for PathJet coordinates.
///
/// Same as Reflect but outputs Discrete (packed RGBA) for color pipelines.
impl<M> Manifold<PathJet4> for ColorReflect<M>
where
    M: ManifoldCompat<PathJet<Jet3>, Output = Discrete>,
{
    type Output = Discrete;

    #[inline]
    fn eval(&self, p: PathJet4) -> Discrete {
        let (x, y, z, w) = p;
        // Same algorithm as Reflect<M> for PathJet<Jet3>

        // 1. Extract normal from tangent frame
        let tu = (x.val.dx, y.val.dx, z.val.dx);
        let tv = (x.val.dy, y.val.dy, z.val.dy);

        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Build normalized normal as expression tree, clone for reuse
        let n_len_sq = cross_x.clone() * cross_x.clone()
            + cross_y.clone() * cross_y.clone()
            + cross_z.clone() * cross_z.clone();
        let inv_n_len = n_len_sq.max(Field::from(1e-10)).sqrt().rsqrt();

        let nx = (cross_x * inv_n_len.clone()).constant();
        let ny = (cross_y * inv_n_len.clone()).constant();
        let nz = (cross_z * inv_n_len).constant();

        // 2. Normalize incoming direction
        let d_len_sq = x.dir * x.dir + y.dir * y.dir + z.dir * z.dir;
        let inv_d_len = Jet3::constant(Field::from(1.0)) / d_len_sq.sqrt();

        let dx = x.dir * inv_d_len;
        let dy = y.dir * inv_d_len;
        let dz = z.dir * inv_d_len;

        // 3. Curvature scaling
        let d_dot_n_scalar = (dx.val * nx + dy.val * ny + dz.val * nz).constant();
        let cos_incidence = d_dot_n_scalar.abs().max(Field::from(0.1));
        let curvature_scale = (Field::from(2.0) / cos_incidence).constant();

        let fzero = Field::from(0.0);
        let n_jet_x = Jet3 {
            val: nx,
            dx: (x.val.dx * curvature_scale).constant(),
            dy: (x.val.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: (y.val.dx * curvature_scale).constant(),
            dy: (y.val.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: (z.val.dx * curvature_scale).constant(),
            dy: (z.val.dy * curvature_scale).constant(),
            dz: fzero,
        };

        // 4. Householder reflection
        let d_dot_n = dx * n_jet_x + dy * n_jet_y + dz * n_jet_z;
        let two = Jet3::constant(Field::from(2.0));
        let k = two * d_dot_n;

        let r_x = dx - k * n_jet_x;
        let r_y = dy - k * n_jet_y;
        let r_z = dz - k * n_jet_z;

        // 5. Reflected ray
        let reflected_x = PathJet {
            val: x.val,
            dir: r_x,
        };
        let reflected_y = PathJet {
            val: y.val,
            dir: r_y,
        };
        let reflected_z = PathJet {
            val: z.val,
            dir: r_z,
        };
        let reflected_w = PathJet {
            val: w.val,
            dir: w.dir,
        };

        self.inner
            .eval_raw(reflected_x, reflected_y, reflected_z, reflected_w)
    }
}

/// Checkerboard pattern based on X/Z coordinates.
/// Uses Jet3 derivatives for automatic antialiasing at edges.
kernel!(pub struct Checker = || Jet3 -> Field {
    // Which checker cell are we in?
    let cell_x = V(X).floor();
    let cell_z = V(Z).floor();
    let sum = cell_x + cell_z;
    let half = sum * 0.5;
    let fract_half = half - half.floor();
    let is_even = fract_half.abs() < 0.25;

    // Colors
    let color_a = 0.9;
    let color_b = 0.2;
    let base_color = is_even.select(color_a, color_b);

    // AA: distance to nearest grid line in X and Z
    let fx = V(X) - cell_x;
    let fz = V(Z) - cell_z;

    // Distance to nearest edge (0.0 or 1.0 boundary)
    let dx_edge = (fx - 0.5).abs();
    let dz_edge = (fz - 0.5).abs();
    let dist_to_edge = (0.5 - dx_edge).min(0.5 - dz_edge);

    // Gradient magnitude from Jet3 derivatives
    let grad_x = (DX(X) * DX(X) + DY(X) * DY(X) + DZ(X) * DZ(X)).sqrt();
    let grad_z = (DX(Z) * DX(Z) + DY(Z) * DY(Z) + DZ(Z) * DZ(Z)).sqrt();
    let pixel_size = grad_x.max(grad_z) + 0.001;

    // Coverage: how much of the pixel is in this cell vs neighbor
    let coverage = (dist_to_edge / pixel_size).min(1.0).max(0.0);

    // Blend with neighbor color at edges
    let neighbor_color = is_even.select(color_b, color_a);
    base_color * coverage + neighbor_color * (1.0 - coverage)
});

/// Simple Sky Gradient based on Y direction.
///
/// Uses Lift to project Jet3 → Field (discards derivatives - sky doesn't need AA).
pub fn sky() -> Lift<impl Manifold<Field4, Output = Field> + Clone> {
    Lift(kernel!(|| {
        let t = (Y * 0.5 + 0.5).max(0.0).min(1.0);
        t * 0.8 + 0.1
    })())
}

/// Color Sky: Blue gradient, takes a color cube for platform-specific byte order.
///
/// Manual struct implementation because the color_cube parameter needs
/// `Manifold<Field4>` (Field coordinates from V() extraction), not the kernel's
/// Jet3_4 domain.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct ColorSky<C> {
    pub color_cube: C,
}

impl<C> ColorSky<C> {
    pub fn new(color_cube: C) -> Self {
        Self { color_cube }
    }
}

impl<C: Default> Default for ColorSky<C> {
    fn default() -> Self {
        Self::new(C::default())
    }
}

impl<C: ManifoldCompat<Field, Output = Discrete>> Manifold<Jet3_4> for ColorSky<C> {
    type Output = Discrete;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Discrete {
        let (_x, y, _z, _w) = p;

        // t = (V(Y) * 0.5 + 0.5).max(0.0).min(1.0)
        let y_val: Field = y.into();
        let half = Field::from(0.5);
        let zero = Field::from(0.0);
        let one = Field::from(1.0);
        let t = (y_val * half + half).max(zero).min(one).constant();

        // Gradient colors - collapse to Field with .constant()
        let r = (Field::from(0.7) - t * Field::from(0.5)).constant();
        let g = (Field::from(0.85) - t * Field::from(0.45)).constant();
        let b = (one - t * Field::from(0.2)).constant();

        self.color_cube.eval_raw(r, g, b, one)
    }
}

/// Color Checker: Warm/cool checker with AA, takes a color cube for platform-specific byte order.
///
/// Manual struct implementation because the color_cube parameter needs
/// `Manifold<Field4>` (Field coordinates), not the kernel's Jet3_4 domain.
#[derive(Clone, Copy, ManifoldExpr)]
pub struct ColorChecker<C> {
    pub color_cube: C,
}

impl<C> ColorChecker<C> {
    pub fn new(color_cube: C) -> Self {
        Self { color_cube }
    }
}

impl<C: Default> Default for ColorChecker<C> {
    fn default() -> Self {
        Self::new(C::default())
    }
}

impl<C: ManifoldCompat<Field, Output = Discrete>> Manifold<Jet3_4> for ColorChecker<C> {
    type Output = Discrete;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Discrete {
        let (x, _y, z, _w) = p;

        // Extract Field values from Jet3
        let x_val: Field = x.val;
        let z_val: Field = z.val;

        // Which checker cell are we in?
        let cell_x = x_val.clone().floor();
        let cell_z = z_val.clone().floor();
        let sum = (cell_x.clone() + cell_z.clone()).constant();
        let half = Field::from(0.5);
        let sum_half = (sum.clone() * half.clone()).constant();
        let fract_half = (sum_half.clone() - sum_half.floor()).constant();
        let is_even = fract_half.abs().lt(Field::from(0.25));

        // Colors (warm and cool)
        let ra = Field::from(0.95);
        let ga = Field::from(0.9);
        let ba = Field::from(0.8);
        let rb = Field::from(0.2);
        let gb = Field::from(0.25);
        let bb = Field::from(0.3);

        // AA: distance to nearest grid line in X and Z
        let fx = (x_val - cell_x).constant();
        let fz = (z_val - cell_z).constant();

        // Distance to nearest edge (0.0 or 1.0 boundary)
        let dx_edge = (fx - half.clone()).abs().constant();
        let dz_edge = (fz - half.clone()).abs().constant();
        let dist_to_edge = (half.clone() - dx_edge).min(half - dz_edge).constant();

        // Gradient magnitude from Jet3 derivatives
        let grad_x = (x.dx * x.dx + x.dy * x.dy + x.dz * x.dz).sqrt().constant();
        let grad_z = (z.dx * z.dx + z.dy * z.dy + z.dz * z.dz).sqrt().constant();
        let pixel_size = (grad_x.max(grad_z) + Field::from(0.001)).constant();

        // Coverage: how much of the pixel is in this cell vs neighbor
        let zero = Field::from(0.0);
        let one = Field::from(1.0);
        let coverage = (dist_to_edge / pixel_size).min(one.clone()).max(zero).constant();

        // Select and blend colors
        let r_base = is_even.clone().select(ra.clone(), rb.clone());
        let g_base = is_even.clone().select(ga.clone(), gb.clone());
        let b_base = is_even.clone().select(ba.clone(), bb.clone());

        let r_neighbor = is_even.clone().select(rb, ra);
        let g_neighbor = is_even.clone().select(gb, ga);
        let b_neighbor = is_even.select(bb, ba);

        let inv_coverage = (one.clone() - coverage.clone()).constant();
        let r = (r_base * coverage.clone() + r_neighbor * inv_coverage.clone()).constant();
        let g = (g_base * coverage.clone() + g_neighbor * inv_coverage.clone()).constant();
        let b = (b_base * coverage + b_neighbor * inv_coverage).constant();

        // Sample the color cube
        self.color_cube.eval_raw(r, g, b, one)
    }
}
