//! # Fused Derivative Combinators
//!
//! Combinators that evaluate the inner manifold **once** and extract derived quantities.
//!
//! | Combinator | Description | Output |
//! |------------|-------------|--------|
//! | `GradientMag2D(m)` | √(dx² + dy²) | `Field` |
//! | `GradientMag3D(m)` | √(dx² + dy² + dz²) | `Field` |
//! | `Antialias2D(m)` | val / √(dx² + dy²) | `Field` |
//! | `Antialias3D(m)` | val / √(dx² + dy² + dz²) | `Field` |
//! | `Normalized2D(m)` | (dx, dy) / √(dx² + dy²) | `(Field, Field)` |
//! | `Normalized3D(m)` | (dx, dy, dz) / √(dx² + dy² + dz²) | `(Field, Field, Field)` |
//!
//! ## Why Fused?
//!
//! These combinators evaluate the inner manifold exactly once, then compute
//! derived quantities from the Jet components. This avoids redundant evaluation.
//!
//! ## Example
//!
//! ```ignore
//! // Circle SDF with automatic gradient-based antialiasing
//! let sdf = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
//! let aa = Antialias2D(sdf);  // Evaluates sdf ONCE, divides by gradient mag
//! ```

use crate::combinators::binding::{Let, Var, N0, N1, N2};
use crate::ext::ManifoldExt;
use crate::Field;
use crate::Manifold;
use pixelflow_compiler::Element;

// ============================================================================
// Traits for Derivative Access
// ============================================================================

/// Trait for types that have first derivative components (Jet2, Jet3, Jet2H).
///
/// Provides access to the value and first partial derivatives.
pub trait HasDerivatives: Copy {
    /// Extract the value component.
    fn val(&self) -> Field;

    /// Extract ∂f/∂x.
    fn dx(&self) -> Field;

    /// Extract ∂f/∂y.
    fn dy(&self) -> Field;
}

/// Trait for types with Z derivative (Jet3).
pub trait HasDz: HasDerivatives {
    /// Extract ∂f/∂z.
    fn dz(&self) -> Field;
}

/// Trait for types with Hessian (second derivatives) - Jet2H only.
pub trait HasHessian: HasDerivatives {
    /// Extract ∂²f/∂x².
    fn dxx(&self) -> Field;

    /// Extract ∂²f/∂x∂y.
    fn dxy(&self) -> Field;

    /// Extract ∂²f/∂y².
    fn dyy(&self) -> Field;
}

// ============================================================================
// HasDerivatives impl for Field (trivial case)
// ============================================================================
//
// Field is a scalar with no derivative tracking. V(X) for Field domains
// is identity, and DX/DY return zero (no derivative information).

impl HasDerivatives for Field {
    #[inline(always)]
    fn val(&self) -> Field {
        *self
    }

    #[inline(always)]
    fn dx(&self) -> Field {
        Field::from(0.0)
    }

    #[inline(always)]
    fn dy(&self) -> Field {
        Field::from(0.0)
    }
}

// ============================================================================
// Simple Accessor Combinator Structs
// ============================================================================
//
// These extract individual components from Jet-returning manifolds.
// Unlike fused combinators, they do NOT compute - just extract one field.

/// Extract the value component from a Jet-returning manifold.
///
/// # Example
/// ```ignore
/// let sdf = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
/// let value = V(sdf);  // Just the SDF value, no derivatives
/// ```
#[derive(Clone, Debug, Element)]
pub struct ValOf<M>(pub M);

/// Extract ∂f/∂X (the `.dx` component) from a Jet-returning manifold.
///
/// # Example
/// ```ignore
/// let sdf = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
/// let grad_x = DX(sdf);  // ∂sdf/∂X
/// ```
#[derive(Clone, Debug, Element)]
pub struct DxOf<M>(pub M);

/// Extract ∂f/∂Y (the `.dy` component) from a Jet-returning manifold.
#[derive(Clone, Debug, Element)]
pub struct DyOf<M>(pub M);

/// Extract ∂f/∂Z (the `.dz` component) from a Jet3-returning manifold.
///
/// Only works with manifolds that return `Jet3` (3D autodiff).
#[derive(Clone, Debug, Element)]
pub struct DzOf<M>(pub M);

/// Extract ∂²f/∂X² (the `.dxx` component) from a Jet2H-returning manifold.
///
/// Only works with manifolds that return `Jet2H` (2D with Hessian).
#[derive(Clone, Debug, Element)]
pub struct DxxOf<M>(pub M);

/// Extract ∂²f/∂X∂Y (the `.dxy` component) from a Jet2H-returning manifold.
///
/// The mixed partial derivative.
#[derive(Clone, Debug, Element)]
pub struct DxyOf<M>(pub M);

/// Extract ∂²f/∂Y² (the `.dyy` component) from a Jet2H-returning manifold.
#[derive(Clone, Debug, Element)]
pub struct DyyOf<M>(pub M);

// ============================================================================
// Convenience Functions for Accessor Combinators
// ============================================================================
//
// These provide ergonomic syntax: `DX(expr)` instead of `DxOf(expr)`.
// Naming: DX, DY, DZ parallel the coordinate variables X, Y, Z.

/// Extract the value from a Jet-returning manifold.
///
/// # Example
/// ```ignore
/// kernel!(|sdf: kernel| V(sdf) / (DX(sdf) * DX(sdf) + DY(sdf) * DY(sdf)).sqrt())
/// ```
#[allow(non_snake_case)]
#[inline(always)]
pub fn V<M>(m: M) -> ValOf<M> {
    ValOf(m)
}

/// Extract ∂f/∂X from a Jet-returning manifold.
#[allow(non_snake_case)]
#[inline(always)]
pub fn DX<M>(m: M) -> DxOf<M> {
    DxOf(m)
}

/// Extract ∂f/∂Y from a Jet-returning manifold.
#[allow(non_snake_case)]
#[inline(always)]
pub fn DY<M>(m: M) -> DyOf<M> {
    DyOf(m)
}

/// Extract ∂f/∂Z from a Jet3-returning manifold.
#[allow(non_snake_case)]
#[inline(always)]
pub fn DZ<M>(m: M) -> DzOf<M> {
    DzOf(m)
}

/// Extract ∂²f/∂X² from a Jet2H-returning manifold.
#[allow(non_snake_case)]
#[inline(always)]
pub fn DXX<M>(m: M) -> DxxOf<M> {
    DxxOf(m)
}

/// Extract ∂²f/∂X∂Y from a Jet2H-returning manifold.
#[allow(non_snake_case)]
#[inline(always)]
pub fn DXY<M>(m: M) -> DxyOf<M> {
    DxyOf(m)
}

/// Extract ∂²f/∂Y² from a Jet2H-returning manifold.
#[allow(non_snake_case)]
#[inline(always)]
pub fn DYY<M>(m: M) -> DyyOf<M> {
    DyyOf(m)
}

// ============================================================================
// Fused Combinator Structs
// ============================================================================

/// Compute 2D gradient magnitude: √(dx² + dy²)
///
/// Evaluates the inner manifold once and computes the gradient magnitude.
/// Works with Jet2 and Jet2H.
#[derive(Clone, Debug, Element)]
pub struct GradientMag2D<M>(pub M);

/// Compute 3D gradient magnitude: √(dx² + dy² + dz²)
///
/// Evaluates the inner manifold once and computes the 3D gradient magnitude.
/// Only works with Jet3.
#[derive(Clone, Debug, Element)]
pub struct GradientMag3D<M>(pub M);

/// Compute antialiased value: val / √(dx² + dy²)
///
/// Evaluates the inner manifold once, then divides the value by the gradient
/// magnitude. This is the standard SDF antialiasing technique.
/// Works with Jet2 and Jet2H.
#[derive(Clone, Debug, Element)]
pub struct Antialias2D<M>(pub M);

/// Compute antialiased value in 3D: val / √(dx² + dy² + dz²)
///
/// Evaluates the inner manifold once, then divides the value by the 3D gradient
/// magnitude. Only works with Jet3.
#[derive(Clone, Debug, Element)]
pub struct Antialias3D<M>(pub M);

/// Compute normalized 2D gradient: (dx, dy) / √(dx² + dy²)
///
/// Evaluates the inner manifold once and returns the unit gradient vector.
/// Works with Jet2 and Jet2H.
#[derive(Clone, Debug, Element)]
pub struct Normalized2D<M>(pub M);

/// Compute normalized 3D gradient: (dx, dy, dz) / √(dx² + dy² + dz²)
///
/// Evaluates the inner manifold once and returns the unit gradient vector.
/// Only works with Jet3.
#[derive(Clone, Debug, Element)]
pub struct Normalized3D<M>(pub M);

/// Compute 2D curvature from Hessian: κ = (fxx·fy² - 2·fxy·fx·fy + fyy·fx²) / (fx² + fy²)^(3/2)
///
/// Evaluates the inner manifold once and computes the signed curvature.
/// Only works with Jet2H (requires second derivatives).
#[derive(Clone, Debug, Element)]
pub struct Curvature2D<M>(pub M);

// ============================================================================
// Manifold Implementations
// ============================================================================
//
// These use the kernel pattern: build ZST expression tree, bind extracted
// Jet components via Let/Var, then eval. This keeps everything in the
// algebraic framework without needing raw Field operations.

// GradientMag2D: √(dx² + dy²)
// Var layout: N0 = dx, N1 = dy
impl<P, M, J> Manifold<P> for GradientMag2D<M>
where
    P: Copy + Send + Sync,
    J: HasDerivatives,
    M: Manifold<P, Output = J>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        let jet = self.0.eval(p);
        let dx_val = jet.dx();
        let dy_val = jet.dy();

        // ZST expression: √(v0² + v1²)
        let v0 = Var::<N0>::new();
        let v1 = Var::<N1>::new();
        let expr = (v0 * v0 + v1 * v1).sqrt();

        // Bind and eval
        Let::new(dy_val, Let::new(dx_val, expr)).eval(p)
    }
}

// GradientMag3D: √(dx² + dy² + dz²)
// Var layout: N0 = dx, N1 = dy, N2 = dz
impl<P, M, J> Manifold<P> for GradientMag3D<M>
where
    P: Copy + Send + Sync,
    J: HasDz,
    M: Manifold<P, Output = J>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        let jet = self.0.eval(p);
        let dx_val = jet.dx();
        let dy_val = jet.dy();
        let dz_val = jet.dz();

        // ZST expression: √(v0² + v1² + v2²)
        let v0 = Var::<N0>::new();
        let v1 = Var::<N1>::new();
        let v2 = Var::<N2>::new();
        let expr = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();

        // Bind and eval (outermost Let binds first value to deepest Var)
        Let::new(dz_val, Let::new(dy_val, Let::new(dx_val, expr))).eval(p)
    }
}

// Antialias2D: val / √(dx² + dy²)
// Var layout: N0 = val, N1 = dx, N2 = dy
impl<P, M, J> Manifold<P> for Antialias2D<M>
where
    P: Copy + Send + Sync,
    J: HasDerivatives,
    M: Manifold<P, Output = J>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        let jet = self.0.eval(p);
        let val = jet.val();
        let dx_val = jet.dx();
        let dy_val = jet.dy();

        // ZST expression: v0 / √(v1² + v2²)
        let v0 = Var::<N0>::new();
        let v1 = Var::<N1>::new();
        let v2 = Var::<N2>::new();
        let expr = v0 / (v1 * v1 + v2 * v2).sqrt();

        // Bind and eval
        Let::new(dy_val, Let::new(dx_val, Let::new(val, expr))).eval(p)
    }
}

// Antialias3D: val / √(dx² + dy² + dz²)
// Var layout: N0 = val, N1 = dx, N2 = dy, N3 = dz
impl<P, M, J> Manifold<P> for Antialias3D<M>
where
    P: Copy + Send + Sync,
    J: HasDz,
    M: Manifold<P, Output = J>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        use crate::combinators::binding::N3;
        let jet = self.0.eval(p);
        let val = jet.val();
        let dx_val = jet.dx();
        let dy_val = jet.dy();
        let dz_val = jet.dz();

        // ZST expression: v0 / √(v1² + v2² + v3²)
        let v0 = Var::<N0>::new();
        let v1 = Var::<N1>::new();
        let v2 = Var::<N2>::new();
        let v3 = Var::<N3>::new();
        let expr = v0 / (v1 * v1 + v2 * v2 + v3 * v3).sqrt();

        // Bind and eval
        Let::new(dz_val, Let::new(dy_val, Let::new(dx_val, Let::new(val, expr)))).eval(p)
    }
}

// Normalized2D: (dx, dy) / √(dx² + dy²)
// Returns tuple - needs two separate expressions
impl<P, M, J> Manifold<P> for Normalized2D<M>
where
    P: Copy + Send + Sync,
    J: HasDerivatives,
    M: Manifold<P, Output = J>,
{
    type Output = (Field, Field);

    #[inline(always)]
    fn eval(&self, p: P) -> (Field, Field) {
        let jet = self.0.eval(p);
        let dx_val = jet.dx();
        let dy_val = jet.dy();

        // Compute magnitude once
        let v0 = Var::<N0>::new();
        let v1 = Var::<N1>::new();
        let mag_expr = (v0 * v0 + v1 * v1).sqrt();
        let mag = Let::new(dy_val, Let::new(dx_val, mag_expr)).eval(p);

        // Normalize: dx/mag, dy/mag
        let nx_expr = Var::<N0>::new() / Var::<N1>::new();
        let ny_expr = Var::<N0>::new() / Var::<N1>::new();

        let nx = Let::new(mag, Let::new(dx_val, nx_expr)).eval(p);
        let ny = Let::new(mag, Let::new(dy_val, ny_expr)).eval(p);

        (nx, ny)
    }
}

// Normalized3D: (dx, dy, dz) / √(dx² + dy² + dz²)
impl<P, M, J> Manifold<P> for Normalized3D<M>
where
    P: Copy + Send + Sync,
    J: HasDz,
    M: Manifold<P, Output = J>,
{
    type Output = (Field, Field, Field);

    #[inline(always)]
    fn eval(&self, p: P) -> (Field, Field, Field) {
        let jet = self.0.eval(p);
        let dx_val = jet.dx();
        let dy_val = jet.dy();
        let dz_val = jet.dz();

        // Compute magnitude once
        let v0 = Var::<N0>::new();
        let v1 = Var::<N1>::new();
        let v2 = Var::<N2>::new();
        let mag_expr = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
        let mag = Let::new(dz_val, Let::new(dy_val, Let::new(dx_val, mag_expr))).eval(p);

        // Normalize each component
        let norm_expr = Var::<N0>::new() / Var::<N1>::new();

        let nx = Let::new(mag, Let::new(dx_val, norm_expr)).eval(p);
        let ny = Let::new(mag, Let::new(dy_val, norm_expr)).eval(p);
        let nz = Let::new(mag, Let::new(dz_val, norm_expr)).eval(p);

        (nx, ny, nz)
    }
}

// Curvature2D: κ = (fxx·fy² - 2·fxy·fx·fy + fyy·fx²) / (fx² + fy²)^(3/2)
// Var layout: N0=fx, N1=fy, N2=fxx, N3=fxy, N4=fyy
impl<P, M, J> Manifold<P> for Curvature2D<M>
where
    P: Copy + Send + Sync,
    J: HasHessian,
    M: Manifold<P, Output = J>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        use crate::combinators::binding::{N3, N4};
        let jet = self.0.eval(p);
        let fx = jet.dx();
        let fy = jet.dy();
        let fxx = jet.dxx();
        let fxy = jet.dxy();
        let fyy = jet.dyy();

        // ZST expression for curvature
        let v_fx = Var::<N0>::new();
        let v_fy = Var::<N1>::new();
        let v_fxx = Var::<N2>::new();
        let v_fxy = Var::<N3>::new();
        let v_fyy = Var::<N4>::new();

        let fx_sq = v_fx * v_fx;
        let fy_sq = v_fy * v_fy;
        let grad_sq = fx_sq + fy_sq;

        // numerator: fxx·fy² - 2·fxy·fx·fy + fyy·fx²
        let numerator = v_fxx * fy_sq - v_fxy * v_fx * v_fy * 2.0f32 + v_fyy * fx_sq;
        // denominator: (fx² + fy²)^(3/2) = grad_sq * sqrt(grad_sq)
        let denominator = grad_sq * grad_sq.sqrt();
        let expr = numerator / denominator;

        // Bind all values (outermost = deepest Var index)
        Let::new(fyy,
            Let::new(fxy,
                Let::new(fxx,
                    Let::new(fy,
                        Let::new(fx, expr))))).eval(p)
    }
}

// ============================================================================
// Manifold Implementations for Simple Accessor Combinators
// ============================================================================
//
// These use SPECIFIC DOMAIN IMPLS to avoid coherence conflicts with the 0-fill rule.
//
// For Jet domains (Jet2_4, Jet3_4): extract actual derivatives
// For Field4: return 0 (no derivatives available) - the "0-fill rule" from GLSL/HLSL
//
// This allows composed operations like `(DX(sdf) * DX(sdf) + DY(sdf) * DY(sdf)).sqrt()`
// to work with ManifoldExt methods, and CSE will optimize away redundant evaluations.

use crate::jet::{Jet2, Jet3};

type Field4 = (Field, Field, Field, Field);
type Jet2_4 = (Jet2, Jet2, Jet2, Jet2);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

// ============================================================================
// ValOf: Extract .val() from any domain where output has derivatives
// ============================================================================

// Generic impl: works with Jet2_4, Jet3_4, LetExtended<Jet2, _>, etc.
impl<P, M> Manifold<P> for ValOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasDerivatives,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).val()
    }
}


// ============================================================================
// DxOf: Extract .dx() from any domain where output has derivatives
// ============================================================================

// Generic impl: works with Jet2_4, Jet3_4, LetExtended<Jet2, _>, etc.
impl<P, M> Manifold<P> for DxOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasDerivatives,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).dx()
    }
}

// NOTE: No Field4 impl - DX(expr) only makes sense when expr returns a Jet type.

// ============================================================================
// DyOf: Extract .dy() from any domain where output has derivatives
// ============================================================================

// Generic impl: works with Jet2_4, Jet3_4, LetExtended<Jet2, _>, etc.
impl<P, M> Manifold<P> for DyOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasDerivatives,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).dy()
    }
}

// NOTE: No Field4 impl - DY(expr) only makes sense when expr returns a Jet type.

// ============================================================================
// DzOf: Extract .dz() from any domain where output has Z derivative
// ============================================================================

// Generic impl: works with Jet3_4, LetExtended<Jet3, _>, etc.
impl<P, M> Manifold<P> for DzOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasDz,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).dz()
    }
}

// NOTE: No Field4 impl - DZ(expr) only makes sense when expr returns a Jet3 type.

// ============================================================================
// Hessian Accessors: DxxOf, DxyOf, DyyOf
// ============================================================================

// Generic impl for DxxOf: works with any domain where output has Hessian
impl<P, M> Manifold<P> for DxxOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasHessian,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).dxx()
    }
}

// NOTE: No Field4 impl - DXX(expr) only makes sense when expr returns a Jet2H type.

// Generic impl for DxyOf: works with any domain where output has Hessian
impl<P, M> Manifold<P> for DxyOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasHessian,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).dxy()
    }
}

// NOTE: No Field4 impl - DXY(expr) only makes sense when expr returns a Jet2H type.

// Generic impl for DyyOf: works with any domain where output has Hessian
impl<P, M> Manifold<P> for DyyOf<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P>,
    M::Output: HasHessian,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        self.0.eval(p).dyy()
    }
}

// NOTE: No Field4 impl - DYY(expr) only makes sense when expr returns a Jet2H type.
