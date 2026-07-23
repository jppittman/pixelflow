//! Analytical curve rendering with ray-crossing winding numbers.
//!
//! Computes winding number contributions by counting where each curve segment
//! crosses the horizontal scanline at Y with a leftward ray from X.
//!
//! For lines: direct x-intersection with the horizontal scanline.
//! For quadratics: analytical root-finding (quadratic formula).
//!
//! Each crossing contributes a **gradient-normalized ramp** instead of a hard
//! step. With `d = X - x_intersection` computed in the evaluation domain, the
//! per-crossing coverage is:
//!
//! ```text
//! coverage = clamp(d / (‖∇d‖ + ε) + 0.5, 0, 1)
//! ```
//!
//! The same kernel body serves both evaluation domains:
//!
//! - **`Field`** (no derivative tracking): `‖∇d‖ = 0`, so `d / ε` saturates the
//!   clamp and the ramp degenerates to the classic hard 0/1 step.
//! - **`Jet2`** (autodiff): `∇d` chains through every enclosing coordinate
//!   transform (affine scale/flip, translation), so the ramp is exactly one
//!   *screen* pixel wide regardless of glyph scale. This is the antialiased
//!   path; see [`crate::render::aa::Antialiased`] for the seeding combinator.
//!
//! `Geometry::eval` applies `abs().min(1.0)` to convert the summed winding
//! contributions to inside/outside coverage.

use pixelflow_compiler::kernel;
use pixelflow_core::jet::Jet2;
use pixelflow_core::{Field, Manifold};

/// Gradient floor for the crossing ramp. In the `Field` domain the gradient is
/// identically zero, so `d / MIN_GRADIENT` saturates the clamp and reproduces
/// the hard step. In the `Jet2` domain real gradients are ~1, so this floor
/// only guards against division by zero at degenerate tangencies.
const MIN_GRADIENT: f32 = 1e-3;

/// Discriminant floor for the quadratic solver. `Jet2::sqrt` is implemented
/// via `rsqrt`, which is infinite at zero (`0 * inf = NaN`). Clamping the
/// discriminant to a tiny positive value keeps values and derivatives finite
/// at the curve's Y-extremum (tangent point); the resulting root perturbation
/// is ~1e-6 font units. The `disc >= 0` gate still rejects non-intersections.
const MIN_DISC: f32 = 1e-12;

// ═══════════════════════════════════════════════════════════════════════════
// Line Segment (Ray-Crossing Winding)
// ═══════════════════════════════════════════════════════════════════════════

/// Line segment with precomputed ray-crossing coefficients.
///
/// Winding number contribution via horizontal ray intersection:
/// 1. Check if Y is within the segment's vertical extent
/// 2. Compute x_intersection where the segment crosses y = Y
/// 3. Gradient-normalized ramp on `d = X - x_intersection`
/// 4. Multiply by winding direction (±1)
#[derive(Clone, Copy, Debug)]
pub struct AnalyticalLine {
    /// X coordinate of the segment start.
    pub x0: f32,
    /// Y coordinate of the segment start.
    pub y0: f32,
    /// Segment slope as dx/dy (the segment is never horizontal).
    pub dx_over_dy: f32,
    /// Winding direction: -1 for upward segments, +1 for downward.
    pub dir: f32,
    /// Lower Y bound of the segment.
    pub y_min: f32,
    /// Upper Y bound of the segment.
    pub y_max: f32,
}

impl AnalyticalLine {
    /// Create from precomputed coefficients.
    // One argument per segment coefficient; grouping them would just mirror
    // the struct itself. Prefer `from_points` for construction from geometry.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[must_use]
    pub fn new(x0: f32, y0: f32, dx_over_dy: f32, dir: f32, y_min: f32, y_max: f32) -> Self {
        Self {
            x0,
            y0,
            dx_over_dy,
            dir,
            y_min,
            y_max,
        }
    }

    /// Create from two endpoints. Returns None for horizontal/degenerate lines.
    #[must_use]
    pub fn from_points([x0, y0]: [f32; 2], [x1, y1]: [f32; 2]) -> Option<Self> {
        let dy = y1 - y0;
        if dy.abs() < 1e-6 {
            return None;
        }
        let dx = x1 - x0;
        Some(Self::new(
            x0,
            y0,
            dx / dy,
            if dy > 0.0 { -1.0 } else { 1.0 },
            y0.min(y1),
            y0.max(y1),
        ))
    }
}

// One kernel body, stamped per evaluation domain, and — via the same body
// macro — lowered domain-free to IR. `V`/`DX`/`DY` project a domain-valued
// subexpression to its `Field` value/derivative components; over `Field` the
// gradient is zero (hard step), over `Jet2` it is the screen-space ramp, and
// under `Lower` it becomes `Dwrt` that the runtime calculus resolves to the
// same ramp. The `kernel!` macro lifts scalar params/literals to one concrete
// domain per expansion, so a named struct cannot be domain-polymorphic; the
// body macro keeps a single source of truth across all three uses.
//
// The coverage body, defined ONCE. Both the per-domain `Manifold` stamps and
// the domain-free `Lower` impl expand this same source at a chosen scalar
// domain `$n`, so there is no twin to drift: eval calls `.eval(p)`, lower
// calls `.lower(..)` on the identical kernel.
macro_rules! analytical_line_kernel {
    ($self:expr, $n:ty) => {
        kernel!(|x0: f32,
                 y0: f32,
                 dx_over_dy: f32,
                 dir: f32,
                 y_min: f32,
                 y_max: f32,
                 min_grad: f32|
         -> $n {
            // Early rejection: only contributes when Y is in the segment's
            // vertical range. Masks carry no derivatives.
            let in_y = (Y >= y_min) & (Y < y_max);

            // Signed crossing distance in the evaluation domain. Over Jet2 its
            // derivatives chain through every enclosing coordinate transform.
            let d = X - ((Y - y0) * dx_over_dy + x0);

            // Gradient-normalized ramp: hard step over Field (zero gradient),
            // ~1 screen pixel wide over Jet2 / after `lower_dwrt`.
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            let coverage = (V(d) / (grad + V(min_grad)) + V(0.5)).max(V(0.0)).min(V(1.0));

            in_y.select(coverage * V(dir), V(0.0))
        })(
            $self.x0,
            $self.y0,
            $self.dx_over_dy,
            $self.dir,
            $self.y_min,
            $self.y_max,
            MIN_GRADIENT,
        )
    };
}

macro_rules! impl_analytical_line_manifold {
    ($n:ty) => {
        impl Manifold<($n, $n, $n, $n)> for AnalyticalLine {
            type Output = Field;

            #[inline(always)]
            fn eval(&self, p: ($n, $n, $n, $n)) -> Field {
                analytical_line_kernel!(self, $n).eval(p)
            }
        }
    };
}

impl pixelflow_core::Lower for AnalyticalLine {
    fn lower(
        &self,
        arena: &mut pixelflow_ir::arena::ExprArena,
        env: &mut pixelflow_ir::LowerEnv,
    ) -> Option<pixelflow_ir::arena::ExprId> {
        pixelflow_core::Lower::lower(&analytical_line_kernel!(self, Field), arena, env)
    }
}

impl_analytical_line_manifold!(Field);
impl_analytical_line_manifold!(Jet2);

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic Bezier (Analytical Root-Finding with Gradient Ramp)
// ═══════════════════════════════════════════════════════════════════════════

/// Quadratic Bezier with precomputed analytical ray-crossing coefficients.
///
/// Parametric form: P(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
///   x(t) = ax*t^2 + bx*t + cx
///   y(t) = ay*t^2 + by*t + cy
///
/// To find intersections with y = Y, solve: ay*t^2 + by*t + (cy - Y) = 0
/// For each valid root t in [0,1], compute x(t) and gradient-normalized
/// crossing coverage (hard step over `Field`, antialiased over `Jet2`).
#[derive(Clone)]
pub struct AnalyticalQuad {
    // Parametric coefficients: x(t) = ax*t^2 + bx*t + cx
    ax: f32,
    bx: f32,
    cx: f32,
    // Parametric coefficients: y(t) = ay*t^2 + by*t + cy
    ay: f32,
    by: f32,
    cy: f32,
    // Precomputed for quadratic formula
    inv_2ay: f32,
    neg_b_2a: f32,
    disc_const: f32, // by^2 - 4*ay*cy
    disc_slope: f32, // 4*ay (discriminant = disc_slope*Y + disc_const)
    // Degenerate quadratic (actually a line)
    is_linear: bool,
}

impl AnalyticalQuad {
    #[inline]
    #[must_use]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2], [x2, y2]: [f32; 2]) -> Self {
        let ay = y0 - 2.0 * y1 + y2;
        let by = 2.0 * (y1 - y0);
        let cy = y0;
        let ax = x0 - 2.0 * x1 + x2;
        let bx = 2.0 * (x1 - x0);
        let cx = x0;

        let is_linear = ay.abs() < 1e-6;

        let inv_2ay = if is_linear { 0.0 } else { 0.5 / ay };
        let neg_b_2a = -by * inv_2ay;
        let disc_const = by * by - 4.0 * ay * cy;
        let disc_slope = 4.0 * ay;

        Self {
            ax,
            bx,
            cx,
            ay,
            by,
            cy,
            inv_2ay,
            neg_b_2a,
            disc_const,
            disc_slope,
            is_linear,
        }
    }
}

// Two coverage bodies (degenerate line / true quadratic), each defined ONCE
// and expanded at a chosen scalar domain by both the `Manifold` stamps and
// the `Lower` impl — no twin to drift, mirroring the line above.
macro_rules! quad_linear_kernel {
    ($self:expr, $n:ty) => {
        kernel!(|ax: f32, bx: f32, cx: f32, by: f32, cy: f32, dir: f32, min_grad: f32| -> $n {
            let t = (Y - cy) / by;
            let in_t = t.clone().ge(0.0) & t.clone().le(1.0);

            // Signed crossing distance at the intersection.
            let d = X - (t.clone() * t.clone() * ax + t * bx + cx);

            // Gradient-normalized ramp (hard over Field).
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            let coverage = (V(d) / (grad + V(min_grad)) + V(0.5)).max(V(0.0)).min(V(1.0));

            in_t.select(coverage * V(dir), V(0.0))
        })(
            $self.ax,
            $self.bx,
            $self.cx,
            $self.by,
            $self.cy,
            if $self.by > 0.0 { -1.0 } else { 1.0 },
            MIN_GRADIENT,
        )
    };
}

macro_rules! quad_curve_kernel {
    ($self:expr, $n:ty) => {
        kernel!(|ax: f32,
                 bx: f32,
                 cx: f32,
                 ay: f32,
                 by: f32,
                 inv_2a: f32,
                 neg_b_2a: f32,
                 disc_const: f32,
                 disc_slope: f32,
                 min_grad: f32,
                 min_disc: f32|
         -> $n {
            let disc = Y * disc_slope + disc_const;
            // max(min_disc) keeps sqrt finite (value AND derivative) at the
            // tangent point; disc >= 0 below still gates validity.
            let sqrt_disc = disc.clone().max(min_disc).sqrt();

            // Two roots: t = (-by +/- sqrt(disc)) / (2*ay)
            let t_plus = sqrt_disc.clone() * inv_2a + neg_b_2a;
            let t_minus = sqrt_disc * -inv_2a + neg_b_2a;

            // Signed crossing distances at the intersection points.
            let d_plus = X - (t_plus.clone() * t_plus.clone() * ax + t_plus.clone() * bx + cx);
            let d_minus = X - (t_minus.clone() * t_minus.clone() * ax + t_minus.clone() * bx + cx);

            // Tangent dy/dt at each root for winding direction.
            let dy_plus = t_plus.clone() * (ay * 2.0) + by;
            let dy_minus = t_minus.clone() * (ay * 2.0) + by;

            // Gradient-normalized ramps (hard over Field).
            let grad_plus =
                (DX(d_plus.clone()) * DX(d_plus.clone()) + DY(d_plus.clone()) * DY(d_plus.clone()))
                    .sqrt();
            let cov_plus = (V(d_plus) / (grad_plus + V(min_grad)) + V(0.5)).max(V(0.0)).min(V(1.0));
            let grad_minus = (DX(d_minus.clone()) * DX(d_minus.clone())
                + DY(d_minus.clone()) * DY(d_minus.clone()))
            .sqrt();
            let cov_minus =
                (V(d_minus) / (grad_minus + V(min_grad)) + V(0.5)).max(V(0.0)).min(V(1.0));

            // Validity: only count roots with t in [0, 1].
            let valid_plus = t_plus.clone().ge(0.0) & t_plus.le(1.0);
            let valid_minus = t_minus.clone().ge(0.0) & t_minus.le(1.0);

            // Winding sign from tangent direction.
            let sign_plus = dy_plus.gt(0.0).select(V(-1.0), V(1.0));
            let sign_minus = dy_minus.gt(0.0).select(V(-1.0), V(1.0));

            // Valid roots contribute signed coverage, masked by the
            // (unclamped) discriminant.
            let contrib_plus = valid_plus.select(cov_plus * sign_plus, V(0.0));
            let contrib_minus = valid_minus.select(cov_minus * sign_minus, V(0.0));
            disc.ge(0.0).select(contrib_plus + contrib_minus, V(0.0))
        })(
            $self.ax,
            $self.bx,
            $self.cx,
            $self.ay,
            $self.by,
            $self.inv_2ay,
            $self.neg_b_2a,
            $self.disc_const,
            $self.disc_slope,
            MIN_GRADIENT,
            MIN_DISC,
        )
    };
}

macro_rules! impl_analytical_quad_manifold {
    ($n:ty) => {
        impl Manifold<($n, $n, $n, $n)> for AnalyticalQuad {
            type Output = Field;

            #[inline(always)]
            fn eval(&self, p: ($n, $n, $n, $n)) -> Field {
                if self.is_linear {
                    quad_linear_kernel!(self, $n).eval(p)
                } else {
                    quad_curve_kernel!(self, $n).eval(p)
                }
            }
        }
    };
}

impl pixelflow_core::Lower for AnalyticalQuad {
    fn lower(
        &self,
        arena: &mut pixelflow_ir::arena::ExprArena,
        env: &mut pixelflow_ir::LowerEnv,
    ) -> Option<pixelflow_ir::arena::ExprId> {
        if self.is_linear {
            pixelflow_core::Lower::lower(&quad_linear_kernel!(self, Field), arena, env)
        } else {
            pixelflow_core::Lower::lower(&quad_curve_kernel!(self, Field), arena, env)
        }
    }
}

impl_analytical_quad_manifold!(Field);
impl_analytical_quad_manifold!(Jet2);
