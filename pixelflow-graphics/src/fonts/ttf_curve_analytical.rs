//! Analytical curve rendering with ray-crossing winding numbers.
//!
//! Computes winding number contributions by counting where each curve segment
//! crosses the horizontal scanline at Y with a leftward ray from X.
//!
//! For lines: direct x-intersection with the horizontal scanline.
//! For quadratics: analytical root-finding (quadratic formula).
//!
//! The winding number is a pure integer ±1 per crossing. Coverage is a
//! hard step (0 or 1), not a smooth ramp. Geometry::eval applies
//! abs().min(1.0) to convert winding to inside/outside coverage.

use pixelflow_core::{Field, Manifold, ManifoldExt, W, X, Y, Z};
use pixelflow_compiler::kernel;

type Field4 = (Field, Field, Field, Field);

// ═══════════════════════════════════════════════════════════════════════════
// Line Segment (Ray-Crossing Winding)
// ═══════════════════════════════════════════════════════════════════════════

// Line segment with precomputed ray-crossing coefficients.
//
// Winding number contribution via horizontal ray intersection:
// 1. Check if Y is within the segment's vertical extent
// 2. Compute x_intersection where the segment crosses y = Y
// 3. Step: 1 if X >= x_intersection (ray crosses to the left of X), else 0
// 4. Multiply by winding direction (±1)
kernel!(
    pub struct AnalyticalLine = |x0: f32, y0: f32, dx_over_dy: f32,
                                  dir: f32, y_min: f32, y_max: f32| {
        // Early rejection: only contributes when Y is in segment's vertical range
        let in_y = (Y >= y_min) & (Y < y_max);

        // X-coordinate where line segment crosses the horizontal scanline at Y
        let x_int = (Y - y0) * dx_over_dy + x0;

        // Step: 1.0 if X >= x_int (crossing is to the left of or at X)
        let crossed = (X >= x_int).select(1.0, 0.0);

        in_y.select(crossed * dir, 0.0)
    }
);

impl AnalyticalLine {
    /// Create from two endpoints. Returns None for horizontal/degenerate lines.
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

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic Bezier (Analytical Root-Finding with Gradient AA)
// ═══════════════════════════════════════════════════════════════════════════

/// Quadratic Bezier with precomputed analytical ray-crossing coefficients.
///
/// Parametric form: P(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
///   x(t) = ax*t^2 + bx*t + cx
///   y(t) = ay*t^2 + by*t + cy
///
/// To find intersections with y = Y, solve: ay*t^2 + by*t + (cy - Y) = 0
/// For each valid root t in [0,1], compute x(t) and gradient-based coverage.
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

impl Manifold<Field4> for AnalyticalQuad {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        if self.is_linear {
            // Degenerate: quadratic is a line. Solve by*t + (cy - Y) = 0
            let k = kernel!(|ax: f32, bx: f32, cx: f32, by: f32, cy: f32| {
                let t = (Y - cy) / by;
                let in_t = t.clone().ge(0.0) & t.clone().le(1.0);

                // x-coordinate at intersection
                let x_int = t.clone() * t.clone() * ax + t.clone() * bx + cx;

                // Step: 1.0 if crossing is to the left of or at X
                let crossed = (X >= x_int).select(1.0, 0.0);

                let dir = if by > 0.0 { -1.0 } else { 1.0 };
                in_t.select(crossed * dir, 0.0)
            });
            return k(self.ax, self.bx, self.cx, self.by, self.cy).eval(p);
        }

        // True quadratic: solve ay*t^2 + by*t + (cy - Y) = 0
        // discriminant = by^2 - 4*ay*(cy - Y) = disc_const + disc_slope*Y
        let k = kernel!(|ax: f32, bx: f32, cx: f32, ay: f32, by: f32,
                         inv_2a: f32, neg_b_2a: f32, disc_const: f32, disc_slope: f32| {
            let disc = Y * disc_slope + disc_const;
            let sqrt_disc = disc.max(0.0).sqrt();

            // Two roots: t = (-by +/- sqrt(disc)) / (2*ay)
            let t_plus = sqrt_disc.clone() * inv_2a.clone() + neg_b_2a.clone();
            let t_minus = sqrt_disc * -inv_2a + neg_b_2a;

            // X-coordinates at intersection points
            let x_plus = t_plus.clone() * t_plus.clone() * ax.clone() + t_plus.clone() * bx.clone() + cx.clone();
            let x_minus = t_minus.clone() * t_minus.clone() * ax + t_minus.clone() * bx + cx;

            // Tangent dy/dt at each root for winding direction
            let dy_plus = t_plus.clone() * (2.0 * ay.clone()) + by.clone();
            let dy_minus = t_minus.clone() * (2.0 * ay) + by;

            // Step: 1.0 if crossing is to the left of or at X
            let crossed_plus = (X >= x_plus).select(1.0, 0.0);
            let crossed_minus = (X >= x_minus).select(1.0, 0.0);

            // Validity: only count roots with t in [0, 1]
            let valid_plus = t_plus.clone().ge(0.0) & t_plus.clone().le(1.0);
            let valid_minus = t_minus.clone().ge(0.0) & t_minus.clone().le(1.0);

            // Winding sign from tangent direction
            let sign_plus = dy_plus.gt(0.0).select(-1.0, 1.0);
            let sign_minus = dy_minus.gt(0.0).select(-1.0, 1.0);

            // Combine: valid roots contribute signed step, masked by discriminant
            let contrib_plus = valid_plus.select(crossed_plus * sign_plus, 0.0);
            let contrib_minus = valid_minus.select(crossed_minus * sign_minus, 0.0);
            disc.ge(0.0).select(contrib_plus + contrib_minus, 0.0)
        });

        k(self.ax, self.bx, self.cx, self.ay, self.by,
          self.inv_2ay, self.neg_b_2a, self.disc_const, self.disc_slope).eval(p)
    }
}
