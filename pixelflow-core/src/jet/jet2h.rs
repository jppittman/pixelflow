//! # Jet2H: 2D Automatic Differentiation with Hessian (second derivatives)

use crate::Field;
use crate::Manifold;
use crate::ext;
use crate::numeric::{Computational, Numeric, Selectable};

/// The standard 4D Field domain.
type Field4 = (Field, Field, Field, Field);

/// A 2-jet with Hessian: value, first derivatives, and second derivatives.
///
/// Represents f(x,y) along with ∂f/∂x, ∂f/∂y (gradient),
/// and ∂²f/∂x², ∂²f/∂x∂y, ∂²f/∂y² (Hessian matrix).
///
/// The Hessian matrix is:
/// ```text
/// H = [dxx  dxy]
///     [dxy  dyy]
/// ```
///
/// When manifolds are evaluated with Jet2H inputs, both first and second
/// derivatives propagate automatically via the chain rule.
///
/// **Internal type.** Used for second-order optimization and curvature analysis.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2H {
    /// The function value f(x,y)
    pub val: Field,
    /// Partial derivative ∂f/∂x
    pub dx: Field,
    /// Partial derivative ∂f/∂y
    pub dy: Field,
    /// Second partial derivative ∂²f/∂x²
    pub dxx: Field,
    /// Mixed partial derivative ∂²f/∂x∂y
    pub dxy: Field,
    /// Second partial derivative ∂²f/∂y²
    pub dyy: Field,
}

impl Jet2H {
    /// Create a jet seeded for the X variable (∂x/∂x = 1, others = 0)
    #[must_use]
    #[inline(always)]
    pub fn x(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(1.0),
            dy: Field::from(0.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Y variable (∂y/∂y = 1, others = 0)
    #[must_use]
    #[inline(always)]
    pub fn y(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(1.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }

    /// Create a constant jet (no derivatives)
    #[must_use]
    #[inline(always)]
    pub fn constant(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }

    /// Collapse manifold expressions into a Jet2H.
    ///
    /// Evaluates each component at origin to get concrete Field values.
    /// Use sparingly - prefer keeping expressions as manifolds.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new<V, Dx, Dy, Dxx, Dxy, Dyy>(
        val: V,
        dx: Dx,
        dy: Dy,
        dxx: Dxx,
        dxy: Dxy,
        dyy: Dyy,
    ) -> Self
    where
        V: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dx: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dy: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dxx: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dxy: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dyy: ext::ManifoldExt + Manifold<Field4, Output = Field>,
    {
        Self {
            val: val.constant(),
            dx: dx.constant(),
            dy: dy.constant(),
            dxx: dxx.constant(),
            dxy: dxy.constant(),
            dyy: dyy.constant(),
        }
    }

    /// Raw select without early exit (pub(crate) only).
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
            dxx: Field::select_raw(mask.val, if_true.dxx, if_false.dxx),
            dxy: Field::select_raw(mask.val, if_true.dxy, if_false.dxy),
            dyy: Field::select_raw(mask.val, if_true.dyy, if_false.dyy),
        }
    }

    // ========================================================================
    // Public methods for comparison and math operations
    // ========================================================================

    /// Less than comparison (returns mask jet).
    #[inline(always)]
    #[must_use]
    pub fn lt(self, rhs: Self) -> Self {
        Self::constant(self.val.lt(rhs.val))
    }

    /// Less than or equal (returns mask jet).
    #[inline(always)]
    #[must_use]
    pub fn le(self, rhs: Self) -> Self {
        Self::constant(self.val.le(rhs.val))
    }

    /// Greater than comparison (returns mask jet).
    #[inline(always)]
    #[must_use]
    pub fn gt(self, rhs: Self) -> Self {
        Self::constant(self.val.gt(rhs.val))
    }

    /// Greater than or equal (returns mask jet).
    #[inline(always)]
    #[must_use]
    pub fn ge(self, rhs: Self) -> Self {
        Self::constant(self.val.ge(rhs.val))
    }

    /// Square root with first and second derivatives.
    ///
    /// Returns `Jet2HSqrt` which enables automatic rsqrt fusion when divided.
    #[inline(always)]
    #[must_use]
    pub fn sqrt(self) -> Jet2HSqrt {
        Jet2HSqrt(self)
    }

    /// Absolute value with derivatives.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        // |f|'' = f'' * sign(f) + (f'/|f|) * (f' * sign(f) - f')'
        // Simplified: |f|'' = f'' * sign(f)  (derivative of sign is 0 away from 0)
        let sign = self.val / self.val.abs();
        Self::new(
            self.val.abs(),
            self.dx * sign.clone(),
            self.dy * sign.clone(),
            self.dxx * sign.clone(),
            self.dxy * sign.clone(),
            self.dyy * sign,
        )
    }

    /// Element-wise minimum with derivatives.
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
        }
    }

    /// Element-wise maximum with derivatives.
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
        }
    }

    /// Check if any lane of the value is non-zero.
    #[inline(always)]
    #[must_use]
    pub fn any(&self) -> bool {
        self.val.any()
    }

    /// Check if all lanes of the value are non-zero.
    #[inline(always)]
    #[must_use]
    pub fn all(&self) -> bool {
        self.val.all()
    }

    /// Conditional select with early-exit optimization.
    /// Returns if_true where mask is set, if_false elsewhere.
    #[inline(always)]
    #[must_use]
    pub fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.all() {
            return if_true;
        }
        if !mask.any() {
            return if_false;
        }
        Self::select_raw(mask, if_true, if_false)
    }
}

// ============================================================================
// Jet2HSqrt: Enables rsqrt fusion for Jet2H
// ============================================================================

/// Wrapper for sqrt(Jet2H) that enables automatic rsqrt fusion.
///
/// When `Jet2H / Jet2HSqrt` is computed, this automatically uses the faster
/// `rsqrt` path: `a / sqrt(b)` becomes `a * rsqrt(b)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2HSqrt(Jet2H);

impl Jet2HSqrt {
    /// Evaluate to get the actual sqrt result as Jet2H.
    #[inline(always)]
    #[must_use]
    pub fn eval(self) -> Jet2H {
        let rsqrt_val = self.0.val.rsqrt();
        let sqrt_val = self.0.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);

        // sqrt(x)' = rsqrt(x) / 2
        let sqrt_dx = self.0.dx * half_rsqrt.clone();
        let sqrt_dy = self.0.dy * half_rsqrt.clone();

        // sqrt(x)'' = ∂/∂x[x' * rsqrt(x) / 2]
        //           = x'' * rsqrt(x) / 2 + x' * (rsqrt(x) / 2)'
        // where (rsqrt(x))' = -x' * rsqrt(x)³ / 2
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let quarter_rsqrt_cubed = rsqrt_cubed * Field::from(0.25);
        let sqrt_dxx =
            self.0.dxx * half_rsqrt.clone() - self.0.dx * self.0.dx * quarter_rsqrt_cubed.clone();
        let sqrt_dxy =
            self.0.dxy * half_rsqrt.clone() - self.0.dx * self.0.dy * quarter_rsqrt_cubed.clone();
        let sqrt_dyy = self.0.dyy * half_rsqrt - self.0.dy * self.0.dy * quarter_rsqrt_cubed;

        Jet2H::new(sqrt_val, sqrt_dx, sqrt_dy, sqrt_dxx, sqrt_dxy, sqrt_dyy)
    }
}

impl From<Jet2HSqrt> for Jet2H {
    #[inline(always)]
    fn from(s: Jet2HSqrt) -> Jet2H {
        s.eval()
    }
}

/// Rsqrt fusion: `Jet2H / Jet2HSqrt` computes `a * rsqrt(b)` directly.
impl core::ops::Div<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn div(self, rhs: Jet2HSqrt) -> Jet2H {
        let b = rhs.0;
        let rsqrt_b = b.val.rsqrt();
        let result_val = self.val * rsqrt_b;

        // d/dx[a * rsqrt(b)] = a' * rsqrt(b) - a * b' * rsqrt(b)³ / 2
        let rsqrt_cubed = rsqrt_b * rsqrt_b * rsqrt_b;
        let half_rsqrt_cubed = rsqrt_cubed.clone() * Field::from(0.5);

        // First derivatives
        let result_dx =
            self.dx * rsqrt_b + self.val * b.dx * half_rsqrt_cubed.clone() * Field::from(-1.0);
        let result_dy =
            self.dy * rsqrt_b + self.val * b.dy * half_rsqrt_cubed.clone() * Field::from(-1.0);

        // Second derivatives: d²/dx²[a * rsqrt(b)]
        // = a'' * rsqrt(b) + 2 * a' * (rsqrt(b))'
        //   - a * b'' * rsqrt(b)³ / 2
        //   - a * b' * d/dx[rsqrt(b)³ / 2]
        // where d/dx[rsqrt(b)³] = 3 * rsqrt(b)² * (rsqrt(b))'
        //                        = -3 * b' * rsqrt(b)⁵ / 2
        let rsqrt_fifth = rsqrt_cubed * rsqrt_b * rsqrt_b;
        let term = rsqrt_fifth * Field::from(0.75); // 3/2 / 2
        let two = Field::from(2.0);

        let result_dxx = self.dxx * rsqrt_b
            + two * self.dx * b.dx * term.clone() * Field::from(-1.0)
            + self.val * b.dxx * half_rsqrt_cubed.clone() * Field::from(-1.0)
            + self.val * b.dx * b.dx * term.clone();

        let result_dxy = self.dxy * rsqrt_b
            + self.dx * b.dy * term.clone() * Field::from(-1.0)
            + self.dy * b.dx * term.clone() * Field::from(-1.0)
            + self.val * b.dxy * half_rsqrt_cubed.clone() * Field::from(-1.0)
            + self.val * b.dx * b.dy * term.clone();

        let result_dyy = self.dyy * rsqrt_b
            + two * self.dy * b.dy * term.clone() * Field::from(-1.0)
            + self.val * b.dyy * half_rsqrt_cubed * Field::from(-1.0)
            + self.val * b.dy * b.dy * term;

        Jet2H::new(
            result_val, result_dx, result_dy, result_dxx, result_dxy, result_dyy,
        )
    }
}

impl core::ops::Add<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn add(self, rhs: Jet2H) -> Jet2H {
        self.eval() + rhs
    }
}

impl core::ops::Sub<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn sub(self, rhs: Jet2H) -> Jet2H {
        self.eval() - rhs
    }
}

impl core::ops::Mul<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn mul(self, rhs: Jet2H) -> Jet2H {
        self.eval() * rhs
    }
}

impl core::ops::Div<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn div(self, rhs: Jet2H) -> Jet2H {
        self.eval() / rhs
    }
}

impl core::ops::Add<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn add(self, rhs: Jet2HSqrt) -> Jet2H {
        self + rhs.eval()
    }
}

impl core::ops::Sub<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn sub(self, rhs: Jet2HSqrt) -> Jet2H {
        self - rhs.eval()
    }
}

impl core::ops::Mul<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn mul(self, rhs: Jet2HSqrt) -> Jet2H {
        self * rhs.eval()
    }
}

// ============================================================================
// Arithmetic via chain rule (Jet2H)
// ============================================================================

impl core::ops::Add for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // (f + g)'' = f'' + g''
        Self::new(
            self.val + rhs.val,
            self.dx + rhs.dx,
            self.dy + rhs.dy,
            self.dxx + rhs.dxx,
            self.dxy + rhs.dxy,
            self.dyy + rhs.dyy,
        )
    }
}

impl core::ops::Sub for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // (f - g)'' = f'' - g''
        Self::new(
            self.val - rhs.val,
            self.dx - rhs.dx,
            self.dy - rhs.dy,
            self.dxx - rhs.dxx,
            self.dxy - rhs.dxy,
            self.dyy - rhs.dyy,
        )
    }
}

impl core::ops::Mul for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        // Second derivative: (f * g)'' = f'' * g + 2 * f' * g' + f * g''
        let two = Field::from(2.0);
        Self::new(
            self.val * rhs.val,
            self.dx * rhs.val + self.val * rhs.dx,
            self.dy * rhs.val + self.val * rhs.dy,
            self.dxx * rhs.val + two * self.dx * rhs.dx + self.val * rhs.dxx,
            self.dxy * rhs.val + self.dx * rhs.dy + self.dy * rhs.dx + self.val * rhs.dxy,
            self.dyy * rhs.val + two * self.dy * rhs.dy + self.val * rhs.dyy,
        )
    }
}

impl core::ops::Div for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        // For second derivatives, use (f/g) = f * (1/g):
        // d²/dx²[f * g^(-1)] = f'' * g^(-1) + 2 * f' * (g^(-1))'
        //                      + f * (g^(-1))''
        // where (g^(-1))' = -g' * g^(-2)
        //       (g^(-1))'' = -g'' * g^(-2) + 2 * g'² * g^(-3)

        let inv_g = rhs.val.recip();
        let inv_g_sq = inv_g * inv_g;
        let inv_g_cube = inv_g_sq.clone() * inv_g;
        let two = Field::from(2.0);

        // First derivatives
        let dx = self.dx * inv_g + self.val * rhs.dx * inv_g_sq.clone() * Field::from(-1.0);
        let dy = self.dy * inv_g + self.val * rhs.dy * inv_g_sq.clone() * Field::from(-1.0);

        // Second derivatives
        let dxx = self.dxx * inv_g
            + two * self.dx * rhs.dx * inv_g_sq.clone() * Field::from(-1.0)
            + self.val * rhs.dxx * inv_g_sq.clone() * Field::from(-1.0)
            + two * self.val * rhs.dx * rhs.dx * inv_g_cube.clone();

        let dxy = self.dxy * inv_g
            + self.dx * rhs.dy * inv_g_sq.clone() * Field::from(-1.0)
            + self.dy * rhs.dx * inv_g_sq.clone() * Field::from(-1.0)
            + self.val * rhs.dxy * inv_g_sq.clone() * Field::from(-1.0)
            + two * self.val * rhs.dx * rhs.dy * inv_g_cube.clone();

        let dyy = self.dyy * inv_g
            + two * self.dy * rhs.dy * inv_g_sq.clone() * Field::from(-1.0)
            + self.val * rhs.dyy * inv_g_sq * Field::from(-1.0)
            + two * self.val * rhs.dy * rhs.dy * inv_g_cube;

        Self::new(self.val * inv_g, dx, dy, dxx, dxy, dyy)
    }
}

impl core::ops::BitAnd for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // Bitwise AND on masks - derivatives are zero (step function)
        Self::constant(self.val & rhs.val)
    }
}

impl core::ops::BitOr for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // Bitwise OR on masks - derivatives are zero (step function)
        Self::constant(self.val | rhs.val)
    }
}

impl core::ops::Not for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }
}

// ============================================================================
// Computational trait implementation (Jet2H)
// ============================================================================

impl Computational for Jet2H {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self::constant(Field::sequential(start))
    }
}

// Jet2H is a coordinate type
impl crate::numeric::Coordinate for Jet2H {}

// ============================================================================
// Selectable trait implementation (Jet2H)
// ============================================================================

impl Selectable for Jet2H {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self {
            val: <Field as Selectable>::select_raw(mask, if_true.val, if_false.val),
            dx: <Field as Selectable>::select_raw(mask, if_true.dx, if_false.dx),
            dy: <Field as Selectable>::select_raw(mask, if_true.dy, if_false.dy),
            dxx: <Field as Selectable>::select_raw(mask, if_true.dxx, if_false.dxx),
            dxy: <Field as Selectable>::select_raw(mask, if_true.dxy, if_false.dxy),
            dyy: <Field as Selectable>::select_raw(mask, if_true.dyy, if_false.dyy),
        }
    }
}

// ============================================================================
// Numeric trait implementation (Jet2H)
// ============================================================================

impl Numeric for Jet2H {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Chain rule: (√f)' = f' / (2√f)
        // Using rsqrt: (√f)' = f' * rsqrt(f) / 2
        // Second derivative via product rule
        let rsqrt_val = self.val.rsqrt();
        let sqrt_val = self.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);

        // d/dx[f' * rsqrt(f) / 2] = (f'' * rsqrt(f) + f' * rsqrt(f)') / 2
        // where rsqrt(f)' = -f' * rsqrt(f)³ / 2
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let quarter_rsqrt_cubed = rsqrt_cubed * Field::from(0.25);
        let sqrt_dxx =
            self.dxx * half_rsqrt.clone() - self.dx * self.dx * quarter_rsqrt_cubed.clone();
        let sqrt_dyy =
            self.dyy * half_rsqrt.clone() - self.dy * self.dy * quarter_rsqrt_cubed.clone();
        let sqrt_dxy = self.dxy * half_rsqrt.clone() - self.dx * self.dy * quarter_rsqrt_cubed;

        Self::new(
            sqrt_val,
            self.dx * half_rsqrt.clone(),
            self.dy * half_rsqrt,
            sqrt_dxx,
            sqrt_dxy,
            sqrt_dyy,
        )
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let sign = self.val / self.val.abs();
        Self::new(
            self.val.abs(),
            self.dx * sign.clone(),
            self.dy * sign.clone(),
            self.dxx * sign.clone(),
            self.dxy * sign.clone(),
            self.dyy * sign,
        )
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
        }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
        Self::constant(self.val.lt(rhs.val))
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self {
        Self::constant(self.val.le(rhs.val))
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self {
        Self::constant(self.val.gt(rhs.val))
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self {
        Self::constant(self.val.ge(rhs.val))
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.all() {
            return if_true;
        }
        if !mask.any() {
            return if_false;
        }
        Self::select_raw(mask, if_true, if_false)
    }

    #[inline(always)]
    fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
            dxx: Field::select_raw(mask.val, if_true.dxx, if_false.dxx),
            dxy: Field::select_raw(mask.val, if_true.dxy, if_false.dxy),
            dyy: Field::select_raw(mask.val, if_true.dyy, if_false.dyy),
        }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        self.val.any()
    }

    #[inline(always)]
    fn all(&self) -> bool {
        self.val.all()
    }

    #[inline(always)]
    fn from_i32(val: i32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn from_field(field: Field) -> Self {
        Self::constant(field)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        // sin(f)' = cos(f) * f'
        // sin(f)'' = -sin(f) * (f')² + cos(f) * f''
        let sin_val = self.val.sin();
        let cos_val = self.val.cos();
        let neg_sin_val = -sin_val;

        Self::new(
            sin_val,
            self.dx * cos_val,
            self.dy * cos_val,
            neg_sin_val * self.dx * self.dx + self.dxx * cos_val,
            neg_sin_val * self.dx * self.dy + self.dxy * cos_val,
            neg_sin_val * self.dy * self.dy + self.dyy * cos_val,
        )
    }

    #[inline(always)]
    fn cos(self) -> Self {
        // cos(f)' = -sin(f) * f'
        // cos(f)'' = -cos(f) * (f')² - sin(f) * f''
        let cos_val = self.val.cos();
        let sin_val = self.val.sin();
        let neg_sin_val = sin_val * Field::from(-1.0);
        let neg_cos_val = cos_val * Field::from(-1.0);

        Self::new(
            cos_val,
            self.dx * neg_sin_val.clone(),
            self.dy * neg_sin_val.clone(),
            neg_cos_val.clone() * self.dx * self.dx + self.dxx * neg_sin_val.clone(),
            neg_cos_val.clone() * self.dx * self.dy + self.dxy * neg_sin_val.clone(),
            neg_cos_val * self.dy * self.dy + self.dyy * neg_sin_val,
        )
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        let r_sq = self.val * self.val + x.val * x.val;
        let inv_r_sq = Field::from(1.0) / r_sq;
        let dy_darg = x.val * inv_r_sq.clone();
        let dx_darg = self.val * inv_r_sq.clone() * Field::from(-1.0);
        let inv_r_fourth = inv_r_sq.clone() * inv_r_sq.clone();
        let two = Field::from(2.0);
        let term = two * inv_r_fourth;
        let d_dy_darg_y = self.val * dy_darg.clone() * term.clone() * Field::from(-1.0);
        let d_dy_darg_x = inv_r_sq.clone() + x.val * x.val * term.clone() * Field::from(-1.0);
        let d_dx_darg_y = inv_r_sq * Field::from(-1.0) + self.val * self.val * term.clone();
        let d_dx_darg_x = self.val * dx_darg.clone() * term;
        Self::new(
            self.val.atan2(x.val),
            self.dx * dy_darg.clone() + x.dx * dx_darg.clone(),
            self.dy * dy_darg.clone() + x.dy * dx_darg.clone(),
            self.dxx * dy_darg.clone()
                + self.dx * d_dy_darg_y.clone() * self.dx
                + x.dxx * dx_darg.clone()
                + x.dx * d_dx_darg_x.clone() * x.dx
                + self.dx * x.dx * (d_dy_darg_x.clone() + d_dx_darg_y.clone()),
            self.dxy * dy_darg.clone()
                + self.dx * d_dy_darg_y.clone() * self.dy
                + x.dxy * dx_darg.clone()
                + x.dx * d_dx_darg_x.clone() * x.dy
                + self.dy * x.dx * (d_dy_darg_x.clone() + d_dx_darg_y.clone())
                + self.dx * x.dy * (d_dy_darg_x.clone() + d_dx_darg_y.clone()),
            self.dyy * dy_darg
                + self.dy * d_dy_darg_y * self.dy
                + x.dyy * dx_darg
                + x.dy * d_dx_darg_x * x.dy
                + self.dy * x.dy * (d_dy_darg_x + d_dx_darg_y),
        )
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        let val = self.val.pow(exp.val);
        let ln_base = self.val.ln();
        let inv_self = Field::from(1.0) / self.val;
        let coeff = exp.val * inv_self.clone();
        let two = Field::from(2.0);
        Self::new(
            val,
            val * (exp.dx * ln_base + coeff.clone() * self.dx),
            val * (exp.dy * ln_base + coeff.clone() * self.dy),
            self.dxx * val * coeff.clone()
                + two
                    * self.dx
                    * val
                    * inv_self.clone()
                    * (exp.dx * ln_base + coeff.clone() * self.dx)
                + val * exp.dxx * ln_base
                - self.dx * self.dx * val * inv_self.clone() * inv_self.clone(),
            self.dxy * val * coeff.clone()
                + self.dx * val * inv_self.clone() * (exp.dy * ln_base + coeff.clone() * self.dy)
                + self.dy * val * inv_self.clone() * (exp.dx * ln_base + coeff.clone() * self.dx)
                + val * exp.dxy * ln_base,
            self.dyy * val * coeff.clone()
                + two * self.dy * val * inv_self.clone() * (exp.dy * ln_base + coeff * self.dy)
                + val * exp.dyy * ln_base
                - self.dy * self.dy * val * inv_self.clone() * inv_self,
        )
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let exp_val = self.val.exp();
        Self::new(
            exp_val,
            self.dx * exp_val,
            self.dy * exp_val,
            exp_val * self.dx * self.dx + exp_val * self.dxx,
            exp_val * self.dx * self.dy + exp_val * self.dxy,
            exp_val * self.dy * self.dy + exp_val * self.dyy,
        )
    }

    #[inline(always)]
    fn log2(self) -> Self {
        let log2_e = Field::from(core::f32::consts::LOG2_E);
        let inv_val = Field::from(1.0) / self.val;
        let inv_val_sq = inv_val.clone() * inv_val.clone();
        let deriv_coeff = inv_val.clone() * log2_e;
        Self::new(
            self.val.log2(),
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff,
            log2_e * (self.dxx * inv_val.clone() - self.dx * self.dx * inv_val_sq.clone()),
            log2_e * (self.dxy * inv_val.clone() - self.dx * self.dy * inv_val_sq.clone()),
            log2_e * (self.dyy * inv_val - self.dy * self.dy * inv_val_sq),
        )
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // (2^f)' = f' * 2^f * ln(2)
        // (2^f)'' = 2^f * ln(2) * (f'' + (f')² * ln(2))
        let ln_2 = Field::from(core::f32::consts::LN_2);
        let exp2_val = self.val.exp2();
        let deriv_coeff = exp2_val * ln_2;

        Self::new(
            exp2_val,
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff.clone(),
            deriv_coeff.clone() * (self.dxx + self.dx * self.dx * ln_2),
            deriv_coeff.clone() * (self.dxy + self.dx * self.dy * ln_2),
            deriv_coeff * (self.dyy + self.dy * self.dy * ln_2),
        )
    }

    #[inline(always)]
    fn floor(self) -> Self {
        // Floor is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // (a * b + c)' = a' * b + a * b' + c'
        // (a * b + c)'' = a'' * b + 2 * a' * b' + a * b'' + c''
        let two = Field::from(2.0);
        Self::new(
            self.val.mul_add(b.val, c.val),
            self.dx * b.val + self.val * b.dx + c.dx,
            self.dy * b.val + self.val * b.dy + c.dy,
            self.dxx * b.val + two * self.dx * b.dx + self.val * b.dxx + c.dxx,
            self.dxy * b.val + self.dx * b.dy + self.dy * b.dx + self.val * b.dxy + c.dxy,
            self.dyy * b.val + two * self.dy * b.dy + self.val * b.dyy + c.dyy,
        )
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // (1/f)' = -f'/f²
        // (1/f)'' = -f''/f² + 2*(f')²/f³
        let inv = self.val.recip();
        let inv_sq = inv * inv;
        let inv_cube = inv_sq.clone() * inv;
        let neg_inv_sq = inv_sq * Field::from(-1.0);
        let two = Field::from(2.0);
        Self::new(
            inv,
            self.dx * neg_inv_sq.clone(),
            self.dy * neg_inv_sq.clone(),
            self.dxx * neg_inv_sq.clone() + two * self.dx * self.dx * inv_cube.clone(),
            self.dxy * neg_inv_sq.clone() + two * self.dx * self.dy * inv_cube.clone(),
            self.dyy * neg_inv_sq + two * self.dy * self.dy * inv_cube,
        )
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // rsqrt(f)' = -f' * rsqrt(f)³ / 2
        // rsqrt(f)'' = -f'' * rsqrt(f)³ / 2 - 3/2 * f' * rsqrt(f)⁵ * f'
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let rsqrt_fifth = rsqrt_cubed.clone() * rsqrt_val * rsqrt_val;
        let scale = rsqrt_cubed * Field::from(-0.5);
        let scale_second = rsqrt_fifth * Field::from(-1.5);

        Self::new(
            rsqrt_val,
            self.dx * scale.clone(),
            self.dy * scale.clone(),
            self.dxx * scale.clone() + self.dx * self.dx * scale_second.clone(),
            self.dxy * scale.clone() + self.dx * self.dy * scale_second.clone(),
            self.dyy * scale + self.dy * self.dy * scale_second,
        )
    }

    #[inline(always)]
    fn ln(self) -> Self {
        // ln(f)' = f'/f
        // ln(f)'' = f''/f - f'²/f²
        let inv = Field::from(1.0) / self.val;
        let inv_sq = inv.clone() * inv.clone();
        Self::new(
            self.val.ln(),
            self.dx * inv.clone(),
            self.dy * inv.clone(),
            self.dxx * inv.clone() - self.dx * self.dx * inv_sq.clone(),
            self.dxy * inv.clone() - self.dx * self.dy * inv_sq.clone(),
            self.dyy * inv - self.dy * self.dy * inv_sq,
        )
    }

    #[inline(always)]
    fn log10(self) -> Self {
        // log10(f) = ln(f) / ln(10)
        let log10_e = Field::from(0.4342944819032518);
        let inv = Field::from(1.0) / self.val;
        let inv_sq = inv.clone() * inv.clone();
        let scale = inv * log10_e.clone();
        let scale_sq = inv_sq * log10_e;
        Self::new(
            self.val.log10(),
            self.dx * scale.clone(),
            self.dy * scale.clone(),
            self.dxx * scale.clone() - self.dx * self.dx * scale_sq.clone(),
            self.dxy * scale.clone() - self.dx * self.dy * scale_sq.clone(),
            self.dyy * scale - self.dy * self.dy * scale_sq,
        )
    }

    #[inline(always)]
    fn tan(self) -> Self {
        // tan(f)' = sec²(f) * f'
        // tan(f)'' = 2*sec²(f)*tan(f)*f'² + sec²(f)*f''
        let tan_val = self.val.tan();
        let cos_val = self.val.cos();
        let sec_sq = Field::from(1.0) / (cos_val * cos_val);
        let two_sec_sq_tan = Field::from(2.0) * sec_sq.clone() * tan_val;
        Self::new(
            tan_val,
            self.dx * sec_sq.clone(),
            self.dy * sec_sq.clone(),
            self.dxx * sec_sq.clone() + self.dx * self.dx * two_sec_sq_tan.clone(),
            self.dxy * sec_sq.clone() + self.dx * self.dy * two_sec_sq_tan.clone(),
            self.dyy * sec_sq + self.dy * self.dy * two_sec_sq_tan,
        )
    }

    #[inline(always)]
    fn asin(self) -> Self {
        // asin(f)' = f' / sqrt(1-f²)
        use crate::numeric::Numeric as _;
        let one = Field::from(1.0);
        let one_minus_sq = one.raw_sub(self.val.raw_mul(self.val));
        let inv_sqrt = one_minus_sq.rsqrt();
        Self::new(
            self.val.asin(),
            self.dx.raw_mul(inv_sqrt),
            self.dy.raw_mul(inv_sqrt),
            self.dxx.raw_mul(inv_sqrt),
            self.dxy.raw_mul(inv_sqrt),
            self.dyy.raw_mul(inv_sqrt),
        )
    }

    #[inline(always)]
    fn acos(self) -> Self {
        use crate::numeric::Numeric as _;
        let one = Field::from(1.0);
        let one_minus_sq = one.raw_sub(self.val.raw_mul(self.val));
        let neg_inv_sqrt = Field::from(0.0).raw_sub(one_minus_sq.rsqrt());
        Self::new(
            self.val.acos(),
            self.dx.raw_mul(neg_inv_sqrt),
            self.dy.raw_mul(neg_inv_sqrt),
            self.dxx.raw_mul(neg_inv_sqrt),
            self.dxy.raw_mul(neg_inv_sqrt),
            self.dyy.raw_mul(neg_inv_sqrt),
        )
    }

    #[inline(always)]
    fn atan(self) -> Self {
        // atan(f)' = f' / (1+f²)
        use crate::numeric::Numeric as _;
        let one = Field::from(1.0);
        let one_plus_sq = one.raw_add(self.val.raw_mul(self.val));
        let inv = Field::from(1.0).raw_div(one_plus_sq);
        Self::new(
            self.val.atan(),
            self.dx.raw_mul(inv),
            self.dy.raw_mul(inv),
            self.dxx.raw_mul(inv),
            self.dxy.raw_mul(inv),
            self.dyy.raw_mul(inv),
        )
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Self::constant(self.val.ceil())
    }

    #[inline(always)]
    fn round(self) -> Self {
        Self::constant(self.val.round())
    }

    #[inline(always)]
    fn fract(self) -> Self {
        Self::new(self.val.fract(), self.dx, self.dy, self.dxx, self.dxy, self.dyy)
    }

    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        let h = self.val.hypot(y.val);
        let inv_h = Field::from(1.0) / h;
        let dx_coeff = self.val * inv_h.clone();
        let dy_coeff = y.val * inv_h;
        Self::new(
            h,
            self.dx * dx_coeff.clone() + y.dx * dy_coeff.clone(),
            self.dy * dx_coeff.clone() + y.dy * dy_coeff.clone(),
            self.dxx * dx_coeff.clone() + y.dxx * dy_coeff.clone(),
            self.dxy * dx_coeff.clone() + y.dxy * dy_coeff.clone(),
            self.dyy * dx_coeff + y.dyy * dy_coeff,
        )
    }

    #[inline(always)]
    fn mul_rsqrt(self, other: Self) -> Self {
        // mul_rsqrt(a, b) = a * rsqrt(b)
        // Delegate to existing rsqrt and multiplication for correct second derivatives
        self * other.rsqrt()
    }

    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        let mask_low = self.val.lt(lo.val);
        let mask_high = self.val.gt(hi.val);
        let clamped = self.val.clamp(lo.val, hi.val);
        Self {
            val: clamped,
            dx: Field::select_raw(mask_low, lo.dx, Field::select_raw(mask_high, hi.dx, self.dx)),
            dy: Field::select_raw(mask_low, lo.dy, Field::select_raw(mask_high, hi.dy, self.dy)),
            dxx: Field::select_raw(mask_low, lo.dxx, Field::select_raw(mask_high, hi.dxx, self.dxx)),
            dxy: Field::select_raw(mask_low, lo.dxy, Field::select_raw(mask_high, hi.dxy, self.dxy)),
            dyy: Field::select_raw(mask_low, lo.dyy, Field::select_raw(mask_high, hi.dyy, self.dyy)),
        }
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self {
        Self::constant(self.val.eq(rhs.val))
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self {
        Self::constant(self.val.ne(rhs.val))
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        Self {
            val: self.val.add_masked(val.val, mask.val),
            dx: self.dx.add_masked(val.dx, mask.val),
            dy: self.dy.add_masked(val.dy, mask.val),
            dxx: self.dxx.add_masked(val.dxx, mask.val),
            dxy: self.dxy.add_masked(val.dxy, mask.val),
            dyy: self.dyy.add_masked(val.dyy, mask.val),
        }
    }

    #[inline(always)]
    fn raw_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn raw_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn raw_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn raw_div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn raw_neg(self) -> Self {
        Self::new(-self.val, -self.dx, -self.dy, -self.dxx, -self.dxy, -self.dyy)
    }
}

// ============================================================================
// Field/Jet2H conversions
// ============================================================================

/// Explicit lift: Field → Jet2H (constant with zero derivatives)
impl From<Field> for Jet2H {
    #[inline(always)]
    fn from(val: Field) -> Self {
        Self::constant(val)
    }
}

/// Implicit projection: Jet2H → Field (extract value, discard derivatives)
impl From<Jet2H> for Field {
    #[inline(always)]
    fn from(jet: Jet2H) -> Self {
        jet.val
    }
}

// ============================================================================
// HasDerivatives and HasHessian trait implementations (for derivative accessors)
// ============================================================================

impl crate::ops::derivative::HasDerivatives for Jet2H {
    #[inline(always)]
    fn val(&self) -> Field {
        self.val
    }

    #[inline(always)]
    fn dx(&self) -> Field {
        self.dx
    }

    #[inline(always)]
    fn dy(&self) -> Field {
        self.dy
    }
}

impl crate::ops::derivative::HasHessian for Jet2H {
    #[inline(always)]
    fn dxx(&self) -> Field {
        self.dxx
    }

    #[inline(always)]
    fn dxy(&self) -> Field {
        self.dxy
    }

    #[inline(always)]
    fn dyy(&self) -> Field {
        self.dyy
    }
}
