//! # Jet2: 2D Automatic Differentiation (first derivatives)

use crate::Field;
use crate::Manifold;
use crate::ext;
use crate::numeric::{Computational, Numeric, Selectable};

/// The standard 4D Field domain.
type Field4 = (Field, Field, Field, Field);

/// A 2-jet: value and first derivatives.
///
/// Represents f(x,y) along with ∂f/∂x and ∂f/∂y.
/// When manifolds are evaluated with Jet2 inputs, derivatives
/// propagate automatically via the chain rule.
///
/// **Internal type.** Used for antialiasing via automatic differentiation.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2 {
    /// The function value f(x,y)
    pub val: Field,
    /// Partial derivative ∂f/∂x
    pub dx: Field,
    /// Partial derivative ∂f/∂y
    pub dy: Field,
}

impl Jet2 {
    /// Create a jet seeded for the X variable (∂x/∂x = 1, ∂x/∂y = 0)
    #[inline(always)]
    pub fn x(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(1.0),
            dy: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Y variable (∂y/∂x = 0, ∂y/∂y = 1)
    #[inline(always)]
    pub fn y(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(1.0),
        }
    }

    /// Create a constant jet (no derivatives)
    #[inline(always)]
    pub fn constant(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
        }
    }

    /// Collapse manifold expressions into a Jet2.
    ///
    /// Evaluates each component at origin to get concrete Field values.
    /// Use sparingly - prefer keeping expressions as manifolds.
    #[inline(always)]
    pub fn new<V, Dx, Dy>(val: V, dx: Dx, dy: Dy) -> Self
    where
        V: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dx: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dy: ext::ManifoldExt + Manifold<Field4, Output = Field>,
    {
        Self {
            val: val.constant(),
            dx: dx.constant(),
            dy: dy.constant(),
        }
    }

    /// Raw select without early exit (pub(crate) only).
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
        }
    }

    // ========================================================================
    // Public methods for comparison and math operations
    // ========================================================================

    /// Less than comparison (returns mask jet).
    #[inline(always)]
    pub fn lt(self, rhs: Self) -> Self {
        Self::constant(self.val.lt(rhs.val))
    }

    /// Less than or equal (returns mask jet).
    #[inline(always)]
    pub fn le(self, rhs: Self) -> Self {
        Self::constant(self.val.le(rhs.val))
    }

    /// Greater than comparison (returns mask jet).
    #[inline(always)]
    pub fn gt(self, rhs: Self) -> Self {
        Self::constant(self.val.gt(rhs.val))
    }

    /// Greater than or equal (returns mask jet).
    #[inline(always)]
    pub fn ge(self, rhs: Self) -> Self {
        Self::constant(self.val.ge(rhs.val))
    }

    /// Square root with derivative.
    ///
    /// Returns `Jet2Sqrt` which enables automatic rsqrt fusion when divided.
    /// Example: `a / b.sqrt()` computes `a * rsqrt(b)` (faster than `a / sqrt(b)`).
    #[inline(always)]
    pub fn sqrt(self) -> Jet2Sqrt {
        Jet2Sqrt(self)
    }

    /// Absolute value with derivative.
    #[inline(always)]
    pub fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        let sign = self.val / self.val.abs();
        Self::new(self.val.abs(), self.dx * sign.clone(), self.dy * sign)
    }

    /// Element-wise minimum with derivative.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    /// Element-wise maximum with derivative.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    /// Check if any lane of the value is non-zero.
    #[inline(always)]
    pub fn any(&self) -> bool {
        self.val.any()
    }

    /// Check if all lanes of the value are non-zero.
    #[inline(always)]
    pub fn all(&self) -> bool {
        self.val.all()
    }

    /// Conditional select with early-exit optimization.
    /// Returns if_true where mask is set, if_false elsewhere.
    #[inline(always)]
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
// Jet2Sqrt: Enables rsqrt fusion for Jet2
// ============================================================================

/// Wrapper for sqrt(Jet2) that enables automatic rsqrt fusion.
///
/// When `Jet2 / Jet2Sqrt` is computed, this automatically uses the faster
/// `rsqrt` path: `a / sqrt(b)` becomes `a * rsqrt(b)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2Sqrt(Jet2);

impl Jet2Sqrt {
    /// Evaluate to get the actual sqrt result as Jet2.
    #[inline(always)]
    pub fn eval(self) -> Jet2 {
        let rsqrt_val = self.0.val.rsqrt();
        let sqrt_val = self.0.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);
        Jet2::new(
            sqrt_val,
            self.0.dx * half_rsqrt.clone(),
            self.0.dy * half_rsqrt,
        )
    }
}

impl From<Jet2Sqrt> for Jet2 {
    #[inline(always)]
    fn from(s: Jet2Sqrt) -> Jet2 {
        s.eval()
    }
}

/// Rsqrt fusion: `Jet2 / Jet2Sqrt` computes `a * rsqrt(b)` directly.
impl core::ops::Div<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn div(self, rhs: Jet2Sqrt) -> Jet2 {
        let b = rhs.0;
        let rsqrt_b = b.val.rsqrt();
        let result_val = self.val * rsqrt_b;
        let rsqrt_cubed = rsqrt_b * rsqrt_b * rsqrt_b;
        let half_rsqrt_cubed = rsqrt_cubed * Field::from(0.5);
        Jet2::new(
            result_val,
            self.dx * rsqrt_b - self.val * b.dx * half_rsqrt_cubed.clone(),
            self.dy * rsqrt_b - self.val * b.dy * half_rsqrt_cubed,
        )
    }
}

impl core::ops::Add<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn add(self, rhs: Jet2) -> Jet2 {
        self.eval() + rhs
    }
}

impl core::ops::Sub<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn sub(self, rhs: Jet2) -> Jet2 {
        self.eval() - rhs
    }
}

impl core::ops::Mul<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn mul(self, rhs: Jet2) -> Jet2 {
        self.eval() * rhs
    }
}

impl core::ops::Div<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn div(self, rhs: Jet2) -> Jet2 {
        self.eval() / rhs
    }
}

impl core::ops::Add<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn add(self, rhs: Jet2Sqrt) -> Jet2 {
        self + rhs.eval()
    }
}

impl core::ops::Sub<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn sub(self, rhs: Jet2Sqrt) -> Jet2 {
        self - rhs.eval()
    }
}

impl core::ops::Mul<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn mul(self, rhs: Jet2Sqrt) -> Jet2 {
        self * rhs.eval()
    }
}

// ============================================================================
// Arithmetic via chain rule
// ============================================================================

impl core::ops::Add for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // (f + g)' = f' + g'
        Self::new(self.val + rhs.val, self.dx + rhs.dx, self.dy + rhs.dy)
    }
}

impl core::ops::Sub for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // (f - g)' = f' - g'
        Self::new(self.val - rhs.val, self.dx - rhs.dx, self.dy - rhs.dy)
    }
}

impl core::ops::Mul for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        Self::new(
            self.val * rhs.val,
            self.dx * rhs.val + self.val * rhs.dx,
            self.dy * rhs.val + self.val * rhs.dy,
        )
    }
}

impl core::ops::Div for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        let g_sq = rhs.val * rhs.val;
        let inv_g_sq = Field::from(1.0) / g_sq;
        let scale = rhs.val.clone() * inv_g_sq.clone();
        Self::new(
            self.val / rhs.val,
            self.dx * scale.clone() - self.val * rhs.dx.clone() * inv_g_sq.clone(),
            self.dy * scale - self.val * rhs.dy.clone() * inv_g_sq,
        )
    }
}

impl core::ops::BitAnd for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // Bitwise AND on masks - derivatives are zero (step function)
        Self::constant(self.val & rhs.val)
    }
}

impl core::ops::BitOr for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // Bitwise OR on masks - derivatives are zero (step function)
        Self::constant(self.val | rhs.val)
    }
}

impl core::ops::Not for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
        }
    }
}

// ============================================================================
// Computational trait implementation (Public API)
// ============================================================================

impl Computational for Jet2 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        // Zero derivatives - users wrap with Jet2::x() to seed X-differentiation
        Self::constant(Field::sequential(start))
    }
}

// Jet2 is a coordinate type
impl crate::numeric::Coordinate for Jet2 {}

// ============================================================================
// Selectable trait implementation (Jet2)
// ============================================================================

impl Selectable for Jet2 {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self {
            val: <Field as Selectable>::select_raw(mask, if_true.val, if_false.val),
            dx: <Field as Selectable>::select_raw(mask, if_true.dx, if_false.dx),
            dy: <Field as Selectable>::select_raw(mask, if_true.dy, if_false.dy),
        }
    }
}

// ============================================================================
// Numeric trait implementation (Internal)
// ============================================================================

impl Numeric for Jet2 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Chain rule: (√f)' = f' / (2√f)
        // Use rsqrt (4 cycles) instead of sqrt (20-30 cycles)
        // sqrt(x) = x * rsqrt(x), derivative = rsqrt(x) / 2
        let rsqrt_val = self.val.rsqrt();
        let sqrt_val = self.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);
        Self::new(sqrt_val, self.dx * half_rsqrt.clone(), self.dy * half_rsqrt)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        // Note: derivative undefined at f=0, we use sign
        let sign = self.val / self.val.abs(); // NaN at zero, but close enough
        Self::new(self.val.abs(), self.dx * sign.clone(), self.dy * sign)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // min(f,g)' = f' if f < g, g' otherwise
        // The mask determines which derivative to use
        // This is a true blend - both derivatives already computed
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // max(f,g)' = f' if f > g, g' otherwise
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
        // Comparison only looks at values, derivatives are zero
        // (derivative of a step function is 0 almost everywhere)
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
        // Blend values and derivatives
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
        }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        // Check if any lane of the VALUE is true
        // (derivatives don't matter for control flow)
        self.val.any()
    }

    #[inline(always)]
    fn all(&self) -> bool {
        // Check if all lanes of the VALUE are true
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

    // ========================================================================
    // Trigonometric Operations with Automatic Differentiation
    // ========================================================================

    #[inline(always)]
    fn sin(self) -> Self {
        // Chain rule: (sin f)' = cos(f) * f'
        let sin_val = self.val.sin();
        let cos_deriv = self.val.cos();
        Self::new(sin_val, self.dx * cos_deriv.clone(), self.dy * cos_deriv)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        // Chain rule: (cos f)' = -sin(f) * f'
        let cos_val = self.val.cos();
        let neg_sin = -self.val.sin();
        Self::new(cos_val, self.dx * neg_sin.clone(), self.dy * neg_sin)
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        // atan2(y, x) derivatives:
        // ∂/∂y = x / (x² + y²)
        // ∂/∂x = -y / (x² + y²)
        let r_sq = self.val * self.val + x.val * x.val;
        let inv_r_sq = Field::from(1.0) / r_sq;
        let dy_darg = x.val.clone() * inv_r_sq.clone();
        let dx_darg = (-self.val).clone() * inv_r_sq;
        Self::new(
            self.val.atan2(x.val),
            self.dx * dy_darg.clone() + x.dx * dx_darg.clone(),
            self.dy * dy_darg + x.dy * dx_darg,
        )
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // For f^g: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        use crate::numeric::Numeric as _;
        let val = Numeric::pow(self.val, exp.val);
        let ln_base = self.val.ln();
        let inv_self = Field::from(1.0).raw_div(self.val);
        let coeff = exp.val.raw_mul(inv_self);
        Self::new(
            val,
            val * (exp.dx * ln_base + coeff.clone() * self.dx),
            val * (exp.dy * ln_base + coeff * self.dy),
        )
    }

    #[inline(always)]
    fn exp(self) -> Self {
        // Chain rule: (exp f)' = exp(f) * f'
        let exp_val = self.val.exp();
        Self::new(
            exp_val.clone(),
            self.dx * exp_val.clone(),
            self.dy * exp_val,
        )
    }

    #[inline(always)]
    fn log2(self) -> Self {
        // Chain rule: (log2 f)' = f' / (f * ln(2))
        // log2(e) = 1/ln(2) ≈ 1.4426950408889634
        let log2_e = Field::from(1.4426950408889634);
        let inv_val = Field::from(1.0) / self.val;
        let deriv_coeff = inv_val * log2_e;
        Self::new(
            self.val.log2(),
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff,
        )
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // Chain rule: (2^f)' = f' * 2^f * ln(2)
        // ln(2) ≈ 0.6931471805599453
        let ln_2 = Field::from(0.6931471805599453);
        let exp2_val = self.val.exp2();
        let deriv_coeff = exp2_val * ln_2;
        Self::new(
            exp2_val,
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff,
        )
    }

    #[inline(always)]
    fn floor(self) -> Self {
        // Floor is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // (a * b + c)' where a, b, c are jets
        Self::new(
            self.val.mul_add(b.val, c.val),
            self.dx * b.val + self.val * b.dx + c.dx,
            self.dy * b.val + self.val * b.dy + c.dy,
        )
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // (1/f)' = -f'/f²
        let inv = self.val.recip();
        let neg_inv_sq = Field::from(0.0) - inv.clone() * inv;
        Self::new(inv, self.dx * neg_inv_sq.clone(), self.dy * neg_inv_sq)
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // d/dx[1/√f] = -f' * rsqrt(f)³ / 2
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let scale = Field::from(-0.5) * rsqrt_cubed;
        Self::new(rsqrt_val, self.dx * scale.clone(), self.dy * scale)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        // Chain rule: (ln f)' = f' / f
        let inv_val = Field::from(1.0) / self.val;
        Self::new(self.val.ln(), self.dx * inv_val.clone(), self.dy * inv_val)
    }

    #[inline(always)]
    fn log10(self) -> Self {
        // Chain rule: (log10 f)' = f' / (f * ln(10))
        // 1/ln(10) ≈ 0.4342944819032518
        let log10_e = Field::from(0.4342944819032518);
        let inv_val = Field::from(1.0) / self.val;
        let deriv_coeff = inv_val * log10_e;
        Self::new(
            self.val.log10(),
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff,
        )
    }

    #[inline(always)]
    fn tan(self) -> Self {
        // Chain rule: (tan f)' = f' / cos²(f) = f' * sec²(f)
        let tan_val = self.val.tan();
        let cos_val = self.val.cos();
        let sec_sq = Field::from(1.0) / (cos_val * cos_val);
        Self::new(tan_val, self.dx * sec_sq.clone(), self.dy * sec_sq)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        // Chain rule: (asin f)' = f' / sqrt(1 - f²)
        use crate::numeric::Numeric as _;
        let one = Field::from(1.0);
        let one_minus_sq = one.raw_sub(self.val.raw_mul(self.val));
        let inv_sqrt = one_minus_sq.rsqrt();
        Self::new(
            self.val.asin(),
            self.dx.raw_mul(inv_sqrt),
            self.dy.raw_mul(inv_sqrt),
        )
    }

    #[inline(always)]
    fn acos(self) -> Self {
        // Chain rule: (acos f)' = -f' / sqrt(1 - f²)
        use crate::numeric::Numeric as _;
        let one = Field::from(1.0);
        let one_minus_sq = one.raw_sub(self.val.raw_mul(self.val));
        let neg_inv_sqrt = Field::from(0.0).raw_sub(one_minus_sq.rsqrt());
        Self::new(
            self.val.acos(),
            self.dx.raw_mul(neg_inv_sqrt),
            self.dy.raw_mul(neg_inv_sqrt),
        )
    }

    #[inline(always)]
    fn atan(self) -> Self {
        // Chain rule: (atan f)' = f' / (1 + f²)
        use crate::numeric::Numeric as _;
        let one = Field::from(1.0);
        let one_plus_sq = one.raw_add(self.val.raw_mul(self.val));
        let inv = Field::from(1.0).raw_div(one_plus_sq);
        Self::new(self.val.atan(), self.dx.raw_mul(inv), self.dy.raw_mul(inv))
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        // Ceil is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.ceil())
    }

    #[inline(always)]
    fn round(self) -> Self {
        // Round is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.round())
    }

    #[inline(always)]
    fn fract(self) -> Self {
        // fract(f) = f - floor(f), derivative = f' (since floor derivative = 0)
        Self::new(self.val.fract(), self.dx, self.dy)
    }

    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        // hypot(x, y) = sqrt(x² + y²)
        // d/dx[hypot] = x / hypot, d/dy[hypot] = y / hypot
        let h = self.val.hypot(y.val);
        let inv_h = Field::from(1.0) / h;
        let dx_coeff = self.val * inv_h.clone();
        let dy_coeff = y.val * inv_h;
        Self::new(
            h,
            self.dx * dx_coeff.clone() + y.dx * dy_coeff.clone(),
            self.dy * dx_coeff + y.dy * dy_coeff,
        )
    }

    #[inline(always)]
    fn mul_rsqrt(self, other: Self) -> Self {
        // mul_rsqrt(a, b) = a * rsqrt(b) = a * b^(-1/2)
        // d[a * b^(-1/2)] = da * b^(-1/2) + a * (-1/2) * b^(-3/2) * db
        //                 = rsqrt(b) * da - (a * rsqrt(b) / (2 * b)) * db
        use crate::numeric::Numeric as _;
        let rsqrt_b = other.val.rsqrt();
        let result = self.val.raw_mul(rsqrt_b);
        let half_inv_b = rsqrt_b.raw_mul(other.val.recip()).raw_mul(Field::from(0.5));
        let da_coeff = rsqrt_b;
        let db_coeff = result.raw_mul(half_inv_b);
        Self::new(
            result,
            self.dx.raw_mul(da_coeff).raw_sub(other.dx.raw_mul(db_coeff)),
            self.dy.raw_mul(da_coeff).raw_sub(other.dy.raw_mul(db_coeff)),
        )
    }

    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        // clamp is piecewise: lo if x < lo, hi if x > hi, x otherwise
        // Derivative follows the branch taken
        let mask_low = self.val.lt(lo.val);
        let mask_high = self.val.gt(hi.val);
        let clamped = self.val.clamp(lo.val, hi.val);
        // Use lo's derivative if clamped to lo, hi's if clamped to hi, self's otherwise
        let dx = Field::select_raw(
            mask_low,
            lo.dx,
            Field::select_raw(mask_high, hi.dx, self.dx),
        );
        let dy = Field::select_raw(
            mask_low,
            lo.dy,
            Field::select_raw(mask_high, hi.dy, self.dy),
        );
        Self { val: clamped, dx, dy }
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self {
        // Comparison only looks at values, derivatives are zero
        Self::constant(self.val.eq(rhs.val))
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self {
        // Comparison only looks at values, derivatives are zero
        Self::constant(self.val.ne(rhs.val))
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        // For jets, mask.val is the actual mask
        Self {
            val: self.val.add_masked(val.val, mask.val),
            dx: self.dx.add_masked(val.dx, mask.val),
            dy: self.dy.add_masked(val.dy, mask.val),
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
        Self::new(-self.val, -self.dx, -self.dy)
    }
}

// ============================================================================
// Field/Jet2 conversions
// ============================================================================

/// Explicit lift: Field → Jet2 (constant with zero derivatives)
impl From<Field> for Jet2 {
    #[inline(always)]
    fn from(val: Field) -> Self {
        Self::constant(val)
    }
}

/// Explicit lift: f32 → Jet2 (chains through Field)
impl From<f32> for Jet2 {
    #[inline(always)]
    fn from(val: f32) -> Self {
        Self::constant(Field::from(val))
    }
}

/// Implicit projection: Jet2 → Field (extract value, discard derivatives)
impl From<Jet2> for Field {
    #[inline(always)]
    fn from(jet: Jet2) -> Self {
        jet.val
    }
}

// ============================================================================
// HasDerivatives trait implementation (for derivative accessors)
// ============================================================================

impl crate::ops::derivative::HasDerivatives for Jet2 {
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
