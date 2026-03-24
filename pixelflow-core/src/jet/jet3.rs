//! # Jet3: 3D Automatic Differentiation (for surface normals)

use crate::Field;
use crate::Manifold;
use crate::ManifoldExt;
use crate::ext;
use crate::numeric::{Computational, Numeric, Selectable};

/// The standard 4D Field domain.
type Field4 = (Field, Field, Field, Field);

/// A 3-jet: value and first derivatives in 3D.
///
/// Represents f(x,y,z) along with ∂f/∂x, ∂f/∂y, and ∂f/∂z.
/// Essential for computing surface normals from SDF gradients.
///
/// **Internal type.** Used for 3D rendering via automatic differentiation.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet3 {
    /// The function value f(x,y,z)
    pub val: Field,
    /// Partial derivative ∂f/∂x
    pub dx: Field,
    /// Partial derivative ∂f/∂y
    pub dy: Field,
    /// Partial derivative ∂f/∂z
    pub dz: Field,
}

impl Jet3 {
    /// Create a jet seeded for the X variable (∂x/∂x = 1, others = 0)
    #[inline(always)]
    pub fn x(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(1.0),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Y variable (∂y/∂y = 1, others = 0)
    #[inline(always)]
    pub fn y(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(1.0),
            dz: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Z variable (∂z/∂z = 1, others = 0)
    #[inline(always)]
    pub fn z(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dz: Field::from(1.0),
        }
    }

    /// Create a constant jet (no derivatives)
    #[inline(always)]
    pub fn constant(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }

    /// Collapse manifold expressions into a Jet3.
    ///
    /// Evaluates each component at origin to get concrete Field values.
    /// Use sparingly - prefer keeping expressions as manifolds.
    #[inline(always)]
    pub fn new<V, Dx, Dy, Dz>(val: V, dx: Dx, dy: Dy, dz: Dz) -> Self
    where
        V: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dx: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dy: ext::ManifoldExt + Manifold<Field4, Output = Field>,
        Dz: ext::ManifoldExt + Manifold<Field4, Output = Field>,
    {
        Self {
            val: val.constant(),
            dx: dx.constant(),
            dy: dy.constant(),
            dz: dz.constant(),
        }
    }

    /// Get the normalized gradient as a surface normal.
    ///
    /// For an SDF f(p) = 0, the gradient ∇f points outward from the surface.
    /// Returns manifold expressions for the unit normal components.
    /// Use `Jet3::new(nx, ny, nz)` to collapse if needed.
    #[inline(always)]
    pub fn normal(
        &self,
    ) -> (
        impl Manifold<Field, Output = Field>,
        impl Manifold<Field, Output = Field>,
        impl Manifold<Field, Output = Field>,
    ) {
        let len_sq = self.dx * self.dx + self.dy * self.dy + self.dz * self.dz;
        let inv_len = len_sq.rsqrt();
        (
            self.dx.clone() * inv_len.clone(),
            self.dy.clone() * inv_len.clone(),
            self.dz.clone() * inv_len,
        )
    }

    /// Get the raw gradient without normalization.
    #[inline(always)]
    pub fn gradient(&self) -> (Field, Field, Field) {
        (self.dx, self.dy, self.dz)
    }

    /// Raw select without early exit.
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
            dz: Field::select_raw(mask.val, if_true.dz, if_false.dz),
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
    /// Returns `Jet3Sqrt` which enables automatic rsqrt fusion when divided.
    /// Example: `a / b.sqrt()` computes `a * rsqrt(b)` (faster than `a / sqrt(b)`).
    #[inline(always)]
    pub fn sqrt(self) -> Jet3Sqrt {
        Jet3Sqrt(self)
    }

    /// Absolute value with derivative.
    #[inline(always)]
    pub fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        let sign = self.val / self.val.abs();
        Self::new(
            self.val.abs(),
            self.dx * sign.clone(),
            self.dy * sign.clone(),
            self.dz * sign,
        )
    }

    /// Element-wise minimum with derivative.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dz: Field::select_raw(mask, self.dz, rhs.dz),
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
            dz: Field::select_raw(mask, self.dz, rhs.dz),
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
// Jet3Sqrt: Enables rsqrt fusion for Jet3
// ============================================================================

/// Wrapper for sqrt(Jet3) that enables automatic rsqrt fusion.
///
/// When `Jet3 / Jet3Sqrt` is computed, this automatically uses the faster
/// `rsqrt` path: `a / sqrt(b)` becomes `a * rsqrt(b)`.
///
/// Converts to `Jet3` via `Into` when used in other contexts.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet3Sqrt(Jet3);

impl Jet3Sqrt {
    /// Evaluate to get the actual sqrt result as Jet3.
    #[inline(always)]
    pub fn eval(self) -> Jet3 {
        // Chain rule: (√f)' = f' / (2√f) = f' * rsqrt(f) / 2
        let rsqrt_val = self.0.val.rsqrt();
        let sqrt_val = self.0.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);
        Jet3::new(
            sqrt_val,
            self.0.dx * half_rsqrt.clone(),
            self.0.dy * half_rsqrt.clone(),
            self.0.dz * half_rsqrt,
        )
    }
}

impl From<Jet3Sqrt> for Jet3 {
    #[inline(always)]
    fn from(s: Jet3Sqrt) -> Jet3 {
        s.eval()
    }
}

/// Rsqrt fusion: `Jet3 / Jet3Sqrt` computes `a * rsqrt(b)` directly.
impl core::ops::Div<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn div(self, rhs: Jet3Sqrt) -> Jet3 {
        // a / sqrt(b) = a * rsqrt(b)
        // Derivative: d/dx[a * b^(-1/2)] = a' * rsqrt(b) - a * b' * rsqrt(b)³ / 2
        let b = rhs.0;
        let rsqrt_b = b.val.rsqrt();
        let result_val = self.val * rsqrt_b;

        // Derivative scaling factors
        let rsqrt_cubed = rsqrt_b * rsqrt_b * rsqrt_b;
        let half_rsqrt_cubed = rsqrt_cubed * Field::from(0.5);

        Jet3::new(
            result_val,
            self.dx * rsqrt_b - self.val * b.dx * half_rsqrt_cubed.clone(),
            self.dy * rsqrt_b - self.val * b.dy * half_rsqrt_cubed.clone(),
            self.dz * rsqrt_b - self.val * b.dz * half_rsqrt_cubed,
        )
    }
}

/// Jet3Sqrt arithmetic: forward to Jet3 after evaluation
impl core::ops::Add<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn add(self, rhs: Jet3) -> Jet3 {
        self.eval() + rhs
    }
}

impl core::ops::Sub<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn sub(self, rhs: Jet3) -> Jet3 {
        self.eval() - rhs
    }
}

impl core::ops::Mul<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn mul(self, rhs: Jet3) -> Jet3 {
        self.eval() * rhs
    }
}

impl core::ops::Div<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn div(self, rhs: Jet3) -> Jet3 {
        self.eval() / rhs
    }
}

impl core::ops::Add<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn add(self, rhs: Jet3Sqrt) -> Jet3 {
        self + rhs.eval()
    }
}

impl core::ops::Sub<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn sub(self, rhs: Jet3Sqrt) -> Jet3 {
        self - rhs.eval()
    }
}

impl core::ops::Mul<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn mul(self, rhs: Jet3Sqrt) -> Jet3 {
        self * rhs.eval()
    }
}

// Note: Div<Jet3Sqrt> for Jet3 is the rsqrt fusion above

// ============================================================================
// Arithmetic via chain rule (Jet3)
// ============================================================================

impl core::ops::Add for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.val + rhs.val,
            self.dx + rhs.dx,
            self.dy + rhs.dy,
            self.dz + rhs.dz,
        )
    }
}

impl core::ops::Sub for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.val - rhs.val,
            self.dx - rhs.dx,
            self.dy - rhs.dy,
            self.dz - rhs.dz,
        )
    }
}

impl core::ops::Mul for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        Self::new(
            self.val * rhs.val,
            self.dx * rhs.val + self.val * rhs.dx,
            self.dy * rhs.val + self.val * rhs.dy,
            self.dz * rhs.val + self.val * rhs.dz,
        )
    }
}

impl Jet3 {
    /// Scalar multiplication: scales both value and derivatives by a Field.
    ///
    /// This is `pub(crate)` because external code should construct AST nodes
    /// via operator overloads, not call raw SIMD operations directly.
    #[inline(always)]
    pub(crate) fn scale(self, s: Field) -> Jet3 {
        Jet3::new(
            self.val * s,
            self.dx * s,
            self.dy * s,
            self.dz * s,
        )
    }
}

impl core::ops::Div for Jet3 {
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
            self.dy * scale.clone() - self.val * rhs.dy.clone() * inv_g_sq.clone(),
            self.dz * scale - self.val * rhs.dz.clone() * inv_g_sq,
        )
    }
}

impl core::ops::BitAnd for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::constant(self.val & rhs.val)
    }
}

impl core::ops::BitOr for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::constant(self.val | rhs.val)
    }
}

impl core::ops::Not for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }
}

// ============================================================================
// Computational trait implementation (Jet3)
// ============================================================================

impl Computational for Jet3 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self::constant(Field::sequential(start))
    }
}

// Jet3 is a coordinate type
impl crate::numeric::Coordinate for Jet3 {}

// ============================================================================
// Selectable trait implementation (Jet3)
// ============================================================================

impl Selectable for Jet3 {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self {
            val: <Field as Selectable>::select_raw(mask, if_true.val, if_false.val),
            dx: <Field as Selectable>::select_raw(mask, if_true.dx, if_false.dx),
            dy: <Field as Selectable>::select_raw(mask, if_true.dy, if_false.dy),
            dz: <Field as Selectable>::select_raw(mask, if_true.dz, if_false.dz),
        }
    }
}

// ============================================================================
// Numeric trait implementation (Jet3)
// ============================================================================

impl Numeric for Jet3 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Use rsqrt (4 cycles) instead of sqrt (20-30 cycles)
        // sqrt(x) = x * rsqrt(x)
        // d(sqrt(x))/dx = rsqrt(x) / 2
        let rsqrt_val = self.val.rsqrt();
        let sqrt_val = self.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);
        Self::new(
            sqrt_val,
            self.dx * half_rsqrt.clone(),
            self.dy * half_rsqrt.clone(),
            self.dz * half_rsqrt,
        )
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let sign = self.val / self.val.abs();
        Self::new(
            self.val.abs(),
            self.dx * sign.clone(),
            self.dy * sign.clone(),
            self.dz * sign,
        )
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dz: Field::select_raw(mask, self.dz, rhs.dz),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dz: Field::select_raw(mask, self.dz, rhs.dz),
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
            dz: Field::select_raw(mask.val, if_true.dz, if_false.dz),
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
        let sin_val = self.val.sin();
        let cos_deriv = self.val.cos();
        Self::new(
            sin_val,
            self.dx * cos_deriv,
            self.dy * cos_deriv,
            self.dz * cos_deriv,
        )
    }

    #[inline(always)]
    fn cos(self) -> Self {
        let cos_val = self.val.cos();
        let neg_sin = -self.val.sin();
        Self::new(
            cos_val,
            self.dx * neg_sin,
            self.dy * neg_sin,
            self.dz * neg_sin,
        )
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        let r_sq = self.val * self.val + x.val * x.val;
        let inv_r_sq = Field::from(1.0) / r_sq;
        let dy_darg = x.val.clone() * inv_r_sq.clone();
        let dx_darg = (-self.val).clone() * inv_r_sq;
        Self::new(
            self.val.atan2(x.val),
            self.dx * dy_darg.clone() + x.dx * dx_darg.clone(),
            self.dy * dy_darg.clone() + x.dy * dx_darg.clone(),
            self.dz * dy_darg + x.dz * dx_darg,
        )
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        use crate::numeric::Numeric as _;
        let val = Numeric::pow(self.val, exp.val);
        let ln_base = self.val.ln();
        let inv_self = Field::from(1.0).raw_div(self.val);
        let coeff = exp.val.raw_mul(inv_self);
        Self::new(
            val,
            val * (exp.dx * ln_base + coeff.clone() * self.dx),
            val * (exp.dy * ln_base + coeff.clone() * self.dy),
            val * (exp.dz * ln_base + coeff * self.dz),
        )
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let exp_val = self.val.exp();
        Self::new(
            exp_val,
            self.dx * exp_val,
            self.dy * exp_val,
            self.dz * exp_val,
        )
    }

    #[inline(always)]
    fn log2(self) -> Self {
        let log2_e = Field::from(1.4426950408889634);
        let inv_val = Field::from(1.0) / self.val;
        let deriv_coeff = inv_val * log2_e;
        Self::new(
            self.val.log2(),
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff.clone(),
            self.dz * deriv_coeff,
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
            self.dy * deriv_coeff.clone(),
            self.dz * deriv_coeff,
        )
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // (a * b + c)' where a, b, c are jets
        Self::new(
            self.val.mul_add(b.val, c.val),
            self.dx * b.val + self.val * b.dx + c.dx,
            self.dy * b.val + self.val * b.dy + c.dy,
            self.dz * b.val + self.val * b.dz + c.dz,
        )
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let inv = self.val.recip();
        let neg_inv_sq = Field::from(0.0) - inv.clone() * inv;
        Self::new(
            inv,
            self.dx * neg_inv_sq.clone(),
            self.dy * neg_inv_sq.clone(),
            self.dz * neg_inv_sq,
        )
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let scale = Field::from(-0.5) * rsqrt_cubed;
        Self::new(
            rsqrt_val,
            self.dx * scale.clone(),
            self.dy * scale.clone(),
            self.dz * scale,
        )
    }

    #[inline(always)]
    fn ln(self) -> Self {
        // Chain rule: (ln f)' = f' / f
        let inv_val = Field::from(1.0) / self.val;
        Self::new(
            self.val.ln(),
            self.dx * inv_val.clone(),
            self.dy * inv_val.clone(),
            self.dz * inv_val,
        )
    }

    #[inline(always)]
    fn log10(self) -> Self {
        // Chain rule: (log10 f)' = f' / (f * ln(10))
        let log10_e = Field::from(0.4342944819032518);
        let inv_val = Field::from(1.0) / self.val;
        let deriv_coeff = inv_val * log10_e;
        Self::new(
            self.val.log10(),
            self.dx * deriv_coeff.clone(),
            self.dy * deriv_coeff.clone(),
            self.dz * deriv_coeff,
        )
    }

    #[inline(always)]
    fn tan(self) -> Self {
        // Chain rule: (tan f)' = f' * sec²(f)
        let tan_val = self.val.tan();
        let cos_val = self.val.cos();
        let sec_sq = Field::from(1.0) / (cos_val * cos_val);
        Self::new(
            tan_val,
            self.dx * sec_sq.clone(),
            self.dy * sec_sq.clone(),
            self.dz * sec_sq,
        )
    }

    #[inline(always)]
    fn asin(self) -> Self {
        // Chain rule: (asin f)' = f' / sqrt(1 - f²)
        let one = Field::from(1.0);
        let one_minus_sq = one - self.val * self.val;
        let inv_sqrt = one_minus_sq.rsqrt();
        Self::new(
            self.val.asin(),
            self.dx * inv_sqrt.clone(),
            self.dy * inv_sqrt.clone(),
            self.dz * inv_sqrt,
        )
    }

    #[inline(always)]
    fn acos(self) -> Self {
        // Chain rule: (acos f)' = -f' / sqrt(1 - f²)
        let one = Field::from(1.0);
        let one_minus_sq = one - self.val * self.val;
        let neg_inv_sqrt = Field::from(0.0) - one_minus_sq.rsqrt();
        Self::new(
            self.val.acos(),
            self.dx * neg_inv_sqrt.clone(),
            self.dy * neg_inv_sqrt.clone(),
            self.dz * neg_inv_sqrt,
        )
    }

    #[inline(always)]
    fn atan(self) -> Self {
        // Chain rule: (atan f)' = f' / (1 + f²)
        let one = Field::from(1.0);
        let one_plus_sq = one + self.val * self.val;
        let inv = Field::from(1.0) / one_plus_sq;
        Self::new(
            self.val.atan(),
            self.dx * inv.clone(),
            self.dy * inv.clone(),
            self.dz * inv,
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
        // fract(f) = f - floor(f), derivative = f'
        Self::new(self.val.fract(), self.dx, self.dy, self.dz)
    }

    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        // hypot(x, y) = sqrt(x² + y²)
        let h = self.val.hypot(y.val);
        let inv_h = Field::from(1.0) / h;
        let dx_coeff = self.val * inv_h.clone();
        let dy_coeff = y.val * inv_h;
        Self::new(
            h,
            self.dx * dx_coeff.clone() + y.dx * dy_coeff.clone(),
            self.dy * dx_coeff.clone() + y.dy * dy_coeff.clone(),
            self.dz * dx_coeff + y.dz * dy_coeff,
        )
    }

    #[inline(always)]
    fn mul_rsqrt(self, other: Self) -> Self {
        // mul_rsqrt(a, b) = a * rsqrt(b) = a * b^(-1/2)
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
            self.dz.raw_mul(da_coeff).raw_sub(other.dz.raw_mul(db_coeff)),
        )
    }

    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        let mask_low = self.val.lt(lo.val);
        let mask_high = self.val.gt(hi.val);
        let clamped = self.val.clamp(lo.val, hi.val);
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
        let dz = Field::select_raw(
            mask_low,
            lo.dz,
            Field::select_raw(mask_high, hi.dz, self.dz),
        );
        Self { val: clamped, dx, dy, dz }
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
            dz: self.dz.add_masked(val.dz, mask.val),
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
        Self::new(-self.val, -self.dx, -self.dy, -self.dz)
    }
}

// ============================================================================
// Field/Jet3 conversions
// ============================================================================

/// Explicit lift: Field → Jet3 (constant with zero derivatives)
impl From<Field> for Jet3 {
    #[inline(always)]
    fn from(val: Field) -> Self {
        Self::constant(val)
    }
}

/// Explicit lift: f32 → Jet3 (chains through Field)
impl From<f32> for Jet3 {
    #[inline(always)]
    fn from(val: f32) -> Self {
        Self::constant(Field::from(val))
    }
}

/// Implicit projection: Jet3 → Field (extract value, discard derivatives)
impl From<Jet3> for Field {
    #[inline(always)]
    fn from(jet: Jet3) -> Self {
        jet.val
    }
}

// ============================================================================
// HasDerivatives and HasDz trait implementations (for derivative accessors)
// ============================================================================

impl crate::ops::derivative::HasDerivatives for Jet3 {
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

impl crate::ops::derivative::HasDz for Jet3 {
    #[inline(always)]
    fn dz(&self) -> Field {
        self.dz
    }
}
