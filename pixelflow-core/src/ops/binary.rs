//! # Binary Operations
//!
//! AST nodes for arithmetic: Add, Sub, Mul, Div, MulAdd.
//!
//! ## Automatic Optimization: Div by Sqrt → Mul by Rsqrt
//!
//! When dividing by `Sqrt<R>`, the type system automatically fuses this into
//! `Mul<L, Rsqrt<R>>`, using the fast SIMD rsqrt instruction instead of
//! separate sqrt and divide (~8 cycles vs ~25 cycles).
//!
//! ```ignore
//! // These are semantically equivalent but the second is automatically optimized:
//! let slow = x / y.sqrt();           // Div<X, Sqrt<Y>> - uses sqrt + div
//! let fast = x * Rsqrt(y);           // Mul<X, Rsqrt<Y>> - uses rsqrt + mul
//!
//! // The operator overloading makes this automatic:
//! let auto = x / Sqrt(y);            // Becomes Mul<X, Rsqrt<Y>> at compile time!
//! ```

use crate::Manifold;
use crate::jet::Jet3;
use crate::Field;
use pixelflow_compiler::Element;

/// Addition: L + R
#[derive(Clone, Debug, Default, Element)]
pub struct Add<L, R>(pub L, pub R);

/// Subtraction: L - R
#[derive(Clone, Debug, Default, Element)]
pub struct Sub<L, R>(pub L, pub R);

/// Multiplication: L * R
#[derive(Clone, Debug, Default, Element)]
pub struct Mul<L, R>(pub L, pub R);

/// Division: L / R
#[derive(Clone, Debug, Default, Element)]
pub struct Div<L, R>(pub L, pub R);

/// Fused Multiply-Add: A * B + C
///
/// Uses FMA instruction when available (single rounding).
/// This is automatically generated when `Mul + Rhs` or `Lhs + Mul` is written,
/// enabling zero-cost compile-time fusion.
#[derive(Clone, Debug, Default, Element)]
pub struct MulAdd<A, B, C>(pub A, pub B, pub C);

/// Multiply by precomputed reciprocal: M * (1/divisor)
///
/// Optimizes division by constants. The reciprocal is computed once at
/// construction time, turning expensive divisions into fast multiplies.
/// Automatically generated when `Manifold / f32` is written.
#[derive(Clone, Debug, Default, Element)]
pub struct MulRecip<M> {
    /// The inner manifold to evaluate
    pub inner: M,
    /// Precomputed 1/divisor
    pub recip: f32,
}

/// Masked Add: Acc + (Mask ? Val : 0)
///
/// Optimized winding number accumulation. On AVX-512, uses masked add
/// instruction for true single-instruction operation.
#[derive(Clone, Debug, Default, Element)]
pub struct AddMasked<Acc, Val, Mask> {
    /// Accumulator value
    pub acc: Acc,
    /// Value to conditionally add
    pub val: Val,
    /// Mask determining which lanes to add
    pub mask: Mask,
}

/// Scalar multiplication: Jet3 * Field (scales value and all derivatives).
///
/// Essential for ray marching where `hit_point = ray_direction * distance`.
/// The ray direction (Jet3) carries derivatives showing how it varies with
/// screen position, while distance (Field) is a scalar multiplier.
///
/// ```ignore
/// let hx = Scale(X, safe_t);  // X is Jet3, safe_t is Field
/// ```
#[derive(Clone, Debug, Default, Element)]
pub struct Scale<J, S>(pub J, pub S);

// ============================================================================
// Domain-Generic Manifold Implementations
// ============================================================================

impl<P, L, R, O> Manifold<P> for Add<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).raw_add(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Sub<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).raw_sub(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Mul<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).raw_mul(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Div<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).raw_div(self.1.eval(p))
    }
}

impl<P, A, B, C, O> Manifold<P> for MulAdd<A, B, C>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    A: Manifold<P, Output = O>,
    B: Manifold<P, Output = O>,
    C: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        let a = self.0.eval(p);
        let b = self.1.eval(p);
        let c = self.2.eval(p);
        a.mul_add(b, c)
    }
}

impl<P, M, O> Manifold<P> for MulRecip<M>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        // Multiply by precomputed reciprocal - avoids slow division
        self.inner.eval(p).raw_mul(O::from_f32(self.recip))
    }
}

impl<P, Acc, Val, Mask, O> Manifold<P> for AddMasked<Acc, Val, Mask>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    Acc: Manifold<P, Output = O>,
    Val: Manifold<P, Output = O>,
    Mask: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        let acc = self.acc.eval(p);
        let val = self.val.eval(p);
        let mask = self.mask.eval(p);
        acc.add_masked(val, mask)
    }
}

// ============================================================================
// Scalar Multiplication: Jet3 * Field
// ============================================================================

impl<P, J, S> Manifold<P> for Scale<J, S>
where
    P: Copy + Send + Sync,
    J: Manifold<P, Output = Jet3>,
    S: Manifold<P, Output = Field>,
{
    type Output = Jet3;
    #[inline(always)]
    fn eval(&self, p: P) -> Jet3 {
        self.0.eval(p).scale(self.1.eval(p))
    }
}

// ============================================================================
// Automatic Fusion: L / Sqrt(R) → L * Rsqrt(R)
// ============================================================================

/// Multiply by reciprocal square root: L * rsqrt(R).
///
/// This is the optimized form of `L / sqrt(R)`, using fast SIMD rsqrt
/// instruction with Newton-Raphson refinement instead of separate
/// sqrt and divide operations (~8 cycles vs ~25 cycles).
///
/// Created automatically when dividing by `Sqrt<R>`.
#[derive(Clone, Debug, Element)]
pub struct MulRsqrt<L, R>(pub L, pub R);

/// Two-argument arctangent: atan2(y, x).
/// Returns the angle in radians between the positive x-axis and the point (x, y).
#[derive(Clone, Debug, Element)]
pub struct Atan2<Y, X>(pub Y, pub X);

/// Power: base^exponent.
#[derive(Clone, Debug, Element)]
pub struct Pow<Base, Exp>(pub Base, pub Exp);

/// Hypotenuse: sqrt(x² + y²).
#[derive(Clone, Debug, Element)]
pub struct Hypot<X, Y>(pub X, pub Y);

impl<P, L, R, O> Manifold<P> for MulRsqrt<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        // L * rsqrt(R) = L / sqrt(R) but faster
        self.0.eval(p).raw_mul(self.1.eval(p).rsqrt())
    }
}

impl<P, Y, X, O> Manifold<P> for Atan2<Y, X>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    Y: Manifold<P, Output = O>,
    X: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).atan2(self.1.eval(p))
    }
}

impl<P, Base, Exp, O> Manifold<P> for Pow<Base, Exp>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    Base: Manifold<P, Output = O>,
    Exp: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).pow(self.1.eval(p))
    }
}

impl<P, X, Y, O> Manifold<P> for Hypot<X, Y>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    X: Manifold<P, Output = O>,
    Y: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).hypot(self.1.eval(p))
    }
}
