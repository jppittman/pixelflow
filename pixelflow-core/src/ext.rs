//! # DSL Extension Trait: Fluent Manifold Building
//!
//! Provides a fluent method-chaining API for composing manifold expressions.
//!
//! ## Overview
//!
//! `ManifoldExt` is the primary way users build manifold expressions. It provides:
//!
//! - **Method-chaining API**: `x.sqrt().abs().max(y)`
//! - **Operator overloading**: `x * x + y * y`
//! - **Polymorphic evaluation**: Build once, evaluate with `Field` or `Jet2`
//! - **Type-safe composition**: Expression types capture compute graphs
//!
//! ## ManifoldExpr Marker Trait
//!
//! `ManifoldExt` is implemented for types that implement `ManifoldExpr`. This marker
//! trait gates which types get the fluent API, preventing method name conflicts with
//! standard library traits like `Iterator::map` or `Ord::max`.
//!
//! Use `#[derive(ManifoldExpr)]` to enable the fluent API on custom combinator types:
//!
//! ```ignore
//! #[derive(ManifoldExpr)]
//! pub struct MyCustomCombinator<M>(pub M);
//! ```
//!
//! ## Expression Building vs. Evaluation
//!
//! PixelFlow separates these two phases:
//!
//! ### Building Phase
//! ```ignore
//! let circle = (X * X + Y * Y).sqrt() - 1.0;  // Just builds a type tree
//! ```
//!
//! **No computation happens.** The type `Sqrt<Sub<Add<Mul<X,X>, Mul<Y,Y>>, f32>>` is
//! the abstract syntax tree (AST) that represents the computation.
//!
//! ### Evaluation Phase
//! ```ignore
//! // Concrete SIMD evaluation
//! let distance = circle.eval4(3.0, 4.0, 0.0, 0.0);  // Returns Field
//!
//! // Automatic differentiation
//! let result = circle.eval((Jet2::x(3.0), Jet2::y(4.0), ...));
//! // result contains: value, ∂/∂x, ∂/∂y
//! ```
//!
//! ## Method Organization
//!
//! 1. **Unary Operations**: `sqrt`, `abs`, `sin`, `cos`, `floor`, `rsqrt`, etc.
//! 2. **Binary Operations**: `add`, `sub`, `mul`, `div`, `min`, `max`
//! 3. **Comparisons**: `lt`, `le`, `gt`, `ge`
//! 4. **Selection**: `select` (branchless conditional)
//! 5. **Evaluation** (Field4 only): `eval4`, `eval_at`, `constant`
//! 6. **Coordinate Transform** (Field4 only): `at`
//! 7. **Type Erasure** (Field4 only): `boxed`
//! 8. **Functor Operations**: `map`, `lift`

use crate::Manifold;
use crate::combinators::{At, ClosureMap, Map, Select};
use crate::ops::{
    Abs, Acos, Add, Asin, Atan, Atan2, Ceil, Clamp, Cos, Div, Eq, Exp, Exp2, Floor, Fract, Ge, Gt,
    Hypot, Le, Ln, Log10, Log2, Lt, Max, Min, Mul, MulAdd, MulRsqrt, Ne, Neg, Pow, Recip, Round,
    Rsqrt, Sin, Sqrt, Sub, Tan,
};

use alloc::sync::Arc;

/// The standard 4D Field domain for boxed manifolds and legacy APIs.
type Field4 = (crate::Field, crate::Field, crate::Field, crate::Field);

/// Type-erased manifold (returning Field), wrapped in a struct to allow trait implementations.
///
/// Note: `BoxedManifold` is Field4-specific because trait objects require a concrete type.
/// For generic numeric contexts, use static dispatch instead.
#[derive(Clone)]
pub struct BoxedManifold(pub Arc<dyn Manifold<Field4, Output = crate::Field>>);

impl Manifold<Field4> for BoxedManifold {
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: Field4) -> crate::Field {
        self.0.eval(p)
    }
}

// Operator Implementations for BoxedManifold
impl<R: Manifold<Field4>> core::ops::Add<R> for BoxedManifold {
    type Output = Add<Self, R>;
    fn add(self, rhs: R) -> Self::Output {
        Add(self, rhs)
    }
}

impl<R: Manifold<Field4>> core::ops::Sub<R> for BoxedManifold {
    type Output = Sub<Self, R>;
    fn sub(self, rhs: R) -> Self::Output {
        Sub(self, rhs)
    }
}

impl<R: Manifold<Field4>> core::ops::Mul<R> for BoxedManifold {
    type Output = Mul<Self, R>;
    fn mul(self, rhs: R) -> Self::Output {
        Mul(self, rhs)
    }
}

impl<R: Manifold<Field4>> core::ops::Div<R> for BoxedManifold {
    type Output = Div<Self, R>;
    fn div(self, rhs: R) -> Self::Output {
        Div(self, rhs)
    }
}

/// Extension methods for composing manifolds.
///
/// This trait provides a fluent API for building manifold expressions. It is
/// blanket-implemented for **all `Sized` types**, making combinator methods
/// universally available.
///
/// # Domain-Agnostic Combinators
///
/// Methods like `.sqrt()`, `.abs()`, `.add()` just wrap `self` in a combinator
/// struct. They don't need any trait bounds—the resulting struct will implement
/// `Manifold<P>` for whatever domain `P` makes sense.
///
/// # Field4-Specific Methods
///
/// Some methods require the standard `Field4` domain:
/// - `eval4`, `eval_at`, `constant` — convenience evaluation
/// - `at` — coordinate space remapping
/// - `boxed` — type erasure to trait object
///
/// These methods have where clauses restricting them to `Manifold<Field4>`.
///
/// # Example
///
/// ```ignore
/// use pixelflow_core::{ManifoldExt, X, Y};
///
/// // Build expression - works everywhere
/// let circle = (X * X + Y * Y).sqrt() - 1.0;
///
/// // Evaluate with Field4 convenience method
/// let val = circle.eval4(3.0, 4.0, 0.0, 0.0);
///
/// // Inside kernel! macro - also works (no Field4 constraint on .sqrt())
/// kernel!(|m: kernel| (DX(m) * DX(m) + DY(m) * DY(m)).sqrt())
/// ```
pub trait ManifoldExt: Sized {
    // =========================================================================
    // Unary Operations (no domain constraint)
    // =========================================================================

    /// Square root.
    #[inline(always)]
    fn sqrt(self) -> Sqrt<Self> {
        Sqrt(self)
    }

    /// Negation (-x).
    #[inline(always)]
    fn neg(self) -> Neg<Self> {
        Neg(self)
    }

    /// Reciprocal square root (1/sqrt(x)).
    #[inline(always)]
    fn rsqrt(self) -> Rsqrt<Self> {
        Rsqrt(self)
    }

    /// Absolute value.
    #[inline(always)]
    fn abs(self) -> Abs<Self> {
        Abs(self)
    }

    /// Floor (round toward negative infinity).
    #[inline(always)]
    fn floor(self) -> Floor<Self> {
        Floor(self)
    }

    /// Sine function.
    #[inline(always)]
    fn sin(self) -> Sin<Self> {
        Sin(self)
    }

    /// Cosine function.
    #[inline(always)]
    fn cos(self) -> Cos<Self> {
        Cos(self)
    }

    /// Base-2 logarithm.
    #[inline(always)]
    fn log2(self) -> Log2<Self> {
        Log2(self)
    }

    /// Base-2 exponential (2^x).
    #[inline(always)]
    fn exp2(self) -> Exp2<Self> {
        Exp2(self)
    }

    /// Natural exponential (e^x).
    #[inline(always)]
    fn exp(self) -> Exp<Self> {
        Exp(self)
    }

    /// Reciprocal (1/x).
    #[inline(always)]
    fn recip(self) -> Recip<Self> {
        Recip(self)
    }

    /// Ceiling (round toward positive infinity).
    #[inline(always)]
    fn ceil(self) -> Ceil<Self> {
        Ceil(self)
    }

    /// Round to nearest integer.
    #[inline(always)]
    fn round(self) -> Round<Self> {
        Round(self)
    }

    /// Fractional part: x - floor(x).
    #[inline(always)]
    fn fract(self) -> Fract<Self> {
        Fract(self)
    }

    /// Tangent function.
    #[inline(always)]
    fn tan(self) -> Tan<Self> {
        Tan(self)
    }

    /// Arcsine function.
    #[inline(always)]
    fn asin(self) -> Asin<Self> {
        Asin(self)
    }

    /// Arccosine function.
    #[inline(always)]
    fn acos(self) -> Acos<Self> {
        Acos(self)
    }

    /// Arctangent function.
    #[inline(always)]
    fn atan(self) -> Atan<Self> {
        Atan(self)
    }

    /// Natural logarithm.
    #[inline(always)]
    fn ln(self) -> Ln<Self> {
        Ln(self)
    }

    /// Base-10 logarithm.
    #[inline(always)]
    fn log10(self) -> Log10<Self> {
        Log10(self)
    }

    // =========================================================================
    // Binary Operations (no domain constraint)
    // =========================================================================

    /// Add two values.
    #[inline(always)]
    fn add<R>(self, rhs: R) -> Add<Self, R> {
        Add(self, rhs)
    }

    /// Subtract two values.
    #[inline(always)]
    fn sub<R>(self, rhs: R) -> Sub<Self, R> {
        Sub(self, rhs)
    }

    /// Multiply two values.
    #[inline(always)]
    fn mul<R>(self, rhs: R) -> Mul<Self, R> {
        Mul(self, rhs)
    }

    /// Divide two values.
    #[inline(always)]
    fn div<R>(self, rhs: R) -> Div<Self, R> {
        Div(self, rhs)
    }

    /// Element-wise maximum.
    #[inline(always)]
    fn max<R>(self, rhs: R) -> Max<Self, R> {
        Max(self, rhs)
    }

    /// Element-wise minimum.
    #[inline(always)]
    fn min<R>(self, rhs: R) -> Min<Self, R> {
        Min(self, rhs)
    }

    /// Fused multiply-add: self * b + c
    ///
    /// More efficient than separate multiply and add operations on most hardware.
    #[inline(always)]
    fn mul_add<B, C>(self, b: B, c: C) -> MulAdd<Self, B, C> {
        MulAdd(self, b, c)
    }

    /// Two-argument arctangent: atan2(self, x).
    ///
    /// Returns the angle in radians between the positive x-axis and the point (x, self).
    #[inline(always)]
    fn atan2<X>(self, x: X) -> Atan2<Self, X> {
        Atan2(self, x)
    }

    /// Power: self^exp.
    #[inline(always)]
    fn pow<E>(self, exp: E) -> Pow<Self, E> {
        Pow(self, exp)
    }

    /// Hypotenuse: sqrt(self² + y²).
    #[inline(always)]
    fn hypot<Y>(self, y: Y) -> Hypot<Self, Y> {
        Hypot(self, y)
    }

    /// Multiply by reciprocal square root: self * rsqrt(other) = self / sqrt(other).
    ///
    /// This is a common operation in SIMD code for normalization.
    #[inline(always)]
    fn mul_rsqrt<R>(self, other: R) -> MulRsqrt<Self, R> {
        MulRsqrt(self, other)
    }

    // =========================================================================
    // Comparisons (no domain constraint)
    // =========================================================================

    /// Less than comparison.
    #[inline(always)]
    fn lt<R>(self, rhs: R) -> Lt<Self, R> {
        Lt(self, rhs)
    }

    /// Greater than comparison.
    #[inline(always)]
    fn gt<R>(self, rhs: R) -> Gt<Self, R> {
        Gt(self, rhs)
    }

    /// Less than or equal comparison.
    #[inline(always)]
    fn le<R>(self, rhs: R) -> Le<Self, R> {
        Le(self, rhs)
    }

    /// Greater than or equal comparison.
    #[inline(always)]
    fn ge<R>(self, rhs: R) -> Ge<Self, R> {
        Ge(self, rhs)
    }

    /// Equality comparison.
    #[inline(always)]
    fn eq<R>(self, rhs: R) -> Eq<Self, R> {
        Eq(self, rhs)
    }

    /// Inequality comparison.
    #[inline(always)]
    fn ne<R>(self, rhs: R) -> Ne<Self, R> {
        Ne(self, rhs)
    }

    // =========================================================================
    // Ternary Operations (no domain constraint)
    // =========================================================================

    /// Clamp value to range [lo, hi].
    #[inline(always)]
    fn clamp<Lo, Hi>(self, lo: Lo, hi: Hi) -> Clamp<Self, Lo, Hi> {
        Clamp {
            value: self,
            lo,
            hi,
        }
    }

    // =========================================================================
    // Selection (no domain constraint)
    // =========================================================================

    /// Branchless conditional select.
    ///
    /// Returns `if_true` where `self` is non-zero, `if_false` elsewhere.
    /// Both branches are always evaluated (SIMD branchless execution).
    #[inline(always)]
    fn select<T, F>(self, if_true: T, if_false: F) -> Select<Self, T, F> {
        Select {
            cond: self,
            if_true,
            if_false,
        }
    }

    // =========================================================================
    // Functor Operations (no domain constraint)
    // =========================================================================

    /// Transform the output of this manifold using another manifold.
    ///
    /// `self` output becomes the `X` coordinate for `transform`.
    #[inline(always)]
    fn map<T>(self, transform: T) -> Map<Self, T> {
        Map::new(self, transform)
    }

    /// Lift this manifold's output to ray space via a covariant map.
    #[inline(always)]
    fn lift<F>(self, func: F) -> ClosureMap<Self, F>
    where
        F: Fn(crate::Field) -> crate::jet::PathJet<crate::Field> + Send + Sync,
    {
        ClosureMap::new(self, func)
    }

    // =========================================================================
    // Field4-Specific: Evaluation
    // =========================================================================

    /// Evaluate the manifold at the given coordinates.
    ///
    /// Convenience method that accepts types convertible to `Field`.
    #[inline(always)]
    fn eval4<
        A: Into<crate::Field>,
        B: Into<crate::Field>,
        C: Into<crate::Field>,
        D: Into<crate::Field>,
    >(
        &self,
        x: A,
        y: B,
        z: C,
        w: D,
    ) -> crate::Field
    where
        Self: Manifold<Field4, Output = crate::Field>,
    {
        Manifold::eval(self, (x.into(), y.into(), z.into(), w.into()))
    }

    /// Evaluate the manifold at manifold-computed coordinates.
    ///
    /// Takes coordinate expressions, evaluates them at origin, then evaluates
    /// self at those coordinates.
    #[inline(always)]
    fn eval_at<Cx, Cy, Cz, Cw>(&self, x: Cx, y: Cy, z: Cz, w: Cw) -> crate::Field
    where
        Self: Manifold<Field4, Output = crate::Field>,
        Cx: Manifold<Field4, Output = crate::Field>,
        Cy: Manifold<Field4, Output = crate::Field>,
        Cz: Manifold<Field4, Output = crate::Field>,
        Cw: Manifold<Field4, Output = crate::Field>,
    {
        let zero = crate::Field::from(0.0);
        let origin = (zero, zero, zero, zero);
        let new_x = x.eval(origin);
        let new_y = y.eval(origin);
        let new_z = z.eval(origin);
        let new_w = w.eval(origin);
        self.eval((new_x, new_y, new_z, new_w))
    }

    /// Collapse an AST expression to a concrete Field value.
    ///
    /// Evaluates at origin (0,0,0,0).
    #[inline(always)]
    fn constant(&self) -> crate::Field
    where
        Self: Manifold<Field4, Output = crate::Field>,
    {
        let zero = crate::Field::from(0.0);
        self.eval((zero, zero, zero, zero))
    }

    // =========================================================================
    // Field4-Specific: Coordinate Transform
    // =========================================================================

    /// Remap coordinate space before evaluating this manifold.
    ///
    /// Creates a manifold that first remaps input coordinates, then evaluates
    /// `self` at the remapped coordinates.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Scale by 2: sample at (x/2, y/2)
    /// let scaled = circle.at(X / 2.0, Y / 2.0, Z, W);
    /// ```
    /// Domain-generic coordinate transformation (contramap).
    ///
    /// Bounds are checked when the result is used as a Manifold, not here.
    /// This allows `at` to work with any domain P where the coordinate
    /// expressions are `Manifold<P, Output = I>` and self is `Manifold<(I,I,I,I)>`.
    #[inline(always)]
    fn at<Cx, Cy, Cz, Cw>(self, x: Cx, y: Cy, z: Cz, w: Cw) -> At<Cx, Cy, Cz, Cw, Self> {
        At {
            inner: self,
            x,
            y,
            z,
            w,
        }
    }

    // =========================================================================
    // Field4-Specific: Type Erasure
    // =========================================================================

    /// Type-erase this manifold into a boxed trait object.
    ///
    /// Boxing erases the static type and fixes evaluation to `Field4` domain.
    /// For `Jet2` evaluation, keep the expression statically typed.
    #[inline(always)]
    fn boxed(self) -> BoxedManifold
    where
        Self: Manifold<Field4, Output = crate::Field> + 'static,
    {
        BoxedManifold(Arc::new(self))
    }
}

/// Marker trait for types that are manifold expressions.
///
/// `ManifoldExt` methods are only available on types that implement this trait.
/// This prevents method name conflicts with standard library traits like
/// `Iterator::map`, `Iterator::min`, `Iterator::max`, `Ord::min`, `Ord::max`.
///
/// ## Deriving
///
/// Use `#[derive(ManifoldExpr)]` from `pixelflow_compiler` to implement this trait:
///
/// ```ignore
/// use pixelflow_compiler::ManifoldExpr;
///
/// #[derive(ManifoldExpr)]
/// pub struct MyCustomCombinator<M>(pub M);
/// ```
///
/// ## Built-in Implementations
///
/// All standard PixelFlow types implement this trait:
/// - Coordinate variables: `X`, `Y`, `Z`, `W`
/// - Combinators: `Sqrt`, `Add`, `Mul`, `Select`, etc.
/// - Binding: `Let`, `Var`
/// - Scalars: `f32`, `i32`, `Field`
pub trait ManifoldExpr {}

/// Blanket implementation: ManifoldExt is available for ManifoldExpr types.
///
/// This enables method chaining on manifold expression trees while avoiding
/// conflicts with standard library traits. Field4-specific methods are further
/// gated by where clauses on those individual methods.
impl<T: ManifoldExpr> ManifoldExt for T {}

// ============================================================================
// ManifoldExpr implementations for scalar types
// ============================================================================

impl ManifoldExpr for f32 {}
impl ManifoldExpr for i32 {}
impl ManifoldExpr for crate::Field {}
impl ManifoldExpr for BoxedManifold {}

// References to ManifoldExpr types are also ManifoldExpr
// This enables (&manifold).at(...) for borrowed manifolds
impl<M: ManifoldExpr + ?Sized> ManifoldExpr for &M {}
