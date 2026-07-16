//! Unary operations: sqrt, abs, floor, ceil, round, sin, cos, exp, log2, recip.

use crate::Manifold;
use crate::numeric::{Computational, Numeric};
use pixelflow_compiler::Element;

/// Square root.
#[derive(Clone, Debug, Element)]
pub struct Sqrt<M>(pub M);

/// Negation: -M
#[derive(Clone, Debug, Element)]
pub struct Neg<M>(pub M);

/// Absolute value.
#[derive(Clone, Debug, Element)]
pub struct Abs<M>(pub M);

/// Floor (round toward negative infinity).
#[derive(Clone, Debug, Element)]
pub struct Floor<M>(pub M);

/// Reciprocal square root: 1/sqrt(M).
/// Uses fast SIMD rsqrt with Newton-Raphson refinement.
#[derive(Clone, Debug, Element)]
pub struct Rsqrt<M>(pub M);

/// Sine function.
#[derive(Clone, Debug, Element)]
pub struct Sin<M>(pub M);

/// Cosine function.
#[derive(Clone, Debug, Element)]
pub struct Cos<M>(pub M);

/// Base-2 logarithm.
#[derive(Clone, Debug, Element)]
pub struct Log2<M>(pub M);

/// Base-2 exponential.
#[derive(Clone, Debug, Element)]
pub struct Exp2<M>(pub M);

/// Natural exponential (e^x).
#[derive(Clone, Debug, Element)]
pub struct Exp<M>(pub M);

/// Reciprocal (1/x).
#[derive(Clone, Debug, Element)]
pub struct Recip<M>(pub M);

/// Ceiling (round toward positive infinity).
#[derive(Clone, Debug, Element)]
pub struct Ceil<M>(pub M);

/// Round to nearest integer.
#[derive(Clone, Debug, Element)]
pub struct Round<M>(pub M);

/// Fractional part: x - floor(x).
#[derive(Clone, Debug, Element)]
pub struct Fract<M>(pub M);

/// Tangent function.
#[derive(Clone, Debug, Element)]
pub struct Tan<M>(pub M);

/// Arcsine function.
#[derive(Clone, Debug, Element)]
pub struct Asin<M>(pub M);

/// Arccosine function.
#[derive(Clone, Debug, Element)]
pub struct Acos<M>(pub M);

/// Arctangent function.
#[derive(Clone, Debug, Element)]
pub struct Atan<M>(pub M);

/// Natural logarithm.
#[derive(Clone, Debug, Element)]
pub struct Ln<M>(pub M);

/// Base-10 logarithm.
#[derive(Clone, Debug, Element)]
pub struct Log10<M>(pub M);

/// Element-wise maximum.
#[derive(Clone, Debug, Element)]
pub struct Max<L, R>(pub L, pub R);

/// Element-wise minimum.
#[derive(Clone, Debug, Element)]
pub struct Min<L, R>(pub L, pub R);

// ============================================================================
// Domain-Generic Manifold Implementations
// ============================================================================

impl<P, M, O> Manifold<P> for Sqrt<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).sqrt()
    }
}

impl<P, M, O> Manifold<P> for Neg<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).raw_neg()
    }
}

impl<P, M, O> Manifold<P> for Abs<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).abs()
    }
}

impl<P, M, O> Manifold<P> for Floor<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).floor()
    }
}

impl<P, M, O> Manifold<P> for Rsqrt<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).rsqrt()
    }
}

impl<P, M, O> Manifold<P> for Sin<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).sin()
    }
}

impl<P, M, O> Manifold<P> for Cos<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).cos()
    }
}

impl<P, M, O> Manifold<P> for Log2<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).log2()
    }
}

impl<P, M, O> Manifold<P> for Exp2<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).exp2()
    }
}

impl<P, M, O> Manifold<P> for Exp<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).exp()
    }
}

impl<P, M, O> Manifold<P> for Recip<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).recip()
    }
}

impl<P, M, O> Manifold<P> for Ceil<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).ceil()
    }
}

impl<P, M, O> Manifold<P> for Round<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).round()
    }
}

impl<P, M, O> Manifold<P> for Fract<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).fract()
    }
}

impl<P, M, O> Manifold<P> for Tan<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).tan()
    }
}

impl<P, M, O> Manifold<P> for Asin<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).asin()
    }
}

impl<P, M, O> Manifold<P> for Acos<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).acos()
    }
}

impl<P, M, O> Manifold<P> for Atan<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).atan()
    }
}

impl<P, M, O> Manifold<P> for Ln<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).ln()
    }
}

impl<P, M, O> Manifold<P> for Log10<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).log10()
    }
}

impl<P, L, R, O> Manifold<P> for Max<L, R>
where
    P: Copy + Send + Sync,
    O: Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).max(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Min<L, R>
where
    P: Copy + Send + Sync,
    O: Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).min(self.1.eval(p))
    }
}
