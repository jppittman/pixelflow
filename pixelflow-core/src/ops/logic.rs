//! # Logic Operations
//!
//! AST nodes for bitwise logic: And, Or, Not.

use crate::Manifold;
use crate::numeric::Numeric;
use core::ops::{BitAnd, BitOr, Not};
use pixelflow_compiler::Element;

/// Bitwise AND.
#[derive(Clone, Debug, Element)]
pub struct And<L, R>(pub L, pub R);

/// Bitwise OR.
#[derive(Clone, Debug, Element)]
pub struct Or<L, R>(pub L, pub R);

/// Bitwise NOT.
#[derive(Clone, Debug, Element)]
pub struct BNot<M>(pub M);

// ============================================================================
// Domain-Generic Manifold Implementations
// ============================================================================

impl<P, L, R, O> Manifold<P> for And<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Computational,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p) & self.1.eval(p)
    }
}

impl<P, L, R, O> Manifold<P> for Or<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Computational,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p) | self.1.eval(p)
    }
}

impl<P, M, O> Manifold<P> for BNot<M>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Computational,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        !self.0.eval(p)
    }
}
