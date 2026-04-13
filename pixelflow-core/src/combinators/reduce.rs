//! # Reduce Combinator
//!
//! Folds a manifold over a discrete range in a specific dimension.
//! This enables summations, products, and other iterative reductions.
//!
//! ## Unrolling and ZST
//!
//! `Reduce` is implemented using type-level recursion to ensure the loop is
//! fully unrolled into a "line program" of evaluations and additions.
//! If the inner manifold is a ZST, the entire reduction is also a ZST.

use crate::{Field, Manifold, variables::Dimension};
use crate::combinators::binding::{UTerm, UInt, Pred, ToInt};
use core::marker::PhantomData;
use pixelflow_compiler::Element;

/// Reducer: Folds a manifold over a discrete range in dimension `D`.
///
/// `Reduce<M, D, N>` evaluates `M` at `index = 0, 1, ..., N-1` in dimension `D`,
/// passing through other coordinates from the input domain, and sums the results.
#[derive(Clone, Copy, Debug, Default, Element)]
pub struct Reduce<M, D, N> {
    /// The manifold to reduce.
    pub inner: M,
    /// Dimension marker (X, Y, Z, or W).
    pub _dim: PhantomData<D>,
    /// Number of elements to reduce (type-level binary number).
    pub _count: PhantomData<N>,
}

impl<M, D, N> Reduce<M, D, N> {
    /// Create a new Reduce combinator.
    pub const fn new(inner: M) -> Self {
        Self {
            inner,
            _dim: PhantomData,
            _count: PhantomData,
        }
    }
}

// ============================================================================
// Manifold Implementation (Recursive Unrolling)
// ============================================================================

// Base case: Reduce 0 elements -> always zero
impl<P, M, D> Manifold<P> for Reduce<M, D, UTerm>
where
    P: Send + Sync,
    M: Send + Sync,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, _p: P) -> Field {
        Field::from(0.0)
    }
}

// Recursive case: Reduce N elements -> Reduce(N-1) + eval(N-1)
impl<P, M, D, U, B> Manifold<P> for Reduce<M, D, UInt<U, B>>
where
    P: crate::domain::Spatial<Coord = Field> + Copy + Send + Sync,
    M: Manifold<(Field, Field, Field, Field), Output = Field> + Clone,
    D: Dimension + Send + Sync,
    UInt<U, B>: ToInt + Pred + Send + Sync,
    <UInt<U, B> as Pred>::Output: Send + Sync,
    Reduce<M, D, <UInt<U, B> as Pred>::Output>: Manifold<P, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        use crate::ext::ManifoldExt;
        
        // 1. Evaluate the reduction up to N-1
        let prev = Reduce::<M, D, <UInt<U, B> as Pred>::Output>::new(self.inner.clone()).eval(p);

        // 2. Evaluate current index (N-1)
        let k = (UInt::<U, B>::VALUE - 1) as f32;
        
        let current = match D::AXIS {
            crate::variables::Axis::X => self.inner.eval((Field::from(k), p.y(), p.z(), p.w())),
            crate::variables::Axis::Y => self.inner.eval((p.x(), Field::from(k), p.z(), p.w())),
            crate::variables::Axis::Z => self.inner.eval((p.x(), p.y(), Field::from(k), p.w())),
            crate::variables::Axis::W => self.inner.eval((p.x(), p.y(), p.z(), Field::from(k))),
        };

        // AST builder
        (prev + current).eval((Field::from(0.0), Field::from(0.0), Field::from(0.0), Field::from(0.0)))
    }
}
