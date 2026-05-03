//! # Array-Based Context System
//!
//! Provides `WithContext` and `CtxVar` for binding parameters in kernel expressions.
//! Uses array-based indexing with trait abstraction for scalable impl coverage.
//!
//! ## Design
//!
//! Instead of separate impls for each tuple arity, we use:
//! 1. `ContextShape` marker trait - identifies context tuple shapes
//! 2. `ArrayAccess<Pos, I>` trait - provides element access for CtxVar
//!
//! This gives us:
//! - Single `WithContext` Manifold impl (generic over any context shape)
//! - Single `CtxVar` Manifold impl (generic over ArrayAccess)
//! - Single `Spatial` impl for context-extended domains
//!
//! ## CtxVar Indexing
//!
//! `CtxVar<A0, 5>` reads from array A0 at index 5.
//! `CtxVar<A1, 0>` reads from array A1 at index 0.
//!
//! Array position markers: `A0`, `A1`, `A2`, `A3`, etc.

use crate::Manifold;
use crate::domain::Spatial;
use core::marker::PhantomData;

// ============================================================================
// ContextShape: Marker Trait for Context Tuple Shapes
// ============================================================================

/// Marker trait for context tuple shapes.
///
/// This distinguishes context tuples (e.g., `([T; N],)`) from base spatial domains
/// (e.g., `(Field, Field)`), avoiding impl conflicts for `Spatial`.
pub trait ContextShape: Copy + Send + Sync {}

// ============================================================================
// ArrayAccess: Element Access for CtxVar
// ============================================================================

/// Access an element from a context tuple at array position `Pos` and index `I`.
///
/// This trait enables a single generic `CtxVar` Manifold impl instead of
/// separate impls for each (arity, position) combination.
pub trait ArrayAccess<Pos, const I: usize>: ContextShape {
    /// The element type at this position.
    type Element: Copy + Send + Sync;
    /// Get the element at array position `Pos`, index `I`.
    fn access(&self) -> Self::Element;
}

// ============================================================================
// Array Position Markers
// ============================================================================

/// Marker for the first array (index 0) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A0;

/// Marker for the second array (index 1) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A1;

/// Marker for the third array (index 2) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A2;

/// Marker for the fourth array (index 3) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A3;

/// Marker for the fifth array (index 4) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A4;

/// Marker for the sixth array (index 5) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A5;

/// Marker for the seventh array (index 6) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A6;

/// Marker for the eighth array (index 7) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A7;

/// Marker for the ninth array (index 8) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A8;

/// Marker for the tenth array (index 9) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A9;

/// Marker for the eleventh array (index 10) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A10;

/// Marker for the twelfth array (index 11) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A11;

/// Marker for the thirteenth array (index 12) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A12;

/// Marker for the fourteenth array (index 13) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A13;

/// Marker for the fifteenth array (index 14) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A14;

/// Marker for the sixteenth array (index 15) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A15;

// ============================================================================
// ContextShape Implementations
// ============================================================================

// Note: () is NOT a ContextShape - it's a special case that passes through
// without extending the domain.
impl<T: Copy + Send + Sync, const N: usize> ContextShape for ([T; N],) {}
impl<T0: Copy + Send + Sync, T1: Copy + Send + Sync, const N: usize, const M: usize> ContextShape
    for ([T0; N], [T1; M])
{
}
impl<
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    const N: usize,
    const M: usize,
    const K: usize,
> ContextShape for ([T0; N], [T1; M], [T2; K])
{
}
impl<
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    const N: usize,
    const M: usize,
    const K: usize,
    const L: usize,
> ContextShape for ([T0; N], [T1; M], [T2; K], [T3; L])
{
}

// ============================================================================
// ArrayAccess Implementations
// ============================================================================

// Macro to generate ArrayAccess impls for all shapes and positions
macro_rules! impl_array_access {
    // Single array: A0 only
    (1: $($bound:ident: $t:ident),*; $n:ident) => {
        impl<$($t: Copy + Send + Sync,)* const $n: usize, const I: usize>
            ArrayAccess<A0, I> for ([$($t)*; $n],)
        {
            type Element = $($t)*;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.0[I] }
        }
    };
    // Two arrays: A0, A1
    (2: $t0:ident, $t1:ident; $n:ident, $m:ident) => {
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, const $n: usize, const $m: usize, const I: usize>
            ArrayAccess<A0, I> for ([$t0; $n], [$t1; $m])
        {
            type Element = $t0;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.0[I] }
        }
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, const $n: usize, const $m: usize, const I: usize>
            ArrayAccess<A1, I> for ([$t0; $n], [$t1; $m])
        {
            type Element = $t1;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.1[I] }
        }
    };
    // Three arrays: A0, A1, A2
    (3: $t0:ident, $t1:ident, $t2:ident; $n:ident, $m:ident, $k:ident) => {
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const I: usize>
            ArrayAccess<A0, I> for ([$t0; $n], [$t1; $m], [$t2; $k])
        {
            type Element = $t0;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.0[I] }
        }
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const I: usize>
            ArrayAccess<A1, I> for ([$t0; $n], [$t1; $m], [$t2; $k])
        {
            type Element = $t1;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.1[I] }
        }
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const I: usize>
            ArrayAccess<A2, I> for ([$t0; $n], [$t1; $m], [$t2; $k])
        {
            type Element = $t2;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.2[I] }
        }
    };
    // Four arrays: A0, A1, A2, A3
    (4: $t0:ident, $t1:ident, $t2:ident, $t3:ident; $n:ident, $m:ident, $k:ident, $l:ident) => {
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync, $t3: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const $l: usize, const I: usize>
            ArrayAccess<A0, I> for ([$t0; $n], [$t1; $m], [$t2; $k], [$t3; $l])
        {
            type Element = $t0;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.0[I] }
        }
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync, $t3: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const $l: usize, const I: usize>
            ArrayAccess<A1, I> for ([$t0; $n], [$t1; $m], [$t2; $k], [$t3; $l])
        {
            type Element = $t1;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.1[I] }
        }
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync, $t3: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const $l: usize, const I: usize>
            ArrayAccess<A2, I> for ([$t0; $n], [$t1; $m], [$t2; $k], [$t3; $l])
        {
            type Element = $t2;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.2[I] }
        }
        impl<$t0: Copy + Send + Sync, $t1: Copy + Send + Sync, $t2: Copy + Send + Sync, $t3: Copy + Send + Sync,
             const $n: usize, const $m: usize, const $k: usize, const $l: usize, const I: usize>
            ArrayAccess<A3, I> for ([$t0; $n], [$t1; $m], [$t2; $k], [$t3; $l])
        {
            type Element = $t3;
            #[inline(always)]
            fn access(&self) -> Self::Element { self.3[I] }
        }
    };
}

impl_array_access!(1: T: T; N);
impl_array_access!(2: T0, T1; N, M);
impl_array_access!(3: T0, T1, T2; N, M, K);
impl_array_access!(4: T0, T1, T2, T3; N, M, K, L);

// ============================================================================
// Context Combinator
// ============================================================================

/// Context combinator: evaluates manifolds in `Ctx` arrays, passes results to `Body`.
///
/// ## Domain Structure
///
/// After evaluation, creates an extended domain:
/// - Single array: `(([T; N],), P)`
/// - Two arrays: `(([T0; N], [T1; M]), P)`
/// - etc.
#[derive(Clone, Debug)]
pub struct WithContext<Ctx, Body> {
    /// The context tuple of arrays to bind.
    pub ctx: Ctx,
    /// The body manifold that receives the evaluated context.
    pub body: Body,
}

impl<Ctx, Body> WithContext<Ctx, Body> {
    /// Create a new context combinator.
    pub const fn new(ctx: Ctx, body: Body) -> Self {
        Self { ctx, body }
    }
}

// ============================================================================
// CtxVar - Array-Indexed Variable Reference
// ============================================================================

/// Type-level index into a context array.
///
/// `ArrayPos` selects which array (A0, A1, A2, A3).
/// `INDEX` is the position within that array.
///
/// ZST, so expressions using it are Copy.
#[derive(Clone, Copy, Debug, Default)]
pub struct CtxVar<ArrayPos, const INDEX: usize>(PhantomData<ArrayPos>);

impl<ArrayPos, const INDEX: usize> CtxVar<ArrayPos, INDEX> {
    /// Create a new context variable reference.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<ArrayPos, const INDEX: usize> crate::ext::ManifoldExpr for CtxVar<ArrayPos, INDEX> {}

// A parameterized kernel that has been applied to concrete values is itself a
// valid manifold expression and participates in the fluent ManifoldExt API.
impl<Ctx, Body> crate::ext::ManifoldExpr for WithContext<Ctx, Body> {}

// ============================================================================
// Operator Implementations for CtxVar
// ============================================================================

impl<ArrayPos, const INDEX: usize, R> core::ops::Add<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Add<CtxVar<ArrayPos, INDEX>, R>;
    fn add(self, rhs: R) -> Self::Output {
        crate::ops::Add(self, rhs)
    }
}

impl<ArrayPos, const INDEX: usize, R> core::ops::Sub<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Sub<CtxVar<ArrayPos, INDEX>, R>;
    fn sub(self, rhs: R) -> Self::Output {
        crate::ops::Sub(self, rhs)
    }
}

impl<ArrayPos, const INDEX: usize, R> core::ops::Mul<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Mul<CtxVar<ArrayPos, INDEX>, R>;
    fn mul(self, rhs: R) -> Self::Output {
        crate::ops::Mul(self, rhs)
    }
}

impl<ArrayPos, const INDEX: usize, R> core::ops::Div<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Div<CtxVar<ArrayPos, INDEX>, R>;
    fn div(self, rhs: R) -> Self::Output {
        crate::ops::Div(self, rhs)
    }
}

// ============================================================================
// WithContext Manifold Implementations
// ============================================================================

// Special case: empty context passes through without domain extension
impl<P, B, Out> Manifold<P> for WithContext<(), B>
where
    P: Copy + Send + Sync,
    B: Manifold<P, Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval(p)
    }
}

// Generic impl for all non-empty context shapes
impl<Ctx, P, B, Out> Manifold<P> for WithContext<Ctx, B>
where
    Ctx: ContextShape,
    P: Copy + Send + Sync,
    B: Manifold<(Ctx, P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval((self.ctx, p))
    }
}

// Copy impl for all WithContext types
impl<Ctx: Copy, B: Copy> Copy for WithContext<Ctx, B> {}

// ============================================================================
// CtxVar Manifold Implementation (Generic)
// ============================================================================

/// Single generic impl for all CtxVar types.
///
/// Uses ArrayAccess trait to access the correct element from any context shape.
impl<Ctx, Pos, const I: usize, P> Manifold<(Ctx, P)> for CtxVar<Pos, I>
where
    Ctx: ArrayAccess<Pos, I>,
    Pos: Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = Ctx::Element;

    #[inline(always)]
    fn eval(&self, (ctx, _): (Ctx, P)) -> Self::Output {
        ctx.access()
    }
}

// ============================================================================
// ContextFree: Lift Manifold<P> to Manifold<(Ctx, P)>
// ============================================================================

/// Lifts a `Manifold<P>` to `Manifold<(Ctx, P)>` by ignoring the context.
///
/// Used for manifold params in kernels with context variables. The manifold
/// param only knows about the base domain P, but expressions using it alongside
/// context variables need to work at the context-extended domain (Ctx, P).
///
/// # Example
///
/// ```ignore
/// // Manifold param geometry: M where M: Manifold<(Jet3, Jet3, Jet3, Jet3)>
/// // But we need it to work with CtxVar which is Manifold<(([Jet3; 4],), (Jet3, Jet3, Jet3, Jet3))>
/// let geometry = ContextFree(&self.geometry);
/// // Now geometry.gt(CtxVar::<A0, 0>::new()) works!
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ContextFree<M>(pub M);

impl<M> ContextFree<M> {
    /// Create a new context-free wrapper.
    #[inline(always)]
    pub const fn new(inner: M) -> Self {
        Self(inner)
    }
}

impl<M: crate::ext::ManifoldExpr> crate::ext::ManifoldExpr for ContextFree<M> {}

// ============================================================================
// Operator Implementations for ContextFree
// ============================================================================
// Allow ContextFree<M> to participate in arithmetic with CtxVar and other types.

impl<M, R> core::ops::Add<R> for ContextFree<M> {
    type Output = crate::ops::Add<ContextFree<M>, R>;
    fn add(self, rhs: R) -> Self::Output {
        crate::ops::Add(self, rhs)
    }
}

impl<M, R> core::ops::Sub<R> for ContextFree<M> {
    type Output = crate::ops::Sub<ContextFree<M>, R>;
    fn sub(self, rhs: R) -> Self::Output {
        crate::ops::Sub(self, rhs)
    }
}

impl<M, R> core::ops::Mul<R> for ContextFree<M> {
    type Output = crate::ops::Mul<ContextFree<M>, R>;
    fn mul(self, rhs: R) -> Self::Output {
        crate::ops::Mul(self, rhs)
    }
}

impl<M, R> core::ops::Div<R> for ContextFree<M> {
    type Output = crate::ops::Div<ContextFree<M>, R>;
    fn div(self, rhs: R) -> Self::Output {
        crate::ops::Div(self, rhs)
    }
}

// Implement Manifold for all context shapes
impl<Ctx, P, M> Manifold<(Ctx, P)> for ContextFree<M>
where
    Ctx: ContextShape,
    P: Copy + Send + Sync,
    M: Manifold<P>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, (_, p): (Ctx, P)) -> Self::Output {
        self.0.eval(p)
    }
}

// Allow ContextFree to be evaluated on Field4 (e.g. inside At)
impl<M> Manifold<(crate::Field, crate::Field, crate::Field, crate::Field)> for ContextFree<M>
where
    M: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field)>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, p: (crate::Field, crate::Field, crate::Field, crate::Field)) -> Self::Output {
        self.0.eval(p)
    }
}

// ============================================================================
// Spatial Implementations for Context-Extended Domains
// ============================================================================

// Macro to generate Spatial impls for context-extended domains.
// These can't use a blanket impl due to overlap with base domain (I, I) impls.
#[allow(unused_macros)]
macro_rules! impl_spatial_for_context {
    ($($shape:ty),+ $(,)?) => {
        $(
            impl<P: Spatial> Spatial for ($shape, P) {
                type Coord = P::Coord;
                type Scalar = P::Scalar;

                #[inline(always)]
                fn x(&self) -> Self::Coord { self.1.x() }

                #[inline(always)]
                fn y(&self) -> Self::Coord { self.1.y() }

                #[inline(always)]
                fn z(&self) -> Self::Coord { self.1.z() }

                #[inline(always)]
                fn w(&self) -> Self::Coord { self.1.w() }
            }
        )+
    };
}

// Generate Spatial impls for each context shape.
// Note: The type parameters must be concrete for each invocation.
impl<T: Copy + Send + Sync, const N: usize, P: Spatial> Spatial for (([T; N],), P) {
    type Coord = P::Coord;
    type Scalar = P::Scalar;
    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }
    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }
    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }
    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

impl<T0: Copy + Send + Sync, T1: Copy + Send + Sync, const N: usize, const M: usize, P: Spatial>
    Spatial for (([T0; N], [T1; M]), P)
{
    type Coord = P::Coord;
    type Scalar = P::Scalar;
    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }
    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }
    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }
    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

impl<
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    const N: usize,
    const M: usize,
    const K: usize,
    P: Spatial,
> Spatial for (([T0; N], [T1; M], [T2; K]), P)
{
    type Coord = P::Coord;
    type Scalar = P::Scalar;
    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }
    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }
    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }
    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

impl<
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    const N: usize,
    const M: usize,
    const K: usize,
    const L: usize,
    P: Spatial,
> Spatial for (([T0; N], [T1; M], [T2; K], [T3; L]), P)
{
    type Coord = P::Coord;
    type Scalar = P::Scalar;
    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }
    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }
    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }
    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

#[cfg(test)]
mod context_domain_tests {
    use super::*;
    use crate::Field;
    use crate::X;
    use crate::ext::ManifoldExt;
    use crate::jet::Jet3;
    use crate::ops::binary::MulAdd;
    use crate::ops::derivative::DZ;
    use crate::ops::logic::And;

    type CtxDomain = (([Jet3; 3],), (Jet3, Jet3, Jet3, Jet3));

    fn check_manifold<P: Copy + Send + Sync, M: Manifold<P>>(_m: &M) {}















    // Test matching the GeometryMask kernel pattern









}
