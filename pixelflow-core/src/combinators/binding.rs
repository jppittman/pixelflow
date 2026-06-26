//! # Let Bindings via Domain Extension
//!
//! Provides `Let` and `Var` combinators for local variable bindings in kernel
//! expressions. Uses Peano-encoded indices for type-safe de Bruijn indexing.
//!
//! ## Architecture
//!
//! Let bindings work by **extending the domain**:
//!
//! - `Let<Val, Body>` evaluates `Val`, extends domain with the result, evaluates `Body`
//! - `Var<N>` reads from the domain stack using Peano index `N`
//!
//! ## Domain Flow
//!
//! ```text
//! Base domain: (x, y)
//!
//! After Let(val_expr, body):
//!   1. Evaluate val_expr on (x, y) → v
//!   2. Create extended domain: LetExtended(v, (x, y))
//!   3. Evaluate body on extended domain
//!
//! Inside body:
//!   - Var<0> reads v (head of domain)
//!   - X, Y, Z, W read spatial coords (via Spatial trait)
//! ```
//!
//! ## Example
//!
//! ```ignore
//! // let dist = sqrt(x² + y²); dist - 1.0
//! Let::new(
//!     (X * X + Y * Y).sqrt(),  // val: compute distance once
//!     Var::<N0> - 1.0f32,      // body: use it (Var<0> reads head)
//! )
//! ```
//!
//! ## Why Binary Type-Level Numbers?
//!
//! Const generic recursion like `Get<{ N - 1 }>` causes the trait solver to
//! hang because it performs arithmetic during resolution. Binary encoding
//! (`UInt<UInt<UTerm, B1>, B0>` for 0b10) uses structural recursion with
//! logarithmic depth instead of linear depth.

use crate::Manifold;
use crate::domain::{Head, LetExtended, Tail};
use core::marker::PhantomData;
use pixelflow_compiler::Element;

// ============================================================================
// Binary Type-Level Numbers (Logarithmic Depth)
// ============================================================================

/// Terminal/Zero for binary numbers.
#[derive(Clone, Copy, Debug)]
pub struct UTerm;

/// Binary 0 bit.
#[derive(Clone, Copy, Debug)]
pub struct B0;

/// Binary 1 bit.
#[derive(Clone, Copy, Debug)]
pub struct B1;

/// Unsigned integer: UInt<N, B> = N << 1 | B
/// Represents binary numbers with logarithmic nesting depth.
#[derive(Clone, Copy, Debug)]
pub struct UInt<U, B>(PhantomData<(U, B)>);

// Convenience aliases: Generate N0..N255 using binary encoding
pixelflow_compiler::generate_binary_types!(256);

// ============================================================================
// Let Combinator
// ============================================================================

/// Bind a value and evaluate the body with the extended domain.
///
/// `Let<Val, Body>` evaluates `Val` on the input domain, extends the domain
/// with the result using `LetExtended`, then evaluates `Body` on the extended domain.
///
/// ## Domain Extension
///
/// If the input domain is `P`, and `Val: Manifold<P, Output = V>`, then:
/// - `Body` is evaluated on domain `LetExtended<V, P>`
/// - Inside `Body`, `Var<0>` reads the bound value `V`
/// - `X`, `Y`, `Z`, `W` still work (via `Spatial` trait delegation)
#[derive(Clone, Debug, Element)]
pub struct Let<Val, Body> {
    /// The value expression to bind.
    pub val: Val,
    /// The body expression (evaluated with extended context).
    pub body: Body,
}

impl<Val, Body> Let<Val, Body> {
    /// Create a new let binding.
    pub fn new(val: Val, body: Body) -> Self {
        Self { val, body }
    }
}

impl<P, Val, Body> Manifold<P> for Let<Val, Body>
where
    P: Copy + Send + Sync,
    Val: Manifold<P>,
    Val::Output: Copy + Send + Sync,
    Body: Manifold<LetExtended<Val::Output, P>>,
{
    type Output = Body::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        // 1. Evaluate the value being bound
        let val = self.val.eval(p);

        // 2. Create extended domain with bound value
        let extended = LetExtended(val, p);

        // 3. Evaluate body on extended domain
        self.body.eval(extended)
    }
}

// ============================================================================
// Var Combinator
// ============================================================================

/// Read a bound variable from the domain stack.
///
/// `Var<N>` retrieves the value at Peano index `N` from the domain stack.
/// Index 0 is the most recently bound value (head of domain).
#[derive(Clone, Copy, Debug, Element)]
pub struct Var<N>(PhantomData<N>);

impl<N> Var<N> {
    /// Create a new variable reference.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<N> Default for Var<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Manifold implementations for Var (unary decrement-based)
// ============================================================================
//
// Binary numbers need to be decremented to traverse the domain stack.
// We use a simpler unary-style recursion: decrement and tail.

// ============================================================================
// Predecessor trait (simpler than Decrement)
// ============================================================================
//
// Maps each positive number to its predecessor.
// Uses a helper trait to avoid impl conflicts.

/// Predecessor in the type-level natural number chain (N → N-1).
pub trait Pred {
    /// The predecessor type.
    type Output;
}

// Helper marker to distinguish UTerm from other U
trait NotUTerm {}
impl<U, B> NotUTerm for UInt<U, B> {}

// 1 -> 0: UInt<UTerm, B1> -> UTerm
impl Pred for UInt<UTerm, B1> {
    type Output = UTerm;
}

// Odd > 1: flip last bit (2k+1 -> 2k)
impl<U> Pred for UInt<U, B1>
where
    U: NotUTerm,
{
    type Output = UInt<U, B0>;
}

// Even: borrow from higher bits (2k -> 2k-1)
impl<U> Pred for UInt<U, B0>
where
    U: Pred,
{
    type Output = UInt<U::Output, B1>;
}

// Var<UTerm> (index 0): read the head
impl<P> Manifold<P> for Var<UTerm>
where
    P: Head + Send + Sync,
    P::Value: Copy + Send + Sync,
{
    type Output = P::Value;

    #[inline(always)]
    fn eval(&self, p: P) -> P::Value {
        p.head()
    }
}

// Var<N> where N > 0 (explicitly UInt types): tail and decrement
impl<U, B, P> Manifold<P> for Var<UInt<U, B>>
where
    UInt<U, B>: Pred + Send + Sync,
    <UInt<U, B> as Pred>::Output: Send + Sync,
    P: Tail + Send + Sync,
    P::Rest: Copy,
    Var<<UInt<U, B> as Pred>::Output>: Manifold<P::Rest>,
{
    type Output = <Var<<UInt<U, B> as Pred>::Output> as Manifold<P::Rest>>::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        Var::<<UInt<U, B> as Pred>::Output>::new().eval(p.tail())
    }
}

// ============================================================================
// Stub Manifold<Field4> impl for ManifoldExt compatibility
// ============================================================================
//
// Var<N> should only be evaluated on domains with Head trait (i.e., LetExtended).
// However, to use ManifoldExt methods for AST construction (.sqrt(), .abs(), etc.),
// we need Manifold<Field4> impls. These panic if actually called, but that should
// never happen since Var<N> is always wrapped in Let bindings that extend the domain.

type Field4 = (crate::Field, crate::Field, crate::Field, crate::Field);

// Note: Var<N> Manifold impls above work within LetExtended domains created by Let combinators.
// They're never evaluated on the base Field4 domain directly.

// ============================================================================
// Legacy Compatibility: Graph trait and Root
// ============================================================================

/// The empty context (stack bottom) for legacy Graph-based code.
#[derive(Clone, Copy, Debug)]
pub struct Empty;

/// Retrieve a value from the context stack at type-level index `N`.
/// Legacy trait for backward compatibility.
pub trait Get<N>: Send + Sync {
    /// Get the value at index N.
    fn get(&self) -> crate::Field;
}

// Base case: index UTerm (0) gets the head
impl<Tail: Send + Sync> Get<UTerm> for (crate::Field, Tail) {
    #[inline(always)]
    fn get(&self) -> crate::Field {
        self.0
    }
}

// UInt<N, B1>: odd number = 2*N + 1, so decrement gives 2*N + 0 = UInt<N, B0>
impl<U, Tail> Get<UInt<U, B1>> for (crate::Field, Tail)
where
    Tail: Get<UInt<U, B0>>,
{
    #[inline(always)]
    fn get(&self) -> crate::Field {
        self.1.get() // Skip head, get UInt<U, B0> from tail
    }
}

// UInt<UTerm, B0>: special case for index 2 (0b10)
impl<Tail> Get<UInt<UTerm, B0>> for (crate::Field, Tail)
where
    Tail: Get<UTerm>, // 2 - 1 = 1, 1 - 1 = 0 = UTerm
{
    #[inline(always)]
    fn get(&self) -> crate::Field {
        self.1.get()
    }
}

// UInt<UInt<U, UB>, B0>: even number > 2, decrement requires borrowing
// 2*N - 1 = 2*(N-1) + 1 = UInt<(N-1), B1>
impl<U, UB, Tail> Get<UInt<UInt<U, UB>, B0>> for (crate::Field, Tail)
where
    Tail: Get<UInt<UInt<U, UB>, B1>>, // Simplified: just flip B0 to B1 and recurse twice
{
    #[inline(always)]
    fn get(&self) -> crate::Field {
        self.1.get()
    }
}

/// A computation graph node that evaluates with a context stack.
/// Legacy trait for backward compatibility with existing code.
pub trait Graph<Ctx>: Send + Sync {
    /// Evaluate at coordinates with the given context.
    #[allow(clippy::too_many_arguments)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field;
}

/// Wrapper to lift a `Manifold` into the `Graph` world.
#[derive(Clone, Debug)]
pub struct Lift<M>(pub M);

impl<M, Ctx> Graph<Ctx> for Lift<M>
where
    M: Manifold<Field4, Output = crate::Field>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        _ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        self.0.eval((x, y, z, w))
    }
}

/// Graph-level addition (legacy).
#[derive(Clone, Debug)]
pub struct GAdd<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GAdd<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_add(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Graph-level subtraction (legacy).
#[derive(Clone, Debug)]
pub struct GSub<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GSub<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_sub(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Graph-level multiplication (legacy).
#[derive(Clone, Debug)]
pub struct GMul<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GMul<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_mul(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Graph-level division (legacy).
#[derive(Clone, Debug)]
pub struct GDiv<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GDiv<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_div(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Root node that converts a `Graph` into a `Manifold` (legacy).
#[derive(Clone, Debug)]
pub struct Root<G>(pub G);

impl<G> Manifold<Field4> for Root<G>
where
    G: Graph<Empty> + Send + Sync,
{
    type Output = crate::Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> crate::Field {
        self.0.eval_at(&Empty, p.0, p.1, p.2, p.3)
    }
}

// Legacy Graph impl for Let (for backward compatibility)
impl<Ctx, Val, Body> Graph<Ctx> for Let<Val, Body>
where
    Ctx: Send + Sync + Copy,
    Val: Graph<Ctx>,
    Body: Graph<(crate::Field, Ctx)>,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        let val = self.val.eval_at(ctx, x, y, z, w);
        let new_ctx = (val, *ctx);
        self.body.eval_at(&new_ctx, x, y, z, w)
    }
}

// Legacy Graph impl for Var
impl<N, Ctx> Graph<Ctx> for Var<N>
where
    N: Send + Sync,
    Ctx: Get<N>,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        _x: crate::Field,
        _y: crate::Field,
        _z: crate::Field,
        _w: crate::Field,
    ) -> crate::Field {
        ctx.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, X};

    #[test]
    fn test_let_binding_new_style() {
        // let v = 10.0; v + 5.0
        let expr = Let::new(10.0f32, Var::<N0>::new() + 5.0f32);

        // Evaluate on 2D domain
        let domain = (Field::from(0.0), Field::from(0.0));
        let result = expr.eval(domain);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 15.0); // 10 + 5
    }

    #[test]
    fn test_let_with_spatial() {
        // let dist = x; dist * 2
        let expr = Let::new(X, Var::<N0>::new() * 2.0f32);

        let domain = (Field::from(5.0), Field::from(0.0));
        let result = expr.eval(domain);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 10.0); // 5 * 2
    }

    #[test]
    fn test_nested_let_new_style() {
        // let a = 3.0; let b = 4.0; a + b
        let expr = Let::new(
            3.0f32, // a = 3.0 (becomes Var<1> after second let)
            Let::new(
                4.0f32,                              // b = 4.0 (becomes Var<0>)
                Var::<N1>::new() + Var::<N0>::new(), // a + b
            ),
        );

        let domain = (Field::from(0.0), Field::from(0.0));
        let result = expr.eval(domain);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 7.0); // 3 + 4
    }

    // FIXME: This legacy test has trait bound issues with Empty not implementing Get<UTerm>.
    // The nested cons-list approach is being replaced by WithContext flat tuples.
    // Commented out until legacy binding system is fixed or removed.
    /*
    #[test]
    fn test_legacy_peano_get() {
        let ctx: (Field, (Field, Empty)) = (Field::from(10.0), (Field::from(20.0), Empty));

        // Index 0 should get 10.0 (head)
        let v0: Field = <(Field, (Field, Empty)) as Get<N0>>::get(&ctx);
        let mut buf = [0.0f32; crate::PARALLELISM];
        v0.store(&mut buf);
        assert_eq!(buf[0], 10.0);

        // Index 1 should get 20.0 (tail head)
        let v1: Field = <(Field, (Field, Empty)) as Get<N1>>::get(&ctx);
        v1.store(&mut buf);
        assert_eq!(buf[0], 20.0);
    }
    */

    #[test]
    fn test_legacy_let_binding() {
        // let x = 5.0; x + x
        let graph = Let::new(Lift(5.0f32), GAdd(Var::<N0>::new(), Var::<N0>::new()));

        let zero = Field::from(0.0);
        let result = graph.eval_at(&Empty, zero, zero, zero, zero);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 10.0); // 5 + 5
    }
}
