//! # Computed and ManifoldBind Combinators
//!
//! Creates manifolds from closures and binds manifold parameters.
//!
//! ## Computed
//!
//! Wraps a closure `Fn(P) -> Out` as a `Manifold<P>`.
//! Useful for deferred computation that depends on the evaluation point.
//!
//! ## ManifoldBind
//!
//! Binds a manifold parameter into an expression context.
//! At eval time, evaluates the manifold and makes the result available
//! to the body expression via the context system.
//!
//! This is crucial for kernel macros with mixed manifold+scalar parameters.
//! Without it, type inference fails because the closure return type doesn't
//! carry the manifold type information.
//!
//! ## Example
//!
//! ```ignore
//! // kernel!(|inner: kernel, r: f32| inner - r) generates:
//! move |inner, r: f32| {
//!     let body = WithContext::new(([Field::from(r)],), expr);
//!     ManifoldBind::new(inner, body)
//! }
//! // ManifoldBind<M, WithContext<...>> carries M in its type, helping inference.
//! ```

use crate::Field;
use crate::Manifold;

type Field4 = (Field, Field, Field, Field);

/// A manifold created from a closure.
///
/// This combinator wraps a closure `Fn(P) -> Out` and implements `Manifold<P>`.
/// It's useful when you need to defer computation until evaluation time.
#[derive(Clone)]
pub struct Computed<F>(pub F);

impl<F> Computed<F> {
    /// Create a new Computed combinator from a closure.
    #[inline(always)]
    pub const fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F: core::fmt::Debug> core::fmt::Debug for Computed<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Computed").field(&self.0).finish()
    }
}

// Implement Copy when the closure is Copy
impl<F: Copy> Copy for Computed<F> {}

// Generic impl for any domain P
impl<P, F, Out> Manifold<P> for Computed<F>
where
    P: Copy + Send + Sync,
    F: Fn(P) -> Out + Send + Sync,
    Out: Send + Sync,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Out {
        (self.0)(p)
    }
}

// ============================================================================
// ManifoldBind - Bind a manifold parameter into context
// ============================================================================

/// Binds a manifold parameter into an expression context.
///
/// At eval time:
/// 1. Evaluates the manifold at the current point
/// 2. Prepends the result to the context for the body expression
///
/// This combinator carries the manifold type `M` in its type signature,
/// which helps Rust infer types for closure parameters in kernel macros.
///
/// ## Type Structure
///
/// `ManifoldBind<M, Body>` where:
/// - `M`: The manifold to evaluate (e.g., an inner SDF)
/// - `Body`: The expression body that uses the evaluated result
#[derive(Clone, Debug)]
pub struct ManifoldBind<M, Body> {
    /// The manifold to evaluate at each point.
    pub manifold: M,
    /// The body expression that receives the evaluated manifold result.
    pub body: Body,
}

impl<M, Body> ManifoldBind<M, Body> {
    /// Create a new ManifoldBind.
    #[inline(always)]
    pub const fn new(manifold: M, body: Body) -> Self {
        Self { manifold, body }
    }
}

impl<M: Copy, Body: Copy> Copy for ManifoldBind<M, Body> {}

// ManifoldBind with single-array WithContext: M evaluates to Field,
// body receives (([T; N], [Field; 1]), P) - combined scalar context with manifold result
impl<T, const N: usize, M, Body> Manifold<Field4>
    for ManifoldBind<M, super::context::WithContext<([T; N],), Body>>
where
    T: Copy + Send + Sync,
    M: Manifold<Field4, Output = Field>,
    Body: Manifold<(([T; N], [Field; 1]), Field4), Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let m_result = self.manifold.eval(p);
        // Combine: extract the inner array from the 1-tuple, add manifold result array
        let combined_ctx = (self.body.ctx.0, [m_result]);
        self.body.body.eval((combined_ctx, p))
    }
}

// ManifoldBind with empty WithContext (manifold-only kernel): M evaluates to Field,
// body receives (([Field; 1],), P) - just the manifold result
impl<M, Body> Manifold<Field4> for ManifoldBind<M, super::context::WithContext<(), Body>>
where
    M: Manifold<Field4, Output = Field>,
    Body: Manifold<(([Field; 1],), Field4), Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let m_result = self.manifold.eval(p);
        self.body.body.eval((([m_result],), p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variables::X;

    #[test]
    fn test_computed_basic() {
        // Use an expression tree (X) that gets evaluated
        let expr = X;
        let computed = Computed::new(move |p: Field4| -> Field { expr.eval(p) });

        let p = (
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let result: Field = computed.eval(p);

        // X extracts the first component = 3.0
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert!((buf[0] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_computed_captures() {
        let scale = 2.0f32;
        // Use expression tree X * scale
        let expr = X * scale;
        let computed = Computed::new(move |p: Field4| -> Field { expr.eval(p) });

        let p = (
            Field::from(5.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let result: Field = computed.eval(p);

        // X * 2 = 5 * 2 = 10
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert!((buf[0] - 10.0).abs() < 0.001);
    }
}
