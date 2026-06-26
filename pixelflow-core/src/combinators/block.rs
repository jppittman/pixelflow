//! # Block Combinator
//!
//! Forces a scheduling barrier by preventing inlining.
//!
//! ## Purpose
//!
//! Deep expression trees can cause register pressure issues: the compiler
//! keeps many intermediate values in registers simultaneously, leading to
//! spills when registers run out. The `Block` combinator inserts an
//! optimization barrier that:
//!
//! 1. Forces the inner expression to be fully evaluated
//! 2. Spills the result to memory (via the function call ABI)
//! 3. Allows the compiler to reuse registers for subsequent computation
//!
//! ## When to Use
//!
//! - Wide expressions with many parallel subexpressions
//! - Deep nesting where intermediate values accumulate
//! - Before expensive operations that need many registers
//!
//! ## Example
//!
//! ```ignore
//! // Without Block: all intermediates compete for registers
//! let wide = (a + b) * (c + d) * (e + f) * (g + h);
//!
//! // With Block: force evaluation boundaries
//! let left = Block::new((a + b) * (c + d));
//! let right = Block::new((e + f) * (g + h));
//! let result = left * right;
//! ```
//!
//! ## Performance Notes
//!
//! - Adds function call overhead (~1-3 cycles)
//! - May improve overall performance if it prevents excessive spilling
//! - Profile before and after to verify benefit

use crate::Manifold;

/// Scheduling barrier that prevents inlining.
///
/// Wraps a manifold and evaluates it through a non-inlined function,
/// forcing register spills at this boundary.
#[derive(Clone, Copy, Debug)]
pub struct Block<M>(pub M);

impl<M> Block<M> {
    /// Create a new Block combinator.
    #[inline(always)]
    pub fn new(inner: M) -> Self {
        Self(inner)
    }
}

// The key: #[inline(never)] forces a call boundary
#[inline(never)]
fn block_eval<T>(val: T) -> T {
    val
}

impl<I, M> Manifold<(I, I, I, I)> for Block<M>
where
    I: crate::numeric::Computational,
    M: Manifold<(I, I, I, I)>,
    M::Output: Copy,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, p: (I, I, I, I)) -> Self::Output {
        // Evaluate inner, then pass through non-inlined barrier
        block_eval(self.0.eval(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;
    use crate::variables::{X, Y};

    #[test]
    fn test_block_compiles_and_runs() {
        let inner = X + Y;
        let blocked = Block::new(inner);

        let p = (
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        // Just verify it runs without panic
        let _direct = inner.eval(p);
        let _through_block = blocked.eval(p);
    }

    #[test]
    fn test_block_nested() {
        // Block of a Block should still work
        let inner = X * Y;
        let blocked = Block::new(Block::new(inner));

        let p = (
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let _result = blocked.eval(p);
    }

    #[test]
    fn test_block_with_complex_expr() {
        // More complex expression to verify register pressure scenario
        let expr = (X + Y) * (X - Y);
        let blocked = Block::new(expr);

        let p = (
            Field::from(5.0),
            Field::from(3.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let _result = blocked.eval(p);
        // (5+3) * (5-3) = 8 * 2 = 16
    }
}
