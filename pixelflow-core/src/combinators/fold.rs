//! Variable-arity sum: the runtime-length fold the fixed-arity combinator
//! tree cannot express.
//!
//! The ZST op combinators (`Add`, `Mul`, …) are fixed-arity by type. A glyph
//! outline or a line of text is a *runtime* list of coverage manifolds, so it
//! needs a combinator that owns a slice. [`Sum`] is that combinator — the
//! monoid fold over `+`, with both a `Manifold` impl (the evaluation functor)
//! and a `Lower` impl (the IR functor: fold children into `Add` nodes). It is
//! the one primitive graphics needs to express scene-graph sums as pure
//! compositions rather than hand-written evaluators.

use alloc::sync::Arc;

use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::{Lower, LowerEnv, OpKind};

use crate::{Field, Manifold};

/// The monoid sum of a runtime list of coverage manifolds: `Σ mᵢ`, with the
/// empty sum evaluating to `0`. Every summand produces `Field` in every
/// domain, so the accumulation is plain `Field` math regardless of the
/// coordinate type.
#[derive(Clone, Debug)]
pub struct Sum<M>(pub Arc<[M]>);

impl<M> Sum<M> {
    /// Build a sum from any iterator of summands.
    pub fn new(summands: impl IntoIterator<Item = M>) -> Self {
        Self(summands.into_iter().collect())
    }
}

impl<M> FromIterator<M> for Sum<M> {
    fn from_iter<I: IntoIterator<Item = M>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<P, M> Manifold<P> for Sum<M>
where
    P: Copy + Send + Sync,
    M: Manifold<P, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: P) -> Field {
        if self.0.len() == 1 {
            return self.0[0].eval(p);
        }
        let zero = Field::from(0.0);
        self.0.iter().fold(zero, |acc, m| {
            let val = m.eval(p);
            (acc + val).eval(p)
        })
    }
}

/// Fold the lowered summands into a left-leaning `Add` chain. A summand that
/// declines (`None`) collapses the whole sum, matching the all-or-nothing
/// contract of `Lower`: a sum realizes only if every term does.
impl<M: Lower> Lower for Sum<M> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let mut acc: Option<ExprId> = None;
        for m in self.0.iter() {
            let id = m.lower(arena, env)?;
            acc = Some(match acc {
                None => id,
                Some(a) => arena.push_binary(OpKind::Add, a, id),
            });
        }
        Some(acc.unwrap_or_else(|| arena.push_const(0.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variables::X;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval::eval_scalar;

    #[test]
    fn empty_sum_is_zero() {
        let s: Sum<X> = Sum::new(core::iter::empty());
        let mut arena = ExprArena::new();
        let mut env = LowerEnv::default();
        let root = s.lower(&mut arena, &mut env).unwrap();
        assert_eq!(
            eval_scalar(&arena, root, &[9.0, 9.0, 0.0, 0.0], &BindingTable::empty()),
            0.0
        );
    }

    #[test]
    fn sum_lowers_and_matches_eval() {
        use crate::ops::binary::Add;
        // (X+1) + (X+2) + (X+3) = 3X + 6. Homogeneous summand type — the
        // constraint that forces heterogeneous scene-graph segments to share
        // a type (an enum) before they can be summed.
        let s = Sum::new([Add(X, 1.0f32), Add(X, 2.0f32), Add(X, 3.0f32)]);
        let mut arena = ExprArena::new();
        let mut env = LowerEnv::default();
        let root = s.lower(&mut arena, &mut env).unwrap();
        let got = eval_scalar(&arena, root, &[3.0, 0.0, 0.0, 0.0], &BindingTable::empty());
        assert_eq!(got, 3.0 * 3.0 + 6.0);
    }
}
