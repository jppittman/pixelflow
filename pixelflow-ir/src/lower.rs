//! Lowering: the compilation boundary consumers never cross.
//!
//! [`Lower`] maps a manifold value to its expression in an [`ExprArena`] —
//! the functor from the combinator language to the IR, defined once per
//! generator beside each combinator's `Manifold` impl (see
//! docs/designs/2026-07-23-lower-realize-boundary.md). Consumers never call
//! or implement it; it rides as a bound (like `Send`) behind verbs such as
//! `Lattice::realize`, which lowers, JIT-compiles, and falls back to generic
//! evaluation when any node declines.

use crate::arena::{ExprArena, ExprId};
use alloc::vec::Vec;

/// Scope state threaded through a lowering pass: the stack of `Let`-bound
/// values, innermost last. `Var<N>` resolves De Bruijn index `N` against it.
#[derive(Default)]
pub struct LowerEnv {
    lets: Vec<ExprId>,
}

impl LowerEnv {
    /// Enter a `Let` body with `value` bound.
    pub fn bind(&mut self, value: ExprId) {
        self.lets.push(value);
    }

    /// Leave a `Let` body.
    pub fn unbind(&mut self) {
        let popped = self.lets.pop();
        debug_assert!(popped.is_some(), "LowerEnv::unbind without a bind");
    }

    /// Resolve De Bruijn index `n` (0 = innermost binding).
    #[must_use]
    pub fn resolve(&self, n: usize) -> Option<ExprId> {
        let len = self.lets.len();
        if n < len { Some(self.lets[len - 1 - n]) } else { None }
    }
}

/// A manifold that can emit its expression into an arena.
///
/// Returns `None` when the value has no IR form (an opaque evaluator, a
/// non-uniform runtime vector, a context shape the lowering does not model
/// yet) — the caller then falls back to generic evaluation. Implementations
/// live beside the corresponding `Manifold` impls; a type without either has
/// no place in the language.
pub trait Lower {
    /// Emit this manifold's expression into `arena`, resolving `Let`
    /// bindings through `env`.
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId>;
}

/// A bare scalar is a constant manifold.
impl Lower for f32 {
    fn lower(&self, arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        Some(arena.push_const(*self))
    }
}
