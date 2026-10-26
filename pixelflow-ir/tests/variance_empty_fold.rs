//! A fold over nothing varies with nothing.
//!
//! `⊕_{i∈∅} body` is the monoid's identity (`KnownRange::is_empty`): a constant,
//! whatever its body reads. By the variance law (if `b ∉ var(v)` then `⟦v⟧` is
//! constant along `b`) the exact set for a constant is empty, so the fold adds
//! nothing to its readers. Read off the body (`deps(body) \ {i}`), it would
//! name every variable the never-evaluated body reads, and a reader such as
//! `y + ·` would inherit them.

use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::variance::{Variance, compute_arena_variance};
use pixelflow_ir::{ExprArena, OpKind};

/// A fold over nothing is a constant and adds nothing to its reader; a fold
/// over something still frees only its own index.
#[test]
fn a_fold_over_nothing_varies_with_nothing() {
    let i = Binder::from_slot(0).expect("slot 0 exists");
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let index = arena.push_var(i.var());
    let body = arena.push_binary(OpKind::Mul, index, x);
    let fold = Fold::new(Monoid::PRODUCT, i);
    let none = arena.push_reduce(fold, Bound::Lit(0), Bound::Lit(0), body);
    let some = arena.push_reduce(fold, Bound::Lit(0), Bound::Lit(3), body);
    let read = arena.push_binary(OpKind::Add, y, none);

    let variance = compute_arena_variance(&arena);
    assert_eq!(variance[none.0 as usize], Variance::CONST);
    assert_eq!(variance[read.0 as usize], Variance::Y);
    assert_eq!(variance[some.0 as usize], Variance::X);
}
