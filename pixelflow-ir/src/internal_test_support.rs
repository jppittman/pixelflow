//! Fixture builders for `pixelflow-ir`'s own bench/integration-test binaries,
//! and for the one `pixelflow-search` unit test that needs a `Dag<ExprData>`
//! built directly rather than round-tripped through `ExprArena`.
//!
//! `benches/` and `tests/` are separate Cargo targets that link against this
//! crate the same way an external crate would — they see only `pub` items,
//! never `pub(crate)` ones — so they cannot call `dag::Builder`/`dag::Id`
//! directly now that those are sealed to this crate. This module is the one
//! place that still can: it is `pub` (a bench/test binary has no other way
//! in), `#[doc(hidden)]` (it is not API — nothing here is meant to be
//! discovered, only called by the three fixed call sites above), and its
//! functions return only the consumption vocabulary (`Rooted<T>`), never an
//! `Id` or a `Builder`. Calling one of these teaches a caller nothing about
//! how a `Dag` is built.

use crate::dag::{Builder, Rooted};
use crate::expr::{ExprBuilderExt, ExprData};
use crate::kind::OpKind;

/// The shared node payload for the two generic-`Dag` fixtures below. Its
/// shape is arbitrary — nothing reads it as anything but an opaque `T`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Payload {
    Var(u8),
    Const(u32),
    Op(OpKind),
}

/// `benches/dag_vs_arena.rs`'s `build/no_sharing` case: `((x*s0 + y*s0) +
/// (x*s1 + y*s1) + ...)`, N terms, no two subtrees equal — the worst case
/// for consing. `intern` selects `Builder::intern` vs `Builder::push_unique`
/// so the bench can compare them.
#[must_use]
pub fn dag_no_sharing(n: u32, intern: bool) -> Rooted<Payload> {
    let mut b: Builder<Payload> = Builder::with_capacity(6 * n as usize + 4, 0);
    let push = |b: &mut Builder<Payload>, v: Payload, kids: &[_]| {
        if intern {
            b.intern(v, kids)
        } else {
            b.push_unique(v, kids)
        }
    };
    let x = push(&mut b, Payload::Var(0), &[]);
    let y = push(&mut b, Payload::Var(1), &[]);
    let mut acc = push(&mut b, Payload::Const(0.0f32.to_bits()), &[]);
    for i in 0..n {
        let s = push(&mut b, Payload::Const((i as f32 + 1.0).to_bits()), &[]);
        let xs = push(&mut b, Payload::Op(OpKind::Mul), &[x, s]);
        let ys = push(&mut b, Payload::Op(OpKind::Mul), &[y, s]);
        let term = push(&mut b, Payload::Op(OpKind::Add), &[xs, ys]);
        acc = push(&mut b, Payload::Op(OpKind::Add), &[acc, term]);
    }
    b.finish(&[acc])
}

/// `benches/dag_vs_arena.rs`'s `build/naive_sharing` and `traverse` cases:
/// `d0 + d1 + ... + d(N-1)`, each `di` a freshly re-pushed copy of the same
/// 4-node subtree `(x*c + y*c)` — the caller-doesn't-track-sharing case
/// `intern` exists to collapse.
#[must_use]
pub fn dag_naive_sharing(n: u32, intern: bool) -> Rooted<Payload> {
    let mut b: Builder<Payload> = Builder::new();
    let push = |b: &mut Builder<Payload>, v: Payload, kids: &[_]| {
        if intern {
            b.intern(v, kids)
        } else {
            b.push_unique(v, kids)
        }
    };
    let mut acc = push(&mut b, Payload::Const(0.0f32.to_bits()), &[]);
    for _ in 0..n {
        let x = push(&mut b, Payload::Var(0), &[]);
        let y = push(&mut b, Payload::Var(1), &[]);
        let c = push(&mut b, Payload::Const(1.5f32.to_bits()), &[]);
        let xs = push(&mut b, Payload::Op(OpKind::Mul), &[x, c]);
        let ys = push(&mut b, Payload::Op(OpKind::Mul), &[y, c]);
        let d = push(&mut b, Payload::Op(OpKind::Add), &[xs, ys]);
        acc = push(&mut b, Payload::Op(OpKind::Add), &[acc, d]);
    }
    b.finish(&[acc])
}

/// `tests/dag_scratch_allocation.rs`'s fixed 4-node graph: `x`, `y`,
/// `add = x + y`, `root = add * x`.
#[must_use]
pub fn scratch_allocation_fixture() -> Rooted<Payload> {
    let mut b: Builder<Payload> = Builder::new();
    let x = b.intern(Payload::Var(0), &[]);
    let y = b.intern(Payload::Var(1), &[]);
    let add = b.intern(Payload::Op(OpKind::Add), &[x, y]);
    let root = b.intern(Payload::Op(OpKind::Mul), &[add, x]);
    b.finish(&[root])
}

/// `pixelflow-search/src/egraph/template.rs`'s `dag_builder_template_rewrite`
/// test: `(lhs, rhs) = (x - y, x + (-y))`, built directly via `Builder`
/// rather than derived from an `ExprArena` — the thing that test exists to
/// exercise, so this has to construct it the same way, not delegate to
/// `from_arena`.
#[must_use]
pub fn template_rewrite_sub_fixture() -> Rooted<ExprData> {
    let mut b: Builder<ExprData> = Builder::new();
    let v0 = b.push_var(0);
    let v1 = b.push_var(1);
    let lhs = b.push_binary(OpKind::Sub, v0, v1);
    let neg_v1 = b.push_unary(OpKind::Neg, v1);
    let rhs = b.push_binary(OpKind::Add, v0, neg_v1);
    b.finish(&[lhs, rhs])
}
