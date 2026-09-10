//! Fixture builders for `pixelflow-ir`'s own integration-test binaries, and
//! for the one `pixelflow-search` unit test that needs a `Dag<ExprData>`
//! built directly rather than through [`ExprBuilder`](crate::expr::ExprBuilder).
//!
//! `tests/` is a separate Cargo target that links against this crate the same
//! way an external crate would — it sees only `pub` items, never `pub(crate)`
//! ones — so it cannot call `dag::Builder`/`dag::Id` directly. This module is
//! the one place that still can: it is `pub` (a test binary has no other way
//! in), `#[doc(hidden)]` (it is not API — nothing here is meant to be
//! discovered, only called by the fixed call sites above), and its functions
//! return only the consumption vocabulary (`Rooted<T>`), never an `Id` or a
//! `Builder`. Calling one of these teaches a caller nothing about how a `Dag`
//! is built.

use crate::dag::{Builder, Rooted};
use crate::expr::{ExprBuilderExt, ExprData};
use crate::kind::OpKind;

/// The node payload for the generic-`Dag` fixture below. Its shape is
/// arbitrary — nothing reads it as anything but an opaque `T`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Payload {
    Var(u8),
    Const(u32),
    Op(OpKind),
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
/// rather than through `ExprBuilder` — the thing that test exists to
/// exercise, so this has to construct it the same way.
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
