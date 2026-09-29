//! What the language builds from its primitives: one definition each.
//!
//! `fract`, `hypot` and `clamp` are library, not primitives: no instruction
//! computes any of them, so an IR node for one bought nothing but a
//! decomposition every backend and the e-graph wrote for itself (and
//! `Clamp`'s copies disagreed on `lo > hi`). A derivative is a primitive,
//! `Dwrt`, whose axis rides as a `Const` operand — the encoding the
//! e-graph's `ChainRule` and `passes::lower_dwrt` read back.
//!
//! Each is written once, here, over [`Terms`]: wherever a term is built
//! node by node. Both front ends build through the same definition —
//! `kernel!`'s lowering over an [`ExprArena`], and [`Kernel`]'s methods over
//! kernel values — so `X.fract()` in the syntax and `Kernel::x().fract()`
//! are one program (pinned by canonical key in
//! `pixelflow-compiler/tests/the_library_is_the_builders.rs`), and a change
//! to a definition reaches both.
//!
//! Why generic, and not an arena function a `Kernel` round-trips through:
//! a `Kernel` builds one node at a time, each operand copied where it is
//! combined, and its arena is laid out in that order. Spliced into one
//! arena first and combined after, `hypot`'s second operand would land
//! before its first product — one term, laid out differently, and a
//! compile is not promised to be blind to the layout. Over [`Terms`] each
//! representation builds the definition its own way, and a `Kernel`'s is
//! the node sequence it always was.
//!
//! [`Kernel`]: crate::Kernel

use crate::arena::{Axis, ExprArena, ExprId};
use crate::kind::OpKind;

/// Keeps [`Terms`] the IR's: a composite is defined here, over the
/// representations the IR has, and a third is added here too.
pub(crate) mod sealed {
    /// Implemented by each representation [`Terms`](super::Terms) builds.
    pub trait Sealed {}
}

/// Somewhere a term is built node by node: an [`ExprArena`], naming a term
/// by its [`ExprId`], or kernel values, naming one by the `Kernel` it is.
///
/// Sealed: the definitions in this module are written against it so that
/// each has one body, not so that a caller can build them somewhere new.
pub trait Terms: sealed::Sealed {
    /// How a term built here is named.
    type Term: Clone;

    /// The literal `value`.
    fn constant(&mut self, value: f32) -> Self::Term;

    /// `op(operand)`.
    fn unary(&mut self, op: OpKind, operand: Self::Term) -> Self::Term;

    /// `op(a, b)`, for `[a, b]`.
    fn binary(&mut self, op: OpKind, operands: [Self::Term; 2]) -> Self::Term;
}

impl sealed::Sealed for ExprArena {}

impl Terms for ExprArena {
    type Term = ExprId;

    fn constant(&mut self, value: f32) -> ExprId {
        self.push_const(value)
    }

    fn unary(&mut self, op: OpKind, operand: ExprId) -> ExprId {
        self.push_unary(op, operand)
    }

    fn binary(&mut self, op: OpKind, [a, b]: [ExprId; 2]) -> ExprId {
        self.push_binary(op, a, b)
    }
}

/// `x − ⌊x⌋`: the fractional part, in `[0, 1)` for a finite `x`.
pub fn fract<T: Terms>(terms: &mut T, x: T::Term) -> T::Term {
    let floor = terms.unary(OpKind::Floor, x.clone());
    terms.binary(OpKind::Sub, [x, floor])
}

/// `√(x² + y²)`: the length of `(x, y)`.
pub fn hypot<T: Terms>(terms: &mut T, [x, y]: [T::Term; 2]) -> T::Term {
    let xx = terms.binary(OpKind::Mul, [x.clone(), x]);
    let yy = terms.binary(OpKind::Mul, [y.clone(), y]);
    let sum = terms.binary(OpKind::Add, [xx, yy]);
    terms.unary(OpKind::Sqrt, sum)
}

/// `min(max(x, lo), hi)`: `x` held to `[lo, hi]`. Bounds with `lo > hi`
/// give `hi`, as the composition says.
pub fn clamp<T: Terms>(terms: &mut T, x: T::Term, [lo, hi]: [T::Term; 2]) -> T::Term {
    let floored = terms.binary(OpKind::Max, [x, lo]);
    terms.binary(OpKind::Min, [floored, hi])
}

/// `∂e/∂axis`, resolved symbolically when the kernel is baked: `Dwrt(e,
/// axis)`, the axis's `Var` index as a `Const` operand.
///
/// Left unresolved on purpose: a warp applied afterwards substitutes into
/// `e`'s coordinates, and the chain rule then differentiates the warped
/// function (`pixelflow-compiler/tests/derivative_under_warp.rs`).
pub fn derivative<T: Terms>(terms: &mut T, e: T::Term, axis: Axis) -> T::Term {
    let axis = terms.constant(f32::from(axis.var()));
    terms.binary(OpKind::Dwrt, [e, axis])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ExprNode;

    /// Each composite is the nodes its doc says, over an arena.
    #[test]
    fn each_composite_is_the_term_it_denotes() {
        let mut a = ExprArena::new();
        let (x, y) = (a.push_var(Axis::X.var()), a.push_var(Axis::Y.var()));

        let f = fract(&mut a, x);
        let ExprNode::Binary(OpKind::Sub, minuend, floor) = a.node(f) else {
            panic!("x − ⌊x⌋, got {}", a.display(f));
        };
        assert_eq!(minuend, x);
        assert_eq!(a.node(floor), ExprNode::Unary(OpKind::Floor, x));

        let h = hypot(&mut a, [x, y]);
        let ExprNode::Unary(OpKind::Sqrt, sum) = a.node(h) else {
            panic!("√(x² + y²), got {}", a.display(h));
        };
        let ExprNode::Binary(OpKind::Add, xx, yy) = a.node(sum) else {
            panic!("x² + y², got {}", a.display(sum));
        };
        assert_eq!(a.node(xx), ExprNode::Binary(OpKind::Mul, x, x));
        assert_eq!(a.node(yy), ExprNode::Binary(OpKind::Mul, y, y));

        let (lo, hi) = (a.push_const(0.0), a.push_const(1.0));
        let c = clamp(&mut a, x, [lo, hi]);
        let ExprNode::Binary(OpKind::Min, floored, upper) = a.node(c) else {
            panic!("min(max(x, lo), hi), got {}", a.display(c));
        };
        assert_eq!(upper, hi);
        assert_eq!(a.node(floored), ExprNode::Binary(OpKind::Max, x, lo));
    }

    /// A derivative's axis is its `Var` index as a `Const`, the encoding
    /// `passes::lower_dwrt` reads back: it differentiates `x·y` by `y`.
    #[test]
    fn a_derivative_names_its_axis_as_lower_dwrt_reads_it() {
        let mut a = ExprArena::new();
        let (x, y) = (a.push_var(Axis::X.var()), a.push_var(Axis::Y.var()));
        let xy = a.push_binary(OpKind::Mul, x, y);
        let d = derivative(&mut a, xy, Axis::Y);
        let ExprNode::Binary(OpKind::Dwrt, e, axis) = a.node(d) else {
            panic!("Dwrt(e, axis), got {}", a.display(d));
        };
        assert_eq!(e, xy);
        assert!(matches!(a.node(axis), ExprNode::Const(v) if v == 1.0));
        let lowered = crate::passes::lower_dwrt(&mut a, d).expect("lowers");
        let mut stack = alloc::vec![lowered];
        while let Some(id) = stack.pop() {
            assert_ne!(a.kind(id), OpKind::Dwrt, "{}", a.display(lowered));
            stack.extend(a.children(id));
        }
    }
}
