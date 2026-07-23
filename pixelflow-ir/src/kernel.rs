//! `Kernel` — the language's runtime value.
//!
//! A `Kernel` is a handle to an expression fragment: an [`ExprArena`] plus its
//! root. It is the value the front end (the `kernel!` macro) produces and the
//! thing consumers compose — `sum`, `at`, `select`, arithmetic — with the
//! arena hidden entirely behind the methods. This is the "JIT-first" surface:
//! programs are built as `Kernel` values (our own AST), type-checked and
//! monomorphized by our codegen at `compile` time, never encoded in Rust's
//! type system.
//!
//! Composition is arena splicing: every method clones the receiver's arena,
//! splices the operands in (DAG-preserving), and appends the new node. Values
//! are immutable and cheaply cloned (`Arc`); the deep copy happens only when a
//! new node is built, which is construction/bake time, not per pixel.

use alloc::sync::Arc;

use crate::arena::{ExprArena, ExprId};
use crate::kind::OpKind;

/// A composed expression fragment: the front-end value.
#[derive(Clone)]
pub struct Kernel {
    inner: Arc<KernelData>,
}

struct KernelData {
    arena: ExprArena,
    root: ExprId,
}

impl Kernel {
    fn wrap(arena: ExprArena, root: ExprId) -> Self {
        Self {
            inner: Arc::new(KernelData { arena, root }),
        }
    }

    // ─────────────────────────── leaves ───────────────────────────

    /// The X coordinate.
    #[must_use]
    pub fn x() -> Self {
        Self::coord(0)
    }
    /// The Y coordinate.
    #[must_use]
    pub fn y() -> Self {
        Self::coord(1)
    }
    /// The Z coordinate.
    #[must_use]
    pub fn z() -> Self {
        Self::coord(2)
    }
    /// The W coordinate.
    #[must_use]
    pub fn w() -> Self {
        Self::coord(3)
    }

    fn coord(i: u8) -> Self {
        let mut a = ExprArena::new();
        let r = a.push_var(i);
        Self::wrap(a, r)
    }

    /// A constant.
    #[must_use]
    pub fn constant(v: f32) -> Self {
        let mut a = ExprArena::new();
        let r = a.push_const(v);
        Self::wrap(a, r)
    }

    /// Adopt an already-built fragment — the `kernel!` macro's entry point.
    #[must_use]
    pub fn from_parts(arena: ExprArena, root: ExprId) -> Self {
        Self::wrap(arena, root)
    }

    // ───────────────────── the builder seam ───────────────────────

    /// Apply a unary node.
    fn map(&self, op: OpKind) -> Self {
        let mut arena = self.inner.arena.clone();
        let root = arena.push_unary(op, self.inner.root);
        Self::wrap(arena, root)
    }

    /// Apply a binary node with `self` on the left and `rhs` spliced in.
    fn combine(&self, rhs: &Kernel, op: OpKind) -> Self {
        let mut arena = self.inner.arena.clone();
        let rhs_root = arena.splice(&rhs.inner.arena, rhs.inner.root);
        let root = arena.push_binary(op, self.inner.root, rhs_root);
        Self::wrap(arena, root)
    }

    /// Apply a ternary node with `self` first and `b`, `c` spliced in.
    fn combine3(&self, b: &Kernel, c: &Kernel, op: OpKind) -> Self {
        let mut arena = self.inner.arena.clone();
        let b_root = arena.splice(&b.inner.arena, b.inner.root);
        let c_root = arena.splice(&c.inner.arena, c.inner.root);
        let root = arena.push_ternary(op, self.inner.root, b_root, c_root);
        Self::wrap(arena, root)
    }

    // ───────────────────────── arithmetic ─────────────────────────

    /// `self + rhs`.
    #[must_use]
    pub fn add(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Add)
    }
    /// `self - rhs`.
    #[must_use]
    pub fn sub(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Sub)
    }
    /// `self * rhs`.
    #[must_use]
    pub fn mul(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Mul)
    }
    /// `self / rhs`.
    #[must_use]
    pub fn div(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Div)
    }
    /// `min(self, rhs)`.
    #[must_use]
    pub fn min(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Min)
    }
    /// `max(self, rhs)`.
    #[must_use]
    pub fn max(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Max)
    }

    /// `-self`.
    #[must_use]
    pub fn neg(&self) -> Self {
        self.map(OpKind::Neg)
    }
    /// `|self|`.
    #[must_use]
    pub fn abs(&self) -> Self {
        self.map(OpKind::Abs)
    }
    /// `√self`.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        self.map(OpKind::Sqrt)
    }
    /// `1/self`.
    #[must_use]
    pub fn recip(&self) -> Self {
        self.map(OpKind::Recip)
    }

    // ─────────────────────── comparisons / masks ──────────────────

    /// `self < rhs` (a mask).
    #[must_use]
    pub fn lt(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Lt)
    }
    /// `self <= rhs`.
    #[must_use]
    pub fn le(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Le)
    }
    /// `self > rhs`.
    #[must_use]
    pub fn gt(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Gt)
    }
    /// `self >= rhs`.
    #[must_use]
    pub fn ge(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Ge)
    }
    /// Mask AND (canonical masks in both tiers).
    #[must_use]
    pub fn and(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::BitAnd)
    }
    /// Mask OR.
    #[must_use]
    pub fn or(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::BitOr)
    }

    // ─────────────────────────── control ──────────────────────────

    /// `self ? if_true : if_false` — `self` is the mask.
    #[must_use]
    pub fn select(&self, if_true: &Kernel, if_false: &Kernel) -> Self {
        self.combine3(if_true, if_false, OpKind::Select)
    }
    /// `clamp(self, lo, hi)`.
    #[must_use]
    pub fn clamp(&self, lo: &Kernel, hi: &Kernel) -> Self {
        self.combine3(lo, hi, OpKind::Clamp)
    }

    // ───────────────────────── composition ────────────────────────

    /// `Σ kernels`, empty summing to `0` — the variadic monoid fold the
    /// fixed-arity operators cannot express (glyph outlines, text runs).
    #[must_use]
    pub fn sum(kernels: &[Kernel]) -> Self {
        match kernels.split_first() {
            None => Self::constant(0.0),
            Some((head, tail)) => tail.iter().fold(head.clone(), |acc, k| acc.add(k)),
        }
    }

    /// Sample `self` at warped coordinates — contramap / `.at()`. Each of
    /// `cx..cw` is itself a kernel of the outer coordinates; the inner's
    /// `X/Y/Z/W` are substituted by them.
    #[must_use]
    pub fn at(&self, cx: &Kernel, cy: &Kernel, cz: &Kernel, cw: &Kernel) -> Self {
        let mut arena = self.inner.arena.clone();
        let x = arena.splice(&cx.inner.arena, cx.inner.root);
        let y = arena.splice(&cy.inner.arena, cy.inner.root);
        let z = arena.splice(&cz.inner.arena, cz.inner.root);
        let w = arena.splice(&cw.inner.arena, cw.inner.root);
        let root = arena.substitute_vars_with(self.inner.root, &[(0, x), (1, y), (2, z), (3, w)]);
        Self::wrap(arena, root)
    }

    /// The derivative `∂self/∂var` (0=X, 1=Y, 2=Z), resolved symbolically at
    /// compile time. The building block of screen-space antialiasing: no jet
    /// domain, just an expression the calculus differentiates.
    #[must_use]
    pub fn dwrt(&self, var: u8) -> Self {
        let mut arena = self.inner.arena.clone();
        let v = arena.push_const(f32::from(var));
        let root = arena.push_binary(OpKind::Dwrt, self.inner.root, v);
        Self::wrap(arena, root)
    }

    /// `∂self/∂X`.
    #[must_use]
    pub fn dx(&self) -> Self {
        self.dwrt(0)
    }
    /// `∂self/∂Y`.
    #[must_use]
    pub fn dy(&self) -> Self {
        self.dwrt(1)
    }

    // ───────────────────────── back end ───────────────────────────

    /// The underlying fragment — for the lattice bake and inspection. Not part
    /// of the composition surface; consumers use the methods above.
    #[must_use]
    pub fn parts(&self) -> (&ExprArena, ExprId) {
        (&self.inner.arena, self.inner.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::BindingTable;
    use crate::eval::eval_scalar;

    fn eval(k: &Kernel, x: f32, y: f32) -> f32 {
        let (arena, root) = k.parts();
        eval_scalar(arena, root, &[x, y, 0.0, 0.0], &BindingTable::empty())
    }

    #[test]
    fn circle_sdf_composes() {
        // √(x² + y²) − 1, built entirely through the value API.
        let x = Kernel::x();
        let y = Kernel::y();
        let r2 = x.mul(&x).add(&y.mul(&y));
        let sdf = r2.sqrt().sub(&Kernel::constant(1.0));
        assert!((eval(&sdf, 3.0, 4.0) - 4.0).abs() < 1e-5);
        assert!((eval(&sdf, 0.0, 0.0) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn sum_is_variadic_fold() {
        let terms = [
            Kernel::x(),
            Kernel::y(),
            Kernel::constant(10.0),
            Kernel::x(),
        ];
        let s = Kernel::sum(&terms);
        assert_eq!(eval(&s, 3.0, 4.0), 3.0 + 4.0 + 10.0 + 3.0);
        assert_eq!(eval(&Kernel::sum(&[]), 9.0, 9.0), 0.0);
    }

    #[test]
    fn winding_rule_and_select() {
        // min(|Σ|, 1) then a bounds select — the glyph shape in miniature.
        let total = Kernel::sum(&[Kernel::x(), Kernel::y().neg()]);
        let coverage = total.abs().min(&Kernel::constant(1.0));
        let in_bounds = Kernel::x().ge(&Kernel::constant(0.0));
        let masked = in_bounds.select(&coverage, &Kernel::constant(0.0));
        assert_eq!(eval(&masked, 0.3, 0.1), (0.3f32 - 0.1).abs().min(1.0));
        assert_eq!(eval(&masked, -1.0, 0.0), 0.0); // out of bounds
        assert_eq!(eval(&masked, 5.0, 0.0), 1.0); // |5| clamped to 1
    }

    #[test]
    fn at_warps_coordinates() {
        // (x·y) sampled at (x+1, 2y) = (x+1)·2y.
        let body = Kernel::x().mul(&Kernel::y());
        let warped = body.at(
            &Kernel::x().add(&Kernel::constant(1.0)),
            &Kernel::y().mul(&Kernel::constant(2.0)),
            &Kernel::z(),
            &Kernel::w(),
        );
        assert_eq!(eval(&warped, 3.0, 4.0), 4.0 * 8.0);
    }

    #[test]
    fn dx_differentiates_at_compile_time() {
        use crate::backend::emit::lowering::lower_dwrt_owned;
        // d/dx √(x²+y²) = x / √(x²+y²).
        let x = Kernel::x();
        let y = Kernel::y();
        let dist = x.mul(&x).add(&y.mul(&y)).sqrt();
        let ddx = dist.dx();
        let (arena, root) = ddx.parts();
        let (out, oroot) = lower_dwrt_owned(arena, root).expect("calculus");
        let got = eval_scalar(&out, oroot, &[3.0, 4.0, 0.0, 0.0], &BindingTable::empty());
        assert!((got - 0.6).abs() < 1e-5);
    }
}
