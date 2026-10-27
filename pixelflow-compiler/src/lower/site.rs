//! The site an entry that takes no kernel is lowered at: an [`ExprArena`] at
//! macro expansion.

use super::walk::lower_entry;
use super::{FoldBound, Site};
use crate::ast::{Callable, PLAN};
use crate::sema::{AnalyzedKernel, Scalar};
use pixelflow_ir::arena::{
    Axis, ExprArena, ExprId, FoldTerms, OpenFold, UniformDecl, UniformIdentity,
};
use pixelflow_ir::library;
use pixelflow_ir::{Bound, Monoid, OpKind};
use std::convert::Infallible;
use syn::Ident;

/// The default a uniform is declared with here, where no call has supplied
/// one. Nothing at expansion reads it — a uniform is never folded, and
/// emission declares each with the call's value — so it is a NaN, which
/// would poison whatever read it by mistake rather than pass for a number.
const UNBOUND: f32 = f32::NAN;

/// An entry lowered at expansion: its arena, root, and declared scalars.
pub(crate) struct Lowered {
    pub(crate) arena: ExprArena,
    pub(crate) root: ExprId,
    /// The entry's uniform scalars, in the order the arena's uniform table
    /// declares them: the host function's argument each defaults to.
    pub(crate) scalars: Vec<Scalar>,
}

/// The site an entry with no kernel parameters is lowered at.
#[derive(Default)]
pub(crate) struct Expansion {
    arena: ExprArena,
    /// The scalar behind each uniform declared, in declaration order.
    scalars: Vec<Scalar>,
    /// The folds whose bodies are being built, innermost last, each with
    /// the fold it closes into.
    open: Vec<(OpenFold, FoldTerms)>,
}

impl Expansion {
    /// Lower `entry`, an entry that takes no kernel, at expansion: its
    /// parameters are declared first, as uniforms, in
    /// [`AnalyzedKernel::parameters`]' order, so the arena's uniform table is
    /// the entry's declaration order — every scalar, read or not, so that a
    /// positional binding cannot shift when a parameter goes unread.
    ///
    /// # Errors
    ///
    /// When the body has a construct lowering cannot express, or `entry`
    /// takes a kernel — an argument that exists only when its host function
    /// is called.
    pub(crate) fn lower(entry: Callable<'_>, analyzed: &AnalyzedKernel) -> syn::Result<Lowered> {
        let mut site = Expansion::default();
        let root = lower_entry(entry, analyzed, &mut site)?;
        Ok(Lowered {
            arena: site.arena,
            root,
            scalars: site.scalars,
        })
    }
}

impl Site for Expansion {
    type Term = ExprId;
    /// There is no argument at expansion.
    type Argument = Infallible;

    fn constant(&mut self, value: f32) -> ExprId {
        self.arena.push_const(value)
    }

    fn coordinate(&mut self, axis: Axis) -> ExprId {
        self.arena.push_var(axis.var())
    }

    fn unary(&mut self, op: OpKind, operand: ExprId) -> ExprId {
        self.arena.push_unary(op, operand)
    }

    fn binary(&mut self, op: OpKind, [a, b]: [ExprId; 2]) -> ExprId {
        self.arena.push_binary(op, a, b)
    }

    fn ternary(&mut self, op: OpKind, [a, b, c]: [ExprId; 3]) -> ExprId {
        self.arena.push_ternary(op, a, b, c)
    }

    fn fract(&mut self, x: ExprId) -> ExprId {
        library::fract(&mut self.arena, x)
    }

    fn hypot(&mut self, operands: [ExprId; 2]) -> ExprId {
        library::hypot(&mut self.arena, operands)
    }

    fn clamp(&mut self, x: ExprId, bounds: [ExprId; 2]) -> ExprId {
        library::clamp(&mut self.arena, x, bounds)
    }

    fn derivative(&mut self, e: ExprId, axis: Axis) -> ExprId {
        library::derivative(&mut self.arena, e, axis)
    }

    /// A declaration here holds a placeholder default; emission declares
    /// each with the call's value, which the scalar names.
    fn uniform(&mut self, scalar: Scalar) -> ExprId {
        self.scalars.push(scalar);
        let slot = self.arena.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: UNBOUND,
        });
        self.arena.push_uniform(slot)
    }

    fn kernel_parameter(&mut self, name: &Ident) -> syn::Result<Infallible> {
        Err(syn::Error::new(
            name.span(),
            format!(
                "`{name}` is a kernel, and an entry that takes one is lowered when its host \
                 function is called, not at expansion (Phase D-a of {PLAN})"
            ),
        ))
    }

    fn open_fold(
        &mut self,
        depth: usize,
        monoid: Monoid,
        bounds: [FoldBound<ExprId>; 2],
    ) -> ExprId {
        let bound = |bound| match bound {
            FoldBound::Known(count) => Bound::Lit(count),
            FoldBound::Value(term) => Bound::Value(term),
        };
        let [lo, hi] = bounds;
        let open = self
            .arena
            .open_fold(depth)
            .expect("lowering refuses a fold nested past the binders before it opens one");
        let index = open.index();
        self.open.push((
            open,
            FoldTerms {
                monoid,
                lo: bound(lo),
                hi: bound(hi),
            },
        ));
        index
    }

    fn close_fold(&mut self, body: ExprId) -> ExprId {
        let (open, terms) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        self.arena
            .close_fold(open, body, terms)
            .expect("lowering refuses a fold nested past the binders, so one is always free")
    }

    fn apply(&mut self, kernel: Infallible, _at: [ExprId; 2]) -> ExprId {
        match kernel {}
    }

    fn warp(&mut self, field: ExprId, at: [ExprId; 2]) -> ExprId {
        self.arena.warp(field, at)
    }
}
