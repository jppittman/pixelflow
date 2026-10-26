//! The second [`Site`]: lowering's steps as the statements that take them,
//! run when an entry that takes a kernel has its host function called.

use super::arena::{declare_uniform, emitted, fold_from_bits, local, opkind_to_tokens};
use super::host::{argument, kernel_from};
use crate::ast::Callable;
use crate::lower::{FoldBound, Site, lower_entry};
use crate::sema::{AnalyzedKernel, Scalar};
use pixelflow_ir::arena::Axis;
use pixelflow_ir::{Binder, Monoid};
use proc_macro2::TokenStream;
use syn::Ident;

/// The site an entry that takes a kernel is lowered at: the Rust statements
/// that build its arena when its host function is called (plan Phase D-a).
///
/// Each of lowering's steps becomes the statement that takes it, naming a
/// term by the variable that holds its node: the same IR call
/// [`Expansion`](crate::lower::Expansion) makes at expansion — a push, a
/// [`library`](pixelflow_ir::library) definition, [`ExprArena::open_fold`]
/// and [`ExprArena::close_fold`] — in the same order, so a program built at
/// load time is the one lowering would have built at expansion. The steps
/// only `Staged` has are an argument's: [`ExprArena::admit`] when the host
/// function is called, after the entry's own uniforms, and
/// [`ExprArena::apply`] at each `k(x, y)`.
///
/// What is left is compacted ([`ExprArena::compact`]): a node a
/// substitution replaced or nothing read is dropped, as emission drops one
/// from an arena built at expansion, so the kernel holds the program alone.
///
/// A fold staged here closes when the host function runs, and panics then
/// if its body — an argument's folds among them — binds every binder, as
/// `Kernel::over` panics.
///
/// [`ExprArena::open_fold`]: pixelflow_ir::arena::ExprArena::open_fold
/// [`ExprArena::close_fold`]: pixelflow_ir::arena::ExprArena::close_fold
/// [`ExprArena::admit`]: pixelflow_ir::arena::ExprArena::admit
/// [`ExprArena::apply`]: pixelflow_ir::arena::ExprArena::apply
/// [`ExprArena::compact`]: pixelflow_ir::arena::ExprArena::compact
#[derive(Default)]
pub(super) struct Staged {
    statements: Vec<TokenStream>,
    /// How many variables have been named: every one is fresh.
    named: u64,
    /// The folds open, innermost last: the variable holding each, and the
    /// closure that makes its `Fold` once its binder is chosen.
    open: Vec<(Ident, TokenStream)>,
}

impl Staged {
    /// The host function's body for `entry`, an entry that takes a kernel:
    /// statements building its kernel when it is called.
    ///
    /// # Errors
    ///
    /// When the body has a construct lowering cannot express.
    pub(super) fn lower(
        entry: Callable<'_>,
        analyzed: &AnalyzedKernel,
    ) -> syn::Result<TokenStream> {
        let mut site = Staged::default();
        let root = lower_entry(entry, analyzed, &mut site)?;
        let statements = site.statements;
        let kernel = kernel_from(&emitted!(__arena.compact(#root)));
        Ok(emitted! {
            let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
            #(#statements)*
            #kernel
        })
    }

    /// A fresh variable, `prefix` then a number.
    fn fresh(&mut self, prefix: &str) -> Ident {
        let name = local(prefix, self.named);
        self.named += 1;
        name
    }

    /// The statement binding `step`'s node to a fresh variable, which names
    /// it.
    fn step(&mut self, step: TokenStream) -> Ident {
        let term = self.fresh("__t");
        self.statements.push(emitted! { let #term = #step; });
        term
    }

    /// The closure an open fold closes into: its monoid, carried as a
    /// fold's bits (as `arena::node` carries a `Fold`), its binder the one
    /// `close_fold` chooses, and its known or value bounds.
    fn fold_at(monoid: Monoid, bounds: [FoldBound<Ident>; 2]) -> TokenStream {
        let first = Binder::from_slot(0).expect("the IR has a binder");
        let monoid_of = fold_from_bits(pixelflow_ir::Fold::new(monoid, first));
        let bound = |bound| match bound {
            FoldBound::Known(count) => emitted!(::pixelflow_core::__macro::ir::Bound::Lit(#count)),
            FoldBound::Value(term) => emitted!(::pixelflow_core::__macro::ir::Bound::Value(#term)),
        };
        let [lo, hi] = bounds;
        let (lo, hi) = (bound(lo), bound(hi));
        emitted! {{
            ::pixelflow_core::__macro::ir::arena::FoldTerms {
                monoid: (#monoid_of).monoid(),
                lo: #lo,
                hi: #hi,
            }
        }}
    }
}

impl Site for Staged {
    type Term = Ident;
    type Argument = Ident;

    fn constant(&mut self, value: f32) -> Ident {
        // By bit pattern, for `node`'s reason.
        let bits = value.to_bits();
        self.step(emitted!(__arena.push_const(f32::from_bits(#bits))))
    }

    fn coordinate(&mut self, axis: Axis) -> Ident {
        let var = axis.var();
        self.step(emitted!(__arena.push_var(#var)))
    }

    fn unary(&mut self, op: pixelflow_ir::OpKind, a: Ident) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(emitted!(__arena.push_unary(#op, #a)))
    }

    fn binary(&mut self, op: pixelflow_ir::OpKind, [a, b]: [Ident; 2]) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(emitted!(__arena.push_binary(#op, #a, #b)))
    }

    fn ternary(&mut self, op: pixelflow_ir::OpKind, [a, b, c]: [Ident; 3]) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(emitted!(__arena.push_ternary(#op, #a, #b, #c)))
    }

    fn fract(&mut self, x: Ident) -> Ident {
        self.step(emitted!(::pixelflow_core::__macro::ir::library::fract(&mut __arena, #x)))
    }

    fn hypot(&mut self, [a, b]: [Ident; 2]) -> Ident {
        self.step(emitted!(::pixelflow_core::__macro::ir::library::hypot(&mut __arena, [#a, #b])))
    }

    fn clamp(&mut self, x: Ident, [lo, hi]: [Ident; 2]) -> Ident {
        self.step(emitted!(
            ::pixelflow_core::__macro::ir::library::clamp(&mut __arena, #x, [#lo, #hi])
        ))
    }

    fn derivative(&mut self, e: Ident, axis: Axis) -> Ident {
        let axis = match axis {
            Axis::X => emitted!(::pixelflow_core::__macro::ir::arena::Axis::X),
            Axis::Y => emitted!(::pixelflow_core::__macro::ir::arena::Axis::Y),
        };
        self.step(emitted!(
            ::pixelflow_core::__macro::ir::library::derivative(&mut __arena, #e, #axis)
        ))
    }

    /// Declared with the call's value as its default, as
    /// [`arena_to_tokens`](super::arena::arena_to_tokens) declares an expanded
    /// entry's, so baking the kernel draws the call.
    fn uniform(&mut self, scalar: Scalar) -> Ident {
        let slot = self.fresh("__u");
        self.statements
            .push(declare_uniform(&slot, &argument(&scalar)));
        self.step(emitted!(__arena.push_uniform(#slot)))
    }

    fn kernel_parameter(&mut self, name: &Ident) -> syn::Result<Ident> {
        let argument = self.fresh("__k");
        self.statements
            .push(emitted! { let #argument = __arena.admit(#name); });
        Ok(argument)
    }

    fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: [FoldBound<Ident>; 2]) -> Ident {
        let fold = self.fresh("__f");
        let depth = proc_macro2::Literal::usize_unsuffixed(depth);
        self.statements.push(emitted! {
            let #fold = __arena
                .open_fold(#depth)
                .expect("kernel! refused a fold nested past the binders at expansion");
        });
        self.open
            .push((fold.clone(), Self::fold_at(monoid, bounds)));
        self.step(emitted!(#fold.index()))
    }

    fn close_fold(&mut self, body: Ident) -> Ident {
        let (fold, fold_at) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        self.step(emitted! {
            __arena.close_fold(#fold, #body, #fold_at).expect(
                "kernel!: a fold around a kernel's application has no binder left: its body, \
                 the argument's folds among them, binds every one of the IR's indices \
                 (`Binder::COUNT`)"
            )
        })
    }

    fn apply(&mut self, kernel: Ident, [x, y]: [Ident; 2]) -> Ident {
        self.step(emitted!(__arena.apply(&#kernel, [#x, #y])))
    }

    fn warp(&mut self, field: Ident, [x, y]: [Ident; 2]) -> Ident {
        self.step(emitted!(__arena.warp(#field, [#x, #y])))
    }
}
