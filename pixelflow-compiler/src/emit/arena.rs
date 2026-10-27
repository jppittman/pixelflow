//! An arena as the code that rebuilds it at load time, node by node, and
//! the snippets every emitted statement is spelled with.

use super::host::EntryContext;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::{Bound, Fold, FoldHeader, Lit, OpKind};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::Ident;

/// `quote!` for the emission's own tokens: every identifier written in it is
/// at the macro's mixed-site span, so a local it binds (`__arena`, `__t3`,
/// `__lo`) is a different variable from a user's of the same spelling, which
/// keeps the call site's. A user's tokens interpolated into it keep theirs.
///
/// Mixed-site hygiene covers locals and labels only; an item or a path
/// resolves at the call site as ever. So a `const`
/// spelled like an emitted local (`const __lo: usize`) still meets it, and
/// rustc refuses that loudly (E0530, E0158, E0308 or E0005, by the type of
/// the `const`); the `__` prefix is what keeps an
/// unprefixed one from meeting a user's `lo`.
macro_rules! emitted {
    ($($tokens:tt)*) => {
        ::quote::quote_spanned!(::proc_macro2::Span::mixed_site()=> $($tokens)*)
    };
}
pub(super) use emitted;

/// A local of the emitted code, `prefix` then `n`.
pub(super) fn local(prefix: &str, n: u64) -> Ident {
    Ident::new(&format!("{prefix}{n}"), Span::mixed_site())
}

/// The statement declaring `slot`'s uniform in the arena being built, with
/// the call's `default`: an identity minted per call, so an entry expanded at
/// compile time and one staged at run time declare an argument alike.
pub(super) fn declare_uniform(slot: &Ident, default: &TokenStream) -> TokenStream {
    emitted! {
        let #slot = __arena.declare_uniform(::pixelflow_core::__macro::ir::arena::UniformDecl {
            id: ::pixelflow_core::__macro::ir::arena::UniformIdentity::mint(),
            default: #default,
        });
    }
}

/// Decode a fold's header bits. A zero literal range supplies the header's
/// bounds; the `Fold` itself carries only the monoid, binder and stride.
pub(super) fn fold_from_bits(fold: Fold) -> TokenStream {
    let bits = FoldHeader::new(fold, Bound::Lit(0), Bound::Lit(0)).to_bits();
    emitted! {
        ::pixelflow_core::__macro::ir::fold::FoldHeader::from_bits(#bits)
            .expect("kernel! emitted a well-formed fold header").fold()
    }
}

/// Emit the arena as code that rebuilds it at load time: every node the
/// root reaches, in the arena's order, children before parents.
///
/// `Dwrt` nodes are emitted as built: the macro tier never resolves one
/// (`macro_tier`), so a warp applied to the emitted kernel reaches the
/// `Dwrt`'s operand and the chain rule holds
/// (`tests/derivative_under_warp.rs`).
///
/// The arena's uniform table is declared first, one per argument of
/// `context`, in order: each with an identity minted per call and the
/// call's value as its default, so the kernel's arguments are that call's
/// and its program is every call's — every one declared, read or not, so a
/// positional binding cannot shift.
pub(super) fn arena_to_tokens(
    arena: &ExprArena,
    root: ExprId,
    context: &EntryContext,
) -> TokenStream {
    assert_eq!(
        arena.uniforms().len(),
        context.arguments.len(),
        "lowering declares one uniform per argument, and a relink restores the table"
    );
    let decls = context
        .arguments
        .iter()
        .enumerate()
        .map(|(slot, value)| declare_uniform(&uniform_var(slot as u64), value));
    let nodes = reached(arena, root).into_iter().map(|id| {
        let var = node_var(id);
        let expr = node(arena, id, context);
        emitted! {
            let #var = #expr;
        }
    });
    let root_ident = node_var(root);
    emitted! {{
        let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
        #(#decls)*
        #(#nodes)*
        (__arena, #root_ident)
    }}
}

/// The variable holding a uniform slot's `UniformId` in the arena being
/// built.
fn uniform_var(slot: u64) -> Ident {
    local("__u", slot)
}

/// The variable holding a node's id in the arena being built.
fn node_var(id: ExprId) -> Ident {
    local("__e", u64::from(id.0))
}

/// The nodes `root` reaches, in the arena's order, children before parents:
/// what the emitted code builds. A node nothing reads — an unread
/// parameter's leaf — is not rebuilt; its uniform is still declared.
fn reached(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
    let mut reached = vec![false; arena.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reached[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }
    arena
        .nodes()
        .map(|(id, _)| id)
        .filter(|id| reached[id.0 as usize])
        .collect()
}

/// One node, built in `__arena` from the nodes before it.
fn node(arena: &ExprArena, id: ExprId, _context: &EntryContext) -> TokenStream {
    match arena.node(id) {
        ExprNode::Var(i) => emitted! { __arena.push_var(#i) },
        ExprNode::Placeholder(identity) => {
            panic!("emit: fold index scope {identity} escaped without its binder")
        }
        // By bit pattern, not as a decimal literal: `quote`'s `f32`
        // impl goes through `Literal::f32_suffixed`, which asserts
        // `is_finite()` — and non-finite constants are ordinary here. A
        // true comparison mask is all-ones (`OpKind::mask`), which is
        // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
        // the folder produces those. Bits also roundtrip exactly, with
        // no decimal-formatting question to get wrong.
        ExprNode::Const(Lit::Real(v)) => {
            let bits = v.to_bits();
            emitted! { __arena.push_const(f32::from_bits(#bits)) }
        }
        ExprNode::Const(Lit::Mask(value)) => {
            emitted! { __arena.push_lit(::pixelflow_core::__macro::ir::Lit::Mask(#value)) }
        }
        ExprNode::Param(k) => panic!(
            "kernel! produced ExprNode::Param({k}) in an expanded entry; the language has no parameter holes"
        ),
        // No body can spell a buffer: the language has none
        // (docs/DENOTATION.md §11), so one here is a front-end bug. Fail
        // loud rather than emit a node that references a buffer table the
        // emitted code never declares.
        ExprNode::Buffer(b) => {
            panic!(
                "kernel! produced ExprNode::Buffer({}) — the language has no buffers \
                 (docs/DENOTATION.md §11)",
                b.0
            )
        }
        // An argument: the slot declared above for it.
        ExprNode::Uniform(u) => {
            let slot = uniform_var(u.0);
            emitted! { __arena.push_uniform(#slot) }
        }
        // And a reference is minted by `Kernel::by_ref` at composition
        // time — a runtime value, and the key it carries names a store
        // in the *build host's* process, which the compiled program is
        // not. Emitting one would name nothing.
        ExprNode::Ref(k) | ExprNode::Apply(k, ..) => {
            panic!(
                "kernel! produced a reference to {k:?} — a reference names a kernel \
                 interned in this process, which the emitted program does not share"
            )
        }
        ExprNode::Unary(op, child) => {
            let op_code = opkind_to_tokens(op);
            let child = node_var(child);
            emitted! { __arena.push_unary(#op_code, #child) }
        }
        ExprNode::Binary(op, a, b) => {
            let op_code = opkind_to_tokens(op);
            let (a, b) = (node_var(a), node_var(b));
            emitted! { __arena.push_binary(#op_code, #a, #b) }
        }
        ExprNode::Ternary(op, a, b, c) => {
            let op_code = opkind_to_tokens(op);
            let (a, b, c) = (node_var(a), node_var(b), node_var(c));
            emitted! { __arena.push_ternary(#op_code, #a, #b, #c) }
        }
        ExprNode::Nary(op, ..) => {
            let op_code = opkind_to_tokens(op);
            let children: Vec<Ident> = arena.children(id).map(node_var).collect();
            emitted! { __arena.push_nary(#op_code, &[#(#children),*]) }
        }
        // A fold's metadata is a `Fold`, whose fields are private
        // precisely so no caller can assemble one that means nothing —
        // so it travels the way the two cache keys carry it, as bits
        // with a total inverse on the far side.
        ExprNode::Reduce { fold, lo, hi, body } => {
            let body = node_var(body);
            let decoded = fold_from_bits(fold);
            let bound = |bound| match bound {
                Bound::Lit(count) => emitted!(::pixelflow_core::__macro::ir::Bound::Lit(#count)),
                Bound::Value(term) => {
                    let term = node_var(term);
                    emitted!(::pixelflow_core::__macro::ir::Bound::Value(#term))
                }
            };
            let (lo, hi) = (bound(lo), bound(hi));
            emitted! { __arena.push_reduce(#decoded, #lo, #hi, #body) }
        }
    }
}

/// The path naming `kind` in generated code.
///
/// [`OpKind::variant_name`] is generated by `op_table!`, so the identifier
/// cannot drift from the enum and a newly added op needs no edit here.
pub(super) fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    let variant = format_ident!("{}", kind.variant_name());
    quote! { ::pixelflow_core::__macro::ir::OpKind::#variant }
}
