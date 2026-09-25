//! `ExprArena` → the `TokenStream` that rebuilds it at load time.
//!
//! The back end, and there is one. It produces a [`Kernel`] — an arena
//! fragment, the language's own value. Nothing is compiled at
//! macro-expansion time and nothing is compiled at construction: a `Kernel`
//! becomes machine code when a consumer compiles it at a lattice's shape and
//! collapses it (`Lattice::bake`), which is the only way a kernel turns into
//! numbers. So there is no per-batch entry — no `Manifold` impl calling into
//! the JIT once per SIMD batch — and nothing here emits one.
//!
//! Two steps, because the interesting thing happens between them: an arena is
//! lowered from the AST ([`crate::lower`]), *then* optionally rewritten, then
//! emitted. Emission takes an arena rather than an AST so that the optimizer
//! has somewhere to stand.
//!
//! What is emitted depends on how the block was spelled
//! ([`Spelling`](crate::ast::Spelling)). The closure form is an expression:
//! a `Kernel`, or a builder closure over its parameters. The items form is
//! items: one host `fn` per entry, returning a `Kernel`, and one host
//! `const` per `pub const`. A helper is inlined and a private `const` is
//! folded, so neither leaves a trace.
//!
//! [`Kernel`]: pixelflow_core::Kernel

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::optimize::Optimize;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ast::{ConstItem, FnItem, Role, Spelling};
use crate::lower;
use crate::sema::AnalyzedKernel;

/// Emit arena-backend code for an analyzed kernel.
///
/// For the closure form, a token stream evaluating to:
/// - zero params — a [`Kernel`](pixelflow_core::Kernel) value, built at load
///   time from the arena this expansion computed.
/// - N params — a builder `|p0, ..., pN| -> Kernel` whose arguments are
///   anything `Into<Scalar>`: an `f32` is constant-folded into the fragment
///   when the builder runs (no JIT — leaves are bake-time-only and fuse at a
///   root, which is what lets a font build thousands of leaf kernels and
///   compile one arena), and a `Uniform` handle declares a per-call slot
///   instead. The *type* at the call site chooses, so every site that passes
///   an `f32` keeps folding.
///
/// For the items form, one `fn name(params) -> Kernel` per entry, its
/// parameters bound exactly as a builder's are, and one `const` per `pub
/// const`.
///
/// Kernels compose as *values* — `Kernel::at`/`sum`/`select`/arithmetic — not
/// by inlining a manifold through a macro slot, so there is no
/// manifold-typed parameter and nothing here to lower one with.
///
/// `optimizer` rewrites each lowered arena before it is emitted. It is a
/// parameter rather than a branch because "do not optimize" is a value:
/// `kernel_raw!` passes [`Identity`](pixelflow_ir::optimize::Identity), and
/// that is the entire difference between the two macros.
///
/// Returns `Err` if a body contains an operation lowering cannot express.
pub fn emit_kernel(
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> Result<TokenStream, String> {
    match analyzed.def.spelling {
        Spelling::Closure => {
            let [entry] = analyzed.def.fns.as_slice() else {
                unreachable!("the closure form parses to exactly one entry");
            };
            let arena_code = entry_arena(entry, analyzed, optimizer)?;
            Ok(emit_closure(entry, &arena_code))
        }
        Spelling::Items => {
            let mut items = TokenStream::new();
            for c in analyzed.def.consts.iter().filter(|c| is_pub(&c.vis)) {
                items.extend(emit_const(c, analyzed.consts[&c.name.to_string()]));
            }
            for entry in analyzed.def.fns.iter().filter(|f| f.role() == Role::Entry) {
                let arena_code = entry_arena(entry, analyzed, optimizer)?;
                items.extend(emit_entry(entry, &arena_code));
            }
            Ok(items)
        }
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    !matches!(vis, syn::Visibility::Inherited)
}

/// Lower an entry, run the optimizer over it, and emit the code that
/// rebuilds the arena: an expression evaluating to `(arena, root)`.
fn entry_arena(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> Result<TokenStream, String> {
    let mut arena = ExprArena::new();
    let root = lower::lower_entry(entry, analyzed, &mut arena)?;

    // Declining is ordinary and needs no arm: the lowered term stands, and a
    // kernel that reaches the runtime tier unoptimized is optimized there.
    let (arena, root) = optimizer
        .optimize(&arena, root)
        .into_changed()
        .unwrap_or((arena, root));

    Ok(arena_to_tokens(&arena, root))
}

/// The closure form's expansion: a `Kernel`, or a builder closure.
fn emit_closure(entry: &FnItem, arena_code: &TokenStream) -> TokenStream {
    if entry.params.is_empty() {
        return quote! {
            {
                let (__arena, __root) = #arena_code;
                ::pixelflow_core::Kernel::from_parts(__arena, __root)
            }
        };
    }

    // The builder. A closure cannot be generic over its argument types, so
    // the expansion is a generic `fn` returning `impl Fn`: the type parameters
    // are inferred at the call site — `f32` from a float literal or variable,
    // `Uniform` from a handle — and each `let` binding of a builder is one
    // signature. Arguments appear in declaration order.
    let Bound {
        names,
        generics,
        scalar,
        body,
    } = bind_params(entry, arena_code);

    quote! {
        {
            fn __builder< #( #generics: ::core::convert::Into<#scalar> ),* >()
                -> impl Fn( #( #generics ),* ) -> ::pixelflow_core::Kernel
            {
                move | #( #names: #generics ),* | #body
            }
            __builder()
        }
    }
}

/// An entry's expansion: a host function returning a `Kernel`, generic over
/// its parameters exactly as a builder is, with the entry's visibility and
/// doc comments.
fn emit_entry(entry: &FnItem, arena_code: &TokenStream) -> TokenStream {
    let attrs = &entry.attrs;
    let vis = &entry.vis;
    let name = &entry.name;
    if entry.params.is_empty() {
        return quote! {
            #(#attrs)*
            #[must_use]
            #vis fn #name() -> ::pixelflow_core::Kernel {
                let (__arena, __root) = #arena_code;
                ::pixelflow_core::Kernel::from_parts(__arena, __root)
            }
        };
    }
    let Bound {
        names,
        generics,
        scalar,
        body,
    } = bind_params(entry, arena_code);
    quote! {
        #(#attrs)*
        #[must_use]
        #vis fn #name< #( #generics: ::core::convert::Into<#scalar> ),* >( #( #names: #generics ),* )
            -> ::pixelflow_core::Kernel
        #body
    }
}

/// The pieces of a parameterized expansion: the parameter names, one type
/// parameter per name, the `Scalar` path they convert into, and the body
/// that folds them in.
struct Bound<'a> {
    names: Vec<&'a proc_macro2::Ident>,
    generics: Vec<proc_macro2::Ident>,
    scalar: TokenStream,
    body: TokenStream,
}

/// How an entry's parameters reach the arena: each `Param(i)` is
/// substituted with the argument in position `i`, converted to a `Scalar`.
fn bind_params<'a>(entry: &'a FnItem, arena_code: &TokenStream) -> Bound<'a> {
    let names: Vec<&proc_macro2::Ident> = entry.params.iter().map(|p| &p.name).collect();
    let generics: Vec<proc_macro2::Ident> =
        (0..names.len()).map(|i| format_ident!("__A{i}")).collect();
    let scalar = quote! { ::pixelflow_core::__macro::ir::Scalar };
    let arity = names.len();
    let body = quote! {
        {
            let (mut __arena, __root) = #arena_code;
            let __params: [#scalar; #arity] = [ #( #names.into() ),* ];
            let __root = __arena.substitute_params(__root, &__params);
            ::pixelflow_core::Kernel::from_parts(__arena, __root)
        }
    };
    Bound {
        names,
        generics,
        scalar,
        body,
    }
}

/// A `pub const`'s host twin, holding the value `sema` evaluated. By bit
/// pattern, for the reason [`arena_to_tokens`] gives: it is exact, and a
/// const may be non-finite (`1.0 / 0.0`), which a decimal literal cannot
/// spell.
fn emit_const(item: &ConstItem, value: f32) -> TokenStream {
    let attrs = &item.attrs;
    let vis = &item.vis;
    let name = &item.name;
    let bits = value.to_bits();
    quote! {
        #(#attrs)*
        #vis const #name: f32 = f32::from_bits(#bits);
    }
}

/// Emit the arena, node for node, as code that rebuilds it at load time.
///
/// `Dwrt` nodes are emitted as they were built and resolved at bake time, by
/// the one `LowerDwrt` pass in the runtime pipeline. Resolving them at
/// expansion time — which this front end used to do — is not an optimization
/// but a miscompilation under composition: `Kernel::at` warps a kernel by
/// substituting into its `Var` leaves, so a surviving `Dwrt(f, x)` has the
/// warp reach its *operand* and differentiates the warped function, which is
/// the chain rule. A `Dwrt` already resolved to `f'` has no operand left for
/// the warp to reach, and the substitution silently lands inside `f'`.
///
/// See docs/plans/2026-09-08-macro-tier-is-arena-native.md.
pub fn arena_to_tokens(arena: &ExprArena, root: ExprId) -> TokenStream {
    let mut stmts = Vec::new();
    let n = arena.len();
    for idx in 0..n {
        let id = ExprId(idx as u32);
        let ident = format_ident!("__e{}", idx);
        let expr = match arena.node(id) {
            pixelflow_ir::arena::ExprNode::Var(i) => {
                quote! { __arena.push_var(#i) }
            }
            // By bit pattern, not as a decimal literal: `quote`'s `f32`
            // impl goes through `Literal::f32_suffixed`, which asserts
            // `is_finite()` — and non-finite constants are ordinary here. A
            // true comparison mask is all-ones (`OpKind::mask`), which is
            // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
            // the folder now produces those. Bits also roundtrip exactly, with
            // no decimal-formatting question to get wrong.
            pixelflow_ir::arena::ExprNode::Const(v) => {
                let bits = v.to_bits();
                quote! { __arena.push_const(f32::from_bits(#bits)) }
            }
            pixelflow_ir::arena::ExprNode::Param(i) => {
                quote! { __arena.push_param(#i) }
            }
            // The `kernel!` macro has no buffer surface yet, so this is
            // unreachable in practice; fail loud rather than emit a node that
            // references a buffer table `from_raw` does not reconstruct.
            pixelflow_ir::arena::ExprNode::Buffer(b) => {
                panic!(
                    "kernel! produced ExprNode::Buffer({}) — lattice parameters are not wired \
                     into the compiler yet (KERNELS_AND_LATTICES.md M4)",
                    b.0
                )
            }
            // Likewise unreachable: a uniform enters a kernel at the builder
            // call (`substitute_params`), never from the macro's own arena.
            pixelflow_ir::arena::ExprNode::Uniform(u) => {
                panic!(
                    "kernel! produced ExprNode::Uniform({}) — uniforms are chosen at the \
                     builder call site, not in the macro body",
                    u.0
                )
            }
            // And a reference is minted by `Kernel::by_ref` at composition
            // time — a runtime value, and the key it carries names a store
            // in the *build host's* process, which the compiled program is
            // not. Emitting one would name nothing.
            pixelflow_ir::arena::ExprNode::Ref(k) => {
                panic!(
                    "kernel! produced ExprNode::Ref({k:?}) — a reference names a kernel \
                     interned in this process, which the emitted program does not share"
                )
            }
            pixelflow_ir::arena::ExprNode::Unary(op, child) => {
                let op_code = opkind_to_tokens(op);
                let child_ident = format_ident!("__e{}", child.0);
                quote! { __arena.push_unary(#op_code, #child_ident) }
            }
            pixelflow_ir::arena::ExprNode::Binary(op, a, b) => {
                let op_code = opkind_to_tokens(op);
                let a_ident = format_ident!("__e{}", a.0);
                let b_ident = format_ident!("__e{}", b.0);
                quote! { __arena.push_binary(#op_code, #a_ident, #b_ident) }
            }
            pixelflow_ir::arena::ExprNode::Ternary(op, a, b, c) => {
                let op_code = opkind_to_tokens(op);
                let a_ident = format_ident!("__e{}", a.0);
                let b_ident = format_ident!("__e{}", b.0);
                let c_ident = format_ident!("__e{}", c.0);
                quote! { __arena.push_ternary(#op_code, #a_ident, #b_ident, #c_ident) }
            }
            pixelflow_ir::arena::ExprNode::Nary(op, ..) => {
                let op_code = opkind_to_tokens(op);
                let child_idents: Vec<_> = arena
                    .children(id)
                    .map(|c| format_ident!("__e{}", c.0))
                    .collect();
                quote! { __arena.push_nary(#op_code, &[#(#child_idents),*]) }
            }
            // A fold's metadata is a `Fold`, whose fields are private
            // precisely so no caller can assemble one that means nothing —
            // so it travels the way the two cache keys carry it, as bits
            // with a total inverse on the far side.
            pixelflow_ir::arena::ExprNode::Reduce { fold, body } => {
                let bits = fold.to_bits();
                let body_ident = format_ident!("__e{}", body.0);
                quote! {
                    __arena.push_reduce(
                        ::pixelflow_core::__macro::ir::fold::Fold::from_bits(#bits)
                            .expect("kernel! emitted a well-formed fold"),
                        #body_ident,
                    )
                }
            }
            // Unreachable for the same reason `Ref` is: there is no
            // `kernel!` surface syntax for a hard branch. `Guard` is built
            // directly against an `ExprArena` (`ExprArena::push_guard`), not
            // lowered from a macro body — and even if it were, its `on`/
            // `off` keys would name kernels interned in the *build host's*
            // process, which the emitted program does not share, exactly as
            // `Ref`'s panic says.
            pixelflow_ir::arena::ExprNode::Guard { mask, on, off } => {
                panic!(
                    "kernel! produced ExprNode::Guard(mask={mask:?}, on={on:?}, off={off:?}) \
                     — there is no surface syntax for a hard branch yet; it is built directly \
                     against an ExprArena, not lowered from a kernel! body"
                )
            }
            // A store is post-legalize vocabulary: the passes that wrap a
            // kernel in the lattice's folds build one, after extraction,
            // and no kernel! body can spell it.
            pixelflow_ir::arena::ExprNode::Write { .. } => {
                panic!(
                    "kernel! produced ExprNode::Write — a store has no surface syntax; \
                     the legalize passes build one after extraction"
                )
            }
        };
        stmts.push(quote! {
            let #ident = #expr;
        });
    }

    let root_ident = format_ident!("__e{}", root.0);
    quote! {{
        let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
        #(#stmts)*
        (__arena, #root_ident)
    }}
}

/// The path naming `kind` in generated code.
///
/// One line per op used to live here — 40 of the 50, closing with a
/// `_ => panic!("Unsupported OpKind for JIT")` that refused ops the arena
/// holds and codegen emits perfectly well (`Reduce`, the integer-domain ops,
/// `Gather`). It was the fourth independently-maintained copy of the op
/// table, and the third one found drifting from it this week.
///
/// [`OpKind::variant_name`] is generated by `op_table!`, so the identifier
/// cannot drift from the enum and a newly added op needs no edit here.
fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    let variant = format_ident!("{}", kind.variant_name());
    quote! { ::pixelflow_core::__macro::ir::OpKind::#variant }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::analyze;
    use pixelflow_ir::optimize::Identity;
    use quote::quote;

    /// The expansion of `input`, unoptimized, as text.
    fn expansion(input: proc_macro2::TokenStream) -> String {
        let analyzed = analyze(parse(input).expect("parses")).expect("analyzes");
        emit_kernel(&analyzed, &mut Identity)
            .expect("emits")
            .to_string()
    }

    /// The items form expands to items: a `fn` per entry with the entry's
    /// visibility and doc comment, a `const` per `pub const`, and nothing
    /// for a helper or a private `const`.
    #[test]
    fn the_items_form_expands_to_items() {
        let code = expansion(quote! {
            const R: f32 = 1.0;
            /// Twice the radius.
            pub const TWO_R: f32 = R * 2.0;
            fn sq(x: f32) -> f32 { x * x }
            /// The circle.
            pub fn circle(cx: f32) -> f32 { sq(X - cx) - R }
            pub(crate) fn plain() -> f32 { X }
        });
        assert!(
            code.contains("pub const TWO_R : f32 = f32 :: from_bits ("),
            "{code}"
        );
        assert!(
            code.contains("Twice the radius."),
            "the const's doc: {code}"
        );
        assert!(
            !code.contains("const R :"),
            "a private const is folded: {code}"
        );
        assert!(!code.contains("fn sq"), "a helper is inlined: {code}");
        assert!(code.contains("The circle."), "the entry's doc: {code}");
        assert!(code.contains("# [must_use] pub fn circle <"), "{code}");
        assert!(code.contains("(cx : __A0)"), "{code}");
        assert!(code.contains("pub (crate) fn plain () ->"), "{code}");
    }

    /// The closure form expands to an expression, as it always has.
    #[test]
    fn the_closure_form_expands_to_an_expression() {
        let code = expansion(quote! { || X });
        assert!(code.starts_with("{ let (__arena , __root) ="), "{code}");
        let code = expansion(quote! { |r: f32| X - r });
        assert!(code.contains("fn __builder <"), "{code}");
        assert!(code.ends_with("__builder () }"), "{code}");
    }
}
