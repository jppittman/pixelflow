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
//! a `Kernel`, or a closure over its parameters' `f32`s returning one. The
//! items form is items: a host `#[repr(C)]` struct per record, a host `const`
//! per `pub const`, and per entry a host `fn` returning a `Kernel` and, when
//! it has uniform parameters and no kernel-typed one, its `Args` record. A
//! helper is inlined and a private `const` is folded, so neither leaves a
//! trace.
//!
//! **An entry that takes a kernel is staged** (plan Phase D-a). Its
//! argument, `k: impl Fn(f32, f32) -> f32`, is a `&Kernel` the host passes
//! when it calls the entry's host function, so the program does not exist
//! at expansion: the host function *is* lowering, run then. [`Staged`] is
//! the [`Site`] that writes each of lowering's steps as the statement that
//! takes it — the same IR call [`Expansion`] makes now, in the same order —
//! and `k(x, y)` is one more, [`ExprArena::apply`]. No optimizer runs at
//! expansion on such an entry, as on a template: the composed program is
//! optimized when it is baked (§1.8).
//!
//! [`ExprArena::apply`]: pixelflow_ir::arena::ExprArena::apply
//!
//! **Binding times** (docs/plans/2026-09-25-the-language-is-kernel.md §1.4).
//! A host function takes its parameters by their declared types, and every
//! one is a uniform: the kernel it returns declares one uniform per scalar —
//! a record parameter, one per field — in declaration order, with the call's
//! value as its default. So the call's values are the kernel's arguments,
//! never its constants: every call of an entry is one program, keyed and
//! compiled once, and a compiled program is rebound per call from the
//! entry's `Args` record. An entry's structural parameters (`const N:
//! usize`) are the host function's const generics, and each value is its
//! own program.
//!
//! [`Kernel`]: pixelflow_core::Kernel

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::optimize::Optimize;
use pixelflow_ir::{Fold, OpKind};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::ast::{ConstItem, FnItem, RecordItem, Role, Spelling};
use crate::lower::{Expansion, FoldRange, Holes, Lowered, Site, lower_entry};
use crate::sema::{AnalyzedKernel, ConstValue, Scalar};
use pixelflow_ir::arena::Axis;
use pixelflow_ir::{Binder, Monoid};

/// Emit arena-backend code for an analyzed kernel.
///
/// For the closure form, a token stream evaluating to:
/// - zero params — a [`Kernel`](pixelflow_core::Kernel) value, built at load
///   time from the arena this expansion computed.
/// - N params — a closure `move |p0: f32, ...| -> Kernel` whose kernel
///   declares one uniform per parameter, its default the call's argument
///   (no JIT — leaves are bake-time-only and fuse at a root, which is what
///   lets a font build thousands of leaf kernels and compile one arena).
///   The closure sugar has no `Args` record: a program compiled from it is
///   rebound by position, with `UniformBlock::set_declared`.
///
/// For the items form: one struct per record, one `const` per `pub const`,
/// and for each entry its host `fn`, generic over its structural parameters
/// and taking its parameters by their declared types — a kernel-typed one as
/// a `&Kernel` — and its `Args` record. An entry that takes a kernel is
/// [`Staged`]: lowered when its host function is called.
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
            for record in &analyzed.def.records {
                items.extend(emit_record(record));
            }
            for c in analyzed.def.consts.iter().filter(|c| is_pub(&c.vis)) {
                items.extend(emit_const(c, analyzed.consts[&c.name.to_string()]));
            }
            for entry in analyzed.def.fns.iter().filter(|f| f.role() == Role::Entry) {
                let body = match entry.takes_a_kernel() {
                    true => Staged::lower(entry, analyzed)?,
                    false => {
                        let arena_code = entry_arena(entry, analyzed, optimizer)?;
                        quote! {
                            let (__arena, __root) = #arena_code;
                            ::pixelflow_core::Kernel::from_parts(__arena, __root)
                        }
                    }
                };
                items.extend(emit_entry(entry, &body));
                items.extend(emit_args(entry, analyzed));
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
///
/// An entry with structural parameters is a template: its open folds hold
/// placeholder ranges a rewrite would read as ranges (an empty one, to
/// `EmptyFold`), so it is emitted as lowered and optimized when its
/// instantiation is baked — "no optimization runs at expansion unless the
/// instance is declared" (plan §1.8, Phase E).
fn entry_arena(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> Result<TokenStream, String> {
    let Lowered { arena, root, holes } = Expansion::lower(entry, analyzed)?;
    let context = EntryContext {
        arguments: analyzed
            .parameters(entry)
            .into_iter()
            .flat_map(|parameter| parameter.scalars())
            .map(argument)
            .collect(),
        structural: &entry.structural,
        holes: &holes,
    };
    if !entry.structural.is_empty() {
        return Ok(arena_to_tokens(&arena, root, &context));
    }

    // Declining is ordinary and needs no arm: the lowered term stands, and a
    // kernel that reaches the runtime tier unoptimized is optimized there.
    let (arena, root) = match optimizer.optimize(&arena, root).into_changed() {
        // Extraction declares the uniforms it reaches in its own walk order
        // and drops the ones it does not; relinking puts back the entry's
        // declaration order, every scalar in it, which is the order a
        // positional binding supplies them in.
        Some((optimized, optimized_root)) if !arena.uniforms().is_empty() => {
            optimized.relink(optimized_root, arena.buffers(), arena.uniforms())
        }
        Some(changed) => changed,
        None => (arena, root),
    };
    Ok(arena_to_tokens(&arena, root, &context))
}

/// What a scalar argument is in the host function: `param`, or
/// `param.field` for a record parameter's field.
fn argument(scalar: Scalar<'_>) -> TokenStream {
    let param = scalar.param;
    match scalar.field {
        None => quote!(#param),
        Some(field) => quote!(#param.#field),
    }
}

/// What an entry's arena is emitted against: the argument each declared
/// uniform defaults to, in declaration order, and — for a template — the
/// structural parameters and open ranges its holes are filled from.
pub struct EntryContext<'a> {
    arguments: Vec<TokenStream>,
    structural: &'a [Ident],
    holes: &'a Holes,
}

/// The closure form's expansion: a `Kernel`, or a closure over `f32`s.
fn emit_closure(entry: &FnItem, arena_code: &TokenStream) -> TokenStream {
    let body = quote! {
        let (__arena, __root) = #arena_code;
        ::pixelflow_core::Kernel::from_parts(__arena, __root)
    };
    if entry.params.is_empty() {
        return quote! {{ #body }};
    }
    // Every parameter is an `f32`: the closure sugar has no records, and
    // `sema` refuses a `bool` parameter of an entry.
    let names = entry.params.iter().map(|p| &p.name);
    quote! {
        move | #( #names: f32 ),* | -> ::pixelflow_core::Kernel { #body }
    }
}

/// An entry's expansion: a host function returning a `Kernel` — `body`,
/// statements that build it — generic over its structural parameters and
/// taking its parameters by their declared types, with the entry's
/// visibility and doc comments.
///
/// A kernel-typed parameter is taken as a `&Kernel`, not a `Kernel`: the
/// host function only reads it, splicing a copy of its term into the
/// program it builds, and a borrow passes one instance to two parameters,
/// `sum2(&a, &a)`, and halves a list into a tree by reference, as
/// `Kernel::at` and `Lattice::bake` take theirs. Owning it would buy the
/// argument's arena only by building on top of it, which would declare its
/// uniforms before the entry's own.
fn emit_entry(entry: &FnItem, body: &TokenStream) -> TokenStream {
    let attrs = &entry.attrs;
    let vis = &entry.vis;
    let name = &entry.name;
    let (generics, _) = structural_generics(&entry.structural);
    let params = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        match p.is_kernel() {
            true => quote!(#name: &::pixelflow_core::Kernel),
            false => quote!(#name: #ty),
        }
    });
    quote! {
        #(#attrs)*
        #[must_use]
        #vis fn #name #generics ( #(#params),* ) -> ::pixelflow_core::Kernel {
            #body
        }
    }
}

/// An entry's `Args` record (plan §1.4): its parameters as fields, by their
/// declared types, generic over its structural parameters; and
/// `write_into`, which writes them — a record's fields in field order —
/// into a block by the entry's declaration order
/// (`UniformBlock::set_declared`), as an array literal whose length rustc
/// counts. An entry with no parameters has none, and neither has one that
/// takes a kernel ([`FnItem::has_args_record`]).
fn emit_args(entry: &FnItem, analyzed: &AnalyzedKernel) -> TokenStream {
    if !entry.has_args_record() {
        return TokenStream::new();
    }
    let vis = &entry.vis;
    let args = entry.args_record();
    let (generics, arguments) = structural_generics(&entry.structural);
    let entry_name = entry.name.to_string();
    let record_doc = format!(
        "The arguments of `{entry_name}`: a program compiled from its kernel is rebound per \
         call from one of these, with [`{args}::write_into`]."
    );
    let fields = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        let doc = format!("The argument `{name}` of `{entry_name}`.");
        quote! {
            #[doc = #doc]
            #vis #name: #ty,
        }
    });
    let values = analyzed
        .parameters(entry)
        .into_iter()
        .flat_map(|parameter| parameter.scalars())
        .map(|scalar| {
            let value = argument(scalar);
            quote!(self.#value)
        });
    quote! {
        #[doc = #record_doc]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #vis struct #args #generics {
            #(#fields)*
        }

        impl #generics #args #arguments {
            /// This call's values, written into `block` — one made by a
            /// program compiled from this entry's kernel
            /// (`Manifold::block`), kept and rewritten call after call:
            /// every uniform scalar in declaration order, a record's fields
            /// in field order.
            ///
            /// # Errors
            ///
            /// When `block`'s program declares a different number of
            /// arguments — it was compiled from some other kernel, or from
            /// this one composed beside other arguments. Nothing is
            /// written. The count is all that is checked: a block of
            /// another program that declares as many arguments takes these
            /// values by position (plan §1.4).
            #vis fn write_into(
                &self,
                block: &mut ::pixelflow_core::UniformBlock,
            ) -> ::core::result::Result<(), ::pixelflow_core::ArityMismatch> {
                block.set_declared([ #(#values),* ])
            }
        }
    }
}

/// An entry's structural parameters as generics: their declaration,
/// `<const N: usize, …>`, and their use, `<N, …>` — or nothing, for an
/// entry with none.
fn structural_generics(structural: &[Ident]) -> (TokenStream, TokenStream) {
    if structural.is_empty() {
        return (TokenStream::new(), TokenStream::new());
    }
    (
        quote!(< #( const #structural: usize ),* >),
        quote!(< #( #structural ),* >),
    )
}

/// A record's host twin (§1.3): a `#[repr(C)]` struct of the same name,
/// fields and visibility, with its attributes and its fields' — the type an
/// entry's parameter and its `Args` record name.
fn emit_record(record: &RecordItem) -> TokenStream {
    let attrs = &record.attrs;
    let vis = &record.vis;
    let name = &record.name;
    let fields = record.fields.iter().map(|field| {
        let (attrs, vis, name, ty) = (&field.attrs, &field.vis, &field.name, &field.ty);
        quote! {
            #(#attrs)*
            #vis #name: #ty,
        }
    });
    quote! {
        #(#attrs)*
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #vis struct #name {
            #(#fields)*
        }
    }
}

/// A `pub const`'s host twin, holding the value `sema` evaluated. An `f32`
/// by bit pattern, for the reason [`arena_to_tokens`] gives: it is exact,
/// and a const may be non-finite (`1.0 / 0.0`), which a decimal literal
/// cannot spell. A `usize` as the integer it is.
fn emit_const(item: &ConstItem, value: ConstValue) -> TokenStream {
    let attrs = &item.attrs;
    let vis = &item.vis;
    let name = &item.name;
    match value {
        ConstValue::F32(value) => {
            let bits = value.to_bits();
            quote! {
                #(#attrs)*
                #vis const #name: f32 = f32::from_bits(#bits);
            }
        }
        ConstValue::Usize(count) => {
            let count = proc_macro2::Literal::u64_unsuffixed(count);
            quote! {
                #(#attrs)*
                #vis const #name: usize = #count;
            }
        }
    }
}

/// An open fold's range, evaluated when the host function is instantiated:
/// in a `const` block, so rustc checks each operation of the bounds as it
/// checks any `const`, and refuses — per instantiation — the ranges the IR
/// refuses ([`Fold::admits`]): one that runs backwards, and one past
/// [`Fold::EXACT_BOUND`], the last bound a fold's index names exactly.
/// The bound is the IR's, imported; what the `const` block adds is that the
/// refusal is rustc's rather than `Fold::new`'s panic.
///
/// The refusal is rustc's at monomorphization, which `cargo build` reaches
/// and `cargo check` (so clippy) does not: a bad instantiation checks clean
/// and fails to build, in the calling crate, naming the range. It is never
/// a run-time panic.
fn instantiated_range(range: &crate::sema::StructuralRange) -> TokenStream {
    let (lo, hi) = (&range.lo, &range.hi);
    let bound = proc_macro2::Literal::u32_unsuffixed(Fold::EXACT_BOUND);
    let backwards = format!(
        "kernel!: the range `{}` runs backwards at this instantiation",
        range.text()
    );
    let past = format!(
        "kernel!: the range `{}` reaches past 2^24 at this instantiation: a fold's index is an \
         `f32` lane, which names every integer only that far",
        range.text()
    );
    // The locals are the emission's own names, `__`-prefixed as its others
    // are: spelled `lo`, a structural parameter `lo` in scope turned the
    // binding into a pattern naming it, which rustc refuses (E0158).
    quote! {
        const {
            let __lo: usize = #lo;
            let __hi: usize = #hi;
            ::core::assert!(__lo <= __hi, #backwards);
            ::core::assert!(__hi <= #bound, #past);
            (__lo as u32)..(__hi as u32)
        }
    }
}

/// Emit the arena as code that rebuilds it at load time: every node the
/// root reaches, in the arena's order, children before parents.
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
///
/// The arena's uniform table is declared first, one per argument of
/// `context`, in order: each with an identity minted per call and the
/// call's value as its default, so the kernel's arguments are that call's
/// and its program is every call's — every one declared, read or not, so a
/// positional binding cannot shift. A template's holes are filled from the
/// host function's structural parameters: a `Param(k)` is the `k`th one as
/// an `f32`, and an open fold's range is evaluated per instantiation
/// ([`instantiated_range`]).
///
/// # Panics
///
/// If the arena's uniform table is not `context`'s arguments one for one:
/// lowering declares one uniform per argument, and a relink restores the
/// table after optimization, so a mismatch is a front-end bug, not a
/// kernel's.
pub fn arena_to_tokens(arena: &ExprArena, root: ExprId, context: &EntryContext) -> TokenStream {
    assert_eq!(
        arena.uniforms().len(),
        context.arguments.len(),
        "kernel! declared {} uniforms for {} arguments",
        arena.uniforms().len(),
        context.arguments.len()
    );
    let decls = context.arguments.iter().enumerate().map(|(slot, value)| {
        let slot = uniform_var(slot as u64);
        quote! {
            let #slot = __arena.declare_uniform(::pixelflow_core::__macro::ir::arena::UniformDecl {
                id: ::pixelflow_core::__macro::ir::arena::UniformIdentity::mint(),
                default: #value,
            });
        }
    });
    let nodes = reached(arena, root).into_iter().map(|id| {
        let var = node_var(id);
        let expr = node(arena, id, context);
        quote! {
            let #var = #expr;
        }
    });
    let root_ident = node_var(root);
    quote! {{
        let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
        #(#decls)*
        #(#nodes)*
        (__arena, #root_ident)
    }}
}

/// The variable holding a uniform slot's `UniformId` in the arena being
/// built.
fn uniform_var(slot: u64) -> Ident {
    format_ident!("__u{slot}")
}

/// The variable holding a node's id in the arena being built.
fn node_var(id: ExprId) -> Ident {
    format_ident!("__e{}", id.0)
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
    (0..arena.len())
        .filter(|index| reached[*index])
        .map(|index| ExprId(index as u32))
        .collect()
}

/// One node, built in `__arena` from the nodes before it.
fn node(arena: &ExprArena, id: ExprId, context: &EntryContext) -> TokenStream {
    match arena.node(id) {
        ExprNode::Var(i) => quote! { __arena.push_var(#i) },
        // By bit pattern, not as a decimal literal: `quote`'s `f32`
        // impl goes through `Literal::f32_suffixed`, which asserts
        // `is_finite()` — and non-finite constants are ordinary here. A
        // true comparison mask is all-ones (`OpKind::mask`), which is
        // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
        // the folder now produces those. Bits also roundtrip exactly, with
        // no decimal-formatting question to get wrong.
        ExprNode::Const(v) => {
            let bits = v.to_bits();
            quote! { __arena.push_const(f32::from_bits(#bits)) }
        }
        // A template's hole: the structural parameter, as its `f32`, as
        // Rust's `as` rounds it — which is `N as f32` in the body.
        ExprNode::Param(k) => {
            let count = context.structural.get(usize::from(k)).unwrap_or_else(|| {
                panic!(
                    "kernel! produced ExprNode::Param({k}) outside a template's \
                     structural parameters"
                )
            });
            quote! { __arena.push_const(#count as f32) }
        }
        // The `kernel!` macro has no buffer surface yet, so this is
        // unreachable in practice; fail loud rather than emit a node that
        // references a buffer table `from_raw` does not reconstruct.
        ExprNode::Buffer(b) => {
            panic!(
                "kernel! produced ExprNode::Buffer({}) — lattice parameters are not wired \
                 into the compiler yet (KERNELS_AND_LATTICES.md M4)",
                b.0
            )
        }
        // An argument: the slot declared above for it.
        ExprNode::Uniform(u) => {
            let slot = uniform_var(u.0);
            quote! { __arena.push_uniform(#slot) }
        }
        // And a reference is minted by `Kernel::by_ref` at composition
        // time — a runtime value, and the key it carries names a store
        // in the *build host's* process, which the compiled program is
        // not. Emitting one would name nothing.
        ExprNode::Ref(k) => {
            panic!(
                "kernel! produced ExprNode::Ref({k:?}) — a reference names a kernel \
                 interned in this process, which the emitted program does not share"
            )
        }
        ExprNode::Unary(op, child) => {
            let op_code = opkind_to_tokens(op);
            let child = node_var(child);
            quote! { __arena.push_unary(#op_code, #child) }
        }
        ExprNode::Binary(op, a, b) => {
            let op_code = opkind_to_tokens(op);
            let (a, b) = (node_var(a), node_var(b));
            quote! { __arena.push_binary(#op_code, #a, #b) }
        }
        ExprNode::Ternary(op, a, b, c) => {
            let op_code = opkind_to_tokens(op);
            let (a, b, c) = (node_var(a), node_var(b), node_var(c));
            quote! { __arena.push_ternary(#op_code, #a, #b, #c) }
        }
        ExprNode::Nary(op, ..) => {
            let op_code = opkind_to_tokens(op);
            let children: Vec<Ident> = arena.children(id).map(node_var).collect();
            quote! { __arena.push_nary(#op_code, &[#(#children),*]) }
        }
        // A fold's metadata is a `Fold`, whose fields are private
        // precisely so no caller can assemble one that means nothing —
        // so it travels the way the two cache keys carry it, as bits
        // with a total inverse on the far side.
        ExprNode::Reduce { fold, body } => {
            let bits = fold.to_bits();
            let body = node_var(body);
            let decoded = quote! {
                ::pixelflow_core::__macro::ir::fold::Fold::from_bits(#bits)
                    .expect("kernel! emitted a well-formed fold")
            };
            // An open fold keeps its monoid and binder, which are
            // structure, and takes its range from this instantiation.
            let emitted = match context.holes.range_of(fold) {
                None => decoded,
                Some(range) => {
                    let range = instantiated_range(range);
                    quote! {{
                        let __placeholder = #decoded;
                        ::pixelflow_core::__macro::ir::fold::Fold::new(
                            __placeholder.monoid(),
                            __placeholder.binder(),
                            #range,
                        )
                    }}
                }
            };
            quote! { __arena.push_reduce(#emitted, #body) }
        }
        // Unreachable for the same reason `Ref` is: there is no
        // `kernel!` surface syntax for a hard branch. `Guard` is built
        // directly against an `ExprArena` (`ExprArena::push_guard`), not
        // lowered from a macro body — and even if it were, its `on`/
        // `off` keys would name kernels interned in the *build host's*
        // process, which the emitted program does not share, exactly as
        // `Ref`'s panic says.
        ExprNode::Guard { mask, on, off } => {
            panic!(
                "kernel! produced ExprNode::Guard(mask={mask:?}, on={on:?}, off={off:?}) \
                 — there is no surface syntax for a hard branch yet; it is built directly \
                 against an ExprArena, not lowered from a kernel! body"
            )
        }
        // A store is post-legalize vocabulary: the passes that wrap a
        // kernel in the lattice's folds build one, after extraction,
        // and no kernel! body can spell it.
        ExprNode::Write { .. } => {
            panic!(
                "kernel! produced ExprNode::Write — a store has no surface syntax; \
                 the legalize passes build one after extraction"
            )
        }
    }
}

/// The site an entry that takes a kernel is lowered at: the Rust statements
/// that build its arena when its host function is called (plan Phase D-a).
///
/// Each of lowering's steps becomes the statement that takes it, naming a
/// term by the variable that holds its node: the same IR call [`Expansion`]
/// makes at expansion — a push, a [`library`](pixelflow_ir::library)
/// definition, [`ExprArena::open_fold`] and [`ExprArena::close_fold`] — in
/// the same order, so a program built at load time is the one lowering
/// would have built at expansion. The steps only `Staged` has are an
/// argument's: [`ExprArena::admit`] when the host function is called, after
/// the entry's own uniforms, and [`ExprArena::apply`] at each `k(x, y)`.
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
pub struct Staged {
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
    pub fn lower(entry: &FnItem, analyzed: &AnalyzedKernel) -> Result<TokenStream, String> {
        let mut site = Staged::default();
        let root = lower_entry(entry, analyzed, &mut site)?;
        let statements = site.statements;
        Ok(quote! {
            let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
            #(#statements)*
            let (__arena, __root) = __arena.compact(#root);
            ::pixelflow_core::Kernel::from_parts(__arena, __root)
        })
    }

    /// A fresh variable, `prefix` then a number.
    fn fresh(&mut self, prefix: &str) -> Ident {
        let name = format_ident!("{prefix}{}", self.named);
        self.named += 1;
        name
    }

    /// The statement binding `step`'s node to a fresh variable, which names
    /// it.
    fn step(&mut self, step: TokenStream) -> Ident {
        let term = self.fresh("__t");
        self.statements.push(quote! { let #term = #step; });
        term
    }

    /// The closure an open fold closes into: its monoid, carried as a
    /// fold's bits (as [`node`] carries a `Fold`), its binder the one
    /// `close_fold` chooses, and its range — known, or this
    /// instantiation's ([`instantiated_range`]).
    fn fold_at(monoid: Monoid, range: FoldRange) -> TokenStream {
        let first = Binder::from_slot(0).expect("the IR has a binder");
        let bits = pixelflow_ir::Fold::new(monoid, first, 0..0).to_bits();
        let range = match range {
            FoldRange::Known(range) => {
                let (lo, hi) = (range.start, range.end);
                quote!(#lo..#hi)
            }
            FoldRange::Structural(range) => instantiated_range(&range),
        };
        quote! {
            |__binder| ::pixelflow_core::__macro::ir::fold::Fold::new(
                ::pixelflow_core::__macro::ir::fold::Fold::from_bits(#bits)
                    .expect("kernel! emitted a well-formed fold")
                    .monoid(),
                __binder,
                #range,
            )
        }
    }
}

impl Site for Staged {
    type Term = Ident;
    type Argument = Ident;

    fn constant(&mut self, value: f32) -> Ident {
        // By bit pattern, for `node`'s reason.
        let bits = value.to_bits();
        self.step(quote!(__arena.push_const(f32::from_bits(#bits))))
    }

    fn coordinate(&mut self, axis: Axis) -> Ident {
        let var = axis.var();
        self.step(quote!(__arena.push_var(#var)))
    }

    fn unary(&mut self, op: pixelflow_ir::OpKind, a: Ident) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(quote!(__arena.push_unary(#op, #a)))
    }

    fn binary(&mut self, op: pixelflow_ir::OpKind, [a, b]: [Ident; 2]) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(quote!(__arena.push_binary(#op, #a, #b)))
    }

    fn ternary(&mut self, op: pixelflow_ir::OpKind, [a, b, c]: [Ident; 3]) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(quote!(__arena.push_ternary(#op, #a, #b, #c)))
    }

    fn fract(&mut self, x: Ident) -> Ident {
        self.step(quote!(::pixelflow_core::__macro::ir::library::fract(&mut __arena, #x)))
    }

    fn hypot(&mut self, [a, b]: [Ident; 2]) -> Ident {
        self.step(quote!(::pixelflow_core::__macro::ir::library::hypot(&mut __arena, [#a, #b])))
    }

    fn clamp(&mut self, x: Ident, [lo, hi]: [Ident; 2]) -> Ident {
        self.step(quote!(
            ::pixelflow_core::__macro::ir::library::clamp(&mut __arena, #x, [#lo, #hi])
        ))
    }

    fn derivative(&mut self, e: Ident, axis: Axis) -> Ident {
        let axis = match axis {
            Axis::X => quote!(::pixelflow_core::__macro::ir::arena::Axis::X),
            Axis::Y => quote!(::pixelflow_core::__macro::ir::arena::Axis::Y),
        };
        self.step(quote!(
            ::pixelflow_core::__macro::ir::library::derivative(&mut __arena, #e, #axis)
        ))
    }

    /// Declared with the call's value as its default, as [`arena_to_tokens`]
    /// declares an expanded entry's, so baking the kernel draws the call.
    fn uniform(&mut self, scalar: Scalar<'_>) -> Ident {
        let slot = self.fresh("__u");
        let value = argument(scalar);
        self.statements.push(quote! {
            let #slot = __arena.declare_uniform(::pixelflow_core::__macro::ir::arena::UniformDecl {
                id: ::pixelflow_core::__macro::ir::arena::UniformIdentity::mint(),
                default: #value,
            });
        });
        self.step(quote!(__arena.push_uniform(#slot)))
    }

    fn kernel_parameter(&mut self, name: &Ident) -> Result<Ident, String> {
        let argument = self.fresh("__k");
        self.statements
            .push(quote! { let #argument = __arena.admit(#name); });
        Ok(argument)
    }

    /// The host function's own `N as f32`: it is generic over `N`.
    fn count(&mut self, _position: usize, name: &Ident) -> Result<Ident, String> {
        Ok(self.step(quote!(__arena.push_const(#name as f32))))
    }

    fn open_fold(
        &mut self,
        depth: usize,
        monoid: Monoid,
        range: FoldRange,
    ) -> Result<Ident, String> {
        let fold = self.fresh("__f");
        let depth = proc_macro2::Literal::usize_unsuffixed(depth);
        self.statements.push(quote! {
            let #fold = __arena
                .open_fold(#depth)
                .expect("kernel! refused a fold nested past the binders at expansion");
        });
        self.open.push((fold.clone(), Self::fold_at(monoid, range)));
        Ok(self.step(quote!(#fold.index())))
    }

    fn close_fold(&mut self, body: Ident) -> Result<Ident, String> {
        let (fold, fold_at) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        Ok(self.step(quote! {
            __arena.close_fold(#fold, #body, #fold_at).expect(
                "kernel!: a fold around a kernel's application has no binder left: its body, \
                 the argument's folds among them, binds every one of the IR's indices \
                 (`Binder::COUNT`)"
            )
        }))
    }

    fn apply(&mut self, kernel: &Ident, [x, y]: [Ident; 2]) -> Ident {
        self.step(quote!(__arena.apply(&#kernel, [#x, #y])))
    }
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
        assert!(
            code.contains("# [must_use] pub fn circle (cx : f32) -> :: pixelflow_core :: Kernel"),
            "an entry takes its parameters by their declared types: {code}"
        );
        assert!(code.contains("pub struct CircleArgs"), "{code}");
        assert!(code.contains("pub (crate) fn plain () ->"), "{code}");
        assert!(
            !code.contains("PlainArgs"),
            "an entry with no parameters has no `Args` record: {code}"
        );
        assert!(
            !code.contains("Scalar") && !code.contains("Into"),
            "no call-site type decides anything: {code}"
        );
    }

    /// The closure form expands to an expression: a `Kernel`, or a closure
    /// over `f32`s whose kernel declares each as a uniform with the call's
    /// value.
    #[test]
    fn the_closure_form_expands_to_an_expression() {
        let code = expansion(quote! { || X });
        assert!(code.starts_with("{ let (__arena , __root) ="), "{code}");
        let code = expansion(quote! { |r: f32| X - r });
        assert!(
            code.starts_with("move | r : f32 | -> :: pixelflow_core :: Kernel"),
            "{code}"
        );
        assert!(code.contains("default : r"), "{code}");
    }

    /// A record is a host `#[repr(C)]` struct with its docs, fields and
    /// visibility; an entry's `Args` record holds its parameters by their
    /// declared types, generic over its structural parameters, and streams
    /// them — the record's fields in field order — in declaration order.
    #[test]
    fn a_record_and_an_args_record_expand_to_host_structs() {
        let code = expansion(quote! {
            /// A box.
            pub struct Bounds { pub x0: f32, pub x1: f32 }
            pub fn inside<const N: usize>(b: Bounds, r: f32) -> f32 {
                if (X > b.x0) & (X < b.x1) { r } else { 0.0 }
            }
        });
        assert!(
            code.contains(
                "A box.\"] # [repr (C)] # [derive (Clone , Copy , Debug , Default , PartialEq)] \
                 pub struct Bounds { pub x0 : f32 , pub x1 : f32 , }"
            ),
            "{code}"
        );
        assert!(
            code.contains("pub fn inside < const N : usize > (b : Bounds , r : f32)"),
            "{code}"
        );
        assert!(
            code.contains("pub struct InsideArgs < const N : usize >"),
            "{code}"
        );
        assert!(
            code.contains("block . set_declared ([self . b . x0 , self . b . x1 , self . r])"),
            "{code}"
        );
        assert!(
            code.contains("# [derive (Clone , Copy , Debug , PartialEq)] pub struct InsideArgs"),
            "{code}"
        );
        assert!(
            code.contains("default : b . x0") && code.contains("default : r"),
            "{code}"
        );
        // Spelled as `to_string` spaces a path: `"::std"` matches nothing.
        // `items_block.rs` expands a block in a `#[no_implicit_prelude]`
        // module.
        assert!(
            !code.contains(":: std ::"),
            "no `::std` path, so a `no_std` crate expands it: {code}"
        );
    }

    /// A template's open fold takes its range from the instantiation, in a
    /// `const` block that refuses what lowering refuses of a known range;
    /// its `N as f32` is the structural parameter's.
    #[test]
    fn a_templates_holes_are_filled_by_the_instantiation() {
        let code = expansion(quote! {
            pub fn mean<const N: usize>() -> f32 {
                (0..N).map(|i| X * (i as f32)).sum::<f32>() / (N as f32)
            }
        });
        assert!(
            code.contains("const { let __lo : usize = 0 ; let __hi : usize = N ;"),
            "{code}"
        );
        assert!(
            code.contains("runs backwards at this instantiation"),
            "{code}"
        );
        assert!(code.contains("__hi <= 16777216"), "{code}");
        assert!(code.contains("push_const (N as f32)"), "{code}");
    }

    /// An optimizer that counts the arenas it is offered, and declines each.
    struct Offered(usize);

    impl Optimize for Offered {
        fn optimize(&mut self, _arena: &ExprArena, _root: ExprId) -> pixelflow_ir::Rewritten {
            self.0 += 1;
            pixelflow_ir::Rewritten::Declined
        }
    }

    /// No optimizer sees a template, an entry with structural parameters:
    /// its open folds hold placeholder ranges a rewrite would read as
    /// ranges, so it is emitted as lowered and optimized when its
    /// instantiation is baked; an entry without one is offered.
    #[test]
    fn a_template_is_offered_to_no_optimizer() {
        let analyzed = analyze(
            parse(quote! {
                pub fn counted<const N: usize>(r: f32) -> f32 {
                    r + (0..N).map(|i| (i as f32) * X).sum::<f32>()
                }
                pub fn plain(r: f32) -> f32 { r + X }
            })
            .expect("parses"),
        )
        .expect("analyzes");
        let mut offered = Offered(0);
        emit_kernel(&analyzed, &mut offered).expect("emits");
        assert_eq!(offered.0, 1, "`plain` alone");
    }

    /// An entry that takes a kernel is staged: its host function takes the
    /// argument as a `&Kernel`, admits it after declaring its own uniforms,
    /// applies it where the body does, and compacts what it built. It has no
    /// `Args` record, and no optimizer sees it — its program exists only
    /// once the host function is called.
    #[test]
    fn an_entry_that_takes_a_kernel_is_staged() {
        let input = quote! {
            pub struct Bounds { pub x0: f32, pub x1: f32 }
            pub fn glyph(ink: impl Fn(f32, f32) -> f32, b: Bounds) -> f32 {
                if (X > b.x0) & (X < b.x1) { ink(X + 0.5, Y) } else { 0.0 }
            }
            pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
                a(X, Y) + b(X, Y)
            }
            pub fn plain(r: f32) -> f32 { r + X }
        };
        let code = expansion(input.clone());
        assert!(
            code.contains(
                "pub fn glyph (ink : & :: pixelflow_core :: Kernel , b : Bounds) -> :: \
                 pixelflow_core :: Kernel"
            ),
            "{code}"
        );
        assert!(
            code.contains(
                "pub fn sum2 (a : & :: pixelflow_core :: Kernel , b : & :: pixelflow_core :: \
                 Kernel)"
            ),
            "{code}"
        );
        let declared_last = code.find("default : b . x1").expect("b.x1 is declared");
        let admitted = code.find("__arena . admit (ink)").expect("ink is admitted");
        assert!(declared_last < admitted, "own uniforms first: {code}");
        assert!(code.contains("__arena . apply (&"), "{code}");
        assert!(code.contains("__arena . compact ("), "{code}");
        assert!(!code.contains("GlyphArgs"), "no `Args` record: {code}");
        assert!(!code.contains("Sum2Args"), "no `Args` record: {code}");
        assert!(code.contains("PlainArgs"), "{code}");
        assert!(
            !code.contains(":: std ::"),
            "no `::std` path, so a `no_std` crate expands it: {code}"
        );

        let analyzed = analyze(parse(input).expect("parses")).expect("analyzes");
        let mut offered = Offered(0);
        emit_kernel(&analyzed, &mut offered).expect("emits");
        assert_eq!(offered.0, 1, "`plain` alone");
    }
}
