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
//! expansion on such an entry: the composed program is optimized when it is
//! baked (§1.8).
//!
//! [`ExprArena::apply`]: pixelflow_ir::arena::ExprArena::apply
//!
//! **Binding times** (docs/language/2026-09-25-the-language-is-kernel.md §1.4).
//! A host function takes each `f32`, count and record parameter by its
//! declared type, and a kernel-typed one as a `&Kernel`; every `f32`, count
//! and record one is a uniform: the kernel it returns declares one uniform
//! per scalar — a record parameter, one per field — in declaration order,
//! with the call's value as its default (a count's as `n as f32`, refused
//! past 2²⁴ first). So the call's values are the kernel's arguments, never
//! its constants: every call of an entry that takes no kernel is one
//! program, keyed and compiled once, and a compiled program is rebound per
//! call from the entry's `Args` record.
//!
//! [`Kernel`]: pixelflow_core::Kernel

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::optimize::Optimize;
use pixelflow_ir::{Bound, Lit, OpKind};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::BOUND_PLAN;
use crate::ast::{ConstItem, FnItem, RecordItem, Role, Spelling};
use crate::lower::{Expansion, Site, lower_entry};
use crate::sema::{AnalyzedKernel, ConstValue, ParameterKind, Scalar};
use pixelflow_ir::arena::Axis;
use pixelflow_ir::{Binder, Monoid};
use syn::ext::IdentExt;

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
/// and for each entry its host `fn`, taking its parameters by their declared
/// types — a kernel-typed one as a `&Kernel` — and its `Args` record,
/// refusing a count parameter past 2²⁴ before anything is built. An entry
/// that takes a kernel is [`Staged`]: lowered when its host function is
/// called.
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
                items.extend(emit_entry(entry, analyzed, &body));
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
fn entry_arena(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> Result<TokenStream, String> {
    let (arena, root) = Expansion::lower(entry, analyzed)?;
    let arguments: Vec<TokenStream> = analyzed
        .parameters(entry)
        .into_iter()
        .flat_map(|parameter| parameter.scalars())
        .map(argument)
        .collect();

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
    Ok(arena_to_tokens(&arena, root, &arguments))
}

/// What a scalar argument's value is in host code: `param`, `param.field`
/// for a record parameter's field, or `param as f32` for a count — the `f32`
/// its uniform holds, which the host has refused past 2²⁴ before this
/// conversion could make it another count. `write_into` reads each through
/// `self.`: `as` binds looser than a field access, so `self.n as f32` is
/// `(self.n) as f32`.
fn argument(scalar: Scalar<'_>) -> TokenStream {
    match scalar {
        Scalar::Value(param) => quote!(#param),
        Scalar::Count(param) => quote!(#param as f32),
        Scalar::Field(param, field) => quote!(#param.#field),
    }
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
    // Every parameter is an `f32`: the closure sugar has no records, `sema`
    // refuses a `bool` parameter of an entry, and a count is an entry's of
    // the items form.
    let names = entry.params.iter().map(|p| &p.name);
    quote! {
        move | #( #names: f32 ),* | -> ::pixelflow_core::Kernel { #body }
    }
}

/// An entry's expansion: a host function returning a `Kernel` — `body`,
/// statements that build it — taking its parameters by their declared types,
/// with the entry's visibility and doc comments. An entry with a count
/// parameter refuses one past 2²⁴ before building anything, and says so in a
/// `# Panics` section.
///
/// A kernel-typed parameter is taken as a `&Kernel`, not a `Kernel`: the
/// host function only reads it, splicing a copy of its term into the
/// program it builds, and a borrow passes one instance to two parameters,
/// `sum2(&a, &a)`, and halves a list into a tree by reference, as
/// `Kernel::at` and `Lattice::bake` take theirs. Owning it would buy the
/// argument's arena only by building on top of it, which would declare its
/// uniforms before the entry's own.
fn emit_entry(entry: &FnItem, analyzed: &AnalyzedKernel, body: &TokenStream) -> TokenStream {
    let attrs = &entry.attrs;
    let vis = &entry.vis;
    let name = &entry.name;
    let params = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        match p.is_kernel() {
            true => quote!(#name: &::pixelflow_core::Kernel),
            false => quote!(#name: #ty),
        }
    });
    let counts = count_parameters(entry, analyzed);
    let refusals = counts.iter().map(|count| {
        let message = count_refusal(entry, count);
        quote! {
            ::core::assert!(
                ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..#count as u64)),
                #message,
                #count
            );
        }
    });
    let panics = (!counts.is_empty()).then(|| {
        panics_section(&format!(
            " If a count argument is past 2^24: {COUNT_DOMAIN} (§1.4 of {BOUND_PLAN})."
        ))
    });
    quote! {
        #(#attrs)*
        #panics
        #[must_use]
        #vis fn #name ( #(#params),* ) -> ::pixelflow_core::Kernel {
            #(#refusals)*
            #body
        }
    }
}

/// A `# Panics` section as doc attributes, for a generated item: `text` is
/// its one paragraph, beginning with a space as a doc comment's text does.
fn panics_section(text: &str) -> TokenStream {
    quote! { #[doc = ""] #[doc = " # Panics"] #[doc = ""] #[doc = #text] }
}

/// An entry's count parameters, in declaration order.
fn count_parameters<'a>(entry: &'a FnItem, analyzed: &'a AnalyzedKernel) -> Vec<&'a Ident> {
    analyzed
        .parameters(entry)
        .into_iter()
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Count))
        .map(|parameter| parameter.name)
        .collect()
}

/// Why the host refuses a count past 2²⁴ where it writes one: the domain of
/// a count. Written as the `f32` its uniform holds, `2^24 + 1` is `2^24`, a
/// count, and a fold bounded by it would run one trip short with no NaN to
/// say so, which only the host can refuse (plan §1.4).
const COUNT_DOMAIN: &str = "a count is an integer in `0..=2^24`; `2^24 + 1` would be written as \
    `2^24`, a fold bounded by it would run one trip short with no NaN, and a larger count would \
    name none";

/// The refusal of a count past 2²⁴ where the host writes it, as a format
/// string with one `{}`, the count.
fn count_refusal(entry: &FnItem, param: &Ident) -> String {
    let bound = pixelflow_ir::Fold::EXACT_BOUND;
    format!(
        "kernel!: `{entry}`'s count `{param}` is {{}}, past {bound} (2^24): {COUNT_DOMAIN} \
         (§1.4 of {BOUND_PLAN})",
        entry = entry.name.unraw(),
        param = param.unraw(),
    )
}

/// An entry's `Args` record (plan §1.4): its parameters as fields, by their
/// declared types (a count's is `usize`); and `write_into`, which refuses a
/// count past 2²⁴ and then writes them — a record's fields in field order —
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
    let counts = count_parameters(entry, analyzed);
    let refusals = counts.iter().map(|count| {
        let message = count_refusal(entry, count);
        quote! {
            ::core::assert!(
                ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..self.#count as u64)),
                #message,
                self.#count
            );
        }
    });
    let panics = (!counts.is_empty()).then(|| {
        panics_section(&format!(
            " If a count in this record is past 2^24: {COUNT_DOMAIN} (§1.4 of {BOUND_PLAN}). \
             Nothing is written."
        ))
    });
    quote! {
        #[doc = #record_doc]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #vis struct #args {
            #(#fields)*
        }

        impl #args {
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
            /// written. Of the block, only the number of arguments is
            /// checked: a block of another program that declares as many
            /// takes these values by position (plan §1.4).
            #panics
            #vis fn write_into(
                &self,
                block: &mut ::pixelflow_core::UniformBlock,
            ) -> ::core::result::Result<(), ::pixelflow_core::ArityMismatch> {
                #(#refusals)*
                block.set_declared([ #(#values),* ])
            }
        }
    }
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
/// See docs/archive/plans/2026-09-08-macro-tier-is-arena-native.md.
///
/// The arena's uniform table is declared first, one per entry of
/// `arguments`, in order: each with an identity minted per call and the
/// call's value as its default, so the kernel's arguments are that call's
/// and its program is every call's — every one declared, read or not, so a
/// positional binding cannot shift.
///
/// # Panics
///
/// If the arena's uniform table is not `arguments` one for one:
/// lowering declares one uniform per argument, and a relink restores the
/// table after optimization, so a mismatch is a front-end bug, not a
/// kernel's.
pub fn arena_to_tokens(arena: &ExprArena, root: ExprId, arguments: &[TokenStream]) -> TokenStream {
    assert_eq!(
        arena.uniforms().len(),
        arguments.len(),
        "kernel! declared {} uniforms for {} arguments",
        arena.uniforms().len(),
        arguments.len()
    );
    let decls = arguments.iter().enumerate().map(|(slot, value)| {
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
        let expr = node(arena, id);
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
fn node(arena: &ExprArena, id: ExprId) -> TokenStream {
    match arena.node(id) {
        ExprNode::Var(i) => quote! { __arena.push_var(#i) },
        // By bit pattern, not as a decimal literal: `quote`'s `f32`
        // impl goes through `Literal::f32_suffixed`, which asserts
        // `is_finite()` — and non-finite constants are ordinary here. Bits
        // also roundtrip exactly, with no decimal-formatting question to get
        // wrong.
        ExprNode::Const(Lit::Real(v)) => {
            let bits = v.to_bits();
            quote! { __arena.push_const(f32::from_bits(#bits)) }
        }
        // A mask is its own literal: a folded comparison, or `BitAnd`'s
        // monoid identity and therefore `all_over`'s seed.
        ExprNode::Const(Lit::Mask(m)) => {
            quote! { __arena.push_lit(::pixelflow_core::__macro::ir::Lit::Mask(#m)) }
        }
        // A slot with no value: lowering builds none.
        ExprNode::Param(k) => {
            panic!("kernel! produced ExprNode::Param({k}): lowering builds none")
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
        // A fold's header — its fold and how its bounds are held — is a
        // `FoldHeader`, whose fields are private precisely so no caller can
        // assemble one that means nothing — so it travels the way the two
        // cache keys carry it, as bits with a total inverse on the far side.
        ExprNode::Reduce { lo, hi, body, .. } => {
            let header = arena.fold_header(id);
            let bits = header.to_bits();
            let body = node_var(body);
            let decoded = quote! {
                ::pixelflow_core::__macro::ir::fold::FoldHeader::from_bits(#bits)
                    .expect("kernel! emitted a well-formed fold")
            };
            // A bound that is a value is a child after the body, `lo`'s first,
            // as the header's bits say.
            let values: Vec<Ident> = [lo, hi]
                .into_iter()
                .filter_map(Bound::value)
                .map(node_var)
                .collect();
            let children = match values.as_slice() {
                [] => quote! { ::core::iter::empty() },
                _ => quote! { [#(#values),*] },
            };
            quote! {{
                let __header = #decoded;
                let (__lo, __hi) = __header.attach(#children);
                __arena.push_reduce(__header.fold(), __lo, __hi, #body)
            }}
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
    /// The folds open, innermost last: the variable holding each open fold,
    /// and the code building the `FoldTerms` it is closed with.
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

    /// The [`FoldTerms`](pixelflow_ir::arena::FoldTerms) an open fold closes
    /// over: its monoid, carried as a header's bits (as [`node`] carries a
    /// `FoldHeader`), and its bounds: a literal, or the variable holding a
    /// value bound's term, built before the fold opened. The binder is the
    /// one `close_fold` chooses.
    fn fold_terms(monoid: Monoid, bounds: [Bound<Ident>; 2]) -> TokenStream {
        let first = Binder::from_slot(0).expect("the IR has a binder");
        let bits = pixelflow_ir::FoldHeader::new(
            pixelflow_ir::Fold::new(monoid, first),
            pixelflow_ir::Bound::Lit(0),
            pixelflow_ir::Bound::Lit(0),
        )
        .to_bits();
        let [lo, hi] = bounds.map(|bound| match bound {
            Bound::Lit(count) => quote!(::pixelflow_core::__macro::ir::fold::Bound::Lit(#count)),
            Bound::Value(term) => quote!(::pixelflow_core::__macro::ir::fold::Bound::Value(#term)),
        });
        quote! {
            ::pixelflow_core::__macro::ir::arena::FoldTerms {
                monoid: ::pixelflow_core::__macro::ir::fold::FoldHeader::from_bits(#bits)
                    .expect("kernel! emitted a well-formed fold")
                    .fold()
                    .monoid(),
                lo: #lo,
                hi: #hi,
            }
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

    fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: [Bound<Ident>; 2]) -> Ident {
        let fold = self.fresh("__f");
        let depth = proc_macro2::Literal::usize_unsuffixed(depth);
        self.statements.push(quote! {
            let #fold = __arena
                .open_fold(#depth)
                .expect("kernel! refused a fold nested past the binders at expansion");
        });
        self.open
            .push((fold.clone(), Self::fold_terms(monoid, bounds)));
        self.step(quote!(#fold.index()))
    }

    fn close_fold(&mut self, body: Ident) -> Result<Ident, String> {
        let (fold, terms) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        Ok(self.step(quote! {
            __arena.close_fold(#fold, #body, #terms).expect(
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
    use pixelflow_ir::Fold;
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
    /// declared types, and streams them — the record's fields in field order —
    /// in declaration order.
    #[test]
    fn a_record_and_an_args_record_expand_to_host_structs() {
        let code = expansion(quote! {
            /// A box.
            pub struct Bounds { pub x0: f32, pub x1: f32 }
            pub fn inside(b: Bounds, r: f32) -> f32 {
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
            code.contains("pub fn inside (b : Bounds , r : f32)"),
            "{code}"
        );
        assert!(code.contains("pub struct InsideArgs {"), "{code}");
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

    /// An optimizer that counts the arenas it is offered, and declines each.
    struct Offered(usize);

    impl Optimize for Offered {
        fn optimize(&mut self, _arena: &ExprArena, _root: ExprId) -> pixelflow_ir::Rewritten {
            self.0 += 1;
            pixelflow_ir::Rewritten::Declined
        }
    }

    /// Every entry that takes no kernel is offered to the optimizer at
    /// expansion, one with a fold among them: nothing is exempt by its shape.
    #[test]
    fn every_entry_that_takes_no_kernel_is_offered_to_the_optimizer() {
        let analyzed = analyze(
            parse(quote! {
                const N: usize = 4;
                pub fn counted(r: f32) -> f32 {
                    r + (0..N).map(|i| (i as f32) * X).sum::<f32>()
                }
                pub fn plain(r: f32) -> f32 { r + X }
            })
            .expect("parses"),
        )
        .expect("analyzes");
        let mut offered = Offered(0);
        emit_kernel(&analyzed, &mut offered).expect("emits");
        assert_eq!(offered.0, 2, "`counted` and `plain`");
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

    /// A count parameter is the host function's declared `usize`, refused past
    /// 2²⁴ before anything is built; its uniform's default and its `Args`
    /// value are `n as f32`, and `write_into` refuses before it writes. A
    /// staged fold closes over its count's term.
    #[test]
    fn a_count_parameter_is_refused_on_the_host_and_declared_as_an_f32() {
        let code = expansion(quote! {
            pub fn sum_to(n: usize, s: f32) -> f32 {
                (0..n).map(|i| X * (i as f32)).sum::<f32>() * s
            }
            pub fn columns(k: impl Fn(f32, f32) -> f32, n: usize) -> f32 {
                (0..n).map(|i| k(X + (i as f32), Y)).sum()
            }
        });
        for expected in [
            "pub fn sum_to (n : usize , s : f32)",
            "KnownRange :: admits (& (0 .. n as u64))",
            "past 16777216 (2^24)",
            "§1.4 of docs/language/2026-10-20-a-bound-is-a-value.md",
            "# Panics",
            "default : n as f32",
            "pub n : usize",
            "self . n as f32 , self . s",
            "Bound :: Value (__t",
        ] {
            assert!(code.contains(expected), "expected `{expected}`: {code}");
        }
        let refused = code
            .find("KnownRange :: admits (& (0 .. self . n as u64))")
            .expect("write_into refuses");
        let written = code
            .find("block . set_declared")
            .expect("write_into writes");
        assert!(
            refused < written,
            "refused before anything is written: {code}"
        );
        let args = &code[code
            .find("impl SumToArgs")
            .expect("the `Args` record's impl")..];
        let write_into = &args[..args.find("fn write_into").expect("write_into")];
        assert!(
            write_into.contains("# Panics")
                && write_into.contains("a count in this record is past 2^24")
                && write_into.contains(COUNT_DOMAIN),
            "write_into documents its panic: {write_into}"
        );
        assert!(
            !code.contains("ColumnsArgs"),
            "an entry that takes a kernel has no `Args`: {code}"
        );
        assert!(!code.contains(":: std ::"), "{code}");
    }

    /// `Σ_{k<hi} k` as an arena, the fold's id, and the uniform that is its
    /// `hi` when it is a value.
    fn sum_to(count: Option<u64>) -> (ExprArena, ExprId, ExprId) {
        let mut arena = ExprArena::new();
        let slot = arena.declare_uniform(pixelflow_ir::Uniform::new(3.0).decl());
        let uniform = arena.push_uniform(slot);
        let binder = pixelflow_ir::Binder::from_slot(0).expect("the first binder");
        let index = arena.push_var(binder.var());
        let hi = count.map_or(Bound::Value(uniform), Bound::Lit);
        let fold = arena.push_reduce(
            Fold::new(pixelflow_ir::Monoid::SUM, binder),
            Bound::Lit(0),
            hi,
            index,
        );
        (arena, fold, uniform)
    }

    /// A fold with a bound that is a value rebuilds it with the bound's
    /// child, which the header's bits cannot hold.
    #[test]
    fn a_value_bounded_fold_emits_its_children() {
        let (arena, fold, uniform) = sum_to(None);
        let code = node(&arena, fold).to_string();
        let child = node_var(uniform);
        assert!(code.contains(&format!("attach ([{child}])")), "{code}");
    }

    /// A fold whose bounds are literals emits what it always did: no children.
    #[test]
    fn a_literal_fold_emits_no_children() {
        let (arena, fold, _) = sum_to(Some(3));
        let code = node(&arena, fold).to_string();
        assert!(
            code.contains("attach (:: core :: iter :: empty ())"),
            "{code}"
        );
    }
}
