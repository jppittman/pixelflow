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
//! it has parameters, its `Args` record. A helper is inlined and a private
//! `const` is folded, so neither leaves a trace.
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
//! **Families** (plan §1.6). A family `pieces: [Row; N]` is `N` elements'
//! uniforms, declared by the host function element-major, and an iteration
//! of it is its body's copies, one per element, made when the host function
//! runs: the body is emitted once, as an arena of its own — a *template*,
//! a function of an abstract element and of the terms every copy shares,
//! which are built once, outside it — and each copy is that template
//! spliced in with the element's uniforms and the shared terms in its
//! inputs' places, combined under the monoid as [`Chain`] combines
//! distinct terms. So the instantiation builds `O(N·|body|)` nodes and
//! allocates only arenas, and the program it builds has no fold, no binder
//! and no index for the family. Templates are built innermost first: one
//! whose body iterates a family itself holds that iteration's copies, and
//! is one instantiation's; Phase B6 closes each as it is built, before its
//! copies (plan §1.8).
//!
//! [`Kernel`]: pixelflow_core::Kernel
//! [`Chain`]: pixelflow_ir::Chain

use std::collections::{BTreeSet, HashMap, HashSet};

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode, UniformId};
use pixelflow_ir::optimize::Optimize;
use pixelflow_ir::{Binder, OpKind, RangeFold};
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use syn::Ident;

use crate::ast::{ConstItem, FnItem, Param, RecordItem, Role, Spelling};
use crate::lower::{self, Families, Holes, Lowered};
use crate::sema::{AnalyzedKernel, ConstValue, Parameter, Scalar};

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
/// and taking its parameters by their declared types, and its `Args` record.
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
            Ok(emit_closure(entry, analyzed, &arena_code))
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
                let arena_code = entry_arena(entry, analyzed, optimizer)?;
                items.extend(emit_entry(entry, analyzed, &arena_code));
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
/// instance is declared" (plan §1.8, Phase E). So is an entry with a
/// family, whose count is structural whatever it is written as: its
/// iterations' copies are made when its host function runs, and a family's
/// node read as a program is the NaN [`Families`] documents.
fn entry_arena(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> Result<TokenStream, String> {
    let Lowered {
        arena,
        root,
        holes,
        families,
    } = lower::lower_entry(entry, analyzed)?;
    let parameters = analyzed.parameters(entry);
    let context = EntryContext {
        declarations: parameters.iter().map(declaration).collect(),
        structural: &entry.structural,
        holes: &holes,
        families: &families,
    };
    let has_a_family = parameters.iter().any(|p| p.family.is_some());
    if !entry.structural.is_empty() || has_a_family {
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

/// What a scalar of a family's element is, for the element `__element` the
/// host function's declaration walks: `*__element`, or `__element.field`.
fn element_argument(scalar: Scalar<'_>) -> TokenStream {
    match scalar.field {
        None => quote!(*__element),
        Some(field) => quote!(__element.#field),
    }
}

/// What one of an entry's parameters declares, in the host function, in
/// declaration order.
enum Declaration {
    /// One value's scalars: each one's value in the host function.
    Value(Vec<TokenStream>),
    /// A family's: `N` elements' scalars, element-major, each element's in
    /// field order.
    Family {
        /// The host function's array.
        array: Ident,
        /// How many elements, as the host type counts them.
        count: TokenStream,
        /// One element's scalars, on `__element` ([`element_argument`]).
        fields: Vec<TokenStream>,
    },
}

/// The declaration of `parameter`: [`AnalyzedKernel::parameters`]' order,
/// which is the one definition of it.
fn declaration(parameter: &Parameter<'_>) -> Declaration {
    match &parameter.family {
        None => Declaration::Value(parameter.scalars().map(argument).collect()),
        Some(count) => Declaration::Family {
            array: parameter.name.clone(),
            count: count.tokens(),
            fields: parameter.scalars().map(element_argument).collect(),
        },
    }
}

/// What an entry's arena is emitted against: what each parameter declares,
/// in declaration order, and — for an entry with structural parameters —
/// those parameters and the open ranges its holes are filled from, and its
/// families' iterations.
pub struct EntryContext<'a> {
    declarations: Vec<Declaration>,
    structural: &'a [Ident],
    holes: &'a Holes,
    families: &'a Families,
}

/// A parameter's type in the host function: as written, or, for a family,
/// `[element; count]` with the count its evaluated one — a structural
/// parameter's name, or the number a literal or a `const` is, since a
/// private `const` is no host item.
fn host_type(param: &Param, parameter: &Parameter<'_>) -> TokenStream {
    let written = param.ty.written();
    match &parameter.family {
        None => quote!(#written),
        Some(count) => {
            let count = count.tokens();
            quote!([#written; #count])
        }
    }
}

/// The closure form's expansion: a `Kernel`, or a closure over `f32`s and
/// families of them.
fn emit_closure(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    arena_code: &TokenStream,
) -> TokenStream {
    let body = quote! {
        let (__arena, __root) = #arena_code;
        ::pixelflow_core::Kernel::from_parts(__arena, __root)
    };
    if entry.params.is_empty() {
        return quote! {{ #body }};
    }
    // The closure sugar has no records, and `sema` refuses a `bool`
    // parameter of an entry: every parameter is an `f32` or a family of
    // them.
    let params = entry
        .params
        .iter()
        .zip(analyzed.parameters(entry))
        .map(|(param, parameter)| {
            let (name, ty) = (&param.name, host_type(param, &parameter));
            quote!(#name: #ty)
        });
    quote! {
        move | #( #params ),* | -> ::pixelflow_core::Kernel { #body }
    }
}

/// An entry's expansion: a host function returning a `Kernel`, generic over
/// its structural parameters and taking its parameters by their declared
/// types, with the entry's visibility and doc comments.
fn emit_entry(entry: &FnItem, analyzed: &AnalyzedKernel, arena_code: &TokenStream) -> TokenStream {
    let attrs = &entry.attrs;
    let vis = &entry.vis;
    let name = &entry.name;
    let (generics, _) = structural_generics(&entry.structural);
    let params = entry
        .params
        .iter()
        .zip(analyzed.parameters(entry))
        .map(|(param, parameter)| {
            let (name, ty) = (&param.name, host_type(param, &parameter));
            quote!(#name: #ty)
        });
    quote! {
        #(#attrs)*
        #[must_use]
        #vis fn #name #generics ( #(#params),* ) -> ::pixelflow_core::Kernel {
            let (__arena, __root) = #arena_code;
            ::pixelflow_core::Kernel::from_parts(__arena, __root)
        }
    }
}

/// An entry's `Args` record (plan §1.4): its parameters as fields, by their
/// declared types, generic over its structural parameters; and
/// `write_into`, which streams them — a record's fields in field order —
/// into a block by the entry's declaration order
/// (`UniformBlock::set_declared`). An entry with no parameters has none.
///
/// The values are streamed — never a declared `[f32; K]` — and the record
/// derives no `Default`, so that a family's field, `[Row; N]`, needs neither
/// a generic const expression nor a `Default` for `[Row; N]`, and neither is
/// stable Rust. With no family the stream is an array literal, whose length
/// rustc counts; with one it is a chain of iterators, each family's
/// elements flat-mapped to their fields — element-major, as the host
/// function declares them — with no allocation, and a length the iterator
/// knows, so a wrong count is refused before anything is written.
fn emit_args(entry: &FnItem, analyzed: &AnalyzedKernel) -> TokenStream {
    if entry.params.is_empty() {
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
    let parameters = analyzed.parameters(entry);
    let fields = entry
        .params
        .iter()
        .zip(&parameters)
        .map(|(param, parameter)| {
            let (name, ty) = (&param.name, host_type(param, parameter));
            let doc = format!("The argument `{name}` of `{entry_name}`.");
            quote! {
                #[doc = #doc]
                #vis #name: #ty,
            }
        });
    let values = argument_stream(&parameters);
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
                block.set_declared(#values)
            }
        }
    }
}

/// An `Args` record's scalars, in declaration order, as `write_into`
/// streams them: an array literal of its values when it has no family, and
/// otherwise a chain — each run of values an array, each family its
/// elements flat-mapped to their fields (an `f32` family's elements copied
/// out), a family of field-less records nothing.
///
/// The iterator's methods are called by path, `Iterator::chain(…)`, as
/// every name an expansion uses is spelled: a block written in a
/// `#[no_implicit_prelude]` module has no `Iterator` in scope to call a
/// method through.
fn argument_stream(parameters: &[Parameter<'_>]) -> TokenStream {
    let value = |scalar: Scalar<'_>| {
        let value = argument(scalar);
        quote!(self.#value)
    };
    if parameters
        .iter()
        .all(|parameter| parameter.family.is_none())
    {
        let values = parameters.iter().flat_map(|p| p.scalars()).map(value);
        return quote!([ #(#values),* ]);
    }
    let iterator = quote!(::core::iter::Iterator);
    let mut segments: Vec<TokenStream> = Vec::with_capacity(parameters.len());
    let mut run: Vec<TokenStream> = Vec::new();
    for parameter in parameters {
        if parameter.family.is_none() {
            run.extend(parameter.scalars().map(value));
            continue;
        }
        if !run.is_empty() {
            segments.push(quote!([ #(#run),* ]));
            run.clear();
        }
        let array = parameter.name;
        let fields: Vec<TokenStream> = parameter.scalars().map(element_argument).collect();
        match (parameter.record, fields.as_slice()) {
            (_, []) => {}
            (None, _) => segments.push(quote!(#iterator::copied(self.#array.iter()))),
            (Some(_), _) => segments.push(quote! {
                #iterator::flat_map(self.#array.iter(), |__element| [ #(#fields),* ])
            }),
        }
    }
    if !run.is_empty() {
        segments.push(quote!([ #(#run),* ]));
    }
    segments.into_iter().fold(
        quote!(::core::iter::empty::<f32>()),
        |stream, segment| quote!(#iterator::chain(#stream, #segment)),
    )
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
/// refuses ([`RangeFold::admits`]): one that runs backwards, and one past
/// [`RangeFold::EXACT_BOUND`], the last bound a fold's index names exactly.
/// The bound is the IR's, imported; what the `const` block adds is that the
/// refusal is rustc's rather than `Fold::new`'s panic.
///
/// The refusal is rustc's at monomorphization, which `cargo build` reaches
/// and `cargo check` (so clippy) does not: a bad instantiation checks clean
/// and fails to build, in the calling crate, naming the range. It is never
/// a run-time panic.
fn instantiated_range(range: &crate::sema::StructuralRange) -> TokenStream {
    let (lo, hi) = (&range.lo, &range.hi);
    let bound = proc_macro2::Literal::u32_unsuffixed(RangeFold::EXACT_BOUND);
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
/// The program's uniforms are declared first, in `context`'s declaration
/// order: each with an identity minted per call and the call's value as its
/// default, so the kernel's arguments are that call's and its program is
/// every call's — a family's, `N` elements of them, element-major, by the
/// host function as it runs. A template's holes are filled from the host
/// function's structural parameters: a `Param(k)` is the `k`th one as an
/// `f32`, and an open fold's range is evaluated per instantiation
/// ([`instantiated_range`]). A family's iteration is its copies
/// ([`Emission::instantiation`]).
///
/// # Panics
///
/// If the arena's uniforms other than its families' are not `context`'s
/// scalars one for one: lowering declares one uniform per scalar, and a
/// relink restores the table after optimization, so a mismatch is a
/// front-end bug, not a kernel's.
pub fn arena_to_tokens(arena: &ExprArena, root: ExprId, context: &EntryContext) -> TokenStream {
    // The arguments are the uniforms lowering declared first, in
    // declaration order; the rest are its families' abstract elements and
    // markers, which no host function declares.
    let arguments: Vec<u64> = arena
        .uniforms()
        .iter()
        .enumerate()
        .filter(|(_, decl)| !context.families.holds(decl.id))
        .map(|(slot, _)| slot as u64)
        .collect();
    let values: Vec<&TokenStream> = context
        .declarations
        .iter()
        .flat_map(|declaration| match declaration {
            Declaration::Value(values) => values.as_slice(),
            Declaration::Family { .. } => &[],
        })
        .collect();
    assert_eq!(
        arguments.len(),
        values.len(),
        "kernel! declared {} uniforms for {} arguments",
        arguments.len(),
        values.len()
    );
    let mut slots = arguments.iter().copied();
    let mut stmts: Vec<TokenStream> = Vec::with_capacity(context.declarations.len());
    for (position, declaration) in context.declarations.iter().enumerate() {
        match declaration {
            Declaration::Value(values) => {
                for value in values {
                    let slot = uniform_var(slots.next().expect("one slot per argument"));
                    stmts.push(quote! {
                        let #slot = __arena.declare_uniform(::pixelflow_core::__macro::ir::arena::UniformDecl {
                            id: ::pixelflow_core::__macro::ir::arena::UniformIdentity::mint(),
                            default: #value,
                        });
                    });
                }
            }
            Declaration::Family { array, fields, .. } => {
                let base = family_base(position);
                stmts.push(quote! {
                    let #base = __arena.uniforms().len() as u64;
                    for __element in &#array {
                        #(
                            __arena.declare_uniform(::pixelflow_core::__macro::ir::arena::UniformDecl {
                                id: ::pixelflow_core::__macro::ir::arena::UniformIdentity::mint(),
                                default: #fields,
                            });
                        )*
                    }
                });
            }
        }
    }
    let program = Scope {
        arena: format_ident!("__arena"),
        depth: 0,
        declared: arguments.into_iter().collect(),
    };
    let emission = Emission::new(arena, context);
    let nodes = emission.nodes(&emission.reached(root, &[]), &program);
    let root_ident = node_var(root);
    quote! {{
        let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
        #(#stmts)*
        #(#nodes)*
        (__arena, #root_ident)
    }}
}

/// The variable holding a lowered uniform slot's `UniformId` in the arena
/// being built where it is declared.
fn uniform_var(slot: u64) -> Ident {
    format_ident!("__u{slot}")
}

/// The variable holding a node's id in the arena being built.
fn node_var(id: ExprId) -> Ident {
    format_ident!("__e{}", id.0)
}

/// The variable holding the first slot of the family declared `position`th
/// among the entry's parameters: element `k`'s field `f` is that plus
/// `k·width + f`, as the host function declares them.
fn family_base(position: usize) -> Ident {
    format_ident!("__family{position}")
}

/// Where emitted code builds: the program's arena, or a family's template
/// inside it.
struct Scope {
    /// The variable holding the arena being built.
    arena: Ident,
    /// How many templates deep: the program is `0`.
    depth: usize,
    /// The lowered uniform slots declared in this arena: a node may read
    /// these and no others.
    declared: HashSet<u64>,
}

/// What a node reads that the program binds inside itself — a fold's or an
/// integral's index, or a field of a family's abstract element — so that a
/// term reading none of what an iteration binds is the same in every copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Bound {
    /// The index a `Reduce` binds.
    Index(Binder),
    /// A field of an abstract element, by its lowered slot.
    Field(UniformId),
}

/// One lowered arena, emitted: the nodes each scope reaches, a family's
/// iteration as its copies.
struct Emission<'a> {
    arena: &'a ExprArena,
    context: &'a EntryContext<'a>,
    /// Per node, what it reads free of what the program binds: its
    /// children's, less what it binds itself — a `Reduce` its index, an
    /// iteration its element's fields.
    free: Vec<BTreeSet<Bound>>,
    /// Per iteration's node, the terms its copies share: its template's
    /// inputs, in arena order ([`Emission::shared_terms`]).
    shared: HashMap<ExprId, Vec<ExprId>>,
}

impl<'a> Emission<'a> {
    fn new(arena: &'a ExprArena, context: &'a EntryContext<'a>) -> Self {
        let mut emission = Self {
            arena,
            context,
            free: Vec::with_capacity(arena.len()),
            shared: HashMap::new(),
        };
        for index in 0..arena.len() {
            let free = emission.free_in(ExprId(index as u32));
            emission.free.push(free);
        }
        for index in 0..arena.len() {
            let id = ExprId(index as u32);
            if let Some((_, body)) = context.families.node(arena, id) {
                let shared = emission.shared_terms(id, body);
                emission.shared.insert(id, shared);
            }
        }
        emission
    }

    /// What `id` reads free of what the program binds, its children's
    /// already known: the arena lists children before parents.
    ///
    /// Exact, which is why this is not `pixelflow_ir::variance`'s table, the
    /// IR's statement of the same scoping rule: a variance is an
    /// over-approximation — a structural hole, `N as f32`, is every binder
    /// there — and an over-approximated free set of an iteration's node
    /// would admit as a shared term one reading a binder its body binds
    /// (pinned by `an_input_reads_no_binder_its_template_holds`). The match
    /// names every node, so a new binding form is a decision here too.
    fn free_in(&self, id: ExprId) -> BTreeSet<Bound> {
        let without = |child: ExprId, bound: &[Bound]| -> BTreeSet<Bound> {
            let free = &self.free[child.0 as usize];
            free.iter()
                .filter(|b| !bound.contains(b))
                .copied()
                .collect()
        };
        let children = || -> BTreeSet<Bound> {
            self.arena
                .children(id)
                .flat_map(|child| self.free[child.0 as usize].iter().copied())
                .collect()
        };
        if let Some((iteration, body)) = self.context.families.node(self.arena, id) {
            let element: Vec<Bound> = iteration
                .element
                .iter()
                .filter_map(|field| self.slot_of(*field))
                .map(|slot| Bound::Field(UniformId(slot)))
                .collect();
            return without(body, &element);
        }
        match self.arena.node(id) {
            ExprNode::Var(i) => Binder::from_var(i).map(Bound::Index).into_iter().collect(),
            ExprNode::Uniform(slot) => {
                let bound = self
                    .context
                    .families
                    .holds(self.arena.uniform_decl(slot).id);
                bound.then_some(Bound::Field(slot)).into_iter().collect()
            }
            ExprNode::Reduce { fold, body } => without(body, &[Bound::Index(fold.binder())]),
            // A structural hole is a constant of the instantiation.
            ExprNode::Const(_) | ExprNode::Param(_) => BTreeSet::new(),
            ExprNode::Unary(..)
            | ExprNode::Binary(..)
            | ExprNode::Ternary(..)
            | ExprNode::Nary(..) => children(),
            // No lowered arena holds these, and `node` refuses each where it
            // is reached: what they read is never asked.
            ExprNode::Buffer(_)
            | ExprNode::Ref(_)
            | ExprNode::Guard { .. }
            | ExprNode::Write { .. } => children(),
        }
    }

    /// The terms an iteration's copies share: the largest subterms of its
    /// body that read nothing its body binds — not its element, not an
    /// index or an element bound inside it — reached through the iterations
    /// nested in it too, since their copies are its template's. Each is the
    /// same in every copy, so it is built once, where the iteration is, and
    /// is one of its template's inputs: `x = X + ½`, an argument, another
    /// family's iteration. A constant, a coordinate or an index bound
    /// outside is a leaf every arena builds alike, and stays in the
    /// template.
    fn shared_terms(&self, node: ExprId, body: ExprId) -> Vec<ExprId> {
        let outside = &self.free[node.0 as usize];
        let mut seen = vec![false; self.arena.len()];
        let mut stack = vec![body];
        let mut shared = Vec::new();
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if self.free[id.0 as usize].is_subset(outside) {
                let rebuilt_alike = matches!(
                    self.arena.node(id),
                    ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_)
                );
                if !rebuilt_alike {
                    shared.push(id);
                }
                continue;
            }
            match self.context.families.node(self.arena, id) {
                Some((_, nested)) => stack.push(nested),
                None => stack.extend(self.arena.children(id)),
            }
        }
        shared.sort_unstable();
        shared
    }

    /// The nodes a scope builds to reach `root`, in the arena's order,
    /// children before parents: not `inputs` (sorted), which the scope reads
    /// as its template's uniforms, and, for a family's node, the terms its
    /// copies share rather than its body, which is its template's.
    fn reached(&self, root: ExprId, inputs: &[ExprId]) -> Vec<ExprId> {
        let mut reached = vec![false; self.arena.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if inputs.binary_search(&id).is_ok()
                || std::mem::replace(&mut reached[id.0 as usize], true)
            {
                continue;
            }
            match self.shared.get(&id) {
                Some(shared) => stack.extend(shared),
                None => stack.extend(self.arena.children(id)),
            }
        }
        (0..self.arena.len())
            .filter(|index| reached[*index])
            .map(|index| ExprId(index as u32))
            .collect()
    }

    /// The statements building `reached` in `scope`'s arena, each bound to
    /// its [`node_var`]; a family's node is its instantiation.
    fn nodes(&self, reached: &[ExprId], scope: &Scope) -> Vec<TokenStream> {
        reached
            .iter()
            .map(|&id| {
                let expr = self
                    .instantiation(id, scope)
                    .unwrap_or_else(|| self.node(id, scope));
                let var = node_var(id);
                quote! {
                    let #var = #expr;
                }
            })
            .collect()
    }

    /// The lowered slot holding `id`, if the arena declares it: an abstract
    /// element's field the body never reads may not be declared at all.
    fn slot_of(&self, id: pixelflow_ir::arena::UniformIdentity) -> Option<u64> {
        self.arena
            .uniforms()
            .iter()
            .position(|decl| decl.id == id)
            .map(|slot| slot as u64)
    }

    /// The iteration `node` is the node of, instantiated in `scope`'s
    /// arena; `None` if `node` is no family's.
    ///
    /// Its body is built once, as an arena of its own — the *template*, a
    /// function of its uniforms, laid out in three runs:
    /// - first, when the body iterates a family itself, the table of the
    ///   arena its copies are spliced into (`scope`'s), slot for slot — so a
    ///   slot below the count copied is that arena's own. Every arena on
    ///   the way copies the table of the one it is spliced into, starting
    ///   from the program's, so each holds the program's slots at the
    ///   program's numbers, and a nested iteration's copies read their
    ///   elements where the host function declared them, however deep;
    /// - then one input per term the copies share
    ///   ([`Emission::shared_terms`]);
    /// - then the abstract element's fields.
    ///
    /// Each copy is that template spliced into `scope`'s arena with its
    /// uniforms placed (`ExprArena::splice_with`): a copied slot where it
    /// is, a shared term as the node `scope` built once, and the element's
    /// fields as element `k`'s — the family's first slot plus `k·width` —
    /// the copies combined as [`pixelflow_ir::Chain`] combines distinct
    /// terms, under the monoid lowering chose, carried by value. Nothing is
    /// rebuilt per copy but the body, and nothing is allocated but the
    /// arenas.
    ///
    /// Templates are built innermost first: a nested iteration's copies are
    /// made while its enclosing template is built, so a template whose body
    /// iterates a family holds those copies, `N` of them, and is built per
    /// instantiation (plan §1.8, B6).
    fn instantiation(&self, node: ExprId, scope: &Scope) -> Option<TokenStream> {
        let (iteration, body) = self.context.families.node(self.arena, node)?;
        let Some(Declaration::Family { count, fields, .. }) =
            self.context.declarations.get(iteration.parameter)
        else {
            panic!("kernel! iterated a family that is not one of the entry's parameters");
        };
        let ir = quote!(::pixelflow_core::__macro::ir);
        let outer = &scope.arena;
        let depth = scope.depth + 1;
        let arena = format_ident!("__arena{depth}");
        let shared = &self.shared[&node];
        let reached = self.reached(body, shared);
        let element: Vec<Option<u64>> = iteration
            .element
            .iter()
            .map(|id| self.slot_of(*id))
            .collect();
        let inner = Scope {
            arena: arena.clone(),
            depth,
            declared: element.iter().flatten().copied().collect(),
        };
        let nodes = self.nodes(&reached, &inner);
        let root = node_var(body);

        let abstract_uniform = quote! {
            #arena.declare_uniform(#ir::arena::UniformDecl {
                id: #ir::arena::UniformIdentity::mint(),
                default: f32::NAN,
            })
        };
        let iterates_a_family = reached
            .iter()
            .any(|id| self.context.families.node(self.arena, *id).is_some());
        let target_table = if iterates_a_family {
            quote! {
                for __decl in #outer.uniforms() {
                    #arena.declare_uniform(*__decl);
                }
            }
        } else {
            TokenStream::new()
        };
        let shared_vars: Vec<Ident> = shared.iter().map(|id| node_var(*id)).collect();
        let shared_count = Literal::usize_unsuffixed(shared.len());
        let shared_inputs = Literal::u64_unsuffixed(shared.len() as u64);
        let element_fields = element.iter().map(|slot| match slot {
            Some(slot) => {
                let var = uniform_var(*slot);
                quote!(let #var = #abstract_uniform;)
            }
            None => quote!(#abstract_uniform;),
        });
        let base = family_base(iteration.parameter);
        let width = Literal::usize_unsuffixed(fields.len());
        let monoid = lower::monoid(iteration.reduction).marshal().to_bytes();
        Some(quote! {{
            let __shared: [#ir::arena::ExprId; #shared_count] = [ #(#shared_vars),* ];
            let (__template, __template_root, __mirrored) = {
                let mut #arena = #ir::arena::ExprArena::new();
                #target_table
                // The slots below this are the splice target's own.
                let __mirrored: u64 = #arena.uniforms().len() as u64;
                #(
                    let #shared_vars = {
                        let __slot = #abstract_uniform;
                        #arena.push_uniform(__slot)
                    };
                )*
                #( #element_fields )*
                #( #nodes )*
                (#arena, #root, __mirrored)
            };
            let mut __chain = #ir::Chain::new(
                #ir::Monoid::unmarshal(#ir::kind::OpCode::from_bytes([ #(#monoid),* ]))
                    .expect("kernel! emitted a monoid"),
            );
            for __k in 0..#count {
                let __first = #base + (__k * #width) as u64;
                let __copy = #outer.splice_with(&__template, __template_root, |__into, __slot| {
                    match __slot.0.checked_sub(__mirrored) {
                        // The splice target's own slot: a nested
                        // iteration's element.
                        ::core::option::Option::None => __into.push_uniform(__slot),
                        ::core::option::Option::Some(__input) => {
                            match __shared.get(__input as usize) {
                                ::core::option::Option::Some(&__term) => __term,
                                // The element's fields, last: element `k`'s.
                                ::core::option::Option::None => {
                                    __into.push_uniform(#ir::arena::UniformId(
                                        __first + (__input - #shared_inputs),
                                    ))
                                }
                            }
                        }
                    }
                });
                __chain.push(__copy, |__op, __folded, __term| {
                    #outer.push_binary(__op, __folded, __term)
                });
            }
            __chain.finish(|__identity| #outer.push_const(__identity))
        }})
    }

    /// One node, built in `scope`'s arena.
    fn node(&self, id: ExprId, scope: &Scope) -> TokenStream {
        let arena = &scope.arena;
        match self.arena.node(id) {
            ExprNode::Var(i) => quote! { #arena.push_var(#i) },
            // By bit pattern, not as a decimal literal: `quote`'s `f32`
            // impl goes through `Literal::f32_suffixed`, which asserts
            // `is_finite()` — and non-finite constants are ordinary here. A
            // true comparison mask is all-ones (`OpKind::mask`), which is
            // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
            // the folder now produces those. Bits also roundtrip exactly, with
            // no decimal-formatting question to get wrong.
            ExprNode::Const(v) => {
                let bits = v.to_bits();
                quote! { #arena.push_const(f32::from_bits(#bits)) }
            }
            // A template's hole: the structural parameter, as its `f32`, as
            // Rust's `as` rounds it — which is `N as f32` in the body.
            ExprNode::Param(k) => {
                let count = self
                    .context
                    .structural
                    .get(usize::from(k))
                    .unwrap_or_else(|| {
                        panic!(
                            "kernel! produced ExprNode::Param({k}) outside a template's \
                             structural parameters"
                        )
                    });
                quote! { #arena.push_const(#count as f32) }
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
            // An argument, or an abstract element's field: the slot this
            // arena declared for it. Any other is a family's, read outside
            // its iteration — a front-end bug, never a program.
            ExprNode::Uniform(u) => {
                assert!(
                    scope.declared.contains(&u.0),
                    "kernel! read uniform slot {} where no arena being built declares it",
                    u.0
                );
                let slot = uniform_var(u.0);
                quote! { #arena.push_uniform(#slot) }
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
                quote! { #arena.push_unary(#op_code, #child) }
            }
            ExprNode::Binary(op, a, b) => {
                let op_code = opkind_to_tokens(op);
                let (a, b) = (node_var(a), node_var(b));
                quote! { #arena.push_binary(#op_code, #a, #b) }
            }
            ExprNode::Ternary(op, a, b, c) => {
                let op_code = opkind_to_tokens(op);
                let (a, b, c) = (node_var(a), node_var(b), node_var(c));
                quote! { #arena.push_ternary(#op_code, #a, #b, #c) }
            }
            ExprNode::Nary(op, ..) => {
                let op_code = opkind_to_tokens(op);
                let children: Vec<Ident> = self.arena.children(id).map(node_var).collect();
                quote! { #arena.push_nary(#op_code, &[#(#children),*]) }
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
                let emitted = match self.context.holes.range_of(fold) {
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
                quote! { #arena.push_reduce(#emitted, #body) }
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
            "no `Default`, which a family's `[Row; N]` could not derive: {code}"
        );
        assert!(
            code.contains("default : b . x0") && code.contains("default : r"),
            "{code}"
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

    /// A family is its host type, `[R; N]`, in the host function and the
    /// `Args` record; its `N` elements' uniforms are declared as the host
    /// function runs, element-major; an iteration is its body's template,
    /// its shared terms (here `r`) its inputs, spliced per element with the
    /// element's uniforms in its fields' places and combined by the IR's
    /// `Chain` under the monoid, carried by value — and no fold; and
    /// `write_into` streams the family's fields flat-mapped, then the rest.
    #[test]
    fn a_familys_iteration_expands_to_its_copies_at_instantiation() {
        let code = expansion(quote! {
            pub struct Pair { pub a: f32, pub b: f32 }
            pub fn f<const N: usize>(pairs: [Pair; N], r: f32) -> f32 {
                pairs.into_iter().map(|p| p.b * X + r).sum()
            }
        });
        let monoid = |monoid: pixelflow_ir::Monoid| {
            let [byte] = monoid.marshal().to_bytes();
            format!(
                "Monoid :: unmarshal (:: pixelflow_core :: __macro :: ir :: kind :: OpCode :: \
                 from_bytes ([{byte}u8]))"
            )
        };
        for expected in [
            "pub fn f < const N : usize > (pairs : [Pair ; N] , r : f32)",
            "pub pairs : [Pair ; N] ,",
            "let __family0 = __arena . uniforms () . len () as u64 ; for __element in & pairs {",
            "default : __element . a",
            "default : __element . b",
            "let mut __arena1 = :: pixelflow_core :: __macro :: ir :: arena :: ExprArena :: new () ; \
             let __mirrored : u64 = __arena1 . uniforms () . len () as u64 ;",
            "ExprId ; 1] = [__e",
            "for __k in 0 .. N {",
            "let __first = __family0 + (__k * 2) as u64 ;",
            "__arena . splice_with (& __template , __template_root , | __into , __slot |",
            &monoid(pixelflow_ir::Monoid::SUM),
            "set_declared (:: core :: iter :: Iterator :: chain (:: core :: iter :: Iterator :: \
             chain (:: core :: iter :: empty :: < f32 > () , :: core :: iter :: Iterator :: \
             flat_map (self . pairs . iter () , | __element | [__element . a , __element . b])) , \
             [self . r]))",
        ] {
            assert!(code.contains(expected), "expected `{expected}` in: {code}");
        }
        assert!(!code.contains("push_reduce"), "no fold: {code}");
        // Spelled as `to_string` spaces a path: `"::std"` matches nothing.
        // No prelude name either: a_family_is_its_copies.rs expands a block
        // in a `#[no_implicit_prelude]` module.
        assert!(
            !code.contains(":: std ::"),
            "no `::std` path, so a `no_std` crate expands it: {code}"
        );

        let code = expansion(quote! { |v: [f32; 3]| v.into_iter().any(|e| X < e) });
        for expected in [
            "move | v : [f32 ; 3] | -> :: pixelflow_core :: Kernel",
            "default : * __element",
            "ExprId ; 0] = [] ;",
            "for __k in 0 .. 3 {",
            &monoid(pixelflow_ir::Monoid::ANY),
        ] {
            assert!(code.contains(expected), "expected `{expected}` in: {code}");
        }
    }

    /// An optimizer that counts the arenas it is offered, and declines each.
    struct Offered(usize);

    impl Optimize for Offered {
        fn optimize(&mut self, _arena: &ExprArena, _root: ExprId) -> pixelflow_ir::Rewritten {
            self.0 += 1;
            pixelflow_ir::Rewritten::Declined
        }
    }

    /// No optimizer sees an entry with a family. Its lowered arena holds each
    /// iteration as `body + marker`, which is only that iteration while no
    /// rule rewrites it: reassociated, `t + (body + marker)` would read as an
    /// iteration of `t + body`, `t` copied `N` times. So it is emitted as
    /// lowered, as a structural entry is, whatever its count is written as,
    /// and an entry without one is offered as before.
    #[test]
    fn an_entry_with_a_family_is_offered_to_no_optimizer() {
        let analyzed = analyze(
            parse(quote! {
                pub fn counted(v: [f32; 2], r: f32) -> f32 {
                    r + v.into_iter().map(|e| e * X).sum::<f32>()
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

    /// What an iteration's copies share is built once, where the iteration
    /// is, and is its template's input — not rebuilt in the template and
    /// spliced per copy. So an iteration whose body reads another's result
    /// takes it as an input: the first family's copies are made once, in the
    /// program, and the second's template holds its own body alone — no
    /// template inside it, and nothing spliced into one — where rebuilding
    /// it made `N` copies of the first body inside the second's template.
    /// A body that does iterate a family itself holds first the table of
    /// the arena its copies are spliced into, so that iteration finds its
    /// elements where the host declared them, however deep.
    #[test]
    fn what_the_copies_share_is_built_once_as_an_input() {
        let code = expansion(quote! {
            pub struct Pair { pub a: f32, pub b: f32 }
            pub fn weights<const N: usize>(pairs: [Pair; N]) -> f32 {
                let f: f32 = pairs.into_iter().map(|p| p.a * X).sum();
                pairs.into_iter().map(|p| p.b * f).sum()
            }
        });
        assert_eq!(
            code.matches("for __k in 0 .. N").count(),
            2,
            "two iterations: {code}"
        );
        assert!(
            !code.contains("__arena2") && !code.contains("__arena1 . splice_with"),
            "no iteration is instantiated inside another's template: {code}"
        );
        assert!(
            !code.contains("for __decl in"),
            "neither body iterates a family: {code}"
        );

        let code = expansion(quote! {
            pub fn deep<const N: usize>(a: [f32; N], r: f32) -> f32 {
                a.into_iter().map(|p| p + a.into_iter().map(|q| p * q + r).sum::<f32>()).sum()
            }
        });
        for expected in [
            "for __decl in __arena . uniforms () { __arena1 . declare_uniform (* __decl) ; } \
             let __mirrored : u64 = __arena1 . uniforms () . len () as u64 ;",
            "__arena1 . splice_with (& __template , __template_root",
        ] {
            assert!(code.contains(expected), "expected `{expected}` in: {code}");
        }

        // Two deep, the middle template copies the table of the arena its
        // copies are spliced into — the outer template's — and counts what
        // it copied: no slot numbering is assumed shared by two arenas.
        let code = expansion(quote! {
            pub fn triples<const N: usize>(a: [f32; N]) -> f32 {
                a.into_iter()
                    .map(|p| {
                        a.into_iter()
                            .map(|q| a.into_iter().map(|s| p * q * s).sum::<f32>())
                            .sum::<f32>()
                    })
                    .sum()
            }
        });
        for expected in [
            "for __decl in __arena1 . uniforms () { __arena2 . declare_uniform (* __decl) ; } \
             let __mirrored : u64 = __arena2 . uniforms () . len () as u64 ;",
            "__arena2 . splice_with (& __template , __template_root",
        ] {
            assert!(code.contains(expected), "expected `{expected}` in: {code}");
        }
        assert!(
            !code.contains("__arena3 . declare_uniform (* __decl)"),
            "the innermost body iterates nothing, and copies no table: {code}"
        );
    }

    /// Every binder a term reads, over-approximated as the IR's own table
    /// states it (`compute_arena_variance`), against the binders `body`
    /// binds: the reduces it reaches, through the iterations nested in it.
    fn binders_held(arena: &ExprArena, body: ExprId) -> pixelflow_ir::variance::Variance {
        let mut held = pixelflow_ir::variance::Variance::CONST;
        let mut seen = vec![false; arena.len()];
        let mut stack = vec![body];
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if let ExprNode::Reduce { fold, .. } = arena.node(id) {
                held = held.union(pixelflow_ir::variance::Variance::from_var(
                    fold.binder().var(),
                ));
            }
            stack.extend(arena.children(id));
        }
        held
    }

    /// A template's input stands for a term built outside it, so it reads no
    /// binder the template holds — checked against the IR's statement of
    /// the scoping rule, `compute_arena_variance`, beside `N as f32`, a
    /// hole the variance counts as reading every binder. Taken from the
    /// variance, the iteration's node would seem to read the fold's `j` and
    /// the integral's `u` from outside, and `j·r` and `u·r` would be inputs
    /// standing for terms under the binders that bind them. Exactly, `r`
    /// is the one input.
    #[test]
    fn an_input_reads_no_binder_its_template_holds() {
        let analyzed = analyze(
            parse(quote! {
                pub struct Pair { pub a: f32, pub b: f32 }
                pub fn f<const N: usize>(pairs: [Pair; N], r: f32) -> f32 {
                    pairs
                        .into_iter()
                        .map(|p| {
                            (0..2).map(|j| (j as f32) * r + p.a).sum::<f32>() * (N as f32)
                                + integral(0.0..1.0, |u| u * r + p.b)
                        })
                        .sum()
                }
            })
            .expect("parses"),
        )
        .expect("analyzes");
        let [entry] = analyzed.def.fns.as_slice() else {
            panic!("one entry");
        };
        let Lowered {
            arena,
            holes,
            families,
            ..
        } = lower::lower_entry(entry, &analyzed).expect("lowers");
        let context = EntryContext {
            declarations: analyzed.parameters(entry).iter().map(declaration).collect(),
            structural: &entry.structural,
            holes: &holes,
            families: &families,
        };
        let emission = Emission::new(&arena, &context);
        let variance = pixelflow_ir::variance::compute_arena_variance(&arena);
        assert_eq!(emission.shared.len(), 1, "one iteration");
        for (&node, inputs) in &emission.shared {
            let (_, body) = families.node(&arena, node).expect("an iteration's node");
            let held = binders_held(&arena, body);
            assert!(
                held.depends_on_binder(),
                "the body binds `j` and `u`: {held:?}"
            );
            for &input in inputs {
                assert!(
                    variance[input.0 as usize].intersection(held).is_const(),
                    "input {} reads a binder its template holds",
                    arena.display(input)
                );
            }
            let [r] = inputs.as_slice() else {
                panic!("`r` alone: {inputs:?}");
            };
            assert!(
                matches!(arena.node(*r), ExprNode::Uniform(_)),
                "{}",
                arena.display(*r)
            );
        }
    }
}
