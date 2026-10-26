//! The host items: the one function of the stage, and what each spelling
//! of a block becomes in the host crate — an expression, or a `fn`, a
//! record and an `Args` record per entry.

use super::arena::{arena_to_tokens, emitted};
use super::staged::Staged;
use crate::ast::{Callable, ClosureEntry, ConstItem, FnItem, KernelDef, Name, RecordItem, Role};
use crate::lower::Expansion;
use crate::sema::{AnalyzedKernel, ConstValue, Scalar};
use pixelflow_ir::optimize::Optimize;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

/// Emit arena-backend code for an analyzed kernel.
///
/// For the closure form, a token stream evaluating to:
/// - zero params — a `pixelflow_core::Kernel` value, built at load
///   time from the arena this expansion computed.
/// - N params — a closure `move |p0: f32, ...| -> Kernel` whose kernel
///   declares one uniform per parameter, its default the call's argument.
///   The closure sugar has no `Args` record: a program compiled from it is
///   rebound by position, with `UniformBlock::set_declared`.
///
/// For the items form: one struct per record, one `const` per `pub const`,
/// and for each entry its host `fn`, taking its parameters by their declared
/// types — a kernel-typed one as
/// a `&Kernel` — and its `Args` record. An entry with kernel parameters is
/// [`Staged`]: lowered when its host function is called.
///
/// `optimizer` rewrites each lowered arena before it is emitted. It is a
/// parameter rather than a branch because "do not optimize" is a value:
/// `kernel_raw!` passes [`Identity`](pixelflow_ir::optimize::Identity), and
/// that is the entire difference between the two macros.
///
/// # Errors
///
/// When lowering refuses what `sema` accepted: folds nested past
/// `Binder::COUNT` after a helper is inlined.
pub(crate) fn emit_kernel(
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> syn::Result<TokenStream> {
    match &analyzed.def {
        KernelDef::Closure(entry) => {
            // A closure has no `Args` record to supply the scalars to.
            let (arena_code, _) = entry_arena(entry.callable(), analyzed, optimizer)?;
            Ok(emit_closure(entry, &arena_code))
        }
        KernelDef::Items {
            records,
            consts,
            fns,
        } => {
            let mut items = TokenStream::new();
            for record in records {
                items.extend(emit_record(record));
            }
            for c in consts.iter().filter(|c| is_pub(&c.vis)) {
                items.extend(emit_const(c, analyzed.consts[&Name::of(&c.name)]));
            }
            for entry in fns.iter().filter(|f| f.role() == Role::Entry) {
                if entry.takes_a_kernel() {
                    let body = Staged::lower(entry.callable(), analyzed)?;
                    items.extend(emit_entry(entry, &body, analyzed));
                    continue;
                }
                let (arena_code, scalars) = entry_arena(entry.callable(), analyzed, optimizer)?;
                items.extend(emit_entry(entry, &kernel_from(&arena_code), analyzed));
                items.extend(emit_args(entry, &scalars));
            }
            Ok(items)
        }
    }
}

/// The `Kernel` made of the `(arena, root)` that `arena_code` evaluates to.
pub(super) fn kernel_from(arena_code: &TokenStream) -> TokenStream {
    emitted! {
        let (__arena, __root) = #arena_code;
        ::pixelflow_core::Kernel::from_parts(__arena, __root)
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    !matches!(vis, syn::Visibility::Inherited)
}

/// Lower an entry, run the optimizer over it, and emit the code that
/// rebuilds the arena: an expression evaluating to `(arena, root)`, and the
/// scalars the lowering declared its uniforms for, which the host function
/// and its `Args` record supply by position.
///
/// Only entries whose IR is fully available at expansion reach this optimizer.
fn entry_arena(
    entry: Callable<'_>,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> syn::Result<(TokenStream, Vec<Scalar>)> {
    let lowered = Expansion::lower(entry, analyzed)?;
    let context = EntryContext {
        arguments: lowered.scalars.iter().map(argument).collect(),
    };

    // Declining is ordinary and needs no arm: the lowered term stands, and a
    // kernel that reaches the runtime tier unoptimized is optimized there.
    let rewritten = match optimizer
        .optimize(&lowered.arena, lowered.root)
        .into_changed()
    {
        // Extraction declares the uniforms it reaches in its own walk order
        // and drops the ones it does not; relinking puts back the entry's
        // declaration order, every scalar in it, which is the order a
        // positional binding supplies them in.
        Some((optimized, optimized_root)) if !lowered.arena.uniforms().is_empty() => {
            Some(optimized.relink(
                optimized_root,
                lowered.arena.buffers(),
                lowered.arena.uniforms(),
            ))
        }
        changed => changed,
    };
    let (arena, root) = rewritten
        .as_ref()
        .map_or((&lowered.arena, lowered.root), |(arena, root)| {
            (arena, *root)
        });
    let tokens = arena_to_tokens(arena, root, &context);
    Ok((tokens, lowered.scalars))
}

/// What a scalar argument is in the host function: `param`, or
/// `param.field` for a record parameter's field.
pub(super) fn argument(scalar: &Scalar) -> TokenStream {
    match scalar {
        Scalar::Param(param) => quote!(#param),
        Scalar::Count(param) => quote!(#param as f32),
        Scalar::Field(param, field) => quote!(#param.#field),
    }
}

/// What an entry's arena is emitted against: the argument each declared
/// uniform defaults to, in declaration order.
pub(super) struct EntryContext {
    pub(super) arguments: Vec<TokenStream>,
}

/// The closure form's expansion: a `Kernel`, or a closure over `f32`s.
fn emit_closure(entry: &ClosureEntry, arena_code: &TokenStream) -> TokenStream {
    let body = kernel_from(arena_code);
    if entry.params.is_empty() {
        return quote! {{ #body }};
    }
    let params = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        quote!(#name: #ty)
    });
    quote! {
        move | #( #params ),* | -> ::pixelflow_core::Kernel { #body }
    }
}

/// An entry's expansion: a host function returning a `Kernel` — `body`,
/// statements that build it — taking its parameters by their declared types,
/// with the entry's
/// visibility and doc comments.
///
/// A kernel-typed parameter is taken as a `&Kernel`, not a `Kernel`: the
/// host function only reads it, splicing a copy of its term into the
/// program it builds, and a borrow passes one instance to two parameters,
/// `sum2(&a, &a)`, as `Kernel::at` and `Lattice::bake` take theirs. Owning it would buy the
/// argument's arena only by building on top of it, which would declare its
/// uniforms before the entry's own.
fn emit_entry(entry: &FnItem, body: &TokenStream, analyzed: &AnalyzedKernel) -> TokenStream {
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
    let guards = analyzed
        .parameters(&entry.params)
        .into_iter()
        .filter_map(|parameter| {
            let crate::sema::Parameter::Count(count) = parameter else {
                return None;
            };
            let message = count_refusal(entry, count);
            Some(quote! {
                ::core::assert!(#count as u64 <= ::pixelflow_core::__macro::ir::Fold::EXACT_BOUND,
                    #message, #count);
            })
        });
    quote! {
        #(#attrs)*
        #[must_use]
        #vis fn #name ( #(#params),* ) -> ::pixelflow_core::Kernel {
            #(#guards)*
            #body
        }
    }
}

/// An entry's `Args` record (plan §1.4): its parameters as fields, by their
/// declared types; and
/// `write_into`, which writes them — a record's fields in field order —
/// into a block by the entry's declaration order
/// (`UniformBlock::set_declared`), as an array literal whose length rustc
/// counts. An entry with no parameters has none, and neither has one that
/// takes a kernel ([`FnItem::has_args_record`]).
fn emit_args(entry: &FnItem, scalars: &[Scalar]) -> TokenStream {
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
    let values = scalars.iter().map(|scalar| {
        let value = argument(scalar);
        quote!(self.#value)
    });
    let guards = scalars.iter().filter_map(|scalar| {
        let Scalar::Count(count) = scalar else {
            return None;
        };
        let message = count_refusal(entry, count);
        Some(quote! {
            ::core::assert!(self.#count as u64 <= ::pixelflow_core::__macro::ir::Fold::EXACT_BOUND,
                #message, self.#count);
        })
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
            /// written. The count is all that is checked: a block of
            /// another program that declares as many arguments takes these
            /// values by position (plan §1.4).
            #vis fn write_into(
                &self,
                block: &mut ::pixelflow_core::UniformBlock,
            ) -> ::core::result::Result<(), ::pixelflow_core::ArityMismatch> {
                #(#guards)*
                block.set_declared([ #(#values),* ])
            }
        }
    }
}

fn count_refusal(entry: &FnItem, count: &Ident) -> String {
    let bound = pixelflow_ir::Fold::EXACT_BOUND;
    format!(
        "kernel!: `{}`'s count `{count}` is {{}}, past {bound} (2^24)",
        entry.name
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
