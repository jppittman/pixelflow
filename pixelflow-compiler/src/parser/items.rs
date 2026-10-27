//! The items form and the closure form: a `syn::File` of records, consts and
//! `fn`s, or `|params| expr`, each item converted to the vocabulary's or
//! refused at the token that spells nothing.

use super::block::{convert_block, plain_name};
use super::expr::{Nesting, convert_expr};
use crate::ast::{
    ClosureEntry, ConstItem, Expr, FnItem, FunctionType, KernelDef, PLAN, Param, RecordField,
    RecordItem,
};
use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{Pat, Token, Type};

/// Parse a `kernel!` invocation's tokens: the closure form or a block of
/// items.
///
/// # Errors
///
/// A spanned `syn::Error` for syntax the language has no meaning for (the
/// module doc lists them).
pub(crate) fn parse(input: TokenStream) -> syn::Result<KernelDef> {
    syn::parse2(input)
}

impl Parse for KernelDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![|]) {
            return parse_closure(input);
        }
        parse_items(input)
    }
}

/// `|param: Type, ...| body`: one entry, its return type inferred.
fn parse_closure(input: ParseStream) -> syn::Result<KernelDef> {
    input.parse::<Token![|]>()?;

    let mut params = Vec::new();

    if !input.peek(Token![|]) {
        loop {
            let ident: syn::Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let ty: Type = input.parse()?;
            refuse_a_collection(&ty)?;
            refuse_a_kernel_in_a_closure(&ty)?;

            params.push(Param {
                name: ident,
                ty: Box::new(ty),
            });

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                if input.peek(Token![|]) {
                    break;
                }
            } else {
                break;
            }
        }
    }

    input.parse::<Token![|]>()?;

    let syn_expr: syn::Expr = input.parse()?;
    let body = convert_expr(syn_expr, Nesting::ROOT)?;

    Ok(KernelDef::Closure(ClosureEntry { params, body }))
}

/// The items form: records, `const`s and `fn`s, as a `syn::File`.
fn parse_items(input: ParseStream) -> syn::Result<KernelDef> {
    let file: syn::File = input.parse()?;
    if let Some(attr) = file.attrs.first() {
        return Err(syn::Error::new_spanned(
            attr,
            "an inner attribute has no meaning in a `kernel!` block",
        ));
    }
    let (mut records, mut consts, mut fns) = (Vec::new(), Vec::new(), Vec::new());
    for item in file.items {
        match item {
            syn::Item::Struct(item) => records.push(convert_record(item)?),
            syn::Item::Const(item) => consts.push(convert_const(item)?),
            syn::Item::Fn(item) => fns.push(convert_fn(item)?),
            other => return Err(refuse_item(&other)),
        }
    }
    Ok(KernelDef::Items {
        records,
        consts,
        fns,
    })
}

/// An item that is not a record, a `const` or a `fn`, named by its kind.
fn refuse_item(item: &syn::Item) -> syn::Error {
    let kind = match item {
        syn::Item::Enum(_) => "enum",
        syn::Item::Static(_) => "static",
        syn::Item::Type(_) => "type",
        syn::Item::Use(_) => "use",
        syn::Item::Mod(_) => "mod",
        syn::Item::Impl(_) => "impl",
        syn::Item::Trait(_) => "trait",
        syn::Item::Macro(_) => "macro invocation",
        _ => "item",
    };
    syn::Error::new_spanned(
        item,
        format!(
            "a `{kind}` in a `kernel!` block\n\
             \n\
             note: a `kernel!` block holds records (`struct`s of `f32` fields), `const` \
             items and `fn` items: a `pub fn` is an entry, a private `fn` is a helper"
        ),
    )
}

/// `struct R { a: f32, b: f32 }`. A record's shape — named fields, no
/// generics — is checked here; that each field is an `f32` is `sema`'s
/// question, since a field typed as another record needs the block's
/// records to say so.
fn convert_record(item: syn::ItemStruct) -> syn::Result<RecordItem> {
    refuse_layout_attributes(&item.attrs)?;
    if let Some(param) = item.generics.params.first() {
        return Err(syn::Error::new_spanned(
            param,
            format!(
                "a generic record\n\
                 \n\
                 note: a record is named `f32` fields, and nothing else (§1.3 of {PLAN})"
            ),
        ));
    }
    refuse_where_clause(&item.generics, "a record")?;
    let named = match item.fields {
        syn::Fields::Named(named) => named,
        syn::Fields::Unnamed(unnamed) => {
            return Err(syn::Error::new_spanned(
                unnamed,
                format!(
                    "a tuple struct in a `kernel!` block\n\
                     \n\
                     note: a record's fields have names, `struct {} {{ x: f32, y: f32 }}` \
                     (§1.3 of {PLAN})",
                    item.ident
                ),
            ));
        }
        syn::Fields::Unit => {
            return Err(syn::Error::new_spanned(
                &item.ident,
                format!(
                    "a unit struct in a `kernel!` block\n\
                     \n\
                     note: a record is named `f32` fields, `struct {} {{ x: f32 }}` (§1.3 of \
                     {PLAN})",
                    item.ident
                ),
            ));
        }
    };
    let mut fields = Vec::with_capacity(named.named.len());
    for field in named.named {
        refuse_layout_attributes(&field.attrs)?;
        let name = field
            .ident
            .expect("a field of a struct with named fields has a name");
        fields.push(RecordField {
            attrs: field.attrs,
            vis: field.vis,
            name,
            ty: field.ty,
        });
    }
    Ok(RecordItem {
        attrs: item.attrs,
        vis: item.vis,
        name: item.ident,
        fields,
    })
}

/// `const NAME: f32 = expr;`. The initializer is parsed as any body is and
/// `sema` evaluates it, refusing what a const cannot hold.
fn convert_const(item: syn::ItemConst) -> syn::Result<ConstItem> {
    refuse_attributes(&item.attrs)?;
    refuse_generics(&item.generics)?;
    Ok(ConstItem {
        attrs: item.attrs,
        vis: item.vis,
        name: item.ident,
        ty: *item.ty,
        init: convert_expr(*item.expr, Nesting::ROOT)?,
    })
}

/// A `fn` item: its signature is checked here for the shapes the language
/// has no meaning for, and its body is converted as any block is.
fn convert_fn(item: syn::ItemFn) -> syn::Result<FnItem> {
    refuse_attributes(&item.attrs)?;
    let sig = item.sig;
    if let Some(token) = sig.constness {
        return Err(syn::Error::new(
            token.span,
            "`const fn` in a `kernel!` block: every helper is inlined at expansion, and a \
             `const` item is evaluated there; the qualifier adds nothing",
        ));
    }
    if let Some(token) = sig.asyncness {
        return Err(syn::Error::new(
            token.span,
            "`async` has no meaning in a kernel",
        ));
    }
    if let Some(token) = sig.unsafety {
        return Err(syn::Error::new(
            token.span,
            "`unsafe` has no meaning in a kernel",
        ));
    }
    if let Some(abi) = sig.abi {
        return Err(syn::Error::new_spanned(
            abi,
            "an ABI has no meaning in a kernel: an entry's host function is plain Rust",
        ));
    }
    if let Some(variadic) = sig.variadic {
        return Err(syn::Error::new_spanned(
            variadic,
            "a variadic parameter list has no meaning in a kernel",
        ));
    }
    refuse_generics(&sig.generics)?;

    let mut params = Vec::with_capacity(sig.inputs.len());
    for input in sig.inputs {
        params.push(convert_param(input)?);
    }

    let ret = match sig.output {
        syn::ReturnType::Type(_, ty) => *ty,
        syn::ReturnType::Default => {
            return Err(syn::Error::new(
                sig.paren_token.span.close(),
                format!(
                    "`fn {}` declares no return type\n\
                     \n\
                     note: a kernel `fn` declares what it returns, `-> f32` or `-> bool`, \
                     and the body is checked against it",
                    sig.ident
                ),
            ));
        }
    };

    Ok(FnItem {
        attrs: item.attrs,
        vis: item.vis,
        name: sig.ident,
        params,
        ret,
        body: Expr::Block(convert_block(*item.block, Nesting::ROOT)?),
    })
}

/// A `fn` parameter: a plain name and a scalar type.
fn convert_param(input: syn::FnArg) -> syn::Result<Param> {
    let typed = match input {
        syn::FnArg::Typed(typed) => typed,
        syn::FnArg::Receiver(receiver) => {
            return Err(syn::Error::new_spanned(
                receiver,
                "a kernel `fn` has no `self`: it is a function of its arguments",
            ));
        }
    };
    if let Some(attr) = typed.attrs.first() {
        return Err(syn::Error::new_spanned(
            attr,
            "an attribute on a kernel parameter has no meaning",
        ));
    }
    let name = match &*typed.pat {
        Pat::Ident(pat_ident) => plain_name(pat_ident, &typed.pat)?,
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "a kernel parameter is a plain name\n\
                 \n\
                 note: destructuring and other patterns are not allowed here",
            ));
        }
    };
    // A kernel-typed parameter, `impl Fn(f32, f32) -> f32`, and every other
    // spelling of a function are `sema`'s: whether one is the language's
    // type takes the block's records to say, as a record's does.
    refuse_a_collection(&typed.ty)?;
    Ok(Param { name, ty: typed.ty })
}

/// The closure sugar takes no kernel-typed parameter: rustc refuses
/// `impl Trait` in a closure's parameters (E0562), and the language's
/// closure is Rust's, so `|k: impl Fn(f32, f32) -> f32| …` is not a closure
/// rustc would compile. An entry in the items form takes one (plan Phase
/// D-a).
///
/// Every spelling of a function is refused here, by the classification
/// `sema` types a parameter with ([`FunctionType::of`]): through
/// parentheses and a macro's invisible group, a kernel spelled `(impl Fn…)`
/// is still one, and any other function is a misspelled kernel, whose
/// correct spelling the closure cannot take either.
fn refuse_a_kernel_in_a_closure(ty: &Type) -> syn::Result<()> {
    if FunctionType::of(ty) == FunctionType::NotOne {
        return Ok(());
    }
    Err(syn::Error::new_spanned(
        ty,
        format!(
            "a function as the closure's parameter\n\
             \n\
             note: a kernel-typed parameter is `impl Fn(f32, f32) -> f32`, and rustc refuses \
             `impl Trait` in a closure's parameters (E0562): the closure form is Rust's \
             closure, so it takes no kernel\n\
             help: take the kernel in an entry of the items form, `kernel! {{ pub fn f(k: impl \
             Fn(f32, f32) -> f32) -> f32 {{ k(X, Y) }} }}` (Phase D-a of {PLAN})"
        ),
    ))
}

/// A parameter typed as a collection — an array `[R; N]` or a slice `[R]`,
/// through references, parentheses and a macro's invisible group — is refused
/// where it is written: the language has none.
fn refuse_a_collection(ty: &Type) -> syn::Result<()> {
    fn names_a_collection(ty: &Type) -> bool {
        match ty {
            Type::Array(_) | Type::Slice(_) => true,
            Type::Reference(reference) => names_a_collection(&reference.elem),
            Type::Paren(inner) => names_a_collection(&inner.elem),
            Type::Group(inner) => names_a_collection(&inner.elem),
            _ => false,
        }
    }
    if !names_a_collection(ty) {
        return Ok(());
    }
    Err(no_collection_types(
        ty,
        "a collection-typed parameter",
        "a parameter is an `f32`, a `bool` or one of the block's records; several values \
         are several parameters",
    ))
}

/// The refusal of a collection, spelled any way: `what` is the spelling
/// found, and `instead` what the language has in its place. There are no
/// collection types (plan §1.3, §1.6): no array, slice, list or table, so
/// nothing is indexed, measured or iterated but a fold's range.
pub(super) fn no_collection_types(
    tokens: impl quote::ToTokens,
    what: &str,
    instead: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        tokens,
        format!(
            "{what} in a `kernel!` block\n\
             \n\
             note: the language has no collection types (§1.3, §1.6 of {PLAN}): no array, \
             slice, list or table, so nothing is indexed, measured or iterated but a fold's \
             range\n\
             note: {instead}"
        ),
    )
}

/// A doc comment is kept, to be re-emitted on an entry; any other attribute
/// asks for something the expansion cannot honor.
fn refuse_attributes(attrs: &[syn::Attribute]) -> syn::Result<()> {
    match attrs.iter().find(|attr| !attr.path().is_ident("doc")) {
        Some(attr) => Err(syn::Error::new_spanned(
            attr,
            "an attribute in a `kernel!` block\n\
             \n\
             note: only doc comments are kept, and re-emitted on an entry's host function",
        )),
        None => Ok(()),
    }
}

/// A record keeps its attributes, and a field its own, on its host twin — a
/// derive, a lint level, a `cfg_attr` — as rustc reads them there; a derive
/// the twin already has is rustc's conflicting impl. Two are refused because
/// they would make the twin disagree with the program: `repr`, since the
/// layout is the language's (`#[repr(C)]`, the fields in order), and `cfg`,
/// since a record's fields are its entries' uniforms on every
/// configuration.
fn refuse_layout_attributes(attrs: &[syn::Attribute]) -> syn::Result<()> {
    let layout = |attr: &&syn::Attribute| ["repr", "cfg"].iter().any(|n| attr.path().is_ident(n));
    match attrs.iter().find(layout) {
        Some(attr) => Err(syn::Error::new_spanned(
            attr,
            format!(
                "a `repr` or a `cfg` on a record\n\
                 \n\
                 note: a record is `#[repr(C)]`, its fields in order, and its fields are its \
                 entries' uniforms on every configuration (§1.3 of {PLAN}); any other \
                 attribute is kept"
            ),
        )),
        None => Ok(()),
    }
}

/// A `const` takes no generics: it is one value, evaluated at expansion.
fn refuse_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(param) = generics.params.first() {
        return Err(syn::Error::new_spanned(
            param,
            match param {
                syn::GenericParam::Const(_) => {
                    "a `const` generic in a `kernel!` block: a count that changes per call is an entry parameter `n: usize`"
                }
                _ => "generics in a `kernel!` block: nothing in a kernel is generic",
            },
        ));
    }
    refuse_where_clause(generics, "a `kernel!` item")
}

/// The one refusal of a `where` clause, on whatever `what` is: with no
/// generics to bound, a clause has nothing to say.
fn refuse_where_clause(generics: &syn::Generics, what: &str) -> syn::Result<()> {
    match &generics.where_clause {
        Some(where_clause) => Err(syn::Error::new_spanned(
            // An empty clause prints no tokens, so spanning the clause would
            // fall back to the whole invocation; the keyword is always there.
            where_clause.where_token,
            format!("a `where` clause on {what}: there are no generics to bound"),
        )),
        None => Ok(()),
    }
}
