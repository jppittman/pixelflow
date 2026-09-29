//! # Parser
//!
//! Parses the kernel DSL from token stream to AST.
//!
//! ## Grammar
//!
//! A `kernel!` is a block of items, parsed by syn as a `File`; the closure
//! form is sugar for a block with one entry. A body is parsed by syn as a Rust
//! expression, so precedence and associativity are Rust's. Of that syntax,
//! this is the kernel language — what every stage accepts:
//!
//! ```text
//! kernel  ::= item*                          -- the items form
//!           | '|' params '|' expr            -- sugar: one entry, its type inferred
//! item    ::= VIS? 'struct' RECORD '{' (field (',' field)* ','?)? '}'
//!                                            -- a record of `f32` fields
//!           | VIS? 'const' IDENT ':' 'f32' '=' cexpr ';'
//!           | VIS? 'const' IDENT ':' 'usize' '=' iexpr ';'
//!           | VIS 'fn' IDENT structural? '(' params ')' '->' type block
//!                                            -- an entry
//!           | 'fn' IDENT '(' params ')' '->' type block
//!                                            -- a helper: private, no generics
//! field   ::= VIS? IDENT ':' 'f32'
//! structural ::= '<' 'const' IDENT ':' 'usize' (',' 'const' IDENT ':' 'usize')* ','? '>'
//!                                            -- an entry's structural parameters
//! params  ::= (param (',' param)* ','?)?
//! param   ::= IDENT ':' type
//! type    ::= 'f32' | 'bool' | RECORD        -- an entry's parameters are `f32`s
//!                                            -- and records: its uniforms
//!
//! cexpr   ::= cexpr ('+' | '-' | '*' | '/') cexpr   -- an `f32` const's initializer
//!           | '-' cexpr | '(' cexpr ')'            -- or an integral's bounds,
//!           | IDENT 'as' 'f32'                     -- evaluated at expansion,
//!           | IDENT | LITERAL                      -- per operation in f32
//! iexpr   ::= iexpr ('+' | '-' | '*' | '/') iexpr   -- a `usize`: a range's bounds
//!           | '(' iexpr ')'                        -- or a `usize` const's
//!           | IDENT | INTEGER                      -- initializer; each operation
//!                                                  -- checked as rustc checks it,
//!                                                  -- at expansion, or when the
//!                                                  -- host fn is instantiated if
//!                                                  -- it names a structural
//!                                                  -- parameter
//!
//! expr    ::= expr binop expr
//!           | '-' expr
//!           | expr '.' METHOD '(' (expr (',' expr)*)? ')'
//!           | expr '.' IDENT                       -- a record's field
//!           | PROJECTION '(' expr ')'
//!           | IDENT '(' (expr (',' expr)*)? ')'    -- a helper, inlined
//!           | 'if' expr block 'else' (block | 'if' …)   -- the choice
//!           | fold
//!           | integral
//!           | 'monotone_root' '(' expr ',' expr ',' expr ')'   -- τ(δ), from (δ, step, bend)
//!           | IDENT 'as' 'f32'         -- a `usize` (a fold's index, a `usize`
//!                                      -- const or a structural parameter) as
//!                                      -- a value
//!           | '(' expr ')'
//!           | block
//!           | IDENT                    -- X, Y (in an entry), a parameter, an
//!                                      -- `f32` const, or a `let` in scope
//!           | LITERAL                  -- an integer or a float, as its f32
//! fold    ::= range '.map' '(' binder expr ')' '.sum' F32? '(' ')'       -- Σ
//!           | range '.map' '(' binder expr ')' '.product' F32? '(' ')'   -- Π
//!           | range '.map' '(' binder expr ')'
//!                 '.fold' '(' 'f32::INFINITY' ',' 'f32::min' ')'         -- min
//!           | range '.map' '(' binder expr ')'
//!                 '.fold' '(' 'f32::NEG_INFINITY' ',' 'f32::max' ')'     -- max
//!           | range '.any' '(' binder expr ')'                           -- ∃, of bools
//!           | range '.all' '(' binder expr ')'                           -- ∀, of bools
//! range   ::= '(' iexpr '..' iexpr ')'     -- half-open, constant, forwards
//! binder  ::= '|' IDENT '|'                -- the index: a `usize`
//! integral ::= 'integral' '(' cexpr '..' cexpr ',' '|' IDENT '|' expr ')'
//!                                          -- ∫ over [lo, hi); the variable an `f32`
//!           | 'area' '(' '|' IDENT ',' IDENT '|' expr ')'
//!                                          -- the pixel: integral(-H..H, |v|
//!                                          -- integral(-H..H, |u| expr)), H the IR's
//!                                          -- PIXEL_HALF_WIDTH
//! F32     ::= '::' '<' 'f32' '>'
//! binop   ::= '+' | '-' | '*' | '/'
//!           | '<' | '<=' | '>' | '>=' | '==' | '!='    -- a comparison: a bool
//!           | '&' | '|'                                -- bools combine
//! block   ::= '{' stmt* expr '}'
//! stmt    ::= 'let' IDENT (':' type)? '=' expr ';'
//!           | expr ';'
//!
//! VIS        -- `pub`, `pub(crate)`, …: a visibility, kept on the host item
//! RECORD     -- the name of one of the block's records
//! METHOD     -- an `OpKind` method, a `LIBRARY_METHODS` composition, or `clone`
//! PROJECTION -- V, DX, DY, DXX, DXY, DYY
//! ```
//!
//! A `let` is scoped as Rust scopes it (`crate::symbol`), and so is a fold's
//! index: its body sees the enclosing bindings, an enclosing fold's index
//! among them; so is an integral's variable. Nothing shadows `X`, `Y`, a
//! `const`, a structural parameter or a `fn`: a parameter, a `let`, an index
//! or a variable of that name is refused. `X` and `Y` appear only in an
//! entry; a helper takes its coordinates as arguments
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.2). A fold
//! (plan §1.5) denotes ⊕ of its body over `i ∈ [a, b)`, and its monoid's
//! identity when the range is empty; the syntax never unrolls it. An
//! integral denotes ∫ of its body over `u ∈ [lo, hi)`; whether it closes is
//! the e-graph's, and one it leaves open is legalized by quadrature.
//! `integral`, `area` and `monotone_root` are the language's: a call to one
//! always means it, and no `const`, `fn` or record takes one of their names.
//!
//! Binding times (plan §1.4). An entry's `const N: usize` generics are
//! **structural**: each value is its own program, and a body reads one only
//! as a count. Every parameter is a **uniform**: an `f32` is one, a record is
//! one per field, and the host function's arguments are their values for
//! that call — never constants folded in. A record is written by name: a
//! parameter, a `let` alias of one (`let q = p;`), an argument passed on, or
//! the base of a field read.
//!
//! A record and its fields keep their attributes, re-emitted on the host
//! struct; every other item keeps only its doc comments.
//!
//! Refused here, with a span, at the token: an item that is not a `struct`, a
//! `const` or a `fn`; a record that is generic, a tuple struct or a unit
//! struct, and a `repr` or a `cfg` on a record or its field (§1.3); an
//! attribute other than a doc comment on a `const` or a `fn`; generics on a
//! helper (B3) and on a `const`, and any generic of an entry but a
//! `const N: usize` without a default; a parameter typed as a closure
//! (Phase D); a `fn` without a declared return type; an `if` without an
//! `else` or an `if let`; `loop`/`while`/`for`; assignment; `return`; a
//! closure anywhere but a fold's or an integral's; a tuple and a tuple field
//! (B3, D7's front half); a record literal (Phase D, D7); a range anywhere
//! but a fold's or an integral's; an `integral` or an `area` of any other
//! shape than the two above (an argument missing or extra, the interval not
//! a range, the closure's parameters not one plain name for `integral` and
//! two for `area`, a parameter with a type, a qualified closure or one with
//! a return type); an inclusive or open-ended range; a method on a range or
//! a mapped range that is not one of the fold's spellings; `.map`, `.any` or
//! `.all` over anything but a range (iterating a family of records is B3); a
//! `.fold` whose arguments are not one of the two above; a fold's closure
//! with a type annotation, a pattern, more than one parameter, a return type
//! or a qualifier; type arguments on a method (`::<f32>` on `.sum` and
//! `.product` excepted); an `as` to any type but `f32`; a path or a call
//! from outside the block; `%` and `!` (no IR op); a `let` whose pattern is
//! not a plain name (`mut`, `ref`, `@`; destructuring is B3); a `let`
//! without an initializer; `let … else`; an item or a macro inside a block;
//! an operator not in the table above; a literal that is not a number; a
//! literal suffixed with a type other than `f32`; an integer past `u128`; a
//! float past `f32`'s range; and any other Rust expression syntax, named in
//! the refusal. Nothing is passed through for a later stage to refuse: the
//! AST holds only what the language means.
//!
//! Parsed, and refused by `sema`: an unbound or retired name; a coordinate in
//! a helper; a call to an entry or to an unknown function; `monotone_root`
//! with other than three `f32`s; a `const`, `fn` or record named
//! `integral`, `area` or `monotone_root`; recursion; an unknown method or a
//! known one at the wrong arity; a type error (every expression is an `f32`
//! or a `bool`; a `usize` is used only as `i as f32`; a record only by name;
//! an integrand is an `f32`); a record's field that is not an `f32` (§1.3); a
//! record returned, built, chosen by an `if`, or used in arithmetic or a
//! comparison (Phase D, D7); an unknown field; an `as` of anything but a
//! `usize` name; an integer an `f32` does not hold exactly where a value is
//! expected; a `const` whose initializer is not a `cexpr` or an `iexpr`; a
//! negated `usize`; a range whose bounds are not constant or that runs
//! backwards; an integral whose bounds are not constant (a structural
//! parameter among them) or are not an interval the IR admits
//! (`IntervalFold::try_new`: finite, `lo < hi`, a finite length); `.at()`,
//! `.constant()`, `.collapse()`; a block with no final expression; and an
//! `Args` record's name taken twice.
//!
//! Refused by lowering, where the language meets the IR's widths: a range
//! bound past 2²⁴, the last integer bound a fold's index — an `f32` lane —
//! names exactly, and folds and integrals nested deeper than the IR has
//! binders. A range over a structural parameter is refused the same ways
//! when its host function is instantiated, by rustc, in a `const` block.
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.

use crate::PLAN;
use crate::ast::{
    AREA, BinaryExpr, BinaryOp, BlockExpr, CallExpr, CastExpr, ConstItem, Expr, FieldExpr, FnItem,
    FoldExpr, INTEGRAL, IdentExpr, IfExpr, IntegralBounds, IntegralExpr, KernelDef, LetStmt,
    Literal, LiteralExpr, MethodCallExpr, Param, RangeExpr, RecordField, RecordItem, Reduction,
    Spelling, Stmt, UnaryExpr, UnaryOp,
};
use proc_macro2::{Span, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Pat, Token, Type};

/// The name of the closure sugar's one entry. It is never emitted: the
/// closure form expands to an expression, not to a named function.
const SUGAR_ENTRY: &str = "__kernel";

/// Parse kernel input from token stream.
pub fn parse(input: TokenStream) -> syn::Result<KernelDef> {
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

/// `|param: Type, ...| body`: sugar for a block with one entry, its return
/// type inferred.
fn parse_closure(input: ParseStream) -> syn::Result<KernelDef> {
    input.parse::<Token![|]>()?;

    let mut params = Vec::new();

    // Handle empty params: || body
    if !input.peek(Token![|]) {
        // Parse parameter list manually
        loop {
            // Parse identifier
            let ident: syn::Ident = input.parse()?;
            // Parse colon
            input.parse::<Token![:]>()?;
            // Parse type
            let ty: Type = input.parse()?;

            params.push(Param {
                name: ident,
                ty: Box::new(ty),
            });

            // Check for comma or end of params
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                // Allow trailing comma before |
                if input.peek(Token![|]) {
                    break;
                }
            } else {
                break;
            }
        }
    }

    input.parse::<Token![|]>()?;

    // Parse the body expression
    let syn_expr: syn::Expr = input.parse()?;
    let body = convert_expr(syn_expr)?;

    let entry = FnItem {
        attrs: Vec::new(),
        vis: syn::Visibility::Public(Default::default()),
        name: syn::Ident::new(SUGAR_ENTRY, Span::call_site()),
        structural: Vec::new(),
        params,
        ret: None,
        body,
    };
    Ok(KernelDef {
        spelling: Spelling::Closure,
        records: Vec::new(),
        consts: Vec::new(),
        fns: vec![entry],
    })
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
    let mut def = KernelDef {
        spelling: Spelling::Items,
        records: Vec::new(),
        consts: Vec::new(),
        fns: Vec::new(),
    };
    for item in file.items {
        match item {
            syn::Item::Struct(item) => def.records.push(convert_record(item)?),
            syn::Item::Const(item) => def.consts.push(convert_const(item)?),
            syn::Item::Fn(item) => def.fns.push(convert_fn(item)?),
            other => return Err(refuse_item(&other)),
        }
    }
    Ok(def)
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
    if let Some(where_clause) = &item.generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "a `where` clause on a record: there are no generics to bound",
        ));
    }
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
        init: convert_expr(*item.expr)?,
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
    let structural = match item.vis {
        syn::Visibility::Inherited => {
            refuse_a_helpers_generics(&sig.generics)?;
            Vec::new()
        }
        _ => structural_parameters(&sig.generics)?,
    };

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
        structural,
        params,
        ret: Some(ret),
        body: Expr::Block(convert_block(*item.block)?),
    })
}

/// An entry's structural parameters: its generics, each `const N: usize`
/// (plan §1.4). A type or lifetime parameter, a const of another type, a
/// default and a `where` clause are refused: the language's one value bound
/// when its program is compiled is a count.
fn structural_parameters(generics: &syn::Generics) -> syn::Result<Vec<syn::Ident>> {
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "a `where` clause in a `kernel!` block: a structural parameter is a `usize`, and \
             there is nothing to bound",
        ));
    }
    let mut structural = Vec::with_capacity(generics.params.len());
    for param in &generics.params {
        let syn::GenericParam::Const(count) = param else {
            return Err(syn::Error::new_spanned(
                param,
                format!(
                    "a type or lifetime parameter on an entry\n\
                     \n\
                     note: an entry's generics are its structural parameters, `const N: usize`, \
                     and each value is its own program (§1.4 and B3 of {PLAN})"
                ),
            ));
        };
        refuse_attributes(&count.attrs)?;
        if !is_usize(&count.ty) {
            return Err(syn::Error::new_spanned(
                &count.ty,
                "a structural parameter is a `usize`: a count, which sizes a program's shape",
            ));
        }
        if let Some(default) = &count.default {
            return Err(syn::Error::new_spanned(
                default,
                "a default for a structural parameter: a call names the program it wants, \
                 `entry::<4>(…)`",
            ));
        }
        structural.push(count.ident.clone());
    }
    Ok(structural)
}

/// A helper takes no generics: it is inlined into an entry, and reads the
/// entry's structural parameters only through its arguments.
fn refuse_a_helpers_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(param) = generics.params.first() {
        return Err(syn::Error::new_spanned(
            param,
            format!(
                "generics on a helper\n\
                 \n\
                 note: a helper is inlined into an entry, and reads the entry's structural \
                 parameters through its arguments: pass `n as f32`\n\
                 note: structural parameters are an entry's, `pub fn f<const N: usize>`; a \
                 helper's are B3 of {PLAN}"
            ),
        ));
    }
    refuse_generics(generics)
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
    match &*typed.ty {
        Type::ImplTrait(_) | Type::BareFn(_) | Type::TraitObject(_) => {
            return Err(syn::Error::new_spanned(
                &typed.ty,
                format!(
                    "a kernel-typed parameter\n\
                     \n\
                     note: passing a function to a kernel `fn` is Phase D of {PLAN}; this \
                     parameter is an `f32`, a `bool` or one of the block's records"
                ),
            ));
        }
        _ => {}
    }
    Ok(Param { name, ty: typed.ty })
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
            format!(
                "generics in a `kernel!` block\n\
                 \n\
                 note: structural parameters, `const N: usize`, are an entry's (§1.4 of {PLAN})"
            ),
        ));
    }
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "a `where` clause in a `kernel!` block: there are no generics to bound",
        ));
    }
    Ok(())
}

/// Convert syn::Expr to our AST Expr.
fn convert_expr(expr: syn::Expr) -> syn::Result<Expr> {
    match expr {
        syn::Expr::Path(expr_path) => {
            // Simple identifier: X, cx, etc.
            if expr_path.path.segments.len() == 1 && expr_path.qself.is_none() {
                let segment = &expr_path.path.segments[0];
                if segment.arguments.is_empty() {
                    return Ok(Expr::Ident(IdentExpr {
                        name: segment.ident.clone(),
                        span: segment.ident.span(),
                    }));
                }
            }
            // A qualified path names something outside the block.
            Err(syn::Error::new_spanned(
                &expr_path,
                format!(
                    "a path in a kernel body: `{}`\n\
                     \n\
                     note: a kernel body sees X, Y (in an entry), its parameters, the block's \
                     `const`s, and the `let` bindings in scope; a name from the enclosing Rust \
                     scope is not captured\n\
                     help: declare the value as a `const` of this block, or as a parameter of \
                     this kernel",
                    quote::quote!(#expr_path)
                ),
            ))
        }

        syn::Expr::Lit(expr_lit) => Ok(Expr::Literal(LiteralExpr {
            value: literal_value(&expr_lit.lit)?,
            span: expr_lit.lit.span(),
        })),

        syn::Expr::Binary(expr_binary) => {
            let op = binary_op(&expr_binary.op)?;
            let span = expr_binary.op.span();
            let lhs = convert_expr(*expr_binary.left)?;
            let rhs = convert_expr(*expr_binary.right)?;
            Ok(Expr::Binary(BinaryExpr {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            }))
        }

        syn::Expr::Unary(expr_unary) => {
            let op = unary_op(&expr_unary.op)?;
            let span = expr_unary.op.span();
            let operand = convert_expr(*expr_unary.expr)?;
            Ok(Expr::Unary(UnaryExpr {
                op,
                operand: Box::new(operand),
                span,
            }))
        }

        syn::Expr::MethodCall(expr_method) => {
            if let Some(range) = a_range(&expr_method.receiver) {
                return convert_range_method(range, &expr_method);
            }
            if let Some((range, map)) = a_mapped_range(&expr_method.receiver) {
                return convert_fold_terminal(range, map, &expr_method);
            }
            // Built before the receiver is converted, returned after it: the
            // receiver's own refusal, the innermost, comes first, as rustc's
            // does (`(0..4).rev().map(…)` is about `.rev`).
            let family = a_family_iterated(&expr_method);
            if let Some(arguments) = &expr_method.turbofish {
                return Err(syn::Error::new_spanned(
                    arguments,
                    "type arguments on a method in a kernel body\n\
                     \n\
                     note: every value in a kernel body is an `f32` or a `bool`; no method is \
                     generic",
                ));
            }
            let receiver = convert_expr(*expr_method.receiver)?;
            if let Some(refusal) = family {
                return Err(refusal);
            }
            let args = expr_method
                .args
                .into_iter()
                .map(convert_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(receiver),
                span: expr_method.method.span(),
                method: expr_method.method,
                args,
            }))
        }

        syn::Expr::Call(expr_call) => {
            if let Some(integral) = an_integral(&expr_call) {
                return integral;
            }
            // Free function call: V(m), DX(expr), a helper, etc.
            // Extract the function name from the callee
            if let syn::Expr::Path(ref path) = *expr_call.func {
                if path.path.segments.len() == 1 && path.qself.is_none() {
                    let func = path.path.segments[0].ident.clone();
                    let args = expr_call
                        .args
                        .into_iter()
                        .map(convert_expr)
                        .collect::<syn::Result<Vec<_>>>()?;
                    return Ok(Expr::Call(CallExpr {
                        span: func.span(),
                        func,
                        args,
                    }));
                }
            }
            // A qualified callee names something outside the block.
            Err(syn::Error::new_spanned(
                &expr_call.func,
                format!(
                    "a qualified call in a kernel body: `{}`\n\
                     \n\
                     note: a body calls the block's helpers (private `fn`s) and the \
                     projections V, DX, DY, DXX, DXY, DYY, by name; an operation on a value is \
                     a method, `x.sqrt()`",
                    quote::quote!(#expr_call)
                ),
            ))
        }

        syn::Expr::If(expr_if) => convert_if(expr_if),

        syn::Expr::Paren(expr_paren) => {
            let inner = convert_expr(*expr_paren.expr)?;
            Ok(Expr::Paren(Box::new(inner)))
        }

        // A value with parts, built in a body. A record enters a kernel as a
        // parameter and is read by field; a tuple is flattened in the front
        // end as a record is (D7's front half, B3), and a record built or
        // returned is Phase D (D7).
        syn::Expr::Tuple(expr_tuple) => Err(syn::Error::new(
            expr_tuple.paren_token.span.join(),
            format!(
                "a tuple in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32` or a `bool`; a record \
                 parameter is read by field, `p.x0`\n\
                 note: tuples are flattened in the front end as records are (D7's front \
                 half), which is B3 of {PLAN}"
            ),
        )),

        syn::Expr::Struct(expr_struct) => Err(syn::Error::new_spanned(
            &expr_struct.path,
            format!(
                "a record literal in a kernel body\n\
                 \n\
                 note: a record enters a kernel as a parameter, one uniform per field, and a \
                 body reads its fields, `p.x0`\n\
                 note: building a record, and returning one, are Phase D (D7 of {PLAN})"
            ),
        )),

        syn::Expr::Field(expr_field) => convert_field(expr_field),

        syn::Expr::Range(expr_range) => Err(syn::Error::new_spanned(
            expr_range,
            format!(
                "a range in a kernel body\n\
                 \n\
                 note: a range is the domain of a fold: {FOLD_SPELLINGS}\n\
                 note: or of an integral: {INTEGRAL_SPELLINGS}"
            ),
        )),

        syn::Expr::Cast(expr_cast) => convert_cast(expr_cast),

        syn::Expr::Block(expr_block) => {
            let block = convert_block(expr_block.block)?;
            Ok(Expr::Block(block))
        }

        // Iteration with state has no denotation in a DAG. A bounded
        // reduction is a fold over a constant range.
        syn::Expr::Loop(_) | syn::Expr::While(_) | syn::Expr::ForLoop(_) => {
            Err(syn::Error::new_spanned(
                expr,
                format!(
                    "a loop in a kernel body\n\
                     \n\
                     note: the language is a DAG: nothing iterates with state\n\
                     note: a bounded reduction over a constant range is a fold: {FOLD_SPELLINGS}"
                ),
            ))
        }

        syn::Expr::Assign(_) => Err(syn::Error::new_spanned(
            expr,
            "assignment in a kernel body\n\
             \n\
             note: a kernel body binds a name once, with `let`; nothing is mutable",
        )),

        syn::Expr::Return(_) => Err(syn::Error::new_spanned(
            expr,
            "`return` in a kernel body\n\
             \n\
             note: a body is an expression; its value is its final expression",
        )),

        syn::Expr::Closure(_) => Err(syn::Error::new_spanned(
            expr,
            format!(
                "a closure in a kernel body\n\
                 \n\
                 note: a closure is the body of a fold or an integral, and nothing else: \
                 {FOLD_SPELLINGS}; {INTEGRAL_SPELLINGS}\n\
                 note: a function as an argument is Phase D of {PLAN}; a private `fn` in the \
                 block is a helper, called by name"
            ),
        )),

        // `syn::Expr` is `#[non_exhaustive]`, so the arm stays; it refuses at
        // the token, naming the syntax, and nothing is passed through.
        other => Err(syn::Error::new_spanned(
            &other,
            format!(
                "unsupported expression in a kernel body: `{}`",
                quote::quote!(#other)
            ),
        )),
    }
}

/// `p.x0`: a named field. Which record `p` is, and whether it has the field,
/// are `sema`'s questions. A tuple's field, `p.0`, is refused: a record's
/// fields have names.
fn convert_field(expr_field: syn::ExprField) -> syn::Result<Expr> {
    let member = match expr_field.member {
        syn::Member::Named(member) => member,
        syn::Member::Unnamed(index) => {
            return Err(syn::Error::new_spanned(
                index,
                format!(
                    "a tuple's field in a kernel body\n\
                     \n\
                     note: a record's fields have names, `p.x0`; tuples are B3 (D7's front \
                     half) of {PLAN}"
                ),
            ));
        }
    };
    Ok(Expr::Field(FieldExpr {
        base: Box::new(convert_expr(*expr_field.base)?),
        span: member.span(),
        member,
    }))
}

/// `if c { a } else { b }`, and `else if` chains. The `else` is required —
/// there is no unit for an `if` without one to be — and `if let` is not a
/// choice between two values.
fn convert_if(expr_if: syn::ExprIf) -> syn::Result<Expr> {
    let span = expr_if.if_token.span;
    if let syn::Expr::Let(pattern) = *expr_if.cond {
        return Err(syn::Error::new_spanned(
            pattern,
            "`if let` in a kernel body\n\
             \n\
             note: an `if` chooses between two values by a `bool`; nothing here is a pattern",
        ));
    }
    let Some((_, else_expr)) = expr_if.else_branch else {
        return Err(syn::Error::new(
            span,
            "an `if` without an `else`\n\
             \n\
             note: an `if` is the choice between two values, so it needs both; a kernel \
             body has no unit for the missing arm to be",
        ));
    };
    let cond = convert_expr(*expr_if.cond)?;
    let then_branch = convert_block(expr_if.then_branch)?;
    let else_branch = convert_expr(*else_expr)?;
    Ok(Expr::If(IfExpr {
        cond: Box::new(cond),
        then_branch,
        else_branch: Box::new(else_branch),
        span,
    }))
}

// ─────────────────────────────── folds ───────────────────────────────

/// Every spelling of a fold, as a refusal of any other names them.
const FOLD_SPELLINGS: &str = "`(a..b).map(|i| e).sum()`, `.product()`, \
     `.fold(f32::INFINITY, f32::min)` or `.fold(f32::NEG_INFINITY, f32::max)`, and \
     `(a..b).any(|i| m)` or `.all(|i| m)`";

/// The two `.fold`s that name a monoid, by its identity and its operation.
/// A `.fold` with any other arguments is a loop with state in disguise, or a
/// monoid whose identity is not the one written, so it is refused.
const FOLD_MONOIDS: [(Reduction, [&str; 2], [&str; 2]); 2] = [
    (Reduction::Min, ["f32", "INFINITY"], ["f32", "min"]),
    (Reduction::Max, ["f32", "NEG_INFINITY"], ["f32", "max"]),
];

/// The range a method is called on, through its parentheses: the `(a..b)`
/// of `(a..b).any(…)`.
fn a_range(receiver: &syn::Expr) -> Option<&syn::ExprRange> {
    match receiver {
        syn::Expr::Paren(paren) => a_range(&paren.expr),
        syn::Expr::Range(range) => Some(range),
        _ => None,
    }
}

/// `(a..b).map(f)`: the range and the `map` call a fold's terminal method is
/// called on.
fn a_mapped_range(receiver: &syn::Expr) -> Option<(&syn::ExprRange, &syn::ExprMethodCall)> {
    let syn::Expr::MethodCall(map) = receiver else {
        return None;
    };
    if map.method != "map" {
        return None;
    }
    Some((a_range(&map.receiver)?, map))
}

/// The methods that iterate: a fold's `.map`, `.any` and `.all`.
const ITERATING_METHODS: [&str; 3] = ["map", "any", "all"];

/// The refusal of one of [`ITERATING_METHODS`] with a closure, called on
/// anything but a range: the spelling of iterating a family of records
/// (plan §1.6), which is B3, and not a closure out of place, which the
/// closure's own refusal would say.
fn a_family_iterated(call: &syn::ExprMethodCall) -> Option<syn::Error> {
    if !ITERATING_METHODS.iter().any(|name| call.method == name) {
        return None;
    }
    if !call
        .args
        .iter()
        .any(|arg| matches!(arg, syn::Expr::Closure(_)))
    {
        return None;
    }
    let receiver = &call.receiver;
    Some(syn::Error::new_spanned(
        receiver,
        format!(
            "`.{}` over `{}`, which is not a range\n\
             \n\
             note: a fold iterates a constant range: {FOLD_SPELLINGS}\n\
             note: iterating a family of records (`pieces.map(|p| …)`) is B3 of {PLAN}",
            call.method,
            quote::quote!(#receiver)
        ),
    ))
}

/// A method called on a range itself: `.any(|i| m)` and `.all(|i| m)` are
/// folds, and nothing else is a value.
fn convert_range_method(range: &syn::ExprRange, call: &syn::ExprMethodCall) -> syn::Result<Expr> {
    let reduction = match call.method.to_string().as_str() {
        "any" => Reduction::Any,
        "all" => Reduction::All,
        "map" => {
            return Err(syn::Error::new_spanned(
                call,
                format!(
                    "a mapped range is an iterator, not a value\n\
                     \n\
                     note: a fold ends in the reduction that combines its terms: {FOLD_SPELLINGS}"
                ),
            ));
        }
        other => {
            return Err(syn::Error::new(
                call.method.span(),
                format!(
                    "`.{other}` on a range\n\
                     \n\
                     note: a range is the domain of a fold: {FOLD_SPELLINGS}"
                ),
            ));
        }
    };
    refuse_turbofish(call)?;
    let (binder, body) = the_closure(call)?;
    Ok(Expr::Fold(FoldExpr {
        reduction,
        range: convert_range(range)?,
        binder,
        body: Box::new(body),
        span: call.method.span(),
    }))
}

/// The method that ends `(a..b).map(|i| e)`: the reduction the fold combines
/// its terms under.
fn convert_fold_terminal(
    range: &syn::ExprRange,
    map: &syn::ExprMethodCall,
    call: &syn::ExprMethodCall,
) -> syn::Result<Expr> {
    let reduction = match call.method.to_string().as_str() {
        "sum" => sum_or_product(call, Reduction::Sum)?,
        "product" => sum_or_product(call, Reduction::Product)?,
        "fold" => fold_monoid(call)?,
        other => {
            return Err(syn::Error::new(
                call.method.span(),
                format!(
                    "`.{other}` does not end a fold\n\
                     \n\
                     note: a fold is spelled {FOLD_SPELLINGS}"
                ),
            ));
        }
    };
    refuse_turbofish(map)?;
    let (binder, body) = the_closure(map)?;
    Ok(Expr::Fold(FoldExpr {
        reduction,
        range: convert_range(range)?,
        binder,
        body: Box::new(body),
        span: call.method.span(),
    }))
}

/// `.sum()` and `.product()` take no arguments, and a type argument only if
/// it is the one type the terms have: `.sum::<f32>()`, which rustc needs
/// wherever nothing else says what the sum is.
fn sum_or_product(call: &syn::ExprMethodCall, reduction: Reduction) -> syn::Result<Reduction> {
    if let Some(arguments) = &call.turbofish {
        let names_f32 = match arguments.args.iter().collect::<Vec<_>>().as_slice() {
            [syn::GenericArgument::Type(ty)] => is_f32(ty),
            _ => false,
        };
        if !names_f32 {
            return Err(syn::Error::new_spanned(
                arguments,
                format!(
                    "`.{}` over anything but `f32`s\n\
                     \n\
                     note: a fold's terms are `f32`s; the one type argument it takes is \
                     `::<f32>`",
                    call.method
                ),
            ));
        }
    }
    if let Some(arg) = call.args.first() {
        return Err(syn::Error::new_spanned(
            arg,
            format!("`.{}()` takes no arguments", call.method),
        ));
    }
    Ok(reduction)
}

/// Whether `ty` is exactly `f32`: the one type a fold's terms, and the one
/// conversion's target, may name.
fn is_f32(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32"))
}

/// Whether `ty` is exactly `usize`: the one type a structural parameter has.
fn is_usize(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident("usize"))
}

/// `operand as f32`. The conversion's target is always `f32`, so any other
/// is refused here and the AST does not hold one; what may be converted — a
/// `usize`, by name — is `sema`'s question, since it needs the name's type.
fn convert_cast(expr_cast: syn::ExprCast) -> syn::Result<Expr> {
    if !is_f32(&expr_cast.ty) {
        return Err(syn::Error::new_spanned(
            &expr_cast.ty,
            "`as` converts a `usize` to an `f32`, and to nothing else\n\
             \n\
             note: every value in a kernel body is an `f32` or a `bool`, and a `usize` (a \
             fold's index or a `usize` const) becomes one only as `i as f32`\n\
             help: write `as f32`",
        ));
    }
    Ok(Expr::Cast(CastExpr {
        span: expr_cast.as_token.span,
        operand: Box::new(convert_expr(*expr_cast.expr)?),
    }))
}

/// `.fold(identity, operation)` names a monoid only as one of
/// [`FOLD_MONOIDS`]; any other `.fold` is refused, naming them.
fn fold_monoid(call: &syn::ExprMethodCall) -> syn::Result<Reduction> {
    refuse_turbofish(call)?;
    let refusal = || {
        syn::Error::new(
            call.method.span(),
            "this `.fold` names no monoid the language has\n\
             \n\
             note: a `.fold` is `.fold(f32::INFINITY, f32::min)` or \
             `.fold(f32::NEG_INFINITY, f32::max)`: the minimum and the maximum, each with \
             its identity\n\
             note: a sum is `.sum()` and a product `.product()`; any other fold carries \
             state from one term to the next, and the language is a DAG",
        )
    };
    let [identity, operation] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(refusal());
    };
    FOLD_MONOIDS
        .iter()
        .find(|(_, id, op)| is_path(identity, id) && is_path(operation, op))
        .map(|(reduction, ..)| *reduction)
        .ok_or_else(refusal)
}

/// Whether `expr` is exactly the path `segments`: `f32::INFINITY`.
fn is_path(expr: &syn::Expr, segments: &[&str; 2]) -> bool {
    let syn::Expr::Path(path) = expr else {
        return false;
    };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == segments.len()
        && path
            .path
            .segments
            .iter()
            .zip(segments)
            .all(|(segment, name)| segment.arguments.is_empty() && segment.ident == name)
}

/// A method of a fold that takes no type argument.
fn refuse_turbofish(call: &syn::ExprMethodCall) -> syn::Result<()> {
    match &call.turbofish {
        Some(arguments) => Err(syn::Error::new_spanned(
            arguments,
            format!(
                "`.{}` takes no type arguments in a kernel body",
                call.method
            ),
        )),
        None => Ok(()),
    }
}

/// The one argument of a fold's `.map`, `.any` or `.all`: a closure whose
/// parameter is the fold's index and whose body is the fold's body.
fn the_closure(call: &syn::ExprMethodCall) -> syn::Result<(syn::Ident, Expr)> {
    let [syn::Expr::Closure(closure)] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new(
            call.method.span(),
            format!(
                "`.{}` takes one argument, the closure `|i| body`\n\
                 \n\
                 note: its parameter is the fold's index and its body is what the fold \
                 combines",
                call.method
            ),
        ));
    };
    let qualified = !closure.attrs.is_empty()
        || closure.lifetimes.is_some()
        || closure.constness.is_some()
        || closure.movability.is_some()
        || closure.asyncness.is_some()
        || closure.capture.is_some();
    if qualified {
        return Err(syn::Error::new_spanned(
            closure,
            "a fold's closure is `|i| body`, unqualified\n\
             \n\
             note: it is the fold's body, not a value: nothing is captured, moved or awaited",
        ));
    }
    if let syn::ReturnType::Type(_, ty) = &closure.output {
        return Err(syn::Error::new_spanned(
            ty,
            "a fold's closure declares no return type: its body is a term of the fold, an \
             `f32` for a sum, product, min or max, a `bool` for `any` or `all`",
        ));
    }
    let [param] = closure.inputs.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new_spanned(
            &closure.inputs,
            "a fold's closure takes one parameter, the index it ranges over",
        ));
    };
    let binder = match param {
        Pat::Ident(pat_ident) => plain_name(pat_ident, param)?,
        Pat::Type(typed) => {
            return Err(syn::Error::new_spanned(
                &typed.ty,
                "a fold's index is a `usize`, always; write the plain name, `|i|`",
            ));
        }
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "a fold's index is a plain name: `|i|`",
            ));
        }
    };
    Ok((binder, convert_expr((*closure.body).clone())?))
}

/// `a..b`: both bounds, half-open. Whether they are constant, and run
/// forwards, is `sema`'s question: it evaluates them.
fn convert_range(range: &syn::ExprRange) -> syn::Result<RangeExpr> {
    if let syn::RangeLimits::Closed(dots) = &range.limits {
        return Err(syn::Error::new_spanned(
            dots,
            "an inclusive range: a fold ranges over a half-open `a..b`\n\
             \n\
             help: write the end one past the last index",
        ));
    }
    let (Some(lo), Some(hi)) = (&range.start, &range.end) else {
        return Err(syn::Error::new_spanned(
            range,
            "a fold's range names both of its bounds: `a..b`",
        ));
    };
    Ok(RangeExpr {
        lo: Box::new(convert_expr((**lo).clone())?),
        hi: Box::new(convert_expr((**hi).clone())?),
        span: range.limits.span(),
    })
}

// ───────────────────────────── integrals ─────────────────────────────

/// Every spelling of an integral, as a refusal of any other names them.
const INTEGRAL_SPELLINGS: &str =
    "`integral(lo..hi, |u| e)`, and `area(|u, v| e)` over the pixel about the sample";

/// How the closure of `integral` or `area` is spelled, for the refusal of
/// any other shape.
struct ClosureShape {
    /// Whose closure it is.
    function: &'static str,
    /// Its one spelling.
    spelling: &'static str,
    /// What its parameters are.
    parameters: &'static str,
}

/// `integral(lo..hi, |u| body)`.
const INTEGRAL_CLOSURE: ClosureShape = ClosureShape {
    function: INTEGRAL,
    spelling: "|u| body",
    parameters: "one parameter, the variable of integration",
};

/// `area(|u, v| body)`.
const AREA_CLOSURE: ClosureShape = ClosureShape {
    function: AREA,
    spelling: "|u, v| body",
    parameters: "two parameters, the offsets across the pixel along X and along Y",
};

/// A call to `integral` or `area`, converted, or `None` for any other call.
///
/// Their arguments are a range and a closure, which a body holds nowhere
/// else, so they are read here, where the closure is still one: the AST
/// holds no closure to hand a later stage. What the bounds are — constant,
/// an interval — is `sema`'s question, since it evaluates them.
fn an_integral(call: &syn::ExprCall) -> Option<syn::Result<Expr>> {
    let syn::Expr::Path(path) = &*call.func else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let name = path.path.get_ident()?;
    match name.to_string().as_str() {
        INTEGRAL => Some(convert_integral(name, call)),
        AREA => Some(convert_area(name, call)),
        _ => None,
    }
}

/// `integral(lo..hi, |u| e)`: ∫ of `e` over `u ∈ [lo, hi)`.
fn convert_integral(name: &syn::Ident, call: &syn::ExprCall) -> syn::Result<Expr> {
    let [interval, closure] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new(
            name.span(),
            format!(
                "`integral` takes two arguments, its interval and the closure `|u| body`, \
                 but {} were supplied\n\
                 \n\
                 note: `integral(lo..hi, |u| e)` is ∫ of `e` over `u ∈ [lo, hi)`, the bounds \
                 constant `f32`s",
                call.args.len()
            ),
        ));
    };
    let Some(range) = a_range(interval) else {
        return Err(syn::Error::new_spanned(
            interval,
            "an integral's first argument is its interval: a constant `lo..hi`",
        ));
    };
    let syn::Expr::Closure(closure) = closure else {
        return Err(syn::Error::new_spanned(
            closure,
            "an integral's second argument is the closure `|u| body`: its parameter is the \
             variable of integration, and its body the integrand",
        ));
    };
    let ([variable], body) = integral_closure(closure, &INTEGRAL_CLOSURE)?;
    Ok(Expr::Integral(IntegralExpr {
        bounds: IntegralBounds::Written(convert_interval(range)?),
        variable,
        body: Box::new(body),
        span: name.span(),
    }))
}

/// `area(|u, v| e)`, the prelude's pixel: exactly
/// `integral(-H..H, |v| integral(-H..H, |u| e))`, `H` the IR's
/// `PIXEL_HALF_WIDTH`, the `v` integral outermost — as `Kernel::area` builds
/// it. The author writes the shift: `area(|u, v| f(X + u, Y + v))` is the
/// builder's `f.area()`.
fn convert_area(name: &syn::Ident, call: &syn::ExprCall) -> syn::Result<Expr> {
    let [syn::Expr::Closure(closure)] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new(
            name.span(),
            "`area` takes one argument, the closure `|u, v| body`\n\
             \n\
             note: `area(|u, v| e)` is ∫∫ of `e` over the pixel about the sample, `u` across it \
             along X and `v` along Y; the body writes the shift, `f(X + u, Y + v)`",
        ));
    };
    let ([u, v], body) = integral_closure(closure, &AREA_CLOSURE)?;
    let across = |variable, body| {
        Expr::Integral(IntegralExpr {
            bounds: IntegralBounds::Pixel,
            variable,
            body: Box::new(body),
            span: name.span(),
        })
    };
    Ok(across(v, across(u, body)))
}

/// The variables and the body of an integral's closure: `N` plain names,
/// each an `f32` always, so none carries a type. Unqualified and with no
/// return type, as a fold's closure is: it is the integrand, not a value.
fn integral_closure<const N: usize>(
    closure: &syn::ExprClosure,
    shape: &ClosureShape,
) -> syn::Result<([syn::Ident; N], Expr)> {
    let ClosureShape {
        function,
        spelling,
        parameters,
    } = shape;
    let qualified = !closure.attrs.is_empty()
        || closure.lifetimes.is_some()
        || closure.constness.is_some()
        || closure.movability.is_some()
        || closure.asyncness.is_some()
        || closure.capture.is_some();
    if qualified {
        return Err(syn::Error::new_spanned(
            closure,
            format!(
                "`{function}`'s closure is `{spelling}`, unqualified\n\
                 \n\
                 note: it is the integrand, not a value: nothing is captured, moved or awaited"
            ),
        ));
    }
    if let syn::ReturnType::Type(_, ty) = &closure.output {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "`{function}`'s closure declares no return type: its body is the integrand, an `f32`"
            ),
        ));
    }
    let mut variables = Vec::with_capacity(N);
    for param in &closure.inputs {
        let variable = match param {
            Pat::Ident(pat_ident) => plain_name(pat_ident, param)?,
            Pat::Type(typed) => {
                return Err(syn::Error::new_spanned(
                    &typed.ty,
                    "an integral's variable is an `f32`, always; write the plain name, `|u|`",
                ));
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "an integral's variable is a plain name: `|u|`",
                ));
            }
        };
        variables.push(variable);
    }
    let variables = variables.try_into().map_err(|_| {
        syn::Error::new(
            closure.or1_token.span(),
            format!("`{function}`'s closure takes {parameters}: `{spelling}`"),
        )
    })?;
    Ok((variables, convert_expr((*closure.body).clone())?))
}

/// An integral's `lo..hi`: both bounds, half-open. Whether they are
/// constant, and an interval the IR admits, is `sema`'s question: it
/// evaluates them.
fn convert_interval(range: &syn::ExprRange) -> syn::Result<RangeExpr> {
    if let syn::RangeLimits::Closed(dots) = &range.limits {
        return Err(syn::Error::new_spanned(
            dots,
            "an inclusive range: an integral is over a half-open `lo..hi`\n\
             \n\
             note: an end is a point, and a point has no length, so `..=` would denote the same \
             integral\n\
             help: write `lo..hi`",
        ));
    }
    let (Some(lo), Some(hi)) = (&range.start, &range.end) else {
        return Err(syn::Error::new_spanned(
            range,
            "an integral's interval names both of its bounds: `lo..hi`",
        ));
    };
    Ok(RangeExpr {
        lo: Box::new(convert_expr((**lo).clone())?),
        hi: Box::new(convert_expr((**hi).clone())?),
        span: range.limits.span(),
    })
}

/// A binary operator, or the refusal that names it. `%` has no IR op, and a
/// compound assignment assigns.
fn binary_op(op: &syn::BinOp) -> syn::Result<BinaryOp> {
    if let Some(op) = BinaryOp::from_syn(op) {
        return Ok(op);
    }
    let spelling = quote::quote!(#op).to_string();
    let message = match op {
        syn::BinOp::AddAssign(_)
        | syn::BinOp::SubAssign(_)
        | syn::BinOp::MulAssign(_)
        | syn::BinOp::DivAssign(_)
        | syn::BinOp::RemAssign(_)
        | syn::BinOp::BitXorAssign(_)
        | syn::BinOp::BitAndAssign(_)
        | syn::BinOp::BitOrAssign(_)
        | syn::BinOp::ShlAssign(_)
        | syn::BinOp::ShrAssign(_) => format!(
            "`{spelling}` assigns, and a kernel body assigns nothing\n\
             \n\
             note: a kernel body binds a name once, with `let`; nothing is mutable"
        ),
        syn::BinOp::Rem(_) => "`%` has no IR op\n\
             \n\
             help: `a - (a / b).floor() * b` is the remainder with the floor's sign"
            .to_string(),
        _ => format!(
            "unsupported binary operator `{spelling}`\n\
             \n\
             note: the kernel! macro supports these binary operators:\n\
             note:   arithmetic: + - * /\n\
             note:   comparison: < <= > >= == != (a `bool`)\n\
             note:   on `bool`s: & |"
        ),
    };
    Err(syn::Error::new_spanned(op, message))
}

/// A unary operator, or the refusal that names it. `!` has no IR op yet.
fn unary_op(op: &syn::UnOp) -> syn::Result<UnaryOp> {
    if let Some(op) = UnaryOp::from_syn(op) {
        return Ok(op);
    }
    let message = match op {
        syn::UnOp::Not(_) => "`!` has no IR op yet\n\
             \n\
             help: write the comparison that means the negation, or swap the arms of the `if`"
            .to_string(),
        _ => format!(
            "unsupported unary operator `{}`\n\
             \n\
             note: the kernel! macro supports `-` (negation); for other unary operations, \
             use method calls like .abs()",
            quote::quote!(#op)
        ),
    };
    Err(syn::Error::new_spanned(op, message))
}

/// Convert a syn::Block to our BlockExpr.
fn convert_block(block: syn::Block) -> syn::Result<BlockExpr> {
    let span = block.brace_token.span.join();
    let mut stmts = Vec::with_capacity(block.stmts.len());
    let mut final_expr = None;

    for (i, stmt) in block.stmts.iter().enumerate() {
        let is_last = i == block.stmts.len() - 1;

        match stmt {
            syn::Stmt::Local(local) => {
                // let binding
                let name = match &local.pat {
                    Pat::Ident(pat_ident) => plain_name(pat_ident, &local.pat)?,
                    Pat::Type(pat_type) => match &*pat_type.pat {
                        Pat::Ident(pat_ident) => plain_name(pat_ident, &local.pat)?,
                        pattern => return Err(refuse_let_pattern(pattern)),
                    },
                    pattern => return Err(refuse_let_pattern(pattern)),
                };

                let ty = match &local.pat {
                    Pat::Type(pat_type) => Some((*pat_type.ty).clone()),
                    _ => None,
                };

                let init = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(
                        &local.pat,
                        "let binding must have an initializer\n\
                         \n\
                         help: provide a value for this binding:\n\
                         help:   let dx = X - cx;",
                    )
                })?;

                // The `else` branch used to be dropped unread. It could never
                // run — a plain name always matches — so accepting it would
                // compile a branch nobody can reach, silently.
                if let Some((else_token, _)) = &init.diverge {
                    return Err(syn::Error::new_spanned(
                        else_token,
                        "`let ... else` is not supported in a kernel body\n\
                         \n\
                         note: a kernel `let` binds a plain name, which always matches, \
                         so the `else` branch could never run\n\
                         \n\
                         help: remove the `else` branch",
                    ));
                }

                let init_expr = convert_expr((*init.expr).clone())?;

                stmts.push(Stmt::Let(Box::new(LetStmt {
                    span: name.span(),
                    name,
                    ty,
                    init: init_expr,
                })));
            }

            syn::Stmt::Expr(expr, semi) => {
                let converted = convert_expr(expr.clone())?;
                if is_last && semi.is_none() {
                    // Final expression without semicolon - this is the block's value
                    final_expr = Some(Box::new(converted));
                } else {
                    stmts.push(Stmt::Expr(converted));
                }
            }

            syn::Stmt::Item(item) => {
                return Err(syn::Error::new_spanned(
                    item,
                    "item definitions are not allowed inside a kernel body\n\
                     \n\
                     note: a body holds let bindings and expressions\n\
                     \n\
                     help: declare `const`s and helper `fn`s at the top of the kernel! block:\n\
                     help:   kernel! {\n\
                     help:       fn helper(x: f32) -> f32 { x * 2.0 }\n\
                     help:       pub fn entry() -> f32 { helper(X) }\n\
                     help:   }",
                ));
            }

            syn::Stmt::Macro(mac) => {
                return Err(syn::Error::new_spanned(
                    mac,
                    "macro invocations are not allowed inside kernel! blocks\n\
                     \n\
                     note: kernel! needs to analyze the expression at compile time\n\
                     \n\
                     help: expand the macro outside the kernel! or use equivalent expressions:\n\
                     help:   let value = some_macro!();\n\
                     help:   let my_kernel = kernel!(|| value * X);",
                ));
            }
        }
    }

    Ok(BlockExpr {
        stmts,
        expr: final_expr,
        span,
    })
}

/// A `let` whose pattern is not a name — `let (x, y) = …`, `let Row { x0, .. }
/// = …` — takes a value apart; a record's fields are read by name instead.
fn refuse_let_pattern(pattern: &Pat) -> syn::Error {
    syn::Error::new_spanned(
        pattern,
        format!(
            "a pattern in a kernel `let`\n\
             \n\
             note: a kernel `let` binds a plain name: an `f32`, a `bool`, or a record, which \
             it aliases (`let q = p;`) and whose fields are read by name (`q.x0`)\n\
             note: destructuring is flattened in the front end as a record is (D7's front \
             half), which is B3 of {PLAN}"
        ),
    )
}

/// The one type suffix a numeric literal in a kernel body may carry: every
/// value there is an `f32`.
const F32_SUFFIX: &str = "f32";

/// The number a literal denotes, or a spanned refusal saying why it names
/// none.
fn literal_value(lit: &syn::Lit) -> syn::Result<Literal> {
    match lit {
        syn::Lit::Float(float) => float_value(float).map(Literal::F32),
        syn::Lit::Int(int) => int_value(int),
        other => Err(syn::Error::new_spanned(
            other,
            "only numeric literals are allowed in a kernel body\n\
             \n\
             note: every value in a kernel body is an `f32`; a `bool` comes from a comparison",
        )),
    }
}

/// A float literal rounds once, to the nearest `f32` — the value rustc gives
/// the same literal.
///
/// It used to be parsed as `f64` and then cast, which rounds twice, and the
/// two disagree wherever the `f64` lands exactly on the midpoint between two
/// `f32`s and the cast breaks the tie to even: `1.00000005960464477539062500001`
/// is `1 + 2⁻²³` rounded once and `1.0` rounded twice.
fn float_value(float: &syn::LitFloat) -> syn::Result<f32> {
    refuse_a_foreign_suffix(float.suffix(), float)?;
    rounded_once(float.base10_digits(), float)
}

/// Digits rounded once to the nearest `f32` (`str::parse` is correctly
/// rounded). An infinity is refused, as rustc refuses it
/// (`overflowing_literals` is deny-by-default): it is not what anyone wrote.
fn rounded_once(digits: &str, literal: &impl quote::ToTokens) -> syn::Result<f32> {
    let value: f32 = digits
        .parse()
        .map_err(|err| syn::Error::new_spanned(literal, err))?;
    if value.is_infinite() {
        return Err(syn::Error::new_spanned(
            literal,
            format!(
                "literal out of range for `f32`\n\
                 \n\
                 note: `{digits}` rounds to infinity; the largest `f32` is {max:e}",
                max = f32::MAX
            ),
        ));
    }
    Ok(value)
}

/// An integer literal denotes an exact integer, and it is kept as one: which
/// type it is — an `f32` where a value is expected, a `usize` in a range's
/// bounds or a `usize` const — depends on where it stands, and
/// [`LiteralExpr::f32_value`] and [`LiteralExpr::usize_value`] answer for
/// each. Past `u128` it is refused, as rustc refuses it: no type in the
/// language holds it.
///
/// `16777217f32` is a float literal to rustc (the suffix makes it one), and
/// it rounds once like `16777217.0`, so that is what it does here too.
fn int_value(int: &syn::LitInt) -> syn::Result<Literal> {
    refuse_a_foreign_suffix(int.suffix(), int)?;
    if int.suffix() == F32_SUFFIX {
        return rounded_once(int.base10_digits(), int).map(Literal::F32);
    }
    int.base10_parse::<u128>().map(Literal::Int).map_err(|_| {
        syn::Error::new_spanned(
            int,
            format!(
                "integer literal is too large\n\
                 \n\
                 note: `{}` exceeds the largest integer a literal may be, {}",
                int.base10_digits(),
                u128::MAX
            ),
        )
    })
}

/// The plain name a `let` or a parameter binds. `mut`, `ref` and an `@`
/// subpattern are refused rather than dropped: nothing in a kernel body
/// assigns, borrows or matches, so each would be a promise the body cannot
/// keep, and a subpattern can refute a `let`, which rustc refuses too.
fn plain_name(pat_ident: &syn::PatIdent, pat: &Pat) -> syn::Result<syn::Ident> {
    let decorated =
        pat_ident.by_ref.is_some() || pat_ident.mutability.is_some() || pat_ident.subpat.is_some();
    if decorated {
        return Err(syn::Error::new_spanned(
            pat,
            "a `let` or a parameter in a kernel binds a plain name\n\
             \n\
             note: `mut`, `ref` and `@` are refused rather than ignored: nothing in a \
             kernel body assigns, borrows or matches",
        ));
    }
    Ok(pat_ident.ident.clone())
}

/// A suffix naming a type other than `f32` is refused rather than dropped:
/// `2.5f64` or `3u8` asks for a type a kernel body does not have. A count is
/// written unsuffixed: an integer takes its type from where it stands.
fn refuse_a_foreign_suffix(suffix: &str, literal: &impl quote::ToTokens) -> syn::Result<()> {
    match suffix {
        "" | F32_SUFFIX => Ok(()),
        other => Err(syn::Error::new_spanned(
            literal,
            format!(
                "a `{other}` literal in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32`; an unsuffixed integer in a \
                 range's bounds or a `usize` const is a count\n\
                 \n\
                 help: remove the suffix, or write `{F32_SUFFIX}`"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Role;
    use quote::quote;

    /// The closure sugar's one entry.
    fn entry(def: &KernelDef) -> &FnItem {
        assert_eq!(def.spelling, Spelling::Closure);
        assert_eq!(def.fns.len(), 1);
        &def.fns[0]
    }

    #[test]
    fn parse_simple_kernel() {
        let input = quote! { |r: f32| X * X + Y * Y - r };
        let kernel = parse(input).unwrap();
        let entry = entry(&kernel);
        assert_eq!(entry.params.len(), 1);
        assert_eq!(entry.params[0].name.to_string(), "r");
        assert_eq!(entry.role(), Role::Entry);
        assert!(entry.ret.is_none(), "the sugar's type is inferred");
    }

    #[test]
    fn parse_empty_params() {
        let input = quote! { || X * X + Y * Y };
        let kernel = parse(input).unwrap();
        assert_eq!(entry(&kernel).params.len(), 0);
    }

    #[test]
    fn parse_multiple_params() {
        let input = quote! { |cx: f32, cy: f32, r: f32| X - cx };
        let kernel = parse(input).unwrap();
        assert_eq!(entry(&kernel).params.len(), 3);
    }

    #[test]
    fn parse_method_call() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let kernel = parse(input).unwrap();
        // Should successfully parse the .sqrt() method call
        match &entry(&kernel).body {
            Expr::Binary(_) => {} // Expected: sqrt() - r
            _ => panic!("expected binary expression"),
        }
    }

    #[test]
    fn parse_block_expr() {
        let input = quote! {
            |cx: f32, cy: f32| {
                let dx = X - cx;
                let dy = Y - cy;
                dx * dx + dy * dy
            }
        };
        let kernel = parse(input).unwrap();
        match &entry(&kernel).body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 2); // two let statements
                assert!(block.expr.is_some()); // final expression
            }
            _ => panic!("expected block expression"),
        }
    }

    /// A parameter's declared type is kept for `sema` to check; the parser
    /// does not judge it.
    #[test]
    fn parse_scalar_params_keep_their_declared_types() {
        let input = quote! { |cx: f32, m: bool| X * cx };
        let kernel = parse(input).unwrap();
        let entry = entry(&kernel);
        assert_eq!(entry.params.len(), 2);
        assert_eq!(entry.params[0].name.to_string(), "cx");
        assert_eq!(entry.params[1].name.to_string(), "m");
        for (param, want) in entry.params.iter().zip(["f32", "bool"]) {
            let syn::Type::Path(path) = &*param.ty else {
                panic!("expected a path type for {}", param.name);
            };
            assert_eq!(path.path.segments[0].ident.to_string(), want);
        }
    }

    /// A multi-segment path (qself-free but `len() != 1`) is refused as the
    /// path it is, not silently truncated to an `Ident` of its first segment;
    /// so is a call through one.
    #[test]
    fn a_path_from_outside_the_block_is_refused_by_name() {
        let err = refusal(quote! { || std::f32::consts::PI });
        assert!(
            err.contains("a path in a kernel body: `std :: f32 :: consts :: PI`")
                && err.contains("not captured"),
            "got: {err}"
        );
        let err = refusal(quote! { || f32::sqrt(X) });
        assert!(err.contains("a qualified call"), "got: {err}");
    }

    /// A value with parts built or taken apart in a body is refused where it
    /// is written: a `let` pattern, a tuple and a tuple's field naming B3,
    /// where tuples are flattened in the front end as records are (D7's
    /// front half, which §1.7's glyph needs for `let (x, y) = …`); a record
    /// literal naming Phase D (D7).
    #[test]
    fn a_pattern_a_tuple_and_a_record_literal_are_refused_naming_their_phase() {
        let cases: [(TokenStream, &str, &str); 6] = [
            (quote! { || { let (a, b) = (X, Y); a } }, "a pattern", "B3"),
            (
                quote! { || { let (a, b): (f32, f32) = (X, Y); a } },
                "a pattern",
                "B3",
            ),
            (
                quote! { || { let Row { x0, .. } = p; x0 } },
                "a pattern",
                "B3",
            ),
            (quote! { || (X, Y) }, "a tuple in a kernel body", "B3"),
            (quote! { |p: f32| p.0 }, "a tuple's field", "B3"),
            (quote! { || Row { x0: X }.x0 }, "a record literal", "D7"),
        ];
        for (input, expected, phase) in cases {
            let err = refusal(input);
            assert!(
                err.contains(expected) && err.contains(phase),
                "expected `{expected}` and `{phase}`, got: {err}"
            );
        }
    }

    /// `p.x0` parses as a field of whatever `p` is; which record, and
    /// whether it has the field, are `sema`'s questions.
    #[test]
    fn a_named_field_parses_as_a_field_read() {
        let def = parse(quote! { |p: f32| (p).x0 + 1.0 }).expect("parses");
        let Expr::Binary(sum) = &entry(&def).body else {
            panic!("expected the sum, got {:?}", entry(&def).body);
        };
        let Expr::Field(field) = &*sum.lhs else {
            panic!("expected a field, got {:?}", sum.lhs);
        };
        assert_eq!(field.member.to_string(), "x0");
        assert_eq!(
            field.base.named().map(ToString::to_string),
            Some("p".into())
        );
    }

    /// A range anywhere but a fold's or an integral's names the spellings of
    /// both.
    #[test]
    fn a_range_is_refused_naming_the_fold_and_the_integral() {
        for input in [
            quote! { || X + (0.0..1.0) },
            quote! { || DX(0.0..1.0) },
            quote! { || { let r = 0..4; X } },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("a range in a kernel body")
                    && err.contains("(a..b).map(|i| e).sum()")
                    && err.contains("integral(lo..hi, |u| e)"),
                "got: {err}"
            );
        }
    }

    /// Every other Rust expression is refused at the token, named, with
    /// nothing passed through.
    #[test]
    fn any_other_expression_is_refused_at_parse() {
        for other in [
            quote! { || unsafe { X } },
            quote! { || match X { _ => Y } },
            quote! { || &X },
            quote! { || [X, Y] },
            quote! { || X? },
        ] {
            let err = refusal(other);
            assert!(
                err.contains("unsupported expression in a kernel body: `"),
                "got: {err}"
            );
        }
    }

    #[test]
    fn parse_let_with_type_annotation_extracts_both_name_and_type() {
        let input = quote! {
            || {
                let dx: f32 = X;
                dx
            }
        };
        let kernel = parse(input).unwrap();
        match &entry(&kernel).body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 1);
                match &block.stmts[0] {
                    Stmt::Let(let_stmt) => {
                        assert_eq!(let_stmt.name.to_string(), "dx");
                        let ty = let_stmt.ty.as_ref().expect("expected a type annotation");
                        if let Type::Path(type_path) = ty {
                            assert_eq!(type_path.path.segments[0].ident.to_string(), "f32");
                        } else {
                            panic!("expected path type");
                        }
                    }
                    _ => panic!("expected let statement"),
                }
            }
            _ => panic!("expected block expression"),
        }
    }

    #[test]
    fn a_terminal_statement_with_a_trailing_semicolon_is_not_the_blocks_final_expression() {
        let input = quote! { || { X; } };
        let kernel = parse(input).unwrap();
        match &entry(&kernel).body {
            Expr::Block(block) => {
                assert_eq!(
                    block.stmts.len(),
                    1,
                    "the semicolon-terminated X is a statement"
                );
                assert!(
                    block.expr.is_none(),
                    "a block ending in `expr;` has no final expression"
                );
            }
            _ => panic!("expected block expression"),
        }
    }

    /// What the literal body `|| <lit>` denotes as parsed, or the parser's
    /// refusal.
    fn parsed_literal(lit: TokenStream) -> Result<LiteralExpr, String> {
        let def = parse(quote! { || #lit }).map_err(|e| e.to_string())?;
        match &entry(&def).body {
            Expr::Literal(literal) => Ok(literal.clone()),
            other => panic!("expected a literal, got {other:?}"),
        }
    }

    /// The value the literal body `|| <lit>` denotes where a value is
    /// expected, or the refusal: the parser's, or the value position's.
    fn literal(lit: TokenStream) -> Result<f32, String> {
        parsed_literal(lit)?.f32_value().map_err(|e| e.to_string())
    }

    /// The parser's refusal of `input`, as text.
    fn refusal(input: TokenStream) -> String {
        parse(input)
            .expect_err("the input is refused at parse")
            .to_string()
    }

    /// `let … else` is refused, not parsed with its `else` branch dropped.
    #[test]
    fn let_else_is_refused_rather_than_dropped() {
        let input = quote! {
            || {
                let a = X else { return; };
                a
            }
        };
        let err = refusal(input);
        assert!(err.contains("`let ... else`"), "got: {err}");
    }

    /// A float literal rounds once, straight to `f32`, as rustc rounds it.
    /// This literal is just above the midpoint between `1.0` and `1 + 2⁻²³`,
    /// by far less than an `f64` ulp: through `f64` it lands exactly on the
    /// midpoint and the cast ties to even, `1.0`.
    #[test]
    #[allow(clippy::excessive_precision)] // The digits past f32's precision are the witness.
    fn a_float_literal_rounds_once_to_f32() {
        let once = 1.00000005960464477539062500001_f32;
        let twice = 1.00000005960464477539062500001_f64 as f32;
        assert_eq!(once.to_bits(), 0x3f80_0001);
        assert_eq!(twice.to_bits(), 0x3f80_0000);

        let got = literal(quote!(1.00000005960464477539062500001)).expect("in range");
        assert_eq!(got.to_bits(), once.to_bits());
        assert_eq!(literal(quote!(0.1)), Ok(0.1_f32));
        assert_eq!(literal(quote!(2.5f32)), Ok(2.5));
        assert_eq!(literal(quote!(1_000.25)), Ok(1000.25));
        // The suffix makes an integer a float literal to rustc: rounded once.
        assert_eq!(literal(quote!(16777217f32)), Ok(16_777_216.0));
    }

    /// A `let` binds a plain name; `mut`, `ref` and `@` are refused, not
    /// silently dropped.
    #[test]
    fn a_let_binds_a_plain_name() {
        for decorated in [
            quote! { || { let mut a = X; a } },
            quote! { || { let ref a = X; a } },
            quote! { || { let a @ 1.0..=2.0 = X; a } },
        ] {
            let err = refusal(decorated);
            assert!(err.contains("plain name"), "got: {err}");
        }
    }

    #[test]
    fn a_float_literal_past_f32s_range_is_refused() {
        let err = literal(quote!(1e39)).expect_err("rounds to infinity");
        assert!(err.contains("out of range for `f32`"), "got: {err}");
        // The largest finite `f32` is in range.
        assert_eq!(literal(quote!(3.4028235e38)), Ok(f32::MAX));
    }

    /// An integer literal is exact or refused where a value is expected.
    /// Every integer up to 2²⁴ is an `f32`; 2²⁴ + 1 is the first that is not.
    #[test]
    fn an_integer_literal_is_exact_or_refused() {
        assert_eq!(literal(quote!(0)), Ok(0.0));
        assert_eq!(literal(quote!(16777216)), Ok(16_777_216.0));
        // One significant bit, however large: exactly an `f32`.
        assert_eq!(literal(quote!(1099511627776)), Ok(1_099_511_627_776.0));
        assert_eq!(literal(quote!(0x10)), Ok(16.0));

        for inexact in [quote!(16777217), quote!(4294967295)] {
            let err = literal(inexact).expect_err("an f32 cannot hold it");
            assert!(
                err.contains("not exactly representable as an `f32`"),
                "got: {err}"
            );
        }
        // Past `u128` no type holds it, and the parser says so.
        let err =
            literal(quote!(1000000000000000000000000000000000000000000)).expect_err("past u128");
        assert!(err.contains("integer literal is too large"), "got: {err}");
    }

    /// An integer is kept exactly as written, since its type is its
    /// position's: `16777217` is no `f32`, and is a `usize`. A float is never
    /// a count.
    #[test]
    fn an_integer_literal_is_kept_exact_for_its_position() {
        let int = parsed_literal(quote!(16777217)).expect("parses");
        assert_eq!(int.value, Literal::Int(16_777_217));
        assert_eq!(int.usize_value().expect("a count"), 16_777_217);
        assert!(int.f32_value().is_err());

        let past_usize = parsed_literal(quote!(18446744073709551616)).expect("parses");
        let err = past_usize.usize_value().expect_err("2^64").to_string();
        assert!(err.contains("out of range for `usize`"), "got: {err}");

        let float = parsed_literal(quote!(4.0)).expect("parses");
        let err = float.usize_value().expect_err("a float").to_string();
        assert!(err.contains("expected `usize`"), "got: {err}");
    }

    /// A suffix naming another type is refused rather than silently dropped.
    #[test]
    fn a_literal_suffixed_with_a_type_other_than_f32_is_refused() {
        for foreign in [quote!(2.5f64), quote!(3u8), quote!(3i32)] {
            let err = literal(foreign).expect_err("a kernel value is an f32");
            assert!(err.contains("literal in a kernel body"), "got: {err}");
        }
    }

    #[test]
    fn a_literal_that_is_not_a_number_is_refused() {
        for other in [quote!(true), quote!("text"), quote!('c')] {
            let err = literal(other).expect_err("not a number");
            assert!(err.contains("only numeric literals"), "got: {err}");
        }
    }

    #[test]
    fn parse_block_body_with_a_scalar_param() {
        let input = quote! {
            |x: f32| {
                let a = X + x;
                a
            }
        };
        let kernel = parse(input).unwrap();
        assert_eq!(entry(&kernel).params.len(), 1);

        match &entry(&kernel).body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 1, "expected 1 let statement");
                assert!(block.expr.is_some(), "expected final expression");
            }
            other => panic!("expected block expression, got {other:?}"),
        }
    }

    // ───────────────────────── the items form ─────────────────────────

    /// A block of items: `const`s, helpers and entries, told apart by `pub`.
    #[test]
    fn an_items_block_parses_consts_helpers_and_entries() {
        let input = quote! {
            const R: f32 = 1.0;
            pub const TWO_R: f32 = R * 2.0;
            /// A helper: private.
            fn sq(x: f32) -> f32 { x * x }
            pub fn circle(cx: f32) -> f32 { sq(X - cx) - R }
        };
        let def = parse(input).unwrap();
        assert_eq!(def.spelling, Spelling::Items);
        assert_eq!(def.consts.len(), 2);
        assert!(matches!(def.consts[0].vis, syn::Visibility::Inherited));
        assert!(matches!(def.consts[1].vis, syn::Visibility::Public(_)));
        assert_eq!(def.fns.len(), 2);
        assert_eq!(def.fns[0].role(), Role::Helper);
        assert_eq!(def.fns[0].attrs.len(), 1, "the doc comment is kept");
        assert_eq!(def.fns[1].role(), Role::Entry);
        assert_eq!(def.fns[1].params.len(), 1);
        assert!(def.fns[1].ret.is_some(), "the return type is declared");
    }

    /// `if c { a } else { b }` and an `else if` chain parse as the choice.
    #[test]
    fn an_if_with_its_else_parses_as_the_choice() {
        let input = quote! { || if X < Y { X } else if X < 2.0 { 2.0 } else { Y } };
        let def = parse(input).unwrap();
        let Expr::If(outer) = &entry(&def).body else {
            panic!("expected an if, got {:?}", entry(&def).body);
        };
        assert!(matches!(*outer.cond, Expr::Binary(_)));
        assert!(
            matches!(*outer.else_branch, Expr::If(_)),
            "`else if` is an `if` in the else position"
        );
    }

    /// An `if` without an `else` has no value to be, so it is refused where
    /// it stands.
    #[test]
    fn an_if_without_an_else_is_refused() {
        let err = refusal(quote! { || if X < Y { X } });
        assert!(err.contains("without an `else`"), "got: {err}");
        let err = refusal(quote! { || if let a = X { a } else { Y } });
        assert!(err.contains("`if let`"), "got: {err}");
    }

    /// The constructs a DAG has no meaning for are refused with a span, not
    /// passed through for a later stage to refuse without one.
    #[test]
    fn loops_assignment_return_and_closures_are_refused() {
        let cases: [(TokenStream, &str); 7] = [
            (quote! { || loop { X } }, "a loop"),
            (quote! { || { while X < Y { X }; X } }, "a loop"),
            (quote! { || { for i in 0..3 { X }; X } }, "a loop"),
            (quote! { || { let a = X; a = Y; a } }, "assignment"),
            (quote! { || { let a = X; a += Y; a } }, "assigns"),
            (quote! { || { return X; } }, "`return`"),
            (quote! { || DX(|x: f32| x) }, "a closure"),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// `%` and `!` have no IR op; each is refused by name at the operator.
    #[test]
    fn rem_and_not_are_refused_by_name() {
        let err = refusal(quote! { || X % Y });
        assert!(err.contains("`%` has no IR op"), "got: {err}");
        let err = refusal(quote! { || !(X < Y) });
        assert!(err.contains("`!` has no IR op"), "got: {err}");
    }

    /// A record parses with its visibility, its doc comments and its fields
    /// in declaration order, each field's own visibility and docs kept.
    #[test]
    fn a_record_parses_with_its_fields_and_docs() {
        let def = parse(quote! {
            /// One arc piece.
            pub struct Row {
                /// Where it starts.
                pub x0: f32,
                sigma: f32,
            }
            pub fn f(p: Row) -> f32 { p.x0 }
        })
        .expect("parses");
        let [row] = def.records.as_slice() else {
            panic!("one record, got {:?}", def.records);
        };
        assert_eq!(row.name.to_string(), "Row");
        assert!(matches!(row.vis, syn::Visibility::Public(_)));
        assert_eq!(row.attrs.len(), 1, "the record's doc comment is kept");
        let names: Vec<String> = row.fields.iter().map(|f| f.name.to_string()).collect();
        assert_eq!(names, ["x0", "sigma"]);
        assert_eq!(
            row.fields[0].attrs.len(),
            1,
            "a field's doc comment is kept"
        );
        assert!(matches!(row.fields[1].vis, syn::Visibility::Inherited));
    }

    /// A record is named `f32` fields: a tuple struct, a unit struct and a
    /// generic record are refused where they are written, naming §1.3; any
    /// other item names what the block holds.
    #[test]
    fn a_record_that_is_not_named_fields_is_refused_naming_its_section() {
        let cases: [(TokenStream, &str); 3] = [
            (quote! { struct Row(f32, f32); }, "a tuple struct"),
            (quote! { struct Row; }, "a unit struct"),
            (quote! { struct Row<T> { x: T } }, "a generic record"),
        ];
        for (input, expected) in cases {
            let err = refusal(quote! { #input pub fn f() -> f32 { X } });
            assert!(
                err.contains(expected) && err.contains("§1.3"),
                "expected `{expected}` and `§1.3`, got: {err}"
            );
        }
        let err = refusal(quote! {
            enum Row { A }
            pub fn f() -> f32 { X }
        });
        assert!(
            err.contains("a `enum`") && err.contains("records (`struct`s of `f32` fields)"),
            "got: {err}"
        );
        for attr in [quote!(#[repr(packed)]), quote!(#[cfg(any())])] {
            let err = refusal(quote! {
                #attr
                struct Row { x: f32 }
                pub fn f() -> f32 { X }
            });
            assert!(
                err.contains("a `repr` or a `cfg` on a record"),
                "got: {err}"
            );
            let err = refusal(quote! {
                struct Row { #attr x: f32 }
                pub fn f() -> f32 { X }
            });
            assert!(
                err.contains("a `repr` or a `cfg` on a record"),
                "got: {err}"
            );
        }
    }

    /// A record's other attributes, and its fields', are kept for its host
    /// twin: rustc reads them there.
    #[test]
    fn a_records_attributes_are_kept() {
        let def = parse(quote! {
            /// A row.
            #[allow(dead_code)]
            #[cfg_attr(any(), derive(Eq))]
            pub struct Row { #[allow(unused)] pub x0: f32 }
            pub fn f() -> f32 { X }
        })
        .expect("parses");
        let [row] = def.records.as_slice() else {
            panic!("one record, got {}", def.records.len());
        };
        let paths: Vec<String> = row
            .attrs
            .iter()
            .map(|attr| attr.path().get_ident().expect("a plain path").to_string())
            .collect();
        assert_eq!(paths, ["doc", "allow", "cfg_attr"]);
        assert_eq!(row.fields[0].attrs.len(), 1, "the field's `allow`");
    }

    /// An entry's generics are its structural parameters, in declaration
    /// order; a helper has none.
    #[test]
    fn an_entrys_const_generics_are_its_structural_parameters() {
        let def = parse(quote! {
            fn h(x: f32) -> f32 { x }
            pub fn f<const N: usize, const M: usize>(r: f32) -> f32 { h(r) }
        })
        .expect("parses");
        let names: Vec<String> = def.fns[1]
            .structural
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(names, ["N", "M"]);
        assert!(def.fns[0].structural.is_empty());
    }

    // ───────────────────────────── folds ─────────────────────────────

    /// The fold the body `|| <expr>` parses to.
    fn fold(input: TokenStream) -> FoldExpr {
        let def = parse(quote! { || #input }).expect("the fold parses");
        match &entry(&def).body {
            Expr::Fold(fold) => fold.clone(),
            other => panic!("expected a fold, got {other:?}"),
        }
    }

    /// Each spelling of a fold names its monoid; the closure's parameter is
    /// the binder and its body the fold's body.
    #[test]
    fn every_fold_spelling_parses_to_its_reduction() {
        let cases: [(TokenStream, Reduction); 8] = [
            (quote! { (0..4).map(|i| X).sum() }, Reduction::Sum),
            (quote! { (0..4).map(|i| X).sum::<f32>() }, Reduction::Sum),
            (quote! { (0..4).map(|i| X).product() }, Reduction::Product),
            (
                quote! { (0..4).map(|i| X).product::<f32>() },
                Reduction::Product,
            ),
            (
                quote! { (0..4).map(|i| X).fold(f32::INFINITY, f32::min) },
                Reduction::Min,
            ),
            (
                quote! { (0..4).map(|i| X).fold(f32::NEG_INFINITY, f32::max) },
                Reduction::Max,
            ),
            (quote! { (0..4).any(|i| X < Y) }, Reduction::Any),
            (quote! { (0..4).all(|i| X < Y) }, Reduction::All),
        ];
        for (input, want) in cases {
            let parsed = fold(input);
            assert_eq!(parsed.reduction, want);
            assert_eq!(parsed.binder.to_string(), "i");
        }
        let parsed = fold(quote! { (LO..LO + N).map(|k| X).sum() });
        assert!(matches!(*parsed.range.lo, Expr::Ident(_)));
        assert!(matches!(*parsed.range.hi, Expr::Binary(_)));
    }

    /// A fold spelled any other way is refused at parse, naming what the
    /// language accepts: a `.fold` that is not a min or a max, a method that
    /// ends no fold, a range that is inclusive or open, and a closure that is
    /// not `|i| body`.
    #[test]
    fn a_fold_spelled_any_other_way_is_refused() {
        let cases: [(TokenStream, &str); 19] = [
            (quote! { || (0..4).map(|i| X) }, "an iterator, not a value"),
            (quote! { || (0..4).map(|i| X).min() }, "does not end a fold"),
            (
                quote! { || (0..4).map(|i| X).fold(0.0, |a, b| a + b) },
                "names no monoid",
            ),
            (
                quote! { || (0..4).map(|i| X).fold(f32::INFINITY, f32::max) },
                "names no monoid",
            ),
            (
                quote! { || (0..4).map(|i| X).fold(std::f32::INFINITY, f32::min) },
                "names no monoid",
            ),
            (
                quote! { || (0..4).map(|i| X).fold(f32::INFINITY) },
                "names no monoid",
            ),
            (quote! { || (0..4).sum() }, "`.sum` on a range"),
            (
                quote! { || (0..4).rev().map(|i| X).sum() },
                "`.rev` on a range",
            ),
            (quote! { || (0..=4).map(|i| X).sum() }, "an inclusive range"),
            (quote! { || (0..).map(|i| X).sum() }, "both of its bounds"),
            (
                quote! { || (0..4).map(|i: usize| X).sum() },
                "`usize`, always",
            ),
            (quote! { || (0..4).map(|i, j| X).sum() }, "one parameter"),
            (quote! { || (0..4).map(|_| X).sum() }, "a plain name"),
            (quote! { || (0..4).map(move |i| X).sum() }, "unqualified"),
            (
                quote! { || (0..4).map(|i| -> f32 { X }).sum() },
                "declares no return type",
            ),
            (
                quote! { || (0..4).map(|i| X).sum::<f64>() },
                "over anything but `f32`s",
            ),
            (
                quote! { || (0..4).map(|i| X).sum(1.0) },
                "takes no arguments",
            ),
            (quote! { || (0..4).map(X).sum() }, "the closure `|i| body`"),
            (
                quote! { || (0..4).any::<f32>(|i| X < Y) },
                "takes no type arguments",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
        // The refusal of a `.fold` names the two it accepts.
        let err = refusal(quote! { || (0..4).map(|i| X).fold(0.0, f32::max) });
        assert!(
            err.contains("f32::INFINITY, f32::min") && err.contains("f32::NEG_INFINITY, f32::max"),
            "got: {err}"
        );
    }

    /// A closure is the body of a fold or an integral and nothing else:
    /// anywhere else it is refused, naming the spellings of both, and the
    /// phase that brings a function as an argument.
    #[test]
    fn a_closure_outside_a_fold_or_an_integral_is_refused() {
        for input in [
            quote! { || X.max(|i| i) },
            quote! { || { let f = |x: f32| x; X } },
            quote! { || (0..4).map(|i| X).sum() + (|j| Y) },
            quote! { || integral(0.0..1.0, |u| u) * (|w| w) },
            quote! { || area(|u, v| X.max(|w| w)) },
            quote! { || DX(|u| u) },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("a closure in a kernel body")
                    && err.contains("the body of a fold or an integral")
                    && err.contains("(a..b).map(|i| e).sum()")
                    && err.contains("integral(lo..hi, |u| e)")
                    && err.contains("area(|u, v| e)")
                    && err.contains("Phase D"),
                "got: {err}"
            );
        }
    }

    /// A fold's spelling over something that is not a range is iterating a
    /// family of records, and is refused naming B3, where the closure's own
    /// refusal would name the fold and the integral.
    #[test]
    fn iterating_anything_but_a_range_is_refused_naming_b3() {
        for input in [
            quote! { |pieces: f32| pieces.map(|p| p * X).sum() },
            quote! { |pieces: f32| pieces.any(|p| p < X) },
            quote! { || X.all(|p| p < Y) },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("which is not a range")
                    && err.contains("a family of records")
                    && err.contains("B3"),
                "got: {err}"
            );
        }
    }

    /// Type arguments name a type a method would be generic over, and no
    /// method in a kernel body is generic; they are refused, not dropped.
    #[test]
    fn type_arguments_on_a_method_are_refused() {
        let err = refusal(quote! { || X.sqrt::<f32>() });
        assert!(err.contains("type arguments on a method"), "got: {err}");
    }

    /// `i as f32` parses as a cast of its operand; which operands a cast
    /// means anything of is `sema`'s question, since it needs their types.
    #[test]
    fn a_cast_parses_with_its_operand() {
        let def = parse(quote! { || (i) as f32 }).expect("parses");
        let Expr::Cast(cast) = &entry(&def).body else {
            panic!("expected a cast, got {:?}", entry(&def).body);
        };
        assert_eq!(cast.named().map(ToString::to_string), Some("i".to_string()));
        let def = parse(quote! { || (X + 1.0) as f32 }).expect("parses");
        let Expr::Cast(cast) = &entry(&def).body else {
            panic!("expected a cast");
        };
        assert!(cast.named().is_none());
    }

    /// The one conversion is to `f32`; a cast to any other type is refused
    /// at the type, so the AST never holds one.
    #[test]
    fn a_cast_to_anything_but_f32_is_refused() {
        for input in [
            quote! { || (0..4).map(|i| i as u32).sum() },
            quote! { || X as f64 },
            quote! { || X as usize },
            quote! { || X as std::primitive::f32 },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("`as` converts a `usize` to an `f32`, and to nothing else"),
                "got: {err}"
            );
        }
    }

    /// A `fn` signature carries nothing the language cannot honor: an
    /// entry's generics are `const N: usize` and nothing else, and a
    /// helper's are B3.
    #[test]
    fn a_fn_signature_is_plain() {
        let cases: [(TokenStream, &str); 10] = [
            (
                quote! { fn h<const N: usize>(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
                "generics on a helper",
            ),
            (
                quote! { fn h<const N: usize>(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
                "B3",
            ),
            (quote! { pub fn f<T>() -> f32 { X } }, "B3"),
            (quote! { pub fn f<'a>() -> f32 { X } }, "lifetime parameter"),
            (
                quote! { pub fn f<const N: u32>() -> f32 { X } },
                "a structural parameter is a `usize`",
            ),
            (
                quote! { pub fn f<const N: usize = 4>() -> f32 { X } },
                "a default for a structural parameter",
            ),
            (quote! { pub fn f() { X } }, "declares no return type"),
            (quote! { pub fn f(mut x: f32) -> f32 { x } }, "plain name"),
            (
                quote! { pub fn f(g: impl Fn(f32) -> f32) -> f32 { X } },
                "Phase D",
            ),
            (quote! { const fn f() -> f32 { X } }, "`const fn`"),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    // ─────────────────────────── integrals ───────────────────────────

    /// The integral the body `|| <expr>` parses to.
    fn integral(input: TokenStream) -> IntegralExpr {
        let def = parse(quote! { || #input }).expect("the integral parses");
        match &entry(&def).body {
            Expr::Integral(integral) => integral.clone(),
            other => panic!("expected an integral, got {other:?}"),
        }
    }

    /// `integral(lo..hi, |u| e)` is one integral over the bounds written,
    /// its closure's parameter the variable and its body the integrand; the
    /// interval may be parenthesized, as a fold's range may.
    #[test]
    fn an_integral_parses_with_its_interval_variable_and_body() {
        for input in [
            quote! { integral(0.0..1.0, |u| u * X) },
            quote! { integral((0.0..1.0), |u| u * X) },
        ] {
            let parsed = integral(input);
            assert_eq!(parsed.variable.to_string(), "u");
            let IntegralBounds::Written(range) = &parsed.bounds else {
                panic!("the interval is the one written");
            };
            assert!(matches!(*range.lo, Expr::Literal(_)));
            assert!(matches!(*parsed.body, Expr::Binary(_)));
        }
        let parsed = integral(quote! { integral(-H..H * 2.0, |w| w) });
        let IntegralBounds::Written(range) = &parsed.bounds else {
            panic!("the interval is the one written");
        };
        assert!(matches!(*range.lo, Expr::Unary(_)));
        assert!(matches!(*range.hi, Expr::Binary(_)));
    }

    /// `area(|u, v| e)` is exactly `integral(-H..H, |v| integral(-H..H,
    /// |u| e))` over the pixel: two integrals, the `v` one outermost, as
    /// `Kernel::area` builds them.
    #[test]
    fn area_is_two_integrals_over_the_pixel_v_outermost() {
        let outer = integral(quote! { area(|u, v| X + u) });
        assert!(matches!(outer.bounds, IntegralBounds::Pixel));
        assert_eq!(outer.variable.to_string(), "v");
        let Expr::Integral(inner) = &*outer.body else {
            panic!("the outer integral's body is the inner integral");
        };
        assert!(matches!(inner.bounds, IntegralBounds::Pixel));
        assert_eq!(inner.variable.to_string(), "u");
        assert!(matches!(*inner.body, Expr::Binary(_)));
    }

    /// An integral or an `area` spelled any other way is refused where it
    /// is written, naming its one spelling: `area` with one parameter
    /// among them.
    #[test]
    fn an_integral_spelled_any_other_way_is_refused() {
        let cases: [(TokenStream, &str); 15] = [
            (quote! { || integral(0.0..1.0) }, "takes two arguments"),
            (
                quote! { || integral(0.0..1.0, |u| u, 2.0) },
                "takes two arguments",
            ),
            (
                quote! { || integral(X, |u| u) },
                "first argument is its interval",
            ),
            (
                quote! { || integral(0.0..1.0, X) },
                "second argument is the closure `|u| body`",
            ),
            (
                quote! { || integral(0.0..=1.0, |u| u) },
                "an inclusive range",
            ),
            (
                quote! { || integral(0.0.., |u| u) },
                "names both of its bounds",
            ),
            (
                quote! { || integral(0.0..1.0, |u, v| u) },
                "`integral`'s closure takes one parameter",
            ),
            (
                quote! { || integral(0.0..1.0, |u: f32| u) },
                "an `f32`, always",
            ),
            (quote! { || integral(0.0..1.0, |_| X) }, "a plain name"),
            (quote! { || integral(0.0..1.0, move |u| u) }, "unqualified"),
            (
                quote! { || integral(0.0..1.0, |u| -> f32 { u }) },
                "declares no return type",
            ),
            (
                quote! { || area(|u| X + u) },
                "`area`'s closure takes two parameters, the offsets across the pixel",
            ),
            (
                quote! { || area(|u, v, w| X) },
                "`area`'s closure takes two parameters",
            ),
            (quote! { || area(X) }, "`area` takes one argument"),
            (
                quote! { || area(|u, v| X, |u, v| Y) },
                "`area` takes one argument",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }
}
