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
//! type    ::= 'f32' | 'bool' | RECORD        -- an entry's parameters are `f32`s,
//!                                            -- records and families: its uniforms
//!           | '[' (RECORD | 'f32') ';' (INTEGER | IDENT) ']'
//!                                            -- a family, an entry's: N elements'
//!                                            -- uniforms, N a literal, a `usize`
//!                                            -- const or a structural parameter
//!
//! cexpr   ::= cexpr ('+' | '-' | '*' | '/') cexpr   -- an `f32` const's initializer,
//!           | '-' cexpr | '(' cexpr ')'            -- evaluated at expansion,
//!           | IDENT 'as' 'f32'                     -- per operation in f32
//!           | IDENT | LITERAL
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
//!           | iteration
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
//! iteration ::= family '.map' '(' element expr ')' '.sum' F32? '(' ')'           -- Σ
//!           | family '.map' '(' element expr ')' '.product' F32? '(' ')'       -- Π
//!           | family '.map' '(' element expr ')'
//!                 '.fold' '(' 'f32::INFINITY' ',' 'f32::min' ')'             -- min
//!           | family '.map' '(' element expr ')'
//!                 '.fold' '(' 'f32::NEG_INFINITY' ',' 'f32::max' ')'         -- max
//!           | family '.any' '(' element expr ')'                             -- ∃
//!           | family '.all' '(' element expr ')'                             -- ∀
//!                                          -- one copy of the body per element,
//!                                          -- made when the host fn is instantiated
//! family  ::= IDENT '.into_iter' '(' ')' | '(' family ')'
//!                                          -- an entry's family, by value
//! element ::= '|' IDENT '|'                -- one element: a record, or an `f32`
//! F32     ::= '::' '<' 'f32' '>'
//! binop   ::= '+' | '-' | '*' | '/'
//!           | '<' | '<=' | '>' | '>=' | '==' | '!='    -- a comparison: a bool
//!           | '&' | '|'                                -- bools combine
//! block   ::= '{' stmt* expr '}'
//! stmt    ::= 'let' IDENT (':' type)? '=' expr ';'
//!           | 'let' '(' IDENT (',' IDENT)* ','? ')' (':' '(' type (',' type)* ')')?
//!                 '=' '(' expr (',' expr)* ','? ')' ';'
//!                                          -- as many names as expressions, each
//!                                          -- bound to its own, all at once
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
//! among them. Nothing shadows `X`, `Y`, a `const`, a structural parameter
//! or a `fn`: a parameter, a `let` or an index of that name is refused. `X`
//! and `Y` appear only in an entry; a helper takes its coordinates as
//! arguments
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.2). A fold
//! (plan §1.5) denotes ⊕ of its body over `i ∈ [a, b)`, and its monoid's
//! identity when the range is empty; the syntax never unrolls it.
//!
//! Binding times (plan §1.4). An entry's `const N: usize` generics are
//! **structural**: each value is its own program, and a body reads one only
//! as a count. Every parameter is a **uniform**: an `f32` is one, a record is
//! one per field, and the host function's arguments are their values for
//! that call — never constants folded in. A record is written by name: a
//! parameter, a `let` alias of one (`let q = p;`), an argument passed on, or
//! the base of a field read.
//!
//! Families (plan §1.6). An entry's `pieces: [Row; N]` is `N` elements'
//! uniforms at static slots, element-major, and not a table: it is iterated
//! as a whole, `pieces.into_iter().map(|p| e).sum()` and its siblings,
//! which denote ⊕ of `e[p := element k]` over the elements, the monoid's
//! identity when there are none. The program holds one copy of the body per
//! element, made when the host function is instantiated; no fold, binder or
//! index exists for the family, and nothing else is done with one. The
//! spelling is Rust's own — `into_iter()` on an array yields its elements by
//! value, so `p` passes to a helper taking a `Row` — and rustc types it as
//! the kernel does.
//!
//! A tuple is taken apart where it is written, and only there:
//! `let (a, b) = (e1, e2);` binds each name to its expression, every
//! expression read before any name binds, as Rust's does (D7's front half).
//!
//! A record and its fields keep their attributes, re-emitted on the host
//! struct; every other item keeps only its doc comments.
//!
//! Refused here, with a span, at the token: an item that is not a `struct`, a
//! `const` or a `fn`; a record that is generic, a tuple struct or a unit
//! struct, and a `repr` or a `cfg` on a record or its field (§1.3); an
//! attribute other than a doc comment on a `const` or a `fn`; generics on a
//! helper and on a `const`, and any generic of an entry but a
//! `const N: usize` without a default; a parameter typed as a closure
//! (Phase D); a `fn` without a declared return type; an `if` without an
//! `else` or an `if let`; `loop`/`while`/`for`; assignment; `return`; a
//! closure anywhere but a fold's or an iteration's; a tuple
//! anywhere but a tuple `let`'s value, and a tuple's field (a tuple value is
//! Phase D, D7); a record literal (Phase D, D7); indexing and slicing (there
//! are no tables, §1.6); a range anywhere but a fold's; an inclusive or
//! open-ended range; a method on a range or a mapped
//! range that is not one of the fold's spellings; any method on a family's
//! `into_iter()` but `.map`, `.any` and `.all` — `.rev`, `.enumerate`,
//! `.skip`, … (no index, no order, §1.6) — and a mapped family not ended by
//! one of the six reductions; `.map`, `.any` or `.all` over anything but a
//! range or a family's `into_iter()` (an array's own `.map`, `.iter()`);
//! a `.fold` whose arguments are not one of the two above; a fold's or an
//! iteration's closure with a type annotation, a pattern, more than one
//! parameter, a return type or a qualifier; type arguments on a method
//! (`::<f32>` on `.sum` and `.product` excepted); an `as` to any type but
//! `f32`; a path or a call from outside the block; `%` and `!` (no IR op);
//! a `let` whose pattern is neither a plain name nor a tuple of them (`mut`,
//! `ref`, `@`, a record's pattern); a tuple `let` whose value is not a tuple
//! of as many expressions, with a pattern within its pattern, a `_` or a
//! `..`, a name bound twice, no names, or an annotation that is not a tuple
//! of as many types; a `let` without an initializer; `let … else`; an
//! item or a macro inside a block; an operator not in the table above; a
//! literal that is not a number; a literal suffixed with a type other than
//! `f32`; an integer past `u128`; a float past `f32`'s range; and any other
//! Rust expression syntax, named in the refusal. Nothing is passed through
//! for a later stage to refuse: the AST holds only what the language means.
//!
//! Parsed, and refused by `sema`: an unbound or retired name; a coordinate in
//! a helper; a call to an entry or to an unknown function; recursion; an
//! unknown method or a
//! known one at the wrong arity; a type error (every expression is an `f32`
//! or a `bool`; a `usize` is used only as `i as f32`; a record only by name);
//! a record's field that is not an `f32` (§1.3); a
//! record returned, built, chosen by an `if`, or used in arithmetic or a
//! comparison (Phase D, D7); a family whose element is not a record or an
//! `f32`, whose count is not an integer literal, a `usize` const or a
//! structural parameter, or that is a helper's parameter, and a family
//! anywhere but as what `into_iter()` iterates — passed, aliased, compared,
//! returned, converted, read by field — or with an array's method,
//! `.len()`, `.iter()`, `.get` (§1.6); iterating what is not a family; an
//! element outside its closure; an unknown field; an `as` of anything but a
//! `usize` name; an integer an `f32` does not hold exactly where a value is
//! expected; a `const` whose initializer is not a `cexpr` or an `iexpr`; a
//! negated `usize`; a range whose bounds are not constant or that runs
//! backwards; `.at()`,
//! `.constant()`, `.collapse()`; a block with no final expression; and an
//! `Args` record's name taken twice.
//!
//! Refused by lowering, where the language meets the IR's widths: a range
//! bound past 2²⁴, the last integer bound a fold's index — an `f32` lane —
//! names exactly, and folds nested deeper than the IR has binders. A range
//! over a structural parameter is refused the same ways when its host
//! function is instantiated, by rustc, in a `const` block.
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.

use crate::PLAN;
use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, CastExpr, ConstItem, Expr, FAMILY_SUM, FamilyExpr,
    FamilyType, FieldExpr, FnItem, FoldExpr, IdentExpr, IfExpr, KernelDef, LetStmt, Literal,
    LiteralExpr, MethodCallExpr, Param, ParamType, RangeExpr, RecordField, RecordItem, Reduction,
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
                ty: param_type(ty)?,
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
/// entry's structural parameters only through its arguments — a count as
/// `n as f32`, a family one element at a time.
fn refuse_a_helpers_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(param) = generics.params.first() {
        return Err(syn::Error::new_spanned(
            param,
            format!(
                "generics on a helper\n\
                 \n\
                 note: a helper is inlined into an entry, and reads the entry's structural \
                 parameters through its arguments: pass `n as f32`, or iterate a family in \
                 the entry and pass the helper one element\n\
                 note: structural parameters are an entry's, `pub fn f<const N: usize>` \
                 (§1.4 of {PLAN})"
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
    Ok(Param {
        name,
        ty: param_type(*typed.ty)?,
    })
}

/// A parameter's declared type: one value, or `[E; N]`, a family (plan
/// §1.6). What the element and the count are is `sema`'s question; the
/// count is converted as any expression is, so `sema` resolves it through
/// the scopes it resolves every name through.
fn param_type(ty: Type) -> syn::Result<ParamType> {
    let Type::Array(array) = ty else {
        return Ok(ParamType::One(Box::new(ty)));
    };
    Ok(ParamType::Family(FamilyType {
        span: array.bracket_token.span.join(),
        element: array.elem,
        count: Box::new(convert_expr(array.len)?),
    }))
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
            if let Some(family) = a_family(&expr_method.receiver) {
                return convert_family_method(family, &expr_method);
            }
            if let Some((family, map)) = a_mapped_family(&expr_method.receiver) {
                return convert_family_terminal(family, map, &expr_method);
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
        // parameter and is read by field; a tuple is taken apart where it is
        // written, by a `let` (D7's front half), and a tuple or a record
        // built as a value — passed, returned, chosen — is Phase D (D7).
        syn::Expr::Tuple(expr_tuple) => Err(syn::Error::new(
            expr_tuple.paren_token.span.join(),
            format!(
                "a tuple in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32` or a `bool`; a tuple is taken \
                 apart where it is written, by a `let`: {TUPLE_LET}\n\
                 note: a tuple as a value — passed, returned or chosen — is Phase D (D7 of \
                 {PLAN})"
            ),
        )),

        // There are no tables (§1.6): nothing reads a family, or anything
        // else, by index, and a slice is a range of indices.
        syn::Expr::Index(expr_index) => Err(syn::Error::new(
            expr_index.bracket_token.span.join(),
            format!(
                "indexing in a kernel body\n\
                 \n\
                 note: there are no tables (§1.6 of {PLAN}): a family is iterated as a whole, \
                 {FAMILY_SUM}, one copy of the body per element, and nothing reads one \
                 element, or a slice of them, by index"
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
                 note: a range is the domain of a fold: {FOLD_SPELLINGS}"
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
                 note: a closure is the body of a fold or of a family's iteration, and \
                 nothing else: {FOLD_SPELLINGS}; {FAMILY_SUM}\n\
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
                     note: a record's fields have names, `p.x0`; a tuple is taken apart by a \
                     `let`, {TUPLE_LET}\n\
                     note: a tuple as a value is Phase D (D7 of {PLAN})"
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

/// The methods that iterate, over a range or a family: `.map`, `.any` and
/// `.all`.
const ITERATING_METHODS: [&str; 3] = ["map", "any", "all"];

/// The refusal of one of [`ITERATING_METHODS`] with a closure, called on
/// anything but a range or a family's `into_iter()`: iteration spelled
/// another way, and not a closure out of place, which the closure's own
/// refusal would say.
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
    let lent = match a_named_call(receiver, "iter") {
        Some(family) => format!(
            "\nnote: `.iter()` lends references; a family's elements are iterated by value, \
             `{family}.into_iter()`"
        ),
        None => String::new(),
    };
    Some(syn::Error::new_spanned(
        receiver,
        format!(
            "`.{}` over `{}`, which is neither a range nor a family's `into_iter()`\n\
             \n\
             note: a fold iterates a constant range: {FOLD_SPELLINGS}\n\
             note: a family is iterated as a whole, at instantiation (§1.6 of {PLAN}): \
             {FAMILY_SPELLINGS}{lent}",
            call.method,
            quote::quote!(#receiver)
        ),
    ))
}

// ─────────────────────────────── families ───────────────────────────────

/// Every spelling of a family's iteration, as a refusal of any other names
/// them (plan §1.6).
const FAMILY_SPELLINGS: &str = "`pieces.into_iter().map(|p| e).sum()`, `.product()`, \
     `.fold(f32::INFINITY, f32::min)` or `.fold(f32::NEG_INFINITY, f32::max)`, and \
     `pieces.into_iter().any(|p| m)` or `.all(|p| m)`";

/// The one place a tuple is written: a `let` taking it apart.
const TUPLE_LET: &str = "`let (a, b) = (e1, e2);` binds each name to its expression";

/// `name.method()`, through parentheses: the plain name a method with no
/// arguments is called on.
fn a_named_call<'a>(expr: &'a syn::Expr, method: &str) -> Option<&'a syn::Ident> {
    let call = match expr {
        syn::Expr::Paren(paren) => return a_named_call(&paren.expr, method),
        syn::Expr::MethodCall(call) => call,
        _ => return None,
    };
    if call.method != method || !call.args.is_empty() || call.turbofish.is_some() {
        return None;
    }
    a_plain_name(&call.receiver)
}

/// A name, through parentheses: how a family is written.
fn a_plain_name(expr: &syn::Expr) -> Option<&syn::Ident> {
    match expr {
        syn::Expr::Paren(paren) => a_plain_name(&paren.expr),
        syn::Expr::Path(path) if path.qself.is_none() => path.path.get_ident(),
        _ => None,
    }
}

/// `pieces.into_iter()`: the family a method is called on, element by
/// element and by value — the spelling that types as the kernel does, since
/// an element passes to a helper taking the record.
fn a_family(receiver: &syn::Expr) -> Option<&syn::Ident> {
    a_named_call(receiver, "into_iter")
}

/// `pieces.into_iter().map(f)`: the family and the `map` call a reduction
/// is called on.
fn a_mapped_family(receiver: &syn::Expr) -> Option<(&syn::Ident, &syn::ExprMethodCall)> {
    let syn::Expr::MethodCall(map) = receiver else {
        return None;
    };
    if map.method != "map" {
        return None;
    }
    Some((a_family(&map.receiver)?, map))
}

/// A method called on a family's iterator itself: `.any(|p| m)` and
/// `.all(|p| m)` iterate it, and nothing else is a value — there is no index
/// to enumerate, skip or take by, and no order to reverse.
fn convert_family_method(family: &syn::Ident, call: &syn::ExprMethodCall) -> syn::Result<Expr> {
    let reduction = match call.method.to_string().as_str() {
        "any" => Reduction::Any,
        "all" => Reduction::All,
        "map" => {
            return Err(syn::Error::new_spanned(
                call,
                format!(
                    "a mapped family is an iterator, not a value\n\
                     \n\
                     note: a family's iteration ends in the reduction that combines its \
                     copies: {FAMILY_SPELLINGS}"
                ),
            ));
        }
        other => {
            return Err(syn::Error::new(
                call.method.span(),
                format!(
                    "`.{other}` on a family's iterator\n\
                     \n\
                     note: a family is iterated as a whole, its elements by value: \
                     {FAMILY_SPELLINGS}\n\
                     note: a family is not a table (§1.6 of {PLAN}): there is no index to \
                     enumerate, skip or take by, no order to reverse, and no length but the \
                     count it is declared with"
                ),
            ));
        }
    };
    refuse_turbofish(call)?;
    let (element, body) = the_closure(call, &FAMILY_CLOSURE)?;
    Ok(Expr::Family(FamilyExpr {
        reduction,
        family: family.clone(),
        element,
        body: Box::new(body),
        span: call.method.span(),
    }))
}

/// The method that ends `pieces.into_iter().map(|p| e)`: the reduction its
/// copies are combined under.
fn convert_family_terminal(
    family: &syn::Ident,
    map: &syn::ExprMethodCall,
    call: &syn::ExprMethodCall,
) -> syn::Result<Expr> {
    let reduction = mapped_reduction(call)?;
    refuse_turbofish(map)?;
    let (element, body) = the_closure(map, &FAMILY_CLOSURE)?;
    Ok(Expr::Family(FamilyExpr {
        reduction,
        family: family.clone(),
        element,
        body: Box::new(body),
        span: call.method.span(),
    }))
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
    let (binder, body) = the_closure(call, &FOLD_CLOSURE)?;
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
    let reduction = mapped_reduction(call)?;
    refuse_turbofish(map)?;
    let (binder, body) = the_closure(map, &FOLD_CLOSURE)?;
    Ok(Expr::Fold(FoldExpr {
        reduction,
        range: convert_range(range)?,
        binder,
        body: Box::new(body),
        span: call.method.span(),
    }))
}

/// The reduction a method called on a `.map(…)` names: the monoid a fold's
/// terms, or a family's copies, are combined under.
fn mapped_reduction(call: &syn::ExprMethodCall) -> syn::Result<Reduction> {
    match call.method.to_string().as_str() {
        "sum" => sum_or_product(call, Reduction::Sum),
        "product" => sum_or_product(call, Reduction::Product),
        "fold" => fold_monoid(call),
        other => Err(syn::Error::new(
            call.method.span(),
            format!(
                "`.{other}` does not end a fold\n\
                 \n\
                 note: a fold is spelled {FOLD_SPELLINGS}\n\
                 note: a family's iteration is spelled {FAMILY_SPELLINGS}"
            ),
        )),
    }
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

/// What the one parameter of an iteration's closure is — a fold's index or
/// a family's element — for the refusal of any other shape.
struct IterationClosure {
    /// Whose closure it is: `a fold's`.
    whose: &'static str,
    /// Its one spelling: `|i| body`.
    spelling: &'static str,
    /// What its parameter and its body are.
    meaning: &'static str,
    /// What its body is, which is why it declares no return type.
    body: &'static str,
    /// What its one parameter is.
    parameter: &'static str,
    /// The refusal of a typed parameter.
    typed: &'static str,
    /// The refusal of a pattern.
    pattern: &'static str,
}

/// `(a..b).map(|i| body)`: the parameter is the index.
const FOLD_CLOSURE: IterationClosure = IterationClosure {
    whose: "a fold's",
    spelling: "|i| body",
    meaning: "its parameter is the fold's index and its body is what the fold combines",
    body: "a term of the fold",
    parameter: "the index it ranges over",
    typed: "a fold's index is a `usize`, always; write the plain name, `|i|`",
    pattern: "a fold's index is a plain name: `|i|`",
};

/// `pieces.into_iter().map(|p| body)`: the parameter is one element.
const FAMILY_CLOSURE: IterationClosure = IterationClosure {
    whose: "a family's",
    spelling: "|p| body",
    meaning: "its parameter is one element of the family and its body is that element's \
              copy",
    body: "each element's copy",
    parameter: "the element",
    typed: "a family's element has the family's element type, always; write the plain \
            name, `|p|`",
    pattern: "a family's element is a plain name, `|p|`; a record element's fields are \
              read by name, `p.x0`",
};

/// The one argument of an iteration's `.map`, `.any` or `.all`: a closure
/// whose parameter is a fold's index or a family's element, and whose body
/// is what is combined.
fn the_closure(
    call: &syn::ExprMethodCall,
    shape: &IterationClosure,
) -> syn::Result<(syn::Ident, Expr)> {
    let IterationClosure {
        whose,
        spelling,
        meaning,
        body,
        parameter,
        typed,
        pattern,
    } = shape;
    let [syn::Expr::Closure(closure)] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new(
            call.method.span(),
            format!(
                "`.{}` takes one argument, the closure `{spelling}`\n\
                 \n\
                 note: {meaning}",
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
            format!(
                "{whose} closure is `{spelling}`, unqualified\n\
                 \n\
                 note: it is {body}, not a value: nothing is captured, moved or awaited"
            ),
        ));
    }
    if let syn::ReturnType::Type(_, ty) = &closure.output {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "{whose} closure declares no return type: its body is {body}, an `f32` for a \
                 sum, product, min or max, a `bool` for `any` or `all`"
            ),
        ));
    }
    let [param] = closure.inputs.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new_spanned(
            &closure.inputs,
            format!("{whose} closure takes one parameter, {parameter}"),
        ));
    };
    let name = match param {
        Pat::Ident(pat_ident) => plain_name(pat_ident, param)?,
        Pat::Type(annotated) => return Err(syn::Error::new_spanned(&annotated.ty, *typed)),
        other => return Err(syn::Error::new_spanned(other, *pattern)),
    };
    Ok((name, convert_expr((*closure.body).clone())?))
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
            syn::Stmt::Local(local) => stmts.push(convert_let(local)?),

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

/// A `let`: a plain name bound to its expression, or `let (a, b) = (e1,
/// e2);`, each name bound to its own expression (D7's front half).
fn convert_let(local: &syn::Local) -> syn::Result<Stmt> {
    let (pattern, annotation) = match &local.pat {
        Pat::Type(pat_type) => (&*pat_type.pat, Some(&*pat_type.ty)),
        pattern => (pattern, None),
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
    // The `else` branch used to be dropped unread. It could never run — a
    // plain name always matches, and so does a tuple of them — so accepting
    // it would compile a branch nobody can reach, silently.
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
    match pattern {
        Pat::Ident(pat_ident) => {
            let name = plain_name(pat_ident, &local.pat)?;
            Ok(Stmt::Let(Box::new(LetStmt {
                span: name.span(),
                name,
                ty: annotation.cloned(),
                init: convert_expr((*init.expr).clone())?,
            })))
        }
        Pat::Tuple(names) => convert_let_tuple(names, annotation, &init.expr),
        pattern => Err(refuse_let_pattern(pattern)),
    }
}

/// `let (a, b) = (e1, e2);`: plain names, as many as the tuple of
/// expressions has elements, each bound to its own, all at once. A tuple is
/// taken apart where it is written and nowhere else, so nothing of it is
/// left to be a value (D7's front half); a pattern within the pattern, a
/// `_`, and a value that is not a tuple of expressions are refused.
fn convert_let_tuple(
    pattern: &syn::PatTuple,
    annotation: Option<&Type>,
    init: &syn::Expr,
) -> syn::Result<Stmt> {
    let mut names: Vec<syn::Ident> = Vec::with_capacity(pattern.elems.len());
    for element in &pattern.elems {
        let name = match element {
            Pat::Ident(pat_ident) => plain_name(pat_ident, element)?,
            Pat::Wild(wild) => {
                return Err(syn::Error::new_spanned(
                    wild,
                    "`_` in a tuple `let`\n\
                     \n\
                     note: a kernel body has no effects, so a value nothing reads need not be \
                     written: drop it from both tuples",
                ));
            }
            Pat::Tuple(_) | Pat::Paren(_) => {
                return Err(syn::Error::new_spanned(
                    element,
                    format!(
                        "a pattern within a tuple `let`\n\
                         \n\
                         note: a tuple inside a tuple is a tuple value, and a tuple as a value \
                         is Phase D (D7 of {PLAN})\n\
                         help: bind the names flat, `let (a, b, c) = (e1, e2, e3);`"
                    ),
                ));
            }
            other => return Err(refuse_let_pattern(other)),
        };
        if names.contains(&name) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "identifier `{name}` is bound more than once in the same pattern\n\
                     \n\
                     note: rustc refuses it too (E0416)"
                ),
            ));
        }
        names.push(name);
    }
    if names.is_empty() {
        return Err(syn::Error::new_spanned(
            pattern,
            "a tuple `let` of no names binds nothing\n\
             \n\
             note: a kernel body has no unit value to take apart",
        ));
    }
    let values = match tuple_expression(init) {
        Some(values) => values,
        None => {
            return Err(syn::Error::new_spanned(
                init,
                format!(
                    "the value of a tuple `let` is a tuple of expressions, `(e1, e2)`\n\
                     \n\
                     note: a tuple is taken apart where it is written; a tuple computed — \
                     returned, passed or chosen — is a tuple value, which is Phase D (D7 of \
                     {PLAN})"
                ),
            ));
        }
    };
    if values.len() != names.len() {
        return Err(syn::Error::new_spanned(
            init,
            format!(
                "mismatched types: expected a tuple with {} elements, found one with {} \
                 elements\n\
                 \n\
                 note: `let ({}) = …` binds each name to one expression of the tuple",
                names.len(),
                values.len(),
                names
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    let types = element_types(annotation, names.len())?;
    let mut lets = Vec::with_capacity(names.len());
    for ((name, value), ty) in names.into_iter().zip(values).zip(types) {
        lets.push(LetStmt {
            span: name.span(),
            name,
            ty,
            init: convert_expr(value.clone())?,
        });
    }
    Ok(Stmt::LetTuple(lets))
}

/// The expressions of a tuple, through parentheses, or `None` for anything
/// else.
fn tuple_expression(expr: &syn::Expr) -> Option<Vec<&syn::Expr>> {
    match expr {
        syn::Expr::Paren(paren) => tuple_expression(&paren.expr),
        syn::Expr::Tuple(tuple) => Some(tuple.elems.iter().collect()),
        _ => None,
    }
}

/// Each name's annotation in `let (a, b): (f32, f32) = …`: the tuple
/// type's elements, one per name, or none at all when there is no
/// annotation.
fn element_types(annotation: Option<&Type>, arity: usize) -> syn::Result<Vec<Option<Type>>> {
    let Some(annotation) = annotation else {
        return Ok(vec![None; arity]);
    };
    match annotation {
        Type::Tuple(tuple) if tuple.elems.len() == arity => {
            Ok(tuple.elems.iter().cloned().map(Some).collect())
        }
        _ => Err(syn::Error::new_spanned(
            annotation,
            format!(
                "the annotation of a tuple `let` of {arity} names is a tuple of {arity} types, \
                 `(f32, f32)`"
            ),
        )),
    }
}

/// A `let` whose pattern is neither a name nor a tuple of names — `let Row
/// { x0, .. } = …`, `let [a, b] = …` — takes a value apart that the language
/// reads otherwise: a record by field.
fn refuse_let_pattern(pattern: &Pat) -> syn::Error {
    syn::Error::new_spanned(
        pattern,
        format!(
            "a pattern in a kernel `let`\n\
             \n\
             note: a kernel `let` binds a plain name — an `f32`, a `bool`, or a record, which \
             it aliases (`let q = p;`) and whose fields are read by name (`q.x0`) — or takes \
             a tuple apart where it is written, {TUPLE_LET}\n\
             note: a record's fields are read by name, not taken apart by a pattern \
             (§1.3 of {PLAN})"
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
            let syn::Type::Path(path) = param.ty.written() else {
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

    /// A value with parts built or taken apart in a body, anywhere but a
    /// tuple `let`, is refused where it is written: a record's pattern
    /// naming §1.3, since a record is read by field; a tuple as a value and
    /// a tuple's field naming Phase D (D7), since a tuple is taken apart
    /// where it is written and so is never one; a record literal naming
    /// Phase D (D7). (They named B3 until the tuple `let` came, which is
    /// `a_tuple_let_binds_each_name_to_its_expression`.)
    #[test]
    fn a_record_pattern_a_tuple_value_and_a_record_literal_are_refused_naming_their_phase() {
        let cases: [(TokenStream, &str, &str); 4] = [
            (
                quote! { || { let Row { x0, .. } = p; x0 } },
                "a pattern",
                "§1.3",
            ),
            (quote! { || (X, Y) }, "a tuple in a kernel body", "D7"),
            (quote! { |p: f32| p.0 }, "a tuple's field", "D7"),
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

    /// A range anywhere but a fold's names the fold's spellings.
    #[test]
    fn a_range_is_refused_naming_the_fold() {
        for input in [
            quote! { || X + (0.0..1.0) },
            quote! { || DX(0.0..1.0) },
            quote! { || { let r = 0..4; X } },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("a range in a kernel body") && err.contains("(a..b).map(|i| e).sum()"),
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

    /// A closure is the body of a fold or of a family's iteration and
    /// nothing else: anywhere else it is refused, naming the spellings of
    /// both, and the phase that brings a function as an argument.
    #[test]
    fn a_closure_outside_a_fold_or_an_iteration_is_refused() {
        for input in [
            quote! { || X.max(|i| i) },
            quote! { || { let f = |x: f32| x; X } },
            quote! { || (0..4).map(|i| X).sum() + (|j| Y) },
            quote! { || DX(|u| u) },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("a closure in a kernel body")
                    && err.contains("the body of a fold or of a family's iteration")
                    && err.contains("(a..b).map(|i| e).sum()")
                    && err.contains("pieces.into_iter().map(|p| e).sum()")
                    && err.contains("Phase D"),
                "got: {err}"
            );
        }
    }

    /// An iteration's spelling over anything but a range or a family's
    /// `into_iter()` is refused naming both spellings and §1.6, where the
    /// closure's own refusal would name them less precisely: an
    /// array's own `.map`, and `.iter()`, which lends references where a
    /// family's elements are values, among them.
    #[test]
    fn iterating_anything_but_a_range_or_a_family_is_refused_naming_both() {
        for input in [
            quote! { |pieces: f32| pieces.map(|p| p * X).sum() },
            quote! { |pieces: f32| pieces.any(|p| p < X) },
            quote! { || X.all(|p| p < Y) },
            quote! { |v: [f32; 2]| v.map(|e| e * X).sum() },
            quote! { |v: [f32; 2]| (X + v).into_iter().map(|e| e).sum() },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("which is neither a range nor a family's `into_iter()`")
                    && err.contains("(a..b).map(|i| e).sum()")
                    && err.contains("pieces.into_iter().map(|p| e).sum()")
                    && err.contains("§1.6"),
                "got: {err}"
            );
        }
        let err = refusal(quote! { |v: [f32; 2]| v.iter().map(|e| e * X).sum() });
        assert!(
            err.contains("`.iter()` lends references") && err.contains("`v.into_iter()`"),
            "got: {err}"
        );
    }

    // ───────────────────────────── families ─────────────────────────────

    /// A parameter `[E; N]` is a family: its element as written, and its
    /// count converted as any expression is, for `sema` to resolve.
    #[test]
    fn a_family_parameter_parses_as_its_element_and_count() {
        let def = parse(quote! {
            pub fn f<const N: usize>(pieces: [Row; N], r: f32) -> f32 { r }
        })
        .expect("parses");
        let ParamType::Family(family) = &def.fns[0].params[0].ty else {
            panic!("`[Row; N]` is a family: {:?}", def.fns[0].params[0].ty);
        };
        let syn::Type::Path(element) = &*family.element else {
            panic!("the element is `Row`");
        };
        assert!(element.path.is_ident("Row"));
        assert_eq!(
            family.count.named().map(ToString::to_string),
            Some("N".to_string())
        );
        assert!(matches!(def.fns[0].params[1].ty, ParamType::One(_)));

        let def = parse(quote! { |v: [f32; 3]| X }).expect("parses");
        let ParamType::Family(family) = &entry(&def).params[0].ty else {
            panic!("`[f32; 3]` is a family");
        };
        assert!(matches!(*family.count, Expr::Literal(_)));
    }

    /// The family iteration the body `|v: [f32; 2]| <expr>` parses to.
    fn family(input: TokenStream) -> FamilyExpr {
        let def = parse(quote! { |v: [f32; 2]| #input }).expect("the iteration parses");
        match &entry(&def).body {
            Expr::Family(family) => family.clone(),
            other => panic!("expected a family's iteration, got {other:?}"),
        }
    }

    /// Each spelling of a family's iteration names its monoid, as a fold's
    /// does; the closure's parameter is the element, and the family is the
    /// name `into_iter()` is called on, through parentheses too.
    #[test]
    fn every_family_spelling_parses_to_its_reduction() {
        let cases: [(TokenStream, Reduction); 8] = [
            (
                quote! { v.into_iter().map(|e| e * X).sum() },
                Reduction::Sum,
            ),
            (
                quote! { v.into_iter().map(|e| e * X).sum::<f32>() },
                Reduction::Sum,
            ),
            (
                quote! { v.into_iter().map(|e| e * X).product() },
                Reduction::Product,
            ),
            (
                quote! { (v).into_iter().map(|e| e * X).product::<f32>() },
                Reduction::Product,
            ),
            (
                quote! { v.into_iter().map(|e| e * X).fold(f32::INFINITY, f32::min) },
                Reduction::Min,
            ),
            (
                quote! { v.into_iter().map(|e| e * X).fold(f32::NEG_INFINITY, f32::max) },
                Reduction::Max,
            ),
            (quote! { v.into_iter().any(|e| e < X) }, Reduction::Any),
            (quote! { v.into_iter().all(|e| e < X) }, Reduction::All),
        ];
        for (input, want) in cases {
            let parsed = family(input);
            assert_eq!(parsed.reduction, want);
            assert_eq!(parsed.family.to_string(), "v");
            assert_eq!(parsed.element.to_string(), "e");
        }
    }

    /// A family iterated any other way is refused where it is written,
    /// naming the spellings it has: no adaptor that indexes, skips, takes or
    /// reorders (a family is not a table, §1.6), no mapped family left as an
    /// iterator, no reduction but the six, and a closure `|p| body`.
    #[test]
    fn a_family_iterated_any_other_way_is_refused() {
        let cases: [(TokenStream, &str); 13] = [
            (
                quote! { v.into_iter().rev().map(|e| e).sum() },
                "`.rev` on a family's iterator",
            ),
            (
                quote! { v.into_iter().enumerate().map(|e| X).sum() },
                "`.enumerate` on a family's iterator",
            ),
            (quote! { v.into_iter().skip(1).any(|e| e < X) }, "§1.6"),
            (
                quote! { v.into_iter().map(|e| e) },
                "a mapped family is an iterator, not a value",
            ),
            (
                quote! { v.into_iter().map(|e| e).min() },
                "does not end a fold",
            ),
            (
                quote! { v.into_iter().map(|e| e).sum(1.0) },
                "takes no arguments",
            ),
            (
                quote! { v.into_iter().map(|e| e).fold(0.0, |a, b| a + b) },
                "names no monoid",
            ),
            (
                quote! { v.into_iter().map(|e: f32| e).sum() },
                "the family's element type, always",
            ),
            (
                quote! { v.into_iter().map(|(a, b)| a).sum() },
                "a family's element is a plain name",
            ),
            (
                quote! { v.into_iter().map(move |e| e).sum() },
                "a family's closure is `|p| body`, unqualified",
            ),
            (
                quote! { v.into_iter().map(|e, f| e).sum() },
                "a family's closure takes one parameter, the element",
            ),
            (
                quote! { v.into_iter().map(X).sum() },
                "the closure `|p| body`",
            ),
            (
                quote! { v.into_iter().map::<f32>(|e| e).sum() },
                "takes no type arguments",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(quote! { |v: [f32; 2]| #input });
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// Nothing is read by index — a family's element, a slice of them, or
    /// anything else: there are no tables (§1.6).
    #[test]
    fn indexing_and_slicing_are_refused_naming_section_1_6() {
        for input in [
            quote! { |v: [f32; 3]| v[0] },
            quote! { |v: [f32; 3]| v[1..3].into_iter().map(|e| e).sum() },
            quote! { || X[0] },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("indexing in a kernel body")
                    && err.contains("there are no tables")
                    && err.contains("§1.6"),
                "got: {err}"
            );
        }
    }

    // ───────────────────────────── tuple lets ─────────────────────────────

    /// The one tuple `let` of `|| { input; X }`, flattened.
    fn tuple_let(input: TokenStream) -> Vec<LetStmt> {
        let def = parse(quote! { || { #input; X } }).expect("parses");
        let Expr::Block(block) = &entry(&def).body else {
            panic!("a block");
        };
        let [Stmt::LetTuple(lets)] = block.stmts.as_slice() else {
            panic!("one tuple `let`, got {:?}", block.stmts);
        };
        lets.clone()
    }

    /// `let (a, b) = (e1, e2);` is flattened: each name bound to its own
    /// expression, with its own annotation when the tuple type gives one,
    /// in one statement that binds them all at once (D7's front half). It
    /// used to be refused, naming B3.
    #[test]
    fn a_tuple_let_binds_each_name_to_its_expression() {
        let lets = tuple_let(quote! { let (a, b) = (X, Y + 1.0) });
        let names: Vec<String> = lets.iter().map(|l| l.name.to_string()).collect();
        assert_eq!(names, ["a", "b"]);
        assert!(matches!(lets[0].init, Expr::Ident(_)));
        assert!(matches!(lets[1].init, Expr::Binary(_)));
        assert!(lets.iter().all(|l| l.ty.is_none()));

        let lets = tuple_let(quote! { let (m, w): (bool, f32) = ((X < Y), X) });
        let types: Vec<String> = lets
            .iter()
            .map(|l| {
                let ty = l.ty.as_ref().expect("annotated");
                quote!(#ty).to_string()
            })
            .collect();
        assert_eq!(types, ["bool", "f32"]);

        assert_eq!(
            tuple_let(quote! { let (a,) = ((X,)) }).len(),
            1,
            "a tuple of one"
        );
    }

    /// A tuple is written in one place, the tuple `let` that takes it
    /// apart; anywhere else it is a tuple value, refused naming Phase D
    /// (D7), and so is a `let` whose value is not a tuple of expressions.
    #[test]
    fn a_tuple_anywhere_but_a_destructuring_let_is_refused() {
        let cases: [(TokenStream, &str); 6] = [
            (quote! { || (X, Y) }, "a tuple in a kernel body"),
            (
                quote! { || { let t = (X, Y); X } },
                "a tuple in a kernel body",
            ),
            (quote! { || X.max((X, Y)) }, "a tuple in a kernel body"),
            (
                quote! { || { let (a, b) = (X, Y); (b, a) } },
                "a tuple in a kernel body",
            ),
            (
                quote! { || { let (a, b) = t; a } },
                "the value of a tuple `let` is a tuple of expressions",
            ),
            (
                quote! { || { let (a, b) = if X < Y { (X, Y) } else { (Y, X) }; a } },
                "the value of a tuple `let` is a tuple of expressions",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(
                err.contains(expected) && err.contains("Phase D") && err.contains("D7"),
                "expected `{expected}` naming Phase D, got: {err}"
            );
        }
    }

    /// A tuple `let` binds plain names, as many as its tuple has elements,
    /// each once: a mismatched arity (rustc's E0308), a pattern within the
    /// pattern (a tuple value, Phase D), `_`, `..`, a name bound twice
    /// (E0416), no names at all, and an annotation that is not a tuple of
    /// as many types are each refused where they are written.
    #[test]
    fn a_tuple_let_of_another_shape_is_refused() {
        let cases: [(TokenStream, &str); 9] = [
            (
                quote! { let (a, b) = (X, Y, X) },
                "expected a tuple with 2 elements, found one with 3 elements",
            ),
            (
                quote! { let (a, b, c) = (X, Y) },
                "expected a tuple with 3 elements, found one with 2 elements",
            ),
            (
                quote! { let ((a, b), c) = ((X, Y), X) },
                "a pattern within a tuple `let`",
            ),
            (
                quote! { let ((a), b) = (X, Y) },
                "a pattern within a tuple `let`",
            ),
            (quote! { let (a, _) = (X, Y) }, "`_` in a tuple `let`"),
            (
                quote! { let (a, ..) = (X, Y) },
                "a pattern in a kernel `let`",
            ),
            (
                quote! { let (a, a) = (X, Y) },
                "identifier `a` is bound more than once in the same pattern",
            ),
            (quote! { let () = () }, "binds nothing"),
            (
                quote! { let (a, b): (f32, f32, f32) = (X, Y) },
                "the annotation of a tuple `let` of 2 names is a tuple of 2 types",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(quote! { || { #input; X } });
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
        let err = refusal(quote! { || { let ((a, b), c) = ((X, Y), X); a } });
        assert!(err.contains("Phase D (D7"), "got: {err}");
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
    /// entry's generics are `const N: usize` and nothing else, and a helper
    /// has none — it reads its entry's through its arguments. (The helper's
    /// refusal named B3 until B3 was done without helper generics.)
    #[test]
    fn a_fn_signature_is_plain() {
        let cases: [(TokenStream, &str); 10] = [
            (
                quote! { fn h<const N: usize>(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
                "generics on a helper",
            ),
            (
                quote! { fn h<const N: usize>(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
                "pass the helper one element",
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
}
