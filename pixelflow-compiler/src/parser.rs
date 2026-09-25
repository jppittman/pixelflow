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
//! item    ::= 'pub'? 'const' IDENT ':' 'f32' '=' cexpr ';'
//!           | 'pub'? 'fn' IDENT '(' params ')' '->' type block
//!                                            -- `pub`: an entry; private: a helper
//! params  ::= (param (',' param)* ','?)?
//! param   ::= IDENT ':' type
//! type    ::= 'f32' | 'bool'                 -- an entry's parameters are `f32`
//!
//! cexpr   ::= cexpr ('+' | '-' | '*' | '/') cexpr   -- a const's initializer,
//!           | '-' cexpr | '(' cexpr ')'            -- evaluated at expansion,
//!           | IDENT | LITERAL                      -- per operation in f32
//!
//! expr    ::= expr binop expr
//!           | '-' expr
//!           | expr '.' METHOD '(' (expr (',' expr)*)? ')'
//!           | PROJECTION '(' expr ')'
//!           | IDENT '(' (expr (',' expr)*)? ')'    -- a helper, inlined
//!           | 'if' expr block 'else' (block | 'if' …)   -- the choice
//!           | '(' expr ')'
//!           | block
//!           | IDENT                    -- X, Y (in an entry), a parameter, a
//!                                      -- const, or a `let` in scope
//!           | LITERAL                  -- an integer or a float, as its f32
//! binop   ::= '+' | '-' | '*' | '/'
//!           | '<' | '<=' | '>' | '>=' | '==' | '!='    -- a comparison: a bool
//!           | '&' | '|'                                -- bools combine
//! block   ::= '{' stmt* expr '}'
//! stmt    ::= 'let' IDENT (':' type)? '=' expr ';'
//!           | expr ';'
//!
//! METHOD     -- an `OpKind` method, a `LIBRARY_METHODS` composition, or `clone`
//! PROJECTION -- V, DX, DY, DXX, DXY, DYY
//! ```
//!
//! A `let` is scoped as Rust scopes it (`crate::symbol`). Nothing shadows
//! `X`, `Y`, a `const` or a `fn`: a parameter or a `let` of that name is
//! refused. `X` and `Y` appear only in an entry; a helper takes its
//! coordinates as arguments (docs/plans/2026-09-25-the-language-is-kernel.md
//! §1.2).
//!
//! Refused here, with a span, at the token: an item that is not a `const` or
//! a `fn` (records are B3 of the plan above), generics on either (B3), a
//! parameter typed as a closure (B4), a `fn` without a declared return type,
//! an `if` without an `else` or an `if let`, `loop`/`while`/`for`,
//! assignment, `return`, a closure (B4), a tuple (D7, B3), a field access
//! (B3), a range (B2; `integral` is B4), a path or a call from outside the
//! block, `%` and `!` (no IR op), a `let` whose pattern is not a plain name
//! (`mut`, `ref`, `@`; destructuring is B3), a `let` without an initializer,
//! `let … else`, an item or a macro inside a block, an operator not in the
//! table above, a literal that is not a number, a literal suffixed with a
//! type other than `f32`, an integer an `f32` does not hold exactly, a float
//! past `f32`'s range, and any other Rust expression syntax, named in the
//! refusal. Nothing is passed through for a later stage to refuse: the AST
//! holds only what the language means.
//!
//! Parsed, and refused by `sema`: an unbound or retired name, a coordinate in
//! a helper, a call to an entry or to an unknown function (`integral`,
//! `area` and `monotone_root` name B4), recursion, an unknown method or a
//! known one at the wrong arity, a type error (every expression is an `f32`
//! or a `bool`), a `const` whose initializer is not a `cexpr`, `.at()`,
//! `.constant()`, `.collapse()`, and a block with no final expression.
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.

use crate::PLAN;
use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, ConstItem, Expr, FnItem, IdentExpr, IfExpr,
    KernelDef, LetStmt, LiteralExpr, MethodCallExpr, Param, Spelling, Stmt, UnaryExpr, UnaryOp,
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
        params,
        ret: None,
        body,
    };
    Ok(KernelDef {
        spelling: Spelling::Closure,
        consts: Vec::new(),
        fns: vec![entry],
    })
}

/// The items form: `const`s and `fn`s, as a `syn::File`.
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
        consts: Vec::new(),
        fns: Vec::new(),
    };
    for item in file.items {
        match item {
            syn::Item::Const(item) => def.consts.push(convert_const(item)?),
            syn::Item::Fn(item) => def.fns.push(convert_fn(item)?),
            other => return Err(refuse_item(&other)),
        }
    }
    Ok(def)
}

/// An item that is neither a `const` nor a `fn`, named by its kind so the
/// message can say which phase brings it.
fn refuse_item(item: &syn::Item) -> syn::Error {
    let kind = match item {
        syn::Item::Struct(_) => "struct",
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
             note: a `kernel!` block holds `const` items and `fn` items: a `pub fn` is an \
             entry, a private `fn` is a helper\n\
             note: records (`struct`) are B3 of {PLAN}"
        ),
    )
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
        ret: Some(ret),
        body: Expr::Block(convert_block(*item.block)?),
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
    match &*typed.ty {
        Type::ImplTrait(_) | Type::BareFn(_) | Type::TraitObject(_) => {
            return Err(syn::Error::new_spanned(
                &typed.ty,
                format!(
                    "a kernel-typed parameter\n\
                     \n\
                     note: passing a function to a kernel `fn` is B4 and Phase D of {PLAN}; \
                     this parameter is a scalar, `f32` or `bool`"
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

/// Generics and const generics are the structural parameters of B3; until
/// then a `fn` or a `const` takes none.
fn refuse_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(param) = generics.params.first() {
        return Err(syn::Error::new_spanned(
            param,
            format!(
                "generics in a `kernel!` block\n\
                 \n\
                 note: structural parameters (`const N: usize`) are B3 of {PLAN}"
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
            let receiver = convert_expr(*expr_method.receiver)?;
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

        // A value with parts. Records and tuples are flattened in the front
        // end (D7), which is B3.
        syn::Expr::Tuple(expr_tuple) => Err(syn::Error::new(
            expr_tuple.paren_token.span.join(),
            format!(
                "a tuple in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32` or a `bool`\n\
                 note: records and tuples are flattened in the front end (D7 of {PLAN}), \
                 which is B3"
            ),
        )),

        syn::Expr::Field(expr_field) => Err(syn::Error::new_spanned(
            expr_field,
            format!(
                "a field access in a kernel body\n\
                 \n\
                 note: records and their fields are B3 of {PLAN}"
            ),
        )),

        syn::Expr::Range(expr_range) => Err(syn::Error::new_spanned(
            expr_range,
            format!(
                "a range in a kernel body\n\
                 \n\
                 note: a fold over a constant range is B2 of {PLAN}; `integral(lo..hi, |u| e)` \
                 is B4"
            ),
        )),

        syn::Expr::Block(expr_block) => {
            let block = convert_block(expr_block.block)?;
            Ok(Expr::Block(block))
        }

        // Iteration with state has no denotation in a DAG. A bounded
        // reduction is a fold over a constant range, which B2 spells.
        syn::Expr::Loop(_) | syn::Expr::While(_) | syn::Expr::ForLoop(_) => {
            Err(syn::Error::new_spanned(
                expr,
                format!(
                    "a loop in a kernel body\n\
                     \n\
                     note: the language is a DAG: nothing iterates with state\n\
                     note: a bounded reduction over a constant range is B2 of {PLAN}"
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
                 note: a function as an argument is B4 of {PLAN}; a private `fn` in the \
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
/// = …` — takes a value apart, and a kernel value has no parts yet.
fn refuse_let_pattern(pattern: &Pat) -> syn::Error {
    syn::Error::new_spanned(
        pattern,
        format!(
            "a pattern in a kernel `let`\n\
             \n\
             note: a kernel `let` binds a plain name: every value in a kernel body is an \
             `f32` or a `bool`\n\
             note: records and their fields are B3 of {PLAN}"
        ),
    )
}

/// The one type suffix a numeric literal in a kernel body may carry: every
/// value there is an `f32`.
const F32_SUFFIX: &str = "f32";

/// The `f32` a literal denotes, or a spanned refusal saying why it names none.
fn literal_value(lit: &syn::Lit) -> syn::Result<f32> {
    match lit {
        syn::Lit::Float(float) => float_value(float),
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

/// An integer literal denotes an exact integer, so it is refused unless an
/// `f32` holds it exactly: at most [`f32::MANTISSA_DIGITS`] significant bits.
///
/// An integer's digits claim exactness, and rustc has no rounding of its own
/// to borrow here: it refuses an unsuffixed integer where an `f32` is
/// expected. `16777217` (2²⁴ + 1) is the first integer refused;
/// `1099511627776` (2⁴⁰) is accepted.
///
/// `16777217f32` is a float literal to rustc (the suffix makes it one), and
/// it rounds once like `16777217.0`, so that is what it does here too.
fn int_value(int: &syn::LitInt) -> syn::Result<f32> {
    refuse_a_foreign_suffix(int.suffix(), int)?;
    if int.suffix() == F32_SUFFIX {
        return rounded_once(int.base10_digits(), int);
    }
    let inexact = || {
        syn::Error::new_spanned(
            int,
            format!(
                "`{int}` is not exactly representable as an `f32`\n\
                 \n\
                 note: an `f32` holds an integer exactly only when it has at most {} \
                 significant bits\n\
                 \n\
                 help: write the `f32` you mean as a float literal, e.g. `{digits}.0`, \
                 which rounds to the nearest one",
                f32::MANTISSA_DIGITS,
                digits = int.base10_digits(),
            ),
        )
    };
    let n = int.base10_parse::<u128>().map_err(|_| inexact())?;
    if significant_bits(n) > f32::MANTISSA_DIGITS {
        return Err(inexact());
    }
    // Exact, and finite: below 2¹²⁸, an integer with at most `MANTISSA_DIGITS`
    // significant bits is at most `f32::MAX`.
    Ok(n as f32)
}

/// The bits between an integer's highest and lowest set bits, inclusive:
/// what a binary significand must hold to represent it exactly.
fn significant_bits(n: u128) -> u32 {
    if n == 0 {
        return 0;
    }
    u128::BITS - n.leading_zeros() - n.trailing_zeros()
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
/// `2.5f64` or `3u8` asks for a type a kernel body does not have.
fn refuse_a_foreign_suffix(suffix: &str, literal: &impl quote::ToTokens) -> syn::Result<()> {
    match suffix {
        "" | F32_SUFFIX => Ok(()),
        other => Err(syn::Error::new_spanned(
            literal,
            format!(
                "a `{other}` literal in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32`\n\
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

    /// A value with parts is refused where it is written, naming the phase
    /// that brings records: a `let` pattern, a field access, a tuple.
    #[test]
    fn a_pattern_a_field_and_a_tuple_are_refused_naming_b3() {
        let cases: [(TokenStream, &str); 5] = [
            (quote! { || { let (a, b) = (X, Y); a } }, "a pattern"),
            (
                quote! { || { let (a, b): (f32, f32) = (X, Y); a } },
                "a pattern",
            ),
            (quote! { |p: f32| p.e0y }, "a field access"),
            (quote! { || (X, Y) }, "a tuple"),
            (quote! { || (X, Y) }, "D7"),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(
                err.contains(expected) && err.contains("B3"),
                "expected `{expected}` and `B3`, got: {err}"
            );
        }
    }

    /// A range names the fold (B2) and the integral (B4) it will belong to.
    #[test]
    fn a_range_is_refused_naming_the_fold_and_the_integral() {
        let err = refusal(quote! { || integral(0.0..1.0, X) });
        assert!(
            err.contains("a range") && err.contains("B2") && err.contains("B4"),
            "got: {err}"
        );
    }

    /// Every other Rust expression is refused at the token, named, with
    /// nothing passed through.
    #[test]
    fn any_other_expression_is_refused_at_parse() {
        for other in [
            quote! { || X as f32 },
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

    /// The value of the literal body `|| <lit>`, or the parser's refusal.
    fn literal(lit: TokenStream) -> Result<f32, String> {
        let def = parse(quote! { || #lit }).map_err(|e| e.to_string())?;
        match &entry(&def).body {
            Expr::Literal(literal) => Ok(literal.value),
            other => panic!("expected a literal, got {other:?}"),
        }
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

    /// An integer literal is exact or refused. Every integer up to 2²⁴ is an
    /// `f32`; 2²⁴ + 1 is the first that is not.
    #[test]
    fn an_integer_literal_is_exact_or_refused() {
        assert_eq!(literal(quote!(0)), Ok(0.0));
        assert_eq!(literal(quote!(16777216)), Ok(16_777_216.0));
        // One significant bit, however large: exactly an `f32`.
        assert_eq!(literal(quote!(1099511627776)), Ok(1_099_511_627_776.0));
        assert_eq!(literal(quote!(0x10)), Ok(16.0));

        for inexact in [
            quote!(16777217),
            quote!(4294967295),
            quote!(1000000000000000000000000000000000000000000),
        ] {
            let err = literal(inexact).expect_err("an f32 cannot hold it");
            assert!(
                err.contains("not exactly representable as an `f32`"),
                "got: {err}"
            );
        }
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

    /// An item the block has no meaning for names the phase that brings it.
    #[test]
    fn a_struct_item_is_refused_naming_b3() {
        let err = refusal(quote! {
            struct Row { x: f32 }
            pub fn f() -> f32 { X }
        });
        assert!(
            err.contains("a `struct`") && err.contains("B3"),
            "got: {err}"
        );
    }

    /// A `fn` signature carries nothing the language cannot honor.
    #[test]
    fn a_fn_signature_is_plain() {
        let cases: [(TokenStream, &str); 6] = [
            (quote! { pub fn f<const N: usize>() -> f32 { X } }, "B3"),
            (quote! { pub fn f<T>() -> f32 { X } }, "B3"),
            (quote! { pub fn f() { X } }, "declares no return type"),
            (quote! { pub fn f(mut x: f32) -> f32 { x } }, "plain name"),
            (
                quote! { pub fn f(g: impl Fn(f32) -> f32) -> f32 { X } },
                "B4",
            ),
            (quote! { const fn f() -> f32 { X } }, "`const fn`"),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }
}
