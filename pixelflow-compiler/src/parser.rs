//! # Parser
//!
//! Parses the kernel DSL from token stream to AST.
//!
//! ## Grammar
//!
//! The parameter list is parsed by hand; the body is parsed by syn as a Rust
//! expression, so precedence and associativity are Rust's. Of that syntax,
//! this is the kernel language — what every stage accepts:
//!
//! ```text
//! kernel  ::= '|' params '|' expr
//! params  ::= (param (',' param)* ','?)?
//! param   ::= IDENT ':' type
//!
//! expr    ::= expr binop expr
//!           | '-' expr
//!           | expr '.' METHOD '(' (expr (',' expr)*)? ')'
//!           | PROJECTION '(' expr ')'
//!           | '(' expr ')'
//!           | block
//!           | IDENT                    -- X, Y, a parameter, or a `let` in scope
//!           | LITERAL                  -- an integer or a float, as its f32
//! binop   ::= '+' | '-' | '*' | '/'
//!           | '<' | '<=' | '>' | '>=' | '==' | '!='    -- a comparison: a mask
//!           | '&' | '|'                                -- masks combine
//! block   ::= '{' stmt* expr '}'
//! stmt    ::= 'let' IDENT (':' type)? '=' expr ';'   -- the type is not checked
//!           | expr ';'
//!
//! METHOD     -- an `OpKind` method, a `LIBRARY_METHODS` composition, or `clone`
//! PROJECTION -- V, DX, DY, DXX, DXY, DYY
//! ```
//!
//! A `let` is scoped as Rust scopes it (`crate::symbol`), and neither a `let`
//! nor a parameter may be named X or Y.
//!
//! Refused here, with a span: a `let` whose pattern is not a plain name
//! (`mut`, `ref`, `@`, or destructuring), a `let` without an initializer,
//! `let … else`, an item or a macro inside a
//! block, an operator that is neither in the table above nor `%` or `!`, a
//! literal that is not a number, a literal suffixed with a type other than
//! `f32`, an integer an `f32` does not hold exactly, and a float past `f32`'s
//! range.
//!
//! Parsed, and refused by a later stage: an unbound or retired name, `let X`,
//! an unknown method or a known one at the wrong arity (`sema`); and `%`, `!`,
//! a tuple, `.at()`, `.constant()`, `.collapse()`, an unknown projection, a
//! block with no
//! final expression, and any other Rust syntax, which the parser keeps whole
//! as [`Expr::Verbatim`] so the refusal can name it (lowering).
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, KernelDef, LetStmt, LiteralExpr,
    MethodCallExpr, Param, Stmt, TupleExpr, UnaryExpr, UnaryOp,
};
use proc_macro2::{Span, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::{Pat, Token, Type};

/// Parse kernel input from token stream.
pub fn parse(input: TokenStream) -> syn::Result<KernelDef> {
    syn::parse2(input)
}

impl Parse for KernelDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: |param: Type, ...| body
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

        Ok(KernelDef { params, body })
    }
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
            // Complex path - pass through verbatim
            Ok(Expr::Verbatim(syn::Expr::Path(expr_path)))
        }

        syn::Expr::Lit(expr_lit) => Ok(Expr::Literal(LiteralExpr {
            value: literal_value(&expr_lit.lit)?,
            span: expr_lit.lit.span(),
        })),

        syn::Expr::Binary(expr_binary) => {
            let op = BinaryOp::from_syn(&expr_binary.op).ok_or_else(|| {
                let op_str = quote::quote!(#expr_binary.op).to_string();
                syn::Error::new_spanned(
                    expr_binary.op,
                    format!(
                        "unsupported binary operator `{}`\n\
                         \n\
                         note: the kernel! macro only supports these binary operators:\n\
                         note:   arithmetic: + - * / %\n\
                         note:   comparison: < <= > >= == !=\n\
                         note:   logical: & |\n\
                         \n\
                         help: if you need bitwise operations or other operators, extract them to a helper function",
                        op_str
                    ),
                )
            })?;
            let lhs = convert_expr(*expr_binary.left)?;
            let rhs = convert_expr(*expr_binary.right)?;
            Ok(Expr::Binary(BinaryExpr {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::call_site(),
            }))
        }

        syn::Expr::Unary(expr_unary) => {
            let op = UnaryOp::from_syn(&expr_unary.op).ok_or_else(|| {
                let op_str = quote::quote!(#expr_unary.op).to_string();
                syn::Error::new_spanned(
                    expr_unary.op,
                    format!(
                        "unsupported unary operator `{}`\n\
                         \n\
                         note: the kernel! macro supports these unary operators:\n\
                         note:   - (negation)   example: -X\n\
                         note:   ! (logical not) example: !condition\n\
                         \n\
                         help: for other unary operations, use method calls like .abs() or helper functions",
                        op_str
                    ),
                )
            })?;
            let operand = convert_expr(*expr_unary.expr)?;
            Ok(Expr::Unary(UnaryExpr {
                op,
                operand: Box::new(operand),
                span: Span::call_site(),
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
                method: expr_method.method,
                args,
                span: Span::call_site(),
            }))
        }

        syn::Expr::Call(expr_call) => {
            // Free function call: V(m), DX(expr), etc.
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
                        func,
                        args,
                        span: Span::call_site(),
                    }));
                }
            }
            // Complex call (qualified path, etc.) - pass through verbatim
            Ok(Expr::Verbatim(syn::Expr::Call(expr_call)))
        }

        syn::Expr::Paren(expr_paren) => {
            let inner = convert_expr(*expr_paren.expr)?;
            Ok(Expr::Paren(Box::new(inner)))
        }

        syn::Expr::Tuple(expr_tuple) => {
            let elems = expr_tuple
                .elems
                .into_iter()
                .map(convert_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(Expr::Tuple(TupleExpr {
                elems,
                span: Span::call_site(),
            }))
        }

        syn::Expr::Block(expr_block) => {
            let block = convert_block(expr_block.block)?;
            Ok(Expr::Block(block))
        }

        // Anything else - pass through verbatim for codegen to handle
        other => Ok(Expr::Verbatim(other)),
    }
}

/// Convert a syn::Block to our BlockExpr.
fn convert_block(block: syn::Block) -> syn::Result<BlockExpr> {
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
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &local.pat,
                                "complex pattern not supported in let binding\n\
                                 \n\
                                 note: kernel! only supports simple identifier patterns\n\
                                 \n\
                                 help: use a simple identifier like:\n\
                                 help:   let dx = X - cx;\n\
                                 help:   let result: f32 = calculation;",
                            ));
                        }
                    },
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "complex pattern not supported in let binding\n\
                             \n\
                             note: kernel! only supports simple identifier patterns\n\
                             \n\
                             help: destructuring, tuples, and other patterns are not allowed\n\
                             help: use a simple identifier like:\n\
                             help:   let value = expression;",
                        ));
                    }
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
                    name,
                    ty,
                    init: init_expr,
                    span: Span::call_site(),
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
                    "item definitions are not allowed inside kernel! blocks\n\
                     \n\
                     note: kernel! blocks can only contain let bindings and expressions\n\
                     \n\
                     help: define functions, structs, and other items outside the kernel! macro:\n\
                     help:   fn helper(x: f32) -> f32 { x * 2.0 }\n\
                     help:   let my_kernel = kernel!(|| helper(X));",
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
        span: Span::call_site(),
    })
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
             note: every value in a kernel body is an `f32`; a mask comes from a comparison",
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

/// The plain name a `let` binds. `mut`, `ref` and an `@` subpattern are
/// refused rather than dropped: nothing in a kernel body assigns, borrows or
/// matches, so each would be a promise the body cannot keep, and a subpattern
/// can refute a `let`, which rustc refuses too.
fn plain_name(pat_ident: &syn::PatIdent, pat: &Pat) -> syn::Result<syn::Ident> {
    let decorated =
        pat_ident.by_ref.is_some() || pat_ident.mutability.is_some() || pat_ident.subpat.is_some();
    if decorated {
        return Err(syn::Error::new_spanned(
            pat,
            "a `let` in a kernel body binds a plain name\n\
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
    use quote::quote;

    #[test]
    fn parse_simple_kernel() {
        let input = quote! { |r: f32| X * X + Y * Y - r };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 1);
        assert_eq!(kernel.params[0].name.to_string(), "r");
    }

    #[test]
    fn parse_empty_params() {
        let input = quote! { || X * X + Y * Y };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 0);
    }

    #[test]
    fn parse_multiple_params() {
        let input = quote! { |cx: f32, cy: f32, r: f32| X - cx };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 3);
    }

    #[test]
    fn parse_method_call() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let kernel = parse(input).unwrap();
        // Should successfully parse the .sqrt() method call
        match kernel.body {
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
        match kernel.body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 2); // two let statements
                assert!(block.expr.is_some()); // final expression
            }
            _ => panic!("expected block expression"),
        }
    }

    /// A parameter's declared type is a scalar type and nothing else: the
    /// `kernel` keyword that used to mark a manifold-typed slot is gone with
    /// the tier that spliced one, and kernels compose as `Kernel` values.
    #[test]
    fn parse_scalar_params_keep_their_declared_types() {
        let input = quote! { |cx: f32, n: i32| X * cx + n };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 2);
        assert_eq!(kernel.params[0].name.to_string(), "cx");
        assert_eq!(kernel.params[1].name.to_string(), "n");
        for (param, want) in kernel.params.iter().zip(["f32", "i32"]) {
            let syn::Type::Path(path) = &*param.ty else {
                panic!("expected a path type for {}", param.name);
            };
            assert_eq!(path.path.segments[0].ident.to_string(), want);
        }
    }

    #[test]
    fn parse_rejects_a_multi_segment_path_as_a_plain_identifier() {
        // A multi-segment path (qself-free but len() != 1) must fall through
        // to Verbatim, not be silently truncated to an Ident of its first
        // segment.
        let input = quote! { || std::f32::consts::PI };
        let kernel = parse(input).unwrap();
        assert!(
            matches!(kernel.body, Expr::Verbatim(_)),
            "expected Verbatim for a multi-segment path, got {:?}",
            kernel.body
        );
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
        match kernel.body {
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
        match kernel.body {
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
        match parse(quote! { || #lit }).map_err(|e| e.to_string())?.body {
            Expr::Literal(literal) => Ok(literal.value),
            other => panic!("expected a literal, got {other:?}"),
        }
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
        let err = parse(input).expect_err("the else branch cannot be honored");
        assert!(err.to_string().contains("`let ... else`"), "got: {err}");
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
            let err = parse(decorated).expect_err("the pattern is not a plain name");
            assert!(err.to_string().contains("plain name"), "got: {err}");
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
        assert_eq!(kernel.params.len(), 1);

        match kernel.body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 1, "expected 1 let statement");
                assert!(block.expr.is_some(), "expected final expression");
            }
            other => panic!("expected block expression, got {other:?}"),
        }
    }
}
