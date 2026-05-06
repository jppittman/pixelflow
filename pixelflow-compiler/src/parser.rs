//! # Parser
//!
//! Parses the kernel DSL from token stream to AST.
//!
//! ## Grammar
//!
//! ```text
//! kernel     ::= '|' params '|' expr
//! params     ::= (param (',' param)*)?
//! param      ::= IDENT ':' type
//!
//! expr       ::= binary
//! binary     ::= unary (('+' | '-' | '*' | '/' | '%') unary)*
//! unary      ::= ('-' | '!')? postfix
//! postfix    ::= primary ('.' method_call)*
//! method_call::= IDENT '(' args? ')'
//! primary    ::= IDENT | LITERAL | '(' expr ')' | block
//! block      ::= '{' stmt* expr? '}'
//! stmt       ::= 'let' IDENT (':' type)? '=' expr ';'
//!              | expr ';'
//! ```
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, KernelDef, LetStmt, LiteralExpr,
    MethodCallExpr, Param, ParamKind, Stmt, StructDecl, TupleExpr, UnaryExpr, UnaryOp,
};
use proc_macro2::{Span, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::{Pat, Token, Type, Visibility};

/// Parse kernel input from token stream.
pub fn parse(input: TokenStream) -> syn::Result<KernelDef> {
    syn::parse2(input)
}

/// Parser state for the closure-like syntax.
struct KernelParser;

impl Parse for KernelDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Try to parse optional struct declaration: [visibility] struct Name =
        let struct_decl = parse_struct_decl(input)?;

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

                // Detect `kernel` keyword as manifold parameter marker
                let kind = if is_kernel_keyword(&ty) {
                    ParamKind::Manifold
                } else {
                    ParamKind::Scalar(ty)
                };

                params.push(Param { name: ident, kind });

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

        // Parse optional domain and return types:
        // - `DomainType -> OutputType` (both)
        // - `-> OutputType` (just output)
        // - (nothing) (neither)
        let (domain_ty, return_ty) = parse_type_annotations(input)?;

        // Parse the body expression
        let syn_expr: syn::Expr = input.parse()?;
        let body = convert_expr(syn_expr)?;

        Ok(KernelDef {
            struct_decl,
            params,
            domain_ty,
            return_ty,
            body,
        })
    }
}

/// Try to parse an optional struct declaration.
///
/// Grammar: [visibility] 'struct' IDENT '='
///
/// Returns None if the input doesn't start with a struct declaration.
fn parse_struct_decl(input: ParseStream) -> syn::Result<Option<StructDecl>> {
    // Check if we're looking at a struct declaration
    // This could be: `pub struct Foo =`, `pub(crate) struct Foo =`, or `struct Foo =`

    // First, try to peek ahead to see if there's a `struct` keyword
    // We need to handle visibility first since it can consume tokens

    // Use a fork to speculatively parse
    let fork = input.fork();

    // Try to parse visibility (this handles pub, pub(crate), etc.)
    let visibility: Visibility = fork.parse()?;

    // Check for `struct` keyword
    if fork.peek(Token![struct]) {
        // Commit to parsing the struct declaration
        input.parse::<Visibility>()?; // consume visibility in real stream
        input.parse::<Token![struct]>()?;

        let name: syn::Ident = input.parse()?;
        input.parse::<Token![=]>()?;

        Ok(Some(StructDecl { visibility, name }))
    } else {
        // Not a struct declaration, leave input unchanged
        Ok(None)
    }
}

/// Parse optional domain and return type annotations.
///
/// Grammar:
/// - `DomainType -> OutputType` → (Some(domain), Some(output))
/// - `-> OutputType` → (None, Some(output))
/// - (nothing) → (None, None)
///
/// This allows syntax like `Field -> Discrete` where Field is the domain
/// type (used for coordinates) and Discrete is the output type.
fn parse_type_annotations(input: ParseStream) -> syn::Result<(Option<Type>, Option<Type>)> {
    // Check if we have `->` directly (no domain type)
    if input.peek(Token![->]) {
        input.parse::<Token![->]>()?;
        let output_ty = input.parse::<Type>()?;
        return Ok((None, Some(output_ty)));
    }

    // Try to parse a type followed by `->`
    // Use a fork to speculatively check if this is `Type ->`
    let fork = input.fork();

    // Try to parse a type
    if let Ok(ty) = fork.parse::<Type>() {
        // Check if followed by `->`
        if fork.peek(Token![->]) {
            // Yes! This is `DomainType -> OutputType`
            // Consume from the real stream
            let domain_ty = input.parse::<Type>()?;
            input.parse::<Token![->]>()?;
            let output_ty = input.parse::<Type>()?;
            return Ok((Some(domain_ty), Some(output_ty)));
        }
    }

    // No type annotations
    Ok((None, None))
}

/// Check if a type is the `kernel` keyword (manifold parameter marker).
fn is_kernel_keyword(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        type_path.path.is_ident("kernel")
    } else {
        false
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
            span: expr_lit.lit.span(),
            lit: expr_lit.lit,
        })),

        syn::Expr::Binary(expr_binary) => {
            let op = BinaryOp::from_syn(&expr_binary.op).ok_or_else(|| {
                let op_str = quote::quote!(#expr_binary.op).to_string();
                syn::Error::new_spanned(
                    &expr_binary.op,
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
                    &expr_unary.op,
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
    let mut stmts = Vec::new();
    let mut final_expr = None;

    for (i, stmt) in block.stmts.iter().enumerate() {
        let is_last = i == block.stmts.len() - 1;

        match stmt {
            syn::Stmt::Local(local) => {
                // let binding
                let name = match &local.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                    Pat::Type(pat_type) => match &*pat_type.pat {
                        Pat::Ident(pat_ident) => pat_ident.ident.clone(),
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

                let init_expr = convert_expr((*init.expr).clone())?;

                stmts.push(Stmt::Let(LetStmt {
                    name,
                    ty,
                    init: init_expr,
                    span: Span::call_site(),
                }));
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

    #[test]
    fn parse_return_type() {
        let input = quote! { |cx: f32| -> Jet3 X - cx };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.return_ty.is_some());
        // Verify the return type is "Jet3"
        let ty = kernel.return_ty.unwrap();
        if let syn::Type::Path(type_path) = ty {
            assert_eq!(type_path.path.segments[0].ident.to_string(), "Jet3");
        } else {
            panic!("expected path type");
        }
    }

    #[test]
    fn parse_no_return_type() {
        let input = quote! { |cx: f32| X - cx };
        let kernel = parse(input).unwrap();
        assert!(kernel.return_ty.is_none());
        assert!(kernel.domain_ty.is_none());
    }

    #[test]
    fn parse_domain_and_output_type() {
        // Field -> Discrete syntax: Field is domain, Discrete is output
        let input = quote! { |cx: f32| Field -> Discrete X - cx };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 1);

        // Verify domain type is "Field"
        let domain = kernel.domain_ty.expect("expected domain type");
        if let syn::Type::Path(type_path) = domain {
            assert_eq!(type_path.path.segments[0].ident.to_string(), "Field");
        } else {
            panic!("expected path type for domain");
        }

        // Verify output type is "Discrete"
        let output = kernel.return_ty.expect("expected return type");
        if let syn::Type::Path(type_path) = output {
            assert_eq!(type_path.path.segments[0].ident.to_string(), "Discrete");
        } else {
            panic!("expected path type for output");
        }
    }

    #[test]
    fn parse_manifold_param() {
        // `kernel` keyword marks a manifold parameter
        let input = quote! { |inner: kernel, r: f32| inner - r };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 2);

        // First param should be Manifold
        assert!(
            matches!(kernel.params[0].kind, ParamKind::Manifold),
            "expected inner to be Manifold param"
        );
        assert_eq!(kernel.params[0].name.to_string(), "inner");

        // Second param should be Scalar(f32)
        assert!(
            matches!(kernel.params[1].kind, ParamKind::Scalar(_)),
            "expected r to be Scalar param"
        );
        assert_eq!(kernel.params[1].name.to_string(), "r");
    }

    #[test]
    fn parse_multiple_manifold_params() {
        let input = quote! { |a: kernel, b: kernel| a + b };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 2);
        assert!(matches!(kernel.params[0].kind, ParamKind::Manifold));
        assert!(matches!(kernel.params[1].kind, ParamKind::Manifold));
    }

    #[test]
    fn parse_domain_with_block() {
        // This is the syntax that's failing
        let input = quote! {
            |x: f32| Field -> Discrete {
                let a = X + x;
                a
            }
        };
        let kernel = parse(input).unwrap();

        eprintln!("Domain: {:?}", kernel.domain_ty);
        eprintln!("Return: {:?}", kernel.return_ty);
        eprintln!("Body: {:?}", kernel.body);

        // Verify domain is Field
        assert!(kernel.domain_ty.is_some(), "expected domain type");

        // Verify return is Discrete
        assert!(kernel.return_ty.is_some(), "expected return type");

        // The body should be a block with let binding
        match kernel.body {
            Expr::Block(block) => {
                eprintln!("Block stmts: {:?}", block.stmts);
                eprintln!("Block expr: {:?}", block.expr);
                assert_eq!(block.stmts.len(), 1, "expected 1 let statement");
                assert!(block.expr.is_some(), "expected final expression");
            }
            other => panic!("expected block expression, got {:?}", other),
        }
    }
}
