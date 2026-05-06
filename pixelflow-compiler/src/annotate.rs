//! # Expression Annotation Pass
//!
//! Transforms the raw AST into an annotated form where literals have their
//! Var binding indices resolved. This is a pure functional pass.
//!
//! ## Design
//!
//! The annotation pass threads context through return values (functional state):
//! ```text
//! annotate(expr, ctx) -> (AnnotatedExpr, ctx')
//! ```
//!
//! This avoids mutation while still tracking state (literal counter).
//!
//! ## Why Annotate?
//!
//! In Jet domains (Jet3, Jet2), literals cannot be inlined as actual Jet values
//! because they'd break the ZST expression tree. Instead, literals become
//! `Var<N>` references bound via `Let`. The annotation pass assigns each
//! literal its Var index.

use crate::ast::{BinaryOp, BlockExpr, Expr, IdentExpr, Stmt, UnaryOp};
use proc_macro2::Span;
use syn::{Ident, Lit, Type};

/// Context threaded through the annotation pass.
#[derive(Clone)]
pub struct AnnotationCtx {
    /// Next literal index to assign (0, 1, 2, ...).
    /// These are "collection order" indices, inverted at emit time.
    pub next_literal: usize,
}

impl AnnotationCtx {
    pub fn new() -> Self {
        Self { next_literal: 0 }
    }

    /// Total number of literals collected.
    pub fn literal_count(&self) -> usize {
        self.next_literal
    }
}

/// Annotated expression tree.
///
/// Mirrors `Expr` but literals carry their resolved Var index.
#[derive(Debug, Clone)]
pub enum AnnotatedExpr {
    Ident(IdentExpr),
    Literal(AnnotatedLiteral),
    Binary(AnnotatedBinary),
    Unary(AnnotatedUnary),
    MethodCall(AnnotatedMethodCall),
    Call(AnnotatedCall),
    Block(AnnotatedBlock),
    Tuple(AnnotatedTuple),
    Paren(Box<AnnotatedExpr>),
    Verbatim(syn::Expr),
}

#[derive(Debug, Clone)]
pub struct AnnotatedTuple {
    pub elems: Vec<AnnotatedExpr>,
    pub span: Span,
}

/// A literal with its binding information.
#[derive(Debug, Clone)]
pub struct AnnotatedLiteral {
    pub lit: Lit,
    pub span: Span,
    /// If Some(idx), this literal should be emitted as `Var::<N{idx}>::new()`.
    /// If None, emit the literal directly.
    pub var_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct AnnotatedBinary {
    pub op: BinaryOp,
    pub lhs: Box<AnnotatedExpr>,
    pub rhs: Box<AnnotatedExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct AnnotatedUnary {
    pub op: UnaryOp,
    pub operand: Box<AnnotatedExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct AnnotatedMethodCall {
    pub receiver: Box<AnnotatedExpr>,
    pub method: Ident,
    pub args: Vec<AnnotatedExpr>,
    pub span: Span,
}

/// A free function call (V(m), DX(expr), etc.).
#[derive(Debug, Clone)]
pub struct AnnotatedCall {
    pub func: Ident,
    pub args: Vec<AnnotatedExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct AnnotatedBlock {
    pub stmts: Vec<AnnotatedStmt>,
    pub expr: Option<Box<AnnotatedExpr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum AnnotatedStmt {
    Let(AnnotatedLet),
    Expr(AnnotatedExpr),
}

#[derive(Debug, Clone)]
pub struct AnnotatedLet {
    pub name: Ident,
    pub ty: Option<Type>,
    pub init: AnnotatedExpr,
    pub span: Span,
}

/// Collected literal for Let binding generation.
#[derive(Debug, Clone)]
pub struct CollectedLiteral {
    pub index: usize,
    pub lit: Lit,
}

/// Result of annotation: the annotated tree plus collected literals.
pub struct AnnotationResult {
    pub expr: AnnotatedExpr,
    pub literals: Vec<CollectedLiteral>,
}

/// Annotate an expression tree, resolving literal Var indices.
///
/// This is a pure function - context flows through return values.
pub fn annotate(
    expr: &Expr,
    ctx: AnnotationCtx,
) -> (AnnotatedExpr, AnnotationCtx, Vec<CollectedLiteral>) {
    let mut literals = Vec::new();
    let (annotated, final_ctx) = annotate_expr(expr, ctx, &mut literals);
    (annotated, final_ctx, literals)
}

fn annotate_expr(
    expr: &Expr,
    ctx: AnnotationCtx,
    literals: &mut Vec<CollectedLiteral>,
) -> (AnnotatedExpr, AnnotationCtx) {
    match expr {
        Expr::Ident(ident) => (AnnotatedExpr::Ident(ident.clone()), ctx),

        Expr::Literal(lit_expr) => {
            // Always assign a collection-order index to literals.
            // This makes literals into expression tree nodes (Var<N>), enabling:
            // 1. `0.1 + expr` to work (both sides are expression trees)
            // 2. FMA pattern matching (MulAdd recognition)
            // 3. Uniform treatment across Field and Jet domains
            //
            // The actual Var<N> index is computed at emit time by inverting:
            // Var index = (literal_count - 1) - collection_index
            // This ensures the last literal bound (innermost Let) is at N0.
            let collection_index = ctx.next_literal;
            literals.push(CollectedLiteral {
                index: collection_index,
                lit: lit_expr.lit.clone(),
            });
            let new_ctx = AnnotationCtx {
                next_literal: ctx.next_literal + 1,
                ..ctx
            };
            (
                AnnotatedExpr::Literal(AnnotatedLiteral {
                    lit: lit_expr.lit.clone(),
                    span: lit_expr.span,
                    var_index: Some(collection_index),
                }),
                new_ctx,
            )
        }

        Expr::Binary(binary) => {
            let (lhs, ctx1) = annotate_expr(&binary.lhs, ctx, literals);
            let (rhs, ctx2) = annotate_expr(&binary.rhs, ctx1, literals);
            (
                AnnotatedExpr::Binary(AnnotatedBinary {
                    op: binary.op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span: binary.span,
                }),
                ctx2,
            )
        }

        Expr::Unary(unary) => {
            let (operand, ctx1) = annotate_expr(&unary.operand, ctx, literals);
            (
                AnnotatedExpr::Unary(AnnotatedUnary {
                    op: unary.op,
                    operand: Box::new(operand),
                    span: unary.span,
                }),
                ctx1,
            )
        }

        Expr::MethodCall(call) => {
            let (receiver, mut ctx1) = annotate_expr(&call.receiver, ctx, literals);
            let mut args = Vec::with_capacity(call.args.len());
            for arg in &call.args {
                let (annotated_arg, new_ctx) = annotate_expr(arg, ctx1, literals);
                args.push(annotated_arg);
                ctx1 = new_ctx;
            }
            (
                AnnotatedExpr::MethodCall(AnnotatedMethodCall {
                    receiver: Box::new(receiver),
                    method: call.method.clone(),
                    args,
                    span: call.span,
                }),
                ctx1,
            )
        }

        Expr::Tuple(tuple) => {
            let mut ctx1 = ctx;
            let mut elems = Vec::with_capacity(tuple.elems.len());
            for elem in &tuple.elems {
                let (annotated_elem, new_ctx) = annotate_expr(elem, ctx1, literals);
                elems.push(annotated_elem);
                ctx1 = new_ctx;
            }
            (
                AnnotatedExpr::Tuple(AnnotatedTuple {
                    elems,
                    span: tuple.span,
                }),
                ctx1,
            )
        }

        Expr::Block(block) => {
            let (annotated_block, ctx1) = annotate_block(block, ctx, literals);
            (AnnotatedExpr::Block(annotated_block), ctx1)
        }

        Expr::Paren(inner) => {
            let (annotated, ctx1) = annotate_expr(inner, ctx, literals);
            (AnnotatedExpr::Paren(Box::new(annotated)), ctx1)
        }

        Expr::Call(call) => {
            // Annotate all arguments (this is where manifold param rewriting happens)
            let mut ctx1 = ctx;
            let mut args = Vec::with_capacity(call.args.len());
            for arg in &call.args {
                let (annotated_arg, new_ctx) = annotate_expr(arg, ctx1, literals);
                args.push(annotated_arg);
                ctx1 = new_ctx;
            }
            (
                AnnotatedExpr::Call(AnnotatedCall {
                    func: call.func.clone(),
                    args,
                    span: call.span,
                }),
                ctx1,
            )
        }

        Expr::Verbatim(syn_expr) => (AnnotatedExpr::Verbatim(syn_expr.clone()), ctx),
    }
}

fn annotate_block(
    block: &BlockExpr,
    ctx: AnnotationCtx,
    literals: &mut Vec<CollectedLiteral>,
) -> (AnnotatedBlock, AnnotationCtx) {
    let mut current_ctx = ctx;
    let mut stmts = Vec::with_capacity(block.stmts.len());

    for stmt in &block.stmts {
        let (annotated_stmt, new_ctx) = annotate_stmt(stmt, current_ctx, literals);
        stmts.push(annotated_stmt);
        current_ctx = new_ctx;
    }

    let (final_expr, final_ctx) = match &block.expr {
        Some(e) => {
            let (annotated, ctx1) = annotate_expr(e, current_ctx, literals);
            (Some(Box::new(annotated)), ctx1)
        }
        None => (None, current_ctx),
    };

    (
        AnnotatedBlock {
            stmts,
            expr: final_expr,
            span: block.span,
        },
        final_ctx,
    )
}

fn annotate_stmt(
    stmt: &Stmt,
    ctx: AnnotationCtx,
    literals: &mut Vec<CollectedLiteral>,
) -> (AnnotatedStmt, AnnotationCtx) {
    match stmt {
        Stmt::Let(let_stmt) => {
            let (init, ctx1) = annotate_expr(&let_stmt.init, ctx, literals);
            (
                AnnotatedStmt::Let(AnnotatedLet {
                    name: let_stmt.name.clone(),
                    ty: let_stmt.ty.clone(),
                    init,
                    span: let_stmt.span,
                }),
                ctx1,
            )
        }
        Stmt::Expr(expr) => {
            let (annotated, ctx1) = annotate_expr(expr, ctx, literals);
            (AnnotatedStmt::Expr(annotated), ctx1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use quote::quote;

    #[test]
    fn annotate_no_literals() {
        let input = quote! { || X + Y };
        let kernel = parse(input).unwrap();
        let ctx = AnnotationCtx::new();
        let (_, _, literals) = annotate(&kernel.body, ctx);
        assert!(literals.is_empty());
    }

    #[test]
    fn annotate_single_literal() {
        let input = quote! { || X + 1.0 };
        let kernel = parse(input).unwrap();
        let ctx = AnnotationCtx::new();
        let (annotated, _, literals) = annotate(&kernel.body, ctx);

        // Literals are always collected now (for both Field and Jet modes)
        assert_eq!(literals.len(), 1);
        assert_eq!(literals[0].index, 0); // collection order index

        // Check the annotated tree has the var_index
        if let AnnotatedExpr::Binary(binary) = annotated {
            if let AnnotatedExpr::Literal(lit) = &*binary.rhs {
                assert_eq!(lit.var_index, Some(0));
            } else {
                panic!("expected literal");
            }
        } else {
            panic!("expected binary");
        }
    }

    #[test]
    fn annotate_multiple_literals() {
        let input = quote! { || 1.0 / X.sqrt() + 2.0 };
        let kernel = parse(input).unwrap();
        let ctx = AnnotationCtx::new();
        let (_, _, literals) = annotate(&kernel.body, ctx);

        assert_eq!(literals.len(), 2);
        // Collection-order indices: first literal → 0, second → 1
        // The actual Var<N> index is computed at emit time by inverting
        assert_eq!(literals[0].index, 0); // collection order
        assert_eq!(literals[1].index, 1); // collection order
    }
}
