//! # Expression Fold
//!
//! A catamorphism (fold) over the expression AST.
//!
//! ## Categorical Perspective
//!
//! All compiler phases perform the same structural recursion over Expr:
//! - **sema**: Expr → () (validate, build symbol table)
//! - **optimize**: Expr → Expr (algebraic simplification)
//! - **annotate**: Expr → AnnotatedExpr (assign binding indices)
//! - **codegen**: AnnotatedExpr → TokenStream (emit code)
//!
//! This is the classic "catamorphism" pattern from category theory.
//! Each phase defines how to handle each node type, and the traversal
//! logic is factored out into a generic fold.
//!
//! ## Design
//!
//! The `ExprFold` trait defines a transformation `Expr → T` where T is
//! the output type. Implementors define how to transform each node type.
//! The fold machinery handles the recursion.
//!
//! For phases that need state (like sema's symbol table), the trait
//! methods take `&mut self`.

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, LiteralExpr, MethodCallExpr, Stmt,
    UnaryExpr, UnaryOp,
};
use syn::Ident;

/// A fold (catamorphism) over the expression AST.
///
/// Implementors define how to transform each node type. The default
/// implementations handle the structural recursion, calling the
/// appropriate methods for children.
///
/// ## Type Parameter
///
/// - `T`: The output type of the fold (e.g., `Expr`, `TokenStream`, `()`)
pub trait ExprFold {
    /// The output type of the fold.
    type Output;

    // ========================================================================
    // Leaf Nodes
    // ========================================================================

    /// Transform an identifier reference.
    fn fold_ident(&mut self, name: &Ident) -> Self::Output;

    /// Transform a literal value.
    fn fold_literal(&mut self, lit: &syn::Lit) -> Self::Output;

    // ========================================================================
    // Composite Nodes (default implementations recurse)
    // ========================================================================

    /// Transform a binary expression.
    ///
    /// Default: fold both children, then combine.
    fn fold_binary(&mut self, op: BinaryOp, lhs: Self::Output, rhs: Self::Output) -> Self::Output;

    /// Transform a unary expression.
    ///
    /// Default: fold child, then transform.
    fn fold_unary(&mut self, op: UnaryOp, operand: Self::Output) -> Self::Output;

    /// Transform a method call.
    fn fold_method_call(
        &mut self,
        receiver: Self::Output,
        method: &Ident,
        args: Vec<Self::Output>,
    ) -> Self::Output;

    /// Transform a free function call.
    fn fold_call(&mut self, func: &Ident, args: Vec<Self::Output>) -> Self::Output;

    /// Transform a parenthesized expression.
    ///
    /// Default: unwrap (parens are just grouping).
    fn fold_paren(&mut self, inner: Self::Output) -> Self::Output {
        inner
    }

    /// Transform a block expression.
    fn fold_block(
        &mut self,
        stmts: Vec<Self::Output>,
        final_expr: Option<Self::Output>,
    ) -> Self::Output;

    /// Transform a tuple expression.
    fn fold_tuple(&mut self, elems: Vec<Self::Output>) -> Self::Output;

    /// Transform a let statement's initializer.
    fn fold_let(&mut self, name: &Ident, init: Self::Output) -> Self::Output;

    /// Transform a verbatim expression (pass-through).
    fn fold_verbatim(&mut self, expr: &syn::Expr) -> Self::Output;
}

/// Perform a fold over an expression tree.
///
/// This drives the recursion, calling the appropriate trait methods.
pub fn fold_expr<F: ExprFold>(folder: &mut F, expr: &Expr) -> F::Output {
    match expr {
        Expr::Ident(ident) => folder.fold_ident(&ident.name),
        Expr::Literal(lit) => folder.fold_literal(&lit.lit),
        Expr::Binary(binary) => {
            let lhs = fold_expr(folder, &binary.lhs);
            let rhs = fold_expr(folder, &binary.rhs);
            folder.fold_binary(binary.op, lhs, rhs)
        }
        Expr::Unary(unary) => {
            let operand = fold_expr(folder, &unary.operand);
            folder.fold_unary(unary.op, operand)
        }
        Expr::MethodCall(call) => {
            let receiver = fold_expr(folder, &call.receiver);
            let args: Vec<_> = call.args.iter().map(|a| fold_expr(folder, a)).collect();
            folder.fold_method_call(receiver, &call.method, args)
        }
        Expr::Call(call) => {
            let args: Vec<_> = call.args.iter().map(|a| fold_expr(folder, a)).collect();
            folder.fold_call(&call.func, args)
        }
        Expr::Paren(inner) => {
            let inner_out = fold_expr(folder, inner);
            folder.fold_paren(inner_out)
        }
        Expr::Block(block) => {
            let stmts: Vec<_> = block
                .stmts
                .iter()
                .filter_map(|stmt| match stmt {
                    Stmt::Let(let_stmt) => {
                        let init = fold_expr(folder, &let_stmt.init);
                        Some(folder.fold_let(&let_stmt.name, init))
                    }
                    Stmt::Expr(e) => Some(fold_expr(folder, e)),
                })
                .collect();
            let final_expr = block.expr.as_ref().map(|e| fold_expr(folder, e));
            folder.fold_block(stmts, final_expr)
        }
        Expr::Tuple(tuple) => {
            let elems: Vec<_> = tuple.elems.iter().map(|e| fold_expr(folder, e)).collect();
            folder.fold_tuple(elems)
        }
        Expr::Verbatim(syn_expr) => folder.fold_verbatim(syn_expr),
    }
}

// ============================================================================
// Example: Identity Fold (for testing)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A fold that counts nodes (for testing the traversal).
    struct NodeCounter {
        count: usize,
    }

    impl ExprFold for NodeCounter {
        type Output = ();

        fn fold_ident(&mut self, _name: &Ident) -> () {
            self.count += 1;
        }

        fn fold_literal(&mut self, _lit: &syn::Lit) -> () {
            self.count += 1;
        }

        fn fold_binary(&mut self, _op: BinaryOp, _lhs: (), _rhs: ()) -> () {
            self.count += 1;
        }

        fn fold_unary(&mut self, _op: UnaryOp, _operand: ()) -> () {
            self.count += 1;
        }

        fn fold_method_call(&mut self, _receiver: (), _method: &Ident, _args: Vec<()>) -> () {
            self.count += 1;
        }

        fn fold_call(&mut self, _func: &Ident, _args: Vec<()>) -> () {
            self.count += 1;
        }

        fn fold_block(&mut self, _stmts: Vec<()>, _final_expr: Option<()>) -> () {
            self.count += 1;
        }

        fn fold_tuple(&mut self, _elems: Vec<()>) -> () {
            self.count += 1;
        }

        fn fold_let(&mut self, _name: &Ident, _init: ()) -> () {
            self.count += 1;
        }

        fn fold_verbatim(&mut self, _expr: &syn::Expr) -> () {
            self.count += 1;
        }
    }
}
