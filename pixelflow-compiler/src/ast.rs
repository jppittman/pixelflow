//! # Abstract Syntax Tree
//!
//! The AST represents the structure of a kernel expression after parsing.
//!
//! ## Design Philosophy
//!
//! The AST is a **source-level representation** that preserves the structure
//! the user wrote. It does NOT attempt to mirror PixelFlow's type-level AST
//! (the `Sqrt<Add<Mul<X,X>,...>>` trees) - that's what the generated code produces.
//!
//! The compiler's job is to transform this source AST into Rust code that
//! rebuilds the corresponding arena fragment at load time.
//!
//! ## AST Structure
//!
//! ```text
//! KernelDef
//!   ├── params: [(name, type), ...]    // Closure parameters (scalars)
//!   └── body: Expr                     // The kernel expression
//!
//! Expr
//!   ├── Ident(name)                    // Variable reference: X, cx, etc.
//!   ├── Literal(value)                 // Numeric literal: 1.0, 2.5
//!   ├── Binary(op, lhs, rhs)           // a + b, x * y
//!   ├── Unary(op, operand)             // -x
//!   ├── Call(method, receiver, args)   // x.sqrt(), a.max(b)
//!   ├── Block(stmts, expr)             // { let dx = ...; dx * dx }
//!   └── Paren(inner)                   // (a + b)
//! ```

use proc_macro2::Span;
use syn::{Ident, Type};

/// A complete kernel definition.
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// Parameters captured from the closure syntax.
    pub params: Vec<Param>,
    /// The kernel body expression.
    pub body: Expr,
}

/// A captured scalar parameter.
///
/// There is one parameter kind, because there is one thing a parameter can
/// be: a number folded into the arena when the builder runs. Kernels compose
/// as `Kernel` values (`Kernel::at`/`sum`/`select`/arithmetic), not by
/// splicing a manifold through a macro slot.
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name.
    pub name: Ident,
    /// The declared scalar type (`f32`, `i32`).
    pub ty: Box<Type>,
}

/// An expression in the kernel body.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A variable reference (X, Y, cx, etc.).
    Ident(IdentExpr),

    /// A numeric literal (1.0, 2.5f32, etc.).
    Literal(LiteralExpr),

    /// A binary operation (a + b, x * y, etc.).
    Binary(BinaryExpr),

    /// A unary operation (-x, !b).
    Unary(UnaryExpr),

    /// A method call (x.sqrt(), a.max(b), etc.).
    MethodCall(MethodCallExpr),

    /// A free function call (V(m), DX(expr), sin(x), etc.).
    Call(CallExpr),

    /// A block expression ({ let dx = ...; dx * dx }).
    Block(BlockExpr),

    /// A tuple expression: (a, b, c)
    Tuple(TupleExpr),

    /// A parenthesized expression ((a + b)).
    Paren(Box<Expr>),

    /// Passthrough for expressions we don't specially handle.
    /// The codegen phase will emit these verbatim.
    Verbatim(syn::Expr),
}

#[derive(Debug, Clone)]
pub struct TupleExpr {
    pub elems: Vec<Expr>,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// An identifier expression.
#[derive(Debug, Clone)]
pub struct IdentExpr {
    pub name: Ident,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// A literal expression.
#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub lit: syn::Lit,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// Binary operators we recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Comparison (for future use, currently handled via method calls)
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    // Boolean/bitwise operations
    BitAnd,
    BitOr,
}

/// A binary expression.
#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// A unary expression.
#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Box<Expr>,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// A method call expression.
#[derive(Debug, Clone)]
pub struct MethodCallExpr {
    /// The receiver (what the method is called on).
    pub receiver: Box<Expr>,
    /// The method name (sqrt, sin, max, etc.).
    pub method: Ident,
    /// Method arguments (empty for sqrt, one arg for max, etc.).
    pub args: Vec<Expr>,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// A free function call expression (V(m), DX(expr), etc.).
#[derive(Debug, Clone)]
pub struct CallExpr {
    /// The function being called (V, DX, DY, etc.).
    pub func: Ident,
    /// Function arguments.
    pub args: Vec<Expr>,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// A statement in a block.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// A let binding: `let dx = X - cx;`
    Let(Box<LetStmt>),
    /// An expression statement: `foo();`
    Expr(Expr),
}

/// A let statement.
#[derive(Debug, Clone)]
pub struct LetStmt {
    pub name: Ident,
    pub ty: Option<Type>,
    pub init: Expr,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

/// A block expression.
#[derive(Debug, Clone)]
pub struct BlockExpr {
    pub stmts: Vec<Stmt>,
    /// The final expression (if any).
    pub expr: Option<Box<Expr>>,
    // Kept for AST-node uniformity; not all node types' spans are read today.
    #[allow(dead_code)]
    pub span: Span,
}

impl BinaryOp {
    /// Convert from syn's BinOp.
    pub fn from_syn(op: &syn::BinOp) -> Option<Self> {
        match op {
            syn::BinOp::Add(_) => Some(BinaryOp::Add),
            syn::BinOp::Sub(_) => Some(BinaryOp::Sub),
            syn::BinOp::Mul(_) => Some(BinaryOp::Mul),
            syn::BinOp::Div(_) => Some(BinaryOp::Div),
            syn::BinOp::Rem(_) => Some(BinaryOp::Rem),
            syn::BinOp::Lt(_) => Some(BinaryOp::Lt),
            syn::BinOp::Le(_) => Some(BinaryOp::Le),
            syn::BinOp::Gt(_) => Some(BinaryOp::Gt),
            syn::BinOp::Ge(_) => Some(BinaryOp::Ge),
            syn::BinOp::Eq(_) => Some(BinaryOp::Eq),
            syn::BinOp::Ne(_) => Some(BinaryOp::Ne),
            syn::BinOp::BitAnd(_) => Some(BinaryOp::BitAnd),
            syn::BinOp::BitOr(_) => Some(BinaryOp::BitOr),
            _ => None,
        }
    }
}

impl UnaryOp {
    /// Convert from syn's UnOp.
    pub fn from_syn(op: &syn::UnOp) -> Option<Self> {
        match op {
            syn::UnOp::Neg(_) => Some(UnaryOp::Neg),
            syn::UnOp::Not(_) => Some(UnaryOp::Not),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_op_from_syn_maps_every_supported_syn_binop_to_its_own_variant() {
        let cases: &[(syn::BinOp, BinaryOp)] = &[
            (syn::parse_quote!(+), BinaryOp::Add),
            (syn::parse_quote!(-), BinaryOp::Sub),
            (syn::parse_quote!(*), BinaryOp::Mul),
            (syn::parse_quote!(/), BinaryOp::Div),
            (syn::parse_quote!(%), BinaryOp::Rem),
            (syn::parse_quote!(<), BinaryOp::Lt),
            (syn::parse_quote!(<=), BinaryOp::Le),
            (syn::parse_quote!(>), BinaryOp::Gt),
            (syn::parse_quote!(>=), BinaryOp::Ge),
            (syn::parse_quote!(==), BinaryOp::Eq),
            (syn::parse_quote!(!=), BinaryOp::Ne),
            (syn::parse_quote!(&), BinaryOp::BitAnd),
            (syn::parse_quote!(|), BinaryOp::BitOr),
        ];
        for (syn_op, expected) in cases {
            assert_eq!(BinaryOp::from_syn(syn_op), Some(*expected), "{syn_op:?}");
        }
    }

    #[test]
    fn binary_op_from_syn_rejects_an_unsupported_syn_binop() {
        let syn_op: syn::BinOp = syn::parse_quote!(+=);
        assert_eq!(BinaryOp::from_syn(&syn_op), None);
    }

    #[test]
    fn unary_op_from_syn_maps_neg_and_not_to_their_own_variants() {
        let neg: syn::UnOp = syn::parse_quote!(-);
        let not: syn::UnOp = syn::parse_quote!(!);
        assert_eq!(UnaryOp::from_syn(&neg), Some(UnaryOp::Neg));
        assert_eq!(UnaryOp::from_syn(&not), Some(UnaryOp::Not));
    }
}
