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
//! constructs the appropriate PixelFlow type trees at runtime.
//!
//! ## AST Structure
//!
//! ```text
//! KernelDef
//!   ├── params: [(name, type), ...]    // Closure parameters
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
use syn::{Ident, Type, Visibility};

/// Optional struct declaration for named kernels.
///
/// When present, the kernel emits a named struct instead of an anonymous one.
/// Example: `kernel! pub struct Circle = |cx: f32, cy: f32, r: f32| -> Field { ... }`
#[derive(Debug, Clone)]
pub struct StructDecl {
    /// Visibility (pub, pub(crate), or inherited).
    pub visibility: Visibility,
    /// The struct name.
    pub name: Ident,
}

/// A complete kernel definition.
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// Optional struct declaration for named kernels.
    /// If None, an anonymous closure-based kernel is emitted.
    pub struct_decl: Option<StructDecl>,
    /// Parameters captured from the closure syntax.
    pub params: Vec<Param>,
    /// Optional domain type annotation (e.g., `Field` in `Field -> Discrete`).
    /// When specified separately from return type, allows non-Coordinate output types.
    pub domain_ty: Option<Type>,
    /// Optional return type annotation (e.g., `-> Jet3` or `-> Discrete`).
    pub return_ty: Option<Type>,
    /// The kernel body expression.
    pub body: Expr,
}

/// Parameter kind - scalar (f32/i32) or manifold (generic).
#[derive(Debug, Clone)]
pub enum ParamKind {
    /// Scalar parameter - use Let/Var binding with concrete type.
    /// Example: `r: f32` → struct field, bound via Let::new(self.r, ...)
    Scalar(Box<Type>),
    /// Manifold parameter - generic type with trait bounds.
    /// Example: `inner: kernel` → generic M0, evaluated then bound via Let
    Manifold,
}

/// A captured parameter.
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name.
    pub name: Ident,
    /// Parameter kind (scalar with type, or manifold).
    pub kind: ParamKind,
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
    pub span: Span,
}

/// An identifier expression.
#[derive(Debug, Clone)]
pub struct IdentExpr {
    pub name: Ident,
    pub span: Span,
}

/// A literal expression.
#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub lit: syn::Lit,
    pub span: Span,
}

/// Binary operators we recognize.
#[derive(Debug, Clone, Copy)]
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
    pub span: Span,
}

/// Unary operators.
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// A unary expression.
#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Box<Expr>,
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
    pub span: Span,
}

/// A free function call expression (V(m), DX(expr), etc.).
#[derive(Debug, Clone)]
pub struct CallExpr {
    /// The function being called (V, DX, DY, etc.).
    pub func: Ident,
    /// Function arguments.
    pub args: Vec<Expr>,
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
    pub span: Span,
}

/// A block expression.
#[derive(Debug, Clone)]
pub struct BlockExpr {
    pub stmts: Vec<Stmt>,
    /// The final expression (if any).
    pub expr: Option<Box<Expr>>,
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
