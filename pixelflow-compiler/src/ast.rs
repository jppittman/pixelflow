//! # Abstract Syntax Tree
//!
//! The AST represents the structure of a kernel block after parsing.
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
//!   ├── spelling: Closure | Items     // how the macro's value is spelled
//!   ├── consts: [ConstItem, ...]      // `const NAME: f32 = expr;`
//!   └── fns: [FnItem, ...]            // `pub fn` entries and private helpers
//!
//! FnItem
//!   ├── vis                           // `pub` makes an entry; private is a helper
//!   ├── params: [(name, type), ...]   // scalar parameters
//!   ├── ret: type                     // declared; absent only for the closure sugar
//!   └── body: Expr
//!
//! Expr
//!   ├── Ident(name)                    // Variable reference: X, cx, etc.
//!   ├── Literal(value)                 // Numeric literal, as its f32: 1.0, 2.5
//!   ├── Binary(op, lhs, rhs)           // a + b, x * y
//!   ├── Unary(op, operand)             // -x
//!   ├── MethodCall(receiver, method, args) // x.sqrt(), a.max(b)
//!   ├── Call(func, args)               // DX(e), a helper: f(x, y)
//!   ├── If(cond, then, else)           // if c { a } else { b }
//!   ├── Block(stmts, expr)             // { let dx = ...; dx * dx }
//!   └── Paren(inner)                   // (a + b)
//! ```

use proc_macro2::Span;
use syn::{Ident, Type};

/// A complete kernel definition: the items of a `kernel!` block.
///
/// The closure form `|a: f32, …| e` is sugar for a block with one entry, so
/// every stage after the parser sees one shape; only emission asks how the
/// result is spelled.
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// How the macro's value is spelled.
    pub spelling: Spelling,
    /// The block's `const` items, in declaration order.
    pub consts: Vec<ConstItem>,
    /// The block's `fn` items — entries and helpers — in declaration order.
    pub fns: Vec<FnItem>,
}

/// How a `kernel!` invocation spells its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spelling {
    /// `kernel!(|a: f32, …| e)`: one entry, and the expansion is an
    /// expression — a `Kernel` with no parameters, a builder closure with
    /// some.
    Closure,
    /// `kernel! { const …; fn …; pub fn … }`: the expansion is items, one
    /// host `fn` per entry and one host `const` per `pub const`.
    Items,
}

/// A `const NAME: f32 = expr;` item, evaluated at expansion.
#[derive(Debug, Clone)]
pub struct ConstItem {
    /// Doc comments, re-emitted on a `pub const`'s host twin.
    pub attrs: Vec<syn::Attribute>,
    /// `pub` makes the value a host `const` too.
    pub vis: syn::Visibility,
    pub name: Ident,
    /// The declared type; sema requires `f32`.
    pub ty: Type,
    pub init: Expr,
}

/// A `fn` item: an entry when `pub`, a helper otherwise.
#[derive(Debug, Clone)]
pub struct FnItem {
    /// Doc comments, re-emitted on an entry's host function.
    pub attrs: Vec<syn::Attribute>,
    /// `pub` makes an entry — the macro emits a host function for it. A
    /// private `fn` is a helper, inlined at each call.
    pub vis: syn::Visibility,
    pub name: Ident,
    pub params: Vec<Param>,
    /// The declared return type. `None` only for the closure sugar, whose
    /// type is inferred.
    pub ret: Option<Type>,
    /// The kernel body expression.
    pub body: Expr,
}

/// What a `fn` item is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A program: it may read `X` and `Y`, and the macro emits a host
    /// function for it.
    Entry,
    /// A function of its arguments only, inlined at each call.
    Helper,
}

impl FnItem {
    /// An entry is any `fn` with a visibility; a helper has none.
    pub fn role(&self) -> Role {
        match self.vis {
            syn::Visibility::Inherited => Role::Helper,
            _ => Role::Entry,
        }
    }
}

/// A declared scalar parameter.
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name.
    pub name: Ident,
    /// The declared type (`f32`; `bool` in a helper).
    pub ty: Box<Type>,
}

/// An expression in the kernel body.
///
/// Every variant is syntax the language gives a meaning to. What it does not
/// — a tuple, a field, a range, a path from outside the block — the parser
/// refuses at the token, with a span; nothing is carried along to be refused
/// by a later stage.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A variable reference (X, Y, cx, etc.).
    Ident(IdentExpr),

    /// A numeric literal (1.0, 2.5f32, etc.).
    Literal(LiteralExpr),

    /// A binary operation (a + b, x * y, etc.).
    Binary(BinaryExpr),

    /// A unary operation (-x).
    Unary(UnaryExpr),

    /// A method call (x.sqrt(), a.max(b), etc.).
    MethodCall(MethodCallExpr),

    /// A free function call: a projection (`DX(e)`) or a helper (`f(x, y)`).
    Call(CallExpr),

    /// The choice: `if c { a } else { b }`.
    If(IfExpr),

    /// A block expression ({ let dx = ...; dx * dx }).
    Block(BlockExpr),

    /// A parenthesized expression ((a + b)).
    Paren(Box<Expr>),
}

impl Expr {
    /// Where the expression is, for a diagnostic to point at: an operator's
    /// token, a name, a block's braces.
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident(e) => e.span,
            Expr::Literal(e) => e.span,
            Expr::Binary(e) => e.span,
            Expr::Unary(e) => e.span,
            Expr::MethodCall(e) => e.span,
            Expr::Call(e) => e.span,
            Expr::If(e) => e.span,
            Expr::Block(e) => e.span,
            Expr::Paren(inner) => inner.span(),
        }
    }
}

/// An identifier expression.
#[derive(Debug, Clone)]
pub struct IdentExpr {
    pub name: Ident,
    /// The name's span.
    pub span: Span,
}

/// A numeric literal, already the `f32` it denotes.
///
/// The parser decides the value — rounding a float once, as rustc does, and
/// refusing a literal that names no `f32` — so no later stage holds the
/// source text or rounds it again.
#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub value: f32,
    pub span: Span,
}

/// Binary operators we recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    // Comparisons: `f32`s in, a `bool` out.
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    // `bool`s combine.
    BitAnd,
    BitOr,
}

/// A binary expression.
#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    /// The operator's span.
    pub span: Span,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

/// A unary expression.
#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Box<Expr>,
    /// The operator's span.
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
    /// The method name's span.
    pub span: Span,
}

/// A free function call expression (DX(expr), a helper, etc.).
#[derive(Debug, Clone)]
pub struct CallExpr {
    /// The function being called (V, DX, DY, a helper's name).
    pub func: Ident,
    /// Function arguments.
    pub args: Vec<Expr>,
    /// The function name's span.
    pub span: Span,
}

/// `if cond { then } else { otherwise }`. The `else` is required: a body is
/// an expression, and there is no unit. An `else if` chain is an `If` in
/// the `else` position.
#[derive(Debug, Clone)]
pub struct IfExpr {
    pub cond: Box<Expr>,
    pub then_branch: BlockExpr,
    pub else_branch: Box<Expr>,
    /// The `if` token's span.
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
    // Kept for AST-node uniformity; the name carries its own span.
    #[allow(dead_code)]
    pub span: Span,
}

/// A block expression.
#[derive(Debug, Clone)]
pub struct BlockExpr {
    pub stmts: Vec<Stmt>,
    /// The final expression (if any).
    pub expr: Option<Box<Expr>>,
    /// The braces' span.
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

    /// `%` has no IR op and `+=` assigns: neither is a kernel operator.
    #[test]
    fn binary_op_from_syn_rejects_an_unsupported_syn_binop() {
        for unsupported in [
            syn::parse_quote!(+=),
            syn::parse_quote!(%),
            syn::parse_quote!(^),
            syn::parse_quote!(&&),
        ] {
            let syn_op: syn::BinOp = unsupported;
            assert_eq!(BinaryOp::from_syn(&syn_op), None, "{syn_op:?}");
        }
    }

    /// `!` has no IR op, so it is not a kernel operator either.
    #[test]
    fn unary_op_from_syn_maps_neg_and_nothing_else() {
        let neg: syn::UnOp = syn::parse_quote!(-);
        let not: syn::UnOp = syn::parse_quote!(!);
        assert_eq!(UnaryOp::from_syn(&neg), Some(UnaryOp::Neg));
        assert_eq!(UnaryOp::from_syn(&not), None);
    }
}
