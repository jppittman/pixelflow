//! A body's expressions and statements: every variant is syntax the language
//! gives a meaning to, carrying what was written and where.

use super::{Method, Projection};
use proc_macro2::Span;
use syn::{Ident, Type};

/// An expression in the kernel body.
///
/// Every variant is syntax the language gives a meaning to. What it does not
/// — a tuple, a tuple's field, a range, a path from outside the block — the parser
/// refuses at the token, with a span; nothing is carried along to be refused
/// by a later stage.
#[derive(Debug, Clone)]
pub(crate) enum Expr {
    /// A variable reference (X, Y, cx, etc.).
    Ident(Ident),

    /// A numeric literal (1.0, 2.5f32, etc.).
    Literal(LiteralExpr),

    /// A binary operation (a + b, x * y, etc.).
    Binary(BinaryExpr),

    /// A unary operation (-x).
    Unary(UnaryExpr),

    /// A method call (x.sqrt(), a.max(b), etc.).
    MethodCall(MethodCallExpr),

    /// A derivative projection: `DX(e)`.
    Project(ProjectExpr),

    /// A free function call: a helper (`f(x, y)`) or a kernel (`k(x, y)`).
    Call(CallExpr),

    /// The choice: `if c { a } else { b }`.
    If(IfExpr),

    /// A fold over a constant range: `(0..N).map(|i| e).sum()`.
    Fold(FoldExpr),

    /// A conversion: `i as f32`.
    Cast(CastExpr),

    /// A record's field: `p.x0`.
    Field(FieldExpr),

    /// A block expression ({ let dx = ...; dx * dx }).
    Block(BlockExpr),

    /// A parenthesized expression ((a + b)).
    Paren(Box<Expr>),
}

impl Expr {
    /// Where the expression is, for a diagnostic to point at: an operator's
    /// token, a name, a block's braces.
    pub(crate) fn span(&self) -> Span {
        match self {
            Expr::Ident(name) => name.span(),
            Expr::Literal(e) => e.span,
            Expr::Binary(e) => e.span,
            Expr::Unary(e) => e.span,
            Expr::MethodCall(e) => e.name.span(),
            Expr::Project(e) => e.span,
            Expr::Call(e) => e.func.span(),
            Expr::If(e) => e.span,
            Expr::Fold(e) => e.span,
            Expr::Cast(e) => e.span,
            Expr::Field(e) => e.member.span(),
            Expr::Block(e) => e.span,
            Expr::Paren(inner) => inner.span(),
        }
    }

    /// The name this expression is, through any parentheses, or `None` if
    /// it is not a name. A `usize` and a record are both written only by
    /// name: nothing in a body computes a count (plan §1.6) or builds a
    /// record (Phase D, D7).
    pub(crate) fn named(&self) -> Option<&Ident> {
        match self {
            Expr::Paren(inner) => inner.named(),
            Expr::Ident(name) => Some(name),
            _ => None,
        }
    }
}

/// `base.member`: one field of a record. The base is a record by name — a
/// parameter or a `let` alias of one — which `sema` checks, with the
/// member one of the record's fields.
#[derive(Debug, Clone)]
pub(crate) struct FieldExpr {
    pub(crate) base: Box<Expr>,
    pub(crate) member: Ident,
}

/// A numeric literal: the number it denotes, decided by the parser.
///
/// The parser rounds a float once, as rustc does, so no later stage holds
/// the source text or rounds it again. An integer is kept exactly as
/// written, because its type depends on where it stands: a value where a
/// value is expected ([`LiteralExpr::f32_value`]), a count in a range's
/// bounds or a `usize` const ([`LiteralExpr::usize_value`]). Every stage
/// asks those two questions here, so no two stages can answer them
/// differently.
#[derive(Debug, Clone)]
pub(crate) struct LiteralExpr {
    pub(crate) value: Literal,
    pub(crate) span: Span,
}

/// What a numeric literal denotes, before its position gives it a type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Literal {
    /// A float literal, or an integer suffixed `f32` (a float literal to
    /// rustc): the `f32` it rounds to, once.
    F32(f32),
    /// An unsuffixed integer literal: exactly the integer written.
    Int(u128),
}

impl LiteralExpr {
    /// The `f32` this literal denotes where a value is expected.
    ///
    /// An integer's digits claim exactness, and rustc has no rounding of its
    /// own to borrow here: it refuses an unsuffixed integer where an `f32`
    /// is expected. `16777217` (2²⁴ + 1) is the first integer refused;
    /// `1099511627776` (2⁴⁰) is accepted.
    ///
    /// # Errors
    ///
    /// An integer an `f32` does not hold exactly: more than
    /// [`f32::MANTISSA_DIGITS`] significant bits.
    pub(crate) fn f32_value(&self) -> syn::Result<f32> {
        match self.value {
            Literal::F32(value) => Ok(value),
            // Exact, and finite: below 2¹²⁸, an integer with at most
            // `MANTISSA_DIGITS` significant bits is at most `f32::MAX`.
            Literal::Int(n) if significant_bits(n) <= f32::MANTISSA_DIGITS => Ok(n as f32),
            Literal::Int(n) => Err(syn::Error::new(
                self.span,
                format!(
                    "`{n}` is not exactly representable as an `f32`\n\
                     \n\
                     note: an `f32` holds an integer exactly only when it has at most {} \
                     significant bits\n\
                     \n\
                     help: write the `f32` you mean as a float literal, e.g. `{n}.0`, \
                     which rounds to the nearest one",
                    f32::MANTISSA_DIGITS,
                ),
            )),
        }
    }

    /// The `usize` this literal denotes in a range's bounds or a `usize`
    /// const: an integer, at most `usize::MAX`. A `usize` is 64 bits on
    /// every target this language compiles for (x86-64 and aarch64), and
    /// the control plane is 64-bit.
    ///
    /// # Errors
    ///
    /// A float literal, or an integer past `usize::MAX`.
    pub(crate) fn usize_value(&self) -> syn::Result<u64> {
        let Literal::Int(n) = self.value else {
            return Err(syn::Error::new(
                self.span,
                "mismatched types: expected `usize`, found a float literal\n\
                 \n\
                 note: a range's bounds and a `usize` const are counts, written as integers",
            ));
        };
        u64::try_from(n).map_err(|_| {
            syn::Error::new(
                self.span,
                format!(
                    "literal out of range for `usize`\n\
                     \n\
                     note: `{n}` is past `usize::MAX`, {}",
                    u64::MAX
                ),
            )
        })
    }
}

/// The bits between an integer's highest and lowest set bits, inclusive:
/// what a binary significand must hold to represent it exactly.
fn significant_bits(n: u128) -> u32 {
    if n == 0 {
        return 0;
    }
    u128::BITS - n.leading_zeros() - n.trailing_zeros()
}

/// Binary operators we recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
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
pub(crate) struct BinaryExpr {
    pub(crate) op: BinaryOp,
    pub(crate) lhs: Box<Expr>,
    pub(crate) rhs: Box<Expr>,
    /// The operator's span.
    pub(crate) span: Span,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Neg,
}

/// A unary expression.
#[derive(Debug, Clone)]
pub(crate) struct UnaryExpr {
    pub(crate) op: UnaryOp,
    pub(crate) operand: Box<Expr>,
    /// The operator's span.
    pub(crate) span: Span,
}

/// A method call expression.
#[derive(Debug, Clone)]
pub(crate) struct MethodCallExpr {
    /// The receiver (what the method is called on).
    pub(crate) receiver: Box<Expr>,
    /// The method as written (sqrt, sin, max, etc.): its spelling and span,
    /// for a diagnostic to quote and point at.
    pub(crate) name: Ident,
    /// What the name denotes at this call's arity, resolved once by the
    /// parser: every later stage reads this, never the spelling.
    pub(crate) method: Method,
    /// Method arguments (empty for sqrt, one arg for max, etc.).
    pub(crate) args: Vec<Expr>,
}

/// `DX(e)`: the projection of one argument.
#[derive(Debug, Clone)]
pub(crate) struct ProjectExpr {
    pub(crate) projection: Projection,
    pub(crate) arg: Box<Expr>,
    /// The callee's span.
    pub(crate) span: Span,
}

/// A free function call expression: a helper, or a kernel-typed parameter
/// applied.
#[derive(Debug, Clone)]
pub(crate) struct CallExpr {
    /// The function being called (a helper's name, or a kernel's).
    pub(crate) func: Ident,
    /// Function arguments.
    pub(crate) args: Vec<Expr>,
}

/// How many arguments a callee takes, and how many a call supplied.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Arity {
    pub(crate) want: usize,
    pub(crate) got: usize,
}

impl Arity {
    /// "was" or "were", agreeing with the number supplied.
    pub(crate) fn supplied_verb(self) -> &'static str {
        match self.got {
            1 => "was",
            _ => "were",
        }
    }
}

/// The refusal of a call that supplies `arity.got` arguments to a callee that
/// takes `arity.want`.
pub(crate) fn arity_mismatch(
    span: Span,
    callee: impl std::fmt::Display,
    arity: Arity,
) -> syn::Error {
    let Arity { want, got } = arity;
    let plural = match want {
        1 => "",
        _ => "s",
    };
    syn::Error::new(
        span,
        format!(
            "`{callee}` takes {want} argument{plural}, but {got} {} supplied",
            arity.supplied_verb(),
        ),
    )
}

/// `if cond { then } else { otherwise }`. The `else` is required: a body is
/// an expression, and there is no unit. An `else if` chain is an `If` in
/// the `else` position.
#[derive(Debug, Clone)]
pub(crate) struct IfExpr {
    pub(crate) cond: Box<Expr>,
    pub(crate) then_branch: BlockExpr,
    pub(crate) else_branch: Box<Expr>,
    /// The `if` token's span.
    pub(crate) span: Span,
}

/// `(lo..hi).map(|i| e).sum()` and its siblings: the fold of `e` over
/// `i ∈ [lo, hi)` under a monoid, or the monoid's identity when the range is
/// empty (docs/language/2026-09-25-the-language-is-kernel.md §1.5).
///
/// The closure's parameter is the binder and its body is the fold's body.
/// The closure is not a value, and has no meaning anywhere else. The bounds
/// are constant, evaluated by `sema` at expansion, and the binder is a
/// `usize`, which a body makes a value of only by `i as f32`.
#[derive(Debug, Clone)]
pub(crate) struct FoldExpr {
    pub(crate) reduction: Reduction,
    pub(crate) range: RangeExpr,
    /// The closure's parameter: the index the body reads.
    pub(crate) binder: Ident,
    pub(crate) body: Box<Expr>,
    /// The span of the method that names the reduction (`sum`, `fold`,
    /// `any`, …).
    pub(crate) span: Span,
}

/// The half-open `lo..hi` a fold ranges over.
#[derive(Debug, Clone)]
pub(crate) struct RangeExpr {
    pub(crate) lo: Box<Expr>,
    pub(crate) hi: Box<Expr>,
    /// The `..` token's span.
    pub(crate) span: Span,
}

/// Which monoid a fold combines its terms under, as the source spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reduction {
    /// `.map(|i| e).sum()`: `+`, identity 0.
    Sum,
    /// `.map(|i| e).product()`: `×`, identity 1.
    Product,
    /// `.map(|i| e).fold(f32::INFINITY, f32::min)`: identity +∞.
    Min,
    /// `.map(|i| e).fold(f32::NEG_INFINITY, f32::max)`: identity −∞.
    Max,
    /// `.any(|i| m)`: a mask's `|`, identity all-clear.
    Any,
    /// `.all(|i| m)`: a mask's `&`, identity all-set.
    All,
}

/// `operand as f32`, the language's one conversion. The target is always
/// `f32` — the parser refuses any other — so it is not stored. `sema`
/// accepts one operand: a `usize`, by name, a fold's index, a `usize` const
/// or an entry's per-call count parameter.
#[derive(Debug, Clone)]
pub(crate) struct CastExpr {
    pub(crate) operand: Box<Expr>,
    /// The `as` token's span.
    pub(crate) span: Span,
}

impl CastExpr {
    /// The name being converted, through any parentheses, or `None` if the
    /// operand is not a name. Only a name can be a `usize`: nothing in a
    /// body computes one (plan §1.6), so a `usize` expression is always a
    /// fold's index, a `usize` const or an entry count parameter.
    pub(crate) fn named(&self) -> Option<&Ident> {
        self.operand.named()
    }
}

/// A statement in a block.
#[derive(Debug, Clone)]
pub(crate) enum Stmt {
    /// `let dx = X - cx;`, or `let (a, b) = (e1, e2);` flattened: each name
    /// bound to its expression, and all at once — every expression is
    /// evaluated where the statement stands, before any of its names binds,
    /// as Rust evaluates the tuple before taking it apart (D7's front half).
    /// There is no tuple left behind to be a value, and the one-name `let`
    /// is the list of one.
    Let(Vec<LetStmt>),
    /// An expression statement: `foo();`
    Expr(Expr),
}

/// A let statement.
#[derive(Debug, Clone)]
pub(crate) struct LetStmt {
    pub(crate) name: Ident,
    pub(crate) ty: Option<Type>,
    pub(crate) init: Expr,
}

/// A block expression.
#[derive(Debug, Clone)]
pub(crate) struct BlockExpr {
    pub(crate) stmts: Vec<Stmt>,
    /// The final expression (if any).
    pub(crate) expr: Option<Box<Expr>>,
    /// The braces' span.
    pub(crate) span: Span,
}
