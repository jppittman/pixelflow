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
//!   ├── records: [RecordItem, ...]    // `struct R { a: f32, b: f32 }`
//!   ├── consts: [ConstItem, ...]      // `const NAME: f32 = expr;`
//!   └── fns: [FnItem, ...]            // `pub fn` entries and private helpers
//!
//! FnItem
//!   ├── vis                           // `pub` makes an entry; private is a helper
//!   ├── structural: [name, ...]       // an entry's `const N: usize` parameters
//!   ├── params: [(name, type), ...]   // scalar and record parameters
//!   ├── ret: type                     // declared; absent only for the closure sugar
//!   └── body: Expr
//!
//! Expr
//!   ├── Ident(name)                    // Variable reference: X, cx, etc.
//!   ├── Literal(value)                 // Numeric literal: 1.0, 2.5, 4
//!   ├── Binary(op, lhs, rhs)           // a + b, x * y
//!   ├── Unary(op, operand)             // -x
//!   ├── MethodCall(receiver, method, args) // x.sqrt(), a.max(b)
//!   ├── Call(func, args)               // DX(e), a helper: f(x, y)
//!   ├── If(cond, then, else)           // if c { a } else { b }
//!   ├── Fold(reduction, range, binder, body) // (0..N).map(|i| e).sum()
//!   ├── Cast(operand)                  // i as f32
//!   ├── Field(base, member)            // p.x0, a record's field
//!   ├── Block(stmts, expr)             // { let dx = ...; dx * dx }
//!   └── Paren(inner)                   // (a + b)
//! ```

use proc_macro2::Span;
use syn::ext::IdentExt;
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
    /// The block's records, in declaration order: [`RecordId`] indexes this.
    pub records: Vec<RecordItem>,
    /// The block's `const` items, in declaration order.
    pub consts: Vec<ConstItem>,
    /// The block's `fn` items — entries and helpers — in declaration order.
    pub fns: Vec<FnItem>,
}

impl KernelDef {
    /// The record a declared type names, if it names one of the block's.
    ///
    /// The one resolution of a record's name: `sema` types a parameter by
    /// it, and lowering and emission lay a record parameter out by it, so no
    /// two stages can disagree about which record a type is.
    pub fn record_named(&self, ty: &Type) -> Option<RecordId> {
        let Type::Path(path) = ty else {
            return None;
        };
        if path.qself.is_some() {
            return None;
        }
        let ident = path.path.get_ident()?;
        self.records
            .iter()
            .position(|record| record.name == *ident)
            .map(RecordId)
    }

    /// The record `id` names.
    pub fn record(&self, id: RecordId) -> &RecordItem {
        &self.records[id.0]
    }
}

/// Which of the block's records a type is: an index into
/// [`KernelDef::records`], in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecordId(pub usize);

/// `struct R { a: f32, b: f32 }`: a record of named `f32` fields
/// (docs/plans/2026-09-25-the-language-is-kernel.md §1.3).
///
/// Emitted as a host `#[repr(C)]` struct of the same name, fields and
/// visibility. In a body a record is its fields: an entry's record parameter
/// is one uniform per field, in field order, and a helper's is its
/// argument's fields.
#[derive(Debug, Clone)]
pub struct RecordItem {
    /// Its attributes (any but `repr` and `cfg`), re-emitted on the host
    /// struct.
    pub attrs: Vec<syn::Attribute>,
    pub vis: syn::Visibility,
    pub name: Ident,
    /// The fields, in declaration order: the order a record parameter's
    /// uniforms are declared in.
    pub fields: Vec<RecordField>,
}

/// One named field of a record. `sema` requires its type to be `f32`.
#[derive(Debug, Clone)]
pub struct RecordField {
    /// Its attributes (any but `repr` and `cfg`), re-emitted on the host
    /// struct's field.
    pub attrs: Vec<syn::Attribute>,
    pub vis: syn::Visibility,
    pub name: Ident,
    pub ty: Type,
}

/// How a `kernel!` invocation spells its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spelling {
    /// `kernel!(|a: f32, …| e)`: one entry, and the expansion is an
    /// expression — a `Kernel` with no parameters, a closure over their
    /// `f32`s with some, each one a uniform.
    Closure,
    /// `kernel! { struct …; const …; fn …; pub fn … }`: the expansion is
    /// items — a host struct per record, a host `const` per `pub const`, and
    /// per entry a host `fn` and, when it has parameters, its `Args` record.
    Items,
}

/// A `const NAME: f32 = expr;` or `const NAME: usize = expr;` item,
/// evaluated at expansion.
#[derive(Debug, Clone)]
pub struct ConstItem {
    /// Doc comments, re-emitted on a `pub const`'s host twin.
    pub attrs: Vec<syn::Attribute>,
    /// `pub` makes the value a host `const` too.
    pub vis: syn::Visibility,
    pub name: Ident,
    /// The declared type; sema requires `f32` or `usize`.
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
    /// An entry's structural parameters, its `const N: usize` generics, in
    /// declaration order (plan §1.4). Each value is its own program: the
    /// host function is generic over them, and a body reads one as a count,
    /// in a range's bounds or as `N as f32`. A helper has none.
    pub structural: Vec<Ident>,
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

    /// The name of an entry's `Args` record: the entry's name in
    /// UpperCamelCase, then `Args` — `shifted_radius` has
    /// `ShiftedRadiusArgs` (plan §1.4). `sema` refuses one that collides,
    /// and emission names the record by it.
    pub fn args_record(&self) -> Ident {
        let camel: String = self
            .name
            .unraw()
            .to_string()
            .split('_')
            .flat_map(|word| {
                let mut letters = word.chars();
                letters
                    .next()
                    .map(|first| first.to_uppercase().chain(letters))
                    .into_iter()
                    .flatten()
            })
            .collect();
        Ident::new(&format!("{camel}Args"), self.name.span())
    }
}

/// A declared parameter: a scalar, or one of the block's records.
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name.
    pub name: Ident,
    /// The declared type (`f32` or a record; `bool` in a helper).
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
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident(e) => e.span,
            Expr::Literal(e) => e.span,
            Expr::Binary(e) => e.span,
            Expr::Unary(e) => e.span,
            Expr::MethodCall(e) => e.span,
            Expr::Call(e) => e.span,
            Expr::If(e) => e.span,
            Expr::Fold(e) => e.span,
            Expr::Cast(e) => e.span,
            Expr::Field(e) => e.span,
            Expr::Block(e) => e.span,
            Expr::Paren(inner) => inner.span(),
        }
    }

    /// The name this expression is, through any parentheses, or `None` if
    /// it is not a name. A `usize` and a record are both written only by
    /// name: nothing in a body computes a count (plan §1.6) or builds a
    /// record (Phase D, D7).
    pub fn named(&self) -> Option<&Ident> {
        match self {
            Expr::Paren(inner) => inner.named(),
            Expr::Ident(ident) => Some(&ident.name),
            _ => None,
        }
    }
}

/// `base.member`: one field of a record. The base is a record by name — a
/// parameter or a `let` alias of one — which `sema` checks, with the
/// member one of the record's fields.
#[derive(Debug, Clone)]
pub struct FieldExpr {
    pub base: Box<Expr>,
    pub member: Ident,
    /// The member's span.
    pub span: Span,
}

/// An identifier expression.
#[derive(Debug, Clone)]
pub struct IdentExpr {
    pub name: Ident,
    /// The name's span.
    pub span: Span,
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
pub struct LiteralExpr {
    pub value: Literal,
    pub span: Span,
}

/// What a numeric literal denotes, before its position gives it a type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Literal {
    /// A float literal, or an integer suffixed `f32` (a float literal to
    /// rustc): the `f32` it rounds to, once.
    F32(f32),
    /// An unsuffixed integer literal: exactly the integer written.
    Int(u128),
}

impl LiteralExpr {
    /// The `f32` this literal denotes where a value is expected.
    ///
    /// An integer is refused unless an `f32` holds it exactly: at most
    /// [`f32::MANTISSA_DIGITS`] significant bits. An integer's digits claim
    /// exactness, and rustc has no rounding of its own to borrow here: it
    /// refuses an unsuffixed integer where an `f32` is expected. `16777217`
    /// (2²⁴ + 1) is the first integer refused; `1099511627776` (2⁴⁰) is
    /// accepted.
    pub fn f32_value(&self) -> syn::Result<f32> {
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
    pub fn usize_value(&self) -> syn::Result<u64> {
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

/// `(lo..hi).map(|i| e).sum()` and its siblings: the fold of `e` over
/// `i ∈ [lo, hi)` under a monoid, or the monoid's identity when the range is
/// empty (docs/plans/2026-09-25-the-language-is-kernel.md §1.5).
///
/// The closure's parameter is the binder and its body is the fold's body.
/// The closure is not a value, and has no meaning anywhere else. The bounds
/// are constant, evaluated by `sema` at expansion, and the binder is a
/// `usize`, which a body makes a value of only by `i as f32`.
#[derive(Debug, Clone)]
pub struct FoldExpr {
    pub reduction: Reduction,
    pub range: RangeExpr,
    /// The closure's parameter: the index the body reads.
    pub binder: Ident,
    pub body: Box<Expr>,
    /// The span of the method that names the reduction (`sum`, `fold`,
    /// `any`, …).
    pub span: Span,
}

/// The half-open `lo..hi` a fold ranges over.
#[derive(Debug, Clone)]
pub struct RangeExpr {
    pub lo: Box<Expr>,
    pub hi: Box<Expr>,
    /// The `..` token's span.
    pub span: Span,
}

/// Which monoid a fold combines its terms under, as the source spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
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
/// or an entry's structural parameter.
#[derive(Debug, Clone)]
pub struct CastExpr {
    pub operand: Box<Expr>,
    /// The `as` token's span.
    pub span: Span,
}

impl CastExpr {
    /// The name being converted, through any parentheses, or `None` if the
    /// operand is not a name. Only a name can be a `usize`: nothing in a
    /// body computes one (plan §1.6), so a `usize` expression is always a
    /// fold's index, a `usize` const or a structural parameter.
    pub fn named(&self) -> Option<&Ident> {
        self.operand.named()
    }
}

/// A statement in a block.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// A let binding: `let dx = X - cx;`
    Let(Box<LetStmt>),
    /// `let (a, b) = (e1, e2);`, flattened: each name bound to its
    /// expression, and all at once — every expression is evaluated where the
    /// statement stands, before any of its names binds, as Rust evaluates the
    /// tuple before taking it apart (D7's front half). There is no tuple
    /// left behind to be a value.
    LetTuple(Vec<LetStmt>),
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

    /// An entry's `Args` record is its name in UpperCamelCase, then `Args`;
    /// a raw identifier is named by what it spells.
    #[test]
    fn an_entrys_args_record_is_its_name_in_upper_camel_case() {
        for (entry, args) in [
            ("circle", "CircleArgs"),
            ("shifted_radius2", "ShiftedRadius2Args"),
            ("a__b_", "ABArgs"),
            ("r#type", "TypeArgs"),
        ] {
            let item: FnItem = FnItem {
                attrs: Vec::new(),
                vis: syn::parse_quote!(pub),
                name: syn::parse_str(entry).expect("an identifier"),
                structural: Vec::new(),
                params: Vec::new(),
                ret: None,
                body: Expr::Paren(Box::new(Expr::Ident(IdentExpr {
                    name: syn::parse_quote!(X),
                    span: Span::call_site(),
                }))),
            };
            assert_eq!(item.args_record().to_string(), args, "{entry}");
        }
    }
}
