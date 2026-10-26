//! # Abstract Syntax Tree
//!
//! The AST represents the structure of a kernel block after parsing.
//!
//! ## Design Philosophy
//!
//! The AST is a **source-level representation** that preserves the structure
//! the user wrote. It is not the IR: lowering builds the IR, an `ExprArena`,
//! from it.
//!
//! The compiler's job is to transform this source AST into Rust code that
//! rebuilds the corresponding arena fragment at load time.
//!
//! ## AST Structure
//!
//! ```text
//! KernelDef
//!   ├── Closure(ClosureEntry)         // `|a: f32, …| e`: one entry, its type inferred
//!   │     ├── params: [(name, type), ...]
//!   │     └── body: Expr
//!   └── Items                         // `kernel! { … }`
//!         ├── records: [RecordItem, ...]  // `struct R { a: f32, b: f32 }`
//!         ├── consts: [ConstItem, ...]    // `const NAME: f32 = expr;`
//!         └── fns: [FnItem, ...]          // `pub fn` entries and private helpers
//!
//! FnItem
//!   ├── vis                           // `pub` makes an entry; private is a helper
//!   ├── params: [(name, type), ...]   // scalar, record and kernel parameters
//!   ├── ret: type                     // declared
//!   └── body: Expr
//!
//! Expr
//!   ├── Ident(name)                    // Variable reference: X, cx, etc.
//!   ├── Literal(value)                 // Numeric literal: 1.0, 2.5, 4
//!   ├── Binary(op, lhs, rhs)           // a + b, x * y
//!   ├── Unary(op, operand)             // -x
//!   ├── MethodCall(receiver, method, args) // x.sqrt(), a.max(b)
//!   ├── Project(projection, arg)       // DX(e)
//!   ├── Call(func, args)               // a helper: f(x, y), or a kernel: k(x, y)
//!   ├── If(cond, then, else)           // if c { a } else { b }
//!   ├── Fold(reduction, range, binder, body) // (0..N).map(|i| e).sum()
//!   ├── Cast(operand)                  // i as f32
//!   ├── Field(base, member)            // p.x0, a record's field
//!   ├── Block(stmts, expr)             // { let dx = ...; dx * dx }
//!   └── Paren(inner)                   // (a + b)
//! ```
//!
//! # Contract
//!
//! The vocabulary has no stage function: its contract is the types below,
//! each `pub(crate) use`d here, and the one reading of every question two
//! stages ask of them. Fields the parser builds and the stages read are
//! `pub(crate)`. Every fallible entry returns `syn::Result`, `Err` a
//! `syn::Error` spanned at the construct.
//!
//! ```ignore
//! /// The plan every refusal cites: a refusal of a construct names the phase that brings it.
//! pub(crate) const PLAN: &str;
//!
//! /// What a `kernel!` invocation spells: the closure form is one entry, the items form a block.
//! enum KernelDef { Closure(ClosureEntry), Items { records: Vec<RecordItem>, consts: Vec<ConstItem>, fns: Vec<FnItem> } }
//! impl KernelDef {
//!     /// The block's records, `const`s and `fn`s, in declaration order; a closure's are empty.
//!     fn records(&self) -> &[RecordItem];  fn consts(&self) -> &[ConstItem];  fn fns(&self) -> &[FnItem];
//!     /// The one resolution of a record's name: the record `ty` names bare, if it is one of the block's.
//!     fn record_named(&self, ty: &Type) -> Option<RecordId>;
//!     /// The record `id` names.
//!     fn record(&self, id: RecordId) -> &RecordItem;
//! }
//! /// The view of a function sema and lowering read, from either spelling.
//! impl ClosureEntry { fn callable(&self) -> Callable<'_>; }
//! impl FnItem {
//!     fn callable(&self) -> Callable<'_>;
//!     /// `Entry` iff the item has a visibility; a private `fn` is a `Helper`.
//!     fn role(&self) -> Role;
//!     /// Whether a parameter is kernel-typed.
//!     fn takes_a_kernel(&self) -> bool;
//!     /// Params non-empty and none kernel-typed.
//!     fn has_args_record(&self) -> bool;
//!     /// The name in UpperCamelCase, then `Args`; a leading digit gains one `_`.
//!     fn args_record(&self) -> Ident;
//! }
//! /// Kernel-typed parameters in declaration order: the order their arguments are admitted.
//! impl Callable<'_> { fn kernel_parameters(&self) -> impl Iterator<Item = &Ident>; }
//! impl Param { fn is_kernel(&self) -> bool; }
//!
//! /// `Kernel` iff `ty` is `impl Fn(f32, f32) -> f32`, through `Paren`, `Group` and `Reference`; any
//! /// other function spelling is `Misspelled(what differs)`; `NotOne` otherwise.
//! impl FunctionType { fn of(ty: &Type) -> FunctionType; }
//! /// The one bare name a type is, through a macro's invisible group; `None` for `::p`, a qself, generics.
//! fn bare_name(ty: &Type) -> Option<&Ident>;  fn is_f32(ty: &Type) -> bool;
//!
//! /// A name as the language compares it: `Eq` and `Hash` on the identifier without its `r#`.
//! impl Name {
//!     fn of(ident: &Ident) -> Name;
//!     /// The axis this name is the coordinate of, if it is `X` or `Y`.
//!     fn coordinate(&self) -> Option<Axis>;
//!     /// The two names the language spells itself, each with its axis: `X` then `Y`.
//!     fn coordinates() -> impl Iterator<Item = (Name, Axis)>;
//! }
//! fn same_name(a: &Ident, b: &Ident) -> bool;
//! /// `V`, `DX`, `DY`, `DXX`, `DXY`, `DYY`, by name; `None` for any other.
//! impl Projection { fn from_name(name: &Name) -> Option<Projection>; }
//! /// Rust's lexical scoping over bindings of `V`: a function scope no pop reaches, one map per open block.
//! impl<V> Scopes<V> {
//!     fn default() -> Self;  fn push_scope(&mut self);  fn pop_scope(&mut self);
//!     /// Shadows any binding of `name` in scope, the same block's too.
//!     fn bind(&mut self, name: Name, value: V);
//!     /// The innermost binding in scope, else the function's.
//!     fn lookup(&self, name: &Name) -> Option<&V>;
//! }
//! impl Method {
//!     /// The method `name` is at `arity` arguments (the receiver not counted): `clone`/0,
//!     /// `OpKind::from_method_call`, then `LibraryMethod::ALL`; name and arity together, else `None`.
//!     fn resolve(name: &str, arity: usize) -> Option<Method>;
//!     /// Every `(name, arity)` a body may call: the parser's oracle, and the sweep's.
//!     fn advertised() -> impl Iterator<Item = (&'static str, usize)>;
//! }
//!
//! /// Where the expression is, for a diagnostic: an operator's token, a name, a block's braces.
//! impl Expr { fn span(&self) -> Span;
//!     /// The name this expression is, through parentheses; `None` if it is not a name.
//!     fn named(&self) -> Option<&Ident>; }
//! impl CastExpr { fn named(&self) -> Option<&Ident>; }
//! impl LiteralExpr {
//!     /// Err: an integer with more than `f32::MANTISSA_DIGITS` significant bits.
//!     fn f32_value(&self) -> syn::Result<f32>;
//!     /// Err: a float literal, or an integer past `u64::MAX`.
//!     fn usize_value(&self) -> syn::Result<u64>;
//! }
//! /// "was" or "were", agreeing with the number supplied.
//! impl Arity { fn supplied_verb(self) -> &'static str; }
//! /// The refusal of a call supplying `arity.got` arguments to a callee taking `arity.want`.
//! fn arity_mismatch(span: Span, callee: impl Display, arity: Arity) -> syn::Error;
//! ```

mod def;
mod expr;
mod name;
mod scopes;
mod types;

pub(crate) use def::{
    Callable, ClosureEntry, ConstItem, FnItem, KernelDef, Param, RecordField, RecordId, RecordItem,
    Role,
};
pub(crate) use expr::{
    Arity, BinaryExpr, BinaryOp, BlockExpr, CallExpr, CastExpr, Expr, FieldExpr, FoldExpr, IfExpr,
    LetStmt, Literal, LiteralExpr, MethodCallExpr, ProjectExpr, RangeExpr, Reduction, Stmt,
    UnaryExpr, UnaryOp, arity_mismatch,
};
pub(crate) use name::{LibraryMethod, Method, Name, Projection, same_name};
pub(crate) use scopes::Scopes;
pub(crate) use types::{FunctionType, bare_name, is_f32};

/// The plan that owns the constructs the front end refuses by phase: a
/// refusal of one names the phase that brings it.
pub(crate) const PLAN: &str = "docs/language/2026-09-25-the-language-is-kernel.md";
