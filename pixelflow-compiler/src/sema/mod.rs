//! # Semantic Analysis
//!
//! Analyzes the AST for semantic correctness: every name resolves, every
//! expression has a type, every call is to a helper with the right arity, and
//! the call graph is a DAG.
//!
//! ## Responsibilities
//!
//! 1. **Items**: the block's records, `const`s and `fn`s are collected by
//!    name, a record's fields are checked to be `f32`s, and a `const` is
//!    evaluated at expansion ([`consts::evaluate_consts`]).
//! 2. **Symbol Resolution**: match identifiers to their definitions, with
//!    Rust's lexical scoping ([`crate::ast::Scopes`], which lowering
//!    resolves through too).
//! 3. **Types**: every expression is an `f32` or a `bool` ([`analyze::Ty`]). The IR
//!    keeps its lanes — a `bool` is a mask lane — so the type lives here and
//!    nowhere downstream; what it buys is that `X.select(Y, 7.0)`, which
//!    blended a number as a mask, is a type error rather than plausible
//!    pixels. A third type, `usize`, is a count: a fold's index, a `usize`
//!    const or an entry's count parameter.
//!    A body names one only as a fold bound or to convert it, `i as f32` —
//!    because nothing in the language computes with an index
//!    (docs/language/2026-09-25-the-language-is-kernel.md §1.3, §1.6). A
//!    fourth, a record, is a name for its `f32` fields: written by name,
//!    read by field, and never computed (Phase D, D7).
//! 4. **Calls**: a helper is called at its arity with arguments of its
//!    parameters' types; an entry is not callable; recursion is refused.
//! 5. **Folds**: bounds are known counts or unchanged entry count
//!    parameters ([`range_bounds`]) — never an enclosing fold's index, so a
//!    user fold is rectangular; known bounds run
//!    forwards, while dynamic order is guarded by the IR. The body is a term of its monoid, an `f32` or
//!    a `bool`, in a scope where the closure's parameter is the index.
//! 6. **Binding times** (plan §1.4): entry count parameters are per-call
//!    uniforms. [`AnalyzedKernel::parameters`]
//!    is the order they are declared in.
//! 7. **Tuple `let`s** (D7's front half): a tuple `let` types every
//!    expression before it binds any name.
//! 8. **Kernels** (Phase D-a): a parameter typed `impl Fn(f32, f32) -> f32`
//!    is a kernel the host passes at run time ([`analyze::Ty::Kernel`]). A body
//!    applies one, `k(x, y)`, or passes it by name to a helper's
//!    kernel-typed parameter, and does nothing else with it; and, as rustc
//!    moves an `impl Fn` it passes, it passes one on at most once on each
//!    path (an `if`'s arms are two), uses it after that on that path not at
//!    all, and never inside a fold's body.
//!
//! ## Symbol Classes
//!
//! | Class      | Binding Time        | Arena Representation | Example |
//! |------------|---------------------|----------------------|---------|
//! | Intrinsic  | Collapse time       | `Var(0)`, `Var(1)`   | X, Y    |
//! | Parameter  | Per call (uniform)  | `Uniform(slot)`, one per scalar or record field | cx, n, p |
//! | Const      | Expansion time      | `Const(v)`, or a count | PI, N |
//! | Local      | Expression scope    | A shared `ExprId`, or a record's fields | dx, q |
//! | Index      | A fold's body       | The fold's binder `Var` | i    |
//!
//! A helper's parameters are a further thing at lowering — the argument's
//! own node, or a record argument's fields, bound by name where the helper
//! is inlined — but to `sema` they are parameters like any other: a name
//! with a type.
//!
//! A kernel-typed parameter, `impl Fn(f32, f32) -> f32` (Phase D-a), is a
//! `Parameter` of type [`analyze::Ty::Kernel`]. To `sema` it is a name that is
//! applied, `k(x, y)`, or passed on; at lowering it is an argument admitted
//! when the entry's host function is called, whose term each application
//! splices in at its coordinates.
//!
//! ## Symbol Resolution Rules
//!
//! An identifier resolves to the innermost binding of its name in scope:
//! 1. a `let`-bound local → a shared arena id, or a record's fields
//! 2. a declared parameter → an entry's is a uniform, one per field of a
//!    record, bound per call; a helper's is the argument at the call
//! 3. a fold's index → the fold's binder, a `usize`, visible in the fold's
//!    body and nowhere else
//! 4. a `const` → its value
//! 5. an intrinsic (X, Y) → a coordinate `Var`, in an entry only: a helper
//!    takes its coordinates as arguments, so that application is contramap
//!    (docs/language/2026-09-25-the-language-is-kernel.md §1.2)
//! 6. otherwise → refused. A kernel body does not capture from the caller's
//!    scope, so a name nothing here binds is an error here, with a span —
//!    not a capture that lowering then refuses without one.
//!
//! Nothing shadows X, Y, a `const` or a `fn` — a
//! parameter, a `let` or a fold's index of that name is refused — so a
//! coordinate always means the coordinate and an item always means the
//! item.
//!
//! ## Output
//!
//! The semantic phase produces an [`AnalyzedKernel`]: the AST, validated,
//! and the value of every `const`.
//!
//! # Contract
//!
//! One function, the stage's, and the types it returns: `def` checked and its
//! `const`s evaluated, or the first refusal at its construct. Lowering reads
//! nine of these, emission five; `Ty` is not among them — the type a body's
//! expression has lives in `analyze.rs` and is read nowhere downstream. Every
//! item below is `pub(crate)` where it is implemented and re-exported here;
//! every other item under `sema/` is the implementation and is `pub(super)`
//! or private. Every fallible entry returns `syn::Result`, `Err` a
//! `syn::Error` spanned at the construct.
//!
//! ```ignore
//! /// Implemented by `sema::analyze` and the types below.
//! trait Analyze {
//!     /// `def` checked and its consts evaluated: every name bound in scope; every expression
//!     /// `f32` or `bool`, a `usize` only as `i as f32`, a record only by name, a kernel only
//!     /// applied or passed on once per path; every call a helper at its arity; the call graph
//!     /// acyclic; every `const` the value rustc gives; every fold's bounds admitted at their
//!     /// binding times; at least one entry. Returns `def` unchanged. Err: the first refusal, at its
//!     /// construct (a block with no entry has none, and is refused at the invocation).
//!     fn analyze(def: KernelDef) -> syn::Result<AnalyzedKernel>;
//!     /// An entry's uniforms in declaration order: an `f32` one, a record one per field, a
//!     /// kernel-typed one omitted — the one definition of a program's uniform order.
//!     fn AnalyzedKernel::parameters<'a>(&'a self, params: &'a [Param]) -> Vec<Parameter<'a>>;
//!     /// `lo..hi` over literals, `usize` consts and value counts.
//!     /// Known ends run forward and `KnownRange::admits` them; value ends
//!     /// are checked by the IR guard.
//!     fn range_bounds(range: &RangeExpr, scope: RangeScope<'_>) -> syn::Result<[Count; 2]>;
//!     /// The refusal of a fold opened with `KERNEL_BINDER_CAPACITY` already open.
//!     fn folds_nested_too_deep(span: Span) -> syn::Error;
//! }
//! /// The AST, validated, and the value of every `const`.
//! struct AnalyzedKernel { pub(crate) def: KernelDef, pub(crate) consts: HashMap<Name, ConstValue> }
//! enum ConstValue { F32(f32), Usize(u64) }
//! /// An entry's parameter as its program sees it: a uniform scalar, or a record of them.
//! enum Parameter<'a> { Scalar(&'a Ident), Count(&'a Ident), Record { name: &'a Ident, id: RecordId, fields: &'a [RecordField] } }
//! /// One uniform scalar: a parameter, or one field of a record parameter.
//! enum Scalar { Param(Ident), Count(Ident), Field(Ident, Ident) }
//! /// One bound at the binding time where its count becomes available.
//! enum Count { Known(u64), Name(Ident) }
//! /// What a fold's bound may name: a `usize` const;
//! /// value names are checked against the body's scope.
//! struct RangeScope<'a> { pub(crate) consts: &'a HashMap<Name, ConstValue> }
//! ```

mod analyze;
mod body;
mod calls;
mod consts;

pub(crate) use analyze::{AnalyzedKernel, ConstValue, Parameter, Scalar, analyze};
pub(crate) use consts::{Count, RangeScope, folds_nested_too_deep, range_bounds};
