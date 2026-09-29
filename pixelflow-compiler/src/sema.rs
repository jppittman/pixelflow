//! # Semantic Analysis
//!
//! Analyzes the AST for semantic correctness: every name resolves, every
//! expression has a type, every call is to a helper with the right arity, and
//! the call graph is a DAG.
//!
//! ## Responsibilities
//!
//! 1. **Items**: the block's `const`s and `fn`s are collected by name, and a
//!    `const` is evaluated at expansion ([`evaluate_consts`]).
//! 2. **Symbol Resolution**: match identifiers to their definitions, with
//!    Rust's lexical scoping ([`crate::symbol::Scopes`], which lowering
//!    resolves through too).
//! 3. **Types**: every expression is an `f32` or a `bool` ([`Ty`]). The IR
//!    keeps its lanes — a `bool` is a mask lane — so the type lives here and
//!    nowhere downstream; what it buys is that `X.select(Y, 7.0)`, which
//!    blended a number as a mask, is a type error rather than plausible
//!    pixels. A third type, `usize`, is a count: a fold's index or a
//!    `usize` const. It is never the type of an expression — a body names
//!    one only to convert it, `i as f32` — because nothing in the language
//!    computes with an index (docs/plans/2026-09-25-the-language-is-kernel.md
//!    §1.3, §1.6).
//! 4. **Calls**: a helper is called at its arity with arguments of its
//!    parameters' types; an entry is not callable; recursion is refused.
//! 5. **Folds**: a fold's bounds are constant and run forwards
//!    ([`range_bounds`]), and its body is a term of its monoid, an `f32` or
//!    a `bool`, in a scope where the closure's parameter is the index.
//! 6. **Integrals**: an integral's bounds are constant `f32`s and an
//!    interval the IR admits ([`interval_bounds`]), and its body is an
//!    `f32`, in a scope where the closure's parameter is the variable, an
//!    `f32`. `monotone_root` takes three `f32`s.
//!
//! ## Symbol Resolution Rules
//!
//! An identifier resolves to the innermost binding of its name in scope:
//! 1. a `let`-bound local → a shared arena id
//! 2. a declared parameter → an entry's is a `Param` bound by the host
//!    function, a helper's is the argument at the call
//! 3. a fold's index → the fold's binder, a `usize`, visible in the fold's
//!    body and nowhere else; an integral's variable → the integral's binder,
//!    an `f32`, likewise
//! 4. a `const` → its value
//! 5. an intrinsic (X, Y) → a coordinate `Var`, in an entry only: a helper
//!    takes its coordinates as arguments, so that application is contramap
//!    (docs/plans/2026-09-25-the-language-is-kernel.md §1.2)
//! 6. otherwise → refused. A kernel body does not capture from the caller's
//!    scope, so a name nothing here binds is an error here, with a span —
//!    not a capture that lowering then refuses without one.
//!
//! Nothing shadows X, Y, a `const` or a `fn` — a parameter, a `let`, a
//! fold's index or an integral's variable of that name is refused — so a
//! coordinate always means the coordinate and an item always means the item.
//! No item takes the name of a function of the language (`integral`, `area`,
//! `monotone_root`), so a call to one always means the language's.
//!
//! ## Output
//!
//! The semantic phase produces an [`AnalyzedKernel`]: the AST, validated,
//! and the value of every `const`.

use crate::PLAN;
use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, CastExpr, ConstItem, Expr, FnItem, FoldExpr, IfExpr,
    IntegralBounds, IntegralExpr, KernelDef, LANGUAGE_FUNCTIONS, LetStmt, MONOTONE_ROOT,
    MethodCallExpr, Param, RangeExpr, Reduction, Role, Spelling, Stmt, UnaryOp,
};
use crate::lower::{LIBRARY_METHODS, Projection};
use crate::symbol::{SymbolKind, SymbolTable};
use pixelflow_ir::{Binder, IntervalFold, OpKind, known_method_names};
use proc_macro2::Span;
use std::collections::HashMap;
use syn::{Ident, Type};

/// The type of a name or an expression in a kernel body.
///
/// Two value types, and the IR has one lane for both: a `bool` is an
/// all-ones or all-zero mask (`OpKind::mask`). The distinction is enforced
/// here because it cannot be enforced there — a mask read as a number is a
/// NaN, and a number used as a mask blends bit patterns.
///
/// The third, `usize`, is a count and not a value. A fold's index is one,
/// and so is a `usize` const. A name of this type is refused wherever a
/// value is expected, and converted only by `i as f32`: the index is an
/// `f32` lane already, so the conversion lowers to nothing, but writing it
/// is what keeps index arithmetic out of the language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// A value.
    F32,
    /// A mask: a comparison produces it, `&` and `|` combine it, an `if`
    /// chooses by it.
    Bool,
    /// A count: a fold's index, or a `usize` const. 64 bits, as `usize` is
    /// on every target the language compiles for.
    Usize,
}

impl Ty {
    /// The type a declaration names, or `None` for one the language does not
    /// have.
    pub fn from_syn(ty: &Type) -> Option<Ty> {
        let Type::Path(path) = ty else {
            return None;
        };
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return None;
        }
        let segment = &path.path.segments[0];
        if !segment.arguments.is_empty() {
            return None;
        }
        match segment.ident.to_string().as_str() {
            "f32" => Some(Ty::F32),
            "bool" => Some(Ty::Bool),
            "usize" => Some(Ty::Usize),
            _ => None,
        }
    }

    /// The value type a declaration names, `f32` or `bool`; `what` begins
    /// the refusal of any other ("a kernel parameter is").
    fn of_a_value(ty: &Type, what: &str) -> syn::Result<Ty> {
        match Ty::from_syn(ty) {
            Some(value @ (Ty::F32 | Ty::Bool)) => Ok(value),
            Some(Ty::Usize) => Err(syn::Error::new_spanned(
                ty,
                format!(
                    "{what} an `f32` or a `bool`\n\
                     \n\
                     note: a `usize` is a count, not a value: a fold's index, or a `usize` \
                     const, which a body converts by `i as f32`"
                ),
            )),
            None => Err(syn::Error::new_spanned(
                ty,
                format!(
                    "{what} an `f32` or a `bool`\n\
                     \n\
                     note: every value in a kernel body is one of the two"
                ),
            )),
        }
    }

    /// The type's name, as written.
    pub fn name(self) -> &'static str {
        match self {
            Ty::F32 => "f32",
            Ty::Bool => "bool",
            Ty::Usize => "usize",
        }
    }
}

/// What a fold's body is, and so what the fold is: the terms of a sum, a
/// product, a minimum or a maximum are `f32`s, and those of `any` and `all`
/// are `bool`s. A fold has its terms' type.
fn term_type(reduction: Reduction) -> (Ty, &'static str) {
    match reduction {
        Reduction::Sum | Reduction::Product | Reduction::Min | Reduction::Max => (
            Ty::F32,
            "a sum, a product, a minimum or a maximum combines `f32`s",
        ),
        Reduction::Any | Reduction::All => (
            Ty::Bool,
            "`any` and `all` combine `bool`s; a comparison gives one",
        ),
    }
}

/// How an `OpKind` method types: what its receiver and arguments are, and
/// what it gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MethodTyping {
    /// Every operand an `f32`, and an `f32` out.
    Arithmetic,
    /// Two `f32`s in, a `bool` out.
    Comparison,
    /// A `bool` receiver chooses between two operands of one type.
    Choice,
}

/// The typing of an `OpKind` a kernel body may call as a method.
pub(crate) fn method_typing(op: OpKind) -> MethodTyping {
    match op {
        OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {
            MethodTyping::Comparison
        }
        OpKind::If => MethodTyping::Choice,
        _ => MethodTyping::Arithmetic,
    }
}

/// A `const`'s value, evaluated at expansion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstValue {
    /// `const NAME: f32`: a value, folded into every body that names it.
    F32(f32),
    /// `const NAME: usize`: a count, for a range's bounds or `NAME as f32`.
    Usize(u64),
}

/// The result of semantic analysis.
#[derive(Debug)]
pub struct AnalyzedKernel {
    /// The original kernel definition.
    pub def: KernelDef,
    /// Every `const`'s value, evaluated at expansion.
    pub consts: HashMap<String, ConstValue>,
}

/// Perform semantic analysis on a parsed kernel.
pub fn analyze(def: KernelDef) -> syn::Result<AnalyzedKernel> {
    let items = Items::collect(&def)?;
    let consts = evaluate_consts(&def.consts, &items.consts)?;
    for f in &def.fns {
        FnAnalyzer::new(f, &items, &consts)?.check(f)?;
    }
    refuse_recursion(&def.fns)?;
    require_an_entry(&def)?;
    Ok(AnalyzedKernel { def, consts })
}

/// A block whose expansion would be empty is refused rather than expanded
/// to nothing.
fn require_an_entry(def: &KernelDef) -> syn::Result<()> {
    let has_entry = def.fns.iter().any(|f| f.role() == Role::Entry);
    if def.spelling == Spelling::Items && !has_entry {
        return Err(syn::Error::new(
            Span::call_site(),
            "a `kernel!` block with no entry emits nothing\n\
             \n\
             note: a `pub fn` is an entry, and the macro emits a host function for each; \
             a private `fn` is a helper, inlined where it is called",
        ));
    }
    Ok(())
}

/// Coordinate names a `kernel!` body may not use: they named the Z and W
/// axes, which a lattice no longer has.
const RETIRED_COORDINATES: [&str; 2] = ["Z", "W"];

/// The projection of the retired axis.
const RETIRED_PROJECTION: &str = "DZ";

/// Method names that meant something in a tier that is gone. `.at()` warped
/// a manifold-typed parameter, `.constant()`/`.collapse()` evaluated one to
/// a field; a kernel composes `Kernel` values instead.
const RETIRED_METHODS: [&str; 3] = ["at", "constant", "collapse"];

/// The one method that is neither an `OpKind` nor a library composition:
/// the identity on an arena value.
const CLONE: &str = "clone";

/// `monotone_root`'s arguments: the height `δ`, and the rise's step and bend.
const MONOTONE_ROOT_ARGUMENTS: usize = 3;

/// Maximum per-character difference for a same-length method name to be
/// suggested as a typo fix (e.g. `sqrtt` -> `sqrt`).
const MAX_TYPO_CHAR_DIFF: usize = 2;

/// A `fn`'s declared types.
#[derive(Debug, Clone)]
struct Signature {
    role: Role,
    params: Vec<Ty>,
    /// `None` only for the closure sugar's entry, whose type is inferred; an
    /// entry is never called, so it is never consulted.
    ret: Option<Ty>,
}

/// The block's items by name: what every body can see besides its own
/// scope.
struct Items {
    /// Every `const`'s declared type: `f32` or `usize`.
    consts: HashMap<String, Ty>,
    fns: HashMap<String, Signature>,
}

impl Items {
    /// Collect the items, refusing a duplicate name, a name that is a
    /// coordinate or a projection, and a declared type the language does
    /// not have.
    fn collect(def: &KernelDef) -> syn::Result<Self> {
        let mut items = Items {
            consts: HashMap::with_capacity(def.consts.len()),
            fns: HashMap::with_capacity(def.fns.len()),
        };
        for c in &def.consts {
            items.refuse_a_taken_name(&c.name)?;
            let ty = match Ty::from_syn(&c.ty) {
                Some(ty @ (Ty::F32 | Ty::Usize)) => ty,
                Some(Ty::Bool) | None => {
                    return Err(syn::Error::new_spanned(
                        &c.ty,
                        "a `const` in a `kernel!` block is an `f32` or a `usize`\n\
                         \n\
                         note: it is evaluated at expansion: an `f32` is folded into every \
                         body that names it, and a `usize` is a count, a range's bound; a \
                         `bool` would be a mask constant, which nothing spells yet",
                    ));
                }
            };
            items.consts.insert(c.name.to_string(), ty);
        }
        for f in &def.fns {
            items.refuse_a_taken_name(&f.name)?;
            let signature = Self::signature(f)?;
            items.fns.insert(f.name.to_string(), signature);
        }
        Ok(items)
    }

    /// An item's name is unique in the block, and is not a coordinate or a
    /// projection.
    fn refuse_a_taken_name(&self, name: &Ident) -> syn::Result<()> {
        let text = name.to_string();
        if SymbolTable::COORDINATES.contains(&text.as_str()) {
            return Err(syn::Error::new(
                name.span(),
                format!("`{text}` is the intrinsic coordinate; an item cannot be named after it"),
            ));
        }
        if Projection::from_name(&text).is_some() {
            return Err(syn::Error::new(
                name.span(),
                format!("`{text}` is a projection; an item cannot be named after it"),
            ));
        }
        if LANGUAGE_FUNCTIONS.contains(&text.as_str()) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{text}` is a function of the language; an item cannot be named after it\n\
                     \n\
                     note: `integral`, `area` and `monotone_root` are the language's (§1.5 of \
                     {PLAN}), and a call to one always means it"
                ),
            ));
        }
        if self.consts.contains_key(&text) || self.fns.contains_key(&text) {
            return Err(syn::Error::new(
                name.span(),
                format!("`{text}` is defined twice in this `kernel!` block"),
            ));
        }
        Ok(())
    }

    /// A `fn`'s parameter and return types. An entry's parameters are `f32`:
    /// each is bound through `Into<Scalar>` by the host function. A helper's
    /// may be `bool` too — a mask is an ordinary argument at an inlined call.
    fn signature(f: &FnItem) -> syn::Result<Signature> {
        let role = f.role();
        let mut params = Vec::with_capacity(f.params.len());
        for param in &f.params {
            params.push(Self::param_type(param, role)?);
        }
        let ret = match &f.ret {
            None => None,
            Some(ty) => Some(Ty::of_a_value(ty, "a kernel `fn` returns")?),
        };
        Ok(Signature { role, params, ret })
    }

    fn param_type(param: &Param, role: Role) -> syn::Result<Ty> {
        let ty = Ty::of_a_value(&param.ty, "a kernel parameter is")?;
        match (role, ty) {
            (Role::Entry, Ty::Bool) => Err(syn::Error::new_spanned(
                &param.ty,
                "an entry's parameter is an `f32`\n\
                 \n\
                 note: an entry's parameters are bound by its host function, each through \
                 `Into<Scalar>`, and a `Scalar` is a number\n\
                 help: take the mask's operands as parameters and compare them in the body",
            )),
            _ => Ok(ty),
        }
    }
}

/// The analysis of one `fn` body: its symbols, and the block's items.
struct FnAnalyzer<'a> {
    items: &'a Items,
    /// The `const`s' values, which a fold's bounds are evaluated over.
    consts: &'a HashMap<String, ConstValue>,
    role: Role,
    symbols: SymbolTable,
}

impl<'a> FnAnalyzer<'a> {
    /// The scope a body opens in: the coordinates, the block's `const`s, and
    /// the `fn`'s own parameters.
    fn new(
        f: &FnItem,
        items: &'a Items,
        consts: &'a HashMap<String, ConstValue>,
    ) -> syn::Result<Self> {
        let mut analyzer = FnAnalyzer {
            items,
            consts,
            role: f.role(),
            symbols: SymbolTable::new(),
        };
        for (name, &ty) in &items.consts {
            analyzer.symbols.register_const(name, ty);
        }
        let signature = &items.fns[&f.name.to_string()];
        for (param, &ty) in f.params.iter().zip(&signature.params) {
            analyzer.register_parameter(param, ty)?;
        }
        Ok(analyzer)
    }

    /// Type the body, and check it against the declared return type.
    fn check(mut self, f: &FnItem) -> syn::Result<()> {
        let Some(want) = self.items.fns[&f.name.to_string()].ret else {
            self.type_of(&f.body)?;
            return Ok(());
        };
        self.expect(
            &f.body,
            want,
            &format!("`{}` declares that it returns `{}`", f.name, want.name()),
        )?;
        Ok(())
    }

    /// Register a parameter in the symbol table.
    fn register_parameter(&mut self, param: &Param, ty: Ty) -> syn::Result<()> {
        let name = param.name.to_string();
        self.refuse_shadowing_an_item(&param.name, "parameter")?;
        if self.symbols.lookup(&name).is_some() {
            return Err(syn::Error::new(
                param.name.span(),
                format!(
                    "duplicate parameter '{}'\n\
                     help: each parameter must have a unique name",
                    name
                ),
            ));
        }
        self.symbols.register_parameter(&name, ty);
        Ok(())
    }

    /// A parameter or a `let` may not take the name of a coordinate, a
    /// `const` or a `fn`. A `let` may shadow a parameter or another `let`,
    /// as Rust's does.
    fn refuse_shadowing_an_item(&self, name: &Ident, binder: &str) -> syn::Result<()> {
        let text = name.to_string();
        // A `let X` would make `X` mean the local below it — and lowering
        // used to match the coordinate names before locals, so the kernel
        // silently read the coordinate instead (`{ let X = Y; X }` gave X).
        if self.symbols.is_intrinsic(&text) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "{binder} `{text}` shadows the intrinsic coordinate variable `{text}`\n\
                     note: intrinsics are: X, Y (coordinate variables), and every use of one \
                     in a kernel body means the coordinate\n\
                     help: rename this {binder} to something else"
                ),
            ));
        }
        let shadows_a_const = self
            .symbols
            .lookup(&text)
            .is_some_and(|s| s.kind == SymbolKind::Const);
        if shadows_a_const {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "{binder} `{text}` shadows the `const {text}` of this block\n\
                     help: rename this {binder} to something else"
                ),
            ));
        }
        if self.items.fns.contains_key(&text) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "{binder} `{text}` shadows the `fn {text}` of this block\n\
                     help: rename this {binder} to something else"
                ),
            ));
        }
        Ok(())
    }

    /// The type of an expression, with every name in it resolved: an `f32`
    /// or a `bool`, never a `usize` — a `usize` name is refused here unless
    /// [`Self::type_of_cast`] is converting it.
    fn type_of(&mut self, expr: &Expr) -> syn::Result<Ty> {
        match expr {
            Expr::Ident(ident_expr) => self.resolve_ident(&ident_expr.name),

            Expr::Literal(literal) => {
                literal.f32_value()?;
                Ok(Ty::F32)
            }

            Expr::Binary(binary) => self.type_of_binary(binary),

            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => self.expect(&unary.operand, Ty::F32, "`-` negates an `f32`"),
            },

            Expr::MethodCall(call) => self.type_of_method_call(call),

            Expr::Call(call) => self.type_of_call(call),

            Expr::If(choice) => self.type_of_if(choice),

            Expr::Fold(fold) => self.type_of_fold(fold),

            Expr::Integral(integral) => self.type_of_integral(integral),

            Expr::Cast(cast) => self.type_of_cast(cast),

            Expr::Block(block) => self.type_of_block(block),

            Expr::Paren(inner) => self.type_of(inner),
        }
    }

    /// `(a..b).map(|i| e).sum()` and its siblings: the bounds are constant
    /// and run forwards, and the body is a term of the fold's monoid, typed
    /// in a scope of its own where the closure's parameter is the index. The
    /// body sees every enclosing binding, an enclosing fold's index among
    /// them, as a Rust closure does.
    fn type_of_fold(&mut self, fold: &FoldExpr) -> syn::Result<Ty> {
        range_bounds(&fold.range, self.consts)?;
        self.refuse_shadowing_an_item(&fold.binder, "a fold's index")?;
        let (term, what) = term_type(fold.reduction);
        self.symbols.push_scope();
        self.symbols.register_index(&fold.binder.to_string());
        let typed = self.expect(&fold.body, term, what);
        self.symbols.pop_scope();
        typed
    }

    /// `integral(lo..hi, |u| e)`, and each integral of `area`: the bounds
    /// are constant and an interval, and the body is an `f32` — an integral
    /// of a mask means nothing — typed in a scope of its own where the
    /// closure's parameter is the variable, an `f32`. The body sees every
    /// enclosing binding, as a fold's does.
    fn type_of_integral(&mut self, integral: &IntegralExpr) -> syn::Result<Ty> {
        match &integral.bounds {
            IntegralBounds::Written(range) => {
                interval_bounds(range, self.consts)?;
            }
            IntegralBounds::Pixel => {}
        }
        self.refuse_shadowing_an_item(&integral.variable, "an integral's variable")?;
        self.symbols.push_scope();
        self.symbols
            .register_variable(&integral.variable.to_string());
        let typed = self.expect(
            &integral.body,
            Ty::F32,
            "an integral's body is its integrand, an `f32`; a mask becomes one by a choice, \
             `if m { 1.0 } else { 0.0 }`",
        );
        self.symbols.pop_scope();
        typed
    }

    /// `monotone_root(δ, step, bend)`: three `f32`s in, the parameter `τ(δ)`
    /// out (pixelflow-ir's `integral::monotone_root`).
    fn type_of_monotone_root(&mut self, call: &CallExpr) -> syn::Result<Ty> {
        if call.args.len() != MONOTONE_ROOT_ARGUMENTS {
            return Err(syn::Error::new(
                call.func.span(),
                format!(
                    "`{MONOTONE_ROOT}` takes {MONOTONE_ROOT_ARGUMENTS} arguments, \
                     `(delta, step, bend)`, but {} {} supplied\n\
                     \n\
                     note: `monotone_root(δ, step, bend)` is the parameter at which the rise \
                     `t·(2·step + bend·t)` reaches the height `δ`",
                    call.args.len(),
                    if call.args.len() == 1 { "was" } else { "were" },
                ),
            ));
        }
        for arg in &call.args {
            self.expect(
                arg,
                Ty::F32,
                &format!(
                    "`{MONOTONE_ROOT}` takes `f32`s: the height, and the rise's step and bend"
                ),
            )?;
        }
        Ok(Ty::F32)
    }

    /// `i as f32`: a `usize`, named, as a value. Nothing else converts —
    /// an `f32` needs no conversion, a `bool` is a mask and becomes a number
    /// by a choice, and no expression computes a `usize` to convert. The
    /// target is `f32` by construction: the parser refuses any other.
    fn type_of_cast(&mut self, cast: &CastExpr) -> syn::Result<Ty> {
        let names_a_count = cast.named().is_some_and(|name| {
            self.symbols
                .lookup(&name.to_string())
                .is_some_and(|symbol| symbol.ty == Ty::Usize)
        });
        if names_a_count {
            return Ok(Ty::F32);
        }
        let found = self.type_of(&cast.operand)?;
        let why = match found {
            Ty::Bool => {
                "a `bool` is a mask; it becomes a number by a choice, `if m { 1.0 } else { 0.0 }`"
            }
            Ty::F32 | Ty::Usize => "it is a value already, and needs no conversion",
        };
        Err(syn::Error::new(
            cast.operand.span(),
            format!(
                "`as f32` converts a `usize`, and this expression's type is `{}`\n\
                 \n\
                 note: {why}\n\
                 note: a `usize` is a fold's index or a `usize` const, converted by name: \
                 `i as f32`",
                found.name()
            ),
        ))
    }

    fn type_of_binary(&mut self, binary: &BinaryExpr) -> syn::Result<Ty> {
        let (want, gives, what) = match binary.op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                (Ty::F32, Ty::F32, "arithmetic takes `f32`s")
            }
            BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Ne => (Ty::F32, Ty::Bool, "a comparison takes `f32`s"),
            BinaryOp::BitAnd | BinaryOp::BitOr => (
                Ty::Bool,
                Ty::Bool,
                "`&` and `|` combine `bool`s; a comparison gives one",
            ),
        };
        self.expect(&binary.lhs, want, what)?;
        self.expect(&binary.rhs, want, what)?;
        Ok(gives)
    }

    /// `if c { a } else { b }`: the condition is a `bool`, the arms agree,
    /// and the value has the arms' type.
    fn type_of_if(&mut self, choice: &IfExpr) -> syn::Result<Ty> {
        self.expect(
            &choice.cond,
            Ty::Bool,
            "the condition of an `if` is a `bool`; a comparison gives one",
        )?;
        let then = self.type_of_block(&choice.then_branch)?;
        self.expect(
            &choice.else_branch,
            then,
            "both arms of an `if` have the same type",
        )
    }

    /// Resolve an identifier reference to the innermost binding of its name
    /// in scope, or refuse it.
    ///
    /// An unknown name used to be accepted here as a capture from the
    /// caller's scope, which lowering then refused as `Unknown identifier` —
    /// one stage accepting what a later one refuses, and the later one has no
    /// span to point with. Worse, it hid an out-of-scope local: the leaked
    /// binding lowering never popped was still there to be found, so
    /// `{ { let a = X; a }; a }` compiled.
    fn resolve_ident(&self, ident: &Ident) -> syn::Result<Ty> {
        let name = ident.to_string();
        if let Some(symbol) = self.symbols.lookup(&name) {
            if symbol.kind == SymbolKind::Intrinsic && self.role == Role::Helper {
                return Err(syn::Error::new(
                    ident.span(),
                    format!(
                        "`{name}` in a helper\n\
                         \n\
                         note: a coordinate appears only in an entry (a `pub fn`); a helper is \
                         a function of its arguments, so that applying it to a shifted \
                         coordinate warps it\n\
                         help: take the coordinate as a parameter, and pass `{name}` at the call"
                    ),
                ));
            }
            if symbol.ty == Ty::Usize {
                return Err(a_count_is_not_a_value(ident));
            }
            return Ok(symbol.ty);
        }
        if self.items.fns.contains_key(&name) {
            return Err(syn::Error::new(
                ident.span(),
                format!(
                    "`{name}` is a function, not a value\n\
                     help: call it: `{name}(…)`"
                ),
            ));
        }
        // `Z` and `W` were coordinate intrinsics until a lattice became two
        // axes; say so, rather than that the name is missing.
        if let Some(axis) = RETIRED_COORDINATES.iter().find(|a| **a == name) {
            return Err(syn::Error::new(
                ident.span(),
                format!(
                    "`{axis}` is no longer a coordinate: a lattice has two axes, X and Y\n\
                     note: a scalar that is the same at every sample is a uniform, not an axis\n\
                     help: declare it as a parameter of this kernel and pass a \
                     `Uniform` handle at the call site"
                ),
            ));
        }
        Err(syn::Error::new(
            ident.span(),
            format!(
                "cannot find `{name}` in this kernel body\n\
                 note: a kernel body sees X, Y (in an entry), its parameters, the block's \
                 `const`s, and the `let` bindings in scope; a `let` inside a block goes out \
                 of scope where the block ends\n\
                 note: a value from the enclosing Rust scope is not captured\n\
                 help: to use an outside value, declare it as a parameter of this kernel \
                 and pass it at the call site"
            ),
        ))
    }

    /// Type a method call: the method by name and arity, then its operands
    /// against what it takes.
    fn type_of_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<Ty> {
        let method_name = call.method.to_string();
        let arg_count = call.args.len();

        if RETIRED_METHODS.contains(&method_name.as_str()) {
            return Err(syn::Error::new(
                call.method.span(),
                format!(
                    "`.{method_name}()` inside a kernel body sampled a manifold-typed \
                     parameter, and there are none\n\
                     help: compose `Kernel` values instead: `Kernel::at` is the warp"
                ),
            ));
        }
        // Arena expressions are values, so `.clone()` is the identity.
        if method_name == CLONE && arg_count == 0 {
            return self.type_of(&call.receiver);
        }

        // Validate method name AND arity against known methods (IR ops +
        // library compositions) — `OpKind::from_method_call` checks arity,
        // so `.sqrt(1)` is rejected here rather than slipping through as
        // "known" and failing later with a less specific error.
        if let Some(op) = OpKind::from_method_call(&method_name, arg_count) {
            return self.check_op_method(op, call);
        }
        if LIBRARY_METHODS.contains(&(method_name.as_str(), arg_count)) {
            return self.all_f32(call, &format!("`.{method_name}` takes `f32`s"));
        }

        Err(self.unknown_method(call))
    }

    /// The operands of an `OpKind` method against its typing.
    fn check_op_method(&mut self, op: OpKind, call: &MethodCallExpr) -> syn::Result<Ty> {
        let method_name = call.method.to_string();
        match method_typing(op) {
            MethodTyping::Arithmetic => {
                self.all_f32(call, &format!("`.{method_name}` takes `f32`s"))
            }
            MethodTyping::Comparison => {
                self.all_f32(call, &format!("`.{method_name}` compares `f32`s"))?;
                Ok(Ty::Bool)
            }
            MethodTyping::Choice => {
                self.expect(
                    &call.receiver,
                    Ty::Bool,
                    &format!(
                        "the receiver of `.{method_name}` is the mask, a `bool`; a \
                         comparison gives one"
                    ),
                )?;
                let [then, otherwise] = call.args.as_slice() else {
                    unreachable!("OpKind::from_method_call resolves a choice at two arguments")
                };
                let arm = self.type_of(then)?;
                self.expect(
                    otherwise,
                    arm,
                    &format!("both arms of `.{method_name}` have the same type"),
                )
            }
        }
    }

    /// Every operand of the call an `f32`, and an `f32` out.
    fn all_f32(&mut self, call: &MethodCallExpr, what: &str) -> syn::Result<Ty> {
        self.expect(&call.receiver, Ty::F32, what)?;
        for arg in &call.args {
            self.expect(arg, Ty::F32, what)?;
        }
        Ok(Ty::F32)
    }

    /// The refusal of a method name nothing advertises, or of a known one at
    /// the wrong arity.
    fn unknown_method(&self, call: &MethodCallExpr) -> syn::Error {
        let method_name = call.method.to_string();
        let arg_count = call.args.len();

        // A recognized name at the wrong arity is not an unknown name, and
        // sending it into the typo search below produced the useless
        // `unknown method 'sqrt'; did you mean 'sqrt'?` — the search found
        // the very name it had just declared unknown. Answer the question
        // the caller actually got wrong.
        if let Some(want) = Self::expected_arg_count(&method_name) {
            return syn::Error::new(
                call.method.span(),
                format!(
                    "`{method_name}` takes {want} argument{}, but {arg_count} \
                     {} supplied",
                    if want == 1 { "" } else { "s" },
                    if arg_count == 1 { "was" } else { "were" },
                ),
            );
        }

        // Find similar method for suggestion - collect all known methods
        let all_methods: Vec<&str> = known_method_names()
            .chain(LIBRARY_METHODS.iter().map(|(name, _)| *name))
            .chain(std::iter::once(CLONE))
            .collect();

        let suggestion = all_methods
            .iter()
            .find(|&&m| {
                let m_lower = m.to_lowercase();
                let name_lower = method_name.to_lowercase();
                m_lower == name_lower
                    || (m.len() == method_name.len()
                        && m.chars()
                            .zip(method_name.chars())
                            .filter(|(a, b)| a != b)
                            .count()
                            <= MAX_TYPO_CHAR_DIFF)
            })
            .copied();

        let msg = match suggestion {
            Some(similar) => format!(
                "unknown method '{}'\n\
                 help: did you mean '{}'?",
                method_name, similar
            ),
            None => format!(
                "unknown method '{}'\n\
                 note: common methods: sqrt, abs, sin, cos, exp, min, max, clone\n\
                 help: see Kernel's method surface for what is available",
                method_name
            ),
        };

        syn::Error::new(call.method.span(), msg)
    }

    /// The argument count a known method takes, or `None` if no method has
    /// that name at any arity.
    ///
    /// Name and arity are separate questions. `OpKind::from_method_call`
    /// deliberately answers them together — that is what makes `.sqrt(1.0)` a
    /// hard error rather than something that slips through and fails later —
    /// but a *diagnostic* has to take them apart again to say which one is
    /// wrong.
    ///
    /// Asking `from_method_call` again at the op's own arity is what
    /// distinguishes a DSL method from an op that merely shares a name
    /// (`add`, `shl`), without this module needing to see the private
    /// predicate that decides it.
    fn expected_arg_count(name: &str) -> Option<usize> {
        if let Some(op) = OpKind::from_name(name) {
            let args = op.arity().checked_sub(1)?;
            if OpKind::from_method_call(name, args).is_some() {
                return Some(args);
            }
        }
        LIBRARY_METHODS
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, count)| *count)
    }

    /// A free function call: a helper, inlined by lowering, or a projection.
    fn type_of_call(&mut self, call: &CallExpr) -> syn::Result<Ty> {
        let name = call.func.to_string();
        if let Some(signature) = self.items.fns.get(&name) {
            return self.type_of_helper_call(call, signature.clone());
        }
        if name == RETIRED_PROJECTION {
            return Err(syn::Error::new(
                call.func.span(),
                "`DZ` is no longer a coordinate: a lattice has two axes, X and Y",
            ));
        }
        if Projection::from_name(&name).is_some() {
            let [arg] = call.args.as_slice() else {
                return Err(syn::Error::new(
                    call.func.span(),
                    format!(
                        "`{name}` takes 1 argument, but {} were supplied",
                        call.args.len()
                    ),
                ));
            };
            return self.expect(arg, Ty::F32, "a projection takes an `f32`");
        }
        if name == MONOTONE_ROOT {
            return self.type_of_monotone_root(call);
        }
        Err(syn::Error::new(
            call.func.span(),
            format!(
                "cannot find a function `{name}` in this `kernel!` block\n\
                 note: a body calls the block's helpers (private `fn`s) and the projections \
                 V, DX, DY, DXX, DXY, DYY\n\
                 note: a function from the enclosing Rust scope is not captured"
            ),
        ))
    }

    /// A call to one of the block's `fn`s: a helper, at its arity, with
    /// arguments of its parameters' types.
    fn type_of_helper_call(&mut self, call: &CallExpr, signature: Signature) -> syn::Result<Ty> {
        let name = call.func.to_string();
        if signature.role == Role::Entry {
            return Err(syn::Error::new(
                call.func.span(),
                format!(
                    "`{name}` is an entry, not a helper\n\
                     \n\
                     note: a `pub fn` is a program of its own, over `X` and `Y`; a body \
                     calls helpers, the block's private `fn`s, which take their coordinates \
                     as arguments\n\
                     help: make `{name}` private, or move what this call needs into a helper"
                ),
            ));
        }
        if call.args.len() != signature.params.len() {
            return Err(syn::Error::new(
                call.func.span(),
                format!(
                    "`{name}` takes {} argument{}, but {} {} supplied",
                    signature.params.len(),
                    if signature.params.len() == 1 { "" } else { "s" },
                    call.args.len(),
                    if call.args.len() == 1 { "was" } else { "were" },
                ),
            ));
        }
        for (position, (arg, &want)) in call.args.iter().zip(&signature.params).enumerate() {
            self.expect(
                arg,
                want,
                &format!(
                    "parameter {} of `{name}` is a `{}`",
                    position + 1,
                    want.name()
                ),
            )?;
        }
        Ok(signature
            .ret
            .expect("a helper declares its return type: the parser requires one"))
    }

    /// Analyze a block expression in a scope of its own.
    fn type_of_block(&mut self, block: &BlockExpr) -> syn::Result<Ty> {
        self.symbols.push_scope();
        let analyzed = self.type_of_block_contents(block);
        self.symbols.pop_scope();
        analyzed
    }

    /// A block's statements in order, then its value, in the scope
    /// [`Self::type_of_block`] opened.
    fn type_of_block_contents(&mut self, block: &BlockExpr) -> syn::Result<Ty> {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(let_stmt) => self.analyze_let(let_stmt)?,
                Stmt::Expr(expr) => {
                    self.type_of(expr)?;
                }
            }
        }
        match &block.expr {
            Some(expr) => self.type_of(expr),
            None => Err(syn::Error::new(
                block.span,
                "a block with no final expression has no value\n\
                 \n\
                 note: a kernel body is an expression; the last statement of a block, \
                 without its `;`, is the block's value",
            )),
        }
    }

    /// Analyze a let statement: the initializer is typed, checked against an
    /// annotation if there is one, and the name is bound to that type.
    fn analyze_let(&mut self, let_stmt: &LetStmt) -> syn::Result<()> {
        self.refuse_shadowing_an_item(&let_stmt.name, "`let`")?;

        // The initializer is analyzed before the binding exists, so it sees
        // whatever the name meant before: `let a = a + 1.0;`.
        let found = match &let_stmt.ty {
            None => self.type_of(&let_stmt.init)?,
            Some(annotation) => {
                let want = Ty::of_a_value(annotation, "a `let` in a kernel body is")?;
                self.expect(
                    &let_stmt.init,
                    want,
                    &format!("`{}` is declared as a `{}`", let_stmt.name, want.name()),
                )?
            }
        };
        self.symbols
            .register_local(&let_stmt.name.to_string(), found);
        Ok(())
    }

    /// Type `expr`, and refuse it unless it is `want`, saying what `what`
    /// needed. Every position of the language that needs one type goes
    /// through here, so a type error reads the same everywhere.
    fn expect(&mut self, expr: &Expr, want: Ty, what: &str) -> syn::Result<Ty> {
        let found = self.type_of(expr)?;
        if found == want {
            return Ok(found);
        }
        Err(syn::Error::new(
            expr.span(),
            format!(
                "mismatched types: expected `{}`, found `{}`\n\
                 \n\
                 note: {what}",
                want.name(),
                found.name()
            ),
        ))
    }
}

/// A `usize` where a value is expected. A `usize` — a fold's index, or a
/// `usize` const — is a value only as `i as f32`: nothing in the language
/// computes with one, because there is nothing to index (plan §1.6).
fn a_count_is_not_a_value(name: &Ident) -> syn::Error {
    syn::Error::new(
        name.span(),
        format!(
            "mismatched types: `{name}` is a `usize`, where a value is expected\n\
             \n\
             note: a `usize` (a fold's index, or a `usize` const) becomes a value by an \
             explicit conversion, `{name} as f32`, and by nothing else: there is no arithmetic \
             on a `usize`, no comparison of one, and no table to index (§1.6 of {PLAN})\n\
             help: write `{name} as f32`"
        ),
    )
}

// ───────────────────────────── consts ─────────────────────────────

/// Integer arithmetic at expansion: what a `usize` const's initializer and a
/// fold's bounds are built from. Each operation is checked, as rustc's const
/// evaluator checks it, so a count that would wrap is an error rather than a
/// different count.
///
/// A trait because the two differ only in what a name means: a `const`
/// being evaluated may name another, evaluated on demand; a bound names a
/// `usize` const, already evaluated, or it is not constant.
trait UsizeScope {
    /// The value of `name` in a `usize` expression, or why it has none.
    fn usize_named(&mut self, name: &Ident) -> syn::Result<u64>;

    /// The refusal of something a `usize` expression is not built from.
    fn not_a_count(&self, span: Span) -> syn::Error;

    /// An integer literal, a name, `+ - * /` and parentheses.
    fn eval_usize(&mut self, expr: &Expr) -> syn::Result<u64> {
        let binary = match expr {
            Expr::Literal(literal) => return literal.usize_value(),
            Expr::Ident(ident) => return self.usize_named(&ident.name),
            Expr::Paren(inner) => return self.eval_usize(inner),
            // Constant, but no count: rustc's refusal of the same tokens.
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => {
                    return Err(syn::Error::new(
                        unary.span,
                        "cannot apply unary operator `-` to type `usize`\n\
                         \n\
                         note: unsigned values cannot be negated; a range's bounds and a \
                         `usize` const are counts",
                    ));
                }
            },
            Expr::Binary(binary) => binary,
            other => return Err(self.not_a_count(other.span())),
        };
        let lhs = self.eval_usize(&binary.lhs)?;
        let rhs = self.eval_usize(&binary.rhs)?;
        let (value, symbol) = match binary.op {
            BinaryOp::Add => (lhs.checked_add(rhs), "+"),
            BinaryOp::Sub => (lhs.checked_sub(rhs), "-"),
            BinaryOp::Mul => (lhs.checked_mul(rhs), "*"),
            BinaryOp::Div if rhs == 0 => {
                return Err(syn::Error::new(
                    binary.span,
                    format!("attempt to divide `{lhs}_usize` by zero"),
                ));
            }
            BinaryOp::Div => (Some(lhs / rhs), "/"),
            BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::BitAnd
            | BinaryOp::BitOr => return Err(self.not_a_count(binary.span)),
        };
        value.ok_or_else(|| {
            syn::Error::new(
                binary.span,
                format!(
                    "attempt to compute `{lhs}_usize {symbol} {rhs}_usize`, which would overflow"
                ),
            )
        })
    }
}

/// A fold's bounds, evaluated at expansion: each a `usize` built from
/// integer literals and `usize` consts, and the range running forwards.
///
/// The one evaluation of them: `sema` checks a fold with it, and lowering
/// reads the bounds it returns, so the two cannot disagree on a range.
///
/// Ranges are constant (plan §1.5): the program's shape is known when it is
/// compiled. A reversed range is refused rather than read as the empty fold
/// it would be in Rust: an empty range is written `a..a`, and `b..a` is
/// almost always a slip (clippy's `reversed_empty_ranges` is deny-by-default
/// for the same reason), and the IR's own `RangeFold` refuses one.
pub(crate) fn range_bounds(
    range: &RangeExpr,
    consts: &HashMap<String, ConstValue>,
) -> syn::Result<(u64, u64)> {
    let mut scope = RangeScope { consts };
    let lo = scope.eval_usize(&range.lo)?;
    let hi = scope.eval_usize(&range.hi)?;
    if lo > hi {
        return Err(syn::Error::new(
            range.span,
            format!(
                "the range `{lo}..{hi}` runs backwards\n\
                 \n\
                 note: `{lo}..{hi}` holds no index, so a fold over it would be its monoid's \
                 identity; the empty range is written `a..a`, and a reversed one is refused as \
                 the slip it usually is"
            ),
        ));
    }
    Ok((lo, hi))
}

/// What a fold's bound may name: a `usize` const, already evaluated.
struct RangeScope<'a> {
    consts: &'a HashMap<String, ConstValue>,
}

impl UsizeScope for RangeScope<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<u64> {
        match self.consts.get(&name.to_string()) {
            Some(ConstValue::Usize(value)) => Ok(*value),
            Some(ConstValue::F32(_)) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const, and a range's bounds are `usize`s\n\
                     \n\
                     help: declare the count as `const {name}: usize`"
                ),
            )),
            None => Err(syn::Error::new(
                name.span(),
                format!(
                    "a range's bounds are constant, and `{name}` is not a `const`\n\
                     \n\
                     note: a bound is evaluated at expansion, from integer literals, `usize` \
                     consts, `+ - * /` and parentheses; a fold's index, a parameter, a `let` \
                     and a coordinate are not constant\n\
                     note: ranges are constant: a program's shape is known when it is \
                     compiled (§1.5 of {PLAN})"
                ),
            )),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        syn::Error::new(
            span,
            format!(
                "a range's bounds are constant\n\
                 \n\
                 note: a bound is evaluated at expansion, from integer literals, `usize` \
                 consts, `+ - * /` and parentheses\n\
                 note: ranges are constant: a program's shape is known when it is compiled \
                 (§1.5 of {PLAN})"
            ),
        )
    }
}

// ───────────────────────── f32 constants ─────────────────────────

/// `f32` arithmetic at expansion: what an `f32` const's initializer and an
/// integral's bounds are built from — literals, `f32` consts, a `usize`
/// const `as f32`, `+ - * /`, unary `-` and parentheses — each operation in
/// `f32`, as rustc's const evaluator does it (see [`evaluate_consts`]).
///
/// A trait, as [`UsizeScope`] is, because the two differ only in what a
/// name means: a `const` being evaluated may name another, evaluated on
/// demand; a bound names an `f32` const, already evaluated, or it is not
/// constant.
trait F32Scope: UsizeScope {
    /// The value of `name` in an `f32` expression, or why it has none.
    fn f32_named(&mut self, name: &Ident) -> syn::Result<f32>;

    /// The refusal of something an `f32` constant is not built from.
    fn not_an_f32_constant(&self, span: Span) -> syn::Error;

    fn eval_f32(&mut self, expr: &Expr) -> syn::Result<f32> {
        match expr {
            Expr::Literal(literal) => literal.f32_value(),
            Expr::Ident(ident) => self.f32_named(&ident.name),
            Expr::Paren(inner) => self.eval_f32(inner),
            // Rust's own `as`, which rounds to the nearest `f32` as rustc's
            // does: exact up to 2²⁴.
            Expr::Cast(cast) => {
                let Some(name) = cast.named() else {
                    return Err(self.not_an_f32_constant(cast.span));
                };
                Ok(self.usize_named(name)? as f32)
            }
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => Ok(-self.eval_f32(&unary.operand)?),
            },
            Expr::Binary(binary) => {
                let lhs = self.eval_f32(&binary.lhs)?;
                let rhs = self.eval_f32(&binary.rhs)?;
                match binary.op {
                    BinaryOp::Add => Ok(lhs + rhs),
                    BinaryOp::Sub => Ok(lhs - rhs),
                    BinaryOp::Mul => Ok(lhs * rhs),
                    BinaryOp::Div => Ok(lhs / rhs),
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr => Err(self.not_an_f32_constant(binary.span)),
                }
            }
            other => Err(self.not_an_f32_constant(other.span())),
        }
    }
}

/// An integral's bounds, evaluated at expansion: each an `f32` constant
/// ([`F32Scope`]), and the two an interval pixelflow-ir admits.
///
/// The one evaluation of them: `sema` checks an integral with it, and
/// lowering reads the bounds it returns, so the two cannot disagree on an
/// interval.
///
/// Bounds are constant (plan §1.5), as a range's are. Whether they make an
/// interval is `IntervalFold::try_new`'s to say — the contract's one
/// definition, asked here at expansion so a refusal is an error at the
/// bounds rather than a panic when the kernel is built. The contract does
/// not depend on which index an integral binds, so any binder asks it;
/// lowering asks again, with the binder it chose. Only the explanation of a
/// refusal is this function's own.
pub(crate) fn interval_bounds(
    range: &RangeExpr,
    consts: &HashMap<String, ConstValue>,
) -> syn::Result<(f32, f32)> {
    let mut scope = IntervalScope { consts };
    let lo = scope.eval_f32(&range.lo)?;
    let hi = scope.eval_f32(&range.hi)?;
    let admitted = Binder::all()
        .next()
        .and_then(|binder| IntervalFold::try_new(binder, lo, hi))
        .is_some();
    if admitted {
        return Ok((lo, hi));
    }
    let why = why_not_an_interval(lo, hi);
    Err(syn::Error::new(
        range.span,
        format!(
            "this integral's interval is not one the IR admits\n\
             \n\
             note: {why}\n\
             note: an integral is over `lo..hi` with finite ends, `lo < hi`, and a finite length \
             (`IntervalFold::try_new`)"
        ),
    ))
}

/// Why `lo..hi` is not an interval: the explanation of a refusal
/// `IntervalFold::try_new` made, which is the one that decides.
fn why_not_an_interval(lo: f32, hi: f32) -> String {
    if !(lo.is_finite() && hi.is_finite()) {
        return format!(
            "an end of `{lo:?}..{hi:?}` is not finite: an integral is over a bounded interval"
        );
    }
    if lo == hi {
        return format!(
            "`{lo:?}..{hi:?}` is empty: its integral would be `0.0` whatever the body, and an \
             empty interval is unrepresentable in the IR"
        );
    }
    if lo > hi {
        return format!(
            "the interval `{lo:?}..{hi:?}` runs backwards: an integral is over `lo..hi` with \
             `lo < hi`"
        );
    }
    format!(
        "the length of `{lo:?}..{hi:?}`, `hi - lo`, overflows `f32`: an integral's interval \
         has a finite measure"
    )
}

/// What an integral's bound may name: a const, already evaluated — an `f32`
/// one, or a `usize` one `as f32`.
struct IntervalScope<'a> {
    consts: &'a HashMap<String, ConstValue>,
}

impl IntervalScope<'_> {
    /// The refusal of a name that is not a `const` in a bound.
    fn not_a_const(name: &Ident) -> syn::Error {
        syn::Error::new(
            name.span(),
            format!(
                "an integral's bounds are constant, and `{name}` is not a `const`\n\
                 \n\
                 note: a bound is evaluated at expansion, from literals, `f32` consts, a `usize` \
                 const `as f32`, `+ - * /`, unary `-` and parentheses; a variable, a fold's \
                 index, a parameter, a `let` and a coordinate are not constant\n\
                 note: integral bounds are constant: a program's shape is known when it is \
                 compiled (§1.5 of {PLAN})"
            ),
        )
    }
}

impl UsizeScope for IntervalScope<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<u64> {
        match self.consts.get(&name.to_string()) {
            Some(ConstValue::Usize(value)) => Ok(*value),
            Some(ConstValue::F32(_)) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const already: `as f32` converts a `usize`\n\
                     \n\
                     help: write `{name}`"
                ),
            )),
            None => Err(Self::not_a_const(name)),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        self.not_an_f32_constant(span)
    }
}

impl F32Scope for IntervalScope<'_> {
    fn f32_named(&mut self, name: &Ident) -> syn::Result<f32> {
        match self.consts.get(&name.to_string()) {
            Some(ConstValue::F32(value)) => Ok(*value),
            Some(ConstValue::Usize(_)) => Err(a_count_is_not_a_value(name)),
            None => Err(Self::not_a_const(name)),
        }
    }

    fn not_an_f32_constant(&self, span: Span) -> syn::Error {
        syn::Error::new(
            span,
            format!(
                "integral bounds are constant\n\
                 \n\
                 note: a bound is evaluated at expansion, from literals, `f32` consts, a `usize` \
                 const `as f32`, `+ - * /`, unary `-` and parentheses, each operation in `f32`\n\
                 note: a program's shape is known when it is compiled (§1.5 of {PLAN})"
            ),
        )
    }
}

/// Every `const`'s value.
///
/// A `const` is evaluated here, at expansion, and gets the value rustc gives
/// the same tokens. An `f32` const is evaluated per operation in `f32`,
/// since rustc evaluates an `f32` `const` in `f32` too. The discipline is
/// exactly the one rustc's const evaluator has: each operation rounds once,
/// in `f32`, and a product and a sum are never contracted into one rounding
/// — which matters because this crate is built with `-fp-contract=fast`
/// (`.cargo/config.toml`), and an FMA gives `b * c + d` a value two
/// roundings never reach. Its literals were rounded once by the parser, and
/// nothing here rounds again. A `usize` const is evaluated in 64-bit
/// integers, each operation checked ([`UsizeScope`]). A const may name
/// another declared anywhere in the block; a cycle is refused.
fn evaluate_consts(
    consts: &[ConstItem],
    types: &HashMap<String, Ty>,
) -> syn::Result<HashMap<String, ConstValue>> {
    let mut evaluator = ConstEvaluator {
        items: consts.iter().map(|c| (c.name.to_string(), c)).collect(),
        types,
        values: HashMap::with_capacity(consts.len()),
        in_progress: Vec::new(),
    };
    for c in consts {
        evaluator.value_of(&c.name)?;
    }
    Ok(evaluator.values)
}

struct ConstEvaluator<'a> {
    items: HashMap<String, &'a ConstItem>,
    /// Each const's declared type, which decides how it is evaluated.
    types: &'a HashMap<String, Ty>,
    values: HashMap<String, ConstValue>,
    /// The consts whose initializers are being evaluated, outermost first:
    /// naming one of them again is a cycle.
    in_progress: Vec<String>,
}

impl UsizeScope for ConstEvaluator<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<u64> {
        match self.value_of(name)? {
            ConstValue::Usize(value) => Ok(value),
            ConstValue::F32(_) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const, where a `usize` is expected\n\
                     \n\
                     note: a `usize` const is built from integers and other `usize` consts"
                ),
            )),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        Self::not_constant(span)
    }
}

impl F32Scope for ConstEvaluator<'_> {
    /// The `f32` const `name`'s value; a `usize` const is a count, and is a
    /// value only as `name as f32`.
    fn f32_named(&mut self, name: &Ident) -> syn::Result<f32> {
        match self.value_of(name)? {
            ConstValue::F32(value) => Ok(value),
            ConstValue::Usize(_) => Err(a_count_is_not_a_value(name)),
        }
    }

    fn not_an_f32_constant(&self, span: Span) -> syn::Error {
        Self::not_constant(span)
    }
}

impl ConstEvaluator<'_> {
    /// The value of the const `name`, evaluating it on first demand.
    fn value_of(&mut self, name: &Ident) -> syn::Result<ConstValue> {
        let key = name.to_string();
        if let Some(&value) = self.values.get(&key) {
            return Ok(value);
        }
        let Some(item) = self.items.get(&key).copied() else {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "cannot find a `const` named `{key}`\n\
                     \n\
                     note: a `const` initializer is evaluated at expansion, so it names \
                     only literals and other `const`s of this block"
                ),
            ));
        };
        if self.in_progress.contains(&key) {
            let cycle = self.in_progress.join("` → `");
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{key}` depends on itself\n\
                     \n\
                     note: `{cycle}` → `{key}`"
                ),
            ));
        }
        self.in_progress.push(key.clone());
        let value = match self.types.get(&key) {
            Some(Ty::Usize) => ConstValue::Usize(self.eval_usize(&item.init)?),
            Some(Ty::F32) => ConstValue::F32(self.eval_f32(&item.init)?),
            Some(Ty::Bool) | None => {
                return Err(syn::Error::new_spanned(
                    &item.ty,
                    "a `const` in a `kernel!` block is an `f32` or a `usize`",
                ));
            }
        };
        self.in_progress.pop();
        self.values.insert(key, value);
        Ok(value)
    }

    fn not_constant(span: Span) -> syn::Error {
        syn::Error::new(
            span,
            "a `const` initializer is evaluated at expansion\n\
             \n\
             note: an `f32` const is built from literals, other `f32` consts, a `usize` \
             const `as f32`, `+ - * /`, unary `-` and parentheses, each operation in `f32`\n\
             note: a `usize` const is built from integer literals, other `usize` consts, \
             `+ - * /` and parentheses, each operation checked",
        )
    }
}

// ─────────────────────────── the call graph ───────────────────────────

/// The call graph over the block's `fn`s is a DAG: a helper is inlined at
/// each call, and a cycle would inline forever.
fn refuse_recursion(fns: &[FnItem]) -> syn::Result<()> {
    let names: Vec<String> = fns.iter().map(|f| f.name.to_string()).collect();
    let calls: HashMap<String, Vec<(String, Span)>> = fns
        .iter()
        .map(|f| {
            let mut out = Vec::new();
            collect_calls(&f.body, &names, &mut out);
            (f.name.to_string(), out)
        })
        .collect();
    let mut walk = CallGraphWalk {
        calls: &calls,
        finished: Vec::new(),
        path: Vec::new(),
    };
    for name in &names {
        walk.visit(name)?;
    }
    Ok(())
}

/// A depth-first walk of the call graph, refusing the first edge that
/// closes a cycle.
struct CallGraphWalk<'a> {
    calls: &'a HashMap<String, Vec<(String, Span)>>,
    /// Every `fn` whose reachable calls are known to be acyclic.
    finished: Vec<String>,
    /// The `fn`s whose bodies the walk is inside, outermost first.
    path: Vec<String>,
}

impl CallGraphWalk<'_> {
    fn visit(&mut self, name: &str) -> syn::Result<()> {
        if self.finished.iter().any(|f| f == name) {
            return Ok(());
        }
        self.path.push(name.to_string());
        for (callee, span) in &self.calls[name] {
            if let Some(start) = self.path.iter().position(|f| f == callee) {
                let cycle = self.path[start..].join("` calls `");
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "recursion is refused: the language is a DAG\n\
                         \n\
                         note: `{cycle}` calls `{callee}`\n\
                         note: a helper is inlined at each call, so a cycle would inline \
                         forever; a bounded reduction is a fold, `(a..b).map(|i| e).sum()`"
                    ),
                ));
            }
            self.visit(callee)?;
        }
        self.path.pop();
        self.finished.push(name.to_string());
        Ok(())
    }
}

/// Every call in `expr` to one of the block's `fn`s, with its span.
fn collect_calls(expr: &Expr, fns: &[String], out: &mut Vec<(String, Span)>) {
    match expr {
        Expr::Ident(_) | Expr::Literal(_) => {}
        Expr::Binary(binary) => {
            collect_calls(&binary.lhs, fns, out);
            collect_calls(&binary.rhs, fns, out);
        }
        Expr::Unary(unary) => collect_calls(&unary.operand, fns, out),
        Expr::MethodCall(call) => {
            collect_calls(&call.receiver, fns, out);
            for arg in &call.args {
                collect_calls(arg, fns, out);
            }
        }
        Expr::Call(call) => {
            let name = call.func.to_string();
            if fns.contains(&name) {
                out.push((name, call.func.span()));
            }
            for arg in &call.args {
                collect_calls(arg, fns, out);
            }
        }
        Expr::If(choice) => {
            collect_calls(&choice.cond, fns, out);
            collect_block_calls(&choice.then_branch, fns, out);
            collect_calls(&choice.else_branch, fns, out);
        }
        // A bound is constant, so it calls nothing; a body may.
        Expr::Fold(fold) => collect_calls(&fold.body, fns, out),
        Expr::Integral(integral) => collect_calls(&integral.body, fns, out),
        Expr::Cast(cast) => collect_calls(&cast.operand, fns, out),
        Expr::Block(block) => collect_block_calls(block, fns, out),
        Expr::Paren(inner) => collect_calls(inner, fns, out),
    }
}

fn collect_block_calls(block: &BlockExpr, fns: &[String], out: &mut Vec<(String, Span)>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(let_stmt) => collect_calls(&let_stmt.init, fns, out),
            Stmt::Expr(expr) => collect_calls(expr, fns, out),
        }
    }
    if let Some(expr) = &block.expr {
        collect_calls(expr, fns, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use proc_macro2::TokenStream;
    use quote::quote;

    /// `sema`'s refusal of `input`, as text.
    fn refusal(input: TokenStream) -> String {
        let def = parse(input).expect("the input parses");
        analyze(def).expect_err("sema refuses it").to_string()
    }

    fn accepted(input: TokenStream) -> AnalyzedKernel {
        let def = parse(input).expect("the input parses");
        analyze(def).expect("sema accepts it")
    }

    /// A body that names `Z` or `W` is a compile error that says where the
    /// value goes instead — not a capture from the caller's scope, which is
    /// what an unknown name would otherwise become.
    #[test]
    fn a_body_naming_a_retired_axis_is_refused_with_the_uniform_note() {
        for body in [quote! { || X + Z }, quote! { || X * W }] {
            let text = refusal(body);
            assert!(
                text.contains("no longer a coordinate") && text.contains("Uniform"),
                "the message must point at uniforms, got: {text}"
            );
        }
    }

    /// A parameter may still be *called* Z: the refusal is about the
    /// intrinsic that is gone, not about the letter.
    #[test]
    fn a_parameter_named_z_is_ordinary() {
        accepted(quote! { |Z: f32| X + Z });
    }

    #[test]
    fn analyze_simple_kernel() {
        accepted(quote! { |r: f32| X * X + Y * Y - r });
    }

    /// A name nothing in the kernel binds is refused here, with its span.
    ///
    /// This test used to pin the opposite — `analyze` accepted an unknown name
    /// as a capture from the caller's scope — while documenting that the
    /// kernel did not compile, because arena lowering has no node for a
    /// captured Rust binding and refused it as `Unknown identifier`. That is
    /// the shape of the `round`/`log10`/`pow` and `fract`/`hypot`/`clamp`
    /// defects: one stage accepting what a later stage refuses. It also hid
    /// an out-of-scope local (see the next test). A capture is still
    /// expressible — the emitted tokens sit in the caller's scope, so it
    /// could fold as a `Const` exactly as a parameter does — and when it is
    /// built, it is built in both stages at once. Pass it as a parameter
    /// meanwhile, which is what the message says.
    #[test]
    fn an_unknown_name_is_refused_not_captured() {
        let text = refusal(quote! { |r: f32| X * X + captured_from_env });
        assert!(
            text.contains("cannot find `captured_from_env`") && text.contains("parameter"),
            "the message must name the identifier and the way out, got: {text}"
        );
    }

    /// Probe p15. A `let` inside a block goes out of scope where the block
    /// ends, and a use after it is an error, as rustc makes it one. Before
    /// the fix this compiled and read the leaked binding: 3 at `X = 3`.
    #[test]
    fn a_local_used_after_its_block_ends_is_refused() {
        let err = refusal(quote! {
            || {
                {
                    let a = X;
                    a
                };
                a
            }
        });
        assert!(err.contains("cannot find `a`"), "got: {err}");
    }

    /// Probe p5. `let X` and `let Y` are refused, as a parameter named X or Y
    /// is. Before the fix `{ let X = Y; X }` compiled and read the
    /// coordinate X, because lowering matched the coordinate names before
    /// locals.
    #[test]
    fn a_let_named_after_a_coordinate_is_refused() {
        for input in [
            quote! { || { let X = Y; X } },
            quote! { || { let Y: f32 = X; Y } },
            quote! { |r: f32| r + { let X = r; X } },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("shadows the intrinsic coordinate"),
                "got: {err}"
            );
        }
    }

    /// A parameter shadowed by a `let` in an inner block is visible again
    /// after the block. The table used to drop the name outright when the
    /// inner block ended; with unknown names now refused, that would have
    /// turned a correct kernel into an error.
    #[test]
    fn a_parameter_shadowed_in_an_inner_block_is_in_scope_after_it() {
        accepted(quote! { |r: f32| ({ let r = X; r }) + r });
    }

    /// A `let`'s initializer sees the binding it is about to shadow.
    #[test]
    fn a_let_initializer_sees_the_binding_it_shadows() {
        accepted(quote! { |r: f32| { let r = r * 2.0; let a = X; let a = a + r; a } });
    }

    #[test]
    fn error_on_shadowing_intrinsic() {
        let err = refusal(quote! { |X: f32| X * X }); // X shadows the intrinsic
        assert!(err.contains("shadows the intrinsic"), "got: {err}");
    }

    #[test]
    fn block_scoping() {
        accepted(quote! {
            |cx: f32| {
                let dx = X - cx;
                dx * dx
            }
        });
    }

    #[test]
    fn error_on_unknown_method() {
        let err = refusal(quote! { |r: f32| X.unknownmethod() });
        assert!(err.contains("unknown method"));
    }

    #[test]
    fn typo_suggestion_for_method() {
        // "sqrtt" should suggest "sqrt"
        let err = refusal(quote! { || X.sqrtt() });
        assert!(err.contains("unknown method"));
    }

    #[test]
    fn typo_suggestion_for_method_matches_case_insensitively() {
        // "SQRT" differs from "sqrt" in every char position case-sensitively,
        // so only the case-insensitive fallback catches it.
        let err = refusal(quote! { || X.SQRT() });
        assert!(err.contains("did you mean 'sqrt'"), "{err}");
    }

    #[test]
    fn typo_suggestion_for_method_names_the_exact_match_at_the_two_char_diff_boundary() {
        // "bba" differs from "abs" in exactly 2 chars (position 0: b vs a,
        // position 2: a vs s; position 1 matches) and differs by 3 from
        // every other same-length method name (add, sub, mul, div, neg,
        // min, max, sin, cos, tan, exp, pow, shl, shr) — an unambiguous
        // 2-char-diff match.
        let err = refusal(quote! { || X.bba() });
        assert!(err.contains("did you mean 'abs'"), "{err}");
    }

    #[test]
    fn typo_suggestion_for_method_is_absent_when_no_same_length_method_is_close() {
        // "qqq" shares a length with several 3-letter methods (sin, cos,
        // tan, abs, neg) but differs from every one of them in all 3 chars —
        // matching length alone must not be enough to suggest one.
        let err = refusal(quote! { || X.qqq() });
        assert!(!err.contains("did you mean"), "{err}");
    }

    #[test]
    fn known_methods_accepted() {
        accepted(quote! { || X.sqrt().abs().sin().cos().clone() });
    }

    #[test]
    fn error_on_duplicate_parameter() {
        let err = refusal(quote! { |r: f32, r: f32| X - r });
        assert!(err.contains("duplicate parameter"));
    }

    // ───────────────────────────── types ─────────────────────────────

    /// Probe p16. `X.select(Y, 7.0)` blended a number as a mask and gave 5
    /// at (2, 1): a value neither arm held. It is a type error now, at the
    /// receiver.
    #[test]
    fn a_number_used_as_a_mask_is_a_type_error() {
        let err = refusal(quote! { || X.select(Y, 7.0) });
        assert!(
            err.contains("expected `bool`, found `f32`") && err.contains("mask"),
            "got: {err}"
        );
        let err = refusal(quote! { || if X { Y } else { 7.0 } });
        assert!(
            err.contains("expected `bool`, found `f32`") && err.contains("condition"),
            "got: {err}"
        );
        let err = refusal(quote! { || X & Y });
        assert!(
            err.contains("expected `bool`, found `f32`") && err.contains("`&`"),
            "got: {err}"
        );
    }

    /// The reverse confusion: a mask where a number is needed.
    #[test]
    fn a_mask_used_as_a_number_is_a_type_error() {
        let cases: [(TokenStream, &str); 5] = [
            (quote! { || X + (X < Y) }, "arithmetic"),
            (quote! { || -(X < Y) }, "negates"),
            (quote! { || (X < Y).sqrt() }, "`.sqrt` takes"),
            (quote! { || (X < Y) < Y }, "comparison"),
            (quote! { || DX(X < Y) }, "projection"),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(
                err.contains("expected `f32`, found `bool`") && err.contains(expected),
                "expected `{expected}`, got: {err}"
            );
        }
    }

    /// The arms of a choice agree, whichever spelling chooses.
    #[test]
    fn the_arms_of_a_choice_have_one_type() {
        let err = refusal(quote! { || if X < Y { X } else { X < Y } });
        assert!(err.contains("both arms of an `if`"), "got: {err}");
        let err = refusal(quote! { || (X < Y).select(X < Y, X) });
        assert!(err.contains("both arms of `.select`"), "got: {err}");
        // A choice between masks is a mask.
        accepted(quote! { || if X < Y { X < 2.0 } else { Y < 2.0 } });
        accepted(quote! { || (X < Y).select(X < 2.0, Y < 2.0) });
    }

    /// A `let` annotation is checked, and a `bool` local is a `bool`.
    #[test]
    fn a_let_annotation_is_checked() {
        accepted(quote! { || { let m: bool = X < Y; let v: f32 = X; if m { v } else { Y } } });
        let err = refusal(quote! { || { let m: f32 = X < Y; m } });
        assert!(
            err.contains("expected `f32`, found `bool`") && err.contains("`m` is declared"),
            "got: {err}"
        );
        let err = refusal(quote! { || { let m: u32 = X; m } });
        assert!(err.contains("`f32` or a `bool`"), "got: {err}");
    }

    /// A parameter's declared type is honored: an `f32` is a number, and
    /// there is no other type for an entry's parameters to be.
    #[test]
    fn a_parameter_type_is_honored() {
        let err = refusal(quote! { |n: i32| X + n });
        assert!(err.contains("`f32` or a `bool`"), "got: {err}");
        let err = refusal(quote! { |m: bool| if m { X } else { Y } });
        assert!(
            err.contains("an entry's parameter is an `f32`"),
            "got: {err}"
        );
        // A helper may take a mask.
        accepted(quote! {
            fn pick(m: bool, a: f32, b: f32) -> f32 { if m { a } else { b } }
            pub fn f() -> f32 { pick(X < Y, X, Y) }
        });
    }

    // ───────────────────────── items and helpers ─────────────────────────

    /// A block's `const`s are evaluated at expansion, per operation in `f32`,
    /// in whatever order they name each other.
    #[test]
    fn consts_are_evaluated_at_expansion() {
        let analyzed = accepted(quote! {
            const NEARLY_ONE: f32 = 1.0 - SNAP;
            const SNAP: f32 = 1.0 / 1024.0;
            pub const NEG: f32 = -(SNAP * 2.0);
            pub fn f() -> f32 { X * NEARLY_ONE + NEG }
        });
        assert_eq!(analyzed.consts["SNAP"], ConstValue::F32(1.0 / 1024.0));
        assert_eq!(
            analyzed.consts["NEARLY_ONE"],
            ConstValue::F32(1.0 - 1.0 / 1024.0)
        );
        assert_eq!(analyzed.consts["NEG"], ConstValue::F32(-(2.0 / 1024.0)));
    }

    /// Each operation of a `const` is an `f32` operation, as rustc's is.
    /// The first `+ 1.0` ties to even in `f32` and stays at `2²⁴`, and the
    /// second must too; an evaluator carrying the exact sum in `f64` and
    /// rounding once at the end reaches `2²⁴ + 2`. One operation could not
    /// tell them apart: a single `f64` operation cast once is correctly
    /// rounded, so the witness is two.
    #[test]
    fn a_const_is_evaluated_in_f32_per_operation() {
        const RUSTC: f32 = 16777216.0 + 1.0 + 1.0;
        let analyzed = accepted(quote! {
            const A: f32 = 16777216.0 + 1.0 + 1.0;
            pub fn f() -> f32 { X + A }
        });
        assert_eq!(analyzed.consts["A"], ConstValue::F32(RUSTC));
        assert_eq!(analyzed.consts["A"], ConstValue::F32(16_777_216.0));
        assert_eq!((16777216.0_f64 + 1.0 + 1.0) as f32, 16_777_218.0);
    }

    /// A product and a sum are two roundings, never one, as rustc's const
    /// evaluator never contracts them. `B * C` is exactly `1 + 2⁻¹¹ + 2⁻²⁴`,
    /// a tie that rounds to even, `1 + 2⁻¹¹`, so `+ D` gives `0`; an FMA keeps
    /// the `2⁻²⁴` and gives that instead.
    #[test]
    fn a_const_never_contracts_a_product_and_a_sum() {
        const B: f32 = 1.0 + 1.0 / 4096.0;
        const C: f32 = 1.0 + 1.0 / 4096.0;
        const D: f32 = -(1.0 + 1.0 / 2048.0);
        const RUSTC: f32 = B * C + D;
        let analyzed = accepted(quote! {
            const B: f32 = 1.0 + 1.0 / 4096.0;
            const C: f32 = 1.0 + 1.0 / 4096.0;
            const D: f32 = -(1.0 + 1.0 / 2048.0);
            const A: f32 = B * C + D;
            pub fn f() -> f32 { X + A }
        });
        assert_eq!(analyzed.consts["A"], ConstValue::F32(RUSTC));
        assert_eq!(analyzed.consts["A"], ConstValue::F32(0.0));
        assert_eq!(B.mul_add(C, D), 1.0 / 16_777_216.0, "one rounding: 2^-24");
    }

    /// What a `const` cannot be: a coordinate, a comparison, a call, a cycle,
    /// a `bool`.
    #[test]
    fn a_const_that_is_not_constant_is_refused() {
        let cases: [(TokenStream, &str); 6] = [
            (
                quote! { const A: f32 = X; pub fn f() -> f32 { A } },
                "cannot find a `const` named `X`",
            ),
            (
                quote! { const A: f32 = 1.0 < 2.0; pub fn f() -> f32 { A } },
                "evaluated at expansion",
            ),
            (
                quote! { const A: f32 = (2.0).sqrt(); pub fn f() -> f32 { A } },
                "evaluated at expansion",
            ),
            (
                quote! { const A: f32 = B + 1.0; const B: f32 = A * 2.0; pub fn f() -> f32 { A } },
                "depends on itself",
            ),
            (
                quote! { const A: bool = 1.0; pub fn f() -> f32 { X } },
                "a `const` in a `kernel!` block is an `f32`",
            ),
            (
                quote! { const A: f32 = 1.0; const A: f32 = 2.0; pub fn f() -> f32 { X } },
                "defined twice",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// A helper is a function of its arguments: it may not read a coordinate.
    #[test]
    fn a_coordinate_in_a_helper_is_refused() {
        let err = refusal(quote! {
            fn shifted() -> f32 { X + 0.5 }
            pub fn f() -> f32 { shifted() }
        });
        assert!(
            err.contains("`X` in a helper") && err.contains("take the coordinate as a parameter"),
            "got: {err}"
        );
    }

    /// Recursion, direct, mutual and through a fold's body, is refused at
    /// the call that closes the cycle.
    #[test]
    fn recursion_is_refused() {
        let err = refusal(quote! {
            fn f(x: f32) -> f32 { f(x) }
            pub fn g() -> f32 { f(X) }
        });
        assert!(
            err.contains("recursion is refused: the language is a DAG")
                && err.contains("`f` calls `f`"),
            "got: {err}"
        );
        let err = refusal(quote! {
            fn a(x: f32) -> f32 { b(x) + 1.0 }
            fn b(x: f32) -> f32 { c(x) * 2.0 }
            fn c(x: f32) -> f32 { a(x) }
            pub fn g() -> f32 { a(X) }
        });
        assert!(
            err.contains("recursion is refused")
                && err.contains("`a` calls `b` calls `c` calls `a`"),
            "got: {err}"
        );
        // Through a fold's body: a call there is a call, and lowering would
        // inline it without end.
        let err = refusal(quote! {
            fn f(x: f32) -> f32 { (0..2).map(|i| f(x)).sum() }
            pub fn g() -> f32 { f(X) }
        });
        assert!(
            err.contains("recursion is refused") && err.contains("`f` calls `f`"),
            "got: {err}"
        );
        // And through an integrand, `area`'s among them.
        for input in [
            quote! {
                fn f(x: f32) -> f32 { integral(0.0..1.0, |u| f(x + u)) }
                pub fn g() -> f32 { f(X) }
            },
            quote! {
                fn f(x: f32) -> f32 { area(|u, v| f(x + u + v)) }
                pub fn g() -> f32 { f(X) }
            },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("recursion is refused") && err.contains("`f` calls `f`"),
                "got: {err}"
            );
        }
    }

    /// A call is checked at its arity and its parameters' types; an entry
    /// is not callable; an unknown function is not captured; the language's
    /// `monotone_root` takes three `f32`s.
    #[test]
    fn a_call_is_checked() {
        let cases: [(TokenStream, &str); 9] = [
            (
                quote! { fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h(X, Y) } },
                "`h` takes 1 argument, but 2 were supplied",
            ),
            (
                quote! { fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h(X < Y) } },
                "parameter 1 of `h` is a `f32`",
            ),
            (
                quote! { pub fn e() -> f32 { X } pub fn f() -> f32 { e() } },
                "`e` is an entry, not a helper",
            ),
            (
                quote! { pub fn f() -> f32 { outside(X) } },
                "cannot find a function `outside`",
            ),
            (
                quote! { fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h } },
                "`h` is a function, not a value",
            ),
            (
                quote! { pub fn f() -> f32 { monotone_root(X, 1.0) } },
                "`monotone_root` takes 3 arguments, `(delta, step, bend)`, but 2 were supplied",
            ),
            (
                quote! { pub fn f() -> f32 { monotone_root(X, 1.0, 2.0, 3.0) } },
                "but 4 were supplied",
            ),
            (
                quote! { pub fn f() -> f32 { monotone_root(X < Y, 1.0, 2.0) } },
                "`monotone_root` takes `f32`s",
            ),
            (
                quote! { pub fn f() -> bool { monotone_root(X, 1.0, 2.0) } },
                "expected `bool`, found `f32`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
        accepted(quote! { || monotone_root(Y - 1.0, X.max(0.0), 0.5) * 2.0 });
    }

    /// A `fn` returns what it declares.
    #[test]
    fn a_return_type_is_checked() {
        let err = refusal(quote! { pub fn f() -> bool { X } });
        assert!(
            err.contains("expected `bool`, found `f32`") && err.contains("`f` declares"),
            "got: {err}"
        );
        accepted(quote! { pub fn f() -> bool { X < Y } });
    }

    /// Nothing shadows an item.
    #[test]
    fn an_item_is_not_shadowed() {
        let err = refusal(quote! {
            const R: f32 = 1.0;
            pub fn f(R: f32) -> f32 { R }
        });
        assert!(err.contains("shadows the `const R`"), "got: {err}");
        let err = refusal(quote! {
            fn h(x: f32) -> f32 { x }
            pub fn f() -> f32 { let h = X; h }
        });
        assert!(err.contains("shadows the `fn h`"), "got: {err}");
        let err = refusal(quote! {
            fn X(x: f32) -> f32 { x }
            pub fn f() -> f32 { X(Y) }
        });
        assert!(err.contains("intrinsic coordinate"), "got: {err}");
    }

    /// A block with no entry would expand to nothing.
    #[test]
    fn a_block_with_no_entry_is_refused() {
        let err = refusal(quote! { fn h(x: f32) -> f32 { x } });
        assert!(err.contains("no entry emits nothing"), "got: {err}");
    }

    /// A retired method is refused by name, with the way out.
    #[test]
    fn a_retired_method_is_refused() {
        for input in [
            quote! { || X.at(Y, X) },
            quote! { || X.constant() },
            quote! { || X.collapse() },
        ] {
            let err = refusal(input);
            assert!(err.contains("Kernel::at"), "got: {err}");
        }
    }

    // ─────────────────────── folds and the binder type ───────────────────────

    /// Each fold types as its monoid's terms: a sum, a product, a minimum
    /// and a maximum of `f32`s are an `f32`, and `any` and `all` of `bool`s
    /// are a `bool`.
    #[test]
    fn every_fold_types_as_its_terms() {
        accepted(quote! { || (0..4).map(|i| X * (i as f32)).sum() });
        accepted(quote! { || (0..4).map(|i| X + i as f32).product() });
        accepted(quote! { || (0..4).map(|i| X - i as f32).fold(f32::INFINITY, f32::min) });
        accepted(quote! { || (0..4).map(|i| X - i as f32).fold(f32::NEG_INFINITY, f32::max) });
        accepted(quote! { pub fn f() -> bool { (0..4).any(|i| X < i as f32) } });
        accepted(quote! { pub fn f() -> bool { (0..4).all(|i| X < i as f32) } });
        let cases: [(TokenStream, &str); 3] = [
            (
                quote! { || (0..4).map(|i| X < i as f32).sum() },
                "a sum, a product, a minimum or a maximum combines `f32`s",
            ),
            (
                quote! { || (0..4).any(|i| X + i as f32) },
                "`any` and `all` combine `bool`s",
            ),
            (
                quote! { || (0..4).any(|i| X < i as f32) + 1.0 },
                "arithmetic takes `f32`s",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(
                err.contains("mismatched types") && err.contains(expected),
                "expected `{expected}`, got: {err}"
            );
        }
    }

    /// Ranges are constant (plan §1.5): a bound naming a fold's index, a
    /// parameter, a `let`, a coordinate or an `f32` const, or built from
    /// anything but integers and `+ - * /`, is refused where it is written.
    #[test]
    fn a_range_bound_that_is_not_constant_is_refused() {
        let cases: [(TokenStream, &str); 9] = [
            (
                quote! { |n: f32| (0..n).map(|i| i as f32).sum() },
                "`n` is not a `const`",
            ),
            (
                quote! { || (0..X).map(|i| i as f32).sum() },
                "`X` is not a `const`",
            ),
            (
                quote! { || { let n = 4.0; (0..n).map(|i| i as f32).sum() } },
                "`n` is not a `const`",
            ),
            (
                quote! { || (0..4).map(|i| (0..i).map(|j| j as f32).sum::<f32>()).sum() },
                "`i` is not a `const`",
            ),
            (
                quote! { const R: f32 = 4.0; pub fn f() -> f32 { (0..R).map(|i| i as f32).sum() } },
                "`R` is an `f32` const",
            ),
            (
                quote! { || (0..4.0).map(|i| i as f32).sum() },
                "expected `usize`, found a float literal",
            ),
            (
                quote! { || (0..X.floor()).map(|i| i as f32).sum() },
                "a range's bounds are constant",
            ),
            (
                quote! { || (0..(1 < 2)).map(|i| i as f32).sum() },
                "a range's bounds are constant",
            ),
            (
                quote! { || (0..-1).map(|i| i as f32).sum() },
                "cannot apply unary operator `-` to type `usize`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
        let err = refusal(quote! { |n: f32| (0..n).map(|i| i as f32).sum() });
        assert!(err.contains("ranges are constant"), "got: {err}");
    }

    /// A range runs forwards; a reversed one is refused, saying it would be
    /// the empty fold. The empty range itself is a fold like any other.
    #[test]
    fn a_reversed_range_is_refused_and_an_empty_one_is_not() {
        let err = refusal(quote! { || (5..3).map(|i| i as f32).sum() });
        assert!(
            err.contains("`5..3` runs backwards") && err.contains("identity"),
            "got: {err}"
        );
        accepted(quote! { || (3..3).map(|i| i as f32).sum() });
    }

    /// A range's bounds are evaluated in `usize`, from `usize` consts, each
    /// operation checked as rustc checks it.
    #[test]
    fn a_range_from_usize_consts_is_evaluated_at_expansion() {
        let analyzed = accepted(quote! {
            const LO: usize = 7 / 2;
            const N: usize = LO * 4 - 1;
            pub const MAX: usize = 18446744073709551615;
            pub fn f() -> f32 { (LO..LO + N).map(|i| i as f32).sum() + (N as f32) }
        });
        assert_eq!(analyzed.consts["LO"], ConstValue::Usize(3));
        assert_eq!(analyzed.consts["N"], ConstValue::Usize(11));
        assert_eq!(analyzed.consts["MAX"], ConstValue::Usize(u64::MAX));
        let Expr::Block(body) = &analyzed.def.fns[0].body else {
            panic!("a fn's body is a block");
        };
        let Some(Expr::Binary(sum)) = body.expr.as_deref() else {
            panic!("the body is a sum");
        };
        let Expr::Fold(fold) = &*sum.lhs else {
            panic!("its left operand is the fold");
        };
        assert_eq!(
            range_bounds(&fold.range, &analyzed.consts).expect("constant"),
            (3, 14)
        );
    }

    /// A `usize` const is checked as rustc checks one: an overflow, a
    /// division by zero, a negation, a float and an `f32` const are errors.
    #[test]
    fn a_usize_const_is_checked_as_rustc_checks_it() {
        let cases: [(TokenStream, &str); 6] = [
            (
                quote! { const N: usize = 3 - 4; pub fn f() -> f32 { X } },
                "attempt to compute `3_usize - 4_usize`, which would overflow",
            ),
            (
                quote! { const N: usize = 18446744073709551615 + 1; pub fn f() -> f32 { X } },
                "which would overflow",
            ),
            (
                quote! { const N: usize = 4 / 0; pub fn f() -> f32 { X } },
                "attempt to divide `4_usize` by zero",
            ),
            (
                quote! { const N: usize = -1; pub fn f() -> f32 { X } },
                "cannot apply unary operator `-` to type `usize`",
            ),
            (
                quote! { const N: usize = 2.0; pub fn f() -> f32 { X } },
                "expected `usize`, found a float literal",
            ),
            (
                quote! { const A: f32 = 1.0; const N: usize = A; pub fn f() -> f32 { X } },
                "`A` is an `f32` const, where a `usize` is expected",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
        let err = refusal(quote! {
            const N: usize = 18446744073709551616;
            pub fn f() -> f32 { X }
        });
        assert!(err.contains("out of range for `usize`"), "got: {err}");
    }

    /// An `f32` const names a `usize` const only through `as f32`, which
    /// rounds as Rust's `as` rounds.
    #[test]
    fn an_f32_const_converts_a_usize_const_by_as() {
        let analyzed = accepted(quote! {
            const N: usize = 16777217;
            const HALF: f32 = 1.0 / (N as f32);
            pub fn f() -> f32 { X * HALF }
        });
        assert_eq!(
            analyzed.consts["HALF"],
            ConstValue::F32(1.0 / (16_777_217_usize as f32))
        );
        let err = refusal(quote! {
            const N: usize = 4;
            const A: f32 = N;
            pub fn f() -> f32 { X * A }
        });
        assert!(err.contains("`N` is a `usize`"), "got: {err}");
    }

    /// A fold's index is a `usize`, and a `usize` is not a value: arithmetic
    /// on one, a comparison of one, and one where an `f32` is expected are
    /// type errors at the name, which say `i as f32`. So is a `usize` const.
    #[test]
    fn a_usize_where_a_value_is_expected_is_a_type_error() {
        let cases: [(TokenStream, &str); 9] = [
            (quote! { || (0..4).map(|i| X * i).sum() }, "`i`"),
            (quote! { || (0..4).map(|i| (i + 1) as f32).sum() }, "`i`"),
            (quote! { || (0..4).any(|i| i < 2) }, "`i`"),
            (quote! { || (0..4).map(|i| i).sum() }, "`i`"),
            (quote! { || (0..4).map(|i| i.sqrt()).sum() }, "`i`"),
            (quote! { || (0..4).map(|i| DX(i)).sum() }, "`i`"),
            (
                quote! { || (0..4).map(|i| { let j = i; j as f32 }).sum() },
                "`i`",
            ),
            (
                quote! {
                    fn h(x: f32) -> f32 { x }
                    pub fn f() -> f32 { (0..4).map(|i| h(i)).sum() }
                },
                "`i`",
            ),
            (
                quote! { const N: usize = 4; pub fn f() -> f32 { X * N } },
                "`N`",
            ),
        ];
        for (input, name) in cases {
            let err = refusal(input);
            assert!(
                err.contains(&format!("mismatched types: {name} is a `usize`"))
                    && err.contains("as f32"),
                "expected {name} as a `usize`, got: {err}"
            );
        }
    }

    /// `as f32` converts a `usize`, named, and nothing else. (A target other
    /// than `f32` is the parser's refusal.)
    #[test]
    fn as_f32_converts_a_usize_and_nothing_else() {
        accepted(quote! { || (0..4).map(|i| (i) as f32).sum() });
        accepted(quote! { const N: usize = 4; pub fn f() -> f32 { X * (N as f32) } });
        let cases: [(TokenStream, &str); 4] = [
            (quote! { || X as f32 }, "this expression's type is `f32`"),
            (quote! { || (X < Y) as f32 }, "a mask"),
            (
                quote! { || (0..4).map(|i| (i as f32) as f32).sum() },
                "this expression's type is `f32`",
            ),
            (
                quote! { const A: f32 = 1.0; pub fn f() -> f32 { A as f32 } },
                "this expression's type is `f32`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// A `usize` is a count, not a value: no parameter, `let` or `fn`
    /// return is declared one.
    #[test]
    fn a_usize_is_not_a_declared_value_type() {
        let cases: [(TokenStream, &str); 3] = [
            (
                quote! { pub fn f(n: usize) -> f32 { X } },
                "a kernel parameter is an `f32` or a `bool`",
            ),
            (
                quote! { pub fn f() -> usize { 4 } },
                "a kernel `fn` returns an `f32` or a `bool`",
            ),
            (
                quote! { || { let n: usize = 4; X } },
                "a `let` in a kernel body is an `f32` or a `bool`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(
                err.contains(expected) && err.contains("a `usize` is a count"),
                "expected `{expected}`, got: {err}"
            );
        }
    }

    /// A fold's index is scoped as a closure's parameter is: it shadows a
    /// binding, an enclosing index among them, and nothing past the fold's
    /// body sees it. It shadows no item.
    #[test]
    fn a_fold_index_is_scoped_to_the_folds_body() {
        accepted(quote! { |r: f32| (0..4).map(|r| r as f32).sum() + r });
        accepted(quote! { || (0..2).map(|i| (0..3).map(|i| i as f32).sum::<f32>()).sum() });
        let err = refusal(quote! { || (0..4).map(|i| i as f32).sum() + i as f32 });
        assert!(err.contains("cannot find `i`"), "got: {err}");
        let err = refusal(quote! { || (0..4).map(|X| X).sum() });
        assert!(
            err.contains("a fold's index `X` shadows the intrinsic"),
            "got: {err}"
        );
        let err = refusal(quote! {
            const N: usize = 4;
            pub fn f() -> f32 { (0..N).map(|N| X).sum() }
        });
        assert!(err.contains("shadows the `const N`"), "got: {err}");
    }

    /// Every method the front end advertises has a typing, and the typing
    /// is the one the IR's arity implies: a comparison is binary, a choice
    /// is ternary.
    #[test]
    fn every_advertised_op_method_has_a_typing() {
        for name in known_method_names() {
            let op = OpKind::from_name(name).expect("advertised names parse");
            match method_typing(op) {
                MethodTyping::Comparison => assert_eq!(op.arity(), 2, "{name}"),
                MethodTyping::Choice => assert_eq!(op.arity(), 3, "{name}"),
                MethodTyping::Arithmetic => {}
            }
        }
    }

    // ───────────────────────────── integrals ─────────────────────────────

    /// The bounds of the integral an entry's body ends in, as `sema`
    /// evaluates them for lowering to read.
    fn bounds_of_the_integral(input: TokenStream) -> (f32, f32) {
        let analyzed = accepted(input);
        let mut body = &analyzed.def.fns[0].body;
        while let Expr::Block(block) = body {
            body = block.expr.as_deref().expect("the block has a value");
        }
        let Expr::Integral(integral) = body else {
            panic!("the body is an integral, got {body:?}");
        };
        let IntegralBounds::Written(range) = &integral.bounds else {
            panic!("the bounds are written");
        };
        interval_bounds(range, &analyzed.consts).expect("an interval")
    }

    /// An integral's bounds are `f32` constants, evaluated at expansion as an
    /// `f32` const's initializer is: literals, `f32` consts, a `usize`
    /// const `as f32`, `+ - * /`, unary `-`. An integer literal is its
    /// `f32`, as it is wherever a value is expected.
    #[test]
    fn an_integrals_bounds_are_evaluated_at_expansion() {
        assert_eq!(
            bounds_of_the_integral(quote! { || integral(0..1, |u| u) }),
            (0.0, 1.0)
        );
        assert_eq!(
            bounds_of_the_integral(quote! {
                const H: f32 = 1.0 / 4.0;
                const N: usize = 3;
                pub fn f() -> f32 { integral(-H..H * (N as f32) + 0.5, |u| u * X) }
            }),
            (-0.25, 1.25)
        );
    }

    /// Integral bounds are constant (plan §1.5): a bound naming a
    /// parameter, a coordinate, a `let`, a fold's index or an enclosing
    /// integral's variable, or built from anything but constant arithmetic,
    /// is refused where it is written.
    #[test]
    fn an_integral_bound_that_is_not_constant_is_refused() {
        let cases: [(TokenStream, &str); 8] = [
            (
                quote! { |c: f32| integral(0.0..c, |u| u) },
                "an integral's bounds are constant, and `c` is not a `const`",
            ),
            (
                quote! { || integral(0.0..X, |u| u) },
                "and `X` is not a `const`",
            ),
            (
                quote! { || { let h = 1.0; integral(0.0..h, |u| u) } },
                "and `h` is not a `const`",
            ),
            (
                quote! { || (0..4).map(|i| integral(0.0..(i as f32), |u| u)).sum() },
                "and `i` is not a `const`",
            ),
            (
                quote! { || integral(0.0..1.0, |u| integral(0.0..u, |v| v)) },
                "and `u` is not a `const`",
            ),
            (
                quote! { || integral(0.0..X.floor(), |u| u) },
                "integral bounds are constant",
            ),
            (
                quote! { || integral(0.0..(1.0 < 2.0), |u| u) },
                "integral bounds are constant",
            ),
            (
                quote! { const N: usize = 2; pub fn f() -> f32 { integral(0.0..N, |u| u) } },
                "`N` is a `usize`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// The interval is one pixelflow-ir admits (`IntervalFold::try_new`),
    /// or a spanned error at the bounds saying why: it runs backwards, it is
    /// empty, an end is not finite, or its length overflows. It was a panic
    /// in the IR's constructor when the kernel was built.
    #[test]
    fn an_interval_the_ir_refuses_is_a_spanned_error() {
        let cases: [(TokenStream, &str); 5] = [
            (
                quote! { || integral(1.0..0.0, |u| u) },
                "`1.0..0.0` runs backwards",
            ),
            (
                quote! { || integral(1.0..1.0, |u| u) },
                "`1.0..1.0` is empty",
            ),
            (quote! { || integral(-0.0..0.0, |u| u) }, "is empty"),
            (
                quote! {
                    const FAR: f32 = 1.0 / 0.0;
                    pub fn f() -> f32 { integral(0.0..FAR, |u| u) }
                },
                "is not finite",
            ),
            (
                quote! { || integral(-3.0e38..3.0e38, |u| u) },
                "overflows `f32`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(
                err.contains(expected) && err.contains("IntervalFold::try_new"),
                "expected `{expected}`, got: {err}"
            );
        }
        // The narrowest interval an `f32` spells is admitted.
        let above_one = f32::from_bits(1.0_f32.to_bits() + 1);
        assert_eq!(
            bounds_of_the_integral(quote! { || integral(1.0..1.00000012, |u| u) }),
            (1.0, above_one)
        );
    }

    /// An integral's variable is scoped as a fold's index is: its body sees
    /// it and every enclosing binding, it shadows a parameter there, and
    /// nothing past the body sees it — `area`'s two variables alike. It
    /// shadows no item.
    #[test]
    fn an_integrals_variable_is_scoped_to_its_body() {
        accepted(quote! { |u: f32| integral(0.0..1.0, |u| u * X) + u });
        accepted(quote! { || integral(0.0..1.0, |u| area(|v, w| u * v * w)) });
        let cases: [(TokenStream, &str); 5] = [
            (
                quote! { || integral(0.0..1.0, |u| u) + u },
                "cannot find `u`",
            ),
            (quote! { || area(|u, v| X + u) + v }, "cannot find `v`"),
            (quote! { || area(|u, v| X + u) * u }, "cannot find `u`"),
            (
                quote! { || integral(0.0..1.0, |X| X) },
                "an integral's variable `X` shadows the intrinsic",
            ),
            (
                quote! {
                    const C: f32 = 1.0;
                    pub fn f() -> f32 { area(|C, v| X + C) }
                },
                "shadows the `const C`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// An integral's variable is an `f32` a body computes with, unlike a
    /// fold's index; its integrand is an `f32`, and so is the integral.
    #[test]
    fn an_integral_types_as_an_f32_of_an_f32() {
        accepted(quote! { || integral(0.0..1.0, |u| (u * u + X).sqrt()) });
        accepted(quote! { pub fn f() -> bool { area(|u, v| X + u) < 0.5 } });
        let cases: [(TokenStream, &str); 3] = [
            (
                quote! { || integral(0.0..1.0, |u| u < X) },
                "an integral's body is its integrand, an `f32`",
            ),
            (
                quote! { || integral(0.0..1.0, |u| u as f32) },
                "this expression's type is `f32`",
            ),
            (
                quote! { || if area(|u, v| X) { X } else { Y } },
                "expected `bool`, found `f32`",
            ),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
    }

    /// `integral`, `area` and `monotone_root` are the language's: no `const`
    /// or `fn` of a block takes one of their names, so a call to one always
    /// means it.
    #[test]
    fn the_language_functions_are_reserved() {
        for input in [
            quote! { fn integral(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
            quote! { const area: f32 = 1.0; pub fn f() -> f32 { X } },
            quote! { fn monotone_root(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
            quote! { pub fn area() -> f32 { X } },
        ] {
            let err = refusal(input);
            assert!(
                err.contains("is a function of the language; an item cannot be named after it"),
                "got: {err}"
            );
        }
    }
}
