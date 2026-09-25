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
//!    pixels.
//! 4. **Calls**: a helper is called at its arity with arguments of its
//!    parameters' types; an entry is not callable; recursion is refused.
//!
//! ## Symbol Resolution Rules
//!
//! An identifier resolves to the innermost binding of its name in scope:
//! 1. a `let`-bound local → a shared arena id
//! 2. a declared parameter → an entry's is a `Param` bound by the host
//!    function, a helper's is the argument at the call
//! 3. a `const` → its value
//! 4. an intrinsic (X, Y) → a coordinate `Var`, in an entry only: a helper
//!    takes its coordinates as arguments, so that application is contramap
//!    (docs/plans/2026-09-25-the-language-is-kernel.md §1.2)
//! 5. otherwise → refused. A kernel body does not capture from the caller's
//!    scope, so a name nothing here binds is an error here, with a span —
//!    not a capture that lowering then refuses without one.
//!
//! Nothing shadows X, Y, a `const` or a `fn` — a parameter or a `let` of
//! that name is refused — so a coordinate always means the coordinate and an
//! item always means the item.
//!
//! ## Output
//!
//! The semantic phase produces an [`AnalyzedKernel`]: the AST, validated,
//! and the value of every `const`.

use crate::PLAN;
use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, ConstItem, Expr, FnItem, IfExpr, KernelDef, LetStmt,
    MethodCallExpr, Param, Role, Spelling, Stmt, UnaryOp,
};
use crate::lower::{LIBRARY_METHODS, Projection};
use crate::symbol::{SymbolKind, SymbolTable};
use pixelflow_ir::{OpKind, known_method_names};
use proc_macro2::Span;
use std::collections::HashMap;
use syn::{Ident, Type};

/// The type of an expression in a kernel body.
///
/// Two types, and the IR has one lane for both: a `bool` is an all-ones or
/// all-zero mask (`OpKind::mask`). The distinction is enforced here because
/// it cannot be enforced there — a mask read as a number is a NaN, and a
/// number used as a mask blends bit patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// A value.
    F32,
    /// A mask: a comparison produces it, `&` and `|` combine it, an `if`
    /// chooses by it.
    Bool,
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
            _ => None,
        }
    }

    /// The type's name, as written.
    pub fn name(self) -> &'static str {
        match self {
            Ty::F32 => "f32",
            Ty::Bool => "bool",
        }
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

/// The result of semantic analysis.
#[derive(Debug)]
pub struct AnalyzedKernel {
    /// The original kernel definition.
    pub def: KernelDef,
    /// Every `const`'s value, evaluated at expansion.
    pub consts: HashMap<String, f32>,
}

/// Perform semantic analysis on a parsed kernel.
pub fn analyze(def: KernelDef) -> syn::Result<AnalyzedKernel> {
    let items = Items::collect(&def)?;
    let consts = evaluate_consts(&def.consts)?;
    for f in &def.fns {
        FnAnalyzer::new(f, &items)?.check(f)?;
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

/// The functions of the language that B4 brings (plan §1.5): an integral,
/// the pixel's area, and the monotone root. A call to one names the phase.
const B4_FUNCTIONS: [&str; 3] = ["integral", "area", "monotone_root"];

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
    const_names: Vec<String>,
    fns: HashMap<String, Signature>,
}

impl Items {
    /// Collect the items, refusing a duplicate name, a name that is a
    /// coordinate or a projection, and a declared type the language does
    /// not have.
    fn collect(def: &KernelDef) -> syn::Result<Self> {
        let mut items = Items {
            const_names: Vec::with_capacity(def.consts.len()),
            fns: HashMap::with_capacity(def.fns.len()),
        };
        for c in &def.consts {
            items.refuse_a_taken_name(&c.name)?;
            if Ty::from_syn(&c.ty) != Some(Ty::F32) {
                return Err(syn::Error::new_spanned(
                    &c.ty,
                    "a `const` in a `kernel!` block is an `f32`\n\
                     \n\
                     note: it is evaluated at expansion and folded into every body that \
                     names it; a `bool` there would be a mask constant, which nothing spells yet",
                ));
            }
            items.const_names.push(c.name.to_string());
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
        if self.const_names.contains(&text) || self.fns.contains_key(&text) {
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
            Some(ty) => Some(Ty::from_syn(ty).ok_or_else(|| {
                syn::Error::new_spanned(
                    ty,
                    "a kernel `fn` returns an `f32` or a `bool`\n\
                     \n\
                     note: every value in a kernel body is one of the two",
                )
            })?),
        };
        Ok(Signature { role, params, ret })
    }

    fn param_type(param: &Param, role: Role) -> syn::Result<Ty> {
        let ty = Ty::from_syn(&param.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &param.ty,
                "a kernel parameter is an `f32` or a `bool`\n\
                 \n\
                 note: every value in a kernel body is one of the two",
            )
        })?;
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
    role: Role,
    symbols: SymbolTable,
}

impl<'a> FnAnalyzer<'a> {
    /// The scope a body opens in: the coordinates, the block's `const`s, and
    /// the `fn`'s own parameters.
    fn new(f: &FnItem, items: &'a Items) -> syn::Result<Self> {
        let mut analyzer = FnAnalyzer {
            items,
            role: f.role(),
            symbols: SymbolTable::new(),
        };
        for name in &items.const_names {
            analyzer.symbols.register_const(name);
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

    /// The type of an expression, with every name in it resolved.
    fn type_of(&mut self, expr: &Expr) -> syn::Result<Ty> {
        match expr {
            Expr::Ident(ident_expr) => self.resolve_ident(&ident_expr.name),

            Expr::Literal(_) => Ok(Ty::F32),

            Expr::Binary(binary) => self.type_of_binary(binary),

            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => self.expect(&unary.operand, Ty::F32, "`-` negates an `f32`"),
            },

            Expr::MethodCall(call) => self.type_of_method_call(call),

            Expr::Call(call) => self.type_of_call(call),

            Expr::If(choice) => self.type_of_if(choice),

            Expr::Block(block) => self.type_of_block(block),

            Expr::Paren(inner) => self.type_of(inner),
        }
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
        if B4_FUNCTIONS.contains(&name.as_str()) {
            return Err(syn::Error::new(
                call.func.span(),
                format!(
                    "`{name}` is not yet a function of the language\n\
                     \n\
                     note: `integral`, `area` and `monotone_root` are B4 of {PLAN}"
                ),
            ));
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
                let want = Ty::from_syn(annotation).ok_or_else(|| {
                    syn::Error::new_spanned(
                        annotation,
                        "a `let` in a kernel body is an `f32` or a `bool`\n\
                         \n\
                         note: every value in a kernel body is one of the two",
                    )
                })?;
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

// ───────────────────────────── consts ─────────────────────────────

/// Every `const`'s value.
///
/// A `const` is evaluated here, at expansion, per operation in `f32` — the
/// value rustc gives the same expression, since rustc evaluates an `f32`
/// `const` in `f32` too. The discipline is exactly the one rustc's const
/// evaluator has: each operation rounds once, in `f32`, and a product and a
/// sum are never contracted into one rounding — which matters because this
/// crate is built with `-fp-contract=fast` (`.cargo/config.toml`), and an
/// FMA gives `b * c + d` a value two roundings never reach. Its literals were
/// rounded once by the parser, and nothing here rounds again. A const may
/// name another declared anywhere in the block; a cycle is refused.
fn evaluate_consts(consts: &[ConstItem]) -> syn::Result<HashMap<String, f32>> {
    let mut evaluator = ConstEvaluator {
        items: consts.iter().map(|c| (c.name.to_string(), c)).collect(),
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
    values: HashMap<String, f32>,
    /// The consts whose initializers are being evaluated, outermost first:
    /// naming one of them again is a cycle.
    in_progress: Vec<String>,
}

impl ConstEvaluator<'_> {
    /// The value of the const `name`, evaluating it on first demand.
    fn value_of(&mut self, name: &Ident) -> syn::Result<f32> {
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
        let value = self.eval(&item.init)?;
        self.in_progress.pop();
        self.values.insert(key, value);
        Ok(value)
    }

    /// An initializer's value: literals, other consts, `+ - * /`, unary `-`
    /// and parentheses, each operation in `f32`.
    fn eval(&mut self, expr: &Expr) -> syn::Result<f32> {
        match expr {
            Expr::Literal(literal) => Ok(literal.value),
            Expr::Ident(ident) => self.value_of(&ident.name),
            Expr::Paren(inner) => self.eval(inner),
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => Ok(-self.eval(&unary.operand)?),
            },
            Expr::Binary(binary) => {
                let lhs = self.eval(&binary.lhs)?;
                let rhs = self.eval(&binary.rhs)?;
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
                    | BinaryOp::BitOr => Err(Self::not_constant(binary.span)),
                }
            }
            other => Err(Self::not_constant(other.span())),
        }
    }

    fn not_constant(span: Span) -> syn::Error {
        syn::Error::new(
            span,
            "a `const` initializer is evaluated at expansion\n\
             \n\
             note: it is built from literals, other `const`s, `+ - * /`, unary `-` and \
             parentheses, each operation in `f32`",
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
                         forever; a bounded reduction is a fold (B2 of {PLAN})"
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
        assert_eq!(analyzed.consts["SNAP"], 1.0 / 1024.0);
        assert_eq!(analyzed.consts["NEARLY_ONE"], 1.0 - 1.0 / 1024.0);
        assert_eq!(analyzed.consts["NEG"], -(2.0 / 1024.0));
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
        assert_eq!(analyzed.consts["A"], RUSTC);
        assert_eq!(analyzed.consts["A"], 16_777_216.0);
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
        assert_eq!(analyzed.consts["A"], RUSTC);
        assert_eq!(analyzed.consts["A"], 0.0);
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

    /// Recursion, direct and mutual, is refused at the call that closes the
    /// cycle.
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
    }

    /// A call is checked at its arity and its parameters' types; an entry
    /// is not callable; an unknown function is not captured; a function B4
    /// brings names the phase.
    #[test]
    fn a_call_is_checked() {
        let cases: [(TokenStream, &str); 8] = [
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
                quote! { pub fn f() -> f32 { monotone_root(X, 1.0, 2.0) } },
                "`monotone_root` is not yet a function of the language",
            ),
            (quote! { pub fn f() -> f32 { area(X) } }, "B4"),
            (quote! { pub fn f() -> f32 { integral(X) } }, "B4"),
        ];
        for (input, expected) in cases {
            let err = refusal(input);
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        }
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
}
