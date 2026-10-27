//! One `fn` body, typed: every name resolved to its binding, every
//! expression an `f32` or a `bool`, every kernel applied or passed on once per
//! path.

use crate::ast::{
    Arity, BinaryExpr, BinaryOp, BlockExpr, CallExpr, Callable, CastExpr, Expr, FieldExpr,
    FoldExpr, FunctionType, IfExpr, LetStmt, Method, MethodCallExpr, Name, PLAN, Param,
    ProjectExpr, Projection, RecordId, Role, Scopes, Stmt, UnaryOp, arity_mismatch, same_name,
};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::KERNEL_BINDER_CAPACITY;
use proc_macro2::Span;
use std::collections::HashMap;
use syn::Ident;

use super::analyze::{
    ConstValue, Items, MethodTyping, RETIRED_COORDINATES, RETIRED_PROJECTION, Signature, Ty,
    method_typing, term_type,
};
use super::consts::{RangeScope, bound_names, folds_nested_too_deep, range_bounds};

/// The binding class of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymbolKind {
    /// Intrinsic coordinate variable (X, Y).
    /// Bound at collapse time, as a `Var` node.
    Intrinsic,

    /// A declared parameter (e.g., `r: f32`): an entry's is bound when its
    /// host function runs, a helper's where the helper is inlined.
    Parameter,

    /// A `const` item, evaluated at expansion.
    Const,

    /// Local variable introduced by `let`.
    /// Scoped to the containing block.
    Local,

    /// A fold's index, the closure's parameter: a `usize`, scoped to the
    /// fold's body.
    Index,
}

/// A name's binding, as the body's analysis sees it.
#[derive(Debug, Clone, Copy)]
struct Symbol {
    /// What kind of symbol this is.
    kind: SymbolKind,
    /// The symbol's type.
    ty: Ty,
}

/// The analysis of one `fn` body: its symbols, and the block's items.
pub(super) struct FnAnalyzer<'a> {
    items: &'a Items<'a>,
    /// The `const`s' values, which a fold's bounds are evaluated over.
    consts: &'a HashMap<Name, ConstValue>,
    role: Role,
    /// What the body is checked against: the `fn`'s name and declared return
    /// type, or nothing for the closure form, whose type is inferred.
    declared: Option<(&'a Ident, Ty)>,
    body: &'a Expr,
    symbols: Scopes<Symbol>,
    /// How many folds' bodies the walk is inside: a fold's body is a Rust
    /// closure, and a closure `.map` calls again cannot move what it
    /// captured.
    folds_open: usize,
    /// Each kernel-typed parameter passed on to a helper on the path the
    /// walk is on, by name, where: rustc moves an `impl Fn` it passes, so
    /// nothing after it on that path uses it (E0382). Per path, as rustc's
    /// check is: an `if`'s arms are two paths, and after the `if` a kernel
    /// either one moved is moved ([`FnAnalyzer::type_of_if`]).
    moved: HashMap<Name, Span>,
}

impl<'a> FnAnalyzer<'a> {
    /// The scope a body opens in: the coordinates, the block's `const`s, and
    /// the `fn`'s own parameters.
    pub(super) fn new(
        f: Callable<'a>,
        items: &'a Items<'a>,
        consts: &'a HashMap<Name, ConstValue>,
    ) -> syn::Result<Self> {
        let mut analyzer = FnAnalyzer {
            items,
            consts,
            role: f.role,
            declared: f.name.map(|name| (name, items.fns[&Name::of(name)].ret)),
            body: f.body,
            symbols: Scopes::default(),
            folds_open: 0,
            moved: HashMap::new(),
        };
        for (name, _) in Name::coordinates() {
            analyzer.symbols.bind(
                name,
                Symbol {
                    kind: SymbolKind::Intrinsic,
                    ty: Ty::F32,
                },
            );
        }
        for (name, &ty) in &items.consts {
            analyzer.symbols.bind(
                name.clone(),
                Symbol {
                    kind: SymbolKind::Const,
                    ty,
                },
            );
        }
        for param in f.params {
            let ty = items.param_type(param, f.role)?;
            analyzer.register_parameter(param, ty)?;
        }
        Ok(analyzer)
    }

    /// Type the body, and check it against the declared return type if it
    /// has one.
    pub(super) fn check(mut self) -> syn::Result<()> {
        let body = self.body;
        match self.declared {
            None => self.type_of(body).map(drop),
            Some((name, want)) => self
                .expect(
                    body,
                    want,
                    &format!("`{name}` declares that it returns `{}`", want.name()),
                )
                .map(drop),
        }
    }

    /// Bind a parameter in the function's scope.
    fn register_parameter(&mut self, param: &Param, ty: Ty) -> syn::Result<()> {
        let name = Name::of(&param.name);
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
        self.symbols.bind(
            name,
            Symbol {
                kind: SymbolKind::Parameter,
                ty,
            },
        );
        Ok(())
    }

    /// A parameter or a `let` may not take the name of a coordinate, a
    /// projection, a `const` or a `fn`. A `let` may
    /// shadow a parameter or another `let`, as Rust's does.
    ///
    /// A projection because a call is resolved by its name: a kernel named
    /// `DX` would make `DX(x, y)` an application here and a derivative to a
    /// stage that looked the projection up first.
    fn refuse_shadowing_an_item(&self, name: &Ident, binder: &str) -> syn::Result<()> {
        let text = name.to_string();
        let key = Name::of(name);
        if Projection::from_name(&key).is_some() {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "{binder} `{text}` is named after the projection `{text}(e)`\n\
                     help: rename this {binder} to something else"
                ),
            ));
        }
        // A `let X` would make `X` mean the local below it while a stage that
        // matched the coordinate names first still read the coordinate
        // (`{ let X = Y; X }`): refusing the binding keeps every `X` the
        // coordinate.
        let is_intrinsic = self
            .symbols
            .lookup(&key)
            .is_some_and(|s| s.kind == SymbolKind::Intrinsic);
        if is_intrinsic {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "{binder} `{text}` shadows the intrinsic coordinate variable `{key}`\n\
                     note: intrinsics are: X, Y (coordinate variables), and every use of one \
                     in a kernel body means the coordinate\n\
                     help: rename this {binder} to something else"
                ),
            ));
        }
        let shadowed = match self.symbols.lookup(&key).map(|s| s.kind) {
            Some(SymbolKind::Const) => Some(format!("the `const {text}` of this block")),
            _ => None,
        };
        if let Some(shadowed) = shadowed {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "{binder} `{text}` shadows {shadowed}\n\
                     help: rename this {binder} to something else"
                ),
            ));
        }
        if self.items.fns.contains_key(&key) {
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
            Expr::Ident(name) => self.resolve_ident(name),

            Expr::Literal(literal) => {
                literal.f32_value()?;
                Ok(Ty::F32)
            }

            Expr::Binary(binary) => self.type_of_binary(binary),

            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => self.expect(&unary.operand, Ty::F32, "`-` negates an `f32`"),
            },

            Expr::MethodCall(call) => self.type_of_method_call(call),

            Expr::Project(project) => self.type_of_projection(project),

            Expr::Call(call) => self.type_of_call(call),

            Expr::If(choice) => self.type_of_if(choice),

            Expr::Fold(fold) => self.type_of_fold(fold),

            Expr::Cast(cast) => self.type_of_cast(cast),

            Expr::Field(field) => self.type_of_field(field),

            Expr::Block(block) => self.type_of_block(block),

            Expr::Paren(inner) => self.type_of(inner),
        }
    }

    /// `p.x0`: the base is a record, by name, and the member one of its
    /// fields, an `f32`.
    fn type_of_field(&mut self, field: &FieldExpr) -> syn::Result<Ty> {
        let found = self.type_of(&field.base)?;
        let Ty::Record(record) = found else {
            return Err(syn::Error::new(
                field.base.span(),
                format!(
                    "a field of a `{}`\n\
                     \n\
                     note: only a record has fields; every other value in a kernel body is an \
                     `f32` or a `bool`",
                    self.items.name_of(found)
                ),
            ));
        };
        Self::require_named_record(&field.base)?;
        let def = self.items.def.record(record);
        if def.fields.iter().any(|f| same_name(&f.name, &field.member)) {
            return Ok(Ty::F32);
        }
        let fields: Vec<String> = def.fields.iter().map(|f| format!("`{}`", f.name)).collect();
        Err(syn::Error::new(
            field.member.span(),
            format!(
                "no field `{}` on the record `{}`\n\
                 \n\
                 note: its fields are {}",
                field.member,
                def.name,
                fields.join(", ")
            ),
        ))
    }

    /// A record-typed expression is a name: a parameter or a `let` alias
    /// of one. Anything else that has a record's type — a block ending in
    /// one, a `.clone()` of one — computes a record, and a record is never
    /// computed until Phase D (D7).
    fn require_named_record(expr: &Expr) -> syn::Result<()> {
        if expr.named().is_some() {
            return Ok(());
        }
        Err(syn::Error::new(
            expr.span(),
            format!(
                "a record computed by an expression, where a record is named\n\
                 \n\
                 note: a record is a parameter or a `let` alias of one, written by its name, \
                 and read by field, `p.x0`\n\
                 note: choosing, building or returning a record is Phase D (D7 of {PLAN})"
            ),
        ))
    }

    /// The bounds a fold's range may name: the block's `usize` consts,
    /// enclosing indices and entry count parameters.
    fn range_scope(&self) -> RangeScope<'a> {
        RangeScope {
            consts: self.consts,
        }
    }

    /// `(a..b).map(|i| e).sum()` and its siblings: the bounds are constant
    /// and run forwards, and the body is a term of the fold's monoid, typed
    /// in a scope of its own where the closure's parameter is the index. The
    /// body sees every enclosing binding, an enclosing fold's index among
    /// them, as a Rust closure does.
    fn type_of_fold(&mut self, fold: &FoldExpr) -> syn::Result<Ty> {
        for name in bound_names(&fold.range.lo)
            .into_iter()
            .chain(bound_names(&fold.range.hi))
        {
            let Some(symbol) = self.symbols.lookup(&Name::of(name)) else {
                return Err(syn::Error::new(
                    name.span(),
                    format!("`{name}` is not in scope as a range bound"),
                ));
            };
            if symbol.ty != Ty::Usize {
                return Err(syn::Error::new(
                    name.span(),
                    format!(
                        "`{name}` is a `{}`, where a range bound is a `usize` count",
                        self.items.name_of(symbol.ty)
                    ),
                ));
            }
        }
        let bounds = range_bounds(&fold.range, self.range_scope())?;
        if matches!(
            fold.reduction,
            crate::ast::Reduction::Any | crate::ast::Reduction::All
        ) {
            if let Some(name) = bounds.iter().find_map(super::consts::Count::name) {
                return Err(syn::Error::new(
                    name.span(),
                    format!(
                        "`{}` over a range bounded by `{name}`, a value: `any` and `all` require bounds known at expansion",
                        if fold.reduction == crate::ast::Reduction::Any {
                            "any"
                        } else {
                            "all"
                        }
                    ),
                ));
            }
        }
        if self.folds_open >= KERNEL_BINDER_CAPACITY {
            return Err(folds_nested_too_deep(fold.span));
        }
        self.refuse_shadowing_an_item(&fold.binder, "a fold's index")?;
        let (term, what) = term_type(fold.reduction);
        self.symbols.push_scope();
        // The fold's index is a `usize` in the scope its body is typed in.
        self.symbols.bind(
            Name::of(&fold.binder),
            Symbol {
                kind: SymbolKind::Index,
                ty: Ty::Usize,
            },
        );
        self.folds_open += 1;
        let typed = self.expect(&fold.body, term, what);
        self.folds_open -= 1;
        self.symbols.pop_scope();
        typed
    }

    /// `i as f32`: a `usize`, named, as a value. Nothing else converts —
    /// an `f32` needs no conversion, a `bool` is a mask and becomes a number
    /// by a choice, and no expression computes a `usize` to convert. The
    /// target is `f32` by construction: the parser refuses any other.
    fn type_of_cast(&mut self, cast: &CastExpr) -> syn::Result<Ty> {
        let names_a_count = cast.named().is_some_and(|name| {
            self.symbols
                .lookup(&Name::of(name))
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
            Ty::Record(_) => "a record is its fields; each is an `f32` already, `p.x0`",
            Ty::Kernel => {
                return Err(Self::kernel_is_not_a_value(
                    cast.operand.span(),
                    "the operand of `as f32`, which converts a `usize`",
                ));
            }
            Ty::F32 | Ty::Usize => "it is a value already, and needs no conversion",
        };
        Err(syn::Error::new(
            cast.operand.span(),
            format!(
                "`as f32` converts a `usize`, and this expression's type is `{}`\n\
                 \n\
                 note: {why}\n\
                 note: a `usize` is a fold's index, a `usize` const or an entry count parameter, \
                 converted by name: `i as f32`",
                self.items.name_of(found)
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
    ///
    /// The arms are two paths, and a kernel one passes on is moved on that
    /// path only: the other arm may use it, as rustc's move check, which
    /// follows the flow, lets it. After the `if`, a kernel either arm moved
    /// is moved — rustc's "maybe moved", which it refuses to use (E0382).
    /// The language has no loop, so that join is the whole of the rule.
    fn type_of_if(&mut self, choice: &IfExpr) -> syn::Result<Ty> {
        self.expect(
            &choice.cond,
            Ty::Bool,
            "the condition of an `if` is a `bool`; a comparison gives one",
        )?;
        let before_the_arms = self.moved.clone();
        let then = self.type_of_block(&choice.then_branch)?;
        if then == Ty::Kernel {
            return Err(Self::kernel_is_not_a_value(
                choice.then_branch.span,
                "an arm of an `if`",
            ));
        }
        if let Ty::Record(record) = then {
            return Err(syn::Error::new(
                choice.span,
                format!(
                    "an `if` choosing between records of `{}`\n\
                     \n\
                     note: a record is its fields, and an `if` chooses between values; choose \
                     each field, `if m {{ p.x0 }} else {{ q.x0 }}`\n\
                     note: choosing a whole record is Phase D (D7 of {PLAN})",
                    self.items.def.record(record).name
                ),
            ));
        }
        let moved_by_then = core::mem::replace(&mut self.moved, before_the_arms);
        let ty = self.expect(
            &choice.else_branch,
            then,
            "both arms of an `if` have the same type",
        )?;
        for (name, span) in moved_by_then {
            self.moved.entry(name).or_insert(span);
        }
        Ok(ty)
    }

    /// Resolve an identifier reference to the innermost binding of its name
    /// in scope, or refuse it.
    ///
    /// A name nothing here binds is refused here, with a span, rather than
    /// accepted as a capture from the caller's scope for lowering to refuse
    /// as `Unknown identifier` without one. That includes a local whose block
    /// has ended: `{ { let a = X; a }; a }`.
    fn resolve_ident(&self, ident: &Ident) -> syn::Result<Ty> {
        let name = Name::of(ident);
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
        // A lattice has two axes: say that `Z` and `W` are not coordinates,
        // rather than that the name is missing.
        if let Some(axis) = RETIRED_COORDINATES.into_iter().find(|axis| name == **axis) {
            return Err(syn::Error::new(
                ident.span(),
                format!(
                    "`{axis}` is no longer a coordinate: a lattice has two axes, X and Y\n\
                     note: a scalar that is the same at every sample is a uniform, not an axis\n\
                     help: declare it as a parameter of this kernel: every parameter is a \
                     uniform, bound per call"
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

    /// Type a method call: the method the parser resolved, then its operands
    /// against what it takes.
    fn type_of_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<Ty> {
        match call.method {
            Method::Clone => self.type_of(&call.receiver),
            Method::At => self.type_of_at(call),
            Method::Op(op) => self.check_op_method(op, call),
            Method::Library(_) => self.all_f32(call, &format!("`.{}` takes `f32`s", call.name)),
        }
    }

    /// A field observed at another point keeps its receiver's type.
    fn type_of_at(&mut self, call: &MethodCallExpr) -> syn::Result<Ty> {
        let [x, y] = call.args.as_slice() else {
            unreachable!("Method::At resolves only at two arguments")
        };
        let field = self.type_of(&call.receiver)?;
        match field {
            Ty::F32 | Ty::Bool => {}
            Ty::Usize => unreachable!("a count is refused when named in a value position"),
            Ty::Kernel => {
                return Err(syn::Error::new(
                    call.name.span(),
                    "a kernel is observed by applying it: `k(x, y)` is `k` at `(x, y)`, which is what `k.at(x, y)` would mean",
                ));
            }
            Ty::Record(record) => return Err(self.record_is_not_a_value(&call.receiver, record)),
        }
        let what = "`.at` observes a field at two `f32` coordinates";
        self.expect(x, Ty::F32, what)?;
        self.expect(y, Ty::F32, what)?;
        Ok(field)
    }

    /// The operands of an `OpKind` method against its typing.
    fn check_op_method(&mut self, op: OpKind, call: &MethodCallExpr) -> syn::Result<Ty> {
        let method_name = &call.name;
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

    /// `DX(e)`: a projection of an `f32`.
    fn type_of_projection(&mut self, project: &ProjectExpr) -> syn::Result<Ty> {
        self.expect(&project.arg, Ty::F32, "a projection takes an `f32`")
    }

    /// A free function call: a kernel applied, a helper inlined by
    /// lowering, or a projection. A name in scope is resolved first, as
    /// lowering resolves it: a kernel-typed parameter is applied, and any
    /// other binding is not a function.
    fn type_of_call(&mut self, call: &CallExpr) -> syn::Result<Ty> {
        let name = Name::of(&call.func);
        if let Some(symbol) = self.symbols.lookup(&name).copied() {
            if symbol.ty == Ty::Kernel {
                return self.type_of_application(call);
            }
            return Err(syn::Error::new(
                call.func.span(),
                format!(
                    "`{name}` is a `{}`, and only a kernel is applied\n\
                     \n\
                     note: a kernel is a parameter typed `impl Fn(f32, f32) -> f32`, applied \
                     at two coordinates, `{name}(x, y)` (Phase D-a of {PLAN})",
                    self.items.name_of(symbol.ty)
                ),
            ));
        }
        let items = self.items;
        if let Some(signature) = items.fns.get(&name) {
            return self.type_of_helper_call(call, signature);
        }
        if name == *RETIRED_PROJECTION {
            return Err(syn::Error::new(
                call.func.span(),
                "`DZ` is no longer a coordinate: a lattice has two axes, X and Y",
            ));
        }
        if Projection::from_name(&name).is_some() {
            // The parser keeps a projection's name a call at any arity but one.
            let arity = Arity {
                want: 1,
                got: call.args.len(),
            };
            return Err(arity_mismatch(call.func.span(), &name, arity));
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

    /// `k(x, y)`: a kernel applied at two `f32` coordinates, which is the
    /// kernel's value there, an `f32` — contramap (§1.2), `k.at(x, y)`.
    ///
    /// Applying a kernel borrows it (`Fn::call` takes `&self`), so a body
    /// applies one as often as it likes — but not once it has passed it on,
    /// which moved it (E0382), nor while an argument passes it on, which
    /// rustc refuses as a move out of the borrowed callee.
    fn type_of_application(&mut self, call: &CallExpr) -> syn::Result<Ty> {
        let name = Name::of(&call.func);
        self.refuse_a_moved_kernel(&call.func, "applied")?;
        let [x, y] = call.args.as_slice() else {
            let arity = Arity {
                want: 2,
                got: call.args.len(),
            };
            return Err(syn::Error::new(
                call.func.span(),
                format!(
                    "the kernel `{name}` is applied at the two coordinates, `{name}(x, y)`, and \
                     {} {} supplied",
                    arity.got,
                    arity.supplied_verb(),
                ),
            ));
        };
        let what = "a kernel is applied at two `f32` coordinates";
        self.expect(x, Ty::F32, what)?;
        self.expect(y, Ty::F32, what)?;
        self.refuse_a_moved_kernel(&call.func, "applied")?;
        Ok(Ty::F32)
    }

    /// A kernel passed on to a helper's kernel-typed parameter: by name, and
    /// moved, as rustc moves an `impl Fn` it passes — so at most once on a
    /// path, not after it was moved on that path, and not from inside a
    /// fold's body, a closure `.map` calls again (E0507).
    fn pass_a_kernel(&mut self, arg: &Expr, what: &str) -> syn::Result<()> {
        let found = self.type_of(arg)?;
        if found != Ty::Kernel {
            return Err(syn::Error::new(
                arg.span(),
                format!(
                    "mismatched types: expected `{}`, found `{}`\n\
                     \n\
                     note: {what}",
                    Ty::Kernel.name(),
                    self.items.name_of(found)
                ),
            ));
        }
        let Some(name) = arg.named() else {
            return Err(syn::Error::new(
                arg.span(),
                format!(
                    "a kernel computed by an expression, where a kernel is passed by name\n\
                     \n\
                     note: a kernel is a parameter, passed on by its name (Phase D-a of {PLAN})"
                ),
            ));
        };
        self.refuse_a_moved_kernel(name, "passed on")?;
        if self.folds_open > 0 {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "the kernel `{name}` passed on inside a fold's body\n\
                     \n\
                     note: passing an `impl Fn` moves it, and a fold's body is a closure `.map` \
                     calls once per index, which cannot move what it captured (rustc's E0507)\n\
                     help: apply it in the fold's body, `{name}(x, y)`, or pass it to a helper \
                     whose fold applies it"
                ),
            ));
        }
        self.moved.insert(Name::of(name), name.span());
        Ok(())
    }

    /// Refuse a use of the kernel `name` after it was passed on: rustc's
    /// E0382, which a block of the language is held to because it is Rust.
    fn refuse_a_moved_kernel(&self, name: &Ident, use_: &str) -> syn::Result<()> {
        let Some(moved) = self.moved.get(&Name::of(name)) else {
            return Ok(());
        };
        let mut err = syn::Error::new(
            name.span(),
            format!(
                "the kernel `{name}` {use_} after it was passed on\n\
                 \n\
                 note: a kernel-typed parameter is an `impl Fn`, which moves when it is passed, \
                 so nothing uses it after (rustc's E0382)\n\
                 help: pass it to one helper, and apply it there as often as it is needed"
            ),
        );
        err.combine(syn::Error::new(
            *moved,
            format!("`{name}` was passed on here"),
        ));
        Err(err)
    }

    /// A call to one of the block's `fn`s: a helper, at its arity, with
    /// arguments of its parameters' types.
    fn type_of_helper_call(&mut self, call: &CallExpr, signature: &Signature) -> syn::Result<Ty> {
        let name = Name::of(&call.func);
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
            let arity = Arity {
                want: signature.params.len(),
                got: call.args.len(),
            };
            return Err(arity_mismatch(call.func.span(), &name, arity));
        }
        for (position, (arg, &want)) in call.args.iter().zip(&signature.params).enumerate() {
            let what = format!(
                "parameter {} of `{name}` is a `{}`",
                position + 1,
                self.items.name_of(want)
            );
            if want == Ty::Kernel {
                self.pass_a_kernel(arg, &what)?;
                continue;
            }
            self.expect(arg, want, &what)?;
        }
        Ok(signature.ret)
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
                Stmt::Let(lets) => self.analyze_let(lets)?,
                // A record as a statement is a value lowering has no lane
                // for, so it is refused here rather than there.
                Stmt::Expr(expr) => match self.type_of(expr)? {
                    Ty::Record(record) => return Err(self.record_is_not_a_value(expr, record)),
                    Ty::Kernel => {
                        return Err(Self::kernel_is_not_a_value(expr.span(), "a statement"));
                    }
                    Ty::F32 | Ty::Bool | Ty::Usize => {}
                },
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

    /// `let a = e;` or `let (a, b) = (e1, e2);`: every expression is typed
    /// where the statement stands, before any of its names binds — `let (a,
    /// b) = (b, a);` swaps, as it does in Rust, and `let a = a + 1.0;` sees
    /// the old `a` — and then each name is bound to its expression's type.
    fn analyze_let(&mut self, lets: &[LetStmt]) -> syn::Result<()> {
        let mut found = Vec::with_capacity(lets.len());
        for let_stmt in lets {
            found.push(self.let_type(let_stmt)?);
        }
        for (let_stmt, ty) in lets.iter().zip(found) {
            self.symbols.bind(
                Name::of(&let_stmt.name),
                Symbol {
                    kind: SymbolKind::Local,
                    ty,
                },
            );
        }
        Ok(())
    }

    /// The type a `let` binds its name to, its initializer checked against
    /// its annotation if it has one.
    fn let_type(&mut self, let_stmt: &LetStmt) -> syn::Result<Ty> {
        self.refuse_shadowing_an_item(&let_stmt.name, "`let`")?;

        // The initializer is analyzed before the binding exists, so it sees
        // whatever the name meant before: `let a = a + 1.0;`. A record is
        // aliased, `let q = p;`, and so is written by name.
        let found = match &let_stmt.ty {
            None => match self.type_of(&let_stmt.init)? {
                record @ Ty::Record(_) => {
                    Self::require_named_record(&let_stmt.init)?;
                    record
                }
                Ty::Kernel => {
                    return Err(Self::kernel_is_not_a_value(let_stmt.init.span(), "a `let`"));
                }
                value => value,
            },
            Some(annotation) if FunctionType::of(annotation) != FunctionType::NotOne => {
                return Err(syn::Error::new_spanned(
                    annotation,
                    format!(
                        "a `let` of a function\n\
                         \n\
                         note: a kernel is a parameter, applied by name, `k(x, y)`; a `let` binds \
                         an `f32`, a `bool` or a record (Phase D-a of {PLAN})"
                    ),
                ));
            }
            Some(annotation) => {
                let want = match self.items.declared(annotation) {
                    Some(record @ Ty::Record(_)) => record,
                    _ => Ty::of_a_value(annotation, "a `let` in a kernel body is")?,
                };
                self.expect(
                    &let_stmt.init,
                    want,
                    &format!(
                        "`{}` is declared as a `{}`",
                        let_stmt.name,
                        self.items.name_of(want)
                    ),
                )?
            }
        };
        Ok(found)
    }

    /// Type `expr`, and refuse it unless it is `want`, saying what `what`
    /// needed. Every position of the language that needs one type goes
    /// through here, so a type error reads the same everywhere. A record is
    /// accepted only by name ([`Self::require_named_record`]).
    fn expect(&mut self, expr: &Expr, want: Ty, what: &str) -> syn::Result<Ty> {
        let found = self.type_of(expr)?;
        if found == want {
            if matches!(want, Ty::Record(_)) {
                Self::require_named_record(expr)?;
            }
            return Ok(found);
        }
        if let (Ty::Record(record), Ty::F32 | Ty::Bool) = (found, want) {
            return Err(self.record_is_not_a_value(expr, record));
        }
        if found == Ty::Kernel {
            return Err(Self::kernel_is_not_a_value(expr.span(), what));
        }
        Err(syn::Error::new(
            expr.span(),
            format!(
                "mismatched types: expected `{}`, found `{}`\n\
                 \n\
                 note: {what}",
                self.items.name_of(want),
                self.items.name_of(found)
            ),
        ))
    }

    /// A kernel where a value is expected — in arithmetic, a comparison, a
    /// method's operand, a fold's body, a `let`, an `if`'s arm, a return, a
    /// statement: `where_` says which. A kernel is applied, or passed on.
    fn kernel_is_not_a_value(span: Span, where_: &str) -> syn::Error {
        syn::Error::new(
            span,
            format!(
                "a kernel where a value is expected\n\
                 \n\
                 note: {where_}\n\
                 note: a kernel is a function of the two coordinates, and is not a value: a \
                 body applies it, `k(x, y)` — its value at `(x, y)`, an `f32` — or passes it by \
                 name to a helper's kernel-typed parameter (Phase D-a of {PLAN})\n\
                 help: `k(X, Y)` is its value at the sample"
            ),
        )
    }

    /// A record where a value is expected: in arithmetic, a comparison, a
    /// method's operand, a fold's body. A record is its fields, and a whole
    /// record as one value is Phase D (D7).
    fn record_is_not_a_value(&self, expr: &Expr, record: RecordId) -> syn::Error {
        syn::Error::new(
            expr.span(),
            format!(
                "mismatched types: the record `{}` where a value is expected\n\
                 \n\
                 note: a record is its fields, each an `f32`: read one by name, `p.x0`\n\
                 note: arithmetic, comparison or a choice over whole records is Phase D (D7 of \
                 {PLAN})",
                self.items.def.record(record).name
            ),
        )
    }
}

/// A `usize` where a value is expected. A `usize` — a fold's index, a
/// `usize` const or an entry count parameter — is a value only as `i as f32`:
/// nothing in the language computes with one, because there is nothing to
/// index (plan §1.6).
pub(super) fn a_count_is_not_a_value(name: &Ident) -> syn::Error {
    syn::Error::new(
        name.span(),
        format!(
            "mismatched types: `{name}` is a `usize`, where a value is expected\n\
             \n\
             note: a `usize` (a fold's index, a `usize` const or an entry count parameter) \
             becomes a value by an explicit conversion, `{name} as f32`, and by nothing else: \
             there is no arithmetic on a `usize`, no comparison of one, and no table to index \
             (§1.6 of {PLAN})\n\
             help: write `{name} as f32`"
        ),
    )
}
