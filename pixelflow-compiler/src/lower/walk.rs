//! The one walk from an entry's AST to a term at a [`Site`]: bindings in
//! scope, helpers inlined, `const`s folded, each fold one `Reduce`.

use super::{FoldBound, Site};
use crate::ast::{
    BinaryOp, BlockExpr, Callable, CastExpr, Expr, FieldExpr, FnItem, FoldExpr, LetStmt,
    LibraryMethod, Method, MethodCallExpr, Name, ProjectExpr, Projection, RecordId, Reduction,
    Role, Scopes, Stmt, UnaryOp, same_name,
};
use crate::sema::{
    AnalyzedKernel, ConstValue, Count, Parameter, RangeScope, Scalar, folds_nested_too_deep,
    range_bounds,
};
use pixelflow_ir::arena::Axis;
use pixelflow_ir::{Binder, Monoid, OpKind};
use std::collections::HashMap;
use std::rc::Rc;
use syn::Ident;

/// Why a method's operands are all there: [`Method::resolve`] admits a name
/// only at the arity it takes.
const RESOLVED_AT_ITS_ARITY: &str = "`Method::resolve` admits a method only at its arity";

impl Projection {
    /// The axes the projection differentiates along, innermost first: `DXY`
    /// is `∂/∂Y` of `∂e/∂X`, as `Kernel`'s `e.dx().dy()` is.
    fn axes(self) -> &'static [Axis] {
        match self {
            Self::Value => &[],
            Self::Dx => &[Axis::X],
            Self::Dy => &[Axis::Y],
            Self::Dxx => &[Axis::X, Axis::X],
            Self::Dxy => &[Axis::X, Axis::Y],
            Self::Dyy => &[Axis::Y, Axis::Y],
        }
    }
}

/// A name in scope while a body is lowered.
#[derive(Debug, Clone)]
enum Binding<T, A> {
    /// A value: a `let`'s node, an entry's parameter's uniform, or a
    /// helper's parameter bound to its argument's node.
    Value(T),
    /// A fold's index, a `usize`: its placeholder `Var` while the fold's
    /// body is built. A body reads it only as `i as f32`.
    Count(T),
    /// A record: its fields' nodes, in field order — an entry's record
    /// parameter's uniforms, a helper's record argument, or a `let` alias of
    /// either. A record has no node of its own.
    Record(RecordId, Rc<[T]>),
    /// A kernel: an entry's kernel-typed parameter, admitted, or a helper's
    /// bound to the one passed to it. It has no node of its own either; each
    /// application builds one.
    Kernel(A),
}

/// A record as a binding holds it: which record, and its fields' terms in
/// field order.
type RecordFields<T> = (RecordId, Rc<[T]>);

/// The monoid a fold's spelling names.
fn monoid(reduction: Reduction) -> Monoid {
    match reduction {
        Reduction::Sum => Monoid::SUM,
        Reduction::Product => Monoid::PRODUCT,
        Reduction::Min => Monoid::MIN,
        Reduction::Max => Monoid::MAX,
        Reduction::Any => Monoid::ANY,
        Reduction::All => Monoid::ALL,
    }
}

/// Lower an entry's body at `site`, inlining the block's helpers and folding
/// its `const`s. Its uniform parameters are declared first, in
/// [`AnalyzedKernel::parameters`]' order, then its kernel-typed parameters
/// are admitted, in [`Callable::kernel_parameters`]' — each
/// argument's uniforms after the entry's own, in the argument's order.
/// Children are lowered first so that a parent's operands always exist.
///
/// # Errors
///
/// When the body has a construct lowering cannot express there.
pub(crate) fn lower_entry<S: Site>(
    entry: Callable<'_>,
    analyzed: &AnalyzedKernel,
    site: &mut S,
) -> syn::Result<S::Term> {
    let helpers = analyzed
        .def
        .fns()
        .iter()
        .filter(|f| f.role() == Role::Helper)
        .map(|f| (Name::of(&f.name), f))
        .collect();
    let mut locals = Scopes::default();
    for parameter in analyzed.parameters(entry.params) {
        let (name, binding) = match parameter {
            Parameter::Scalar(name) => (
                name,
                Binding::Value(site.uniform(Scalar::Param(name.clone()))),
            ),
            Parameter::Count(name) => (
                name,
                Binding::Count(site.uniform(Scalar::Count(name.clone()))),
            ),
            Parameter::Record { name, id, fields } => {
                let uniforms = fields
                    .iter()
                    .map(|field| site.uniform(Scalar::Field(name.clone(), field.name.clone())))
                    .collect();
                (name, Binding::Record(id, uniforms))
            }
        };
        locals.bind(Name::of(name), binding);
    }
    for name in entry.kernel_parameters() {
        let argument = site.kernel_parameter(name)?;
        locals.bind(Name::of(name), Binding::Kernel(argument));
    }
    let mut lowering = Lowering {
        program: Program { analyzed, helpers },
        frame: Frame {
            role: entry.role,
            locals,
        },
        site,
        open_folds: 0,
    };
    lowering.lower(entry.body)
}

/// The block's items: what every body can name besides its own scope.
struct Program<'a> {
    /// The block, analyzed: its records and its `const`s' values.
    analyzed: &'a AnalyzedKernel,
    helpers: HashMap<Name, &'a FnItem>,
}

/// The function being lowered: the bindings
/// in scope — its parameters in the function's own scope, then the
/// `let`-bound locals and fold indices of the blocks being walked (one scope
/// per block, and one per fold body, with Rust's lexical scoping). An
/// entry's parameters are bound to their uniforms and admitted arguments; an
/// inlined helper's, to its argument nodes and the kernels passed to it.
struct Frame<S: Site> {
    role: Role,
    locals: Scopes<Binding<S::Term, S::Argument>>,
}

/// State threaded through the AST → term walk.
struct Lowering<'a, S: Site> {
    program: Program<'a>,
    frame: Frame<S>,
    site: &'a mut S,
    /// How many folds' bodies are being built, across inlined helpers too:
    /// the depth that picks the next one's placeholder.
    open_folds: usize,
}

impl<S: Site> Lowering<'_, S> {
    /// Translate an AST node into a term at the site, resolving `let`-bound
    /// locals via the frame's scopes. Each binding maps to a single term, so
    /// a local used twice is one node and the arena is a DAG rather than
    /// duplicated subtrees.
    fn lower(&mut self, expr: &Expr) -> syn::Result<S::Term> {
        match expr {
            Expr::Ident(name) => self.resolve(name),

            Expr::Literal(lit) => {
                let value = lit
                    .f32_value()
                    .expect("sema refuses a literal that is no `f32`");
                Ok(self.site.constant(value))
            }

            Expr::Binary(binary) => {
                let lhs = self.lower(&binary.lhs)?;
                let rhs = self.lower(&binary.rhs)?;

                let op = match binary.op {
                    BinaryOp::Add => OpKind::Add,
                    BinaryOp::Sub => OpKind::Sub,
                    BinaryOp::Mul => OpKind::Mul,
                    BinaryOp::Div => OpKind::Div,
                    BinaryOp::Lt => OpKind::Lt,
                    BinaryOp::Le => OpKind::Le,
                    BinaryOp::Gt => OpKind::Gt,
                    BinaryOp::Ge => OpKind::Ge,
                    BinaryOp::Eq => OpKind::Eq,
                    BinaryOp::Ne => OpKind::Ne,
                    // A `bool` is a canonical mask lane — all-ones or all-zero
                    // — so bitwise AND/OR is logical AND/OR exactly. `sema`
                    // types both operands as `bool`.
                    BinaryOp::BitAnd => OpKind::BitAnd,
                    BinaryOp::BitOr => OpKind::BitOr,
                };

                Ok(self.site.binary(op, [lhs, rhs]))
            }

            Expr::Unary(unary) => {
                let operand = self.lower(&unary.operand)?;
                let op = match unary.op {
                    UnaryOp::Neg => OpKind::Neg,
                };
                Ok(self.site.unary(op, operand))
            }

            Expr::MethodCall(call) => self.lower_method_call(call),

            Expr::Project(project) => self.lower_projection(project),

            // A name in scope is resolved first, as `sema` resolves it: a
            // kernel is applied, and any other binding is not a function.
            // Then a helper is inlined.
            Expr::Call(call) => {
                let func = Name::of(&call.func);
                if let Some(binding) = self.frame.locals.lookup(&func) {
                    let Binding::Kernel(kernel) = binding else {
                        unreachable!("sema refuses applying a name that is not a kernel");
                    };
                    let kernel = kernel.clone();
                    return self.apply(kernel, &call.args);
                }
                let helper = self.program.helpers.get(&func).copied().expect(
                    "`sema` refuses a call to anything but a kernel, a helper or a projection",
                );
                self.inline(helper, &call.args)
            }

            // The choice, the same node `.select` lowers to: `If(m, a, b)`
            // is `if m then a else b`.
            Expr::If(choice) => {
                let cond = self.lower(&choice.cond)?;
                let then = self.lower_block(&choice.then_branch)?;
                let otherwise = self.lower(&choice.else_branch)?;
                Ok(self.site.ternary(OpKind::If, [cond, then, otherwise]))
            }

            Expr::Fold(fold) => self.lower_fold(fold),

            Expr::Cast(cast) => self.lower_cast(cast),

            Expr::Field(field) => Ok(self.lower_field(field)),

            Expr::Paren(inner) => self.lower(inner),

            Expr::Block(block) => self.lower_block(block),
        }
    }

    /// `k(x, y)`: the kernel `k` applied at the coordinates, which are
    /// lowered first, in order — contramap (§1.2).
    fn apply(&mut self, kernel: S::Argument, args: &[Expr]) -> syn::Result<S::Term> {
        let [x, y] = args else {
            unreachable!("sema refuses a kernel applied at other than two coordinates");
        };
        let x = self.lower(x)?;
        let y = self.lower(y)?;
        Ok(self.site.apply(kernel, [x, y]))
    }

    /// `⊕_{i ∈ [lo, hi)} body` as one `Reduce` over its [`pixelflow_ir::Fold`]: a
    /// known range, or one whose ends are value counts.
    ///
    /// The body is built against the fold's index, a placeholder, and the
    /// fold is closed after it, by the IR's one definition of how a binder
    /// is chosen and a placeholder renamed ([`pixelflow_ir::arena::ExprArena::close_over`]),
    /// which `Kernel`'s folds are built through too.
    fn lower_fold(&mut self, fold: &FoldExpr) -> syn::Result<S::Term> {
        let scope = RangeScope {
            consts: &self.program.analyzed.consts,
        };
        let counts = range_bounds(&fold.range, scope)?;
        let bound = |count: Count| -> syn::Result<FoldBound<S::Term>> {
            match count {
                Count::Known(value) => Ok(FoldBound::Known(value)),
                Count::Name(name) => {
                    let Binding::Count(term) =
                        self.frame.locals.lookup(&Name::of(&name)).ok_or_else(|| {
                            syn::Error::new(
                                name.span(),
                                format!("`{name}` is not bound as a count"),
                            )
                        })?
                    else {
                        return Err(syn::Error::new(
                            name.span(),
                            format!("`{name}` is not a count bound"),
                        ));
                    };
                    Ok(FoldBound::Value(term.clone()))
                }
            }
        };
        let [lo, hi] = counts;
        let bounds = [bound(lo)?, bound(hi)?];
        // One placeholder per fold open at once, the `n`th for `n` open, so a
        // nested fold's rename never reaches its enclosing fold's index. No
        // more open than the IR has binders: the refusal names the depth,
        // where running out of placeholders would name nothing the author
        // wrote.
        if self.open_folds >= Binder::COUNT {
            return Err(folds_nested_too_deep(fold.span));
        }
        let index = self
            .site
            .open_fold(self.open_folds, monoid(fold.reduction), bounds);
        self.frame.locals.push_scope();
        self.frame
            .locals
            .bind(Name::of(&fold.binder), Binding::Count(index));
        self.open_folds += 1;
        let body = self.lower(&fold.body);
        self.open_folds -= 1;
        self.frame.locals.pop_scope();
        Ok(self.site.close_fold(body?))
    }

    /// `i as f32`. A fold's index is an `f32` lane already, so its
    /// conversion is its binder's `Var`; a `usize` const is its value's
    /// `f32`, rounded as Rust's `as` rounds it.
    fn lower_cast(&mut self, cast: &CastExpr) -> syn::Result<S::Term> {
        let name = cast
            .named()
            .expect("sema refuses an `as f32` of anything but a name");
        if let Some(binding) = self.frame.locals.lookup(&Name::of(name)) {
            let Binding::Count(index) = binding else {
                unreachable!("sema refuses `{name} as f32` where `{name}` is no `usize`");
            };
            return Ok(index.clone());
        }
        let Some(&ConstValue::Usize(count)) = self.program.analyzed.consts.get(&Name::of(name))
        else {
            unreachable!("sema refuses `{name} as f32` where `{name}` is no `usize`");
        };
        Ok(self.site.constant(count as f32))
    }

    /// `p.x0`: the node of one field of a record binding.
    ///
    /// A record is only ever written by name, and read at a field it has
    /// (`sema` refuses anything else).
    fn lower_field(&self, field: &FieldExpr) -> S::Term {
        let (record, fields) = self
            .record_named(&field.base)
            .expect("sema refuses a field read of anything but a record by name");
        let index = self
            .program
            .analyzed
            .def
            .record(record)
            .fields
            .iter()
            .position(|f| same_name(&f.name, &field.member))
            .expect("sema refuses a field the record does not have");
        fields[index].clone()
    }

    /// The record binding `expr` names, if it names one.
    fn record_named(&self, expr: &Expr) -> Option<RecordFields<S::Term>> {
        match self.frame.locals.lookup(&Name::of(expr.named()?))? {
            Binding::Record(record, fields) => Some((*record, Rc::clone(fields))),
            Binding::Value(_) | Binding::Count(_) | Binding::Kernel(_) => None,
        }
    }

    /// The kernel binding `expr` names, if it names one: a kernel is only
    /// ever passed by name (`sema` refuses anything else).
    fn kernel_named(&self, expr: &Expr) -> Option<S::Argument> {
        match self.frame.locals.lookup(&Name::of(expr.named()?))? {
            Binding::Kernel(kernel) => Some(kernel.clone()),
            Binding::Value(_) | Binding::Count(_) | Binding::Record(..) => None,
        }
    }

    /// `receiver.method(args)`: the operands are lowered in order, receiver
    /// first, then the method builds its one node. The parser resolved the
    /// method at this arity, so a method always finds the operands it takes.
    fn lower_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<S::Term> {
        let receiver = self.lower(&call.receiver)?;
        let mut args = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            args.push(self.lower(arg)?);
        }
        let mut args = args.into_iter();
        let mut operand = || args.next().expect(RESOLVED_AT_ITS_ARITY);
        Ok(match call.method {
            // Arena expressions are values, so `.clone()` is the identity.
            Method::Clone => receiver,
            Method::At => self.site.warp(receiver, [operand(), operand()]),
            // Primitive ops: one `OpKind` per (name, arity), read from the
            // single table `OpKind::from_method_call` resolves against.
            Method::Op(op) => match call.args.len() {
                0 => self.site.unary(op, receiver),
                1 => self.site.binary(op, [receiver, operand()]),
                2 => self.site.ternary(op, [receiver, operand(), operand()]),
                _ => unreachable!("OpKind::from_method_call only resolves ops of arity 1..=3"),
            },
            // Library, not primitives: the IR's one definition of each,
            // which `Kernel`'s method of the same name builds too.
            Method::Library(LibraryMethod::Fract) => self.site.fract(receiver),
            Method::Library(LibraryMethod::Hypot) => self.site.hypot([receiver, operand()]),
            Method::Library(LibraryMethod::Clamp) => {
                self.site.clamp(receiver, [operand(), operand()])
            }
        })
    }

    /// Derivative projections (V/DX/DY and the Hessian family) map to
    /// [`pixelflow_ir::library::derivative`]s, one per axis, as `Kernel::dx`/`dy` build
    /// them: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites them into
    /// chain-rule arithmetic before codegen.
    fn lower_projection(&mut self, project: &ProjectExpr) -> syn::Result<S::Term> {
        let mut e = self.lower(&project.arg)?;
        for &axis in project.projection.axes() {
            e = self.site.derivative(e, axis);
        }
        Ok(e)
    }

    /// β-reduction: `helper(args)` is the helper's body with each parameter
    /// bound to its argument's node, to a record argument's fields, or to
    /// the kernel passed to it.
    ///
    /// The arguments are lowered in the caller's frame, once each, so an
    /// argument used twice in the body is one node. The body is lowered in a
    /// frame of its own: the helper's parameters are its base scope, and
    /// nothing of the caller's — no local or entry parameter
    /// parameter — is visible.
    fn inline(&mut self, helper: &FnItem, args: &[Expr]) -> syn::Result<S::Term> {
        assert_eq!(
            args.len(),
            helper.params.len(),
            "sema refuses a call to `{}` at another arity",
            helper.name
        );
        let mut locals = Scopes::default();
        for (param, arg) in helper.params.iter().zip(args) {
            let binding = match (self.record_named(arg), self.kernel_named(arg)) {
                (Some((record, fields)), _) => Binding::Record(record, fields),
                (None, Some(kernel)) => Binding::Kernel(kernel),
                (None, None) => Binding::Value(self.lower(arg)?),
            };
            locals.bind(Name::of(&param.name), binding);
        }
        let callee = Frame {
            role: Role::Helper,
            locals,
        };
        // The frame is swapped and the fold depth is not: a helper's fold
        // inlined inside a fold is nested in it, and its placeholder must
        // not be the caller's, or its rename would reach the caller's index
        // through the argument (pinned against rustc and the builder).
        let caller = std::mem::replace(&mut self.frame, callee);
        let body = self.lower(&helper.body);
        self.frame = caller;
        body
    }

    /// The node a name refers to: the innermost binding of it in scope — a
    /// `let`, a fold's index, a parameter — else a `const`, else a
    /// coordinate.
    ///
    /// Bindings come first because that is what lexical scoping means. `sema`
    /// refuses a `let` named X or Y, so today the order only decides a
    /// local against a parameter it shadows; but matching `"X"` before the
    /// locals would let `{ let X = Y; X }` read the coordinate, and a
    /// lowering that is right only because an earlier stage refused its
    /// input is one refactor away from that bug. For the same reason a
    /// coordinate in a helper is refused here too, not only in `sema`.
    fn resolve(&mut self, ident: &Ident) -> syn::Result<S::Term> {
        let name = &Name::of(ident);
        if let Some(binding) = self.frame.locals.lookup(name) {
            let Binding::Value(id) = binding else {
                unreachable!("sema refuses `{name}` read as a value: it is no `f32` or `bool`");
            };
            return Ok(id.clone());
        }
        // The same order sema documents: a binding, then a const, then
        // the coordinates. Sema refuses a
        // parameter named X or Y, so the two stages agree without one
        // relying on the other's refusal.
        match self.program.analyzed.consts.get(name) {
            Some(&ConstValue::F32(value)) => return Ok(self.site.constant(value)),
            Some(ConstValue::Usize(_)) => {
                unreachable!("sema refuses the `usize` const `{name}` read as a value")
            }
            None => {}
        }
        let axis = name
            .coordinate()
            .unwrap_or_else(|| unreachable!("sema refuses the unknown name `{name}`"));
        match self.frame.role {
            Role::Entry => Ok(self.site.coordinate(axis)),
            Role::Helper => Err(syn::Error::new(
                ident.span(),
                format!("`{name}` in a helper: a helper takes its coordinates as arguments"),
            )),
        }
    }

    /// A block's `let`s live in a scope of their own, which ends with it.
    fn lower_block(&mut self, block: &BlockExpr) -> syn::Result<S::Term> {
        self.frame.locals.push_scope();
        let value = self.lower_block_contents(block);
        self.frame.locals.pop_scope();
        value
    }

    /// What a `let` binds its name to: its initializer's node, or — a record
    /// being aliased — the same fields.
    fn let_binding(&mut self, let_stmt: &LetStmt) -> syn::Result<Binding<S::Term, S::Argument>> {
        match self.record_named(&let_stmt.init) {
            Some((record, fields)) => Ok(Binding::Record(record, fields)),
            None => Ok(Binding::Value(self.lower(&let_stmt.init)?)),
        }
    }

    /// A block's statements in order, then its value, in the scope the
    /// caller opened for it.
    fn lower_block_contents(&mut self, block: &BlockExpr) -> syn::Result<S::Term> {
        for stmt in &block.stmts {
            match stmt {
                // Every initializer before any name: `let (a, b) = (b, a);`
                // swaps, as Rust's does, and `let a = a + 1.0;` sees whatever
                // the name meant before.
                Stmt::Let(lets) => {
                    let mut bindings = Vec::with_capacity(lets.len());
                    for let_stmt in lets {
                        bindings.push(self.let_binding(let_stmt)?);
                    }
                    for (let_stmt, binding) in lets.iter().zip(bindings) {
                        self.frame.locals.bind(Name::of(&let_stmt.name), binding);
                    }
                }
                // A non-binding statement has no value to thread; lower it so
                // any nested error surfaces, then discard the id.
                Stmt::Expr(e) => {
                    self.lower(e)?;
                }
            }
        }
        let value = block
            .expr
            .as_ref()
            .expect("sema refuses a block with no final expression");
        self.lower(value)
    }
}
