//! Macro AST → `ExprArena`.
//!
//! The front end's one lowering step: the surface syntax a user wrote becomes
//! the IR everything downstream speaks. `let` bindings resolve to the
//! [`ExprId`] they name, so the arena is a DAG and a shared subexpression is
//! one node; operators and DSL methods resolve through [`OpKind`], so the op
//! table is not restated here.
//!
//! A helper is inlined at each call — β-reduction. Its arguments are lowered
//! in the caller's scope, once each, and its body is lowered in a scope of
//! its own where its parameters name those nodes: the callee sees its
//! parameters, the block's `const`s and nothing of the caller's, which is
//! what lexical scoping means. A `const` lowers to the value `sema` gave it.
//! An `if` lowers to [`OpKind::If`], the same node `.select` does.
//!
//! A fold lowers to one `Reduce` node over a [`Fold::Range`], built as
//! `Kernel::over` builds it: the body against a placeholder index, then the
//! binder chosen inside-out, the lowest slot no fold in the body binds. So a
//! fold written here and the same fold built with `Kernel::sum_over` and its
//! siblings are one arena. Nothing here unrolls: that is the e-graph's
//! (`HalveFold`, `PeelFold`), when the kernel is baked.
//!
//! An integral lowers the same way to one `Reduce` over a
//! [`Fold::Interval`], its interval built by [`IntervalFold::try_new`], and
//! `area`'s pair over the IR's own pixel, [`PIXEL_HALF_WIDTH`], so an `area`
//! written here and `Kernel::area` are one arena. Nothing here integrates:
//! closing an integral is the e-graph's (`FactorFold`, `NarrowInterval`,
//! `ArcMoment`), and one left open is legalized by quadrature
//! (`passes::resolve`). `monotone_root` lowers to
//! [`integral::monotone_root`], the one definition the rule that closes an
//! arc's integral reads back.
//!
//! An entry's parameters are its uniforms
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.4): each scalar — an
//! `f32` parameter, or one field of a record parameter — is declared as a
//! uniform, in [`AnalyzedKernel::parameters`]' order, and read through its
//! `Uniform` leaf. Nothing a call passes is a constant of the program, so
//! every call of an entry is one program. A declaration here holds a
//! placeholder default; emission declares each with the call's value.
//!
//! An entry with structural parameters lowers to a *template*: its folds
//! over a range that names one, and its `N as f32`s, are left open
//! ([`Holes`]) and filled when its host function is instantiated.
//!
//! Emission — arena to the `TokenStream` that rebuilds it — is [`crate::emit`].

use crate::PLAN;
use crate::ast::{
    BinaryOp, BlockExpr, CastExpr, Expr, FieldExpr, FnItem, FoldExpr, IntegralBounds, IntegralExpr,
    MONOTONE_ROOT, RecordId, Reduction, Role, Stmt, UnaryOp,
};
use crate::sema::{
    AnalyzedKernel, Bounds, ConstValue, RangeScope, StructuralRange, interval_bounds, range_bounds,
};
use crate::symbol::Scopes;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode, UniformDecl, UniformIdentity};
use pixelflow_ir::integral::{self, ROOT_FLOOR, Rise, RootFloor};
use pixelflow_ir::kernel::PIXEL_HALF_WIDTH;
use pixelflow_ir::{Binder, Fold, IntervalFold, Monoid, OpKind, Variance};
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use syn::Ident;

/// DSL method calls that denote a fixed composition of primitive ops rather
/// than a single [`OpKind`] — `(name, arg_count)`, `arg_count` excluding the
/// receiver.
///
/// Lowering builds the composition; this list is the one place that says
/// which names and arities exist, so `sema`'s validation and lowering's
/// dispatch cannot silently drift on which library methods a kernel body may
/// call. They did once, in both directions at once — see
/// `every_advertised_method_compiles` in the crate root.
pub(crate) const LIBRARY_METHODS: &[(&str, usize)] = &[("fract", 0), ("hypot", 1), ("clamp", 2)];

/// A derivative projection, called as `DX(e)`.
///
/// One definition of the names: `sema` accepts a call by asking here, and
/// lowering builds the `Dwrt` chain by matching on the variant, so the two
/// cannot disagree about which projections exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Projection {
    /// `V(e)`: the value itself. Every arena expression is already
    /// value-space, so it is the identity.
    Value,
    Dx,
    Dy,
    Dxx,
    Dxy,
    Dyy,
}

impl Projection {
    /// The projection a name denotes, if any.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "V" => Some(Self::Value),
            "DX" => Some(Self::Dx),
            "DY" => Some(Self::Dy),
            "DXX" => Some(Self::Dxx),
            "DXY" => Some(Self::Dxy),
            "DYY" => Some(Self::Dyy),
            _ => None,
        }
    }
}

/// The coordinate axes, as `Dwrt` names them.
const AXIS_X: u8 = 0;
const AXIS_Y: u8 = 1;

/// The `Var` index of the first placeholder: a fold's index while its body
/// is built, before its slot is chosen.
///
/// Past every index a real binder can take — the reduction index space ends
/// at [`Variance::VARIABLES`] — as `Kernel::over`'s placeholders are, so no
/// rename of a placeholder can reach a binder an inner fold has already
/// chosen. One placeholder per fold open at once, `PLACEHOLDER_BASE + depth`,
/// so that a nested fold's rename never reaches its enclosing fold's index:
/// sharing one would make `Σ_i Σ_j f(i, j)` into `Σ_i Σ_j f(j, j)`.
const PLACEHOLDER_BASE: usize = Variance::VARIABLES as usize;

/// A name in scope while a body is lowered.
#[derive(Debug, Clone)]
enum Binding {
    /// A value: a `let`'s node, an entry's parameter's uniform, or a
    /// helper's parameter bound to its argument's node.
    Value(ExprId),
    /// A fold's index, a `usize`: its placeholder `Var` while the fold's
    /// body is built. A body reads it only as `i as f32`.
    Index(ExprId),
    /// A record: its fields' nodes, in field order — an entry's record
    /// parameter's uniforms, a helper's record argument, or a `let` alias of
    /// either. A record has no node of its own.
    Record(RecordId, Rc<[ExprId]>),
}

/// What a fold or an integral binds, and the body it binds it in: the name
/// the body reads, what the name is there, and the body.
struct Abstraction<'e> {
    name: &'e syn::Ident,
    /// A fold's index is a [`Binding::Index`]; an integral's variable, an
    /// `f32` a body computes with, is a [`Binding::Value`].
    reads_as: fn(ExprId) -> Binding,
    body: &'e Expr,
}

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

/// The largest bound a fold's index reaches exactly: 2²⁴.
///
/// The index is an `f32` lane — the binder's `Var`, and the counter the JIT
/// steps by adding `1.0` — and an `f32` names every integer up to 2²⁴ and
/// not every one past it, so past it indices round together and the fold is
/// not the one written. Measured before this bound:
/// `(16777100..16777300).map(|i| ((i as f32) - 16777000.0) * X).sum()` gave
/// 39890 at `X = 1`, where rustc gives 39900; below 2²⁴ the two agree.
pub(crate) const EXACT_INDEX_BOUND: u64 = 1 << f32::MANTISSA_DIGITS;

/// A fold's bounds at the IR's width. `sema` holds a bound in 64 bits, as
/// the control plane is. The IR narrows it twice: `RangeFold`'s ends are
/// `u32` today (widening them is A5 of the plan, deprioritized), and the
/// index is an `f32` lane, exact to [`EXACT_INDEX_BOUND`]. The lane is the
/// tighter of the two and the one A5 would not lift, so a bound past it is
/// refused here, naming both, rather than narrowed.
fn ir_range(lo: u64, hi: u64) -> Result<Range<u32>, String> {
    let exact = |bound: u64| {
        u32::try_from(bound)
            .ok()
            .filter(|_| bound <= EXACT_INDEX_BOUND)
    };
    match (exact(lo), exact(hi)) {
        (Some(lo), Some(hi)) => Ok(lo..hi),
        _ => Err(format!(
            "the range `{lo}..{hi}` reaches past {EXACT_INDEX_BOUND} (2^24): a fold's index is \
             an `f32` lane, which names every integer only that far, so past it indices would \
             round together and the fold would not be the one written\n\
             note: a fold's ends are also `u32` in the IR today (`RangeFold`); 64-bit fold ends \
             are A5 of {PLAN}, deprioritized, and would not widen the lane"
        )),
    }
}

/// The lowest binder no `Reduce` reachable from `body` binds.
///
/// `Kernel::over`'s rule, and a fold built here follows it so that the two
/// constructions are one program: binders are chosen inside-out, so a fold
/// sees every inner fold's slot and takes the next free one, and distinct
/// live binders never share an index. `pixelflow-ir`'s own
/// (`lowest_free_binder` in `kernel.rs`) is private and walks a `Dag`; this
/// restatement goes with lowering's other copies in B5 of the plan, and
/// `tests/fold_is_kernel_over.rs` pins the two to one canonical key.
fn lowest_free_binder(arena: &ExprArena, body: ExprId) -> Result<Binder, String> {
    let mut bound = [false; Binder::COUNT];
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![body];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if let ExprNode::Reduce { fold, .. } = arena.node(id) {
            bound[usize::from(fold.binder().slot())] = true;
        }
        stack.extend(arena.children(id));
    }
    Binder::all()
        .find(|binder| !bound[usize::from(binder.slot())])
        .ok_or_else(|| {
            format!(
                "a fold whose body already binds all {} of the IR's indices (`Binder::COUNT`)",
                Binder::COUNT
            )
        })
}

/// The default a uniform is declared with here, where no call has supplied
/// one. Nothing at expansion reads it — a uniform is never folded, and
/// emission declares each with the call's value — so it is a NaN, which
/// would poison whatever read it by mistake rather than pass for a number.
const UNBOUND: f32 = f32::NAN;

/// The first placeholder range end of a template's open fold ([`Holes`]).
/// Past every bound a known fold can have — [`ir_range`] refuses one past
/// [`EXACT_INDEX_BOUND`] — so no known fold interns with an open one.
const HOLE_BASE: u32 = EXACT_INDEX_BOUND as u32 + 1;

/// What a structural entry's template leaves open, for its host function to
/// fill per instantiation (plan §1.4): each fold whose range names a
/// structural parameter, and — as `Param(k)` leaves — each `N as f32`,
/// `k` the parameter's position.
///
/// An open fold is built as any fold is, over the empty placeholder range
/// `h..h`, `h = HOLE_BASE + k` for the `k`th distinct range text. Two folds
/// over one range text, of one monoid and one body, are one fold, and are
/// interned as one; two over different ranges never are. The range lives in
/// the fold's bits rather than beside a node id because lowering splices a
/// fold's arena into its enclosing one, which renumbers every id.
#[derive(Debug, Default)]
pub struct Holes {
    ranges: Vec<StructuralRange>,
}

impl Holes {
    /// The placeholder range an open fold over `range` is built with.
    fn placeholder(&mut self, range: StructuralRange) -> Result<Range<u32>, String> {
        let text = range.text();
        let index = match self.ranges.iter().position(|r| r.text() == text) {
            Some(index) => index,
            None => {
                self.ranges.push(range);
                self.ranges.len() - 1
            }
        };
        let end = u32::try_from(index)
            .ok()
            .and_then(|index| HOLE_BASE.checked_add(index))
            .ok_or_else(|| {
                format!(
                    "more distinct ranges over structural parameters than a placeholder can \
                     name ({index})"
                )
            })?;
        Ok(end..end)
    }

    /// The range an open fold of this template is over, or `None` if `fold`
    /// is a known one.
    ///
    /// # Panics
    ///
    /// On a placeholder range no hole names — one from another template, or
    /// a fold a pass rebuilt. Emitted as it stands it would be an empty
    /// fold, its monoid's identity, with plausible pixels.
    pub fn range_of(&self, fold: Fold) -> Option<&StructuralRange> {
        let Fold::Range(range) = fold else {
            return None;
        };
        let index = range.range().start.checked_sub(HOLE_BASE)?;
        let hole = usize::try_from(index)
            .ok()
            .and_then(|index| self.ranges.get(index));
        Some(hole.unwrap_or_else(|| {
            panic!(
                "kernel!: a fold over the placeholder range {:?} that no structural range of \
                 this entry names",
                range.range()
            )
        }))
    }
}

/// An entry lowered: its arena and root, and what the arena leaves open if
/// it is a template.
pub struct Lowered {
    pub arena: ExprArena,
    pub root: ExprId,
    pub holes: Holes,
}

/// Lower an entry's body, inlining the block's helpers and folding its
/// `const`s. Its parameters are declared first, as uniforms, in
/// [`AnalyzedKernel::parameters`]' order, so the arena's uniform table is
/// the entry's declaration order — every scalar, read or not, so that a
/// positional binding cannot shift when a parameter goes unread. Children
/// are recursed first so that parent nodes always reference
/// already-interned [`ExprId`]s.
pub fn lower_entry(entry: &FnItem, analyzed: &AnalyzedKernel) -> Result<Lowered, String> {
    let helpers = analyzed
        .def
        .fns
        .iter()
        .filter(|f| f.role() == Role::Helper)
        .map(|f| (f.name.to_string(), f))
        .collect();
    let mut arena = ExprArena::new();
    let mut locals = Scopes::default();
    for parameter in analyzed.parameters(entry) {
        let mut uniforms = parameter.scalars().map(|_| {
            let slot = arena.declare_uniform(UniformDecl {
                id: UniformIdentity::mint(),
                default: UNBOUND,
            });
            arena.push_uniform(slot)
        });
        let binding = match parameter.record {
            Some((record, _)) => Binding::Record(record, uniforms.collect()),
            None => Binding::Value(uniforms.next().expect("a scalar parameter is one uniform")),
        };
        locals.bind(parameter.name.to_string(), binding);
    }
    let mut lowering = Lowering {
        program: Program { analyzed, helpers },
        frame: Frame {
            role: entry.role(),
            structural: &entry.structural,
            locals,
        },
        arena: &mut arena,
        open_folds: 0,
        holes: Holes::default(),
    };
    let root = lowering.lower(&entry.body)?;
    let holes = lowering.holes;
    Ok(Lowered { arena, root, holes })
}

/// The block's items: what every body can name besides its own scope.
struct Program<'a> {
    /// The block, analyzed: its records and its `const`s' values.
    analyzed: &'a AnalyzedKernel,
    helpers: HashMap<String, &'a FnItem>,
}

/// The function being lowered: its structural parameters, and the bindings
/// in scope — its parameters in the function's own scope, then the
/// `let`-bound locals and fold indices of the blocks being walked (one scope
/// per block, and one per fold body, with Rust's lexical scoping). An
/// entry's parameters are bound to their uniforms; an inlined helper's, to
/// its argument nodes.
struct Frame<'a> {
    role: Role,
    /// An entry's structural parameters; a helper has none.
    structural: &'a [Ident],
    locals: Scopes<Binding>,
}

/// State threaded through the AST → arena walk.
struct Lowering<'a> {
    program: Program<'a>,
    frame: Frame<'a>,
    arena: &'a mut ExprArena,
    /// How many folds' and integrals' bodies are being built, across
    /// inlined helpers too: the depth that picks the next one's placeholder.
    open_folds: usize,
    /// What a template leaves open: the ranges of its folds over structural
    /// parameters.
    holes: Holes,
}

impl Lowering<'_> {
    /// Translate an AST node into the arena, resolving `let`-bound locals via
    /// the frame's scopes. Each binding maps to a single [`ExprId`], so a
    /// local used twice is one node and the arena is a DAG rather than
    /// duplicated subtrees.
    fn lower(&mut self, expr: &Expr) -> Result<ExprId, String> {
        match expr {
            Expr::Ident(ident) => self.resolve(&ident.name.to_string()),

            Expr::Literal(lit) => {
                let value = lit.f32_value().map_err(|e| e.to_string())?;
                Ok(self.arena.push_const(value))
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

                Ok(self.arena.push_binary(op, lhs, rhs))
            }

            Expr::Unary(unary) => {
                let operand = self.lower(&unary.operand)?;
                let op = match unary.op {
                    UnaryOp::Neg => OpKind::Neg,
                };
                Ok(self.arena.push_unary(op, operand))
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();
                let receiver = self.lower(&call.receiver)?;
                let arg_count = call.args.len();

                // Arena expressions are values, so `.clone()` is the identity.
                if method == "clone" && arg_count == 0 {
                    return Ok(receiver);
                }

                // Primitive ops: one `OpKind` per (name, arity), read from
                // the single table `OpKind::from_method_call` resolves
                // against — not re-listed here as a second copy that could
                // silently drift from it (see `LIBRARY_METHODS` below for
                // the one part of this dispatch that table doesn't cover).
                if let Some(op) = OpKind::from_method_call(&method, arg_count) {
                    let mut args = Vec::with_capacity(arg_count);
                    for arg in &call.args {
                        args.push(self.lower(arg)?);
                    }
                    return Ok(match *args.as_slice() {
                        [] => self.arena.push_unary(op, receiver),
                        [a] => self.arena.push_binary(op, receiver, a),
                        [a, b] => self.arena.push_ternary(op, receiver, a, b),
                        _ => unreachable!(
                            "OpKind::from_method_call only resolves ops of arity 1..=3"
                        ),
                    });
                }

                match (method.as_str(), arg_count) {
                    // `fract(x) = x - floor(x)`.
                    ("fract", 0) => {
                        let f = self.arena.push_unary(OpKind::Floor, receiver);
                        Ok(self.arena.push_binary(OpKind::Sub, receiver, f))
                    }
                    // `hypot(x, y) = sqrt(x² + y²)`.
                    ("hypot", 1) => {
                        let arg = self.lower(&call.args[0])?;
                        let xx = self.arena.push_binary(OpKind::Mul, receiver, receiver);
                        let yy = self.arena.push_binary(OpKind::Mul, arg, arg);
                        let sum = self.arena.push_binary(OpKind::Add, xx, yy);
                        Ok(self.arena.push_unary(OpKind::Sqrt, sum))
                    }
                    // `clamp` is library, not a primitive: it denotes
                    // `min(max(x, lo), hi)` and is built as that composition.
                    ("clamp", 2) => {
                        let lo = self.lower(&call.args[0])?;
                        let hi = self.lower(&call.args[1])?;
                        let floored = self.arena.push_binary(OpKind::Max, receiver, lo);
                        Ok(self.arena.push_binary(OpKind::Min, floored, hi))
                    }

                    _ => Err(format!("Unsupported method: {}", method)),
                }
            }

            // A helper is inlined; a projection becomes a `Dwrt` chain.
            Expr::Call(call) => {
                let func = call.func.to_string();
                if let Some(helper) = self.program.helpers.get(&func).copied() {
                    return self.inline(helper, &call.args);
                }
                if func == MONOTONE_ROOT {
                    return self.lower_monotone_root(&call.args);
                }
                self.lower_projection(&func, &call.args)
            }

            // The choice, the same node `.select` lowers to: `If(m, a, b)`
            // is `if m then a else b`.
            Expr::If(choice) => {
                let cond = self.lower(&choice.cond)?;
                let then = self.lower_block(&choice.then_branch)?;
                let otherwise = self.lower(&choice.else_branch)?;
                Ok(self.arena.push_ternary(OpKind::If, cond, then, otherwise))
            }

            Expr::Fold(fold) => self.lower_fold(fold),

            Expr::Integral(integral) => self.lower_integral(integral),

            Expr::Cast(cast) => self.lower_cast(cast),

            Expr::Field(field) => self.lower_field(field),

            // Parentheses are transparent - just recurse into the inner expression
            Expr::Paren(inner) => self.lower(inner),

            Expr::Block(block) => self.lower_block(block),
        }
    }

    /// `⊕_{i ∈ [lo, hi)} body` as one `Reduce` over a [`Fold::Range`].
    fn lower_fold(&mut self, fold: &FoldExpr) -> Result<ExprId, String> {
        let scope = RangeScope {
            consts: &self.program.analyzed.consts,
            structural: self.frame.structural,
        };
        let range = match range_bounds(&fold.range, scope).map_err(|e| e.to_string())? {
            Bounds::Known(lo, hi) => ir_range(lo, hi)?,
            Bounds::Structural(range) => self.holes.placeholder(range)?,
        };
        let monoid = monoid(fold.reduction);
        let index = Abstraction {
            name: &fold.binder,
            reads_as: Binding::Index,
            body: &fold.body,
        };
        self.lower_abstraction(index, |binder| Ok(Fold::new(monoid, binder, range)))
    }

    /// `∫_{u ∈ [lo, hi)} body` as one `Reduce` over a [`Fold::Interval`]:
    /// the interval written, or the pixel `[-H, H)` for `H` the IR's
    /// [`PIXEL_HALF_WIDTH`], each of the two `area` is.
    ///
    /// The interval is built by [`IntervalFold::try_new`], the IR's own
    /// contract. `sema` refused the bounds it does not admit, with a span;
    /// lowering refuses them too rather than rely on that, as it refuses an
    /// index where a value is expected.
    fn lower_integral(&mut self, integral: &IntegralExpr) -> Result<ExprId, String> {
        let (lo, hi) = match &integral.bounds {
            IntegralBounds::Written(range) => {
                interval_bounds(range, &self.program.analyzed.consts).map_err(|e| e.to_string())?
            }
            IntegralBounds::Pixel => (-PIXEL_HALF_WIDTH, PIXEL_HALF_WIDTH),
        };
        let variable = Abstraction {
            name: &integral.variable,
            reads_as: Binding::Value,
            body: &integral.body,
        };
        self.lower_abstraction(variable, |binder| {
            IntervalFold::try_new(binder, lo, hi)
                .map(Fold::Interval)
                .ok_or_else(|| format!("`{lo:?}..{hi:?}` is not an interval the IR admits"))
        })
    }

    /// A fold or an integral: `fold_at`'s fold of the abstraction's body, as
    /// `Kernel`'s `bind_fresh` builds every fold.
    ///
    /// The body is built in a copy of the arena, against a placeholder
    /// index, and only the finished fold is spliced back: the placeholder's
    /// nodes, which renaming it to the chosen binder leaves unreachable,
    /// stay in the copy and are dropped with it, so the arena this emits
    /// holds only the program. Every id bound before the fold means the
    /// same node in the copy, which is what lets the body read them.
    fn lower_abstraction(
        &mut self,
        abstraction: Abstraction,
        fold_at: impl FnOnce(Binder) -> Result<Fold, String>,
    ) -> Result<ExprId, String> {
        let copy = self.arena.clone();
        let enclosing = std::mem::replace(&mut *self.arena, copy);
        let built = self.bind_fresh(abstraction, fold_at);
        let copy = std::mem::replace(&mut *self.arena, enclosing);
        Ok(self.arena.splice(&copy, built?))
    }

    /// [`Self::lower_abstraction`]'s fold, built in the arena it swapped
    /// in: the body with the name bound to a placeholder, then the binder
    /// chosen — after the body exists, since which slots its folds bind
    /// decides it — and the placeholder renamed to it.
    fn bind_fresh(
        &mut self,
        abstraction: Abstraction,
        fold_at: impl FnOnce(Binder) -> Result<Fold, String>,
    ) -> Result<ExprId, String> {
        if self.open_folds >= Binder::COUNT {
            return Err(format!(
                "folds nested more than {} deep: the IR binds at most that many indices at once \
                 (`Binder::COUNT`)",
                Binder::COUNT
            ));
        }
        let placeholder_var = u8::try_from(PLACEHOLDER_BASE + self.open_folds)
            .map_err(|_| format!("fold placeholder past `u8`: {} open", self.open_folds))?;
        let placeholder = self.arena.push_var(placeholder_var);

        self.frame.locals.push_scope();
        self.frame.locals.bind(
            abstraction.name.to_string(),
            (abstraction.reads_as)(placeholder),
        );
        self.open_folds += 1;
        let body = self.lower(abstraction.body);
        self.open_folds -= 1;
        self.frame.locals.pop_scope();
        let body = body?;

        let binder = lowest_free_binder(self.arena, body)?;
        let fold = fold_at(binder)?;
        let index = self.arena.push_var(binder.var());
        let body = self
            .arena
            .substitute_vars_with(body, &[(placeholder_var, index)]);
        Ok(self.arena.push_reduce(fold, body))
    }

    /// `monotone_root(δ, step, bend)`: [`integral::monotone_root`], the one
    /// definition, under the floor an author writes, [`ROOT_FLOOR`] — as
    /// the glyph's builder does (`fonts/loop_blinn.rs`).
    fn lower_monotone_root(&mut self, args: &[Expr]) -> Result<ExprId, String> {
        let [delta, step, bend] = args else {
            return Err(format!(
                "`{MONOTONE_ROOT}` takes `(delta, step, bend)`, but {} arguments were supplied",
                args.len()
            ));
        };
        let delta = self.lower(delta)?;
        let rise = Rise {
            step: self.lower(step)?,
            bend: self.lower(bend)?,
        };
        let floor = RootFloor::new(ROOT_FLOOR)
            .ok_or("`ROOT_FLOOR` is not a floor `RootFloor` admits".to_string())?;
        Ok(integral::monotone_root(self.arena, delta, rise, floor))
    }

    /// `i as f32`. A fold's index is an `f32` lane already, so its
    /// conversion is its binder's `Var`; a `usize` const is its value's
    /// `f32`, rounded as Rust's `as` rounds it; a structural parameter is a
    /// template's `Param` hole, which its host function fills with the same
    /// `N as f32`.
    fn lower_cast(&mut self, cast: &CastExpr) -> Result<ExprId, String> {
        let Some(name) = cast.named() else {
            return Err("`as f32` converts a `usize`, which is a name".to_string());
        };
        if let Some(binding) = self.frame.locals.lookup(&name.to_string()) {
            return match binding {
                Binding::Index(index) => Ok(*index),
                Binding::Value(_) | Binding::Record(..) => Err(format!(
                    "`{name} as f32`: `{name}` is a value, and `as f32` converts a `usize`"
                )),
            };
        }
        if let Some(position) = self.frame.structural.iter().position(|n| n == name) {
            // Borrowed: `Param(u8)` is a rewrite rule's metavariable too, and
            // a `u8` is narrower than the control plane allows. It is loud
            // past 256 rather than wrapping, and it goes with `Param` when
            // D2 of the plan deletes it — a structural hole wants a leaf of
            // its own then.
            let hole = u8::try_from(position).map_err(|_| {
                format!(
                    "`{name} as f32`: an entry reads at most {} structural parameters as values, \
                     the width of the IR's `Param` leaf",
                    usize::from(u8::MAX) + 1
                )
            })?;
            return Ok(self.arena.push_param(hole));
        }
        match self.program.analyzed.consts.get(&name.to_string()) {
            Some(&ConstValue::Usize(count)) => Ok(self.arena.push_const(count as f32)),
            _ => Err(format!("`{name} as f32`: `{name}` is not a `usize`")),
        }
    }

    /// `p.x0`: the node of one field of a record binding.
    fn lower_field(&mut self, field: &FieldExpr) -> Result<ExprId, String> {
        let (record, fields) = self.record(&field.base)?;
        let def = self.program.analyzed.def.record(record);
        def.fields
            .iter()
            .position(|f| f.name == field.member)
            .map(|index| fields[index])
            .ok_or_else(|| format!("no field `{}` on the record `{}`", field.member, def.name))
    }

    /// The record `expr` names, through any parentheses: a record is only
    /// ever written by name (`sema` refuses anything else).
    fn record(&self, expr: &Expr) -> Result<(RecordId, Rc<[ExprId]>), String> {
        match self.record_named(expr) {
            Some(record) => Ok(record),
            None => Err(format!(
                "a record, by name, where the body has `{}`",
                expr.named()
                    .map_or_else(|| "an expression".to_string(), ToString::to_string)
            )),
        }
    }

    /// The record binding `expr` names, if it names one.
    fn record_named(&self, expr: &Expr) -> Option<(RecordId, Rc<[ExprId]>)> {
        match self.frame.locals.lookup(&expr.named()?.to_string())? {
            Binding::Record(record, fields) => Some((*record, Rc::clone(fields))),
            Binding::Value(_) | Binding::Index(_) => None,
        }
    }

    /// Derivative projections (V/DX/DY and the Hessian family) map to
    /// `Dwrt` nodes: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites
    /// them into chain-rule arithmetic before codegen.
    fn lower_projection(&mut self, func: &str, args: &[Expr]) -> Result<ExprId, String> {
        let Some(projection) = Projection::from_name(func) else {
            return Err(format!("Unsupported call: {func}"));
        };
        let [arg] = args else {
            return Err(format!(
                "Unsupported call: {func}/{} (projections take one argument)",
                args.len()
            ));
        };
        let inner = self.lower(arg)?;
        Ok(match projection {
            Projection::Value => inner,
            Projection::Dx => push_dwrt(self.arena, inner, AXIS_X),
            Projection::Dy => push_dwrt(self.arena, inner, AXIS_Y),
            Projection::Dxx => {
                let d = push_dwrt(self.arena, inner, AXIS_X);
                push_dwrt(self.arena, d, AXIS_X)
            }
            Projection::Dxy => {
                let d = push_dwrt(self.arena, inner, AXIS_X);
                push_dwrt(self.arena, d, AXIS_Y)
            }
            Projection::Dyy => {
                let d = push_dwrt(self.arena, inner, AXIS_Y);
                push_dwrt(self.arena, d, AXIS_Y)
            }
        })
    }

    /// β-reduction: `helper(args)` is the helper's body with each parameter
    /// bound to its argument's node, or to a record argument's fields.
    ///
    /// The arguments are lowered in the caller's frame, once each, so an
    /// argument used twice in the body is one node. The body is lowered in a
    /// frame of its own: the helper's parameters are its base scope, and
    /// nothing of the caller's — no local, no entry parameter, no structural
    /// parameter — is visible.
    fn inline(&mut self, helper: &FnItem, args: &[Expr]) -> Result<ExprId, String> {
        if args.len() != helper.params.len() {
            return Err(format!(
                "`{}` takes {} arguments, but {} were supplied",
                helper.name,
                helper.params.len(),
                args.len()
            ));
        }
        let mut locals = Scopes::default();
        for (param, arg) in helper.params.iter().zip(args) {
            let binding = match self.record_named(arg) {
                Some((record, fields)) => Binding::Record(record, fields),
                None => Binding::Value(self.lower(arg)?),
            };
            locals.bind(param.name.to_string(), binding);
        }
        let callee = Frame {
            role: Role::Helper,
            structural: &[],
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
    /// locals was how `{ let X = Y; X }` came to read the coordinate, and a
    /// lowering that is right only because an earlier stage refused its
    /// input is one refactor away from that bug again. For the same reason a
    /// coordinate in a helper is refused here too, not only in `sema`.
    fn resolve(&mut self, name: &str) -> Result<ExprId, String> {
        if let Some(binding) = self.frame.locals.lookup(name) {
            return match binding {
                Binding::Value(id) => Ok(*id),
                Binding::Index(_) => Err(format!(
                    "`{name}` is a fold's index, a `usize`, where a value is expected: \
                     `{name} as f32`"
                )),
                Binding::Record(..) => Err(format!(
                    "`{name}` is a record, where a value is expected: read a field, \
                     `{name}.x0`"
                )),
            };
        }
        // The same order sema documents: a binding, then a const, then a
        // structural parameter, then the coordinates. Sema refuses a
        // parameter named X or Y, so the two stages agree without one
        // relying on the other's refusal.
        match self.program.analyzed.consts.get(name) {
            Some(&ConstValue::F32(value)) => return Ok(self.arena.push_const(value)),
            Some(ConstValue::Usize(_)) => {
                return Err(format!(
                    "`{name}` is a `usize` const, where a value is expected: `{name} as f32`"
                ));
            }
            None => {}
        }
        if self.frame.structural.iter().any(|n| n == name) {
            return Err(format!(
                "`{name}` is a structural parameter, a `usize`, where a value is expected: \
                 `{name} as f32`"
            ));
        }
        let axis = match name {
            "X" => AXIS_X,
            "Y" => AXIS_Y,
            _ => return Err(format!("Unknown identifier: {name}")),
        };
        match self.frame.role {
            Role::Entry => Ok(self.arena.push_var(axis)),
            Role::Helper => Err(format!(
                "`{name}` in a helper: a helper takes its coordinates as arguments"
            )),
        }
    }

    /// A block's `let`s live in a scope of their own, which ends with it.
    fn lower_block(&mut self, block: &BlockExpr) -> Result<ExprId, String> {
        self.frame.locals.push_scope();
        let value = self.lower_block_contents(block);
        self.frame.locals.pop_scope();
        value
    }

    /// A block's statements in order, then its value, in the scope the
    /// caller opened for it.
    fn lower_block_contents(&mut self, block: &BlockExpr) -> Result<ExprId, String> {
        for stmt in &block.stmts {
            match stmt {
                // The initializer is lowered before the binding exists, so it
                // sees whatever the name meant before: `let a = a + 1.0;`. A
                // record is aliased: the new name binds the same fields.
                Stmt::Let(let_stmt) => {
                    let binding = match self.record_named(&let_stmt.init) {
                        Some((record, fields)) => Binding::Record(record, fields),
                        None => Binding::Value(self.lower(&let_stmt.init)?),
                    };
                    self.frame.locals.bind(let_stmt.name.to_string(), binding);
                }
                // A non-binding statement has no value to thread; lower it so
                // any nested error surfaces, then discard the id.
                Stmt::Expr(e) => {
                    self.lower(e)?;
                }
            }
        }
        match &block.expr {
            Some(final_expr) => self.lower(final_expr),
            None => Err("Block has no final expression".to_string()),
        }
    }
}

/// Push `Dwrt(expr, var)` — the variable index rides as a `Const` operand,
/// matching the encoding the e-graph `ChainRule` and `lower_dwrt` read.
fn push_dwrt(arena: &mut ExprArena, expr: ExprId, var: u8) -> ExprId {
    let v = arena.push_const(var as f32);
    arena.push_binary(OpKind::Dwrt, expr, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::Ty;
    use pixelflow_ir::arena::{ExprNode, UniformId};
    use quote::quote;

    /// Lower a block straight from the parser, with no `sema` in front: what
    /// lowering itself makes of a name, whatever an earlier stage would
    /// refuse. The `const`s are evaluated, since lowering reads their values.
    /// A refusal comes back as its text: the arena has no `Debug` to unwrap
    /// through.
    fn lower_unanalyzed(input: proc_macro2::TokenStream) -> Result<(ExprArena, ExprId), String> {
        let def = parse(input).expect("the body parses");
        let consts = def
            .consts
            .iter()
            .map(|c| {
                let Expr::Literal(lit) = &c.init else {
                    panic!("this harness holds literal consts only");
                };
                let value = match Ty::from_syn(&c.ty) {
                    Some(Ty::Usize) => ConstValue::Usize(lit.usize_value().expect("a count")),
                    _ => ConstValue::F32(lit.f32_value().expect("a value")),
                };
                (c.name.to_string(), value)
            })
            .collect();
        let unanalyzed = AnalyzedKernel { def, consts };
        let entry = unanalyzed
            .def
            .fns
            .iter()
            .find(|f| f.role() == Role::Entry)
            .expect("one entry");
        let Lowered { arena, root, .. } = lower_entry(entry, &unanalyzed)?;
        Ok((arena, root))
    }

    fn lowered(input: proc_macro2::TokenStream) -> (ExprArena, ExprId) {
        lower_unanalyzed(input).expect("the body lowers")
    }

    /// Probe p5, at the stage that got it wrong. `sema` now refuses
    /// `let X`, but lowering resolves a name through its scopes before the
    /// coordinates regardless: `{ let X = Y; X }` is the local, which is `Y`.
    /// It was `Var(0)`.
    #[test]
    fn a_binding_is_resolved_before_a_coordinate_of_the_same_name() {
        let (arena, root) = lowered(quote! { || { let X = Y; X } });
        assert!(
            matches!(arena.node(root), ExprNode::Var(1)),
            "`X` is the local bound to Y, got {:?}",
            arena.node(root)
        );
    }

    /// A local shadows the parameter it is named after, and only inside its
    /// block.
    #[test]
    fn a_binding_shadows_a_parameter_only_inside_its_block() {
        let (arena, root) = lowered(quote! { |r: f32| ({ let r = X; r }) + r });
        let ExprNode::Binary(OpKind::Add, inner, outer) = arena.node(root) else {
            panic!("expected the sum, got {:?}", arena.node(root));
        };
        assert!(
            matches!(arena.node(inner), ExprNode::Var(0)),
            "inner `r` is X"
        );
        assert!(
            matches!(arena.node(outer), ExprNode::Uniform(UniformId(0))),
            "outer `r` is the parameter's uniform"
        );
    }

    /// Every parameter is a uniform, declared in order — a record's fields
    /// in field order — whether or not the body reads it; a field read is
    /// its field's uniform, through a `let` alias and a helper's record
    /// parameter alike.
    #[test]
    fn every_parameter_is_a_uniform_declared_in_order() {
        let (arena, root) = lowered(quote! {
            struct Pair { a: f32, b: f32 }
            fn second(p: Pair) -> f32 { p.b }
            pub fn f(unread: f32, p: Pair, r: f32) -> f32 { let q = p; second(q) * r }
        });
        assert_eq!(
            arena.uniforms().len(),
            4,
            "unread, p.a, p.b, r: every scalar is declared"
        );
        let ExprNode::Binary(OpKind::Mul, field, r) = arena.node(root) else {
            panic!("expected the product, got {}", arena.display(root));
        };
        assert!(
            matches!(arena.node(field), ExprNode::Uniform(UniformId(2))),
            "`second(q)` is p.b, the third scalar"
        );
        assert!(matches!(arena.node(r), ExprNode::Uniform(UniformId(3))));
    }

    /// A template's `N as f32` is its structural parameter's `Param` hole,
    /// and a fold over a range naming one is built over a placeholder its
    /// holes map back to the range; a known fold is not a hole.
    #[test]
    fn a_structural_parameter_leaves_the_template_open() {
        let def = parse(quote! {
            pub fn f<const N: usize, const M: usize>() -> f32 {
                (0..M * 2).map(|i| X * (i as f32)).sum::<f32>()
                    + (0..3).map(|i| i as f32).sum::<f32>()
                    + (M as f32)
            }
        })
        .expect("parses");
        let analyzed = crate::sema::analyze(def).expect("analyzes");
        let Lowered { arena, root, holes } =
            lower_entry(&analyzed.def.fns[0], &analyzed).expect("lowers");
        let mut open = Vec::new();
        let mut known = Vec::new();
        let mut params = Vec::new();
        for (_, node) in arena.nodes() {
            match node {
                ExprNode::Reduce { fold, .. } => match holes.range_of(fold) {
                    Some(range) => open.push(range.text()),
                    None => known.push(fold),
                },
                ExprNode::Param(k) => params.push(k),
                _ => {}
            }
        }
        assert_eq!(open, ["0..(M * 2)"]);
        assert_eq!(
            known.len(),
            1,
            "`0..3` is known here: {}",
            arena.display(root)
        );
        assert_eq!(params, [1], "`M` is the second structural parameter");
    }

    /// A placeholder range no hole names is a front-end bug, never a known
    /// fold: emitted as it stands it would be empty, its monoid's identity.
    #[test]
    #[should_panic(expected = "no structural range of this entry names")]
    fn a_placeholder_no_hole_names_is_refused() {
        let binder = Binder::all().next().expect("a binder");
        let stray = Fold::new(Monoid::SUM, binder, HOLE_BASE..HOLE_BASE);
        let _unreachable = Holes::default().range_of(stray);
    }

    /// A literal lowers to the value the parser gave it, bit for bit.
    #[test]
    fn a_literal_lowers_to_the_value_the_parser_rounded_once() {
        let (arena, root) = lowered(quote! { || 1.00000005960464477539062500001 });
        let ExprNode::Const(value) = arena.node(root) else {
            panic!("expected a constant, got {:?}", arena.node(root));
        };
        assert_eq!(value.to_bits(), 0x3f80_0001);
    }

    /// `if` and `.select` are the same node.
    #[test]
    fn an_if_lowers_to_the_if_node() {
        let (arena, root) = lowered(quote! { || if X < Y { X } else { Y } });
        let ExprNode::Ternary(OpKind::If, cond, a, b) = arena.node(root) else {
            panic!("expected If, got {:?}", arena.node(root));
        };
        assert!(matches!(
            arena.node(cond),
            ExprNode::Binary(OpKind::Lt, _, _)
        ));
        assert!(matches!(arena.node(a), ExprNode::Var(0)));
        assert!(matches!(arena.node(b), ExprNode::Var(1)));
    }

    /// A `const` lowers to its value.
    #[test]
    fn a_const_lowers_to_its_value() {
        let (arena, root) = lowered(quote! {
            const R: f32 = 2.5;
            pub fn f() -> f32 { X * R }
        });
        let ExprNode::Binary(OpKind::Mul, _, r) = arena.node(root) else {
            panic!("expected the product, got {:?}", arena.node(root));
        };
        assert!(matches!(arena.node(r), ExprNode::Const(v) if v == 2.5));
    }

    /// A helper's body is lowered in a frame of its own: its parameter
    /// names the argument's node, and a caller's local of the same name as
    /// something the helper reads is not visible to it. Here `sq`'s `x` is
    /// the argument `Y`, not the caller's `let x = X`, and the argument is
    /// one node used twice.
    #[test]
    fn a_helper_is_inlined_in_its_own_scope() {
        let (arena, root) = lowered(quote! {
            fn sq(x: f32) -> f32 { x * x }
            pub fn f() -> f32 { let x = X; sq(Y) + x }
        });
        let ExprNode::Binary(OpKind::Add, call, local) = arena.node(root) else {
            panic!("expected the sum, got {:?}", arena.node(root));
        };
        let ExprNode::Binary(OpKind::Mul, a, b) = arena.node(call) else {
            panic!("expected the square, got {:?}", arena.node(call));
        };
        assert_eq!(a, b, "the argument is one node, used twice");
        assert!(
            matches!(arena.node(a), ExprNode::Var(1)),
            "`x` in `sq` is Y"
        );
        assert!(
            matches!(arena.node(local), ExprNode::Var(0)),
            "`x` in `f` is X"
        );
    }

    /// The caller's locals are not in scope in the helper, at this stage
    /// too: a helper naming one the caller happens to bind is an error, not
    /// dynamic scoping.
    #[test]
    fn a_helper_does_not_see_the_callers_locals() {
        let Err(err) = lower_unanalyzed(quote! {
            fn leak() -> f32 { a }
            pub fn f() -> f32 { let a = X; leak() }
        }) else {
            panic!("`a` is not in the helper's scope");
        };
        assert!(err.contains("Unknown identifier: a"), "got: {err}");
    }

    /// A coordinate in a helper is refused by lowering itself, not only by
    /// `sema`.
    #[test]
    fn a_coordinate_in_a_helper_is_refused_here_too() {
        let Err(err) = lower_unanalyzed(quote! {
            fn shifted() -> f32 { X }
            pub fn f() -> f32 { shifted() }
        }) else {
            panic!("a helper reads no coordinate");
        };
        assert!(err.contains("`X` in a helper"), "got: {err}");
    }

    // ───────────────────────────── folds ─────────────────────────────

    /// Whether any node in the arena — reachable or not — is a
    /// placeholder index.
    fn holds_a_placeholder(arena: &ExprArena) -> bool {
        arena
            .nodes()
            .any(|(_, node)| matches!(node, ExprNode::Var(v) if usize::from(v) >= PLACEHOLDER_BASE))
    }

    /// A fold is one `Reduce` over its range, its body reading the binder's
    /// `Var`, and the arena holds nothing else: the placeholder the body was
    /// built against stayed in the copy it was built in.
    #[test]
    fn a_fold_lowers_to_one_reduce_and_leaves_nothing_behind() {
        let (arena, root) = lowered(quote! { || (2..6).map(|i| X * (i as f32)).sum() });
        let ExprNode::Reduce { fold, body } = arena.node(root) else {
            panic!("expected a fold, got {}", arena.display(root));
        };
        let Fold::Range(range) = fold else {
            panic!("a fold over a range, got {fold}");
        };
        assert_eq!(range.range(), 2..6);
        assert_eq!(range.monoid(), Monoid::SUM);
        assert_eq!(range.binder().slot(), 0);
        let ExprNode::Binary(OpKind::Mul, x, index) = arena.node(body) else {
            panic!("expected the product, got {}", arena.display(body));
        };
        assert!(matches!(arena.node(x), ExprNode::Var(0)));
        assert!(matches!(arena.node(index), ExprNode::Var(v) if v == range.binder().var()));
        assert!(!holds_a_placeholder(&arena));
        assert_eq!(
            arena.len(),
            4,
            "X, the index, the product, the fold, and nothing else"
        );
    }

    /// Nested folds each take their own slot, inside-out, and no
    /// placeholder of either survives.
    #[test]
    fn nested_folds_take_distinct_slots_inside_out() {
        let (arena, root) = lowered(quote! {
            || (0..3).map(|i| (0..4).map(|j| (i as f32) * (j as f32)).sum::<f32>()).sum()
        });
        let ExprNode::Reduce { fold: outer, body } = arena.node(root) else {
            panic!("expected the outer fold, got {}", arena.display(root));
        };
        let ExprNode::Reduce { fold: inner, .. } = arena.node(body) else {
            panic!("expected the inner fold, got {}", arena.display(body));
        };
        assert_eq!(outer.binder().slot(), 1);
        assert_eq!(inner.binder().slot(), 0);
        assert!(!holds_a_placeholder(&arena));
    }

    /// Lowering refuses a fold's index where a value is expected, as `sema`
    /// does, without relying on it.
    #[test]
    fn an_index_where_a_value_is_expected_is_refused_here_too() {
        let Err(err) = lower_unanalyzed(quote! { || (0..4).map(|i| X * i).sum() }) else {
            panic!("`i` is a `usize`");
        };
        assert!(
            err.contains("`i` is a fold's index, a `usize`"),
            "got: {err}"
        );
        let Err(err) = lower_unanalyzed(quote! {
            const N: usize = 4;
            pub fn f() -> f32 { X * N }
        }) else {
            panic!("`N` is a `usize`");
        };
        assert!(err.contains("`N` is a `usize` const"), "got: {err}");
    }

    /// A `usize` const `as f32` is its value's `f32`.
    #[test]
    fn a_usize_const_as_f32_is_its_value() {
        let (arena, root) = lowered(quote! {
            const N: usize = 4;
            pub fn f() -> f32 { N as f32 }
        });
        assert!(matches!(arena.node(root), ExprNode::Const(v) if v == 4.0));
    }

    /// `sema` holds a bound in 64 bits; the IR's index is an `f32` lane,
    /// exact to 2²⁴, and its fold ends are `u32`. A bound past the lane is
    /// refused — past `u32` too, naming the plan's A5 — and 2²⁴ itself is
    /// the last bound accepted.
    #[test]
    fn a_bound_past_the_exact_index_is_refused_naming_a5() {
        for past in [
            quote! { || (0..16777217).map(|i| i as f32).sum() },
            quote! { || (16777217..16777217).map(|i| i as f32).sum() },
            quote! { || (0..4294967296).map(|i| i as f32).sum() },
        ] {
            let Err(err) = lower_unanalyzed(past) else {
                panic!("past 2^24 an f32 lane does not name every index");
            };
            assert!(
                err.contains("past 16777216 (2^24)") && err.contains("A5") && err.contains(PLAN),
                "got: {err}"
            );
        }
        let (arena, root) = lowered(quote! { || (16777215..16777216).map(|i| i as f32).sum() });
        let ExprNode::Reduce {
            fold: Fold::Range(range),
            ..
        } = arena.node(root)
        else {
            panic!("expected a fold, got {}", arena.display(root));
        };
        assert_eq!(range.range(), 16_777_215..16_777_216);
    }

    /// Folds nest as deep as the IR has binders, and no deeper: one more is
    /// refused, not a panic.
    #[test]
    fn folds_nest_as_deep_as_the_ir_has_binders() {
        let nest = |depth: usize| {
            let mut body = quote! { X };
            for _ in 0..depth {
                body = quote! { (0..1).map(|i| #body).sum::<f32>() };
            }
            quote! { || #body }
        };
        let (arena, root) = lowered(nest(Binder::COUNT));
        let ExprNode::Reduce { fold, .. } = arena.node(root) else {
            panic!("expected a fold");
        };
        assert_eq!(usize::from(fold.binder().slot()), Binder::COUNT - 1);
        let Err(err) = lower_unanalyzed(nest(Binder::COUNT + 1)) else {
            panic!("one fold deeper than the index space");
        };
        assert!(err.contains("nested more than"), "got: {err}");
    }

    /// Helpers calling helpers: each call is its own inlining, over its own
    /// arguments.
    #[test]
    fn helpers_call_helpers() {
        let (arena, root) = lowered(quote! {
            fn twice(x: f32) -> f32 { x + x }
            fn quad(x: f32) -> f32 { twice(twice(x)) }
            pub fn f() -> f32 { quad(X) }
        });
        let ExprNode::Binary(OpKind::Add, a, b) = arena.node(root) else {
            panic!("expected the outer sum, got {:?}", arena.node(root));
        };
        assert_eq!(a, b);
        let ExprNode::Binary(OpKind::Add, c, d) = arena.node(a) else {
            panic!("expected the inner sum, got {:?}", arena.node(a));
        };
        assert_eq!(c, d);
        assert!(matches!(arena.node(c), ExprNode::Var(0)));
    }

    // ─────────────────────────── integrals ───────────────────────────

    /// The interval fold `IntervalFold::try_new` builds over `lo..hi` at
    /// binder `slot`.
    fn interval(slot: u8, lo: f32, hi: f32) -> Fold {
        let binder = Binder::from_slot(slot).expect("a binder slot");
        Fold::Interval(IntervalFold::try_new(binder, lo, hi).expect("an interval"))
    }

    /// An integral is one `Reduce` over its interval, its body reading the
    /// binder's `Var` as a value, and the arena holds nothing else.
    #[test]
    fn an_integral_lowers_to_one_reduce_and_leaves_nothing_behind() {
        let (arena, root) = lowered(quote! { || integral(0.25..2.0, |u| X * u) });
        let ExprNode::Reduce { fold, body } = arena.node(root) else {
            panic!("expected an integral, got {}", arena.display(root));
        };
        assert_eq!(fold, interval(0, 0.25, 2.0));
        let ExprNode::Binary(OpKind::Mul, x, u) = arena.node(body) else {
            panic!("expected the product, got {}", arena.display(body));
        };
        assert!(matches!(arena.node(x), ExprNode::Var(0)));
        assert!(matches!(arena.node(u), ExprNode::Var(v) if v == fold.binder().var()));
        assert!(!holds_a_placeholder(&arena));
        assert_eq!(arena.len(), 4, "X, u, the product, the integral");
    }

    /// `area` is two integrals over the IR's pixel, `[-PIXEL_HALF_WIDTH,
    /// PIXEL_HALF_WIDTH)`: the inner one, `u`'s, at slot 0, and the outer,
    /// `v`'s, at slot 1 — `Kernel::area`'s construction.
    #[test]
    fn area_lowers_to_two_integrals_over_the_irs_pixel() {
        let (arena, root) = lowered(quote! { || area(|u, v| (X + u) * (Y + v)) });
        let ExprNode::Reduce { fold: outer, body } = arena.node(root) else {
            panic!("expected the outer integral, got {}", arena.display(root));
        };
        let ExprNode::Reduce { fold: inner, body } = arena.node(body) else {
            panic!("expected the inner integral, got {}", arena.display(body));
        };
        assert_eq!(outer, interval(1, -PIXEL_HALF_WIDTH, PIXEL_HALF_WIDTH));
        assert_eq!(inner, interval(0, -PIXEL_HALF_WIDTH, PIXEL_HALF_WIDTH));
        let ExprNode::Binary(OpKind::Mul, x_side, y_side) = arena.node(body) else {
            panic!("expected the product, got {}", arena.display(body));
        };
        let shifted = |side: ExprId, axis: u8, by: u8| {
            matches!(arena.node(side), ExprNode::Binary(OpKind::Add, a, b)
                if arena.node(a) == ExprNode::Var(axis) && arena.node(b) == ExprNode::Var(by))
        };
        assert!(shifted(x_side, 0, inner.binder().var()), "X + u");
        assert!(shifted(y_side, 1, outer.binder().var()), "Y + v");
        assert!(!holds_a_placeholder(&arena));
    }

    /// Lowering refuses an interval the IR does not admit, as `sema` does,
    /// without relying on it.
    #[test]
    fn an_interval_the_ir_refuses_is_refused_here_too() {
        let Err(err) = lower_unanalyzed(quote! { || integral(1.0..0.0, |u| u) }) else {
            panic!("`1.0..0.0` runs backwards");
        };
        assert!(err.contains("runs backwards"), "got: {err}");
    }

    /// An integral's variable is a value, and not a `usize`: `u as f32`
    /// converts nothing, here as in `sema`.
    #[test]
    fn an_integrals_variable_is_a_value() {
        let Err(err) = lower_unanalyzed(quote! { || integral(0.0..1.0, |u| u as f32) }) else {
            panic!("`u` is a value");
        };
        assert!(err.contains("`u` is a value"), "got: {err}");
    }

    /// `monotone_root(δ, step, bend)` is `integral::monotone_root` on the
    /// three operands under `ROOT_FLOOR`: its output ends in `δ·(1/d)`, the
    /// floor a literal of the denominator.
    #[test]
    fn monotone_root_lowers_to_the_one_definition() {
        let (arena, root) = lowered(quote! { || monotone_root(Y, X, 0.5) });
        let ExprNode::Binary(OpKind::Mul, delta, reciprocal) = arena.node(root) else {
            panic!("expected δ·(1/d), got {}", arena.display(root));
        };
        assert!(matches!(arena.node(delta), ExprNode::Var(1)));
        let ExprNode::Binary(OpKind::Div, _, denominator) = arena.node(reciprocal) else {
            panic!("expected 1/d, got {}", arena.display(reciprocal));
        };
        let ExprNode::Binary(OpKind::Max, _, floor) = arena.node(denominator) else {
            panic!(
                "expected the floored denominator, got {}",
                arena.display(denominator)
            );
        };
        assert!(matches!(arena.node(floor), ExprNode::Const(v) if v == ROOT_FLOOR));
    }
}
