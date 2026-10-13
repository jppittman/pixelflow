//! Macro AST → a term, built at a [`Site`].
//!
//! The front end's one lowering step: the surface syntax a user wrote becomes
//! the IR everything downstream speaks. `let` bindings resolve to the term
//! they name, so the arena is a DAG and a shared subexpression is one node;
//! operators and DSL methods resolve through [`OpKind`], so the op table is
//! not restated here.
//!
//! A helper is inlined at each call — β-reduction. Its arguments are lowered
//! in the caller's scope, once each, and its body is lowered in a scope of
//! its own where its parameters name those nodes: the callee sees its
//! parameters, the block's `const`s and nothing of the caller's, which is
//! what lexical scoping means. A `const` lowers to the value `sema` gave it.
//! An `if` lowers to [`OpKind::If`], the same node `.select` does.
//!
//! **What the IR defines, lowering calls; it restates none of it** (plan
//! §1.1, B5). The library methods (`fract`, `hypot`, `clamp`) and the
//! derivative projections build through [`library`], the definitions
//! `Kernel`'s methods build through; the coordinates are [`Axis`]'s; a
//! fold is opened and closed by [`ExprArena::open_fold`] and
//! [`ExprArena::close_fold`], whose binder is chosen and placeholder renamed
//! by [`ExprArena::close_over`], as `Kernel::over`'s are; a range is one
//! [`Fold::admits`]; a kernel is applied by [`ExprArena::apply`]. So each
//! construction written here and the same one built with the builder are
//! one program (`tests/the_library_is_the_builders.rs`,
//! `tests/fold_is_kernel_over.rs`, `tests/kernel_typed_parameters.rs`).
//!
//! A fold lowers to one `Reduce` node over its [`Fold`], built as
//! `Kernel::over` builds it: the body against a placeholder, then closed
//! over it. Nothing here unrolls: that is the e-graph's (`HalveFold`,
//! `PeelFold`), when the kernel is baked.
//!
//! An entry's parameters are its uniforms
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.4): each scalar — an
//! `f32` parameter, or one field of a record parameter — is declared as a
//! uniform, in [`AnalyzedKernel::parameters`]' order, and read through its
//! `Uniform` leaf. Nothing a call passes is a constant of the program, so
//! every call of an entry is one program.
//!
//! **Where a term is built is a [`Site`]** (Phase D-a). Lowering is one walk,
//! and there are two places it can build:
//!
//! - [`Expansion`]: an [`ExprArena`], at macro expansion — every entry that
//!   takes no kernel. Its declarations hold a placeholder default, and
//!   emission declares each with the call's value; an entry with structural
//!   parameters lowers to a *template* whose folds over a range that names
//!   one, and its `N as f32`s, are left open ([`Holes`]) and filled when its
//!   host function is instantiated.
//! - [`Staged`](crate::emit::Staged): Rust statements that build the arena
//!   when the entry's host function is called — an entry that takes a
//!   kernel-typed parameter, whose argument exists only then. Each step is
//!   the step [`Expansion`] takes, run later: the same IR calls, in the same
//!   order.
//!
//! A kernel-typed parameter is admitted when the host function is called,
//! after the entry's own uniforms ([`AnalyzedKernel::kernel_parameters`]),
//! and `k(x, y)` is [`Site::apply`]. [`Expansion`] has no argument to apply
//! — its [`Site::Argument`] is uninhabited — so no entry that takes one can
//! be lowered there.

use crate::PLAN;
use crate::ast::{
    BinaryOp, BlockExpr, CastExpr, Expr, FieldExpr, FnItem, FoldExpr, LetStmt, RecordId, Reduction,
    Role, Stmt, UnaryOp,
};
use crate::sema::{
    AT, AnalyzedKernel, Bounds, ConstValue, RangeScope, Scalar, StructuralRange, range_bounds,
};
use crate::symbol::Scopes;
use pixelflow_ir::arena::{
    Axis, ExprArena, ExprId, IndexSpaceFull, OpenFold, UniformDecl, UniformIdentity,
};
use pixelflow_ir::library;
use pixelflow_ir::{Binder, Fold, Monoid, OpKind};
use std::collections::HashMap;
use std::convert::Infallible;
use std::ops::Range;
use std::rc::Rc;
use syn::Ident;

/// DSL method calls that denote a fixed composition of primitive ops rather
/// than a single [`OpKind`] — `(name, arg_count)`, `arg_count` excluding the
/// receiver.
///
/// The syntax's names for [`library`]'s definitions, which lowering calls
/// and does not restate. This list is the one place that says which names
/// and arities exist, so `sema`'s validation and lowering's dispatch cannot
/// silently drift on which library methods a kernel body may call. They did
/// once, in both directions at once — see `every_advertised_method_compiles`
/// in the crate root.
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

/// The range a fold runs over, as lowering hands it to a [`Site`]: known
/// here, and admitted ([`known_range`]), or naming an entry's structural
/// parameters, known when its host function is instantiated.
#[derive(Debug, Clone)]
pub(crate) enum FoldRange {
    Known(Range<u32>),
    Structural(StructuralRange),
}

/// Somewhere lowering builds a term: each step of the walk, as a call.
///
/// Two of them, and they differ only in *when* a step runs ([`Expansion`]
/// now, `Staged` when the entry's host function is called) — every step is
/// the same IR call either way, so the walk is written once, here, and a
/// program an entry builds at load time is the one it would have built at
/// expansion.
pub(crate) trait Site {
    /// How a term built here is named.
    type Term: Clone;
    /// What a kernel-typed parameter is bound to here.
    type Argument: Clone;

    /// The literal `value`.
    fn constant(&mut self, value: f32) -> Self::Term;
    /// The coordinate `axis`.
    fn coordinate(&mut self, axis: Axis) -> Self::Term;
    /// `op(operand)`.
    fn unary(&mut self, op: OpKind, operand: Self::Term) -> Self::Term;
    /// `op(a, b)`.
    fn binary(&mut self, op: OpKind, operands: [Self::Term; 2]) -> Self::Term;
    /// `op(a, b, c)`.
    fn ternary(&mut self, op: OpKind, operands: [Self::Term; 3]) -> Self::Term;
    /// [`library::fract`].
    fn fract(&mut self, x: Self::Term) -> Self::Term;
    /// [`library::hypot`].
    fn hypot(&mut self, operands: [Self::Term; 2]) -> Self::Term;
    /// [`library::clamp`].
    fn clamp(&mut self, x: Self::Term, bounds: [Self::Term; 2]) -> Self::Term;
    /// [`library::derivative`].
    fn derivative(&mut self, e: Self::Term, axis: Axis) -> Self::Term;
    /// One of the entry's uniform scalars, declared next, and its leaf.
    fn uniform(&mut self, scalar: Scalar<'_>) -> Self::Term;
    /// The entry's kernel-typed parameter `name`, admitted next
    /// ([`ExprArena::admit`]).
    ///
    /// # Errors
    ///
    /// Where there is no argument to admit.
    fn kernel_parameter(&mut self, name: &Ident) -> Result<Self::Argument, String>;
    /// `name as f32`, the entry's `position`th structural parameter.
    ///
    /// # Errors
    ///
    /// Where the position has no room.
    fn count(&mut self, position: usize, name: &Ident) -> Result<Self::Term, String>;
    /// Open a fold over `range` under `monoid`, `depth` folds being open
    /// already: its index ([`ExprArena::open_fold`]). Lowering has refused
    /// a depth past the binders.
    ///
    /// # Errors
    ///
    /// Where the range has no room.
    fn open_fold(
        &mut self,
        depth: usize,
        monoid: Monoid,
        range: FoldRange,
    ) -> Result<Self::Term, String>;
    /// Close the innermost open fold over `body` ([`ExprArena::close_fold`]).
    ///
    /// # Errors
    ///
    /// When its body binds every binder, where that is known.
    fn close_fold(&mut self, body: Self::Term) -> Result<Self::Term, String>;
    /// `kernel` applied at `(x, y)` ([`ExprArena::apply`]).
    fn apply(&mut self, kernel: &Self::Argument, at: [Self::Term; 2]) -> Self::Term;
    /// The field `field` observed at `(x, y)` ([`ExprArena::warp`]).
    fn warp(&mut self, field: Self::Term, at: [Self::Term; 2]) -> Self::Term;
}

/// A name in scope while a body is lowered.
#[derive(Debug, Clone)]
enum Binding<T, A> {
    /// A value: a `let`'s node, an entry's parameter's uniform, or a
    /// helper's parameter bound to its argument's node.
    Value(T),
    /// A fold's index, a `usize`: its placeholder `Var` while the fold's
    /// body is built. A body reads it only as `i as f32`.
    Index(T),
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

/// A known fold's bounds at the IR's width, if [`Fold::admits`] them.
///
/// `sema` holds a bound in 64 bits, as the control plane is, and has
/// refused a range that runs backwards. The IR narrows a bound twice: its
/// ends are `u32` today (widening them is A5 of the plan, deprioritized),
/// and its index is an `f32` lane, exact to [`Fold::EXACT_BOUND`]. The
/// lane is the tighter of the two, the one A5 would not lift, and the IR's
/// to state; lowering asks it before building the body, so the refusal
/// names the bounds, in the words of both narrowings.
fn known_range(lo: u64, hi: u64) -> Result<Range<u32>, String> {
    let range = match (u32::try_from(lo), u32::try_from(hi)) {
        (Ok(lo), Ok(hi)) => Some(lo..hi),
        _ => None,
    };
    range.filter(Fold::admits).ok_or_else(|| {
        let bound = Fold::EXACT_BOUND;
        format!(
            "the range `{lo}..{hi}` reaches past {bound} (2^24): a fold's index is an `f32` \
             lane, which names every integer only that far, so past it indices would round \
             together and the fold would not be the one written (`Fold::EXACT_BOUND`)\n\
             note: a fold's ends are also `u32` in the IR today (`Fold`); 64-bit fold ends \
             are A5 of {PLAN}, deprioritized, and would not widen the lane"
        )
    })
}

/// The default a uniform is declared with here, where no call has supplied
/// one. Nothing at expansion reads it — a uniform is never folded, and
/// emission declares each with the call's value — so it is a NaN, which
/// would poison whatever read it by mistake rather than pass for a number.
const UNBOUND: f32 = f32::NAN;

/// The range every open fold of a template is built over ([`Holes`]):
/// empty, so that read as a program an open fold is its monoid's identity,
/// never plausible pixels.
const OPEN_RANGE: Range<u32> = 0..0;

/// The stride an open fold over a template's first distinct range text is
/// built with; the `k`th's is this plus `k`. No known fold has it: lowering
/// builds every known fold with `Fold::new`, whose stride is 1, so no known
/// fold interns with an open one, and none reads as one.
const HOLE_STRIDE: u32 = 2;

/// What a structural entry's template leaves open, for its host function to
/// fill per instantiation (plan §1.4): each fold whose range names a
/// structural parameter, and — as `Param(k)` leaves — each `N as f32`,
/// `k` the parameter's position.
///
/// An open fold is built as any fold is, over [`OPEN_RANGE`] strided by
/// [`HOLE_STRIDE`]` + k` for the `k`th distinct range text: the ends of a
/// range [`Fold::admits`] have no room for a placeholder, and its
/// stride does. Two folds over one range text, of one monoid and one body,
/// are one fold, and are interned as one; two over different ranges never
/// are. The range text lives in the fold's bits rather than beside a node
/// id because lowering splices a fold's arena into its enclosing one, which
/// renumbers every id.
#[derive(Debug, Default)]
pub struct Holes {
    ranges: Vec<StructuralRange>,
}

impl Holes {
    /// The stride an open fold over `range` is built with, over
    /// [`OPEN_RANGE`].
    fn stride(&mut self, range: StructuralRange) -> Result<u32, String> {
        let text = range.text();
        let index = match self.ranges.iter().position(|r| r.text() == text) {
            Some(index) => index,
            None => {
                self.ranges.push(range);
                self.ranges.len() - 1
            }
        };
        u32::try_from(index)
            .ok()
            .and_then(|index| HOLE_STRIDE.checked_add(index))
            .ok_or_else(|| {
                format!(
                    "more distinct ranges over structural parameters than a placeholder can \
                     name ({index})"
                )
            })
    }

    /// The range an open fold of this template is over, or `None` if `fold`
    /// is a known one.
    ///
    /// # Panics
    ///
    /// On a placeholder stride no hole names — one from another template.
    /// Emitted as it stands it would be an empty fold, its monoid's
    /// identity, with plausible pixels.
    pub fn range_of(&self, range: Fold) -> Option<&StructuralRange> {
        if range.range() != OPEN_RANGE {
            return None;
        }
        let index = range.stride().checked_sub(HOLE_STRIDE)?;
        let hole = usize::try_from(index)
            .ok()
            .and_then(|index| self.ranges.get(index));
        Some(hole.unwrap_or_else(|| {
            panic!(
                "kernel!: a fold over the placeholder stride {} that no structural range of \
                 this entry names",
                range.stride()
            )
        }))
    }
}

/// An entry lowered at expansion: its arena and root, and what the arena
/// leaves open if it is a template.
pub struct Lowered {
    pub arena: ExprArena,
    pub root: ExprId,
    pub holes: Holes,
}

/// The site an entry that takes no kernel is lowered at: an [`ExprArena`],
/// at macro expansion, with the folds open in it and the template's holes.
#[derive(Default)]
pub struct Expansion {
    arena: ExprArena,
    holes: Holes,
    /// The folds whose bodies are being built, innermost last, each with
    /// the fold it closes into.
    open: Vec<(OpenFold, ExpansionFold)>,
}

/// The fold an open one closes into at expansion: a known range, or a
/// template's hole ([`Holes`]).
#[derive(Debug, Clone)]
enum ExpansionFold {
    Known(Monoid, Range<u32>),
    Hole(Monoid, u32),
}

impl Expansion {
    /// Lower `entry`, an entry that takes no kernel, at expansion: its
    /// parameters are declared first, as uniforms, in
    /// [`AnalyzedKernel::parameters`]' order, so the arena's uniform table is
    /// the entry's declaration order — every scalar, read or not, so that a
    /// positional binding cannot shift when a parameter goes unread.
    ///
    /// # Errors
    ///
    /// When the body has a construct lowering cannot express, or `entry`
    /// takes a kernel — an argument that exists only when its host function
    /// is called.
    pub fn lower(entry: &FnItem, analyzed: &AnalyzedKernel) -> Result<Lowered, String> {
        let mut site = Expansion::default();
        let root = lower_entry(entry, analyzed, &mut site)?;
        Ok(Lowered {
            arena: site.arena,
            root,
            holes: site.holes,
        })
    }
}

impl Site for Expansion {
    type Term = ExprId;
    /// There is no argument at expansion.
    type Argument = Infallible;

    fn constant(&mut self, value: f32) -> ExprId {
        self.arena.push_const(value)
    }

    fn coordinate(&mut self, axis: Axis) -> ExprId {
        self.arena.push_var(axis.var())
    }

    fn unary(&mut self, op: OpKind, operand: ExprId) -> ExprId {
        self.arena.push_unary(op, operand)
    }

    fn binary(&mut self, op: OpKind, [a, b]: [ExprId; 2]) -> ExprId {
        self.arena.push_binary(op, a, b)
    }

    fn ternary(&mut self, op: OpKind, [a, b, c]: [ExprId; 3]) -> ExprId {
        self.arena.push_ternary(op, a, b, c)
    }

    fn fract(&mut self, x: ExprId) -> ExprId {
        library::fract(&mut self.arena, x)
    }

    fn hypot(&mut self, operands: [ExprId; 2]) -> ExprId {
        library::hypot(&mut self.arena, operands)
    }

    fn clamp(&mut self, x: ExprId, bounds: [ExprId; 2]) -> ExprId {
        library::clamp(&mut self.arena, x, bounds)
    }

    fn derivative(&mut self, e: ExprId, axis: Axis) -> ExprId {
        library::derivative(&mut self.arena, e, axis)
    }

    /// A declaration here holds a placeholder default; emission declares
    /// each with the call's value.
    fn uniform(&mut self, _scalar: Scalar<'_>) -> ExprId {
        let slot = self.arena.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: UNBOUND,
        });
        self.arena.push_uniform(slot)
    }

    fn kernel_parameter(&mut self, name: &Ident) -> Result<Infallible, String> {
        Err(format!(
            "`{name}` is a kernel, and an entry that takes one is lowered when its host function \
             is called, not at expansion (Phase D-a of {PLAN})"
        ))
    }

    /// A template's `Param` hole, which its host function fills with the
    /// same `N as f32`.
    fn count(&mut self, position: usize, name: &Ident) -> Result<ExprId, String> {
        // Borrowed: `Param(u8)` is a rewrite rule's metavariable too, and a
        // `u8` is narrower than the control plane allows. It is loud past
        // 256 rather than wrapping, and it goes with `Param` when D2 of the
        // plan deletes it — a structural hole wants a leaf of its own then.
        let hole = u8::try_from(position).map_err(|_| {
            format!(
                "`{name} as f32`: an entry reads at most {} structural parameters as values, \
                 the width of the IR's `Param` leaf",
                usize::from(u8::MAX) + 1
            )
        })?;
        Ok(self.arena.push_param(hole))
    }

    fn open_fold(
        &mut self,
        depth: usize,
        monoid: Monoid,
        range: FoldRange,
    ) -> Result<ExprId, String> {
        let fold = match range {
            FoldRange::Known(range) => ExpansionFold::Known(monoid, range),
            FoldRange::Structural(range) => ExpansionFold::Hole(monoid, self.holes.stride(range)?),
        };
        let open = self
            .arena
            .open_fold(depth)
            .expect("lowering refuses a fold nested past the binders before it opens one");
        let index = open.index();
        self.open.push((open, fold));
        Ok(index)
    }

    fn close_fold(&mut self, body: ExprId) -> Result<ExprId, String> {
        let (open, fold) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        let fold_at = |binder: Binder| match fold {
            ExpansionFold::Known(monoid, range) => Fold::new(monoid, binder, range),
            ExpansionFold::Hole(monoid, stride) => {
                Fold::strided(monoid, binder, OPEN_RANGE, stride)
            }
        };
        self.arena
            .close_fold(open, body, fold_at)
            .map_err(|IndexSpaceFull| {
                format!(
                    "a fold whose body already binds all {} of the IR's indices \
                     (`Binder::COUNT`)",
                    Binder::COUNT
                )
            })
    }

    fn apply(&mut self, kernel: &Infallible, _at: [ExprId; 2]) -> ExprId {
        match *kernel {}
    }

    fn warp(&mut self, field: ExprId, at: [ExprId; 2]) -> ExprId {
        self.arena.warp(field, at)
    }
}

/// Lower an entry's body at `site`, inlining the block's helpers and folding
/// its `const`s. Its uniform parameters are declared first, in
/// [`AnalyzedKernel::parameters`]' order, then its kernel-typed parameters
/// are admitted, in [`AnalyzedKernel::kernel_parameters`]' — each
/// argument's uniforms after the entry's own, in the argument's order.
/// Children are lowered first so that a parent's operands always exist.
///
/// # Errors
///
/// When the body has a construct lowering cannot express there.
pub(crate) fn lower_entry<S: Site>(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    site: &mut S,
) -> Result<S::Term, String> {
    let helpers = analyzed
        .def
        .fns
        .iter()
        .filter(|f| f.role() == Role::Helper)
        .map(|f| (f.name.to_string(), f))
        .collect();
    let mut locals = Scopes::default();
    for parameter in analyzed.parameters(entry) {
        let mut uniforms = parameter.scalars().map(|scalar| site.uniform(scalar));
        let binding = match parameter.record {
            Some((record, _)) => Binding::Record(record, uniforms.collect()),
            None => Binding::Value(uniforms.next().expect("a scalar parameter is one uniform")),
        };
        locals.bind(parameter.name.to_string(), binding);
    }
    for name in analyzed.kernel_parameters(entry) {
        let argument = site.kernel_parameter(name)?;
        locals.bind(name.to_string(), Binding::Kernel(argument));
    }
    let mut lowering = Lowering {
        program: Program { analyzed, helpers },
        frame: Frame {
            role: entry.role(),
            structural: &entry.structural,
            locals,
        },
        site,
        open_folds: 0,
    };
    lowering.lower(&entry.body)
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
/// entry's parameters are bound to their uniforms and admitted arguments; an
/// inlined helper's, to its argument nodes and the kernels passed to it.
struct Frame<'a, S: Site> {
    role: Role,
    /// An entry's structural parameters; a helper has none.
    structural: &'a [Ident],
    locals: Scopes<Binding<S::Term, S::Argument>>,
}

/// State threaded through the AST → term walk.
struct Lowering<'a, S: Site> {
    program: Program<'a>,
    frame: Frame<'a, S>,
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
    fn lower(&mut self, expr: &Expr) -> Result<S::Term, String> {
        match expr {
            Expr::Ident(ident) => self.resolve(&ident.name.to_string()),

            Expr::Literal(lit) => {
                let value = lit.f32_value().map_err(|e| e.to_string())?;
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

            Expr::MethodCall(call) => {
                let method = call.method.to_string();
                let receiver = self.lower(&call.receiver)?;
                let arg_count = call.args.len();

                // Arena expressions are values, so `.clone()` is the identity.
                if method == "clone" && arg_count == 0 {
                    return Ok(receiver);
                }

                // The contramap: the receiver observed at the coordinates,
                // which are lowered after it, in order. It builds no node of
                // its own; it rewrites the receiver's.
                if method == AT {
                    let [x, y] = call.args.as_slice() else {
                        return Err(format!(
                            "`.at` observes a field at the two coordinates, and {} were supplied",
                            arg_count
                        ));
                    };
                    let x = self.lower(x)?;
                    let y = self.lower(y)?;
                    return Ok(self.site.warp(receiver, [x, y]));
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
                    let mut args = args.into_iter();
                    return Ok(match (args.next(), args.next(), args.next()) {
                        (None, ..) => self.site.unary(op, receiver),
                        (Some(a), None, _) => self.site.binary(op, [receiver, a]),
                        (Some(a), Some(b), None) => self.site.ternary(op, [receiver, a, b]),
                        (Some(_), Some(_), Some(_)) => unreachable!(
                            "OpKind::from_method_call only resolves ops of arity 1..=3"
                        ),
                    });
                }

                // Library, not primitives: the IR's one definition of each,
                // which `Kernel`'s method of the same name builds too.
                match (method.as_str(), call.args.as_slice()) {
                    ("fract", []) => Ok(self.site.fract(receiver)),
                    ("hypot", [other]) => {
                        let other = self.lower(other)?;
                        Ok(self.site.hypot([receiver, other]))
                    }
                    ("clamp", [lo, hi]) => {
                        let lo = self.lower(lo)?;
                        let hi = self.lower(hi)?;
                        Ok(self.site.clamp(receiver, [lo, hi]))
                    }
                    _ => Err(format!("Unsupported method: {}", method)),
                }
            }

            // A name in scope is resolved first, as `sema` resolves it: a
            // kernel is applied, and any other binding is not a function.
            // Then a helper is inlined, and a projection becomes a `Dwrt`
            // chain.
            Expr::Call(call) => {
                let func = call.func.to_string();
                if let Some(binding) = self.frame.locals.lookup(&func) {
                    let Binding::Kernel(kernel) = binding else {
                        return Err(format!(
                            "`{func}` is not a kernel; only a kernel is applied"
                        ));
                    };
                    let kernel = kernel.clone();
                    return self.apply(&kernel, &func, &call.args);
                }
                if let Some(helper) = self.program.helpers.get(&func).copied() {
                    return self.inline(helper, &call.args);
                }
                self.lower_projection(&func, &call.args)
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

            Expr::Field(field) => self.lower_field(field),

            // Parentheses are transparent - just recurse into the inner expression
            Expr::Paren(inner) => self.lower(inner),

            Expr::Block(block) => self.lower_block(block),
        }
    }

    /// `k(x, y)`: the kernel `k` applied at the coordinates, which are
    /// lowered first, in order — contramap (§1.2).
    fn apply(
        &mut self,
        kernel: &S::Argument,
        name: &str,
        args: &[Expr],
    ) -> Result<S::Term, String> {
        let [x, y] = args else {
            return Err(format!(
                "the kernel `{name}` is applied at the two coordinates, and {} were supplied",
                args.len()
            ));
        };
        let x = self.lower(x)?;
        let y = self.lower(y)?;
        Ok(self.site.apply(kernel, [x, y]))
    }

    /// `⊕_{i ∈ [lo, hi)} body` as one `Reduce` over its [`Fold`]: a
    /// known range, or one naming the entry's structural parameters.
    ///
    /// The body is built against the fold's index, a placeholder, and the
    /// fold is closed after it, by the IR's one definition of how a binder
    /// is chosen and a placeholder renamed ([`ExprArena::close_over`]),
    /// which `Kernel`'s folds are built through too.
    fn lower_fold(&mut self, fold: &FoldExpr) -> Result<S::Term, String> {
        let scope = RangeScope {
            consts: &self.program.analyzed.consts,
            structural: self.frame.structural,
        };
        let bounds = range_bounds(&fold.range, scope).map_err(|e| e.to_string())?;
        let range = match bounds {
            Bounds::Known(lo, hi) => FoldRange::Known(known_range(lo, hi)?),
            Bounds::Structural(range) => FoldRange::Structural(range),
        };
        // One placeholder per fold open at once, the `n`th for `n` open, so a
        // nested fold's rename never reaches its enclosing fold's index. No
        // more open than the IR has binders: the refusal names the depth,
        // where running out of placeholders would name nothing the author
        // wrote.
        if self.open_folds >= Binder::COUNT {
            return Err(format!(
                "folds nested more than {} deep: the IR binds at most that many indices at \
                 once (`Binder::COUNT`)",
                Binder::COUNT
            ));
        }
        let index = self
            .site
            .open_fold(self.open_folds, monoid(fold.reduction), range)?;
        self.frame.locals.push_scope();
        self.frame
            .locals
            .bind(fold.binder.to_string(), Binding::Index(index));
        self.open_folds += 1;
        let body = self.lower(&fold.body);
        self.open_folds -= 1;
        self.frame.locals.pop_scope();
        self.site.close_fold(body?)
    }

    /// `i as f32`. A fold's index is an `f32` lane already, so its
    /// conversion is its binder's `Var`; a `usize` const is its value's
    /// `f32`, rounded as Rust's `as` rounds it; a structural parameter is the
    /// site's [`Site::count`], the same `N as f32` its host function has.
    fn lower_cast(&mut self, cast: &CastExpr) -> Result<S::Term, String> {
        let Some(name) = cast.named() else {
            return Err("`as f32` converts a `usize`, which is a name".to_string());
        };
        if let Some(binding) = self.frame.locals.lookup(&name.to_string()) {
            return match binding {
                Binding::Index(index) => Ok(index.clone()),
                Binding::Value(_) | Binding::Record(..) => Err(format!(
                    "`{name} as f32`: `{name}` is a value, and `as f32` converts a `usize`"
                )),
                Binding::Kernel(_) => Err(format!(
                    "`{name} as f32`: `{name}` is a kernel, and `as f32` converts a `usize`"
                )),
            };
        }
        if let Some(position) = self.frame.structural.iter().position(|n| n == name) {
            return self.site.count(position, name);
        }
        match self.program.analyzed.consts.get(&name.to_string()) {
            Some(&ConstValue::Usize(count)) => Ok(self.site.constant(count as f32)),
            _ => Err(format!("`{name} as f32`: `{name}` is not a `usize`")),
        }
    }

    /// `p.x0`: the node of one field of a record binding.
    fn lower_field(&mut self, field: &FieldExpr) -> Result<S::Term, String> {
        let (record, fields) = self.record(&field.base)?;
        let def = self.program.analyzed.def.record(record);
        def.fields
            .iter()
            .position(|f| f.name == field.member)
            .map(|index| fields[index].clone())
            .ok_or_else(|| format!("no field `{}` on the record `{}`", field.member, def.name))
    }

    /// The record `expr` names, through any parentheses: a record is only
    /// ever written by name (`sema` refuses anything else).
    fn record(&self, expr: &Expr) -> Result<RecordFields<S::Term>, String> {
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
    fn record_named(&self, expr: &Expr) -> Option<RecordFields<S::Term>> {
        match self.frame.locals.lookup(&expr.named()?.to_string())? {
            Binding::Record(record, fields) => Some((*record, Rc::clone(fields))),
            Binding::Value(_) | Binding::Index(_) | Binding::Kernel(_) => None,
        }
    }

    /// The kernel binding `expr` names, if it names one: a kernel is only
    /// ever passed by name (`sema` refuses anything else).
    fn kernel_named(&self, expr: &Expr) -> Option<S::Argument> {
        match self.frame.locals.lookup(&expr.named()?.to_string())? {
            Binding::Kernel(kernel) => Some(kernel.clone()),
            Binding::Value(_) | Binding::Index(_) | Binding::Record(..) => None,
        }
    }

    /// Derivative projections (V/DX/DY and the Hessian family) map to
    /// [`library::derivative`]s, one per axis, as `Kernel::dx`/`dy` build
    /// them: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites them into
    /// chain-rule arithmetic before codegen.
    fn lower_projection(&mut self, func: &str, args: &[Expr]) -> Result<S::Term, String> {
        let Some(projection) = Projection::from_name(func) else {
            return Err(format!("Unsupported call: {func}"));
        };
        let [arg] = args else {
            return Err(format!(
                "Unsupported call: {func}/{} (projections take one argument)",
                args.len()
            ));
        };
        let mut e = self.lower(arg)?;
        for &axis in projection.axes() {
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
    /// nothing of the caller's — no local, no entry parameter, no structural
    /// parameter — is visible.
    fn inline(&mut self, helper: &FnItem, args: &[Expr]) -> Result<S::Term, String> {
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
            let binding = match (self.record_named(arg), self.kernel_named(arg)) {
                (Some((record, fields)), _) => Binding::Record(record, fields),
                (None, Some(kernel)) => Binding::Kernel(kernel),
                (None, None) => Binding::Value(self.lower(arg)?),
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
    fn resolve(&mut self, name: &str) -> Result<S::Term, String> {
        if let Some(binding) = self.frame.locals.lookup(name) {
            return match binding {
                Binding::Value(id) => Ok(id.clone()),
                Binding::Index(_) => Err(format!(
                    "`{name}` is a fold's index, a `usize`, where a value is expected: \
                     `{name} as f32`"
                )),
                Binding::Record(..) => Err(format!(
                    "`{name}` is a record, where a value is expected: read a field, \
                     `{name}.x0`"
                )),
                Binding::Kernel(_) => Err(format!(
                    "`{name}` is a kernel, where a value is expected: apply it, \
                     `{name}(X, Y)`"
                )),
            };
        }
        // The same order sema documents: a binding, then a const, then a
        // structural parameter, then the coordinates. Sema refuses a
        // parameter named X or Y, so the two stages agree without one
        // relying on the other's refusal.
        match self.program.analyzed.consts.get(name) {
            Some(&ConstValue::F32(value)) => return Ok(self.site.constant(value)),
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
            "X" => Axis::X,
            "Y" => Axis::Y,
            _ => return Err(format!("Unknown identifier: {name}")),
        };
        match self.frame.role {
            Role::Entry => Ok(self.site.coordinate(axis)),
            Role::Helper => Err(format!(
                "`{name}` in a helper: a helper takes its coordinates as arguments"
            )),
        }
    }

    /// A block's `let`s live in a scope of their own, which ends with it.
    fn lower_block(&mut self, block: &BlockExpr) -> Result<S::Term, String> {
        self.frame.locals.push_scope();
        let value = self.lower_block_contents(block);
        self.frame.locals.pop_scope();
        value
    }

    /// What a `let` binds its name to: its initializer's node, or — a record
    /// being aliased — the same fields.
    fn let_binding(&mut self, let_stmt: &LetStmt) -> Result<Binding<S::Term, S::Argument>, String> {
        match self.record_named(&let_stmt.init) {
            Some((record, fields)) => Ok(Binding::Record(record, fields)),
            None => Ok(Binding::Value(self.lower(&let_stmt.init)?)),
        }
    }

    /// A block's statements in order, then its value, in the scope the
    /// caller opened for it.
    fn lower_block_contents(&mut self, block: &BlockExpr) -> Result<S::Term, String> {
        for stmt in &block.stmts {
            match stmt {
                // The initializer is lowered before the binding exists, so it
                // sees whatever the name meant before: `let a = a + 1.0;`.
                Stmt::Let(let_stmt) => {
                    let binding = self.let_binding(let_stmt)?;
                    self.frame.locals.bind(let_stmt.name.to_string(), binding);
                }
                // Every initializer before any name: `let (a, b) = (b, a);`
                // swaps, as Rust's does.
                Stmt::LetTuple(lets) => {
                    let mut bindings = Vec::with_capacity(lets.len());
                    for let_stmt in lets {
                        bindings.push(self.let_binding(let_stmt)?);
                    }
                    for (let_stmt, binding) in lets.iter().zip(bindings) {
                        self.frame.locals.bind(let_stmt.name.to_string(), binding);
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::Ty;
    use pixelflow_ir::Placeholder;
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
        let Lowered { arena, root, .. } = Expansion::lower(entry, &unanalyzed)?;
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
    /// holes map back to the range — two different ranges to two different
    /// holes; a known fold is not a hole.
    #[test]
    fn a_structural_parameter_leaves_the_template_open() {
        let def = parse(quote! {
            pub fn f<const N: usize, const M: usize>() -> f32 {
                (0..M * 2).map(|i| X * (i as f32)).sum::<f32>()
                    + (0..N).map(|i| Y * (i as f32)).sum::<f32>()
                    + (0..3).map(|i| i as f32).sum::<f32>()
                    + (M as f32)
            }
        })
        .expect("parses");
        let analyzed = crate::sema::analyze(def).expect("analyzes");
        let Lowered {
            arena, root, holes, ..
        } = Expansion::lower(&analyzed.def.fns[0], &analyzed).expect("lowers");
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
        open.sort();
        assert_eq!(open, ["0..(M * 2)", "0..N"], "each range is its own hole");
        assert_eq!(
            known.len(),
            1,
            "`0..3` is known here: {}",
            arena.display(root)
        );
        assert_eq!(params, [1], "`M` is the second structural parameter");
    }

    /// A placeholder no hole names is a front-end bug, never a known fold:
    /// emitted as it stands it would be empty, its monoid's identity.
    #[test]
    #[should_panic(expected = "no structural range of this entry names")]
    fn a_placeholder_no_hole_names_is_refused() {
        let binder = Binder::all().next().expect("a binder");
        let stray = Fold::strided(Monoid::SUM, binder, OPEN_RANGE, HOLE_STRIDE);
        let _unreachable = Holes::default().range_of(stray);
    }

    /// A known fold is never read as a hole, whatever its range: empty at
    /// the open range's position, or strided by a pass.
    #[test]
    fn a_known_fold_is_no_hole() {
        let binder = Binder::all().next().expect("a binder");
        let holes = Holes::default();
        for known in [
            Fold::new(Monoid::SUM, binder, OPEN_RANGE),
            Fold::strided(Monoid::SUM, binder, 0..8, HOLE_STRIDE),
        ] {
            assert!(holes.range_of(known).is_none(), "{known}");
        }
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
        let first = Placeholder::nth(0).expect("a placeholder").var();
        arena
            .nodes()
            .any(|(_, node)| matches!(node, ExprNode::Var(v) if v >= first))
    }

    /// A fold is one `Reduce` over its range, its body reading the binder's
    /// `Var`, and the arena holds nothing else: the placeholder the body was
    /// built against stayed in the copy it was built in.
    #[test]
    fn a_fold_lowers_to_one_reduce_and_leaves_nothing_behind() {
        let (arena, root) = lowered(quote! { || (2..6).map(|i| X * (i as f32)).sum() });
        let ExprNode::Reduce { fold: range, body } = arena.node(root) else {
            panic!("expected a fold, got {}", arena.display(root));
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
        let ExprNode::Reduce { fold: range, .. } = arena.node(root) else {
            panic!("expected a fold, got {}", arena.display(root));
        };
        assert_eq!(range.range(), 16_777_215..16_777_216);
    }

    /// Folds nest as deep as the IR has binders, and no deeper: one more is
    /// refused, not a panic.
    #[test]
    fn folds_nest_as_deep_as_the_ir_has_binders() {
        use proc_macro2::TokenStream;
        let fold = |body: TokenStream| quote! { (0..1).map(|i| #body).sum::<f32>() };
        let nest = |depth: usize| (0..depth).fold(quote! { X }, |body, _| fold(body));
        let closure = |body: TokenStream| quote! { || #body };
        let (arena, root) = lowered(closure(nest(Binder::COUNT)));
        let ExprNode::Reduce { fold: outer, .. } = arena.node(root) else {
            panic!("expected a fold");
        };
        assert_eq!(usize::from(outer.binder().slot()), Binder::COUNT - 1);
        let Err(err) = lower_unanalyzed(closure(nest(Binder::COUNT + 1))) else {
            panic!("one binder deeper than the IR has");
        };
        assert!(err.contains("folds nested more than"), "got: {err}");
    }

    /// `let (a, b) = (b, a);` binds both names at once, to what each
    /// expression meant before the statement: it swaps, as Rust's does.
    #[test]
    fn a_tuple_let_binds_every_name_at_once() {
        let (arena, root) = lowered(quote! {
            || { let (a, b) = (X, Y); let (a, b) = (b, a); a - b }
        });
        let ExprNode::Binary(OpKind::Sub, a, b) = arena.node(root) else {
            panic!("`a - b`, got {}", arena.display(root));
        };
        assert!(matches!(arena.node(a), ExprNode::Var(1)), "`a` is Y");
        assert!(matches!(arena.node(b), ExprNode::Var(0)), "`b` is X");
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

    /// A value is not converted: `h as f32` of a `let` is refused here, as
    /// in `sema`, without relying on it — `as f32` converts a `usize`.
    #[test]
    fn a_value_is_not_converted() {
        let Err(err) = lower_unanalyzed(quote! { || { let h = X; h as f32 } }) else {
            panic!("`h` is a value");
        };
        assert!(err.contains("`h` is a value"), "got: {err}");
    }
}
