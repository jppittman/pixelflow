//! The program the emitter is handed: the vocabulary every stage between the
//! arena and the machine code shares.
//!
//! A scheduled value ([`Def`]) and the operation that defines it ([`ScheduledOp`]),
//! the scopes a schedule is split into ([`ScopedSchedule`]) and the blocks each
//! scope's arms are ([`Flow`]). None of it names a register or a byte: it is what lowering
//! produces and what allocation and emission consume, which is why it lives
//! under neither. The stages that build it are [`Nest`], [`Own`] and [`Layout`].

mod layout;
mod lower;
mod ownership;
mod scopes;
mod tree;
pub(crate) use layout::Layout;
pub(crate) use lower::arena_to_schedule;
pub(crate) use ownership::Own;
pub(crate) use scopes::Nest;

use alloc::vec::Vec;

use pixelflow_ir::Lit;
use pixelflow_ir::fold::{Binder, Bound, Fold};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::variance::Variance;

/// The lane a literal tabulates to, as the `f32` a [`ScheduledOp::Const`]
/// carries: a real is itself, a true mask is the all-ones lane and a false
/// mask is zero. The backends key a constant by its bits, so the `f32` here
/// is the data plane's 32-bit pattern and nothing is read as a number.
pub(crate) fn lane_pattern(lit: Lit) -> f32 {
    f32::from_bits(lit.bits())
}

/// A value in the program (SSA-style).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ValueId(pub(super) u64);

/// One step of a schedule: a value, and the operation that defines it.
///
/// Every step defines exactly one value — the DAG is in SSA form — so a
/// schedule is a sequence of these, and a value's program point is its index
/// in that sequence.
#[derive(Clone, Debug)]
pub(crate) struct Def {
    /// The value this step defines.
    pub(super) value: ValueId,
    /// The operation that computes it.
    pub(super) op: ScheduledOp,
    /// What the value varies with: the variance of the arena node it was
    /// lowered from (`compute_arena_variance`), which scoping places it by.
    pub(super) variance: Variance,
}

/// Info about an operation in the schedule.
#[derive(Debug, Clone)]
pub(crate) enum ScheduledOp {
    /// A fold's binder, read as a value: the counter of the innermost
    /// enclosing loop that binds it.
    Var(Binder),
    /// Constant value
    Const(f32),
    /// Unary op with input value
    Unary(UnaryOp, ValueId),
    /// Binary op with input values
    Binary(BinaryOp, ValueId, ValueId),
    /// A comparison: a mask, not a number.
    Compare(CompareOp, ValueId, ValueId),
    /// `a·b + c`, one rounding.
    MulAdd(ValueId, ValueId, ValueId),
    /// `if mask then a else b`, the two arms second and third.
    If(ValueId, ValueId, ValueId),
    /// Bit-shift by a compile-time immediate: the value is `ValueId`, and the
    /// shift count is folded out of the `Const` RHS by `lower::arena_to_schedule`
    /// (so it never becomes a scheduled value / register).
    ShiftImm(ShiftOp, ValueId, u8),
    /// Table gather: read the buffer whose base is the second operand
    /// at the lane index computed by the first. Lowered from
    /// `RawGather(Buffer(slot), index)`; the `Buffer` leaf *is* the base — a
    /// [`ScheduledOp::Context`] def, a pointer value
    /// the allocator places like any other — so the index is the one vector
    /// operand and the base the one pointer operand.
    Gather(ValueId, ValueId),
    /// A `Gather` whose index is the same in every lane: one scalar load,
    /// broadcast. The same `RawGather(Buffer(slot), index)`, split from
    /// [`ScheduledOp::Gather`] by `lower::arena_to_schedule` on the index's
    /// variance — it lacks the lane binder's bit, so lane 0 *is* the index
    /// and the other lanes are copies of it. A glyph's per-piece table
    /// reads are addressed by its fold's own binder and nothing else, which
    /// makes them this and not a gather; the split is what turns a per-lane
    /// address sequence (`vpextrd`/`vinsertps` ×4, `vgatherdps`, four
    /// `umov`/`ldr`/`ins`) into `cvttss2si` + `vbroadcastss [base + idx*4]`.
    /// Index first, base second, as `Gather`.
    Broadcast(ValueId, ValueId),
    /// Per-call scalar, broadcast from a block: the value at `4 * offset`
    /// from the block whose base is the pointer operand — the link's
    /// uniform block, or the origin's. Not a leaf to the placement, since
    /// the load is an instruction worth doing once per call rather than
    /// once per batch. The offset is a `UniformId`'s slot, at its width.
    Uniform(ValueId, u64),
    /// The `k`-th pointer of the context the kernel is called with: a
    /// buffer's base for `k` below the buffer count, the link's uniform
    /// block and the origin block after. The definition of every
    /// pointer value; no operands, variance `CONST`,
    /// so it is placed in the per-call scope and carried into the loops
    /// inside by `plan_carries` on the strength of its reads there — one
    /// load per call (docs/archive/plans/2026-09-22-a-pointer-is-a-value.md).
    Context(u16),
    /// The lane fold's binder: the constant `[0, 1, …, L−1]`. The fold
    /// whose binder this is executes by lanes (its body is inlined into its
    /// parent's schedule — see `lower::arena_to_schedule`), so the binder is a
    /// leaf here rather than a loop counter. Its def's variance is the lane
    /// binder's bit alone (the arena `Var`'s), so every value built on it
    /// carries that bit; lowering reads lane-uniformity off the arena's table,
    /// and scoping keeps lane-varying values in the storing scope by it.
    Lanes,
    /// The store the lattice's folds wrap a kernel in: `value`'s first
    /// `lanes` lanes at `out + 4·(row·pitch + col)`, `row` and `col` being
    /// the enclosing folds' binders. This def *is* the lane fold, executed
    /// by lanes: `lane` is that fold's binder, which it closes over the way
    /// any `Reduce` closes over its own, and `lanes` its trip count — the
    /// full batch, or a row's remainder — so two lane folds sharing one
    /// arena `Write` are two defs of different widths reading one value.
    Write {
        row: Binder,
        col: Binder,
        lane: Binder,
        lanes: u32,
        value: ValueId,
    },
    /// Two effects, the first then the second: the unit monoid's own
    /// combine, which is what a `SEQ` fold over a row's main batches and its
    /// remainder is. Reads no register — the schedule's order *is* the
    /// sequencing — and defines no value.
    Seq(ValueId, ValueId),
    /// A surviving bounded fold: `⊕` over the indices from `lo` to `hi`, under
    /// `fold`'s algebra and binder, whose body is the value named by `body` —
    /// in *this schedule's* numbering (`lower::arena_to_schedule` maps it like
    /// any other child). `scopes::extract_folds` carves the body's closure out
    /// into its own [`ScopeFold`]; `body` is then the root of that scope's
    /// schedule, the value each trip accumulates, which selection reads when
    /// it opens the loop for the [`Scope::Fold`] this def opens. It is never
    /// an operand of the scope this def sits in.
    ///
    /// `lo` and `hi` are the arena node's bounds, copied where the node is in
    /// hand (`lower::arena_to_schedule`). A literal is selection's constant; a
    /// value is an operand ([`operands`]): `lo` is the loop's seed, read where
    /// the loop is entered, and `hi` is read at its latch, both in the scope
    /// this def sits in. Selection reads a known range with `KnownRange::of`.
    Reduce {
        fold: Fold,
        lo: Bound<ValueId>,
        hi: Bound<ValueId>,
        body: ValueId,
    },
}

/// The ops a backend selects, as types: [`ScheduledOp`] carries these and not
/// an [`OpKind`], so a backend's `match` over one is exhaustive and the
/// compiler is what says a target owes an encoding. `legalize` leaves only
/// ops that have a place here, and `lower::arena_to_schedule` is the one
/// conversion; an op with none is a bypassed pipeline.
pub(crate) trait Legal: Sized {
    /// The legal op `kind` is, if it is one.
    fn of(kind: OpKind) -> Option<Self>;
}

macro_rules! legal_ops {
    ($($(#[$meta:meta])* $name:ident { $($op:ident),+ $(,)? })+) => {$(
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub(crate) enum $name { $($op),+ }

        #[cfg(test)]
        impl $name {
            /// Every op of this type, for a test that must cover each.
            pub(crate) const ALL: &'static [Self] = &[$(Self::$op),+];
        }

        impl Legal for $name {
            fn of(kind: OpKind) -> Option<Self> {
                match kind {
                    $(OpKind::$op => Some(Self::$op),)+
                    _ => None,
                }
            }
        }
    )+};
}

legal_ops! {
    /// One operand in, one lane out.
    UnaryOp { Neg, Abs, Sqrt, Rsqrt, Recip, Floor, Ceil, Round, TruncToInt, IntToFloat }
    /// Two lanes in, a number or a bit pattern out (`IAdd`, `BitAnd` and
    /// `BitOr` act on the bits).
    BinaryOp { Add, Sub, Mul, Div, Min, Max, IAdd, BitAnd, BitOr }
    /// Two lanes in, a mask out.
    CompareOp { Lt, Le, Gt, Ge, Eq, Ne }
    /// A lane shifted by an immediate.
    ShiftOp { Shl, Shr }
}

/// A bitwise op on two masks that leaves a mask: the part of [`BinaryOp`] a
/// target with a register file for masks spells in it ([`BinaryOp::on_masks`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MaskOp {
    And,
    Or,
}

impl BinaryOp {
    /// The mask op this op is on `a` and `b`, each its operand as a mask when
    /// it is one, with the two masks: a bitwise op on two masks leaves a mask,
    /// and every other pairing yields a number, or a bit pattern read as one.
    /// A comparison is a mask whatever it compares, so it takes no operands to
    /// say so.
    ///
    /// The one place that says which binary ops keep a mask a mask. A
    /// selector that holds masks apart from numbers (AVX-512's opmasks) hands
    /// it each operand it holds as one and takes the file of the result from
    /// the answer; [`ScheduledOp::is_mask`] hands it what the schedule says of
    /// each operand, to ask that choice before a selector is reached.
    pub(crate) fn on_masks<M>(self, a: Option<M>, b: Option<M>) -> Option<(MaskOp, M, M)> {
        let op = match self {
            Self::BitAnd => MaskOp::And,
            Self::BitOr => MaskOp::Or,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Min | Self::Max | Self::IAdd => {
                return None;
            }
        };
        Some((op, a?, b?))
    }
}

/// The legal op `op` is. `passes::legalize` leaves only these, so none is a
/// bad kernel: it is a bypassed pipeline, and nothing a caller could act on.
pub(crate) fn legal<T: Legal>(op: OpKind) -> T {
    T::of(op).unwrap_or_else(|| not_legal(op))
}

#[cold]
#[inline(never)]
fn not_legal(op: OpKind) -> ! {
    panic!(
        "{op:?} is not a backend-legal op -- `passes::legalize` \
         leaves only backend-legal ops, so the pipeline was bypassed, not the kernel \
         refused"
    )
}

/// Which scope of a loop nest: the body the call runs once, or one of the
/// folds nested in it.
///
/// A **name**, not a coordinate: nesting is not a total order of scopes, since
/// a scope opens in the *middle* of another, which is what a surviving
/// `Reduce` is — the parent's own defs sit on both sides of the fold's. Order
/// is meaningful only within one scope; this is the key that says which one.
///
/// The nest is a **tree**: every fold hangs off whichever scope holds its
/// def. The lattice's own rows and columns are folds too
/// (docs/archive/plans/2026-09-16-collapse-is-a-fold.md), so there is the body, run
/// once per call, and folds all the way down. The derived `Ord` is therefore **not** nesting order — it
/// is a total order over names, for deterministic keying.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Scope {
    /// The whole function: what runs once per call, and holds the outermost
    /// fold's def.
    Body,
    /// A surviving `Reduce`'s loop body. It opens in the *middle* of its parent's
    /// schedule, which is what makes the nest a tree.
    Fold(usize),
}

/// A schedule split by scope: the body the call runs once, and the folds
/// nested in it.
///
/// This is the loop nest as data. `body` is what happens once per call — the
/// per-call values, and the outermost fold's def — and each fold's schedule
/// is what happens once per trip of its loop. A value is computed by the
/// outermost scope binding every binder it depends on, so a value depending
/// on nothing is computed once per call and one depending only on the row
/// binder once per row (docs/archive/plans/2026-09-16-collapse-is-a-fold.md §2.2);
/// a fold that reads it names it in [`ScopeFold::outer`]. A scope's roots —
/// what it computes for the scopes inside it — are its defs some fold inside
/// names there: derived, not stored.
///
/// The body is a field rather than `folds[0]` because it is genuinely a
/// different thing: it wraps everything and opens nowhere.
///
/// `A` is what each stage after [`Nest`] works out about every scope: nothing
/// as [`Nest`] leaves it, then its [`ownership::Ownership`] ([`Own`]), then its
/// [`Flow`] ([`Layout`]). A stage cannot be handed a nest its predecessor has
/// not run on.
pub(crate) struct ScopedSchedule<A> {
    /// What runs once per call.
    pub(super) body: ScopeRegion<A>,
    /// The surviving folds, in [`Scope::Fold`] order — every one of them,
    /// the lattice's own included.
    pub(super) folds: Vec<ScopeFold<A>>,
}

/// One scope of a [`ScopedSchedule`] that opens nowhere: the body.
pub(crate) struct ScopeRegion<A> {
    /// What it computes, in topological order.
    pub(super) schedule: Vec<Def>,
    /// What the stages since scoping have worked out about it.
    pub(super) analysis: A,
}

/// One surviving fold of a [`ScopedSchedule`]: a scope that opens in the
/// middle of another scope.
pub(crate) struct ScopeFold<A> {
    /// The scope whose schedule holds this loop's def.
    pub(super) parent: Scope,
    /// The `Reduce` def of `parent` this is the body of.
    pub(super) reduce: ValueId,
    /// The fold's free variables: the values an enclosing scope computes that
    /// this fold reads, which `schedule` — only what the fold defines — does
    /// not hold.
    pub(super) outer: Vec<ValueId>,
    /// The loop body, in topological order.
    pub(super) schedule: Vec<Def>,
    /// What the stages since scoping have worked out about it.
    pub(super) analysis: A,
}

impl<A> ScopeFold<A> {
    /// The ids the fold names: the ones it defines and the ones it reads
    /// from outside. Those its parent's schedule also holds are what the
    /// fold reads from there.
    pub(super) fn reads(&self) -> impl Iterator<Item = ValueId> + '_ {
        let defined = self.schedule.iter().map(|def| def.value);
        defined.chain(self.outer.iter().copied())
    }
}

impl<A> ScopedSchedule<A> {
    /// What `scope` holds: its defs, the ids it reads from outside (none for
    /// the body) and its analysis.
    fn at(&self, scope: Scope) -> (&[Def], &[ValueId], &A) {
        match scope {
            Scope::Body => (&self.body.schedule, &[], &self.body.analysis),
            Scope::Fold(j) => {
                let fold = &self.folds[j];
                (&fold.schedule, &fold.outer, &fold.analysis)
            }
        }
    }

    /// This nest with each scope's analysis replaced by what `f` works out
    /// about that scope, reading the whole nest as it was.
    fn analyze<B>(self, f: impl Fn(&Self, Scope) -> B) -> ScopedSchedule<B> {
        let body = f(&self, Scope::Body);
        let analyses: Vec<B> = (0..self.folds.len())
            .map(|j| f(&self, Scope::Fold(j)))
            .collect();
        let body = ScopeRegion {
            schedule: self.body.schedule,
            analysis: body,
        };
        let folds = self
            .folds
            .into_iter()
            .zip(analyses)
            .map(|(fold, analysis)| ScopeFold {
                parent: fold.parent,
                reduce: fold.reduce,
                outer: fold.outer,
                schedule: fold.schedule,
                analysis,
            })
            .collect();
        ScopedSchedule { body, folds }
    }
}

/// Which arm of an `If` node a branch skips.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum IfArm {
    /// The `if_true` arm: skipped when all lanes of the mask are false.
    True,
    /// The `if_false` arm: skipped when all lanes of the mask are true.
    False,
}

/// How a scope's schedule runs: its defs in order, with each arm that owns
/// work nested as the block it is, immediately before its `If` (`layout`).
/// Selection walks it and emits the guard and the block after it around each
/// arm; nothing recovers a region from a position.
#[derive(Debug)]
pub(crate) struct Flow {
    /// What the scope runs, outermost.
    pub(super) items: Vec<Item>,
    /// Every arm that owns work, in the order of their `If`s.
    pub(super) arms: Vec<ArmBlock>,
}

/// One step of a block.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Item {
    /// The def at this position of the scope's schedule.
    Def(usize),
    /// The block of this entry of [`Flow::arms`], run where it stands.
    Arm(usize),
}

/// An arm of an `If` that owns work of its own: skipped when no lane of the
/// mask selects it.
#[derive(Debug)]
pub(crate) struct ArmBlock {
    /// Which arm of its `If`.
    pub(super) arm: IfArm,
    /// The `If`'s mask, bound before the block begins.
    pub(super) mask: ValueId,
    /// What the arm runs.
    pub(super) items: Vec<Item>,
}

impl ScheduledOp {
    /// Whether the value this op defines is a mask: a comparison, or a
    /// bitwise op on two operands that are masks ([`BinaryOp::on_masks`]),
    /// `operand_is_mask` saying which. A reinterpretation defines no value
    /// (lowering maps it to its operand), so it keeps its operand's answer.
    ///
    /// A mask is a bit pattern, not a number, and a target may hold it in a
    /// file of its own (AVX-512's opmasks), so
    /// what a position that needs a number may take is asked here, before any
    /// target is chosen. The bitwise rule is the selector's own (`on_masks`);
    /// that a comparison is a mask, and that nothing else is, is the selector's
    /// choice of file restated, and the AVX-512 test
    /// `a_value_is_an_opmask_exactly_when_is_mask_says_it_is_a_mask` holds the
    /// two equal.
    pub(crate) fn is_mask(&self, operand_is_mask: impl Fn(ValueId) -> bool) -> bool {
        match self {
            Self::Compare(..) => true,
            Self::Binary(op, a, b) => op
                .on_masks(
                    operand_is_mask(*a).then_some(()),
                    operand_is_mask(*b).then_some(()),
                )
                .is_some(),
            _ => false,
        }
    }
}

/// The values an operation reads *as registers*, in operand order.
///
/// A `Reduce`'s operands are its value bounds, the seed and the end its loop
/// reads; its body is not an operand: by the time one reaches a schedule this
/// function walks, `scopes::extract_folds` has already carved its body out
/// into its own `ScopeFold`; the `ValueId` `ScheduledOp::Reduce` still
/// carries names that scope's root, never an operand this scope's allocation
/// resolves. What the loop it opens reads from this scope is a dependency
/// all the same, and the ownership has it as one (`ownership::FoldReads`). A
/// `Seq` sequences two effects and reads no register; a `Write` reads the one
/// value it stores — its row and column are binders, found where their folds
/// keep them, not operands.
pub(crate) fn operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let (a, b, c) = match sop {
        ScheduledOp::Var(_)
        | ScheduledOp::Lanes
        | ScheduledOp::Const(_)
        | ScheduledOp::Context(_)
        | ScheduledOp::Uniform(..)
        | ScheduledOp::Seq(..) => (None, None, None),
        ScheduledOp::Reduce { lo, hi, .. } => (lo.value(), hi.value(), None),
        // A gather's base is a pointer, read through `pointer_operand`; the
        // index is its one vector operand.
        ScheduledOp::Unary(_, a)
        | ScheduledOp::ShiftImm(_, a, _)
        | ScheduledOp::Gather(a, _)
        | ScheduledOp::Broadcast(a, _) => (Some(*a), None, None),
        ScheduledOp::Write { value, .. } => (Some(*value), None, None),
        ScheduledOp::Binary(_, a, b) | ScheduledOp::Compare(_, a, b) => (Some(*a), Some(*b), None),
        ScheduledOp::MulAdd(a, b, c) | ScheduledOp::If(a, b, c) => (Some(*a), Some(*b), Some(*c)),
    };
    [a, b, c].into_iter().flatten()
}

/// The address an operation reads, if it reads one: a gather's, a
/// broadcast's or a uniform load's base, always a pointer value.
/// One at most.
pub(crate) fn pointer_operand(sop: &ScheduledOp) -> Option<ValueId> {
    match sop {
        ScheduledOp::Gather(_, base) | ScheduledOp::Broadcast(_, base) => Some(*base),
        ScheduledOp::Uniform(base, _) => Some(*base),
        _ => None,
    }
}

/// Every value an operation reads, of either class: [`operands`] then
/// [`pointer_operand`]. What a liveness question that does not care which
/// file a value lives in asks — how many times a root is read, whether a
/// schedule is topological.
pub(crate) fn all_operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    operands(sop).chain(pointer_operand(sop))
}

/// Every value an operation is *built from*: its operands of both classes,
/// plus the body a `Reduce` folds (its value bounds are operands already) and
/// the two effects a `Seq` orders — the children a walk of the DAG's
/// structure follows, as opposed to the registers an instruction reads
/// ([`operands`]).
fn structural_children(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let extra = match sop {
        ScheduledOp::Reduce { body, .. } => [Some(*body), None],
        ScheduledOp::Seq(a, b) => [Some(*a), Some(*b)],
        _ => [None, None],
    };
    all_operands(sop).chain(extra.into_iter().flatten())
}
