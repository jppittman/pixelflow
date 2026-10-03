//! The program the emitter is handed: the vocabulary every stage between the
//! arena and the machine code shares.
//!
//! A scheduled value ([`Def`]) and the operation that defines it ([`ScheduledOp`]),
//! the scopes a schedule is split into ([`ScopedSchedule`]) and the branches over
//! them ([`IfGuard`]). None of it names a register or a byte: it is what lowering
//! produces and what allocation and emission consume, which is why it lives
//! under neither.

pub(crate) mod guards;
pub(crate) mod layout;
pub(crate) mod ownership;
pub(crate) mod tree;

use alloc::vec::Vec;

use pixelflow_ir::fold::{Binder, Fold};
use pixelflow_ir::kind::OpKind;

/// A value in the program (SSA-style).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// Which register file a value lives in: a vector of `f32` lanes, or an
/// address.
///
/// A function of the defining op ([`ScheduledOp::class`]), and every
/// consumer knows by position which it reads — a `Gather`'s index is a
/// vector and its base is a pointer. The two classes never compete for a
/// register, so allocation runs once per class over one schedule
/// (`LinearScan`), each pass blind to the other's values, and the assembler
/// gets a `PtrReg` where it demands one because the allocator never held
/// the address anywhere else (docs/plans/2026-09-22-a-pointer-is-a-value.md).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Class {
    /// One SIMD batch of `f32`, in a `Reg`.
    Vector,
    /// An address, in a `PtrReg`.
    Pointer,
}

/// One step of a schedule: a value, and the operation that defines it.
///
/// Every step defines exactly one value — the DAG is in SSA form — so a
/// schedule is a sequence of these, and a value's program point is its index
/// in that sequence.
#[derive(Clone, Debug)]
pub struct Def {
    /// The value this step defines.
    pub value: ValueId,
    /// The operation that computes it.
    pub op: ScheduledOp,
}

/// Info about an operation in the schedule.
#[derive(Debug, Clone)]
pub enum ScheduledOp {
    /// Variable reference (input register)
    Var(u8),
    /// Constant value
    Const(f32),
    /// Unary op with input value
    Unary(OpKind, ValueId),
    /// Binary op with input values
    Binary(OpKind, ValueId, ValueId),
    /// Ternary op with input values
    Ternary(OpKind, ValueId, ValueId, ValueId),
    /// Bit-shift by a compile-time immediate: `op` is `Shl` or `Shr`, the value
    /// is `ValueId`, and the shift count is folded out of the `Const` RHS by
    /// `arena_to_schedule` (so it never becomes a scheduled value / register).
    ShiftImm(OpKind, ValueId, u8),
    /// Bound-memory gather: read the buffer whose base is the second operand
    /// at the lane index computed by the first. Lowered from
    /// `RawGather(Buffer(slot), index)`; the `Buffer` leaf *is* the base — a
    /// [`ScheduledOp::Context`] def, a [`Class::Pointer`] value
    /// the allocator places like any other — so the index is the one vector
    /// operand and the base the one pointer operand.
    Gather(ValueId, ValueId),
    /// A `Gather` whose index is the same in every lane: one scalar load,
    /// broadcast. The same `RawGather(Buffer(slot), index)`, split from
    /// [`ScheduledOp::Gather`] by `arena_to_schedule` on the index's
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
    /// [`Class::Pointer`] value; no operands, variance `CONST`,
    /// so it is placed in the per-call scope and carried into the loops
    /// inside by `plan_carries` on the strength of its reads there — one
    /// load per call where every gather used to reload it
    /// (docs/plans/2026-09-22-a-pointer-is-a-value.md).
    Context(u16),
    /// The lane fold's binder: the constant `[0, 1, …, L−1]`. The fold
    /// whose binder this is executes by lanes (its body is inlined into its
    /// parent's schedule — see `arena_to_schedule`), so the binder is a
    /// leaf here rather than a loop counter. Carries the binder so its
    /// variance bit is the fold's, which is what "lane-uniform" is read off.
    Lanes(Binder),
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
    /// A surviving bounded fold: `⊕` over `fold`'s visited indices, whose
    /// body is the value named by the second field — in *this schedule's*
    /// numbering (`arena_to_schedule` maps it like any other child), before
    /// `extract_folds` carves the body out into its own
    /// [`ScopeFold`]. Kept only so `schedule_variance` can look
    /// the body's variance up (`Reduce`'s own result is the body's variance
    /// with the binder's own bit removed) and so `extract_folds` can find
    /// the body's closure; the emitter never resolves it as an operand —
    /// the loop's result comes from `regalloc::Allocation::opens_at`
    /// naming the [`Scope::Fold`] this def opens, not from this
    /// `ValueId`.
    Reduce(Fold, ValueId),
    /// A surviving `Guard`: the mask, and its two arms' names. `mask` is a
    /// real value in *this* schedule (`arena_to_schedule` maps it like any
    /// other child); the two `KernelKey`s are not — they name kernels whose
    /// bodies live in wholly separate arenas, resolved through
    /// `KernelStore::resolve` by `extract_guards`, which schedules each arm
    /// as its own [`Scope::GuardArm`], exactly as `extract_folds`
    /// carves a [`ScheduledOp::Reduce`]'s body into its own
    /// [`Scope::Fold`] — except an arm is not carved *out of*
    /// anything here, since nothing of it was ever in this schedule to carve.
    /// The emitter never resolves this def's operands the ordinary way: its
    /// own `ValueId` is forced to a slot (`regalloc`'s `Scan`, mirroring a
    /// `Reduce`'s accumulator), and the two arms' scopes — found by
    /// `regalloc::Allocation::guard_opening_at` — are each emitted as a
    /// nested scope bracketed by a branch instead of a loop
    /// (docs/plans/2026-09-12-emit-should-just-emit.md §3).
    Guard(
        ValueId,
        pixelflow_ir::key::KernelKey,
        pixelflow_ir::key::KernelKey,
    ),
}

impl ScheduledOp {
    /// Which register file the value this op defines lives in: a
    /// [`ScheduledOp::Context`] is an address, everything else is a vector
    /// (an effect's "value" included, which is never placed anywhere).
    #[must_use]
    pub fn class(&self) -> Class {
        match self {
            ScheduledOp::Context(_) => Class::Pointer,
            _ => Class::Vector,
        }
    }
}

/// Which scope of a loop nest: the body the call runs once, or one of the
/// folds nested in it.
///
/// A **name**, not a coordinate. It used to be half of one — `Point` was
/// `(scope, index)` ordered lexicographically — and that only worked while
/// the nest was a *chain*: scopes totally ordered by nesting, and all of an
/// outer scope's code preceding all of an inner scope's. The second stops
/// being true the moment a scope opens in the *middle* of another, which is
/// what a surviving `Reduce` is: the parent's own defs sit on both sides of
/// the fold's. So the ordering moved to where it is always meaningful —
/// within one scope — and this is now only the key that says which one.
///
/// The nest is a **tree**: every fold hangs off whichever scope holds its
/// def. The lattice's own rows and columns are folds too
/// (docs/plans/2026-09-16-collapse-is-a-fold.md), so there is no spine of
/// regions any more — there is the body, run once per call, and folds all
/// the way down. The derived `Ord` is therefore **not** nesting order — it
/// is a total order over names, for deterministic keying.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// The whole function: what runs once per call, and holds the outermost
    /// fold's def.
    Body,
    /// A surviving `Reduce`'s loop body, indexing
    /// `NestAllocation::folds`. It opens in the *middle* of its parent's
    /// schedule, which is what makes the nest a tree.
    Fold(usize),
    /// One arm of a surviving `Guard`, indexing
    /// `NestAllocation::guard_arms`. Also opens in the middle of its
    /// parent's schedule — at the `Guard` def, exactly as a fold opens at
    /// its `Reduce` — but it is a branch, not a loop: it runs at most once
    /// per time its parent's def is reached, carries nothing in (a guard
    /// arm's arena is wholly separate from its parent's — a name, not a
    /// closure), and hands back one result the parent stores to a shared
    /// slot (docs/plans/2026-09-12-emit-should-just-emit.md §3). Additive
    /// to [`Scope::Fold`] rather than unified with it: the two are the same
    /// shape to the frame layout (a region opening mid-schedule) but
    /// different in what crosses the boundary, and every existing `Fold`
    /// consumer already assumes a `Reduce` at the opening def — keeping
    /// this a separate variant means those consumers need no change to
    /// keep answering exactly as they did before a `Guard` ever reached
    /// this allocator (the G2 byte-identity gate).
    GuardArm(usize),
}

/// A schedule split by scope: the body the call runs once, and the folds
/// nested in it.
///
/// This is the loop nest as data. `body` is what happens once per call — the
/// per-call values, and the outermost fold's def — and each fold's schedule
/// is what happens once per trip of its loop. Each scope's `roots` are the
/// values it computes for the scopes inside it: placed here, by the
/// outermost scope binding every binder the value depends on, so a value
/// depending on nothing is computed once per call and one depending only on
/// the row binder once per row (docs/plans/2026-09-16-collapse-is-a-fold.md
/// §2.2).
///
/// The body is a field rather than `folds[0]` because it is genuinely a
/// different thing: it wraps everything and opens nowhere.
pub struct ScopedSchedule {
    /// What runs once per call.
    pub body: ScopeRegion,
    /// The surviving folds, in [`Scope::Fold`] order — every one of them,
    /// the lattice's own included.
    pub folds: Vec<ScopeFold>,
    /// The surviving `Guard`s' arms, in [`Scope::GuardArm`] order. Built
    /// separately from `folds` — after `RegisterAllocator::allocate_nest`'s
    /// caller has already carved the folds out and clustered their arms —
    /// because a guard arm's schedule does not come from carving anything
    /// out of this nest's own; it comes from resolving a wholly separate
    /// `KernelKey` (see [`Scope::GuardArm`]'s doc).
    pub guard_arms: Vec<ScopeGuardArm>,
}

/// One scope of a [`ScopedSchedule`] that opens nowhere: the body.
pub struct ScopeRegion {
    /// Values this scope computes for the ones inside it.
    pub roots: Vec<ValueId>,
    /// What it computes, in topological order.
    pub schedule: Vec<Def>,
    /// This scope's `If` guards: which entries of `schedule` each branch
    /// skips. A table over `schedule`, handed in with it — the allocator
    /// places split ranges around the arms it names and the emitter branches
    /// over them, and neither derives them.
    pub(crate) guards: Vec<IfGuard>,
}

/// One surviving fold of a [`ScopedSchedule`]: a scope that opens in the
/// middle of another scope.
pub struct ScopeFold {
    /// The scope whose schedule holds this loop's def.
    pub parent: Scope,
    /// Which def of `parent` — the `Reduce` this is the body of.
    pub at: usize,
    /// Values this fold computes for the scopes inside it.
    pub roots: Vec<ValueId>,
    /// The loop body, in topological order.
    pub schedule: Vec<Def>,
    /// The body's `If` guards, as [`ScopeRegion::guards`].
    pub(crate) guards: Vec<IfGuard>,
}

/// One arm of a surviving `Guard`, as an input to
/// `RegisterAllocator::allocate_nest`: a scope that opens in the middle of
/// another scope, exactly like [`ScopeFold`], but with no `roots` — its
/// schedule is wholly self-contained (a separate arena's own, freshly
/// numbered), so it has nothing to read from its parent beyond the branch
/// condition the parent resolves before ever reaching this scope.
pub struct ScopeGuardArm {
    /// The scope whose schedule holds the `Guard` def this is an arm of.
    pub parent: Scope,
    /// The `Guard` def's position in `parent`'s schedule.
    pub at: usize,
    /// Which of the `Guard`'s two arms this is.
    pub arm: IfArm,
    /// This arm's own evaluation order, in topological order, ending at the
    /// value the parent stores to the `Guard`'s result slot.
    pub schedule: Vec<Def>,
    /// The arm's `If` guards, as [`ScopeRegion::guards`].
    pub(crate) guards: Vec<IfGuard>,
}

/// Which arm of an `If` node a guard branch skips or targets.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IfArm {
    /// The `if_true` arm: skipped when all lanes of the mask are false.
    True,
    /// The `if_false` arm: skipped when all lanes of the mask are true.
    False,
}

impl IfArm {
    /// Both arms of an `If`.
    pub const ALL: [Self; 2] = [Self::True, Self::False];
}

/// A value associated with each arm of an `If` node (`True` and `False`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArmPair<T> {
    pub true_arm: T,
    pub false_arm: T,
}

impl<T> ArmPair<T> {
    /// Construct a pair from true-arm and false-arm values.
    #[inline]
    pub const fn new(true_arm: T, false_arm: T) -> Self {
        Self {
            true_arm,
            false_arm,
        }
    }

    /// Access the value for `arm`.
    #[inline]
    pub const fn get(&self, arm: IfArm) -> &T {
        match arm {
            IfArm::True => &self.true_arm,
            IfArm::False => &self.false_arm,
        }
    }

    /// Mutably access the value for `arm`.
    #[inline]
    pub fn get_mut(&mut self, arm: IfArm) -> &mut T {
        match arm {
            IfArm::True => &mut self.true_arm,
            IfArm::False => &mut self.false_arm,
        }
    }

    /// Iterate over references to both arm values.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &T> {
        [&self.true_arm, &self.false_arm].into_iter()
    }
}

impl<T> core::ops::Index<IfArm> for ArmPair<T> {
    type Output = T;
    #[inline]
    fn index(&self, arm: IfArm) -> &Self::Output {
        self.get(arm)
    }
}

impl<T> core::ops::IndexMut<IfArm> for ArmPair<T> {
    #[inline]
    fn index_mut(&mut self, arm: IfArm) -> &mut Self::Output {
        self.get_mut(arm)
    }
}

/// Describes an If node's short-circuit structure in the schedule.
///
/// For `If(mask, if_true, if_false)`, identifies contiguous ranges of
/// schedule entries that are exclusive to each arm (not shared with mask
/// or the other arm). These ranges can be guarded by conditional branches.
#[derive(Debug, Clone)]
pub(crate) struct IfGuard {
    /// Schedule index of the If node itself.
    pub(crate) if_idx: usize,
    /// ValueId of the mask operand (already computed before arms).
    pub(crate) mask_vid: ValueId,
    /// Range of schedule indices exclusive to each arm: `[start, end)`.
    /// Empty if `start == end`.
    pub(crate) ranges: ArmPair<(usize, usize)>,
}

impl IfGuard {
    /// Schedule index range exclusive to the given arm: `[start, end)`.
    #[must_use]
    #[inline]
    pub(crate) const fn range(&self, arm: IfArm) -> (usize, usize) {
        match arm {
            IfArm::True => self.ranges.true_arm,
            IfArm::False => self.ranges.false_arm,
        }
    }

    /// Schedule index range exclusive to the true arm: `[start, end)`.
    #[must_use]
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn true_range(&self) -> (usize, usize) {
        self.ranges.true_arm
    }

    /// Schedule index range exclusive to the false arm: `[start, end)`.
    #[must_use]
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn false_range(&self) -> (usize, usize) {
        self.ranges.false_arm
    }

    /// Whether this arm is guarded (has a non-empty range).
    #[must_use]
    #[inline]
    pub(crate) fn is_guarded(&self, arm: IfArm) -> bool {
        let (s, e) = self.range(arm);
        s != e
    }

    /// Whether either arm is guarded.
    #[must_use]
    #[inline]
    pub(crate) fn has_guarded_arm(&self) -> bool {
        IfArm::ALL.iter().any(|&arm| self.is_guarded(arm))
    }

    /// Total entries skipped across both arms.
    #[must_use]
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn total_guarded_entries(&self) -> usize {
        (self.ranges.true_arm.1 - self.ranges.true_arm.0)
            + (self.ranges.false_arm.1 - self.ranges.false_arm.0)
    }
}

/// The values an operation reads *as registers*, in operand order.
///
/// A `Reduce` is a leaf here, the same as `Uniform` — by the time one reaches
/// a schedule this function walks, `extract_folds` has already carved its
/// body out into its own `ScopeFold`; the `ValueId` `ScheduledOp::Reduce`
/// still carries is `schedule_variance`'s and `extract_folds`'s own concern
/// (they run before extraction, and after respectively, over different
/// schedules), never an operand this scope's allocation resolves. What the
/// loop it opens reads from this scope is a dependency all the same, and the
/// guard analysis has it as one (`guards::FoldReads`). A `Seq`
/// sequences two effects and reads no register; a `Write` reads the one
/// value it stores — its row and column are binders, found where their
/// folds keep them, not operands.
pub(crate) fn operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let (a, b, c) = match sop {
        ScheduledOp::Var(_)
        | ScheduledOp::Lanes(_)
        | ScheduledOp::Const(_)
        | ScheduledOp::Context(_)
        | ScheduledOp::Uniform(..)
        | ScheduledOp::Reduce(..)
        | ScheduledOp::Seq(..) => (None, None, None),
        // A gather's base is a pointer, read through `pointer_operand`; the
        // index is its one vector operand.
        ScheduledOp::Unary(_, a)
        | ScheduledOp::ShiftImm(_, a, _)
        | ScheduledOp::Gather(a, _)
        | ScheduledOp::Broadcast(a, _) => (Some(*a), None, None),
        ScheduledOp::Write { value, .. } => (Some(*value), None, None),
        // A `Guard`'s mask is the one register operand its own def reads —
        // its two arms are names into a wholly separate arena, not values in
        // this schedule, exactly as a `Reduce`'s body is not (see the doc
        // above) but without even that much: an arm's operands are its own
        // scope's concern (`LinearScan::allocate_nest`'s guard-arm loop),
        // never this scope's.
        ScheduledOp::Guard(mask, _, _) => (Some(*mask), None, None),
        ScheduledOp::Binary(_, a, b) => (Some(*a), Some(*b), None),
        ScheduledOp::Ternary(_, a, b, c) => (Some(*a), Some(*b), Some(*c)),
    };
    [a, b, c].into_iter().flatten()
}

/// The address an operation reads, if it reads one: a gather's, a
/// broadcast's or a uniform load's base, always a [`Class::Pointer`] value.
/// One at most, which is what lets `Scratch::ptr_reload` be a single
/// register.
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
/// plus the body a `Reduce` folds and the two effects a `Seq` orders — the
/// children a walk of the DAG's structure follows, as opposed to the
/// registers an instruction reads ([`operands`]).
pub(crate) fn structural_children(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let extra = match sop {
        ScheduledOp::Reduce(_, body) => [Some(*body), None],
        ScheduledOp::Seq(a, b) => [Some(*a), Some(*b)],
        _ => [None, None],
    };
    all_operands(sop).chain(extra.into_iter().flatten())
}
