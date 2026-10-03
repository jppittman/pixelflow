//! Which schedule entries an `If`'s short-circuit branch may skip, and
//! the ordering that lets one branch span them.
//!
//! A property of the **schedule**, not of allocation and not of emission: it
//! asks only which values each arm of an `If` computes for itself, and the
//! answer is the same whatever registers those values end up in. Both sides
//! read it — the emitter to place the branches, the allocator to keep a split
//! live range from naming a register a skipped arm never loaded — so it lives
//! beside them rather than inside either.
//!
//! Exclusivity is necessary and not sufficient: a branch skips a *range*, so
//! an arm is only guardable when the values it owns are one contiguous run.
//! An arm can own two hundred values and be refused because forty entries
//! belonging to some other expression happen to sit between its first and its
//! last. [`cluster_if_arms`] is the answer to exactly that case, and only
//! that case — it stable-partitions the region between the mask and the
//! `If` into shared, then true-exclusive, then false-exclusive values. That
//! is always a legal topological order, because a shared value can never
//! depend on an arm-exclusive one (if it did, the exclusivity filter would
//! have rejected the value: it has a consumer outside the arm).
//!
//! It runs only where it buys a branch. Moving every shared value ahead of
//! both arms stretches live ranges across the skipped arm, and that pressure
//! should be paid where a guard is bought and nowhere else.
//!
//! And a guard is only bought where it can pay: see
//! [`MISPREDICT_PENALTY_CYCLES`]. Whether a mask is *coherent* — uniform
//! across a batch often enough for the branch to fire and to predict — is a
//! property of the data, which no static analysis can know. Its worst case,
//! though, is known exactly, and bounding the downside by the upside is
//! enough to keep the analysis honest without a tuned number anywhere.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use pixelflow_ir::kind::OpKind;
use pixelflow_search::egraph::CostModel;

use super::ScheduledOp;
use super::regalloc::{Def, ValueId};
pub(crate) use crate::program::IfGuard;
#[cfg(any(debug_assertions, feature = "layout-shadow"))]
use crate::program::layout::Layout;
#[cfg(any(debug_assertions, feature = "layout-shadow"))]
use crate::program::ownership::Ownership;
pub use crate::program::{ArmPair, IfArm};

/// A dense bitset over `0..capacity`.
///
/// Every set this module builds is over one of two spaces this file already
/// treats as dense and sequential — `ValueId.0` (`schedule_ops`,
/// `vid_to_sched_idx`, `consumers` are all dense `Vec`s indexed by it) or a
/// schedule position — so `contains` is one shift and one mask instead of a
/// tree descent, `difference` is one word-wise pass instead of a walk with a
/// rebuild, and there is nothing here for an allocator to do per element:
/// `transitive_deps` alone runs 1,445 times over one glyph's worst compile,
/// and a `BTreeSet` node is a heap allocation per member.
#[derive(Clone)]
struct IndexSet {
    bits: alloc::vec::Vec<u64>,
}

impl IndexSet {
    const BITS: usize = u64::BITS as usize;

    fn empty(capacity: usize) -> Self {
        Self {
            bits: alloc::vec![0u64; capacity.div_ceil(Self::BITS)],
        }
    }

    #[inline]
    fn contains(&self, i: usize) -> bool {
        self.bits
            .get(i / Self::BITS)
            .is_some_and(|word| word & (1u64 << (i % Self::BITS)) != 0)
    }

    #[inline]
    fn insert(&mut self, i: usize) {
        self.bits[i / Self::BITS] |= 1u64 << (i % Self::BITS);
    }

    #[inline]
    fn remove(&mut self, i: usize) {
        if let Some(word) = self.bits.get_mut(i / Self::BITS) {
            *word &= !(1u64 << (i % Self::BITS));
        }
    }

    /// Every member this set has in common with `other`, `self` loses.
    ///
    /// Word-wise: an `&!` per word this set and `other` both cover, not a
    /// walk of `other` with a lookup into `self` per member.
    fn difference_with(&mut self, other: &Self) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= !b;
        }
    }

    /// The smallest member, if any.
    fn min(&self) -> Option<usize> {
        self.bits
            .iter()
            .enumerate()
            .find(|&(_, &w)| w != 0)
            .map(|(wi, &w)| wi * Self::BITS + w.trailing_zeros() as usize)
    }

    /// The largest member, if any.
    fn max(&self) -> Option<usize> {
        self.bits
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &w)| w != 0)
            .map(|(wi, &w)| wi * Self::BITS + (Self::BITS - 1 - w.leading_zeros() as usize))
    }

    /// Every member, ascending — the bit pattern read back out.
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(wi, &w)| {
            (0..Self::BITS as u32)
                .filter(move |b| w & (1u64 << b) != 0)
                .map(move |b| wi * Self::BITS + b as usize)
        })
    }
}

/// What each fold a scope opens reads from that scope, keyed by the fold's
/// `Reduce` def.
///
/// To [`regalloc::operands`](super::regalloc::operands) a `Reduce` def is a
/// leaf, and for the registers the def's own instruction reads that is right:
/// once `extract_folds` has carved a fold's body into a scope of its own, the
/// def is only where the loop opens. But the loop *runs* there, and its body
/// reads this scope — a sibling fold's result from its accumulator slot, a
/// value invariant in the fold from the park this scope leaves it in. Nothing
/// in this scope's schedule recorded those reads, so to everything here that
/// asks what a value is needed by, or needs, the fold read nothing: a guard
/// could skip a fold whose result a sibling fold's body still reads, and
/// clustering could sink a fold's input to after the fold
/// (`pixelflow-core/tests/guard_sibling_fold.rs` pins both). These are the
/// missing edges, and with them the def consumes what its fold reads the way
/// any def consumes its operands.
///
/// They take no walk of a body. A fold's schedule is carved out of its
/// parent's and keeps the parent's `ValueId`s, so what a fold reads from its
/// parent is among the ids the two schedules share; and a fold nested in the
/// fold is carved out of *that* schedule in turn, so whatever it reads from
/// here is shared with the outer one too — the edges are transitive by
/// construction. Every shared id is an edge, a leaf the body rebuilds for
/// itself included: that over-approximates (such a leaf is kept out of an arm
/// the fold does not run in), and asks nothing of which values `place_roots`
/// later parks and which it leaves to be rebuilt.
///
/// **And what each loop costs.** To the latency table a `Reduce` costs 0 — a
/// fold's price depends on its trip count and its body, and an `OpKind`
/// carries neither — so the def, which is all this scope's schedule holds of
/// the loop, priced an arm owning a 64-trip fold at nothing, and the arm was
/// refused a branch as too cheap to pay for one. The price is the extractor's
/// own formula, [`CostModel::fold_cost`], with the body priced over the ids
/// the two schedules do *not* share — what the fold computes on each trip —
/// each def as an arm's entries are ([`def_cycles`]), and a fold nested in it
/// by that fold's own `FoldReads`. That is why each construction site builds
/// the innermost scope's first.
#[derive(Default)]
pub(crate) struct FoldReads(BTreeMap<ValueId, OpenedFold>);

/// One loop a scope opens, as that scope sees it.
struct OpenedFold {
    /// This scope's values the loop reads ([`FoldReads`]).
    reads: Vec<ValueId>,
    /// One run of the whole loop, in latency-prior cycles.
    cycles: usize,
}

impl FoldReads {
    /// The folds opened in `scope`, each given as its `Reduce` def's
    /// `ValueId`, the fold's own schedule, and the `FoldReads` of that
    /// schedule — the folds opened inside it, which the fold's price is made
    /// of too.
    ///
    /// # Panics
    ///
    /// If a fold's `ValueId` is not a `Reduce` def of `scope`: a loop opens
    /// where its def is, and nowhere else.
    pub(crate) fn new<'a>(
        scope: &[Def],
        folds: impl IntoIterator<Item = (ValueId, &'a [Def], &'a FoldReads)>,
    ) -> Self {
        let cycles = CostModel::latency_prior();
        let capacity = scope
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut here = IndexSet::empty(capacity);
        let mut opens = BTreeMap::new();
        for def in scope {
            here.insert(def.value.0 as usize);
            if let ScheduledOp::Reduce(fold, _) = def.op {
                opens.insert(def.value, fold);
            }
        }
        let opened = folds
            .into_iter()
            .map(|(reduce, body, inner)| {
                let fold = *opens.get(&reduce).unwrap_or_else(|| {
                    panic!("{reduce:?} opens a fold but is not a Reduce def of its scope")
                });
                let (shared, per_trip): (Vec<&Def>, Vec<&Def>) = body
                    .iter()
                    .partition(|def| here.contains(def.value.0 as usize));
                let body_cycles = per_trip
                    .into_iter()
                    .map(|def| def_cycles(def, inner, &cycles))
                    .fold(0, usize::saturating_add);
                let opened = OpenedFold {
                    reads: shared.into_iter().map(|def| def.value).collect(),
                    cycles: cycles.fold_cost(fold, body_cycles),
                };
                (reduce, opened)
            })
            .collect();
        Self(opened)
    }

    /// Everything the def of `value` by `op` reads in this scope: its
    /// operands of both register classes — a gather's base pointer is a read
    /// as much as its index is — and, for a `Reduce` that opens a fold here,
    /// what the fold reads.
    pub(crate) fn reads<'a>(
        &'a self,
        value: ValueId,
        op: &'a ScheduledOp,
    ) -> impl Iterator<Item = ValueId> + 'a {
        let fold: &[ValueId] = match op {
            ScheduledOp::Reduce(..) => self.0.get(&value).map_or(&[], |f| f.reads.as_slice()),
            _ => &[],
        };
        super::regalloc::all_operands(op).chain(fold.iter().copied())
    }

    /// One run of the loop the `Reduce` def of `value` opens here, or
    /// nothing for a def that opens none: a fold hoisted out of this scope,
    /// read from the accumulator slot its own scope's loop left it in.
    fn cycles(&self, value: ValueId) -> usize {
        self.0.get(&value).map_or(0, |f| f.cycles)
    }
}

/// What executing `def` once costs where it is scheduled, in latency-prior
/// cycles — the table's price for its op, and for a `Reduce` the price of the
/// loop it opens here ([`FoldReads`]). The summand of an `If` arm's price
/// and of a fold body's, which are one question: what running these entries
/// costs.
pub(crate) fn def_cycles(def: &Def, folds: &FoldReads, cycles: &CostModel) -> usize {
    match &def.op {
        ScheduledOp::Var(_)
        | ScheduledOp::Lanes(_)
        | ScheduledOp::Const(_)
        | ScheduledOp::Seq(..) => 0,
        // One store, priced as the load a gather is.
        ScheduledOp::Write { .. } => cycles.cost(OpKind::RawGather),
        // One broadcast load; priced as the leaf it is in the prologue, where
        // it lands. A context pointer's one load lands there too.
        ScheduledOp::Uniform(..) | ScheduledOp::Context(_) => cycles.cost(OpKind::Uniform),
        ScheduledOp::Unary(op, _) | ScheduledOp::Binary(op, _, _) => cycles.cost(*op),
        ScheduledOp::ShiftImm(op, _, _) => cycles.cost(*op),
        ScheduledOp::Ternary(op, _, _, _) => cycles.cost(*op),
        // A broadcast is the arena's `RawGather` with a lane-uniform index,
        // and the extractor priced it as that read. It is not a uniform's
        // prologue leaf: it runs where its address varies, which in a fold's
        // body is every trip.
        ScheduledOp::Gather(_, _) | ScheduledOp::Broadcast(_, _) => cycles.cost(OpKind::RawGather),
        ScheduledOp::Reduce(..) => folds.cycles(def.value),
        // A hard branch whose arms are scopes of their own, which no walk of
        // this schedule reaches (G2,
        // docs/plans/2026-09-12-emit-should-just-emit.md): unpriced rather
        // than guessed at, so an arm holding one is priced by what else it
        // owns.
        ScheduledOp::Guard(..) => 0,
    }
}

/// What `vid` reads in this scope ([`FoldReads::reads`]), or nothing for a leaf
/// or a hole (a `ValueId` absent from `schedule_ops`).
///
/// `schedule_ops` is a dense Vec indexed by `ValueId.0`, pre-built by the
/// caller so each lookup is O(1) instead of O(n).
fn reads_of<'a>(
    vid: ValueId,
    schedule_ops: &'a [Option<ScheduledOp>],
    folds: &'a FoldReads,
) -> impl Iterator<Item = ValueId> + 'a {
    schedule_ops
        .get(vid.0 as usize)
        .and_then(Option::as_ref)
        .into_iter()
        .flat_map(move |op| folds.reads(vid, op))
}

/// Compute the transitive dependencies of a ValueId in the schedule.
fn transitive_deps(
    vid: ValueId,
    schedule_ops: &[Option<ScheduledOp>],
    folds: &FoldReads,
) -> IndexSet {
    let mut deps = IndexSet::empty(schedule_ops.len());
    let mut worklist = alloc::vec![vid];
    while let Some(v) = worklist.pop() {
        let idx = v.0 as usize;
        if deps.contains(idx) {
            continue;
        }
        deps.insert(idx);
        worklist.extend(reads_of(v, schedule_ops, folds));
    }
    deps
}

/// One `If`'s arms as schedule positions: the entries each arm computes
/// for itself and nothing else.
///
/// Exclusivity only — whether a branch can actually span an arm is
/// [`IfArms::range`], which is where the *order* gets its say.
struct IfArms {
    if_idx: usize,
    #[cfg(test)]
    if_vid: ValueId,
    mask_vid: ValueId,
    /// Where the mask lands, or `usize::MAX` when it is not in this scope's
    /// schedule (a live-in from an enclosing one).
    mask_idx: usize,
    indices: ArmPair<IndexSet>,
    /// Everything the `If` reads, transitively, as schedule positions —
    /// which is also, by complement, everything between the mask and the
    /// `If` that the `If` does NOT need.
    #[cfg(test)]
    cone: IndexSet,
    /// What each arm's own entries cost, in latency-prior cycles — what a
    /// guard on that arm could save, against what the branch costs when it
    /// does not.
    cycles: ArmPair<usize>,
}

impl IfArms {
    /// The half-open range a branch may skip for `arm`, or an empty range
    /// at the `If` when it may not.
    ///
    /// The branch skips the WHOLE range when the mask is uniform, so every
    /// index in it must belong to this arm; and the uniformity test reads the
    /// mask's register at the range's start, so the mask must be computed by
    /// then. (Schedules from the macro pipeline emit the mask before both
    /// arms, but arena-composed kernels may schedule an arm BEFORE it —
    /// guarding that would branch on an uninitialized register. The `If`
    /// still evaluates correctly through the unconditional blend.)
    fn range(&self, arm: IfArm) -> (usize, usize) {
        let indices = &self.indices[arm];
        let cycles = self.cycles[arm];
        if cycles <= MISPREDICT_PENALTY_CYCLES {
            return (self.if_idx, self.if_idx);
        }
        let (Some(start), Some(last)) = (indices.min(), indices.max()) else {
            return (self.if_idx, self.if_idx);
        };
        let end = last + 1;
        let one_run = (start..end).all(|idx| indices.contains(idx));
        if one_run && self.mask_idx < start {
            (start, end)
        } else {
            (self.if_idx, self.if_idx)
        }
    }

    fn true_range(&self) -> (usize, usize) {
        self.range(IfArm::True)
    }

    fn false_range(&self) -> (usize, usize) {
        self.range(IfArm::False)
    }

    fn ranges(&self) -> ArmPair<(usize, usize)> {
        ArmPair::new(self.true_range(), self.false_range())
    }

    #[cfg(any(test, debug_assertions, feature = "layout-shadow"))]
    /// An arm the ORDER refuses: it is worth guarding and no branch can span
    /// it. Distinct from an arm that owns nothing, and from one too cheap to
    /// guard — no reordering helps either of those.
    fn refused_for_order(&self) -> bool {
        let refused = |cycles: usize, range: (usize, usize)| {
            cycles > MISPREDICT_PENALTY_CYCLES && range.0 == range.1
        };
        IfArm::ALL
            .iter()
            .any(|&arm| refused(self.cycles[arm], self.range(arm)))
    }
}

/// Analyze the schedule for If nodes and compute short-circuit guard ranges.
///
/// For each If, partitions schedule entries into:
/// - Shared: needed by mask, or by both arms (must always execute)
/// - True-exclusive: only needed by the true arm (skip if mask all-false)
/// - False-exclusive: only needed by the false arm (skip if mask all-true)
///
/// `external` are values read *outside* this schedule — a scope's roots,
/// read by the loops inside it — so no arm may own one: a guard skipping the
/// arm would leave the value unwritten for a loop that runs regardless.
/// `folds` is what each loop this scope opens reads from it, which makes the
/// loop's `Reduce` def a consumer of each of those values ([`FoldReads`]).
///
/// Returns guards sorted by if_idx (ascending).
pub(crate) fn analyze_if_guards(
    schedule: &[Def],
    external: &[ValueId],
    folds: &FoldReads,
) -> Vec<IfGuard> {
    let per_if = if_arms(schedule, external, folds);
    #[cfg(any(debug_assertions, feature = "layout-shadow"))]
    {
        assert_ownership_agrees(schedule, external, folds, &per_if);
        assert_layout_agrees(schedule, external, folds, &per_if);
    }
    guards_from(&per_if)
}

/// The guards `per_if`'s arms earn: one per `If` with at least one arm a
/// branch can span, sorted by `if_idx` (ascending).
fn guards_from(per_if: &[IfArms]) -> Vec<IfGuard> {
    let mut guards = Vec::new();

    for arms in per_if {
        let ranges = arms.ranges();

        // Only create a guard if at least one arm has exclusive nodes
        if ranges.true_arm.0 != ranges.true_arm.1 || ranges.false_arm.0 != ranges.false_arm.1 {
            guards.push(IfGuard {
                if_idx: arms.if_idx,
                mask_vid: arms.mask_vid,
                ranges,
            });
        }
    }

    guards
}

/// The old analysis and [`Ownership`] answer one question two ways: per `If`
/// and arm, the same entries and the same price.
///
/// Run on every call in a debug build and, in release, under the
/// `layout-shadow` feature: the equality is what lets the ownership pass
/// replace this analysis. An inequality is a stop — ownership is then not the
/// exclusivity relation for that input, and the pass is what gets fixed.
#[cfg(any(debug_assertions, feature = "layout-shadow"))]
fn assert_ownership_agrees(
    schedule: &[Def],
    external: &[ValueId],
    folds: &FoldReads,
    per_if: &[IfArms],
) {
    // Both stages read a schedule in which every value precedes its readers,
    // which the allocator has always required; a hand-built fixture that
    // breaks it has no ownership to compare. No compile reaches this: the
    // 95-kernel glyph table on both tiers, the units font and chrome ran with
    // the skip counted, and it never fired.
    if !crate::program::layout::reads_follow(0..schedule.len(), schedule, folds) {
        return;
    }
    let own = Ownership::of(schedule, external, folds);
    assert_eq!(
        own.arms().len(),
        2 * per_if.len(),
        "ownership found a different number of `If`s"
    );
    let (pairs, _) = own.arms().as_chunks::<2>();
    for (old, pair) in per_if.iter().zip(pairs) {
        for new in pair {
            assert_eq!(new.if_pos, old.if_idx, "ownership met a different `If`");
            let owned: Vec<usize> = (0..schedule.len())
                .filter(|&pos| own.is_within(own.region_of(pos), new.region))
                .collect();
            let exclusive: Vec<usize> = old.indices[new.arm].iter().collect();
            assert_eq!(
                owned, exclusive,
                "{:?} arm of the If at {}: ownership and exclusivity disagree on what it owns",
                new.arm, old.if_idx
            );
            assert_eq!(
                new.cycles, old.cycles[new.arm],
                "{:?} arm of the If at {}: ownership and exclusivity price it differently",
                new.arm, old.if_idx
            );
        }
    }
}

/// The layout is valid exactly when the old analysis, run on the order the
/// layout chose, finds the runs the layout says it made.
///
/// So the old analysis is the oracle for the new order, on every call: it must
/// find the same guards at the same positions over the same ranges; every
/// arm it guarded on the order it was given the layout must guard too; and a
/// scope in which it refused nothing for its order must come back unmoved. An
/// inequality is a stop — the layout is what gets fixed, never the bound.
#[cfg(any(debug_assertions, feature = "layout-shadow"))]
fn assert_layout_agrees(
    schedule: &[Def],
    external: &[ValueId],
    folds: &FoldReads,
    per_if: &[IfArms],
) {
    // As `assert_ownership_agrees`: never fires on a compile.
    if !crate::program::layout::reads_follow(0..schedule.len(), schedule, folds) {
        return;
    }
    let layout = Layout::of(schedule, external, folds);
    assert!(
        layout.is_topological(schedule, folds),
        "the layout put a value ahead of an operand"
    );
    let permuted: Vec<Def> = layout
        .order
        .iter()
        .map(|&old| schedule[old].clone())
        .collect();
    for (old, &new) in layout.position.iter().enumerate() {
        assert_eq!(
            permuted[new].value, schedule[old].value,
            "the layout's two maps are not inverses"
        );
    }
    let found = guards_from(&if_arms(&permuted, external, folds));
    assert_eq!(
        found.len(),
        layout.guards.len(),
        "the old analysis finds a different number of branches in the laid-out order"
    );
    for (found, laid) in found.iter().zip(&layout.guards) {
        assert_eq!(
            (found.if_idx, found.mask_vid, found.ranges),
            (laid.if_idx, laid.mask_vid, laid.ranges),
            "the laid-out order is not the runs the layout says"
        );
    }
    for old in guards_from(per_if) {
        let value = schedule[old.if_idx].value;
        let kept = layout
            .guards
            .iter()
            .find(|laid| permuted[laid.if_idx].value == value)
            .unwrap_or_else(|| panic!("the layout lost the branch of {value:?}"));
        for arm in IfArm::ALL {
            assert!(
                !old.is_guarded(arm) || kept.is_guarded(arm),
                "the layout lost the {arm:?} arm's branch of {value:?}"
            );
        }
    }
    if per_if.iter().all(|arms| !arms.refused_for_order()) {
        assert!(
            layout.is_identity(),
            "nothing was refused for its order, and the layout moved something"
        );
    }
}

/// Every `If` in the schedule, with the entries exclusive to each arm.
///
/// `external` values have a consumer outside the schedule (see
/// [`analyze_if_guards`]), which no arm's closure can contain; a value a
/// fold reads has that fold's `Reduce` def as a consumer ([`FoldReads`]).
fn if_arms(schedule: &[Def], external: &[ValueId], folds: &FoldReads) -> Vec<IfArms> {
    let mut arms = Vec::new();

    if schedule.is_empty() {
        return arms;
    }

    // The extraction cost model's table, which is the workspace's one answer
    // to "what does this op cost" — the guard's bound is denominated in the
    // same cycles the optimizer chose the expression with.
    let cycles = CostModel::latency_prior();

    // Build dense lookup: schedule_ops[vid.0] = Some(&ScheduledOp) for O(1) child traversal.
    // ValueIds are sequential starting from 0 (guaranteed by arena_to_schedule).
    let max_vid = schedule.iter().map(|def| def.value.0).max().unwrap_or(0) as usize;
    let mut schedule_ops: alloc::vec::Vec<Option<ScheduledOp>> = alloc::vec![None; max_vid + 1];
    for def in schedule {
        schedule_ops[def.value.0 as usize] = Some(def.op.clone());
    }

    // Build dense lookup: vid_to_sched_idx[vid.0] = schedule position (u32::MAX = absent).
    let mut vid_to_sched_idx: alloc::vec::Vec<usize> = alloc::vec![usize::MAX; max_vid + 1];
    for (i, def) in schedule.iter().enumerate() {
        vid_to_sched_idx[def.value.0 as usize] = i;
    }

    // Global consumer map: consumers[v.0] = every value that reads v as an
    // operand. A node may only be guarded (skipped when its arm's mask is
    // uniform) if EVERY consumer is inside that arm's subtree (or the `If`
    // itself) — otherwise an outer/sibling expression reads a register the
    // branch never computed. Subtree-local exclusivity (below) is necessary but
    // NOT sufficient; this is the global check that was missing.
    let mut consumers: alloc::vec::Vec<alloc::vec::Vec<ValueId>> =
        alloc::vec![alloc::vec::Vec::new(); max_vid + 1];
    for def in schedule {
        let vid = def.value;
        for child in folds.reads(vid, &def.op) {
            if (child.0 as usize) <= max_vid {
                consumers[child.0 as usize].push(vid);
            }
        }
    }
    // A reader outside the schedule is a consumer no arm can contain: a name
    // no def here has, so it is never "in the set" and never the `If`.
    const OUTSIDE: ValueId = ValueId(u32::MAX);
    for root in external {
        if (root.0 as usize) <= max_vid {
            consumers[root.0 as usize].push(OUTSIDE);
        }
    }

    for (i, def) in schedule.iter().enumerate() {
        let (sel_vid, sop) = (&def.value, &def.op);
        if let ScheduledOp::Ternary(OpKind::If, mask_vid, true_vid, false_vid) = sop {
            // (the exclusivity analysis, unchanged)
            // Compute transitive deps for each subtree using the dense O(1) lookup
            let mask_deps = transitive_deps(*mask_vid, &schedule_ops, folds);
            let true_deps = transitive_deps(*true_vid, &schedule_ops, folds);
            let false_deps = transitive_deps(*false_vid, &schedule_ops, folds);

            // A node is safe to skip under this arm only if every one of its
            // consumers is ALSO skipped under it — or is the `If` itself.
            // Reaching the arm is not enough: a value can be inside the arm's
            // cone and still be shared with the world outside it, and a
            // dependency of such a value would then be skipped while its
            // consumer runs, reading a register the branch never wrote.
            //
            // So exclusivity is a closure, not a filter. Seed it with the
            // values only this arm's cone reaches, then drop any whose
            // consumers are not themselves in the set — the greatest set
            // closed under "my consumers are skipped with me".
            //
            // A worklist rather than a rescan-to-fixpoint: `v` can only
            // become newly doomed when a consumer of it just left the set
            // (removing `u` never *adds* an in-set consumer to anything, so
            // doomed-ness only ever gains evidence), and the only values a
            // removal can affect that way are `u`'s own operands — `u` was
            // one of *their* consumers. So the initial set is the seed
            // (nothing has been triggered yet, but a consumer outside the
            // set from the start still dooms its producer), and every
            // removal pushes that value's operands back on to be
            // re-examined, rather than re-walking everyone.
            let closed_exclusive = |cone: &IndexSet, other: &IndexSet| {
                let mut set = cone.clone();
                set.difference_with(&mask_deps);
                set.difference_with(other);
                let mut worklist: alloc::vec::Vec<ValueId> =
                    set.iter().map(|i| ValueId(i as u32)).collect();
                while let Some(v) = worklist.pop() {
                    let idx = v.0 as usize;
                    if !set.contains(idx) {
                        continue; // already removed by an earlier pop
                    }
                    let doomed = consumers[idx]
                        .iter()
                        .any(|c| *c != *sel_vid && !set.contains(c.0 as usize));
                    if !doomed {
                        continue;
                    }
                    set.remove(idx);
                    for operand in reads_of(v, &schedule_ops, folds) {
                        if set.contains(operand.0 as usize) {
                            worklist.push(operand);
                        }
                    }
                }
                set
            };

            let true_exclusive = closed_exclusive(&true_deps, &false_deps);
            let false_exclusive = closed_exclusive(&false_deps, &true_deps);

            // Map to schedule indices using dense O(1) lookup
            let to_schedule_indices = |exclusive: &IndexSet| -> IndexSet {
                let mut indices = IndexSet::empty(schedule.len());
                for vid in exclusive.iter() {
                    if let Some(&idx) = vid_to_sched_idx.get(vid)
                        && idx != usize::MAX
                    {
                        indices.insert(idx);
                    }
                }
                indices
            };
            let true_indices = to_schedule_indices(&true_exclusive);
            let false_indices = to_schedule_indices(&false_exclusive);

            let mask_idx = vid_to_sched_idx
                .get(mask_vid.0 as usize)
                .copied()
                .unwrap_or(usize::MAX);

            #[cfg(test)]
            let cone = {
                let mut cone = IndexSet::empty(schedule.len());
                for vid in mask_deps
                    .iter()
                    .chain(true_deps.iter())
                    .chain(false_deps.iter())
                {
                    if let Some(&idx) = vid_to_sched_idx.get(vid)
                        && idx != usize::MAX
                    {
                        cone.insert(idx);
                    }
                }
                cone
            };

            let arm_cycles = |indices: &IndexSet| -> usize {
                indices
                    .iter()
                    .map(|idx| def_cycles(&schedule[idx], folds, &cycles))
                    .fold(0, usize::saturating_add)
            };
            let (true_cycles, false_cycles) =
                (arm_cycles(&true_indices), arm_cycles(&false_indices));

            arms.push(IfArms {
                if_idx: i,
                #[cfg(test)]
                if_vid: *sel_vid,
                mask_vid: *mask_vid,
                mask_idx,
                indices: ArmPair::new(true_indices, false_indices),
                #[cfg(test)]
                cone,
                cycles: ArmPair::new(true_cycles, false_cycles),
            });
        }
    }

    arms
}

/// What a guard costs when it never fires: the uniformity test, plus a
/// branch the hardware cannot predict because the mask is incoherent.
///
/// Taken as ~16 cycles, which is the mispredict penalty on the cores this
/// compiler targets — 15–20 on Intel since Skylake and on AMD since Zen
/// (Agner Fog, *The microarchitecture of Intel, AMD and VIA CPUs*, §"Branch
/// prediction"), 13–16 on ARM's recent out-of-order cores (Cortex-A76 and
/// Neoverse software optimization guides). It is an architectural figure, not
/// a knob: **do not sweep it**, and do not move it to make a kernel faster.
///
/// It is used as a *bound*, which is why one number for two architectures is
/// honest. A guard's upside depends on how often the mask is uniform, which
/// is data and unknowable here; its downside does not. An arm whose work
/// costs less than the penalty cannot pay for its own branch even if the
/// branch always fires, so guarding it is a loss in every world — while an
/// arm that costs far more is capped at this much loss and may save all of
/// it. Measured, that is the whole difference between a glyph's coverage
/// mask (a handful of ops per arm, varying per lane, 3.6x slower with a
/// guard) and a sphere's silhouette (214 entries, uniformly false in 97% of
/// batches, 3.2x faster with one).
pub(crate) const MISPREDICT_PENALTY_CYCLES: usize = 16;

/// Reorder a scope's schedule so that an `If`'s arm-exclusive entries form
/// one run — where that, and only that, is what stands between the arm and a
/// branch.
///
/// The transformation per `If` is a stable partition of the entries between
/// the mask and the `If` into shared, then true-exclusive, then
/// false-exclusive. It is always a legal topological order:
///
/// - No shared entry depends on an arm-exclusive one. If it did, that value
///   would have a consumer outside the arm and `only_used_within` would not
///   have called it exclusive.
/// - No true-exclusive entry depends on a false-exclusive one, or the reverse,
///   for the same reason.
/// - Relative order is preserved inside each group, and every group's
///   dependencies now precede it.
///
/// Outermost first — an `If` is scheduled after everything in its arms, so an
/// enclosing `If` comes later — and each `If` is partitioned **once**,
/// unconditionally.
///
/// # This used to be a search, and that was the wrong shape
///
/// It ran up to eight rounds of hill-climbing: recompute every arm's closure,
/// try a partition, recompute every closure *again* to score it, keep it only
/// if strictly more entries ended up under a guard, restart. Measured on a
/// glyph it was **73% of an entire bake** — 1,540 ms of 2,127 on `8`@32 — and
/// what it found was a constant 282 bytes of emitted code, the same 282 for
/// `A`, `O` and `8`. The search scaled; what it found did not. See
/// `docs/BACKLOG.md`, X1.
///
/// The accept/reject test existed to protect register pressure: partitioning
/// moves shared values ahead of both arms, so they live across the skipped
/// arm, and that was judged "a cost worth paying for a branch and not
/// otherwise". That reasoning takes `If` to be a blend with the branch as
/// an upsell to be justified. It is not — a uniform mask *takes an arm*, and
/// the blend is the fallback for a mask that varies by lane (CLAUDE.md,
/// "`If` contains an if"). The branch is not optional, so neither is the
/// live-range cost of admitting it, and the test has nothing left to decide.
///
/// What still decides something is [`MISPREDICT_PENALTY_CYCLES`] — one
/// comparison per `If`, not a search: an arm too cheap to pay for its own
/// branch is never partitioned, because [`IfArms::refused_for_order`] is
/// false for it. That bound is measured (a glyph's coverage mask is 3.6x
/// *slower* guarded), so "always admit the jump" is a statement about what
/// `If` means, not a licence to branch on a two-instruction arm.
///
/// `folds` is what each loop this scope opens reads from it ([`FoldReads`]):
/// a permutation is only legal if it keeps those reads ahead of the loop, and
/// a loop's `Reduce` def names none of them as an operand.
#[cfg(test)]
pub(crate) fn cluster_if_arms(schedule: Vec<Def>, folds: &FoldReads) -> Vec<Def> {
    let mut current = schedule;
    // Keyed by the `If`'s *value*: the one identity that survives a
    // reordering, where a schedule position does not. Each `If` is
    // partitioned at most once, so this terminates in at most one pass per
    // `If` and there is no round cap to choose.
    let mut partitioned: alloc::collections::BTreeSet<ValueId> =
        alloc::collections::BTreeSet::new();

    loop {
        // Recomputed each time because `partition_around` returns a new
        // schedule and `IfArms` holds indices into the old one. Only the
        // region it rewrites moves, and relative order is preserved within
        // each group, so an `If` already contiguous inside that region stays
        // contiguous.
        // Clustering knows which values each loop reads, but not which of
        // them `place_roots` will park: it runs before the roots are placed.
        // That only costs a guard — the analysis that ranges an arm is told
        // about the roots, and excludes them — never correctness: a parked
        // value is one a loop reads, and `folds` keeps it ahead of the loop.
        let arms = if_arms(&current, &[], folds);
        let Some(candidate) = arms
            .iter()
            .rev()
            .find(|c| c.refused_for_order() && !partitioned.contains(&c.if_vid))
        else {
            break;
        };
        partitioned.insert(candidate.if_vid);
        current = partition_around(&current, candidate, folds);
    }

    current
}

/// The schedule with the region of the `If` that `arms` describes
/// stable-partitioned into shared, then
/// true-exclusive, then false-exclusive entries.
#[cfg(test)]
fn partition_around(schedule: &[Def], arms: &IfArms, folds: &FoldReads) -> Vec<Def> {
    let first_arm = arms.indices[IfArm::True]
        .iter()
        .chain(arms.indices[IfArm::False].iter())
        .min();
    let Some(first_arm) = first_arm else {
        return schedule.to_vec();
    };
    // The mask is shared, so it lands in the first group wherever it started;
    // the region begins at whichever of the two comes first.
    let start = first_arm.min(arms.mask_idx);
    let region = start..arms.if_idx;

    // A scope's result is its last entry, so nothing may be placed after it:
    // when the `If` IS the root, the strangers stay ahead of the arms.
    let sink_past_if = arms.if_idx + 1 < schedule.len();
    let in_any_arm = |i: &usize| {
        arms.indices[IfArm::True].contains(*i) || arms.indices[IfArm::False].contains(*i)
    };
    let stays_before = |i: &usize| (arms.cone.contains(*i) || !sink_past_if) && !in_any_arm(i);

    let mut out = Vec::with_capacity(schedule.len());
    out.extend_from_slice(&schedule[..start]);
    // What the `If` reads and neither arm owns: it must be computed before
    // the arms, because the arms read it.
    out.extend(
        region
            .clone()
            .filter(stays_before)
            .map(|i| schedule[i].clone()),
    );
    for arm in IfArm::ALL {
        out.extend(
            arms.indices[arm]
                .iter()
                .filter(|i| region.contains(i))
                .map(|i| schedule[i].clone()),
        );
    }
    out.push(schedule[arms.if_idx].clone());
    // What the `If` does NOT read sinks past it, keeping its order. Legal
    // for the same reason the partition is: nothing the `If` reads can read
    // one of these, or it would be in the cone. And it is the better place —
    // hoisting a stranger ahead of both arms would keep it live across the
    // arm a branch is there to skip, which is pressure bought for nothing.
    out.extend(
        region
            .clone()
            .filter(|i| !stays_before(i) && !in_any_arm(i))
            .map(|i| schedule[i].clone()),
    );
    out.extend_from_slice(&schedule[arms.if_idx + 1..]);

    debug_assert_eq!(
        out.len(),
        schedule.len(),
        "a partition moves entries, never adds or drops them"
    );
    debug_assert!(
        is_topological(&out, folds),
        "a partition reordered a value ahead of an operand"
    );
    out
}

/// Every operand is defined before it is read — the property a partition must
/// preserve and the one a wrong exclusivity rule silently breaks. A value a
/// loop reads is an operand of the loop's `Reduce` def here ([`FoldReads`]),
/// or a partition that sinks it past the loop goes unnoticed.
///
/// This caught a real bug the day it was written: "exclusive" used to mean
/// every consumer *reaches* the arm, which admits a value whose consumer is
/// shared with the world outside it. Moving such a value behind its consumer
/// produced a kernel that read an undefined register, and the emitted code was
/// wrong in a way no unit test of the analysis would have shown.
#[cfg(test)]
fn is_topological(schedule: &[Def], folds: &FoldReads) -> bool {
    let defined: alloc::collections::BTreeSet<ValueId> =
        schedule.iter().map(|def| def.value).collect();
    let mut seen = alloc::collections::BTreeSet::new();
    for def in schedule {
        let ready = |c: &ValueId| seen.contains(c) || !defined.contains(c);
        let ok = folds.reads(def.value, &def.op).all(|c| ready(&c));
        if !ok {
            return false;
        }
        seen.insert(def.value);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    mod index_set {
        use super::IndexSet;

        /// Round-trips across a word boundary: 64 is the first bit that must
        /// land in the second `u64`, so this alone would catch an off-by-one
        /// in the `/`/`%` split.
        #[test]
        fn insert_contains_remove_survive_a_word_boundary() {
            let mut set = IndexSet::empty(130);
            for i in [0usize, 1, 63, 64, 65, 127, 128, 129] {
                set.insert(i);
            }
            for i in [0usize, 1, 63, 64, 65, 127, 128, 129] {
                assert!(set.contains(i), "{i} was inserted");
            }
            for i in [2usize, 62, 66, 126] {
                assert!(!set.contains(i), "{i} was never inserted");
            }
            set.remove(64);
            assert!(!set.contains(64));
            assert!(set.contains(65), "removing 64 must not touch its neighbor");
        }

        #[test]
        fn min_and_max_span_words() {
            let mut set = IndexSet::empty(200);
            assert_eq!(set.min(), None, "an empty set has no minimum");
            assert_eq!(set.max(), None, "an empty set has no maximum");
            set.insert(150);
            set.insert(3);
            set.insert(70);
            assert_eq!(set.min(), Some(3));
            assert_eq!(set.max(), Some(150));
        }

        #[test]
        fn difference_with_is_word_wise_and_asymmetric() {
            let mut a = IndexSet::empty(128);
            for i in [1usize, 64, 100] {
                a.insert(i);
            }
            let mut b = IndexSet::empty(128);
            for i in [64usize, 65] {
                b.insert(i);
            }
            a.difference_with(&b);
            let left: alloc::vec::Vec<usize> = a.iter().collect();
            assert_eq!(left, alloc::vec![1, 100], "only b's members leave a");
        }

        #[test]
        fn iter_matches_insertion_ascending_regardless_of_order() {
            let mut set = IndexSet::empty(80);
            for i in [70usize, 0, 65, 3, 1] {
                set.insert(i);
            }
            let seen: alloc::vec::Vec<usize> = set.iter().collect();
            assert_eq!(seen, alloc::vec![0, 1, 3, 65, 70]);
        }
    }

    fn def(value: u32, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    // The arm op in each fixture is load-bearing, not decoration.
    // `IfArms::range` refuses any arm costing `<= MISPREDICT_PENALTY_CYCLES`
    // — guarding one cannot pay for a mispredict — so the op has to clear that
    // bar for a guard to exist at all to pin the range of. `Rsqrt` is 21 cycles
    // in `latency_prior`; a `Neg` is 3, and every assertion here would read
    // zero guards.

    /// An `If` whose true arm alone does work exclusive to it — the false
    /// arm is just the mask again, so it contributes nothing beyond
    /// `mask_deps`. Pins the exact range rather than only "a guard formed
    /// somewhere," which the whole-kernel `assert_guard_forms`-style tests
    /// in `emit/mod.rs` already cover.
    #[test]
    fn range_the_true_arm_when_only_it_is_exclusive() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(
                3,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(2), ValueId(0)),
            ),
        ];

        let guards = analyze_if_guards(&schedule, &[], &FoldReads::default());

        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].if_idx, 3);
        assert_eq!(guards[0].mask_vid, ValueId(0));
        assert_eq!(guards[0].true_range(), (1, 3));
        assert_eq!(guards[0].false_range(), (3, 3));
    }

    /// Symmetric to the above: the false arm alone is exclusive.
    #[test]
    fn range_the_false_arm_when_only_it_is_exclusive() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(
                3,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(0), ValueId(2)),
            ),
        ];

        let guards = analyze_if_guards(&schedule, &[], &FoldReads::default());

        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].if_idx, 3);
        assert_eq!(guards[0].true_range(), (3, 3));
        assert_eq!(guards[0].false_range(), (1, 3));
    }

    /// An operand reachable only through a value the schedule never defines
    /// (a "hole" — legitimate for a schedule spliced from arbitrary
    /// fragments, per this module's doc comment) must not be mistaken for a
    /// real schedule position. Regression test for treating the sentinel
    /// `usize::MAX` (marking "not in this schedule") as a valid index, which
    /// would corrupt the range or overflow computing its end.
    #[test]
    fn ignore_a_false_operand_missing_from_the_schedule() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(3))), // ValueId(3) has no Def
            def(
                4,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(0), ValueId(1)),
            ),
        ];

        let guards = analyze_if_guards(&schedule, &[], &FoldReads::default());

        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].if_idx, 2);
        assert_eq!(guards[0].false_range(), (1, 2));
    }

    /// The cost gate is `<=`, and this pins that boundary rather than a value
    /// safely past it. `Recip` is exactly `MISPREDICT_PENALTY_CYCLES` in
    /// `latency_prior`, and an arm that costs exactly the mispredict penalty
    /// is refused: the branch can save at most what it costs when it is
    /// wrong, so guarding it is never a win. Turning the gate into `<` admits
    /// this arm and this test says so; the `Rsqrt` fixtures above cannot,
    /// since 21 is on the same side of the bar either way.
    #[test]
    fn refuse_an_arm_that_costs_exactly_the_mispredict_penalty() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Recip, ValueId(1))),
            def(
                3,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(2), ValueId(0)),
            ),
        ];

        assert_eq!(
            pixelflow_search::egraph::CostModel::latency_prior().cost(OpKind::Recip),
            MISPREDICT_PENALTY_CYCLES,
            "fixture assumes Recip sits exactly on the gate",
        );

        assert!(analyze_if_guards(&schedule, &[], &FoldReads::default()).is_empty());
    }

    /// A four-trip fold; the scope its body was carved into is not what
    /// these tests look at, so the body names a hole.
    fn reduce() -> ScheduledOp {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        let fold = Fold::new(
            Monoid::SUM,
            Binder::from_slot(0).expect("slot 0 exists"),
            0..4,
        );
        ScheduledOp::Reduce(fold, ValueId(99))
    }

    /// `W` is read by the true arm and by a sibling fold's body, which runs
    /// whatever the mask: the arm may not own `W`. Its `Reduce` def names no
    /// operand, so without the sibling's reads the arm owned it, and a
    /// uniformly-false mask skipped the loop the sibling then read.
    #[test]
    fn keep_a_fold_a_sibling_fold_reads_out_of_the_arm() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, reduce()),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Binary(OpKind::Mul, ValueId(1), ValueId(2))),
            def(
                4,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(3), ValueId(0)),
            ),
            def(5, reduce()),
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(4), ValueId(5))),
        ];
        // The sibling's body holds `W`'s id, as a placeholder read from its
        // accumulator slot.
        let sibling = [def(1, reduce())];
        let folds = FoldReads::new(
            &schedule,
            [(ValueId(5), &sibling[..], &FoldReads::default())],
        );

        let blind = analyze_if_guards(&schedule, &[], &FoldReads::default());
        assert_eq!(blind[0].true_range(), (1, 4), "the arm owned `W` unseen");

        let guards = analyze_if_guards(&schedule, &[], &folds);
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].true_range(), (2, 4), "`W` is not the arm's");
    }

    /// `s` is read only by `W`'s body, and `W` is in the true arm: `s` is in
    /// the `If`'s cone, not a stranger to be sunk past it — which would put
    /// it after the loop that reads it.
    #[test]
    fn cluster_keeps_what_a_fold_reads_ahead_of_the_fold() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(1))),
            def(4, reduce()),
            def(5, ScheduledOp::Binary(OpKind::Mul, ValueId(2), ValueId(4))),
            def(
                6,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(5), ValueId(0)),
            ),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(6), ValueId(1))),
        ];
        let body = [def(3, ScheduledOp::Const(0.0))];
        let folds = FoldReads::new(&schedule, [(ValueId(4), &body[..], &FoldReads::default())]);
        let at = |order: &[Def], v: u32| {
            order
                .iter()
                .position(|d| d.value == ValueId(v))
                .expect("a permutation keeps every def")
        };

        let blind = cluster_if_arms(schedule.clone(), &FoldReads::default());
        assert!(
            at(&blind, 3) > at(&blind, 4),
            "`s` sank past its loop unseen"
        );

        let clustered = cluster_if_arms(schedule, &folds);
        assert!(at(&clustered, 3) < at(&clustered, 4));
        assert!(is_topological(&clustered, &folds));
    }

    /// A gather's base is a pointer operand, not a vector one, and it is a
    /// read all the same: the `Context` the arm's `Broadcast` addresses
    /// through is in the `If`'s cone, not a stranger to be sunk past the
    /// `If` — which would put the pointer's definition after its reader.
    /// `Y + Y` sits between the arm's entries and is read by the root, so the
    /// arm is refused for order and clustering runs.
    #[test]
    fn cluster_keeps_a_pointer_ahead_of_its_reader() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(1))),
            def(4, ScheduledOp::Context(0)),
            def(5, ScheduledOp::Broadcast(ValueId(2), ValueId(4))),
            def(
                6,
                ScheduledOp::Ternary(OpKind::If, ValueId(0), ValueId(5), ValueId(0)),
            ),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(6), ValueId(3))),
        ];
        let folds = FoldReads::default();
        let at = |order: &[Def], v: u32| {
            order
                .iter()
                .position(|d| d.value == ValueId(v))
                .expect("a permutation keeps every def")
        };

        let clustered = cluster_if_arms(schedule, &folds);
        assert!(
            at(&clustered, 4) < at(&clustered, 5),
            "the pointer sank past the broadcast that reads it: {clustered:?}"
        );
        assert!(is_topological(&clustered, &folds));
    }

    /// The arm fold's trip count below: long enough that pricing the loop
    /// as one instruction, or as nothing, is off by a factor this large.
    const ARM_TRIPS: u32 = 64;

    /// A sum of `trips` terms binding `slot`.
    fn sum_over(slot: u8, trips: u32) -> pixelflow_ir::fold::Fold {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        let binder = Binder::from_slot(slot).expect("a live binder slot");
        Fold::new(Monoid::SUM, binder, 0..trips)
    }

    /// `select(X < 20, F, 0) + Y`, with `F` the `Reduce` def `fold` opening
    /// at position 3 — the true arm's only entry of its own.
    fn if_over_a_fold(fold: pixelflow_ir::fold::Fold, body_root: u32) -> Vec<Def> {
        alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Const(20.0)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Reduce(fold, ValueId(body_root))),
            def(4, ScheduledOp::Const(0.0)),
            def(
                5,
                ScheduledOp::Ternary(OpKind::If, ValueId(2), ValueId(3), ValueId(4)),
            ),
            def(6, ScheduledOp::Var(1)),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(5), ValueId(6))),
        ]
    }

    /// `|x − j|` per trip, `j` the binder of `fold` and `x` read from the
    /// enclosing scope as `ValueId(0)`; the ids from `first` up are the
    /// body's own.
    fn distance_body(fold: pixelflow_ir::fold::Fold, first: u32) -> Vec<Def> {
        let (j, diff, abs) = (first, first + 1, first + 2);
        alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(j, ScheduledOp::Var(fold.binder().var())),
            def(
                diff,
                ScheduledOp::Binary(OpKind::Sub, ValueId(0), ValueId(j))
            ),
            def(abs, ScheduledOp::Unary(OpKind::Abs, ValueId(diff))),
        ]
    }

    /// An arm that owns a fold is priced by the loop: `n` trips of a `k`-cycle
    /// body are `n·k`, plus the `n − 1` combines — not the `Reduce` def's
    /// table price of 0, which refused the arm a branch however long the
    /// loop. With the price, the arm clears the mispredict bound and is
    /// guarded.
    #[test]
    fn price_an_arm_that_owns_a_fold_by_its_trips() {
        let cycles = CostModel::latency_prior();
        let fold = sum_over(0, ARM_TRIPS);
        let body = distance_body(fold, 10);
        let schedule = if_over_a_fold(fold, 12);
        let folds = FoldReads::new(&schedule, [(ValueId(3), &body[..], &FoldReads::default())]);

        let n = ARM_TRIPS as usize;
        let k = cycles.cost(OpKind::Sub) + cycles.cost(OpKind::Abs);
        let combine = cycles.cost(OpKind::Add);
        let arms = if_arms(&schedule, &[], &folds);
        assert_eq!(arms.len(), 1);
        assert_eq!(
            arms[0].cycles.true_arm,
            n * k + (n - 1) * combine,
            "the arm is its loop: {n} trips of a {k}-cycle body, and a combine between each"
        );
        assert_eq!(arms[0].cycles.true_arm, cycles.fold_cost(fold, k));

        let blind = if_arms(&schedule, &[], &FoldReads::default());
        assert_eq!(
            blind[0].cycles.true_arm, 0,
            "a Reduce def that opens no loop here is a slot read, and the table prices it 0"
        );

        let guards = analyze_if_guards(&schedule, &[], &folds);
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].true_range(), (3, 4), "the loop is skipped whole");
        assert!(analyze_if_guards(&schedule, &[], &FoldReads::default()).is_empty());
    }

    /// A table read whose address the lane binder does not reach is a
    /// `Broadcast`, and in a fold's body it runs every trip: `Σ_j |x − t[j]|`
    /// is priced `n` reads, as the extractor priced the arena's `RawGather`,
    /// not `n` of a uniform's prologue leaf (0). Its base pointer is the
    /// scope's root and read by the loop, so the arm is the loop alone.
    #[test]
    fn price_a_table_read_per_trip_as_the_read_it_is() {
        let cycles = CostModel::latency_prior();
        let fold = sum_over(0, ARM_TRIPS);
        let (x, ctx, j, t, diff, abs) = (0, 8, 10, 11, 12, 13);
        let schedule = alloc::vec![
            def(x, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Const(20.0)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(x), ValueId(1))),
            def(ctx, ScheduledOp::Context(0)),
            def(3, ScheduledOp::Reduce(fold, ValueId(abs))),
            def(4, ScheduledOp::Const(0.0)),
            def(
                5,
                ScheduledOp::Ternary(OpKind::If, ValueId(2), ValueId(3), ValueId(4)),
            ),
            def(6, ScheduledOp::Var(1)),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(5), ValueId(6))),
        ];
        let body = [
            def(x, ScheduledOp::Var(0)),
            def(ctx, ScheduledOp::Context(0)),
            def(j, ScheduledOp::Var(fold.binder().var())),
            def(t, ScheduledOp::Broadcast(ValueId(j), ValueId(ctx))),
            def(
                diff,
                ScheduledOp::Binary(OpKind::Sub, ValueId(x), ValueId(t)),
            ),
            def(abs, ScheduledOp::Unary(OpKind::Abs, ValueId(diff))),
        ];
        let folds = FoldReads::new(&schedule, [(ValueId(3), &body[..], &FoldReads::default())]);
        let roots = [ValueId(ctx)];

        let k =
            cycles.cost(OpKind::RawGather) + cycles.cost(OpKind::Sub) + cycles.cost(OpKind::Abs);
        let arms = if_arms(&schedule, &roots, &folds);
        assert_eq!(arms[0].cycles.true_arm, cycles.fold_cost(fold, k));

        let guards = analyze_if_guards(&schedule, &roots, &folds);
        assert_eq!(guards[0].true_range(), (4, 5), "the loop, not its pointer");
    }

    /// A fold nested in the arm's fold is priced by its own trips inside
    /// every trip of the outer one: `m · (n·k + …)`, recursively — the inner
    /// loop's price comes from the outer body's own `FoldReads`.
    #[test]
    fn price_a_nested_fold_by_the_product_of_its_trips() {
        const OUTER_TRIPS: u32 = 4;
        let cycles = CostModel::latency_prior();
        let (outer, inner) = (sum_over(0, OUTER_TRIPS), sum_over(1, ARM_TRIPS));
        // Outer body: `|I − i|`, `I` the inner fold, `i` the outer binder.
        let outer_body = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(20, ScheduledOp::Var(outer.binder().var())),
            def(21, ScheduledOp::Reduce(inner, ValueId(32))),
            def(
                22,
                ScheduledOp::Binary(OpKind::Sub, ValueId(21), ValueId(20))
            ),
            def(23, ScheduledOp::Unary(OpKind::Abs, ValueId(22))),
        ];
        let inner_body = distance_body(inner, 30);
        let inside = FoldReads::new(
            &outer_body,
            [(ValueId(21), &inner_body[..], &FoldReads::default())],
        );
        let schedule = if_over_a_fold(outer, 23);
        let folds = FoldReads::new(&schedule, [(ValueId(3), &outer_body[..], &inside)]);

        let k = cycles.cost(OpKind::Sub) + cycles.cost(OpKind::Abs);
        let inner_loop = cycles.fold_cost(inner, k);
        let outer_loop = cycles.fold_cost(outer, inner_loop + k);
        let arms = if_arms(&schedule, &[], &folds);
        assert_eq!(arms[0].cycles.true_arm, outer_loop);
        assert!(
            outer_loop >= (OUTER_TRIPS * ARM_TRIPS) as usize * k,
            "every trip of the outer loop runs the whole inner one"
        );
    }
}
