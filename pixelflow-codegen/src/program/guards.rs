//! What a scope's `If` arms cost, and what a loop a scope opens reads from it:
//! the two facts about a schedule that layout prices a branch by.
//!
//! [`FoldReads`] is the fold table — the edges a `Reduce` def's body adds to
//! the scope that opens it, and the price of the loop — and
//! [`MISPREDICT_PENALTY_CYCLES`] the bound an arm's price must clear to earn a
//! branch. Which arms those are, and the order that makes each one a run, is
//! `layout`'s: it is chosen from who owns what (`ownership`), not recovered
//! afterwards by a search over the order.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use pixelflow_ir::kind::OpKind;
use pixelflow_search::egraph::CostModel;

use crate::program::{Def, ScheduledOp, ValueId};

/// What each fold a scope opens reads from that scope, keyed by the fold's
/// `Reduce` def.
///
/// To [`operands`](crate::program::operands) a `Reduce` def is a
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
        // Ids are dense (handed out sequentially), so membership is one index.
        let mut here = alloc::vec![false; capacity];
        let mut opens = BTreeMap::new();
        for def in scope {
            here[def.value.0 as usize] = true;
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
                    .partition(|def| here.get(def.value.0 as usize).copied().unwrap_or(false));
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
        crate::program::all_operands(op).chain(fold.iter().copied())
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
    }
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
