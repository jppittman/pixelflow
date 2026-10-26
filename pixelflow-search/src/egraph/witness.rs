//! Extraction witnesses: reading the extractor's own choices back out.
//!
//! A **witness** for kernel `K` at budget `b_hi` is a term the same pipeline
//! produced at a smaller budget `b_lo` that is cheaper than what it produced
//! at `b_hi`. The e-graph is monotone, so every subterm of that term has an
//! e-class in `G(b_hi)` and the extractor walked past it. This module is the
//! read-only half of finding out where: it maps a term into a graph by
//! hash-cons lookup, and it re-runs the two production DP passes with a
//! recorder attached so the per-candidate cost tables that produced a choice
//! can be read.
//!
//! Research only — `#[cfg(feature = "provenance-journal")]`, the feature
//! every harness in `pixelflow-pipeline` already builds with and
//! `scripts/check-provenance-journal-scope.sh` keeps out of downstream
//! builds. Nothing here is on a production path, and the seams it uses
//! (`Dp`, `TieBreak`, `StageRecorder`) are monomorphized so production's
//! instantiation is byte-identical to the passes before they existed.
//!
//! Denotation: `docs/archive/plans/2026-09-08-extraction-witnesses.md`.

use std::collections::HashMap;

use pixelflow_ir::{Ir, LatticeShape, Shape, ShiftCount};

use super::and_or::Frozen;
use super::cost::CostFunction;
use super::extract::{
    Canonical, ChoiceCost, Dp, DpOutcome, ExtractionObjective, Insertion, Pricing,
    SHARED_DAG_PASS_BYTE_BUDGET, Selection, StageRecorder, TieBreak, shared_dag_dp_pass,
    tree_dp_pass,
};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Vocabulary;
use super::shift::{self, ShiftDirection};

// ---------------------------------------------------------------------------
// Recording a pass
// ---------------------------------------------------------------------------

/// Every candidate cost a DP pass computed, indexed by e-class then by node
/// index within the class.
#[derive(Clone, Debug, Default)]
pub struct CandidateTable {
    /// `cost[class][idx]` — the DP cost the pass assigned that candidate.
    pub cost: Vec<Vec<usize>>,
    /// `own[class][idx]` — that candidate's weighted own cost.
    pub own: Vec<Vec<usize>>,
    /// The index the class settled on.
    pub settled: Vec<Option<usize>>,
}

impl CandidateTable {
    fn with_classes(n: usize) -> Self {
        Self {
            cost: vec![Vec::new(); n],
            own: vec![Vec::new(); n],
            settled: vec![None; n],
        }
    }

    /// The DP cost of candidate `idx` in `class`, or `None` if the pass
    /// never priced it (the class was not reached).
    #[must_use]
    pub fn cost_of(&self, class: EClassId, idx: usize) -> Option<usize> {
        self.cost.get(class.index())?.get(idx).copied()
    }

    /// True iff `class` had two or more candidates at the settled cost —
    /// i.e. insertion order, not cost, decided it.
    #[must_use]
    pub fn is_tie(&self, class: EClassId) -> bool {
        let Some(idx) = self.settled.get(class.index()).copied().flatten() else {
            return false;
        };
        let row = &self.cost[class.index()];
        let Some(&best) = row.get(idx) else {
            return false;
        };
        row.iter().filter(|&&c| c == best).count() > 1
    }
}

impl StageRecorder for CandidateTable {
    fn candidate(&mut self, class: EClassId, idx: usize, cost: usize, own: usize) {
        let row = &mut self.cost[class.index()];
        if row.len() <= idx {
            row.resize(idx + 1, usize::MAX);
        }
        row[idx] = cost;
        let row = &mut self.own[class.index()];
        if row.len() <= idx {
            row.resize(idx + 1, usize::MAX);
        }
        row[idx] = own;
    }

    fn settled(&mut self, class: EClassId, idx: usize) {
        self.settled[class.index()] = Some(idx);
    }
}

// ---------------------------------------------------------------------------
// One traced extraction
// ---------------------------------------------------------------------------

/// One DP pass, traced: what it priced, what it picked, and what the term it
/// picked costs.
#[derive(Clone, Debug)]
pub struct PassTrace {
    /// The map production would return from this arm.
    pub choices: Vec<Option<usize>>,
    /// The cost of `choices`.
    pub cost: ChoiceCost,
    /// Per-candidate costs the pass computed.
    pub table: CandidateTable,
}

/// `extract_dag_scoped`, re-run with both arms traced.
///
/// The final `choices`/`cost` are what production returns for this graph —
/// asserted equal to `Optimizer::run`'s by the harness, so a drift between
/// this and the shipped path is loud rather than quietly analysed.
#[derive(Clone, Debug)]
pub struct StageTrace {
    pub tree: PassTrace,
    /// `None` when the shared pass ran out of its byte budget, which is
    /// production's `ExtractionObjective::TreeOnly`.
    pub shared: Option<PassTrace>,
    /// Which arm min-of-two kept.
    pub objective: ExtractionObjective,
    pub choices: Vec<Option<usize>>,
    pub cost: ChoiceCost,
}

/// Which extraction stage settled a class on the node it holds in the final
/// map — the answer to "who lost the witness here".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The winning arm's DP chose it.
    Dp,
    /// Both arms were computed and the *losing* arm held the witness's node
    /// here: min-of-two threw it away.
    MinOfTwo,
}

impl StageTrace {
    /// The winning arm.
    #[must_use]
    pub fn winner(&self) -> &PassTrace {
        match (self.objective, self.shared.as_ref()) {
            (ExtractionObjective::Shared, Some(s)) => s,
            _ => &self.tree,
        }
    }

    /// The losing arm, when there were two.
    #[must_use]
    pub fn loser(&self) -> Option<&PassTrace> {
        match (self.objective, self.shared.as_ref()) {
            (ExtractionObjective::Shared, Some(_)) => Some(&self.tree),
            (ExtractionObjective::TreeCheaper, Some(s)) => Some(s),
            _ => None,
        }
    }

    /// Which stage put `self.choices[class]` where it is, given the node the
    /// witness wanted there.
    #[must_use]
    pub fn stage_of(&self, class: EClassId, wanted: usize) -> Stage {
        let i = class.index();
        if self
            .loser()
            .and_then(|l| l.choices.get(i).copied().flatten())
            == Some(wanted)
        {
            return Stage::MinOfTwo;
        }
        Stage::Dp
    }
}

/// Re-run production's extraction on `egraph`, tracing both DP passes.
///
/// Identical in structure to `extract_dag_scoped`; the only difference is
/// that the passes are handed a recorder instead of `()`.
pub fn trace<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> StageTrace {
    let n = egraph.num_classes();
    let frozen = Frozen::new(egraph.as_frozen(), root);

    let pass_trace = |outcome: DpOutcome, table: CandidateTable| PassTrace {
        cost: outcome
            .selection
            .price(egraph.as_frozen(), &Pricing { costs, shape }),
        choices: outcome.choices,
        table,
    };

    let mut dp = Dp::new(costs, shape, Insertion, CandidateTable::with_classes(n));
    let tree = pass_trace(tree_dp_pass(frozen.problem(), &mut dp), dp.into_recorder());

    let mut dp = Dp::new(costs, shape, Insertion, CandidateTable::with_classes(n));
    let pass = shared_dag_dp_pass(frozen.problem(), &mut dp, SHARED_DAG_PASS_BYTE_BUDGET);
    let table = dp.into_recorder();
    let Some(shared_outcome) = pass.outcome else {
        let (choices, cost) = (tree.choices.clone(), tree.cost);
        return StageTrace {
            tree,
            shared: None,
            objective: ExtractionObjective::TreeOnly,
            choices,
            cost,
        };
    };
    let shared = pass_trace(shared_outcome, table);

    let (objective, winner) = if shared.cost.dag < tree.cost.dag {
        (ExtractionObjective::Shared, &shared)
    } else {
        (ExtractionObjective::TreeCheaper, &tree)
    };
    let (choices, cost) = (winner.choices.clone(), winner.cost);
    StageTrace {
        tree,
        shared: Some(shared),
        objective,
        choices,
        cost,
    }
}

// ---------------------------------------------------------------------------
// The tie-break A/B
// ---------------------------------------------------------------------------

/// How a variant extraction breaks ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ties {
    /// Production: the first admissible node — insertion order.
    Insertion,
    /// A total order on the node's own content.
    Canonical,
}

/// `extract_dag_scoped` under `ties`, returning the same pair production
/// reports. With `Ties::Insertion` this is production, arithmetic for
/// arithmetic.
#[must_use]
pub fn extract_under<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
    ties: Ties,
) -> (Vec<Option<usize>>, ChoiceCost) {
    match ties {
        Ties::Insertion => run_arms(egraph, root, costs, shape, Insertion),
        Ties::Canonical => run_arms(egraph, root, costs, shape, Canonical),
    }
}

fn run_arms<C: CostFunction, T: TieBreak + Copy>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
    ties: T,
) -> (Vec<Option<usize>>, ChoiceCost) {
    let frozen = Frozen::new(egraph.as_frozen(), root);
    let mut dp = Dp::new(costs, shape, ties, ());
    let tree = tree_dp_pass(frozen.problem(), &mut dp);
    let tree_cost = tree
        .selection
        .price(egraph.as_frozen(), &Pricing { costs, shape });

    let mut dp = Dp::new(costs, shape, ties, ());
    let pass = shared_dag_dp_pass(frozen.problem(), &mut dp, SHARED_DAG_PASS_BYTE_BUDGET);
    let Some(shared) = pass.outcome else {
        return (tree.choices, tree_cost);
    };
    let shared_cost = shared
        .selection
        .price(egraph.as_frozen(), &Pricing { costs, shape });
    if shared_cost.dag < tree_cost.dag {
        return (shared.choices, shared_cost);
    }
    (tree.choices, tree_cost)
}

// ---------------------------------------------------------------------------
// Mapping a term into a graph
// ---------------------------------------------------------------------------

/// A subterm of the witness that has no e-class in the bigger graph.
///
/// Monotonicity says this cannot happen; it is reported rather than assumed
/// away, because the assumption is exactly what the instrument is testing.
#[derive(Clone, Debug)]
pub struct TermMiss {
    /// The node shape that was not found, rendered.
    pub node: String,
    /// How many subterms had already been mapped when it was hit.
    pub mapped: usize,
}

/// A term's induced choice map over `egraph`'s classes.
#[derive(Clone, Debug)]
pub struct InducedChoices {
    /// `C_T`: one node index per class the term occupies.
    pub choices: Vec<Option<usize>>,
    /// The class the term's root landed in.
    pub root: EClassId,
    /// How many of the term's distinct nodes collapsed onto a class another
    /// of its nodes already occupied — equal in the bigger graph, distinct
    /// in the smaller one. The first in post-order keeps the class, so the
    /// induced map names a term no dearer than the one it was read from.
    pub merges: usize,
    /// Distinct classes the term occupies.
    pub occupied: usize,
}

/// Look `term` up in `egraph`, node by node, without inserting anything.
///
/// The table is built read-only from `class_ids()` × `nodes()` with operands
/// canonicalized by `find`, a fold's body excepted — the memo's content,
/// reached without touching the memo. A subterm that is absent is returned as
/// a [`TermMiss`], never silently skipped.
///
/// # Errors
///
/// [`TermMiss`] when a subterm of `term` has no e-class in `egraph`.
pub fn induce<I: Ir>(
    egraph: &EGraph,
    term: &I,
    root: I::Ref,
    vocab: Vocabulary,
) -> Result<InducedChoices, TermMiss> {
    let mut table: HashMap<ENode, (EClassId, usize)> = HashMap::new();
    for class in egraph.class_ids() {
        let canonical = egraph.find(class);
        if canonical != class {
            continue;
        }
        for (idx, node) in egraph.nodes(canonical).enumerate() {
            // A fold's body is keyed as stored. This walk has never read it
            // through `find`, and the miss counts in
            // docs/results/2026-09-08-extraction-witnesses.md were measured
            // that way; reading it would change what the bin reports, which
            // is a decision of its own.
            let key = match node {
                ENode::Reduce { .. } => *node,
                _ => egraph.canonical_node(*node),
            };
            table.entry(key).or_insert((canonical, idx));
        }
    }

    let mut choices: Vec<Option<usize>> = vec![None; egraph.num_classes()];
    let mut memo: std::collections::BTreeMap<I::Ref, EClassId> = std::collections::BTreeMap::new();
    let mut merges = 0usize;
    let mut occupied = 0usize;
    let mut root_class = None;

    for r in post_order_term(term, root) {
        if memo.contains_key(&r) {
            continue;
        }
        let node = match term.project(r) {
            Shape::Var(i) => ENode::Var(i),
            Shape::Const(v) => ENode::Const(v),
            Shape::Param(i) => ENode::Param(i),
            Shape::Buffer(d) => ENode::Buffer(d),
            Shape::Uniform(d) => ENode::Uniform(d),
            Shape::Op(kind, children) => {
                if !vocab.admits(kind) {
                    return Err(TermMiss {
                        node: format!("{kind:?} (not in vocabulary)"),
                        mapped: memo.len(),
                    });
                }
                if children.len() != kind.arity() {
                    return Err(TermMiss {
                        node: format!("{kind:?} (applied to {} operands)", children.len()),
                        mapped: memo.len(),
                    });
                }
                match ShiftDirection::of(kind) {
                    Some(direction) => {
                        let Some((operand, count)) = shift::spelled(term, children) else {
                            return Err(TermMiss {
                                node: format!(
                                    "{kind:?} (its count is not a literal integer in 0..{})",
                                    ShiftCount::WORD_BITS
                                ),
                                mapped: memo.len(),
                            });
                        };
                        let operand = *memo.get(&operand).expect("post-order maps children first");
                        ENode::Shift {
                            direction,
                            count,
                            operand,
                        }
                    }
                    None => {
                        let mut operands = Vec::with_capacity(children.len());
                        for i in 0..children.len() {
                            let c = children.get(i).expect("index < len");
                            let class = *memo.get(&c).expect("post-order maps children first");
                            operands.push(class);
                        }
                        ENode::apply(kind, &operands)
                    }
                }
            }
            Shape::Reduce { fold, lo, hi, body } => {
                let class = |r| *memo.get(&r).expect("post-order maps children first");
                ENode::reduce(fold, lo.map(class), hi.map(class), class(body))
            }
            // Not a term a witness graph holds: only the runtime tier's
            // unit walk admits a reference (as a unit's opaque leaf), so a
            // witness term carrying one came from somewhere that skipped
            // the pipeline.
            Shape::Ref(key) => {
                return Err(TermMiss {
                    node: format!("{key:?} (a reference; expand_refs first)"),
                    mapped: memo.len(),
                });
            }
            // Nor a `Write`: an effect the legalize passes build after
            // extraction, which `insert` declines on principle.
            Shape::Write { .. } => {
                return Err(TermMiss {
                    node: "Write (insert declines a Write)".to_string(),
                    mapped: memo.len(),
                });
            }
        };
        let Some(&(class, idx)) = table.get(&node) else {
            return Err(TermMiss {
                node: format!("{node:?}"),
                mapped: memo.len(),
            });
        };
        memo.insert(r, class);
        match choices[class.index()] {
            Some(_) => merges += 1,
            None => {
                choices[class.index()] = Some(idx);
                occupied += 1;
            }
        }
        root_class = Some(class);
    }

    Ok(InducedChoices {
        choices,
        root: root_class.expect("a term has at least one node"),
        merges,
        occupied,
    })
}

/// The term's nodes, children before parents, each once.
fn post_order_term<I: Ir>(term: &I, root: I::Ref) -> Vec<I::Ref> {
    enum Task<R> {
        Visit(R),
        Emit(R),
    }
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<I::Ref> = std::collections::BTreeSet::new();
    let mut stack = vec![Task::Visit(root)];
    while let Some(t) = stack.pop() {
        match t {
            Task::Visit(r) => {
                if seen.contains(&r) {
                    continue;
                }
                stack.push(Task::Emit(r));
                match term.project(r) {
                    Shape::Op(kind, children) => {
                        // A shift's count is a parameter of the node, as in
                        // `insert`; a shift that cannot be read that way
                        // is visited whole and missed by `induce`.
                        let shift =
                            ShiftDirection::of(kind).and_then(|_| shift::spelled(term, children));
                        match shift {
                            Some((operand, _)) => stack.push(Task::Visit(operand)),
                            None => {
                                for i in (0..children.len()).rev() {
                                    stack.push(Task::Visit(children.get(i).expect("index < len")));
                                }
                            }
                        }
                    }
                    // A fold's children are its body and each value bound.
                    // Missing one would leave it unmapped and the fold
                    // unmatchable. Reverse, as for an op, so they pop in
                    // `children()` order: the body, then `lo`'s, then `hi`'s.
                    Shape::Reduce { body, lo, hi, .. } => {
                        for child in [hi.value(), lo.value()].into_iter().flatten() {
                            stack.push(Task::Visit(child));
                        }
                        stack.push(Task::Visit(body));
                    }
                    Shape::Write { value, .. } => stack.push(Task::Visit(value)),
                    _ => {}
                }
            }
            Task::Emit(r) => {
                if seen.insert(r) {
                    out.push(r);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Costing a hypothetical map
// ---------------------------------------------------------------------------

/// The price of the term `choices` names, or `None` when it names none
/// (a cyclic or incomplete map).
///
/// A single swap can make a choice graph cyclic — that is a fact about the
/// swap, not a broken invariant, so this is the form the swap search needs.
#[must_use]
pub fn cost_if_well_founded<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
    costs: &C,
    shape: LatticeShape,
) -> Option<ChoiceCost> {
    let term = Selection::of_map(egraph.as_frozen(), root, choices).ok()?;
    Some(term.price(egraph.as_frozen(), &Pricing { costs, shape }))
}

/// The classes `choices` reaches from `root`, children before parents.
///
/// # Panics
///
/// If `choices` is not well-founded from `root`.
#[must_use]
pub fn reachable_under(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Vec<EClassId> {
    well_founded_term(egraph, root, choices)
        .picks()
        .map(|(class, _)| class)
        .collect()
}

/// Materialize a choice map as an arena — `Optimized::to_arena` for a map
/// this module produced rather than one `Optimizer::run` returned.
///
/// # Panics
///
/// If `choices` is not well-founded from `root`.
#[must_use]
pub fn arena_of(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> (pixelflow_ir::arena::ExprArena, pixelflow_ir::arena::ExprId) {
    well_founded_term(egraph, root, choices).to_arena(egraph.as_frozen())
}

fn well_founded_term(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> Selection {
    Selection::of_map(egraph.as_frozen(), root, choices).unwrap_or_else(|refused| {
        panic!(
            "witness: the choice map is not well-founded from root {}: {refused}",
            root.0
        )
    })
}
