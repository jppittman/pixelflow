//! Extraction: materialise a concrete arena expression from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model and materialises it as an
//! [`pixelflow_ir::ExprArena`].

use super::cost::{CostFunction, CostModel};
use super::deps::var_variance;
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use pixelflow_ir::{LatticeShape, Variance};

/// A witnessed selection: an e-graph, a root e-class, and a well-founded
/// choice function from every e-class reachable from `root` to the node
/// index selected for it.
///
/// "Well-founded" means: every reachable e-class has a recorded choice, and
/// the choice graph is acyclic (bottom-up realizable). Those two properties
/// are exactly what let a choice function be materialised at all — an
/// unvalidated `Vec<Option<usize>>` can loop forever when walked (a real
/// 2.7GB OOM, see `choices_to_arena`'s doc comment), and a call site that
/// forgets to repair or backfill it produces that bug silently.
///
/// `Extraction` makes the bug class unrepresentable: the only ways to
/// obtain a value of this type are [`Extraction::from_dp`] (wraps
/// [`repair_choices_well_founded`]) and [`Extraction::from_backfill`]
/// (wraps [`backfill_well_founded`]) — both establish well-foundedness as
/// part of construction, so a bare unvalidated vector can never cross into
/// [`choices_to_arena`] or the edge walker
/// ([`crate::nnue::EdgeTrace::from_extraction`]), which accept only
/// `&Extraction`. See docs/plans/2026-08-17-cost-model-domain.md §1
/// "Extraction (J2)".
pub struct Extraction<'g> {
    egraph: &'g EGraph,
    root: EClassId,
    choices: Vec<Option<usize>>,
}

impl<'g> Extraction<'g> {
    /// The DP path's smart constructor: makes `choices` well-founded via
    /// [`repair_choices_well_founded`] (resolving any mutual cycles the
    /// bottom-up DP recorded — `CYCLE_COST` penalizes only
    /// self-references, so two merged classes can each cheapest-pick a
    /// node through the other), then seals the result.
    pub(crate) fn from_dp(
        egraph: &'g EGraph,
        root: EClassId,
        mut choices: Vec<Option<usize>>,
    ) -> Self {
        let root = egraph.find(root);
        repair_choices_well_founded(egraph, root, &mut choices);
        Self {
            egraph,
            root,
            choices,
        }
    }

    /// The hand-built path's smart constructor: fills any e-class
    /// reachable from `root` still lacking a choice via
    /// [`backfill_well_founded`], then seals the result. Tests and research
    /// harnesses that pin specific node choices come in here; production
    /// extraction comes in through [`Extraction::from_dp`].
    ///
    /// # Panics
    ///
    /// Panics if the result is cyclic. A well-founded backfill cannot
    /// itself introduce a cycle — reaching this means the caller sealed a
    /// choice state that was never cycle-checked, which is a bug at the
    /// call site, not a recoverable outcome (NO SILENT FAILURES).
    pub(crate) fn from_backfill(
        egraph: &'g EGraph,
        root: EClassId,
        mut choices: Vec<Option<usize>>,
    ) -> Self {
        let root = egraph.find(root);
        backfill_well_founded(egraph, root, &mut choices);
        assert!(
            !choices_have_cycle_from(egraph, root, &choices),
            "Extraction::from_backfill: choice graph is cyclic after backfill for root {} — \
             a well-founded backfill cannot itself introduce a cycle; the caller sealed a \
             state that was never cycle-checked",
            root.0
        );
        Self {
            egraph,
            root,
            choices,
        }
    }

    /// The e-graph this extraction selects nodes from.
    pub fn egraph(&self) -> &'g EGraph {
        self.egraph
    }

    /// The extraction's (canonical) root e-class.
    pub fn root(&self) -> EClassId {
        self.root
    }

    /// The chosen node index for `class`, if `class` is reachable from
    /// [`Self::root`]. `None` for classes outside the extraction.
    pub fn choice(&self, class: EClassId) -> Option<usize> {
        let idx = self.egraph.find(class).0 as usize;
        self.choices.get(idx).copied().flatten()
    }

    /// Read-only view of the raw choice vector, indexed by canonical
    /// e-class id. Still only reachable through a sealed `Extraction`.
    pub(crate) fn choices(&self) -> &[Option<usize>] {
        &self.choices
    }

    /// The choice vector [`choices_to_arena`] will actually materialise:
    /// every `Shl`/`Shr` count child re-pinned to a `Const` representative of
    /// its class ([`pin_shift_counts`]), because the emitter's shift lowering
    /// requires an immediate and a count class can legitimately hold
    /// arithmetic that is value-equal to a constant without being one
    /// (e.g. `Y - Y` merged with `Const(0)`).
    ///
    /// Any consumer that walks an `Extraction` — not just
    /// [`choices_to_arena`] itself — must walk this view, not [`Self::choices`]
    /// directly: describing the raw (possibly non-`Const`) choice for a
    /// count class would describe a DAG that is not the one actually
    /// compiled. `ChoicesCostDag` in `crate::nnue::factored` takes this view
    /// rather than re-deriving it (one definition, imported, not restated).
    pub(crate) fn pinned_choices(&self) -> Vec<Option<usize>> {
        pin_shift_counts(self.egraph, self.root, &self.choices)
    }

    /// Variance histogram (fraction const / frame-uniform / scanline-uniform
    /// / pixel-varying) of the CHOSEN nodes, not the class-wide meet
    /// [`super::DepsAnalysis`] would compute over the whole e-graph.
    ///
    /// Materialises the choice function once via [`choices_to_arena`] and
    /// classifies that arena — P1(c) of
    /// docs/plans/2026-08-17-cost-model-domain.md: once a rewrite merges a
    /// pixel-varying node into a class alongside a constant one, the
    /// class-wide meet reports CONST regardless of which node the
    /// extraction actually chose, so only the materialised DAG describes
    /// what was picked.
    #[must_use]
    pub fn chosen_variance(&self) -> [f32; crate::nnue::factored::SCALAR_FEATURE_COUNT] {
        let (arena, _root) = choices_to_arena(self);
        crate::nnue::factored::variance_histogram(&arena)
    }

    /// Unwrap into the raw choice vector.
    ///
    /// Kept for legacy raw-vector consumers (`ExtractionPolicy::choices`,
    /// `compute_ref_counts`, `build_extracted_dag_from_choices`) that
    /// predate this type; new code should consume `&Extraction` instead.
    pub(crate) fn into_choices(self) -> Vec<Option<usize>> {
        self.choices
    }

    /// Attempt to build a candidate extraction that swaps `class`'s choice
    /// to `node_idx`, backfilling any newly-introduced children and
    /// rejecting (returning `None`) if the swap closes a cycle through
    /// already-chosen classes.
    ///
    /// This is [`IncrementalExtractor::extract_choices_only`]'s
    /// per-candidate constructor — unlike [`Extraction::from_backfill`], a
    /// cycle here is a normal search outcome (reject this candidate, try
    /// another), not a bug.
    ///
    /// ## Acyclicity check scope, and the one case it does NOT match
    /// `choices_have_cycle_from` bit-for-bit
    ///
    /// The check is scoped to `canonical`'s own new outgoing edges
    /// ([`choices_have_cycle_through`]) rather than re-walking the whole
    /// tree from `root`: this swap changes exactly one vertex's outgoing
    /// edge — `canonical`'s — plus adds fresh backfilled subtrees hanging
    /// off `node_idx`'s children, each internally acyclic by construction
    /// ([`backfill_well_founded`] never revisits a class that already has a
    /// choice, so a backfilled region can only ever terminate by joining
    /// the pre-existing tree, not by cutting back into it). PROVIDED
    /// `canonical` is itself currently reachable from `root`, a graph
    /// mutated at exactly one vertex's outgoing edge gains a cycle if and
    /// only if that vertex becomes reachable from itself through the new
    /// edge, so checking forward reachability from `node_idx`'s children
    /// back to `canonical` is equivalent to (but bounded by the
    /// forward-reachable set from the new children, not the whole
    /// root-reachable tree, unlike) a full re-walk from root.
    ///
    /// That proviso can fail: `extract_choices_only`'s refinement loop
    /// takes its `active` class list once per pass and then mutates
    /// `current_extraction` as it goes, so a class visited later in the
    /// same pass can, by the time its own `try_swap` call runs, no longer
    /// be root-reachable at all — an earlier accepted swap in the same
    /// pass severed the only path to it. For such a `canonical`, this
    /// check and a full `choices_have_cycle_from(root)` walk can disagree
    /// (this one may reject a "cycle" the root walk would call
    /// unreachable-hence-irrelevant, or the reverse). That disagreement
    /// never reaches an observable output, though: every consumer that
    /// scores or materialises a candidate walks forward from `root` only
    /// ([`choices_to_arena`], which is what a [`Reranker`]'s score is
    /// computed from), so a `canonical` unreachable from root is invisible
    /// to it regardless of what this function decides; accept or reject,
    /// the candidate's score is bit-identical to `current_cost` and it
    /// never wins the refinement loop's strict `<` comparison.
    pub(crate) fn try_swap(&self, class: EClassId, node_idx: usize) -> Option<Self> {
        let canonical = self.egraph.find(class);
        let mut choices = self.choices.clone();
        choices[canonical.0 as usize] = Some(node_idx);

        let new_children: &[EClassId] = match self.egraph.nodes(canonical).get(node_idx) {
            Some(ENode::Op { children, .. }) => children,
            _ => &[],
        };

        for &child in new_children {
            backfill_well_founded(self.egraph, child, &mut choices);
        }

        // A leaf swap (Var/Const/Buffer, or an Op with no children) adds no
        // outgoing edge at all, so it cannot possibly close a cycle — only
        // removes `canonical`'s old edges, which can only ever break a
        // cycle, never create one. Skip the walk entirely rather than
        // paying for a reachability check with nothing to find.
        if !new_children.is_empty() {
            let has_cycle =
                choices_have_cycle_through(self.egraph, canonical, new_children, &choices);
            if has_cycle {
                return None;
            }
        }

        Some(Self {
            egraph: self.egraph,
            root: self.root,
            choices,
        })
    }
}

// ============================================================================
// Swap-refinement search (Reranker seam)
// ============================================================================
//
// The extraction-head program that used to drive this search with a trained
// NNUE tied the static table on schedule-free kernels (workshop paper on
// branch `claude/workshop-writeup`, PR #1072, closed without merging — not
// in this tree; see docs/plans/2026-09-01-schedule-cost-model-denotation.md)
// — but JP's ruling on the
// program's SHAPE was narrower than "delete everything it touched":
// "I don't think what we have was the correct shape. I think it's right as
// an idea. [...] Egraph extraction is the place where code gen's schedule
// choice is going to go." What was wrong was a bag-of-edges MLP that
// predicted TOTAL cost and tried to replace the additive table outright;
// what was right is this local search — swap one e-class's choice, rescore,
// keep strict improvements — as the reranking primitive a future
// non-additive cost model needs. So the search stays, generic over a
// [`Reranker`] seam with no implementation shipped: the NNUE-specific
// scoring is deleted with the program that trained it, not the search that
// used to call it.

/// A pluggable scoring function for [`IncrementalExtractor`]'s
/// swap-refinement search. Lower is better — the search treats this as a
/// cost, not a probability or utility, and accepts a candidate only on a
/// strict improvement.
///
/// No implementation ships in this crate. This trait is the seam a future
/// residual reranker over the additive latency-prior table plugs into; only
/// test-only rerankers exist today (see `egraph::extract::tests`).
pub trait Reranker {
    /// Score the DAG `extraction` selects, materialised as `arena` — the
    /// same `(arena, root)` pair [`choices_to_arena`] would return for
    /// `extraction`, computed once per candidate by the search loop.
    fn score(&self, extraction: &Extraction<'_>, arena: &pixelflow_ir::ExprArena) -> f64;
}

/// Incremental swap-refinement extractor, generic over a [`Reranker`].
///
/// - **Pass 1 (Bootstrap)**: extract a well-founded starting choice per
///   e-class via [`Extraction::from_backfill`].
/// - **Passes 2..=`MAX_PASSES` (Refine)**: for each active e-class, try
///   alternative nodes (up to `top_k`) via [`Extraction::try_swap`],
///   rescoring each candidate through `reranker`. Accept the single best
///   strict improvement per class per pass. Repeat until a pass makes no
///   improvement (fixpoint) or `MAX_PASSES` is reached.
pub struct IncrementalExtractor<'a, R: Reranker + ?Sized> {
    reranker: &'a R,
    top_k: usize,
}

impl<'a, R: Reranker + ?Sized> IncrementalExtractor<'a, R> {
    /// `top_k` bounds how many alternative nodes per e-class are evaluated
    /// during refinement passes.
    pub fn new(reranker: &'a R, top_k: usize) -> Self {
        Self { reranker, top_k }
    }

    /// Run the extraction refinement loop and return `(score, extraction)`.
    ///
    /// Call [`choices_to_arena`] on the returned [`Extraction`] to
    /// materialise the extracted DAG.
    pub fn extract_choices_only<'g>(
        &self,
        egraph: &'g EGraph,
        root_class: EClassId,
    ) -> (f64, Extraction<'g>) {
        const MAX_PASSES: usize = 10;

        // Pass 1: Bootstrap with a well-founded choice per reachable e-class.
        // For unmerged classes this is the original expression's node; where
        // saturation merges reordered node lists it is the first admissible
        // node instead — refinement below then improves whatever valid
        // start this provides.
        let num_classes = egraph.num_classes();
        let choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut current_extraction = Extraction::from_backfill(egraph, root_class, choices);
        let mut current_cost = self.score(&current_extraction);

        // Refinement passes: for each e-class, try ALL alternatives (up to
        // top_k), accept the BEST improvement (not first). Repeat until
        // fixpoint or max passes.
        for _pass in 0..MAX_PASSES {
            let active = self.get_active_classes(&current_extraction);
            let mut improved = false;

            for &class in &active {
                let canonical = egraph.find(class);
                let nodes = egraph.nodes(canonical);
                if nodes.len() <= 1 {
                    continue;
                }

                let current_node_idx = current_extraction.choice(canonical).unwrap_or_else(|| {
                    panic!(
                        "extract_choices_only: e-class {} is active (reachable from root) \
                         but has no recorded choice — backfill_well_founded should have \
                         populated every class returned by get_active_classes",
                        canonical.0
                    )
                });
                let candidates_to_try = nodes.len().min(self.top_k);

                // Best-improvement: evaluate ALL candidates, pick the
                // cheapest. Each candidate is evaluated on a COMPLETE choice
                // state (the swap applied AND the newly reachable subtree
                // backfilled, cycle-checked via `Extraction::try_swap`), so
                // the state scored is the state adopted.
                let mut best_swap_cost = current_cost;
                let mut best_swap: Option<Extraction<'g>> = None;

                for node_idx in 0..candidates_to_try {
                    if node_idx == current_node_idx {
                        continue;
                    }

                    // Skip self-referential candidates (would create cycles).
                    {
                        let children = (&nodes[node_idx]).children_slice();
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            continue;
                        }
                    }

                    // Rejects candidates that close a cycle through classes
                    // that already held choices.
                    let Some(candidate) = current_extraction.try_swap(canonical, node_idx) else {
                        continue;
                    };

                    let test_cost = self.score(&candidate);
                    if test_cost < best_swap_cost {
                        best_swap_cost = test_cost;
                        best_swap = Some(candidate);
                    }
                }

                if let Some(swapped) = best_swap {
                    // Adopt EXACTLY the state that was cycle-checked and
                    // scored — no post-acceptance re-derivation.
                    current_extraction = swapped;
                    current_cost = best_swap_cost;
                    improved = true;
                }
            }

            if !improved {
                break; // Fixpoint
            }
        }

        (current_cost, current_extraction)
    }

    /// Materialise `extraction` and hand it to `self.reranker`.
    fn score(&self, extraction: &Extraction<'_>) -> f64 {
        let (arena, _root) = choices_to_arena(extraction);
        self.reranker.score(extraction, &arena)
    }

    /// Walk the current best tree and collect active (reachable) e-class IDs.
    fn get_active_classes(&self, extraction: &Extraction<'_>) -> Vec<EClassId> {
        let egraph = extraction.egraph();
        let root = extraction.root();

        let mut active = Vec::new();
        let mut visited: alloc::vec::Vec<bool> = alloc::vec![false; egraph.num_classes()];
        let mut stack = vec![root];

        while let Some(class) = stack.pop() {
            let canonical = egraph.find(class);
            let idx = canonical.0 as usize;
            if visited[idx] {
                continue;
            }
            visited[idx] = true;

            active.push(canonical);

            let node_idx = extraction.choice(canonical).unwrap_or_else(|| {
                panic!(
                    "get_active_classes: e-class {} reachable from root has no recorded \
                     choice — extract_choices_only must call backfill_well_founded \
                     transitively before invoking get_active_classes",
                    canonical.0
                )
            });
            let nodes = egraph.nodes(canonical);
            if node_idx < nodes.len() {
                {
                    let children = (&nodes[node_idx]).children_slice();
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }

        active
    }
}

/// Fill in a **well-founded** choice for every e-class reachable from
/// `start` that doesn't yet have a recorded choice.
///
/// This restores the invariant relied on throughout `extract_choices_only`
/// and its helpers (`get_active_classes`, the refinement loop, and
/// `choices_to_arena`): every e-class reachable from the root via the
/// *currently chosen* nodes has a `Some` entry in `choices`, and the choice
/// graph is a DAG.
///
/// The previous version filled `Some(0)` (the "original/first" node) — which
/// is only the original expression's node for an UNMERGED class. After
/// saturation unions, a class's node list is a merge, so two classes can each
/// hold a node referencing the other at index 0 and the node-0 choice
/// function is CYCLIC. A cyclic bootstrap is unrecoverable downstream: every
/// refinement swap fails the cycle check (the pre-existing cycle is reachable
/// no matter what is swapped) and the cyclic set flows to `choices_to_arena`
/// (observed: a full-DEV bench run died by OOM there; the restart-DFS
/// `break_choice_cycles` repair did not terminate on the same graph).
///
/// So the fill is constructed well-founded instead of repaired after the
/// fact: Kahn-style admission. A class is admitted once ANY of its nodes has
/// all children admitted (leaves admit immediately; classes that already hold
/// a choice count as admitted, since their subtrees are complete by this same
/// invariant). The admitted node is recorded as the choice. Every well-formed
/// e-graph admits all reachable classes — each class was created holding a
/// node whose children existed before it, so creation order is a topological
/// witness — and a class that never admits is a corrupt graph, reported by
/// panic rather than papered over.
///
/// # Panics
///
/// Panics if some reachable class cannot be admitted (the e-graph holds a
/// class none of whose nodes has admissible children — a structurally corrupt
/// graph, never a rewrite judgment call).
fn backfill_well_founded(egraph: &EGraph, start: EClassId, choices: &mut [Option<usize>]) {
    let num_classes = choices.len();

    // Scope: canonical classes reachable from `start` through EVERY node of
    // each not-yet-chosen class. Classes with an existing choice are complete
    // subtrees and stop the walk.
    let mut scope_pos: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut scope: Vec<u32> = Vec::new();
    let mut stack = alloc::vec![egraph.find(start)];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || scope_pos[idx].is_some() || choices[idx].is_some() {
            continue;
        }
        scope_pos[idx] = Some(scope.len());
        scope.push(canonical.0);
        for node in egraph.nodes(canonical) {
            {
                let children = (node).children_slice();
                for &child in children {
                    stack.push(egraph.find(child));
                }
            }
        }
    }
    if scope.is_empty() {
        return;
    }

    // Per (scope class, node): how many child references still await
    // admission. Duplicate children count once per occurrence. Reverse edges
    // record which (class, node) counters to decrement when a class admits.
    let mut pending: Vec<Vec<usize>> = Vec::with_capacity(scope.len());
    let mut reverse: Vec<Vec<(usize, usize)>> = alloc::vec![Vec::new(); scope.len()];
    let mut ready: Vec<u32> = Vec::new();
    for (pos, &cid) in scope.iter().enumerate() {
        let canonical = EClassId(cid);
        let nodes = egraph.nodes(canonical);
        let mut per_node = Vec::with_capacity(nodes.len());
        let mut any_ready = false;
        for (node_idx, node) in nodes.iter().enumerate() {
            let mut count = 0usize;
            {
                let children = (node).children_slice();
                for &child in children {
                    let child_idx = egraph.find(child).0 as usize;
                    if let Some(child_pos) = scope_pos[child_idx] {
                        count += 1;
                        reverse[child_pos].push((pos, node_idx));
                    }
                }
            }
            if count == 0 {
                any_ready = true;
            }
            per_node.push(count);
        }
        pending.push(per_node);
        if any_ready {
            ready.push(cid);
        }
    }

    let mut admitted = 0usize;
    let mut queue: alloc::collections::VecDeque<u32> = ready.into_iter().collect();
    while let Some(cid) = queue.pop_front() {
        let idx = cid as usize;
        if choices[idx].is_some() {
            continue; // Admitted through an earlier queue entry.
        }
        let pos = scope_pos[idx].expect("queued class is in scope");
        let node_idx = pending[pos]
            .iter()
            .position(|&count| count == 0)
            .expect("queued class has a zero-pending node");
        choices[idx] = Some(node_idx);
        admitted += 1;
        for &(parent_pos, parent_node) in &reverse[pos] {
            pending[parent_pos][parent_node] -= 1;
            if pending[parent_pos][parent_node] == 0 {
                queue.push_back(scope[parent_pos]);
            }
        }
    }

    assert!(
        admitted == scope.len(),
        "backfill_well_founded: {} of {} reachable e-classes cannot be given a well-founded \
         choice — the e-graph holds classes none of whose nodes has admissible children, which \
         is structural corruption, not a rewrite outcome (root {})",
        scope.len() - admitted,
        scope.len(),
        start.0
    );
}

/// Check whether the current extraction choices contain a cycle reachable from `root`.
fn choices_have_cycle_from(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> bool {
    let num_classes = egraph.num_classes();
    let mut color: Vec<u8> = alloc::vec![0; num_classes];
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes {
            continue;
        }

        if children_done {
            color[idx] = 2;
            continue;
        }

        match color[idx] {
            1 => return true,
            2 => continue,
            _ => {}
        }

        color[idx] = 1;
        stack.push((canonical, true));

        // `unwrap_or(0)` here is an ANALYSIS default, not an identity
        // sentinel, and the difference is why it stays: a class with no
        // recorded choice contributes no edge to the extracted DAG, so
        // walking its node 0 can only add phantom edges — which can only
        // report a cycle that is not there (a rejected swap, i.e. cost),
        // never miss one that is. Nothing materialised comes out of here;
        // the one place a missing choice would become a wrong *term* is
        // `choices_to_arena`, which panics on it instead.
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        if let Some(ENode::Op { children, .. }) = egraph.nodes(canonical).get(node_idx) {
            for &child in children.iter().rev() {
                stack.push((child, false));
            }
        }
    }

    false
}

/// Check whether `canonical` is forward-reachable from `new_children` by
/// following `choices`. Used by [`Extraction::try_swap`] as the equivalent,
/// but far cheaper, replacement for a full [`choices_have_cycle_from`]
/// re-walk from root — see that method's doc comment for why scoping the
/// check to the one changed vertex's new edges is sound. Plain reachability
/// (no gray/black coloring) is enough here, unlike
/// [`choices_have_cycle_from`]: we are not distinguishing "cycle" from
/// "revisited via legitimate DAG sharing" for the whole tree, only asking
/// whether ANY forward path from the new edges leads back to `canonical` —
/// which is exactly what a cycle through the swapped vertex means.
///
/// Bounded by the size of the forward-reachable set from `new_children`,
/// not `egraph.num_classes()`.
fn choices_have_cycle_through(
    egraph: &EGraph,
    canonical: EClassId,
    new_children: &[EClassId],
    choices: &[Option<usize>],
) -> bool {
    let num_classes = choices.len();
    let mut visited: alloc::collections::BTreeSet<u32> = alloc::collections::BTreeSet::new();
    let mut stack: Vec<EClassId> = new_children.to_vec();

    while let Some(class) = stack.pop() {
        let c = egraph.find(class);
        if c == canonical {
            return true;
        }
        let idx = c.0 as usize;
        if idx >= num_classes || !visited.insert(c.0) {
            continue;
        }
        // Same analysis default as `choices_have_cycle_from`, safe for the
        // same reason: extra edges can only over-report reachability, which
        // costs a rejected swap and never a wrong term.
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        if let Some(ENode::Op { children, .. }) = egraph.nodes(c).get(node_idx) {
            for &child in children {
                stack.push(child);
            }
        }
    }

    false
}

/// Make an existing choice function well-founded, keeping every recorded
/// choice that participates in no cycle.
///
/// The bottom-up DP can record mutual cycles (class 68 picks `neg(69)` while
/// class 69 picks `neg(68)`): `CYCLE_COST` penalizes only SELF-references, so
/// two merged classes can each cheapest-pick a node through the other. The
/// previous repair (`break_choice_cycles`, a restart-DFS that broke one cycle
/// per pass) did not terminate on saturated FULL-tier graphs — its Strategy-3
/// fallback can leave the cycle intact, and the restart then rediscovers the
/// same cycle forever (observed live on two DEV kernels, each pinning a core
/// for minutes before the run was killed).
///
/// So, like [`backfill_well_founded`], the repair is a construction rather
/// than a patch loop — Kahn admission in two interleaved phases:
///
/// 1. **Drain**: admit every class whose RECORDED node has all children
///    admitted. Acyclic regions of the DP's choice graph are admitted here
///    unchanged, so their cost-optimal selection is kept.
/// 2. **Unstick**: when the drain stalls before every class is admitted, the
///    remaining classes all sit on cycles (or behind them). Admit ONE ready
///    class through its first admissible node — rewriting its choice — and
///    return to draining. Each unstick admits a class, so the loop is bounded
///    by the class count; total work is linear in e-graph edges.
///
/// # Panics
///
/// Panics if admission exhausts with classes left over — a class none of
/// whose nodes has admissible children is a structurally corrupt e-graph
/// (every well-formed class holds a creation-order witness node).
pub(crate) fn repair_choices_well_founded(
    egraph: &EGraph,
    root: EClassId,
    choices: &mut [Option<usize>],
) {
    let num_classes = choices.len();

    // Scope: every canonical class reachable from `root` through ANY node —
    // the repair may switch a class to a node whose children the recorded
    // graph never visited, so the full downward closure is the safe scope.
    let mut scope_pos: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut scope: Vec<u32> = Vec::new();
    let mut stack = alloc::vec![egraph.find(root)];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || scope_pos[idx].is_some() {
            continue;
        }
        scope_pos[idx] = Some(scope.len());
        scope.push(canonical.0);
        for node in egraph.nodes(canonical) {
            {
                let children = (node).children_slice();
                for &child in children {
                    stack.push(egraph.find(child));
                }
            }
        }
    }
    if scope.is_empty() {
        return;
    }

    // Pending child-admissions per (scope class, node); reverse edges say
    // which counters an admission decrements. Duplicate children count per
    // occurrence.
    let mut pending: Vec<Vec<usize>> = Vec::with_capacity(scope.len());
    let mut reverse: Vec<Vec<(usize, usize)>> = alloc::vec![Vec::new(); scope.len()];
    for (pos, &cid) in scope.iter().enumerate() {
        let nodes = egraph.nodes(EClassId(cid));
        let mut per_node = Vec::with_capacity(nodes.len());
        for (node_idx, node) in nodes.iter().enumerate() {
            let mut count = 0usize;
            {
                let children = (node).children_slice();
                for &child in children {
                    let child_pos = scope_pos[egraph.find(child).0 as usize]
                        .expect("child of a scope class is in scope");
                    count += 1;
                    reverse[child_pos].push((pos, node_idx));
                }
            }
            per_node.push(count);
        }
        pending.push(per_node);
    }

    let mut admitted: Vec<bool> = alloc::vec![false; scope.len()];
    let mut admitted_count = 0usize;
    // Classes whose RECORDED node is ready (drain phase pulls from here).
    let mut recorded_ready: Vec<usize> = Vec::new();
    // Classes with ANY ready node (unstick phase pulls from here).
    let mut any_ready: Vec<usize> = Vec::new();
    for (pos, &cid) in scope.iter().enumerate() {
        let recorded = choices[cid as usize];
        for (node_idx, &count) in pending[pos].iter().enumerate() {
            if count == 0 {
                if recorded == Some(node_idx) {
                    recorded_ready.push(pos);
                }
                any_ready.push(pos);
            }
        }
    }

    // Admit `pos` through `node_idx`, propagating readiness to parents.
    let admit = |pos: usize,
                 node_idx: usize,
                 admitted: &mut Vec<bool>,
                 admitted_count: &mut usize,
                 recorded_ready: &mut Vec<usize>,
                 any_ready: &mut Vec<usize>,
                 pending: &mut Vec<Vec<usize>>,
                 choices: &mut [Option<usize>]| {
        admitted[pos] = true;
        *admitted_count += 1;
        choices[scope[pos] as usize] = Some(node_idx);
        for &(parent_pos, parent_node) in &reverse[pos] {
            pending[parent_pos][parent_node] -= 1;
            if pending[parent_pos][parent_node] == 0 && !admitted[parent_pos] {
                if choices[scope[parent_pos] as usize] == Some(parent_node) {
                    recorded_ready.push(parent_pos);
                }
                any_ready.push(parent_pos);
            }
        }
    };

    while admitted_count < scope.len() {
        // Phase 1 — drain: keep recorded choices wherever they admit.
        let mut progressed = false;
        while let Some(pos) = recorded_ready.pop() {
            if admitted[pos] {
                continue;
            }
            let node_idx = choices[scope[pos] as usize]
                .expect("recorded_ready holds only classes with a recorded choice");
            admit(
                pos,
                node_idx,
                &mut admitted,
                &mut admitted_count,
                &mut recorded_ready,
                &mut any_ready,
                &mut pending,
                choices,
            );
            progressed = true;
        }
        if admitted_count == scope.len() {
            break;
        }
        // Phase 2 — unstick: everything left sits on or behind a cycle of
        // recorded choices. Rewrite ONE class to its first admissible node.
        while let Some(pos) = any_ready.pop() {
            if admitted[pos] {
                continue;
            }
            let node_idx = pending[pos]
                .iter()
                .position(|&count| count == 0)
                .expect("any_ready holds only classes with a zero-pending node");
            admit(
                pos,
                node_idx,
                &mut admitted,
                &mut admitted_count,
                &mut recorded_ready,
                &mut any_ready,
                &mut pending,
                choices,
            );
            progressed = true;
            break;
        }
        assert!(
            progressed,
            "repair_choices_well_founded: {} of {} reachable e-classes cannot be admitted — \
             the e-graph holds classes none of whose nodes has admissible children, which is \
             structural corruption, not a rewrite outcome (root {})",
            scope.len() - admitted_count,
            scope.len(),
            root.0
        );
    }
}

/// Extract the minimum-cost arena expression from an e-class.
///
/// Uses dynamic programming: cost(class) = min over all nodes in class.
///
/// The third return value is the **tree** cost of the arena returned
/// alongside it — every child summed at every use, sharing never priced,
/// which is the objective the DP minimizes. The arena itself is a DAG and
/// the emitted kernel pays each distinct node once; for that number use
/// [`extract_dag`] and read [`ExtractedDAG::dag_cost`].
///
/// # Type Parameter
///
/// The cost function can be any type implementing `CostFunction`:
/// - `CostModel` for hardcoded costs
/// - Custom domain-specific cost functions
pub fn extract<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, usize) {
    // Cap for cycle/self-referential costs - high but not astronomical
    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Iterative bottom-up cost computation using topological order
    // We use a work stack to avoid recursion
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)]; // (class, children_processed)
    let mut on_stack: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        // Already computed
        if best_cost[canonical.0 as usize].is_some() {
            continue;
        }

        if !children_done {
            // First visit: push self back (to process after children), then push children
            if on_stack[canonical.0 as usize] {
                // Cycle detected - don't cache, parent will handle with high cost
                continue;
            }
            on_stack[canonical.0 as usize] = true;

            stack.push((canonical, true)); // Come back after children

            // Push all children that need processing
            for node in egraph.nodes(canonical) {
                {
                    let children = (node).children_slice();
                    for &child in children {
                        let child_canonical = egraph.find(child);
                        if best_cost[child_canonical.0 as usize].is_none() {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            // Second visit: all children are computed, now compute this class
            on_stack[canonical.0 as usize] = false;

            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let this_node_cost = match node {
                    ENode::Var(_)
                    | ENode::Const(_)
                    | ENode::Buffer(_)
                    | ENode::Uniform(_)
                    | ENode::Param(_) => costs.node_cost(node, None),
                    // A fold is, for costing, a node with one child: its
                    // metadata is not an operand, so `children_slice` is the
                    // whole of what this arm needs to know about either.
                    ENode::Op { .. } | ENode::Reduce { .. } => {
                        let children = node.children_slice();
                        // Check for self-referential children
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
                            // Saturating fold, not `.sum()`: a child's own
                            // `best_cost` can already sit at a prohibitive
                            // sentinel (`Dwrt`'s `usize::MAX / 4` from
                            // `CostModel::node_op_cost`, or this function's
                            // own `CYCLE_COST`), so a node with several such
                            // children overflows a plain `usize` sum. A real
                            // `Dwrt`-bearing e-graph reaches that here.
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                                })
                                .fold(0usize, usize::saturating_add);
                            op_cost.saturating_add(children_cost)
                        }
                    }
                };

                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                }
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    // Seals `best_node` into an `Extraction`, repairing any mutual cycles
    // the DP recorded before the tree is built.
    let extraction = Extraction::from_dp(egraph, root, best_node);
    // Costed from the repaired choices, so the returned number is the cost of
    // the returned arena. Reading `best_cost[root]` here (as this did before
    // #1111) reports the pre-repair DP total, which names a different term
    // whenever the repair rewrote a pick.
    let total_cost = cost_of_choices(
        egraph,
        root,
        extraction.choices(),
        costs,
        LatticeShape::POINT,
    )
    .tree;
    let (arena, root_id) = choices_to_arena(&extraction);
    (arena, root_id, total_cost)
}

// ============================================================================
// DAG-Aware Reference Counting (for NNUE extraction)
// ============================================================================

/// Count how many times each canonical e-class is referenced by the current
/// extraction choices, walking from `root`.
///
/// A class with `ref_count > 1` is referenced by multiple parents and should
/// be treated as shared (let-bound) in the DAG. The function uses `expanded`
/// tracking so each e-class is recursed into only once, but its count is
/// incremented every time it is referenced.
///
/// Returns a `Vec<u32>` indexed by canonical e-class ID.
pub fn compute_ref_counts(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> Vec<u32> {
    let num_classes = egraph.num_classes();
    let mut counts: Vec<u32> = alloc::vec![0u32; num_classes];
    let mut expanded: Vec<bool> = alloc::vec![false; num_classes];
    let mut stack: Vec<EClassId> = alloc::vec![root];

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes {
            continue;
        }

        counts[idx] += 1;

        // Only recurse into children on first visit (DAG, not tree).
        if !expanded[idx] {
            expanded[idx] = true;
            if let Some(node_idx) = choices[idx] {
                let nodes = egraph.nodes(canonical);
                if node_idx < nodes.len() {
                    {
                        let children = (&nodes[node_idx]).children_slice();
                        for &child in children {
                            stack.push(child);
                        }
                    }
                }
            }
        }
    }

    counts
}

/// Build an `ExtractedDAG` from extraction choices + reference counts.
///
/// Bridges an extractor that produces per-e-class choices with DAG codegen
/// (which needs `ExtractedDAG`'s sharing info for let-bindings).
///
/// `cost` is passed in rather than recomputed because this function has no
/// cost model: the caller that made these choices is the one that knows what
/// they were priced against. It must be [`cost_of_choices`] of *these*
/// choices — [`Optimized::cost`](super::Optimized::cost) is exactly that.
/// Fabricating a zero here (as this did before #1111) puts a number in
/// [`ExtractedDAG::total_cost`] that describes nothing.
pub fn build_extracted_dag_from_choices(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
    ref_counts: &[u32],
    cost: ChoiceCost,
) -> ExtractedDAG {
    let canonical_root = egraph.find(root);

    // Shared e-classes: ref_count > 1
    let shared: Vec<(EClassId, usize)> = ref_counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 1)
        .map(|(i, c)| (EClassId(i as u32), *c as usize))
        .collect();

    // Topological schedule: shared classes before their dependents (post-order).
    let mut schedule = Vec::new();
    let mut visited = alloc::vec![false; egraph.num_classes()];

    fn topo_walk(
        egraph: &EGraph,
        class: EClassId,
        choices: &[Option<usize>],
        ref_counts: &[u32],
        visited: &mut Vec<bool>,
        schedule: &mut Vec<EClassId>,
    ) {
        let canonical = egraph.find(class);
        let idx = canonical.index();
        if idx >= visited.len() || visited[idx] {
            return;
        }
        visited[idx] = true;

        if let Some(node_idx) = choices.get(idx).copied().flatten() {
            if let Some(node) = egraph.nodes(canonical).get(node_idx) {
                {
                    let children = (node).children_slice();
                    for &child in children {
                        topo_walk(egraph, child, choices, ref_counts, visited, schedule);
                    }
                }
            }
        }

        if ref_counts.get(idx).copied().unwrap_or(0) > 1 {
            schedule.push(canonical);
        }
    }

    topo_walk(
        egraph,
        root,
        choices,
        ref_counts,
        &mut visited,
        &mut schedule,
    );

    ExtractedDAG {
        root: canonical_root,
        shared,
        schedule,
        choices: choices.to_vec(),
        total_cost: cost.tree,
        dag_cost: cost.dag,
        report: ExtractionReport::external(),
    }
}

// ============================================================================
// Arena-Direct Extraction (EGraph → ExprArena)
// ============================================================================

/// Walk extraction choices and materialise directly into an [`pixelflow_ir::ExprArena`].
///
/// Each reachable e-class maps to exactly one [`pixelflow_ir::ExprId`]. Shared
/// e-classes naturally share `ExprId`s (DAG output — nodes are not duplicated).
///
/// ## Algorithm
///
/// Iterative post-order traversal with a `Vec<Option<ExprId>>` cache indexed by
/// canonical e-class id:
///
/// - If an e-class already has a cached `ExprId`, reuse it (O(1), `ExprId` is `Copy`).
/// - Otherwise push children for visiting (in reverse so they are processed
///   left-to-right), then push a `Complete` task for the current e-class.
/// - On `Complete`: pop the children `ExprId`s from the result stack, push a new
///   node into the arena, and record the `ExprId` in the cache.
///
/// Post-order guarantees nodes are appended in topological order (children before
/// parents), which is a requirement of [`pixelflow_ir::ExprArena`].
/// Re-pin every `Shl`/`Shr` count child to a `Const` representative.
///
/// The emitter lowers shifts to hardware immediates, so the count child MUST
/// extract as a `Const`. But a count's e-class can legitimately hold
/// arithmetic as well — a reachable `4 + 4` folds into the same class as `8`
/// — and extraction picks by COST, so a cost model that prices the `Add`
/// lower (a learned one, or any future retuning) hands codegen a non-constant
/// child and it panics. Substituting a `Const` from the same class is sound
/// by definition: same class means equal value.
///
/// Scoped to classes reachable from `root` via the ORIGINAL (unpinned)
/// `choices` — the same traversal [`choices_to_arena`] performs once pinning
/// has settled. `choices` can
/// (and, on a graph whose choices were built up by several backfill
/// passes, routinely does) hold `Some` entries for classes no longer
/// reachable from `root` under the CURRENT choice function — a backfill
/// only ever adds entries, never retracts a stale one from an earlier
/// candidate. Walking `0..egraph.num_classes()`
/// unconditionally, as this used to, re-derives and re-pins every one of
/// those stale entries even though nothing downstream ever reads them
/// (`choices_to_arena` only ever visits classes reachable
/// from `root`) — pure wasted work on a saturated e-graph's full class
/// count, not the reachable subtree's.
///
/// Using unpinned `choices` (rather than the pins already decided so far in
/// this same walk) to decide which children to descend into is deliberately
/// a superset of the classes [`choices_to_arena`] will actually visit once
/// pinning is final: pinning a count class to a `Const` can only ever REMOVE
/// reachability (a `Const` has no children to recurse into), never add it,
/// so this walk's reachable set is never missing a class the final pinned
/// tree needs. The decision written into `pinned[ci]` for any one count
/// class does not depend on visitation order either — it is "keep the
/// existing choice if already `Const`, else the class's first `Const`
/// node", the same answer no matter which of possibly several referencing
/// `Shl`/`Shr` nodes is processed first — so interleaving the decision with
/// the traversal (instead of a full resolve-then-walk pass) cannot change
/// the returned vector's values, only which unreachable entries are left
/// untouched (they pass through from `choices` unread either way).
///
/// # Panics
///
/// Panics if a shift-count class holds no `Const` at all. That cannot arise
/// from a well-formed arena (the count entered as a literal) and would panic
/// in the emitter regardless — failing here names the real cause.
fn pin_shift_counts(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> alloc::vec::Vec<Option<usize>> {
    let num_classes = choices.len();
    let mut pinned = choices.to_vec();
    let mut visited: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];
    let mut stack: Vec<EClassId> = alloc::vec![egraph.find(root)];

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || visited[idx] {
            continue;
        }
        visited[idx] = true;

        // Deliberately reads the ORIGINAL (unpinned) choice, not `pinned`,
        // to decide what this class's traversal children are — see the
        // "superset" reasoning in the doc comment above.
        let Some(node_idx) = choices.get(idx).and_then(|o| *o) else {
            continue;
        };
        let Some(ENode::Op { op, children }) = egraph.nodes(canonical).get(node_idx) else {
            continue;
        };
        if matches!(
            op.kind(),
            pixelflow_ir::OpKind::Shl | pixelflow_ir::OpKind::Shr
        ) && let Some(&count) = children.get(1)
        {
            let count_class = egraph.find(count);
            let ci = count_class.0 as usize;
            let count_nodes = egraph.nodes(count_class);
            let already_const = pinned
                .get(ci)
                .and_then(|o| *o)
                .is_some_and(|chosen| matches!(count_nodes.get(chosen), Some(ENode::Const(_))));
            if !already_const {
                let const_idx = count_nodes
                    .iter()
                    .position(|n| matches!(n, ENode::Const(_)))
                    .unwrap_or_else(|| {
                        panic!(
                            "pin_shift_counts: shift-count e-class {ci} holds no Const; \
                             the emitter's immediate-only shift lowering cannot be met"
                        )
                    });
                pinned[ci] = Some(const_idx);
            }
        }

        for &child in children {
            stack.push(child);
        }
    }

    pinned
}

pub fn choices_to_arena(
    extraction: &Extraction<'_>,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    use pixelflow_ir::{Children, ExprArena, ExprId, Ir, Shape};

    let egraph = extraction.egraph();
    let root = extraction.root();

    // Shifts must reach codegen with a constant count — see
    // `Extraction::pinned_choices` / `pin_shift_counts`.
    let pinned = extraction.pinned_choices();
    let choices: &[Option<usize>] = &pinned;

    enum Task {
        /// Visit an e-class: push it to the result stack if cached, otherwise
        /// schedule children + a Complete task.
        Visit(EClassId),
        /// All children of this e-class have been processed; pop their ExprIds,
        /// push a new arena node, and cache the result.
        Complete { canonical_id: u32, node_idx: usize },
    }

    let num_classes = egraph.num_classes();
    let mut arena = ExprArena::with_capacity(num_classes);
    // Cache: canonical e-class id → ExprId (None = not yet visited).
    let mut id_map: Vec<Option<ExprId>> = alloc::vec![None; num_classes];
    // DFS color per canonical class: 0 = unvisited, 1 = on the current path
    // (children scheduled, Complete pending). Re-entering a gray class means
    // the choice graph reaches a class through its own descendants — a CYCLE.
    // Without this check the walk re-schedules the cycle forever and the
    // process dies by OOM instead of an error (observed: a full-DEV bench run
    // SIGKILLed at 2.7GB inside this loop). A cyclic choice set is an
    // extractor bug and must be reported as one, loudly, with the class id.
    let mut color: Vec<u8> = alloc::vec![0; num_classes];
    let mut result_stack: Vec<ExprId> = Vec::new();
    let mut task_stack: Vec<Task> = alloc::vec![Task::Visit(root)];

    while let Some(task) = task_stack.pop() {
        match task {
            Task::Visit(class) => {
                let canonical = egraph.find(class);
                let idx = canonical.0 as usize;

                // Already materialised — reuse without any clone (ExprId is Copy).
                if let Some(cached_id) = id_map.get(idx).and_then(|o| *o) {
                    result_stack.push(cached_id);
                    continue;
                }

                // No recorded choice for a reachable e-class means the extractor
                // that produced `choices` violated the invariant that every class
                // reachable from `root` (via chosen nodes) has an entry — e.g. a
                // saturation-introduced child that wasn't transitively backfilled.
                // Silently materialising node 0 here would paper over that bug by
                // emitting a node that may not even be the reachable/consistent
                // variant. Panic loudly instead so the extractor bug gets fixed
                // at the source rather than surfacing as a subtly wrong kernel.
                let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
                    panic!(
                        "choices_to_arena: e-class {} is reachable from root {} but has \
                         no recorded extraction choice — the extractor that produced \
                         `choices` must guarantee every reachable e-class has Some(idx)",
                        idx, root.0
                    )
                });

                let nodes = egraph.nodes(canonical);
                assert!(
                    node_idx < nodes.len(),
                    "choices_to_arena: node_idx {} out of bounds ({}) for e-class {}",
                    node_idx,
                    nodes.len(),
                    idx
                );
                let node = &nodes[node_idx];

                match node {
                    ENode::Var(var_idx) => {
                        let expr_id = arena.embed(Shape::Var(*var_idx));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Const(bits) => {
                        let expr_id = arena.embed(Shape::Const(f32::from_bits(*bits)));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Buffer(decl) => {
                        // One slot per distinct identity, and the assertion
                        // that a repeat identity agrees on extents, both live
                        // in `ExprArena`'s `embed`: declaring a buffer is what
                        // the destination representation does, not what the
                        // walk over the e-graph does.
                        let expr_id = arena.embed(Shape::Buffer(*decl));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Uniform(decl) => {
                        // Same rule: the decl (identity and default) is
                        // redeclared by the destination arena, one slot per
                        // identity.
                        let expr_id = arena.embed(Shape::Uniform(*decl));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Param(i) => {
                        // The slot index is the whole node; it means the same
                        // thing in the destination arena, and the builder
                        // substitutes it there.
                        let expr_id = arena.embed(Shape::Param(*i));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Op { .. } | ENode::Reduce { .. } => {
                        let children = node.children_slice();
                        assert!(
                            color[idx] != 1,
                            "choices_to_arena: extraction choices are CYCLIC — e-class {} is \
                             reached again through its own chosen descendants (root {}). The \
                             extractor that produced these choices must guarantee a \
                             well-founded choice DAG; materializing this one would loop until \
                             the process is OOM-killed",
                            idx,
                            root.0
                        );
                        color[idx] = 1;
                        // Schedule completion after children are processed.
                        task_stack.push(Task::Complete {
                            canonical_id: canonical.0,
                            node_idx,
                        });
                        // Push children in reverse so they are popped left-to-right.
                        for &child in children.iter().rev() {
                            task_stack.push(Task::Visit(child));
                        }
                    }
                }
            }

            Task::Complete {
                canonical_id,
                node_idx,
            } => {
                let idx = canonical_id as usize;

                // Another branch may have filled the cache between scheduling this
                // Complete and executing it (diamond sharing). Reuse if so.
                if let Some(cached_id) = id_map.get(idx).and_then(|o| *o) {
                    result_stack.push(cached_id);
                    continue;
                }

                let canonical = EClassId(canonical_id);
                let nodes = egraph.nodes(canonical);
                let node = &nodes[node_idx];

                if let ENode::Reduce { fold, .. } = node {
                    let body = result_stack
                        .pop()
                        .expect("choices_to_arena: a fold's body is built before it");
                    let expr_id = arena.embed(Shape::Reduce { fold: *fold, body });
                    if idx < id_map.len() {
                        id_map[idx] = Some(expr_id);
                    }
                    result_stack.push(expr_id);
                    continue;
                }
                let ENode::Op { op, children } = node else {
                    // Leaves are handled in Visit; reaching here would be a bug.
                    panic!(
                        "choices_to_arena: Complete task for non-Op node (e-class {})",
                        canonical_id
                    );
                };

                let arity = children.len();
                let start = result_stack.len().checked_sub(arity).unwrap_or_else(|| {
                    panic!(
                        "choices_to_arena: result_stack underflow (arity={}, len={}, e-class={})",
                        arity,
                        result_stack.len(),
                        canonical_id
                    )
                });
                let child_ids: Vec<pixelflow_ir::ExprId> = result_stack.drain(start..).collect();

                // Arity dispatch is the destination's business: `embed` picks
                // the unary/binary/ternary/n-ary node. A zero-arity `Op` is
                // malformed and `embed` says so — where this used to
                // materialise it as the constant 0, which is a wrong answer
                // wearing a right answer's clothes.
                let expr_id = arena.embed(Shape::Op(op.kind(), Children::Many(&child_ids)));

                if idx < id_map.len() {
                    id_map[idx] = Some(expr_id);
                }
                result_stack.push(expr_id);
            }
        }
    }

    let root_id = result_stack
        .pop()
        .unwrap_or_else(|| panic!("choices_to_arena: empty result stack after traversal"));
    (arena, root_id)
}

// ============================================================================
// DAG-Aware Extraction
// ============================================================================

/// Result of DAG-aware extraction with sharing information.
///
/// Unlike regular extraction which produces a tree, this tracks:
/// - Which e-classes are used multiple times (candidates for let-binding)
/// - The topological order for emission (dependencies first)
/// - The best node choice per e-class
///
/// # Example
///
/// For `sin(X) * sin(X) + sin(X)`:
/// - E-class containing `sin(X)` is used 3 times
/// - DAG extraction identifies this for let-binding
/// - Codegen emits: `let __0 = X.sin().eval(__p); (__0 * __0 + __0).eval(__p)`
#[derive(Clone, Debug)]
pub struct ExtractedDAG {
    /// The root e-class of the expression.
    pub root: EClassId,

    /// E-classes used more than once: (class_id, use_count).
    /// These are candidates for let-binding in codegen.
    pub shared: Vec<(EClassId, usize)>,

    /// Topological order for emission (dependencies before dependents).
    /// Shared e-classes appear before e-classes that use them.
    pub schedule: Vec<EClassId>,

    /// Best node choice per e-class (indexed by canonical e-class ID).
    pub choices: Vec<Option<usize>>,

    /// **Tree** cost of the term in [`Self::choices`]: every child summed at
    /// every use, so sharing is never priced. This is the objective the
    /// extraction DP minimizes, and it is *not* what the emitted kernel pays
    /// — see [`Self::dag_cost`], which is the number a caller asking "what
    /// will this kernel cost?" wants. On `shader:julia_set` the two are
    /// ~1.4e7 and 716 (`docs/results/2026-09-02-extraction-gap.md`).
    ///
    /// Read from the *repaired* choices, so it describes the term this
    /// struct returns (#1111). Before that fix it was the pre-repair DP
    /// total, which named a different term on 132 of 302 measured kernels.
    pub total_cost: usize,

    /// **DAG** cost of the term in [`Self::choices`]: each distinct chosen
    /// e-class priced once, which is what the emitted kernel pays.
    /// [`choices_to_arena`] materializes one arena node per reachable
    /// e-class and codegen let-binds the shared ones, so under
    /// [`LatticeShape::POINT`] this equals the latency-prior cost of that
    /// arena — the property every measurement in this repo assumes when it
    /// re-costs the materialized arena instead of reading a field here.
    ///
    /// The DP does not minimize this (#1116); it is the honest price of what
    /// the DP happened to choose.
    pub dag_cost: usize,

    /// Which objective the term in [`Self::choices`] came from, and what the
    /// sharing-aware pass cost to find out. A number quoted from this struct
    /// without its objective is a number from an unknown extractor.
    pub report: ExtractionReport,
}

/// Which of [`extract_dag_scoped`]'s objectives produced the returned term.
///
/// The two-objective no-regression property (the sharing-aware term is
/// returned only when it is cheaper by true [`ChoiceCost::dag`], else the
/// tree term) holds for [`Self::Shared`] and [`Self::TreeCheaper`], where
/// both objectives ran. It is **not attempted** for [`Self::TreeOnly`]: the
/// sharing-aware pass was abandoned at [`SHARED_DAG_PASS_BYTE_BUDGET`] and
/// the tree term is all there is. That case is loud by construction — it
/// is a variant, not a silently identical `Vec<Option<usize>>` — so a
/// measurement above the budget can never be quoted as if it were on the
/// production objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionObjective {
    /// Both objectives ran; the sharing-aware term was cheaper.
    Shared,
    /// Both objectives ran; the tree term was no dearer (ties go to it).
    TreeCheaper,
    /// Only the tree objective ran: the sharing-aware pass's reach sets
    /// outgrew [`SHARED_DAG_PASS_BYTE_BUDGET`] and it was abandoned.
    TreeOnly,
    /// The choices were supplied from outside the two-objective DP — a
    /// [`Reranker`](super::Reranker), or a caller's own choice map through
    /// [`build_extracted_dag_from_choices`]. Neither objective's pass ran.
    External,
}

impl ExtractionObjective {
    /// The name the telemetry record and the measurement harnesses print.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::TreeCheaper => "tree_cheaper",
            Self::TreeOnly => "tree_only",
            Self::External => "external",
        }
    }
}

/// What [`shared_dag_dp_pass`] cost, whether or not it finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedPassStats {
    /// Classes reachable from the root — the set the pass's reach sets
    /// range over. On a saturated glyph a third of the e-graph.
    pub live_classes: usize,
    /// Bytes the reach sets held when the pass ended: at its completion,
    /// or at the point it crossed [`SHARED_DAG_PASS_BYTE_BUDGET`] and was
    /// abandoned (then the first total above the budget). A deterministic
    /// function of the e-graph, so two hosts report the same number.
    pub reach_bytes: usize,
}

/// The objective behind an [`ExtractedDAG`], with the sharing-aware pass's
/// accounting beside it. Carried by [`ExtractedDAG::report`] and
/// [`Optimized::extraction`](super::Optimized::extraction), and emitted by
/// the `saturation-telemetry` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractionReport {
    /// Which term was returned.
    pub objective: ExtractionObjective,
    /// The sharing-aware pass's accounting; `None` only under
    /// [`ExtractionObjective::External`], where no pass ran.
    pub shared_pass: Option<SharedPassStats>,
    /// What the winning arm's DP believed it had chosen, against the term it
    /// actually named. `None` only under [`ExtractionObjective::External`],
    /// where no DP ran. See [`ClaimAudit`].
    pub audit: Option<ClaimAudit>,
}

impl ExtractionReport {
    /// Choices supplied from outside the DP: no objective, no pass.
    #[must_use]
    pub fn external() -> Self {
        Self {
            objective: ExtractionObjective::External,
            shared_pass: None,
            audit: None,
        }
    }
}

/// Which column of [`ChoiceCost`] a DP arm's own minimized value is on.
///
/// The two arms of [`extract_dag_scoped`] minimize *different quantities* —
/// [`tree_dp_pass`] a tree cost, [`shared_dag_dp_pass`] a DAG cost — and a
/// claim read off one arm's table means nothing beside the other column. The
/// scale is carried rather than inferred from [`ExtractionObjective`] because
/// a number whose units live in a comment is a number something will
/// eventually compare wrongly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostScale {
    /// [`ChoiceCost::tree`] — every child summed at every use.
    Tree,
    /// [`ChoiceCost::dag`] — each distinct chosen e-class priced once.
    Dag,
}

impl CostScale {
    /// The column of `cost` this scale names.
    #[must_use]
    pub fn of(self, cost: ChoiceCost) -> usize {
        match self {
            Self::Tree => cost.tree,
            Self::Dag => cost.dag,
        }
    }

    /// The name the measurement harnesses print.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tree => "tree",
            Self::Dag => "dag",
        }
    }
}

/// The winning DP arm's own minimized value at the root, and the scale it is
/// on — the objective the extractor **minimized**, beside the price the
/// kernel **pays** ([`ExtractedDAG::dag_cost`]).
///
/// Those have to be the same number. `settle_in_cost_order` settles a class
/// strictly after the children of the candidate it settles on, so its map is
/// well-founded, nothing downstream rewrites a pick, and every reach set is
/// final before a parent reads it. When they are not the same number the
/// extractor is minimizing something no one pays — which is what the DFS
/// post-order it replaced did, claiming **281** for a chrome term costing
/// **4,564,003,324** at a 50,000-class cap
/// (`docs/results/2026-09-08-cse-mispricing.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimAudit {
    /// The winning arm's minimized value at the root, read from its own DP.
    pub claimed: usize,
    /// Which of [`ChoiceCost`]'s columns [`Self::claimed`] is comparable to.
    pub scale: CostScale,
}

impl ClaimAudit {
    /// The signed error of the claim against the term actually returned:
    /// `claimed - actual` on [`Self::scale`]'s column. Negative means the DP
    /// believed the term cheaper than it is — the direction in which more
    /// graph buys a worse choice.
    #[must_use]
    pub fn signed_error(&self, cost: ChoiceCost) -> i128 {
        self.claimed as i128 - self.scale.of(cost) as i128
    }
}

impl ExtractedDAG {
    /// Check if an e-class is shared (used more than once).
    pub fn is_shared(&self, class: EClassId) -> bool {
        self.shared.iter().any(|(id, _)| *id == class)
    }

    /// Get the use count for an e-class.
    pub fn use_count(&self, class: EClassId) -> usize {
        self.shared
            .iter()
            .find(|(id, _)| *id == class)
            .map(|(_, count)| *count)
            .unwrap_or(1)
    }

    /// Get the index of the best node for an e-class.
    pub fn best_node_idx(&self, class: EClassId) -> Option<usize> {
        self.choices.get(class.0 as usize).and_then(|o| *o)
    }

    /// The two reported costs as the pair they are.
    #[must_use]
    pub fn cost(&self) -> ChoiceCost {
        ChoiceCost {
            tree: self.total_cost,
            dag: self.dag_cost,
        }
    }
}

/// Extract a DAG with sharing information from an e-class.
///
/// This is the DAG-aware version of `extract()`. It returns structural
/// information about sharing that codegen can use to emit let-bindings.
///
/// # Arguments
///
/// * `egraph` - The e-graph to extract from
/// * `root` - The root e-class
/// * `costs` - The cost function for choosing best nodes
///
/// # Returns
///
/// An `ExtractedDAG` containing:
/// - Best node per e-class
/// - Shared e-classes (for let-binding)
/// - Topological order for emission
/// The variance of one e-node, given the variance already chosen for the
/// classes below it: the union of its children's, with leaves naming their
/// own. A child whose form is not settled yet (a cycle under repair) counts
/// as fully varying — the conservative direction, since it can only make a
/// form look more expensive, never less.
fn node_variance(
    egraph: &EGraph,
    node: &ENode,
    best_var: &[Variance],
    canonical: EClassId,
) -> Variance {
    match node {
        ENode::Var(v) => var_variance(*v),
        // A buffer's contents are fixed for the kernel's lifetime; a read of
        // one varies with its index, which is the `Gather`'s other child.
        ENode::Const(_) | ENode::Buffer(_) | ENode::Uniform(_) | ENode::Param(_) => Variance::CONST,
        ENode::Op { children, .. } => children.iter().fold(Variance::CONST, |acc, &child| {
            let c = egraph.find(child);
            if c == canonical {
                return Variance::ALL;
            }
            acc.union(best_var[c.0 as usize])
        }),
        // The one node that *shrinks* the set. Its index is bound, so it is
        // not free in the result — which is what makes `Σ_i f(i)` frame-
        // uniform when `f` reads nothing but the index, and therefore
        // hoistable out of the pixel loop.
        ENode::Reduce { fold, body } => {
            let c = egraph.find(*body);
            if c == canonical {
                return Variance::ALL;
            }
            best_var[c.0 as usize].without(Variance::from_var(fold.binder().var()))
        }
    }
}

/// The cost of one *settled* extraction, in both of the shapes that matter.
///
/// The two numbers differ by exactly how much sharing the choice function
/// induces, and the gap is not cosmetic: on `shader:julia_set` the tree cost
/// is ~1.4e7 against a DAG cost of 716, a 20,000x sharing ratio
/// (`docs/results/2026-09-02-extraction-gap.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChoiceCost {
    /// Every child summed at every use — **sharing is never priced**, so a
    /// subterm reached ten times is paid ten times. This is the objective
    /// [`extract_dag_scoped`]'s DP minimizes, which is why it is reported:
    /// comparing the DP against a reference means comparing this number.
    pub tree: usize,

    /// Each distinct chosen e-class priced **once** — what the emitted kernel
    /// actually pays, since [`choices_to_arena`] materializes exactly one
    /// arena node per reachable e-class and codegen let-binds the shared ones.
    ///
    /// A caller asking "what will this kernel cost?" wants this number.
    pub dag: usize,
}

/// Cost the term named by `choices`, under `costs`, weighted by `shape`.
///
/// This costs *the choice function it is given* — nothing is minimized here,
/// no node is reconsidered. Pass a settled, well-founded choice map (the
/// output of [`repair_choices_well_founded`], or [`Extraction::choices`]);
/// costing the raw DP table instead is exactly the #1111 bug this exists to
/// close, so a cyclic map panics rather than returning a number for a term
/// that cannot be materialized.
///
/// Weighting matches [`extract_dag_scoped`]: a node's op cost is multiplied
/// by [`LatticeShape::evals`] of the variance of the *chosen* form below it,
/// so a Z-only subexpression is priced once per frame and an X-dependent one
/// once per sample. Leaves are free ([`CostModel::node_op_cost`]), so under
/// [`LatticeShape::POINT`] `ChoiceCost::dag` equals the latency-prior cost of
/// the arena `choices_to_arena` builds from the same map.
///
/// # Panics
///
/// If a reachable e-class has no recorded choice, if a recorded index is out
/// of bounds, or if the choice graph is cyclic — all three are broken
/// invariants of the producing extractor, not recoverable states.
pub fn cost_of_choices<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
    costs: &C,
    shape: LatticeShape,
) -> ChoiceCost {
    let chosen = |canonical: EClassId| -> &ENode {
        let idx = canonical.0 as usize;
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "cost_of_choices: e-class {} is reachable from root {} but has no recorded \
                 choice — cost a settled choice map (post-repair), never a partial one",
                idx, root.0
            )
        });
        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "cost_of_choices: node_idx {} out of bounds ({}) for e-class {}",
            node_idx,
            nodes.len(),
            idx
        );
        &nodes[node_idx]
    };

    let num_classes = egraph.num_classes();
    let mut tree: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut var: Vec<Variance> = alloc::vec![Variance::CONST; num_classes];
    // 0 = unvisited, 1 = on the current path, 2 = costed.
    let mut color: Vec<u8> = alloc::vec![0u8; num_classes];
    let mut dag = 0usize;

    let root_canonical = egraph.find(root);
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root_canonical, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;

        if !children_done {
            if color[idx] == 2 {
                continue;
            }
            assert!(
                color[idx] != 1,
                "cost_of_choices: the choice graph is CYCLIC — e-class {} is reached again \
                 through its own chosen descendants (root {}). A cyclic map names no term, \
                 so it has no cost; repair it before costing it",
                idx,
                root.0
            );
            color[idx] = 1;
            stack.push((canonical, true));
            {
                let children = (chosen(canonical)).children_slice();
                for &child in children {
                    stack.push((child, false));
                }
            }
            continue;
        }

        color[idx] = 2;
        let node = chosen(canonical);
        let node_var = node_variance(egraph, node, &var, canonical);
        let weight = shape.evals(node_var);
        // Saturating throughout: `Dwrt` is priced `usize::MAX / 4` and a tree
        // cost is exponential in the sharing it refuses to price, so both
        // sums reach the ceiling on real inputs.
        let own = usize::try_from((costs.node_cost(node, None) as u64).saturating_mul(weight))
            .unwrap_or(usize::MAX);
        let children_cost = node
            .children_slice()
            .iter()
            .map(|&child| {
                let c = egraph.find(child).0 as usize;
                tree[c].expect("post-order visits every child before its parent")
            })
            .fold(0usize, usize::saturating_add);
        tree[idx] = Some(own.saturating_add(children_cost));
        var[idx] = node_var;
        dag = dag.saturating_add(own);
    }

    ChoiceCost {
        tree: tree[root_canonical.0 as usize]
            .expect("the root is costed by the walk that starts at it"),
        dag,
    }
}

pub fn extract_dag<C: CostFunction>(egraph: &EGraph, root: EClassId, costs: &C) -> ExtractedDAG {
    extract_dag_scoped(egraph, root, costs, LatticeShape::POINT)
}

/// [`extract_dag`], pricing each node by how often the lattice evaluates it.
///
/// The cost of a program is not the cost of its text but of its execution:
/// a node's op cost is multiplied by [`LatticeShape::evals`] of the variance
/// of the form chosen for it, so a subexpression that depends only on Z is
/// priced once per frame while one that touches X is priced once per sample.
/// Every extent is known at compile time, so this is the exact instruction
/// count of the unrolled program rather than an ordinal preference.
///
/// That single change is what makes extraction the thing that *decides* the
/// factorization: given `(X + Z) + Z` and its reassociation `X + (Z + Z)`,
/// both two adds, the second leaves one of them outside the pixel loop and
/// is therefore cheaper by a factor of the frame — which loop-invariant code
/// motion after the fact can only discover, never choose between.
///
/// [`LatticeShape::POINT`] weights everything by one, so `extract_dag`'s
/// behavior is unchanged.
///
/// # The objective (#1116)
///
/// The DP this used to be summed each child's `best_cost`, which is a
/// **tree** cost: a class used ten times was charged ten times in the
/// objective and emitted once in the kernel, so the thing minimized was not
/// the thing paid. On `shader:julia_set` the two numbers were ~1.4e7 and 716.
///
/// The fix is to carry, alongside each class's cost, **the set of classes
/// its chosen sub-DAG contains**, and to price that set — each member once.
/// A parent unions its children's sets, so a class two siblings both reach is
/// paid for once, and `Mul(a, a)` pays for `a` once. At every class the number
/// being minimized is then the true DAG cost of the sub-DAG rooted there, and
/// at the root it is exactly [`ExtractedDAG::dag_cost`]: the objective and the
/// price are the same quantity. See [`shared_dag_dp_pass`].
///
/// It remains an approximation of the extraction optimum, which is NP-hard —
/// the choice is still made greedily bottom-up, so a locally dear class that
/// would have paid for itself upstream is still passed over. What it is not is
/// an approximation of the *objective*: nothing here charges for sharing.
///
/// Both objectives are run and **the cheaper term by true `dag_cost` wins**,
/// ties going to the tree pass. So the returned DAG cost can only be lower
/// than the pre-#1116 extractor's, never higher — no-regression is structural
/// rather than empirical, at the price of a second DP pass over a graph
/// extraction walks once per compile. The one exception is loud: the shared
/// pass holds its reach sets under [`SHARED_DAG_PASS_BYTE_BUDGET`], and a
/// graph whose sets outgrow it gets the tree term with
/// [`ExtractionObjective::TreeOnly`] in [`ExtractedDAG::report`] — never
/// the same `Vec` under a different objective.
pub fn extract_dag_scoped<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let tree = costed(
        egraph,
        root,
        tree_dp_pass(egraph, root, &mut Dp::production(costs, shape)),
        CostScale::Tree,
        costs,
        shape,
    );
    let pass = shared_dag_dp_pass(
        egraph,
        root,
        &mut Dp::production(costs, shape),
        SHARED_DAG_PASS_BYTE_BUDGET,
    );
    let stats = Some(pass.stats);
    let Some(dp) = pass.outcome else {
        return assemble(egraph, root, tree, ExtractionObjective::TreeOnly, stats);
    };
    let shared = costed(egraph, root, dp, CostScale::Dag, costs, shape);
    // The comparison is between two *re-costed* terms, never between the two
    // DPs' own tables: those are on different scales (`CostScale`). Both sides
    // here are `cost_of_choices` of a settled map under the same `costs` and
    // the same `shape`, so the min is like against like.
    //
    // Only the winner is assembled: the reference counts and the emission
    // schedule describe a term, and one of these two is not going to be one.
    if shared.cost.dag < tree.cost.dag {
        return assemble(egraph, root, shared, ExtractionObjective::Shared, stats);
    }
    assemble(egraph, root, tree, ExtractionObjective::TreeCheaper, stats)
}

/// The most memory [`shared_dag_dp_pass`] may hold in reach sets before it
/// gives up and [`extract_dag_scoped`] returns the tree term as
/// [`ExtractionObjective::TreeOnly`].
///
/// The pass's memory is the sum of its reach sets, each held in whichever
/// form is smaller (see [`Reach`]), so the worst case is the dense bound
/// `live_classes² / 8` bytes — a chain, where every class reaches every
/// class below it — and real kernels sit far under it: the reach sets are
/// the sub-DAGs of the chosen terms, and most of a saturated glyph's live
/// classes are variants deep inside one Bézier segment with a sub-DAG of a
/// few hundred classes. Calibrated on the 2026-09-08 class-cap sweep
/// (`docs/results/2026-09-08-class-cap-sweep.md`): the number is set so
/// that no DEV kernel at the shipped `classical` cap comes near it and the
/// dense worst case is still bounded to a size a glyph bake can hold
/// transiently — the sets are allocated once per extraction and dropped at
/// its end. A budget in bytes rather than classes because bytes are what
/// the gate protects, and the class count was a proxy that fired on the
/// whole e-graph while the pass was sized by the third of it the root
/// reaches.
pub const SHARED_DAG_PASS_BYTE_BUDGET: usize = 256 << 20;

/// A settled choice map, the cost of the term it names, and what the DP that
/// produced it claimed that term was worth.
struct CostedChoices {
    choices: Vec<Option<usize>>,
    cost: ChoiceCost,
    audit: ClaimAudit,
}

/// The two terms [`extract_dag_scoped`] chooses between: `(tree, shared)`.
///
/// `tree` is the pre-#1116 extractor, arithmetic for arithmetic, and exists
/// so the A/B that justifies the sharing-aware objective can be run against
/// the thing it replaced rather than against a remembered number.
pub(crate) fn extract_dag_objectives<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> (ExtractedDAG, ExtractedDAG) {
    (
        extract_dag_tree_arm(egraph, root, costs, shape),
        extract_dag_shared_arm(egraph, root, costs, shape),
    )
}

/// The tree-cost arm on its own — the pre-#1116 extractor, and the control
/// the objective A/B and its cost measurement are run against.
pub(crate) fn extract_dag_tree_arm<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let term = costed(
        egraph,
        root,
        tree_dp_pass(egraph, root, &mut Dp::production(costs, shape)),
        CostScale::Tree,
        costs,
        shape,
    );
    // `shared_pass: None` — the pass was not run, as opposed to run and
    // abandoned, which is what production's `TreeOnly` carries.
    assemble(egraph, root, term, ExtractionObjective::TreeOnly, None)
}

/// The sharing-aware arm on its own (#1116). Runs the pass to completion
/// whatever it costs — an A/B that silently swapped its arm for the tree
/// term above some size would be measuring nothing — so a graph that would
/// exceed [`SHARED_DAG_PASS_BYTE_BUDGET`] in production is the caller's
/// memory to spend here.
pub(crate) fn extract_dag_shared_arm<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let pass = shared_dag_dp_pass(egraph, root, &mut Dp::production(costs, shape), usize::MAX);
    let choices = pass
        .outcome
        .expect("an unbounded shared pass cannot run out of budget");
    let term = costed(egraph, root, choices, CostScale::Dag, costs, shape);
    assemble(
        egraph,
        root,
        term,
        ExtractionObjective::Shared,
        Some(pass.stats),
    )
}

/// Cost the term a settled DP choice map names.
///
/// There is no repair stage here any more. A map out of
/// [`settle_in_cost_order`] is well-founded by construction — a class is
/// settled strictly after the children of the candidate it settles on — so
/// [`repair_choices_well_founded`] had nothing left to do but relabel
/// classes no term reaches, and every choice it used to make on the DP's
/// behalf was a cost decision taken with no cost model. It survives for
/// [`Extraction::from_dp`], whose input is an arbitrary caller's map.
/// `the_dp_map_is_well_founded_so_the_repair_is_a_no_op` is the gate.
///
/// The cost is of the choices being RETURNED, not of the DP table that
/// produced them — the distinction #1111 had to make when a repair could
/// switch a class under it, kept because it costs one walk.
fn costed<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    dp: DpOutcome,
    scale: CostScale,
    costs: &C,
    shape: LatticeShape,
) -> CostedChoices {
    let DpOutcome { choices, root_cost } = dp;
    let cost = cost_of_choices(egraph, root, &choices, costs, shape);
    let audit = ClaimAudit {
        claimed: root_cost,
        scale,
    };
    // The DP minimizes `claimed`; the caller pays `cost`. `settle_in_cost_order`
    // settles a class strictly after the children of the candidate it settles
    // on, so nothing rewrites a pick and every reach set is final before its
    // parent reads it — they are the same term, so they must be the same
    // number. An extractor whose objective differs from the price of what it
    // returns is optimizing something no one pays, which is exactly what the
    // DFS post-order did (docs/results/2026-09-08-cse-mispricing.md).
    //
    // A `debug_assert` rather than a test: this way every extraction any test
    // in the workspace performs is a self-consistency check, which is the only
    // way to cover corpora this crate cannot name. The defect this closes was
    // found by it firing on core-term's terminal scene.
    debug_assert_eq!(
        audit.claimed,
        audit.scale.of(cost),
        "extraction claim/price mismatch on the {:?} scale: the DP settled the root at \
         {claimed}, but the term its map names costs tree {tree} / dag {dag} — the objective \
         and the price have come apart",
        audit.scale,
        claimed = audit.claimed,
        tree = cost.tree,
        dag = cost.dag,
    );
    CostedChoices {
        choices,
        cost,
        audit,
    }
}

/// Build the sharing and emission schedule around a settled choice map.
fn assemble(
    egraph: &EGraph,
    root: EClassId,
    costed: CostedChoices,
    objective: ExtractionObjective,
    shared_pass: Option<SharedPassStats>,
) -> ExtractedDAG {
    let CostedChoices {
        choices,
        cost,
        audit,
    } = costed;
    let report = ExtractionReport {
        objective,
        shared_pass,
        audit: Some(audit),
    };
    let mut ref_counts: Vec<usize> = alloc::vec![0; egraph.num_classes()];
    count_refs_recursive(egraph, root, &choices, &mut ref_counts);

    let shared: Vec<(EClassId, usize)> = ref_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 1)
        .map(|(idx, count)| (EClassId(idx as u32), *count))
        .collect();

    let schedule = toposort_dag(egraph, root, &choices, &shared);

    ExtractedDAG {
        root: egraph.find(root),
        shared,
        schedule,
        choices,
        total_cost: cost.tree,
        dag_cost: cost.dag,
        report,
    }
}

// Above any weighted cost a real program can reach, so a self-reference never
// looks cheaper than an expensive-but-legitimate form. (A flat 1_000_000 was
// safely above every *unweighted* cost; weighting by a frame's sample count
// clears that by orders of magnitude.)
pub(crate) const CYCLE_COST: usize = usize::MAX / 4;

/// One node's weighted own cost under `shape`.
fn weighted_own<C: CostFunction>(costs: &C, node: &ENode, weight: u64) -> usize {
    usize::try_from((costs.node_cost(node, None) as u64).saturating_mul(weight))
        .unwrap_or(usize::MAX)
}

/// The two policy knobs the DP passes read, plus the trace they write.
///
/// Grouped rather than passed as four more arguments, and monomorphized
/// rather than dispatched: production instantiates it as
/// `Dp<C, Insertion, ()>`, where both `T` and `R` are ZSTs whose methods are
/// empty, so the emitted passes are byte-identical to the ones that took
/// `(costs, shape)` alone. Every knob is a *type*, so a research arm is a
/// second `impl` rather than a mode flag threaded through the inner loop.
pub(crate) struct Dp<'a, C, T, R> {
    costs: &'a C,
    shape: LatticeShape,
    ties: T,
    rec: R,
}

impl<'a, C> Dp<'a, C, Insertion, ()> {
    /// Production's instance: ties to insertion order, nothing recorded.
    pub(crate) fn production(costs: &'a C, shape: LatticeShape) -> Self {
        Self {
            costs,
            shape,
            ties: Insertion,
            rec: (),
        }
    }
}

impl<'a, C, T, R> Dp<'a, C, T, R> {
    /// A research instance: the same DP under another tie-break, writing to
    /// `rec`.
    pub(crate) fn new(costs: &'a C, shape: LatticeShape, ties: T, rec: R) -> Self {
        Self {
            costs,
            shape,
            ties,
            rec,
        }
    }

    /// The trace the passes wrote, once they are done with it.
    pub(crate) fn into_recorder(self) -> R {
        self.rec
    }
}

/// Which of two nodes a class keeps when the DP prices them **equally**.
///
/// The strict `<` the passes have always used answers this implicitly —
/// the earlier index wins, and index is insertion order — which is why the
/// `'8'` bisect saw extraction move under a semantically-null change to the
/// input. Naming the decision makes the alternative a second `impl` instead
/// of a fork of the pass.
pub(crate) trait TieBreak {
    /// At equal cost, does `challenger` displace the `incumbent` node index
    /// in `class`?
    fn prefer(&self, egraph: &EGraph, class: EClassId, incumbent: usize, challenger: usize)
    -> bool;
}

/// Production: never — the first admissible node stands, so the choice is
/// the e-graph's insertion order.
pub(crate) struct Insertion;

impl TieBreak for Insertion {
    #[inline]
    fn prefer(&self, _: &EGraph, _: EClassId, _: usize, _: usize) -> bool {
        false
    }
}

/// The research arm: a total order on the node's own content, so a tie is
/// broken by what the node *is* rather than by when it was inserted.
///
/// Key, ascending: leaf/op tag, then the leaf's payload or the `OpKind`
/// ordinal, then arity, then the canonical child ids. Two distinct nodes in
/// one class always differ somewhere in that key (a class holds no
/// duplicates after `rebuild`), so the order is total and the result is
/// independent of insertion order.
pub(crate) struct Canonical;

/// `node`'s position in [`Canonical`]'s order.
fn canonical_key(egraph: &EGraph, node: &ENode) -> (u8, u64, usize, Vec<u32>) {
    let children: Vec<u32> = node
        .children_slice()
        .iter()
        .map(|&c| egraph.find(c).0)
        .collect();
    match node {
        ENode::Var(i) => (0, u64::from(*i), 0, children),
        ENode::Const(bits) => (1, u64::from(*bits), 0, children),
        ENode::Buffer(_) => (2, 0, 0, children),
        ENode::Uniform(_) => (3, 0, 0, children),
        ENode::Param(i) => (4, u64::from(*i), 0, children),
        ENode::Op { op, .. } => (5, op.kind() as u64, children.len(), children),
        // The fold *is* the discriminating part: two folds over one body
        // differ only in their metadata, so that is what orders them.
        ENode::Reduce { fold, .. } => (6, fold.to_bits(), children.len(), children),
    }
}

impl TieBreak for Canonical {
    fn prefer(
        &self,
        egraph: &EGraph,
        class: EClassId,
        incumbent: usize,
        challenger: usize,
    ) -> bool {
        let nodes = egraph.nodes(class);
        canonical_key(egraph, &nodes[challenger]) < canonical_key(egraph, &nodes[incumbent])
    }
}

/// What a DP pass writes about each candidate it priced, for the research
/// harness that asks *why* a class went the way it did.
///
/// Production's instance is `()`, whose methods are empty and inline away —
/// the pass allocates nothing and branches nowhere for a trace nobody reads.
pub(crate) trait StageRecorder {
    /// One candidate priced: its DP cost and its weighted own cost.
    fn candidate(&mut self, class: EClassId, idx: usize, cost: usize, own: usize);
    /// The candidate the class settled on.
    fn settled(&mut self, class: EClassId, idx: usize);
}

impl StageRecorder for () {
    #[inline]
    fn candidate(&mut self, _: EClassId, _: usize, _: usize, _: usize) {}
    #[inline]
    fn settled(&mut self, _: EClassId, _: usize) {}
}

/// A class is settled when its cheapest **admissible** candidate has every
/// child settled — Knuth's AND-OR generalisation of Dijkstra, and the
/// denotation both DP passes below compute.
///
/// The passes used to walk one DFS post-order, which cannot express that. A
/// class whose child was still on the stack got priced at a sentinel and was
/// never revisited, so on a saturated graph — commutativity alone closes
/// cycles — a large fraction of classes carried no opinion at all, and the
/// pick fell out of [`repair_choices_well_founded`], whose job is
/// well-foundedness, not cost. 29 % of the frontier classes holding the
/// witnesses of `docs/results/2026-09-08-extraction-witnesses.md`, and 73 %
/// of them on the shaders, were decided that way; the sentinel is gone with
/// the traversal that needed it.
///
/// Settling in cost order is exact whenever a candidate costs at least as
/// much as each of its children, which both passes satisfy: the tree pass
/// adds its children's costs to a non-negative own cost, and the shared pass
/// prices the union of its children's reach sets, a superset of each of
/// them. A candidate that mentions its own class is never admissible, and a
/// class none of whose candidates ever becomes admissible is one no
/// well-founded term reaches — [`settle_in_cost_order`] refuses to return
/// with the root in that state rather than inventing a choice for it.
trait Settling {
    /// Price candidate `idx` of `class`. Every child class is settled, and
    /// none of them is `class` itself.
    fn price(&mut self, class: EClassId, idx: usize, node: &ENode) -> usize;

    /// Which of two candidates priced **equally** the class keeps.
    fn prefer(&self, class: EClassId, incumbent: usize, challenger: usize) -> bool;

    /// Settle `class` on candidate `idx`, priced at `cost`. Returning
    /// `false` abandons the pass.
    fn settle(&mut self, class: EClassId, idx: usize, node: &ENode, cost: usize) -> bool;
}

/// A candidate that mentions its own class: no decrement ever takes this
/// counter to zero, so the candidate is never priced.
const NEVER_ADMISSIBLE: u32 = u32::MAX;

/// Compact ids for the live classes — `u32::MAX` for a class the root does
/// not reach.
///
/// The compact ids are what the shared pass indexes its reach sets by: they
/// only ever hold classes the root reaches, and on a saturated glyph that is
/// a third of the e-graph, so sizing them by `num_classes` would pay for the
/// rest of the graph in every union.
fn compact_ids(num_classes: usize, live: &[EClassId]) -> Vec<u32> {
    let mut compact: Vec<u32> = alloc::vec![u32::MAX; num_classes];
    for (i, c) in live.iter().enumerate() {
        compact[c.0 as usize] = i as u32;
    }
    compact
}

/// Price one candidate and let it take its class's incumbent if it is
/// cheaper, or equal and preferred.
fn relax<S: Settling>(
    egraph: &EGraph,
    class: EClassId,
    idx: usize,
    s: &mut S,
    best: &mut [Option<(usize, usize)>],
    heap: &mut BinaryHeap<Reverse<(usize, u32)>>,
) {
    let cost = s.price(class, idx, &egraph.nodes(class)[idx]);
    let takes = match best[class.0 as usize] {
        None => true,
        Some((incumbent_cost, incumbent)) => {
            cost < incumbent_cost || (cost == incumbent_cost && s.prefer(class, incumbent, idx))
        }
    };
    if !takes {
        return;
    }
    best[class.0 as usize] = Some((cost, idx));
    heap.push(Reverse((cost, class.0)));
}

/// Settle every live class in increasing cost order, or abandon.
///
/// `None` is [`Settling::settle`] asking to stop — the shared pass over its
/// memory budget. Otherwise every class some well-founded term reaches has a
/// choice, and the map is acyclic **by construction**: a class is settled
/// strictly after the children of the candidate it settles on, so no repair
/// stage is required to make the result materialisable.
///
/// The [`DpOutcome`] carries the root's settled cost beside the map: that is
/// the value this driver *minimized*, and it is the only number that can be
/// checked against the price of the term the map names.
fn settle_in_cost_order<S: Settling>(
    egraph: &EGraph,
    root: EClassId,
    order: &[EClassId],
    s: &mut S,
) -> Option<DpOutcome> {
    let num_classes = egraph.num_classes();
    let compact = compact_ids(num_classes, order);
    let mut choice: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut settled: Vec<bool> = alloc::vec![false; num_classes];
    // The cheapest candidate priced so far for each class, and its index.
    let mut best: Vec<Option<(usize, usize)>> = alloc::vec![None; num_classes];
    // Unsettled distinct child classes per candidate, and — per live class —
    // the candidates waiting on it.
    let mut waiting: Vec<Vec<u32>> = alloc::vec![Vec::new(); num_classes];
    let mut parents: Vec<Vec<(EClassId, usize)>> = alloc::vec![Vec::new(); order.len()];

    let mut distinct: Vec<EClassId> = Vec::new();
    for &class in order {
        let nodes = egraph.nodes(class);
        let mut per_node: Vec<u32> = Vec::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().enumerate() {
            distinct.clear();
            let mut self_referential = false;
            {
                let children = (node).children_slice();
                for &child in children.iter() {
                    let c = egraph.find(child);
                    if c == class {
                        self_referential = true;
                        break;
                    }
                    if !distinct.contains(&c) {
                        distinct.push(c);
                    }
                }
            }
            if self_referential {
                per_node.push(NEVER_ADMISSIBLE);
                continue;
            }
            for &c in &distinct {
                let ci = compact[c.0 as usize];
                assert!(
                    ci != u32::MAX,
                    "settle_in_cost_order: e-class {} is a child of live class {} but was not \
                     enumerated as live — the two traversals have drifted",
                    c.0,
                    class.0
                );
                parents[ci as usize].push((class, idx));
            }
            per_node.push(distinct.len() as u32);
        }
        waiting[class.0 as usize] = per_node;
    }

    let mut heap: BinaryHeap<Reverse<(usize, u32)>> = BinaryHeap::new();
    for &class in order {
        for idx in 0..waiting[class.0 as usize].len() {
            if waiting[class.0 as usize][idx] == 0 {
                relax(egraph, class, idx, s, &mut best, &mut heap);
            }
        }
    }

    while let Some(Reverse((cost, cid))) = heap.pop() {
        let ci = cid as usize;
        if settled[ci] {
            continue;
        }
        let (incumbent_cost, incumbent) =
            best[ci].expect("a class in the heap has been priced at least once");
        if incumbent_cost != cost {
            // A superseded entry: the class has since been priced cheaper,
            // and that entry is still in the heap.
            continue;
        }
        let class = EClassId(cid);
        if !s.settle(
            class,
            incumbent,
            &egraph.nodes(class)[incumbent],
            incumbent_cost,
        ) {
            return None;
        }
        settled[ci] = true;
        choice[ci] = Some(incumbent);

        // Taken, not borrowed: a class settles once, so nothing reads its
        // parent list again, and the memory goes back as the pass proceeds.
        let ps = core::mem::take(&mut parents[compact[ci] as usize]);
        for (parent, idx) in ps {
            let pi = parent.0 as usize;
            if settled[pi] {
                continue;
            }
            let w = &mut waiting[pi][idx];
            assert!(
                *w != 0 && *w != NEVER_ADMISSIBLE,
                "settle_in_cost_order: candidate {idx} of e-class {} was decremented past its \
                 child count — the parent index and the waiting counts disagree",
                parent.0
            );
            *w -= 1;
            if *w == 0 {
                relax(egraph, parent, idx, s, &mut best, &mut heap);
            }
        }
    }

    assert!(
        settled[egraph.find(root).0 as usize],
        "settle_in_cost_order: root e-class {} has no well-founded term — every candidate of \
         every class it reaches sits behind a cycle, which is structural corruption rather than \
         a rewrite outcome",
        root.0
    );
    let root_cost = best[egraph.find(root).0 as usize]
        .expect("a settled class has been priced")
        .0;
    Some(DpOutcome {
        choices: choice,
        root_cost,
    })
}

/// A DP pass's choice map beside the value its own table holds at the root —
/// the number the pass *minimized*, before anything downstream re-costs the
/// term. Kept together because quoting either without the other is how a
/// claim gets mistaken for a price.
pub(crate) struct DpOutcome {
    pub(crate) choices: Vec<Option<usize>>,
    pub(crate) root_cost: usize,
}

/// The pre-#1116 DP: cheapest node per class where a child costs its whole
/// subtree, at every use. The control arm of the objective A/B, and the
/// floor [`extract_dag_scoped`] never returns worse than.
pub(crate) fn tree_dp_pass<C: CostFunction, T: TieBreak, R: StageRecorder>(
    egraph: &EGraph,
    root: EClassId,
    dp: &mut Dp<'_, C, T, R>,
) -> DpOutcome {
    let order = post_order(egraph, root);
    let num_classes = egraph.num_classes();
    let mut pricer = TreePricer {
        egraph,
        dp,
        cost: alloc::vec![None; num_classes],
        // The variance of the form chosen for each class, which is what its
        // scope — and so its weight — is read from. Carried in the same DP
        // as the cost because the two determine each other: a child's
        // variance sets its parent's weight, and a parent's weight is part
        // of what makes one child's form worth choosing over another's.
        var: alloc::vec![Variance::CONST; num_classes],
    };
    settle_in_cost_order(egraph, root, &order, &mut pricer)
        .expect("the tree pass has no budget and never abandons")
}

/// [`tree_dp_pass`]'s pricing: a candidate costs its own weighted cost plus
/// each child's settled cost, summed at every use.
struct TreePricer<'a, 'c, C, T, R> {
    egraph: &'a EGraph,
    dp: &'a mut Dp<'c, C, T, R>,
    cost: Vec<Option<usize>>,
    var: Vec<Variance>,
}

impl<C: CostFunction, T: TieBreak, R: StageRecorder> Settling for TreePricer<'_, '_, C, T, R> {
    fn price(&mut self, class: EClassId, idx: usize, node: &ENode) -> usize {
        let node_var = node_variance(self.egraph, node, &self.var, class);
        let own = weighted_own(self.dp.costs, node, self.dp.shape.evals(node_var));
        let cost = match node {
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_) => own,
            // Saturating fold, not `.sum()`: a child's own cost can already
            // sit at a prohibitive sentinel (`Dwrt`'s `usize::MAX / 4` from
            // `CostModel::node_op_cost`), so a node with several such
            // children overflows a plain `usize` sum.
            ENode::Op { .. } | ENode::Reduce { .. } => own.saturating_add(
                node.children_slice()
                    .iter()
                    .map(|&child| {
                        self.cost[self.egraph.find(child).0 as usize]
                            .expect("a priced candidate's children are settled")
                    })
                    .fold(0usize, usize::saturating_add),
            ),
        };
        self.dp.rec.candidate(class, idx, cost, own);
        cost
    }

    fn prefer(&self, class: EClassId, incumbent: usize, challenger: usize) -> bool {
        self.dp
            .ties
            .prefer(self.egraph, class, incumbent, challenger)
    }

    fn settle(&mut self, class: EClassId, idx: usize, node: &ENode, cost: usize) -> bool {
        // Read the children's variances before writing this class's, which
        // `node_variance` never consults for an admissible candidate.
        let node_var = node_variance(self.egraph, node, &self.var, class);
        self.cost[class.0 as usize] = Some(cost);
        self.var[class.0 as usize] = node_var;
        self.dp.rec.settled(class, idx);
        true
    }
}

/// The sharing-aware DP (#1116): cheapest node per class where the cost of a
/// candidate is the cost of **the set of classes its sub-DAG contains**, each
/// member priced once.
///
/// Same driver as [`tree_dp_pass`]; the only change is what a candidate
/// costs. Each class carries the set of classes its chosen sub-DAG reaches
/// (a [`Reach`]), and a candidate unions its children's sets, adding a
/// class's own cost the first time that class enters the union. Two siblings
/// that both reach `sin(X)` therefore pay for it once, which is what the
/// emitted kernel does: `choices_to_arena` materializes one node per
/// reachable class and codegen let-binds the shared ones.
///
/// A union is taken by stamping: every member of every child's set is
/// visited once, and a per-class epoch mark says whether it has been seen
/// under this candidate, so the cost of a candidate is the size of its
/// children's sets, not the size of the graph. Memory is the sum of the
/// sets, each in the smaller of its two forms, and is held under `budget`:
/// the first class whose set would carry the total past it ends the pass
/// with [`SharedPassOutcome::choices`] `None`, and the caller returns the
/// tree term as [`ExtractionObjective::TreeOnly`]. The dense form bounds
/// the worst case at `live² / 8` bytes (the 2026-09-08 memory profile's
/// measured quadratic); real kernels hold a small fraction of that because
/// most live classes are variants deep inside one sub-DAG.
pub(crate) fn shared_dag_dp_pass<C: CostFunction, T: TieBreak, R: StageRecorder>(
    egraph: &EGraph,
    root: EClassId,
    dp: &mut Dp<'_, C, T, R>,
    budget: usize,
) -> SharedPassOutcome {
    let order = post_order(egraph, root);
    let live = order.len();
    let mut pricer = SharedPricer {
        egraph,
        dp,
        var: alloc::vec![Variance::CONST; egraph.num_classes()],
        sets: ReachSets {
            compact: compact_ids(egraph.num_classes(), &order),
            reach: (0..live).map(|_| None).collect(),
            own: alloc::vec![0; live],
            stamp: alloc::vec![0; live],
            epoch: 0,
            scratch: Vec::new(),
            words: live.div_ceil(REACH_WORD_BITS),
            bytes: 0,
            budget,
        },
    };
    let outcome = settle_in_cost_order(egraph, root, &order, &mut pricer);
    SharedPassOutcome {
        outcome,
        stats: SharedPassStats {
            live_classes: live,
            reach_bytes: pricer.sets.bytes,
        },
    }
}

/// The reach sets, indexed by compact live id, and the budget they are held
/// under.
struct ReachSets {
    compact: Vec<u32>,
    /// `reach[i]` is the set of classes the chosen sub-DAG at live class `i`
    /// contains, itself included — `None` until the class settles.
    reach: Vec<Option<Reach>>,
    /// The weighted own cost of each live class's chosen node: what a union
    /// pays when that class first enters it.
    own: Vec<usize>,
    /// `stamp[i] == epoch` iff live class `i` has entered the union being
    /// taken for the current candidate. One epoch per candidate; never
    /// cleared, so a union costs its members and nothing else.
    stamp: Vec<usize>,
    epoch: usize,
    scratch: Vec<u32>,
    words: usize,
    bytes: usize,
    budget: usize,
}

impl ReachSets {
    /// Union the children's reach sets into `scratch`, returning what those
    /// classes cost with each member paid once.
    fn union_below(&mut self, egraph: &EGraph, node: &ENode) -> usize {
        self.epoch += 1;
        self.scratch.clear();
        let ENode::Op { children, .. } = node else {
            return 0;
        };
        let mut below = 0usize;
        for &child in children.iter() {
            let ci = self.compact[egraph.find(child).0 as usize] as usize;
            // Taken and put back: the closure below needs `self` mutably
            // while the set is read, and a set is never its own member's.
            let set = self.reach[ci]
                .take()
                .expect("a priced candidate's children are settled");
            set.for_each(|member| {
                let m = member as usize;
                if self.stamp[m] != self.epoch {
                    self.stamp[m] = self.epoch;
                    below = below.saturating_add(self.own[m]);
                    self.scratch.push(member);
                }
            });
            self.reach[ci] = Some(set);
        }
        below
    }
}

/// [`shared_dag_dp_pass`]'s pricing.
struct SharedPricer<'a, 'c, C, T, R> {
    egraph: &'a EGraph,
    dp: &'a mut Dp<'c, C, T, R>,
    var: Vec<Variance>,
    sets: ReachSets,
}

impl<C: CostFunction, T: TieBreak, R: StageRecorder> Settling for SharedPricer<'_, '_, C, T, R> {
    fn price(&mut self, class: EClassId, idx: usize, node: &ENode) -> usize {
        let node_var = node_variance(self.egraph, node, &self.var, class);
        let own = weighted_own(self.dp.costs, node, self.dp.shape.evals(node_var));
        let below = self.sets.union_below(self.egraph, node);
        let cost = own.saturating_add(below);
        self.dp.rec.candidate(class, idx, cost, own);
        cost
    }

    fn prefer(&self, class: EClassId, incumbent: usize, challenger: usize) -> bool {
        self.dp
            .ties
            .prefer(self.egraph, class, incumbent, challenger)
    }

    fn settle(&mut self, class: EClassId, idx: usize, node: &ENode, _cost: usize) -> bool {
        let node_var = node_variance(self.egraph, node, &self.var, class);
        let own = weighted_own(self.dp.costs, node, self.dp.shape.evals(node_var));
        // Rebuild the winner's union rather than carrying one per class in
        // flight: the driver prices candidates for many unsettled classes
        // before any of them settles, so a stored winner set would be a
        // second copy of the frontier's reach.
        let _ = self.sets.union_below(self.egraph, node);
        let me = self.sets.compact[class.0 as usize];
        self.sets.scratch.push(me);
        let set = Reach::smaller_of(&self.sets.scratch, self.sets.words);
        self.sets.bytes = self.sets.bytes.saturating_add(set.bytes());
        if self.sets.bytes > self.sets.budget {
            return false;
        }
        self.sets.reach[me as usize] = Some(set);
        self.sets.own[me as usize] = own;
        self.var[class.0 as usize] = node_var;
        self.dp.rec.settled(class, idx);
        true
    }
}

/// What [`shared_dag_dp_pass`] returns: its choice map when it finished
/// under budget, and its accounting either way.
pub(crate) struct SharedPassOutcome {
    /// `None` when the reach sets crossed the byte budget and the pass was
    /// abandoned.
    pub(crate) outcome: Option<DpOutcome>,
    pub(crate) stats: SharedPassStats,
}

const REACH_WORD_BITS: usize = u64::BITS as usize;

/// One live class's reach set — the compact ids of the classes its chosen
/// sub-DAG contains — in whichever of two forms is smaller.
///
/// The sparse form is the members themselves, four bytes each; the dense
/// form is one bit per live class. [`Reach::smaller_of`] picks per set, so
/// a leaf's set is four bytes and the root's is a bitset, and the total
/// held by [`shared_dag_dp_pass`] is never above the dense bound and is far
/// below it on any graph that is not a chain.
enum Reach {
    Sparse(Vec<u32>),
    Dense(Vec<u64>),
}

impl Reach {
    /// `members` as the smaller of the two forms over `words` dense words.
    /// `members` must be distinct.
    fn smaller_of(members: &[u32], words: usize) -> Self {
        let sparse_bytes = members.len() * core::mem::size_of::<u32>();
        let dense_bytes = words * core::mem::size_of::<u64>();
        if sparse_bytes <= dense_bytes {
            return Self::Sparse(members.to_vec());
        }
        let mut bits = alloc::vec![0u64; words];
        for &m in members {
            let m = m as usize;
            bits[m / REACH_WORD_BITS] |= 1u64 << (m % REACH_WORD_BITS);
        }
        Self::Dense(bits)
    }

    fn bytes(&self) -> usize {
        match self {
            Self::Sparse(v) => v.len() * core::mem::size_of::<u32>(),
            Self::Dense(v) => v.len() * core::mem::size_of::<u64>(),
        }
    }

    fn for_each(&self, mut f: impl FnMut(u32)) {
        match self {
            Self::Sparse(v) => v.iter().copied().for_each(f),
            Self::Dense(v) => {
                for (w, &word) in v.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        f((w * REACH_WORD_BITS + bit) as u32);
                    }
                }
            }
        }
    }
}

/// The classes reachable from `root`.
///
/// The order is a DFS post-order, which no longer decides anything: the DP
/// passes settle in **cost** order ([`settle_in_cost_order`]), and this is
/// the live set and the compact numbering the shared pass's reach sets are
/// indexed by. It used to be the DP's evaluation order, which is why a class
/// whose descendants reached it back — commutativity alone is enough on a
/// saturated graph — appeared before them and got priced at a sentinel.
fn post_order(egraph: &EGraph, root: EClassId) -> Vec<EClassId> {
    // Dense bitset over canonical class ids, not `BTreeSet<u32>`: every id
    // here is already bounded by `egraph.num_classes()`, so a `Vec<bool>`
    // index is O(1) and allocation-free per probe, versus an O(log n)
    // tree-node alloc per insert on a set this file already indexes by plain
    // `Vec` elsewhere (`cost_of_choices`'s `color: Vec<u8>`). `post_order`
    // runs twice per extraction (once per DP pass), so this is on the same
    // hot path as `shared_dag_dp_pass`.
    let num_classes = egraph.num_classes();
    let mut order: Vec<EClassId> = Vec::new();
    let mut settled: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];
    let mut on_stack: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;

        if settled[idx] {
            continue;
        }

        if !children_done {
            if on_stack[idx] {
                continue;
            }
            on_stack[idx] = true;

            stack.push((canonical, true));

            for node in egraph.nodes(canonical) {
                {
                    let children = (node).children_slice();
                    for &child in children {
                        let child_canonical = egraph.find(child);
                        if !settled[child_canonical.0 as usize] {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            on_stack[idx] = false;
            settled[idx] = true;
            order.push(canonical);
        }
    }

    order
}

/// Count references to each e-class in the extracted expression.
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow.
fn count_refs_recursive(
    egraph: &EGraph,
    class: EClassId,
    best_node: &[Option<usize>],
    ref_counts: &mut [usize],
) {
    let mut stack: Vec<EClassId> = alloc::vec![class];

    while let Some(cls) = stack.pop() {
        let canonical = egraph.find(cls);
        ref_counts[canonical.0 as usize] += 1;

        // Only recurse on first visit to count true structural refs
        if ref_counts[canonical.0 as usize] == 1 {
            if let Some(node_idx) = best_node[canonical.0 as usize] {
                let node = &egraph.nodes(canonical)[node_idx];
                {
                    let children = (node).children_slice();
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }
    }
}

/// Topological sort of e-classes for emission order.
///
/// Returns e-classes in order such that dependencies come before dependents.
/// Shared e-classes are prioritized to appear early.
///
/// Uses iterative post-order traversal to avoid thread stack overflow.
fn toposort_dag(
    egraph: &EGraph,
    root: EClassId,
    best_node: &[Option<usize>],
    shared: &[(EClassId, usize)],
) -> Vec<EClassId> {
    // Dense bitsets over canonical class ids (bounded by `best_node.len()`,
    // itself sized to `egraph.num_classes()` by the DP pass that built it) —
    // see `post_order`'s doc comment for why this beats `BTreeSet<u32>` here.
    let num_classes = best_node.len();
    let mut shared_set: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];
    for (id, _) in shared {
        shared_set[id.0 as usize] = true;
    }
    let mut visited: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];
    let mut result = Vec::new();

    // Iterative post-order: (class, children_pushed)
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;

        if visited[idx] {
            continue;
        }

        if !children_done {
            stack.push((canonical, true));

            if let Some(node_idx) = best_node.get(idx).and_then(|o| *o) {
                let node = &egraph.nodes(canonical)[node_idx];
                {
                    let children = (node).children_slice();
                    for &child in children {
                        let child_can = egraph.find(child);
                        if !visited[child_can.0 as usize] {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            visited[idx] = true;

            if shared_set[idx] {
                result.push(canonical);
            }
        }
    }

    // Add root if not already included
    let root_canonical = egraph.find(root);
    if !result.iter().any(|id| *id == root_canonical) {
        result.push(root_canonical);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scope-weighted extraction un-fuses an FMA to move work out of the loop.
    ///
    /// One e-class holds two equal forms of `X + 2Z`: the fused
    /// `MulAdd(2, Z, X)` — one instruction — and `X + (Z + Z)`, which is two.
    /// Priced by text the FMA wins and always should. Priced against a
    /// 256×256 frame it loses: its single instruction runs at every one of
    /// the 65 536 samples, while the pair runs one add per sample and lifts
    /// `Z + Z` out of the loop entirely, because `Z` does not vary across it.
    ///
    /// No saturation here — both forms are placed in the class by hand — so
    /// this is a property of the cost model alone, and the choice flipping
    /// with the lattice is the whole claim of scope-weighted extraction: the
    /// factorization is *chosen*, not discovered afterwards by a hoisting
    /// pass that arrives once the FMA has already welded `Z` into the
    /// per-sample expression.
    #[test]
    fn scope_weighting_unfuses_an_fma_to_hoist_the_z_term() {
        use crate::egraph::ops::op_from_kind;
        use pixelflow_ir::OpKind;

        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let z = egraph.add(ENode::Var(2));
        let two = egraph.add(ENode::Const(2.0f32.to_bits()));
        let op = |kind| op_from_kind(kind).expect("op is modelled");

        let fused = egraph.add(ENode::Op {
            op: op(OpKind::MulAdd),
            children: vec![two, z, x],
        });
        let z_plus_z = egraph.add(ENode::Op {
            op: op(OpKind::Add),
            children: vec![z, z],
        });
        let unfused = egraph.add(ENode::Op {
            op: op(OpKind::Add),
            children: vec![x, z_plus_z],
        });
        let root = egraph.union(fused, unfused);
        egraph.rebuild();

        let costs = CostModel::latency_prior();
        let chosen = |shape| {
            let dag = extract_dag_scoped(&egraph, root, &costs, shape);
            let idx = dag.choices[egraph.find(root).0 as usize].expect("root is chosen");
            match &egraph.nodes(egraph.find(root))[idx] {
                ENode::Op { op, .. } => op.kind(),
                other => panic!("root should be an op, got {other:?}"),
            }
        };

        assert_eq!(
            chosen(LatticeShape::POINT),
            OpKind::MulAdd,
            "with no lattice the fused form is one instruction against two"
        );
        assert_eq!(
            chosen(LatticeShape::new([256, 256])),
            OpKind::Add,
            "over a frame the fused form pays for Z at every sample"
        );
    }

    /// A chain of `n` distinct-const `Add` nodes: exactly `n + 1` live
    /// classes with no saturation (each `add()` is a fresh, non-folding
    /// node), and the adversarial shape for the reach sets — every class
    /// reaches every class below it, so the sets are as large as they can
    /// be. What the tree-only test above the old class-count gate used, and
    /// what the byte budget is bounded on.
    fn add_chain(n: usize) -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let mut cur = egraph.add(ENode::Var(0));
        for i in 0..n {
            let c = egraph.add(ENode::constant(i as f32 + 1.0));
            cur = egraph.add(ENode::Op {
                op: crate::egraph::ops::op_from_kind(pixelflow_ir::OpKind::Add)
                    .expect("Add is modelled"),
                children: vec![cur, c],
            });
        }
        egraph.rebuild();
        (egraph, cur)
    }

    /// The shared pass abandons itself at its byte budget and says so: the
    /// choice map is `None`, the stats carry the first total over the
    /// budget, and `extract_dag_scoped` — whose budget is the production
    /// constant — reports the objective it actually used rather than
    /// returning the tree term under a shared label.
    #[test]
    fn shared_pass_over_budget_is_abandoned_loudly() {
        const CHAIN: usize = 2_000;
        let (egraph, root) = add_chain(CHAIN);
        let costs = CostModel::latency_prior();

        let full = shared_dag_dp_pass(
            &egraph,
            root,
            &mut Dp::production(&costs, LatticeShape::POINT),
            usize::MAX,
        );
        let full_choices = full
            .outcome
            .as_ref()
            .expect("unbounded pass finishes")
            .choices
            .clone();
        assert_eq!(full.stats.live_classes, 2 * CHAIN + 1);
        // The dense bound, plus one word of rounding per set.
        let live = full.stats.live_classes;
        let dense_bound = live * live.div_ceil(REACH_WORD_BITS) * 8;
        assert!(
            full.stats.reach_bytes <= dense_bound,
            "reach sets ({}) must never exceed the dense bound ({dense_bound})",
            full.stats.reach_bytes
        );

        let budget = full.stats.reach_bytes / 2;
        let cut = shared_dag_dp_pass(
            &egraph,
            root,
            &mut Dp::production(&costs, LatticeShape::POINT),
            budget,
        );
        assert!(
            cut.outcome.is_none(),
            "a pass over budget returns no choices"
        );
        assert!(
            cut.stats.reach_bytes > budget,
            "stats carry the total that crossed the budget ({} vs {budget})",
            cut.stats.reach_bytes
        );
        assert_eq!(cut.stats.live_classes, full.stats.live_classes);

        // Under the production budget this chain fits, and the report says
        // which term came back and what it cost to know.
        let scoped = extract_dag_scoped(&egraph, root, &costs, LatticeShape::POINT);
        let stats = scoped
            .report
            .shared_pass
            .expect("production extraction always runs the pass");
        assert_eq!(stats, full.stats);
        assert!(
            full.stats.reach_bytes <= SHARED_DAG_PASS_BYTE_BUDGET,
            "fixture must fit the production budget for this half of the test"
        );
        assert_ne!(scoped.report.objective, ExtractionObjective::TreeOnly);
        assert_ne!(scoped.report.objective, ExtractionObjective::External);
        let shared = extract_dag_shared_arm(&egraph, root, &costs, LatticeShape::POINT);
        assert_eq!(shared.choices, full_choices);
    }

    /// Sparse and dense reach sets are one set: a union taken through
    /// either form visits the same members, so a class whose set flips
    /// form (the chain's upper half) prices its children identically.
    #[test]
    fn reach_forms_agree() {
        let words = 4;
        let few: Vec<u32> = vec![3, 200, 77];
        let many: Vec<u32> = (0..words as u32 * 40).collect();
        let sparse = Reach::smaller_of(&few, words);
        let dense = Reach::smaller_of(&many, words);
        assert!(matches!(sparse, Reach::Sparse(_)));
        assert!(matches!(dense, Reach::Dense(_)));
        assert_eq!(sparse.bytes(), few.len() * 4);
        assert_eq!(dense.bytes(), words * 8);
        let mut seen = Vec::new();
        sparse.for_each(|m| seen.push(m));
        assert_eq!(seen, few);
        seen.clear();
        dense.for_each(|m| seen.push(m));
        assert_eq!(seen, many);
    }

    /// The sharing-aware DP as it shipped before the reach sets went hybrid
    /// **and** before the pass settled in cost order: one dense bitset per
    /// live class, one DFS post-order, `CYCLE_COST` for a class whose child
    /// is still on the stack. Kept verbatim as the reference the budgeted
    /// pass is held to — its results are the ones every committed extraction
    /// row was taken with
    /// (`docs/results/2026-09-07-egraph-off-vs-on-real-shaders-rows`).
    ///
    /// Its agreement with the shipped pass is therefore conditional, and the
    /// condition is checked rather than assumed: the fixpoint reproduces
    /// this reference exactly **wherever this reference had an opinion**, so
    /// the caller asserts the fixture holds no class it could only price at
    /// the sentinel. Without that assertion the comparison would quietly go
    /// vacuous the day a fixture grew a cycle.
    fn dense_reference_pass<C: CostFunction>(
        egraph: &EGraph,
        root: EClassId,
        costs: &C,
        shape: LatticeShape,
    ) -> (Vec<Option<usize>>, usize) {
        const BITS: usize = usize::BITS as usize;
        let num_classes = egraph.num_classes();
        let order = post_order(egraph, root);
        let live = order.len();
        let words = live.div_ceil(BITS);
        let mut compact: Vec<u32> = alloc::vec![u32::MAX; num_classes];
        for (i, c) in order.iter().enumerate() {
            compact[c.0 as usize] = i as u32;
        }
        let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut best_var: Vec<Variance> = alloc::vec![Variance::CONST; num_classes];
        let mut best_own: Vec<usize> = alloc::vec![0; live];
        let mut reach: Vec<usize> = alloc::vec![0; live * words];
        let mut scratch: Vec<usize> = alloc::vec![0; words];
        for canonical in order.iter().copied() {
            let me = compact[canonical.0 as usize] as usize;
            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;
            let mut min_var = Variance::CONST;
            let mut min_own = 0usize;
            for (idx, node) in nodes.iter().enumerate() {
                let node_var = node_variance(egraph, node, &best_var, canonical);
                let own = weighted_own(costs, node, shape.evals(node_var));
                let this_node_cost = match node {
                    ENode::Var(_)
                    | ENode::Const(_)
                    | ENode::Buffer(_)
                    | ENode::Uniform(_)
                    | ENode::Param(_) => own,
                    ENode::Op { .. } | ENode::Reduce { .. } => {
                        let children = node.children_slice();
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            scratch.fill(0);
                            let mut below = 0usize;
                            let mut unresolved = false;
                            for &child in children.iter() {
                                let c = egraph.find(child).0 as usize;
                                if best_cost[c].is_none() {
                                    unresolved = true;
                                    break;
                                }
                                let base = compact[c] as usize * words;
                                for w in 0..words {
                                    let fresh = reach[base + w] & !scratch[w];
                                    if fresh == 0 {
                                        continue;
                                    }
                                    scratch[w] |= fresh;
                                    let mut bits = fresh;
                                    while bits != 0 {
                                        let bit = bits.trailing_zeros() as usize;
                                        bits &= bits - 1;
                                        below = below.saturating_add(best_own[w * BITS + bit]);
                                    }
                                }
                            }
                            if unresolved {
                                CYCLE_COST
                            } else {
                                own.saturating_add(below)
                            }
                        }
                    }
                };
                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                    min_var = node_var;
                    min_own = own;
                }
            }
            let base = me * words;
            reach[base..base + words].fill(0);
            {
                let children = (&nodes[min_idx]).children_slice();
                if min_cost != CYCLE_COST {
                    for &child in children.iter() {
                        let cbase = compact[egraph.find(child).0 as usize] as usize * words;
                        for w in 0..words {
                            reach[base + w] |= reach[cbase + w];
                        }
                    }
                }
            }
            reach[base + me / BITS] |= 1usize << (me % BITS);
            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
            best_var[canonical.0 as usize] = min_var;
            best_own[me] = min_own;
        }
        let root_cost = best_cost[egraph.find(root).0 as usize].expect("the root is settled");
        (best_node, root_cost)
    }

    /// An SDF-shaped arena with real sharing (the same op mix as
    /// `egraph_profile`'s), saturated by the production optimizer under a
    /// small explicit budget: a graph with variants, cycles and shared
    /// classes, as the passes meet them in production.
    fn saturated_sdf_egraph(target_nodes: usize) -> (EGraph, EClassId) {
        use pixelflow_ir::{ExprArena, OpKind};
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let c = arena.push_const(0.37);
        let dx = arena.push_binary(OpKind::Sub, x, c);
        let dy = arena.push_binary(OpKind::Sub, y, c);
        let dx2 = arena.push_binary(OpKind::Mul, dx, dx);
        let mut cur = arena.push_ternary(OpKind::MulAdd, dy, dy, dx2);
        let mut step = 0usize;
        while arena.len() < target_nodes {
            cur = match step % 8 {
                0 => {
                    let inside = arena.push_binary(OpKind::Lt, cur, c);
                    arena.push_ternary(OpKind::Select, inside, dx2, cur)
                }
                1 => arena.push_unary(OpKind::Sqrt, cur),
                2 => arena.push_binary(OpKind::Mul, cur, dx),
                3 => arena.push_binary(OpKind::Add, cur, dy),
                4 => arena.push_ternary(OpKind::MulAdd, cur, c, dx2),
                5 => arena.push_binary(OpKind::Max, cur, dx),
                6 => arena.push_binary(OpKind::Sub, cur, c),
                _ => arena.push_binary(OpKind::Mul, cur, cur),
            };
            step += 1;
        }
        let mut optimizer =
            super::super::Optimizer::production().budget(super::super::Budget::Explicit {
                iterations: 4,
                classes: 3_000,
                applications: Some(20_000),
            });
        let mut egraph = optimizer.egraph();
        let root =
            super::super::insert(&arena, cur, &mut egraph, super::super::Vocabulary::Runtime)
                .expect("the SDF arena is representable");
        let node_count = super::super::reachable_count(&arena, cur);
        let optimized = optimizer.run(&mut egraph, root, node_count);
        assert!(
            optimized.stats.classes > 500,
            "fixture must saturate into a real graph ({} classes)",
            optimized.stats.classes
        );
        (egraph, root)
    }

    /// How many classes the pre-fixpoint DP left at the cycle sentinel: a
    /// class every one of whose candidates mentions itself or a class the
    /// single DFS post-order had not yet priced.
    fn sentinel_priced(egraph: &EGraph, root: EClassId) -> Vec<EClassId> {
        let order = post_order(egraph, root);
        let mut priced: Vec<bool> = alloc::vec![false; egraph.num_classes()];
        let mut sentinel = Vec::new();
        for &class in &order {
            let all_cyclic = egraph.nodes(class).iter().all(|node| match node {
                ENode::Op { children, .. } => children
                    .iter()
                    .any(|&c| egraph.find(c) == class || !priced[egraph.find(c).0 as usize]),
                _ => false,
            });
            if all_cyclic {
                sentinel.push(class);
            }
            priced[class.0 as usize] = true;
        }
        sentinel
    }

    /// The defect the fixpoint exists to remove: the post-order DP priced a
    /// class in a cycle at the sentinel and expressed no preference, so the
    /// pick fell to `repair_choices_well_founded`, which has no cost model.
    /// The fixpoint prices it — `neg(neg(x)) = x` means the `Neg` class's
    /// only child is the class that holds it, and settling in cost order
    /// reaches the child first.
    #[test]
    fn the_fixpoint_prices_a_class_the_post_order_dp_left_at_the_sentinel() {
        #[derive(Default)]
        struct Prices(Vec<(EClassId, usize)>);
        impl StageRecorder for Prices {
            fn candidate(&mut self, class: EClassId, _idx: usize, cost: usize, _own: usize) {
                self.0.push((class, cost));
            }
            fn settled(&mut self, _: EClassId, _: usize) {}
        }

        let (egraph, merged, n1) = cyclic_capable_egraph();
        let n1c = egraph.find(n1);
        assert!(
            sentinel_priced(&egraph, merged).contains(&n1c),
            "the fixture must hold a class the post-order DP could only price at the sentinel"
        );

        let costs = CostModel::latency_prior();
        let mut dp = Dp::new(&costs, LatticeShape::POINT, Insertion, Prices::default());
        let choices = tree_dp_pass(&egraph, merged, &mut dp).choices;
        let priced = dp.into_recorder().0;

        assert!(
            priced.iter().any(|&(c, _)| c == n1c),
            "the class the post-order DP left at the sentinel was never priced"
        );
        assert!(
            priced.iter().all(|&(_, cost)| cost < CYCLE_COST),
            "the fixpoint priced a candidate at the cycle sentinel: {priced:?}"
        );
        assert!(
            choices[n1c.0 as usize].is_some(),
            "the class was priced but not settled"
        );
    }

    /// The gate on deleting the repair stage from the extraction path: a
    /// choice map out of either DP pass is already well-founded, so
    /// `repair_choices_well_founded` has nothing to change.
    #[test]
    fn the_dp_map_is_well_founded_so_the_repair_is_a_no_op() {
        let costs = CostModel::latency_prior();
        for (nodes, shape) in [
            (64, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            for raw in [
                tree_dp_pass(&egraph, root, &mut Dp::production(&costs, shape)).choices,
                shared_dag_dp_pass(
                    &egraph,
                    root,
                    &mut Dp::production(&costs, shape),
                    usize::MAX,
                )
                .outcome
                .expect("unbounded")
                .choices,
            ] {
                let mut repaired = raw.clone();
                repair_choices_well_founded(&egraph, root, &mut repaired);
                assert_eq!(
                    repaired, raw,
                    "{nodes} nodes at {shape:?}: the repair moved a class the DP had settled"
                );
            }
        }
    }

    /// The research tie-break seam changes nothing under production's
    /// instance.
    ///
    /// `Ties::Insertion` is the `impl` the shipped extractor uses, so
    /// `witness::extract_under` with it must return the same choice map and
    /// the same cost as `extract_dag_scoped` — on a saturated graph, at a
    /// frame shape, where ties are dense (the 2026-09-08 witness run found
    /// 47–81 % of live classes tied). Without this, a refactor of either DP
    /// pass could silently move production's extraction and only the
    /// research harness would see it.
    #[cfg(feature = "provenance-journal")]
    #[test]
    fn insertion_tie_break_is_productions_extraction() {
        use crate::egraph::witness::{Ties, extract_under};
        let costs = CostModel::latency_prior();
        for (nodes, shape) in [
            (64, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            let production = extract_dag_scoped(&egraph, root, &costs, shape);
            let (choices, cost) = extract_under(&egraph, root, &costs, shape, Ties::Insertion);
            assert_eq!(
                choices, production.choices,
                "{nodes} nodes at {shape:?}: the Insertion tie-break moved production's choices"
            );
            assert_eq!(
                (cost.tree, cost.dag),
                (production.total_cost, production.dag_cost),
                "{nodes} nodes at {shape:?}: the Insertion tie-break moved production's cost"
            );
        }
    }

    /// The budgeted hybrid-set pass and the dense pass it replaced choose
    /// the same node in every class of a saturated graph — same objective,
    /// same tie-breaking, priced through sparse and dense sets alike —
    /// under both the point and a frame lattice. This is what lets the
    /// committed extraction rows stand as this pass's regression baseline.
    #[test]
    fn shared_pass_matches_the_dense_reference_on_a_saturated_graph() {
        let costs = CostModel::latency_prior();
        for (nodes, shape) in [
            (64, LatticeShape::POINT),
            (256, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            assert!(
                sentinel_priced(&egraph, root).is_empty(),
                "{nodes} nodes: the reference could only price some class at the sentinel, so \
                 it has no opinion there and this comparison would be vacuous — see \
                 `the_fixpoint_prices_a_class_the_post_order_dp_left_at_the_sentinel`"
            );
            let (reference_choices, reference_cost) =
                dense_reference_pass(&egraph, root, &costs, shape);
            let pass = shared_dag_dp_pass(
                &egraph,
                root,
                &mut Dp::production(&costs, shape),
                usize::MAX,
            );
            let live = pass.stats.live_classes;
            assert!(live > 100, "{nodes} nodes: only {live} live classes");
            let dp = pass.outcome.expect("unbounded");
            assert_eq!(
                dp.choices, reference_choices,
                "{nodes} nodes at {shape:?}: the hybrid pass disagrees with the dense reference"
            );
            // #1229 pinned the choices; the claim beside them is the number
            // the pass minimized, and a sparse/dense split that agreed on the
            // map while disagreeing on its price would be a mispricing this
            // test was blind to.
            assert_eq!(
                dp.root_cost, reference_cost,
                "{nodes} nodes at {shape:?}: the hybrid pass and the dense reference price the \
                 same map differently"
            );
            let dense_bytes = live * live.div_ceil(REACH_WORD_BITS) * 8;
            assert!(
                pass.stats.reach_bytes <= dense_bytes,
                "{nodes} nodes: hybrid sets ({}) above the dense bound ({dense_bytes})",
                pass.stats.reach_bytes
            );
        }
    }

    #[test]
    fn extract_simple() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));

        let costs = CostModel::default();
        let (arena, root, cost) = extract(&egraph, x, &costs);

        assert_eq!(arena.len(), 1);
        assert_eq!(root.0, 0);
        assert_eq!(cost, 0); // Leaf nodes (Var/Const) have cost 0
    }

    #[test]
    fn extract_with_ops() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let (arena, root, _cost) = extract(&egraph, sum, &costs);

        assert_eq!(arena.len(), 3); // Add + X + Y
        assert_eq!(root.0, 2);
    }

    #[test]
    fn extract_latency_prior_picks_cheaper_equivalent_form() {
        // x + x and x * 2 are equivalent, but under the latency-prior cost
        // model Add (4 cycles) is cheaper than Mul (5 cycles), so once the
        // two forms are unioned into one e-class, extraction must pick the
        // Add form.
        //
        // This is the extraction-side counterpart to the existing
        // NNUE latency-prior tests: it exercises `CostModel::latency_prior`
        // (the static cost table), not the neural model.
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let two = egraph.add(ENode::constant(2.0));

        let x_plus_x = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, x],
        });
        let x_times_2 = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, two],
        });

        egraph.union(x_plus_x, x_times_2);

        let costs = CostModel::latency_prior();
        assert!(
            costs.cost(pixelflow_ir::OpKind::Add) < costs.cost(pixelflow_ir::OpKind::Mul),
            "test assumes Add is strictly cheaper than Mul in the latency prior"
        );

        let (arena, root, cost) = extract(&egraph, egraph.find(x_plus_x), &costs);

        // Cheapest form is `x + x`: Add(4) + Var(0) + Var(0) = 4.
        assert_eq!(cost, costs.cost(pixelflow_ir::OpKind::Add));

        let root_node = arena.node(root);
        assert!(
            matches!(
                root_node,
                pixelflow_ir::arena::ExprNode::Binary(pixelflow_ir::OpKind::Add, _, _)
            ),
            "extraction with the latency-prior cost model should pick the Add form, got {root_node:?}"
        );
    }

    // ========================================================================
    // Swap-refinement search (Reranker seam)
    // ========================================================================

    /// A trivial, test-only [`Reranker`]: the sum of `costs.cost(op)` over
    /// every node the candidate's materialised arena contains — the same
    /// additive latency-prior table [`extract_dag`] minimizes, just summed
    /// over the arena instead of folded bottom-up through the e-graph. No
    /// sharing discount (each arena node is already deduped by
    /// `choices_to_arena`, so this is a DAG cost, not a tree cost) — fine
    /// for the sharing-free graphs these tests build.
    struct TableReranker<'a> {
        costs: &'a CostModel,
    }

    impl Reranker for TableReranker<'_> {
        fn score(&self, _extraction: &Extraction<'_>, arena: &pixelflow_ir::ExprArena) -> f64 {
            let mut total = 0.0f64;
            for i in 0..arena.len() {
                let id = pixelflow_ir::ExprId(i as u32);
                total += self.costs.cost(arena.kind(id)) as f64;
            }
            total
        }
    }

    #[test]
    fn swap_search_reproduces_extract_dags_choice_on_a_cost_ambiguous_class() {
        // Same setup as `extract_latency_prior_picks_cheaper_equivalent_form`:
        // x+x and x*2 are unioned, so the class has a genuine choice to make.
        // The bootstrap pass may start on either node; the swap-refinement
        // loop must walk to the same cheaper form `extract_dag` reaches via
        // its (unrelated) bottom-up DP — same table, same answer, two
        // different search strategies.
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let two = egraph.add(ENode::constant(2.0));
        let x_plus_x = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, x],
        });
        let x_times_2 = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, two],
        });
        let merged = egraph.union(x_plus_x, x_times_2);
        egraph.rebuild();

        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, merged, &costs);

        let reranker = TableReranker { costs: &costs };
        let extractor = IncrementalExtractor::new(&reranker, 8);
        let (search_cost, extraction) = extractor.extract_choices_only(&egraph, merged);
        let (arena, root) = choices_to_arena(&extraction);

        assert_eq!(
            search_cost, dag.total_cost as f64,
            "swap-refinement search must reach the same additive cost as extract_dag's DP"
        );
        let root_node = arena.node(root);
        assert!(
            matches!(
                root_node,
                pixelflow_ir::arena::ExprNode::Binary(pixelflow_ir::OpKind::Add, _, _)
            ),
            "swap search should have converged on the Add form, got {root_node:?}"
        );
    }

    #[test]
    fn swap_search_bootstrap_survives_a_merge_reordered_class() {
        // The real round-1 failure this search once hit: the old bootstrap
        // picked node index 0 everywhere, but after saturation merges "node
        // 0" of two classes can reference each other — a CYCLIC bootstrap.
        // `Extraction::from_backfill` (which the search's bootstrap pass
        // uses) must return a well-founded choice set for ANY node
        // ordering, which materialises without panicking regardless of
        // what the reranker says.
        let (egraph, merged, _n1) = cyclic_capable_egraph();
        let costs = CostModel::latency_prior();
        let reranker = TableReranker { costs: &costs };
        let extractor = IncrementalExtractor::new(&reranker, 8);
        let (_cost, extraction) = extractor.extract_choices_only(&egraph, merged);
        let (arena, root) = choices_to_arena(&extraction);
        assert!(arena.len() >= 1);
        assert!(root.0 < arena.len() as u32);
    }

    // ========================================================================
    // Cyclic choice sets (2026-08 round 1: full-DEV bench OOM)
    // ========================================================================

    /// An e-graph whose merged root class holds a node referencing a class
    /// that references it back: x unioned with neg(neg(x)). Returns
    /// (egraph, merged_root, inner_neg_class).
    fn cyclic_capable_egraph() -> (EGraph, EClassId, EClassId) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let n1 = egraph.add(ENode::Op {
            op: &super::super::ops::Neg,
            children: alloc::vec![x],
        });
        let n2 = egraph.add(ENode::Op {
            op: &super::super::ops::Neg,
            children: alloc::vec![n1],
        });
        let merged = egraph.union(x, n2); // neg(neg(x)) = x
        egraph.rebuild();
        (egraph, merged, n1)
    }

    /// Node index of the Neg op inside a class, if present.
    fn neg_index(egraph: &EGraph, class: EClassId) -> Option<usize> {
        egraph
            .nodes(egraph.find(class))
            .iter()
            .position(|n| matches!(n, ENode::Op { .. }))
    }

    #[test]
    #[should_panic(expected = "CYCLIC")]
    fn choices_to_arena_refuses_a_cyclic_choice_set() {
        // Before the gray-marking assert, this walk re-scheduled the cycle
        // forever: a full-DEV bench run grew to 2.7GB and died by SIGKILL
        // with zero diagnostics. The cycle must be a loud extractor
        // accusation instead. `Extraction`'s own constructors now refuse a
        // cyclic choice set before this point is ever reached (see
        // `extraction_constructors_refuse_a_cyclic_choice_set`) — this test
        // exercises `choices_to_arena`'s own belt-and-suspenders check by
        // constructing the `Extraction` directly (private-field literal,
        // valid from within this module), bypassing the smart constructors
        // on purpose.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        let extraction = Extraction {
            egraph: &egraph,
            root: egraph.find(merged),
            choices,
        };
        let _ = choices_to_arena(&extraction);
    }

    #[test]
    #[should_panic(expected = "cyclic")]
    fn extraction_constructors_refuse_a_cyclic_choice_set() {
        // The type-level guarantee J2 adds: a cyclic choice vector can no
        // longer become an `Extraction` at all, so the 2.7GB-OOM class of
        // bug can't reach `choices_to_arena` in the first place.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        let _ = Extraction::from_backfill(&egraph, merged, choices);
    }

    #[test]
    fn repair_keeps_acyclic_choices_and_terminates_on_a_recorded_cycle() {
        // The static DP's failure shape: a recorded mutual cycle (merged
        // class picks Neg(n1), n1 picks Neg(merged)). The old restart-DFS
        // breaker could rediscover the same cycle forever — two DEV kernels
        // pinned a core for minutes. The repair must terminate, produce a
        // well-founded set, and leave already-acyclic choices alone.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));

        repair_choices_well_founded(&egraph, merged, &mut choices);
        assert!(
            !choices_have_cycle_from(&egraph, merged, &choices),
            "repair must leave a well-founded choice set"
        );
        // Materialization is the proof of well-foundedness.
        let extraction = Extraction {
            egraph: &egraph,
            root: egraph.find(merged),
            choices,
        };
        let (arena, root) = choices_to_arena(&extraction);
        assert!(root.0 < arena.len() as u32);

        // And a set that is ALREADY acyclic passes through untouched.
        let mut acyclic: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        backfill_well_founded(&egraph, merged, &mut acyclic);
        let before = acyclic.clone();
        repair_choices_well_founded(&egraph, merged, &mut acyclic);
        assert_eq!(
            before, acyclic,
            "an acyclic choice function must be kept verbatim (drain phase only)"
        );
    }

    // ========================================================================
    // DAG Extraction Tests
    // ========================================================================

    #[test]
    fn extract_dag_simple() {
        // X + Y: no sharing
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, sum, &costs);

        assert!(
            dag.shared.is_empty(),
            "X + Y should have no shared subexprs"
        );
        assert_eq!(dag.root, egraph.find(sum));
    }

    #[test]
    fn extract_dag_shared_subexpr() {
        // X * X: X is used twice
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x], // X used twice!
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, x_squared, &costs);

        // X should be marked as shared (used 2 times)
        assert!(!dag.shared.is_empty(), "X * X should have X as shared");
        assert!(dag.is_shared(x), "X should be shared");
        assert_eq!(dag.use_count(x), 2);
    }

    #[test]
    fn extract_dag_triple_use() {
        // sin(X) * sin(X) + sin(X): sin(X) used 3 times
        // We simulate this structure without actual sin
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        // Simulate sin(X) as sqrt(X) for test purposes
        let sin_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let sin_x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sin_x, sin_x],
        });
        let result = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![sin_x_squared, sin_x],
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, result, &costs);

        // sin_x should be shared (used 3 times: twice in Mul, once in Add)
        assert!(
            dag.is_shared(sin_x),
            "sqrt(X) should be shared (used 3 times)"
        );
        assert_eq!(dag.use_count(sin_x), 3);

        // Schedule should have sin_x before the operations that use it
        let sin_x_idx = dag.schedule.iter().position(|&id| id == egraph.find(sin_x));
        assert!(sin_x_idx.is_some(), "sin_x should be in schedule");
    }

    #[test]
    fn extract_dag_nested_sharing() {
        // (X + Y) * (X + Y): (X + Y) is shared
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sum, sum], // sum used twice
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, product, &costs);

        // (X + Y) should be shared
        assert!(dag.is_shared(sum), "(X + Y) should be shared");
        assert_eq!(dag.use_count(sum), 2);
    }

    // ========================================================================
    // compute_ref_counts Tests
    // ========================================================================

    #[test]
    fn compute_ref_counts_no_sharing() {
        // X + Y: no sharing
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(sum).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let rc = compute_ref_counts(&egraph, sum, &choices);
        assert_eq!(
            rc[egraph.find(sum).0 as usize],
            1,
            "root should have ref_count 1"
        );
        assert_eq!(
            rc[egraph.find(x).0 as usize],
            1,
            "X should have ref_count 1"
        );
        assert_eq!(
            rc[egraph.find(y).0 as usize],
            1,
            "Y should have ref_count 1"
        );
    }

    #[test]
    fn compute_ref_counts_shared() {
        // X * X: X is used twice
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(x_squared).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let rc = compute_ref_counts(&egraph, x_squared, &choices);
        assert_eq!(rc[egraph.find(x_squared).0 as usize], 1, "root ref_count");
        assert_eq!(
            rc[egraph.find(x).0 as usize],
            2,
            "X should have ref_count 2"
        );
    }

    #[test]
    fn compute_ref_counts_triple_use() {
        // sqrt(X) * sqrt(X) + sqrt(X): sqrt(X) referenced 3 times
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sqrt_x, sqrt_x],
        });
        let result = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![product, sqrt_x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(result).0 as usize] = Some(0);
        choices[egraph.find(product).0 as usize] = Some(0);
        choices[egraph.find(sqrt_x).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let rc = compute_ref_counts(&egraph, result, &choices);
        assert_eq!(
            rc[egraph.find(sqrt_x).0 as usize],
            3,
            "sqrt(X) should have ref_count 3"
        );
        assert_eq!(
            rc[egraph.find(x).0 as usize],
            1,
            "X should have ref_count 1 (only 1 parent)"
        );
    }

    // =========================================================================
    // Train/deploy feature-path equivalence (2026-08 round-0 skew guard)
    // =========================================================================

    #[test]
    fn edge_trace_records_a_reload_for_a_shared_subexpression() {
        use crate::nnue::EdgeTrace;

        // sqrt(X) * sqrt(X): tree has 2x sqrt edges, DAG has 1x sqrt + 1x reload
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sqrt_x, sqrt_x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(product).0 as usize] = Some(0);
        choices[egraph.find(sqrt_x).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, product, choices);
        let trace = EdgeTrace::from_extraction(&extraction);

        assert_eq!(trace.node_count(), 3, "3 unique nodes");
        assert_eq!(
            trace.edges().len(),
            3,
            "shared reuse should contribute a reload edge"
        );
        let reloads = trace
            .edges()
            .iter()
            .filter(|e| {
                e.parent == pixelflow_ir::OpKind::Mul && e.child == pixelflow_ir::OpKind::Var
            })
            .count();
        assert_eq!(reloads, 1);
    }

    // =========================================================================
    // Arena/extraction edge-walk equivalence (2026-08 round-0 skew guard)
    // =========================================================================

    /// The arena walk (`EdgeTrace::from_arena_dag`) and the e-graph walk
    /// (`EdgeTrace::from_extraction`) are thin adapters over one walker, and
    /// this test pins that: for the same DAG — shared subexpressions, shared
    /// leaves — the two paths must record the identical edge stream. If a
    /// future change gives either path its own edge policy, this fails.
    #[test]
    fn arena_and_extraction_walks_record_the_same_edge_stream() {
        use crate::nnue::EdgeTrace;
        use pixelflow_ir::{ExprArena, OpKind};

        // Arena: (sin(Z * 0.3) * (X + Y) + sin(Z * 0.3)) + Y * 0.3
        // - sin(Z * 0.3) is SHARED (register reload on the second reference)
        // - Y and 0.3 are shared leaves (leaf reload policy)
        let mut arena = ExprArena::new();
        let z = arena.push_var(2);
        let c = arena.push_const(0.3);
        let zm = arena.push_binary(OpKind::Mul, z, c);
        let sin = arena.push_unary(OpKind::Sin, zm);
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let xy = arena.push_binary(OpKind::Add, x, y);
        let m = arena.push_binary(OpKind::Mul, sin, xy);
        let a = arena.push_binary(OpKind::Add, m, sin);
        let yc = arena.push_binary(OpKind::Mul, y, c);
        let root = arena.push_binary(OpKind::Add, a, yc);

        // The SAME DAG as an e-graph (sharing preserved node for node).
        use crate::egraph::ops;
        let mut eg = EGraph::new();
        let ez = eg.add(ENode::Var(2));
        let ec = eg.add(ENode::constant(0.3));
        let ezm = eg.add(ENode::Op {
            op: &ops::Mul,
            children: alloc::vec![ez, ec],
        });
        let esin = eg.add(ENode::Op {
            op: &ops::Sin,
            children: alloc::vec![ezm],
        });
        let ex = eg.add(ENode::Var(0));
        let ey = eg.add(ENode::Var(1));
        let exy = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ex, ey],
        });
        let em = eg.add(ENode::Op {
            op: &ops::Mul,
            children: alloc::vec![esin, exy],
        });
        let ea = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![em, esin],
        });
        let eyc = eg.add(ENode::Op {
            op: &ops::Mul,
            children: alloc::vec![ey, ec],
        });
        let eroot = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ea, eyc],
        });

        let choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        let extraction = Extraction::from_backfill(&eg, eroot, choices);

        let from_arena = EdgeTrace::from_arena_dag(&arena, root);
        let from_egraph = EdgeTrace::from_extraction(&extraction);

        assert_eq!(from_arena.node_count(), 11, "11 distinct nodes");
        assert_eq!(
            from_arena, from_egraph,
            "the arena walk and the e-graph walk must record the identical edge stream"
        );

        // ---------------------------------------------------------------
        // Shift-count pinning (review thread on PR #1019): a Shl/Shr count
        // e-class can legitimately hold both a Const and a value-equal
        // varying-shaped alternative. `choices_to_arena` always pins that
        // child to the Const representative (`pin_shift_counts` — the
        // emitter's shift lowering requires an immediate), so if the
        // extraction chose the varying node, the e-graph walk must describe
        // the PINNED arena, not the raw choice — otherwise it records nodes
        // `choices_to_arena` never emits.
        // ---------------------------------------------------------------
        struct ShlOp;
        impl crate::egraph::ops::Op for ShlOp {
            fn kind(&self) -> OpKind {
                OpKind::Shl
            }
        }

        // The arena `choices_to_arena` will actually materialise: pinning
        // always wins, so the compiled form is `Shl(X, Const(0))` no matter
        // which node the count class's extraction chose.
        let mut arena3 = ExprArena::new();
        let x3 = arena3.push_var(0);
        let zero3 = arena3.push_const(0.0);
        let shl3 = arena3.push_binary(OpKind::Shl, x3, zero3);
        let from_arena3 = EdgeTrace::from_arena_dag(&arena3, shl3);

        let mut eg3 = EGraph::new();
        let ex3 = eg3.add(ENode::Var(0));
        let ey3 = eg3.add(ENode::Var(1));
        let esub3 = eg3.add(ENode::Op {
            op: &ops::Sub,
            children: alloc::vec![ey3, ey3],
        });
        let econst3 = eg3.add(ENode::constant(0.0));
        let count3 = eg3.union(esub3, econst3); // Sub(Y, Y) = 0, same class as Const(0)
        eg3.rebuild();
        let eshl3 = eg3.add(ENode::Op {
            op: &ShlOp,
            children: alloc::vec![ex3, count3],
        });

        // Choose the varying Sub(Y, Y) node for the count class, not the
        // Const — exactly the scenario `pin_shift_counts` exists to correct
        // at emission time.
        let canonical_count3 = eg3.find(count3);
        let sub_idx3 = eg3
            .nodes(canonical_count3)
            .iter()
            .position(|n| matches!(n, ENode::Op { .. }))
            .expect("merged count class holds the Sub node");
        let mut choices3: Vec<Option<usize>> = alloc::vec![None; eg3.num_classes()];
        choices3[eg3.find(eshl3).0 as usize] = Some(0);
        choices3[canonical_count3.0 as usize] = Some(sub_idx3);
        choices3[eg3.find(ex3).0 as usize] = Some(0);
        choices3[eg3.find(ey3).0 as usize] = Some(0);
        let extraction3 = Extraction::from_backfill(&eg3, eshl3, choices3);

        let from_egraph3 = EdgeTrace::from_extraction(&extraction3);

        assert_eq!(
            from_arena3, from_egraph3,
            "the e-graph walk must not walk into Sub(Y, Y) once the count is pinned to \
             Const(0), or its stream will disagree with the arena choices_to_arena emits"
        );
    }

    // =========================================================================
    // Extraction::chosen_variance (2026-09-01: denotation kept, accumulator
    // shape deleted — see the module doc comment above
    // `crate::nnue::factored::variance_histogram`)
    // =========================================================================

    /// A known const/frame/scanline/pixel mix, built once as an arena and
    /// once as the equivalent e-graph, must classify identically through
    /// both entry points — mirroring
    /// `arena_and_extraction_walks_record_the_same_edge_stream` above.
    #[test]
    fn arena_and_extraction_classify_a_known_variance_mix_identically() {
        use crate::nnue::factored::variance_histogram;
        use pixelflow_ir::{ExprArena, OpKind};

        // Add(Add(Const(2.0), W), Add(Y, X)):
        // - Const(2.0)        -> const
        // - W (var 3)         -> frame     (no X, no Y)
        // - Add(Const, W)     -> frame
        // - Y (var 1)         -> scanline  (no X)
        // - X (var 0)         -> pixel
        // - Add(Y, X)         -> pixel     (depends on X)
        // - root Add          -> pixel     (depends on X)
        // 1 const, 2 frame, 1 scanline, 3 pixel of 7 nodes.
        let mut arena = ExprArena::new();
        let c = arena.push_const(2.0);
        let w = arena.push_var(3);
        let frame_sum = arena.push_binary(OpKind::Add, c, w);
        let y = arena.push_var(1);
        let x = arena.push_var(0);
        let xy = arena.push_binary(OpKind::Add, y, x);
        let root = arena.push_binary(OpKind::Add, frame_sum, xy);

        use crate::egraph::ops;
        let mut eg = EGraph::new();
        let ec = eg.add(ENode::constant(2.0));
        let ew = eg.add(ENode::Var(3));
        let efs = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ec, ew],
        });
        let ey = eg.add(ENode::Var(1));
        let ex = eg.add(ENode::Var(0));
        let exy = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ey, ex],
        });
        let eroot = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![efs, exy],
        });

        let choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        let extraction = Extraction::from_backfill(&eg, eroot, choices);

        let from_arena = variance_histogram(&arena);
        let from_extraction = extraction.chosen_variance();

        assert_eq!(
            from_arena, from_extraction,
            "arena and extraction must classify the same DAG identically"
        );
        assert_eq!(
            from_arena,
            [1.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 3.0 / 7.0],
            "known const/frame/scanline/pixel mix: {from_arena:?}"
        );
        let _ = root; // arena root; classification is over every node.
    }

    // =========================================================================
    // choices_to_arena tests
    // =========================================================================

    /// X + Y should produce an arena with exactly 3 nodes: Var(0), Var(1), Add.
    #[test]
    fn choices_to_arena_simple() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let add = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(add).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, add, choices);
        let (arena, root_id) = choices_to_arena(&extraction);

        assert_eq!(arena.len(), 3, "X + Y should have exactly 3 arena nodes");
        // Root should be the last node (post-order: X, Y, Add)
        assert_eq!(root_id.0, 2, "root ExprId should be 2 (the Add node)");
    }

    /// X * X should produce an arena with exactly 2 nodes: Var(0) and Mul.
    /// The shared Var(0) e-class must reuse one ExprId rather than being duplicated.
    #[test]
    fn choices_to_arena_shared() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let mul = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(mul).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, mul, choices);
        let (arena, root_id) = choices_to_arena(&extraction);

        assert_eq!(
            arena.len(),
            2,
            "X * X should have exactly 2 arena nodes (X shared)"
        );
        assert_eq!(root_id.0, 1, "root ExprId should be 1 (the Mul node)");
    }

    /// Direct extraction and explicit `choices_to_arena` should agree for tree-shaped inputs.
    #[test]
    fn extract_matches_choices_to_arena() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let add = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(add).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, add, choices);
        let (arena, root_id) = choices_to_arena(&extraction);
        let (extracted_arena, extracted_root, _cost) = extract(&egraph, add, &CostModel::default());
        assert_eq!(arena.len(), extracted_arena.len());
        assert_eq!(root_id, extracted_root);
    }

    // ========================================================================
    // The reported cost describes the returned term (#1111)
    // ========================================================================

    /// Latency-prior DAG cost of a materialized arena: every reachable
    /// operation priced once, leaves free — the independent statement of
    /// what [`ExtractedDAG::dag_cost`] claims, computed from the arena
    /// instead of from the choices. Deliberately a second implementation:
    /// the point of `dag_cost_equals_the_materialized_arenas_cost` is that
    /// two walks over two representations agree.
    fn arena_dag_cost(
        arena: &pixelflow_ir::ExprArena,
        root: pixelflow_ir::ExprId,
        costs: &CostModel,
    ) -> usize {
        use pixelflow_ir::arena::ExprNode;
        let mut seen = alloc::vec![false; arena.nodes_raw().len()];
        let mut stack = alloc::vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => None,
                ExprNode::Unary(k, _)
                | ExprNode::Binary(k, _, _)
                | ExprNode::Ternary(k, _, _, _) => Some(*k),
                other => panic!("unexpected extracted node {other:?}"),
            };
            if let Some(k) = kind {
                total = total.saturating_add(costs.cost(k));
            }
            stack.extend(arena.children(id));
        }
        total
    }

    /// `sin(X) * sin(X) + sin(X)` — one `Sin` reached three times, so the
    /// tree and DAG costs of the same term are genuinely different numbers.
    /// One e-class, two forms: `sin(X) * sin(X)`, which reuses one `Sin`,
    /// and `ln(X)`, which reuses nothing. The tree objective charges the
    /// `Sin` twice and picks `ln`; the emitted kernel would then compute a
    /// `Ln` where a `Mul` over an already-live `Sin` was cheaper.
    ///
    /// This is #1116 in four e-nodes.
    fn sharing_vs_flat_egraph() -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::Op {
            op: &super::super::ops::Sin,
            children: alloc::vec![x],
        });
        let squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![s, s],
        });
        let flat = egraph.add(ENode::Op {
            op: &super::super::ops::Ln,
            children: alloc::vec![x],
        });
        egraph.union(squared, flat);
        egraph.rebuild();
        let root = egraph.find(squared);
        (egraph, root)
    }

    #[test]
    fn the_sharing_objective_picks_the_form_that_reuses_a_subterm() {
        let (egraph, root) = sharing_vs_flat_egraph();
        let costs = CostModel::latency_prior();

        let sin = costs.cost(pixelflow_ir::OpKind::Sin);
        let mul = costs.cost(pixelflow_ir::OpKind::Mul);
        let ln = costs.cost(pixelflow_ir::OpKind::Ln);
        // The fixture only says anything if `ln` sits strictly between the
        // DAG price of the shared form and its tree price. Assert that here
        // rather than let a cost-table refresh quietly defuse the test.
        assert!(
            mul + sin < ln && ln < mul + sin + sin,
            "fixture defused by the cost table: mul {mul} + sin {sin} vs ln {ln}"
        );

        let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, LatticeShape::POINT);
        assert_eq!(
            tree.dag_cost, ln,
            "the tree objective charges the shared Sin twice and takes the Ln"
        );
        assert_eq!(
            shared.dag_cost,
            mul + sin,
            "the sharing objective prices the Sin once and takes the Mul"
        );
        assert_eq!(
            extract_dag(&egraph, root, &costs).dag_cost,
            mul + sin,
            "extract_dag returns the cheaper of the two by DAG cost"
        );
    }

    /// The no-regression property `extract_dag_scoped` is built on: it
    /// returns the cheaper of the two objectives by true DAG cost, so it can
    /// never be worse than the extractor it replaced.
    #[test]
    fn extract_dag_is_never_worse_than_the_tree_objective_it_replaced() {
        let costs = CostModel::latency_prior();
        for (label, (egraph, root)) in [
            ("shared_sin", shared_sin_egraph()),
            ("sharing_vs_flat", sharing_vs_flat_egraph()),
        ] {
            let (tree, _) = extract_dag_objectives(&egraph, root, &costs, LatticeShape::POINT);
            let chosen = extract_dag(&egraph, root, &costs);
            assert!(
                chosen.dag_cost <= tree.dag_cost,
                "{label}: extract_dag returned {} against the tree arm's {}",
                chosen.dag_cost,
                tree.dag_cost
            );
        }
    }

    /// With nothing shared, the two objectives are the same function, so the
    /// sharing pass must not perturb an unshared kernel at all.
    #[test]
    fn the_two_objectives_agree_when_nothing_is_shared() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let a = egraph.add(ENode::Op {
            op: &super::super::ops::Sin,
            children: alloc::vec![x],
        });
        let b = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![a],
        });
        egraph.rebuild();
        let costs = CostModel::latency_prior();
        let (tree, shared) = extract_dag_objectives(&egraph, b, &costs, LatticeShape::POINT);
        assert_eq!(tree.dag_cost, shared.dag_cost);
        assert_eq!(tree.total_cost, shared.total_cost);
        assert_eq!(tree.choices, shared.choices);
    }

    fn shared_sin_egraph() -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::Op {
            op: &super::super::ops::Sin,
            children: alloc::vec![x],
        });
        let sq = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![s, s],
        });
        let root = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![sq, s],
        });
        egraph.rebuild();
        (egraph, root)
    }

    /// The property every measurement in this repo assumes when it re-costs
    /// the materialized arena rather than reading the extraction's own field
    /// (`runtime.rs`'s `arena_cost`, `ir_bridge.rs`'s namesake, #1101's
    /// harness): `dag_cost` IS that number, so the workaround and the field
    /// agree.
    #[test]
    fn dag_cost_equals_the_materialized_arenas_cost() {
        let (egraph, root) = shared_sin_egraph();
        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, root, &costs);

        let extraction = Extraction::from_dp(&egraph, root, dag.choices.clone());
        let (arena, arena_root) = choices_to_arena(&extraction);

        assert_eq!(
            dag.dag_cost,
            arena_dag_cost(&arena, arena_root, &costs),
            "ExtractedDAG::dag_cost must equal the latency-prior cost of the arena \
             choices_to_arena builds from the same choices"
        );
    }

    /// The two reported numbers are not two spellings of one quantity: the
    /// DP minimizes tree cost, the kernel pays DAG cost, and on a term with
    /// any sharing they differ. (`shader:julia_set` is the extreme: ~1.4e7
    /// against 716 — `docs/results/2026-09-02-extraction-gap.md`.)
    #[test]
    fn tree_cost_prices_a_shared_subterm_once_per_use_and_dag_cost_once() {
        let (egraph, root) = shared_sin_egraph();
        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, root, &costs);

        let sin = costs.cost(pixelflow_ir::OpKind::Sin);
        let mul = costs.cost(pixelflow_ir::OpKind::Mul);
        let add = costs.cost(pixelflow_ir::OpKind::Add);
        assert!(sin > 0, "the fixture needs a Sin that costs something");

        assert_eq!(
            dag.dag_cost,
            sin + mul + add,
            "the DAG cost pays the one shared Sin once"
        );
        assert_eq!(
            dag.total_cost,
            3 * sin + mul + add,
            "the tree cost pays it at each of its three uses"
        );
    }

    /// A cost function that prices `Sin` above `extract_dag_scoped`'s
    /// `CYCLE_COST` sentinel (`usize::MAX / 4`), so a class holding a
    /// self-referential node alongside a `Sin` records the *self-reference*
    /// as its DP minimum. That is the state `repair_choices_well_founded`
    /// exists to rewrite, and the only way to reach it with an off-the-shelf
    /// table would be a lattice big enough to saturate the weighting — this
    /// says the same thing in one line.
    struct SinAboveTheCycleSentinel;

    impl CostFunction for SinAboveTheCycleSentinel {
        fn node_cost(&self, node: &ENode, _parent: Option<pixelflow_ir::OpKind>) -> usize {
            match node {
                ENode::Op { op, .. } if op.kind() == pixelflow_ir::OpKind::Sin => usize::MAX / 2,
                ENode::Op { .. } | ENode::Reduce { .. } => 1,
                ENode::Var(_)
                | ENode::Const(_)
                | ENode::Buffer(_)
                | ENode::Uniform(_)
                | ENode::Param(_) => 0,
            }
        }
    }

    /// `sin(X)` unioned with `neg(sin(X))`, so one class holds both `Sin(x)`
    /// and a `Neg` whose child is that same class. Returns
    /// (egraph, the merged class, index of `Sin` in it, index of `Neg`).
    fn self_referential_pick_egraph() -> (EGraph, EClassId, usize, usize) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::Op {
            op: &super::super::ops::Sin,
            children: alloc::vec![x],
        });
        let n = egraph.add(ENode::Op {
            op: &super::super::ops::Neg,
            children: alloc::vec![s],
        });
        let merged = egraph.union(s, n);
        egraph.rebuild();

        let canonical = egraph.find(merged);
        let nodes = egraph.nodes(canonical);
        let sin_idx = nodes
            .iter()
            .position(
                |nd| matches!(nd, ENode::Op { op, .. } if op.kind() == pixelflow_ir::OpKind::Sin),
            )
            .expect("merged class holds Sin(x)");
        let neg_idx = nodes
            .iter()
            .position(
                |nd| matches!(nd, ENode::Op { op, .. } if op.kind() == pixelflow_ir::OpKind::Neg),
            )
            .expect("merged class holds Neg(merged)");
        (egraph, canonical, sin_idx, neg_idx)
    }

    /// #1111, the regression this fix exists for: when the repair rewrites a
    /// choice, the reported cost must be the cost of the term that comes
    /// back — not the DP's pre-repair number for a term nobody receives.
    ///
    /// Here the DP's minimum for the merged class is the self-referential
    /// `Neg`, priced at `CYCLE_COST` (`usize::MAX / 4`); the repair rewrites
    /// it to the only admissible node, `Sin`, priced at `usize::MAX / 2`.
    /// Reading `best_cost[root]` (what this did before) reports
    /// `usize::MAX / 4` for a returned term that costs twice that.
    #[test]
    fn reported_cost_follows_the_choice_the_repair_rewrote() {
        let (egraph, merged, sin_idx, neg_idx) = self_referential_pick_egraph();
        assert_ne!(sin_idx, neg_idx);

        // What the DP records before the repair: the self-referential Neg,
        // which names no term at all.
        let mut pre_repair: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        pre_repair[merged.0 as usize] = Some(neg_idx);
        assert!(
            choices_have_cycle_from(&egraph, merged, &pre_repair),
            "the fixture must actually put the DP in the state the repair fixes"
        );

        let costs = SinAboveTheCycleSentinel;
        let sin_node = &egraph.nodes(merged)[sin_idx];
        assert!(
            costs.node_cost(sin_node, None) > usize::MAX / 4,
            "the premise: only a Sin priced above CYCLE_COST makes the DP prefer the              self-reference, which is what puts the repair on the path at all"
        );

        let dag = extract_dag(&egraph, merged, &costs);

        assert_eq!(
            dag.choices[merged.0 as usize],
            Some(sin_idx),
            "the repair must have rewritten the self-referential pick"
        );
        assert_eq!(
            dag.total_cost,
            usize::MAX / 2,
            "the reported tree cost must be Sin's, the node actually returned"
        );
        assert_eq!(
            dag.dag_cost,
            usize::MAX / 2,
            "and so must the DAG cost — one op, reached once"
        );
        assert_ne!(
            dag.total_cost,
            usize::MAX / 4,
            "the pre-repair CYCLE_COST total describes a term that was thrown away"
        );

        // The whole term still materializes, and still costs what was said.
        let extraction = Extraction::from_dp(&egraph, merged, dag.choices.clone());
        let (arena, arena_root) = choices_to_arena(&extraction);
        assert!(matches!(
            arena.node(arena_root),
            pixelflow_ir::arena::ExprNode::Unary(pixelflow_ir::OpKind::Sin, _)
        ));
    }

    /// `cost_of_choices` costs the map it is handed and nothing else — a
    /// cyclic map names no term, so it must be an accusation rather than a
    /// number. (Costing the DP's raw table is exactly the #1111 bug.)
    #[test]
    #[should_panic(expected = "CYCLIC")]
    fn cost_of_choices_refuses_a_cyclic_choice_map() {
        let (egraph, merged, _sin_idx, neg_idx) = self_referential_pick_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        choices[merged.0 as usize] = Some(neg_idx);
        let _ = cost_of_choices(
            &egraph,
            merged,
            &choices,
            &CostModel::latency_prior(),
            LatticeShape::POINT,
        );
    }
    // -----------------------------------------------------------------
    // The extractor's objective against the price of what it returns.
    // The hypothesis these pin (JP, 2026-09-08): "the DP's internal cost for
    // the choice map it selects does not equal the true `dag_cost` of the
    // term that map materializes, and the error grows with graph size."
    // It was true of the DFS post-order DP — chrome at a 50,000-class cap
    // claimed 281 for a term costing 4,564,003,324. See
    // docs/results/2026-09-08-cse-mispricing.md.
    // -----------------------------------------------------------------

    /// The claim a DP arm reports is the price of the term it returns, on
    /// that arm's own scale — across sizes and shapes, so a divergence that
    /// only appears on bigger graphs is caught here rather than inferred from
    /// a corpus run.
    #[test]
    fn the_dp_claim_prices_the_term_the_arm_returns() {
        let costs = CostModel::latency_prior();
        for nodes in [64usize, 256, 1024, 4096] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            for shape in [LatticeShape::POINT, LatticeShape::new([256, 256])] {
                let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, shape);
                for (label, dag) in [("tree arm", &tree), ("shared arm", &shared)] {
                    let audit = dag.report.audit.expect("both arms run a DP");
                    assert_eq!(
                        audit.claimed,
                        audit.scale.of(dag.cost()),
                        "{label} at {nodes} nodes / {shape:?}: claimed {} on the {:?} scale \
                         but the term costs tree {} / dag {}",
                        audit.claimed,
                        audit.scale,
                        dag.total_cost,
                        dag.dag_cost,
                    );
                }
                // The sharing-aware arm's scale is the one the kernel pays,
                // so its claim IS `dag_cost` — the property #1116 bought and
                // the post-order traversal took back.
                let audit = shared.report.audit.expect("the shared arm runs a DP");
                assert_eq!(audit.scale, CostScale::Dag);
                assert_eq!(audit.claimed, shared.dag_cost);
                assert_eq!(audit.signed_error(shared.cost()), 0);
            }
        }
    }

    /// `extract_dag_scoped` chooses between the arms on **one** scale — the
    /// true `dag_cost` of each term — never on the arms' own DP tables, which
    /// are on different scales and can be saturated besides.
    #[test]
    fn the_arms_are_compared_on_the_price_not_on_their_claims() {
        let costs = CostModel::latency_prior();
        for nodes in [64usize, 256, 1024, 4096] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            for shape in [LatticeShape::POINT, LatticeShape::new([256, 256])] {
                let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, shape);
                let scoped = extract_dag_scoped(&egraph, root, &costs, shape);
                assert_eq!(
                    scoped.dag_cost,
                    tree.dag_cost.min(shared.dag_cost),
                    "{nodes} nodes at {shape:?}: production returned {} where the cheaper arm \
                     costs tree {} / shared {}",
                    scoped.dag_cost,
                    tree.dag_cost,
                    shared.dag_cost
                );
                // Ties go to the tree arm, and only ties: a `Shared` verdict
                // means the sharing-aware term was *strictly* cheaper.
                match scoped.report.objective {
                    ExtractionObjective::Shared => {
                        assert!(shared.dag_cost < tree.dag_cost);
                        assert_eq!(scoped.choices, shared.choices);
                    }
                    ExtractionObjective::TreeCheaper => {
                        assert!(tree.dag_cost <= shared.dag_cost);
                        assert_eq!(scoped.choices, tree.choices);
                    }
                    other => panic!("{nodes} nodes: unexpected objective {other:?}"),
                }
            }
        }
    }

    /// The tree arm's DP objective **saturates** — its claim reaches
    /// `usize::MAX` on a graph this size, at which point every candidate ties
    /// at the ceiling and the settling keeps whichever the tie-break prefers.
    /// A degenerate objective, pinned so the fact is a test rather than a
    /// surprise.
    ///
    /// It cannot degrade production: the arms are chosen between by true
    /// `dag_cost` (the test above), so a saturated tree claim buys the tree
    /// arm nothing. The pin is what keeps that reasoning honest — if the
    /// comparison ever moved onto the arms' own tables, this says what it
    /// would be comparing.
    #[test]
    fn the_tree_arms_objective_saturates_on_a_real_sized_graph() {
        let costs = CostModel::latency_prior();
        let (egraph, root) = saturated_sdf_egraph(256);
        let tree = extract_dag_tree_arm(&egraph, root, &costs, LatticeShape::POINT);
        let audit = tree.report.audit.expect("the tree arm runs a DP");
        assert_eq!(audit.scale, CostScale::Tree);
        assert!(
            audit.claimed > 1e18 as usize,
            "the tree objective was expected at or near its ceiling, not {}",
            audit.claimed
        );
        assert_eq!(audit.claimed, tree.total_cost);
        // ...while the price of that same term is an ordinary number.
        assert!(
            tree.dag_cost < 1_000_000,
            "dag_cost should be a real number, not a ceiling: {}",
            tree.dag_cost
        );
    }

    /// A class two paths reach is priced **once** by the sharing-aware DP —
    /// not twice (the tree objective's error) and not zero times (a reach set
    /// that lost a member, which is what the post-order DP did to every class
    /// it left at the sentinel). Checked against a hand-computed sum so the
    /// test does not restate the implementation.
    #[test]
    fn the_shared_dp_prices_a_doubly_reached_class_exactly_once() {
        // sin(X) * sin(X): the `sin` class is reached by both children.
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::Op {
            op: &super::super::ops::Sin,
            children: alloc::vec![x],
        });
        let root = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![s, s],
        });
        egraph.rebuild();
        let costs = CostModel::latency_prior();
        let shape = LatticeShape::POINT;
        let dp = shared_dag_dp_pass(
            &egraph,
            root,
            &mut Dp::production(&costs, shape),
            usize::MAX,
        )
        .outcome
        .expect("unbounded");

        let own = |c: EClassId| -> usize {
            weighted_own(
                &costs,
                &egraph.nodes(c)[dp.choices[c.0 as usize].unwrap()],
                1,
            )
        };
        assert_eq!(
            dp.root_cost,
            own(root) + own(s) + own(x),
            "the shared class must enter the sum once"
        );
        // The tree objective is the same sum with `sin(X)` charged twice.
        let tree = tree_dp_pass(&egraph, root, &mut Dp::production(&costs, shape));
        assert_eq!(tree.root_cost, own(root) + 2 * (own(s) + own(x)));
    }

    /// Over [`SHARED_DAG_PASS_BYTE_BUDGET`] the pass returns no map and the
    /// caller reports `TreeOnly` **on the `Tree` scale** — never a
    /// shared-priced answer under a tree label.
    #[test]
    fn the_budget_fallback_reports_the_tree_scale_it_actually_used() {
        const CHAIN: usize = 2_000;
        let (egraph, root) = add_chain(CHAIN);
        let costs = CostModel::latency_prior();
        let shape = LatticeShape::POINT;
        let full = shared_dag_dp_pass(
            &egraph,
            root,
            &mut Dp::production(&costs, shape),
            usize::MAX,
        );
        let budget = full.stats.reach_bytes / 2;
        assert!(
            shared_dag_dp_pass(&egraph, root, &mut Dp::production(&costs, shape), budget)
                .outcome
                .is_none()
        );

        let tree = extract_dag_tree_arm(&egraph, root, &costs, shape);
        assert_eq!(tree.report.objective, ExtractionObjective::TreeOnly);
        let audit = tree.report.audit.expect("the tree arm runs a DP");
        assert_eq!(audit.scale, CostScale::Tree);
        assert_eq!(audit.claimed, tree.total_cost);
    }

    /// The smallest e-graph whose DFS post-order asks a DP to price a class
    /// before any form of it exists — and, before extraction was settled in
    /// cost order, the smallest one on which the sharing-aware pass returned
    /// a term costing more than it claimed.
    ///
    /// ```text
    /// R = { Mul(Z, W) , Cos(C) }     Z = { Neg(P) }     W = Var(2)
    /// C = { Cos(D)    , Sin(P) }     P = { Sqrt(C) }    D = Var(1)
    /// ```
    ///
    /// `C` and `P` are mutually reachable, so a DFS entering `C` through
    /// `Sin(P)` opens `P` while `C` is still on the stack. The post-order DP
    /// settled `P` there, priced it at the cycle sentinel and gave it a reach
    /// set of `{P}` — and `C`, which only the chosen term's `P` reaches, went
    /// unpaid.
    fn blind_dfs_egraph() -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let d = egraph.add(ENode::Var(1));
        let w = egraph.add(ENode::Var(2));
        let c0 = egraph.add(ENode::Op {
            op: &super::super::ops::Cos,
            children: alloc::vec![d],
        });
        let p = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![c0],
        });
        let c1 = egraph.add(ENode::Op {
            op: &super::super::ops::Sin,
            children: alloc::vec![p],
        });
        let c = egraph.union(c0, c1);
        let z = egraph.add(ENode::Op {
            op: &super::super::ops::Neg,
            children: alloc::vec![p],
        });
        let r0 = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![z, w],
        });
        let r1 = egraph.add(ENode::Op {
            op: &super::super::ops::Cos,
            children: alloc::vec![c],
        });
        let root = egraph.union(r0, r1);
        egraph.rebuild();
        (egraph, root)
    }

    /// The claim/price identity on the graph a DFS order could not price —
    /// exactly where it used to fail, and the premise is asserted rather than
    /// remembered, so a fixture that stops exercising the case says so.
    #[test]
    fn the_claim_is_exact_on_the_graph_a_dfs_order_could_not_price() {
        let costs = CostModel::latency_prior();
        let (egraph, root) = blind_dfs_egraph();

        // Premise: some class of this graph has no form whose children the
        // DFS post-order settles before it.
        let mut settled = alloc::vec![false; egraph.num_classes()];
        let mut blind = 0usize;
        for class in post_order(&egraph, root) {
            let has_form = egraph.nodes(class).iter().any(|node| match node {
                ENode::Op { children, .. } => {
                    children.iter().all(|&c| settled[egraph.find(c).0 as usize])
                }
                _ => true,
            });
            if !has_form {
                blind += 1;
            }
            settled[class.0 as usize] = true;
        }
        assert!(blind > 0, "premise: the DFS order is blind on this fixture");

        for shape in [LatticeShape::POINT, LatticeShape::new([1920, 1080])] {
            let dag = extract_dag_scoped(&egraph, root, &costs, shape);
            let audit = dag.report.audit.expect("a DP ran");
            assert_eq!(
                audit.claimed,
                audit.scale.of(dag.cost()),
                "at {shape:?}: claimed {} against a term costing tree {} / dag {}",
                audit.claimed,
                dag.total_cost,
                dag.dag_cost
            );
            assert_eq!(
                cost_of_choices(&egraph, root, &dag.choices, &costs, shape),
                dag.cost(),
                "the reported pair must be `cost_of_choices` of the returned map"
            );
        }
    }
}
