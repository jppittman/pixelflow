//! Extraction: materialise a concrete arena expression from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model and materialises it as an
//! [`pixelflow_ir::ExprArena`].

use super::cost::{CostFunction, CostModel};
use super::deps::var_variance;
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use alloc::vec::Vec;
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
// NNUE tied the static table on schedule-free kernels
// (docs/paper/2026-08-egraph-nnue-parity.md) — but JP's ruling on the
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
                    if let ENode::Op { children, .. } = &nodes[node_idx] {
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
        use alloc::collections::BTreeSet;

        let mut active = Vec::new();
        let mut visited = BTreeSet::new();
        let mut stack = vec![root];

        while let Some(class) = stack.pop() {
            let canonical = egraph.find(class);
            if !visited.insert(canonical.0) {
                continue;
            }

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
                if let ENode::Op { children, .. } = &nodes[node_idx] {
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
            if let ENode::Op { children, .. } = node {
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
            if let ENode::Op { children, .. } = node {
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
fn repair_choices_well_founded(egraph: &EGraph, root: EClassId, choices: &mut [Option<usize>]) {
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
            if let ENode::Op { children, .. } = node {
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
            if let ENode::Op { children, .. } = node {
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
    let mut admit = |pos: usize,
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
    use alloc::collections::BTreeSet;

    // Cap for cycle/self-referential costs - high but not astronomical
    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Iterative bottom-up cost computation using topological order
    // We use a work stack to avoid recursion
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)]; // (class, children_processed)
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        // Already computed
        if best_cost[canonical.0 as usize].is_some() {
            continue;
        }

        if !children_done {
            // First visit: push self back (to process after children), then push children
            if !on_stack.insert(canonical.0) {
                // Cycle detected - don't cache, parent will handle with high cost
                continue;
            }

            stack.push((canonical, true)); // Come back after children

            // Push all children that need processing
            for node in egraph.nodes(canonical) {
                if let ENode::Op { children, .. } = node {
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
            on_stack.remove(&canonical.0);

            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => {
                        costs.node_cost(node, None)
                    }
                    ENode::Op { children, .. } => {
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
                    if let ENode::Op { children, .. } = &nodes[node_idx] {
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
                if let ENode::Op { children, .. } = node {
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
    use pixelflow_ir::{ExprArena, ExprId};

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
    // Buffer identity → declared slot in the output arena. Distinct e-classes
    // can carry the same identity only if their decls differ, which is a
    // corrupt graph (one memory, two extents) — assert, never alias silently.
    let mut buffer_slots: alloc::collections::BTreeMap<
        pixelflow_ir::arena::BufferIdentity,
        (
            pixelflow_ir::arena::BufferId,
            pixelflow_ir::arena::BufferDecl,
        ),
    > = alloc::collections::BTreeMap::new();
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
                        let expr_id = arena.push_var(*var_idx);
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Const(bits) => {
                        let expr_id = arena.push_const(f32::from_bits(*bits));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Buffer(decl) => {
                        // One slot per distinct identity: e-classes already
                        // dedupe equal decls, so a repeat identity here means
                        // two decls disagreeing on extents.
                        let buf_id = match buffer_slots.get(&decl.id) {
                            Some(&(buf_id, prior)) => {
                                assert!(
                                    prior == *decl,
                                    "choices_to_arena: BufferIdentity declared twice with \
                                     different extents ({prior:?} vs {decl:?})"
                                );
                                buf_id
                            }
                            None => {
                                let buf_id = arena.declare_buffer(*decl);
                                buffer_slots.insert(decl.id, (buf_id, *decl));
                                buf_id
                            }
                        };
                        let expr_id = arena.push_buffer(buf_id);
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Op { children, .. } => {
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

                let op_kind = op.kind();

                let expr_id = match arity {
                    0 => arena.push_const(0.0), // Degenerate zero-arity Op — treat as 0.
                    1 => arena.push_unary(op_kind, child_ids[0]),
                    2 => arena.push_binary(op_kind, child_ids[0], child_ids[1]),
                    3 => arena.push_ternary(op_kind, child_ids[0], child_ids[1], child_ids[2]),
                    _ => arena.push_nary(op_kind, &child_ids),
                };

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
        ENode::Const(_) | ENode::Buffer(_) => Variance::CONST,
        ENode::Op { children, .. } => children.iter().fold(Variance::CONST, |acc, &child| {
            let c = egraph.find(child);
            if c == canonical {
                return Variance::ALL;
            }
            acc.union(best_var[c.0 as usize])
        }),
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
            if let ENode::Op { children, .. } = chosen(canonical) {
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
        let children_cost = match node {
            ENode::Op { children, .. } => children
                .iter()
                .map(|&child| {
                    let c = egraph.find(child).0 as usize;
                    tree[c].expect("post-order visits every child before its parent")
                })
                .fold(0usize, usize::saturating_add),
            ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => 0,
        };
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
/// extraction walks once per compile.
pub fn extract_dag_scoped<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let tree = repaired_and_costed(
        egraph,
        root,
        tree_dp_pass(egraph, root, costs, shape),
        costs,
        shape,
    );
    let shared = repaired_and_costed(
        egraph,
        root,
        shared_dag_dp_pass(egraph, root, costs, shape),
        costs,
        shape,
    );
    // Only the winner is assembled: the reference counts and the emission
    // schedule describe a term, and one of these two is not going to be one.
    assemble(
        egraph,
        root,
        if shared.cost.dag < tree.cost.dag {
            shared
        } else {
            tree
        },
    )
}

/// A repaired choice map and the cost of the term it names.
struct CostedChoices {
    choices: Vec<Option<usize>>,
    cost: ChoiceCost,
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
    let costed = repaired_and_costed(
        egraph,
        root,
        tree_dp_pass(egraph, root, costs, shape),
        costs,
        shape,
    );
    assemble(egraph, root, costed)
}

/// The sharing-aware arm on its own (#1116).
pub(crate) fn extract_dag_shared_arm<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let costed = repaired_and_costed(
        egraph,
        root,
        shared_dag_dp_pass(egraph, root, costs, shape),
        costs,
        shape,
    );
    assemble(egraph, root, costed)
}

/// Repair a raw DP choice map and cost the term it names.
fn repaired_and_costed<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    mut choices: Vec<Option<usize>>,
    costs: &C,
    shape: LatticeShape,
) -> CostedChoices {
    // Repair any mutual cycles in the choice graph before anything reads it.
    repair_choices_well_founded(egraph, root, &mut choices);

    // Cost the choices we are about to RETURN, not the DP table that produced
    // them: the repair can switch a class to a different node, and reading the
    // DP's own total here (as this did before #1111) reported the cost of a
    // term that is not the one returned — measurably so, on 132 of 302
    // kernels. The recomputed number is also free of the `CYCLE_COST`
    // inflation, since the repaired map is well-founded and holds no
    // self-referential pick.
    let cost = cost_of_choices(egraph, root, &choices, costs, shape);
    CostedChoices { choices, cost }
}

/// Build the sharing and emission schedule around a settled choice map.
fn assemble(egraph: &EGraph, root: EClassId, costed: CostedChoices) -> ExtractedDAG {
    let CostedChoices { choices, cost } = costed;
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
    }
}

// Above any weighted cost a real program can reach, so a self-reference never
// looks cheaper than an expensive-but-legitimate form. (A flat 1_000_000 was
// safely above every *unweighted* cost; weighting by a frame's sample count
// clears that by orders of magnitude.)
const CYCLE_COST: usize = usize::MAX / 4;

/// One node's weighted own cost under `shape`.
fn weighted_own<C: CostFunction>(costs: &C, node: &ENode, weight: u64) -> usize {
    usize::try_from((costs.node_cost(node, None) as u64).saturating_mul(weight))
        .unwrap_or(usize::MAX)
}

/// The pre-#1116 DP: cheapest node per class where a child costs its whole
/// subtree, at every use. Kept exactly as it was, as the control arm of the
/// objective A/B and as the floor [`extract_dag_scoped`] never returns worse
/// than.
fn tree_dp_pass<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> Vec<Option<usize>> {
    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];
    // The variance of the form chosen for each class, which is what its
    // scope — and so its weight — is read from. Carried in the same DP as
    // the cost because the two determine each other: a child's variance sets
    // its parent's weight, and a parent's weight is part of what makes one
    // child's form worth choosing over another's.
    let mut best_var: Vec<Variance> = alloc::vec![Variance::CONST; num_classes];

    for canonical in post_order(egraph, root) {
        let nodes = egraph.nodes(canonical);
        let mut min_cost = usize::MAX;
        let mut min_idx = 0;
        let mut min_var = Variance::CONST;

        for (idx, node) in nodes.iter().enumerate() {
            let node_var = node_variance(egraph, node, &best_var, canonical);
            let weight = shape.evals(node_var);
            let this_node_cost = match node {
                ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => {
                    weighted_own(costs, node, weight)
                }
                ENode::Op { children, .. } => {
                    if children.iter().any(|&c| egraph.find(c) == canonical) {
                        CYCLE_COST
                    } else {
                        // Saturating fold, not `.sum()`: a child's own
                        // `best_cost` can already sit at a prohibitive
                        // sentinel (`Dwrt`'s `usize::MAX / 4` from
                        // `CostModel::node_op_cost`, or this function's own
                        // `CYCLE_COST`), so a node with several such children
                        // overflows a plain `usize` sum.
                        let children_cost: usize = children
                            .iter()
                            .map(|&child| {
                                let c = egraph.find(child);
                                best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                            })
                            .fold(0usize, usize::saturating_add);
                        weighted_own(costs, node, weight).saturating_add(children_cost)
                    }
                }
            };

            if this_node_cost < min_cost {
                min_cost = this_node_cost;
                min_idx = idx;
                min_var = node_var;
            }
        }

        best_cost[canonical.0 as usize] = Some(min_cost);
        best_node[canonical.0 as usize] = Some(min_idx);
        best_var[canonical.0 as usize] = min_var;
    }

    best_node
}

/// The sharing-aware DP (#1116): cheapest node per class where the cost of a
/// candidate is the cost of **the set of classes its sub-DAG contains**, each
/// member priced once.
///
/// Same skeleton as [`tree_dp_pass`]; the only change is what a candidate
/// costs. Each class carries a bitset of the classes its chosen sub-DAG
/// reaches, and a candidate unions its children's bitsets, adding a class's
/// own cost the first time that class enters the union. Two siblings that
/// both reach `sin(X)` therefore pay for it once, which is what the emitted
/// kernel does: `choices_to_arena` materializes one node per reachable class
/// and codegen let-binds the shared ones.
///
/// Space is one bit per REACHABLE class per reachable class. A production
/// preset caps the e-graph at 10,000 classes (`saturate.rs`), so the ceiling
/// is ~12.5 MB; a median production glyph reaches 1,755 of them and uses
/// ~385 KB, allocated once per extraction and dropped at the end of it.
fn shared_dag_dp_pass<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> Vec<Option<usize>> {
    const BITS: usize = usize::BITS as usize;

    let num_classes = egraph.num_classes();
    let order = post_order(egraph, root);

    // Index the bitsets by position in `order`, not by e-class id: the sets
    // only ever hold classes the root reaches, and on a saturated glyph that
    // is a third of the e-graph (1,352 of 4,703 on `glyph16:U+0021`). Sizing
    // them by `num_classes` would pay for the rest of the graph in every
    // union.
    let live = order.len();
    let words = live.div_ceil(BITS);
    let mut compact: Vec<u32> = alloc::vec![u32::MAX; num_classes];
    for (i, c) in order.iter().enumerate() {
        compact[c.0 as usize] = i as u32;
    }

    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_var: Vec<Variance> = alloc::vec![Variance::CONST; num_classes];
    // The weighted own cost of each live class's chosen node — what a union
    // pays when that class first enters it. Indexed by compact id.
    let mut best_own: Vec<usize> = alloc::vec![0; live];
    // `reach[i * words .. (i + 1) * words]` is the set of classes the chosen
    // sub-DAG at the `i`th live class contains, itself included.
    let mut reach: Vec<usize> = alloc::vec![0; live.saturating_mul(words)];
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
            let weight = shape.evals(node_var);
            let own = weighted_own(costs, node, weight);
            let this_node_cost = match node {
                ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => own,
                ENode::Op { children, .. } => {
                    if children.iter().any(|&c| egraph.find(c) == canonical) {
                        CYCLE_COST
                    } else {
                        scratch.fill(0);
                        let mut below = 0usize;
                        let mut unresolved = false;
                        for &child in children.iter() {
                            let c = egraph.find(child).0 as usize;
                            if best_cost[c].is_none() {
                                // A class still on the DFS stack: the same
                                // cycle the tree pass prices at the sentinel.
                                unresolved = true;
                                break;
                            }
                            let ci = compact[c];
                            assert!(
                                ci != u32::MAX,
                                "shared_dag_dp_pass: e-class {c} is a costed child but was \
                                 not enumerated by post_order — the two traversals have \
                                 drifted"
                            );
                            let base = ci as usize * words;
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

        // Rebuild the winner's reach set. Recomputing it costs one more union
        // over the winning node's children and saves keeping a full bitset
        // per candidate alive through the loop above.
        let base = me * words;
        reach[base..base + words].fill(0);
        if let ENode::Op { children, .. } = &nodes[min_idx] {
            if min_cost != CYCLE_COST {
                for &child in children.iter() {
                    let ci = compact[egraph.find(child).0 as usize] as usize;
                    let cbase = ci * words;
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

    best_node
}

/// The classes reachable from `root`, children before parents.
///
/// A class whose own descendants reach it back appears before them — the
/// e-graphs saturation produces are cyclic (commutativity alone is enough),
/// and the DP prices such a class at the cycle sentinel exactly as it always
/// has. Shared by both DP passes so their traversal, and therefore which
/// classes end up cycle-priced, cannot drift apart.
fn post_order(egraph: &EGraph, root: EClassId) -> Vec<EClassId> {
    use alloc::collections::BTreeSet;

    let mut order: Vec<EClassId> = Vec::new();
    let mut settled: BTreeSet<u32> = BTreeSet::new();
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if settled.contains(&canonical.0) {
            continue;
        }

        if !children_done {
            if !on_stack.insert(canonical.0) {
                continue;
            }

            stack.push((canonical, true));

            for node in egraph.nodes(canonical) {
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_canonical = egraph.find(child);
                        if !settled.contains(&child_canonical.0) {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            on_stack.remove(&canonical.0);
            settled.insert(canonical.0);
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
                if let ENode::Op { children, .. } = node {
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
    use alloc::collections::BTreeSet;

    let shared_set: BTreeSet<u32> = shared.iter().map(|(id, _)| id.0).collect();
    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut result = Vec::new();

    // Iterative post-order: (class, children_pushed)
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if visited.contains(&canonical.0) {
            continue;
        }

        if !children_done {
            stack.push((canonical, true));

            if let Some(node_idx) = best_node.get(canonical.0 as usize).and_then(|o| *o) {
                let node = &egraph.nodes(canonical)[node_idx];
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_can = egraph.find(child);
                        if !visited.contains(&child_can.0) {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            visited.insert(canonical.0);

            if shared_set.contains(&canonical.0) {
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
            chosen(LatticeShape::new([256, 256, 1, 1])),
            OpKind::Add,
            "over a frame the fused form pays for Z at every sample"
        );
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
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Buffer(_) => None,
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
                ENode::Op { .. } => 1,
                ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => 0,
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
}
