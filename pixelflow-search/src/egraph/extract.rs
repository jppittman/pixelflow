//! Extraction: materialise a concrete arena expression from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model and materialises it as an
//! [`pixelflow_ir::ExprArena`].

use super::cost::{CostFunction, CostModel};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use crate::nnue::{EdgeAccumulator, ExprNnue};
use alloc::vec::Vec;

/// Incremental 3-pass neural extractor.
///
/// Instead of recomputing the full `EdgeAccumulator` from scratch for each
/// candidate, performs an O(Δ) swap: remove old subtree edges, add new.
/// This makes each candidate evaluation O(subtree_size) instead of O(whole_tree).
///
/// ## Algorithm
///
/// - **Pass 1 (Bootstrap)**: extract shallowest tree (minimum AST node count per
///   e-class). This is fast and gives a reasonable starting point.
/// - **Passes 2-3 (Refine)**: for each active e-class, try alternative nodes
///   using O(Δ) accumulator updates. Accept if strictly lower cost.
///   Repeat until fixpoint or `MAX_PASSES` (3) reached.
pub struct IncrementalExtractor<'a> {
    nnue: &'a ExprNnue,
    top_k: usize,
}

impl<'a> IncrementalExtractor<'a> {
    /// Create a new incremental extractor.
    ///
    /// `top_k` bounds how many alternative nodes per e-class are evaluated
    /// during refinement passes (default 8 is a good trade-off).
    pub fn new(nnue: &'a ExprNnue, top_k: usize) -> Self {
        Self { nnue, top_k }
    }

    /// Run the extraction refinement loop and return only `(cost, choices)`.
    ///
    /// The `choices` vector maps canonical e-class ID to the chosen node index.
    /// Call [`choices_to_arena`] to materialise the extracted DAG.
    pub fn extract_choices_only(
        &self,
        egraph: &EGraph,
        root_class: EClassId,
    ) -> (f32, Vec<Option<usize>>) {
        const MAX_PASSES: usize = 10;

        // Pass 1: Bootstrap with the ORIGINAL expression's nodes (index 0 per
        // e-class = the first node added, which is the original). This ensures
        // we start from the input expression and only move to alternatives the
        // NNUE extraction head certifies as cheaper.
        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        backfill_reachable_defaults(egraph, root_class, &mut choices);

        // Run variance analysis once — O(n) over e-graph, provides
        // per-e-class coordinate dependency info to the extraction head.
        let variance_analysis = super::deps::DepsAnalysis::analyze(egraph);

        // Build initial DAG-aware accumulator using ref counts.
        // This avoids the tree-bloating problem: shared subexpressions are
        // counted once (computation) + (ref_count-1) var_ref edges (register loads).
        let ref_count = compute_ref_counts(egraph, root_class, &choices);
        let current_acc = EdgeAccumulator::from_dag_choices_with_variance(
            egraph,
            root_class,
            &choices,
            &ref_count,
            &self.nnue.embeddings,
            Some(&variance_analysis),
        );
        let mut current_cost = self.nnue.predict_log_cost_with_features(&current_acc);

        // Refinement passes: for each e-class, try ALL alternatives (up to top_k),
        // accept the BEST improvement (not first). Repeat until fixpoint or max passes.
        //
        // DAG-aware: each swap may change ref counts (new children may be shared
        // differently), so we rebuild the accumulator from scratch for each candidate.
        // This is O(reachable_classes) per candidate, same as the old tree-based path,
        // but now sharing-aware. True incremental updates can be added later.
        for _pass in 0..MAX_PASSES {
            let active = self.get_active_classes(egraph, root_class, &choices);
            let mut improved = false;

            for &class in &active {
                let canonical = egraph.find(class);
                let nodes = egraph.nodes(canonical);
                if nodes.len() <= 1 {
                    continue;
                }

                let current_node_idx = choices[canonical.0 as usize].unwrap_or_else(|| {
                    panic!(
                        "extract_choices_only: e-class {} is active (reachable from root) \
                         but has no recorded choice — backfill_reachable_defaults should have \
                         populated every class returned by get_active_classes",
                        canonical.0
                    )
                });
                let candidates_to_try = nodes.len().min(self.top_k);

                // Best-improvement: evaluate ALL candidates, pick the cheapest.
                let mut best_swap_cost = current_cost;
                let mut best_swap_idx: Option<usize> = None;

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

                    // Tentatively apply the swap. First check it doesn't create a cycle.
                    let old_choice = choices[canonical.0 as usize];
                    choices[canonical.0 as usize] = Some(node_idx);
                    if choices_have_cycle_from(egraph, root_class, &choices) {
                        choices[canonical.0 as usize] = old_choice;
                        continue;
                    }

                    let test_refs = compute_ref_counts(egraph, root_class, &choices);
                    let test_acc = EdgeAccumulator::from_dag_choices_with_variance(
                        egraph,
                        root_class,
                        &choices,
                        &test_refs,
                        &self.nnue.embeddings,
                        Some(&variance_analysis),
                    );
                    let test_cost = self.nnue.predict_log_cost_with_features(&test_acc);

                    // Restore original choice.
                    choices[canonical.0 as usize] = old_choice;

                    if test_cost < best_swap_cost {
                        best_swap_cost = test_cost;
                        best_swap_idx = Some(node_idx);
                    }
                }

                if let Some(idx) = best_swap_idx {
                    choices[canonical.0 as usize] = Some(idx);
                    current_cost = best_swap_cost;
                    improved = true;

                    // The newly-chosen node may have children in e-classes
                    // that weren't reachable before (saturation merges can
                    // introduce new children). Backfill the ENTIRE newly
                    // reachable subtree (not just direct children) so the
                    // invariant "every class in `active` has Some(choice)"
                    // actually holds — a shallow, direct-children-only
                    // backfill left grandchildren `None`, which callers
                    // like `get_active_classes` and the refinement loop
                    // below were then silently defaulting to node 0 for,
                    // masking a real gap instead of restoring it.
                    let nodes = egraph.nodes(canonical);
                    if let Some(ENode::Op { children, .. }) = nodes.get(idx) {
                        for &child in children {
                            backfill_reachable_defaults(egraph, child, &mut choices);
                        }
                    }
                }
            }

            if !improved {
                break; // Fixpoint
            }
        }

        (current_cost, choices)
    }

    /// Pass 1: Bottom-up DP choosing the node with fewest total AST nodes.
    fn extract_shallowest(&self, egraph: &EGraph, root: EClassId) -> Vec<Option<usize>> {
        use alloc::collections::BTreeSet;

        const CYCLE_COUNT: usize = 1_000_000;

        let num_classes = egraph.num_classes();
        let mut best_count: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

        let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];
        let mut on_stack: BTreeSet<u32> = BTreeSet::new();

        while let Some((class, children_done)) = stack.pop() {
            let canonical = egraph.find(class);

            if best_count[canonical.0 as usize].is_some() {
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
                            let child_can = egraph.find(child);
                            if best_count[child_can.0 as usize].is_none() {
                                stack.push((child, false));
                            }
                        }
                    }
                }
            } else {
                on_stack.remove(&canonical.0);

                let nodes = egraph.nodes(canonical);
                let mut min_count = usize::MAX;
                let mut min_idx = 0;

                for (idx, node) in nodes.iter().enumerate() {
                    let count = match node {
                        ENode::Var(_) | ENode::Const(_) => 1,
                        ENode::Op { children, .. } => {
                            if children.iter().any(|&c| egraph.find(c) == canonical) {
                                CYCLE_COUNT
                            } else {
                                let child_sum: usize = children
                                    .iter()
                                    .map(|&c| {
                                        let cc = egraph.find(c);
                                        best_count[cc.0 as usize].unwrap_or(CYCLE_COUNT)
                                    })
                                    .sum();
                                1usize.saturating_add(child_sum)
                            }
                        }
                    };

                    if count < min_count {
                        min_count = count;
                        min_idx = idx;
                    }
                }

                best_count[canonical.0 as usize] = Some(min_count);
                best_node[canonical.0 as usize] = Some(min_idx);
            }
        }

        break_choice_cycles(egraph, root, &mut best_node);

        best_node
    }

    /// Walk the current best tree and collect active (reachable) e-class IDs.
    fn get_active_classes(
        &self,
        egraph: &EGraph,
        root: EClassId,
        choices: &[Option<usize>],
    ) -> Vec<EClassId> {
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

            let node_idx = choices[canonical.0 as usize].unwrap_or_else(|| {
                panic!(
                    "get_active_classes: e-class {} reachable from root has no recorded \
                     choice — extract_choices_only must call backfill_reachable_defaults \
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

/// Transitively fill in `Some(0)` (the original/first node) for every
/// e-class reachable from `start` that doesn't yet have a recorded choice.
///
/// This restores the invariant relied on throughout `extract_choices_only`
/// and its helpers (`get_active_classes`, the refinement loop, and
/// `choices_to_arena`): every e-class reachable from the root via the
/// *currently chosen* nodes has `Some` entry in `choices`. Saturation merges
/// and NNUE-guided swaps can both introduce children that were not part of
/// the original bootstrap walk; if those children are left `None`, callers
/// fall back to `unwrap_or(0)`, which silently (and possibly incorrectly,
/// since node 0 may not be the reachable/consistent variant) treats an
/// unrecorded choice as if it were recorded. Making the backfill transitive
/// here means that fallback becomes unreachable in practice, and any future
/// gap is a real bug caught loudly rather than papered over downstream.
///
/// Already-visited classes (`choices[idx].is_some()`) stop the walk — this
/// keeps the traversal to genuinely new subtrees.
fn backfill_reachable_defaults(egraph: &EGraph, start: EClassId, choices: &mut [Option<usize>]) {
    let num_classes = choices.len();
    let mut stack = alloc::vec![start];

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || choices[idx].is_some() {
            continue;
        }
        choices[idx] = Some(0); // Original/first node in the e-class.
        if let Some(node) = egraph.nodes(canonical).first() {
            if let ENode::Op { children, .. } = node {
                for &child in children {
                    stack.push(child);
                }
            }
        }
    }
}

/// Extract directly into an [`pixelflow_ir::ExprArena`].
#[must_use]
pub fn extract_neural_to_arena(
    egraph: &EGraph,
    root: EClassId,
    nnue: &ExprNnue,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, f32) {
    let extractor = IncrementalExtractor::new(nnue, 8);
    let (cost, choices) = extractor.extract_choices_only(egraph, root);
    let (arena, root_id) = choices_to_arena(egraph, root, &choices);
    (arena, root_id, cost)
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

        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        if let Some(ENode::Op { children, .. }) = egraph.nodes(canonical).get(node_idx) {
            for &child in children.iter().rev() {
                stack.push((child, false));
            }
        }
    }

    false
}

/// Post-pass: detect and break cycles in the choice graph.
///
/// After the bottom-up DP, mutual cycles can exist (e.g. class 68 picks
/// neg(69) and class 69 picks neg(68)). This function performs a DFS from
/// `root` following the choice graph, detects back-edges via 3-color
/// marking (white/gray/black), and breaks each cycle by swapping a cycle
/// member's choice to a leaf or a non-cycle-referencing node.
///
/// Restarts from root after each break to handle nested cycles.
fn break_choice_cycles(egraph: &EGraph, root: EClassId, best_node: &mut Vec<Option<usize>>) {
    // Restart-DFS approach (like the original) but with dense Vecs instead
    // of BTreeSet. Each iteration finds one cycle and breaks it, then restarts.
    // Correctness requires restart because breaking a cycle changes the choice
    // graph — new cycles may appear or old ones may disappear.
    //
    // Complexity: O(cycles × classes). Dense Vecs make each DFS pass O(classes)
    // instead of O(classes × log classes) with BTreeSet.

    let capacity = best_node.len();
    let root_can = egraph.find(root).0 as usize;
    if root_can >= capacity {
        return;
    }

    // Reusable buffers (cleared each iteration, not reallocated)
    let mut color: Vec<u8> = vec![0; capacity]; // 0=white, 1=gray, 2=black
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(256);

    loop {
        // Reset for this DFS pass
        color.iter_mut().for_each(|c| *c = 0);
        stack.clear();
        stack.push((root_can, 0));
        color[root_can] = 1;

        let mut cycle_found = false;

        'dfs: while !stack.is_empty() {
            let (cid, ref_ci) = {
                let top = stack.last().unwrap();
                (top.0, top.1)
            };

            let node_idx = best_node[cid].unwrap_or(0);
            let canonical = egraph.find(EClassId(cid as u32));
            let nodes = egraph.nodes(canonical);

            let children: Vec<usize> = if node_idx < nodes.len() {
                if let ENode::Op { children, .. } = &nodes[node_idx] {
                    children
                        .iter()
                        .map(|&c| egraph.find(c).0 as usize)
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let mut found_child = false;
            let mut ci = ref_ci;
            while ci < children.len() {
                let child_can = children[ci];
                ci += 1;
                if child_can >= capacity {
                    continue;
                }

                match color[child_can] {
                    0 => {
                        stack.last_mut().unwrap().1 = ci;
                        color[child_can] = 1;
                        stack.push((child_can, 0));
                        found_child = true;
                        break;
                    }
                    1 => {
                        // Back-edge → cycle. Extract and break it.
                        let cycle: Vec<usize> = stack
                            .iter()
                            .map(|&(s, _)| s)
                            .skip_while(|&s| s != child_can)
                            .collect();
                        break_single_cycle(egraph, &cycle, best_node);
                        cycle_found = true;
                        break 'dfs;
                    }
                    _ => {} // black, skip
                }
            }

            if !found_child {
                stack.last_mut().unwrap().1 = ci;
                color[cid] = 2;
                stack.pop();
            }
        }

        if !cycle_found {
            break; // No cycles remain
        }
    }
}

/// Break a cycle by switching the chosen node for one member to a leaf
/// or a node whose children are all outside the cycle.
fn break_single_cycle(egraph: &EGraph, cycle: &[usize], best_node: &mut Vec<Option<usize>>) {
    if cycle.is_empty() {
        return;
    }

    // Build cycle membership set for O(1) lookup
    let max_id = cycle.iter().copied().max().unwrap_or(0);
    let mut in_cycle = vec![false; max_id + 1];
    for &cid in cycle {
        in_cycle[cid] = true;
    }

    // Strategy 1: find ANY cycle member with a leaf node
    for &cid in cycle {
        let canonical = egraph.find(EClassId(cid as u32));
        let nodes = egraph.nodes(canonical);
        for (idx, node) in nodes.iter().enumerate() {
            if matches!(node, ENode::Var(_) | ENode::Const(_)) {
                best_node[cid] = Some(idx);
                return;
            }
        }
    }

    // Strategy 2: find a cycle member with an Op whose children
    // are ALL outside the cycle
    for &cid in cycle {
        let canonical = egraph.find(EClassId(cid as u32));
        let nodes = egraph.nodes(canonical);
        for (idx, node) in nodes.iter().enumerate() {
            if let ENode::Op { children, .. } = node {
                let all_outside = children.iter().all(|&c| {
                    let cc = egraph.find(c).0 as usize;
                    cc >= in_cycle.len() || !in_cycle[cc]
                });
                if all_outside {
                    best_node[cid] = Some(idx);
                    return;
                }
            }
        }
    }

    // Strategy 3: pick the first cycle member, choose its node with
    // fewest children in the cycle (minimize remaining cycle edges)
    let cid = cycle[0];
    let canonical = egraph.find(EClassId(cid as u32));
    let nodes = egraph.nodes(canonical);
    let mut best_idx = 0;
    let mut best_in_cycle_count = usize::MAX;
    for (idx, node) in nodes.iter().enumerate() {
        let count = match node {
            ENode::Var(_) | ENode::Const(_) => 0,
            ENode::Op { children, .. } => children
                .iter()
                .filter(|&&c| {
                    let cc = egraph.find(c).0 as usize;
                    cc < in_cycle.len() && in_cycle[cc]
                })
                .count(),
        };
        if count < best_in_cycle_count {
            best_in_cycle_count = count;
            best_idx = idx;
        }
    }
    best_node[cid] = Some(best_idx);
}

/// Extract the minimum-cost arena expression from an e-class.
///
/// Uses dynamic programming: cost(class) = min over all nodes in class.
///
/// # Type Parameter
///
/// The cost function can be any type implementing `CostFunction`:
/// - `CostModel` for hardcoded costs
/// - Neural cost models (e.g., `ExprNnue` via adapter)
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
                    ENode::Var(_) | ENode::Const(_) => costs.node_cost(node, None),
                    ENode::Op { children, .. } => {
                        // Check for self-referential children
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                                })
                                .sum();
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

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Break any mutual cycles in the choice graph before building the tree.
    break_choice_cycles(egraph, root, &mut best_node);

    let (arena, root_id) = choices_to_arena(egraph, root, &best_node);
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

/// Build an `ExtractedDAG` from NNUE extraction choices + reference counts.
///
/// Bridges the NNUE hill-climbing extractor (which produces per-e-class choices)
/// with DAG codegen (which needs `ExtractedDAG` with sharing info for let-bindings).
pub fn build_extracted_dag_from_choices(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
    ref_counts: &[u32],
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
        total_cost: 0,
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
pub fn choices_to_arena(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    use pixelflow_ir::{ExprArena, ExprId};

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
                    ENode::Op { children, .. } => {
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

    /// Total cost of the extracted expression.
    pub total_cost: usize,
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
pub fn extract_dag<C: CostFunction>(egraph: &EGraph, root: EClassId, costs: &C) -> ExtractedDAG {
    use alloc::collections::BTreeSet;

    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Compute best node per e-class (same as regular extraction)
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if best_cost[canonical.0 as usize].is_some() {
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
                        if best_cost[child_canonical.0 as usize].is_none() {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            on_stack.remove(&canonical.0);

            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) => costs.node_cost(node, None),
                    ENode::Op { children, .. } => {
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                                })
                                .sum();
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

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Break any mutual cycles in the choice graph before counting refs.
    break_choice_cycles(egraph, root, &mut best_node);

    // Phase 2: Count references to each e-class in the extracted DAG
    let mut ref_counts: Vec<usize> = alloc::vec![0; num_classes];
    count_refs_recursive(egraph, root, &best_node, &mut ref_counts);

    // Phase 3: Identify shared e-classes (count > 1)
    let shared: Vec<(EClassId, usize)> = ref_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 1)
        .map(|(idx, count)| (EClassId(idx as u32), *count))
        .collect();

    // Phase 4: Topological sort for emission order
    let schedule = toposort_dag(egraph, root, &best_node, &shared);

    ExtractedDAG {
        root: egraph.find(root),
        shared,
        schedule,
        choices: best_node,
        total_cost,
    }
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

    #[test]
    fn dag_accumulator_handles_shared_subexpressions() {
        use crate::nnue::{EdgeAccumulator, ExprNnue};

        // sin(X) * sin(X): tree has 2x sin edges, DAG has 1x sin + 1x var_ref
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

        let nnue = ExprNnue::new_with_latency_prior(42);

        // DAG accumulator
        let ref_counts = compute_ref_counts(&egraph, product, &choices);
        let dag_acc = EdgeAccumulator::from_dag_choices(
            &egraph,
            product,
            &choices,
            &ref_counts,
            &nnue.embeddings,
        );

        assert_eq!(dag_acc.node_count, 3, "DAG acc should count 3 unique nodes");
        assert_eq!(
            dag_acc.edge_count, 3,
            "shared reuse should contribute a var_ref edge"
        );
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

        let (arena, root_id) = choices_to_arena(&egraph, add, &choices);

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

        let (arena, root_id) = choices_to_arena(&egraph, mul, &choices);

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

        let (arena, root_id) = choices_to_arena(&egraph, add, &choices);
        let (extracted_arena, extracted_root, _cost) = extract(&egraph, add, &CostModel::default());
        assert_eq!(arena.len(), extracted_arena.len());
        assert_eq!(root_id, extracted_root);
    }
}
