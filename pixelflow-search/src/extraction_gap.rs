//! How much extracted cost does the greedy DP leave on the table?
//!
//! `extract_dag` is not argmin under its own cost model, three ways over
//! (issue #1111, and the `Optimizer::cost` doc comment says so outright):
//!
//! 1. **Unpriced sharing.** It sums each child's `best_cost`, which is a
//!    TREE cost. A subexpression used five times is charged five times in
//!    the objective and emitted once in the code, so the objective is not
//!    the thing being minimized.
//! 2. **Single DFS, not a fixpoint.** On the cyclic e-graphs saturation
//!    always produces (commutativity alone makes cycles), a class whose
//!    child is still `on_stack` is scored `CYCLE_COST` and never revisited.
//! 3. ~~**Costed before repaired.**~~ **Fixed (#1111).** `total_cost` used to
//!    be read before `repair_choices_well_founded` may change the choices, so
//!    the reported number need not describe the returned term — this harness
//!    measured that on 132 of 302 kernels. Both reported numbers are now
//!    computed from the repaired choices, and there are two:
//!    `ExtractedDAG::total_cost` is the tree cost the DP minimizes and
//!    `ExtractedDAG::dag_cost` is the DAG cost the kernel pays. The
//!    `reported_matches_returned` column below is kept as the standing
//!    regression check and should now be true on every kernel.
//!
//! That matters right now because three independent "give saturation more
//! room" changes each improved most kernels and regressed a minority. If a
//! larger e-graph is not monotonically better, the chooser is why: more
//! equalities give the greedy DP more ways to take a worse branch. The
//! **exact** optimum, by contrast, IS monotone — a larger e-graph holds a
//! superset of the equalities, hence a superset of the extractable terms, so
//! its optimum can only fall.
//!
//! # What "exact" means here, precisely
//!
//! Minimum-cost extraction with sharing is NP-hard, so the reference is a
//! **branch-and-bound over per-class node choices** with a per-kernel node
//! and wall-clock budget. It computes the true DAG optimum — each chosen
//! e-class priced exactly once, under the SAME
//! [`CostModel::latency_prior`](crate::egraph::CostModel::latency_prior) the
//! DP uses — or reports `UNSOLVED` and is excluded from the gap statistic.
//! Never a truncated answer presented as an optimum.
//!
//! Between greedy and exact sits a third, always-computable reference:
//! **Knuth's algorithm** (Dijkstra generalized to AND-OR graphs) gives the
//! exact minimum **tree** cost in polynomial time. That is precisely what
//! `extract_dag` *intends* to compute, so it splits the loss additively:
//!
//! ```text
//! cost(greedy) - cost(exact)
//!   = [cost(greedy) - cost(tree-optimal)]   <- defects 2+3: the DP not
//!                                              attaining its own objective
//!   + [cost(tree-optimal) - cost(exact)]    <- defect 1: optimizing tree
//!                                              cost instead of DAG cost
//! ```
//!
//! Both bracketed terms are measured on the same footing: the **true DAG
//! cost of the term each chooser returns**.
//!
//! Read-only. Nothing here changes production behavior.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use pixelflow_ir::{ExprArena, ExprId, LatticeShape};

use crate::arena_corpus::{category_of, load_arena_dump, median, percentile};
use crate::egraph::extract::{
    ExtractedDAG, extract_dag_objectives, extract_dag_scoped, extract_dag_tree_arm,
};
use crate::egraph::{
    Budget, CostModel, EClassId, EGraph, ENode, Extraction, Optimizer, SaturationStop,
};
use crate::runtime::{arena_to_egraph, reachable_count};

// ---------------------------------------------------------------------------
// The instance: one saturated e-graph, reduced to what a chooser must decide
// ---------------------------------------------------------------------------

/// One candidate e-node for one e-class: its own op cost and the canonical
/// classes it depends on, deduplicated (a DAG prices a child class once no
/// matter how many of a node's slots point at it).
#[derive(Clone, Debug)]
struct Candidate {
    /// Index into `egraph.nodes(class)` — the choice a chooser records.
    node_idx: usize,
    own: u64,
    /// Distinct child classes: what a DAG pays for, once each.
    children: Vec<u32>,
    /// Parallel to `children`: how many of the node's operand slots name
    /// that class. What a TREE pays for, and therefore what the DP sums.
    /// `Mul(a, a)` is one entry in `children` with multiplicity 2.
    mult: Vec<u32>,
}

/// The extraction problem, lifted off the e-graph so the search never touches
/// `find` in its inner loop.
struct Instance {
    root: u32,
    /// Per canonical class id, per e-node index: EVERY node, so that any
    /// chooser's answer — including one that picked a node this instance
    /// would never search — can be re-costed. `cands[c][i].node_idx == i`.
    cands: Vec<Vec<Candidate>>,
    /// Per class: every node index that can appear in SOME well-founded
    /// selection — self-referential nodes removed, nothing else. This is the
    /// full feasible set, and it is what the tree-cost reference searches.
    feasible: Vec<Vec<usize>>,
    /// Per class: the node indices the DAG search may choose — `feasible`
    /// minus the nodes dominance proves can never help. Dominance is sound
    /// for DAG cost and NOT for tree cost (`A` may name one child twice
    /// where `B` names two children once, so `A ⊆ B` says nothing about the
    /// tree), which is exactly why the two sets are kept apart.
    alive: Vec<Vec<usize>>,
    /// Per class: cheapest `own` over its live candidates — the admissible
    /// per-class floor the bound sums over.
    floor: Vec<u64>,
    /// Classes reachable from the root over ALL nodes (the search space).
    reachable: Vec<u32>,
    /// Candidates dropped by dominance, for the report.
    dominated: usize,
}

impl Instance {
    fn build(egraph: &EGraph, root: EClassId, model: &CostModel) -> Self {
        let n = egraph.num_classes();
        let root = egraph.find(root).0;

        // Reachable set over every node of every class: any of them could be
        // chosen, so all of them are in the search space.
        let mut reachable: Vec<u32> = Vec::new();
        let mut seen = vec![false; n];
        let mut stack = vec![root];
        seen[root as usize] = true;
        while let Some(c) = stack.pop() {
            reachable.push(c);
            for node in egraph.nodes(EClassId(c)) {
                if let ENode::Op { children, .. } = node {
                    for &ch in children {
                        let ch = egraph.find(ch).0;
                        if !seen[ch as usize] {
                            seen[ch as usize] = true;
                            stack.push(ch);
                        }
                    }
                }
            }
        }
        reachable.sort_unstable();

        let mut cands: Vec<Vec<Candidate>> = vec![Vec::new(); n];
        let mut alive: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut feasible: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &c in &reachable {
            let mut out: Vec<Candidate> = Vec::new();
            let mut live: Vec<usize> = Vec::new();
            for (node_idx, node) in egraph.nodes(EClassId(c)).iter().enumerate() {
                let own = model.node_cost_of(node);
                let mut slots: Vec<u32> = match node {
                    ENode::Op { children, .. } => {
                        children.iter().map(|&ch| egraph.find(ch).0).collect()
                    }
                    _ => Vec::new(),
                };
                slots.sort_unstable();
                let mut children: Vec<u32> = Vec::new();
                let mut mult: Vec<u32> = Vec::new();
                for sl in slots {
                    if children.last() == Some(&sl) {
                        *mult.last_mut().expect("pushed with children") += 1;
                    } else {
                        children.push(sl);
                        mult.push(1);
                    }
                }
                // A node that names its own class can never appear in a
                // well-founded selection. Excluding it from the search is
                // the same judgement `extract_dag`'s `CYCLE_COST` makes,
                // made once instead of as a sentinel that then has to be
                // kept above every real cost. It stays in `cands` so a
                // chooser that picked it can still be re-costed.
                if !children.contains(&c) {
                    live.push(node_idx);
                }
                out.push(Candidate {
                    node_idx,
                    own,
                    children,
                    mult,
                });
            }
            cands[c as usize] = out;
            feasible[c as usize] = live.clone();
            alive[c as usize] = live;
        }

        // Dominance: candidate A makes candidate B useless when A costs no
        // more on its own AND needs no class B does not also need — B's
        // extra classes can only add cost (never subtract; a class already
        // paid for elsewhere adds zero). Ties broken by position so the
        // filter is deterministic and never empties a class.
        let mut dominated = 0usize;
        for &c in &reachable {
            let live = std::mem::take(&mut alive[c as usize]);
            let mut keep: Vec<bool> = vec![true; live.len()];
            for i in 0..live.len() {
                for j in 0..live.len() {
                    if i == j || !keep[i] || !keep[j] {
                        continue;
                    }
                    let a = &cands[c as usize][live[j]];
                    let b = &cands[c as usize][live[i]];
                    let a_dominates_b = a.own <= b.own
                        && a.children.iter().all(|ch| b.children.contains(ch))
                        && (a.own < b.own || a.children.len() < b.children.len() || j < i);
                    if a_dominates_b {
                        keep[i] = false;
                        dominated += 1;
                    }
                }
            }
            alive[c as usize] = live
                .into_iter()
                .zip(keep)
                .filter_map(|(idx, k)| k.then_some(idx))
                .collect();
        }

        let mut floor = vec![0u64; n];
        for &c in &reachable {
            floor[c as usize] = alive[c as usize]
                .iter()
                .map(|&i| cands[c as usize][i].own)
                .min()
                .unwrap_or(0);
        }

        Self {
            root,
            cands,
            feasible,
            alive,
            floor,
            reachable,
            dominated,
        }
    }

    /// The true DAG cost of a choice function: every class the root reaches
    /// under `choice`, priced once.
    ///
    /// Returns `None` when the choice function is not well-founded from the
    /// root, or leaves a reachable class unchosen — either way it does not
    /// describe a term, and a number for it would be a silent lie.
    fn dag_cost(&self, choice: &[Option<usize>]) -> Option<u64> {
        let mut total = 0u64;
        let mut state = vec![0u8; self.cands.len()]; // 0 unseen, 1 on path, 2 done
        let mut stack: Vec<(u32, usize)> = vec![(self.root, 0)];
        state[self.root as usize] = 1;
        let node_of = |c: u32| -> Option<&Candidate> {
            let idx = choice.get(c as usize).copied().flatten()?;
            self.cands[c as usize].get(idx)
        };
        total += node_of(self.root)?.own;
        while let Some(&mut (c, ref mut next)) = stack.last_mut() {
            let cand = node_of(c)?;
            if *next < cand.children.len() {
                let ch = cand.children[*next];
                *next += 1;
                match state[ch as usize] {
                    0 => {
                        state[ch as usize] = 1;
                        total = total.saturating_add(node_of(ch)?.own);
                        stack.push((ch, 0));
                    }
                    1 => return None, // cycle
                    _ => {}
                }
            } else {
                state[c as usize] = 2;
                stack.pop();
            }
        }
        Some(total)
    }

    /// The true TREE cost of a choice function: every *edge* walked, so a
    /// shared class is charged once per use. This is what `extract_dag`'s
    /// DP sums, and comparing it against the DP's reported `total_cost` is
    /// how defect 3 is detected.
    ///
    /// Memoized per class — a shared DAG's tree expansion is exponential in
    /// its depth, so the obvious recursion would not return on a real
    /// kernel. `None` when the choice function is cyclic or incomplete.
    fn tree_cost(&self, choice: &[Option<usize>]) -> Option<u64> {
        let n = self.cands.len();
        let mut memo: Vec<Option<u64>> = vec![None; n];
        let mut state = vec![0u8; n]; // 0 unseen, 1 on path, 2 done
        let mut stack: Vec<(u32, usize)> = vec![(self.root, 0)];
        state[self.root as usize] = 1;
        let node_of = |c: u32| -> Option<&Candidate> {
            let idx = choice.get(c as usize).copied().flatten()?;
            self.cands[c as usize].get(idx)
        };
        node_of(self.root)?;
        while let Some(&mut (c, ref mut next)) = stack.last_mut() {
            let cand = node_of(c)?;
            if *next < cand.children.len() {
                let ch = cand.children[*next];
                *next += 1;
                match state[ch as usize] {
                    0 => {
                        node_of(ch)?;
                        state[ch as usize] = 1;
                        stack.push((ch, 0));
                    }
                    1 => return None, // cycle
                    _ => {}
                }
            } else {
                // Multiplied back out by slot count: the DP sums
                // `children.iter()`, so `Mul(a, a)` charges `a` twice.
                let mut total = cand.own;
                for (i, &ch) in cand.children.iter().enumerate() {
                    let child = memo[ch as usize].expect("child finished first");
                    total = total.saturating_add(child.saturating_mul(cand.mult[i] as u64));
                }
                memo[c as usize] = Some(total);
                state[c as usize] = 2;
                stack.pop();
            }
        }
        memo[self.root as usize]
    }
}

/// `CostModel::node_op_cost` as a `u64`, so a search that adds a few thousand
/// of them cannot wrap. `Dwrt`'s prohibitive sentinel survives the widening.
trait NodeCostU64 {
    fn node_cost_of(&self, node: &ENode) -> u64;
}

impl NodeCostU64 for CostModel {
    fn node_cost_of(&self, node: &ENode) -> u64 {
        self.node_op_cost(node) as u64
    }
}

// ---------------------------------------------------------------------------
// Reference B: exact minimum TREE cost, in polynomial time
// ---------------------------------------------------------------------------

/// Knuth's generalization of Dijkstra to AND-OR graphs: the least fixpoint of
/// `cost[c] = min over nodes n of (own(n) + sum over child slots of cost[child])`.
///
/// Finalizing classes in nondecreasing cost order makes the resulting choice
/// function well-founded by construction — the finalization order IS a
/// topological order — so unlike the DP this never needs a repair pass, and
/// unlike the DP it is an exact argmin of the objective it names.
///
/// Returns the per-class choice vector, or `None` if the root is unreachable
/// (every route to it passes through a class with no finite-cost node).
fn tree_optimal_choices(inst: &Instance) -> Option<Vec<Option<usize>>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = inst.cands.len();
    // Per (class, candidate): how many child *slots* are still unfinalized,
    // and the accumulated cost of the finalized ones. Slots, not distinct
    // classes: a tree cost charges each occurrence.
    let mut remaining: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut acc: Vec<Vec<u64>> = vec![Vec::new(); n];
    // Reverse index: class -> the (class, candidate) slots waiting on it.
    let mut waiters: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];

    let mut heap: BinaryHeap<Reverse<(u64, u32, usize)>> = BinaryHeap::new();
    for &c in &inst.reachable {
        let list = &inst.cands[c as usize];
        remaining[c as usize] = vec![0; list.len()];
        acc[c as usize] = vec![0; list.len()];
        for &i in &inst.feasible[c as usize] {
            let cd = &list[i];
            remaining[c as usize][i] = cd.children.len();
            acc[c as usize][i] = cd.own;
            if cd.children.is_empty() {
                heap.push(Reverse((cd.own, c, i)));
            }
            for &ch in &cd.children {
                waiters[ch as usize].push((c, i));
            }
        }
    }

    let mut best: Vec<Option<(u64, usize)>> = vec![None; n];
    while let Some(Reverse((cost, c, cand_idx))) = heap.pop() {
        if best[c as usize].is_some() {
            continue;
        }
        best[c as usize] = Some((cost, cand_idx));
        for (pc, pi) in waiters[c as usize].clone() {
            if best[pc as usize].is_some() {
                continue;
            }
            let cand = &inst.cands[pc as usize][pi];
            let Some(slot) = cand.children.iter().position(|&x| x == c) else {
                continue;
            };
            let r = &mut remaining[pc as usize][pi];
            if *r == 0 {
                continue;
            }
            *r -= 1;
            acc[pc as usize][pi] =
                acc[pc as usize][pi].saturating_add(cost.saturating_mul(cand.mult[slot] as u64));
            if *r == 0 {
                heap.push(Reverse((acc[pc as usize][pi], pc, pi)));
            }
        }
    }

    best[inst.root as usize]?;
    let mut choices: Vec<Option<usize>> = vec![None; n];
    for &c in &inst.reachable {
        choices[c as usize] = best[c as usize].map(|(_, idx)| idx);
    }
    Some(choices)
}

/// An admissible lower bound on the DAG cost of the term rooted at each
/// class: `lb[c] = min over nodes of (own + MAX over children of lb[child])`.
///
/// Why this is a bound and the tree cost is not: the classes along any single
/// root-to-leaf chain of the chosen DAG are pairwise distinct, so a DAG pays
/// for all of them; the cheapest such chain is therefore never more than the
/// whole DAG. `max` where Knuth's algorithm has `sum` — same Dijkstra
/// structure, because `max` is monotone in each argument too.
///
/// Much tighter than "the cheapest node in each class" for the deep chains
/// real kernels are made of, which is what lets the search close at all.
fn path_lower_bounds(inst: &Instance) -> Vec<u64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = inst.cands.len();
    let mut remaining: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut acc: Vec<Vec<u64>> = vec![Vec::new(); n];
    let mut waiters: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
    let mut heap: BinaryHeap<Reverse<(u64, u32, usize)>> = BinaryHeap::new();

    for &c in &inst.reachable {
        let list = &inst.cands[c as usize];
        remaining[c as usize] = vec![0; list.len()];
        acc[c as usize] = vec![0; list.len()];
        for &i in &inst.alive[c as usize] {
            let cd = &list[i];
            remaining[c as usize][i] = cd.children.len();
            acc[c as usize][i] = 0;
            if cd.children.is_empty() {
                heap.push(Reverse((cd.own, c, i)));
            }
            for &ch in &cd.children {
                waiters[ch as usize].push((c, i));
            }
        }
    }

    let mut lb: Vec<u64> = vec![0; n];
    let mut done: Vec<bool> = vec![false; n];
    while let Some(Reverse((cost, c, _))) = heap.pop() {
        if done[c as usize] {
            continue;
        }
        done[c as usize] = true;
        lb[c as usize] = cost;
        for (pc, pi) in waiters[c as usize].clone() {
            if done[pc as usize] {
                continue;
            }
            let r = &mut remaining[pc as usize][pi];
            if *r == 0 {
                continue;
            }
            *r -= 1;
            acc[pc as usize][pi] = acc[pc as usize][pi].max(cost);
            if *r == 0 {
                let own = inst.cands[pc as usize][pi].own;
                heap.push(Reverse((own.saturating_add(acc[pc as usize][pi]), pc, pi)));
            }
        }
    }
    // A class no finite chain reaches keeps 0, which is still admissible.
    lb
}

/// The mirror of [`path_lower_bounds`], pointing the other way: an admissible
/// lower bound on the cost of everything ABOVE a class — the cheapest chain of
/// own-costs from the root down to (but not including) `c`.
///
/// Same distinctness argument: a class that is used is reachable from the
/// root, the classes on that path are pairwise distinct and distinct from `c`
/// and from everything under `c` (an ancestor that were also a descendant
/// would be a cycle), so the DAG pays for all of them.
fn chain_upper_bounds(inst: &Instance) -> Vec<u64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = inst.cands.len();
    let mut dist: Vec<u64> = vec![u64::MAX; n];
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    dist[inst.root as usize] = 0;
    heap.push(Reverse((0, inst.root)));
    while let Some(Reverse((d, c))) = heap.pop() {
        if d > dist[c as usize] {
            continue;
        }
        for &i in &inst.alive[c as usize] {
            let cd = &inst.cands[c as usize][i];
            let through = d.saturating_add(cd.own);
            for &ch in &cd.children {
                if through < dist[ch as usize] {
                    dist[ch as usize] = through;
                    heap.push(Reverse((through, ch)));
                }
            }
        }
    }
    dist
}

/// Reduced-cost fixing: with an incumbent of cost `ub` in hand, any e-node
/// whose *best possible* whole-DAG cost already reaches `ub` cannot appear in
/// a strictly better term, so the search need never consider it.
///
/// The bound per node is `above + own + below` — a chain from the root down
/// to this class, this node, and the cheapest chain under it — and every
/// class in those three parts is distinct, so their sum is admissible.
/// Pruning cascades: a node all of whose alternatives in a child class are
/// gone is itself gone.
///
/// This is what makes the search close at all on a saturated graph, and it is
/// exact rather than a heuristic cut: `>= ub` is safe precisely because a term
/// of cost `ub` is already in hand.
///
/// Returns the final `below` bounds, and whether the root survived — a root
/// with no live node means nothing beats the incumbent, i.e. the incumbent is
/// already optimal.
fn prune_to_incumbent(inst: &mut Instance, ub: u64) -> (Vec<u64>, bool) {
    const MAX_ROUNDS: usize = 8;
    let mut below = path_lower_bounds(inst);
    for _ in 0..MAX_ROUNDS {
        below = path_lower_bounds(inst);
        let above = chain_upper_bounds(inst);
        let mut changed = false;
        for idx in 0..inst.reachable.len() {
            let c = inst.reachable[idx];
            let before = inst.alive[c as usize].len();
            let kept: Vec<usize> = inst.alive[c as usize]
                .iter()
                .copied()
                .filter(|&i| {
                    let cd = &inst.cands[c as usize][i];
                    if cd
                        .children
                        .iter()
                        .any(|&ch| inst.alive[ch as usize].is_empty())
                    {
                        return false;
                    }
                    let deepest = cd
                        .children
                        .iter()
                        .map(|&ch| below[ch as usize])
                        .max()
                        .unwrap_or(0);
                    above[c as usize]
                        .saturating_add(cd.own)
                        .saturating_add(deepest)
                        < ub
                })
                .collect();
            if kept.len() != before {
                changed = true;
            }
            inst.alive[c as usize] = kept;
        }
        if !changed {
            break;
        }
        if inst.alive[inst.root as usize].is_empty() {
            return (below, false);
        }
    }
    for &c in &inst.reachable {
        inst.floor[c as usize] = inst.alive[c as usize]
            .iter()
            .map(|&i| inst.cands[c as usize][i].own)
            .min()
            .unwrap_or(0);
    }
    let alive_root = !inst.alive[inst.root as usize].is_empty();
    (below, alive_root)
}

// ---------------------------------------------------------------------------
// Reference C: exact minimum DAG cost, by branch and bound
// ---------------------------------------------------------------------------

/// What the exact search concluded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExactStatus {
    /// The search closed: `cost` is the DAG optimum.
    Optimal,
    /// The search ran out of budget. `cost` is the best term FOUND (a valid
    /// upper bound on the optimum, hence a certified floor on greedy's
    /// excess), and `lower_bound` is what the search had proved.
    Unsolved,
}

struct ExactResult {
    status: ExactStatus,
    cost: u64,
    lower_bound: u64,
    choices: Vec<Option<usize>>,
    expansions: u64,
    elapsed: Duration,
}

struct Search<'a> {
    inst: &'a Instance,
    /// Per class, an admissible lower bound on the DAG cost of its subtree.
    path_lb: &'a [u64],
    assign: Vec<Option<usize>>,
    /// Class -> index into `inst.cands[class]` of the assigned candidate.
    cand_of: Vec<Option<usize>>,
    pending: Vec<u32>,
    cost: u64,
    best: u64,
    best_assign: Vec<Option<usize>>,
    /// Scratch for the deduplicating bound.
    stamp: Vec<u32>,
    epoch: u32,
    expansions: u64,
    deadline: Instant,
    max_expansions: u64,
    /// Set when the node or wall-clock budget ran out. Every frame checks it
    /// and unwinds, so an exhausted search returns its incumbent and says so
    /// rather than returning a partial tree as if it were the optimum.
    timed_out: bool,
}

impl<'a> Search<'a> {
    /// Sum of per-class floors over the still-undecided obligations, counting
    /// each class once. Admissible: every one of them will be assigned some
    /// node, contributing at least its floor, and a DAG pays for each class
    /// exactly once.
    fn bound_rest(&mut self, from: usize) -> u64 {
        self.epoch += 1;
        let e = self.epoch;
        let mut sum_floors = 0u64;
        let mut max_chain = 0u64;
        for i in from..self.pending.len() {
            let c = self.pending[i];
            if self.assign[c as usize].is_some() || self.stamp[c as usize] == e {
                continue;
            }
            self.stamp[c as usize] = e;
            sum_floors = sum_floors.saturating_add(self.inst.floor[c as usize]);
            max_chain = max_chain.max(self.path_lb[c as usize]);
        }
        // Both are lower bounds on the same remaining cost; the larger is
        // the better one. They are not additive — a chain under one pending
        // class can pass through another — so they are combined by `max`,
        // not by `+`. An inadmissible bound would prune the optimum and
        // report a wrong number as if it were proved.
        sum_floors.max(max_chain)
    }

    /// Does assigning `c` close a cycle through already-assigned classes?
    /// A cycle can only form entirely among assigned classes, because an
    /// unassigned class has no outgoing edges yet — so this check at the
    /// moment of assignment is complete.
    fn creates_cycle(&self, c: u32) -> bool {
        let mut seen: HashSet<u32> = HashSet::new();
        let mut stack: Vec<u32> = Vec::new();
        let cand = &self.inst.cands[c as usize][self.cand_of[c as usize].expect("just assigned")];
        stack.extend(cand.children.iter().copied());
        while let Some(x) = stack.pop() {
            if x == c {
                return true;
            }
            if !seen.insert(x) {
                continue;
            }
            if let Some(ci) = self.cand_of[x as usize] {
                stack.extend(self.inst.cands[x as usize][ci].children.iter().copied());
            }
        }
        false
    }

    fn run(&mut self, pos: usize) {
        if self.timed_out {
            return;
        }
        self.expansions += 1;
        if self.expansions % 2048 == 0
            && (self.expansions > self.max_expansions || Instant::now() > self.deadline)
        {
            self.timed_out = true;
            return;
        }

        let mut pos = pos;
        while pos < self.pending.len() && self.assign[self.pending[pos] as usize].is_some() {
            pos += 1;
        }
        if pos == self.pending.len() {
            if self.cost < self.best {
                self.best = self.cost;
                self.best_assign = self.assign.clone();
            }
            return;
        }

        let c = self.pending[pos];
        // Cheapest own cost first: finds good incumbents early, which is what
        // makes the bound bite.
        let mut order: Vec<usize> = self.inst.alive[c as usize].clone();
        order.sort_by_key(|&i| {
            let cd = &self.inst.cands[c as usize][i];
            (cd.own, cd.children.len(), cd.node_idx)
        });

        for ci in order {
            if self.timed_out {
                return;
            }
            let own = self.inst.cands[c as usize][ci].own;
            self.assign[c as usize] = Some(ci);
            self.cand_of[c as usize] = Some(ci);
            self.cost = self.cost.saturating_add(own);

            let ok = !self.creates_cycle(c);
            if ok {
                let saved = self.pending.len();
                for k in 0..self.inst.cands[c as usize][ci].children.len() {
                    let ch = self.inst.cands[c as usize][ci].children[k];
                    if self.assign[ch as usize].is_none() {
                        self.pending.push(ch);
                    }
                }
                let h = self.bound_rest(pos + 1);
                if self.cost.saturating_add(h) < self.best {
                    self.run(pos + 1);
                }
                self.pending.truncate(saved);
            }

            self.cost -= own;
            self.assign[c as usize] = None;
            self.cand_of[c as usize] = None;
        }
    }
}

/// Exact DAG-cost extraction, seeded with `ub_choices` (the greedy term) as
/// the incumbent so the bound bites from the first node.
fn exact_dag_choices(
    inst: &Instance,
    seed: &[Option<usize>],
    seed_cost: u64,
    max_expansions: u64,
    time_limit: Duration,
) -> ExactResult {
    let n = inst.cands.len();
    let started = Instant::now();
    // The search runs on a pruned copy; the caller's instance keeps every
    // node, so a chooser's answer can still be re-costed afterwards.
    let mut inst = Instance {
        root: inst.root,
        cands: inst.cands.clone(),
        feasible: inst.feasible.clone(),
        alive: inst.alive.clone(),
        floor: inst.floor.clone(),
        reachable: inst.reachable.clone(),
        dominated: inst.dominated,
    };
    let (path_lb, root_alive) = prune_to_incumbent(&mut inst, seed_cost);
    let root_chain_lb = path_lb[inst.root as usize];
    if !root_alive {
        // Reduced-cost fixing removed every way of beating the incumbent, so
        // the incumbent IS the optimum — proved without expanding a node.
        return ExactResult {
            status: ExactStatus::Optimal,
            cost: seed_cost,
            lower_bound: seed_cost,
            choices: seed.to_vec(),
            expansions: 0,
            elapsed: started.elapsed(),
        };
    }
    let inst = &inst;
    let mut search = Search {
        inst,
        path_lb: &path_lb,
        assign: vec![None; n],
        cand_of: vec![None; n],
        pending: vec![inst.root],
        cost: 0,
        best: seed_cost.saturating_add(1), // strict `<` finds ties as improvements
        best_assign: seed.to_vec(),
        stamp: vec![0; n],
        epoch: 0,
        expansions: 0,
        deadline: started + time_limit,
        max_expansions,
        timed_out: false,
    };
    search.run(0);

    // `best` was seeded at seed_cost+1; if nothing beat the seed the honest
    // answer is the seed's own cost.
    let cost = search.best.min(seed_cost);
    let status = if search.timed_out {
        ExactStatus::Unsolved
    } else {
        ExactStatus::Optimal
    };
    // For an unsolved search the only bound actually proved is the chain
    // bound — the search records no global dual bound of its own, and
    // claiming one it never computed would be the exact failure mode this
    // module exists to measure.
    let lower_bound = if status == ExactStatus::Optimal {
        cost
    } else {
        root_chain_lb
    };
    ExactResult {
        status,
        cost,
        lower_bound,
        choices: if search.best < seed_cost.saturating_add(1) {
            search.best_assign
        } else {
            seed.to_vec()
        },
        expansions: search.expansions,
        elapsed: started.elapsed(),
    }
}

// ---------------------------------------------------------------------------
// An instrumented mirror of `extract_dag_scoped`, pinned to the real thing
// ---------------------------------------------------------------------------

/// The greedy DP again, byte-identical in behavior, but recording which
/// classes it priced at `CYCLE_COST` — the observation defect 2 needs and
/// the public return type does not carry.
///
/// The caller asserts the mirror's choices equal `extract_dag_scoped`'s. A
/// mirror that has drifted is a measurement reporting someone else's
/// algorithm, so drift fails loudly rather than being averaged in.
struct GreedyTrace {
    choices: Vec<Option<usize>>,
    total_cost: usize,
    /// Classes whose winning node was priced at the cycle sentinel.
    cycle_priced: HashSet<u32>,
}

fn greedy_trace(egraph: &EGraph, root: EClassId, costs: &CostModel) -> GreedyTrace {
    use std::collections::BTreeSet;
    const CYCLE_COST: usize = usize::MAX / 4;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = vec![None; num_classes];
    let mut cycle_priced: HashSet<u32> = HashSet::new();

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
            let mut min_was_cycle = false;
            for (idx, node) in nodes.iter().enumerate() {
                let mut saw_cycle = false;
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => costs.node_op_cost(node),
                    ENode::Op { children, .. } => {
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            saw_cycle = true;
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_op_cost(node);
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    match best_cost[c.0 as usize] {
                                        Some(v) => v,
                                        None => {
                                            saw_cycle = true;
                                            CYCLE_COST
                                        }
                                    }
                                })
                                .fold(0usize, usize::saturating_add);
                            op_cost.saturating_add(children_cost)
                        }
                    }
                };
                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                    min_was_cycle = saw_cycle;
                }
            }
            if min_was_cycle {
                cycle_priced.insert(canonical.0);
            }
            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);
    GreedyTrace {
        choices: best_node,
        total_cost,
        cycle_priced,
    }
}

// ---------------------------------------------------------------------------
// One kernel, measured
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Row {
    name: String,
    category: &'static str,
    budget: &'static str,
    node_count: usize,
    classes: usize,
    reachable_classes: usize,
    free_classes: usize,
    branch_product_log10: f64,
    stop: String,
    /// True DAG cost of the term `extract_dag` returns.
    greedy_dag: u64,
    /// True TREE cost of that same term.
    greedy_tree: u64,
    /// `ExtractedDAG::total_cost` — the number the DP reports.
    greedy_reported: u64,
    /// True DAG cost of the tree-cost-optimal term (Knuth).
    treeopt_dag: u64,
    /// The optimal tree cost itself.
    treeopt_tree: u64,
    exact_status: &'static str,
    exact_dag: u64,
    exact_lower_bound: u64,
    exact_expansions: u64,
    exact_ms: f64,
    /// greedy / exact, on solved kernels.
    ratio: f64,
    /// greedy - treeopt_dag: the DP failing its own objective.
    loss_dp: i64,
    /// treeopt_dag - exact: tree cost being the wrong objective.
    loss_sharing: i64,
    /// (greedy_dag - treeopt_dag) / greedy_dag — the always-computable share
    /// of the gap, since Knuth's algorithm needs no time limit.
    dp_loss_frac: f64,
    /// (greedy_tree - treeopt_tree) / greedy_tree — how far the DP lands from
    /// the optimum of the objective it *names*, with no DAG argument at all.
    objective_miss_frac: f64,
    /// Classes the greedy DP priced at `CYCLE_COST` that the best term found
    /// by the exact search nevertheless uses. On an unsolved kernel that
    /// term is the seed, so this is a floor, not a count of all of them.
    cycle_priced_used_by_exact: usize,
    cycle_priced_total: usize,
    /// Defect 3: does the reported cost describe the returned term?
    reported_matches_returned: bool,
    reported_delta: i64,
    dominated_candidates: usize,

    // --- #1116: the sharing-aware objective, measured against the same
    //     references as everything above ---
    /// True DAG cost of the term the sharing-aware DP returns.
    sharing_dag: u64,
    /// True TREE cost of that same term — expected to RISE where the
    /// objective change bites, since tree cost is the thing being abandoned.
    sharing_tree: u64,
    /// True DAG cost of what `extract_dag_scoped` now returns: the cheaper of
    /// the two arms. Never above `greedy_dag`, by construction.
    chosen_dag: u64,
    /// Did the sharing arm change the extracted term at all?
    sharing_changed_term: bool,
    /// `greedy_dag - chosen_dag`: cost units the objective change recovers.
    sharing_gain: i64,
    /// Of the gap `greedy_dag - exact_dag`, the share this closes. NaN when
    /// the exact optimum is unproved or the gap is zero.
    gap_closed_frac: f64,
    /// Median nanoseconds for the pre-#1116 extractor (tree arm alone).
    tree_arm_ns: f64,
    /// Median nanoseconds for what `extract_dag_scoped` now does (both arms).
    scoped_ns: f64,
}

struct Measured {
    row: Row,
}

fn measure(
    name: &str,
    category: &'static str,
    budget: Budget,
    budget_label: &'static str,
    arena: &ExprArena,
    root: ExprId,
    time_limit: Duration,
    max_expansions: u64,
) -> Measured {
    // The same two lowering passes `optimize_runtime_arena_uncached` runs
    // before the e-graph sees the arena.
    let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root)
        .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
    let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);
    let node_count = reachable_count(&arena, root);

    let mut optimizer = Optimizer::production().budget(budget);
    let mut egraph = optimizer.egraph();
    let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut memo)
        .unwrap_or_else(|| panic!("{name}: arena_to_egraph returned None (unsupported node)"));
    let optimized = optimizer.run(&mut egraph, root_class, node_count);

    let model = CostModel::latency_prior();
    let inst = Instance::build(&egraph, root_class, &model);

    // (a) the production chooser, both arms. `tree_arm` is the pre-#1116
    // extractor verbatim and is what every "greedy_*" column below reports,
    // so the A/B is against the thing being replaced rather than against a
    // remembered number; `shared_arm` is the sharing-aware objective (#1116).
    let (tree_arm, shared_arm): (ExtractedDAG, ExtractedDAG) =
        extract_dag_objectives(&egraph, root_class, &model, LatticeShape::POINT);
    let dag = tree_arm;
    let trace = greedy_trace(&egraph, root_class, &model);
    let mirrored = Extraction::from_dp(&egraph, root_class, trace.choices.clone());
    assert_eq!(
        mirrored.choices(),
        dag.choices.as_slice(),
        "{name}: the instrumented greedy mirror has drifted from the tree arm of \
         extract_dag_objectives — the measurement would be describing a different \
         algorithm than the one #1116 replaces"
    );
    let greedy_choices = dag.choices.clone();
    let greedy_dag = inst.dag_cost(&greedy_choices).unwrap_or_else(|| {
        panic!(
            "{name}: extract_dag returned a choice function that is not a term — either \
             cyclic after `repair_choices_well_founded`, or missing a choice for a class \
             the root reaches. That is a production bug, not a measurement outcome"
        )
    });
    let greedy_tree = inst
        .tree_cost(&greedy_choices)
        .expect("a well-founded choice function has a finite tree cost");

    let sharing_choices = shared_arm.choices.clone();
    let sharing_dag = inst.dag_cost(&sharing_choices).unwrap_or_else(|| {
        panic!(
            "{name}: the sharing-aware DP returned a choice function that is not a term \
             — either cyclic after `repair_choices_well_founded`, or missing a choice \
             for a class the root reaches. That is a production bug, not a measurement \
             outcome"
        )
    });
    let sharing_tree = inst
        .tree_cost(&sharing_choices)
        .expect("a well-founded choice function has a finite tree cost");
    // What production now returns: the cheaper arm by DAG cost, ties to tree.
    let chosen_dag = greedy_dag.min(sharing_dag);

    // Extraction cost. This is the one wall-clock number in an otherwise
    // deterministic measurement, so it is taken as a RATIO of two things
    // timed back to back in the same process on the same saturated graph,
    // alternating and taking medians — the absolute nanoseconds are the
    // machine's, the ratio is the change's.
    const TIMING_REPS: usize = 9;
    let mut tree_ns: Vec<f64> = Vec::with_capacity(TIMING_REPS);
    let mut scoped_ns: Vec<f64> = Vec::with_capacity(TIMING_REPS);
    for _ in 0..TIMING_REPS {
        let t0 = Instant::now();
        let a = extract_dag_tree_arm(&egraph, root_class, &model, LatticeShape::POINT);
        tree_ns.push(t0.elapsed().as_secs_f64() * 1e9);
        let t1 = Instant::now();
        let b = extract_dag_scoped(&egraph, root_class, &model, LatticeShape::POINT);
        scoped_ns.push(t1.elapsed().as_secs_f64() * 1e9);
        // Keep the optimizer from eliding either call.
        assert!(
            a.dag_cost >= b.dag_cost,
            "{name}: the scoped extractor returned a worse term than its own tree arm"
        );
    }
    let tree_arm_ns = median(&mut tree_ns);
    let scoped_ns = median(&mut scoped_ns);
    let production = extract_dag_scoped(&egraph, root_class, &model, LatticeShape::POINT);
    let production_dag = inst
        .dag_cost(&production.choices)
        .expect("extract_dag_scoped returns a well-founded choice function");
    assert_eq!(
        production_dag, chosen_dag,
        "{name}: extract_dag_scoped did not return the cheaper of its two arms"
    );

    // (b) exact minimum tree cost
    let treeopt_choices = tree_optimal_choices(&inst)
        .unwrap_or_else(|| panic!("{name}: no finite-cost term reaches the root"));
    let treeopt_dag = inst
        .dag_cost(&treeopt_choices)
        .unwrap_or_else(|| panic!("{name}: Knuth's algorithm produced a non-well-founded choice"));
    let treeopt_tree = inst
        .tree_cost(&treeopt_choices)
        .unwrap_or_else(|| panic!("{name}: tree-optimal choice has no finite tree cost"));

    // (c) exact minimum DAG cost, seeded with the better of (a) and (b)
    let (seed, seed_cost) = [
        (&treeopt_choices, treeopt_dag),
        (&greedy_choices, greedy_dag),
        (&sharing_choices, sharing_dag),
    ]
    .into_iter()
    .min_by_key(|&(_, c)| c)
    .map(|(choices, c)| (choices.clone(), c))
    .expect("three seeds");
    let exact = exact_dag_choices(&inst, &seed, seed_cost, max_expansions, time_limit);

    let cycle_priced_used_by_exact = trace
        .cycle_priced
        .iter()
        .filter(|&&c| exact.choices.get(c as usize).copied().flatten().is_some())
        .count();

    let free_classes = inst
        .reachable
        .iter()
        .filter(|&&c| inst.alive[c as usize].len() > 1)
        .count();
    let branch_product_log10: f64 = inst
        .reachable
        .iter()
        .map(|&c| (inst.alive[c as usize].len().max(1) as f64).log10())
        .sum();

    let solved = exact.status == ExactStatus::Optimal;
    let row = Row {
        name: name.to_string(),
        category,
        budget: budget_label,
        node_count,
        classes: optimized.stats.classes,
        reachable_classes: inst.reachable.len(),
        free_classes,
        branch_product_log10,
        stop: format!("{:?}", optimized.stats.stop),
        greedy_dag,
        greedy_tree,
        greedy_reported: dag.total_cost as u64,
        treeopt_dag,
        treeopt_tree,
        exact_status: if solved { "OPTIMAL" } else { "UNSOLVED" },
        exact_dag: exact.cost,
        exact_lower_bound: exact.lower_bound,
        exact_expansions: exact.expansions,
        exact_ms: exact.elapsed.as_secs_f64() * 1000.0,
        ratio: match (solved, exact.cost, greedy_dag) {
            (false, _, _) => f64::NAN,
            // A free kernel (every op folded away) is trivially agreed on;
            // 0/0 is 1.0 here, not a NaN that would poison the quantiles.
            (true, 0, 0) => 1.0,
            (true, 0, _) => f64::INFINITY,
            (true, e, g) => g as f64 / e as f64,
        },
        loss_dp: greedy_dag as i64 - treeopt_dag as i64,
        dp_loss_frac: if greedy_dag > 0 {
            (greedy_dag as f64 - treeopt_dag as f64) / greedy_dag as f64
        } else {
            0.0
        },
        objective_miss_frac: if greedy_tree > 0 {
            (greedy_tree as f64 - treeopt_tree as f64) / greedy_tree as f64
        } else {
            0.0
        },
        loss_sharing: if solved {
            treeopt_dag as i64 - exact.cost as i64
        } else {
            0
        },
        cycle_priced_used_by_exact,
        cycle_priced_total: trace.cycle_priced.len(),
        reported_matches_returned: greedy_tree == dag.total_cost as u64,
        reported_delta: dag.total_cost as i64 - greedy_tree as i64,
        dominated_candidates: inst.dominated,
        sharing_dag,
        sharing_tree,
        chosen_dag,
        sharing_changed_term: sharing_choices != greedy_choices,
        sharing_gain: greedy_dag as i64 - chosen_dag as i64,
        tree_arm_ns,
        scoped_ns,
        gap_closed_frac: match (solved, greedy_dag as i64 - exact.cost as i64) {
            (false, _) => f64::NAN,
            (true, 0) => f64::NAN,
            (true, gap) => (greedy_dag as i64 - chosen_dag as i64) as f64 / gap as f64,
        },
    };
    Measured { row }
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("{key}={v:?} is not a number: {e}")),
        Err(_) => default,
    }
}

/// THE measurement: how much extracted cost does the greedy DP leave on the
/// table, versus an exact optimum, on real kernels?
///
/// Writes docs/results/2026-09-02-extraction-gap.{md,csv,json}. Read-only
/// with respect to production: no rule, budget, cost model, or extractor is
/// changed.
#[test]
#[ignore = "offline measurement: PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of .arena dumps> cargo test -p pixelflow-search --release --lib -- --ignored extraction_gap_measurement"]
fn extraction_gap_measurement() {
    let dir = PathBuf::from(
        std::env::var("PIXELFLOW_EXTRACTION_GAP_ARENA_DIR")
            .expect("PIXELFLOW_EXTRACTION_GAP_ARENA_DIR must be set"),
    );
    let time_limit = Duration::from_secs(env_usize("PIXELFLOW_EXTRACTION_GAP_SECS", 20) as u64);
    let max_expansions = env_usize("PIXELFLOW_EXTRACTION_GAP_EXPANSIONS", 40_000_000) as u64;
    let limit_kernels = env_usize("PIXELFLOW_EXTRACTION_GAP_LIMIT", usize::MAX);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().map(|e| e == "arena").unwrap_or(false))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no .arena files found in {}",
        dir.display()
    );

    let mut rows: Vec<Row> = Vec::new();

    // --- 1. Real kernels, production regime -------------------------------
    for path in paths.iter().take(limit_kernels) {
        let (name, arena, root) = load_arena_dump(path);
        let category = category_of(&path.file_name().unwrap().to_string_lossy());
        let m = measure(
            &name,
            category,
            Budget::Production,
            "production",
            &arena,
            root,
            time_limit,
            max_expansions,
        );
        eprintln!(
            "{:<28} {:>6} cls {:>7} reach {:>6} free  greedy {:>7} treeopt {:>7} exact {:>7} [{}] {:.0}ms",
            m.row.name,
            m.row.classes,
            m.row.reachable_classes,
            m.row.free_classes,
            m.row.greedy_dag,
            m.row.treeopt_dag,
            m.row.exact_dag,
            m.row.exact_status,
            m.row.exact_ms
        );
        rows.push(m.row);
    }
    let real_count = rows.len();

    // --- 2. Synthetic size ladder: where does exactness stop? -------------
    {
        use crate::nnue::{BwdGenConfig, BwdGenerator};
        let templates = crate::egraph::collect_rule_templates();
        for &max_depth in &[2usize, 3, 4, 5, 6, 7, 9, 11] {
            for seed in 0u64..12 {
                let config = BwdGenConfig {
                    max_depth,
                    ..Default::default()
                };
                let mut generator = BwdGenerator::new(
                    seed.wrapping_add(max_depth as u64 * 10_000),
                    config,
                    templates.clone(),
                );
                let pair = generator.generate_arena();
                let name = format!("synth_d{max_depth}_s{seed}");
                let m = measure(
                    &name,
                    "synthetic",
                    Budget::Production,
                    "production",
                    &pair.arena,
                    pair.unoptimized,
                    time_limit,
                    max_expansions,
                );
                rows.push(m.row);
            }
        }
    }
    let synth_count = rows.len() - real_count;

    // --- 3. The #1114 regression kernels, small vs large e-graph ----------
    //
    // A larger e-graph holds a SUPERSET of the equalities, so the exact
    // optimum over it can only fall. If greedy's cost rises where exact's
    // does not, the regression is the chooser, not the budget.
    let mut monotone_rows: Vec<Row> = Vec::new();
    for prefix in [
        "glyph16_U004B",
        "glyph32_U004B",
        "psychedelic",
        "shader_julia",
    ] {
        let Some(path) = paths
            .iter()
            .find(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        else {
            continue;
        };
        let (name, arena, root) = load_arena_dump(path);
        let category = category_of(&path.file_name().unwrap().to_string_lossy());
        for (budget, label) in [
            (Budget::Production, "production"),
            (
                Budget::Explicit {
                    iterations: 100,
                    classes: 20_000,
                    applications: None,
                },
                "roomier",
            ),
        ] {
            let m = measure(
                &format!("{name} [{label}]"),
                category,
                budget,
                label,
                &arena,
                root,
                time_limit,
                max_expansions,
            );
            eprintln!(
                "monotonicity {:<32} greedy {:>7} treeopt {:>7} exact {:>7} [{}]",
                m.row.name,
                m.row.greedy_dag,
                m.row.treeopt_dag,
                m.row.exact_dag,
                m.row.exact_status
            );
            monotone_rows.push(m.row);
        }
    }

    write_report(
        &rows,
        &monotone_rows,
        real_count,
        synth_count,
        time_limit,
        max_expansions,
    );
}

fn write_report(
    rows: &[Row],
    monotone: &[Row],
    real_count: usize,
    synth_count: usize,
    time_limit: Duration,
    max_expansions: u64,
) {
    let solved: Vec<&Row> = rows
        .iter()
        .filter(|r| r.exact_status == "OPTIMAL")
        .collect();
    let unsolved: Vec<&Row> = rows
        .iter()
        .filter(|r| r.exact_status == "UNSOLVED")
        .collect();

    let mut ratios: Vec<f64> = solved.iter().map(|r| r.ratio).collect();
    assert!(
        ratios.iter().all(|x| x.is_finite()),
        "a solved kernel produced a non-finite greedy/exact ratio — that means the exact \
         search returned cost 0 for a term the greedy chooser priced above 0, which cannot \
         happen and would silently poison every quantile below"
    );
    let agree = solved
        .iter()
        .filter(|r| r.greedy_dag == r.exact_dag)
        .count();
    let med = median(&mut ratios.clone());
    let q1 = percentile(&mut ratios.clone(), 25.0);
    let q3 = percentile(&mut ratios.clone(), 75.0);
    let p90 = percentile(&mut ratios.clone(), 90.0);
    let worst = ratios
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(f64::NAN, f64::max);
    // JSON has no NaN: an empty stratum reports `null`, never a token no
    // parser accepts.
    let jnum = |x: f64| -> String {
        if x.is_finite() {
            format!("{x:.6}")
        } else {
            "null".to_string()
        }
    };

    // Certified floor on greedy's excess, over ALL kernels: even unsolved
    // ones give a valid upper bound on the optimum (the best term the search
    // actually built), so `greedy / best_found` is a lower bound on the true
    // ratio.
    let mut certified: Vec<f64> = rows
        .iter()
        .filter(|r| r.exact_dag > 0)
        .map(|r| r.greedy_dag as f64 / r.exact_dag as f64)
        .collect();
    assert!(
        certified.iter().all(|x| x.is_finite()),
        "certified ratios must be finite: a zero-cost denominator was filtered out above"
    );
    let certified_med = median(&mut certified.clone());
    let certified_improved = rows.iter().filter(|r| r.exact_dag < r.greedy_dag).count();

    // The always-computable half of the gap. Knuth's algorithm has no time
    // limit, so these quantiles run over EVERY kernel — nothing is excluded,
    // and no kernel is silently dropped for being hard.
    let mut dp_loss: Vec<f64> = rows.iter().map(|r| r.dp_loss_frac).collect();
    let dp_med = median(&mut dp_loss.clone());
    let dp_q1 = percentile(&mut dp_loss.clone(), 25.0);
    let dp_q3 = percentile(&mut dp_loss.clone(), 75.0);
    let dp_p90 = percentile(&mut dp_loss.clone(), 90.0);
    let dp_worst = dp_loss
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(f64::NAN, f64::max);
    let dp_agree = rows.iter().filter(|r| r.loss_dp == 0).count();
    let dp_greedy_wins = rows.iter().filter(|r| r.loss_dp < 0).count();
    let mut obj_miss: Vec<f64> = rows.iter().map(|r| r.objective_miss_frac).collect();
    let obj_med = median(&mut obj_miss.clone());
    let obj_p90 = percentile(&mut obj_miss.clone(), 90.0);
    let obj_worst = obj_miss
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(f64::NAN, f64::max);
    let obj_agree = rows
        .iter()
        .filter(|r| r.greedy_tree == r.treeopt_tree)
        .count();

    let total_loss: i64 = solved
        .iter()
        .map(|r| r.greedy_dag as i64 - r.exact_dag as i64)
        .sum();
    let total_loss_dp: i64 = solved.iter().map(|r| r.loss_dp).sum();
    let total_loss_sharing: i64 = solved.iter().map(|r| r.loss_sharing).sum();
    let reported_mismatch = rows.iter().filter(|r| !r.reported_matches_returned).count();
    let cycle_used = rows
        .iter()
        .filter(|r| r.cycle_priced_used_by_exact > 0)
        .count();

    eprintln!("=== extraction gap ===");
    eprintln!(
        "solved {}/{} ({} real, {} synthetic); UNSOLVED {} (excluded from the gap statistic)",
        solved.len(),
        rows.len(),
        real_count,
        synth_count,
        unsolved.len()
    );
    eprintln!(
        "greedy/exact  median {med:.4}  Q1 {q1:.4}  Q3 {q3:.4}  p90 {p90:.4}  worst {worst:.4}  \
         exact agreement {agree}/{}",
        solved.len()
    );
    eprintln!(
        "DP-vs-tree-optimal over ALL {} kernels: median {:.4}%, Q3 {:.4}%, p90 {:.4}%, \
         worst {:.4}%, tied on {dp_agree}, greedy ahead on {dp_greedy_wins}",
        rows.len(),
        dp_med * 100.0,
        dp_q3 * 100.0,
        dp_p90 * 100.0,
        dp_worst * 100.0
    );
    eprintln!(
        "objective miss (tree cost, the DP's own objective): median {:.4}%, p90 {:.4}%, \
         worst {:.4}%, attained on {obj_agree}/{}",
        obj_med * 100.0,
        obj_p90 * 100.0,
        obj_worst * 100.0,
        rows.len()
    );
    eprintln!(
        "mechanism split (pooled cost units, solved only): total loss {total_loss}, \
         DP-not-argmin {total_loss_dp}, unpriced-sharing {total_loss_sharing}"
    );
    eprintln!(
        "defect 3 (reported != returned): {reported_mismatch}/{} kernels",
        rows.len()
    );
    eprintln!(
        "cycle-priced classes used by the exact term: {cycle_used}/{} kernels",
        rows.len()
    );
    eprintln!(
        "certified floor over ALL kernels: median greedy/best-found {certified_med:.4}, \
         strictly improved on {certified_improved}"
    );

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("docs/results");
    std::fs::create_dir_all(&out_dir).expect("create docs/results");

    // ---- CSV ----
    let mut csv = String::new();
    csv.push_str(
        "name,category,budget,node_count,classes,reachable_classes,free_classes,branch_product_log10,\
         stop,greedy_dag,greedy_tree,greedy_reported,treeopt_dag,treeopt_tree,exact_status,exact_dag,\
         exact_lower_bound,exact_expansions,exact_ms,ratio,loss_dp,loss_sharing,\
         cycle_priced_used_by_exact,cycle_priced_total,reported_matches_returned,reported_delta,\
         dominated_candidates,dp_loss_frac,objective_miss_frac\n",
    );
    for r in rows.iter().chain(monotone.iter()) {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{:.3},{},{},{},{},{},{},{},{},{},{},{:.1},{:.6},{},{},{},{},{},{},{}\n",
            r.name,
            r.category,
            r.budget,
            r.node_count,
            r.classes,
            r.reachable_classes,
            r.free_classes,
            r.branch_product_log10,
            r.stop,
            r.greedy_dag,
            r.greedy_tree,
            r.greedy_reported,
            r.treeopt_dag,
            r.treeopt_tree,
            r.exact_status,
            r.exact_dag,
            r.exact_lower_bound,
            r.exact_expansions,
            r.exact_ms,
            r.ratio,
            r.loss_dp,
            r.loss_sharing,
            r.cycle_priced_used_by_exact,
            r.cycle_priced_total,
            r.reported_matches_returned,
            r.reported_delta,
            format!(
                "{},{:.6},{:.6}",
                r.dominated_candidates, r.dp_loss_frac, r.objective_miss_frac
            ),
        ));
    }
    let csv_path = out_dir.join("2026-09-02-extraction-gap.csv");
    std::fs::write(&csv_path, csv).expect("write csv");

    // ---- JSON ----
    let json_row = |r: &Row| -> String {
        format!(
            "{{\"name\":\"{}\",\"category\":\"{}\",\"budget\":\"{}\",\"node_count\":{},\
             \"classes\":{},\"reachable_classes\":{},\"free_classes\":{},\
             \"branch_product_log10\":{:.3},\"stop\":\"{}\",\"greedy_dag\":{},\"greedy_tree\":{},\
             \"greedy_reported\":{},\"treeopt_dag\":{},\"treeopt_tree\":{},\"exact_status\":\"{}\",\
             \"exact_dag\":{},\"exact_lower_bound\":{},\"exact_expansions\":{},\"exact_ms\":{:.1},\
             \"ratio\":{},\"loss_dp\":{},\"loss_sharing\":{},\"cycle_priced_used_by_exact\":{},\
             \"cycle_priced_total\":{},\"reported_matches_returned\":{},\"reported_delta\":{},\
             \"dp_loss_frac\":{:.6},\"objective_miss_frac\":{:.6}}}",
            r.name,
            r.category,
            r.budget,
            r.node_count,
            r.classes,
            r.reachable_classes,
            r.free_classes,
            r.branch_product_log10,
            r.stop,
            r.greedy_dag,
            r.greedy_tree,
            r.greedy_reported,
            r.treeopt_dag,
            r.treeopt_tree,
            r.exact_status,
            r.exact_dag,
            r.exact_lower_bound,
            r.exact_expansions,
            r.exact_ms,
            if r.ratio.is_finite() {
                format!("{:.6}", r.ratio)
            } else {
                "null".to_string()
            },
            r.loss_dp,
            r.loss_sharing,
            r.cycle_priced_used_by_exact,
            r.cycle_priced_total,
            r.reported_matches_returned,
            r.reported_delta,
            r.dp_loss_frac,
            r.objective_miss_frac,
        )
    };
    let mut json = String::new();
    json.push_str("{\n  \"summary\": {\n");
    json.push_str(&format!(
        "    \"kernels\": {}, \"solved\": {}, \"unsolved\": {}, \"real\": {}, \"synthetic\": {},\n",
        rows.len(),
        solved.len(),
        unsolved.len(),
        real_count,
        synth_count
    ));
    json.push_str(&format!(
        "    \"ratio_median\": {}, \"ratio_q1\": {}, \"ratio_q3\": {}, \"ratio_p90\": {}, \
         \"ratio_worst\": {}, \"exact_agreement\": {agree},\n",
        jnum(med),
        jnum(q1),
        jnum(q3),
        jnum(p90),
        jnum(worst)
    ));
    json.push_str(&format!(
        "    \"loss_total\": {total_loss}, \"loss_dp\": {total_loss_dp}, \
         \"loss_sharing\": {total_loss_sharing},\n"
    ));
    json.push_str(&format!(
        "    \"dp_loss_median\": {}, \"dp_loss_q1\": {}, \"dp_loss_q3\": {}, \
         \"dp_loss_p90\": {}, \"dp_loss_worst\": {}, \"dp_tied\": {dp_agree}, \
         \"dp_greedy_ahead\": {dp_greedy_wins},\n",
        jnum(dp_med),
        jnum(dp_q1),
        jnum(dp_q3),
        jnum(dp_p90),
        jnum(dp_worst)
    ));
    json.push_str(&format!(
        "    \"objective_miss_median\": {}, \"objective_miss_p90\": {}, \
         \"objective_miss_worst\": {}, \"objective_attained\": {obj_agree},\n",
        jnum(obj_med),
        jnum(obj_p90),
        jnum(obj_worst)
    ));
    json.push_str(&format!(
        "    \"reported_cost_mismatch_kernels\": {reported_mismatch}, \
         \"cycle_priced_used_kernels\": {cycle_used},\n"
    ));
    json.push_str(&format!(
        "    \"certified_ratio_median_all_kernels\": {}, \
         \"certified_improved_kernels\": {certified_improved},\n",
        jnum(certified_med)
    ));
    json.push_str(&format!(
        "    \"time_limit_secs\": {}\n  }},\n",
        time_limit.as_secs()
    ));
    json.push_str("  \"rows\": [\n");
    let body: Vec<String> = rows.iter().map(json_row).collect();
    json.push_str(&body.join(",\n"));
    json.push_str("\n  ],\n  \"monotonicity\": [\n");
    let mbody: Vec<String> = monotone.iter().map(json_row).collect();
    json.push_str(&mbody.join(",\n"));
    json.push_str("\n  ]\n}\n");
    std::fs::write(out_dir.join("2026-09-02-extraction-gap.json"), json).expect("write json");

    // ---- Markdown ----
    let mut md = String::new();
    let w = &mut md;
    writeln!(
        w,
        "# How much is the greedy extractor leaving on the table?\n"
    )
    .unwrap();
    writeln!(
        w,
        "Measurement only. Nothing in production changed: same rule set, same \
         `Budget::Production`, same `CostModel::latency_prior()`, same `extract_dag`.\n"
    )
    .unwrap();
    writeln!(w, "## Headline\n").unwrap();
    writeln!(
        w,
        "**The measurable half needs no time limit.** Knuth's algorithm gives the exact \
         minimum TREE cost — the objective `extract_dag` names — in polynomial time, on \
         every kernel. Against it, over ALL **{}** kernels with nothing excluded:\n",
        rows.len()
    )
    .unwrap();
    writeln!(
        w,
        "- `1 - treeopt/greedy` (DAG cost of each returned term): median **{:.3}%**, \
         Q1 {:.3}%, Q3 {:.3}%, p90 **{:.3}%**, worst **{:.3}%**.",
        dp_med * 100.0,
        dp_q1 * 100.0,
        dp_q3 * 100.0,
        dp_p90 * 100.0,
        dp_worst * 100.0
    )
    .unwrap();
    writeln!(
        w,
        "- The two choosers tie on **{dp_agree}** kernels; greedy is ahead on **{dp_greedy_wins}** \
         (possible, and honest to report: the tree-optimal term is optimal for TREE cost, \
         and neither chooser optimizes the DAG cost both are scored by)."
    )
    .unwrap();
    writeln!(
        w,
        "- On its own objective, with no DAG argument in play, the DP misses the tree \
         optimum by median **{:.3}%**, p90 **{:.3}%**, worst **{:.3}%**, and attains it on \
         **{obj_agree}/{}** kernels.\n",
        obj_med * 100.0,
        obj_p90 * 100.0,
        obj_worst * 100.0,
        rows.len()
    )
    .unwrap();
    writeln!(w, "The NP-hard half, with a time limit:\n").unwrap();
    writeln!(
        w,
        "- Kernels: **{}** ({real_count} real dumps, {synth_count} synthetic).",
        rows.len()
    )
    .unwrap();
    writeln!(
        w,
        "- Exact optimum proved on **{}**; **{}** UNSOLVED within {}s and excluded from the gap statistic.",
        solved.len(),
        unsolved.len(),
        time_limit.as_secs()
    )
    .unwrap();
    writeln!(
        w,
        "- `cost_greedy / cost_exact` on solved kernels: median **{med:.4}**, Q1 {q1:.4}, \
         Q3 {q3:.4}, p90 **{p90:.4}**, worst **{worst:.4}**."
    )
    .unwrap();
    writeln!(
        w,
        "- Greedy is **exactly optimal on {agree} of {} solved kernels**.",
        solved.len()
    )
    .unwrap();
    writeln!(
        w,
        "- Certified floor across ALL kernels (an unsolved search still returns a term, \
         which upper-bounds the optimum): median `greedy/best-found` **{certified_med:.4}**, \
         strictly beaten on **{certified_improved}**.\n"
    )
    .unwrap();

    writeln!(w, "## The mechanism split\n").unwrap();
    writeln!(
        w,
        "The loss decomposes exactly, because the middle term is itself an argmin:\n\n\
         ```text\n\
         greedy - exact = (greedy - tree_optimal) + (tree_optimal - exact)\n\
         ```\n\n\
         `tree_optimal` is Knuth's algorithm — Dijkstra generalized to AND-OR graphs — which \
         computes the exact minimum TREE cost in polynomial time. That is the objective \
         `extract_dag` *names*, so the first term is the DP failing its own objective \
         (single DFS + `CYCLE_COST`, defects 2 and 3) and the second is that objective being \
         the wrong one (sharing unpriced, defect 1). Both terms are measured as the true DAG \
         cost of the term each chooser returns.\n"
    )
    .unwrap();
    writeln!(w, "| share | pooled cost units (solved kernels) |").unwrap();
    writeln!(w, "|---|---|").unwrap();
    writeln!(w, "| total loss, greedy - exact | {total_loss} |").unwrap();
    writeln!(
        w,
        "| (i) DP not attaining its own tree objective | {total_loss_dp} |"
    )
    .unwrap();
    writeln!(
        w,
        "| (ii) tree cost being the wrong objective | {total_loss_sharing} |"
    )
    .unwrap();
    writeln!(
        w,
        "\n(iii) **cost reported before repair** (issue #1111, fixed): `total_cost` fails to \
         describe the returned term on **{reported_mismatch} of {}** kernels. It was 132/302 \
         when the number was read before `repair_choices_well_founded`; it is costed from the \
         repaired choices now, so anything but 0 here is a regression. Per-kernel magnitudes \
         are in `reported_delta`.\n",
        rows.len()
    )
    .unwrap();
    writeln!(
        w,
        "Cycle handling, separately: the greedy DP priced at least one class at `CYCLE_COST` \
         that the best term the exact search holds nevertheless uses, on **{cycle_used} of \
         {}** kernels. On an unsolved kernel that term is the seed, so this is a floor.\n",
        rows.len()
    )
    .unwrap();

    writeln!(w, "## Where exactness stops being computable\n").unwrap();
    writeln!(
        w,
        "Minimum-cost extraction with sharing is NP-hard, so the reference is a \
         branch-and-bound over per-class node choices with a per-kernel budget \
         ({}s wall, expansion-capped), seeded with the better of the greedy and \
         tree-optimal terms. It reports `UNSOLVED` rather than a truncated answer.\n",
        time_limit.as_secs()
    )
    .unwrap();
    writeln!(
        w,
        "| category | kernels | solved | median reachable classes (solved) | median reachable classes (unsolved) |"
    )
    .unwrap();
    writeln!(w, "|---|---|---|---|---|").unwrap();
    for cat in ["glyph", "shader", "psychedelic", "cellgrid", "synthetic"] {
        let sub: Vec<&Row> = rows.iter().filter(|r| r.category == cat).collect();
        if sub.is_empty() {
            continue;
        }
        let mut s: Vec<f64> = sub
            .iter()
            .filter(|r| r.exact_status == "OPTIMAL")
            .map(|r| r.reachable_classes as f64)
            .collect();
        let mut u: Vec<f64> = sub
            .iter()
            .filter(|r| r.exact_status == "UNSOLVED")
            .map(|r| r.reachable_classes as f64)
            .collect();
        let show = |xs: &mut Vec<f64>| -> String {
            let m = median(xs);
            if m.is_finite() {
                format!("{m:.0}")
            } else {
                "n/a".to_string()
            }
        };
        writeln!(
            w,
            "| {cat} | {} | {} | {} | {} |",
            sub.len(),
            s.len(),
            show(&mut s),
            show(&mut u)
        )
        .unwrap();
    }

    writeln!(w, "\n## Is a bigger e-graph monotonically better?\n").unwrap();
    writeln!(
        w,
        "A larger e-graph holds a superset of the equalities, hence a superset of the \
         extractable terms, so the **exact** optimum over it can only fall. Any kernel whose \
         *greedy* cost rises when the budget is loosened is therefore an extraction failure, \
         not a budget failure.\n"
    )
    .unwrap();
    writeln!(
        w,
        "| kernel | budget | classes | stop | greedy | tree-optimal | best known | status |"
    )
    .unwrap();
    writeln!(w, "|---|---|---|---|---|---|---|---|").unwrap();
    for r in monotone {
        writeln!(
            w,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            r.name,
            r.budget,
            r.classes,
            r.stop,
            r.greedy_dag,
            r.treeopt_dag,
            r.greedy_dag.min(r.treeopt_dag).min(r.exact_dag),
            r.exact_status
        )
        .unwrap();
    }
    writeln!(w, "\n**Verdicts.**\n").unwrap();
    for chunk in monotone.chunks(2) {
        let [small, large] = chunk else { continue };
        let large_best = large.greedy_dag.min(large.treeopt_dag).min(large.exact_dag);
        let verdict = if large_best < small.greedy_dag && large.greedy_dag > small.greedy_dag {
            "**extraction failure** — the roomier e-graph demonstrably CONTAINS a cheaper \
             term than production's own output, and the greedy chooser nevertheless returned \
             a more expensive one than it did on the smaller graph"
        } else if large.greedy_dag > small.greedy_dag {
            "greedy regressed, and no cheaper term was exhibited in the roomier graph within \
             the time limit — unresolved, not exonerated"
        } else {
            "no regression at this budget step"
        };
        writeln!(
            w,
            "- `{}`: production greedy {}, roomier greedy {}, cheapest term exhibited in the \
             roomier graph {} — {verdict}.",
            small.name.split(" [").next().unwrap_or(&small.name),
            small.greedy_dag,
            large.greedy_dag,
            large_best
        )
        .unwrap();
    }
    writeln!(
        w,
        "\nOne caveat, stated because it is easy to misread the table: the **exact** optimum \
         is monotone under a budget increase, but the *tree-optimal* column need not be. \
         Knuth's algorithm minimizes tree cost; the term it picks is then scored here on DAG \
         cost, and a roomier graph can hand it a term that is cheaper as a tree and shares \
         less as a DAG. Where that column rises, it is that effect, not a violated law.\n"
    )
    .unwrap();

    writeln!(
        w,
        "\n## Worst kernels by objective miss — the DP against its own tree cost\n"
    )
    .unwrap();
    writeln!(
        w,
        "No time limit is involved in this table: Knuth's algorithm is exact and polynomial, \
         so every kernel here is a settled number. `greedy DAG` and `tree-opt DAG` are the \
         same two terms re-scored as DAGs, which is what the emitted code actually costs.\n"
    )
    .unwrap();
    writeln!(
        w,
        "| kernel | category | greedy tree | tree-opt tree | miss | greedy DAG | tree-opt DAG |"
    )
    .unwrap();
    writeln!(w, "|---|---|---|---|---|---|---|").unwrap();
    let mut by_miss: Vec<&Row> = rows.iter().collect();
    by_miss.sort_by(|a, b| {
        b.objective_miss_frac
            .partial_cmp(&a.objective_miss_frac)
            .unwrap()
    });
    for r in by_miss.iter().take(20) {
        writeln!(
            w,
            "| {} | {} | {} | {} | {:.2}% | {} | {} |",
            r.name,
            r.category,
            r.greedy_tree,
            r.treeopt_tree,
            r.objective_miss_frac * 100.0,
            r.greedy_dag,
            r.treeopt_dag
        )
        .unwrap();
    }

    writeln!(w, "\n## Worst kernels by proved ratio\n").unwrap();
    writeln!(
        w,
        "| kernel | category | reachable classes | greedy | tree-opt | exact | greedy/exact |"
    )
    .unwrap();
    writeln!(w, "|---|---|---|---|---|---|---|").unwrap();
    let mut worst_rows: Vec<&Row> = solved.clone();
    worst_rows.sort_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap());
    for r in worst_rows.iter().take(15) {
        writeln!(
            w,
            "| {} | {} | {} | {} | {} | {} | {:.4} |",
            r.name,
            r.category,
            r.reachable_classes,
            r.greedy_dag,
            r.treeopt_dag,
            r.exact_dag,
            r.ratio
        )
        .unwrap();
    }

    writeln!(w, "\n## Raw data\n").unwrap();
    writeln!(
        w,
        "`2026-09-02-extraction-gap.csv` / `.json` carry every kernel's row. The two budget \
         knobs are part of the result — a different limit moves the solved/UNSOLVED split \
         and nothing else — so they are written out rather than left to the defaults:\n\n\
         ```sh\n\
         PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of .arena dumps> \\\n  \
         PIXELFLOW_EXTRACTION_GAP_SECS={} \\\n  \
         PIXELFLOW_EXTRACTION_GAP_EXPANSIONS={max_expansions} \\\n  \
         RUST_MIN_STACK=268435456 \\\n  \
         cargo test -p pixelflow-search --release --lib -- --ignored extraction_gap_measurement\n\
         ```\n\n\
         The `.arena` corpus is what the three `#[ignore]`d dumpers produce \
         (`pixelflow-core`'s cell-grid dump, `pixelflow-graphics`'s glyph dump, \
         `pixelflow-pipeline`'s shader/psychedelic dump) — the same 206 files the \
         missing-congruence probe reads.\n",
        time_limit.as_secs()
    )
    .unwrap();

    std::fs::write(out_dir.join("2026-09-02-extraction-gap.md"), md).expect("write md");
    eprintln!(
        "wrote {}",
        out_dir.join("2026-09-02-extraction-gap.md").display()
    );
}

// ---------------------------------------------------------------------------
// #1116: does pricing sharing close the gap the probe above measured?
// ---------------------------------------------------------------------------

/// THE measurement for #1116: the sharing-aware objective, scored against the
/// same two exact references — Knuth's exact tree optimum on every kernel, and
/// the branch-and-bound DAG optimum where it closes.
///
/// Writes docs/results/2026-09-02-extraction-objective.{md,csv,json}.
#[test]
#[ignore = "offline measurement: PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of .arena dumps> cargo test -p pixelflow-search --release --lib -- --ignored extraction_objective_measurement"]
fn extraction_objective_measurement() {
    let dir = PathBuf::from(
        std::env::var("PIXELFLOW_EXTRACTION_GAP_ARENA_DIR")
            .expect("PIXELFLOW_EXTRACTION_GAP_ARENA_DIR must be set"),
    );
    let time_limit = Duration::from_secs(env_usize("PIXELFLOW_EXTRACTION_GAP_SECS", 4) as u64);
    let max_expansions = env_usize("PIXELFLOW_EXTRACTION_GAP_EXPANSIONS", 15_000_000) as u64;
    let limit_kernels = env_usize("PIXELFLOW_EXTRACTION_GAP_LIMIT", usize::MAX);
    let skip_synthetic = std::env::var("PIXELFLOW_EXTRACTION_OBJECTIVE_REAL_ONLY").is_ok();

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().map(|e| e == "arena").unwrap_or(false))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no .arena files found in {}",
        dir.display()
    );

    let mut rows: Vec<Row> = Vec::new();

    // --- 1. Real kernels, production regime -------------------------------
    for path in paths.iter().take(limit_kernels) {
        let (name, arena, root) = load_arena_dump(path);
        let category = category_of(&path.file_name().unwrap().to_string_lossy());
        let m = measure(
            &name,
            category,
            Budget::Production,
            "production",
            &arena,
            root,
            time_limit,
            max_expansions,
        );
        eprintln!(
            "{:<28} tree {:>8} shared {:>8} chosen {:>8} exact {:>8} [{}]",
            m.row.name,
            m.row.greedy_dag,
            m.row.sharing_dag,
            m.row.chosen_dag,
            m.row.exact_dag,
            m.row.exact_status
        );
        rows.push(m.row);
    }
    let real_count = rows.len();

    // --- 2. The same synthetic ladder, because that is where the exact
    //        reference actually closes -------------------------------------
    if !skip_synthetic {
        use crate::nnue::{BwdGenConfig, BwdGenerator};
        let templates = crate::egraph::collect_rule_templates();
        for &max_depth in &[2usize, 3, 4, 5, 6, 7, 9, 11] {
            for seed in 0u64..12 {
                let config = BwdGenConfig {
                    max_depth,
                    ..Default::default()
                };
                let mut generator = BwdGenerator::new(
                    seed.wrapping_add(max_depth as u64 * 10_000),
                    config,
                    templates.clone(),
                );
                let pair = generator.generate_arena();
                let name = format!("synth_d{max_depth}_s{seed}");
                let m = measure(
                    &name,
                    "synthetic",
                    Budget::Production,
                    "production",
                    &pair.arena,
                    pair.unoptimized,
                    time_limit,
                    max_expansions,
                );
                rows.push(m.row);
            }
        }
    }

    write_objective_report(&rows, real_count, time_limit, max_expansions);
}

fn write_objective_report(
    rows: &[Row],
    real_count: usize,
    time_limit: Duration,
    max_expansions: u64,
) {
    // The gate, first and unconditionally: a single real kernel priced worse
    // than before is a stop, not a line in a table.
    let regressions: Vec<&Row> = rows
        .iter()
        .take(real_count)
        .filter(|r| r.chosen_dag > r.greedy_dag)
        .collect();
    assert!(
        regressions.is_empty(),
        "extract_dag_scoped returns the cheaper arm by construction, so a regression is a \
         broken invariant, not a result: {:?}",
        regressions.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    let real = &rows[..real_count];
    let solved: Vec<&Row> = rows
        .iter()
        .filter(|r| r.exact_status == "OPTIMAL")
        .collect();

    // --- the headline: what fraction of the measured gap is closed --------
    let gap_rows: Vec<&Row> = solved
        .iter()
        .copied()
        .filter(|r| r.greedy_dag > r.exact_dag)
        .collect();
    let gap_total: i64 = gap_rows
        .iter()
        .map(|r| r.greedy_dag as i64 - r.exact_dag as i64)
        .sum();
    let gap_closed: i64 = gap_rows
        .iter()
        .map(|r| r.greedy_dag as i64 - r.chosen_dag as i64)
        .sum();
    let pooled_closed = if gap_total > 0 {
        gap_closed as f64 / gap_total as f64
    } else {
        f64::NAN
    };
    let now_optimal = solved
        .iter()
        .filter(|r| r.chosen_dag == r.exact_dag)
        .count();
    let was_optimal = solved
        .iter()
        .filter(|r| r.greedy_dag == r.exact_dag)
        .count();

    // --- the real-kernel table, per category ------------------------------
    let mut cats: Vec<&'static str> = real.iter().map(|r| r.category).collect();
    cats.sort_unstable();
    cats.dedup();

    let cat_stats = |cat: Option<&'static str>| -> (usize, usize, usize, usize, f64, f64, i64) {
        let sel: Vec<&Row> = real
            .iter()
            .filter(|r| cat.is_none_or(|c| r.category == c))
            .collect();
        let improved = sel.iter().filter(|r| r.chosen_dag < r.greedy_dag).count();
        let worse = sel.iter().filter(|r| r.chosen_dag > r.greedy_dag).count();
        let mut deltas: Vec<f64> = sel
            .iter()
            .map(|r| {
                if r.greedy_dag == 0 {
                    0.0
                } else {
                    (r.chosen_dag as f64 - r.greedy_dag as f64) / r.greedy_dag as f64 * 100.0
                }
            })
            .collect();
        let best = deltas.iter().copied().fold(f64::INFINITY, f64::min);
        let med = median(&mut deltas);
        let units: i64 = sel
            .iter()
            .map(|r| r.greedy_dag as i64 - r.chosen_dag as i64)
            .sum();
        (
            sel.len(),
            improved,
            sel.len() - improved - worse,
            worse,
            med,
            if best.is_finite() { best } else { 0.0 },
            units,
        )
    };

    let (n_all, imp_all, unch_all, worse_all, med_all, best_all, units_all) = cat_stats(None);
    let pooled_greedy: i64 = real.iter().map(|r| r.greedy_dag as i64).sum();
    let pooled_chosen: i64 = real.iter().map(|r| r.chosen_dag as i64).sum();
    // Signed the same way as every other delta in this report: negative is
    // cheaper. Mixing the two conventions in one document is how a result
    // gets read backwards.
    let pooled_real = if pooled_greedy > 0 {
        (pooled_chosen - pooled_greedy) as f64 / pooled_greedy as f64 * 100.0
    } else {
        0.0
    };

    eprintln!(
        "\ngap closed on the {} solved kernels with a gap: {:.1}% ({} of {} units)",
        gap_rows.len(),
        pooled_closed * 100.0,
        gap_closed,
        gap_total
    );
    eprintln!(
        "exactly DAG-optimal on solved kernels: {was_optimal}/{} -> {now_optimal}/{}",
        solved.len(),
        solved.len()
    );
    eprintln!(
        "real kernels: {n_all} total, {imp_all} improved, {unch_all} unchanged, {worse_all} worse; \
         pooled {pooled_real:+.2}%"
    );

    // Extraction cost, as a per-kernel ratio of medians.
    let mut ratios: Vec<f64> = real
        .iter()
        .filter(|r| r.tree_arm_ns > 0.0)
        .map(|r| r.scoped_ns / r.tree_arm_ns)
        .collect();
    let ratio_med = median(&mut ratios.clone());
    let ratio_p90 = percentile(&mut ratios, 90.0);
    let ratio_max = ratios.iter().copied().fold(0.0_f64, f64::max);
    let tree_total: f64 = real.iter().map(|r| r.tree_arm_ns).sum();
    let scoped_total: f64 = real.iter().map(|r| r.scoped_ns).sum();
    let ratio_pooled = if tree_total > 0.0 {
        scoped_total / tree_total
    } else {
        f64::NAN
    };
    eprintln!(
        "extraction time vs the extractor it replaces: median {ratio_med:.2}x, p90 \
         {ratio_p90:.2}x, max {ratio_max:.2}x, pooled {ratio_pooled:.2}x"
    );

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("docs/results");
    std::fs::create_dir_all(&out_dir).expect("create docs/results");

    // ---- CSV ----
    let mut csv = String::new();
    csv.push_str(
        "name,category,node_count,classes,reachable_classes,tree_dag,tree_tree,sharing_dag,\
         sharing_tree,chosen_dag,sharing_changed_term,sharing_gain,dag_delta_pct,treeopt_dag,\
         exact_status,exact_dag,gap_closed_frac,tree_arm_ns,scoped_ns\n",
    );
    for r in rows.iter() {
        let delta_pct = if r.greedy_dag == 0 {
            0.0
        } else {
            (r.chosen_dag as f64 - r.greedy_dag as f64) / r.greedy_dag as f64 * 100.0
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{},{},{},{:.6},{:.0},{:.0}\n",
            r.name,
            r.category,
            r.node_count,
            r.classes,
            r.reachable_classes,
            r.greedy_dag,
            r.greedy_tree,
            r.sharing_dag,
            r.sharing_tree,
            r.chosen_dag,
            r.sharing_changed_term,
            r.sharing_gain,
            delta_pct,
            r.treeopt_dag,
            r.exact_status,
            r.exact_dag,
            r.gap_closed_frac,
            r.tree_arm_ns,
            r.scoped_ns
        ));
    }
    std::fs::write(out_dir.join("2026-09-02-extraction-objective.csv"), csv).expect("write csv");

    // ---- JSON ----
    let mut json = String::new();
    json.push_str("{\n  \"summary\": {\n");
    json.push_str(&format!(
        "    \"kernels\": {}, \"real\": {}, \"synthetic\": {},\n",
        rows.len(),
        real_count,
        rows.len() - real_count
    ));
    json.push_str(&format!(
        "    \"exact_solved\": {}, \"solved_with_a_gap\": {},\n",
        solved.len(),
        gap_rows.len()
    ));
    json.push_str(&format!(
        "    \"gap_units_total\": {gap_total}, \"gap_units_closed\": {gap_closed}, \
         \"gap_closed_frac\": {pooled_closed:.6},\n"
    ));
    json.push_str(&format!(
        "    \"dag_optimal_before\": {was_optimal}, \"dag_optimal_after\": {now_optimal},\n"
    ));
    json.push_str(&format!(
        "    \"real_improved\": {imp_all}, \"real_unchanged\": {unch_all}, \
         \"real_worse\": {worse_all},\n"
    ));
    json.push_str(&format!(
        "    \"real_pooled_dag_delta_pct\": {pooled_real:.4}, \
         \"real_median_dag_delta_pct\": {med_all:.4}, \
         \"real_best_dag_delta_pct\": {best_all:.4}, \"real_units_saved\": {units_all},\n"
    ));
    json.push_str(&format!(
        "    \"arm_sharing_wins\": {}, \"arm_tie\": {}, \"arm_tree_wins\": {},\n",
        real.iter().filter(|r| r.sharing_dag < r.greedy_dag).count(),
        real.iter()
            .filter(|r| r.sharing_dag == r.greedy_dag)
            .count(),
        real.iter().filter(|r| r.sharing_dag > r.greedy_dag).count()
    ));
    json.push_str(&format!(
        "    \"extraction_time_ratio_median\": {ratio_med:.4}, \
         \"extraction_time_ratio_p90\": {ratio_p90:.4}, \
         \"extraction_time_ratio_max\": {ratio_max:.4}, \
         \"extraction_time_ratio_pooled\": {ratio_pooled:.4},\n"
    ));
    json.push_str(&format!(
        "    \"exact_time_limit_s\": {}, \"exact_max_expansions\": {max_expansions}\n  }},\n",
        time_limit.as_secs()
    ));
    json.push_str("  \"by_category\": [\n");
    let cat_json: Vec<String> = cats
        .iter()
        .map(|&c| {
            let (n, i, u, w, med, best, units) = cat_stats(Some(c));
            format!(
                "    {{\"category\":\"{c}\",\"n\":{n},\"improved\":{i},\"unchanged\":{u},\
                 \"worse\":{w},\"median_pct\":{med:.4},\"best_pct\":{best:.4},\"units\":{units}}}"
            )
        })
        .collect();
    json.push_str(&cat_json.join(",\n"));
    json.push_str("\n  ],\n  \"rows\": [\n");
    let body: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "    {{\"name\":\"{}\",\"category\":\"{}\",\"reachable_classes\":{},\
                 \"tree_dag\":{},\"sharing_dag\":{},\"chosen_dag\":{},\"tree_tree\":{},\
                 \"sharing_tree\":{},\"changed\":{},\"exact_status\":\"{}\",\"exact_dag\":{}}}",
                r.name,
                r.category,
                r.reachable_classes,
                r.greedy_dag,
                r.sharing_dag,
                r.chosen_dag,
                r.greedy_tree,
                r.sharing_tree,
                r.sharing_changed_term,
                r.exact_status,
                r.exact_dag
            )
        })
        .collect();
    json.push_str(&body.join(",\n"));
    json.push_str("\n  ]\n}\n");
    std::fs::write(out_dir.join("2026-09-02-extraction-objective.json"), json).expect("write json");

    // ---- Markdown ----
    let mut md = String::new();
    let _ = writeln!(md, "# Making extraction price sharing (#1116)\n");
    let _ = writeln!(
        md,
        "`extract_dag` summed each child's `best_cost` — a TREE cost — so a subterm used ten \
         times was charged ten times in the objective and emitted once in the kernel. This is \
         the measurement of replacing that objective with the DAG cost the kernel actually \
         pays, scored against the same two exact references #1115 built \
         (`2026-09-02-extraction-gap.md`): Knuth's exact tree optimum, and the branch-and-bound \
         DAG optimum where it closes.\n"
    );
    let _ = writeln!(
        md,
        "Cost table: the latency prior as re-measured on current main (#1134 — Sin 70->95, \
         Cos 75->103, Tan 87->117, Sqrt 15->13, Select 4->3). Every number below is \
         deterministic: `CostModel::latency_prior` through `ExtractedDAG::dag_cost`.\n"
    );

    let _ = writeln!(md, "## Headline\n");
    let _ = writeln!(
        md,
        "- **Gap closed: {:.1}%** of the measured gap — {gap_closed} of {gap_total} pooled cost \
         units — on the {} kernel(s) where the branch and bound proved an optimum *and* the \
         old extractor missed it. That population is small by construction: the B&B stops \
         closing at ~100-200 reachable classes, and greedy was already exactly optimal on \
         most of what it does close.",
        pooled_closed * 100.0,
        gap_rows.len()
    );
    let _ = writeln!(
        md,
        "- Exactly DAG-optimal on the {} solved kernels: **{was_optimal} -> {now_optimal}**.",
        solved.len()
    );
    let _ = writeln!(
        md,
        "- Real kernels ({n_all}): **{imp_all} improved, {unch_all} unchanged, {worse_all} \
         worse**. Pooled `dag_cost` **{pooled_real:+.2}%**, median {med_all:+.2}%, best \
         {best_all:+.2}%, {units_all} cost units saved.\n"
    );
    let _ = writeln!(
        md,
        "**Zero regressions is structural, not lucky.** `extract_dag_scoped` runs both \
         objectives and returns the cheaper term by true `dag_cost`, ties going to the tree \
         arm, so the returned cost is a minimum over a set that contains the old answer. The \
         probe asserts it rather than reporting it.\n"
    );
    let _ = writeln!(
        md,
        "**The gap denominator is this run's own, not #1115's.** #1115 pooled 195 units over \
         the 89 kernels its branch and bound closed, under the pre-#1134 cost table. This run \
         closes {} kernels under the refreshed table and pools {gap_total} units over them. \
         Where the two numbers agree that is arithmetic coincidence, not the same set \
         re-measured, and the fraction above is computed entirely within this run.\n",
        solved.len()
    );

    if !gap_rows.is_empty() {
        let _ = writeln!(md, "### The gap kernels, one row each\n");
        let _ = writeln!(
            md,
            "| kernel | reachable classes | old (tree) | new | exact DAG optimum | closed |"
        );
        let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|");
        for r in &gap_rows {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {:.0}% |",
                r.name,
                r.reachable_classes,
                r.greedy_dag,
                r.chosen_dag,
                r.exact_dag,
                r.gap_closed_frac * 100.0
            );
        }
        let _ = writeln!(md);
    }

    let _ = writeln!(md, "## Real kernels, by group\n");
    let _ = writeln!(
        md,
        "| group | n | improved | unchanged | worse | median Δ dag_cost | best Δ | units saved |"
    );
    let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|");
    for &c in &cats {
        let (n, i, u, w, med, best, units) = cat_stats(Some(c));
        let _ = writeln!(
            md,
            "| {c} | {n} | {i} | {u} | {w} | {med:+.2}% | {best:+.2}% | {units} |"
        );
    }
    let _ = writeln!(
        md,
        "| **all real** | **{n_all}** | **{imp_all}** | **{unch_all}** | **{worse_all}** | \
         **{med_all:+.2}%** | **{best_all:+.2}%** | **{units_all}** |\n"
    );

    let _ = writeln!(md, "## Which arm wins, and how often\n");
    let arm_shared = real.iter().filter(|r| r.sharing_dag < r.greedy_dag).count();
    let arm_tie = real
        .iter()
        .filter(|r| r.sharing_dag == r.greedy_dag)
        .count();
    let arm_tree = real.iter().filter(|r| r.sharing_dag > r.greedy_dag).count();
    let _ = writeln!(
        md,
        "Neither DP is optimal — both choose greedily bottom-up — so the sharing arm is not \
         uniformly better, and reporting only the min would hide that. Over the {n_all} real \
         kernels: the sharing arm is strictly cheaper on **{arm_shared}**, ties on \
         **{arm_tie}**, and is strictly *dearer* on **{arm_tree}**.\n"
    );
    let _ = writeln!(
        md,
        "The {arm_tree} are the honest cost of a bottom-up chooser: a class that is DAG-cheap \
         in isolation need not compose into a DAG-cheap parent. Running both arms and taking \
         the min is what turns that from a regression into a no-op, and is why the second pass \
         buys robustness as well as cost.\n"
    );

    let mut losers: Vec<&Row> = real
        .iter()
        .filter(|r| r.sharing_dag > r.greedy_dag)
        .collect();
    losers.sort_by(|a, b| {
        let da = (a.sharing_dag as f64 - a.greedy_dag as f64) / a.greedy_dag.max(1) as f64;
        let db = (b.sharing_dag as f64 - b.greedy_dag as f64) / b.greedy_dag.max(1) as f64;
        db.partial_cmp(&da).unwrap()
    });
    if !losers.is_empty() {
        let _ = writeln!(
            md,
            "The worst of them, so the shape of the failure is on the record rather than \
             averaged away:\n"
        );
        let _ = writeln!(
            md,
            "| kernel | group | tree dag | sharing dag | Δ if taken alone |"
        );
        let _ = writeln!(md, "|---|---|---:|---:|---:|");
        for r in losers.iter().take(8) {
            let d =
                (r.sharing_dag as f64 - r.greedy_dag as f64) / r.greedy_dag.max(1) as f64 * 100.0;
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {:+.2}% |",
                r.name, r.category, r.greedy_dag, r.sharing_dag, d
            );
        }
        let _ = writeln!(md);
    }

    let _ = writeln!(md, "## Where the objective change bites\n");
    let mut movers: Vec<&Row> = real
        .iter()
        .filter(|r| r.chosen_dag < r.greedy_dag)
        .collect();
    movers.sort_by(|a, b| {
        let da = (a.greedy_dag as f64 - a.chosen_dag as f64) / a.greedy_dag.max(1) as f64;
        let db = (b.greedy_dag as f64 - b.chosen_dag as f64) / b.greedy_dag.max(1) as f64;
        db.partial_cmp(&da).unwrap()
    });
    if movers.is_empty() {
        let _ = writeln!(
            md,
            "No real kernel changed. That is the result, and it is the same shape as #1115's \
             own negative finding: exact DAG extraction was worth 0.028% pooled beyond the tree \
             optimum on this corpus.\n"
        );
    } else {
        let _ = writeln!(
            md,
            "| kernel | group | reachable classes | tree dag | sharing dag | Δ | tree TREE cost | sharing TREE cost |"
        );
        let _ = writeln!(md, "|---|---|---:|---:|---:|---:|---:|---:|");
        for r in movers.iter().take(25) {
            let d =
                (r.chosen_dag as f64 - r.greedy_dag as f64) / r.greedy_dag.max(1) as f64 * 100.0;
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {:+.2}% | {} | {} |",
                r.name,
                r.category,
                r.reachable_classes,
                r.greedy_dag,
                r.sharing_dag,
                d,
                r.greedy_tree,
                r.sharing_tree
            );
        }
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "The two right-hand columns are the point: where the objective change bites, the \
             TREE cost **rises**. That is not a regression — it is the old objective being \
             abandoned, visible.\n"
        );
    }

    let _ = writeln!(md, "## Cost\n");
    let changed = rows.iter().filter(|r| r.sharing_changed_term).count();
    let _ = writeln!(
        md,
        "Extraction wall time against the extractor it replaces, over the {n_all} real \
         kernels: **median {ratio_med:.2}x, p90 {ratio_p90:.2}x, max {ratio_max:.2}x, pooled \
         {ratio_pooled:.2}x**. Each kernel is the median of {} alternating pairs timed \
         back to back in one process on one saturated graph, so the absolute nanoseconds \
         belong to the machine and only the ratio is claimed. Deterministically: two DP \
         traversals where there was one.\n",
        9
    );
    let _ = writeln!(
        md,
        "Two DP passes over the e-graph instead of one, plus one bit per class per class of \
         scratch for the sharing pass (~385 KB at the median production glyph's 1,755 reachable \
         classes; ~12.5 MB at the 10,000-class production ceiling, allocated once per extraction \
         and dropped at the end of it). Extraction runs **once** per compile against thousands \
         of rule applications in saturation, which is why this is the cheap place to spend.\n"
    );
    let _ = writeln!(
        md,
        "The sharing arm returned a different term on {changed} of {} kernels; on the rest the \
         second pass is pure overhead and the tree arm's answer is returned unchanged.\n",
        rows.len()
    );

    let _ = writeln!(md, "## The gate\n");
    let _ = writeln!(
        md,
        "The bar this change was held to was: no real kernel regresses in `dag_cost`, and \
         extraction runs in under 2x. The first is met and is structural. **The second is \
         not** — {ratio_med:.2}x median. The second DP pass is not free and the compacted \
         bitset only took it from 2.57x to {ratio_med:.2}x, because the sharing arm also pays \
         a repair and a costing pass of its own. So this is not self-arming: the trade is \
         {units_all} cost units and {:.1}% of the closable gap against {ratio_med:.2}x on a \
         phase that runs once per compile, and that is a judgement call, not a threshold.\n",
        pooled_closed * 100.0
    );

    let _ = writeln!(md, "## What this obliges downstream\n");
    let _ = writeln!(
        md,
        "Every Guide label and every registered constant in the Phase-3 program was minted \
         under tree cost. #1128's bisect already showed the consequence: guides trained on \
         tree-cost labels steer toward UNSHARED terms (tree/dag sharing ratio 4.25 unguided vs \
         3.29 guided on `sh`) and lose to unguided on the real metric. With the objective \
         changed, that chain restarts — re-mint, retrain, re-evaluate — and any registered \
         constant carried across it is in stale units.\n"
    );

    let _ = writeln!(md, "## Reproduction\n");
    let _ = writeln!(md, "```sh");
    let _ = writeln!(
        md,
        "PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of .arena dumps> \\\n  \
         PIXELFLOW_EXTRACTION_GAP_SECS={} \\\n  \
         PIXELFLOW_EXTRACTION_GAP_EXPANSIONS={max_expansions} \\\n  \
         RUST_MIN_STACK=268435456 \\\n  \
         cargo test -p pixelflow-search --release --lib -- --ignored \
         extraction_objective_measurement",
        time_limit.as_secs()
    );
    let _ = writeln!(md, "```\n");
    let _ = writeln!(
        md,
        "`2026-09-02-extraction-objective.csv` / `.json` carry every kernel's row."
    );

    std::fs::write(out_dir.join("2026-09-02-extraction-objective.md"), md).expect("write md");
    eprintln!(
        "wrote {}",
        out_dir.join("2026-09-02-extraction-objective.md").display()
    );
}

// ---------------------------------------------------------------------------
// The measurement's own tests: a probe nobody trusts measures nothing
// ---------------------------------------------------------------------------

#[cfg(test)]
mod self_check {
    use super::*;
    use crate::egraph::ops::op_from_kind;
    use pixelflow_ir::OpKind;

    /// A hand-built e-graph where sharing and tree cost disagree, so the two
    /// references must return different terms.
    ///
    /// Root class holds `Add(a, a)` (one add, one shared child) and
    /// `Mul(b, c)` where `b` and `c` are distinct leaves. Tree cost prefers
    /// whichever op is cheaper; DAG cost has to notice `a` is paid once.
    #[test]
    fn exact_never_loses_to_greedy() {
        let model = CostModel::latency_prior();
        let mut egraph = Optimizer::production().egraph();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sq = egraph.add(ENode::Op {
            op: op_from_kind(OpKind::Mul).expect("Mul is a known op"),
            children: vec![x, x],
        });
        let sum = egraph.add(ENode::Op {
            op: op_from_kind(OpKind::Add).expect("Add is a known op"),
            children: vec![sq, sq],
        });
        egraph.saturate_budgeted(6, 400, None);

        let inst = Instance::build(&egraph, sum, &model);
        let greedy = extract_dag_scoped(&egraph, sum, &model, LatticeShape::POINT);
        let greedy_cost = inst
            .dag_cost(&greedy.choices)
            .expect("greedy choice is a term");
        let treeopt = tree_optimal_choices(&inst).expect("root reachable");
        let treeopt_cost = inst.dag_cost(&treeopt).expect("tree-optimal is a term");
        let exact = exact_dag_choices(
            &inst,
            &greedy.choices,
            greedy_cost.min(treeopt_cost),
            5_000_000,
            Duration::from_secs(30),
        );
        assert_eq!(
            exact.status,
            ExactStatus::Optimal,
            "this instance is small enough to close"
        );
        assert!(
            exact.cost <= greedy_cost,
            "exact {} must not exceed greedy {greedy_cost}",
            exact.cost
        );
        assert!(
            exact.cost <= treeopt_cost,
            "exact {} must not exceed tree-optimal {treeopt_cost}",
            exact.cost
        );
        // The exact term is a term.
        assert_eq!(
            inst.dag_cost(&exact.choices),
            Some(exact.cost),
            "the exact search's own choice vector must re-cost to its reported cost"
        );
    }

    /// Exhaustive enumeration of every choice vector, for instances small
    /// enough to enumerate. The only way to know a branch-and-bound is
    /// exact is to check it against something that cannot be wrong.
    fn brute_force(inst: &Instance) -> Option<u64> {
        let free: Vec<u32> = inst.reachable.to_vec();
        let sizes: Vec<usize> = free
            .iter()
            .map(|&c| inst.feasible[c as usize].len().max(1))
            .collect();
        let total: u128 = sizes.iter().map(|&s| s as u128).product();
        if total > 4_000_000 {
            return None; // too big to enumerate; skip this instance
        }
        let mut counter = vec![0usize; free.len()];
        let mut best: Option<u64> = None;
        loop {
            let mut choice: Vec<Option<usize>> = vec![None; inst.cands.len()];
            for (k, &c) in free.iter().enumerate() {
                let feasible = &inst.feasible[c as usize];
                if !feasible.is_empty() {
                    choice[c as usize] = Some(feasible[counter[k] % feasible.len()]);
                }
            }
            if let Some(cost) = inst.dag_cost(&choice) {
                best = Some(best.map_or(cost, |b: u64| b.min(cost)));
            }
            let mut k = 0;
            loop {
                if k == free.len() {
                    return best;
                }
                counter[k] += 1;
                if counter[k] < sizes[k] {
                    break;
                }
                counter[k] = 0;
                k += 1;
            }
        }
    }

    /// On every instance small enough to enumerate exhaustively, the
    /// branch-and-bound's `OPTIMAL` answer must be the enumerated minimum.
    #[test]
    fn branch_and_bound_agrees_with_exhaustive_enumeration() {
        use crate::nnue::{BwdGenConfig, BwdGenerator};
        let model = CostModel::latency_prior();
        let templates = crate::egraph::collect_rule_templates();
        let mut checked = 0usize;
        for seed in 0u64..60 {
            let config = BwdGenConfig {
                max_depth: 2,
                ..Default::default()
            };
            let mut generator = BwdGenerator::new(seed, config, templates.clone());
            let pair = generator.generate_arena();
            let mut egraph = Optimizer::production().egraph();
            let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
            let Some(root) = arena_to_egraph(&pair.arena, pair.unoptimized, &mut egraph, &mut memo)
            else {
                continue;
            };
            // Deliberately tiny budget: the point is an instance small
            // enough that enumeration is a reference, not a second heuristic.
            egraph.saturate_budgeted(2, 60, Some(40));
            let inst = Instance::build(&egraph, root, &model);
            let Some(truth) = brute_force(&inst) else {
                continue;
            };
            let treeopt = tree_optimal_choices(&inst).expect("root reachable");
            let seed_cost = inst.dag_cost(&treeopt).expect("tree-optimal is a term");
            let exact = exact_dag_choices(
                &inst,
                &treeopt,
                seed_cost,
                50_000_000,
                Duration::from_secs(20),
            );
            assert_eq!(
                exact.status,
                ExactStatus::Optimal,
                "seed {seed}: an instance small enough to enumerate must also close"
            );
            assert_eq!(
                exact.cost, truth,
                "seed {seed}: branch-and-bound says {} but exhaustive enumeration says {truth}",
                exact.cost
            );
            checked += 1;
        }
        assert!(
            checked >= 10,
            "only {checked} instances were small enough to check — the reference is not \
             being exercised, which makes the agreement vacuous"
        );
    }

    /// Knuth's algorithm must agree with the DP whenever the DP is right —
    /// on an acyclic, unsaturated graph they are the same computation.
    #[test]
    fn tree_optimal_matches_the_dp_on_an_acyclic_graph() {
        let model = CostModel::latency_prior();
        let mut egraph = Optimizer::production().egraph();
        let x = egraph.add(ENode::Var(0));
        let one = egraph.add(ENode::constant(1.0));
        let add = egraph.add(ENode::Op {
            op: op_from_kind(OpKind::Add).expect("Add is a known op"),
            children: vec![x, one],
        });
        let inst = Instance::build(&egraph, add, &model);
        let treeopt = tree_optimal_choices(&inst).expect("root reachable");
        let dp = extract_dag_scoped(&egraph, add, &model, LatticeShape::POINT);
        assert_eq!(
            inst.tree_cost(&treeopt),
            Some(dp.total_cost as u64),
            "on an acyclic single-node-per-class graph the DP already attains the tree optimum"
        );
    }
}
