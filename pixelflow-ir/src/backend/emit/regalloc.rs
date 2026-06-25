//! Graph coloring register allocator for DAG expressions.
//!
//! Sethi-Ullman is optimal for trees but suboptimal for DAGs with sharing.
//! Graph coloring handles shared subexpressions properly.
//!
//! ## Algorithm (Hack et al. 2006)
//!
//! Based on "Register Allocation for Programs in SSA Form" by Hack, Grund, and Goos.
//! Key insight: SSA-form programs produce **chordal** interference graphs.
//!
//! 1. **Liveness analysis**: Compute live ranges for each value
//! 2. **Build interference graph**: Values live at same point interfere
//! 3. **Simplicial elimination ordering**: Maximum cardinality search (MCS)
//! 4. **Greedy coloring**: Optimal for chordal graphs
//!
//! For chordal graphs, this produces an optimal coloring in O(V + E) time.
//! Expression DAGs from e-graph extraction are always in SSA form.

use alloc::collections::BTreeMap;
use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;

use super::Reg;

/// A value in the program (SSA-style).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// Interference graph for register allocation.
///
/// Two values interfere if they're both live at the same program point.
/// The graph is undirected - if A interferes with B, B interferes with A.
///
/// Internally uses dense Vecs indexed by `ValueId.0` for O(1) lookup.
#[derive(Debug, Default)]
pub struct InterferenceGraph {
    /// All values in the graph, in insertion order.
    values: Vec<ValueId>,
    /// Adjacency list: `neighbors[vid.0]` = Vec of interfering ValueIds.
    /// Empty Vec for values with no neighbors.
    neighbors: Vec<Vec<ValueId>>,
    /// Pre-colored values: `precolored[vid.0]` = Some(Reg) if pre-colored.
    precolored: Vec<Option<Reg>>,
    /// Capacity: `max(ValueId.0) + 1` across all inserted values.
    capacity: usize,
}

impl InterferenceGraph {
    /// Create an empty interference graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow dense Vecs to accommodate a ValueId.
    fn ensure_capacity(&mut self, v: ValueId) {
        let idx = v.0 as usize + 1;
        if idx > self.capacity {
            self.capacity = idx;
            self.neighbors.resize_with(idx, Vec::new);
            self.precolored.resize(idx, None);
        }
    }

    /// Add a value to the graph (with no interferences yet).
    pub fn add_value(&mut self, v: ValueId) {
        self.ensure_capacity(v);
        // Only push to the values list if not already present.
        // The neighbors Vec already has an entry from ensure_capacity.
        if !self.values.contains(&v) {
            self.values.push(v);
        }
    }

    /// Add an interference edge between two values.
    pub fn add_edge(&mut self, a: ValueId, b: ValueId) {
        if a == b {
            return;
        }
        self.ensure_capacity(a);
        self.ensure_capacity(b);

        let ai = a.0 as usize;
        let bi = b.0 as usize;

        // Push unconditionally — dedup happens in finalize().
        // The old code did .contains() which is O(degree) per insertion,
        // making the entire build O(n × degree²). With push + sort/dedup
        // at the end, it's O(edges × log(degree)).
        self.neighbors[ai].push(b);
        self.neighbors[bi].push(a);

        // Ensure both appear in the values list.
        if !self.values.contains(&a) {
            self.values.push(a);
        }
        if !self.values.contains(&b) {
            self.values.push(b);
        }
    }

    /// Mark a value as pre-colored (must use specific register).
    pub fn precolor(&mut self, v: ValueId, reg: Reg) {
        self.add_value(v);
        self.precolored[v.0 as usize] = Some(reg);
    }

    /// Get the degree of a value (number of interferences).
    #[must_use]
    pub fn degree(&self, v: ValueId) -> usize {
        let idx = v.0 as usize;
        if idx < self.capacity {
            self.neighbors[idx].len()
        } else {
            0
        }
    }

    /// Get all values in the graph.
    /// Sort and deduplicate all neighbor lists. Must be called after
    /// all edges are added (add_edge pushes duplicates for speed).
    pub fn dedup_edges(&mut self) {
        for neighbors in &mut self.neighbors {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
    }

    #[must_use]
    pub fn values(&self) -> &[ValueId] {
        &self.values
    }

    /// Get neighbors of a value.
    #[must_use]
    pub fn neighbors(&self, v: ValueId) -> &[ValueId] {
        let idx = v.0 as usize;
        if idx < self.capacity {
            &self.neighbors[idx]
        } else {
            &[]
        }
    }

    /// Check if a value is pre-colored.
    #[must_use]
    pub fn is_precolored(&self, v: ValueId) -> bool {
        let idx = v.0 as usize;
        idx < self.capacity && self.precolored[idx].is_some()
    }

    /// Get the pre-assigned register for a value, if any.
    #[must_use]
    pub fn precolor_of(&self, v: ValueId) -> Option<Reg> {
        let idx = v.0 as usize;
        if idx < self.capacity {
            self.precolored[idx]
        } else {
            None
        }
    }

    /// Number of values in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Result of register allocation.
#[derive(Debug)]
pub struct RegAllocation {
    /// Mapping from value to assigned register.
    pub assignment: BTreeMap<ValueId, Reg>,
    /// Values that couldn't be colored (need spilling to stack).
    pub spilled: Vec<ValueId>,
    /// Values that were evicted but can be rematerialized (constants).
    /// These don't need spill slots — just re-emit the constant load.
    pub rematerialize: BTreeMap<ValueId, u32>,
    /// Number of registers used.
    pub num_regs: u8,
}

/// Greedy graph coloring with simplicial elimination ordering.
///
/// This is optimal for chordal graphs, which expression DAGs always produce.
/// For non-chordal graphs, it's a good heuristic.
///
/// # Arguments
/// * `graph` - The interference graph
/// * `num_regs` - Number of available registers
/// * `scratch_base` - First scratch register index
#[must_use]
pub fn color_graph(graph: &InterferenceGraph, num_regs: u8, scratch_base: u8) -> RegAllocation {
    // Dense assignment Vec: assignment[vid.0] = Some(Reg) if colored.
    let capacity = graph.capacity;
    let mut assignment_dense: Vec<Option<Reg>> = vec![None; capacity.max(1)];
    let mut spilled: Vec<ValueId> = Vec::new();

    // Copy pre-colored assignments.
    for &v in graph.values() {
        if let Some(reg) = graph.precolor_of(v) {
            assignment_dense[v.0 as usize] = Some(reg);
        }
    }

    // Build simplicial elimination ordering (reverse order for greedy coloring).
    let ordering = simplicial_elimination_order(graph);

    // Greedy coloring in the computed order.
    // used_colors[c] = true if color c is taken by a neighbor.
    let mut used_colors: Vec<bool> = vec![false; (scratch_base as usize) + (num_regs as usize)];

    for v in ordering {
        if assignment_dense[v.0 as usize].is_some() {
            continue; // Already pre-colored.
        }

        // Mark colors used by neighbors.
        let mut marked: Vec<usize> = Vec::new();
        for &neighbor in graph.neighbors(v) {
            if let Some(Reg(c)) = assignment_dense[neighbor.0 as usize] {
                let ci = c as usize;
                if ci < used_colors.len() && !used_colors[ci] {
                    used_colors[ci] = true;
                    marked.push(ci);
                }
            }
        }

        // Find first available color in scratch range.
        let mut color = None;
        for c in scratch_base..(scratch_base + num_regs) {
            if !used_colors[c as usize] {
                color = Some(c);
                break;
            }
        }

        // Clear marks for next iteration.
        for ci in marked {
            used_colors[ci] = false;
        }

        match color {
            Some(c) => {
                assignment_dense[v.0 as usize] = Some(Reg(c));
            }
            None => {
                spilled.push(v);
            }
        }
    }

    // Build the public BTreeMap assignment from the dense Vec.
    let mut assignment: BTreeMap<ValueId, Reg> = BTreeMap::new();
    for &v in graph.values() {
        if let Some(reg) = assignment_dense[v.0 as usize] {
            assignment.insert(v, reg);
        }
    }

    // Count registers used.
    let max_reg = assignment
        .values()
        .filter(|r| r.0 >= scratch_base)
        .map(|r| r.0 - scratch_base + 1)
        .max()
        .unwrap_or(0);

    RegAllocation {
        assignment,
        spilled,
        rematerialize: BTreeMap::new(),
        num_regs: max_reg,
    }
}

/// Compute a simplicial elimination ordering using maximum cardinality search (MCS).
///
/// MCS produces a perfect elimination ordering for chordal graphs, making
/// subsequent greedy coloring optimal. Uses a max-heap for O(n log n) rather
/// than the naive O(n²) scan.
fn simplicial_elimination_order(graph: &InterferenceGraph) -> Vec<ValueId> {
    let n = graph.len();
    if n == 0 {
        return vec![];
    }

    let capacity = graph.capacity;

    // Dense weight and membership arrays indexed by ValueId.0.
    let mut weight: Vec<usize> = vec![0; capacity];
    let mut in_remaining: Vec<bool> = vec![false; capacity];

    for &v in graph.values() {
        in_remaining[v.0 as usize] = true;
    }

    // Max-heap of (weight, ValueId). Stale entries (weight mismatch) are skipped.
    // ValueId derives Ord, so (usize, ValueId) is ordered lexicographically —
    // weight is the primary key, ValueId is the tiebreaker.
    let mut heap: BinaryHeap<(usize, ValueId)> =
        graph.values().iter().map(|&v| (0usize, v)).collect();

    let mut order = Vec::with_capacity(n);

    for _ in 0..n {
        // Pop max-weight vertex still in remaining; skip stale heap entries.
        let v = loop {
            let (w, v) = heap.pop().expect("heap must be non-empty during MCS");
            let vi = v.0 as usize;
            if in_remaining[vi] && w == weight[vi] {
                break v;
            }
            // Stale entry — the real weight is higher; skip.
        };

        in_remaining[v.0 as usize] = false;
        order.push(v);

        // Increment weights of remaining neighbors and push updated entries.
        for &neighbor in graph.neighbors(v) {
            let ni = neighbor.0 as usize;
            if in_remaining[ni] {
                weight[ni] += 1;
                heap.push((weight[ni], neighbor));
            }
        }
    }

    // Reverse so that simplicial vertices (colored last in MCS) are colored first.
    order.reverse();
    order
}

/// Build interference graph from a scheduled DAG.
///
/// # Arguments
/// * `schedule` - Topologically sorted list of (value_id, definition)
/// * `uses` - For each value, list of values that use it
///
/// Returns an interference graph where two values interfere if
/// their live ranges overlap.
pub fn build_interference_graph<D, F>(schedule: &[(ValueId, D)], uses_of: F) -> InterferenceGraph
where
    F: Fn(ValueId) -> Vec<ValueId>,
{
    let mut graph = InterferenceGraph::new();

    // Add all values.
    for (v, _) in schedule {
        graph.add_value(*v);
    }

    // Compute max ValueId for the dense live bitvec.
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let live_capacity = max_vid + 1;

    // Dense live set: live[vid.0] = true if the value is currently live.
    // live_list tracks which values are live for O(|live|) iteration.
    let mut live: Vec<bool> = vec![false; live_capacity];
    let mut live_list: Vec<ValueId> = Vec::new();

    // Walk schedule in reverse (backward liveness analysis).
    for (v, _) in schedule.iter().rev() {
        let vi = v.0 as usize;

        // All currently live values interfere with v.
        for &other in &live_list {
            graph.add_edge(*v, other);
        }

        // v is now live (just defined).
        if vi < live_capacity && !live[vi] {
            live[vi] = true;
            live_list.push(*v);
        }

        // Add uses of v to the live set.
        for used in uses_of(*v) {
            let ui = used.0 as usize;
            if ui < live_capacity && !live[ui] {
                live[ui] = true;
                live_list.push(used);
            }
        }
    }

    graph.dedup_edges();
    graph
}

// ============================================================================
// Linear scan register allocator
// ============================================================================

/// Linear scan register allocation for scheduled DAGs.
///
/// Walks the schedule forward in program order. At each definition:
/// 1. Free registers holding values whose last use has passed.
/// 2. Assign a free scratch register if available.
/// 3. If no register is free, spill the currently-live value whose next use
///    is farthest in the future (Belady's optimal eviction).
///
/// Pre-colored values (variables bound to input registers) are handled by
/// marking those registers as occupied for the duration of the value's
/// live range.
///
/// This is O(n * k) where n = schedule length, k = number of registers.
/// For our use case (k=22 scratch regs), this is effectively O(n).
#[must_use]
pub fn linear_scan(
    schedule: &[(ValueId, super::ScheduledOp)],
    _uses_map: &[Vec<ValueId>],
    precolored: &BTreeMap<ValueId, super::Reg>,
    num_regs: u8,
    scratch_base: u8,
) -> RegAllocation {
    use super::Reg;

    let n = schedule.len();
    if n == 0 {
        return RegAllocation {
            assignment: BTreeMap::new(),
            spilled: Vec::new(),
            rematerialize: BTreeMap::new(),
            num_regs: 0,
        };
    }

    // Dense Vec indexed by ValueId.0 for O(1) lookups.
    // Find the maximum ValueId to size our vectors.
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let vec_len = max_vid + 1;

    // Build schedule index for each ValueId (definition point).
    // usize::MAX means "not defined" (should never be read for undefined ids).
    let mut def_idx: Vec<usize> = vec![usize::MAX; vec_len];
    for (i, (vid, _)) in schedule.iter().enumerate() {
        def_idx[vid.0 as usize] = i;
    }

    // Compute last-use index for each value.
    // A value's last use is the latest schedule index of any operation that
    // reads it. If nothing uses it (root), last use = its own definition.
    let mut last_use: Vec<usize> = vec![usize::MAX; vec_len];
    for (i, (vid, _)) in schedule.iter().enumerate() {
        // Default: last use is own definition (covers the root).
        if last_use[vid.0 as usize] == usize::MAX {
            last_use[vid.0 as usize] = i;
        }
    }
    // For each value, find which schedule entries use it as an operand.
    // uses_map: definer → list of values that USE this definer.
    // We need the reverse: for each value V, what schedule indices read V?
    // Walk the schedule and inspect operands.
    for (i, (_, sop)) in schedule.iter().enumerate() {
        let operands: &[ValueId] = match sop {
            super::ScheduledOp::Var(_) | super::ScheduledOp::Const(_) => &[],
            super::ScheduledOp::Unary(_, a) | super::ScheduledOp::ShiftImm(_, a, _) => {
                core::slice::from_ref(a)
            }
            super::ScheduledOp::Binary(_, a, b) => {
                // Can't make a slice from two refs, handle below
                let lu_a = &mut last_use[a.0 as usize];
                *lu_a = (*lu_a).max(i);
                let lu_b = &mut last_use[b.0 as usize];
                *lu_b = (*lu_b).max(i);
                continue;
            }
            super::ScheduledOp::Ternary(_, a, b, c) => {
                let lu_a = &mut last_use[a.0 as usize];
                *lu_a = (*lu_a).max(i);
                let lu_b = &mut last_use[b.0 as usize];
                *lu_b = (*lu_b).max(i);
                let lu_c = &mut last_use[c.0 as usize];
                *lu_c = (*lu_c).max(i);
                continue;
            }
        };
        for operand in operands {
            let lu = &mut last_use[operand.0 as usize];
            *lu = (*lu).max(i);
        }
    }

    // Identify rematerializable values (constants). These never need spill
    // slots — just re-emit the constant load instruction on use.
    // Dense Vec<Option<u32>> indexed by ValueId.0.
    let mut const_bits: Vec<Option<u32>> = vec![None; vec_len];
    for (vid, sop) in schedule {
        if let super::ScheduledOp::Const(val) = sop {
            const_bits[vid.0 as usize] = Some(val.to_bits());
        }
    }

    let mut assignment: BTreeMap<ValueId, Reg> = BTreeMap::new();
    let mut spilled: Vec<ValueId> = Vec::new();
    let mut rematerialize: BTreeMap<ValueId, u32> = BTreeMap::new();

    // reg_owner[r - scratch_base] = Some(vid) if register r holds value vid.
    let mut reg_owner: Vec<Option<ValueId>> = vec![None; num_regs as usize];

    // Handle pre-colored values (variables).
    for (&vid, &reg) in precolored {
        assignment.insert(vid, reg);
    }

    let mut max_reg_used: u8 = 0;

    for (i, (vid, _sop)) in schedule.iter().enumerate() {
        // Skip pre-colored (already assigned).
        if assignment.contains_key(vid) {
            continue;
        }

        // Free registers whose values have expired (last use < i).
        for slot in reg_owner.iter_mut() {
            if let Some(owner) = *slot {
                let lu = last_use[owner.0 as usize];
                if lu < i {
                    *slot = None;
                }
            }
        }

        // Try to find a free scratch register.
        let mut free_reg = None;
        for r in 0..num_regs {
            if reg_owner[r as usize].is_none() {
                free_reg = Some(r);
                break;
            }
        }

        if let Some(r) = free_reg {
            let reg = Reg(r + scratch_base);
            assignment.insert(*vid, reg);
            reg_owner[r as usize] = Some(*vid);
            if r + 1 > max_reg_used {
                max_reg_used = r + 1;
            }
        } else {
            // No free register — must evict.
            //
            // Eviction priority (prefer to evict cheap values):
            // 1. Constants (rematerializable — free to recompute, no memory traffic)
            // 2. Non-constants with farthest next use (Belady's optimal)

            // First: try evicting a constant with farthest next use.
            let mut best_const_slot: Option<usize> = None;
            let mut best_const_lu = 0usize;
            let mut best_any_slot = 0usize;
            let mut best_any_lu = 0usize;

            for (slot_idx, slot) in reg_owner.iter().enumerate() {
                if let Some(owner) = *slot {
                    let lu = last_use[owner.0 as usize];
                    if const_bits[owner.0 as usize].is_some() && lu > best_const_lu {
                        best_const_lu = lu;
                        best_const_slot = Some(slot_idx);
                    }
                    if lu > best_any_lu {
                        best_any_lu = lu;
                        best_any_slot = slot_idx;
                    }
                }
            }

            // Prefer evicting a constant (free rematerialization) over a
            // non-constant (expensive spill+reload).
            let evict_slot = best_const_slot.unwrap_or(best_any_slot);
            let evicted = reg_owner[evict_slot].expect("all slots occupied but none found");
            let evicted_lu = last_use[evicted.0 as usize];
            let my_lu = last_use[vid.0 as usize];

            // Check if the new value itself should be evicted instead.
            // But: if the new value is a constant, prefer evicting it (cheaper).
            let evict_new = if const_bits[vid.0 as usize].is_some() {
                // New value is a constant — always evict it (free to rematerialize).
                true
            } else if const_bits[evicted.0 as usize].is_some() {
                // Occupant is a constant — evict it (free) to make room.
                false
            } else {
                // Neither is a constant — use Belady: evict whoever is used later.
                my_lu >= evicted_lu
            };

            if evict_new {
                // Evict the new value.
                if let Some(bits) = const_bits[vid.0 as usize] {
                    rematerialize.insert(*vid, bits);
                } else {
                    spilled.push(*vid);
                }
            } else {
                // Evict the occupant, take its register.
                assignment.remove(&evicted);
                if let Some(bits) = const_bits[evicted.0 as usize] {
                    rematerialize.insert(evicted, bits);
                } else {
                    spilled.push(evicted);
                }

                let reg = Reg(evict_slot as u8 + scratch_base);
                assignment.insert(*vid, reg);
                reg_owner[evict_slot] = Some(*vid);
            }
        }
    }

    RegAllocation {
        assignment,
        spilled,
        rematerialize,
        num_regs: max_reg_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    #[test]
    fn verify_empty_graph() {
        let graph = InterferenceGraph::new();
        let alloc = color_graph(&graph, 8, 4);
        assert!(alloc.assignment.is_empty());
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn verify_no_interference() {
        let mut graph = InterferenceGraph::new();
        graph.add_value(ValueId(0));
        graph.add_value(ValueId(1));
        graph.add_value(ValueId(2));
        // No edges - no interference

        let alloc = color_graph(&graph, 8, 4);
        assert_eq!(alloc.assignment.len(), 3);
        assert!(alloc.spilled.is_empty());
        // All can use the same register since they don't interfere
    }

    #[test]
    fn verify_chain_interference() {
        let mut graph = InterferenceGraph::new();
        // Linear chain: 0 -- 1 -- 2
        graph.add_value(ValueId(0));
        graph.add_value(ValueId(1));
        graph.add_value(ValueId(2));
        graph.add_edge(ValueId(0), ValueId(1));
        graph.add_edge(ValueId(1), ValueId(2));

        let alloc = color_graph(&graph, 8, 4);
        assert_eq!(alloc.assignment.len(), 3);
        assert!(alloc.spilled.is_empty());

        // 0 and 2 can share a color, 1 needs different
        let c0 = alloc.assignment[&ValueId(0)];
        let c1 = alloc.assignment[&ValueId(1)];
        let c2 = alloc.assignment[&ValueId(2)];

        assert_ne!(c0, c1);
        assert_ne!(c1, c2);
        // c0 and c2 could be same or different
    }

    #[test]
    fn verify_clique_needs_more_colors() {
        let mut graph = InterferenceGraph::new();
        // Triangle (3-clique): all interfere with all
        for i in 0..3 {
            graph.add_value(ValueId(i));
        }
        graph.add_edge(ValueId(0), ValueId(1));
        graph.add_edge(ValueId(1), ValueId(2));
        graph.add_edge(ValueId(0), ValueId(2));

        let alloc = color_graph(&graph, 8, 4);
        assert_eq!(alloc.assignment.len(), 3);
        assert!(alloc.spilled.is_empty());
        assert!(alloc.num_regs >= 3);

        // All must have different colors
        let colors: BTreeSet<_> = alloc.assignment.values().collect();
        assert_eq!(colors.len(), 3);
    }

    #[test]
    fn verify_precolored() {
        let mut graph = InterferenceGraph::new();
        graph.precolor(ValueId(0), Reg(0)); // Input register
        graph.add_value(ValueId(1));
        graph.add_edge(ValueId(0), ValueId(1));

        let alloc = color_graph(&graph, 8, 4);
        assert_eq!(alloc.assignment[&ValueId(0)], Reg(0));
        assert_ne!(alloc.assignment[&ValueId(1)], Reg(0));
    }

    #[test]
    fn verify_spilling() {
        let mut graph = InterferenceGraph::new();
        // 4-clique with only 2 registers available
        for i in 0..4 {
            graph.add_value(ValueId(i));
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                graph.add_edge(ValueId(i), ValueId(j));
            }
        }

        let alloc = color_graph(&graph, 2, 4); // Only 2 registers!
        assert_eq!(alloc.spilled.len(), 2); // 2 must spill
        assert_eq!(alloc.assignment.len(), 2); // 2 got colors
    }
}
