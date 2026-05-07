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
use alloc::collections::BTreeSet;
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
#[derive(Debug, Default)]
pub struct InterferenceGraph {
    /// Adjacency list: value → set of interfering values
    edges: BTreeMap<ValueId, BTreeSet<ValueId>>,
    /// Pre-colored values (e.g., function arguments in specific registers)
    precolored: BTreeMap<ValueId, Reg>,
}

impl InterferenceGraph {
    /// Create an empty interference graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value to the graph (with no interferences yet).
    pub fn add_value(&mut self, v: ValueId) {
        self.edges.entry(v).or_default();
    }

    /// Add an interference edge between two values.
    pub fn add_edge(&mut self, a: ValueId, b: ValueId) {
        if a != b {
            self.edges.entry(a).or_default().insert(b);
            self.edges.entry(b).or_default().insert(a);
        }
    }

    /// Mark a value as pre-colored (must use specific register).
    pub fn precolor(&mut self, v: ValueId, reg: Reg) {
        self.add_value(v);
        self.precolored.insert(v, reg);
    }

    /// Get the degree of a value (number of interferences).
    pub fn degree(&self, v: ValueId) -> usize {
        self.edges.get(&v).map(|s| s.len()).unwrap_or(0)
    }

    /// Get all values in the graph.
    pub fn values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.edges.keys().copied()
    }

    /// Get neighbors of a value.
    pub fn neighbors(&self, v: ValueId) -> impl Iterator<Item = ValueId> + '_ {
        self.edges
            .get(&v)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }

    /// Check if a value is pre-colored.
    pub fn is_precolored(&self, v: ValueId) -> bool {
        self.precolored.contains_key(&v)
    }

    /// Get the pre-assigned register for a value, if any.
    pub fn precolor_of(&self, v: ValueId) -> Option<Reg> {
        self.precolored.get(&v).copied()
    }

    /// Number of values in the graph.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
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
pub fn color_graph(graph: &InterferenceGraph, num_regs: u8, scratch_base: u8) -> RegAllocation {
    let mut assignment: BTreeMap<ValueId, Reg> = BTreeMap::new();
    let mut spilled: Vec<ValueId> = Vec::new();

    // Copy pre-colored assignments
    for (&v, &reg) in &graph.precolored {
        assignment.insert(v, reg);
    }

    // Build simplicial elimination ordering (reverse order for greedy coloring)
    let ordering = simplicial_elimination_order(graph);

    // Greedy coloring in the computed order
    for v in ordering {
        if assignment.contains_key(&v) {
            continue; // Already pre-colored
        }

        // Find colors used by neighbors
        let mut used_colors: BTreeSet<u8> = BTreeSet::new();
        for neighbor in graph.neighbors(v) {
            if let Some(&Reg(c)) = assignment.get(&neighbor) {
                used_colors.insert(c);
            }
        }

        // Find first available color in scratch range
        let mut color = None;
        for c in scratch_base..(scratch_base + num_regs) {
            if !used_colors.contains(&c) {
                color = Some(c);
                break;
            }
        }

        match color {
            Some(c) => {
                assignment.insert(v, Reg(c));
            }
            None => {
                // No register available - need to spill
                spilled.push(v);
            }
        }
    }

    // Count registers used
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

/// Compute a simplicial elimination ordering.
///
/// Uses maximum cardinality search (MCS), which produces a perfect
/// elimination ordering for chordal graphs.
fn simplicial_elimination_order(graph: &InterferenceGraph) -> Vec<ValueId> {
    let n = graph.len();
    if n == 0 {
        return vec![];
    }

    let mut order = Vec::with_capacity(n);
    let mut weight: BTreeMap<ValueId, usize> = BTreeMap::new();
    let mut remaining: BTreeSet<ValueId> = BTreeSet::new();

    // Initialize
    for v in graph.values() {
        weight.insert(v, 0);
        remaining.insert(v);
    }

    // MCS: repeatedly pick the vertex with maximum weight
    for _ in 0..n {
        // Find max-weight unordered vertex
        let v = remaining
            .iter()
            .max_by_key(|&&v| weight.get(&v).unwrap_or(&0))
            .copied()
            .expect("remaining should not be empty");

        remaining.remove(&v);
        order.push(v);

        // Increment weight of remaining neighbors
        for neighbor in graph.neighbors(v) {
            if remaining.contains(&neighbor) {
                *weight.entry(neighbor).or_default() += 1;
            }
        }
    }

    // Return in reverse order for greedy coloring
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

    // Add all values
    for (v, _) in schedule {
        graph.add_value(*v);
    }

    // Live set tracking
    let mut live: BTreeSet<ValueId> = BTreeSet::new();

    // Walk schedule in reverse (backward liveness analysis)
    for (v, _) in schedule.iter().rev() {
        // v is defined here, so it's live after this point until its last use
        // All currently live values interfere with v
        for &other in &live {
            graph.add_edge(*v, other);
        }

        // v is now live (just defined)
        live.insert(*v);

        // Remove v's uses from live set (they end here if this is their last use)
        // Actually, we need to add uses TO the live set
        for used in uses_of(*v) {
            live.insert(used);
        }
    }

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
            super::ScheduledOp::Unary(_, a) => core::slice::from_ref(a),
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

    #[test]
    fn test_empty_graph() {
        let graph = InterferenceGraph::new();
        let alloc = color_graph(&graph, 8, 4);
        assert!(alloc.assignment.is_empty());
        assert!(alloc.spilled.is_empty());
    }

    #[test]
    fn test_no_interference() {
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
    fn test_chain_interference() {
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
    fn test_clique_needs_more_colors() {
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
    fn test_precolored() {
        let mut graph = InterferenceGraph::new();
        graph.precolor(ValueId(0), Reg(0)); // Input register
        graph.add_value(ValueId(1));
        graph.add_edge(ValueId(0), ValueId(1));

        let alloc = color_graph(&graph, 8, 4);
        assert_eq!(alloc.assignment[&ValueId(0)], Reg(0));
        assert_ne!(alloc.assignment[&ValueId(1)], Reg(0));
    }

    #[test]
    fn test_spilling() {
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
