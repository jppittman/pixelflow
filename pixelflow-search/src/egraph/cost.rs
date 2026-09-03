//! Cost model for e-graph extraction.
//!
//! The cost model controls which equivalent expression the e-graph extracts.
//! It uses `OpKind` from `pixelflow-ir` as the canonical operation enumeration.
//!
//! # Architecture
//!
//! The module provides two levels of abstraction:
//!
//! 1. **`CostFunction` trait**: Pluggable interface for any cost estimator
//! 2. **`CostModel` struct**: Hardcoded O(1) lookup table based on OpKind
//!
//! This allows the e-graph extraction to use either:
//! - Fast hardcoded costs (`CostModel`)
//! - Custom domain-specific cost models

use super::node::ENode;
use pixelflow_ir::OpKind;
use pixelflow_ir::kind::OpMap;

// ============================================================================
// Latency Prior — single source of truth
// ============================================================================

/// Handcrafted per-op cycle-latency estimates, indexed by `OpKind::index()`.
///
/// This is the ONE place these numbers are allowed to live. Both the static
/// [`CostModel`] (via [`CostModel::latency_prior`]) and the Guide's op
/// embedding initialization
/// (`nnue::factored::OpEmbeddings::init_with_latency_prior`) derive their
/// costs from this table so the two representations cannot drift apart. If
/// you're tempted to hand-tune a number in one place, change it here instead.
///
/// Handcrafted cycle estimates, one per op.
///
/// Written as an exhaustive `match` rather than a positional per-op array.
/// The array form aligned to the discriminants only
/// by convention, nothing checked the convention, and it drifted: while the
/// discriminants had gaps the table was written densely, so 13 of 50 ops read
/// their neighbour's cycle count. `Dwrt` came back 10 instead of 1000 — cheap
/// enough for extraction to pick an unlowered derivative, which is exactly what
/// the 1000 exists to prevent — and `Shr`, which `expand_log2` emits, came back
/// 1000 instead of 1.
///
/// A `match` cannot drift: adding an op is a compile error until it is priced.
///
/// # Measured basis (2026-08-10, Apple M2 Max, aarch64 NEON JIT)
///
/// The non-trivial entries below are **measured, not guessed**: serial chains
/// of K=8 vs K=32 applications of each op, JIT-compiled through
/// `compile` (so transcendentals are timed in their
/// `expand_transcendentals` lowered form — the form this table is actually
/// pricing) and timed under `BenchMode::Latency`; the per-stage slope cancels
/// call overhead exactly. Units are normalized so `Add = 4` (the table's
/// historical unit; measured FADD slope 0.87ns/stage ≈ 3 real cycles at
/// ~3.4GHz). Two independent runs agreed within 3% on every corrected entry.
/// Protocol: `pixelflow-pipeline/examples/measure_latency_prior.rs`.
///
/// The headline correction: `Pow` was priced 12 — cheaper than a hardware
/// `Sqrt` at 15 — but `expand_transcendentals` lowers `Pow(a,b)` to
/// `exp2(b·log2 a)`, two bit-manipulation polynomial kernels. Measured: 196
/// (and the measurement is internally consistent: Log2 121 + Mul 5 + Exp2 69
/// ≈ 196, likewise Ln ≈ Log2+Mul, Asin ≈ Atan2+Sqrt+Mul+Sub). With the old
/// table, extraction preferred `Pow(x, 0.5)` over `Sqrt(x)` — a measured
/// 2.8x kernel slowdown shipped by the DEFAULT cost model.
///
/// Also corrected by the same measurement: `Recip` (10 → 16) and `Rsqrt`
/// (5 → 21) are *slower* serially than the hardware `Div` (11) and `Sqrt`
/// (15) they approximate on this backend — the NEON lowering is
/// estimate + Newton steps, a serial chain, not a cheap single instruction.
///
/// Caveat: measured on one machine (aarch64 NEON). The lowered subgraphs are
/// the same shape on x86, so the ordering is expected to transfer, but the
/// exact ratios are host-specific — see `cargo xtask isa-matrix`.
///
/// # Re-measurement (2026-09-02, same machine, 7 load-gated runs)
///
/// Re-run against current `main` because the code generator was substantially
/// rewritten after Round 1 (#1071 collapse-JIT unification, #1082 one compile
/// entry, #1092/#1119 the loop nest, #1075–#1077 memory operands and the ABI).
/// Extraction is argmin over this table, so stale coefficients would put every
/// cost number this repo reports in stale units.
///
/// **Most of the table survived.** 24 of the 29 measured ops came back within
/// their own cross-run spread of the Round 1 value, so those numbers describe
/// the current machine and are left alone. Five moved:
///
/// | op | was | now | why |
/// |---|---|---|---|
/// | `Sin` | 70 | 95 | Cody-Waite range reduction (#992) |
/// | `Cos` | 75 | 103 | same reduction |
/// | `Tan` | 87 | 117 | same reduction |
/// | `Sqrt` | 15 | 13 | codegen |
/// | `Select` | 4 | 3 | codegen |
///
/// The trig row is not drift, it is a *different function*: #992 landed 62
/// seconds before Round 1's own measurement commit, so Round 1 almost
/// certainly priced the pre-fix reduction. The fingerprint says so — the only
/// ops that moved are the three sharing `expand_sin`'s reduction, while
/// `Atan`, `Atan2`, `Asin` and `Acos`, which do not use it, came back at
/// +0.6%, +0.7%, −3.7% and −3.5%. Correct reduction costs ~25 cycles more
/// than the reduction it replaced, and this table now says so.
///
/// The Round 1 consistency identity still holds under the new numbers:
/// Log2 116.8 + Mul 5.5 + Exp2 66.1 = 188.4 against a directly measured
/// `Pow` of 189.3.
///
/// **What this protocol does NOT measure**, and therefore what no re-run can
/// refresh: 21 of the 50 entries are unmeasured, including `Gather` and
/// `RawGather` at 10 apiece. `measure_latency_prior` calls every kernel with
/// a NULL context pointer and a single-batch tile, and gathers read their
/// buffer bases out of that context register — so a probe containing one
/// would segfault. The memory-operand and loop-nest work (#1075, #1092,
/// #1119) therefore cannot show up in this table's memory entries: those
/// entries are, and remain, guesses. The comparisons (`Lt`..`Ne`) and the
/// integer/bit ops are unmeasured for the same protocol reason.
///
/// Two further caveats on the instrument, both recorded in
/// `docs/results/2026-09-02-latency-prior-remeasure.md`: `BenchMode::Latency`
/// divides by lane count, so its ABSOLUTE ns are 4× optimistic (the example
/// prints an impossible "13.66GHz" clock estimate off them) — harmless here
/// only because this table is normalized to `Add = 4` and a uniform scale
/// cancels; and `Select`'s number rests on the assumption that its compare
/// overlaps the Mul the protocol subtracts, which is the weakest measurement
/// in the set.
#[must_use]
pub fn latency_prior_cycles() -> OpMap<usize> {
    OpMap::from_fn(|op| match op {
        OpKind::Var => 0,     // free
        OpKind::Const => 0,   // free
        OpKind::Add => 4,     // measured 4.0 (anchor)
        OpKind::Sub => 4,     // measured 4.0
        OpKind::Mul => 5,     // measured 5.3
        OpKind::Div => 11,    // measured 11.3 (was 15)
        OpKind::Neg => 3,     // measured 3.1 (was 1)
        OpKind::Sqrt => 13,   // remeasured 13.4 (was 15)
        OpKind::Rsqrt => 21,  // measured 21.5 — estimate + NR chain (was 5)
        OpKind::Abs => 3,     // measured 3.1 (was 1)
        OpKind::Min => 3,     // measured 2.7 (was 4)
        OpKind::Max => 3,     // measured 2.7 (was 4)
        OpKind::MulAdd => 5,  // fused; measured 5.5, kept at Mul parity
        OpKind::Recip => 16,  // measured 16.0 — estimate + NR chain (was 10)
        OpKind::Floor => 4,   // measured ~4
        OpKind::Ceil => 4,    // measured 4.3
        OpKind::Round => 4,   // measured 4.3
        OpKind::Sin => 95,    // remeasured 95.0 — Cody-Waite reduction (was 70)
        OpKind::Cos => 103,   // remeasured 103.0 — Cody-Waite reduction (was 75)
        OpKind::Tan => 117,   // remeasured 116.8 — Cody-Waite reduction (was 87)
        OpKind::Asin => 103,  // measured 102.8 (was 10)
        OpKind::Acos => 103,  // measured 102.9 (was 10)
        OpKind::Atan => 79,   // measured 79.3 (was 10)
        OpKind::Exp => 75,    // measured 74.7 (was 10)
        OpKind::Exp2 => 69,   // measured 69.1 (was 10)
        OpKind::Ln => 128,    // measured 127.6 (was 10)
        OpKind::Log2 => 122,  // measured 122.0 (was 10)
        OpKind::Log10 => 134, // measured 133.6 (was 10)
        OpKind::Atan2 => 79,  // measured 78.8 (was 10)
        OpKind::Pow => 196,   // measured 196.2 ≈ Log2 + Mul + Exp2 (was 12)
        OpKind::Lt => 3,
        OpKind::Le => 3,
        OpKind::Gt => 3,
        OpKind::Ge => 3,
        OpKind::Eq => 3,
        OpKind::Ne => 3,
        OpKind::Select => 3,     // remeasured 3.3 (was 4)
        OpKind::Tuple => 0,      // free (structural)
        OpKind::TruncToInt => 1, // cvttps2dq
        OpKind::IntToFloat => 1, // cvtdq2ps
        OpKind::IAdd => 1,       // paddd
        OpKind::Shl => 1,
        OpKind::Shr => 1,
        OpKind::BitAnd => 1,
        OpKind::BitOr => 1,
        OpKind::Dwrt => 1000,
        OpKind::Buffer => 0,     // leaf, free
        OpKind::Gather => 10,    // memory read
        OpKind::RawGather => 10, // primitive memory read
        OpKind::Reduce => 0,     // lowered (unrolled) before costing
    })
}

// ============================================================================
// Cost Function Trait
// ============================================================================

/// Trait for pluggable cost functions in e-graph extraction.
///
/// Implementors provide a cost estimate for ENodes, enabling different
/// cost models (hardcoded, learned neural, domain-specific) to be used
/// interchangeably during extraction.
///
/// # Contract
///
/// - `node_cost` returns a cost in arbitrary units (lower is better)
/// - Leaves (Var, Const) should typically return 0
/// - Costs should be consistent: same input → same output
///
/// # Example
///
/// ```ignore
/// // Using the hardcoded cost model
/// let costs = CostModel::default();
/// let (tree, cost) = extract(&egraph, root, &costs);
/// ```
pub trait CostFunction {
    /// Estimate the cost of a single ENode given its parent context.
    ///
    /// This is the atomic operation cost, NOT including children.
    /// The extraction algorithm sums child costs separately.
    ///
    /// `parent` is the OpKind of the operation using this result.
    /// This allows for 'sliding window' optimizations (e.g. FMA detection).
    fn node_cost(&self, node: &ENode, parent: Option<OpKind>) -> usize;

    /// Get the cost of an operation by OpKind (optional, for interop).
    fn cost_by_kind(&self, op: OpKind, parent: Option<OpKind>) -> usize {
        panic!("CostFunction::cost_by_kind not implemented");
    }
}

/// Cost model indexed by OpKind.
///
/// Uses [`OpMap`] internally for O(1) lookup.
/// Includes depth penalty for compile-time optimization.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Cost per operation.
    costs: OpMap<usize>,
    /// Depth at which to start applying penalties.
    pub depth_threshold: usize,
    /// Penalty per depth level beyond threshold.
    pub depth_penalty: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel {
    /// Create a cost model seeded with the handcrafted latency-prior cycle
    /// table ([`latency_prior_cycles`]).
    ///
    /// This is the default: an all-zero cost model makes every expression
    /// "free" and extraction degenerates to an arbitrary tie-break, so
    /// zero-cost is never a useful baseline for real extraction. Use
    /// [`CostModel::zero`] explicitly if you actually want all-zero costs
    /// (e.g. to test structural properties independent of cost).
    pub fn new() -> Self {
        Self::latency_prior()
    }

    /// Create a cost model from the handcrafted latency-prior cycle table.
    ///
    /// Source of truth: [`latency_prior_cycles`], shared with
    /// `nnue::factored::OpEmbeddings::init_with_latency_prior` so the static
    /// table and the Guide's op-embedding prior cannot drift apart.
    pub fn latency_prior() -> Self {
        Self {
            costs: latency_prior_cycles(),
            depth_threshold: 1024, // Effectively disabled
            depth_penalty: 0,
        }
    }

    /// Create an all-zero cost model.
    ///
    /// Every expression costs nothing, so extraction can't distinguish
    /// equivalent forms on cost alone. Only useful for tests that check
    /// structural extraction behavior (DAG sharing, cycle handling, etc.)
    /// independent of any particular cost table.
    pub fn zero() -> Self {
        Self {
            costs: OpMap::splat(0),
            depth_threshold: 1024, // Effectively disabled
            depth_penalty: 0,
        }
    }

    /// Create with aggressive depth penalty for complex kernels.
    pub fn shallow() -> Self {
        Self {
            depth_threshold: 16,
            depth_penalty: 500,
            ..Self::new()
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get cost for an OpKind.
    #[inline]
    pub fn cost(&self, op: OpKind) -> usize {
        self.costs[op]
    }

    /// Set cost for an OpKind.
    #[inline]
    pub fn set_cost(&mut self, op: OpKind, cost: usize) {
        self.costs[op] = cost;
    }

    /// Get the raw costs array.
    pub fn costs(&self) -> &OpMap<usize> {
        &self.costs
    }

    /// Get mutable reference to costs array.
    pub fn costs_mut(&mut self) -> &mut OpMap<usize> {
        &mut self.costs
    }

    /// Calculate the hinge penalty for a given depth.
    #[inline]
    pub fn depth_cost(&self, depth: usize) -> usize {
        // `>` is an equivalent mutant to `>=` under cargo-mutants: at
        // `depth == self.depth_threshold` the multiplier `(depth -
        // self.depth_threshold)` is 0 either way, so both branches return the
        // same 0. No test can distinguish them — left as `>` to match the
        // "strictly past the threshold" reading of a hinge penalty.
        if depth > self.depth_threshold {
            (depth - self.depth_threshold) * self.depth_penalty
        } else {
            0
        }
    }

    /// Get cost for an ENode.
    ///
    /// Uses `op.kind()` to convert at the boundary from `&dyn Op` to `OpKind`.
    pub fn node_op_cost(&self, node: &ENode) -> usize {
        match node {
            // Buffer is a leaf like Var/Const: the cost of the read lives on
            // the Gather that consumes it.
            ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => 0,
            // `Dwrt` is the internal autodiff marker. It is rewritten away by
            // the chain rule; a surviving one is the (not-yet-wired) jet
            // fallback. Either way extraction must never choose it, so it is
            // prohibitively expensive regardless of the learned weight table.
            ENode::Op { op, .. } if op.kind() == OpKind::Dwrt => usize::MAX / 4,
            ENode::Op { op, .. } => self.cost(op.kind()),
        }
    }

    /// Get cost by operation name (for backward compatibility).
    ///
    /// # Panics
    ///
    /// Panics if `name` does not match a known `OpKind`. Silently mapping
    /// an unrecognized name to `Add`'s cost would let typos and stale
    /// callers pass through with a wrong-but-plausible number — fail loud
    /// instead.
    pub fn cost_by_name(&self, name: &str) -> usize {
        let op = OpKind::from_name(name)
            .unwrap_or_else(|| panic!("CostModel::cost_by_name: unknown op name {name:?}"));
        self.cost(op)
    }
}

// ============================================================================
// CostFunction Implementation for CostModel
// ============================================================================

impl CostFunction for CostModel {
    fn node_cost(&self, node: &ENode, _parent: Option<OpKind>) -> usize {
        self.node_op_cost(node)
    }

    fn cost_by_kind(&self, op: OpKind, _parent: Option<OpKind>) -> usize {
        self.cost(op)
    }
}

#[cfg(test)]
mod every_op_is_priceable {
    use super::{CostModel, latency_prior_cycles};
    use pixelflow_ir::kind::{OpKind, OpMap};

    /// `cost`/`set_cost` subscript a positional per-op array with
    /// `OpKind::index()`. That is only total while the discriminants are dense.
    ///
    /// They were not, until 2026-08-02: `Gather`/`RawGather`/`Reduce` sat past
    /// three gaps and indexed 50..=52 into a 50-slot array. Nothing caught it
    /// because `arena_to_egraph` refuses `Buffer` leaves before extraction
    /// runs, and every `Gather` has one — so the panic was reachable only by
    /// pricing an op the front end happened never to hand over. Pricing the
    /// whole enum here does not depend on that accident holding.
    /// The prices that are load-bearing rather than advisory.
    ///
    /// `Dwrt` is priced prohibitively so extraction never selects an unlowered
    /// derivative — `arena_to_schedule` panics on one that reaches codegen, and
    /// that panic is supposed to be unreachable. When the table was positional
    /// and `index()` was sparse, `Dwrt` came back 10, which is cheaper than
    /// `Div`. `Shr` got Dwrt's 1000 in the same shift, which taught the
    /// optimizer to avoid the shifts `expand_log2` is built from.
    #[test]
    fn the_prohibitive_prices_are_actually_prohibitive() {
        let cycles = latency_prior_cycles();

        assert_eq!(cycles[OpKind::Dwrt], 1000, "Dwrt must stay unselectable");
        assert!(
            cycles[OpKind::Dwrt] > 100 * cycles[OpKind::Mul],
            "Dwrt at {} is not prohibitive next to Mul at {}",
            cycles[OpKind::Dwrt],
            cycles[OpKind::Mul]
        );

        // The bit-manipulation atoms exp/log lower into are single
        // instructions and must be priced like it.
        for op in [
            OpKind::Shl,
            OpKind::Shr,
            OpKind::BitAnd,
            OpKind::BitOr,
            OpKind::IAdd,
            OpKind::TruncToInt,
            OpKind::IntToFloat,
        ] {
            assert_eq!(cycles[op], 1, "{op:?} is a single instruction");
        }
    }

    /// Pricing is total: every op has a cost, and asking for one cannot fail.
    ///
    /// `Gather`/`RawGather`/`Reduce` are named outright rather than left to
    /// the walk over `all()`. They sit at the end of the table, which is where
    /// an op goes missing from a table-filling loop without anyone noticing,
    /// so naming them tests the walk as much as the pricing.
    #[test]
    fn every_op_can_be_priced() {
        let mut model = CostModel::latency_prior();

        for op in [OpKind::Gather, OpKind::RawGather, OpKind::Reduce] {
            let _ = model.cost(op);
            model.set_cost(op, 1);
        }

        for op in OpKind::all() {
            let _ = model.cost(op);
            model.set_cost(op, 1);
        }
    }

    /// `CostModel::zero()` exists specifically so extraction tests can check
    /// structural behavior (DAG sharing, cycle handling) independent of any
    /// cost table — see its doc comment. That property only holds if every
    /// op is actually priced at 0; if `zero()` ever returned the
    /// `latency_prior` table instead (an easy mix-up, since both build a
    /// `CostModel { costs, .. }` with the same shape), every extraction test
    /// built on "cost is irrelevant here" would start silently depending on
    /// real cycle counts instead.
    #[test]
    fn zero_prices_every_op_at_zero_cost() {
        let model = CostModel::zero();
        for op in OpKind::all() {
            assert_eq!(
                model.cost(op),
                0,
                "{op:?} is not zero-priced under CostModel::zero()"
            );
        }
    }

    /// `cost_by_name` used to hand-roll its own `&str -> OpKind` table,
    /// separate from `OpKind::from_name`, and that table's whitelist simply
    /// stopped at `tuple` — every op past it (memory/lattice/bit-manip ops
    /// included) was unreachable by name even though `cost`/`set_cost`
    /// already handled it fine by value. Delegating to `OpKind::from_name`
    /// means `cost_by_name` covers exactly what `cost` covers, nothing more
    /// or less — check that against a sample spanning the enum, not just the
    /// ops the old table happened to list.
    #[test]
    fn cost_by_name_matches_cost_for_every_sampled_op() {
        let model = CostModel::latency_prior();
        for name in [
            "sin",
            "log10",
            "pow",
            "lt",
            "select",
            "tuple",
            "dwrt",
            "buffer",
            "gather",
            "raw_gather",
            "reduce",
            "iadd",
            "shl",
            "trunc_to_int",
        ] {
            let op = OpKind::from_name(name).unwrap_or_else(|| panic!("unknown op name {name:?}"));
            assert_eq!(
                model.cost_by_name(name),
                model.cost(op),
                "cost_by_name({name:?}) should match cost(OpKind::{op:?})"
            );
        }
    }
}

#[cfg(test)]
mod cost_model_accessors {
    use super::{CostFunction, CostModel};
    use crate::egraph::ops::op_from_kind;
    use crate::egraph::{EGraph, ENode};
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};

    /// `set_cost` followed by `cost` for the same op should observe the
    /// value just written, and must not disturb any other op's price —
    /// otherwise `set_cost`/`cost` could be indexing different slots
    /// without any test noticing.
    #[test]
    fn set_cost_then_cost_returns_the_value_just_set_without_disturbing_other_ops() {
        let mut model = CostModel::latency_prior();
        let mul_before = model.cost(OpKind::Mul);

        model.set_cost(OpKind::Add, 777);

        assert_eq!(model.cost(OpKind::Add), 777);
        assert_eq!(
            model.cost(OpKind::Mul),
            mul_before,
            "set_cost(Add, _) must not leak into Mul's price"
        );
    }

    /// `costs()` is a read-only view onto the same table `cost()` reads
    /// from, seeded by the latency prior — not a stub or a freshly
    /// allocated all-zero map.
    #[test]
    fn costs_accessor_reflects_the_latency_prior_table() {
        let model = CostModel::latency_prior();
        assert_eq!(model.costs()[OpKind::Sin], model.cost(OpKind::Sin));
        assert_eq!(model.costs()[OpKind::Add], model.cost(OpKind::Add));
    }

    /// `costs_mut()` must hand back a view into the model's own table:
    /// writing through it should be visible via `cost()` afterward.
    #[test]
    fn costs_mut_edits_are_visible_through_cost() {
        let mut model = CostModel::latency_prior();
        model.costs_mut()[OpKind::Div] = 321;
        assert_eq!(model.cost(OpKind::Div), 321);
    }

    /// `shallow()` keeps the latency-prior op costs (it only overrides the
    /// depth penalty fields), and its depth parameters must differ from
    /// `new()`'s effectively-disabled defaults or the two constructors
    /// would be indistinguishable.
    #[test]
    fn shallow_keeps_latency_prior_op_costs_but_tightens_the_depth_penalty() {
        let shallow = CostModel::shallow();
        let latency_prior = CostModel::latency_prior();

        assert_eq!(shallow.cost(OpKind::Sin), latency_prior.cost(OpKind::Sin));
        assert_eq!(shallow.depth_threshold, 16);
        assert_eq!(shallow.depth_penalty, 500);
        assert_ne!(shallow.depth_threshold, CostModel::new().depth_threshold);
    }

    /// At or below the threshold, depth carries no penalty at all.
    #[test]
    fn depth_cost_is_zero_at_and_below_the_threshold() {
        let model = CostModel::shallow(); // depth_threshold=16, depth_penalty=500
        assert_eq!(model.depth_cost(16), 0);
        assert_eq!(model.depth_cost(10), 0);
    }

    /// Past the threshold, the penalty scales linearly with how far past —
    /// picked so `+`/`-`/`*`/`/` substitutions on either operator each
    /// produce a different, wrong number instead of coincidentally
    /// agreeing with `(depth - threshold) * penalty`.
    #[test]
    fn depth_cost_charges_penalty_per_level_past_the_threshold() {
        let model = CostModel::shallow(); // depth_threshold=16, depth_penalty=500
        assert_eq!(model.depth_cost(18), 1000); // (18 - 16) * 500
    }

    /// `Var`/`Const`/`Buffer` are leaves; the cost of reading a `Buffer` is
    /// charged to the `Gather` that consumes it, so all three price at
    /// zero regardless of what the op-cost table says.
    ///
    /// `Buffer` is asserted alongside the other two rather than left to the
    /// shared match arm: runtime arenas do contain buffer leaves, so if that
    /// arm ever started taking the table price, the buffer *and* its
    /// consuming gather would both be charged and extraction could change
    /// its choices.
    #[test]
    fn node_op_cost_should_price_var_const_and_buffer_leaves_at_zero() {
        let model = CostModel::latency_prior();
        assert_eq!(model.node_op_cost(&ENode::Var(0)), 0);
        assert_eq!(model.node_op_cost(&ENode::constant(2.0)), 0);

        let decl = BufferDecl {
            id: BufferIdentity::mint(),
            width: 8,
            height: 4,
        };
        assert_eq!(model.node_op_cost(&ENode::Buffer(decl)), 0);
    }

    /// `Dwrt` is the unlowered-autodiff marker and must never look cheap to
    /// extraction, however the op-cost table happens to price it — so
    /// `node_op_cost` overrides the table for it specifically.
    #[test]
    fn node_op_cost_makes_a_dwrt_node_prohibitively_expensive() {
        let model = CostModel::latency_prior();
        let op = op_from_kind(OpKind::Dwrt).expect("Dwrt has an Op impl");
        let node = ENode::Op {
            op,
            children: vec![],
        };
        assert_eq!(model.node_op_cost(&node), usize::MAX / 4);
    }

    /// An ordinary op node (not Dwrt, not a leaf) prices straight from the
    /// op-cost table, not from the Dwrt override path.
    #[test]
    fn node_op_cost_prices_an_ordinary_op_node_from_the_cost_table() {
        let mut egraph = EGraph::new();
        let lhs = egraph.add(ENode::constant(1.0));
        let rhs = egraph.add(ENode::constant(2.0));
        let model = CostModel::latency_prior();
        let op = op_from_kind(OpKind::Add).expect("Add has an Op impl");
        let node = ENode::Op {
            op,
            children: vec![lhs, rhs],
        };
        assert_eq!(model.node_op_cost(&node), model.cost(OpKind::Add));
    }

    /// The `CostFunction` trait impl for `CostModel` is a thin delegation
    /// layer; `node_cost` must route to `node_op_cost` rather than return a
    /// constant. The node is an `Add` (cost 4) rather than a leaf (cost 0)
    /// so that a delegation which quietly returned zero is distinguishable
    /// from one that works.
    #[test]
    fn node_cost_trait_method_delegates_to_node_op_cost() {
        let mut egraph = EGraph::new();
        let lhs = egraph.add(ENode::constant(1.0));
        let rhs = egraph.add(ENode::constant(2.0));
        let op = op_from_kind(OpKind::Add).expect("Add has an Op impl");
        let node = ENode::Op {
            op,
            children: vec![lhs, rhs],
        };

        let model = CostModel::latency_prior();
        assert_eq!(
            CostFunction::node_cost(&model, &node, None),
            model.node_op_cost(&node)
        );
        assert_ne!(CostFunction::node_cost(&model, &node, None), 0);
    }

    /// `CostFunction::cost_by_kind` has a default body (`panic!`) for
    /// implementors that don't provide their own — documented as "not
    /// implemented" rather than silently returning a made-up number.
    /// `CostModel` overrides it (tested above); this exercises the trait's
    /// own default via a minimal second implementor.
    #[test]
    #[should_panic(expected = "not implemented")]
    fn cost_by_kind_default_trait_method_panics_when_not_overridden() {
        struct NoOverride;
        impl CostFunction for NoOverride {
            fn node_cost(&self, _node: &ENode, _parent: Option<OpKind>) -> usize {
                0
            }
        }

        let _ = CostFunction::cost_by_kind(&NoOverride, OpKind::Add, None);
    }

    /// Likewise `cost_by_kind` must route to `cost`, not return a constant.
    #[test]
    fn cost_by_kind_trait_method_delegates_to_cost() {
        let model = CostModel::latency_prior();
        assert_eq!(
            CostFunction::cost_by_kind(&model, OpKind::Log2, None),
            model.cost(OpKind::Log2)
        );
    }
}
