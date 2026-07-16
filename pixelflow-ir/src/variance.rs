//! # Variance Analysis
//!
//! 4-bit bitset tracking which coordinates an expression depends on.
//! This is the shared type used by both the e-graph analysis
//! (`pixelflow-search`) and the compiler codegen (`pixelflow-compiler`).
//!
//! ## Coordinate Mapping
//!
//! - Bit 0 (X): pixel column — varies per pixel
//! - Bit 1 (Y): pixel row — varies per scanline
//! - Bit 2 (Z): time/frame — varies per frame
//! - Bit 3 (W): layer/channel — varies per instance
//!
//! ## Evaluation Scopes
//!
//! The variance bitset maps to evaluation scopes via the Lattice type:
//!
//! | Variance | Scope | Meaning |
//! |----------|-------|---------|
//! | `0b0000` | Const | Compile-time constant |
//! | `0b1000` | Frame | W-only, compute once per frame |
//! | `0b0100` | Frame | Z-only, compute once per frame |
//! | `0b0010` | Scanline | Y-only, compute once per scanline |
//! | `0b0001` | Pixel | X-dependent, compute per pixel |
//! | `0b0111` | Pixel | X+Y+Z, full spatial varying |
//!
//! The rule: the shallowest scope that binds all variables in the bitset.

/// 4-bit variance bitset: which coordinates an expression depends on.
///
/// X=bit0, Y=bit1, Z=bit2, W=bit3. Operations:
/// - `union`: bitwise OR (join — a binary op depends on both operands' vars)
/// - `meet`: minimum across e-class representatives (pick lowest-deps form)
///
/// This type is `no_std` compatible and zero-cost (single `u8`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Variance(u8);

/// Bit positions for each coordinate variable.
impl Variance {
    /// No dependencies — compile-time constant.
    pub const CONST: Self = Self(0);

    /// Depends on X (pixel column). Bit 0.
    pub const X: Self = Self(1 << 0);

    /// Depends on Y (pixel row). Bit 1.
    pub const Y: Self = Self(1 << 1);

    /// Depends on Z (time/frame). Bit 2.
    pub const Z: Self = Self(1 << 2);

    /// Depends on W (layer/channel). Bit 3.
    pub const W: Self = Self(1 << 3);

    /// All spatial coordinates (X, Y, Z). Common for per-pixel expressions.
    pub const SPATIAL: Self = Self(0b0111);

    /// All coordinates (X, Y, Z, W).
    pub const ALL: Self = Self(0b1111);

    /// Create from a variable index (0=X, 1=Y, 2=Z, 3=W).
    ///
    /// # Panics
    ///
    /// Panics if `var_idx >= 4`.
    #[inline]
    #[must_use]
    pub const fn from_var(var_idx: u8) -> Self {
        assert!(var_idx < 4, "variable index must be 0..4");
        Self(1 << var_idx)
    }

    /// Create from raw bits. Only the low 4 bits are used.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0b1111)
    }

    /// Get the raw bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    // --- Lattice operations ---

    /// Union (join): the result depends on everything either operand depends on.
    /// Used WITHIN a single expression node (a binary op joins its children).
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Meet: the minimum-variance representative.
    /// Used ACROSS e-nodes in the same e-class (pick the cheapest representation).
    ///
    /// Compares by popcount first (fewer deps = better), then by raw value for
    /// determinism.
    #[inline]
    #[must_use]
    pub const fn meet(self, other: Self) -> Self {
        let a_pop = self.0.count_ones();
        let b_pop = other.0.count_ones();
        if a_pop < b_pop {
            self
        } else if b_pop < a_pop {
            other
        } else if self.0 <= other.0 {
            self
        } else {
            other
        }
    }

    // --- Queries ---

    /// True if no variable dependencies (compile-time constant).
    #[inline]
    #[must_use]
    pub const fn is_const(self) -> bool {
        self.0 == 0
    }

    /// True if depends on X (must be in the inner pixel loop).
    #[inline]
    #[must_use]
    pub const fn depends_on_x(self) -> bool {
        self.0 & Self::X.0 != 0
    }

    /// True if does NOT depend on X (can be hoisted out of the pixel loop).
    #[inline]
    #[must_use]
    pub const fn is_x_invariant(self) -> bool {
        !self.depends_on_x()
    }

    /// True if depends on Y.
    #[inline]
    #[must_use]
    pub const fn depends_on_y(self) -> bool {
        self.0 & Self::Y.0 != 0
    }

    /// True if depends on Z.
    #[inline]
    #[must_use]
    pub const fn depends_on_z(self) -> bool {
        self.0 & Self::Z.0 != 0
    }

    /// True if depends on W.
    #[inline]
    #[must_use]
    pub const fn depends_on_w(self) -> bool {
        self.0 & Self::W.0 != 0
    }

    /// True if this is "uniform" in the older three-level sense: depends only on W
    /// (or is const). Compatible with the old `Deps::is_uniform()`.
    #[inline]
    #[must_use]
    pub const fn is_frame_uniform(self) -> bool {
        // No spatial bits (X, Y, Z) set
        self.0 & Self::SPATIAL.0 == 0
    }

    /// True if this is "varying" in the older three-level sense: depends on any
    /// spatial coordinate (X, Y, or Z).
    #[inline]
    #[must_use]
    pub const fn is_spatially_varying(self) -> bool {
        self.0 & Self::SPATIAL.0 != 0
    }

    /// Number of coordinates this expression depends on (0-4).
    #[inline]
    #[must_use]
    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }
}

impl core::fmt::Debug for Variance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_const() {
            return write!(f, "Variance(CONST)");
        }
        write!(f, "Variance{{")?;
        let mut first = true;
        for (bit, name) in [(0, 'X'), (1, 'Y'), (2, 'Z'), (3, 'W')] {
            if self.0 & (1 << bit) != 0 {
                if !first {
                    write!(f, ",")?;
                }
                write!(f, "{name}")?;
                first = false;
            }
        }
        write!(f, "}}")
    }
}

impl core::fmt::Display for Variance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

// ───────────────────── Arena-level variance analysis ─────────────────────

use alloc::vec::Vec;

/// Compute variance for every node in an `ExprArena`.
///
/// Returns a `Vec<Variance>` indexed by `ExprId`. Because the arena is
/// append-only in topological order, a single forward pass suffices —
/// when we visit node `i`, all its children `j < i` are already computed.
///
/// Cost: O(n) where n = `arena.len()`. No allocations beyond the result vec.
#[must_use]
pub fn compute_arena_variance(arena: &crate::arena::ExprArena) -> Vec<Variance> {
    use crate::arena::{ExprId, ExprNode};

    let n = arena.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let id = ExprId(i as u32);
        let v = match arena.node(id) {
            ExprNode::Var(idx) => {
                if *idx < 4 {
                    Variance::from_var(*idx)
                } else {
                    Variance::ALL
                }
            }
            ExprNode::Const(_) => Variance::CONST,
            ExprNode::Param(_) => {
                // Parameters are substituted before JIT compilation.
                // If we see one here, treat conservatively as all-varying.
                Variance::ALL
            }
            ExprNode::Unary(_, child) => result[child.0 as usize],
            ExprNode::Binary(_, a, b) => result[a.0 as usize].union(result[b.0 as usize]),
            ExprNode::Ternary(_, a, b, c) => result[a.0 as usize]
                .union(result[b.0 as usize])
                .union(result[c.0 as usize]),
            ExprNode::Nary(_, start, len) => {
                let children = arena.nary_children_slice(*start, *len);
                let mut v = Variance::CONST;
                for &child in children {
                    v = v.union(result[child.0 as usize]);
                }
                v
            }
        };
        result.push(v);
    }

    result
}

/// Find arena nodes that should be hoisted out of the X-loop.
///
/// Returns up to `max` `ExprId`s that are:
/// 1. X-invariant (don't depend on X)
/// 2. Non-trivial (not Var or Const — actual computation worth hoisting)
/// 3. Used by at least one X-dependent node (worth the register cost)
///
/// Results are sorted by estimated cost (transcendentals first).
#[must_use]
pub fn find_hoistable_arena_nodes(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    variance: &[Variance],
    max: usize,
) -> Vec<crate::arena::ExprId> {
    use crate::arena::{ExprId, ExprNode};
    use crate::kind::OpKind;

    let n = arena.len();

    // Mark which nodes are reachable from root
    let mut reachable = alloc::vec![false; n];
    let mut stack = alloc::vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if idx >= n || reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        for child in arena.children(id) {
            stack.push(child);
        }
    }

    // Mark which reachable nodes have at least one X-dependent parent
    let mut has_x_dependent_parent = alloc::vec![false; n];
    for i in 0..n {
        if !reachable[i] {
            continue;
        }
        let id = ExprId(i as u32);
        if variance[i].depends_on_x() {
            // This node depends on X — mark all its children as "used by X-dependent"
            for child in arena.children(id) {
                has_x_dependent_parent[child.0 as usize] = true;
            }
        }
    }

    // Collect hoistable candidates
    let mut candidates: Vec<(ExprId, u8)> = Vec::new(); // (id, priority)
    for i in 0..n {
        if !reachable[i] || !has_x_dependent_parent[i] {
            continue;
        }
        let v = variance[i];
        if !v.is_x_invariant() || v.is_const() {
            continue; // Must be X-invariant and non-const
        }
        let id = ExprId(i as u32);
        let node = arena.node(id);

        // Skip trivial nodes (Var, Const, Param) — not worth a register
        let priority = match node {
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => continue,
            #[allow(clippy::collapsible_match)]
            ExprNode::Unary(op, _) => match *op {
                OpKind::Sin
                | OpKind::Cos
                | OpKind::Exp
                | OpKind::Exp2
                | OpKind::Ln
                | OpKind::Log2
                | OpKind::Log10
                | OpKind::Sqrt
                | OpKind::Asin
                | OpKind::Acos
                | OpKind::Atan
                | OpKind::Atan2
                | OpKind::Pow
                | OpKind::Tan => 3, // Transcendentals: highest priority
                _ => 1,
            },
            ExprNode::Binary(op, _, _) => match *op {
                OpKind::Div => 2, // Division is expensive
                OpKind::Pow | OpKind::Atan2 | OpKind::Hypot => 3,
                _ => 1, // Add, Sub, Mul are cheap
            },
            _ => 1,
        };

        candidates.push((id, priority));
    }

    // Sort by priority (highest first), then by ExprId (topological order)
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    candidates.into_iter().take(max).map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_var() {
        assert_eq!(Variance::from_var(0), Variance::X);
        assert_eq!(Variance::from_var(1), Variance::Y);
        assert_eq!(Variance::from_var(2), Variance::Z);
        assert_eq!(Variance::from_var(3), Variance::W);
    }

    #[test]
    fn test_union() {
        assert_eq!(Variance::X.union(Variance::Y), Variance::from_bits(0b0011));
        assert_eq!(Variance::CONST.union(Variance::Z), Variance::Z);
        assert_eq!(Variance::X.union(Variance::X), Variance::X);
        assert_eq!(
            Variance::X.union(Variance::Y).union(Variance::Z),
            Variance::SPATIAL
        );
    }

    #[test]
    fn test_meet() {
        // Fewer deps wins
        assert_eq!(Variance::CONST.meet(Variance::X), Variance::CONST);
        assert_eq!(Variance::X.meet(Variance::CONST), Variance::CONST);

        // Same popcount: lower raw value wins (deterministic)
        assert_eq!(Variance::X.meet(Variance::Y), Variance::X); // 0b0001 < 0b0010
        assert_eq!(Variance::Y.meet(Variance::X), Variance::X);

        // 2-bit vs 1-bit: 1-bit wins
        let xy = Variance::X.union(Variance::Y);
        assert_eq!(xy.meet(Variance::Z), Variance::Z);
    }

    #[test]
    fn test_queries() {
        assert!(Variance::CONST.is_const());
        assert!(!Variance::X.is_const());

        assert!(Variance::X.depends_on_x());
        assert!(!Variance::Y.depends_on_x());

        assert!(Variance::Y.is_x_invariant());
        assert!(!Variance::X.is_x_invariant());

        assert!(Variance::W.is_frame_uniform());
        assert!(Variance::CONST.is_frame_uniform());
        assert!(!Variance::X.is_frame_uniform());
        assert!(!Variance::X.union(Variance::W).is_frame_uniform());

        assert!(Variance::X.is_spatially_varying());
        assert!(!Variance::W.is_spatially_varying());
    }

    #[test]
    fn test_debug_format() {
        assert_eq!(format!("{:?}", Variance::CONST), "Variance(CONST)");
        assert_eq!(format!("{:?}", Variance::X), "Variance{X}");
        assert_eq!(
            format!("{:?}", Variance::X.union(Variance::Z)),
            "Variance{X,Z}"
        );
        assert_eq!(format!("{:?}", Variance::ALL), "Variance{X,Y,Z,W}");
    }

    #[test]
    fn test_popcount() {
        assert_eq!(Variance::CONST.popcount(), 0);
        assert_eq!(Variance::X.popcount(), 1);
        assert_eq!(Variance::X.union(Variance::Y).popcount(), 2);
        assert_eq!(Variance::ALL.popcount(), 4);
    }

    #[test]
    fn test_compute_arena_variance() {
        use crate::arena::ExprArena;
        use crate::kind::OpKind;

        let mut arena = ExprArena::new();
        // Build: sin(Z * 0.3) * (X + Y)
        let z = arena.push_var(2); // Z → {Z}
        let c03 = arena.push_const(0.3); // 0.3 → {}
        let z_mul = arena.push_binary(OpKind::Mul, z, c03); // Z*0.3 → {Z}
        let sin_z = arena.push_unary(OpKind::Sin, z_mul); // sin(Z*0.3) → {Z}
        let x = arena.push_var(0); // X → {X}
        let y = arena.push_var(1); // Y → {Y}
        let x_add_y = arena.push_binary(OpKind::Add, x, y); // X+Y → {X,Y}
        let result = arena.push_binary(OpKind::Mul, sin_z, x_add_y); // → {X,Y,Z}

        let v = super::compute_arena_variance(&arena);

        assert_eq!(v[z.0 as usize], Variance::Z);
        assert_eq!(v[c03.0 as usize], Variance::CONST);
        assert_eq!(v[z_mul.0 as usize], Variance::Z);
        assert_eq!(v[sin_z.0 as usize], Variance::Z);
        assert!(v[sin_z.0 as usize].is_x_invariant());
        assert_eq!(v[x.0 as usize], Variance::X);
        assert_eq!(v[y.0 as usize], Variance::Y);
        assert_eq!(v[x_add_y.0 as usize], Variance::X.union(Variance::Y));
        assert_eq!(
            v[result.0 as usize],
            Variance::X.union(Variance::Y).union(Variance::Z)
        );
    }

    #[test]
    fn test_find_hoistable_arena_nodes() {
        use crate::arena::ExprArena;
        use crate::kind::OpKind;

        let mut arena = ExprArena::new();
        // sin(Z * 0.3) + X
        let z = arena.push_var(2);
        let c03 = arena.push_const(0.3);
        let z_mul = arena.push_binary(OpKind::Mul, z, c03);
        let sin_z = arena.push_unary(OpKind::Sin, z_mul);
        let x = arena.push_var(0);
        let result = arena.push_binary(OpKind::Add, sin_z, x);

        let v = super::compute_arena_variance(&arena);
        let hoistable = super::find_hoistable_arena_nodes(&arena, result, &v, 8);

        // sin(Z*0.3) should be hoistable (X-invariant, transcendental, used by X-dependent Add)
        assert!(
            hoistable.contains(&sin_z),
            "sin(Z*0.3) should be hoistable, got: {:?}",
            hoistable
        );
        // Z*0.3 might also be hoistable (X-invariant Mul used by sin which is used by X-dependent)
        // But X and Z and 0.3 should NOT be hoistable (trivial nodes)
        assert!(!hoistable.contains(&x), "X should not be hoistable");
        assert!(!hoistable.contains(&z), "Z should not be hoistable");
        assert!(!hoistable.contains(&c03), "0.3 should not be hoistable");
    }

    /// Verify that `arena_to_hoisted_schedule` correctly partitions
    /// `sin(Z * 0.3) * (X + Y)` into setup and loop phases.
    ///
    /// Expected partitioning with default predicate (X-invariant, non-const):
    /// - Setup: Z, 0.3, Z*0.3, sin(Z*0.3)  [Z and 0.3 are trivial but promoted
    ///   as transitive deps of the non-trivial setup nodes Z*0.3 and sin(Z*0.3)]
    /// - Loop: X, Y, X+Y, sin(Z*0.3)*(X+Y)
    ///
    /// The split index should separate setup from loop, and the hoisted count
    /// should reflect only non-trivial setup values that cross the boundary
    /// (sin(Z*0.3) is used by the loop's multiply, and Z*0.3 is used by sin
    /// which is itself in setup, so only sin(Z*0.3) crosses).
    #[test]
    fn test_hoisted_schedule_partitioning() {
        use crate::arena::ExprArena;
        use crate::backend::emit;
        use crate::kind::OpKind;

        let mut arena = ExprArena::new();
        // Build: sin(Z * 0.3) * (X + Y)
        let z = arena.push_var(2); // Z
        let c03 = arena.push_const(0.3); // 0.3
        let z_mul = arena.push_binary(OpKind::Mul, z, c03); // Z*0.3
        let sin_z = arena.push_unary(OpKind::Sin, z_mul); // sin(Z*0.3)
        let x = arena.push_var(0); // X
        let y = arena.push_var(1); // Y
        let x_add_y = arena.push_binary(OpKind::Add, x, y); // X+Y
        let root = arena.push_binary(OpKind::Mul, sin_z, x_add_y); // sin(Z*0.3)*(X+Y)

        let hoisted = emit::arena_to_hoisted_schedule(&arena, root, emit::default_hoist_predicate);

        // The split should be > 0 (we have setup nodes).
        assert!(
            hoisted.split > 0,
            "expected non-zero split, got 0 (no setup nodes detected)"
        );

        // Setup phase should contain at least Z, 0.3, Z*0.3, sin(Z*0.3) = 4 nodes.
        assert!(
            hoisted.split >= 4,
            "expected at least 4 setup nodes (Z, 0.3, Z*0.3, sin(Z*0.3)), got split={}",
            hoisted.split
        );

        // Loop phase should contain at least X, Y, X+Y, the final multiply = 4 nodes.
        let loop_count = hoisted.schedule.len() - hoisted.split;
        assert!(
            loop_count >= 4,
            "expected at least 4 loop nodes (X, Y, X+Y, multiply), got {}",
            loop_count
        );

        // The hoisted count (boundary crossers) should be >= 1 (at least sin(Z*0.3)
        // crosses from setup into the loop body via the final multiply).
        assert!(
            hoisted.num_hoisted >= 1,
            "expected at least 1 hoisted boundary value, got {}",
            hoisted.num_hoisted
        );

        // Verify topological order within each phase: for every operation,
        // its operands must appear earlier in the schedule.
        let mut defined = alloc::collections::BTreeSet::new();
        for (vid, op) in &hoisted.schedule {
            let operands: alloc::vec::Vec<emit::regalloc::ValueId> = match op {
                emit::ScheduledOp::Var(_) | emit::ScheduledOp::Const(_) => alloc::vec![],
                emit::ScheduledOp::Unary(_, a) => alloc::vec![*a],
                emit::ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
                emit::ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
            };
            for operand in &operands {
                assert!(
                    defined.contains(operand),
                    "topological order violated: {:?} uses {:?} which is not yet defined",
                    vid,
                    operand
                );
            }
            defined.insert(*vid);
        }

        // Verify that all setup nodes come before all loop nodes.
        // (This is guaranteed by construction but let's be explicit.)
        let setup_vids: alloc::collections::BTreeSet<_> = hoisted.schedule[..hoisted.split]
            .iter()
            .map(|(v, _)| *v)
            .collect();
        let loop_vids: alloc::collections::BTreeSet<_> = hoisted.schedule[hoisted.split..]
            .iter()
            .map(|(v, _)| *v)
            .collect();
        assert!(
            setup_vids.is_disjoint(&loop_vids),
            "setup and loop phases must not share ValueIds"
        );
    }

    /// Verify that a purely X-dependent expression produces an empty setup phase.
    #[test]
    fn test_hoisted_schedule_no_setup_for_x_only() {
        use crate::arena::ExprArena;
        use crate::backend::emit;
        use crate::kind::OpKind;

        let mut arena = ExprArena::new();
        // Build: X * X + X
        let x1 = arena.push_var(0);
        let x2 = arena.push_var(0); // same variable, different node
        let x_sq = arena.push_binary(OpKind::Mul, x1, x2);
        let x3 = arena.push_var(0);
        let root = arena.push_binary(OpKind::Add, x_sq, x3);

        let hoisted = emit::arena_to_hoisted_schedule(&arena, root, emit::default_hoist_predicate);

        // Nothing is X-invariant (except the Var(0) nodes themselves, which are trivial).
        assert_eq!(
            hoisted.split, 0,
            "expected split=0 for purely X-dependent expression, got {}",
            hoisted.split
        );
        assert_eq!(
            hoisted.num_hoisted, 0,
            "expected 0 hoisted values for purely X-dependent expression, got {}",
            hoisted.num_hoisted
        );
    }

    /// Verify that constants alone do not trigger hoisting (they are rematerialized).
    #[test]
    fn test_hoisted_schedule_constants_not_hoisted() {
        use crate::arena::ExprArena;
        use crate::backend::emit;
        use crate::kind::OpKind;

        let mut arena = ExprArena::new();
        // Build: X + 42.0
        let x = arena.push_var(0);
        let c = arena.push_const(42.0);
        let root = arena.push_binary(OpKind::Add, x, c);

        let hoisted = emit::arena_to_hoisted_schedule(&arena, root, emit::default_hoist_predicate);

        // 42.0 is X-invariant but is a Const (trivial) — not worth hoisting.
        assert_eq!(
            hoisted.split, 0,
            "expected split=0 when only constants are X-invariant, got {}",
            hoisted.split
        );
        assert_eq!(hoisted.num_hoisted, 0);
    }
}
