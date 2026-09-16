//! # Variance Analysis
//!
//! Which variables an expression depends on, as a bitset. This is the shared
//! type used by both the e-graph analysis (`pixelflow-search`) and the compiler
//! codegen (`pixelflow-compiler`).
//!
//! ## Variable Mapping
//!
//! - Bit 0 (X): pixel column — varies per pixel
//! - Bit 1 (Y): pixel row — varies per scanline
//! - Bits 2..4: retired. They were the Z and W axes; a lattice has
//!   [`COORD_AXES`](crate::arena::COORD_AXES) axes and a per-call scalar is a
//!   uniform, whose variance is `CONST`.
//! - Bits 4..8: the four reduction index slots — vary per step of the binder
//!   that binds them
//!
//! ## Scopes are binders
//!
//! A scope is something that binds a variable, and the bitset says which scopes
//! enclose an expression. Coordinates are bound by the lattice nest, reduction
//! indices by [`Kernel::over`](crate::Kernel::over) — the same kind of thing,
//! so they share the bitset.
//!
//! | Variance | Scope | Meaning |
//! |----------|-------|---------|
//! | `0b0000_0000` | Const | Compile-time constant — a uniform is here |
//! | `0b0000_0010` | Scanline | Y-only, compute once per scanline |
//! | `0b0000_0001` | Pixel | X-dependent, compute per pixel |
//! | `0b0001_0000` | Binder | varies with reduction slot 4 only |
//!
//! The rule: the shallowest scope that binds every variable in the bitset. That
//! one rule is loop-invariant code motion, hoisting out of a reduction, and
//! constant folding — see `docs/designs/lattice-scheduling-types.md`.

/// Which variables an expression depends on: one bit per variable.
///
/// Coordinates X=bit0, Y=bit1; reduction index slots `4..8` in bits `4..8`.
/// Bits 2 and 3 are the retired Z and W axes and are never set. Operations:
/// - `union`: bitwise OR (join — a binary op depends on both operands' vars)
/// - `meet`: minimum across e-class representatives (pick lowest-deps form)
/// - `without`: set difference — what a binder does to its own index
///
/// This type is `no_std` compatible and zero-cost (single `u8`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Variance(u8);

/// Bit positions for each variable.
impl Variance {
    /// No dependencies — compile-time constant.
    pub const CONST: Self = Self(0);

    /// Depends on X (pixel column). Bit 0.
    pub const X: Self = Self(1 << 0);

    /// Depends on Y (pixel row). Bit 1.
    pub const Y: Self = Self(1 << 1);

    /// The coordinates the lattice nest binds (X, Y).
    pub const COORDS: Self = Self(0b0000_0011);

    /// The four reduction index slots a binder can bind.
    pub const BINDERS: Self = Self(0b1111_0000);

    /// Every variable — the top of the lattice, and the answer whenever the
    /// analysis cannot prove something narrower.
    pub const ALL: Self = Self(0b1111_1111);

    /// Create from a variable index: `0..2` are the coordinates X/Y, `4..8`
    /// the reduction index slots.
    ///
    /// Indices 2 and 3 were the Z and W axes.
    /// [`ExprArena::push_var`](crate::arena::ExprArena::push_var) still
    /// builds them — `Var` is also a rewrite metavariable — and it is
    /// [`emit::compile`] that refuses one, so the bits stay constructible
    /// and belong to no scope.
    ///
    /// **Note what that costs, because it is a trap.** Bits 2 and 3 are
    /// outside *both* [`Self::COORDS`] and [`Self::BINDERS`], so a stray
    /// retired axis reads as [`Self::is_frame_uniform`] and LICM hoists it
    /// into the per-call prologue. When they were `Z`/`W` inside `COORDS`
    /// they were never hoisted. The failure mode got strictly worse when the
    /// axes retired, which is exactly why the emitter's refusal is
    /// unconditional rather than advisory.
    ///
    /// # Panics
    ///
    /// Panics if `var_idx >= 8`.
    #[inline]
    #[must_use]
    pub const fn from_var(var_idx: u8) -> Self {
        assert!(var_idx < 8, "variable index must be 0..8");
        Self(1 << var_idx)
    }

    /// Create from raw bits. All eight bits are meaningful.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
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

    /// Set difference: drop `other`'s variables from this set.
    ///
    /// This is what a binder does. Every other node either preserves its
    /// children's variance or unions it; `⊕_{i∈D}` is the only construct that
    /// *shrinks* the set, because it binds `i` and so `i` is no longer free in
    /// the result:
    ///
    /// ```text
    /// deps(⊕_{i∈D} body) = deps(body) \ {i}
    /// ```
    #[inline]
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
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

    /// True if this depends on variable `var_idx` — i.e. it must be evaluated
    /// inside the scope that binds it.
    ///
    /// # Panics
    ///
    /// Panics if `var_idx >= 8`.
    #[inline]
    #[must_use]
    pub const fn depends_on(self, var_idx: u8) -> bool {
        self.0 & Self::from_var(var_idx).0 != 0
    }

    /// True if this can be hoisted out of the scope binding `var_idx`.
    ///
    /// The one question the schedule asks, at every level: the X loop asks it of
    /// `0`, a scanline of `1`, a reduction of its own index slot. `is_x_invariant`
    /// is this with `0` written in.
    ///
    /// # Panics
    ///
    /// Panics if `var_idx >= 8`.
    #[inline]
    #[must_use]
    pub const fn is_invariant_in(self, var_idx: u8) -> bool {
        !self.depends_on(var_idx)
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

    /// True if this depends on any reduction index — it lives inside a binder
    /// body and cannot be hoisted past it.
    #[inline]
    #[must_use]
    pub const fn depends_on_binder(self) -> bool {
        self.0 & Self::BINDERS.0 != 0
    }

    /// True if depends on Y.
    #[inline]
    #[must_use]
    pub const fn depends_on_y(self) -> bool {
        self.0 & Self::Y.0 != 0
    }

    /// True if this can be computed once per call: it depends on no
    /// coordinate and no binder, which is where a uniform's arithmetic lives.
    ///
    /// A binder index disqualifies it as surely as a coordinate does — a
    /// value that changes per step of a reduction cannot be lifted to frame
    /// scope, which sits outside the binder.
    #[inline]
    #[must_use]
    pub const fn is_frame_uniform(self) -> bool {
        self.0 & (Self::COORDS.0 | Self::BINDERS.0) == 0
    }

    /// True if this varies across the lattice: it depends on a coordinate.
    #[inline]
    #[must_use]
    pub const fn is_spatially_varying(self) -> bool {
        self.0 & Self::COORDS.0 != 0
    }

    /// Number of variables this expression depends on (0-8).
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
        for bit in 0..8u8 {
            if self.0 & (1 << bit) == 0 {
                continue;
            }
            if !first {
                write!(f, ",")?;
            }
            first = false;
            match bit {
                0 => write!(f, "X")?,
                1 => write!(f, "Y")?,
                // A retired axis: no arena can name one, but the macro
                // tier's e-graph indexes its names in the same space.
                2 | 3 => write!(f, "?{bit}")?,
                // Reduction index slots print as the slot they bind, so a
                // variance set reads back as the binders that enclose the node.
                slot => write!(f, "i{slot}")?,
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

/// A reference denotes what its referent denotes, sampled at the same
/// coordinates, so it varies exactly as the referent does. Resolving is the
/// only way to know that — a key carries a kernel's identity, not its
/// structure — and where it cannot be resolved the answer is the conservative
/// one: refuse to claim an invariance that cannot be proved. An unresolvable
/// key is a corrupt graph, which `passes::expand_refs` reports at the layer
/// that can name it.
#[cfg(feature = "std")]
fn referent_variance(key: crate::key::KernelKey) -> Variance {
    crate::store::KernelStore::resolve(key).map_or(Variance::ALL, |referent| {
        let (ref_arena, ref_root) = referent.parts();
        compute_arena_variance(ref_arena)[ref_root.0 as usize]
    })
}

/// The same, with no store to resolve against — always the conservative
/// answer, and never actually asked: `Kernel::by_ref` is the `std` feature
/// too, so a `no_std` build cannot hold the key this would look up.
#[cfg(not(feature = "std"))]
fn referent_variance(_key: crate::key::KernelKey) -> Variance {
    Variance::ALL
}

/// Compute variance for every node in an `ExprArena`.
///
/// Returns a `Vec<Variance>` indexed by `ExprId`. Because the arena is
/// append-only in topological order, a single forward pass suffices —
/// when we visit node `i`, all its children `j < i` are already computed.
///
/// Cost: O(n) where n = `arena.len()`. No allocations beyond the result vec.
///
/// Public because it is the input the public `find_hoistable_*` functions
/// require, and because `pixelflow-search`'s NNUE featurizer uses it to
/// populate the variance histogram on arena-built accumulators (the same
/// classification `egraph::deps::DepsAnalysis` provides on e-graphs).
#[must_use]
pub fn compute_arena_variance(arena: &crate::arena::ExprArena) -> Vec<Variance> {
    use crate::arena::{ExprId, ExprNode};

    let n = arena.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let id = ExprId(i as u32);
        let v = match arena.node(id) {
            // Coordinates (0..4) and reduction index slots (4..8) each get their
            // own bit. Anything above that is not a variable this analysis knows.
            ExprNode::Var(idx) => {
                if *idx < 8 {
                    Variance::from_var(*idx)
                } else {
                    Variance::ALL
                }
            }
            ExprNode::Const(_) => Variance::CONST,
            // A buffer leaf is constant; a Gather's variance is the union of
            // its index expressions (handled by the Ternary arm below).
            ExprNode::Buffer(_) => Variance::CONST,
            // A uniform is invariant on the lattice — that one line is what
            // moves everything computed from it alone into the per-call
            // prologue — and unknown on the parameter space, which is why it
            // is not a `Const`.
            ExprNode::Uniform(_) => Variance::CONST,
            ExprNode::Ref(key) => referent_variance(*key),
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
            // A binder is the only node that shrinks the set: it binds its
            // index, so the index is not free in the result. There is no
            // "malformed binder" case to be conservative about any more —
            // `Fold` cannot hold an index that is not a binder, so this arm
            // no longer has to decline an analysis it used to be unable to
            // trust.
            ExprNode::Reduce { fold, body } => {
                result[body.0 as usize].without(Variance::from_var(fold.binder().var()))
            }
            // A `Guard` varies with its mask (a real child, in this arena)
            // and with whatever either arm varies with — resolved through
            // the same `referent_variance` a `Ref` leaf uses, and for the
            // same reason: the arms are names, and resolving is the only way
            // to know what a name denotes. Both arms, not just the taken
            // one: nothing here knows which arm a mask selects per-lane (and
            // a `Guard`'s whole point is that lanes may disagree), so the
            // honest answer is the union of every value the branch could
            // read, exactly as `Select`'s soft form already does.
            ExprNode::Guard { mask, on, off } => result[mask.0 as usize]
                .union(referent_variance(*on))
                .union(referent_variance(*off)),
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

/// Compute variance for every node in a DAG.
///
/// Because `dag.iter()` visits nodes strictly in children-before-parents order,
/// a single forward pass over `dag.iter()` suffices.
///
/// Returns a [`SideTable<Variance>`] indexed directly by [`Node<'_, ExprData>`].
#[must_use]
pub fn compute_dag_variance(
    dag: &crate::dag::Dag<crate::expr::ExprData>,
) -> crate::dag::SideTable<Variance> {
    use crate::expr::ExprData;

    let mut table = dag.side_table(Variance::CONST);

    for node in dag.iter() {
        let v = match *node {
            ExprData::Var(idx) => {
                if idx < 8 {
                    Variance::from_var(idx)
                } else {
                    Variance::ALL
                }
            }
            ExprData::Const(_) | ExprData::Buffer(_) | ExprData::Uniform(_) => Variance::CONST,
            ExprData::Param(_) => Variance::ALL,
            // The only node that *removes* a dependency: its binder is
            // bound here, so the body's variance on that one slot does not
            // escape.
            //
            // The binder comes off `Fold`, not out of a `Const` child. That
            // is the whole of the difference: this arm used to read a float,
            // ask `floorf` whether it was really an integer, check the
            // integer against a magic range to see whether it was really a
            // binder slot, and fall back to `Variance::ALL` when any of that
            // failed — a widening that was silent and unfalsifiable.
            ExprData::Reduce(fold) => {
                let body = node.children().next();
                let body_v = body.map_or(Variance::ALL, |b| table[b]);
                body_v.without(Variance::from_var(fold.binder().var()))
            }
            // A name has no variance of its own to compute. `Ref` is a leaf
            // whose referent lives in another graph, so nothing here can see
            // what it depends on; `ALL` is the sound answer, and the linker
            // (`passes::expand_refs`) is what turns it into a real one.
            ExprData::Ref(_) => Variance::ALL,
            // A `Guard` varies with its mask (the one real child here) and
            // with whatever either arm varies with, resolved the same way a
            // `Ref` leaf's variance is: both arms union in, since nothing at
            // this level knows which one a lane-varying mask will take.
            ExprData::Guard { on, off } => {
                let mask = node
                    .children()
                    .next()
                    .expect("Guard has one child: the mask");
                table[mask]
                    .union(referent_variance(on))
                    .union(referent_variance(off))
            }
            ExprData::Op(_) => {
                let mut v = Variance::CONST;
                for child in node.children() {
                    v = v.union(table[child]);
                }
                v
            }
        };
        table[node] = v;
    }

    table
}

/// Find arena nodes that should be hoisted out of the X-loop.
///
/// [`find_hoistable_out_of`] with `0` — the pixel loop's question.
#[must_use]
/// NOTE (2026-08-03): this is the ARENA-side hoisting analysis, and it has no
/// callers. The live loop-invariant-code-motion in the collapse compile path
/// does the same job over the *schedule* instead — see `schedule_variance` and
/// `plan_collapse_hoist` in pixelflow-codegen's `emit/mod.rs`, which
/// `compile_via_backend` runs at two scopes (whole-nest, then
/// per-row). So this is one analysis implemented twice at two tiers, the same
/// shape as the chain rule was. Which copy survives is an open question, not a
/// dormant feature.
pub fn find_hoistable_arena_nodes(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    variance: &[Variance],
    max: usize,
) -> Vec<crate::arena::ExprId> {
    find_hoistable_out_of(0, arena, root, variance, max)
}

/// Find arena nodes that should be hoisted out of the scope binding `var`.
///
/// Returns up to `max` `ExprId`s that are:
/// 1. invariant in `var` (so hoisting is legal)
/// 2. non-trivial (not Var or Const — actual computation worth hoisting)
/// 3. used by at least one `var`-dependent node (so hoisting pays)
///
/// Results are sorted by estimated cost (transcendentals first).
///
/// One function serves every level of the nest, because "can this leave the
/// loop" is one question asked of different variables: `0` for the pixel loop,
/// `1` for a scanline, a reduction's own slot for hoisting out of the fold —
/// which is the rewrite `⊕_i (f(i) · c) = c · ⊕_i f(i)` stated as an analysis.
///
/// # Panics
///
/// Panics if `var >= 8`.
#[must_use]
pub fn find_hoistable_out_of(
    var: u8,
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    variance: &[Variance],
    max: usize,
) -> Vec<crate::arena::ExprId> {
    use crate::arena::{ExprId, ExprNode};
    use crate::kind::OpKind;

    assert!(var < 8, "variable index must be 0..8");
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

    // Mark which reachable nodes are consumed by a `var`-dependent node: those
    // are the ones whose value has to cross the loop boundary to be useful.
    let mut feeds_dependent = alloc::vec![false; n];
    for i in 0..n {
        if !reachable[i] {
            continue;
        }
        let id = ExprId(i as u32);
        if variance[i].depends_on(var) {
            for child in arena.children(id) {
                feeds_dependent[child.0 as usize] = true;
            }
        }
    }

    // Collect hoistable candidates
    let mut candidates: Vec<(ExprId, u8)> = Vec::new(); // (id, priority)
    for i in 0..n {
        if !reachable[i] || !feeds_dependent[i] {
            continue;
        }
        let v = variance[i];
        if !v.is_invariant_in(var) || v.is_const() {
            continue; // Must be invariant in `var` and non-const
        }
        let id = ExprId(i as u32);
        let node = arena.node(id);

        // Skip trivial nodes (Var, Const, Param, Buffer, Uniform) — not worth a register
        let priority = match node {
            ExprNode::Var(_)
            | ExprNode::Const(_)
            | ExprNode::Param(_)
            | ExprNode::Buffer(_)
            | ExprNode::Uniform(_) => {
                continue;
            }
            // A loop-invariant memory read is well worth a register.
            ExprNode::Ternary(OpKind::Gather, _, _, _) => 2,
            ExprNode::Unary(
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
                | OpKind::Tan,
                _,
            ) => 3, // Transcendentals: highest priority
            ExprNode::Unary(_, _) => 1,
            ExprNode::Binary(op, _, _) => match *op {
                OpKind::Div => 2, // Division is expensive
                OpKind::Pow | OpKind::Atan2 => 3,
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

// Several tests below call `compute_arena_variance` directly. That is testing
// the public API, not an exception to it: the function is `pub`, and
// `pixelflow-search`'s `nnue::factored` calls it from outside this crate. Its
// in-crate callers are `passes::unroll_reduce` and `eval::eval_scalar`, both of
// which consume the whole per-node table; `find_hoistable_out_of` and
// `find_hoistable_arena_nodes` are not callers at all — they take an
// already-computed variance slice from theirs.
/// The extents of the lattice a kernel is compiled for: samples per axis,
/// `[x, y]`.
///
/// A kernel together with its lattice is a closed, finite, loop-free
/// expression: every axis has a static extent, so the whole program is one
/// straight-line DAG and a loop is only its run-length-encoded form. This is
/// the compile-time half of `pixelflow_core::Lattice` — the same `extent`,
/// with `origin` erased and nothing else erased. It is part of every compile
/// cache key, so a lattice of a different size is a different kernel:
/// resizing is recompilation, by decision
/// (`docs/plans/2026-09-01-loop-aware-codegen.md`).
///
/// An axis of extent 1 is a per-call constant; an axis of larger extent is a
/// binder the emitted code either distributes (unrolls) or factors (loops).
/// [`varying`](Self::varying) names the binders as a [`Variance`], so
/// `deps(node) ∩ shape.varying()` is the scope a node's value lives at.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LatticeShape([u32; crate::arena::COORD_AXES]);

impl LatticeShape {
    /// No lattice: one batch of caller-supplied points per call. Nothing is a
    /// binder.
    pub const POINT: Self = Self([1; crate::arena::COORD_AXES]);

    /// A lattice with these samples per axis.
    #[inline]
    #[must_use]
    pub const fn new(extent: [u32; crate::arena::COORD_AXES]) -> Self {
        Self(extent)
    }

    /// Samples per axis, `[x, y]`.
    #[inline]
    #[must_use]
    pub const fn extent(self) -> [u32; crate::arena::COORD_AXES] {
        self.0
    }

    /// The binders: one bit per axis of extent above 1, in [`Variance`]'s
    /// order. Equal to `pixelflow_core::Lattice::loop_mask()` for the same
    /// extents.
    #[inline]
    #[must_use]
    pub const fn varying(self) -> Variance {
        let mut bits = 0u8;
        let mut axis = 0;
        while axis < crate::arena::COORD_AXES {
            if self.0[axis] > 1 {
                bits |= 1 << axis;
            }
            axis += 1;
        }
        Variance(bits)
    }

    /// How many times this lattice evaluates a value whose dependencies are
    /// `deps` — the weight its cost carries in the whole program.
    ///
    /// The loop nest runs W outermost to X innermost, and nothing is
    /// materialized, so a value is recomputed once per iteration of the
    /// innermost binder it depends on: the product of the extents from that
    /// axis outward. A value depending on X runs at every sample; one
    /// depending only on Z runs once per Z plane; one depending on nothing
    /// runs once per call. That single rule is loop-invariant code motion,
    /// hoisting out of a reduction, and constant folding, priced.
    ///
    /// A dependency on a reduction binder counts as the innermost scope: the
    /// binder sits inside the coordinate nest and this type does not carry
    /// its extent. Binders are distributed before the e-graph sees them, so
    /// the case does not arise in practice.
    #[inline]
    #[must_use]
    pub const fn evals(self, deps: Variance) -> u64 {
        let innermost = if deps.depends_on_binder() {
            0
        } else {
            let bits = deps.bits();
            if bits == 0 {
                return 1;
            }
            bits.trailing_zeros() as usize
        };
        let mut count: u64 = 1;
        let mut axis = innermost;
        while axis < crate::arena::COORD_AXES {
            count *= self.0[axis] as u64;
            axis += 1;
        }
        count
    }

    /// The extents serialized little-endian, for cache keys.
    #[inline]
    #[must_use]
    pub const fn key_bytes(self) -> [u8; 4 * crate::arena::COORD_AXES] {
        let mut out = [0u8; 4 * crate::arena::COORD_AXES];
        let mut axis = 0;
        while axis < crate::arena::COORD_AXES {
            let b = self.0[axis].to_le_bytes();
            let mut k = 0;
            while k < 4 {
                out[axis * 4 + k] = b[k];
                k += 1;
            }
            axis += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_from_var() {
        assert_eq!(Variance::from_var(0), Variance::X);
        assert_eq!(Variance::from_var(1), Variance::Y);
        // The reduction index slots are variables in the same space, so the
        // hoisting question can be asked of them too.
        assert_eq!(
            Variance::from_var(4)
                .union(Variance::from_var(5))
                .union(Variance::from_var(6))
                .union(Variance::from_var(7)),
            Variance::BINDERS
        );
        // The retired axes' bits sit between the two, in no scope at all.
        assert!(Variance::COORDS.union(Variance::BINDERS) != Variance::ALL);
    }

    #[test]
    fn verify_union() {
        assert_eq!(Variance::X.union(Variance::Y), Variance::from_bits(0b0011));
        assert_eq!(Variance::CONST.union(Variance::Y), Variance::Y);
        assert_eq!(Variance::X.union(Variance::X), Variance::X);
        assert_eq!(Variance::X.union(Variance::Y), Variance::COORDS);
    }

    #[test]
    fn verify_meet() {
        // Fewer deps wins
        assert_eq!(Variance::CONST.meet(Variance::X), Variance::CONST);
        assert_eq!(Variance::X.meet(Variance::CONST), Variance::CONST);

        // Same popcount: lower raw value wins (deterministic)
        assert_eq!(Variance::X.meet(Variance::Y), Variance::X); // 0b0001 < 0b0010
        assert_eq!(Variance::Y.meet(Variance::X), Variance::X);

        // 2-bit vs 1-bit: 1-bit wins
        let xy = Variance::X.union(Variance::Y);
        assert_eq!(xy.meet(Variance::from_var(4)), Variance::from_var(4));
    }

    #[test]
    fn queries() {
        assert!(Variance::CONST.is_const());
        assert!(!Variance::X.is_const());

        assert!(Variance::X.depends_on_x());
        assert!(!Variance::Y.depends_on_x());

        assert!(Variance::Y.is_x_invariant());
        assert!(!Variance::X.is_x_invariant());

        // A uniform is CONST, and CONST is what lands in the per-call
        // prologue; nothing else is frame-uniform now that Z and W are gone.
        assert!(Variance::CONST.is_frame_uniform());
        assert!(!Variance::X.is_frame_uniform());
        assert!(!Variance::from_var(4).is_frame_uniform());

        assert!(Variance::X.is_spatially_varying());
        assert!(!Variance::CONST.is_spatially_varying());
    }

    #[test]
    fn debug_format() {
        assert_eq!(format!("{:?}", Variance::CONST), "Variance(CONST)");
        assert_eq!(format!("{:?}", Variance::X), "Variance{X}");
        assert_eq!(
            format!("{:?}", Variance::X.union(Variance::from_var(4))),
            "Variance{X,i4}"
        );
        assert_eq!(format!("{:?}", Variance::COORDS), "Variance{X,Y}");
        // Binder slots print as the slot they bind.
        assert_eq!(format!("{:?}", Variance::from_var(4)), "Variance{i4}");
        assert_eq!(
            format!("{:?}", Variance::Y.union(Variance::from_var(5))),
            "Variance{Y,i5}"
        );
        assert_eq!(
            format!("{:?}", Variance::ALL),
            "Variance{X,Y,?2,?3,i4,i5,i6,i7}"
        );
    }

    #[test]
    fn verify_popcount() {
        assert_eq!(Variance::CONST.popcount(), 0);
        assert_eq!(Variance::X.popcount(), 1);
        assert_eq!(Variance::X.union(Variance::Y).popcount(), 2);
        assert_eq!(Variance::COORDS.popcount(), 2);
        assert_eq!(Variance::BINDERS.popcount(), 4);
        assert_eq!(Variance::ALL.popcount(), 8);
    }

    /// A reference's variance is its referent's — resolved, not guessed. A
    /// blanket `ALL` would be safe but would stop LICM hoisting anything a
    /// named kernel feeds; a blanket `CONST` would hoist a Y-varying value
    /// out of the row loop and render the wrong picture.
    #[test]
    fn a_reference_varies_as_its_referent_does() {
        use crate::arena::ExprArena;
        use crate::kernel::Kernel;

        // X-only, Y-only, and constant referents: each must come back with
        // exactly the referent's own free coordinates.
        let cases = [
            (Kernel::x().sqrt(), Variance::X),
            (Kernel::y().neg(), Variance::Y),
            (Kernel::constant(4.0), Variance::CONST),
            (
                Kernel::x().add(&Kernel::y()),
                Variance::X.union(Variance::Y),
            ),
        ];
        for (referent, expected) in cases {
            let named = referent.by_ref();
            let (arena, root) = named.parts();
            let v = super::compute_arena_variance(arena);
            assert_eq!(v[root.0 as usize], expected);
        }

        // And it composes: a reference to an X-only kernel plus Y varies in
        // both, exactly as the spliced form would.
        let mixed = Kernel::x().sqrt().by_ref().add(&Kernel::y());
        let (arena, root) = mixed.parts();
        let v = super::compute_arena_variance(arena);
        assert_eq!(v[root.0 as usize], Variance::X.union(Variance::Y));

        // An unresolvable key claims nothing. `KernelKey::of` on a kernel
        // nobody interned is the honest way to get one: no store entry, so
        // no referent to read a variance off.
        let never = Kernel::x().mul(&Kernel::constant(1.0e-27));
        let (never_arena, never_root) = never.parts();
        let mut orphaned = ExprArena::new();
        let orphan = orphaned.push_ref(crate::key::KernelKey::of(never_arena, never_root));
        assert_eq!(
            super::compute_arena_variance(&orphaned)[orphan.0 as usize],
            Variance::ALL
        );
    }

    /// A `Guard` varies with its mask (a real child) *and* with whatever
    /// either arm varies with — both, not just one, because nothing at this
    /// level knows which arm a lane-varying mask will actually take. Same
    /// resolution path as a `Ref`: the arms are names, and `referent_variance`
    /// is how their variance is read at all.
    #[test]
    fn a_guard_varies_with_its_mask_and_both_arms() {
        use crate::arena::ExprArena;
        use crate::kernel::Kernel;
        use crate::kind::OpKind;
        use crate::store::KernelStore;

        // mask: Y > 0 → {Y}. on: X-only. off: constant.
        let on_key = KernelStore::intern(&Kernel::x().sqrt());
        let off_key = KernelStore::intern(&Kernel::constant(1.0));

        let mut arena = ExprArena::new();
        let y = arena.push_var(1);
        let zero = arena.push_const(0.0);
        let mask = arena.push_binary(OpKind::Gt, y, zero);
        let guard = arena.push_guard(mask, on_key, off_key);

        let v = super::compute_arena_variance(&arena);
        assert_eq!(
            v[guard.0 as usize],
            Variance::Y.union(Variance::X),
            "mask contributes Y, `on` contributes X, `off` contributes nothing"
        );

        // Swap in a Y-varying `off` arm too: now every one of the three
        // sources agrees on Y, and the union still carries X from `on`.
        let off_key_y = KernelStore::intern(&Kernel::y().neg());
        let guard_all_y_and_x = arena.push_guard(mask, on_key, off_key_y);
        assert_eq!(
            super::compute_arena_variance(&arena)[guard_all_y_and_x.0 as usize],
            Variance::X.union(Variance::Y)
        );

        // An unresolvable arm claims everything, exactly as an unresolvable
        // `Ref` does — the conservative answer when nothing can be resolved.
        let never = Kernel::x().mul(&Kernel::constant(1.0e-27));
        let (never_arena, never_root) = never.parts();
        let orphan_key = crate::key::KernelKey::of(never_arena, never_root);
        let orphan_guard = arena.push_guard(mask, orphan_key, off_key);
        assert_eq!(
            super::compute_arena_variance(&arena)[orphan_guard.0 as usize],
            Variance::ALL
        );
    }

    #[test]
    fn verify_compute_arena_variance() {
        use crate::arena::ExprArena;
        use crate::kind::OpKind;

        let mut arena = ExprArena::new();
        // Build: sin(u * 0.3) * (X + Y), where `u` is a uniform — the shape
        // the old `sin(Z * 0.3)` becomes, and CONST rather than a coordinate.
        let u = arena.declare_uniform(crate::Uniform::new(0.0).decl());
        let z = arena.push_uniform(u); // u → {}
        let c03 = arena.push_const(0.3); // 0.3 → {}
        let z_mul = arena.push_binary(OpKind::Mul, z, c03); // u*0.3 → {}
        let sin_z = arena.push_unary(OpKind::Sin, z_mul); // sin(u*0.3) → {}
        let x = arena.push_var(0); // X → {X}
        let y = arena.push_var(1); // Y → {Y}
        let x_add_y = arena.push_binary(OpKind::Add, x, y); // X+Y → {X,Y}
        let result = arena.push_binary(OpKind::Mul, sin_z, x_add_y); // → {X,Y}

        let v = super::compute_arena_variance(&arena);

        assert_eq!(v[z.0 as usize], Variance::CONST);
        assert_eq!(v[c03.0 as usize], Variance::CONST);
        assert_eq!(v[z_mul.0 as usize], Variance::CONST);
        assert_eq!(v[sin_z.0 as usize], Variance::CONST);
        assert!(v[sin_z.0 as usize].is_x_invariant());
        assert_eq!(v[x.0 as usize], Variance::X);
        assert_eq!(v[y.0 as usize], Variance::Y);
        assert_eq!(v[x_add_y.0 as usize], Variance::X.union(Variance::Y));
        assert_eq!(v[result.0 as usize], Variance::X.union(Variance::Y));
    }

    /// `deps(⊕_{i∈D} body) = deps(body) \ {i}` — REDUCTIONS_AND_FOLDS.md:32.
    /// The binder is the only construct that shrinks the set.
    #[test]
    fn binder_consumes_its_index() {
        use crate::Kernel;

        // Σ_{i<4} (i + X) depends on X, not on the index it binds.
        let k = Kernel::sum_over(4, |i| i.add(&Kernel::x()));
        let (arena, root) = k.parts();
        let v = super::compute_arena_variance(arena);
        assert_eq!(v[root.0 as usize], Variance::X);

        // Σ_{i<4} i depends on nothing at all: the index is bound, and it was
        // the body's only variable. This is the case that used to come back as
        // ALL — the analysis claimed maximal dependency for a closed term.
        let closed = Kernel::sum_over(4, Clone::clone);
        let (arena, root) = closed.parts();
        let v = super::compute_arena_variance(arena);
        assert!(
            v[root.0 as usize].is_const(),
            "Σ_i i has no free variables, got {:?}",
            v[root.0 as usize]
        );
    }

    /// Inside the body, the index IS free — and it is distinct from the
    /// coordinates, so `Y`-invariance and index-invariance are separate facts.
    #[test]
    fn binder_index_is_a_variable_like_any_other() {
        use crate::Kernel;
        use crate::arena::ExprNode;

        // Σ_{i<4} (i · Y): the product depends on both the index and Y.
        let k = Kernel::sum_over(4, |i| i.mul(&Kernel::y()));
        let (arena, root) = k.parts();
        let v = super::compute_arena_variance(arena);

        // The body — a fold's one child.
        let ExprNode::Reduce { body, .. } = arena.node(root) else {
            panic!("expected a Reduce at the root");
        };
        let body_v = v[body.0 as usize];

        assert!(
            body_v.depends_on_binder(),
            "body reads the index: {body_v:?}"
        );
        assert!(body_v.depends_on(4), "slot 4 is the only live binder");
        assert!(body_v.depends_on_y());
        assert!(body_v.is_x_invariant(), "nothing here reads X");
        assert!(
            !body_v.is_frame_uniform(),
            "a value that changes per fold step cannot lift to frame scope"
        );
        // The result, by contrast, has lost the index.
        assert_eq!(v[root.0 as usize], Variance::Y);
    }

    /// Nested binders occupy distinct slots, and each consumes exactly its own.
    #[test]
    fn nested_binders_consume_one_index_each() {
        use crate::Kernel;

        // Σ_{i<3} Σ_{j<4} (i + j + X) — both indices bound, X survives.
        let k = Kernel::sum_over(3, |i| {
            let i = i.clone();
            Kernel::sum_over(4, move |j| i.add(j).add(&Kernel::x()))
        });
        let (arena, root) = k.parts();
        let v = super::compute_arena_variance(arena);
        assert_eq!(v[root.0 as usize], Variance::X);
    }

    /// The LICM rule and the reduction-hoisting rule are one query at different
    /// variables: `⊕_i (f(i) · c) = c · ⊕_i f(i)` when `deps(c) ∩ {i} = {}`
    /// (REDUCTIONS_AND_FOLDS.md:109) is `find_hoistable_out_of(i, …)`.
    #[test]
    fn hoisting_out_of_a_binder_is_the_same_query_as_licm() {
        use crate::Kernel;
        use crate::arena::ExprNode;
        use crate::kind::OpKind;

        // Σ_{i<8} (i · sin(Y)) — sin(Y) is invariant in the index, so it can
        // leave the fold; it is NOT invariant in Y.
        let k = Kernel::sum_over(8, |i| i.mul(&Kernel::y().sin()));
        let (arena, root) = k.parts();
        let v = super::compute_arena_variance(arena);

        let ExprNode::Reduce { body, .. } = arena.node(root) else {
            panic!("expected a Reduce at the root");
        };
        let body = *body;

        let out_of_binder = super::find_hoistable_out_of(4, arena, body, &v, 8);
        let sin = out_of_binder
            .iter()
            .find(|id| matches!(arena.node(**id), ExprNode::Unary(OpKind::Sin, _)));
        assert!(
            sin.is_some(),
            "sin(Y) must be hoistable out of the fold, got {out_of_binder:?}"
        );

        // Asking about Y instead finds nothing: sin(Y) cannot cross that scope.
        let out_of_y = super::find_hoistable_out_of(1, arena, body, &v, 8);
        assert!(
            !out_of_y
                .iter()
                .any(|id| matches!(arena.node(*id), ExprNode::Unary(OpKind::Sin, _))),
            "sin(Y) must not be hoistable out of Y, got {out_of_y:?}"
        );
    }
}

#[cfg(test)]
mod lattice_shape_tests {
    use super::{LatticeShape, Variance, compute_dag_variance};

    #[test]
    fn binders_are_the_axes_with_extent_above_one() {
        assert_eq!(LatticeShape::POINT.varying(), Variance::CONST);
        assert_eq!(LatticeShape::new([37, 1]).varying(), Variance::X);
        assert_eq!(LatticeShape::new([1, 37]).varying(), Variance::Y);
        assert_eq!(LatticeShape::new([8, 8]).varying(), Variance::COORDS);
    }

    #[test]
    fn evals_counts_the_iterations_of_the_innermost_binder_depended_on() {
        let frame = LatticeShape::new([1920, 1080]);
        // Per sample, per row, per call — the three scopes the collapse loop
        // already has, as numbers.
        assert_eq!(frame.evals(Variance::X), 1920 * 1080);
        assert_eq!(frame.evals(Variance::X.union(Variance::Y)), 1920 * 1080);
        assert_eq!(frame.evals(Variance::Y), 1080);
        // A uniform is CONST, so it is evaluated once per call — which is the
        // whole reason a per-call scalar stopped being an axis of extent 1.
        assert_eq!(frame.evals(Variance::CONST), 1);

        // No lattice: everything is evaluated exactly once, so weighting a
        // cost by `evals` leaves it unchanged.
        for deps in [Variance::CONST, Variance::X, Variance::COORDS] {
            assert_eq!(LatticeShape::POINT.evals(deps), 1);
        }

        // A reduction binder sits inside the coordinate nest, and this type
        // does not carry its extent: count it as the innermost scope.
        assert_eq!(frame.evals(Variance::from_var(4)), 1920 * 1080);
    }

    #[test]
    fn key_bytes_are_the_extents_and_distinguish_sizes() {
        let a = LatticeShape::new([8, 8]);
        let b = LatticeShape::new([9, 8]);
        assert_eq!(a.key_bytes()[..4], 8u32.to_le_bytes());
        assert_eq!(a.key_bytes()[4..8], 8u32.to_le_bytes());
        assert_ne!(a.key_bytes(), b.key_bytes());
        assert_eq!(a.extent(), [8, 8]);
    }

    #[test]
    fn verify_compute_dag_variance() {
        use crate::dag::Builder;
        use crate::expr::ExprBuilderExt;
        use crate::kind::OpKind;

        let mut b = Builder::new();
        let x = b.push_var(0); // Variance::X
        let y = b.push_var(1); // Variance::Y
        let add = b.push_binary(OpKind::Add, x, y); // Variance::COORDS
        let c = b.push_const(5.0); // Variance::CONST
        let mul = b.push_binary(OpKind::Mul, add, c); // Variance::COORDS
        let rooted = b.finish(&[mul]);

        let var_table = compute_dag_variance(&rooted);
        assert_eq!(var_table[rooted.entry()], Variance::COORDS);
    }
}
