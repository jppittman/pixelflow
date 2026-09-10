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
//!   [`COORD_AXES`](crate::decl::COORD_AXES) axes and a per-call scalar is a
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
    /// [`ExprBuilder::push_var`](crate::expr::ExprBuilder::push_var) still
    /// builds them — `Var` is also a rewrite metavariable — and it is
    /// `emit::compile` that refuses one, so the bits stay constructible
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
                // A retired axis: no compiled kernel can name one, but the
                // macro tier's e-graph indexes its names in the same space.
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

// ───────────────────── DAG-level variance analysis ─────────────────────

/// Compute variance for every node in a DAG.
///
/// Because `dag.iter()` visits nodes strictly in children-before-parents order,
/// a single forward pass suffices — when a node is visited, every child's
/// answer is already in the table. O(V + E), one allocation.
///
/// Returns a [`SideTable<Variance>`](crate::dag::SideTable) indexed directly by
/// [`Node<'_, ExprData>`](crate::dag::Node).
///
/// There used to be a second copy of this over `ExprArena`, plus a
/// `find_hoistable_out_of` built on it that nothing called — the live
/// loop-invariant code motion runs over the *schedule* instead
/// (`schedule_variance`/`plan_collapse_hoist` in pixelflow-codegen's
/// `emit/mod.rs`). The arena copy went with the arena; the hoisting question
/// stays where it is answered.
#[must_use]
pub fn compute_dag_variance(
    dag: &crate::dag::Dag<crate::expr::ExprData>,
) -> crate::dag::SideTable<Variance> {
    use crate::expr::ExprData;
    use crate::kind::OpKind;

    let mut table = dag.side_table(Variance::CONST);

    for node in dag.iter() {
        let v = match *node {
            // Coordinates (0..2) and reduction index slots (4..8) each get
            // their own bit. Anything above that is not a variable this
            // analysis knows, so it claims nothing.
            ExprData::Var(idx) => {
                if idx < 8 {
                    Variance::from_var(idx)
                } else {
                    Variance::ALL
                }
            }
            // A buffer leaf is constant; a Gather's variance is the union of
            // its index expressions, which the `Op` arm below computes.
            //
            // A uniform is invariant on the lattice — that one line is what
            // moves everything computed from it alone into the per-call
            // prologue — and unknown on the parameter space, which is why it
            // is not folded like a `Const`.
            ExprData::Const(_) | ExprData::Buffer(_) | ExprData::Uniform(_) => Variance::CONST,
            // Parameters are substituted before compilation. Seeing one here
            // is conservatively all-varying.
            ExprData::Param(_) => Variance::ALL,
            // A binder is the only node that shrinks the set: it binds its
            // index, so the index is not free in the result.
            ExprData::Op(OpKind::Reduce) => {
                let mut kids = node.children();
                let (_combiner, bound, _extent, body) =
                    (kids.next(), kids.next(), kids.next(), kids.next());
                let body_v = body.map_or(Variance::ALL, |b| table[b]);
                match bound.and_then(bound_index_slot) {
                    Some(s) => body_v.without(Variance::from_var(s)),
                    // Malformed binder — refuse to claim invariance we cannot
                    // prove.
                    None => Variance::ALL,
                }
            }
            ExprData::Op(_) => node
                .children()
                .fold(Variance::CONST, |v, child| v.union(table[child])),
        };
        table[node] = v;
    }

    table
}

/// The index slot a `Reduce` binds, read from its second child (a `Const`
/// holding the slot number). `None` if that is not a well-formed binder index.
fn bound_index_slot(child: crate::dag::Node<'_, crate::expr::ExprData>) -> Option<u8> {
    let val = child.as_f32()?;
    if val != libm::floorf(val) || !(0.0..256.0).contains(&val) {
        return None;
    }
    let slot = val as u8;
    crate::decl::reduce_binders()
        .contains(&slot)
        .then_some(slot)
}

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
pub struct LatticeShape([u32; crate::decl::COORD_AXES]);

impl LatticeShape {
    /// No lattice: one batch of caller-supplied points per call. Nothing is a
    /// binder.
    pub const POINT: Self = Self([1; crate::decl::COORD_AXES]);

    /// A lattice with these samples per axis.
    #[inline]
    #[must_use]
    pub const fn new(extent: [u32; crate::decl::COORD_AXES]) -> Self {
        Self(extent)
    }

    /// Samples per axis, `[x, y]`.
    #[inline]
    #[must_use]
    pub const fn extent(self) -> [u32; crate::decl::COORD_AXES] {
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
        while axis < crate::decl::COORD_AXES {
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
        while axis < crate::decl::COORD_AXES {
            count *= self.0[axis] as u64;
            axis += 1;
        }
        count
    }

    /// The extents serialized little-endian, for cache keys.
    #[inline]
    #[must_use]
    pub const fn key_bytes(self) -> [u8; 4 * crate::decl::COORD_AXES] {
        let mut out = [0u8; 4 * crate::decl::COORD_AXES];
        let mut axis = 0;
        while axis < crate::decl::COORD_AXES {
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

    #[test]
    fn variance_of_a_uniform_expression_is_const() {
        use crate::expr::ExprBuilder;
        use crate::kind::OpKind;

        // sin(u * 0.3) * (X + Y), where `u` is a uniform — the shape the old
        // `sin(Z * 0.3)` becomes, and CONST rather than a coordinate.
        let mut b = ExprBuilder::new();
        let u = b.declare_uniform(crate::Uniform::new(0.0).decl());
        let z = b.push_uniform(u); // u → {}
        let c03 = b.push_const(0.3); // 0.3 → {}
        let z_mul = b.push_binary(OpKind::Mul, z, c03); // u*0.3 → {}
        let sin_z = b.push_unary(OpKind::Sin, z_mul); // sin(u*0.3) → {}
        let x = b.push_var(0); // X → {X}
        let y = b.push_var(1); // Y → {Y}
        let x_add_y = b.push_binary(OpKind::Add, x, y); // X+Y → {X,Y}
        let result = b.push_binary(OpKind::Mul, sin_z, x_add_y); // → {X,Y}
        let (rooted, _) = b.finish(&[z, c03, z_mul, sin_z, x, y, x_add_y, result]);

        let v = compute_dag_variance(&rooted);
        let at = |i: usize| v[rooted.entry_at(i)];
        for slot in 0..4 {
            assert_eq!(at(slot), Variance::CONST, "entry {slot}");
        }
        assert!(at(3).is_x_invariant(), "sin(u*0.3) leaves the pixel loop");
        assert_eq!(at(4), Variance::X);
        assert_eq!(at(5), Variance::Y);
        assert_eq!(at(6), Variance::COORDS);
        assert_eq!(at(7), Variance::COORDS);
    }

    /// `deps(⊕_{i∈D} body) = deps(body) \ {i}` — REDUCTIONS_AND_FOLDS.md:32.
    /// The binder is the only construct that shrinks the set.
    #[test]
    fn binder_consumes_its_index() {
        use crate::Kernel;

        // Σ_{i<4} (i + X) depends on X, not on the index it binds.
        let k = Kernel::sum_over(4, |i| i.add(&Kernel::x()));
        let v = compute_dag_variance(k.dag());
        assert_eq!(v[k.root()], Variance::X);

        // Σ_{i<4} i depends on nothing at all: the index is bound, and it was
        // the body's only variable. This is the case that used to come back as
        // ALL — the analysis claimed maximal dependency for a closed term.
        let closed = Kernel::sum_over(4, Clone::clone);
        let v = compute_dag_variance(closed.dag());
        assert!(
            v[closed.root()].is_const(),
            "Σ_i i has no free variables, got {:?}",
            v[closed.root()]
        );
    }

    /// Inside the body, the index IS free — and it is distinct from the
    /// coordinates, so `Y`-invariance and index-invariance are separate facts.
    #[test]
    fn binder_index_is_a_variable_like_any_other() {
        use crate::Kernel;

        // Σ_{i<4} (i · Y): the product depends on both the index and Y.
        let k = Kernel::sum_over(4, |i| i.mul(&Kernel::y()));
        let v = compute_dag_variance(k.dag());

        // The body is the Reduce's fourth child.
        let body = k.root().children().nth(3).expect("Reduce has a body");
        let body_v = v[body];

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
        assert_eq!(v[k.root()], Variance::Y);
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
        let v = compute_dag_variance(k.dag());
        assert_eq!(v[k.root()], Variance::X);
    }

    /// A malformed binder claims nothing: an index slot that is not a `Const`
    /// naming one leaves the analysis unable to prove the index is bound, and
    /// "unable to prove" is ALL rather than a guess.
    #[test]
    fn a_malformed_binder_is_all_varying() {
        use crate::expr::ExprBuilder;
        use crate::kind::OpKind;

        let mut b = ExprBuilder::new();
        let combiner = b.push_const(0.0);
        let not_a_slot = b.push_var(1); // where a Const(4..8) belongs
        let extent = b.push_const(3.0);
        let body = b.push_var(4);
        let root = b.push_nary(OpKind::Reduce, &[combiner, not_a_slot, extent, body]);
        let (rooted, _) = b.finish(&[root]);
        assert_eq!(compute_dag_variance(&rooted)[rooted.entry()], Variance::ALL);
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
