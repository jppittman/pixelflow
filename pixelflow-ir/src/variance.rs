//! # Variance Analysis
//!
//! Which variables an expression depends on, as a bitset. This is the shared
//! type used by both the e-graph's extractor (`pixelflow-search`, which prices
//! a node by [`LatticeShape::evals`] of it) and codegen's loop placement
//! (`pixelflow-codegen`'s `schedule_variance` and `place_roots`).
//!
//! ## Variable Mapping
//!
//! - Bit 0 (X): pixel column — varies per pixel
//! - Bit 1 (Y): pixel row — varies per scanline
//! - Bits 2..4: retired. They were the Z and W axes; a lattice has
//!   [`COORD_AXES`](crate::arena::COORD_AXES) axes and a per-call scalar is a
//!   uniform, whose variance is `CONST`.
//! - Bits 4..64: the reduction index slots — vary per step of the binder
//!   that binds them. Every bit the word has past the axes: the control
//!   plane is 64-bit, and how deep folds may nest is what the word holds,
//!   not a count chosen on its own.
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
/// Coordinates X=bit0, Y=bit1; reduction index slots in bits `4..64`.
/// Bits 2 and 3 are the retired Z and W axes and are never set. Operations:
/// - `union`: bitwise OR (join — a binary op depends on both operands' vars)
/// - `intersection`: bitwise AND (meet — what every one of several equal
///   terms is free of, the whole term is free of)
/// - `without`: set difference — what a binder does to its own index
///
/// This type is `no_std` compatible and zero-cost (single `u64`). It was a
/// `u8`, which capped nesting at four folds — a width no measurement asked
/// for, and one the lattice's own three folds would have all but used up.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Variance(u64);

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

    /// How many variable indices the bitset names: the coordinates, the two
    /// retired axes, and every reduction index slot. [`Self::from_var`]
    /// accepts `0..VARIABLES`, and a `Var` past it is not a variable this
    /// analysis knows.
    pub const VARIABLES: u8 = u64::BITS as u8;

    /// The reduction index slots a binder can bind: every bit past the
    /// coordinates and the retired axes.
    pub const BINDERS: Self = Self(u64::MAX << crate::arena::REDUCE_BINDER_BASE);

    /// Every variable — the top of the lattice, and the answer whenever the
    /// analysis cannot prove something narrower.
    pub const ALL: Self = Self(u64::MAX);

    /// Create from a variable index: `0..2` are the coordinates X/Y,
    /// `4..VARIABLES` the reduction index slots.
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
    /// Panics if `var_idx >= VARIABLES`.
    #[inline]
    #[must_use]
    pub const fn from_var(var_idx: u8) -> Self {
        assert!(
            var_idx < Self::VARIABLES,
            "variable index must be below Variance::VARIABLES"
        );
        Self(1 << var_idx)
    }

    /// Create from raw bits. Every bit is a variable.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Get the raw bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u64 {
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

    /// Intersection (meet): the variables both operands depend on.
    ///
    /// Used ACROSS terms known to be equal. Each term's variance is an
    /// over-approximation of the one function they all denote, so a variable
    /// absent from *any* of them is absent from the function:
    ///
    /// ```text
    /// var(C) = ⋂_{n ∈ C} var(n)
    /// ```
    ///
    /// This is the lattice meet, not a choice of representative: `X ∩ Y` is
    /// `CONST`, because a function that is constant along `Y` (the first
    /// term says so) and constant along `X` (the second does) is constant.
    #[inline]
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
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
    /// Panics if `var_idx >= VARIABLES`.
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
    /// Panics if `var_idx >= VARIABLES`.
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

    /// Number of variables this expression depends on (`0..=VARIABLES`).
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
        for bit in 0..Self::VARIABLES {
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
/// Public because `pixelflow-search` calls it from outside this crate: its
/// `nnue::factored::variance_histogram`, the classification behind
/// `Extraction::chosen_variance`, reads the whole per-node table (the e-graph
/// keeps the same fact per class, as `EGraph::variance`). In this crate
/// `passes::unroll_reduce` and `passes::lower_dwrt`'s tabulation rule read it
/// the same way.
#[must_use]
pub fn compute_arena_variance(arena: &crate::arena::ExprArena) -> Vec<Variance> {
    use crate::arena::{ExprId, ExprNode};

    let n = arena.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let id = ExprId(i as u32);
        let v = match arena.node(id) {
            // Coordinates and reduction index slots each get their own bit.
            // Anything past them is not a variable this analysis knows.
            ExprNode::Var(idx) => {
                if idx < Variance::VARIABLES {
                    Variance::from_var(idx)
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
            ExprNode::Ref(key) => referent_variance(key),
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
                .union(referent_variance(on))
                .union(referent_variance(off)),
            // A store varies with what it stores and with where: its three
            // binders are read for the address, so it sits inside all three
            // folds — which is the whole of why the lattice's loops can be
            // placed by the same rule as everything else.
            ExprNode::Write {
                row,
                col,
                lane,
                value,
            } => result[value.0 as usize]
                .union(Variance::from_var(row.var()))
                .union(Variance::from_var(col.var()))
                .union(Variance::from_var(lane.var())),
            ExprNode::Nary(..) => {
                let mut v = Variance::CONST;
                for child in arena.children(id) {
                    v = v.union(result[child.0 as usize]);
                }
                v
            }
        };
        result.push(v);
    }

    result
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
/// binder the emitted code loops over: `passes::lattice::collapse` wraps the
/// kernel in one fold per axis, and codegen emits each as a loop.
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
        let mut bits = 0u64;
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
    /// The loop nest runs Y outermost to X innermost, and nothing is
    /// materialized, so a value is recomputed once per iteration of the
    /// innermost axis it depends on: the product of the extents from that
    /// axis outward. A value depending on X runs at every sample; one
    /// depending only on Y runs once per row; one depending on nothing — a
    /// uniform's arithmetic included — runs once per call. That single rule
    /// is loop-invariant code motion and constant folding, priced.
    ///
    /// A dependency on a reduction binder counts as the innermost scope,
    /// every sample: this type carries neither the binder's trip count nor
    /// where codegen places its fold (one that reads no coordinate runs once
    /// per call, outside the lattice's folds). The case arises whenever a fold
    /// survives saturation — `pixelflow-search`'s extractor weights every node
    /// of a fold's body by this — so a node that reads a binder weighs what a
    /// per-sample node weighs, whatever else it reads, and the fold's trip
    /// count is not in the weight at all.
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

    // Several tests below call `compute_arena_variance` directly. That is
    // testing the public API, not an exception to it: the function is `pub`,
    // and `pixelflow-search`'s `nnue::factored` calls it from outside this
    // crate (see its doc).

    #[test]
    fn verify_from_var() {
        assert_eq!(Variance::from_var(0), Variance::X);
        assert_eq!(Variance::from_var(1), Variance::Y);
        // The reduction index slots are variables in the same space, so the
        // hoisting question can be asked of them too — every slot a binder
        // can take, and no other bit.
        let binders = crate::fold::Binder::all()
            .map(|b| Variance::from_var(b.var()))
            .fold(Variance::CONST, Variance::union);
        assert_eq!(binders, Variance::BINDERS);
        assert_eq!(
            Variance::BINDERS.popcount() as usize,
            crate::fold::Binder::COUNT
        );
        // The retired axes' bits sit between the two, in no scope at all.
        assert!(Variance::COORDS.union(Variance::BINDERS) != Variance::ALL);
        assert_eq!(
            Variance::COORDS
                .union(Variance::from_var(2))
                .union(Variance::from_var(3))
                .union(Variance::BINDERS),
            Variance::ALL
        );
    }

    #[test]
    fn verify_union() {
        assert_eq!(Variance::X.union(Variance::Y), Variance::from_bits(0b0011));
        assert_eq!(Variance::CONST.union(Variance::Y), Variance::Y);
        assert_eq!(Variance::X.union(Variance::X), Variance::X);
        assert_eq!(Variance::X.union(Variance::Y), Variance::COORDS);
    }

    #[test]
    fn verify_intersection() {
        assert_eq!(Variance::CONST.intersection(Variance::X), Variance::CONST);
        assert_eq!(Variance::X.intersection(Variance::CONST), Variance::CONST);

        // Two single, different variables share nothing: a function constant
        // along each is constant. A popcount minimum would have answered `X`.
        assert_eq!(Variance::X.intersection(Variance::Y), Variance::CONST);
        assert_eq!(Variance::Y.intersection(Variance::X), Variance::CONST);

        let xy = Variance::X.union(Variance::Y);
        let slot = Variance::from_var(4);
        assert_eq!(xy.intersection(slot), Variance::CONST);
        assert_eq!(xy.intersection(Variance::Y), Variance::Y);
        assert_eq!(xy.union(slot).intersection(slot), slot);
        assert_eq!(Variance::ALL.intersection(xy), xy);
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
        // Every variable, the last binder included.
        let all = format!("{:?}", Variance::ALL);
        assert!(all.starts_with("Variance{X,Y,?2,?3,i4,i5,"), "{all}");
        let last = Variance::VARIABLES - 1;
        assert!(all.ends_with(&format!(",i{last}}}")), "{all}");
    }

    #[test]
    fn verify_popcount() {
        assert_eq!(Variance::CONST.popcount(), 0);
        assert_eq!(Variance::X.popcount(), 1);
        assert_eq!(Variance::X.union(Variance::Y).popcount(), 2);
        assert_eq!(Variance::COORDS.popcount(), 2);
        assert_eq!(
            Variance::BINDERS.popcount(),
            u32::from(Variance::VARIABLES) - 4
        );
        assert_eq!(Variance::ALL.popcount(), u32::from(Variance::VARIABLES));
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

    /// A store varies with what it stores and with the three binders it
    /// stores under — it is inside all three lattice folds by its bits,
    /// which is what places it by the same rule as everything else.
    #[test]
    fn a_write_varies_with_its_value_and_its_binders() {
        use crate::fold::Binder;
        use crate::kind::OpKind;

        let slot = |s: u8| Binder::from_slot(s).expect("a binder slot");
        let (row, col, lane) = (slot(0), slot(1), slot(2));
        let mut arena = crate::arena::ExprArena::new();
        let y = arena.push_var(1);
        let l = arena.push_var(lane.var());
        let value = arena.push_binary(OpKind::Add, y, l);
        let write = arena.push_write(row, col, lane, value);

        let v = super::compute_arena_variance(&arena);
        assert_eq!(
            v[value.0 as usize],
            Variance::Y.union(Variance::from_var(lane.var()))
        );
        assert_eq!(
            v[write.0 as usize],
            Variance::Y
                .union(Variance::from_var(row.var()))
                .union(Variance::from_var(col.var()))
                .union(Variance::from_var(lane.var())),
            "the store reads every binder for its address"
        );
    }

    /// Folds nest past four. Each level takes the lowest free slot, so six
    /// nested sums bind six slots — the fifth and sixth were unrepresentable
    /// when the bitset was a byte — and every one is bound by the time the
    /// root is reached.
    #[test]
    fn six_nested_binders_each_take_a_slot() {
        use crate::Kernel;
        use crate::arena::ExprNode;

        // Past four: the depth the byte-wide bitset could not hold.
        const DEPTH: usize = 6;
        let mut k = Kernel::x();
        for _ in 0..DEPTH {
            let inner = k.clone();
            k = Kernel::sum_over(2, move |i| inner.add(i));
        }
        let (arena, root) = k.parts();
        let v = super::compute_arena_variance(arena);
        assert_eq!(
            v[root.0 as usize],
            Variance::X,
            "every binder is bound at the root"
        );

        let mut slots: alloc::vec::Vec<u8> = (0..arena.len())
            .filter_map(|i| match arena.node(crate::arena::ExprId(i as u32)) {
                ExprNode::Reduce { fold, .. } => Some(fold.binder().slot()),
                _ => None,
            })
            .collect();
        slots.sort_unstable();
        assert_eq!(slots, (0..DEPTH as u8).collect::<alloc::vec::Vec<u8>>());

        // And no fold's rename captured an inner fold's index: every fold's
        // body still reads the `Var` of the binder that fold chose. A body
        // is built against a placeholder that is renamed to a real slot once
        // the inner folds have taken theirs; were a placeholder's index a
        // slot an inner fold could take, the outer rename would rewrite the
        // inner index too — `Σ_i Σ_j f(i, j)` as `Σ_i Σ_j f(i, i)` — and
        // nothing downstream could tell. The placeholders sit past the whole
        // binder space, and this is the check that keeps them there.
        for i in 0..arena.len() {
            let id = crate::arena::ExprId(i as u32);
            let ExprNode::Reduce { fold, body } = arena.node(id) else {
                continue;
            };
            let own = fold.binder().var();
            let mut reads_own = false;
            let mut stack = alloc::vec![body];
            while let Some(n) = stack.pop() {
                if matches!(arena.node(n), ExprNode::Var(v) if v == own) {
                    reads_own = true;
                    break;
                }
                stack.extend(arena.children(n));
            }
            assert!(
                reads_own,
                "the fold binding slot {} lost its own index",
                fold.binder().slot()
            );
        }
    }
}

#[cfg(test)]
mod lattice_shape_tests {
    use super::{LatticeShape, Variance};

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
}
