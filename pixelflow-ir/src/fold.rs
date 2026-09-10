//! What a bounded reduction *is*: an algebra, an index, and a range.
//!
//! A `Reduce` node used to carry these three as `Const(f32)` children — an
//! `OpKind` reinterpreted through its discriminant, a binder slot, and a trip
//! count, all as floats, decoded by four separate readers each with its own
//! "malformed binder" fallback. CLAUDE.md lists that encoding among the places
//! where "the meaning lives in a comment instead of a type", and it is the
//! reason `Reduce` could not enter an e-graph: a metadata child is an e-class
//! like any other, so `Const(4.0)` naming the `Add` combiner is the *same
//! class* as any literal `4.0` in the kernel, and every arithmetic rule in the
//! set would have been free to rewrite it.
//!
//! So the metadata is a type, and it is part of the node rather than beneath
//! it. Nothing folds a [`Fold`], nothing can name it as a number, and
//! `Reduce`'s only child is its body.
//!
//! **The range is the load-bearing half.** A reduction over an *extent* has an
//! implicit lower bound of zero, which makes its decompositions expensive:
//! peeling the first term off `⊕_{[0,n)} f` leaves `⊕_{[0,n-1)} f(·+1)`, a
//! substitution through the whole body, and in an e-graph a rebuilt subgraph.
//! Over a *range* the same peel leaves `⊕_{[1,n)} f` — the body unchanged, and
//! therefore shared. One end of an interval is the difference between a rule
//! that fires and a rule nobody can afford.

use core::ops::Range;

use crate::arena::{REDUCE_BINDER_BASE, REDUCE_BINDERS};
use crate::kind::OpKind;

/// The algebra a reduction folds under: an associative combining operation
/// together with the identity an empty domain folds to.
///
/// [`Kernel::over`](crate::Kernel::over) is parametrized by this, so the binder
/// is one construct and the monoid is the knob — adding an algebra is adding a
/// constant here, not a new kind of fold. The named constructors
/// ([`Kernel::sum_over`](crate::Kernel::sum_over) and friends) are helpers over
/// that primitive.
///
/// Only associative operations with an identity qualify: associativity is what
/// lets the backend reassociate and vectorize the fold, and the identity is
/// what an empty domain denotes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Monoid(OpKind);

impl Monoid {
    /// `+`, identity `0` — contraction, integration, projection, accumulation.
    pub const SUM: Self = Self(OpKind::Add);
    /// `×`, identity `1`.
    pub const PRODUCT: Self = Self(OpKind::Mul);
    /// `max`, identity `−∞` — softmax's stabilizer, "best of a bounded set".
    pub const MAX: Self = Self(OpKind::Max);
    /// `min`, identity `+∞` — nearest hit over a bounded set of SDFs.
    pub const MIN: Self = Self(OpKind::Min);
    /// Mask `∨`, identity all-clear — the existential quantifier over a
    /// bounded domain.
    pub const ANY: Self = Self(OpKind::BitOr);
    /// Mask `∧`, identity all-set — the universal quantifier over a bounded
    /// domain.
    pub const ALL: Self = Self(OpKind::BitAnd);

    /// The combining operation. Crate-private: the op set is an IR concept,
    /// and consumers name algebras, not opcodes.
    pub(crate) fn op(self) -> OpKind {
        debug_assert!(
            self.0.is_monoid(),
            "Monoid must wrap an associative op with an identity"
        );
        self.0
    }

    /// The algebra `op` generates, or `None` if it generates none.
    ///
    /// The only way back from an opcode, and the reason nothing else needs to
    /// call [`OpKind::is_monoid`]: a `Monoid` that exists is one that passed.
    pub(crate) fn of(op: OpKind) -> Option<Self> {
        op.is_monoid().then_some(Self(op))
    }

    /// What an empty domain folds to.
    #[must_use]
    pub fn identity(self) -> f32 {
        self.op()
            .monoid_identity()
            .expect("a Monoid's operator has an identity")
    }
}

/// Which of the reserved index slots a fold binds.
///
/// Stores the slot (`0..`[`Binder::COUNT`]), not the [`Var`] index the body
/// reads, so an index outside the binder space is *unrepresentable* rather
/// than rejected: the arithmetic that used to recover a slot from a raw `u8`
/// (`(v as usize).checked_sub(BINDER_BASE)`, with a comment explaining what
/// wrapping would cost) has nowhere left to go wrong.
///
/// This is the one piece of the old encoding whose failure was genuinely
/// silent, which is why it gets a type and the trip count does not: a binder
/// index below the base names a *lattice coordinate*, so substituting a
/// literal for it would replace every `X` in the body and produce plausible,
/// wrong pixels.
///
/// [`Var`]: crate::arena::ExprNode::Var
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Binder(u8);

impl Binder {
    /// How many folds may be live at once — the depth of nesting a kernel may
    /// carry, and the size of the reserved index space.
    pub const COUNT: usize = REDUCE_BINDERS as usize;

    /// The `slot`-th binder, or `None` past [`Binder::COUNT`].
    #[must_use]
    pub fn from_slot(slot: u8) -> Option<Self> {
        ((slot as usize) < Self::COUNT).then_some(Self(slot))
    }

    /// The binder a [`Var`](crate::arena::ExprNode::Var) index names, or
    /// `None` if that index is a lattice coordinate or a retired axis.
    #[must_use]
    pub fn from_var(index: u8) -> Option<Self> {
        Self::from_slot(index.checked_sub(REDUCE_BINDER_BASE)?)
    }

    /// The `Var` index the body reads this binder through.
    #[must_use]
    pub fn var(self) -> u8 {
        REDUCE_BINDER_BASE + self.0
    }

    /// Which slot this is, for a caller tracking which are live.
    #[must_use]
    pub fn slot(self) -> u8 {
        self.0
    }

    /// Every binder, in slot order.
    pub fn all() -> impl Iterator<Item = Self> + Clone {
        (0..REDUCE_BINDERS).map(Self)
    }
}

/// The fold a [`Reduce`] performs: a monoid, the index it binds, and the
/// arithmetic progression that index runs over.
///
/// `⟦Reduce { fold, body }⟧ = ⊕_{k ∈ lo, lo+s, lo+2s, …, hi-s} ⟦body⟧[fold.binder() := k]`
///
/// `s` is [`Fold::stride`]: 1 until something calls [`Fold::halve`], the only
/// thing that ever changes it. [`Fold::range`] still names the bounding
/// `[lo, hi)`, which is why it is a poor stand-in for "every index visited"
/// once the stride is not 1 — [`Fold::len`] is the count that stays exact.
///
/// [`Reduce`]: crate::arena::ExprNode::Reduce
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Fold {
    monoid: Monoid,
    binder: Binder,
    /// Half-open `[lo, hi)`, `lo <= hi` by construction.
    lo: u16,
    hi: u16,
    /// The step between visited indices. `1` for every [`Fold::new`]; only
    /// [`Fold::halve`] ever doubles it, and only when `hi - lo` stays an
    /// exact multiple of it — an invariant [`Fold::new`] establishes
    /// (`stride` starts at 1, which divides anything) and [`Fold::halve`]
    /// preserves (it only fires on an even trip count, so the new stride
    /// still divides `hi - lo` exactly). That invariant is what makes
    /// [`Fold::len`] integer division rather than an approximation.
    stride: u16,
}

impl Fold {
    /// The largest index a fold may run to.
    ///
    /// This is not an arbitrary truncation of a `u32`. **A `Reduce` is not a
    /// loop**: the language is a DAG with no iteration binder, so a fold that
    /// survives to codegen is emitted as `len()` copies of its body. A trip
    /// count is therefore bounded by what one can afford to *emit* — the
    /// e-graph's `max_classes` budget dies orders of magnitude below this
    /// ceiling — and a range that does not fit is a program that was never
    /// going to compile. [`Fold::new`] rejects it rather than wrapping, and
    /// the two ends fit the node in the 16 bytes `ExprNode` is capped at.
    pub const MAX_INDEX: u32 = u16::MAX as u32;

    /// The fold of `monoid` over `range`, binding `binder`.
    ///
    /// # Panics
    ///
    /// Panics if `range` is reversed or runs past [`Fold::MAX_INDEX`].
    #[must_use]
    pub fn new(monoid: Monoid, binder: Binder, range: Range<u32>) -> Self {
        assert!(
            range.start <= range.end,
            "a fold's range runs forwards: {range:?}"
        );
        assert!(
            range.end <= Self::MAX_INDEX,
            "a fold over {range:?} would emit more than {} copies of its body",
            Self::MAX_INDEX
        );
        Self {
            monoid,
            binder,
            lo: range.start as u16,
            hi: range.end as u16,
            stride: 1,
        }
    }

    /// The algebra combining the terms.
    #[must_use]
    pub fn monoid(self) -> Monoid {
        self.monoid
    }

    /// The index this fold binds.
    #[must_use]
    pub fn binder(self) -> Binder {
        self.binder
    }

    /// The half-open range every visited index lies within.
    ///
    /// The *bound*, not the visited set: once [`Fold::halve`] has run, the
    /// indices in `[lo, hi)` that are actually visited are `lo`,
    /// `lo + stride`, `lo + 2·stride`, … — [`Fold::stride`] apart, not
    /// consecutive. [`Fold::len`] is the count that stays exact either way.
    #[must_use]
    pub fn range(self) -> Range<u32> {
        u32::from(self.lo)..u32::from(self.hi)
    }

    /// The step between one visited index and the next. `1` until
    /// [`Fold::halve`] doubles it.
    #[must_use]
    pub fn stride(self) -> u32 {
        u32::from(self.stride)
    }

    /// How many terms the fold combines — the *trip count*, which
    /// [`Fold::halve`] halves without touching [`Fold::range`]. Distinct from
    /// the index span `hi - lo`: they coincide only at `stride == 1`, and
    /// every caller here wants the trip count (how many times the body is
    /// evaluated), never the span.
    #[must_use]
    pub fn len(self) -> u32 {
        (u32::from(self.hi) - u32::from(self.lo)) / u32::from(self.stride)
    }

    /// Whether the domain is empty — in which case the fold *is* its monoid's
    /// identity, whatever the body says.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.lo == self.hi
    }

    /// The first index, and the fold over everything after it — or `None` when
    /// the domain is empty.
    ///
    /// ```text
    /// ⊕_{[lo,hi) step s} f  =  f(lo) ⊕ ⊕_{[lo+s,hi) step s} f
    /// ```
    ///
    /// The decomposition, as one method. Only the range moves, so the tail
    /// folds *the same body* — which is what lets an e-graph state this as a
    /// rule (the tail shares the body's e-class) and what makes unrolling and
    /// peeling one operation rather than two.
    #[must_use]
    pub fn peel(self) -> Option<(u32, Self)> {
        (!self.is_empty()).then(|| {
            let rest = Self {
                lo: self.lo + self.stride,
                ..self
            };
            (u32::from(self.lo), rest)
        })
    }

    /// Halve the trip count by doubling the stride: `[lo,hi) step s` becomes
    /// `[lo,hi) step 2s`. `None` when there are fewer than two terms, or an
    /// odd number of them — [`Fold::peel_back`] is the epilogue for the
    /// latter (peel one off the back, then halve the even remainder), not a
    /// second one.
    ///
    /// **Doubles the stride, does not split the range.** `[lo,hi) step s`
    /// visits `lo, lo+s, lo+2s, …` in order; doubling the stride re-groups
    /// that same sequence into adjacent pairs —
    /// `(f(lo)⊕f(lo+s)) ⊕ (f(lo+2s)⊕f(lo+3s)) ⊕ …` — the original order,
    /// only re-bracketed, so it needs associativity alone and holds for every
    /// [`Monoid`] here. Halving the *range* into `[lo,mid)` and `[mid,hi)`
    /// instead pairs `f(lo)⊕f(mid)`, `f(lo+1)⊕f(mid+1)`, … — it interleaves
    /// the sequence rather than re-bracketing it, and needs commutativity on
    /// top of associativity to still equal the original fold.
    ///
    /// Run to exhaustion (peeling the odd remainder as it arises) this
    /// reaches the same fully-unrolled term [`Fold::peel_back`] does, in
    /// `⌈log₂ n⌉` steps rather than `n`.
    #[must_use]
    pub fn halve(self) -> Option<Self> {
        let n = self.len();
        (n >= 2 && n.is_multiple_of(2)).then(|| Self {
            // `n` even means `hi - lo` is an exact multiple of `2 * stride`
            // (it was one of `stride`), so `len()` stays exact integer
            // division on the result.
            stride: self.stride * 2,
            ..self
        })
    }

    /// This fold as opaque bits, and [`Fold::from_bits`] back.
    ///
    /// Three callers need exactly this and nothing else: the runtime tier's
    /// JIT-cache key, [`canonical`](crate::key::canonical)'s content digest,
    /// and the `kernel!` macro, which emits an arena as tokens that rebuild
    /// it at load time. Each of them is *serializing* a fold rather than
    /// reasoning about one, so this is what they get — not an accessor for
    /// the combining opcode, which stays crate-private because the op set is
    /// an IR concept and a consumer names algebras.
    #[must_use]
    pub fn to_bits(self) -> u64 {
        u64::from(self.stride) << 48
            | u64::from(self.monoid.op().index() as u8) << 40
            | u64::from(self.binder.slot()) << 32
            | u64::from(self.lo) << 16
            | u64::from(self.hi)
    }

    /// The fold [`Fold::to_bits`] wrote, or `None` if the bits do not name
    /// one — an op that generates no algebra, an index outside the binder
    /// space, a reversed range, a zero stride, or a stride that does not
    /// divide `hi - lo` exactly (the invariant every constructor here
    /// maintains, and which [`Fold::len`] relies on).
    ///
    /// Total, so a corrupt cache key or a hand-written token stream is a
    /// `None` at the boundary rather than a fold that means something else.
    #[must_use]
    pub fn from_bits(bits: u64) -> Option<Self> {
        let stride = ((bits >> 48) & 0xffff) as u16;
        let monoid = Monoid::of(OpKind::from_index(((bits >> 40) & 0xff) as usize)?)?;
        let binder = Binder::from_slot(((bits >> 32) & 0xff) as u8)?;
        let lo = ((bits >> 16) & 0xffff) as u16;
        let hi = (bits & 0xffff) as u16;
        (lo <= hi && stride != 0 && (hi - lo).is_multiple_of(stride)).then_some(Self {
            monoid,
            binder,
            lo,
            hi,
            stride,
        })
    }

    /// The fold over everything *before* the last index, and that index — or
    /// `None` when the domain is empty.
    ///
    /// ```text
    /// ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi-s) step s} f ⊕ f(hi-s)
    /// ```
    ///
    /// [`Fold::peel`]'s mirror, and the one a rewrite should use: run to
    /// exhaustion over a `stride`-1 fold it builds the **left**-leaning chain
    /// `((f(lo) ⊕ f(lo+1)) ⊕ …)`. Peeling from the front instead yields the
    /// same value in the opposite association, and an e-graph then has to
    /// spend reassociation rules — and the classes they mint — to reach the
    /// shape everything downstream was tuned on.
    #[must_use]
    pub fn peel_back(self) -> Option<(Self, u32)> {
        (!self.is_empty()).then(|| {
            let rest = Self {
                hi: self.hi - self.stride,
                ..self
            };
            (rest, u32::from(rest.hi))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binder_is_a_slot_not_a_var_index() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(b.var(), REDUCE_BINDER_BASE);
        assert_eq!(Binder::from_var(b.var()), Some(b));
        // A coordinate axis is not a binder, and cannot be mistaken for one.
        assert_eq!(Binder::from_var(0), None);
        assert_eq!(Binder::from_var(REDUCE_BINDER_BASE + REDUCE_BINDERS), None);
        assert_eq!(Binder::all().count(), Binder::COUNT);
    }

    #[test]
    fn peeling_to_exhaustion_is_unrolling() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let mut fold = Fold::new(Monoid::SUM, b, 3..7);
        let mut indices = alloc::vec::Vec::new();
        while let Some((k, rest)) = fold.peel() {
            indices.push(k);
            fold = rest;
        }
        assert_eq!(indices, [3, 4, 5, 6]);
        assert!(fold.is_empty());
        assert_eq!(fold.len(), 0);
    }

    #[test]
    fn an_empty_fold_is_its_identity() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let empty = Fold::new(Monoid::SUM, b, 5..5);
        assert!(empty.is_empty());
        assert_eq!(empty.peel(), None);
        assert_eq!(empty.monoid().identity(), 0.0);
        assert_eq!(Monoid::PRODUCT.identity(), 1.0);
        assert_eq!(Monoid::MIN.identity(), f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "runs forwards")]
    fn a_reversed_range_is_refused() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        // Built through variables: a literal `7..3` is a lint about an empty
        // range, and the point here is the *constructor's* refusal.
        let (lo, hi) = (7u32, 3u32);
        assert!(Fold::new(Monoid::SUM, b, lo..hi).is_empty());
    }

    #[test]
    #[should_panic(expected = "copies of its body")]
    fn a_range_past_the_emit_ceiling_is_refused() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert!(Fold::new(Monoid::SUM, b, 0..Fold::MAX_INDEX + 1).is_empty());
    }

    #[test]
    fn halve_doubles_the_stride_and_halves_the_trip_count() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::new(Monoid::SUM, b, 0..8);
        assert_eq!(fold.stride(), 1);
        let once = fold.halve().expect("8 is even and has more than one term");
        assert_eq!(once.stride(), 2);
        assert_eq!(once.len(), 4);
        assert_eq!(once.range(), 0..8, "halve moves the stride, not the bound");
        let twice = once.halve().expect("4 is still even");
        assert_eq!(twice.stride(), 4);
        assert_eq!(twice.len(), 2);
        let thrice = twice.halve().expect("2 is still even");
        assert_eq!(thrice.stride(), 8);
        assert_eq!(thrice.len(), 1);
        // One term left: nothing to pair it with.
        assert_eq!(thrice.halve(), None);
    }

    #[test]
    fn halve_declines_on_an_odd_or_too_short_fold() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(
            Fold::new(Monoid::SUM, b, 0..0).halve(),
            None,
            "zero terms: nothing to pair"
        );
        assert_eq!(
            Fold::new(Monoid::SUM, b, 0..1).halve(),
            None,
            "one term: nothing to pair it with"
        );
        assert_eq!(
            Fold::new(Monoid::SUM, b, 0..7).halve(),
            None,
            "an odd trip count has no even pairing"
        );
        assert!(Fold::new(Monoid::SUM, b, 0..2).halve().is_some());
    }

    /// **The load-bearing property.** Halving to exhaustion (falling back to
    /// [`Fold::peel_back`] for the odd remainder, the preference
    /// `passes::expand_reduce` and `egraph::fold_rules::HalveFold` both give
    /// it) must visit the same terms, in the same left-to-right order, as
    /// [`Fold::peel`] does one at a time — for an even trip count (pure
    /// halving, no remainder ever arises) and an odd one (forces the
    /// peel-back epilogue at more than one level of the recursion).
    ///
    /// "Same order" here means the same *sequence* of leaves read
    /// left-to-right, not the same bracketing: halving `[lo,hi)` re-groups
    /// `(f₀⊕f₁) ⊕ (f₂⊕f₃) ⊕ …` rather than peeling's `((f₀⊕f₁)⊕f₂)⊕…`, and
    /// those are genuinely different trees, sound only because the monoid is
    /// associative. Every [`Monoid`] this crate has — `SUM`, `PRODUCT`,
    /// `MIN`, `MAX`, and the two mask quantifiers `ANY`/`ALL` — is also
    /// *commutative*, so a numeric total cannot even see this order property:
    /// a genuinely reordering decomposition (the rejected "halve-and-offset",
    /// module doc of `egraph::fold_rules`) would sum to the identical value.
    /// So this checks the sequence directly — with "term" specialized to its
    /// own raw index and "combine" to list concatenation, `combine` below is
    /// the shape `passes::expand_reduce::combine_halved` uses for real, just
    /// instantiated to make the order legible without an arena or a JIT.
    #[test]
    fn halving_to_exhaustion_visits_the_same_terms_in_the_same_order_as_peeling() {
        fn combine(fold: Fold, terms: &[alloc::vec::Vec<u32>]) -> alloc::vec::Vec<u32> {
            assert_eq!(fold.len() as usize, terms.len());
            if let Some(halved) = fold.halve() {
                let paired: alloc::vec::Vec<alloc::vec::Vec<u32>> = terms
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|[a, b]| [a.as_slice(), b.as_slice()].concat())
                    .collect();
                return combine(halved, &paired);
            }
            let (rest, _) = fold.peel_back().expect("a non-empty fold has a last term");
            let last = terms.last().expect("len matches terms.len()").clone();
            if rest.is_empty() {
                return last;
            }
            let mut acc = combine(rest, &terms[..terms.len() - 1]);
            acc.extend(last);
            acc
        }

        let b = Binder::from_slot(0).expect("slot 0 exists");
        for &(lo, hi) in &[(0u32, 8u32), (3, 11), (0, 7), (2, 9), (0, 1), (5, 6)] {
            let fold = Fold::new(Monoid::SUM, b, lo..hi);
            let want: alloc::vec::Vec<u32> = (lo..hi).collect();

            let terms: alloc::vec::Vec<alloc::vec::Vec<u32>> =
                (lo..hi).map(|k| alloc::vec![k]).collect();
            let halved_order = combine(fold, &terms);
            assert_eq!(
                halved_order, want,
                "halving {lo}..{hi} must visit every index once, in ascending order"
            );

            let mut peeled_order = alloc::vec::Vec::new();
            let mut rest = fold;
            while let Some((k, shorter)) = rest.peel() {
                peeled_order.push(k);
                rest = shorter;
            }
            assert_eq!(
                halved_order, peeled_order,
                "halving and peeling {lo}..{hi} to exhaustion must visit identical sequences"
            );
        }
    }

    #[test]
    fn to_bits_round_trips_a_halved_fold() {
        let b = Binder::from_slot(2).expect("slot 2 exists");
        let fold = Fold::new(Monoid::MAX, b, 10..26)
            .halve()
            .expect("16 is even")
            .halve()
            .expect("8 is still even");
        assert_eq!(fold.stride(), 4);
        assert_eq!(fold.len(), 4);

        let back = Fold::from_bits(fold.to_bits()).expect("a fold's own bits name a fold");
        assert_eq!(back, fold);
        assert_eq!(back.stride(), 4);
        assert_eq!(back.monoid(), Monoid::MAX);
        assert_eq!(back.binder(), b);
        assert_eq!(back.range(), 10..26);
    }

    #[test]
    fn from_bits_refuses_a_stride_that_does_not_divide_the_range() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let mut bits = Fold::new(Monoid::SUM, b, 0..5).to_bits();
        // Corrupt the stride field (bits 48..64) to 2, which does not divide
        // `5 - 0`: a fold that claims this would silently drop or duplicate
        // an index, so `from_bits` must refuse it rather than build one.
        bits = (bits & !(0xffffu64 << 48)) | (2u64 << 48);
        assert_eq!(Fold::from_bits(bits), None);
    }

    /// `ExprNode` is capped at 16 bytes (see the static assertion in
    /// `arena.rs`), and a `Reduce` node holds a `Fold` plus one `ExprId`.
    /// Adding `stride` grew `Fold` by a `u16` — pinned here as a byte count
    /// rather than left to the crate-wide assertion alone, so a future field
    /// that also fits *that* check but pushes `Fold` itself past what a
    /// `Reduce` node can spare fails here with a number, not just "too big".
    #[test]
    fn fold_and_expr_node_stay_within_the_sixteen_byte_budget() {
        assert_eq!(
            core::mem::size_of::<Fold>(),
            8,
            "Fold: monoid(1) + binder(1) + lo(2) + hi(2) + stride(2), no padding"
        );
        assert!(
            core::mem::size_of::<crate::arena::ExprNode>() <= 16,
            "a Reduce node is a Fold plus one ExprId; see arena.rs's own assertion"
        );
    }
}
