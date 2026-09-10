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

use crate::declarations::{REDUCE_BINDER_BASE, REDUCE_BINDERS};
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
/// half-open range that index runs over.
///
/// `⟦Reduce { fold, body }⟧ = ⊕_{k ∈ fold.range()} ⟦body⟧[fold.binder() := k]`
///
/// [`Reduce`]: crate::arena::ExprNode::Reduce
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Fold {
    monoid: Monoid,
    binder: Binder,
    /// Half-open `[lo, hi)`, `lo <= hi` by construction.
    lo: u16,
    hi: u16,
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

    /// The half-open range the index runs over.
    #[must_use]
    pub fn range(self) -> Range<u32> {
        u32::from(self.lo)..u32::from(self.hi)
    }

    /// How many terms the fold combines.
    #[must_use]
    pub fn len(self) -> u32 {
        u32::from(self.hi - self.lo)
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
    /// ⊕_{[lo,hi)} f  =  f(lo) ⊕ ⊕_{[lo+1,hi)} f
    /// ```
    ///
    /// The decomposition, as one method. Only the range moves, so the tail
    /// folds *the same body* — which is what lets an e-graph state this as a
    /// rule (the tail shares the body's e-class) and what makes unrolling and
    /// peeling one operation rather than two: **the legalizer is `peel` run to
    /// exhaustion.**
    #[must_use]
    pub fn peel(self) -> Option<(u32, Self)> {
        (!self.is_empty()).then(|| {
            let rest = Self {
                lo: self.lo + 1,
                ..self
            };
            (u32::from(self.lo), rest)
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
        u64::from(self.monoid.op().index() as u8) << 40
            | u64::from(self.binder.slot()) << 32
            | u64::from(self.lo) << 16
            | u64::from(self.hi)
    }

    /// The fold [`Fold::to_bits`] wrote, or `None` if the bits do not name
    /// one — an op that generates no algebra, an index outside the binder
    /// space, a reversed range.
    ///
    /// Total, so a corrupt cache key or a hand-written token stream is a
    /// `None` at the boundary rather than a fold that means something else.
    #[must_use]
    pub fn from_bits(bits: u64) -> Option<Self> {
        let monoid = Monoid::of(OpKind::from_index(((bits >> 40) & 0xff) as usize)?)?;
        let binder = Binder::from_slot(((bits >> 32) & 0xff) as u8)?;
        let lo = ((bits >> 16) & 0xffff) as u16;
        let hi = (bits & 0xffff) as u16;
        (lo <= hi).then_some(Self {
            monoid,
            binder,
            lo,
            hi,
        })
    }

    /// The fold over everything *before* the last index, and that index — or
    /// `None` when the domain is empty.
    ///
    /// ```text
    /// ⊕_{[lo,hi)} f  =  ⊕_{[lo,hi-1)} f ⊕ f(hi-1)
    /// ```
    ///
    /// [`Fold::peel`]'s mirror, and the one a rewrite should use: run to
    /// exhaustion it builds the **left**-leaning chain
    /// `((f(lo) ⊕ f(lo+1)) ⊕ …)`, which is the association
    /// `passes::expand_reduce` produces. Peeling from the front instead
    /// yields the same value in the opposite association, and an e-graph then
    /// has to spend reassociation rules — and the classes they mint — to
    /// reach the shape everything downstream was tuned on.
    #[must_use]
    pub fn peel_back(self) -> Option<(Self, u32)> {
        (!self.is_empty()).then(|| {
            let rest = Self {
                hi: self.hi - 1,
                ..self
            };
            (rest, u32::from(self.hi) - 1)
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
}
