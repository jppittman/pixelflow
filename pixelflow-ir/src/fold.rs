//! What a bounded reduction *is*: an algebra, an index, and a domain.
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
//!
//! **An integral is the same fold over a different measure.** The domain is
//! either a [`RangeFold`] — integers, counted, run as a loop — or an
//! [`IntervalFold`] — a real `[lo, hi)`, measured by length, which cannot
//! run and is closed by a rule or legalized by quadrature. The binder is the
//! same kind of thing in both, a fresh index the body reads
//! (docs/plans/2026-09-23-an-integral-is-a-fold.md §2).

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
    /// Sequencing, the **unit monoid**: combine is "then", identity is
    /// nothing. The algebra a lattice's folds are over — rows, batches and
    /// lanes wrapped around a [`Write`](crate::arena::ExprNode::Write)
    /// (docs/plans/2026-09-16-collapse-is-a-fold.md §2.4). A fold over it
    /// goes through the fold machinery unchanged; what its combine emits is
    /// no bytes, and the accumulator it would seed with
    /// [`identity`](Self::identity) is one nothing ever reads.
    pub const SEQ: Self = Self(OpKind::Seq);

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

/// The fold a [`Reduce`] performs: the index it binds, and the domain that
/// index ranges over — integers under a monoid, or a real interval under `Σ`.
///
/// ```text
/// ⟦Reduce { Fold::Range(r), body }⟧    = ⊕_{k ∈ lo, lo+s, …, hi-s} ⟦body⟧[r.binder() := k]
/// ⟦Reduce { Fold::Interval(i), body }⟧ = ∫_lo^hi ⟦body⟧[i.binder() := u] du
/// ```
///
/// The two domains differ in their measure — counting against length — and
/// that has one consequence: a range can run as a loop, and an interval
/// cannot. A continuous fold is closed by a rewrite rule, or legalized by
/// quadrature (`passes::resolve`) into arithmetic a loop nest can run
/// (docs/plans/2026-09-23-an-integral-is-a-fold.md §2).
///
/// An enum, not a struct with a domain field, because that is where the
/// difference can be refused rather than checked: a range accessor
/// ([`RangeFold::len`], [`RangeFold::stride`], [`RangeFold::peel`], …) on an
/// interval does not compile, since the caller has to match
/// [`Fold::Range`] to reach one, and an integral under any monoid but `Σ`
/// has nowhere to be written, since an [`IntervalFold`] has no monoid field.
/// What *is* common to both — the binder, the algebra, whether the domain is
/// empty, how many times legalized code evaluates the body — is a method
/// here.
///
/// [`Reduce`]: crate::arena::ExprNode::Reduce
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Fold {
    /// `⊕` over an arithmetic progression of integers — the counting
    /// measure. Runs as a loop.
    Range(RangeFold),
    /// `∫` over a real interval — the length measure, under `Σ` only. Never
    /// runs: a rule closes it, or legalization replaces it by its quadrature.
    Interval(IntervalFold),
}

/// Where [`Fold::to_bits`] writes which domain the rest of the bits describe.
///
/// A range's own layout ends at bit 112 and always left the bits above it
/// zero, so a zero tag there is every range ever serialized — a corpus
/// written before intervals existed decodes unchanged.
const DOMAIN_TAG_SHIFT: u32 = 112;
/// The domain tag of a [`Fold::Range`].
const RANGE_TAG: u128 = 0;
/// The domain tag of a [`Fold::Interval`].
const INTERVAL_TAG: u128 = 1;

impl Fold {
    /// The fold of `monoid` over `range`, binding `binder`, visiting every
    /// index in it — [`RangeFold::new`], as a [`Fold`].
    ///
    /// # Panics
    ///
    /// Panics if `range` is reversed.
    #[must_use]
    pub fn new(monoid: Monoid, binder: Binder, range: Range<u32>) -> Self {
        Self::Range(RangeFold::new(monoid, binder, range))
    }

    /// The fold of `monoid` over `range`, binding `binder`, visiting every
    /// `stride`-th index from `range.start` — the strip-mining `pack` uses to
    /// carve a lattice's column fold into a lane-width main fold and a
    /// narrower remainder (docs/plans/2026-09-16-collapse-is-a-fold.md §2.3).
    ///
    /// # Panics
    ///
    /// Panics if `range` is reversed, `stride` is `0`, or `stride` does not
    /// divide `range.end - range.start` exactly — the invariant
    /// [`RangeFold::len`] and [`Fold::from_bits`] both rely on.
    #[must_use]
    pub fn strided(monoid: Monoid, binder: Binder, range: Range<u32>, stride: u32) -> Self {
        Self::Range(RangeFold::strided(monoid, binder, range, stride))
    }

    /// The algebra combining the terms. An interval's is `Σ`, the only one
    /// an integral has.
    #[must_use]
    pub fn monoid(self) -> Monoid {
        match self {
            Self::Range(range) => range.monoid(),
            Self::Interval(_) => Monoid::SUM,
        }
    }

    /// The index this fold binds.
    #[must_use]
    pub fn binder(self) -> Binder {
        match self {
            Self::Range(range) => range.binder(),
            Self::Interval(interval) => interval.binder(),
        }
    }

    /// Whether the domain is empty — in which case the fold *is* its monoid's
    /// identity, whatever the body says.
    ///
    /// A question about the measure, so it has an answer for both domains:
    /// an interval is never empty, because [`IntervalFold`] cannot be built
    /// with `lo >= hi`.
    #[must_use]
    pub fn is_empty(self) -> bool {
        match self {
            Self::Range(range) => range.is_empty(),
            Self::Interval(_) => false,
        }
    }

    /// How many times legalized code evaluates the body for one evaluation
    /// of the fold: a range's trip count, or the number of nodes in the
    /// quadrature rule an interval is legalized by.
    ///
    /// The multiple a price scales the body by. For an interval it is a
    /// *legalization* count — what the fold costs if no rule closes it — and
    /// never an accuracy knob.
    #[must_use]
    pub fn evaluations(self) -> u64 {
        match self {
            Self::Range(range) => u64::from(range.len()),
            Self::Interval(interval) => interval.quadrature().len() as u64,
        }
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
    ///
    /// The domain is a tag at bit 112 (`DOMAIN_TAG_SHIFT`); below it, a range is
    /// `stride << 80 | op << 72 | binder << 64 | lo << 32 | hi` — the layout
    /// every range had before the tag existed — and an interval is
    /// `binder << 64 | lo << 32 | hi`, its endpoints as `f32` bit patterns.
    /// An interval stores no monoid: it has one algebra, so there is nothing
    /// to write.
    #[must_use]
    pub fn to_bits(self) -> u128 {
        match self {
            Self::Range(range) => RANGE_TAG << DOMAIN_TAG_SHIFT | range.to_bits(),
            Self::Interval(interval) => INTERVAL_TAG << DOMAIN_TAG_SHIFT | interval.to_bits(),
        }
    }

    /// The fold [`Fold::to_bits`] wrote, or `None` if the bits do not name
    /// one — an unknown domain tag, or a payload its domain refuses:
    /// [`RangeFold`]'s (an op that generates no algebra, an index outside
    /// the binder space, a reversed range, a zero stride, or a stride that
    /// does not divide `hi - lo` exactly) or [`IntervalFold`]'s (anything in
    /// the monoid or stride fields, a non-finite endpoint, the `-0.0` bit
    /// pattern, `lo >= hi`, or a length that overflows).
    ///
    /// Total, and the exact inverse of [`Fold::to_bits`]: a corrupt cache
    /// key or a hand-written token stream is a `None` at the boundary rather
    /// than a fold that means something else.
    #[must_use]
    pub fn from_bits(bits: u128) -> Option<Self> {
        let payload = bits & ((1u128 << DOMAIN_TAG_SHIFT) - 1);
        match bits >> DOMAIN_TAG_SHIFT {
            RANGE_TAG => RangeFold::from_bits(payload).map(Self::Range),
            INTERVAL_TAG => IntervalFold::from_bits(payload).map(Self::Interval),
            _ => None,
        }
    }
}

/// One fold printed the way every reader of an arena spells it: a range as
/// its combining op, binder and bounds (`add_4[0..8)`, with ` step 2` once
/// the stride is not 1), an interval as `∫_4[-0.5, 0.5)`.
impl core::fmt::Display for Fold {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Range(range) => {
                write!(
                    f,
                    "{}_{}[{}..{})",
                    range.combine_op().name(),
                    range.binder().var(),
                    range.lo,
                    range.hi
                )?;
                // The step is worth stating once it is not 1 — the shape
                // `RangeFold::halve` leaves behind — since the bounds alone
                // would then read as "every index" and aren't.
                if range.stride != 1 {
                    write!(f, " step {}", range.stride)?;
                }
                Ok(())
            }
            Self::Interval(interval) => write!(
                f,
                "∫_{}[{:?}, {:?})",
                interval.binder().var(),
                interval.lo(),
                interval.hi()
            ),
        }
    }
}

/// A fold over an arithmetic progression of integers: a monoid, the index
/// it binds, and the progression that index runs over.
///
/// `⊕_{k ∈ lo, lo+s, lo+2s, …, hi-s} ⟦body⟧[binder := k]`
///
/// `s` is [`RangeFold::stride`]: 1 from [`RangeFold::new`], or whatever
/// [`Fold::strided`] was given directly — `pack` strip-mining a lattice's
/// column fold is the reason a second constructor exists at all. After
/// construction, only [`RangeFold::halve`] ever changes it, by doubling.
/// [`RangeFold::range`] still names the bounding `[lo, hi)`, which is why it
/// is a poor stand-in for "every index visited" once the stride is not 1 —
/// [`RangeFold::len`] is the count that stays exact.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct RangeFold {
    monoid: Monoid,
    binder: Binder,
    /// Half-open `[lo, hi)`, `lo <= hi` by construction.
    ///
    /// `u32`, the width [`RangeFold::range`] speaks. These were `u16`, first
    /// "so the two fit the node in the 16 bytes `ExprNode` is capped at" — a
    /// width nothing depended on, made into a cap on how many terms a
    /// reduction may have — and then because an unrolled fold could not
    /// afford more copies of its body. A surviving `Reduce` is a loop now
    /// (docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md), so a trip
    /// count costs a counter, and the control plane is 64-bit: an extent
    /// describing a program is not narrowed without a measurement asking.
    lo: u32,
    hi: u32,
    /// The step between visited indices. `1` for every [`RangeFold::new`],
    /// or whatever [`Fold::strided`] was constructed with. After that, only
    /// [`RangeFold::halve`] ever doubles it, and only when `hi - lo` stays an
    /// exact multiple of it — an invariant [`Fold::strided`] checks directly
    /// (it refuses a stride that does not divide the span), [`RangeFold::new`]
    /// gets for free by going through it with a stride of 1 (which divides
    /// anything), and [`RangeFold::halve`] preserves (it only fires on an
    /// even trip count, so the new stride still divides `hi - lo` exactly).
    /// That invariant is what makes [`RangeFold::len`] integer division
    /// rather than an approximation.
    stride: u32,
}

impl RangeFold {
    /// The fold of `monoid` over `range`, binding `binder`, visiting every
    /// index in it — [`Fold::strided`] with a stride of `1`.
    ///
    /// # Panics
    ///
    /// Panics if `range` is reversed.
    #[must_use]
    pub fn new(monoid: Monoid, binder: Binder, range: Range<u32>) -> Self {
        Self::strided(monoid, binder, range, 1)
    }

    /// [`Fold::strided`]'s range. Crate-private: [`Fold::strided`] is the
    /// one public spelling.
    pub(crate) fn strided(monoid: Monoid, binder: Binder, range: Range<u32>, stride: u32) -> Self {
        assert!(
            range.start <= range.end,
            "a fold's range runs forwards: {range:?}"
        );
        assert!(stride != 0, "a fold's stride must be nonzero");
        assert!(
            (range.end - range.start).is_multiple_of(stride),
            "a fold's stride must divide its span exactly: {range:?} step {stride}"
        );
        Self {
            monoid,
            binder,
            lo: range.start,
            hi: range.end,
            stride,
        }
    }

    /// The algebra combining the terms.
    #[must_use]
    pub fn monoid(self) -> Monoid {
        self.monoid
    }

    /// The combining opcode, for a backend that has decided to *emit* this
    /// fold as a loop rather than have the e-graph reason about it further.
    ///
    /// `Monoid::op` stays crate-private — "consumers name algebras, not
    /// opcodes" is right for anything still inside the algebra (rewrite
    /// rules, extraction, the cost model). Codegen is not that: once a
    /// `Reduce` has survived extraction, the surviving loop's own combine
    /// instruction has to be *some* opcode, and this is the one narrow door
    /// for the one consumer past the e-graph that needs it (see
    /// `pixelflow-codegen`'s `IsaBackend::alu`, and
    /// docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md's "2a″" for why
    /// the combine could not live in the arena instead).
    #[must_use]
    pub fn combine_op(self) -> OpKind {
        self.monoid.op()
    }

    /// The index this fold binds.
    #[must_use]
    pub fn binder(self) -> Binder {
        self.binder
    }

    /// The half-open range every visited index lies within.
    ///
    /// The *bound*, not the visited set: once [`RangeFold::halve`] has run,
    /// the indices in `[lo, hi)` that are actually visited are `lo`,
    /// `lo + stride`, `lo + 2·stride`, … — [`RangeFold::stride`] apart, not
    /// consecutive. [`RangeFold::len`] is the count that stays exact either
    /// way.
    #[must_use]
    pub fn range(self) -> Range<u32> {
        self.lo..self.hi
    }

    /// The step between one visited index and the next. `1` from
    /// [`RangeFold::new`], or [`Fold::strided`]'s own argument;
    /// [`RangeFold::halve`] is the only thing that doubles it afterward.
    #[must_use]
    pub fn stride(self) -> u32 {
        self.stride
    }

    /// How many terms the fold combines — the *trip count*, which
    /// [`RangeFold::halve`] halves without touching [`RangeFold::range`].
    /// Distinct from the index span `hi - lo`: they coincide only at
    /// `stride == 1`, and every caller here wants the trip count (how many
    /// times the body is evaluated), never the span.
    #[must_use]
    pub fn len(self) -> u32 {
        (self.hi - self.lo) / self.stride
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
            (self.lo, rest)
        })
    }

    /// Halve the trip count by doubling the stride: `[lo,hi) step s` becomes
    /// `[lo,hi) step 2s`. `None` when there are fewer than two terms, or an
    /// odd number of them — [`RangeFold::peel_back`] is the epilogue for the
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
    /// reaches the same fully-unrolled term [`RangeFold::peel_back`] does, in
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

    /// The fold over everything *before* the last index, and that index — or
    /// `None` when the domain is empty.
    ///
    /// ```text
    /// ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi-s) step s} f ⊕ f(hi-s)
    /// ```
    ///
    /// [`RangeFold::peel`]'s mirror, and the one a rewrite should use: run to
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
            (rest, rest.hi)
        })
    }

    /// The payload [`Fold::to_bits`] writes below the domain tag.
    fn to_bits(self) -> u128 {
        u128::from(self.stride) << 80
            | u128::from(self.monoid.op().index() as u8) << 72
            | u128::from(self.binder.slot()) << 64
            | u128::from(self.lo) << 32
            | u128::from(self.hi)
    }

    /// The range [`RangeFold::to_bits`] wrote, or `None` — see
    /// [`Fold::from_bits`] for what is refused.
    fn from_bits(bits: u128) -> Option<Self> {
        let stride = ((bits >> 80) & 0xffff_ffff) as u32;
        let monoid = Monoid::of(OpKind::from_index(((bits >> 72) & 0xff) as usize)?)?;
        let binder = Binder::from_slot(((bits >> 64) & 0xff) as u8)?;
        let lo = ((bits >> 32) & 0xffff_ffff) as u32;
        let hi = (bits & 0xffff_ffff) as u32;
        (lo <= hi && stride != 0 && (hi - lo).is_multiple_of(stride)).then_some(Self {
            monoid,
            binder,
            lo,
            hi,
            stride,
        })
    }
}

/// An end of an [`IntervalFold`]: a finite `f32`, held as its bit pattern
/// the way [`ExprNode::Const`](crate::arena::ExprNode::Const) holds one, so
/// equality, hashing and ordering derive.
///
/// `-0.0` is stored as `+0.0`. The two are one number, and a fold is a key
/// (`canonical`, the JIT cache, hash-consing): with both patterns
/// representable, `[-0, 1)` and `[0, 1)` would be two integrals of one
/// domain. After canonicalizing, bit equality *is* numeric equality on every
/// value this can hold. The derived order is on bit patterns — it exists for
/// interning, not arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct Endpoint(u32);

impl Endpoint {
    /// The bit pattern of `-0.0`, which no endpoint carries.
    const NEGATIVE_ZERO: u32 = 0x8000_0000;

    /// `value` as an endpoint, or `None` if it is not finite.
    fn new(value: f32) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        // `-0.0 == 0.0`, so this one comparison sends both zeros to `+0.0`.
        if value == 0.0 {
            return Some(Self(0));
        }
        Some(Self(value.to_bits()))
    }

    /// The endpoint whose bits [`Endpoint::bits`] wrote, or `None` if they
    /// are not one: a non-finite value, or the `-0.0` pattern [`Endpoint::new`]
    /// never writes.
    fn from_bits(bits: u32) -> Option<Self> {
        (bits != Self::NEGATIVE_ZERO && f32::from_bits(bits).is_finite()).then_some(Self(bits))
    }

    fn bits(self) -> u32 {
        self.0
    }

    fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// One node of a quadrature rule: where the body is sampled, and the weight
/// that sample carries. `∫_lo^hi f ≈ Σ_k weight_k · f(point_k)`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct QuadratureNode {
    pub(crate) point: f32,
    pub(crate) weight: f32,
}

/// A fold over a real interval: `∫_lo^hi ⟦body⟧[binder := u] du`, `lo < hi`.
///
/// **The endpoints are data-plane `f32`, by CLAUDE.md's own division.** The
/// 64-bit rule is for what *describes a program* — an index, an id, a count,
/// an extent, a bound on how many times something runs — and a range's ends
/// are that: they are its trip count. An interval's ends are not. An
/// interval has no trip count: it never runs as a loop, and no index
/// visits it. Its ends are values the kernel computes with — a closing rule
/// writes them into an antiderivative (`(hi² - lo²)/2`), and quadrature
/// substitutes the midpoint for the binder and multiplies by the length —
/// so they become `Const(f32)` leaves in the body, the same width as every
/// other lane value. Widening them to `f64` would describe a precision the
/// arithmetic they feed does not have.
///
/// No monoid: an integral is a sum, and `Σ` is the only algebra with a
/// measure to integrate against. [`Fold::monoid`] answers `Σ` for it.
///
/// Opaque outside this crate, and built only by `Kernel::area` (or decoded
/// by [`Fold::from_bits`]): which interval a kernel integrates is a fact its
/// constructor chose, not one a consumer edits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntervalFold {
    binder: Binder,
    lo: Endpoint,
    hi: Endpoint,
}

impl IntervalFold {
    /// `∫_lo^hi`, binding `binder`.
    ///
    /// # Panics
    ///
    /// Panics unless `lo` and `hi` are finite, `lo < hi`, and `hi - lo` is
    /// finite. Strict: an empty interval is unrepresentable, so an interval
    /// is never an identity (see [`Fold::is_empty`]) and its quadrature
    /// never multiplies a sample by a zero length.
    pub(crate) fn new(binder: Binder, lo: f32, hi: f32) -> Self {
        Self::try_new(binder, lo, hi).unwrap_or_else(|| {
            panic!(
                "an interval fold needs finite ends with lo < hi and a finite \
                 length: [{lo:?}, {hi:?})"
            )
        })
    }

    fn try_new(binder: Binder, lo: f32, hi: f32) -> Option<Self> {
        let (lo, hi) = (Endpoint::new(lo)?, Endpoint::new(hi)?);
        (lo.get() < hi.get() && (hi.get() - lo.get()).is_finite()).then_some(Self {
            binder,
            lo,
            hi,
        })
    }

    /// The index this fold binds — [`Fold::binder`] is the public spelling.
    pub(crate) fn binder(self) -> Binder {
        self.binder
    }

    /// The lower end.
    pub(crate) fn lo(self) -> f32 {
        self.lo.get()
    }

    /// The upper end.
    pub(crate) fn hi(self) -> f32 {
        self.hi.get()
    }

    /// The centre of the interval, `½·lo + ½·hi` — written so, rather than
    /// `(lo + hi)/2`, because it cannot overflow where the ends are finite.
    pub(crate) fn midpoint(self) -> f32 {
        0.5 * self.lo() + 0.5 * self.hi()
    }

    /// The interval's measure, `hi - lo`. Finite and positive by
    /// construction.
    pub(crate) fn length(self) -> f32 {
        self.hi() - self.lo()
    }

    /// The rule legalization replaces this integral by: one node, the
    /// midpoint, weighted by the length — exact for a body affine in the
    /// binder, and the point sample every kernel computed before integrals
    /// existed when the interval is the centred pixel.
    ///
    /// The one place the rule is written. `passes::quadrature` emits
    /// whatever nodes this returns, and [`Fold::evaluations`] counts them,
    /// so the price and the emitted code cannot disagree about how many
    /// samples an unclosed integral costs. The weight is always emitted,
    /// including the pixel's `1`: a unit-weight elision here would be a
    /// branch on a value that sends a mask-domain integrand through
    /// untouched on one interval and scaled on every other. A closing rule
    /// is what removes the multiply, before legalization ever sees the
    /// fold.
    pub(crate) fn quadrature(self) -> [QuadratureNode; 1] {
        [QuadratureNode {
            point: self.midpoint(),
            weight: self.length(),
        }]
    }

    /// The payload [`Fold::to_bits`] writes below the domain tag.
    fn to_bits(self) -> u128 {
        u128::from(self.binder.slot()) << 64
            | u128::from(self.lo.bits()) << 32
            | u128::from(self.hi.bits())
    }

    /// The interval [`IntervalFold::to_bits`] wrote, or `None` — see
    /// [`Fold::from_bits`] for what is refused.
    fn from_bits(bits: u128) -> Option<Self> {
        // Where a range keeps its monoid and stride. An interval has
        // neither, so anything written there is not an interval.
        if bits >> 72 != 0 {
            return None;
        }
        let binder = Binder::from_slot(((bits >> 64) & 0xff) as u8)?;
        let lo = Endpoint::from_bits(((bits >> 32) & 0xffff_ffff) as u32)?;
        let hi = Endpoint::from_bits((bits & 0xffff_ffff) as u32)?;
        Self::try_new(binder, lo.get(), hi.get())
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
        let mut fold = RangeFold::new(Monoid::SUM, b, 3..7);
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
        let empty = RangeFold::new(Monoid::SUM, b, 5..5);
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
        assert!(RangeFold::new(Monoid::SUM, b, lo..hi).is_empty());
    }

    /// A trip count past what a `u16` held. A surviving fold is a loop, so
    /// this costs a counter, not sixty-six thousand copies of a body.
    #[test]
    fn a_range_past_sixteen_bits_is_a_fold_like_any_other() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let wide = RangeFold::new(Monoid::SUM, b, 1_000..1_000_000);
        assert_eq!(wide.len(), 999_000);
        assert_eq!(wide.range(), 1_000..1_000_000);
        let halved = wide.halve().expect("an even trip count halves");
        assert_eq!(halved.len(), 499_500);
        let back =
            Fold::from_bits(Fold::Range(wide).to_bits()).expect("a fold's own bits name a fold");
        assert_eq!(back, Fold::Range(wide));
    }

    /// The unit monoid: a fold over it is a loop whose combine says nothing.
    #[test]
    fn seq_is_a_monoid_whose_combine_is_sequencing() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = RangeFold::new(Monoid::SEQ, b, 0..4);
        assert_eq!(fold.combine_op(), OpKind::Seq);
        assert_eq!(fold.monoid(), Monoid::SEQ);
        // The seed of an accumulator no combine ever reads.
        assert_eq!(Monoid::SEQ.identity(), 0.0);
        assert_eq!(Monoid::of(OpKind::Seq), Some(Monoid::SEQ));
        let back = Fold::from_bits(Fold::Range(fold).to_bits()).expect("a SEQ fold round-trips");
        assert_eq!(back, Fold::Range(fold));
    }

    #[test]
    fn strided_builds_with_the_given_step() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = RangeFold::strided(Monoid::SUM, b, 0..12, 3);
        assert_eq!(fold.stride(), 3);
        assert_eq!(
            fold.range(),
            0..12,
            "range() names the bound, not stride's steps"
        );
        assert_eq!(fold.len(), 4);
        let back = Fold::from_bits(Fold::Range(fold).to_bits())
            .expect("a strided fold's own bits name it");
        assert_eq!(back, Fold::Range(fold));
    }

    #[test]
    #[should_panic(expected = "must divide")]
    fn strided_refuses_a_stride_that_does_not_divide_the_span() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert!(!RangeFold::strided(Monoid::SUM, b, 0..10, 3).is_empty());
    }

    #[test]
    #[should_panic(expected = "nonzero")]
    fn strided_refuses_a_zero_stride() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert!(!RangeFold::strided(Monoid::SUM, b, 0..10, 0).is_empty());
    }

    #[test]
    fn new_is_strided_with_a_stride_of_one() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(
            RangeFold::new(Monoid::SUM, b, 2..9),
            RangeFold::strided(Monoid::SUM, b, 2..9, 1)
        );
    }

    #[test]
    fn halve_doubles_the_stride_and_halves_the_trip_count() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = RangeFold::new(Monoid::SUM, b, 0..8);
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
            RangeFold::new(Monoid::SUM, b, 0..0).halve(),
            None,
            "zero terms: nothing to pair"
        );
        assert_eq!(
            RangeFold::new(Monoid::SUM, b, 0..1).halve(),
            None,
            "one term: nothing to pair it with"
        );
        assert_eq!(
            RangeFold::new(Monoid::SUM, b, 0..7).halve(),
            None,
            "an odd trip count has no even pairing"
        );
        assert!(RangeFold::new(Monoid::SUM, b, 0..2).halve().is_some());
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
        fn combine(fold: RangeFold, terms: &[alloc::vec::Vec<u32>]) -> alloc::vec::Vec<u32> {
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
            let fold = RangeFold::new(Monoid::SUM, b, lo..hi);
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
        let fold = RangeFold::new(Monoid::MAX, b, 10..26)
            .halve()
            .expect("16 is even")
            .halve()
            .expect("8 is still even");
        assert_eq!(fold.stride(), 4);
        assert_eq!(fold.len(), 4);

        let back =
            Fold::from_bits(Fold::Range(fold).to_bits()).expect("a fold's own bits name a fold");
        assert_eq!(back, Fold::Range(fold));
        let Fold::Range(back) = back else {
            panic!("a range's bits decode to a range")
        };
        assert_eq!(back.stride(), 4);
        assert_eq!(back.monoid(), Monoid::MAX);
        assert_eq!(back.binder(), b);
        assert_eq!(back.range(), 10..26);
    }

    #[test]
    fn from_bits_refuses_a_stride_that_does_not_divide_the_range() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let mut bits = Fold::new(Monoid::SUM, b, 0..5).to_bits();
        // Corrupt the stride field (bits 80..112) to 2, which does not divide
        // `5 - 0`: a fold that claims this would silently drop or duplicate
        // an index, so `from_bits` must refuse it rather than build one.
        bits = (bits & !(0xffff_ffffu128 << 80)) | (2u128 << 80);
        assert_eq!(Fold::from_bits(bits), None);
    }

    /// Corpus compatibility, against a literal assembled by hand from the
    /// layout every range had before the domain tag existed — tag bits
    /// zero, stride 1 at bit 80, `Add` (index 2) at 72, slot 1 at 64,
    /// `lo = 3` at 32, `hi = 11` at 0 — rather than against `to_bits`,
    /// which would only check the encoder agrees with itself.
    #[test]
    fn a_range_serialized_before_intervals_existed_decodes_unchanged() {
        let literal: u128 = 0x0000_0000_0001_0201_0000_0003_0000_000b;
        let b = Binder::from_slot(1).expect("slot 1 exists");
        let want = Fold::new(Monoid::SUM, b, 3..11);
        assert_eq!(Fold::from_bits(literal), Some(want));
        assert_eq!(want.to_bits(), literal, "and a range still writes it");
    }

    fn pixel(b: Binder) -> IntervalFold {
        IntervalFold::new(b, -0.5, 0.5)
    }

    #[test]
    fn an_interval_round_trips_through_its_bits() {
        let b = Binder::from_slot(3).expect("slot 3 exists");
        for interval in [
            pixel(b),
            IntervalFold::new(b, 1.0, 3.0),
            IntervalFold::new(b, -7.25, 1e30),
        ] {
            let fold = Fold::Interval(interval);
            assert_eq!(Fold::from_bits(fold.to_bits()), Some(fold));
        }
    }

    /// An interval's bits, assembled by hand: tag 1 at bit 112, slot 3 at
    /// 64, `lo = -0.5` (`0xbf00_0000`) at 32, `hi = 0.5` (`0x3f00_0000`) at 0,
    /// nothing in between — no monoid is stored for an integral.
    #[test]
    fn an_interval_writes_its_tag_binder_and_endpoint_bits_and_nothing_else() {
        let b = Binder::from_slot(3).expect("slot 3 exists");
        let literal: u128 = 0x0001_0000_0000_0003_bf00_0000_3f00_0000;
        assert_eq!(Fold::Interval(pixel(b)).to_bits(), literal);
        assert_eq!(Fold::from_bits(literal), Some(Fold::Interval(pixel(b))));
    }

    #[test]
    fn from_bits_refuses_what_is_not_an_interval() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let good = Fold::Interval(IntervalFold::new(b, 1.0, 3.0)).to_bits();
        let with_endpoints = |lo: f32, hi: f32| {
            (good & !0xffff_ffff_ffff_ffffu128)
                | u128::from(lo.to_bits()) << 32
                | u128::from(hi.to_bits())
        };
        let refused = [
            // An unknown domain tag.
            ("tag 2", (good & !(0xffffu128 << 112)) | 2u128 << 112),
            ("the top tag bit", good | 1u128 << 127),
            // A monoid in an interval: `Max`'s index where a range keeps its op.
            ("a non-SUM monoid", good | 11u128 << 72),
            // Even `Add`'s own index: an interval stores no monoid at all.
            ("a stored SUM", good | 2u128 << 72),
            ("a stride", good | 1u128 << 80),
            ("an out-of-space binder", good | 0xffu128 << 64),
            ("lo = inf", with_endpoints(f32::INFINITY, 3.0)),
            ("hi = NaN", with_endpoints(1.0, f32::NAN)),
            ("lo = -0.0's pattern", with_endpoints(-0.0, 1.0)),
            ("lo == hi", with_endpoints(1.0, 1.0)),
            ("lo > hi", with_endpoints(3.0, 1.0)),
            ("a length that overflows", with_endpoints(-3e38, 3e38)),
        ];
        assert!(
            Fold::from_bits(good).is_some(),
            "the unmodified bits decode"
        );
        for (what, bits) in refused {
            assert_eq!(Fold::from_bits(bits), None, "{what} must be refused");
        }
    }

    /// `-0.0` and `+0.0` are one number, so one interval, one key.
    #[test]
    fn signed_zero_endpoints_are_one_interval() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let negative = IntervalFold::new(b, -0.0, 1.0);
        let positive = IntervalFold::new(b, 0.0, 1.0);
        assert_eq!(negative, positive);
        assert_eq!(
            Fold::Interval(negative).to_bits(),
            Fold::Interval(positive).to_bits()
        );
        assert_eq!(
            IntervalFold::new(b, -1.0, -0.0),
            IntervalFold::new(b, -1.0, 0.0)
        );
    }

    /// The centred pixel, the corner cell, and the one-term range over the
    /// same binder are three different folds.
    #[test]
    fn the_pixel_is_not_the_corner_cell_or_a_one_term_range() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let centred = Fold::Interval(pixel(b));
        let corner = Fold::Interval(IntervalFold::new(b, 0.0, 1.0));
        let range = Fold::new(Monoid::SUM, b, 0..1);
        assert_ne!(centred, corner);
        assert_ne!(centred.to_bits(), corner.to_bits());
        assert_ne!(centred.to_bits(), range.to_bits());
        assert_ne!(corner.to_bits(), range.to_bits());
    }

    #[test]
    #[should_panic(expected = "lo < hi")]
    fn an_empty_interval_is_refused() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert!(!Fold::Interval(IntervalFold::new(b, 1.0, 1.0)).is_empty());
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn an_infinite_interval_is_refused() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert!(!Fold::Interval(IntervalFold::new(b, 0.0, f32::INFINITY)).is_empty());
    }

    /// What both domains answer: an integral is a `Σ` over a domain that is
    /// never empty.
    #[test]
    fn an_interval_is_a_nonempty_sum() {
        let b = Binder::from_slot(5).expect("slot 5 exists");
        let fold = Fold::Interval(IntervalFold::new(b, 1.0, 3.0));
        assert_eq!(fold.monoid(), Monoid::SUM);
        assert_eq!(fold.binder(), b);
        assert!(!fold.is_empty());
        assert!(Fold::new(Monoid::SUM, b, 4..4).is_empty());
        assert_eq!(Fold::new(Monoid::SUM, b, 4..10).evaluations(), 6);
    }

    /// The rule a legalized interval is replaced by must integrate an affine
    /// body exactly — the property that makes it a quadrature rule and not
    /// a sample. Checked in `f64` against the antiderivative, on an
    /// interval neither centred nor of unit length, where a dropped weight
    /// or a corner point would show.
    #[test]
    fn the_quadrature_rule_is_exact_on_an_affine_body() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let interval = IntervalFold::new(b, 1.0, 3.0);
        // ∫_1^3 (2u + 5) du = [u² + 5u]_1^3 = 24 - 6 = 18.
        let f = |u: f64| 2.0 * u + 5.0;
        let rule: f64 = interval
            .quadrature()
            .iter()
            .map(|node| f64::from(node.weight) * f(f64::from(node.point)))
            .sum();
        assert_eq!(rule, 18.0);
        assert_eq!(
            Fold::Interval(interval).evaluations(),
            interval.quadrature().len() as u64,
            "a price counts the samples the rule takes"
        );
    }

    #[test]
    fn a_fold_prints_its_domain() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(
            alloc::format!("{}", Fold::new(Monoid::SUM, b, 0..8)),
            "add_4[0..8)"
        );
        assert_eq!(
            alloc::format!("{}", Fold::strided(Monoid::MAX, b, 0..8, 2)),
            "max_4[0..8) step 2"
        );
        assert_eq!(
            alloc::format!("{}", Fold::Interval(pixel(b))),
            "∫_4[-0.5, 0.5)"
        );
    }

    /// `ExprNode`'s crate-wide budget (see the static assertion in
    /// `arena.rs`) is a ceiling every variant shares, not a per-variant
    /// promise — it grew from 16 to 24 when `Guard` arrived with two
    /// `KernelKey`s, and a `Fold` fits a `Reduce` node in that same 24.
    /// Pinned here as a byte count rather than left to the crate-wide
    /// assertion alone, so a future field that also fits the crate-wide
    /// check but pushes `Fold` itself past what a `Reduce` node ought to
    /// need fails here with a number, not just "too big".
    ///
    /// The enum costs nothing over the range `Fold` used to be: an
    /// [`IntervalFold`] (a binder and two `f32` bit patterns) fits beside a
    /// [`RangeFold`]'s monoid byte, and rustc keeps the discriminant in that
    /// byte's unused `OpKind` values.
    #[test]
    fn fold_and_expr_node_stay_within_the_node_budget() {
        assert_eq!(
            core::mem::size_of::<RangeFold>(),
            16,
            "RangeFold: monoid(1) + binder(1) + pad(2) + lo(4) + hi(4) + stride(4)"
        );
        assert_eq!(
            core::mem::size_of::<IntervalFold>(),
            12,
            "IntervalFold: binder(1) + pad(3) + lo(4) + hi(4)"
        );
        assert_eq!(
            core::mem::size_of::<Fold>(),
            16,
            "Fold: the larger domain, its discriminant in RangeFold's monoid niche"
        );
        assert!(
            core::mem::size_of::<Fold>() + core::mem::size_of::<crate::arena::ExprId>() <= 24,
            "a Reduce node is a Fold plus one ExprId, and fits the width a \
             Guard already needs — a fact about Reduce, not a budget Fold's \
             fields were chosen to meet"
        );
        assert!(
            core::mem::size_of::<crate::arena::ExprNode>() <= 32,
            "see arena.rs's assertion: a tripwire against an accident, not a \
             width anything depends on"
        );
    }
}
