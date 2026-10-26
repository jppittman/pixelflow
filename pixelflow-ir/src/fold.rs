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
//! it. Nothing folds a [`Fold`], nothing can name it as a number, and a
//! `Reduce`'s children are its body and any bound that is a value: a known
//! bound is the node's data ([`Bound::Lit`]), never a child.
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
use crate::kind::{OpCode, OpKind};
use crate::lit::Lit;
use crate::variance::Variance;

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
    /// (docs/archive/plans/2026-09-16-collapse-is-a-fold.md §2.4). A fold over it
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

    /// What an empty domain folds to: a real for the arithmetic monoids, a
    /// mask for the quantifiers.
    #[must_use]
    pub fn identity(self) -> Lit {
        self.op()
            .monoid_identity()
            .expect("a Monoid's operator has an identity")
    }

    /// Encode for transmission: its combining operation's [`OpCode`]. As
    /// for [`OpKind::marshal`], the round trip is what is promised, not the
    /// bytes. A fold header's bits hold their monoid as this code
    /// ([`FoldHeader::to_bits`]); crate-private, so it hands no consumer an
    /// opcode to reason with.
    #[must_use]
    pub(crate) fn marshal(self) -> OpCode {
        self.op().marshal()
    }

    /// Decode. `None` if the code names no op, or an op that generates no
    /// algebra.
    #[must_use]
    pub(crate) fn unmarshal(code: OpCode) -> Option<Self> {
        OpKind::unmarshal(code).and_then(Self::of)
    }

    /// Whether a fold under this algebra may take a bound that is a value:
    /// one whose sort has an answer for a bound outside the domain, NaN
    /// (docs/language/2026-10-20-a-bound-is-a-value.md §1.4). `ANY` and `ALL`
    /// fold masks, which have none (decision 8, open); `SEQ` is the lattice's,
    /// whose ranges stay constant (§4, "Deferred").
    pub(crate) const fn takes_a_value_bound(self) -> bool {
        matches!(
            self.0,
            OpKind::Add | OpKind::Mul | OpKind::Min | OpKind::Max
        )
    }
}

/// `⊕` of distinct terms under a [`Monoid`], built as the terms arrive:
/// `((t₀ ⊕ t₁) ⊕ t₂) ⊕ …`, left to right — the first term alone, no identity
/// in front of it — and the monoid's identity when no term arrives.
///
/// The one shape a fold of *distinct* terms has. Not [`Kernel::over`]'s,
/// which folds one body over an index, and not a halved fold's, which
/// pairs its terms (`KnownRange::halve`): those are one body `N`
/// times. [`Kernel::fold`] builds through it.
///
/// Generic over how a node is named, and handed each node to build as a
/// callback, so the builder builds its terms between steps. A callback is
/// handed the monoid's own operation to build with; no caller names one.
///
/// [`Kernel::over`]: crate::Kernel::over
/// [`Kernel::fold`]: crate::Kernel::fold
#[derive(Clone, Copy, Debug)]
pub(crate) struct Chain<R> {
    monoid: Monoid,
    /// The terms so far, combined; `None` before the first.
    folded: Option<R>,
}

impl<R> Chain<R> {
    /// No terms yet, under `monoid`.
    #[must_use]
    pub(crate) fn new(monoid: Monoid) -> Self {
        Self {
            monoid,
            folded: None,
        }
    }

    /// Fold in `term`. The first term is the fold so far; after it,
    /// `combine` builds `folded ⊕ term` from the monoid's operation and the
    /// two operands, in that order.
    pub(crate) fn push(&mut self, term: R, combine: impl FnOnce(OpKind, R, R) -> R) {
        self.folded = Some(match self.folded.take() {
            None => term,
            Some(folded) => combine(self.monoid.op(), folded, term),
        });
    }

    /// The fold of every term pushed, or — when none was — what `identity`
    /// builds from the monoid's identity.
    #[must_use]
    pub(crate) fn finish(self, identity: impl FnOnce(Lit) -> R) -> R {
        let monoid = self.monoid;
        self.folded.unwrap_or_else(|| identity(monoid.identity()))
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
    /// `None` if that index is a lattice coordinate, a retired axis, a
    /// [`Placeholder`] (a fold index before its fold is closed), or past the
    /// index space.
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

/// The first `Var` index a [`Placeholder`] takes: past the whole index space
/// a [`Binder`] can name ([`Variance::VARIABLES`]), so renaming a placeholder
/// can never reach a binder an inner fold has already chosen.
///
/// [`Variance::VARIABLES`]: crate::Variance::VARIABLES
pub(crate) const PLACEHOLDER_BASE: u8 = crate::variance::Variance::VARIABLES;

/// A fold's index while its body is built, before its [`Binder`] is chosen.
///
/// The binder is chosen after the body exists — the lowest slot no fold in
/// the body binds — so the body is built against a placeholder `Var` and
/// the placeholder renamed once the slot is known
/// ([`ExprArena::close_over`], the one place that is done). A placeholder
/// must be unique among the binders being built *at once*: a nested fold
/// renames every occurrence of its own, so if it shared one with the fold
/// enclosing it, it would capture the outer index — `Σ_i Σ_j f(i, j)` would
/// become `Σ_i Σ_j f(j, j)`. How one is kept unique is the builder's: a
/// `Kernel` claims one from a set shared across threads, `kernel!`'s
/// lowering takes one per fold it has open.
///
/// A type rather than a `Var` index so that closing over anything else — an
/// `X`, which would bind the coordinate — is unrepresentable.
///
/// [`ExprArena::close_over`]: crate::ExprArena::close_over
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Placeholder(u8);

impl Placeholder {
    /// How many binders may be under construction at once, across every
    /// builder: the placeholders past the binders' index space a `u64` set can
    /// track, which is more than the [`Binder::COUNT`] a program can nest.
    pub const COUNT: usize = u64::BITS as usize;

    /// The `n`th placeholder, or `None` past [`Placeholder::COUNT`].
    #[must_use]
    pub fn nth(n: usize) -> Option<Self> {
        if n >= Self::COUNT {
            return None;
        }
        PLACEHOLDER_BASE
            .checked_add(u8::try_from(n).ok()?)
            .map(Self)
    }

    /// The `Var` index a body reads this placeholder through.
    #[must_use]
    pub fn var(self) -> u8 {
        self.0
    }
}

// A program nests at most `Binder::COUNT` folds, so a builder that takes one
// placeholder per fold it has open never runs out before the index space
// does.
const _: () = assert!(Placeholder::COUNT >= Binder::COUNT);

/// A fold's bound: an integer known when the node is built, or the value of
/// a child term.
///
/// One type for every representation that holds a fold, parametrized by how
/// that representation names a child: `ExprId` in [`ExprNode`], `R` in
/// [`Shape`], `EClassId` in the e-graph, `ValueId` in the schedule, a rule
/// plan's `HeadRef` — and `()` in a stored [`FoldHeader`], where a value
/// bound's child is the node's next edge after the body rather than a field.
///
/// A known bound is node data, as a shift's count is: no `Const` is interned
/// for it, so no literal elsewhere in the kernel can share its id or its
/// class (docs/language/2026-10-20-a-bound-is-a-value.md, L1 and §3.1).
///
/// [`ExprNode`]: crate::arena::ExprNode
/// [`Shape`]: crate::term::Shape
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Bound<R> {
    /// Known: an integer in `0..=Fold::EXACT_BOUND`, held at the control
    /// plane's width. `EXACT_BOUND` is the domain, a data-plane cap from the
    /// `f32` counter; it is not this field's width.
    Lit(u64),
    /// A term: the child that computes the bound.
    Value(R),
}

impl<R> Bound<R> {
    /// The same bound with its child renamed: how a copier carries a bound
    /// from one representation's names to another's.
    #[must_use]
    pub fn map<S>(self, f: impl FnOnce(R) -> S) -> Bound<S> {
        match self {
            Self::Lit(n) => Bound::Lit(n),
            Self::Value(r) => Bound::Value(f(r)),
        }
    }

    /// The literal, if this bound is one.
    #[must_use]
    pub fn lit(self) -> Option<u64> {
        match self {
            Self::Lit(n) => Some(n),
            Self::Value(_) => None,
        }
    }

    /// The child, if this bound is a value.
    #[must_use]
    pub fn value(self) -> Option<R> {
        match self {
            Self::Lit(_) => None,
            Self::Value(r) => Some(r),
        }
    }

    /// This bound with its child forgotten: what a header stores.
    #[must_use]
    pub fn erased(self) -> Bound<()> {
        self.map(|_| ())
    }
}

impl<R: Copy> Bound<R> {
    /// `lo` and `hi` with their constness discovered (plan §3.1: "constness is
    /// discovered, never declared"): a value bound whose term is a literal
    /// naming a count (`Fold::count`) is that count, and every other bound is
    /// returned as it was — a term that is not a literal, or a real naming no
    /// count, which stays a value for its guard to make NaN (L5).
    ///
    /// A pair this read makes known runs forwards: `[lo, hi)` with `lo > hi` is
    /// the empty range at `lo` (L2), as Rust's `5..3` is empty. A pair passed
    /// as two literals is returned untouched and `literal` is not called, so a
    /// reversed one stays `KnownRange::new`'s to refuse.
    ///
    /// `literal` reads a value bound's term: its literal, or `None` when the
    /// term is not one. The one reader of constness for every representation
    /// that builds a fold — the arena's `push_reduce` and `close_over` read a
    /// `Const` node, the e-graph's `insert` a `Shape::Const`, `KnownBound` a
    /// class's members, the extractor a class's chosen form.
    ///
    /// # Errors
    ///
    /// [`BoundRefusal::BoundSort`] when a value bound's term is a mask literal:
    /// a mask names no count.
    pub fn discover(
        lo: Self,
        hi: Self,
        literal: impl Fn(R) -> Option<Lit>,
    ) -> Result<(Self, Self), BoundRefusal> {
        if let (Self::Lit(_), Self::Lit(_)) = (lo, hi) {
            return Ok((lo, hi));
        }
        let lo = lo.read(BoundEnd::Lo, &literal)?;
        let hi = hi.read(BoundEnd::Hi, &literal)?;
        // A value was read, so if the pair is known now, this read made it so.
        Ok(match (lo, hi) {
            (Self::Lit(l), Self::Lit(h)) if l > h => (lo, Self::Lit(l)),
            pair => pair,
        })
    }

    /// This bound with its term read: a value whose term is a literal naming a
    /// count is that count.
    fn read(
        self,
        end: BoundEnd,
        literal: &impl Fn(R) -> Option<Lit>,
    ) -> Result<Self, BoundRefusal> {
        let Self::Value(term) = self else {
            return Ok(self);
        };
        match literal(term) {
            Some(Lit::Mask(_)) => Err(BoundRefusal::BoundSort { end }),
            Some(real) => Ok(Fold::count(real).map_or(self, Self::Lit)),
            None => Ok(self),
        }
    }
}

/// Which end of a fold's range a bound is: what a refusal names, so the
/// message says which of the two the caller wrote wrong.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoundEnd {
    /// The first bound: the index the fold starts at.
    Lo,
    /// The last bound: the index the fold stops before.
    Hi,
}

impl core::fmt::Display for BoundEnd {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Lo => "lo",
            Self::Hi => "hi",
        })
    }
}

/// Why a fold refuses a bound: each the law of
/// docs/language/2026-10-20-a-bound-is-a-value.md it breaks, and the end it
/// breaks it at.
///
/// Every constructor that refuses a bound refuses with one of these, and
/// panics with its `Display`: a refusal is a builder's error, and the message
/// names the law. Where each is refused is the plan's §1.4 and L3, and this
/// type's variants say no more than which.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BoundRefusal {
    /// A real literal that names no count — not an integer, or outside
    /// `0..=Fold::EXACT_BOUND` (L5). Refused at the user surfaces only:
    /// `Kernel::over_values`, and `kernel!` in its own words, at expansion for
    /// a literal (a count the host writes is refused at the write, before it is
    /// a value); every other constructor
    /// keeps such a literal as a value bound, whose guard makes the fold NaN
    /// (`passes::legalize`'s guard), so a literal a rule made never panics.
    BoundOutOfDomain { end: BoundEnd, literal: f32 },
    /// A bound of a sort the fold cannot take (§1.4): a mask, which names no
    /// count, at either end under any fold; or any value bound under `ANY`,
    /// `ALL` or `SEQ`, whose sorts have no out-of-domain answer (decision 8) or
    /// whose ranges stay constant (the lattice's). The fold is in the panic's
    /// context, not here: the mask case has no algebra to blame.
    ///
    /// The sort is read off the term as written where a fold is built: a mask
    /// *literal* is refused there, and a computed bound that is a mask is not a
    /// literal and is admitted. Lowering refuses it by this same variant once
    /// the emitter holds it as one (`ScheduledOp::is_mask`: a comparison, or a
    /// bitwise and/or of two such), on every tier; a select between two masks
    /// is read as a float. The arena giving a bound a sort (T6) closes the
    /// rest.
    BoundSort { end: BoundEnd },
    /// A bound that varies with a coordinate or with its own fold's binder
    /// (L3): `reads` is what it varies with.
    BoundVaries { end: BoundEnd, reads: Variance },
    /// A bound that is arithmetic, at a user surface (§1.4): until bounds are
    /// `Int` (T6), a computed bound is `f32` arithmetic that can round into the
    /// domain, so a user writes a leaf.
    BoundArithmetic { end: BoundEnd },
    /// A bound that is not a number at all: a `Param` slot, which has no
    /// value, or a `Buffer`'s address, read through a gather.
    BoundNotANumber { end: BoundEnd },
}

impl core::fmt::Display for BoundRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const PLAN: &str = "docs/language/2026-10-20-a-bound-is-a-value.md";
        match *self {
            Self::BoundOutOfDomain { end, literal } => write!(
                f,
                "BoundOutOfDomain: a fold's {end} bound is {literal}, which names no count: a \
                 count is an integer in 0..={} (2^24), where an `f32` counter is exact (L5, \
                 §1.4 of {PLAN})",
                Fold::EXACT_BOUND
            ),
            Self::BoundSort { end } => write!(
                f,
                "BoundSort: a fold's {end} bound has a sort the fold cannot take: a mask names \
                 no count, and an ANY, ALL or SEQ fold takes literal bounds only (§1.4 and \
                 decision 8 of {PLAN})"
            ),
            Self::BoundVaries { end, reads } => write!(
                f,
                "BoundVaries: a fold's {end} bound varies with {reads:?}: a bound reads no \
                 coordinate and not its own fold's binder (L3 of {PLAN})"
            ),
            Self::BoundArithmetic { end } => write!(
                f,
                "BoundArithmetic: a fold's {end} bound is arithmetic: until bounds are `Int` (T6 \
                 of docs/language/2026-10-17-the-arena-is-typed.md) a bound is a leaf — a \
                 constant, a uniform or an enclosing fold's index — whose value is the count \
                 exactly (§1.4 of {PLAN})"
            ),
            Self::BoundNotANumber { end } => write!(
                f,
                "BoundNotANumber: a fold's {end} bound is an unbound slot or a table's \
                 address, not a number: a `Param` has no value and a `Buffer` is read \
                 through a gather (§1.4 of {PLAN})"
            ),
        }
    }
}

/// The fold a [`Reduce`] performs: a monoid, the index it binds, and the step
/// that index takes. Its bounds are the node's (`ExprNode::Reduce`'s `lo` and
/// `hi`), not the fold's: a bound is a term, which a known one is stored as
/// (docs/language/2026-10-20-a-bound-is-a-value.md §3.1).
///
/// `⟦Reduce { fold, lo, hi, body }⟧ = ⊕_{k ∈ lo, lo+s, lo+2s, …, hi-s} ⟦body⟧[fold.binder() := k]`
///
/// `s` is [`KnownRange::stride`]: 1 from [`Fold::new`], or whatever
/// [`Fold::strided`] was given directly — `pack` strip-mining a lattice's
/// column fold is the reason a second constructor exists at all. After
/// construction, only [`KnownRange::halve`] ever changes it, by doubling.
/// [`KnownRange::range`] still names the bounding `[lo, hi)`, which is why it
/// is a poor stand-in for "every index visited" once the stride is not 1 —
/// [`KnownRange::len`] is the count that stays exact.
///
/// **A range is read through the node that holds the bounds** —
/// [`ExprArena::known_range`](crate::arena::ExprArena::known_range), the
/// e-graph node's, or the schedule's — which share one reader,
/// [`KnownRange::of`]. Nothing else that reasons about a range reads a bound
/// (serializing and naming a fold do), so when a bound becomes a value, those
/// readers change and their callers do not.
///
/// [`Reduce`]: crate::arena::ExprNode::Reduce
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Fold {
    monoid: Monoid,
    binder: Binder,
    /// The step between visited indices: 1 from [`Fold::new`]. Not 1 only
    /// over a known range it divides, which the constructors that meet the
    /// bounds check ([`FoldHeader::new`], `KnownRange::new`); only `pack` and
    /// `HalveFold` make one. At most [`Fold::EXACT_BOUND`]: it is added to an
    /// `f32` counter.
    stride: u64,
}

impl Fold {
    /// The largest end a range may have: 2²⁴, the last integer to which an
    /// `f32` names every integer.
    ///
    /// A fold's index is an `f32` lane — the binder's `Var`, and the counter
    /// the JIT steps by adding `1.0` and compares against the end, both
    /// loaded as `f32` — so past 2²⁴ neighbouring indices round together
    /// and the fold is not the one written. Measured through `kernel!`
    /// before `kernel!` refused such a bound:
    /// `(16777100..16777300).map(|i| ((i as f32) - 16777000.0) * X).sum()`
    /// gave 39890 at `X = 1`, where rustc gives 39900; below 2²⁴ the two
    /// agree. It is refused here, beside the reversed range, so that every
    /// constructor — `Kernel::over` as much as `kernel!`'s lowering — is
    /// refused alike.
    ///
    /// The ends are `u64`, the control plane's width (CLAUDE.md). This is
    /// their domain, not their width: the lane is the narrower of the two,
    /// and no width lifts it.
    pub const EXACT_BOUND: u64 = 1 << f32::MANTISSA_DIGITS;

    /// The trips a fold is priced at when its range is not known: 32.
    ///
    /// LLVM's `BranchProbabilityInfo` weights a loop's back edge 124 against
    /// its exit's 4 (`LBH_TAKEN_WEIGHT`, `LBH_NONTAKEN_WEIGHT`), a back-edge
    /// probability of 31/32, so a loop it knows nothing about runs
    /// `1 / (1 − 31/32) = 32` times. Policy, not a measurement of these
    /// kernels: the static model prices a fold before its bound has a value
    /// (docs/language/2026-10-20-a-bound-is-a-value.md §1.5), and a ranking at
    /// 32 may differ from one at the real count (§4). Nested folds whose
    /// ranges are not known multiply, as their trips do.
    pub const ESTIMATED_TRIPS: u64 = 32;

    /// The fold of `monoid` binding `binder`, visiting every index its bounds
    /// name — [`Fold::strided`] with a stride of `1`.
    #[must_use]
    pub fn new(monoid: Monoid, binder: Binder) -> Self {
        Self::strided(monoid, binder, 1)
    }

    /// The fold of `monoid` binding `binder`, visiting every `stride`-th
    /// index from its first bound — the strip-mining `pack` uses to carve a
    /// lattice's column fold into a lane-width main fold and a narrower
    /// remainder (docs/archive/plans/2026-09-16-collapse-is-a-fold.md §2.3).
    ///
    /// # Panics
    ///
    /// Panics if `stride` is `0`, or past [`Fold::EXACT_BOUND`]. That the
    /// stride divides the bounds' span is checked where the bounds meet it:
    /// `KnownRange::new`, and [`FoldHeader::new`] through it.
    #[must_use]
    pub fn strided(monoid: Monoid, binder: Binder, stride: u64) -> Self {
        assert_stride(stride);
        Self {
            monoid,
            binder,
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
    /// `pixelflow-codegen`'s selection (`emit/select.rs`), and
    /// docs/codegen/2026-09-10-a-surviving-reduce-is-a-loop.md's "2a″" for why
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

    /// This fold over `range`: its monoid and binder, `range`'s stride, and
    /// `range`'s ends as literal bounds — the node fields a rewrite builds the
    /// fold it decomposed from ([`KnownRange::peel`], [`KnownRange::peel_back`],
    /// [`KnownRange::halve`]).
    ///
    /// Total, with nothing to check: every [`KnownRange`] already has what
    /// `KnownRange::new` asks of one (forwards, within
    /// [`Fold::EXACT_BOUND`], a stride that divides its span), because the
    /// only ones that exist were checked there, or decomposed from one that
    /// was.
    #[must_use]
    pub fn over<R>(self, range: KnownRange) -> (Self, Bound<R>, Bound<R>) {
        (
            Self {
                stride: range.stride,
                ..self
            },
            Bound::Lit(range.lo),
            Bound::Lit(range.hi),
        )
    }

    /// Whether an `f32` counter holds `n` exactly: `n ≤ EXACT_BOUND`. The one
    /// statement of the domain's ceiling — an end (`KnownRange::admits`), a
    /// literal (`Fold::count`), a stride (`assert_stride`,
    /// `FoldHeader::from_bits`) and a literal beside a value bound
    /// (`FoldHeader::new`) all ask it.
    const fn is_exact(n: u64) -> bool {
        n <= Self::EXACT_BOUND
    }

    /// The count `literal` names: a real that is a non-negative integer an
    /// `f32` counter holds exactly (`Fold::is_exact`), or `None` — a mask, a
    /// fraction, a negative, a NaN, an infinity, or an integer past 2²⁴.
    ///
    /// `-0.0` is the integer `0`, as it is to a shift count
    /// (`ShiftCount::from_literal`). Crate-private: another crate reads a count
    /// off a bound through [`Bound::discover`], which is where a count is a
    /// bound.
    #[must_use]
    pub(crate) fn count(literal: Lit) -> Option<u64> {
        let real = literal.real()?;
        // NaN and the negatives fail here; `-0.0 < 0.0` is false.
        if real.is_nan() || real < 0.0 {
            return None;
        }
        // `as` truncates a fraction and saturates past `u64::MAX`, so the round
        // trip is exact for an integer and only for one.
        let count = real as u64;
        (count as f32 == real && Self::is_exact(count)).then_some(count)
    }

    /// L3: whether a bound of this fold may vary with `reads`. It may when
    /// `reads` shares no bit with the coordinates, which would make the trip
    /// count a function of the pixel, or with this fold's own binder, which
    /// would make it a function of itself. An enclosing fold's binder is a
    /// bound's to read (a triangular `0..i`), and a placeholder names no bit
    /// ([`Variance::of_var`]).
    ///
    /// The one statement of the law, and the same at both ends of a range:
    /// [`Self::admits_bound`] names the end a refusal is for, and a reader that
    /// only chooses among terms (the e-graph's rebuild of a bound) asks this.
    #[inline]
    #[must_use]
    pub fn bound_may_read(self, reads: Variance) -> bool {
        let refused = Variance::COORDS.union(Variance::from_var(self.binder.var()));
        reads.intersection(refused).is_const()
    }

    /// L3 ([`Self::bound_may_read`]) asked of the bound at `end`: the arena
    /// asks it of a bound's term (`ExprArena::push_reduce`), the e-graph of a
    /// bound's class (`EGraph::add`).
    ///
    /// # Errors
    ///
    /// [`BoundRefusal::BoundVaries`], with what the bound varies with.
    pub fn admits_bound(self, end: BoundEnd, reads: Variance) -> Result<(), BoundRefusal> {
        match self.bound_may_read(reads) {
            true => Ok(()),
            false => Err(BoundRefusal::BoundVaries { end, reads }),
        }
    }
}

/// The step `stride` is, or the panic that says why not: nonzero, and at most
/// [`Fold::EXACT_BOUND`] because it is added to an `f32` counter. Shared by
/// the two constructors that meet a stride, [`Fold::strided`] and
/// `KnownRange::new`.
fn assert_stride(stride: u64) {
    assert!(stride != 0, "a fold's stride must be nonzero");
    assert!(
        Fold::is_exact(stride),
        "a fold's stride is at most {} (2^24): {stride} — it is added to an `f32` index",
        Fold::EXACT_BOUND
    );
}

/// How a [`FoldHeader`] holds one bound.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum Held {
    /// The header's count field is the bound.
    Lit,
    /// The bound is a child of the node, after the body; the header's count
    /// field is 0.
    Value,
}

impl Held {
    /// A bound as the pair a header stores: how it is held, and its count —
    /// 0 for a value, whose bound is not a number, so that two headers that
    /// hold the same bounds are equal in every field `Eq`, `Hash` and `Ord`
    /// read. The one writer of either.
    fn hold(bound: Bound<()>) -> (Self, u64) {
        match bound {
            Bound::Lit(n) => (Self::Lit, n),
            Bound::Value(()) => (Self::Value, 0),
        }
    }

    /// The bound a `count` held this way is.
    fn bound(self, count: u64) -> Bound<()> {
        match self {
            Self::Lit => Bound::Lit(count),
            Self::Value => Bound::Value(()),
        }
    }
}

/// What a `Reduce` node holds besides its children: its [`Fold`], and each
/// bound as the literal it is or as the mark of a child that follows the
/// body (body first, then `lo`'s child, then `hi`'s).
///
/// A fold's identity: what the arena interns on, what the e-graph
/// hash-conses on, and — as [`FoldHeader::to_bits`] — what the canonical
/// key, `kernel!`'s tokens, the dumps and the corpora write (plan L9). Only
/// checked ones exist: [`FoldHeader::new`] holds the contract
/// `KnownRange::new` holds, and [`FoldHeader::from_bits`] refuses bits that
/// break it.
///
/// **Stored packed, read as a `Fold` and two `Bound<()>`s.** The fields are
/// private: the three counts, then the monoid, the binder and how each bound
/// is held, one byte apiece. A node enum is as wide as its widest variant, so
/// what a `Reduce` stores, every node pays.
///
/// **The counts are stored in 32 bits**, against the control plane's 64-bit
/// rule, for a measured reason (2026-10-22). The readers return 64 bits; only
/// the storage is narrow. Stored at 64 bits (A2b of
/// docs/language/2026-10-20-a-bound-is-a-value.md as it landed), the header
/// was 32 bytes, `ENode` 48 and `NodeData`/`ExprData` 32; at 32 bits it is 16,
/// and they are 32 and 16. Measured on a native release build, the cold
/// compile of the 29 programs ASCII needs, 12 rounds of the three binaries in
/// rotating order, each run its own process, peak RSS read from the parent
/// (`ru_maxrss`, KiB, shown here divided by 1000). The harness is out of tree.
///
/// - Peak RSS, medians: 90.0 MB before A2b, 104.8 MB with 64-bit storage
///   (+16.4%), 95.2 MB with 32-bit storage (+5.8%): narrowing recovers about
///   65% of the growth.
/// - Instructions (callgrind, deterministic), the same compile: 5,155 M before
///   A2b, 5,424 M at 64 bits (+5.2%), 5,355 M at 32 bits (+3.9%).
///
/// The width does not bind. Every count a header holds is at most
/// [`Fold::EXACT_BOUND`] (2²⁴, the `f32` counter's exact range), which the
/// constructors check, and [`FoldHeader::to_bits`] already writes each count
/// in a 32-bit field of the key. Lifting `EXACT_BOUND` (T6 of
/// docs/language/2026-10-17-the-arena-is-typed.md, once the counter is an
/// integer) is a key-format change that widens these fields and that storage
/// together; the const assert beside `to_bits` is where a build says so.
/// [`FoldHeader::fold`], [`FoldHeader::lo`] and [`FoldHeader::hi`] are the
/// readers. A bound that is a value keeps its count at 0, so that equal bounds
/// are equal in every field `Eq`, `Hash` and `Ord` read, and writes the all-ones
/// marker in its bits. The fields are declared in the order `Fold` derived `Ord`
/// while it held the ends — monoid, binder, `lo`, `hi`, stride — so the derived
/// order on a literal fold is the old one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FoldHeader {
    monoid: Monoid,
    binder: Binder,
    lo: u32,
    hi: u32,
    stride: u32,
    lo_held: Held,
    hi_held: Held,
}

impl FoldHeader {
    /// `fold` between `lo` and `hi`, with what the fold asks of its bounds
    /// checked: a literal pair by `KnownRange::new` (order, domain, stride), a
    /// value bound against the monoid, the stride, and any literal beside it.
    ///
    /// # Panics
    ///
    /// On a literal pair `KnownRange::new` refuses (backwards, ending past
    /// [`Fold::EXACT_BOUND`], a stride that does not divide the span), with its
    /// messages. On a value bound the fold cannot take: under `ANY`, `ALL` or
    /// `SEQ` ([`BoundRefusal::BoundSort`]); at a stride other than 1, which
    /// only a known range it divides may have; or beside a literal past
    /// [`Fold::EXACT_BOUND`].
    #[must_use]
    pub fn new(fold: Fold, lo: Bound<()>, hi: Bound<()>) -> Self {
        let (Bound::Lit(l), Bound::Lit(h)) = (lo, hi) else {
            return Self::with_a_value(fold, lo, hi);
        };
        Self::known(fold, KnownRange::new(l..h, fold.stride))
    }

    /// `fold` between `lo` and `hi`, at least one of them a value: what a value
    /// bound asks of its fold, checked, and the header stored.
    fn with_a_value(fold: Fold, lo: Bound<()>, hi: Bound<()>) -> Self {
        let end = match lo {
            Bound::Value(()) => BoundEnd::Lo,
            Bound::Lit(_) => BoundEnd::Hi,
        };
        assert!(
            fold.monoid.takes_a_value_bound(),
            "{}: {fold:?}",
            BoundRefusal::BoundSort { end }
        );
        assert!(
            fold.stride == 1,
            "a fold's stride is not 1 only over a known range it divides: {fold:?} from {lo:?} \
             to {hi:?}"
        );
        for literal in [lo.lit(), hi.lit()].into_iter().flatten() {
            assert!(
                Fold::is_exact(literal),
                "a fold's literal bound is at most {} (2^24): {literal} — its index is an \
                 `f32` lane, which names every integer only that far",
                Fold::EXACT_BOUND
            );
        }
        Self::store(fold, lo, hi)
    }

    /// `fold` over `range`, taking `range`'s stride. Total.
    #[must_use]
    pub(crate) fn known(fold: Fold, range: KnownRange) -> Self {
        let (fold, lo, hi) = fold.over(range);
        Self::store(fold, lo, hi)
    }

    /// The one writer of a header's fields: it checks nothing, and its callers
    /// ([`FoldHeader::known`], `FoldHeader::new`'s value arm and
    /// [`FoldHeader::from_bits`]'s) have.
    fn store(fold: Fold, lo: Bound<()>, hi: Bound<()>) -> Self {
        let ((lo_held, lo), (hi_held, hi)) = (Held::hold(lo), Held::hold(hi));
        Self {
            monoid: fold.monoid,
            binder: fold.binder,
            lo: field(lo),
            hi: field(hi),
            stride: field(fold.stride),
            lo_held,
            hi_held,
        }
    }

    /// The fold: its monoid, its binder and its stride.
    #[must_use]
    pub fn fold(self) -> Fold {
        Fold {
            monoid: self.monoid,
            binder: self.binder,
            stride: u64::from(self.stride),
        }
    }

    /// The first bound, as the literal it is or the mark of a child.
    #[must_use]
    pub fn lo(self) -> Bound<()> {
        self.lo_held.bound(u64::from(self.lo))
    }

    /// The last bound, as the literal it is or the mark of a child.
    #[must_use]
    pub fn hi(self) -> Bound<()> {
        self.hi_held.bound(u64::from(self.hi))
    }

    /// The range, when both bounds are literals: [`KnownRange::of`].
    #[must_use]
    pub fn known_range(self) -> Option<KnownRange> {
        KnownRange::of(self.fold(), self.lo(), self.hi())
    }

    /// The trips this fold is priced at by a reader holding only its header:
    /// its known range's length, or [`Fold::ESTIMATED_TRIPS`] when a bound is a
    /// value. The extractor prices a fold whose value bounds were chosen as
    /// literals through the fold over those literals, whose header this is
    /// (`pixelflow-search`'s `ENode::literal_twin`).
    #[must_use]
    pub fn priced_trips(self) -> u64 {
        self.known_range()
            .map_or(Fold::ESTIMATED_TRIPS, KnownRange::len)
    }

    /// How many children a node with this header has: its body, and one per
    /// value bound. What the arena's edge decode and `ExprData`'s builder
    /// check a node's edges against.
    #[must_use]
    pub fn arity(self) -> usize {
        1 + usize::from(self.lo().value().is_some()) + usize::from(self.hi().value().is_some())
    }

    /// The bounds with their children: `values` are the node's children
    /// after its body, in order (`lo`'s child before `hi`'s).
    ///
    /// # Panics
    ///
    /// Unless `values` holds exactly `arity() - 1` items: a node whose edges
    /// disagree with its header ("a fold header with {n} value bounds was
    /// given {m} children").
    #[must_use]
    pub fn attach<R>(self, values: impl IntoIterator<Item = R>) -> (Bound<R>, Bound<R>) {
        let wanted = self.arity() - 1;
        let mut values = values.into_iter();
        let mut given = 0;
        let mut take = |bound: Bound<()>| match bound {
            Bound::Lit(n) => Bound::Lit(n),
            Bound::Value(()) => {
                let child = values.next().unwrap_or_else(|| {
                    panic!("a fold header with {wanted} value bounds was given {given} children")
                });
                given += 1;
                Bound::Value(child)
            }
        };
        let (lo, hi) = (take(self.lo()), take(self.hi()));
        let extra = values.count();
        assert!(
            extra == 0,
            "a fold header with {wanted} value bounds was given {} children",
            wanted + extra
        );
        (lo, hi)
    }

    /// The monoid's byte in [`FoldHeader::to_bits`]: its code
    /// (`Monoid::marshal`), the one encoding of a monoid.
    fn monoid_byte(self) -> u8 {
        let [byte] = self.monoid.marshal().to_bytes();
        byte
    }

    /// This header as opaque bits, and [`FoldHeader::from_bits`] back.
    ///
    /// Three callers need exactly this and nothing else: the runtime tier's
    /// JIT-cache key, [`canonical`](crate::key::canonical)'s content digest,
    /// and the `kernel!` macro, which emits an arena as tokens that rebuild
    /// it at load time. Each of them is *serializing* a fold rather than
    /// reasoning about one, so this is what they get — not an accessor for
    /// the combining opcode: the op set is an IR concept, a consumer names
    /// algebras, and `Monoid::op` is crate-private. The opcode itself is
    /// reachable — these bits decode, the monoid's code in them round-trips
    /// through `OpKind::unmarshal`, and a backend emitting a surviving loop
    /// reads it through [`Fold::combine_op`] — so what is withheld is
    /// an accessor offered to anything that reasons about a fold.
    ///
    /// The layout is `stride << 80 | op << 72 | binder << 64 | lo << 32 |
    /// hi`, and nothing is written at or above bit `UNUSED_BITS` (112). The
    /// counts are 32-bit fields because [`Fold::EXACT_BOUND`] fits them:
    /// lifting it is a key-format change, which widens these fields and
    /// re-keys everything that carries them.
    ///
    /// Total. A bound that is a value writes the all-ones count field
    /// (`VALUE_FIELD`), which no literal reaches. The child it marks is not in
    /// the bits: it is the node's next edge after the body, for whoever
    /// serializes the node's children to write ([`FoldHeader::arity`] says how
    /// many follow).
    #[must_use]
    pub fn to_bits(self) -> u128 {
        u128::from(self.stride) << 80
            | u128::from(self.monoid_byte()) << 72
            | u128::from(self.binder.slot()) << 64
            | u128::from(count_field(self.lo())) << 32
            | u128::from(count_field(self.hi()))
    }

    /// The header [`FoldHeader::to_bits`] wrote, or `None` if the bits do not
    /// name one: anything at or above bit `UNUSED_BITS` (112), an op that
    /// generates no algebra, an index outside the binder space, a zero stride
    /// or one past [`Fold::EXACT_BOUND`], a literal range [`KnownRange::admits`]
    /// does not (reversed, or ending past the bound) or that the stride does
    /// not divide exactly, or a value bound (`VALUE_FIELD`, a count field of
    /// all ones) its fold cannot take: under `ANY`, `ALL` or `SEQ`, at a stride
    /// other than 1, or beside a literal past the bound.
    ///
    /// Total, and the exact inverse of [`FoldHeader::to_bits`]: a corrupt
    /// cache key or a hand-written token stream is a `None` at the boundary
    /// rather than a fold that means something else. A literal never decodes
    /// as a value (no count is all ones) and a value never as a literal.
    #[must_use]
    pub fn from_bits(bits: u128) -> Option<Self> {
        if bits >> UNUSED_BITS != 0 {
            return None;
        }
        let stride = u64::from(((bits >> 80) & 0xffff_ffff) as u32);
        let monoid = Monoid::unmarshal(OpCode::from_bytes([((bits >> 72) & 0xff) as u8]))?;
        let binder = Binder::from_slot(((bits >> 64) & 0xff) as u8)?;
        let lo = field_bound(((bits >> 32) & 0xffff_ffff) as u32);
        let hi = field_bound((bits & 0xffff_ffff) as u32);
        if stride == 0 || !Fold::is_exact(stride) {
            return None;
        }
        let fold = Fold::strided(monoid, binder, stride);
        // A value bound settles the header by what the fold asks of one; the
        // rest is a literal range.
        let (Bound::Lit(first), Bound::Lit(end)) = (lo, hi) else {
            return admits_a_value(fold, lo, hi).then(|| Self::store(fold, lo, hi));
        };
        (KnownRange::admits(&(first..end)) && (end - first).is_multiple_of(stride)).then(|| {
            let range = KnownRange {
                lo: first,
                hi: end,
                stride,
            };
            Self::known(fold, range)
        })
    }
}

/// A count's 32-bit field, in a stored [`FoldHeader`] and in
/// [`FoldHeader::to_bits`]. Every count a header holds is at most
/// [`Fold::EXACT_BOUND`], which the constructors check.
fn field(count: u64) -> u32 {
    u32::try_from(count)
        .expect("a fold's counts are at most Fold::EXACT_BOUND, which fits its 32-bit field")
}

/// A count field's bits when its bound is a value: all ones, which no count
/// reaches (the assert below), so a literal and a value never share a field.
/// The value's child is the node's next edge after the body, written by whoever
/// writes the node's children.
const VALUE_FIELD: u32 = u32::MAX;

/// A bound's count field in [`FoldHeader::to_bits`]: its count, or
/// `VALUE_FIELD`.
fn count_field(bound: Bound<()>) -> u32 {
    match bound {
        Bound::Lit(count) => field(count),
        Bound::Value(()) => VALUE_FIELD,
    }
}

/// The bound a count field of [`FoldHeader::to_bits`] names: the inverse of
/// `count_field`. A field of `VALUE_FIELD` is a value; any other is a literal,
/// whose range [`FoldHeader::from_bits`] checks.
fn field_bound(field: u32) -> Bound<()> {
    match field {
        VALUE_FIELD => Bound::Value(()),
        count => Bound::Lit(u64::from(count)),
    }
}

/// What a bound that is a value asks of its fold, as one predicate: its
/// monoid takes one (`Monoid::takes_a_value_bound`), its stride is 1, and a
/// literal beside it is one an `f32` counter holds ([`Fold::is_exact`]).
/// [`FoldHeader::from_bits`] refuses where it fails, and `with_a_value` panics
/// there naming which part; `new_panics_exactly_where_the_predicate_refuses`
/// holds the two equal.
fn admits_a_value(fold: Fold, lo: Bound<()>, hi: Bound<()>) -> bool {
    fold.monoid.takes_a_value_bound()
        && fold.stride == 1
        && [lo.lit(), hi.lit()]
            .into_iter()
            .flatten()
            .all(Fold::is_exact)
}

/// Where [`FoldHeader::to_bits`]'s layout ends: every bit from here up is
/// zero, and [`FoldHeader::from_bits`] refuses bits that say otherwise.
const UNUSED_BITS: u32 = 112;

// The 32-bit fields hold every count a header may, with the all-ones field
// left over: that is the value marker (`VALUE_FIELD`), which no count reaches.
// They are also what a header stores (`FoldHeader::store` narrows through
// `field`), so a count that does not fit is impossible while this holds.
// Lifting `EXACT_BOUND` to the marker is the key-format change `to_bits`
// documents, and this is where a build says so.
const _: () = assert!(Fold::EXACT_BOUND < VALUE_FIELD as u64);

/// One header printed the way every reader of an arena spells it: its
/// combining op, binder and bounds (`add_4[0..8)`, with ` step 2` once the
/// stride is not 1). A bound that is a value prints `_` in its place
/// (`add_4[3.._)`): its child is not part of the header, and the node's
/// display prints it after the body.
impl core::fmt::Display for FoldHeader {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}_{}[",
            self.fold().combine_op().name(),
            self.binder.var()
        )?;
        write_bound(f, self.lo())?;
        f.write_str("..")?;
        write_bound(f, self.hi())?;
        f.write_str(")")?;
        // The step is worth stating once it is not 1 — the shape
        // `KnownRange::halve` leaves behind — since the bounds alone would then
        // read as "every index" and aren't.
        if self.stride != 1 {
            write!(f, " step {}", self.stride)?;
        }
        Ok(())
    }
}

/// A bound as [`FoldHeader`]'s display spells it: the literal's count, or `_`
/// for the child that holds it.
fn write_bound(f: &mut core::fmt::Formatter<'_>, bound: Bound<()>) -> core::fmt::Result {
    match bound {
        Bound::Lit(count) => write!(f, "{count}"),
        Bound::Value(()) => f.write_str("_"),
    }
}

/// The message of an `expect` on a range a site's folds always have. A site
/// that reads a range and has no answer of its own for an unknown one reads it
/// where every fold is literal: a lattice's fold (`SEQ`, which takes literal
/// bounds only), a production glyph's before B2, or a test's own literal fold.
/// One grep target for the steps that make the `None` reachable there.
pub const KNOWN_RANGE: &str = "this fold's bounds are literals: a lattice's fold, a production \
     glyph's before B2, or a test's (docs/language/2026-10-20-a-bound-is-a-value.md)";

/// A fold's range when it is known: the indices `lo, lo + stride, …` below
/// `hi`. The view every reader of a range reads.
///
/// ```text
/// ⊕_{k ∈ lo, lo+s, lo+2s, …, hi-s} f(k)
/// ```
///
/// Forwards (`lo <= hi`), ending at or before [`Fold::EXACT_BOUND`], with a
/// stride that divides `hi - lo` exactly, so [`KnownRange::len`] is integer
/// division. The fields are `u64`, the control plane's width; their domain is
/// `0..=Fold::EXACT_BOUND`. Those are the invariants `KnownRange::new`
/// checks, and the only ways to hold one are that checked constructor, the
/// reader ([`KnownRange::of`]) and the decompositions
/// ([`KnownRange::peel`], [`KnownRange::peel_back`], [`KnownRange::halve`]),
/// each of which preserves them. There is no reversed inhabitant to
/// normalize.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KnownRange {
    lo: u64,
    hi: u64,
    stride: u64,
}

impl KnownRange {
    /// The range `range` visits every `stride`-th index of.
    ///
    /// # Panics
    ///
    /// Panics unless [`KnownRange::admits`] `range`, and unless `stride` is
    /// nonzero, at most [`Fold::EXACT_BOUND`], and divides `range`'s span
    /// exactly — the invariant [`KnownRange::len`] and
    /// [`FoldHeader::from_bits`] both rely on.
    #[must_use]
    pub(crate) fn new(range: Range<u64>, stride: u64) -> Self {
        assert!(
            range.start <= range.end,
            "a fold's range runs forwards: {range:?}"
        );
        assert!(
            Self::admits(&range),
            "a fold's range ends at most at {} (2^24): {range:?} — its index is an `f32` \
             lane, which names every integer only that far",
            Fold::EXACT_BOUND
        );
        assert_stride(stride);
        assert!(
            (range.end - range.start).is_multiple_of(stride),
            "a fold's stride must divide its span exactly: {range:?} step {stride}"
        );
        Self {
            lo: range.start,
            hi: range.end,
            stride,
        }
    }

    /// The range of `fold` between `lo` and `hi`, when both are literals —
    /// `None` when either is a value. The one reader of a fold's range: the
    /// arena's, the e-graph node's and the schedule's all bottom out here, so
    /// when a bound becomes a value, this is where the answer is decided.
    ///
    /// # Panics
    ///
    /// On a literal pair `KnownRange::new` refuses. A pair a node's
    /// constructor admitted cannot be one; only a hand-made pair is.
    #[must_use]
    pub fn of<R>(fold: Fold, lo: Bound<R>, hi: Bound<R>) -> Option<Self> {
        Some(Self::new(lo.lit()?..hi.lit()?, fold.stride))
    }

    /// Whether `range` is a domain a fold may range over: forwards, and
    /// ending at or before [`Fold::EXACT_BOUND`], so that every index
    /// it names, and its end, is an exact `f32`.
    ///
    /// The one statement of the contract: the constructors panic where it
    /// fails, [`FoldHeader::from_bits`] refuses where it fails, and a front end
    /// asks it to refuse a range with its own words before building one.
    #[must_use]
    pub fn admits(range: &Range<u64>) -> bool {
        range.start <= range.end && Fold::is_exact(range.end)
    }

    /// The half-open range every visited index lies within.
    ///
    /// The *bound*, not the visited set: once [`KnownRange::halve`] has run,
    /// the indices in `[lo, hi)` that are actually visited are `lo`,
    /// `lo + stride`, `lo + 2·stride`, … — [`KnownRange::stride`] apart, not
    /// consecutive. [`KnownRange::len`] is the count that stays exact either
    /// way.
    #[must_use]
    pub fn range(self) -> Range<u64> {
        self.lo..self.hi
    }

    /// The step between one visited index and the next. `1` from
    /// [`Fold::new`], or [`Fold::strided`]'s own argument;
    /// [`KnownRange::halve`] is the only thing that doubles it afterward.
    #[must_use]
    pub fn stride(self) -> u64 {
        self.stride
    }

    /// How many terms the fold combines — the *trip count*, which
    /// [`KnownRange::halve`] halves without touching [`KnownRange::range`].
    /// Distinct from the index span `hi - lo`: they coincide only at
    /// `stride == 1`, and every caller here wants the trip count (how many
    /// times the body is evaluated), never the span.
    #[must_use]
    pub fn len(self) -> u64 {
        (self.hi - self.lo) / self.stride
    }

    /// Whether the domain is empty — in which case the fold *is* its monoid's
    /// identity, whatever the body says.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.lo == self.hi
    }

    /// The first index, and the range over everything after it — or `None`
    /// when the domain is empty.
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
    pub fn peel(self) -> Option<(u64, Self)> {
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
    /// odd number of them — [`KnownRange::peel_back`] is the epilogue for the
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
    /// reaches the same fully-unrolled term [`KnownRange::peel_back`] does, in
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

    /// The range over everything *before* the last index, and that index — or
    /// `None` when the domain is empty.
    ///
    /// ```text
    /// ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi-s) step s} f ⊕ f(hi-s)
    /// ```
    ///
    /// [`KnownRange::peel`]'s mirror, and the one a rewrite should use: run to
    /// exhaustion over a `stride`-1 fold it builds the **left**-leaning chain
    /// `((f(lo) ⊕ f(lo+1)) ⊕ …)`. Peeling from the front instead yields the
    /// same value in the opposite association, and an e-graph then has to
    /// spend reassociation rules — and the classes they mint — to reach the
    /// shape everything downstream was tuned on.
    #[must_use]
    pub fn peel_back(self) -> Option<(Self, u64)> {
        (!self.is_empty()).then(|| {
            let rest = Self {
                hi: self.hi - self.stride,
                ..self
            };
            (rest, rest.hi)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The range of a stride-1 fold between `range`'s ends, with no node behind
    /// it.
    fn known(range: Range<u64>) -> KnownRange {
        KnownRange::new(range, 1)
    }

    /// `fold` between two literal bounds.
    fn header(fold: Fold, lo: u64, hi: u64) -> FoldHeader {
        FoldHeader::new(fold, Bound::Lit(lo), Bound::Lit(hi))
    }

    /// The `slot`-th binder.
    fn b(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("the slot exists")
    }

    /// The sum binding the `slot`-th binder.
    fn sum(slot: u8) -> Fold {
        Fold::new(Monoid::SUM, b(slot))
    }

    /// The monoid's byte in a header's bits: its code.
    fn monoid_byte(monoid: Monoid) -> u8 {
        let [byte] = monoid.marshal().to_bytes();
        byte
    }

    /// What a test's terms (named by a `u8`) are, for [`Bound::discover`]: a
    /// count, a second count, no literal at all, a fraction, an integer past the
    /// exact bound, a mask.
    fn term(id: u8) -> Option<Lit> {
        match id {
            0 => Some(Lit::Real(2.0)),
            1 => Some(Lit::Real(9.0)),
            3 => Some(Lit::Real(2.5)),
            4 => Some(Lit::Real(16_777_218.0)),
            5 => Some(Lit::Mask(true)),
            _ => None,
        }
    }

    /// Terms for the reversed pair: five, then three.
    fn reversed_term(id: u8) -> Option<Lit> {
        Some(Lit::Real(if id == 0 { 5.0 } else { 3.0 }))
    }

    /// A count is a real that is a non-negative integer, held exactly: `-0.0`
    /// is the integer `0`, and the exact bound itself is the last count.
    #[test]
    fn a_count_is_a_non_negative_integer_an_f32_counter_holds_exactly() {
        let count = |real| Fold::count(Lit::Real(real));
        assert_eq!(count(0.0), Some(0));
        assert_eq!(count(-0.0), Some(0));
        assert_eq!(count(7.0), Some(7));
        assert_eq!(count(16_777_216.0), Some(Fold::EXACT_BOUND));
        for no_count in [16_777_218.0, 2.5, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(count(no_count), None, "{no_count} names no count");
        }
        assert_eq!(Fold::count(Lit::Mask(true)), None);
        assert_eq!(Fold::count(Lit::Mask(false)), None);
    }

    /// The ceiling of the domain is asked of one function: a range's end, a
    /// literal's count and `Fold::is_exact` agree at, below and past it.
    #[test]
    fn the_exact_bound_is_stated_once() {
        for n in [0, 7, Fold::EXACT_BOUND, Fold::EXACT_BOUND + 2] {
            let exact = Fold::is_exact(n);
            assert_eq!(KnownRange::admits(&(0..n)), exact, "a range ending at {n}");
            assert_eq!(
                Fold::count(Lit::Real(n as f32)).is_some(),
                exact,
                "a literal {n}"
            );
        }
    }

    /// Discovery reads a literal term into the bound and keeps every other: a
    /// term with no literal, a fraction and a count past the domain stay values
    /// (their guard makes the fold NaN), a mask is a sort error, and a pair of
    /// literals is not read at all.
    #[test]
    fn discovery_reads_a_literal_term_and_keeps_every_other() {
        use Bound::{Lit as Known, Value};
        assert_eq!(
            Bound::discover(Value(0u8), Value(1), term),
            Ok((Known(2), Known(9)))
        );
        assert_eq!(
            Bound::discover(Value(2u8), Value(1), term),
            Ok((Value(2), Known(9)))
        );
        for no_count in [3u8, 4] {
            assert_eq!(
                Bound::discover(Known(0), Value(no_count), term),
                Ok((Known(0), Value(no_count))),
                "term {no_count} names no count and stays a value"
            );
        }
        assert_eq!(
            Bound::discover(Known(0), Value(5u8), term),
            Err(BoundRefusal::BoundSort { end: BoundEnd::Hi })
        );
        assert_eq!(
            Bound::discover(Value(5u8), Known(3), term),
            Err(BoundRefusal::BoundSort { end: BoundEnd::Lo })
        );
        assert_eq!(
            Bound::<u8>::discover(Known(7), Known(3), |_| panic!(
                "a literal pair reads no term"
            )),
            Ok((Known(7), Known(3)))
        );
    }

    /// L2: a pair this read makes known runs forwards, so `5..3` is the empty
    /// range at 5 — while the pair a caller passed as two literals is left for
    /// `KnownRange::new` to refuse.
    #[test]
    fn a_pair_discovery_makes_known_runs_forwards() {
        use Bound::{Lit as Known, Value};
        assert_eq!(
            Bound::discover(Value(0u8), Value(1), reversed_term),
            Ok((Known(5), Known(5)))
        );
        assert_eq!(
            Bound::discover(Known(9), Value(1u8), reversed_term),
            Ok((Known(9), Known(9)))
        );
        let empty = FoldHeader::new(sum(0), Known(5), Known(5))
            .known_range()
            .expect("two literals");
        assert!(empty.is_empty());
    }

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
        let mut range = known(3..7);
        let mut indices = alloc::vec::Vec::new();
        while let Some((k, rest)) = range.peel() {
            indices.push(k);
            range = rest;
        }
        assert_eq!(indices, [3, 4, 5, 6]);
        assert!(range.is_empty());
        assert_eq!(range.len(), 0);
    }

    /// A monoid round-trips through its code, every one of them, and a code
    /// naming an op that generates no algebra decodes to none.
    #[test]
    fn a_monoid_round_trips_through_its_code() {
        for monoid in [
            Monoid::SUM,
            Monoid::PRODUCT,
            Monoid::MAX,
            Monoid::MIN,
            Monoid::ANY,
            Monoid::ALL,
            Monoid::SEQ,
        ] {
            assert_eq!(Monoid::unmarshal(monoid.marshal()), Some(monoid));
        }
        assert_eq!(Monoid::unmarshal(OpKind::Sub.marshal()), None);
    }

    /// A chain of distinct terms is left-nested, the first term alone and
    /// the monoid's own operation between each pair — `((a ⊕ b) ⊕ c)`, no
    /// identity in front — and a chain of none is the monoid's identity.
    #[test]
    fn a_chain_folds_its_terms_left_to_right_and_none_to_the_identity() {
        let spelled = |monoid: Monoid, terms: &[&str]| {
            let mut chain = Chain::new(monoid);
            for term in terms {
                chain.push(alloc::string::String::from(*term), |op, folded, term| {
                    alloc::format!("({folded} {op:?} {term})")
                });
            }
            chain.finish(|identity| alloc::format!("{identity}"))
        };
        assert_eq!(spelled(Monoid::SUM, &["a", "b", "c"]), "((a Add b) Add c)");
        assert_eq!(spelled(Monoid::MIN, &["a", "b"]), "(a Min b)");
        assert_eq!(spelled(Monoid::PRODUCT, &["a"]), "a");
        assert_eq!(spelled(Monoid::PRODUCT, &[]), "1");
        assert_eq!(spelled(Monoid::MAX, &[]), "-inf");
    }

    #[test]
    fn an_empty_fold_is_its_identity() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let empty = known(5..5);
        assert!(empty.is_empty());
        assert_eq!(empty.peel(), None);
        assert_eq!(
            Fold::new(Monoid::SUM, b).monoid().identity(),
            Lit::Real(0.0)
        );
        assert_eq!(Monoid::PRODUCT.identity(), Lit::Real(1.0));
        assert_eq!(Monoid::MIN.identity(), Lit::Real(f32::INFINITY));
        assert_eq!(Monoid::ALL.identity(), Lit::Mask(true));
        assert_eq!(Monoid::ANY.identity(), Lit::Mask(false));
    }

    #[test]
    #[should_panic(expected = "runs forwards")]
    fn a_reversed_range_is_refused() {
        // Built through variables: a literal `7..3` is a lint about an empty
        // range, and the point here is the *constructor's* refusal.
        let (lo, hi) = (7u64, 3u64);
        assert!(known(lo..hi).is_empty());
    }

    /// A range ends at most at 2²⁴, where an `f32` index stops naming every
    /// integer: `16777216` is the last end admitted, forwards only, and an
    /// empty range past it is refused too — its end is loaded as an `f32`
    /// like any other's.
    #[test]
    fn a_range_ends_at_most_at_the_exact_bound() {
        let bound = Fold::EXACT_BOUND;
        assert_eq!(bound, 16_777_216);
        assert_eq!(bound as f32 as u64, bound, "the bound itself is exact");
        assert_ne!((bound + 1) as f32 as u64, bound + 1, "and one past is not");
        for admitted in [0..0, 0..bound, bound - 1..bound, bound..bound] {
            assert!(KnownRange::admits(&admitted), "{admitted:?}");
        }
        let (lo, hi) = (7u64, 3u64);
        for refused in [
            0..bound + 1,
            bound + 1..bound + 1,
            0..u64::from(u32::MAX),
            0..u64::MAX,
            lo..hi,
        ] {
            assert!(!KnownRange::admits(&refused), "{refused:?}");
        }
    }

    /// A range's ends are held at the control plane's 64 bits: the whole of
    /// the domain is a count, and nothing narrower than the domain is there to
    /// cap it.
    #[test]
    fn the_ends_are_sixty_four_bits() {
        assert_eq!(known(0..Fold::EXACT_BOUND).len(), 1 << 24);
        assert_eq!(known(0..Fold::EXACT_BOUND).range().end, 1 << 24);
        assert!(!KnownRange::admits(&(0..u64::MAX)));
    }

    /// `Kernel::over`'s constructor refuses a range past the bound, as
    /// `kernel!` refuses one: the builder is covered by the same contract.
    #[test]
    #[should_panic(expected = "ends at most at 16777216 (2^24)")]
    fn a_range_past_the_exact_bound_is_refused() {
        let past = Fold::EXACT_BOUND + 1;
        assert!(!known(16_777_100..past).is_empty());
    }

    /// Bits naming a range past the bound name no fold: a corrupt key is a
    /// `None` at the boundary, not a fold whose index rounds.
    #[test]
    fn from_bits_refuses_a_range_past_the_exact_bound() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let bits = header(Fold::new(Monoid::SUM, b), 0, 4).to_bits();
        let past = (bits & !0xffff_ffffu128) | u128::from(Fold::EXACT_BOUND + 1);
        assert_eq!(FoldHeader::from_bits(past), None);
        let at = (bits & !0xffff_ffffu128) | u128::from(Fold::EXACT_BOUND);
        assert!(FoldHeader::from_bits(at).is_some());
    }

    /// An end field of all ones is the mark of a value bound, wherever it
    /// stands: nothing in the domain is `u32::MAX` (the build asserts it), so
    /// no literal shares it.
    #[test]
    fn an_end_field_of_all_ones_is_a_value_bound() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::new(Monoid::SUM, b);
        let bits = header(fold, 0, 4).to_bits();
        let hi_ones = bits | 0xffff_ffffu128;
        assert_eq!(
            FoldHeader::from_bits(hi_ones),
            Some(FoldHeader::new(fold, Bound::Lit(0), Bound::Value(()))),
            "hi"
        );
        let lo_ones = bits | (0xffff_ffffu128 << 32);
        assert_eq!(
            FoldHeader::from_bits(lo_ones),
            Some(FoldHeader::new(fold, Bound::Value(()), Bound::Lit(4))),
            "lo"
        );
    }

    /// Every placeholder is past every binder — a rename of one never
    /// reaches a slot an inner fold chose — and there are as many as a
    /// `u64` set tracks.
    #[test]
    fn a_placeholder_is_past_every_binder() {
        let first = Placeholder::nth(0).expect("a placeholder");
        assert_eq!(first.var(), crate::Variance::VARIABLES);
        assert!(Binder::all().all(|binder| binder.var() < first.var()));
        let last = Placeholder::nth(Placeholder::COUNT - 1).expect("the last placeholder");
        assert_eq!(
            usize::from(last.var() - first.var()),
            Placeholder::COUNT - 1
        );
        assert_eq!(Placeholder::nth(Placeholder::COUNT), None);
    }

    /// A trip count past what a `u16` held. A surviving fold is a loop, so
    /// this costs a counter, not sixty-six thousand copies of a body.
    #[test]
    fn a_range_past_sixteen_bits_is_a_fold_like_any_other() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let wide = header(Fold::new(Monoid::SUM, b), 1_000, 1_000_000);
        let range = wide.known_range().expect(KNOWN_RANGE);
        assert_eq!(range.len(), 999_000);
        assert_eq!(range.range(), 1_000..1_000_000);
        let halved = range.halve().expect("an even trip count halves");
        assert_eq!(halved.len(), 499_500);
        let back = FoldHeader::from_bits(wide.to_bits()).expect("a header's own bits name it");
        assert_eq!(back, wide);
    }

    /// The unit monoid: a fold over it is a loop whose combine says nothing.
    #[test]
    fn seq_is_a_monoid_whose_combine_is_sequencing() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::new(Monoid::SEQ, b);
        assert_eq!(fold.combine_op(), OpKind::Seq);
        assert_eq!(fold.monoid(), Monoid::SEQ);
        // The seed of an accumulator no combine ever reads.
        assert_eq!(Monoid::SEQ.identity(), Lit::Real(0.0));
        assert_eq!(Monoid::of(OpKind::Seq), Some(Monoid::SEQ));
        let seq = header(fold, 0, 4);
        let back = FoldHeader::from_bits(seq.to_bits()).expect("a SEQ fold round-trips");
        assert_eq!(back, seq);
        assert_eq!(back.fold(), fold);
    }

    #[test]
    fn strided_builds_with_the_given_step() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::strided(Monoid::SUM, b, 3);
        let strided = header(fold, 0, 12);
        let range = strided.known_range().expect(KNOWN_RANGE);
        assert_eq!(range.stride(), 3);
        assert_eq!(
            range.range(),
            0..12,
            "range() names the bound, not stride's steps"
        );
        assert_eq!(range.len(), 4);
        let back = FoldHeader::from_bits(strided.to_bits()).expect("a strided header's own bits");
        assert_eq!(back, strided);
        assert_eq!(back.fold(), fold);
    }

    #[test]
    #[should_panic(expected = "must divide")]
    fn strided_refuses_a_stride_that_does_not_divide_the_span() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert!(
            header(Fold::strided(Monoid::SUM, b, 3), 0, 10)
                .known_range()
                .is_some()
        );
    }

    #[test]
    #[should_panic(expected = "nonzero")]
    fn strided_refuses_a_zero_stride() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(Fold::strided(Monoid::SUM, b, 0).binder(), b);
    }

    /// A stride is added to an `f32` counter, so it is capped where an end is:
    /// at 2²⁴. Its width is the control plane's; its domain is the lane's.
    #[test]
    #[should_panic(expected = "a fold's stride is at most 16777216 (2^24)")]
    fn a_stride_past_the_exact_bound_is_refused() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(
            Fold::strided(Monoid::SUM, b, Fold::EXACT_BOUND + 1).binder(),
            b
        );
    }

    /// Bits naming a stride past the bound name no fold, however empty the
    /// range they carry: 2²⁴ + 1 divides the span `0` and is still refused.
    #[test]
    fn from_bits_refuses_a_stride_past_the_exact_bound() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let empty = header(Fold::new(Monoid::SUM, b), 0, 0).to_bits();
        let stride = |s: u64| (empty & !(0xffff_ffffu128 << 80)) | (u128::from(s) << 80);
        assert!(FoldHeader::from_bits(stride(Fold::EXACT_BOUND)).is_some());
        assert_eq!(FoldHeader::from_bits(stride(Fold::EXACT_BOUND + 1)), None);
    }

    #[test]
    fn new_is_strided_with_a_stride_of_one() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(Fold::new(Monoid::SUM, b), Fold::strided(Monoid::SUM, b, 1));
    }

    #[test]
    fn halve_doubles_the_stride_and_halves_the_trip_count() {
        let range = known(0..8);
        assert_eq!(range.stride(), 1);
        let once = range.halve().expect("8 is even and has more than one term");
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
        assert_eq!(known(0..0).halve(), None, "zero terms: nothing to pair");
        assert_eq!(
            known(0..1).halve(),
            None,
            "one term: nothing to pair it with"
        );
        assert_eq!(
            known(0..7).halve(),
            None,
            "an odd trip count has no even pairing"
        );
        assert!(known(0..2).halve().is_some());
    }

    /// **The load-bearing property.** Halving to exhaustion (falling back to
    /// [`KnownRange::peel_back`] for the odd remainder, the preference
    /// `egraph::fold_rules::HalveFold` gives it) must visit the same terms,
    /// in the same left-to-right order, as [`KnownRange::peel`] does one at a time
    /// — for an even trip count (pure halving, no remainder ever arises) and
    /// an odd one (forces the peel-back epilogue at more than one level of
    /// the recursion).
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
    /// the shape `egraph::fold_rules::HalveFold` builds for real, just
    /// instantiated to make the order legible without an arena or a JIT.
    #[test]
    fn halving_to_exhaustion_visits_the_same_terms_in_the_same_order_as_peeling() {
        fn combine(range: KnownRange, terms: &[alloc::vec::Vec<u64>]) -> alloc::vec::Vec<u64> {
            assert_eq!(range.len() as usize, terms.len());
            if let Some(halved) = range.halve() {
                let paired: alloc::vec::Vec<alloc::vec::Vec<u64>> = terms
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|[a, b]| [a.as_slice(), b.as_slice()].concat())
                    .collect();
                return combine(halved, &paired);
            }
            let (rest, _) = range.peel_back().expect("a non-empty fold has a last term");
            let last = terms.last().expect("len matches terms.len()").clone();
            if rest.is_empty() {
                return last;
            }
            let mut acc = combine(rest, &terms[..terms.len() - 1]);
            acc.extend(last);
            acc
        }

        for &(lo, hi) in &[(0u64, 8u64), (3, 11), (0, 7), (2, 9), (0, 1), (5, 6)] {
            let range = known(lo..hi);
            let want: alloc::vec::Vec<u64> = (lo..hi).collect();

            let terms: alloc::vec::Vec<alloc::vec::Vec<u64>> =
                (lo..hi).map(|k| alloc::vec![k]).collect();
            let halved_order = combine(range, &terms);
            assert_eq!(
                halved_order, want,
                "halving {lo}..{hi} must visit every index once, in ascending order"
            );

            let mut peeled_order = alloc::vec::Vec::new();
            let mut rest = range;
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

    /// `over` swaps only the range: the monoid and binder stay, a decomposition
    /// that leaves the stride alone leaves the fold alone, and the bounds come
    /// back as the literals the node holds.
    #[test]
    fn over_replaces_the_range_and_nothing_else() {
        let b = Binder::from_slot(1).expect("slot 1 exists");
        let fold = Fold::new(Monoid::PRODUCT, b);
        let whole = known(2..9);
        let (same, lo, hi) = fold.over::<()>(whole);
        assert_eq!(same, fold);
        assert_eq!((lo, hi), (Bound::Lit(2), Bound::Lit(9)));

        let (first, rest) = whole.peel().expect("2..9 is not empty");
        assert_eq!(first, 2);
        let (tail, lo, hi) = fold.over::<()>(rest);
        assert_eq!(tail, fold, "a peel leaves the stride alone");
        assert_eq!((lo, hi), (Bound::Lit(3), Bound::Lit(9)));
        assert_eq!(KnownRange::of(tail, lo, hi), Some(rest));
        assert_eq!(tail.monoid(), Monoid::PRODUCT);
        assert_eq!(tail.binder(), b);

        // A halving moves the stride, and the fold with it.
        let halved = known(2..10).halve().expect("8 is even");
        let (strided, _, _) = fold.over::<()>(halved);
        assert_eq!(strided, Fold::strided(Monoid::PRODUCT, b, 2));
    }

    #[test]
    fn to_bits_round_trips_a_halved_fold() {
        let b = Binder::from_slot(2).expect("slot 2 exists");
        let whole = Fold::new(Monoid::MAX, b);
        let range = known(10..26)
            .halve()
            .expect("16 is even")
            .halve()
            .expect("8 is still even");
        let halved = FoldHeader::known(whole, range);
        let range = halved.known_range().expect(KNOWN_RANGE);
        assert_eq!(range.stride(), 4);
        assert_eq!(range.len(), 4);

        let back = FoldHeader::from_bits(halved.to_bits()).expect("a header's own bits name it");
        assert_eq!(back, halved);
        let range = back.known_range().expect(KNOWN_RANGE);
        assert_eq!(range.stride(), 4);
        assert_eq!(back.fold().monoid(), Monoid::MAX);
        assert_eq!(back.fold().binder(), b);
        assert_eq!(range.range(), 10..26);
    }

    #[test]
    fn from_bits_refuses_a_stride_that_does_not_divide_the_range() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let mut bits = header(Fold::new(Monoid::SUM, b), 0, 5).to_bits();
        // Corrupt the stride field (bits 80..112) to 2, which does not divide
        // `5 - 0`: a fold that claims this would silently drop or duplicate
        // an index, so `from_bits` must refuse it rather than build one.
        bits = (bits & !(0xffff_ffffu128 << 80)) | (2u128 << 80);
        assert_eq!(FoldHeader::from_bits(bits), None);
    }

    /// Corpus compatibility, against a literal assembled by hand from the
    /// layout — nothing above bit 112, stride 1 at bit 80, `Add` (index 2)
    /// at 72, slot 1 at 64, `lo = 3` at 32, `hi = 11` at 0 — rather than
    /// against `to_bits`, which would only check the encoder agrees with
    /// itself. The same literal a fold has written since before the bounds
    /// left it and before the interval domain came and went, so a corpus
    /// serialized by any of them decodes unchanged; a bit set above the layout
    /// names no fold.
    #[test]
    fn a_header_decodes_from_its_layout_assembled_by_hand() {
        let literal: u128 = 0x0000_0000_0001_0201_0000_0003_0000_000b;
        let b = Binder::from_slot(1).expect("slot 1 exists");
        let want = header(Fold::new(Monoid::SUM, b), 3, 11);
        assert_eq!(FoldHeader::from_bits(literal), Some(want));
        assert_eq!(want.to_bits(), literal, "and a header still writes it");
        for bit in [112, 127] {
            assert_eq!(
                FoldHeader::from_bits(literal | 1u128 << bit),
                None,
                "bit {bit}"
            );
        }
    }

    /// The same layout with a bound that is a value, assembled by hand: stride 1
    /// at bit 80, `Add` at 72, slot 0 at 64, `lo = 0`, and `hi` a count field of
    /// all ones. The mark is a field no count reaches, so a header writes it and
    /// reads it back as a value, not as a literal.
    #[test]
    fn a_value_header_decodes_from_its_layout_assembled_by_hand() {
        let marked: u128 = 0x0000_0000_0001_0200_0000_0000_ffff_ffff;
        let want = FoldHeader::new(sum(0), Bound::Lit(0), Bound::Value(()));
        assert_eq!(FoldHeader::from_bits(marked), Some(want));
        assert_eq!(want.to_bits(), marked, "and a header still writes it");
        let lo_marked: u128 = 0x0000_0000_0001_0200_ffff_ffff_0000_0003;
        let want = FoldHeader::new(sum(0), Bound::Value(()), Bound::Lit(3));
        assert_eq!(FoldHeader::from_bits(lo_marked), Some(want));
        assert_eq!(want.to_bits(), lo_marked);
    }

    /// The same layout at the far end of its domain: stride 2 at bit 80, `Add`
    /// at 72, slot 0 at 64, `lo = 0`, `hi = 2^24` at 0 — the widest end a fold
    /// may name, written as the 32-bit field it has always been.
    #[test]
    fn a_header_at_the_exact_bound_decodes_from_its_layout_assembled_by_hand() {
        let literal: u128 = 0x0000_0000_0002_0200_0000_0000_0100_0000;
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let want = header(Fold::strided(Monoid::SUM, b, 2), 0, Fold::EXACT_BOUND);
        assert_eq!(FoldHeader::from_bits(literal), Some(want));
        assert_eq!(want.to_bits(), literal);
    }

    #[test]
    fn a_header_prints_its_domain() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        assert_eq!(
            alloc::format!("{}", header(Fold::new(Monoid::SUM, b), 0, 8)),
            "add_4[0..8)"
        );
        assert_eq!(
            alloc::format!("{}", header(Fold::strided(Monoid::MAX, b, 2), 0, 8)),
            "max_4[0..8) step 2"
        );
    }

    /// A bound that is a value prints as `_`, wherever it stands, and the
    /// stride after it as for a literal pair: the header says which bounds are
    /// children, and the node's display prints the children.
    #[test]
    fn a_header_prints_a_value_bound_as_a_hole() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::new(Monoid::SUM, b);
        let show = |lo, hi| alloc::format!("{}", FoldHeader::store(fold, lo, hi));
        assert_eq!(show(Bound::Lit(3), Bound::Value(())), "add_4[3.._)");
        assert_eq!(show(Bound::Value(()), Bound::Lit(3)), "add_4[_..3)");
        assert_eq!(show(Bound::Value(()), Bound::Value(())), "add_4[_.._)");
        let strided = Fold::strided(Monoid::MAX, b, 2);
        assert_eq!(
            alloc::format!(
                "{}",
                FoldHeader::store(strided, Bound::Lit(1), Bound::Value(()))
            ),
            "max_4[1.._) step 2"
        );
    }

    /// A fold over an arithmetic algebra takes a bound that is a value at
    /// stride one: its header holds the mark of a child, knows no range, and is
    /// priced at the estimate; the algebras whose sorts have no out-of-domain
    /// answer (`ANY`, `ALL`) and the lattice's (`SEQ`) take none.
    #[test]
    fn a_value_bound_takes_an_arithmetic_fold_at_stride_one() {
        for monoid in [Monoid::SUM, Monoid::PRODUCT, Monoid::MIN, Monoid::MAX] {
            assert!(monoid.takes_a_value_bound());
            let value = FoldHeader::new(Fold::new(monoid, b(0)), Bound::Lit(0), Bound::Value(()));
            assert_eq!(value.arity(), 2);
            assert_eq!(value.known_range(), None);
            assert_eq!(value.hi(), Bound::Value(()));
            assert_eq!(value.priced_trips(), Fold::ESTIMATED_TRIPS);
        }
        for monoid in [Monoid::ANY, Monoid::ALL, Monoid::SEQ] {
            assert!(!monoid.takes_a_value_bound());
        }
    }

    /// A value bound on a fold of masks is refused by its sort (decision 8).
    #[test]
    #[should_panic(expected = "BoundSort")]
    fn a_value_bound_on_an_any_fold_is_refused() {
        let _refused = FoldHeader::new(
            Fold::new(Monoid::ANY, b(0)),
            Bound::Lit(0),
            Bound::Value(()),
        );
    }

    /// As for `ANY`: a mask has no NaN to answer an out-of-domain bound.
    #[test]
    #[should_panic(expected = "BoundSort")]
    fn a_value_bound_on_an_all_fold_is_refused() {
        let _refused = FoldHeader::new(
            Fold::new(Monoid::ALL, b(0)),
            Bound::Value(()),
            Bound::Lit(3),
        );
    }

    /// The lattice's folds keep constant ranges.
    #[test]
    #[should_panic(expected = "BoundSort")]
    fn a_value_bound_on_a_seq_fold_is_refused() {
        let _refused = FoldHeader::new(
            Fold::new(Monoid::SEQ, b(0)),
            Bound::Lit(0),
            Bound::Value(()),
        );
    }

    /// A stride other than 1 belongs to a known range it divides.
    #[test]
    #[should_panic(expected = "only over a known range it divides")]
    fn a_value_bound_needs_stride_one() {
        let _refused = FoldHeader::new(
            Fold::strided(Monoid::SUM, b(0), 2),
            Bound::Lit(0),
            Bound::Value(()),
        );
    }

    /// The literal beside a value is checked against the domain as any literal
    /// is: no discovery or rule makes one past it, a hand-built node does.
    #[test]
    #[should_panic(expected = "a fold's literal bound is at most 16777216 (2^24)")]
    fn a_literal_beside_a_value_is_in_the_domain() {
        let _refused = FoldHeader::new(sum(0), Bound::Lit(Fold::EXACT_BOUND + 1), Bound::Value(()));
    }

    /// A known range is priced at its length, an unknown one at the estimate.
    #[test]
    fn priced_trips_is_the_known_length_or_the_estimate() {
        assert_eq!(header(sum(0), 0, 40).priced_trips(), 40);
        assert_eq!(header(sum(0), 3, 3).priced_trips(), 0);
        let value = FoldHeader::new(sum(0), Bound::Lit(0), Bound::Value(()));
        assert_eq!(value.priced_trips(), 32);
        assert_eq!(Fold::ESTIMATED_TRIPS, 32);
    }

    /// L3, stated once: a fold admits a bound that reads neither a coordinate
    /// nor its own binder, and an enclosing fold's binder is a bound's to read.
    #[test]
    fn a_fold_admits_a_bound_reading_neither_a_coordinate_nor_its_own_binder() {
        let fold = sum(0);
        assert_eq!(fold.admits_bound(BoundEnd::Lo, Variance::CONST), Ok(()));
        assert_eq!(
            fold.admits_bound(BoundEnd::Hi, Variance::from_var(5)),
            Ok(())
        );
        for (end, reads) in [
            (BoundEnd::Hi, Variance::X),
            (BoundEnd::Lo, Variance::Y),
            (BoundEnd::Hi, Variance::from_var(b(0).var())),
        ] {
            assert_eq!(
                fold.admits_bound(end, reads),
                Err(BoundRefusal::BoundVaries { end, reads })
            );
        }
    }

    /// Each refusal starts with its name, which is what a `should_panic` matches,
    /// and holds the law it breaks.
    #[test]
    fn a_bound_refusal_names_its_law() {
        let end = BoundEnd::Hi;
        for (refusal, name, law) in [
            (
                BoundRefusal::BoundOutOfDomain { end, literal: 2.5 },
                "BoundOutOfDomain",
                "L5",
            ),
            (BoundRefusal::BoundSort { end }, "BoundSort", "decision 8"),
            (
                BoundRefusal::BoundVaries {
                    end,
                    reads: Variance::X,
                },
                "BoundVaries",
                "L3",
            ),
            (
                BoundRefusal::BoundArithmetic { end },
                "BoundArithmetic",
                "T6",
            ),
            (
                BoundRefusal::BoundNotANumber { end },
                "BoundNotANumber",
                "`Param`",
            ),
        ] {
            let shown = alloc::format!("{refusal}");
            assert!(shown.starts_with(name), "{shown}");
            assert!(shown.contains(law), "{shown} names {law}");
            assert!(shown.contains("a fold's hi bound"), "{shown}");
        }
        let out_of_domain = BoundRefusal::BoundOutOfDomain {
            end: BoundEnd::Lo,
            literal: 2.5,
        };
        assert!(alloc::format!("{out_of_domain}").contains("a fold's lo bound is 2.5"));
        assert_eq!(alloc::format!("{} {}", BoundEnd::Lo, BoundEnd::Hi), "lo hi");
    }

    /// A value bound's count field is 0, and the header still tells it from the
    /// literal 0: how a bound is held is a field `Eq`, `Hash` and `Ord` read.
    #[test]
    fn a_value_bound_is_not_the_literal_zero() {
        let fold = sum(0);
        let zero = header(fold, 0, 4);
        let value = FoldHeader::new(fold, Bound::Value(()), Bound::Lit(4));
        assert_ne!(
            zero, value,
            "the count field alone does not tell them apart"
        );
        assert_eq!(zero.lo(), Bound::Lit(0));
        assert_eq!(value.lo(), Bound::Value(()));
        assert_eq!(value.hi(), Bound::Lit(4));
        assert_eq!(value.known_range(), None);
        assert_eq!(value.fold(), fold);
        assert_eq!(
            value,
            FoldHeader::new(fold, Bound::Value(()), Bound::Lit(4)),
            "and a value is one header however it was written"
        );
    }

    /// A value bound writes the all-ones mark in its count field and reads back
    /// as a value; the mark stands in for a child the bits do not hold, and no
    /// literal writes it.
    #[test]
    fn a_value_bound_writes_the_marker_and_reads_back() {
        let marker = u128::from(u32::MAX);
        for (lo, hi) in [
            (Bound::Value(()), Bound::Lit(4)),
            (Bound::Lit(3), Bound::Value(())),
            (Bound::Value(()), Bound::Value(())),
        ] {
            let value = FoldHeader::new(sum(0), lo, hi);
            let bits = value.to_bits();
            assert_eq!(FoldHeader::from_bits(bits), Some(value), "{value}");
            let (lo_field, hi_field) = ((bits >> 32) & marker, bits & marker);
            assert_eq!(lo_field == marker, lo.value().is_some(), "{value}: lo");
            assert_eq!(hi_field == marker, hi.value().is_some(), "{value}: hi");
        }
        let literal = header(sum(0), 3, Fold::EXACT_BOUND);
        assert_ne!(
            literal.to_bits() & marker,
            marker,
            "the largest count is not the mark"
        );
    }

    /// The mark means a value only where the fold can hold one: not under a fold
    /// of masks or the lattice's, not at a stride other than 1, and not beside a
    /// literal past the domain. A field one short of the mark is a count past
    /// the bound, not the mark.
    #[test]
    fn from_bits_refuses_a_marker_its_fold_cannot_take() {
        let marked = |monoid, stride: u32, lo: u64, hi: Option<u64>| {
            let fold = u128::from(stride) << 80
                | u128::from(monoid_byte(monoid)) << 72
                | u128::from(lo) << 32
                | hi.map_or(u128::from(u32::MAX), u128::from);
            FoldHeader::from_bits(fold)
        };
        assert!(marked(Monoid::SUM, 1, 0, None).is_some(), "the control");
        for monoid in [Monoid::ANY, Monoid::ALL, Monoid::SEQ] {
            assert_eq!(marked(monoid, 1, 0, None), None, "{monoid:?}");
        }
        assert_eq!(marked(Monoid::SUM, 2, 0, None), None, "stride 2");
        assert_eq!(
            marked(Monoid::SUM, 1, Fold::EXACT_BOUND + 1, None),
            None,
            "a literal past the bound"
        );
        assert!(marked(Monoid::SUM, 1, Fold::EXACT_BOUND, None).is_some());
        assert_eq!(
            marked(Monoid::SUM, 1, 0, Some(u64::from(u32::MAX) - 1)),
            None,
            "one short of the mark is a count past the bound"
        );
    }

    /// What a value bound asks of its fold is stated twice: as one predicate,
    /// which `from_bits` asks, and a part at a time, which `new` asserts to say
    /// which part failed. They agree on every monoid, stride and pair of
    /// bounds with a value in it, the literal beside it inside the domain and
    /// just outside it.
    #[cfg(feature = "std")]
    #[test]
    fn new_panics_exactly_where_the_predicate_refuses() {
        let past = Fold::EXACT_BOUND + 1;
        for monoid in [
            Monoid::SUM,
            Monoid::PRODUCT,
            Monoid::MIN,
            Monoid::MAX,
            Monoid::ANY,
            Monoid::ALL,
            Monoid::SEQ,
        ] {
            for stride in [1, 2] {
                for (lo, hi) in [
                    (Bound::Value(()), Bound::Value(())),
                    (Bound::Lit(0), Bound::Value(())),
                    (Bound::Value(()), Bound::Lit(Fold::EXACT_BOUND)),
                    (Bound::Lit(past), Bound::Value(())),
                    (Bound::Value(()), Bound::Lit(past)),
                ] {
                    let fold = Fold::strided(monoid, b(0), stride);
                    let built = std::panic::catch_unwind(|| FoldHeader::new(fold, lo, hi));
                    assert_eq!(
                        built.is_ok(),
                        admits_a_value(fold, lo, hi),
                        "{monoid:?} step {stride} from {lo:?} to {hi:?}"
                    );
                }
            }
        }
    }

    /// `from_bits` inverts `to_bits` for every header `FoldHeader::new` admits:
    /// each monoid that takes a value beside a literal pair, a value at either
    /// end, and both.
    #[test]
    fn from_bits_inverts_to_bits_on_every_header_shape() {
        let shapes = [
            (Bound::Lit(0), Bound::Lit(8)),
            (Bound::Lit(3), Bound::Lit(3)),
            (Bound::Lit(0), Bound::Lit(Fold::EXACT_BOUND)),
            (Bound::Value(()), Bound::Lit(4)),
            (Bound::Lit(3), Bound::Value(())),
            (Bound::Value(()), Bound::Value(())),
            (Bound::Lit(Fold::EXACT_BOUND), Bound::Value(())),
        ];
        for monoid in [
            Monoid::SUM,
            Monoid::PRODUCT,
            Monoid::MIN,
            Monoid::MAX,
            Monoid::ANY,
            Monoid::ALL,
            Monoid::SEQ,
        ] {
            for (lo, hi) in shapes {
                let holds_a_value = lo.value().is_some() || hi.value().is_some();
                if holds_a_value && !monoid.takes_a_value_bound() {
                    continue;
                }
                let value = FoldHeader::new(Fold::new(monoid, b(2)), lo, hi);
                assert_eq!(
                    FoldHeader::from_bits(value.to_bits()),
                    Some(value),
                    "{monoid:?} {value}"
                );
            }
        }
    }

    /// Each value bound is handed the next child after the body, `lo`'s before
    /// `hi`'s, and a literal is handed none.
    #[test]
    fn attach_hands_each_value_bound_its_child() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::new(Monoid::SUM, b);

        let literal = header(fold, 0, 4);
        assert_eq!(literal.arity(), 1);
        assert_eq!(
            literal.attach(core::iter::empty::<u32>()),
            (Bound::Lit(0), Bound::Lit(4))
        );

        let value_hi = FoldHeader::store(fold, Bound::Lit(3), Bound::Value(()));
        assert_eq!(value_hi.arity(), 2);
        assert_eq!(value_hi.attach([9u32]), (Bound::Lit(3), Bound::Value(9)));

        let value_lo = FoldHeader::store(fold, Bound::Value(()), Bound::Lit(3));
        assert_eq!(value_lo.arity(), 2);
        assert_eq!(value_lo.attach([9u32]), (Bound::Value(9), Bound::Lit(3)));

        let both = FoldHeader::store(fold, Bound::Value(()), Bound::Value(()));
        assert_eq!(both.arity(), 3);
        assert_eq!(
            both.attach([7u32, 9]),
            (Bound::Value(7), Bound::Value(9)),
            "lo's child before hi's"
        );
    }

    /// A node whose edges disagree with its header is refused: too many.
    #[test]
    #[should_panic(expected = "value bounds was given")]
    fn attach_refuses_a_child_a_literal_header_has_no_bound_for() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let _refused = header(Fold::new(Monoid::SUM, b), 0, 4).attach([9u32]);
    }

    /// A node whose edges disagree with its header is refused: too few.
    #[test]
    #[should_panic(expected = "value bounds was given")]
    fn attach_refuses_a_value_bound_with_no_child() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let value = FoldHeader::store(Fold::new(Monoid::SUM, b), Bound::Lit(3), Bound::Value(()));
        let _refused = value.attach(core::iter::empty::<u32>());
    }

    /// The one reader of a range answers `Some` exactly when both bounds are
    /// literals, whatever names a child bound would have, and refuses a pair
    /// no constructor admits with the constructor's words.
    #[test]
    fn of_reads_a_range_from_two_literal_bounds_only() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let fold = Fold::strided(Monoid::SUM, b, 2);
        let range = KnownRange::of(fold, Bound::<u8>::Lit(0), Bound::Lit(8)).expect("two literals");
        assert_eq!((range.range(), range.stride()), (0..8, 2));
        assert_eq!(KnownRange::of(fold, Bound::Value(0u8), Bound::Lit(8)), None);
        assert_eq!(KnownRange::of(fold, Bound::Lit(0), Bound::Value(0u8)), None);
        assert_eq!(
            KnownRange::of(fold, Bound::Value(0u8), Bound::Value(1u8)),
            None
        );
    }

    #[test]
    #[should_panic(expected = "runs forwards")]
    fn of_refuses_a_hand_made_pair_that_runs_backwards() {
        let b = Binder::from_slot(0).expect("slot 0 exists");
        let _refused = KnownRange::of(
            Fold::new(Monoid::SUM, b),
            Bound::<u8>::Lit(7),
            Bound::Lit(3),
        );
    }

    #[test]
    fn a_bound_maps_erases_and_reads() {
        assert_eq!(Bound::<u32>::Lit(3).map(|v| v + 1), Bound::Lit(3));
        assert_eq!(Bound::Value(4u32).map(|v| v + 1), Bound::Value(5));
        assert_eq!(Bound::<u32>::Lit(3).lit(), Some(3));
        assert_eq!(Bound::Value(4u32).lit(), None);
        assert_eq!(Bound::<u32>::Lit(3).value(), None);
        assert_eq!(Bound::Value(4u32).value(), Some(4));
        assert_eq!(Bound::Value(4u32).erased(), Bound::Value(()));
        assert_eq!(Bound::<u32>::Lit(3).erased(), Bound::Lit(3));
    }

    /// What a fold costs a node, pinned as byte counts so a field that still
    /// fits `arena.rs`'s tripwire fails here with a number: `Fold` is a monoid,
    /// a binder and a 64-bit stride; a header packs three 32-bit counts and
    /// four bytes of the rest. Measured, not budgeted — facts about the types,
    /// not widths their fields were chosen to meet; the 32-bit counts have
    /// their reason in [`FoldHeader`]'s doc.
    #[test]
    fn a_fold_header_is_measured_not_budgeted() {
        assert_eq!(
            core::mem::size_of::<Fold>(),
            16,
            "Fold: monoid(1) + binder(1) + pad(6) + stride(8)"
        );
        assert_eq!(
            core::mem::size_of::<Bound<()>>(),
            16,
            "Bound: tag(1) + pad(7) + literal(8)"
        );
        assert_eq!(
            core::mem::size_of::<KnownRange>(),
            24,
            "KnownRange: lo(8) + hi(8) + stride(8)"
        );
        assert_eq!(
            core::mem::size_of::<FoldHeader>(),
            16,
            "FoldHeader: lo(4) + hi(4) + stride(4) + monoid(1) + binder(1) + two Held(2)"
        );
        assert_eq!(
            core::mem::size_of::<crate::expr::ExprData>(),
            16,
            "ExprData: the FoldHeader's 16 bytes and no more; the tag fits its niches"
        );
        assert!(
            core::mem::size_of::<crate::arena::ExprNode>() <= 64,
            "see arena.rs's assertion: a tripwire against an accident, not a \
             width anything depends on"
        );
    }
}
