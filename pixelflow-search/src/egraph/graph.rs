//! The E-Graph data structure and operations.

use rustc_hash::FxHashMap as HashMap;

use super::node::{EClassId, ENode};
use super::ops;
use super::provenance::{ENodeId, Provenance};
use super::rewrite::{RewriteAction, RootShapes};
use alloc::collections::BTreeMap;
use columns::{Birth, Columns, Kept, SaturationColumns, Slot};
use pixelflow_ir::fold::{Bound, BoundEnd, BoundRefusal, Fold};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{KernelKey, Lit, Variance};

mod columns;
mod runner;
pub use runner::{
    ApplicationMask, HARD_MINTED_LIMIT, Limits, MaskScope, RewriteTarget, Saturation,
    SaturationStats, SaturationStop,
};

/// The store: a `FrozenGraph` (classes, the union-find over them, the
/// per-class facts, the units) and what only saturation reads beside it: the
/// memo that hash-conses their nodes, the worklist, the graph's own mutation
/// log and the columns the scan and rebuild keep.
///
/// It holds no rules and counts no applications — those belong to the runner
/// that sweeps it ([`Saturation`]). What a store does know about rewriting is
/// the change log the runner's dirty tracking reads
/// (`mutation_counter`, `last_changed`) and, when one is
/// kept, the journal that attributes its mutations.
#[derive(Clone)]
pub struct EGraph {
    /// The classes, their facts and the units: what freezing keeps
    /// (`FrozenGraph`). Written only by this type.
    graph: FrozenGraph,
    /// The per-id columns only saturation reads, grown with `graph`'s by
    /// `Columns::push`.
    saturation: SaturationColumns,
    /// Canonical node forms to the class each was in when computed (law 2).
    /// Each key is a copy, not a node id: `rebuild` rewrites the arena's
    /// stored operands in place, and a key keeps the form it was computed in.
    memo: HashMap<ENode, EClassId>,
    worklist: Vec<Slot>,
    /// Unions REFUSED because they would assert two numerically unequal
    /// constants equal — a proved falsehood the graph declines to absorb.
    /// Distinct pairs of literals, kept for reporting and tests; see
    /// [`EGraph::union`].
    refused_const_unions: Vec<(Lit, Lit)>,
    /// Global monotonic counter, bumped once per class-content mutation
    /// (`add`'s memo miss, `union`'s merge) — never reset, never bumped by
    /// anything that only re-canonicalizes existing content (rewriting a
    /// child's stored id to its current representative during rebuild changes
    /// a name, not the class's content, and `class_is_dirty` follows every id
    /// through `find`). See [`Self::class_is_dirty`].
    mutation_counter: u64,
    /// The graph's mutation log, or `None` when the graph was built without
    /// one ([`Self::without_journal`]).
    ///
    /// Origins and merges are recorded as they happen — `add` and `union` are
    /// the only writers — so the log cannot be switched on after the fact,
    /// and a graph built without it has nothing to query rather than an
    /// empty or misattributed answer. The runner writes its cursor (the
    /// current step and the application being executed); see
    /// [`Provenance::open_application`].
    journal: Option<Provenance>,
}

/// An e-graph's classes once nothing will add to them: the node arena, the
/// union-find, the member lists and the live set, the per-class facts, and the
/// units the graph admitted. Everything extraction reads, and only that.
///
/// It has no `add`, `union` or `rebuild`, and no method that takes `&mut self`.
/// [`EGraph`] holds one and is the only thing that writes it;
/// [`EGraph::freeze`] hands it over and drops what only saturation reads (the
/// memo, the worklist, the change log, the scan's and rebuild's columns, the
/// journal and the refusal record), and [`EGraph::as_frozen`] lends it. A cache
/// of saturated graphs keeps one of these, so a cached graph cannot be added to:
/// there is no method to call.
#[derive(Clone)]
pub(crate) struct FrozenGraph {
    /// The per-id columns extraction reads. See the `columns` module.
    columns: Columns,
    /// The units this graph may hold as leaves, with the variance each
    /// carries — told to it by the runtime tier's unit walk
    /// ([`EGraph::admit_unit`]) before the term is inserted. A
    /// [`ENode::Ref`] is only its key, so this is where its variance is read
    /// from. A unit's body is not in this graph, so its variance cannot be
    /// computed here; and a
    /// reference the graph was not told about is one [`insert`](super::insert)
    /// declines, which is what keeps every other path — the macro tier,
    /// research tools — exactly as it was (docs/language/2026-09-25-the-language-is-kernel.md
    /// §4, O1).
    units: BTreeMap<KernelKey, Variance>,
}

impl FrozenGraph {
    /// [`EGraph::find`], on the frozen graph.
    pub(crate) fn find(&self, id: EClassId) -> EClassId {
        self.columns.root(id).id()
    }

    /// [`EGraph::canonical_node`], on the frozen graph.
    pub(crate) fn canonical_node(&self, mut node: ENode) -> ENode {
        for child in node.children_slice_mut() {
            *child = self.find(*child);
        }
        node
    }

    /// [`EGraph::unit_variance`], on the frozen graph.
    pub(crate) fn unit_variance(&self, key: KernelKey) -> Option<Variance> {
        self.units.get(&key).copied()
    }

    /// [`EGraph::ref_variance`], on the frozen graph.
    pub(crate) fn ref_variance(&self, key: KernelKey) -> Variance {
        self.unit_variance(key)
            .unwrap_or_else(|| panic!("EGraph: a Ref to {key:?}, which no unit walk admitted"))
    }

    /// [`EGraph::nodes`], on the frozen graph.
    pub(crate) fn nodes(&self, id: EClassId) -> impl Iterator<Item = &ENode> + Clone + '_ {
        let class = self.columns.root(id);
        self.columns
            .members(class)
            .map(move |tag| self.columns.node(tag))
    }

    /// [`EGraph::tags`], on the frozen graph.
    pub(crate) fn tags(&self, id: EClassId) -> impl Iterator<Item = ENodeId> + Clone + '_ {
        self.columns.members(self.columns.root(id))
    }

    /// [`EGraph::node_for_tag`], on the frozen graph.
    pub(crate) fn node_for_tag(&self, id: EClassId, tag: ENodeId) -> Option<&ENode> {
        let minted_as = self.columns.class_of(tag)?;
        if self.columns.is_retired(tag) || self.find(minted_as) != self.find(id) {
            return None;
        }
        Some(self.columns.node(tag))
    }

    /// The node `tag` names in this graph's arena, as stored. For a tag the
    /// caller read from [`Self::tags`] or that an extraction recorded, so known
    /// to be a member: this is not a membership test, and a retired node is
    /// answered too. A tag of unknown standing asks [`Self::node_for_tag`].
    /// Panics past the end. O(1).
    pub(crate) fn node(&self, tag: ENodeId) -> &ENode {
        self.columns.node(tag)
    }

    /// [`EGraph::num_classes`], on the frozen graph.
    pub(crate) fn num_classes(&self) -> usize {
        self.columns.len()
    }
}

/// Why a research walk over the journal refused a graph that keeps none.
#[cfg(feature = "provenance-journal")]
const NO_JOURNAL: &str = "this graph was built without a journal (Saturation::without_journal, or \
     an Optimizer with no observer attached) — a derivation walk has nothing to read";

/// [`EGraph::class_is_dirty`] checks this many hops of a class's forward
/// neighborhood (the class itself, its direct children, and their
/// children) against a rule's last-swept generation.
///
/// This is the audited maximum matching depth across every rule in
/// [`super::all_rules`] — how many levels beyond the single node handed to
/// `Rewrite::apply` a rule inspects via `egraph.nodes()` on a child (or
/// grandchild) e-class. `math/trig.rs`'s `Pythagorean` and
/// `ReverseAngleAddition` are the only rules that reach a grandchild
/// (through `extract_trig_arg`/`extract_squared_trig`/`extract_sin_cos_mul`);
/// every other rule in the production set — all of `math/algebra.rs`,
/// `math/parity.rs`, `math/exp.rs`, `math/power.rs`, `math/fusion.rs`, and
/// `egraph/derivative.rs`'s `ChainRule` — is depth 0 or 1. A dirty-tracking
/// scheme that only propagated 1 hop would be UNSOUND for those two rules:
/// a class could become newly matchable when a grandchild changes without
/// its own node list, or its direct children's, changing at all, and a
/// 1-hop scheme would silently never re-check it — under-saturation that no
/// correctness test can see, only a comparison against the un-skipped
/// extraction cost can. The runtime tier's `FactorFold` (outside
/// `all_rules`, in `fold_rules`) is depth 2 as well: it reads a fold's body
/// class and then its operands' variance facts, and a fact changes only when
/// its own class is unioned, which bumps that class's `last_changed`.
///
/// This is a single, uniform, crate-wide constant rather than a per-rule
/// depth precisely so a future rule cannot silently exceed it the way a
/// per-rule value could go stale unnoticed: every rule is checked to this
/// depth regardless of what it actually needs, so a new rule that matches
/// deeper than this is caught the same way `Pythagorean` would be — by
/// `docs/archive/plans/2026-09-08-egraph-cpu-memory-profile.md`'s methodology
/// (compare extracted cost with dirty-tracking on vs. off), not by silent
/// trust in a number nobody re-audits. Raise it if that audit is ever
/// redone and finds a deeper rule; do not lower it without redoing the
/// audit above.
const DIRTY_TRACKING_MAX_DEPTH: u8 = 2;

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    /// Create an empty store.
    ///
    /// With `provenance-journal` the mutation log is kept, so every direct
    /// caller keeps what it had. Rules are the runner's: see
    /// [`Saturation::over`].
    pub fn new() -> Self {
        Self {
            graph: FrozenGraph {
                columns: Columns::default(),
                units: BTreeMap::new(),
            },
            saturation: SaturationColumns::default(),
            memo: HashMap::default(),
            worklist: Vec::new(),
            refused_const_unions: Vec::new(),
            mutation_counter: 0,
            journal: Some(Provenance::new()),
        }
    }

    /// The same store with its mutation log turned off.
    ///
    /// A choice made when the graph is built: an origin or a merge recorded
    /// without the application that caused it reads as a seed or a congruence
    /// union, so a log that could be flipped mid-run would hand out wrong
    /// answers rather than missing ones. Production builds its graphs this
    /// way, and a cached graph then carries no log at all.
    #[must_use]
    pub fn without_journal(mut self) -> Self {
        assert!(
            self.graph.columns.len() == 0,
            "a journal is chosen when the graph is built: this graph already holds {} classes, \
             whose births it would have to forget",
            self.graph.columns.len(),
        );
        self.journal = None;
        self
    }

    /// The graph without what only saturation reads: the memo, the worklist,
    /// the change log, the scan's and rebuild's columns, the journal and the
    /// refusal record are dropped, and what extraction reads is moved, not
    /// copied. What a cache of saturated graphs keeps: a [`FrozenGraph`] has
    /// no `add`, so nothing can add to a graph whose memo is gone.
    #[must_use]
    pub(crate) fn freeze(self) -> FrozenGraph {
        self.graph
    }

    /// What [`Self::freeze`] would keep, borrowed: for extraction over a graph
    /// that stays live.
    #[must_use]
    pub(crate) fn as_frozen(&self) -> &FrozenGraph {
        &self.graph
    }

    /// Advance the journal's step, if one is kept. The runner's cursor: the
    /// one place the store is told something about the sweep that is
    /// mutating it.
    fn begin_step(&mut self) {
        #[cfg(feature = "provenance-journal")]
        if let Some(journal) = &mut self.journal {
            journal.begin_step();
        }
    }

    pub fn find(&self, id: EClassId) -> EClassId {
        self.graph.find(id)
    }

    /// `node` with every operand class replaced by its canonical class: the
    /// form the memo keys on, and the form two nodes are compared in.
    ///
    /// Every operand, a fold's body included. A graph between a `union` and
    /// the `rebuild` that follows it holds nodes naming merged-away classes,
    /// and only this copy can be compared with another.
    pub(crate) fn canonical_node(&self, node: ENode) -> ENode {
        self.graph.canonical_node(node)
    }

    /// Tell this graph that `key` names a **unit** varying as `variance` says,
    /// so a reference to it inserts as an opaque leaf
    /// ([`ENode::Ref`]) rather than being declined.
    ///
    /// `runtime::saturate` calls this for every unit of the program, before it
    /// inserts a job's term, with the variance the runtime tier's unit walk
    /// (`runtime::Program::of`) computed: that walk holds each unit's body, so
    /// it is the only place the variance can be computed
    /// (docs/language/2026-09-25-the-language-is-kernel.md §4, O1).
    ///
    /// # Panics
    ///
    /// Panics if `variance` names a binder or a retired axis — a unit is a
    /// closed term over the coordinates and uniforms, and nothing else can be
    /// optimized out of its context, because no rewrite outside it can reach
    /// a binder inside it. Panics too if `key` was admitted before with a
    /// different variance: a key names one referent, so its variance is one
    /// value, and two `Ref` nodes for one key would otherwise differ only in
    /// what they claim about it.
    ///
    /// This sees a variance and nothing else, and a fold index that escaped
    /// its builder (a placeholder) has an empty one (`Variance::of_var`).
    /// Closedness is judged before the call, on the term: `Program::of`
    /// refuses an open unit by `ExprArena::free_index` when it resolves the
    /// body.
    pub(crate) fn admit_unit(&mut self, key: KernelKey, variance: Variance) {
        assert!(
            variance.without(Variance::COORDS).is_const(),
            "EGraph::admit_unit: {key:?} varies as {variance:?} — a unit must be \
             closed over the coordinates, and this one reads a binder no \
             rewrite outside it could substitute"
        );
        if let Some(admitted) = self.graph.units.insert(key, variance) {
            assert_eq!(
                admitted, variance,
                "EGraph::admit_unit: {key:?} was admitted as varying as {admitted:?} and is \
                 admitted again as {variance:?}"
            );
        }
    }

    /// The variance a unit admitted by [`Self::admit_unit`] carries, or
    /// `None` if `key` was never admitted.
    pub(crate) fn unit_variance(&self, key: KernelKey) -> Option<Variance> {
        self.graph.unit_variance(key)
    }

    /// The variance of the unit a [`ENode::Ref`] to `key` names: what
    /// [`Self::unit_variance`] answers, for a node the graph holds.
    ///
    /// # Panics
    ///
    /// Panics if `key` was never admitted. [`Self::add`] refuses a `Ref` to
    /// such a unit, so no node in the graph names one.
    pub(crate) fn ref_variance(&self, key: KernelKey) -> Variance {
        self.graph.ref_variance(key)
    }

    /// Insert `node`, returning the e-class that contains it.
    ///
    /// **Total, and the law depends on it.** The returned class always
    /// contains `node`: either the memo already had it, or this call
    /// allocated a fresh class holding exactly it. There is no size at
    /// which `add` returns something else, because there is nothing else
    /// it could correctly return — this is law L1 (`add` is a homomorphism
    /// from the term algebra onto the e-graph quotient,
    /// `docs/archive/plans/2026-09-02-optimizer-api.md` §1), and L4 (any saturation
    /// policy preserves denotation) is proved from it.
    ///
    /// The class budget therefore is not enforced here. It is enforced
    /// where growth is *decided* — the saturation drivers, each of which
    /// clamps its caller's cap to [`HARD_MINTED_LIMIT`] and stops scanning
    /// before it calls `add`. A driver that stops early leaves a graph
    /// holding fewer equalities than an exhaustive run would; every one it
    /// does hold is still true.
    ///
    /// # Panics
    ///
    /// Panics on an [`ENode::Ref`] to a unit no walk has admitted
    /// (`EGraph::admit_unit`): the node is only a key, and the graph holds
    /// nothing to say what it varies as.
    ///
    /// On a fold whose value bound varies with a coordinate or with the fold's
    /// own binder (`pixelflow_ir::fold::BoundRefusal::BoundVaries`, L3 of
    /// docs/language/2026-10-20-a-bound-is-a-value.md): the
    /// backstop behind `ExprArena::push_reduce`, for a node a rule or a caller
    /// built. It does not refuse bound arithmetic
    /// (docs/language/2026-10-20-a-bound-is-a-value.md §1.4): `HalveFold`'s
    /// `binder + s` in an inner bound is the rules' own, and exact.
    ///
    /// A class's fact is the meet of its members' variances, and a member's is
    /// computed from its children's facts, so a bound class inserted from an
    /// arena that passed `push_reduce`'s L3 passes here, and a merge only
    /// shrinks a fact, so a node the graph holds never becomes illegal. Two
    /// exceptions. The arena types a fold over a known empty range `CONST` and
    /// `ENode::variance` has no such arm, so a hand-built bound holding such a
    /// fold, which `push_reduce` admitted, may be refused here. And a fact is
    /// never repaired upward, so the class a lawful bound term lands in can be
    /// seeded wider than the term (`var_fact`'s law): the class it hash-conses
    /// into, or one `add` mints for it over a child that did. The class's fact
    /// is what is read. A rule's plan does not reach this panic by the second
    /// road: the graph asks whether it would refuse a plan's fold before making
    /// it (`NodeSink::refuses`), and the rewrite is declined.
    pub fn add(&mut self, node: ENode) -> EClassId {
        let node = self.canonical_node(node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        // L3's backstop (docs/language/2026-10-20-a-bound-is-a-value.md §3.2):
        // a value bound read off the classes it is made of, as `push_reduce`
        // reads one off its term, for a node a rule or a caller built. A fold
        // whose bounds are both literals has none and is not looked at. Read
        // before the graph changes: a refusal leaves it as it was. The test is
        // `has_a_value_bound`, spelled in place: through the helper this hot
        // call measured 4.5 M more instructions on the 29 programs, for the same
        // predicate.
        if let ENode::Reduce { header, .. } = &node
            && header.arity() > 1
        {
            self.refuse_a_varying_bound(&node);
        }
        // `node` is canonical, so each child is already its class's root; the
        // walk is the identity here and only hands the fact column the `Root`
        // it is indexed by. Read before the graph changes.
        let variance = node.variance(
            |key| self.ref_variance(key),
            |child| self.graph.columns.var_fact[self.graph.columns.root(child)],
        );
        self.mutation_counter += 1;
        let id = self.graph.columns.push(
            &mut self.saturation,
            Birth {
                node,
                const_fact: node.literal(),
                variance,
                shape: node.shape(),
                stamp: self.mutation_counter,
            },
        );
        #[cfg(feature = "provenance-journal")]
        if let Some(journal) = &mut self.journal {
            journal.record_birth(columns::tag_of(id));
        }
        self.memo.insert(node, id);
        id
    }

    /// Panic unless the value bounds of `node`, canonical and not in the memo,
    /// vary with neither a coordinate nor the fold's own binder (L3), read off
    /// the bound classes' facts. Out of line: `add` is the hottest call in the
    /// crate, and the message is formatted here.
    #[inline(never)]
    fn refuse_a_varying_bound(&self, node: &ENode) {
        if let Some(refused) = varying_bound(node, |class| self.variance(class)) {
            panic!("{refused}");
        }
    }

    /// Refused constant unions: the pairs of literals the graph declined to
    /// assert equal. Non-empty means some kernel hit the folding-vs-algebra
    /// collision — see the refusal block in [`EGraph::union`].
    #[must_use]
    pub fn refused_const_unions(&self) -> &[(Lit, Lit)] {
        &self.refused_const_unions
    }

    /// The binders `class` may depend on: a superset of what the function it
    /// denotes varies with, so `!variance(c).depends_on(b)` proves `c` is
    /// constant along `b` (see `Columns::var_fact` for the law and why it
    /// holds without repair).
    ///
    /// The side condition every loop-transformation rule over a fold asks —
    /// `⊕_i (c ⊗ f) = c ⊗ ⊕_i f` iff `i ∉ var(c)` — asked of the class
    /// rather than of one representative.
    #[must_use]
    pub(crate) fn variance(&self, class: EClassId) -> Variance {
        self.graph.columns.var_fact[self.graph.columns.root(class)]
    }

    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let a = self.graph.columns.root_mut(a);
        let b = self.graph.columns.root_mut(b);
        if a == b {
            return a.id();
        }
        // An e-class asserts "these are all equal", so a union placing two
        // numerically UNEQUAL constants in one class would be a proved
        // falsehood — and congruence closure amplifies any proved falsehood
        // into everything-equals-everything, with extraction tie-breaking
        // arbitrarily between unequal constants. The falsehoods are real:
        // constant folding computes f32-truths (`x + 2²⁴ = 2²⁴`) while the
        // algebraic rules compute ℝ-truths (`(x+y)−y = x`), each sound
        // alone. At the collision, REFUSE the union: an under-merged e-graph
        // is still sound — every class still holds only provably-equal
        // terms, and the cost is missed CSE at exactly the collision points
        // — while a false merge is unbounded. First prover keeps the class;
        // the refusal is journaled and reported so ill-conditioned kernels
        // are loud instead of subtly nondeterministic. (Folding constants in
        // f64 is the planned complement: it makes the folder agree with the
        // algebra at f32-observable scales, so collisions become
        // vanishingly rare and this valve becomes a pure detector.)
        //
        // Two literals are the same constant when `Lit` says so — a real by
        // its bits, so a NaN payload is unionable with itself, a mask by its
        // truth — and also when two reals are `==`: that admits ±0.0, and
        // signed zero selection is already platform-unspecified, so an
        // optimizer choosing a sign is within contract, same license as
        // commuting Min/Max over NaN. A mask and a real are never the same
        // constant, whatever lane they share.
        // Read the class-level fact, NOT the members: `rebuild` detaches a
        // class's members before performing congruence unions, so a
        // member-scanning guard is blind exactly when congruence closure runs.
        let (fa, fb) = (
            self.graph.columns.const_fact[a],
            self.graph.columns.const_fact[b],
        );
        if let (Some(la), Some(lb)) = (fa, fb) {
            let same_real = matches!((la, lb), (Lit::Real(x), Lit::Real(y)) if x == y);
            if la != lb && !same_real {
                let pair = (la, lb);
                if !self.refused_const_unions.contains(&pair) {
                    // Journal ALWAYS; print only on request. This fires
                    // during ordinary kernel compilation — two computation
                    // orders of one real quantity disagreeing by an ulp is
                    // enough — so unconditional stderr would spam every
                    // build. `refused_const_unions()` is the durable record.
                    if std::env::var_os("PIXELFLOW_REPORT_CONST_REFUSALS").is_some() {
                        std::eprintln!(
                            "pixelflow e-graph: refusing a union that would assert \
                             {la} = {lb} (lanes {:#010x} vs {:#010x}) — the rule set \
                             derived contradictory constants (f32 folding vs \
                             algebra at an ill-conditioned input); keeping the \
                             classes separate costs only missed CSE",
                            la.bits(),
                            lb.bits()
                        );
                    }
                    self.refused_const_unions.push(pair);
                }
                return a.min(b).id();
            }
        }
        let (parent, child) = if a < b { (a, b) } else { (b, a) };
        // Everything read at `child` happens before `link`, while it is still
        // a representative.
        self.saturation.merge_kinds(child, parent);
        self.graph.columns.absorb(child, parent);
        // The guard above proved the two facts agree (or that at most one
        // exists), so `or` is a merge, not a choice.
        if self.graph.columns.const_fact[parent].is_none() {
            self.graph.columns.const_fact[parent] = self.graph.columns.const_fact[child];
        }
        // Intersection, not a choice: see `var_fact`'s law.
        self.graph.columns.var_fact[parent] =
            self.graph.columns.var_fact[parent].intersection(self.graph.columns.var_fact[child]);
        self.graph.columns.link(child, parent);
        // `parent`'s own node list just changed (gained `child`'s nodes) —
        // see `class_is_dirty`/`DIRTY_TRACKING_MAX_DEPTH`.
        self.mutation_counter += 1;
        self.saturation.last_changed[parent] = self.mutation_counter;
        let slot = Slot::at(parent);
        if !self.saturation.queued[slot] {
            self.saturation.queued[slot] = true;
            self.worklist.push(slot);
        }
        #[cfg(feature = "provenance-journal")]
        if let Some(journal) = &mut self.journal {
            journal.record_union(parent.id(), child.id());
        }
        parent.id()
    }

    /// Rebuild the e-graph completely. Processes the entire worklist.
    pub fn rebuild(&mut self) {
        self.rebuild_budgeted(usize::MAX);
        debug_assert!(
            self.kind_index_covers_members(),
            "a live class holds a node whose kind its kind mask or the per-kind class set lacks"
        );
    }

    /// Process up to `budget` worklist items. Each item canonicalizes the
    /// stored nodes of one class and probes the memo with each canonical
    /// form: a miss records the form, and a hit in another class unions the
    /// two, or, when that union is refused, takes the node out of the class.
    /// Nothing is deduplicated: a node whose form another member already has
    /// stays, so a class can hold equal members (see [`Self::nodes`]).
    ///
    /// The graph is consistent after each item — partially rebuilt is safe.
    /// Unprocessed classes may have stale canonical forms (rule matching
    /// might miss some equivalences) but won't produce wrong results.
    ///
    /// Returns the number of worklist items remaining.
    pub fn rebuild_budgeted(&mut self, budget: usize) -> usize {
        let mut processed = 0;
        while processed < budget {
            let slot = match self.worklist.pop() {
                Some(slot) => slot,
                None => break,
            };
            processed += 1;
            // Cleared by the slot pushed at (not the class it now
            // canonicalizes to) — matches `union`'s guard, which sets it by
            // the same slot.
            self.saturation.queued[slot] = false;
            // The item's class as it is now. A union inside the item can
            // merge it away, and the item keeps this name on purpose: the
            // memo takes it for the nodes the item keeps, `existing != id`
            // compares against it, and only the write-back asks again.
            let id = self.graph.columns.root_of(slot);
            let mut detached = self.graph.columns.detach(id);
            let mut kept = Kept::new();
            while let Some(tag) = self.graph.columns.walk(&mut detached) {
                let node = self.canonical_node(*self.graph.columns.node(tag));
                // The stored names are state: the arena keeps the form
                // canonicalized now, retired or kept.
                self.graph.columns.set_node(tag, node);
                if let Some(&existing) = self.memo.get(&node) {
                    let existing = self.graph.columns.root(existing);
                    if existing != id {
                        // `union` keeps `min(id, existing)`, so this call
                        // can go either way, and the write-back at the
                        // bottom of the loop has to survive both:
                        //   - `id` survives: `union` splices `existing`'s
                        //     members onto `id`'s list, which `detach` just
                        //     emptied — so the write-back must splice after
                        //     them, not overwrite, or those are dropped.
                        //   - `existing` survives: `id` is merged away and
                        //     its list stops being reachable through `find`
                        //     — so the write-back must target `root(id)`,
                        //     which `write_back` resolves itself.
                        // This is a rebuild-time
                        // congruence-closure union, not a rule firing, so it
                        // carries no rule_idx in the provenance journal
                        // (no application bracket is open during `rebuild`).
                        self.union(id.id(), existing.id());

                        // `union` REFUSES a merge that would assert two
                        // unequal constants equal, and a refusal here needs
                        // handling that a rule-driven refusal does not: the
                        // node below would be pushed back into `id` while
                        // `memo` names `existing`, leaving ONE ENode in two
                        // contradictory classes with only one of them
                        // reachable by lookup — later matching or extraction
                        // could then give the same expression different
                        // values. Retire the node: it stays in the arena and
                        // in no class (law 7). `id` loses a node congruent to
                        // one it can no longer be merged with, which is
                        // under-merging: it costs CSE, not correctness.
                        if self.find(id.id()) != self.find(existing.id()) {
                            self.graph.columns.retire(tag);
                            continue;
                        }
                    }
                } else {
                    self.memo.insert(node, id.id());
                }
                self.graph.columns.keep(&mut kept, tag);
            }
            // After what the class `id` is in now holds: see
            // `Columns::write_back`.
            self.graph.columns.write_back(id, kept);
        }
        self.worklist.len()
    }

    /// Number of pending worklist items (classes needing rebuild).
    pub fn pending_rebuilds(&self) -> usize {
        self.worklist.len()
    }

    /// The nodes of the class `id` belongs to, in the order the graph gives
    /// them. The order is semantics, not layout: rules that return on the
    /// first matching node, the node substitution reads and the candidate
    /// extraction keeps on a tie all depend on it.
    ///
    /// - [`add`](Self::add) gives `[node]`.
    /// - [`union`](Self::union) gives the surviving class's nodes, then the
    ///   merged-away class's.
    /// - A [`rebuild`](Self::rebuild) item detaches the nodes of the class it
    ///   processes, and any union inside the item appends to the class as
    ///   `union` does. At the end the kept nodes, in the order they were
    ///   walked, follow whatever the class then holds: `[absorbed] ++ [kept]`
    ///   when the class survives its unions. When the class loses one, it is
    ///   the survivor's prior nodes, then any the survivor absorbs later in
    ///   the same item, then `[kept]`.
    ///
    /// A node whose canonical form collides with a class it cannot merge with
    /// leaves the class, and rebuild deduplicates nothing: two members may be
    /// equal.
    ///
    /// An iterator and nothing more: no length, no reverse, no slice, no
    /// index, so a caller depends on the order and not on how the members
    /// are stored. A count is `nodes(id).count()` and a position
    /// `nodes(id).nth(k)`; both walk the members. Extraction reads the node it
    /// chose by id (`FrozenGraph::node`), not by position. The members are a list
    /// threaded through the arena, so a walk is one dependent load per member.
    pub fn nodes(&self, id: EClassId) -> impl Iterator<Item = &ENode> + Clone + '_ {
        self.graph.nodes(id)
    }

    /// The stable `ENodeId` tags of the class's nodes, yielded in step with
    /// [`Self::nodes`]: the k-th tag names the k-th node. A tag is the node's
    /// id in this graph: the k-th tag is the id of the k-th node.
    pub fn tags(&self, id: EClassId) -> impl Iterator<Item = ENodeId> + Clone + '_ {
        self.graph.tags(id)
    }

    /// [`Self::nodes`] for a class that is already its own representative, as
    /// [`Self::candidate_classes`] yields: the rule scan's read, without the
    /// walk to the representative it has already done.
    fn nodes_of_canonical(&self, canonical: EClassId) -> impl Iterator<Item = &ENode> + Clone + '_ {
        debug_assert_eq!(
            self.find(canonical),
            canonical,
            "nodes_of_canonical is for a representative"
        );
        let class = self.graph.columns.root(canonical);
        self.graph
            .columns
            .members(class)
            .map(move |tag| self.graph.columns.node(tag))
    }

    /// The mutation log (origins, application log, union journal), or `None`
    /// for a graph built [`without_journal`](Self::without_journal). See the
    /// `provenance` module for details.
    pub fn provenance(&self) -> Option<&Provenance> {
        self.journal.as_ref()
    }

    /// The mutation log of a graph that keeps one: for a walk that has
    /// nothing to compute without it (the hindsight labeler, the research
    /// harnesses). [`Self::provenance`] is the question to ask first.
    ///
    /// # Panics
    ///
    /// On a graph built without a journal.
    #[cfg(feature = "provenance-journal")]
    pub fn journal(&self) -> &Provenance {
        self.journal.as_ref().expect(NO_JOURNAL)
    }

    /// The node `tag` names, if it is a member of `id`'s class now: `None`
    /// for a node another class holds, a retired node (law 7), or a tag past
    /// this graph's end. O(1).
    pub fn node_for_tag(&self, id: EClassId, tag: ENodeId) -> Option<&ENode> {
        self.graph.node_for_tag(id, tag)
    }

    /// Whether node `tag` is a member of a class now: false exactly for a
    /// retired node (law 7) and for a tag past this graph's end.
    pub fn is_member(&self, tag: ENodeId) -> bool {
        self.graph.columns.class_of(tag).is_some() && !self.graph.columns.is_retired(tag)
    }

    /// The operand classes node `tag` names as stored, or none for a node that
    /// is not a member (retired, or past the end): what the derivation walks
    /// read. O(1), where the scan it replaces visited every class.
    #[cfg(feature = "provenance-journal")]
    fn children_of(&self, tag: ENodeId) -> Vec<EClassId> {
        match self.is_member(tag) {
            true => self.graph.columns.node(tag).children(),
            false => Vec::new(),
        }
    }

    /// Compute the transitive rewrite-application ancestry of a set of
    /// chosen `(EClassId, ENodeId)` pairs (typically the nodes an extraction
    /// pass selected). See [`super::provenance::derivation_ancestors`] for
    /// the exact over-approximation made.
    #[cfg(feature = "provenance-journal")]
    pub fn derivation_ancestors(
        &self,
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        let tags_of = |class: EClassId| -> Vec<ENodeId> { self.tags(class).collect() };
        let children_of = |tag: ENodeId| -> Vec<EClassId> { self.children_of(tag) };
        super::provenance::derivation_ancestors(
            &tags_of,
            &children_of,
            self.journal(),
            chosen_nodes,
        )
    }

    /// The tightened counterpart to [`EGraph::derivation_ancestors`] — see
    /// [`super::provenance::derivation_ancestors_tight`] for exactly which
    /// over-approximation axes are narrowed and why the result is still safe
    /// (a subset of `derivation_ancestors`'s result, a superset of the
    /// strict node-on-path bound). Additive: does not change what
    /// `derivation_ancestors` computes, so both can be run on the same
    /// episode for comparison.
    #[cfg(feature = "provenance-journal")]
    pub fn derivation_ancestors_tight(
        &self,
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        self.derivation_ancestors_tight_from(chosen_nodes, chosen_nodes)
    }

    /// [`EGraph::derivation_ancestors_tight`] for a walk that starts from a
    /// SUBSET of the extraction — one application's chosen output node, say
    /// — while still pruning with the extraction's complete choice map. See
    /// [`super::provenance::derivation_ancestors_tight`] for why passing
    /// only the seeds as the choice map degrades the walk back to the loose
    /// bound for every class it descends into.
    #[cfg(feature = "provenance-journal")]
    pub fn derivation_ancestors_tight_from(
        &self,
        seeds: &[(EClassId, ENodeId)],
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        let tags_of = |class: EClassId| -> Vec<ENodeId> { self.tags(class).collect() };
        let children_of = |tag: ENodeId| -> Vec<EClassId> { self.children_of(tag) };
        let canonical_of = |class: EClassId| -> EClassId { self.find(class) };
        super::provenance::derivation_ancestors_tight(
            super::provenance::GraphView {
                tags_of: &tags_of,
                children_of: &children_of,
                canonical_of: &canonical_of,
            },
            self.journal(),
            super::provenance::TightWalk {
                seeds,
                chosen_nodes,
            },
        )
    }

    /// Get the number of e-class ids minted.
    ///
    /// Ids are never reclaimed — a union repoints a slot rather than freeing
    /// it — so this is the count [`Limits::minted`] bounds, not the number of
    /// live classes ([`Self::class_ids`] yields those).
    pub fn num_classes(&self) -> usize {
        self.graph.num_classes()
    }

    /// Iterate over all canonical e-class IDs.
    ///
    /// Returns an iterator of all e-class IDs that are canonical (i.e., roots
    /// of their union-find tree) and have at least one node. Read from the
    /// live set the member store keeps.
    pub fn class_ids(&self) -> impl Iterator<Item = EClassId> + '_ {
        self.graph.columns.live().iter()
    }

    /// The live classes a rule rooted in `roots` can match at, in ascending
    /// id order: every live class that has held a node of one of its kinds.
    ///
    /// A superset of the classes holding such a node now (see
    /// `SaturationColumns::kind_mask`), so a visited class is still checked node by
    /// node. [`RootShapes::Any`] names no kind and visits every live class.
    pub(super) fn candidate_classes(
        &self,
        roots: RootShapes,
    ) -> impl Iterator<Item = EClassId> + '_ {
        self.graph
            .columns
            .live()
            .iter_within(move |word| match roots {
                RootShapes::Any => u64::MAX,
                RootShapes::Kinds(kinds) => self.saturation.by_kind.word_of_any(kinds, word),
            })
    }

    /// Whether every member of every live class has its kind in the class's
    /// mask and the class in that kind's set: the superset law
    /// `candidate_classes` is sound by.
    fn kind_index_covers_members(&self) -> bool {
        self.class_ids().all(|id| {
            let root = self.graph.columns.root(id);
            self.nodes_of_canonical(id)
                .filter_map(ENode::shape)
                .all(|kind| {
                    self.saturation.kind_mask[root].contains(kind)
                        && self.saturation.by_kind.contains(kind, id)
                })
        })
    }

    /// Nodes minted and not retired (law 7), duplicates counted: every member
    /// of every live class.
    pub fn node_count(&self) -> usize {
        self.graph.columns.len() - self.graph.columns.retired_count()
    }

    pub fn contains_const(&self, id: EClassId, val: f32) -> bool {
        self.contains_lit(id, Lit::Real(val))
    }

    /// Whether `id` holds the literal `lit`, as [`ENode::is_lit`] reads it.
    pub fn contains_lit(&self, id: EClassId, lit: Lit) -> bool {
        self.nodes(id).any(|n| n.is_lit(lit))
    }

    /// Whether `canonical`'s forward neighborhood — itself, its direct
    /// children, and their children (see [`DIRTY_TRACKING_MAX_DEPTH`]) —
    /// contains anything that changed since `swept`, a
    /// [`Self::mutation_counter`] value from some rule's last complete scan.
    ///
    /// Reads only `EClassId`s and `last_changed` entries, never clones a
    /// node or calls a rule's `apply` — this is the cheap check
    /// the runner's rule scan uses to decide whether the
    /// expensive path (clone the class's nodes, run every rule against
    /// every one) is worth paying for at all. Canonicalizes every id it
    /// follows: `parent`/`child` fields on a stored node can lag a union
    /// that happened after the node was written, and `last_changed` is only
    /// ever indexed by canonical class slots.
    fn class_is_dirty(&self, canonical: EClassId, swept: u64) -> bool {
        let canonical = self.graph.columns.root(canonical);
        if self.saturation.last_changed[canonical] > swept {
            return true;
        }
        for tag in self.graph.columns.members(canonical) {
            let node = self.graph.columns.node(tag);
            for &child in node.children_slice() {
                let child = self.graph.columns.root(child);
                if self.saturation.last_changed[child] > swept {
                    return true;
                }
                if DIRTY_TRACKING_MAX_DEPTH < 2 {
                    continue;
                }
                for gtag in self.graph.columns.members(child) {
                    let gnode = self.graph.columns.node(gtag);
                    for &gchild in gnode.children_slice() {
                        if self.saturation.last_changed[self.graph.columns.root(gchild)] > swept {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Union `a` and `b`, reporting whether the graph actually changed.
    ///
    /// [`EGraph::union`] REFUSES merges that would assert two numerically
    /// unequal constants equal, so "the classes differed beforehand" is not
    /// evidence that anything merged. Counting a refusal as a change made
    /// saturation believe it had made progress, so it rebuilt and re-applied
    /// the same refused rewrite every iteration until its limit.
    fn union_counted(&mut self, a: EClassId, b: EClassId) -> usize {
        if self.find(a) == self.find(b) {
            return 0;
        }
        self.union(a, b);
        usize::from(self.find(a) == self.find(b))
    }

    /// Predict exactly how many new e-classes committing `action` would add
    /// to the graph **as it stands right now** — without mutating anything.
    ///
    /// Every [`RewriteAction`] arm builds a node shape of known structure
    /// (`Distribute`'s two products and a sum, `Instantiate`'s template,
    /// `Differentiate`'s one chain-rule step, …), and the e-graph is
    /// hash-consed, so the e-classes an application would add are exactly
    /// the shape's nodes that are not already in the memo. This walks that
    /// same shape — literally the same code [`EGraph::apply_action`] runs,
    /// via [`NodeSink`] — against a [`GrowthPredictor`] that probes the memo
    /// exactly as [`EGraph::add`] does on a hit, instead of inserting into
    /// it.
    ///
    /// Because `add` pushes each new node at the arena's length and returns
    /// that index as its class, "e-classes added" and "e-nodes added" are the
    /// same number here, which is why this returns one `usize` rather than a
    /// pair.
    ///
    /// **Only exact against the graph this is called on.** The runner sums
    /// the predictions of every action one scan finds, before any of them
    /// commits, and that sum bounds the batch's real growth in neither
    /// direction. Two actions that build the same node each count it, so the
    /// sum can over-count. And an earlier commit's `union` changes what a
    /// later action's nodes canonicalize to, while the memo keeps the names
    /// its keys were computed in until a rebuild re-inserts them, so a node
    /// predicted as a hit can miss and the sum can under-count. Pinned exact
    /// per single application by the `assert_eq!` in the runner's
    /// `apply_measured`, which calls this immediately before executing the
    /// same action for real.
    ///
    /// `pending` is the caller's scratch table for the walk: emptied here
    /// before use, so what a previous prediction left in it is never read, and
    /// owned by the caller so the table's allocation outlives the prediction.
    pub(crate) fn predicted_growth(
        &self,
        action: &RewriteAction,
        pending: &mut HashMap<ENode, EClassId>,
    ) -> usize {
        match action {
            RewriteAction::Union(_) => 0,
            RewriteAction::Create(node) => self.predict(pending, |s| {
                s.make(*node);
            }),
            RewriteAction::Instantiate {
                template,
                entry,
                bindings,
            } => self.predict(pending, |s| {
                instantiate_template(s, template.0.entry_at(*entry), bindings);
            }),
            RewriteAction::Distribute {
                outer,
                inner,
                a,
                b,
                c,
            } => self.predict(pending, |s| {
                distribute_shape(
                    s,
                    Distributive {
                        outer: *outer,
                        inner: *inner,
                    },
                    [*a, *b, *c],
                );
            }),
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => self.predict(pending, |s| {
                factor_shape(
                    s,
                    Distributive {
                        outer: *outer,
                        inner: *inner,
                    },
                    [*common, *unique_l, *unique_r],
                );
            }),
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => self.predict(pending, |s| {
                canonicalize_shape(s, *target, *inverse, *a, *b);
            }),
            RewriteAction::Associate { op, a, b, c } => self.predict(pending, |s| {
                associate_shape(s, *op, *a, *b, *c);
            }),
            RewriteAction::ReverseAssociate { op, a, b, c } => self.predict(pending, |s| {
                reverse_associate_shape(s, *op, *a, *b, *c);
            }),
            RewriteAction::OddParity { func, inner } => self.predict(pending, |s| {
                odd_parity_shape(s, *func, *inner);
            }),
            RewriteAction::AngleAddition {
                term1_op1,
                term1_op2,
                term2_op1,
                term2_op2,
                term2_sign,
                a,
                b,
            } => self.predict(pending, |s| {
                angle_addition_shape(
                    s,
                    AngleTerms {
                        term1: Product {
                            on_a: *term1_op1,
                            on_b: *term1_op2,
                        },
                        term2: Product {
                            on_a: *term2_op1,
                            on_b: *term2_op2,
                        },
                        term2_sign: *term2_sign,
                    },
                    [*a, *b],
                );
            }),
            RewriteAction::Homomorphism {
                func,
                target_op,
                a,
                b,
            } => self.predict(pending, |s| {
                homomorphism_shape(s, *func, *target_op, *a, *b);
            }),
            RewriteAction::PowerCombine { base, exp_a, exp_b } => self.predict(pending, |s| {
                power_combine_shape(s, *base, *exp_a, *exp_b);
            }),
            RewriteAction::ReverseAngleAddition { trig_op, a, b } => self.predict(pending, |s| {
                reverse_angle_addition_shape(s, *trig_op, *a, *b);
            }),
            RewriteAction::HalfAngleProduct { x } => self.predict(pending, |s| {
                half_angle_product_shape(s, *x);
            }),
            RewriteAction::Doubling { a } => self.predict(pending, |s| {
                doubling_shape(s, *a);
            }),
            RewriteAction::Halving { a } => self.predict(pending, |s| {
                halving_shape(s, *a);
            }),
            RewriteAction::PowerRecurrence { base, exponent } => self.predict(pending, |s| {
                power_recurrence_shape(s, *base, *exponent);
            }),
            RewriteAction::LogPower {
                log_op,
                base,
                exponent,
            } => self.predict(pending, |s| {
                log_power_shape(s, *log_op, *base, *exponent);
            }),
            RewriteAction::ExpandSquare { a, b } => self.predict(pending, |s| {
                expand_square_shape(s, *a, *b);
            }),
            RewriteAction::DiffOfSquares { a, b } => self.predict(pending, |s| {
                diff_of_squares_shape(s, *a, *b);
            }),
            RewriteAction::Differentiate { inner, var } => self.predict(pending, |s| {
                derivative_shape(s, inner, *var);
            }),
            RewriteAction::Plan(plan) => self.predict(pending, |s| {
                // A declined plan has minted what came before the refusal,
                // which is what the count reads.
                plan_shape(s, plan);
            }),
        }
    }

    /// Run `build` against a [`GrowthPredictor`] over this graph, with
    /// `pending` emptied first, and report how many e-classes it would have
    /// minted. See [`EGraph::predicted_growth`].
    fn predict(
        &self,
        pending: &mut HashMap<ENode, EClassId>,
        build: impl FnOnce(&mut GrowthPredictor<'_>),
    ) -> usize {
        pending.clear();
        let mut sink = GrowthPredictor {
            egraph: self,
            pending,
        };
        build(&mut sink);
        sink.pending.len()
    }

    /// Apply a rewrite action and return 1 if a union was made, 0 otherwise.
    ///
    /// Internal executor. Rule-driven callers must go through the runner's
    /// `commit` so the journal's attribution stays correct; this function
    /// itself has no notion of "which rule" — it only knows how to execute
    /// the `RewriteAction` variants.
    fn apply_action(&mut self, class_id: EClassId, action: RewriteAction) -> usize {
        match action {
            RewriteAction::Union(target_id) => self.union_counted(class_id, target_id),
            RewriteAction::Instantiate {
                template,
                entry,
                bindings,
            } => {
                let node = template.0.entry_at(entry);
                let result_id = instantiate_template(self, node, &bindings);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Create(new_node) => {
                let new_id = self.add(new_node);
                self.union_counted(class_id, new_id)
            }
            RewriteAction::Distribute {
                outer,
                inner,
                a,
                b,
                c,
            } => {
                let result_id = distribute_shape(self, Distributive { outer, inner }, [a, b, c]);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => {
                let result_id = factor_shape(
                    self,
                    Distributive { outer, inner },
                    [common, unique_l, unique_r],
                );
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => {
                let result_id = canonicalize_shape(self, target, inverse, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Associate { op, a, b, c } => {
                let result_id = associate_shape(self, op, a, b, c);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ReverseAssociate { op, a, b, c } => {
                let result_id = reverse_associate_shape(self, op, a, b, c);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::OddParity { func, inner } => {
                let result_id = odd_parity_shape(self, func, inner);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::AngleAddition {
                term1_op1,
                term1_op2,
                term2_op1,
                term2_op2,
                term2_sign,
                a,
                b,
            } => {
                let terms = AngleTerms {
                    term1: Product {
                        on_a: term1_op1,
                        on_b: term1_op2,
                    },
                    term2: Product {
                        on_a: term2_op1,
                        on_b: term2_op2,
                    },
                    term2_sign,
                };
                let result_id = angle_addition_shape(self, terms, [a, b]);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Homomorphism {
                func,
                target_op,
                a,
                b,
            } => {
                let result_id = homomorphism_shape(self, func, target_op, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::PowerCombine { base, exp_a, exp_b } => {
                let result_id = power_combine_shape(self, base, exp_a, exp_b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ReverseAngleAddition { trig_op, a, b } => {
                let result_id = reverse_angle_addition_shape(self, trig_op, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::HalfAngleProduct { x } => {
                let result_id = half_angle_product_shape(self, x);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Doubling { a } => {
                let result_id = doubling_shape(self, a);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Halving { a } => {
                let result_id = halving_shape(self, a);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::PowerRecurrence { base, exponent } => {
                let result_id = power_recurrence_shape(self, base, exponent);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::LogPower {
                log_op,
                base,
                exponent,
            } => {
                let result_id = log_power_shape(self, log_op, base, exponent);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ExpandSquare { a, b } => {
                let result_id = expand_square_shape(self, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::DiffOfSquares { a, b } => {
                let result_id = diff_of_squares_shape(self, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Differentiate { inner, var } => {
                let deriv_id = derivative_shape(self, &inner, var);
                self.union_counted(class_id, deriv_id)
            }
            RewriteAction::Plan(plan) => match plan_shape(self, &plan) {
                Some(built) => self.union_counted(class_id, built),
                None => 0,
            },
        }
    }
}

/// A fold node the graph will not hold: a value bound of it varies with a
/// coordinate or with the fold's own binder (L3 of
/// docs/language/2026-10-20-a-bound-is-a-value.md), judged on the fact of the
/// class the bound is. Displays as the panic [`EGraph::add`] raises.
#[derive(Debug)]
struct VaryingBound {
    fold: Fold,
    refusal: BoundRefusal,
}

impl core::fmt::Display for VaryingBound {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {:?}", self.refusal, self.fold)
    }
}

/// Whether `node` is a fold with a value bound: the only node L3 reads a class
/// for. A fold over literals has no bound to read and is not looked at.
#[inline]
fn has_a_value_bound(node: &ENode) -> bool {
    matches!(node, ENode::Reduce { header, .. } if header.arity() > 1)
}

/// The first value bound of the fold `node` that varies with a coordinate or
/// with the fold's own binder (L3), read off `fact`, the variance fact of the
/// class a bound is. `None` for a node that is not a fold with a value bound.
///
/// The one place the e-graph states the law over classes: [`EGraph::add`] asks
/// it of the graph's own facts, and the growth predictor of the facts `add`
/// would have seeded, so a prediction declines exactly the node an `add` would
/// refuse.
#[inline(never)]
fn varying_bound(node: &ENode, fact: impl Fn(EClassId) -> Variance) -> Option<VaryingBound> {
    let ENode::Reduce { header, children } = node else {
        return None;
    };
    let fold = header.fold();
    let (lo, hi) = header.attach(children[1..].iter().copied());
    [(BoundEnd::Lo, lo), (BoundEnd::Hi, hi)]
        .into_iter()
        .find_map(|(end, bound)| {
            let Bound::Value(class) = bound else {
                return None;
            };
            let refusal = fold.admits_bound(end, fact(class)).err()?;
            Some(VaryingBound { fold, refusal })
        })
}

// ---------------------------------------------------------------------------
// RewriteAction shapes: one definition per action, run twice
// ---------------------------------------------------------------------------
//
// Every `RewriteAction` (besides `Union`, which builds nothing) constructs a
// node shape of fixed structure and then unions the result with the matched
// class. `apply_action` needs to build that shape for real; `predicted_growth`
// needs to know how many of its nodes are new, without building anything.
// Rather than a second, hand-copied spelling of each shape for the predictor
// (exactly the drift `RewriteAction::Instantiate`'s own doc warns about —
// "two spellings of one shape"), every shape below is written once, generic
// over where its nodes land: a [`NodeSink`].

/// Materializes one [`ENode`], returning the e-class that holds it.
///
/// [`EGraph`] is a sink that actually inserts (`add`); [`GrowthPredictor`] is
/// a sink that only probes the memo. Every `RewriteAction`'s node-building
/// shape (below) is written once against this trait, so committing an
/// action for real and predicting how many e-classes it would add
/// ([`EGraph::predicted_growth`]) run the exact same code.
trait NodeSink {
    fn make(&mut self, node: ENode) -> EClassId;

    /// Whether `make(node)` would refuse it: a fold one of whose value bounds
    /// is a class whose fact varies with a coordinate or with the fold's own
    /// binder (L3 of docs/language/2026-10-20-a-bound-is-a-value.md), which
    /// [`EGraph::add`] panics on. Asked of a plan's fold before it is made, and
    /// answered after the memo lookup `add` makes first, so a node the graph
    /// already holds is never refused. Reads nothing for a fold over literals.
    fn refuses(&self, node: &ENode) -> bool;

    /// The graph the nodes land in, for the facts a shape is conditioned on.
    fn graph(&self) -> &EGraph;
}

impl NodeSink for EGraph {
    fn make(&mut self, node: ENode) -> EClassId {
        self.add(node)
    }

    fn refuses(&self, node: &ENode) -> bool {
        if !has_a_value_bound(node) {
            return false;
        }
        let node = self.canonical_node(*node);
        !self.memo.contains_key(&node)
            && varying_bound(&node, |class| self.variance(class)).is_some()
    }

    fn graph(&self) -> &EGraph {
        self
    }
}

/// A [`NodeSink`] that predicts what [`EGraph::add`] would do without
/// mutating the graph: canonicalize, then check the graph's real memo first
/// and a `pending` table second — the nodes this same prediction has
/// already "created" earlier in the same shape walk, keyed and deduplicated
/// the identical way the real memo would (e.g. `ExpandSquare`'s `a²` and
/// `b²` degenerate to one shared node when `a == b`, and this catches that
/// exactly as a real `add` would).
///
/// `pending`'s virtual ids are minted the same way [`EGraph::add`] mints
/// real ones — sequentially from `egraph.num_classes()` — so a shape that
/// references its own earlier output builds the identical composite node a
/// real run would look up. The table is probed, inserted into and counted,
/// and iterated by one reader, [`Self::fact`], which finds the node a pending
/// id names. Its answer does not depend on the order, since ids are unique, so
/// the capacity the table keeps between predictions is not observable; its
/// cost scales with that capacity, and is paid only for a fold's value bound.
struct GrowthPredictor<'a> {
    egraph: &'a EGraph,
    pending: &'a mut HashMap<ENode, EClassId>,
}

impl GrowthPredictor<'_> {
    /// [`EGraph::canonical_node`], except a child may be a `pending` id
    /// — an id this same prediction minted for a node earlier in the same
    /// shape walk, one that does not exist in the real graph
    /// (`id.index() >= egraph.num_classes()`) and so has no entry in the
    /// union-find at all. `EGraph::find` indexes it directly, so calling it on
    /// such an id is out of bounds rather than merely wrong. A pending id
    /// needs no lookup regardless: nothing has unioned a class that does
    /// not yet exist in the real graph, so it is already its own canonical
    /// form. Only a real child is resolved through the real union-find.
    fn canonicalize(&self, node: &mut ENode) {
        for child in node.children_slice_mut() {
            if child.index() < self.egraph.num_classes() {
                *child = self.egraph.find(*child);
            }
        }
    }

    /// The variance fact `class` holds, or would hold: the graph's own for a
    /// class that exists, and for one this prediction minted the fact
    /// [`EGraph::add`] would have seeded it with, the node's transfer over its
    /// children's facts.
    ///
    /// Asked only of a fold's value bound (`varying_bound`), so a fold over
    /// literals never pays the search for the node a `pending` id names.
    fn fact(&self, class: EClassId) -> Variance {
        if class.index() < self.egraph.num_classes() {
            return self.egraph.variance(class);
        }
        let minted = self
            .pending
            .iter()
            .find_map(|(node, &id)| (id == class).then_some(node))
            .expect("a pending class is one this prediction minted");
        minted.variance(
            |key| self.egraph.ref_variance(key),
            |child| self.fact(child),
        )
    }
}

impl NodeSink for GrowthPredictor<'_> {
    fn make(&mut self, mut node: ENode) -> EClassId {
        self.canonicalize(&mut node);
        if let Some(&id) = self.egraph.memo.get(&node) {
            return self.egraph.find(id);
        }
        if let Some(&id) = self.pending.get(&node) {
            return id;
        }
        let id = EClassId::from_index(self.egraph.num_classes() + self.pending.len());
        self.pending.insert(node, id);
        id
    }

    fn refuses(&self, node: &ENode) -> bool {
        if !has_a_value_bound(node) {
            return false;
        }
        let mut node = *node;
        self.canonicalize(&mut node);
        // The lookups first, as `EGraph::add` orders them: a node the graph
        // holds was admitted when it was added.
        !self.egraph.memo.contains_key(&node)
            && !self.pending.contains_key(&node)
            && varying_bound(&node, |class| self.fact(class)).is_some()
    }

    fn graph(&self) -> &EGraph {
        self.egraph
    }
}

/// The two operators of a distributive law: `outer` is the root of the term
/// the rule matched and `inner` the operator beneath it. `Distribute` matches
/// `*` over `+` and `Factor` matches `+` over `*`, so the roles are fixed by
/// the match, not by which operator is which.
#[derive(Clone, Copy)]
struct Distributive {
    outer: OpKind,
    inner: OpKind,
}

/// `Distribute`: `A * (B + C) -> A*B + A*C`, with `ops` = (outer `*`, inner
/// `+`) and `operands` = `[A, B, C]`.
fn distribute_shape<S: NodeSink>(
    sink: &mut S,
    ops: Distributive,
    operands: [EClassId; 3],
) -> EClassId {
    let [a, b, c] = operands;
    let ab_id = sink.make(ENode::apply(ops.outer, &[a, b]));
    let ac_id = sink.make(ENode::apply(ops.outer, &[a, c]));
    sink.make(ENode::apply(ops.inner, &[ab_id, ac_id]))
}

/// `Factor`: `A*B + A*C -> A * (B + C)`, with `ops` = (outer `+`, inner `*`)
/// and `operands` = `[A, B, C]`.
fn factor_shape<S: NodeSink>(sink: &mut S, ops: Distributive, operands: [EClassId; 3]) -> EClassId {
    let [common, unique_l, unique_r] = operands;
    let sum_id = sink.make(ENode::apply(ops.outer, &[unique_l, unique_r]));
    sink.make(ENode::apply(ops.inner, &[common, sum_id]))
}

/// `Canonicalize`: `Sub(a,b) -> Add(a, Neg(b))`.
fn canonicalize_shape<S: NodeSink>(
    sink: &mut S,
    target: OpKind,
    inverse: OpKind,
    a: EClassId,
    b: EClassId,
) -> EClassId {
    let inv_id = sink.make(ENode::apply(inverse, &[b]));
    sink.make(ENode::apply(target, &[a, inv_id]))
}

/// `Associate`: `(a op b) op c -> a op (b op c)`.
fn associate_shape<S: NodeSink>(
    sink: &mut S,
    op: OpKind,
    a: EClassId,
    b: EClassId,
    c: EClassId,
) -> EClassId {
    let bc_id = sink.make(ENode::apply(op, &[b, c]));
    sink.make(ENode::apply(op, &[a, bc_id]))
}

/// `ReverseAssociate`: `a op (b op c) -> (a op b) op c`.
fn reverse_associate_shape<S: NodeSink>(
    sink: &mut S,
    op: OpKind,
    a: EClassId,
    b: EClassId,
    c: EClassId,
) -> EClassId {
    let ab_id = sink.make(ENode::apply(op, &[a, b]));
    sink.make(ENode::apply(op, &[ab_id, c]))
}

/// `OddParity`: `Op(neg(x)) -> neg(Op(x))`.
fn odd_parity_shape<S: NodeSink>(sink: &mut S, func: OpKind, inner: EClassId) -> EClassId {
    let func_id = sink.make(ENode::apply(func, &[inner]));
    sink.make(ENode::apply(OpKind::Neg, &[func_id]))
}

/// `on_a(a) * on_b(b)`: one term of an angle-addition identity.
#[derive(Clone, Copy)]
struct Product {
    on_a: OpKind,
    on_b: OpKind,
}

/// The shape of an angle-addition identity: two products of single-argument
/// functions, joined by `term2_sign`.
#[derive(Clone, Copy)]
struct AngleTerms {
    term1: Product,
    term2: Product,
    term2_sign: crate::math::trig::Sign,
}

/// `AngleAddition`: `sin(a+b) -> sin(a)cos(b) + cos(a)sin(b)` (or the `cos`
/// case, sign-selected).
fn angle_addition_shape<S: NodeSink>(
    sink: &mut S,
    terms: AngleTerms,
    angles: [EClassId; 2],
) -> EClassId {
    use crate::math::trig::Sign;

    let [a, b] = angles;
    let AngleTerms {
        term1,
        term2,
        term2_sign,
    } = terms;

    let t1_left_id = sink.make(ENode::apply(term1.on_a, &[a]));
    let t1_right_id = sink.make(ENode::apply(term1.on_b, &[b]));
    let term1_id = sink.make(ENode::apply(OpKind::Mul, &[t1_left_id, t1_right_id]));

    let t2_left_id = sink.make(ENode::apply(term2.on_a, &[a]));
    let t2_right_id = sink.make(ENode::apply(term2.on_b, &[b]));
    let term2_id = sink.make(ENode::apply(OpKind::Mul, &[t2_left_id, t2_right_id]));

    match term2_sign {
        Sign::Plus => sink.make(ENode::apply(OpKind::Add, &[term1_id, term2_id])),
        Sign::Minus => sink.make(ENode::apply(OpKind::Sub, &[term1_id, term2_id])),
    }
}

/// `Homomorphism`: `f(a ⊕ b) -> f(a) ⊗ f(b)`.
fn homomorphism_shape<S: NodeSink>(
    sink: &mut S,
    func: OpKind,
    target_op: OpKind,
    a: EClassId,
    b: EClassId,
) -> EClassId {
    let func_a_id = sink.make(ENode::apply(func, &[a]));
    let func_b_id = sink.make(ENode::apply(func, &[b]));
    sink.make(ENode::apply(target_op, &[func_a_id, func_b_id]))
}

/// `PowerCombine`: `x^a * x^b -> x^(a+b)`.
fn power_combine_shape<S: NodeSink>(
    sink: &mut S,
    base: EClassId,
    exp_a: EClassId,
    exp_b: EClassId,
) -> EClassId {
    let sum_id = sink.make(ENode::apply(OpKind::Add, &[exp_a, exp_b]));
    sink.make(ENode::apply(OpKind::Pow, &[base, sum_id]))
}

/// `ReverseAngleAddition`: `sin(a)cos(b) + cos(a)sin(b) -> sin(a + b)` (or
/// the `cos` case).
fn reverse_angle_addition_shape<S: NodeSink>(
    sink: &mut S,
    trig_op: OpKind,
    a: EClassId,
    b: EClassId,
) -> EClassId {
    let sum_id = sink.make(ENode::apply(OpKind::Add, &[a, b]));
    sink.make(ENode::apply(trig_op, &[sum_id]))
}

/// `HalfAngleProduct`: `sin(x) * cos(x) -> sin(x + x) / 2`.
fn half_angle_product_shape<S: NodeSink>(sink: &mut S, x: EClassId) -> EClassId {
    let two_x_id = sink.make(ENode::apply(OpKind::Add, &[x, x]));
    let sin_2x_id = sink.make(ENode::apply(OpKind::Sin, &[two_x_id]));
    let two_id = sink.make(ENode::constant(2.0));
    sink.make(ENode::apply(OpKind::Div, &[sin_2x_id, two_id]))
}

/// `Doubling`: `a + a -> 2 * a`.
fn doubling_shape<S: NodeSink>(sink: &mut S, a: EClassId) -> EClassId {
    let two_id = sink.make(ENode::constant(2.0));
    sink.make(ENode::apply(OpKind::Mul, &[two_id, a]))
}

/// `Halving`: `2 * a -> a + a`.
fn halving_shape<S: NodeSink>(sink: &mut S, a: EClassId) -> EClassId {
    sink.make(ENode::apply(OpKind::Add, &[a, a]))
}

/// `PowerRecurrence`: `pow(x, n) -> x * pow(x, n-1)` for integer `n >= 3`.
fn power_recurrence_shape<S: NodeSink>(sink: &mut S, base: EClassId, exponent: i32) -> EClassId {
    let n_minus_1_id = sink.make(ENode::constant((exponent - 1) as f32));
    let pow_id = sink.make(ENode::apply(OpKind::Pow, &[base, n_minus_1_id]));
    sink.make(ENode::apply(OpKind::Mul, &[base, pow_id]))
}

/// `LogPower`: `log(pow(x, n)) -> n * log(x)`.
fn log_power_shape<S: NodeSink>(
    sink: &mut S,
    log_op: OpKind,
    base: EClassId,
    exponent: EClassId,
) -> EClassId {
    let log_x_id = sink.make(ENode::apply(log_op, &[base]));
    sink.make(ENode::apply(OpKind::Mul, &[exponent, log_x_id]))
}

/// `ExpandSquare`: `(a+b)² -> a² + 2ab + b²`.
fn expand_square_shape<S: NodeSink>(sink: &mut S, a: EClassId, b: EClassId) -> EClassId {
    let a2_id = sink.make(ENode::apply(OpKind::Mul, &[a, a]));
    let b2_id = sink.make(ENode::apply(OpKind::Mul, &[b, b]));
    let ab_id = sink.make(ENode::apply(OpKind::Mul, &[a, b]));
    let two_id = sink.make(ENode::constant(2.0));
    let two_ab_id = sink.make(ENode::apply(OpKind::Mul, &[two_id, ab_id]));
    let sum1_id = sink.make(ENode::apply(OpKind::Add, &[a2_id, two_ab_id]));
    sink.make(ENode::apply(OpKind::Add, &[sum1_id, b2_id]))
}

/// `DiffOfSquares`: `a² - b² -> (a+b)(a-b)`.
fn diff_of_squares_shape<S: NodeSink>(sink: &mut S, a: EClassId, b: EClassId) -> EClassId {
    let sum_id = sink.make(ENode::apply(OpKind::Add, &[a, b]));
    let diff_id = sink.make(ENode::apply(OpKind::Sub, &[a, b]));
    sink.make(ENode::apply(OpKind::Mul, &[sum_id, diff_id]))
}

/// Materialize `template`'s subtree at `id`, bottom-up, reading each
/// metavariable leaf from `bindings` rather than creating a node for it —
/// the same convention every hand-written multi-node shape above uses, just
/// driven by a runtime pattern instead of a fixed shape.
///
/// # Panics
///
/// On a `Param`/`Buffer` node (a rewrite RHS template must never contain
/// either), on an `OpKind` that `ops::template_may_name` refuses (a template
/// naming something no rule may name is a bug in the rule, not a runtime
/// condition), on a template node whose child count is not its op's arity
/// (`Operands::new`), or on a metavariable with no binding.
fn instantiate_template<S: NodeSink>(
    sink: &mut S,
    node: pixelflow_ir::Node<'_, pixelflow_ir::expr::ExprData>,
    bindings: &[EClassId],
) -> EClassId {
    use pixelflow_ir::expr::ExprData;

    match *node {
        ExprData::Var(mv) => {
            let mv = mv as usize;
            assert!(
                mv < bindings.len(),
                "instantiate_template: metavariable {mv} has no binding \
                 ({} supplied) — a TemplateRewrite construction bug, since \
                 apply() refuses to instantiate a binding it cannot fill",
                bindings.len()
            );
            bindings[mv]
        }
        ExprData::Const(lit) => sink.make(ENode::Const(lit)),
        ExprData::Param(p) => {
            panic!("instantiate_template: Param({p}) in a rewrite RHS template")
        }
        ExprData::Buffer(b) => {
            panic!(
                "instantiate_template: Buffer({}) in a rewrite RHS template",
                b.0
            )
        }
        ExprData::Uniform(u) => {
            panic!(
                "instantiate_template: Uniform({}) in a rewrite RHS template",
                u.0
            )
        }
        ExprData::Ref(k) => {
            panic!("instantiate_template: Ref({k:?}) in a rewrite RHS template")
        }
        ExprData::Reduce(header) => {
            let mut kids = node
                .children()
                .map(|c| instantiate_template(sink, c, bindings));
            let body = kids
                .next()
                .expect("a Reduce template has its body as its first child");
            let (lo, hi) = header.attach(kids);
            sink.make(ENode::reduce(header.fold(), lo, hi, body))
        }
        ExprData::Op(kind) => {
            assert!(
                ops::template_may_name(kind),
                "instantiate_template: a template cannot name {kind:?}"
            );
            let children: Vec<EClassId> = node
                .children()
                .map(|c| instantiate_template(sink, c, bindings))
                .collect();
            sink.make(ENode::apply(kind, &children))
        }
    }
}

/// Build a [`Plan`](super::fold_rules::Plan) — every fold and integration
/// rule's right-hand side — and return the class of its root.
///
/// The plan arrives already computed because computing it needs to *read*
/// the graph (walking a body's classes to substitute a binder, recognizing an
/// integrand), which a [`NodeSink`] cannot do; the rule does that half and
/// this replays it, node by node in build order.
///
/// `None` when a fold of the plan has a value bound whose class fact varies
/// with a coordinate or with the fold's own binder (L3 of
/// docs/language/2026-10-20-a-bound-is-a-value.md), where [`EGraph::add`] would
/// panic. The rule chose the bound's members by their own variance, and that is
/// the bound as a term; the fact of the class it is is the graph's to say, and
/// a fact is never repaired upward (`var_fact`), so the class can be seeded
/// wider than the term: because hash-consing resolved the bound to a class that
/// already exists, or because the bound is a class this plan minted over a
/// child that did. The nodes built before the refusal stay, as any `add`
/// leaves them, and the caller makes no union. A prediction declines at the
/// same node, so it counts the same classes.
fn plan_shape<S: NodeSink>(sink: &mut S, plan: &super::fold_rules::Plan) -> Option<EClassId> {
    use super::fold_rules::{HeadNode, HeadRef};
    let mut planned: Vec<EClassId> = Vec::with_capacity(plan.nodes.len());
    let resolve = |r: HeadRef, planned: &[EClassId]| match r {
        HeadRef::Plan(i) => planned[i as usize],
        HeadRef::Class(c) => c,
    };
    for entry in &plan.nodes {
        let id = match entry {
            HeadNode::Const(lit) => sink.make(ENode::Const(*lit)),
            HeadNode::Op { op, children } => {
                let operands: Vec<EClassId> =
                    children.iter().map(|c| resolve(*c, &planned)).collect();
                sink.make(ENode::apply(*op, &operands))
            }
            HeadNode::Reduce { fold, lo, hi, body } => {
                let node = ENode::reduce(
                    *fold,
                    lo.map(|h| resolve(h, &planned)),
                    hi.map(|h| resolve(h, &planned)),
                    resolve(*body, &planned),
                );
                if sink.refuses(&node) {
                    return None;
                }
                sink.make(node)
            }
            HeadNode::Shift {
                direction,
                count,
                operand,
            } => sink.make(ENode::Shift {
                direction: *direction,
                count: *count,
                operand: resolve(*operand, &planned),
            }),
        };
        planned.push(id);
    }
    Some(resolve(plan.root, &planned))
}

/// Build the e-class of the derivative of `inner` with respect to variable
/// `var`, one chain-rule step deep. Sub-expressions are wrapped in fresh
/// `Dwrt` nodes so equality saturation continues the expansion; the leaf
/// cases (`Var`, `Const`) terminate it. Operators whose derivative is not
/// (yet) known reconstruct the original `Dwrt`, leaving it to survive
/// saturation as the jet fallback.
fn derivative_shape<S: NodeSink>(sink: &mut S, inner: &ENode, var: u8) -> EClassId {
    let (op, children) = match inner {
        ENode::Const(_) => return sink.make(ENode::constant(0.0)),
        ENode::Var(i) => {
            return sink.make(ENode::constant(if *i == var { 1.0 } else { 0.0 }));
        }
        // A Buffer leaf is not a value — it only ever appears as Gather's
        // first child, and no rewrite builds `Dwrt(buffer)`. Reaching one
        // here means the graph is malformed; fail loudly. (`Dwrt(gather)`
        // is the Op arm below: Gather has no derivative table entry, so it
        // reconstructs the Dwrt as the jet fallback.)
        ENode::Buffer(decl) => {
            panic!("derivative_shape: Dwrt applied to a Buffer leaf ({decl:?})")
        }
        // Invariant across the lattice: ∂u/∂x = 0, as for a constant.
        ENode::Uniform(_) => return sink.make(ENode::constant(0.0)),
        // Likewise ∂p/∂x = 0: a slot does not vary along the axis, as a
        // uniform does not.
        ENode::Param(_) => return sink.make(ENode::constant(0.0)),
        // A unit that does not vary along the axis has derivative zero, as
        // a uniform does. One that does keeps its `Dwrt`: a name has no
        // structure to differentiate. The unit walk links every reference
        // a written `Dwrt` reaches before insertion, so the chain rule runs
        // here on every derivative the program wrote; a `Dwrt` over a unit
        // can only come of rewriting a class it shares, and one extraction
        // keeps is lowered by `lower_dwrt` on the linked body, after the
        // link, like any other `Dwrt` the rules did not reach.
        ENode::Ref { key } => {
            if !sink.graph().ref_variance(*key).depends_on(var) {
                return sink.make(ENode::constant(0.0));
            }
            let var_const = sink.make(ENode::constant(var as f32));
            let inner = sink.make(*inner);
            return sink.make(ENode::apply(OpKind::Dwrt, &[inner, var_const]));
        }
        ENode::Op { op, children } => (*op, *children),
        // `d(⊕_k f) = ⊕_k d(f)` is linearity, which holds for `Σ` and for
        // nothing else in the monoid set — `Π` wants the product rule, and
        // `min`/`max` are selections. It is a rule this table does not have,
        // so the `Dwrt` is reconstructed below and survives as the fallback,
        // exactly as `Gather` does. A shift has no derivative rule either
        // (it moves bits, which have no slope), and is kept the same way.
        ENode::Reduce { .. } | ENode::Shift { .. } => {
            let var_const = sink.make(ENode::constant(var as f32));
            let inner = sink.make(*inner);
            return sink.make(ENode::apply(OpKind::Dwrt, &[inner, var_const]));
        }
    };

    let var_const = sink.make(ENode::constant(var as f32));
    // d(child)/dvar as a fresh Dwrt node (saturation expands it later).
    let dwrt = |s: &mut S, c: EClassId| s.make(ENode::apply(OpKind::Dwrt, &[c, var_const]));
    let op2 = |s: &mut S, o: OpKind, a: EClassId, b: EClassId| s.make(ENode::apply(o, &[a, b]));
    let un = |s: &mut S, o: OpKind, a: EClassId| s.make(ENode::apply(o, &[a]));
    let cst = |s: &mut S, v: f32| s.make(ENode::constant(v));

    match op {
        // Linearity: D(a + b) = D(a) + D(b); D(a - b) = D(a) - D(b).
        OpKind::Add | OpKind::Sub => {
            let da = dwrt(sink, children[0]);
            let db = dwrt(sink, children[1]);
            op2(sink, op, da, db)
        }
        OpKind::Neg => {
            let da = dwrt(sink, children[0]);
            un(sink, OpKind::Neg, da)
        }
        // Product rule: D(a*b) = D(a)*b + a*D(b).
        OpKind::Mul => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let t1 = op2(sink, OpKind::Mul, da, b);
            let t2 = op2(sink, OpKind::Mul, a, db);
            op2(sink, OpKind::Add, t1, t2)
        }
        // Fused multiply-add a*b + c: D = D(a)*b + a*D(b) + D(c).
        OpKind::MulAdd => {
            let (a, b, c) = (children[0], children[1], children[2]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let dc = dwrt(sink, c);
            let t1 = op2(sink, OpKind::Mul, da, b);
            let t2 = op2(sink, OpKind::Mul, a, db);
            let prod = op2(sink, OpKind::Add, t1, t2);
            op2(sink, OpKind::Add, prod, dc)
        }
        // Quotient rule: D(a/b) = (D(a)*b - a*D(b)) / (b*b).
        OpKind::Div => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let t1 = op2(sink, OpKind::Mul, da, b);
            let t2 = op2(sink, OpKind::Mul, a, db);
            let num = op2(sink, OpKind::Sub, t1, t2);
            let den = op2(sink, OpKind::Mul, b, b);
            op2(sink, OpKind::Div, num, den)
        }
        // d(sqrt u) = 0.5 * rsqrt(u) * u'.
        OpKind::Sqrt => {
            let u = children[0];
            let du = dwrt(sink, u);
            let half = cst(sink, 0.5);
            let rs = un(sink, OpKind::Rsqrt, u);
            let factor = op2(sink, OpKind::Mul, half, rs);
            op2(sink, OpKind::Mul, factor, du)
        }
        // d(recip u) = -u' / (u*u).
        OpKind::Recip => {
            let u = children[0];
            let du = dwrt(sink, u);
            let ndu = un(sink, OpKind::Neg, du);
            let u2 = op2(sink, OpKind::Mul, u, u);
            op2(sink, OpKind::Div, ndu, u2)
        }
        // d(|u|) = (u / |u|) * u'.
        OpKind::Abs => {
            let u = children[0];
            let du = dwrt(sink, u);
            let au = un(sink, OpKind::Abs, u);
            let sign = op2(sink, OpKind::Div, u, au);
            op2(sink, OpKind::Mul, sign, du)
        }
        // d(rsqrt u) = -0.5 * rsqrt(u) * recip(u) * u'.
        OpKind::Rsqrt => {
            let u = children[0];
            let du = dwrt(sink, u);
            let neg_half = cst(sink, -0.5);
            let rs = un(sink, OpKind::Rsqrt, u);
            let rc = un(sink, OpKind::Recip, u);
            let t = op2(sink, OpKind::Mul, rs, rc);
            let factor = op2(sink, OpKind::Mul, neg_half, t);
            op2(sink, OpKind::Mul, factor, du)
        }
        // Piecewise: derivative of the branch the primal takes. Masks and
        // tie behavior mirror Jet2 (and the runtime `lower_dwrt` pass).
        OpKind::Min => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let mask = op2(sink, OpKind::Lt, a, b);
            sink.make(ENode::apply(OpKind::If, &[mask, da, db]))
        }
        OpKind::Max => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let mask = op2(sink, OpKind::Gt, a, b);
            sink.make(ENode::apply(OpKind::If, &[mask, da, db]))
        }
        // Blend the branch derivatives on the primal mask; the mask itself
        // is not differentiated.
        OpKind::If => {
            let (m, t, f) = (children[0], children[1], children[2]);
            let dt = dwrt(sink, t);
            let df = dwrt(sink, f);
            sink.make(ENode::apply(OpKind::If, &[m, dt, df]))
        }
        // Masks and rounding are step functions: zero almost everywhere.
        OpKind::Lt
        | OpKind::Le
        | OpKind::Gt
        | OpKind::Ge
        | OpKind::Eq
        | OpKind::Ne
        | OpKind::Floor
        | OpKind::Ceil
        | OpKind::Round => sink.make(ENode::constant(0.0)),
        // d(sin u) = cos(u) * u'.
        OpKind::Sin => {
            let u = children[0];
            let du = dwrt(sink, u);
            let c = un(sink, OpKind::Cos, u);
            op2(sink, OpKind::Mul, c, du)
        }
        // d(cos u) = -sin(u) * u'.
        OpKind::Cos => {
            let u = children[0];
            let du = dwrt(sink, u);
            let s = un(sink, OpKind::Sin, u);
            let ns = un(sink, OpKind::Neg, s);
            op2(sink, OpKind::Mul, ns, du)
        }
        // d(tan u) = u' / cos(u)^2.
        OpKind::Tan => {
            let u = children[0];
            let du = dwrt(sink, u);
            let c = un(sink, OpKind::Cos, u);
            let c2 = op2(sink, OpKind::Mul, c, c);
            op2(sink, OpKind::Div, du, c2)
        }
        // d(atan u) = u' / (1 + u*u).
        OpKind::Atan => {
            let u = children[0];
            let du = dwrt(sink, u);
            let one = cst(sink, 1.0);
            let u2 = op2(sink, OpKind::Mul, u, u);
            let den = op2(sink, OpKind::Add, one, u2);
            op2(sink, OpKind::Div, du, den)
        }
        // d(asin u) = u' / sqrt(1 - u*u).
        OpKind::Asin => {
            let u = children[0];
            let du = dwrt(sink, u);
            let one = cst(sink, 1.0);
            let u2 = op2(sink, OpKind::Mul, u, u);
            let diff = op2(sink, OpKind::Sub, one, u2);
            let s = un(sink, OpKind::Sqrt, diff);
            op2(sink, OpKind::Div, du, s)
        }
        // d(acos u) = -u' / sqrt(1 - u*u).
        OpKind::Acos => {
            let u = children[0];
            let du = dwrt(sink, u);
            let one = cst(sink, 1.0);
            let u2 = op2(sink, OpKind::Mul, u, u);
            let diff = op2(sink, OpKind::Sub, one, u2);
            let s = un(sink, OpKind::Sqrt, diff);
            let q = op2(sink, OpKind::Div, du, s);
            un(sink, OpKind::Neg, q)
        }
        // d(exp u) = exp(u) * u'.
        OpKind::Exp => {
            let u = children[0];
            let du = dwrt(sink, u);
            let e = un(sink, OpKind::Exp, u);
            op2(sink, OpKind::Mul, e, du)
        }
        // d(2^u) = 2^u * ln2 * u'.
        OpKind::Exp2 => {
            let u = children[0];
            let du = dwrt(sink, u);
            let e = un(sink, OpKind::Exp2, u);
            let ln2 = cst(sink, core::f32::consts::LN_2);
            let factor = op2(sink, OpKind::Mul, e, ln2);
            op2(sink, OpKind::Mul, factor, du)
        }
        // d(ln u) = u' / u.
        OpKind::Ln => {
            let u = children[0];
            let du = dwrt(sink, u);
            op2(sink, OpKind::Div, du, u)
        }
        // d(log2 u) = u' / (u * ln2).
        OpKind::Log2 => {
            let u = children[0];
            let du = dwrt(sink, u);
            let ln2 = cst(sink, core::f32::consts::LN_2);
            let den = op2(sink, OpKind::Mul, u, ln2);
            op2(sink, OpKind::Div, du, den)
        }
        // d(log10 u) = u' / (u * ln10).
        OpKind::Log10 => {
            let u = children[0];
            let du = dwrt(sink, u);
            let ln10 = cst(sink, core::f32::consts::LN_10);
            let den = op2(sink, OpKind::Mul, u, ln10);
            op2(sink, OpKind::Div, du, den)
        }
        // d(atan2(y, x)) = (x*y' - y*x') / (x² + y²).
        OpKind::Atan2 => {
            let (y, x) = (children[0], children[1]);
            let dy = dwrt(sink, y);
            let dx = dwrt(sink, x);
            let t1 = op2(sink, OpKind::Mul, x, dy);
            let t2 = op2(sink, OpKind::Mul, y, dx);
            let num = op2(sink, OpKind::Sub, t1, t2);
            let x2 = op2(sink, OpKind::Mul, x, x);
            let y2 = op2(sink, OpKind::Mul, y, y);
            let den = op2(sink, OpKind::Add, x2, y2);
            op2(sink, OpKind::Div, num, den)
        }
        // d(f^g) = f^g * (g'*ln f + g*f'/f)  (Jet2's rule).
        OpKind::Pow => {
            let (f, g) = (children[0], children[1]);
            let df = dwrt(sink, f);
            let dg = dwrt(sink, g);
            let lnf = un(sink, OpKind::Ln, f);
            let t1 = op2(sink, OpKind::Mul, dg, lnf);
            let g_over_f = op2(sink, OpKind::Div, g, f);
            let t2 = op2(sink, OpKind::Mul, g_over_f, df);
            let inner = op2(sink, OpKind::Add, t1, t2);
            let p = op2(sink, OpKind::Pow, f, g);
            op2(sink, OpKind::Mul, p, inner)
        }
        // Unknown derivative: reconstruct the Dwrt and let it survive.
        _ => {
            let reconstructed = sink.make(*inner);
            dwrt(sink, reconstructed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::cost::CostModel;
    use crate::egraph::extract::extract_dag;
    use crate::egraph::provenance::{ApplicationId, Origin};
    use crate::egraph::rewrite::Rewrite;
    use crate::egraph::rules::RuleSet;
    use crate::egraph::saturate::SaturationConfig;
    use crate::math::algebra::{
        AddNeg, Annihilator, Cancellation, Canonicalize, Commutative, Distributive, Identity,
        InverseAnnihilation, Involution, MulRecip,
    };
    use pixelflow_ir::{Kernel, KernelStore};

    /// `class`'s members in order, to compare with a literal.
    fn members(eg: &EGraph, class: EClassId) -> Vec<ENode> {
        eg.nodes(class).copied().collect()
    }

    /// What a rebuild owes a caller whatever the store's layout: every id
    /// `add` handed out is a member of `find(id)`'s class, still the node it
    /// was born as (up to its operands' names), or retired, and no other class
    /// lists it. `added` is every node the recipe added, with the id `add`
    /// returned. Retired nodes are read from the store's retired bit. Called
    /// once the recipe's `rebuild` has emptied the worklist.
    fn assert_every_id_is_a_member(eg: &EGraph, added: &[(EClassId, ENode)]) {
        assert_eq!(
            added.len(),
            eg.num_classes(),
            "the recipe lists every id it minted"
        );
        for &(id, node) in added {
            let tag = ENodeId(id.index() as u64);
            match (
                eg.node_for_tag(eg.find(id), tag),
                eg.graph.columns.is_retired(tag),
            ) {
                (Some(member), false) => assert_eq!(
                    eg.canonical_node(*member),
                    eg.canonical_node(node),
                    "id {} holds the node it was born as",
                    id.index()
                ),
                (None, true) => {}
                (member, retired) => panic!(
                    "id {} must be a member of find(id) or retired, exactly one: \
                     member {member:?}, retired {retired}",
                    id.index()
                ),
            }
        }
        let listed: usize = eg.class_ids().map(|class| eg.tags(class).count()).sum();
        assert_eq!(
            listed,
            eg.num_classes() - eg.graph.columns.retired_count(),
            "each id is listed by exactly one class, or retired"
        );
        assert_eq!(eg.pending_rebuilds(), 0, "the recipe ran the worklist dry");
    }

    /// The canonical copy of a node names the surviving class of every operand
    /// the node has, whichever variant holds it: an operation's operands, a
    /// shift's operand and a fold's body. The graph is not rebuilt between
    /// the union and the reads, which is the state an observer can find it in.
    #[test]
    fn a_canonical_node_reads_every_operand_through_find() {
        use pixelflow_ir::ShiftCount;
        use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};

        let mut eg = EGraph::new();
        let kept = eg.add(ENode::Var(0));
        let merged = eg.add(ENode::Var(1));
        let survivor = eg.union(kept, merged);
        let fold = Fold::new(Monoid::SUM, Binder::from_slot(0).expect("a live binder"));
        let sum_to_four = |body| ENode::reduce(fold, Bound::Lit(0), Bound::Lit(4), body);
        let shift = |operand| ENode::Shift {
            direction: crate::egraph::ShiftDirection::Right,
            count: ShiftCount::from_literal(8.0).expect("in range"),
            operand,
        };
        let stale_and_canonical = [
            (
                ENode::apply(OpKind::Add, &[merged, kept]),
                ENode::apply(OpKind::Add, &[survivor, survivor]),
            ),
            (shift(merged), shift(survivor)),
            (sum_to_four(merged), sum_to_four(survivor)),
        ];
        for (stale, canonical) in stale_and_canonical {
            assert_eq!(eg.canonical_node(stale), canonical);
        }
    }

    /// A rebuild item whose class survives the union its own node causes
    /// loses nothing. The item is class 3 (`Neg(c)`); after `union(b, c)` its
    /// node canonicalizes to `Neg(b)`, which the memo names class 4, and
    /// 3 < 4, so the item absorbs class 4 after detaching its own members.
    /// (The order that gives is pinned by
    /// `a_rebuild_item_that_survives_gives_absorbed_then_kept`.) Stated as the
    /// law rather than as which marker survived: every id is a member of
    /// `find(id)`'s class, or retired, and listed once.
    #[test]
    fn rebuild_budgeted_does_not_drop_nodes_when_current_class_survives_union() {
        let mut eg = EGraph::new();

        let a = eg.add(ENode::Var(0)); // class 0
        let b = eg.add(ENode::Var(1)); // class 1
        let c = eg.add(ENode::Var(2)); // class 2
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 3
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 4
        let d = eg.add(ENode::Var(3)); // class 5
        let marker = eg.add(ENode::Var(9)); // class 6

        assert_eq!(nc.index(), 3, "test assumes nc is class 3");
        assert_eq!(nb.index(), 4, "test assumes nb is class 4");

        eg.union(nc, d); // enqueues class 3
        assert_eq!(eg.pending_rebuilds(), 1);
        eg.union(nb, marker); // class 4 gains a node no other class has
        eg.union(b, c); // `Neg(c)` now canonicalizes to `Neg(b)`
        eg.rebuild();

        assert_eq!(
            eg.find(nc),
            eg.find(nb),
            "nc and nb should have been unioned via the canonicalization collision"
        );
        assert_eq!(eg.find(nb), nc, "the item survived");
        assert_every_id_is_a_member(
            &eg,
            &[
                (a, ENode::Var(0)),
                (b, ENode::Var(1)),
                (c, ENode::Var(2)),
                (nc, ENode::apply(OpKind::Neg, &[c])),
                (nb, ENode::apply(OpKind::Neg, &[b])),
                (d, ENode::Var(3)),
                (marker, ENode::Var(9)),
            ],
        );
    }

    /// The mirror case: a rebuild item whose class loses the union its own
    /// node causes. The item is class 4 (`Neg(c)`); after `union(b, c)` its
    /// node canonicalizes to `Neg(b)`, which the memo names class 3, and
    /// 3 < 4, so class 4 is merged away mid-item. Its kept members belong to
    /// `find(4)`: written back to the item's own id they would sit where
    /// `find` no longer leads. (The order is pinned by
    /// `a_rebuild_item_that_loses_gives_prior_then_later_absorbed_then_kept`.)
    /// Stated as the law: every id is a member of `find(id)`'s class, or
    /// retired, and listed once.
    #[test]
    fn rebuild_budgeted_does_not_orphan_nodes_when_current_class_is_merged_away() {
        let mut eg = EGraph::new();

        let a = eg.add(ENode::Var(0)); // class 0
        let b = eg.add(ENode::Var(1)); // class 1
        let c = eg.add(ENode::Var(2)); // class 2
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 3
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 4
        let marker = eg.add(ENode::Var(9)); // class 5

        assert_eq!(nb.index(), 3, "test assumes nb is class 3");
        assert_eq!(nc.index(), 4, "test assumes nc is class 4");

        eg.union(nc, marker); // enqueues class 4, which gains a node of its own
        assert_eq!(eg.pending_rebuilds(), 1);
        eg.union(b, c); // `Neg(c)` now canonicalizes to `Neg(b)`
        eg.rebuild();

        // nb (3) is the surviving parent: `union(4, 3)` keeps the lower id.
        let surviving = eg.find(nc);
        assert_eq!(
            surviving,
            eg.find(nb),
            "nc and nb should have been unioned via the canonicalization collision"
        );
        assert_eq!(
            surviving, nb,
            "test assumes `existing` (nb, class 3) is the surviving parent, \
             i.e. the `existing.0 < id.0` branch"
        );

        assert_every_id_is_a_member(
            &eg,
            &[
                (a, ENode::Var(0)),
                (b, ENode::Var(1)),
                (c, ENode::Var(2)),
                (nb, ENode::apply(OpKind::Neg, &[b])),
                (nc, ENode::apply(OpKind::Neg, &[c])),
                (marker, ENode::Var(9)),
            ],
        );
    }

    /// A node whose canonical form lands in a class it cannot merge with
    /// leaves its class (law 7). Classes 0 and 1 hold the constants 1 and 2;
    /// `Neg(a)` joins 0 and `Neg(b)` joins 1; `union(a, b)` makes the two
    /// `Neg`s congruent. The worklist is a stack of 0, 1, 2: item 2 (the
    /// variables) changes nothing, item 1 finds `Neg(b)` canonicalizes to
    /// `Neg(a)`, which the memo names class 4, now in class 0, and the union
    /// of 1 and 0 is refused (1 = 2). Node 5 is retired: no class lists it
    /// and the node count leaves it out.
    #[test]
    fn a_node_whose_union_is_refused_leaves_its_class() {
        let mut eg = EGraph::new();
        let one = eg.add(ENode::constant(1.0)); // 0
        let two = eg.add(ENode::constant(2.0)); // 1
        let a = eg.add(ENode::Var(0)); // 2
        let b = eg.add(ENode::Var(1)); // 3
        let neg_a = eg.add(ENode::apply(OpKind::Neg, &[a])); // 4
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // 5
        assert_eq!(neg_b.index(), 5, "the recipe assumes these ids");
        eg.union(neg_a, one);
        eg.union(neg_b, two);
        eg.union(a, b);
        eg.rebuild();

        assert_eq!(
            eg.refused_const_unions(),
            &[(Lit::Real(2.0), Lit::Real(1.0))][..],
            "item 1 asked to merge 2 into 1 and was refused"
        );
        assert_ne!(eg.find(one), eg.find(two));
        let retired = ENodeId(5);
        assert_eq!(eg.node_for_tag(eg.find(neg_b), retired), None);
        assert_eq!(eg.node_for_tag(eg.find(neg_a), retired), None);
        assert_eq!(
            eg.node_for_tag(one, ENodeId(0)),
            Some(&ENode::constant(1.0)),
            "a live node answers in its own class"
        );
        assert_eq!(
            eg.node_for_tag(two, ENodeId(0)),
            None,
            "a live node answers nothing in another class"
        );
        assert!(
            eg.class_ids()
                .all(|class| eg.tags(class).all(|t| t != retired))
        );
        assert_eq!(members(&eg, two), [ENode::constant(2.0)]);
        assert_eq!(
            members(&eg, one),
            [ENode::constant(1.0), ENode::apply(OpKind::Neg, &[a])]
        );
        assert_eq!(
            eg.node_count(),
            eg.num_classes() - 1,
            "the retired node is not counted"
        );
        assert_eq!(
            eg.node_for_tag(one, ENodeId(1_000)),
            None,
            "a tag past the graph names nothing"
        );
        assert!(eg.graph.columns.is_retired(retired));
        assert!(!eg.is_member(retired));
        assert!(
            !eg.is_member(ENodeId(u64::MAX)),
            "a tag past the graph is no member"
        );
        assert!(eg.is_member(ENodeId(4)));
        assert_every_id_is_a_member(
            &eg,
            &[
                (one, ENode::constant(1.0)),
                (two, ENode::constant(2.0)),
                (a, ENode::Var(0)),
                (b, ENode::Var(1)),
                (neg_a, ENode::apply(OpKind::Neg, &[a])),
                (neg_b, ENode::apply(OpKind::Neg, &[b])),
            ],
        );
        #[cfg(feature = "provenance-journal")]
        {
            assert!(eg.children_of(retired).is_empty());
            assert_eq!(eg.children_of(ENodeId(4)), vec![a]);
        }
    }

    /// A retired node that founded its class: the class keeps the members
    /// after it, and its own id names a node that is no member of it (plan
    /// §1: `node(c)` need not be a member of `c`). Class 4 is
    /// `[Neg(b), 2, Var(7)]` with the fact 2; `union(a, b)` makes `Neg(b)`
    /// congruent to `Neg(a)`, which is in class 0 with the fact 1. Item 4 is
    /// refused that union and retires its first member.
    #[test]
    fn a_retired_founder_leaves_the_members_after_it() {
        let mut eg = EGraph::new();
        let one = eg.add(ENode::constant(1.0)); // 0
        let a = eg.add(ENode::Var(0)); // 1
        let b = eg.add(ENode::Var(1)); // 2
        let neg_a = eg.add(ENode::apply(OpKind::Neg, &[a])); // 3
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // 4
        let two = eg.add(ENode::constant(2.0)); // 5
        let z = eg.add(ENode::Var(7)); // 6
        assert_eq!(z.index(), 6, "the recipe assumes these ids");
        eg.union(neg_a, one);
        eg.union(neg_b, two);
        eg.union(neg_b, z);
        eg.union(a, b);
        eg.rebuild();

        assert_eq!(
            eg.refused_const_unions(),
            &[(Lit::Real(2.0), Lit::Real(1.0))][..]
        );
        assert_eq!(eg.find(neg_b), neg_b, "class 4 survives its founder");
        assert_eq!(
            eg.tags(neg_b).collect::<Vec<_>>(),
            [ENodeId(5), ENodeId(6)],
            "the kept members follow in walk order"
        );
        assert_eq!(members(&eg, neg_b), [ENode::constant(2.0), ENode::Var(7)]);
        assert!(eg.graph.columns.is_retired(ENodeId(4)));
        assert!(!eg.is_member(ENodeId(4)));
        assert_eq!(eg.node_for_tag(neg_b, ENodeId(4)), None);
        assert_eq!(
            members(&eg, one),
            [ENode::constant(1.0), ENode::apply(OpKind::Neg, &[a])]
        );
        assert_eq!(eg.node_count(), eg.num_classes() - 1);
        assert_every_id_is_a_member(
            &eg,
            &[
                (one, ENode::constant(1.0)),
                (a, ENode::Var(0)),
                (b, ENode::Var(1)),
                (neg_a, ENode::apply(OpKind::Neg, &[a])),
                (neg_b, ENode::apply(OpKind::Neg, &[b])),
                (two, ENode::constant(2.0)),
                (z, ENode::Var(7)),
            ],
        );
    }

    /// A retired member between two kept ones: the kept two are adjacent in
    /// the class afterwards, the first still first. Class 3 is
    /// `[Var(5), Neg(b), 2]` with the fact 2; item 3 keeps `Var(5)`, retires
    /// `Neg(b)` (refused, as above) and keeps `2`.
    #[test]
    fn a_retired_middle_member_leaves_its_neighbours_adjacent() {
        let mut eg = EGraph::new();
        let one = eg.add(ENode::constant(1.0)); // 0
        let a = eg.add(ENode::Var(0)); // 1
        let b = eg.add(ENode::Var(1)); // 2
        let x = eg.add(ENode::Var(5)); // 3
        let neg_a = eg.add(ENode::apply(OpKind::Neg, &[a])); // 4
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // 5
        let two = eg.add(ENode::constant(2.0)); // 6
        assert_eq!(two.index(), 6, "the recipe assumes these ids");
        eg.union(neg_a, one);
        eg.union(x, neg_b);
        eg.union(x, two);
        eg.union(a, b);
        eg.rebuild();

        assert_eq!(
            eg.refused_const_unions(),
            &[(Lit::Real(2.0), Lit::Real(1.0))][..]
        );
        assert_eq!(eg.tags(x).collect::<Vec<_>>(), [ENodeId(3), ENodeId(6)]);
        assert_eq!(members(&eg, x), [ENode::Var(5), ENode::constant(2.0)]);
        assert!(eg.graph.columns.is_retired(ENodeId(5)));
        assert_eq!(eg.node_count(), eg.num_classes() - 1);
        assert_every_id_is_a_member(
            &eg,
            &[
                (one, ENode::constant(1.0)),
                (a, ENode::Var(0)),
                (b, ENode::Var(1)),
                (x, ENode::Var(5)),
                (neg_a, ENode::apply(OpKind::Neg, &[a])),
                (neg_b, ENode::apply(OpKind::Neg, &[b])),
                (two, ENode::constant(2.0)),
            ],
        );
    }

    /// A clone owns its member lists: a union in the original after the clone
    /// moves nothing in the copy (`EGraph: Clone` is derived; plan §8.4).
    #[test]
    fn a_clone_owns_its_member_lists() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let copy = eg.clone();
        eg.union(x, y);
        eg.rebuild();
        assert_eq!(members(&eg, x), [ENode::Var(0), ENode::Var(1)]);
        assert_eq!(members(&copy, x), [ENode::Var(0)]);
        assert_eq!(members(&copy, y), [ENode::Var(1)]);
        assert_ne!(copy.find(x), copy.find(y));
    }

    /// `union` gives the surviving class's nodes, then the merged-away
    /// class's, whichever argument order the caller used.
    #[test]
    fn union_gives_the_parent_then_the_child() {
        for flipped in [false, true] {
            let mut eg = EGraph::new();
            let parent = eg.add(ENode::Var(0));
            let parent_peer = eg.add(ENode::Var(1));
            let child = eg.add(ENode::Var(2));
            let child_peer = eg.add(ENode::Var(3));
            let parent = eg.union(parent, parent_peer);
            let child = eg.union(child, child_peer);
            assert_eq!(
                members(&eg, parent),
                [ENode::Var(0), ENode::Var(1)],
                "a union of two classes gives the first then the second"
            );

            let survivor = if flipped {
                eg.union(child, parent)
            } else {
                eg.union(parent, child)
            };
            assert_eq!(survivor, parent, "the smaller id survives");
            assert_eq!(
                members(&eg, survivor),
                [ENode::Var(0), ENode::Var(1), ENode::Var(2), ENode::Var(3)],
                "parent then child (flipped argument order: {flipped})"
            );
            assert_eq!(members(&eg, child), members(&eg, parent));
        }
    }

    /// A rebuild item whose class survives a union made during the item gives
    /// `[absorbed] ++ [kept]`: what the union appended, then the item's own
    /// nodes in the order they were walked. The item is class 3, it absorbs
    /// class 4, and 3 < 4.
    #[test]
    fn a_rebuild_item_that_survives_gives_absorbed_then_kept() {
        let mut eg = EGraph::new();
        let _a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 3, the item
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 4, absorbed
        let d = eg.add(ENode::Var(3));
        let marker = eg.add(ENode::Var(9));
        eg.union(nc, d);
        eg.union(nb, marker);
        eg.union(b, c);
        assert_eq!(
            members(&eg, nc),
            [ENode::apply(OpKind::Neg, &[c]), ENode::Var(3)],
            "the item's nodes before the rebuild, the first still naming c"
        );
        assert_eq!(
            members(&eg, nb),
            [ENode::apply(OpKind::Neg, &[b]), ENode::Var(9)]
        );

        // The worklist is a stack of 3, 4, 1: class 1 pops, then 4, then 3.
        assert_eq!(eg.pending_rebuilds(), 3);
        assert_eq!(eg.rebuild_budgeted(3), 1, "the union in item 3 queues 3");
        assert_eq!(eg.find(nb), nc, "the item absorbed class 4 and survived");

        let neg_b = ENode::apply(OpKind::Neg, &[b]);
        let expected = [neg_b, ENode::Var(9), neg_b, ENode::Var(3)];
        assert_eq!(
            members(&eg, nc),
            expected,
            "[absorbed: Neg(b), Var(9)] ++ [kept: Neg(b), Var(3)]"
        );
        eg.rebuild();
        assert_eq!(
            members(&eg, nc),
            expected,
            "a further rebuild keeps the order"
        );
    }

    /// A rebuild item whose class loses a union made during the item gives
    /// the survivor's prior nodes, then whatever the survivor absorbs later in
    /// the same item, then `[kept]`. The item is class 4. Its first node
    /// canonicalizes into class 3, which survives with a prior member; its
    /// second into class 5, which 3 then absorbs; its third is its own.
    #[test]
    fn a_rebuild_item_that_loses_gives_prior_then_later_absorbed_then_kept() {
        let mut eg = EGraph::new();
        let _a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 3, survives
        let neg_c = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 4, the item
        let abs_b = eg.add(ENode::apply(OpKind::Abs, &[b])); // class 5, absorbed later
        let abs_c = eg.add(ENode::apply(OpKind::Abs, &[c]));
        let prior = eg.add(ENode::Var(7));
        let later = eg.add(ENode::Var(8));
        let marker = eg.add(ENode::Var(9));
        eg.union(neg_b, prior);
        eg.union(abs_b, later);
        eg.union(neg_c, abs_c);
        eg.union(neg_c, marker);
        eg.union(b, c);
        assert_eq!(
            [neg_b, neg_c, abs_b].map(EClassId::index),
            [3, 4, 5],
            "the recipe assumes these ids"
        );
        assert_eq!(
            members(&eg, neg_c),
            [
                ENode::apply(OpKind::Neg, &[c]),
                ENode::apply(OpKind::Abs, &[c]),
                ENode::Var(9)
            ]
        );

        // The worklist is a stack of 3, 5, 4, 1: class 1 pops, then 4.
        assert_eq!(
            eg.rebuild_budgeted(2),
            2,
            "the unions in item 4 queue nothing"
        );
        assert_eq!(eg.find(neg_c), neg_b, "the item lost to class 3");
        assert_eq!(eg.find(abs_b), neg_b, "and 3 absorbed class 5 during it");

        let neg = ENode::apply(OpKind::Neg, &[b]);
        let abs = ENode::apply(OpKind::Abs, &[b]);
        let expected = [
            neg,
            ENode::Var(7), // the survivor's prior nodes
            abs,
            ENode::Var(8), // absorbed later in the item
            neg,
            abs,
            ENode::Var(9), // kept, in the order they were walked
        ];
        assert_eq!(members(&eg, neg_b), expected);
        eg.rebuild();
        assert_eq!(
            members(&eg, neg_b),
            expected,
            "a further rebuild keeps the order"
        );
    }

    /// A rebuild item that survives one union made during it and then loses a
    /// later one hands the survivor what it absorbed in between: the
    /// survivor's prior nodes, then the nodes the item absorbed while it still
    /// stood, then whatever the survivor absorbs after the loss, then
    /// `[kept]`. The item is class 4. Its first node canonicalizes into class
    /// 5, which 4 absorbs (4 < 5); its second into class 3, which survives
    /// with a prior member (3 < 4), so the item's list, class 5's nodes, goes
    /// to the survivor; its third into class 7, which 3 then absorbs.
    #[test]
    fn a_rebuild_item_that_survives_then_loses_gives_prior_then_absorbed_then_kept() {
        let mut eg = EGraph::new();
        let _a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 3, survives
        let abs_c = eg.add(ENode::apply(OpKind::Abs, &[c])); // class 4, the item
        let abs_b = eg.add(ENode::apply(OpKind::Abs, &[b])); // class 5, absorbed first
        let neg_c = eg.add(ENode::apply(OpKind::Neg, &[c]));
        let sqrt_b = eg.add(ENode::apply(OpKind::Sqrt, &[b])); // class 7, absorbed later
        let sqrt_c = eg.add(ENode::apply(OpKind::Sqrt, &[c]));
        let prior = eg.add(ENode::Var(7));
        let first = eg.add(ENode::Var(8));
        let later = eg.add(ENode::Var(10));
        let marker = eg.add(ENode::Var(9));
        eg.union(neg_b, prior);
        eg.union(abs_b, first);
        eg.union(sqrt_b, later);
        eg.union(abs_c, neg_c);
        eg.union(abs_c, sqrt_c);
        eg.union(abs_c, marker);
        eg.union(b, c);
        assert_eq!(
            [neg_b, abs_c, abs_b, sqrt_b].map(EClassId::index),
            [3, 4, 5, 7],
            "the recipe assumes these ids"
        );
        assert_eq!(
            members(&eg, abs_c),
            [
                ENode::apply(OpKind::Abs, &[c]),
                ENode::apply(OpKind::Neg, &[c]),
                ENode::apply(OpKind::Sqrt, &[c]),
                ENode::Var(9)
            ]
        );

        // The worklist is a stack of 3, 5, 7, 4, 1: class 1 pops, then 4.
        assert_eq!(
            eg.rebuild_budgeted(2),
            4,
            "the union in item 4 that 4 survives queues 4; the one it loses queues nothing"
        );
        assert_eq!(eg.find(abs_c), neg_b, "the item lost to class 3");
        assert_eq!(eg.find(abs_b), neg_b, "after absorbing class 5");
        assert_eq!(eg.find(sqrt_b), neg_b, "and 3 absorbed class 7 during it");

        let neg = ENode::apply(OpKind::Neg, &[b]);
        let abs = ENode::apply(OpKind::Abs, &[b]);
        let sqrt = ENode::apply(OpKind::Sqrt, &[b]);
        let expected = [
            neg,
            ENode::Var(7), // the survivor's prior nodes
            abs,
            ENode::Var(8), // absorbed by the item before it lost
            sqrt,
            ENode::Var(10), // absorbed by the survivor after
            abs,
            neg,
            sqrt,
            ENode::Var(9), // kept, in the order they were walked
        ];
        assert_eq!(members(&eg, neg_b), expected);
        eg.rebuild();
        assert_eq!(
            members(&eg, neg_b),
            expected,
            "a further rebuild keeps the order"
        );
    }

    /// Create an e-graph with standard algebraic rules for testing.
    fn egraph_with_rules() -> Saturation {
        let rules: Vec<Box<dyn Rewrite>> = vec![
            // InversePair rules
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            // Commutativity
            Commutative::new(OpKind::Add),
            Commutative::new(OpKind::Mul),
            Commutative::new(OpKind::Min),
            Commutative::new(OpKind::Max),
            // Distributivity
            Distributive::new(OpKind::Mul, OpKind::Add),
            Distributive::new(OpKind::Mul, OpKind::Sub),
            // Identity
            Identity::new(OpKind::Add),
            Identity::new(OpKind::Mul),
            // Annihilator
            Annihilator::new(OpKind::Mul),
        ];
        Saturation::over(RuleSet::new(rules))
    }

    #[test]
    fn inverse_add() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::apply(OpKind::Neg, &[x]));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, neg_x]));
        SaturationConfig::compatibility(100).run(&mut eg);
        let zero = eg.add(ENode::constant(0.0));
        assert_eq!(eg.find(sum), eg.find(zero));
    }

    #[test]
    fn inverse_mul() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let recip_x = eg.add(ENode::apply(OpKind::Recip, &[x]));
        let product = eg.add(ENode::apply(OpKind::Mul, &[x, recip_x]));
        SaturationConfig::compatibility(100).run(&mut eg);
        let one = eg.add(ENode::constant(1.0));
        assert_eq!(eg.find(product), eg.find(one));
    }

    #[test]
    fn complex_inverse() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::apply(OpKind::Mul, &[x, five]));
        let div = eg.add(ENode::apply(OpKind::Div, &[prod, x]));
        SaturationConfig::compatibility(100).run(&mut eg);
        assert_eq!(eg.find(div), eg.find(five));
    }

    #[test]
    fn nested_subtraction() {
        // a - (b - c) should equal a - b + c
        // Test: 10 - (6 - 2) = 10 - 4 = 6
        let mut eg = egraph_with_rules();
        let a = eg.add(ENode::constant(10.0)); // a = 10
        let b = eg.add(ENode::constant(6.0)); // b = 6
        let c = eg.add(ENode::constant(2.0)); // c = 2

        // Build a - (b - c)
        let b_minus_c = eg.add(ENode::apply(OpKind::Sub, &[b, c])); // 6 - 2 = 4
        let result = eg.add(ENode::apply(OpKind::Sub, &[a, b_minus_c])); // 10 - 4 = 6

        SaturationConfig::compatibility(100).run(&mut eg);

        // Extract and verify structure
        let costs = CostModel::default();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!("Extracted arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn mul_sub_pattern() {
        // This is the problematic pattern from discriminant:
        // d*d - (c - r) where d=4, c=16, r=1
        let mut eg = egraph_with_rules();
        let d = eg.add(ENode::constant(4.0));
        let c_sq = eg.add(ENode::constant(16.0));
        let r_sq = eg.add(ENode::constant(1.0));

        let d_sq = eg.add(ENode::apply(OpKind::Mul, &[d, d]));
        let inner_sub = eg.add(ENode::apply(OpKind::Sub, &[c_sq, r_sq]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[d_sq, inner_sub]));

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::default();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!("Extracted arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn mul_sub_pattern_with_vars() {
        // x*x - (y - z)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));

        let x_sq = eg.add(ENode::apply(OpKind::Mul, &[x, x]));
        let inner_sub = eg.add(ENode::apply(OpKind::Sub, &[y, z]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[x_sq, inner_sub]));

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::default();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!(
            "Extracted arena with vars: root={:?} len={}",
            root,
            arena.len()
        );
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn mul_sub_pattern_with_fma() {
        // Same pattern but with FMA costs (what the kernel! macro uses)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));

        let x_sq = eg.add(ENode::apply(OpKind::Mul, &[x, x]));
        let inner_sub = eg.add(ENode::apply(OpKind::Sub, &[y, z]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[x_sq, inner_sub]));

        SaturationConfig::compatibility(100).run(&mut eg);

        // Use default costs like the kernel! macro does
        let costs = CostModel::new();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!(
            "Extracted arena with FMA costs: root={:?} len={}",
            root,
            arena.len()
        );
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn discriminant_structure() {
        // Match the actual discriminant structure:
        // d_dot_c² - (c_sq - r_sq) where c_sq = a² + b² and r_sq = r²
        let mut eg = egraph_with_rules();
        let d = eg.add(ENode::Var(0));
        let a = eg.add(ENode::Var(1));
        let b = eg.add(ENode::Var(2));
        let r = eg.add(ENode::Var(3));

        let d_sq = eg.add(ENode::apply(OpKind::Mul, &[d, d]));
        let a_sq = eg.add(ENode::apply(OpKind::Mul, &[a, a]));
        let b_sq = eg.add(ENode::apply(OpKind::Mul, &[b, b]));
        let c_sq = eg.add(ENode::apply(OpKind::Add, &[a_sq, b_sq]));
        let r_sq = eg.add(ENode::apply(OpKind::Mul, &[r, r]));
        let inner = eg.add(ENode::apply(OpKind::Sub, &[c_sq, r_sq]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[d_sq, inner]));

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::new();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!("Discriminant arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn depth_cost_should_apply_linear_penalty_only_above_threshold() {
        let mut costs = CostModel::new();
        costs.depth_threshold = 5;
        costs.depth_penalty = 100;

        assert_eq!(costs.depth_cost(0), 0);
        assert_eq!(costs.depth_cost(5), 0);

        assert_eq!(costs.depth_cost(6), 100);
        assert_eq!(costs.depth_cost(7), 200);
        assert_eq!(costs.depth_cost(10), 500);
    }

    #[test]
    fn shallow_should_set_aggressive_depth_threshold_and_penalty() {
        let costs = CostModel::shallow();
        assert_eq!(costs.depth_threshold, 16);
        assert_eq!(costs.depth_penalty, 500);

        assert_eq!(costs.depth_cost(16), 0);
        assert_eq!(costs.depth_cost(17), 500);
        assert_eq!(costs.depth_cost(20), 2000);
    }

    #[test]
    fn depth_aware_extraction() {
        // Build a deep expression: ((((x + 1) + 1) + 1) + 1)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));

        let mut current = x;
        for _ in 0..10 {
            current = eg.add(ENode::apply(OpKind::Add, &[current, one]));
        }

        SaturationConfig::compatibility(100).run(&mut eg);

        // Extract with default costs (high threshold)
        let default_costs = CostModel::default();
        let dag = extract_dag(&eg, current, &default_costs);
        let (arena, root) = dag.to_arena(&eg);
        assert!(arena.node_count_subtree(root) > 0);

        // Extract with shallow costs (low threshold)
        let mut shallow_costs = CostModel::new();
        shallow_costs.depth_threshold = 3;
        shallow_costs.depth_penalty = 1000;
        let dag = extract_dag(&eg, current, &shallow_costs);
        let (arena2, root2) = dag.to_arena(&eg);
        assert!(arena2.node_count_subtree(root2) > 0);
    }

    // ========================================================================
    // Provenance tests
    // ========================================================================

    /// Find the `RewriteTarget` for a given rule name, failing loudly (not
    /// silently) if it isn't present — a missing match means the test setup
    /// is wrong, not that provenance has nothing to check.
    fn find_target(eg: &Saturation, rule_name: &str) -> RewriteTarget {
        eg.find_rewrite_matches()
            .into_iter()
            .find(|t| eg.rules().rule(t.rule_idx).name() == rule_name)
            .unwrap_or_else(|| panic!("no rewrite match found for rule {rule_name:?}"))
    }

    /// Like `find_target`, but further restricted to matches against a
    /// specific (already-canonical) class — needed once a class holds
    /// multiple nodes that all match the same rule name.
    fn find_target_in_class(eg: &Saturation, rule_name: &str, class: EClassId) -> RewriteTarget {
        eg.find_rewrite_matches()
            .into_iter()
            .find(|t| eg.rules().rule(t.rule_idx).name() == rule_name && t.class_id == class)
            .unwrap_or_else(|| {
                panic!("no rewrite match found for rule {rule_name:?} in class {class:?}")
            })
    }

    #[test]
    fn provenance_tiny_expression_one_rewrite_matches_hand_derivation() {
        // x + y, then apply Commutative -> y + x (a fresh Create'd node).
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));

        // Hand-derivation: x, y, and sum are all seeds.
        for &(class, _) in &[(x, "x"), (y, "y"), (sum, "sum")] {
            for tag in eg.tags(class) {
                assert_eq!(
                    eg.journal().origin(tag),
                    Some(Origin::Seed),
                    "seed node in class {class:?} should be Origin::Seed"
                );
            }
        }
        assert_eq!(eg.journal().recorded_count(), 0);
        assert_eq!(eg.journal().union_count(), 0);

        let target = find_target(&eg, "commutative");
        assert_eq!(target.class_id, eg.find(sum));

        let applied = eg.apply_single_rule(target.rule_idx, target.class_id, target.tag);
        assert!(applied, "commutative rule should have applied to x + y");

        // Hand-derivation: exactly one application recorded, for the
        // "commutative" rule, matched against `sum`'s class.
        assert_eq!(eg.journal().recorded_count(), 1);
        let record = eg.journal().application(ApplicationId(0)).unwrap();
        assert_eq!(record.rule_idx, target.rule_idx);
        assert_eq!(record.match_root, eg.find(sum));
        assert_eq!(eg.rules().rule(record.rule_idx).name(), "commutative");

        // Hand-derivation: the new commuted node (y + x) has Origin::Rule
        // pointing at that one application; it lives in the same class as
        // `sum` post-union.
        let commuted_class = eg.find(sum);
        let rule_origin_tags: Vec<ENodeId> = eg
            .tags(commuted_class)
            .filter(|&t| matches!(eg.journal().origin(t), Some(Origin::Rule(_))))
            .collect();
        assert_eq!(
            rule_origin_tags.len(),
            1,
            "expected exactly one rule-created node in the merged class"
        );
        assert_eq!(
            eg.journal().origin(rule_origin_tags[0]),
            Some(Origin::Rule(ApplicationId(0)))
        );

        // Hand-derivation: exactly one union event, attributed to the
        // commutative rule's index, at the step apply_single_rule ran on.
        assert_eq!(eg.journal().union_count(), 1);
        let union_event = eg.journal().union_events()[0];
        assert_eq!(union_event.rule_idx, Some(target.rule_idx));
    }

    #[test]
    fn provenance_chain_rule_b_consumes_rule_a_product() {
        // Chain: apply Commutative to (x + y) producing (y + x) [rule A],
        // then apply Commutative again to a *different* sum, (y + x) + z,
        // whose match consumes the class that A's product lives in as a
        // child. derivation_ancestors of B's product must include A's
        // application.
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let inner = eg.add(ENode::apply(OpKind::Add, &[x, y])); // x + y
        let outer = eg.add(ENode::apply(OpKind::Add, &[inner, z])); // (x + y) + z

        // Rule A: commute the inner sum (x + y) -> (y + x).
        let target_a = find_target_in_class(&eg, "commutative", eg.find(inner));
        assert!(eg.apply_single_rule(target_a.rule_idx, target_a.class_id, target_a.tag));
        let app_a = ApplicationId(0);

        // Rule B: commute the outer sum ((x+y)+z) -> (z + (x+y)). After
        // rule A, inner's class holds both `x+y` and `y+x`, both of which
        // still match "commutative" — so we must specifically target
        // outer's class rather than take the first "commutative" match.
        let target_b = find_target_in_class(&eg, "commutative", eg.find(outer));
        assert!(eg.apply_single_rule(target_b.rule_idx, target_b.class_id, target_b.tag));
        let app_b = ApplicationId(1);
        assert_eq!(eg.journal().recorded_count(), 2);

        // Find B's produced node: the rule-created node in outer's class
        // whose origin is app_b.
        let outer_class = eg.find(outer);
        let b_product_tag = eg
            .tags(outer_class)
            .find(|&t| eg.journal().origin(t) == Some(Origin::Rule(app_b)))
            .expect("expected a node created by application B in outer's class");

        let ancestors = eg.derivation_ancestors(&[(outer_class, b_product_tag)]);
        assert!(
            ancestors.contains(&app_b),
            "B's own application must be in its own ancestry"
        );
        assert!(
            ancestors.contains(&app_a),
            "A's application (which produced the node B's match consumed as \
             a child of the outer sum) must be included in B's product's ancestry"
        );

        // Sanity: the trace formats without panicking and mentions both
        // applications' rule name.
        let trace = eg.format_derivation_trace(&ancestors);
        assert!(trace.contains("commutative"));
    }

    #[test]
    fn provenance_union_driven_case_includes_congruence_union_event() {
        // Build two structurally-identical-after-canonicalization Neg nodes
        // in different classes purely because their children start out
        // unmerged, then union the children directly (not through a rule).
        //
        // Congruence closure only reconsiders a class's memo key when that
        // class itself is on the rebuild worklist (see rebuild_budgeted) —
        // union(b, c) alone doesn't touch nb/nc's classes. So we also
        // enqueue nc via a harmless union(nc, marker) (nc's numeric id is
        // lower, so nc survives as parent and just gains an extra node);
        // this is the same enqueue trick used in
        // rebuild_budgeted_does_not_drop_nodes_when_current_class_survives_union.
        // When rebuild() then reprocesses nc, canonicalizing Neg([c]) turns
        // it into Neg([b]) (since union(b,c) already ran), which collides
        // with nb's memo entry — a congruence-closure union with
        // rule_idx = None.
        let mut eg = egraph_with_rules();
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b]));
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c]));
        let marker = eg.add(ENode::Var(9));

        assert_eq!(eg.journal().union_count(), 0);

        // Enqueue nc on the worklist without disturbing its semantics.
        eg.union(nc, marker);
        // Direct union, not via a rule: rule_idx must be None on the
        // resulting UnionEvent.
        eg.union(b, c);
        eg.rebuild();

        assert_eq!(
            eg.find(nb),
            eg.find(nc),
            "Neg(b) and Neg(c) should have merged"
        );

        // At least one recorded union event must have rule_idx = None
        // (the direct union(b, c) and/or the congruence-closure union that
        // merged nb/nc during rebuild).
        assert!(
            eg.journal()
                .union_events()
                .iter()
                .any(|e| e.rule_idx.is_none()),
            "expected at least one non-rule-driven (congruence/direct) union event"
        );
    }

    /// Overhead measurement: total saturation time and provenance record
    /// counts on a reasonably deep expression. Not a correctness test —
    /// `#[ignore]`d so normal `cargo test` runs stay fast; run explicitly
    /// with `cargo test -p pixelflow-search --release -- --ignored
    /// provenance_overhead`. See module docs on `provenance` for the
    /// numbers observed when this was last run.
    #[test]
    #[ignore]
    fn provenance_overhead_timing() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));

        // A reasonably deep expression: alternating +/-/* chain over x, y.
        let mut current = x;
        for i in 0..40 {
            let op: OpKind = match i % 3 {
                0 => OpKind::Add,
                1 => OpKind::Mul,
                _ => OpKind::Sub,
            };
            current = eg.add(ENode::apply(op, &[current, y]));
        }

        let start = std::time::Instant::now();
        SaturationConfig::compatibility(100).run(&mut eg);
        let elapsed = start.elapsed();

        eprintln!(
            "provenance overhead: saturation took {:?}; origins={} applications={} unions={} classes={}",
            elapsed,
            eg.journal().origin_count(),
            eg.journal().recorded_count(),
            eg.journal().union_count(),
            eg.num_classes(),
        );
    }

    // ------------------------------------------------------------------
    // The laws. `docs/archive/plans/2026-09-02-optimizer-api.md` §1 states each one
    // and names the test that should pin it; these are those tests. They
    // assert the *property*, never the mechanism — a future fix that moves
    // the class budget somewhere else entirely should keep them passing.
    // ------------------------------------------------------------------

    /// Seed a graph whose rules certainly have work to do, and return the
    /// probe ids the law tests watch. `Sub` guarantees a `Canonicalize`
    /// firing (`a - b → a + (-b)`), which is a *node-minting* action — the
    /// kind that used to route through the over-budget hole in `add`.
    fn seed_probes(eg: &mut EGraph) -> Vec<EClassId> {
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let diff = eg.add(ENode::apply(OpKind::Sub, &[x, y]));
        let prod = eg.add(ENode::apply(OpKind::Mul, &[sum, z]));
        let scaled = eg.add(ENode::apply(OpKind::Mul, &[diff, z]));
        vec![x, y, z, sum, diff, prod, scaled]
    }

    /// The pairs of probes the graph currently calls equal.
    fn equal_pairs(eg: &EGraph, probes: &[EClassId]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..probes.len() {
            for j in (i + 1)..probes.len() {
                if eg.find(probes[i]) == eg.find(probes[j]) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// **L1 — soundness.** `add` is a homomorphism onto the semantic
    /// quotient: the class it names always contains the node it was given,
    /// and no amount of graph growth changes that. Consequently an
    /// over-budget run can discover *fewer* equalities than an exhaustive
    /// one, never a different one.
    ///
    /// This is the regression test for #1105. Before the fix, `add`
    /// answered `EClassId(0)` once the graph reached the hard class limit —
    /// a sentinel, not an insertion — and `RewriteAction`'s node-minting
    /// handlers unioned the matched class against it, asserting that some
    /// unrelated term equals whatever was added first. Both halves are
    /// asserted here: that `add` past the limit still names a class holding
    /// the node, and that a saturation run driven past the limit invents no
    /// equality among terms that are not equal.
    ///
    /// Deliberately mechanism-blind: it says nothing about *how* growth is
    /// refused, only that nothing false comes out.
    #[test]
    fn over_budget_growth_cannot_assert_a_false_equality() {
        let mut eg = egraph_with_rules();

        // Class 0 is the term every false union used to land on, so make it
        // something no probe below is equal to.
        let class_zero = eg.add(ENode::constant(-1.0));
        assert_eq!(class_zero, EClassId(0), "class 0 must be the first add");

        let probes = seed_probes(&mut eg);
        assert!(
            equal_pairs(&eg, &probes).is_empty(),
            "probes start distinct"
        );

        // Pad to one short of the ceiling with distinct constants, so the
        // next rewrite that wants to mint a node is over budget — and so the
        // graph lands *exactly* on the ceiling after the `add` below, which
        // keeps the saturation run below inside the sweep rather than
        // bailing at `saturate_with_limits`'s own pre-sweep check.
        let mut pad = 0.0f32;
        while eg.num_classes() < HARD_MINTED_LIMIT - 1 {
            let _ = eg.add(ENode::constant(pad));
            pad += 1.0;
        }

        // `add` is still total at the ceiling: the class it returns holds
        // the node, and it is not class 0's term.
        let fresh = eg.add(ENode::Var(200));
        assert_eq!(eg.num_classes(), HARD_MINTED_LIMIT);
        assert!(
            matches!(eg.nodes(eg.find(fresh)).next(), Some(ENode::Var(200))),
            "add past the limit returned a class that does not hold the node"
        );
        assert_ne!(
            eg.find(fresh),
            eg.find(class_zero),
            "add past the limit aliased an unrelated class"
        );

        // Drive saturation with a cap no smaller than the graph, so the
        // ceiling — not the caller's budget — is what has to hold.
        let stats = eg.saturate_with_limits(4, usize::MAX, std::time::Duration::from_secs(60));
        eprintln!("over-budget saturation: {stats:?}");

        // The load-bearing assertion: nothing new is equal.
        assert!(
            equal_pairs(&eg, &probes).is_empty(),
            "over-budget saturation invented an equality among distinct terms"
        );
        for (i, &p) in probes.iter().enumerate() {
            assert_ne!(
                eg.find(p),
                eg.find(class_zero),
                "probe {i} was falsely unioned with class 0"
            );
        }
        assert_ne!(eg.find(fresh), eg.find(class_zero));
    }

    /// Seed a graph in which equalities are *derivable* and arrive at
    /// different depths, so that a budget ladder has something to be
    /// monotone about. Returns the probes, several of which the rules
    /// eventually prove congruent.
    fn seed_derivable_probes(eg: &mut EGraph) -> Vec<EClassId> {
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let zero = eg.add(ENode::constant(0.0));
        let one = eg.add(ENode::constant(1.0));

        // x + 0 = x, one Identity firing away.
        let x_plus_zero = eg.add(ENode::apply(OpKind::Add, &[x, zero]));
        // (x + 0) * 1 = x, two firings away.
        let scaled = eg.add(ENode::apply(OpKind::Mul, &[x_plus_zero, one]));
        // x * 0 = 0, via Annihilator.
        let annihilated = eg.add(ENode::apply(OpKind::Mul, &[x, zero]));
        // x + y = y + x, via Commutative.
        let xy = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let yx = eg.add(ENode::apply(OpKind::Add, &[y, x]));

        vec![x, y, zero, one, x_plus_zero, scaled, annihilated, xy, yx]
    }

    /// **L2 — monotonicity.** Saturation only ever adds equalities. A pair
    /// the graph already calls equal is still equal after a run at any
    /// budget, and the partition at a larger budget refines the partition at
    /// a smaller one.
    ///
    /// Budget truncation is allowed to stop discovery; it is not allowed to
    /// retract a discovery. That is what lets a budget stay a
    /// quality/compile-time dial instead of a correctness one, and it is half
    /// the proof of L4 (any saturation policy preserves denotation) — the
    /// other half is L1, above.
    ///
    /// Two things are asserted, in increasing strength: a hand-made union
    /// that no rule could have derived survives every budget, and the set of
    /// probe pairs the graph calls equal only ever grows as the budget does.
    ///
    /// Note what is *not* asserted, because this rebuild does not do it:
    /// upward congruence. `union` enqueues only the merged class, never its
    /// parents, so `a = b` does not by itself re-canonicalize `f(a)` and
    /// `f(b)` into one class — they merge when a later sweep re-walks them,
    /// or not at all. That is an incompleteness (fewer equalities than an
    /// ideal congruence closure), which is exactly the direction L2 permits;
    /// it is filed as #1106 rather than assumed here.
    #[test]
    fn saturation_at_any_budget_never_removes_an_equality() {
        // Strictly increasing, spanning "cannot even finish one rule" to the
        // ceiling itself.
        let ladder = [1usize, 2, 8, 64, 512, 4096, HARD_MINTED_LIMIT];
        let mut previous: Option<(usize, Vec<(usize, usize)>)> = None;
        let mut ever_found = 0usize;

        for max_minted in ladder {
            let mut eg = egraph_with_rules();
            let probes = seed_derivable_probes(&mut eg);

            // A hand-asserted equality between two terms no rule in this
            // vocabulary relates: only a retraction could ever break it.
            let a = eg.add(ENode::Var(30));
            let b = eg.add(ENode::Var(31));
            eg.union(a, b);
            eg.rebuild();
            assert_eq!(eg.find(a), eg.find(b), "union did not take");

            eg.saturate_with_limits(20, max_minted, std::time::Duration::from_secs(30));

            assert_eq!(
                eg.find(a),
                eg.find(b),
                "budget {max_minted} retracted a hand-made equality"
            );

            // Refinement: every pair equal at a smaller budget is still
            // equal at this one.
            let pairs = equal_pairs(&eg, &probes);
            if let Some((smaller, before)) = &previous {
                for pair in before {
                    assert!(
                        pairs.contains(pair),
                        "budget {max_minted} lost probe pair {pair:?} that budget \
                         {smaller} had found — saturation is not monotone"
                    );
                }
            }
            ever_found = ever_found.max(pairs.len());
            previous = Some((max_minted, pairs));
        }

        // Guard against the ladder asserting nothing: if no budget ever
        // derived an equality, the refinement check above was vacuous.
        assert!(
            ever_found > 0,
            "no budget derived any probe equality — the monotonicity ladder \
             is vacuous and no longer tests L2"
        );
    }

    /// A rule's scan visits exactly the live classes that can root it, in
    /// ascending order: a merge carries the absorbed class's kinds to the
    /// survivor.
    #[test]
    fn candidate_classes_are_the_live_classes_that_can_root_the_rule() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let product = eg.add(ENode::apply(OpKind::Mul, &[x, y]));
        let neg = eg.add(ENode::apply(OpKind::Neg, &[sum]));
        // `product` is merged into `sum`'s class (the smaller id survives),
        // so the survivor now roots both `Add` and `Mul`.
        let merged = eg.union(sum, product);
        eg.rebuild();

        let candidates = |roots: RootShapes| eg.candidate_classes(roots).collect::<Vec<_>>();
        assert_eq!(candidates(RootShapes::kind(OpKind::Mul)), vec![merged]);
        assert_eq!(candidates(RootShapes::kind(OpKind::Add)), vec![merged]);
        assert_eq!(candidates(RootShapes::kind(OpKind::Neg)), vec![neg]);
        assert_eq!(
            candidates(RootShapes::kinds(&[OpKind::Neg, OpKind::Var])),
            vec![x, y, neg]
        );
        assert_eq!(candidates(RootShapes::nothing()), Vec::<EClassId>::new());
        assert_eq!(
            candidates(RootShapes::Any),
            eg.class_ids().collect::<Vec<_>>()
        );
    }

    /// A class that merges away keeps its bits in the per-kind sets, but it is
    /// not live, so a scan never visits it.
    #[test]
    fn a_merged_away_class_is_not_a_candidate() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let c = eg.add(ENode::constant(1.0));
        let survivor = eg.union(x, c);
        eg.rebuild();
        assert_eq!(survivor, x);
        let candidates = eg
            .candidate_classes(RootShapes::kind(OpKind::Const))
            .collect::<Vec<_>>();
        assert_eq!(candidates, vec![x], "the survivor now holds the constant");
    }

    /// A prediction does not count what an earlier one left in the scratch
    /// table the caller keeps: its nodes are not this prediction's to mint.
    #[test]
    fn a_prediction_ignores_what_the_scratch_held() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = RewriteAction::Create(ENode::apply(OpKind::Add, &[x, y]));
        let product = RewriteAction::Create(ENode::apply(OpKind::Mul, &[x, y]));

        let mut scratch = HashMap::default();
        assert_eq!(eg.predicted_growth(&sum, &mut scratch), 1);
        assert_eq!(
            eg.predicted_growth(&product, &mut scratch),
            1,
            "the sum the first prediction minted is not the product's growth"
        );
    }

    /// Freezing moves the arena rather than copying it: the cache's peak is the
    /// graph's, not twice it.
    #[test]
    fn freezing_moves_the_arena() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, one]));
        let tag = eg.tags(sum).next().expect("a class has a member");
        let before: *const ENode = eg.as_frozen().node(tag);
        let frozen = eg.freeze();
        assert!(core::ptr::eq(frozen.node(tag), before));
        assert_eq!(frozen.find(sum), sum);
        assert_eq!(frozen.nodes(sum).count(), 1);
    }

    /// A frozen graph keeps what extraction reads, as literal facts about the
    /// recipe of `a_node_whose_union_is_refused_leaves_its_class` (same ids):
    /// each class's members and tags in order, the retired node answered by no
    /// class, a live node answered in its own class only, the ids minted, and
    /// the variance of a unit admitted before the freeze.
    #[test]
    fn a_frozen_graph_keeps_what_extraction_reads() {
        let admitted = KernelStore::intern(&Kernel::x().add(&Kernel::constant(0.75)));
        let never_admitted = KernelStore::intern(&Kernel::y().add(&Kernel::constant(0.75)));
        let mut eg = EGraph::new();
        eg.admit_unit(admitted, Variance::X);
        let one = eg.add(ENode::constant(1.0)); // 0
        let two = eg.add(ENode::constant(2.0)); // 1
        let a = eg.add(ENode::Var(0)); // 2
        let b = eg.add(ENode::Var(1)); // 3
        let neg_a = eg.add(ENode::apply(OpKind::Neg, &[a])); // 4
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // 5
        eg.union(neg_a, one);
        eg.union(neg_b, two);
        eg.union(a, b);
        eg.rebuild();
        let retired = ENodeId(5);
        assert!(
            !eg.is_member(retired),
            "precondition: the recipe retires node 5"
        );

        let frozen = eg.freeze();

        assert_eq!(frozen.num_classes(), 6, "six ids were minted");
        assert_eq!(frozen.find(b), a, "the smaller id survives");
        assert_eq!(frozen.find(neg_a), one);
        assert_eq!(frozen.find(neg_b), two);
        assert_eq!(
            frozen.nodes(one).copied().collect::<Vec<_>>(),
            [ENode::constant(1.0), ENode::apply(OpKind::Neg, &[a])]
        );
        assert_eq!(
            frozen.tags(one).collect::<Vec<_>>(),
            [ENodeId(0), ENodeId(4)]
        );
        assert_eq!(
            frozen.nodes(two).copied().collect::<Vec<_>>(),
            [ENode::constant(2.0)]
        );
        assert_eq!(frozen.tags(two).collect::<Vec<_>>(), [ENodeId(1)]);
        assert_eq!(
            frozen.nodes(b).copied().collect::<Vec<_>>(),
            [ENode::Var(0), ENode::Var(1)],
            "a union gives the survivor's members, then the absorbed ones"
        );
        assert_eq!(frozen.tags(a).collect::<Vec<_>>(), [ENodeId(2), ENodeId(3)]);

        assert_eq!(frozen.node_for_tag(two, retired), None);
        assert_eq!(frozen.node_for_tag(one, retired), None);
        assert_eq!(
            frozen.node_for_tag(one, ENodeId(0)),
            Some(&ENode::constant(1.0)),
            "a live node answers in its own class"
        );
        assert_eq!(
            frozen.node_for_tag(two, ENodeId(0)),
            None,
            "a live node answers nothing in another class"
        );
        assert_eq!(
            frozen.node_for_tag(one, ENodeId(1_000)),
            None,
            "a tag past the graph names nothing"
        );

        assert_eq!(frozen.unit_variance(admitted), Some(Variance::X));
        assert_eq!(frozen.ref_variance(admitted), Variance::X);
        assert_eq!(frozen.unit_variance(never_admitted), None);
    }
}

#[cfg(test)]
mod variance_fact_tests {
    use super::*;
    use crate::egraph::fold_rules::{HeadNode, HeadRef, Plan};
    use pixelflow_ir::{Binder, Bound, Fold, Monoid};

    /// **Free of what any member is free of.** `X − X` is seeded `{X}` from
    /// its children; the graph then learns it equals `0`, and the class is
    /// `{X} ∩ ∅ = ∅`. A parent seeded before the union keeps its larger fact
    /// — no upward repair — which is still a superset of the truth, so a rule
    /// reading it can miss the invariance but never invent one.
    #[test]
    fn a_class_is_free_of_what_any_member_is_free_of() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let x_minus_x = eg.add(ENode::apply(OpKind::Sub, &[x, x]));
        let parent = eg.add(ENode::apply(OpKind::Mul, &[x_minus_x, y]));
        assert_eq!(eg.variance(x_minus_x), Variance::X, "seeded from X and X");
        assert_eq!(eg.variance(parent), Variance::COORDS);

        let zero = eg.add(ENode::constant(0.0));
        eg.union(x_minus_x, zero);
        eg.rebuild();

        assert_eq!(
            eg.variance(x_minus_x),
            Variance::CONST,
            "one member proves the class constant, so the class is"
        );
        assert!(
            eg.variance(parent).depends_on_x(),
            "the parent's seed is stale and says so: still a superset of the \
             truth ({{Y}}), which is all soundness needs"
        );
    }

    /// **The meet is an intersection, not a pick.** A class proved equal to
    /// both a Y-only and an X-only term varies with neither — a popcount
    /// minimum would have kept one of the two.
    #[test]
    fn two_disjoint_members_leave_nothing() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sin_x = eg.add(ENode::apply(OpKind::Sin, &[x]));
        let sin_y = eg.add(ENode::apply(OpKind::Sin, &[y]));
        eg.union(sin_x, sin_y);
        eg.rebuild();
        assert_eq!(eg.variance(sin_x), Variance::CONST);
    }

    /// **A fold frees its binder**, and only its binder: `Σ_i (i · X)` varies
    /// with `X`, its body with both.
    #[test]
    fn a_fold_removes_its_binder() {
        let mut eg = EGraph::new();
        let binder = Binder::from_slot(0).expect("a live binder");
        let i = eg.add(ENode::Var(binder.var()));
        let x = eg.add(ENode::Var(0));
        let body = eg.add(ENode::apply(OpKind::Mul, &[i, x]));
        let fold = eg.add(ENode::reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Lit(0),
            Bound::Lit(8),
            body,
        ));
        assert_eq!(
            eg.variance(body),
            Variance::X.union(Variance::from_var(binder.var()))
        );
        assert_eq!(eg.variance(fold), Variance::X);
    }

    /// A sum over slot 0 whose body is `i`, with `hi` the class `bound`.
    fn sum_to(eg: &mut EGraph, bound: EClassId) -> ENode {
        let binder = Binder::from_slot(0).expect("a live binder");
        let body = eg.add(ENode::Var(binder.var()));
        ENode::reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Lit(0),
            Bound::Value(bound),
            body,
        )
    }

    /// **L3's backstop: a bound that reads a coordinate is refused** (L3 of
    /// docs/language/2026-10-20-a-bound-is-a-value.md, not the optimizer's L3).
    /// The trip count would be a function of the pixel.
    #[test]
    #[should_panic(expected = "BoundVaries")]
    fn add_refuses_a_bound_varying_with_x() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let fold = sum_to(&mut eg, x);
        let _ = eg.add(fold);
    }

    /// **L3's backstop: a bound that reads its own fold's binder is refused.**
    /// The trip count would be a function of itself.
    #[test]
    #[should_panic(expected = "BoundVaries")]
    fn add_refuses_a_bound_reading_its_own_binder() {
        let mut eg = EGraph::new();
        let own = eg.add(ENode::Var(
            Binder::from_slot(0).expect("a live binder").var(),
        ));
        let fold = sum_to(&mut eg, own);
        let _ = eg.add(fold);
    }

    /// **An enclosing fold's index is a bound's to read.** `Σ_{j<i}` over the
    /// outer index `i` (slot 1) is added, and its class varies with the
    /// outer index, which the inner fold does not bind: its body less its own
    /// index, and what its bound reads (L4 of
    /// docs/language/2026-10-20-a-bound-is-a-value.md).
    #[test]
    fn add_takes_a_bound_reading_an_enclosing_binder() {
        let mut eg = EGraph::new();
        let outer = eg.add(ENode::Var(
            Binder::from_slot(1).expect("a live binder").var(),
        ));
        let fold = sum_to(&mut eg, outer);
        let class = eg.add(fold);
        assert_eq!(
            eg.variance(class),
            Variance::from_var(Binder::from_slot(1).expect("a live binder").var())
        );
    }

    /// A plan of one bound node, `bound`, and the sum `Σ_{j < bound} j`, whose
    /// bound is that node.
    fn a_sum_to_the_bound_built_by(bound: HeadNode, body: EClassId) -> Plan {
        let fold = Fold::new(Monoid::SUM, Binder::from_slot(0).expect("a live binder"));
        Plan {
            nodes: alloc::vec![
                bound,
                HeadNode::Reduce {
                    fold,
                    lo: Bound::Lit(0),
                    hi: Bound::Value(HeadRef::Plan(0)),
                    body: HeadRef::Class(body),
                },
            ],
            root: HeadRef::Plan(1),
        }
    }

    /// **A plan whose bound lands in a varying class is declined, by the graph
    /// and by its prediction alike** (L3 of
    /// docs/language/2026-10-20-a-bound-is-a-value.md). A fact is seeded at
    /// birth and never repaired upward, so `(X − X) + 1`, once `X − X` is joined
    /// to `0`, is a class holding the node `0 + 1` whose fact still reads `X`.
    /// A plan that builds `0 + 1` as a bound hash-conses into it, and `add`
    /// would panic on a bound that is, as a term, a constant. The plan builder
    /// asks `refuses` first: it declines, nothing is unioned, the nodes built
    /// before the refusal stay, and the prediction counts exactly those.
    ///
    /// Three bounds, each over the one body `j`: one that lands in the stale
    /// class (nothing is built), one this plan mints that reads `X` (the minted
    /// node stays, and the prediction reads its fact off the node it minted),
    /// and a constant (the fold is built).
    #[test]
    fn a_plan_whose_bound_lands_in_a_varying_class_is_declined_and_predicted_so() {
        let binder = Binder::from_slot(0).expect("a live binder");
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));
        let j = eg.add(ENode::Var(binder.var()));
        let x_minus_x = eg.add(ENode::apply(OpKind::Sub, &[x, x]));
        let stale = eg.add(ENode::apply(OpKind::Add, &[x_minus_x, one]));
        // `x_minus_x` is the older class, so it survives the union and the stale
        // class's node is already in its canonical form: a lookup of `0 + 1`
        // finds it.
        let zero = eg.add(ENode::constant(0.0));
        eg.union(x_minus_x, zero);
        eg.rebuild();
        assert!(
            eg.variance(stale).depends_on_x(),
            "the premise: the class is `0 + 1` and its fact still reads X"
        );

        let landing = a_sum_to_the_bound_built_by(
            HeadNode::Op {
                op: OpKind::Add,
                children: alloc::vec![HeadRef::Class(zero), HeadRef::Class(one)],
            },
            j,
        );
        let minted = a_sum_to_the_bound_built_by(
            HeadNode::Op {
                op: OpKind::Add,
                children: alloc::vec![HeadRef::Class(x), HeadRef::Class(one)],
            },
            j,
        );
        let constant = a_sum_to_the_bound_built_by(
            HeadNode::Op {
                op: OpKind::Add,
                children: alloc::vec![HeadRef::Class(one), HeadRef::Class(one)],
            },
            j,
        );

        let mut scratch = HashMap::default();
        for (name, plan, built, declined) in [
            ("a bound that lands in the stale class", landing, 0, true),
            ("a bound this plan mints reading X", minted, 1, true),
            ("a constant bound", constant, 2, false),
        ] {
            let predicted = eg.predicted_growth(&RewriteAction::Plan(plan.clone()), &mut scratch);
            let classes = eg.num_classes();
            if declined {
                assert_eq!(
                    eg.apply_action(x, RewriteAction::Plan(plan)),
                    0,
                    "{name}: a declined plan unions nothing"
                );
            } else {
                assert!(plan_shape(&mut eg, &plan).is_some(), "{name}: built");
            }
            assert_eq!(eg.num_classes() - classes, built, "{name}: classes built");
            assert_eq!(predicted, built, "{name}: the prediction counts the same");
        }
    }

    /// **A cycle does not certify invariance.** Once `X + 0` is merged with
    /// `X`, the class holds a member whose child is the class itself. Every
    /// fact in it was seeded from facts that already existed — `X`'s own bit
    /// — and never from an assumption about the class, so the cycle leaves
    /// `{X}` standing. A least fixpoint started from `∅` closes at `∅`
    /// instead: `var(C) = {X} ∩ var(C + 0)` and `var(C + 0) = var(C)`, so
    /// `∅` is a fixpoint, and it is false.
    #[test]
    fn a_cycle_does_not_certify_invariance() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let x_plus_0 = eg.add(ENode::apply(OpKind::Add, &[x, zero]));
        eg.union(x_plus_0, x);
        eg.rebuild();

        let class = eg.find(x);
        assert!(
            eg.nodes(class)
                .any(|n| n.children_slice().iter().any(|&c| eg.find(c) == class)),
            "precondition: the class reaches itself"
        );
        assert_eq!(eg.variance(class), Variance::X);
    }

    /// A journal is chosen when the graph is built: dropping it from a graph
    /// that already holds classes would forget their births.
    #[test]
    #[should_panic(expected = "already holds")]
    fn a_journal_cannot_be_dropped_from_a_populated_graph() {
        let mut egraph = EGraph::new();
        let _x = egraph.add(ENode::Var(0));
        let _journalless = egraph.without_journal();
    }

    /// The fact survives a clone — a search branch reads the same facts its
    /// parent had.
    #[test]
    fn a_clone_keeps_the_facts() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let copy = eg.clone();
        assert_eq!(copy.variance(sum), Variance::COORDS);
    }
}
