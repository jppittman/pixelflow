//! The E-Graph data structure and operations.

use std::collections::HashMap;

use super::cost::{CostFunction, CostModel};
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
use super::provenance::{ApplicationRecord, ENodeId, Origin, Provenance, UnionEvent};
use super::rewrite::{Rewrite, RewriteAction};
use super::rules::RuleId;
use pixelflow_ir::kind::OpKind;

/// A potential rewrite target: (rule, e-class, node within class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RewriteTarget {
    /// Index into the e-graph's rule list
    pub rule_idx: usize,
    /// The e-class to apply the rule to
    pub class_id: EClassId,
    /// The node within the e-class that the rule should try to match
    pub node_idx: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EClass {
    pub(crate) nodes: Vec<ENode>,
    /// Stable identity per node, parallel to `nodes`: `tags[i]` names
    /// `nodes[i]`. Must be kept in lockstep with `nodes` through every
    /// mutation (union's extend, rebuild's take/canonicalize/extend) — see
    /// `provenance` module docs.
    pub(crate) tags: Vec<ENodeId>,
}

/// Context describing which rewrite application (if any) is currently
/// responsible for e-nodes/unions created by `add()`/`union()`. Set for the
/// duration of `apply_action_from_rule`, `None` otherwise (e.g. during
/// `rebuild_budgeted`'s congruence-closure unions, or seed insertion).
#[derive(Clone, Copy, Debug)]
struct ActiveApplication {
    rule_idx: usize,
    application_id: super::provenance::ApplicationId,
}

/// Which direction `rebuild` repairs congruence in.
///
/// An e-graph is a congruence closure: if `x = y` then `f(x) = f(y)` for
/// every `f` the graph holds. `union` establishes `x = y` and pushes the
/// merged class; [`Downward`](Congruence::Downward) then re-canonicalizes
/// *that class's own* e-nodes and dedups them through the memo. Nothing looks
/// at the e-nodes that reference the merged class as a **child**, so
/// `Add(x, z)` and `Add(y, z)` merge only if a later rule sweep happens to
/// re-walk them.
///
/// That under-merges. It is sound — every class still holds only
/// provably-equal terms, so law L1 holds and L2 explicitly permits a graph
/// with fewer equalities than an exhaustive run would hold — but it costs
/// CSE, and it makes *which* equalities the graph finds depend on rule order.
///
/// [`Upward`](Congruence::Upward) closes it the standard way: each class
/// records the classes owning an e-node that references it, and repairing a
/// class re-enqueues those owners so their now-stale forms are
/// re-canonicalized and merged through the memo.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Congruence {
    /// Repair only the merged class's own e-nodes. What this graph has always
    /// done, and the default so that constructing an `EGraph` directly is not
    /// silently a different algorithm than it was.
    #[default]
    Downward,
    /// Also repair the e-nodes that reference the merged class (upward
    /// merging), to fixpoint within the rebuild budget.
    Upward,
}

pub struct EGraph {
    pub(crate) classes: Vec<EClass>,
    pub(crate) parent: Vec<EClassId>,
    memo: HashMap<ENode, EClassId>,
    worklist: Vec<EClassId>,
    /// Which direction [`EGraph::rebuild_budgeted`] repairs congruence in.
    congruence: Congruence,
    /// For each class, the classes owning an e-node that names it as a child.
    /// Populated only under [`Congruence::Upward`] — the entry exists for
    /// every class either way (it indexes by class id) but stays empty under
    /// [`Congruence::Downward`], so that arm pays 24 bytes a class and no
    /// work.
    ///
    /// Owner ids, not the `(ENode, EClassId)` pairs egg stores. Re-enqueueing
    /// the owning *class* routes repair through the existing downward loop,
    /// which already canonicalizes a class's nodes and merges memo duplicates
    /// — so the node copy is not needed to find the congruence. egg keeps it
    /// to evict the stale memo key; here that key is unreachable rather than
    /// wrong, because a class merged away never becomes canonical again and
    /// so no canonicalized lookup can ever name it. The entry is garbage that
    /// costs memory, not a result anything can read.
    parents: Vec<Vec<EClassId>>,
    /// Owner classes re-enqueued by upward repair. Cost accounting only —
    /// nothing reads it to decide anything.
    upward_enqueues: u64,
    /// Rules are shared via Arc so EGraph can be cloned for search branching.
    rules: std::sync::Arc<Vec<Box<dyn Rewrite>>>,
    /// Stable id per rule, parallel to `rules`. Kept so the graph can name a
    /// rule by something that survives a reordering; see [`super::rules`].
    rule_ids: std::sync::Arc<Vec<RuleId>>,
    /// Matches found per rule, keyed by identity rather than by family name.
    /// A name-keyed map collapsed all four `Commutative` instances into one
    /// bucket, so every per-rule report built on it was wrong by aggregation.
    pub match_counts: HashMap<RuleId, usize>,
    /// Global monotonic counter minting `ENodeId`s in `add()`.
    next_enode_id: u64,
    /// Saturation-iteration counter, advanced once per `saturate_with_limits`
    /// loop iteration. Recorded on every `ApplicationRecord`/`UnionEvent`.
    step: usize,
    /// Rule provenance: origins, application log, union journal.
    provenance: Provenance,
    /// Which rewrite application (if any) is currently executing — read by
    /// `add()`/`union()` to attribute newly created nodes/unions.
    active_application: Option<ActiveApplication>,
    /// The constant each class is known to equal, as f32 bits, indexed by
    /// class id — maintained independently of `EClass::nodes` on purpose.
    /// `rebuild` drains a class's nodes with `mem::take` and only THEN
    /// performs congruence unions, so a guard that scanned the node vector
    /// saw an empty class and waved contradictory merges through at exactly
    /// the moment congruence closure does its work. The fact must outlive the
    /// nodes.
    const_fact: Vec<Option<u32>>,
    /// Unions REFUSED because they would assert two numerically unequal
    /// constants equal — a proved falsehood the graph declines to absorb.
    /// Distinct (bits, bits) pairs, kept for reporting and tests; see
    /// [`EGraph::union`].
    refused_const_unions: Vec<(u32, u32)>,
    /// Rule applications this graph has performed, counted unconditionally.
    ///
    /// Independent of the provenance log on purpose: an application budget
    /// must be enforceable whether or not anyone is observing, and reading
    /// the budget off `provenance().application_count()` would make
    /// recording a load-bearing part of saturation rather than an
    /// observation of it.
    applications: u64,
    /// Whether to record provenance. Recording had no production consumer
    /// (2026-09-01 integration audit) and a production compile discarded a
    /// median 8 446 records per kernel;
    /// [`super::optimizer::Optimizer`] turns it on exactly when an
    /// [`Observer`](super::optimizer::Observer) is attached. Defaults to
    /// `true`, so every direct `EGraph` caller keeps what it had.
    record_provenance: bool,
    /// Application ceiling for the current run, or `None`. Set for the
    /// duration of [`EGraph::saturate_budgeted`].
    application_cap: Option<u64>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EGraph {
    fn clone(&self) -> Self {
        Self {
            classes: self.classes.clone(),
            parent: self.parent.clone(),
            memo: self.memo.clone(),
            worklist: self.worklist.clone(),
            congruence: self.congruence,
            parents: self.parents.clone(),
            upward_enqueues: self.upward_enqueues,
            rules: self.rules.clone(), // Arc clone - cheap, shares rules
            rule_ids: self.rule_ids.clone(),
            match_counts: self.match_counts.clone(),
            next_enode_id: self.next_enode_id,
            step: self.step,
            provenance: self.provenance.clone(),
            active_application: self.active_application,
            const_fact: self.const_fact.clone(),
            refused_const_unions: self.refused_const_unions.clone(),
            applications: self.applications,
            record_provenance: self.record_provenance,
            application_cap: self.application_cap,
        }
    }
}

/// Result of applying a single rule: changes made and evaluations consumed.
///
/// `changes` counts union/create actions. `evals` counts rule match
/// attempts (one per node checked). Evals model compute cost — the
/// Guide learns to stay within an eval budget just as it learns to
/// stay within a node budget.
pub struct ApplyResult {
    pub changes: usize,
    pub evals: usize,
    /// How the scan ended. `changes == 0` is quiescence only when this is
    /// [`ScanStop::Completed`]; under either budget the rest of the graph
    /// was never looked at, so nothing is known about it.
    pub scan: ScanStop,
}

/// How a single rule's scan over the e-graph ended.
///
/// Two budgets can cut a scan short and they are *different facts*, so this
/// is a three-valued type rather than a `truncated: bool` the caller has to
/// disambiguate afterwards by asking which budget looks closer — that
/// inference is exactly what a stop reason exists to replace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanStop {
    /// Every e-class was visited and every match committed.
    Completed,
    /// The class budget `max_nodes` stopped the scan: the graph (plus the
    /// nodes the pending actions would create) reached the cap.
    ClassCap,
    /// The wall-clock deadline elapsed before the scan finished — either it
    /// cut the walk short, or it passed while a rule's `apply` (or the
    /// commit that follows) was running. Both mean the same thing: this
    /// scan did not finish inside the ceiling.
    Deadline,
    /// The rule-application budget was reached mid-scan.
    ApplicationBudget,
}

/// Why an [`EGraph::saturate_with_limits`] call stopped — an explicit stop
/// reason instead of the `SaturationStats` proxy game every measurement
/// harness had to play (`iterations < max_iters && classes <= cap` ≈
/// "probably quiesced"; `pixelflow-pipeline/src/bin/guide_headroom.rs` names
/// it as the one ambiguity that proxy can't resolve). Read off the loop that
/// decides when to stop, never inferred from the counts afterwards.
/// Budget-only framing: none of these certify a fixpoint;
/// [`SaturationStop::Quiesced`] is a diagnostic condition (one full rule
/// sweep ran to completion and produced zero unions), never a closure claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaturationStop {
    /// A full rule sweep completed with zero unions. Diagnostic, not a
    /// certified fixpoint.
    Quiesced,
    /// The class budget `max_classes` stopped the run (memory protection) —
    /// either the count exceeded it outright, or a sweep produced zero
    /// unions only because every remaining action was discarded for budget.
    ClassCap,
    /// `max_iters` sweeps completed without any other condition firing.
    IterationCeiling,
    /// The wall-clock safety ceiling elapsed. Offline measurement callers
    /// should treat this as a hard error (fail loud), never as data.
    ///
    /// [`EGraph::saturate_budgeted`] can never report this: it takes no
    /// clock, which is what makes it deterministic.
    Timeout,
    /// The rule-application budget ran out. The one budget dimension that
    /// means the same thing to every ordering policy, and so the one a
    /// research arm can hold two policies to.
    ApplicationBudget,
}

/// Result of one [`EGraph::saturate_with_limits`] run: how many rounds it
/// took and how many rule applications fired in total, whichever limit
/// (iteration count, class count, timeout, or convergence) ended the run.
///
/// Deliberately not `Default`: a defaulted `stop` would be an invented
/// stop reason, and nothing constructs an empty stats value.
#[derive(Clone, Copy, Debug)]
pub struct SaturationStats {
    /// Number of rewrite rounds completed before the run stopped.
    pub iterations: usize,
    /// Sum of `ApplyResult::changes` across every round.
    pub total_unions: usize,
    /// Which condition ended the run — READ from the loop that stopped, so a
    /// caller never has to infer "quiesced" from `iterations < max_iters`
    /// (which conflates a timeout or class cap with quiescence).
    pub stop: SaturationStop,
}

/// The ceiling on every class budget: no saturation driver grows the graph
/// past this many e-classes, whatever cap its caller asks for.
///
/// This is memory protection for call sites that pass no meaningful cap
/// (`usize::MAX`, or a number derived from an unbounded input). It lives
/// here — at the growth decision — and NOT inside [`EGraph::add`], because
/// `add` is the homomorphism onto the semantic quotient (law L1,
/// `docs/plans/2026-09-02-optimizer-api.md` §1): a budget that stops `add`
/// must hand the caller something that is not the class of the node it
/// asked for, and there is no such thing. A budget that stops the *driver*
/// leaves every class the graph does hold correct, which is what
/// truncation has always meant here.
///
/// Two orders of magnitude above the production caps (500/2000/5000, see
/// `SaturationConfig`), so no shipping configuration meets it.
///
/// It bounds the graph *approximately*: a sweep estimates the classes a
/// pending action will mint (`RewriteAction::Union` 0, `Create` 1, every
/// other variant 3) and then commits the batch it accepted, so a
/// multi-node action can overshoot by the difference. That is precisely
/// why the limit can live at the driver — it guards memory, not meaning,
/// and a bounded overshoot of a 100 000-class ceiling costs memory nobody
/// was going to miss. The old in-`add` limit bought exactness by returning
/// `EClassId(0)`, a false class id that the `Create` handler then unioned
/// against the match — asserting an equality that does not hold (#1105).
pub const HARD_CLASS_LIMIT: usize = 100_000;

impl EGraph {
    /// Create an empty e-graph with no rewrite rules.
    ///
    /// Rules are application-defined. Use `with_rules()` or `add_rule()` to add them.
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
            congruence: Congruence::Downward,
            parents: Vec::new(),
            upward_enqueues: 0,
            rules: std::sync::Arc::new(Vec::new()),
            rule_ids: std::sync::Arc::new(Vec::new()),
            match_counts: HashMap::new(),
            next_enode_id: 0,
            step: 0,
            provenance: Provenance::new(),
            active_application: None,
            const_fact: Vec::new(),
            refused_const_unions: Vec::new(),
            applications: 0,
            record_provenance: true,
            application_cap: None,
        }
    }

    /// Create an e-graph with the given rewrite rules.
    ///
    /// Rules are owned by the e-graph and shared via Arc when cloned.
    pub fn with_rules(rules: Vec<Box<dyn Rewrite>>) -> Self {
        let ids: Vec<RuleId> = rules.iter().map(|r| RuleId::of(r.as_ref())).collect();
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
            congruence: Congruence::Downward,
            parents: Vec::new(),
            upward_enqueues: 0,
            rules: std::sync::Arc::new(rules),
            rule_ids: std::sync::Arc::new(ids),
            match_counts: HashMap::new(),
            next_enode_id: 0,
            step: 0,
            provenance: Provenance::new(),
            active_application: None,
            const_fact: Vec::new(),
            refused_const_unions: Vec::new(),
            applications: 0,
            record_provenance: true,
            application_cap: None,
        }
    }

    /// Add a rule to this e-graph.
    ///
    /// # Panics
    ///
    /// Panics if the e-graph has been cloned (rules are shared via Arc).
    pub fn add_rule(&mut self, rule: Box<dyn Rewrite>) {
        let id = RuleId::of(rule.as_ref());
        std::sync::Arc::get_mut(&mut self.rules)
            .expect("Cannot add rules after EGraph has been cloned")
            .push(rule);
        std::sync::Arc::get_mut(&mut self.rule_ids)
            .expect("Cannot add rules after EGraph has been cloned")
            .push(id);
    }

    /// An e-graph over a rule vector that is already shared.
    ///
    /// [`super::RuleSet`] holds its rules behind an `Arc` so an
    /// [`super::optimizer::Optimizer`] can hand the same vocabulary to as
    /// many graphs as it likes without rebuilding it, and so the ids it
    /// computed once stay in agreement with the rules the graph applies.
    ///
    /// # Panics
    ///
    /// Panics if the two vectors disagree in length — an id vector that does
    /// not name exactly the rules the graph will apply is a
    /// silently-wrong-answer generator, not a recoverable input.
    #[must_use]
    pub fn with_shared_rules(
        rules: std::sync::Arc<Vec<Box<dyn Rewrite>>>,
        rule_ids: std::sync::Arc<Vec<RuleId>>,
    ) -> Self {
        assert_eq!(
            rules.len(),
            rule_ids.len(),
            "EGraph::with_shared_rules: {} rules but {} ids",
            rules.len(),
            rule_ids.len()
        );
        let mut eg = Self::new();
        eg.rules = rules;
        eg.rule_ids = rule_ids;
        eg
    }

    /// Choose which direction [`rebuild_budgeted`](EGraph::rebuild_budgeted)
    /// repairs congruence in. See [`Congruence`].
    ///
    /// # Panics
    ///
    /// Panics unless the graph is still empty. The upward edge list is built
    /// incrementally by [`add`](EGraph::add), so switching to
    /// [`Congruence::Upward`] on a populated graph would leave every existing
    /// e-node unregistered — an upward repair that silently skips most of the
    /// graph, which is the one failure mode this whole change exists to
    /// remove. Choose at construction.
    #[must_use]
    pub fn with_congruence(mut self, congruence: Congruence) -> Self {
        assert!(
            self.classes.is_empty(),
            "EGraph::with_congruence must be chosen before the first add(): \
             the upward edge list is built by add(), so switching on a graph \
             that already holds {} classes would repair only the nodes added \
             afterwards",
            self.classes.len()
        );
        self.congruence = congruence;
        self
    }

    /// Which direction this graph repairs congruence in.
    #[must_use]
    pub fn congruence(&self) -> Congruence {
        self.congruence
    }

    /// The stable id of the rule at `idx`, if there is one.
    #[must_use]
    pub fn rule_id(&self, idx: usize) -> Option<RuleId> {
        self.rule_ids.get(idx).copied()
    }

    /// Rule applications performed since this graph was created.
    ///
    /// Counted whether or not provenance is being recorded — the budget and
    /// the observation are separate concerns, and conflating them is what
    /// made recording impossible to turn off.
    #[must_use]
    pub fn application_count(&self) -> u64 {
        self.applications
    }

    /// Turn provenance recording on or off. On by default.
    ///
    /// Off, the graph still counts applications and still enforces an
    /// application budget; it just does not build the log. Turning recording
    /// off does not erase what is already recorded.
    pub fn set_provenance_recording(&mut self, on: bool) {
        self.record_provenance = on;
    }

    /// Whether provenance is being recorded.
    #[must_use]
    pub fn provenance_recording(&self) -> bool {
        self.record_provenance
    }

    pub fn find(&self, id: EClassId) -> EClassId {
        let mut current = id;
        while self.parent[current.index()] != current {
            current = self.parent[current.index()];
        }
        current
    }

    fn find_mut(&mut self, id: EClassId) -> EClassId {
        let mut current = id;
        let mut path = Vec::new();
        while self.parent[current.index()] != current {
            path.push(current);
            current = self.parent[current.index()];
        }
        for node in path {
            self.parent[node.index()] = current;
        }
        current
    }

    fn canonicalize_node(&self, node: &mut ENode) {
        match node {
            ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => {}
            ENode::Op { children, .. } => {
                for child in children {
                    *child = self.find(*child);
                }
            }
        }
    }

    /// Insert `node`, returning the e-class that contains it.
    ///
    /// **Total, and the law depends on it.** The returned class always
    /// contains `node`: either the memo already had it, or this call
    /// allocated a fresh class holding exactly it. There is no size at
    /// which `add` returns something else, because there is nothing else
    /// it could correctly return — this is law L1 (`add` is a homomorphism
    /// from the term algebra onto the e-graph quotient,
    /// `docs/plans/2026-09-02-optimizer-api.md` §1), and L4 (any saturation
    /// policy preserves denotation) is proved from it.
    ///
    /// The class budget therefore is not enforced here. It is enforced
    /// where growth is *decided* — the saturation drivers, each of which
    /// clamps its caller's cap to [`HARD_CLASS_LIMIT`] and stops scanning
    /// before it calls `add`. A driver that stops early leaves a graph
    /// holding fewer equalities than an exhaustive run would; every one it
    /// does hold is still true.
    pub fn add(&mut self, mut node: ENode) -> EClassId {
        self.canonicalize_node(&mut node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        let id = EClassId(self.classes.len() as u32);
        let enode_id = ENodeId(self.next_enode_id);
        self.next_enode_id += 1;
        let origin = match self.active_application {
            Some(active) => Origin::Rule(active.application_id),
            None => Origin::Seed,
        };
        self.provenance.record_origin(enode_id, origin);
        self.const_fact.push(node.as_f32().map(f32::to_bits));
        self.classes.push(EClass {
            nodes: vec![node.clone()],
            tags: vec![enode_id],
        });
        self.parent.push(id);
        self.parents.push(Vec::new());
        // Register the upward edges before `memo.insert` consumes `node`.
        // `children()` returns an owned Vec, so this holds no borrow of
        // `node` across the `&mut self` accesses below.
        if self.congruence == Congruence::Upward {
            let mut registered: Vec<EClassId> = Vec::new();
            for child in node.children() {
                let child = self.find(child);
                // Dedup within this node's own children only (arity is 1–3):
                // `Mul(x, x)` must not enqueue its owner twice. Deduping
                // against the whole list would be quadratic in a class's
                // fan-in, and entries from *different* nodes are legitimate
                // duplicates anyway.
                if registered.contains(&child) {
                    continue;
                }
                registered.push(child);
                self.parents[child.index()].push(id);
            }
        }
        self.memo.insert(node, id);
        id
    }

    /// Refused constant unions: `(bits, bits)` pairs the graph declined to
    /// assert equal. Non-empty means some kernel hit the folding-vs-algebra
    /// collision — see the refusal block in [`EGraph::union`].
    #[must_use]
    pub fn refused_const_unions(&self) -> &[(u32, u32)] {
        &self.refused_const_unions
    }

    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let a = self.find_mut(a);
        let b = self.find_mut(b);
        if a == b {
            return a;
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
        // `ca == cb` (not bit equality) deliberately admits ±0.0 — signed
        // zero selection is already platform-unspecified, so an optimizer
        // choosing a sign is within contract, same license as commuting
        // Min/Max over NaN. Bit equality additionally admits identical NaN
        // patterns, which `==` would wrongly reject: an all-ones comparison
        // mask reads as NaN and must be unionable with itself.
        // Read the class-level fact, NOT the node vector: `rebuild` drains
        // nodes with `mem::take` before performing congruence unions, so a
        // node-scanning guard is blind exactly when congruence closure runs.
        let (fa, fb) = (self.const_fact[a.index()], self.const_fact[b.index()]);
        if let (Some(ba), Some(bb)) = (fa, fb) {
            let (ca, cb) = (f32::from_bits(ba), f32::from_bits(bb));
            if ca != cb && ba != bb {
                let pair = (ba, bb);
                if !self.refused_const_unions.contains(&pair) {
                    // Journal ALWAYS; print only on request. This fires
                    // during ordinary kernel compilation — two computation
                    // orders of one real quantity disagreeing by an ulp is
                    // enough — so unconditional stderr would spam every
                    // build. `refused_const_unions()` is the durable record.
                    if std::env::var_os("PIXELFLOW_REPORT_CONST_REFUSALS").is_some() {
                        std::eprintln!(
                            "pixelflow e-graph: refusing a union that would assert \
                             {ca} = {cb} (bits {:#010x} vs {:#010x}) — the rule set \
                             derived contradictory constants (f32 folding vs \
                             algebra at an ill-conditioned input); keeping the \
                             classes separate costs only missed CSE",
                            pair.0,
                            pair.1
                        );
                    }
                    self.refused_const_unions.push(pair);
                }
                return if a.0 < b.0 { a } else { b };
            }
        }
        let (parent, child) = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.parent[child.index()] = parent;
        // The e-nodes that referenced `child` now reference `parent`. Their
        // owners must be reachable from `parent`, or the upward repair that
        // runs when `parent` comes off the worklist would miss exactly the
        // nodes this union just invalidated.
        if self.congruence == Congruence::Upward {
            let child_parents = std::mem::take(&mut self.parents[child.index()]);
            self.parents[parent.index()].extend(child_parents);
        }
        let child_nodes = std::mem::take(&mut self.classes[child.index()].nodes);
        let child_tags = std::mem::take(&mut self.classes[child.index()].tags);
        self.classes[parent.index()].nodes.extend(child_nodes);
        self.classes[parent.index()].tags.extend(child_tags);
        // The guard above proved the two facts agree (or that at most one
        // exists), so `or` is a merge, not a choice.
        if self.const_fact[parent.index()].is_none() {
            self.const_fact[parent.index()] = self.const_fact[child.index()];
        }
        self.worklist.push(parent);
        self.provenance.record_union(UnionEvent {
            rule_idx: self.active_application.map(|a| a.rule_idx),
            step: self.step,
            class_a: parent,
            class_b: child,
        });
        parent
    }

    /// Begin a batch of rule applications. Returns a guard that rebuilds
    /// the e-graph when dropped (RAII). Rules applied through the guard
    /// skip per-rule rebuilds; the single rebuild on drop amortizes the cost.
    ///
    /// ```ignore
    /// {
    ///     let mut batch = egraph.batch();
    ///     batch.apply_rule(0, 500);
    ///     batch.apply_rule(1, 500);
    ///     // rebuild happens here on drop
    /// }
    /// ```
    /// Begin a batch of rule applications with interleaved partial rebuild.
    ///
    /// `rebuild_chunk`: max worklist items to process after each rule.
    /// Higher = more deduplication, slower per rule.
    /// Lower = less deduplication, faster per rule but classes grow.
    /// Default of 256 balances the two.
    pub fn batch(&mut self) -> EGraphBatch<'_> {
        EGraphBatch {
            graph: self,
            any_changes: false,
            rebuild_chunk: 256,
        }
    }

    /// Begin a batch with a custom rebuild chunk size.
    pub fn batch_with_chunk(&mut self, rebuild_chunk: usize) -> EGraphBatch<'_> {
        EGraphBatch {
            graph: self,
            any_changes: false,
            rebuild_chunk,
        }
    }

    /// Rebuild the e-graph completely. Processes the entire worklist.
    pub fn rebuild(&mut self) {
        self.rebuild_budgeted(usize::MAX);
    }

    /// Process up to `budget` worklist items. Each item canonicalizes one
    /// e-class's nodes and deduplicates via the memo table.
    ///
    /// The graph is consistent after each item — partially rebuilt is safe.
    /// Unprocessed classes may have stale canonical forms (rule matching
    /// might miss some equivalences) but won't produce wrong results.
    ///
    /// Returns the number of worklist items remaining.
    pub fn rebuild_budgeted(&mut self, budget: usize) -> usize {
        let mut processed = 0;
        while processed < budget {
            let id = match self.worklist.pop() {
                Some(id) => id,
                None => break,
            };
            processed += 1;
            let id = self.find(id);
            self.repair_class(id);
            self.repair_users_of(id);
        }
        self.worklist.len()
    }

    /// Upward repair: re-canonicalize the e-nodes that name `id` as a child
    /// and merge the memo duplicates among them.
    ///
    /// This is the half `union` never performs. `id` is on the worklist
    /// because its membership just changed, so every e-node holding it as a
    /// child now has a stale form — and two such nodes that differed only in
    /// naming the two classes this union joined are now *the same node*.
    /// [`repair_class`](EGraph::repair_class) is what notices that, through
    /// the memo, so there is one congruence rule in this file rather than a
    /// second copy here.
    ///
    /// **Termination**, on a graph that saturation has made cyclic: this adds
    /// no worklist items of its own. Items come only from `union`, and unions
    /// are bounded by the class count because each one strictly reduces the
    /// number of live classes. Repairing a user may union, which enqueues the
    /// survivor, whose own users are repaired when that item is processed —
    /// so the cascade is carried by the worklist and bounded with it.
    fn repair_users_of(&mut self, id: EClassId) {
        if self.congruence != Congruence::Upward {
            return;
        }
        // `find(id)`: repairing `id` above may have merged it away, in which
        // case `union` has already moved its user list onto the survivor.
        let canonical = self.find(id);
        let mut owners = std::mem::take(&mut self.parents[canonical.index()]);
        for owner in &mut owners {
            *owner = self.find(*owner);
        }
        owners.sort_unstable();
        owners.dedup();
        for &owner in &owners {
            if owner != self.find(canonical) {
                self.repair_class(owner);
                self.upward_enqueues += 1;
            }
        }
        // Extend rather than assign, and re-`find` the slot: a repair above
        // may have merged `canonical` away, and `union` would then have moved
        // fresh users onto the survivor's list.
        let slot = self.find(canonical);
        self.parents[slot.index()].extend(owners);
    }

    /// Canonicalize one e-class's own e-nodes and merge memo duplicates.
    /// Downward repair — what `rebuild` has always done per worklist item.
    fn repair_class(&mut self, id: EClassId) {
        let nodes = std::mem::take(&mut self.classes[id.index()].nodes);
        // `tags` must stay zipped with `nodes` through this loop: no
        // reordering happens (nodes are only appended to `new_nodes` in
        // the same order they're drained from `nodes`), so zipping by
        // index here and pushing to `new_tags` in lockstep with
        // `new_nodes` keeps every tag pointed at the right node.
        let tags = std::mem::take(&mut self.classes[id.index()].tags);
        debug_assert_eq!(
            nodes.len(),
            tags.len(),
            "EClass.nodes and EClass.tags must never desync"
        );
        let mut new_nodes = Vec::new();
        let mut new_tags = Vec::new();
        for (mut node, tag) in nodes.into_iter().zip(tags) {
            self.canonicalize_node(&mut node);
            if let Some(&existing) = self.memo.get(&node) {
                let existing = self.find(existing);
                if existing != id {
                    // `union` may pick either `id` or `existing` as the
                    // surviving parent. If `id` survives, `union`'s
                    // extend() appends `existing`'s nodes (and tags)
                    // directly onto `self.classes[id.index()]` — which we
                    // just emptied via mem::take above. We MUST extend
                    // (not overwrite) below, or those appended
                    // nodes/tags are silently dropped when we write
                    // `new_nodes`/`new_tags` back. This is a rebuild-time
                    // congruence-closure union, not a rule firing, so it
                    // carries no rule_idx in the provenance journal
                    // (`active_application` is whatever the caller left
                    // it as — normally `None` outside rule application).
                    self.union(id, existing);

                    // `union` REFUSES a merge that would assert two
                    // unequal constants equal, and a refusal here needs
                    // handling that a rule-driven refusal does not: the
                    // node below would be pushed back into `id` while
                    // `memo` names `existing`, leaving ONE ENode in two
                    // contradictory classes with only one of them
                    // reachable by lookup — later matching or extraction
                    // could then give the same expression different
                    // values. Leave the node with the class memo already
                    // names. `id` loses a node congruent to one it can no
                    // longer be merged with, which is under-merging: it
                    // costs CSE, not correctness.
                    if self.find(id) != self.find(existing) {
                        continue;
                    }
                }
            } else {
                self.memo.insert(node.clone(), id);
            }
            new_nodes.push(node);
            new_tags.push(tag);
        }
        // Extend, not assign: a mid-loop union() above may have already
        // pushed nodes/tags onto classes[id.index()] (see comment above).
        // Overwriting here would silently discard them.
        self.classes[id.index()].nodes.extend(new_nodes);
        self.classes[id.index()].tags.extend(new_tags);
    }

    /// Owner classes re-enqueued by upward congruence repair since this graph
    /// was created. Zero under [`Congruence::Downward`]. Cost accounting for
    /// the A/B in `docs/results/2026-09-02-upward-congruence-ab.md`.
    #[must_use]
    pub fn upward_enqueues(&self) -> u64 {
        self.upward_enqueues
    }

    /// Total upward-edge entries held across all classes — the memory this
    /// bookkeeping costs, in units of one `EClassId` (4 bytes) each, plus one
    /// `Vec` header per class. Zero under [`Congruence::Downward`].
    #[must_use]
    pub fn upward_edge_count(&self) -> usize {
        self.parents.iter().map(Vec::len).sum()
    }

    /// Number of pending worklist items (classes needing rebuild).
    pub fn pending_rebuilds(&self) -> usize {
        self.worklist.len()
    }

    pub fn nodes(&self, id: EClassId) -> &[ENode] {
        let id = self.find(id);
        &self.classes[id.index()].nodes
    }

    /// Get the stable `ENodeId` tags for the canonical class's nodes,
    /// parallel to `nodes(id)` — `tags(id)[i]` names `nodes(id)[i]`.
    pub fn tags(&self, id: EClassId) -> &[ENodeId] {
        let id = self.find(id);
        &self.classes[id.index()].tags
    }

    /// Access the rule-provenance side tables (origins, application log,
    /// union journal). See the `provenance` module for details.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Look up the `ENode` for a given stable tag within a class, if it's
    /// still present there. `O(class size)` — classes are small in practice
    /// (equality-saturated e-classes rarely exceed a few dozen nodes).
    pub fn node_for_tag(&self, id: EClassId, tag: ENodeId) -> Option<&ENode> {
        let id = self.find(id);
        let class = &self.classes[id.index()];
        class
            .tags
            .iter()
            .position(|&t| t == tag)
            .map(|i| &class.nodes[i])
    }

    /// Compute the transitive rewrite-application ancestry of a set of
    /// chosen `(EClassId, ENodeId)` pairs (typically the nodes an extraction
    /// pass selected). See [`super::provenance::derivation_ancestors`] for
    /// the exact over-approximation made.
    pub fn derivation_ancestors(
        &self,
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        let tags_of = |class: EClassId| -> Vec<ENodeId> { self.tags(class).to_vec() };
        let children_of = |tag: ENodeId| -> Vec<EClassId> {
            for class in self.classes.iter() {
                if let Some(idx) = class.tags.iter().position(|&t| t == tag) {
                    return class.nodes[idx].children();
                }
            }
            Vec::new()
        };
        super::provenance::derivation_ancestors(
            &tags_of,
            &children_of,
            &self.provenance,
            chosen_nodes,
        )
    }

    /// Render a human-readable derivation trace for the given ancestry set
    /// (from [`EGraph::derivation_ancestors`]), resolving rule names via
    /// this e-graph's rule list.
    pub fn format_derivation_trace(
        &self,
        ancestors: &std::collections::BTreeSet<super::provenance::ApplicationId>,
    ) -> String {
        let rule_name =
            |idx: usize| -> Option<String> { self.rule(idx).map(|r| r.name().to_string()) };
        super::provenance::format_derivation_trace(&self.provenance, ancestors, &rule_name)
    }

    /// Get the number of registered rewrite rules.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Get the number of e-classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Iterate over all canonical e-class IDs.
    ///
    /// Returns an iterator of all e-class IDs that are canonical (i.e., roots
    /// of their union-find tree) and have at least one node.
    pub fn class_ids(&self) -> impl Iterator<Item = EClassId> + '_ {
        (0..self.classes.len()).filter_map(move |idx| {
            let id = EClassId(idx as u32);
            let canonical = self.find(id);
            if canonical == id && !self.classes[idx].nodes.is_empty() {
                Some(id)
            } else {
                None
            }
        })
    }

    /// Collect all canonical e-class IDs into a `Vec`.
    ///
    /// Use this instead of `class_ids()` when the caller needs `&mut self`
    /// (since the iterator borrows `&self`). This is the single source of
    /// truth for "which classes are canonical" — delegates to `class_ids()`.
    pub fn canonical_class_ids(&self) -> Vec<EClassId> {
        self.class_ids().collect()
    }

    /// Get the total number of nodes across all e-classes.
    pub fn node_count(&self) -> usize {
        self.classes.iter().map(|c| c.nodes.len()).sum()
    }

    /// Get the OpKind of the canonical representative of an e-class.
    ///
    /// Resolves through union-find to the canonical class, then returns
    /// the OpKind of the first node in that class.
    pub fn canonical_op(&self, id: EClassId) -> pixelflow_ir::OpKind {
        let id = self.find(id);
        let class = &self.classes[id.index()];
        match &class.nodes[0] {
            ENode::Var(_) => pixelflow_ir::OpKind::Var,
            ENode::Const(_) => pixelflow_ir::OpKind::Const,
            ENode::Buffer(_) => pixelflow_ir::OpKind::Buffer,
            ENode::Op { op, .. } => op.kind(),
        }
    }

    /// Debug: dump the entire e-graph structure.
    #[allow(dead_code)]
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (idx, class) in self.classes.iter().enumerate() {
            let id = EClassId(idx as u32);
            let canonical = self.find(id);
            if canonical == id && !class.nodes.is_empty() {
                writeln!(&mut out, "e{}: {:?}", idx, class.nodes).unwrap();
            }
        }
        out
    }

    /// Add an arena-based DAG expression to the e-graph, preserving sharing.
    ///
    /// Each `ExprId` in the arena maps to exactly one `EClassId`. Because the
    /// arena is topologically ordered (children always precede parents), a single
    /// linear scan suffices — no recursion, no stack overflow.
    ///
    /// Returns the `EClassId` of the root node.
    ///
    /// # Panics
    ///
    /// Panics if `root` is not a valid index in `arena`, or if any child
    /// `ExprId` referenced by an interior node has not been processed yet
    /// (which would indicate a malformed arena that violates topological order).
    /// Also panics if an `ExprNode::Param` node is encountered, since Param
    /// nodes are not valid after kernel compilation.
    pub fn add_arena(
        &mut self,
        arena: &pixelflow_ir::ExprArena,
        root: pixelflow_ir::ExprId,
    ) -> EClassId {
        use pixelflow_ir::arena::ExprNode;

        let n = arena.len();
        assert!(
            (root.0 as usize) < n,
            "add_arena: root {:?} out of bounds (arena has {} nodes)",
            root,
            n,
        );

        // Map from arena ExprId index → EClassId. None = not yet processed.
        let mut id_map: Vec<Option<EClassId>> = vec![None; n];

        for idx in 0..n {
            let eid = pixelflow_ir::ExprId(idx as u32);
            let eclass = match arena.node(eid) {
                ExprNode::Var(v) => self.add(ENode::Var(*v)),
                ExprNode::Const(v) => self.add(ENode::constant(*v)),
                ExprNode::Param(i) => {
                    panic!("add_arena: ExprNode::Param({i}) not valid after kernel compilation")
                }
                // The leaf carries the full decl (identity + extents), so
                // hash-consing merges buffer references across arenas iff they
                // name the same memory — see `ENode::Buffer`.
                ExprNode::Buffer(b) => self.add(ENode::Buffer(*arena.buffer_decl(*b))),
                ExprNode::Unary(op, child) => {
                    let child_id = id_map[child.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Unary node at idx={idx} references child {:?} which has not been processed (arena not topologically ordered)",
                            child
                        )
                    });
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: vec![child_id],
                    })
                }
                ExprNode::Binary(op, left, right) => {
                    let left_id = id_map[left.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Binary node at idx={idx} references left child {:?} which has not been processed",
                            left
                        )
                    });
                    let right_id = id_map[right.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Binary node at idx={idx} references right child {:?} which has not been processed",
                            right
                        )
                    });
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: vec![left_id, right_id],
                    })
                }
                ExprNode::Ternary(op, a, b, c) => {
                    let a_id = id_map[a.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Ternary node at idx={idx} references child a={:?} which has not been processed",
                            a
                        )
                    });
                    let b_id = id_map[b.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Ternary node at idx={idx} references child b={:?} which has not been processed",
                            b
                        )
                    });
                    let c_id = id_map[c.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Ternary node at idx={idx} references child c={:?} which has not been processed",
                            c
                        )
                    });
                    // Gather is deliberately absent from `op_from_kind` (no
                    // rewrite rule may name it) but representable as opaque
                    // structure — resolve it directly.
                    let static_op: &'static dyn crate::egraph::ops::Op = match *op {
                        pixelflow_ir::OpKind::Gather => &ops::Gather,
                        other => ops::op_from_kind(other).unwrap_or_else(|| {
                            panic!("add_arena: no static Op for OpKind {other:?}")
                        }),
                    };
                    self.add(ENode::Op {
                        op: static_op,
                        children: vec![a_id, b_id, c_id],
                    })
                }
                ExprNode::Nary(op, start, len) => {
                    let children_slice = arena.nary_children_slice(*start, *len);
                    let child_ids: Vec<EClassId> = children_slice
                        .iter()
                        .enumerate()
                        .map(|(ci, c)| {
                            id_map[c.0 as usize].unwrap_or_else(|| {
                                panic!(
                                    "add_arena: Nary node at idx={idx} references child[{ci}]={:?} which has not been processed",
                                    c
                                )
                            })
                        })
                        .collect();
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: child_ids,
                    })
                }
            };
            id_map[idx] = Some(eclass);
        }

        id_map[root.0 as usize].expect("add_arena: root EClassId missing after full traversal")
    }

    /// Get a rule by index.
    pub fn rule(&self, idx: usize) -> Option<&dyn Rewrite> {
        self.rules.get(idx).map(|r| r.as_ref())
    }

    /// Find all actual rewrite matches in the E-graph.
    ///
    /// Returns only targets where the rule actually matches (produces an action).
    /// Much more efficient than enumerating all combinations - only scores real matches.
    pub fn find_rewrite_matches(&self) -> Vec<RewriteTarget> {
        let mut matches = Vec::new();

        for (rule_idx, rule) in self.rules.iter().enumerate() {
            for class_id in self.class_ids() {
                let nodes = &self.classes[class_id.index()].nodes;

                for (node_idx, node) in nodes.iter().enumerate() {
                    // Check if rule matches this node
                    if rule.apply(self, class_id, node).is_some() {
                        matches.push(RewriteTarget {
                            rule_idx,
                            class_id,
                            node_idx,
                        });
                    }
                }
            }
        }

        matches
    }

    /// Apply a single rule to a specific (class, node) pair.
    ///
    /// Returns true if the rule matched and produced a change.
    /// This is used by guided search to apply rules one at a time.
    ///
    /// Unlike the sweeps, this takes no class budget: one call applies at
    /// most one action, so it cannot itself grow the graph without bound —
    /// the loop that calls it can, and that loop owns the budget. Reaching
    /// [`HARD_CLASS_LIMIT`] here therefore means a caller is driving
    /// rewrites with no budget at all, and it **panics** rather than
    /// returning `false`: `false` already means "the rule did not fire",
    /// and a budget exhaustion that is indistinguishable from a non-match
    /// is the silent failure this ceiling exists to prevent. No production
    /// path calls this, and the production caps are two orders of
    /// magnitude below the limit.
    pub fn apply_single_rule(
        &mut self,
        rule_idx: usize,
        class_id: EClassId,
        node_idx: usize,
    ) -> bool {
        assert!(
            self.classes.len() < HARD_CLASS_LIMIT,
            "apply_single_rule: e-graph is at the hard class limit \
             ({} classes, limit {HARD_CLASS_LIMIT}) — this entry point applies \
             one action per call and carries no budget, so the caller's \
             rewrite loop must impose one (see `saturate_with_limits`)",
            self.classes.len(),
        );
        let Some(rule) = self.rules.get(rule_idx) else {
            return false;
        };

        let class_id = self.find(class_id);
        let nodes = self.classes[class_id.index()].nodes.clone();
        let Some(node) = nodes.get(node_idx) else {
            return false;
        };

        let Some(action) = rule.apply(self, class_id, node) else {
            return false;
        };

        // Guided search calls this once per discrete rewrite decision — the
        // step counter's granularity here is "one apply_single_rule call",
        // mirroring "one saturate_with_limits iteration" for the batched path.
        self.step += 1;

        // The batched path already knows how to execute every action; reuse
        // it so the single-step path can't drift out of sync.
        let changed = self.apply_action_from_rule(rule_idx, class_id, action) > 0;

        if changed {
            self.rebuild();
        }
        changed
    }

    pub fn contains_const(&self, id: EClassId, val: f32) -> bool {
        self.nodes(id).iter().any(|n| n.is_const(val))
    }

    /// The one rewrite-until-budget-exhausted loop.
    ///
    /// Every saturation in the workspace bottoms out here, and nothing else
    /// re-decides when a run stops. Callers reach it through one of two
    /// entry points:
    ///
    /// - [`Optimizer::run`](super::optimizer::Optimizer::run) — the
    ///   production path, through [`Self::saturate_budgeted`]. It picks its
    ///   limits from a [`Budget`](super::optimizer::Budget), so the budget
    ///   scales with the input and carries no clock. All three production
    ///   tiers use it.
    /// - This method, called directly only where a budget must be pinned
    ///   rather than inherited — tests, the hindsight labeler, and
    ///   measurement harnesses, which spell it
    ///   [`SaturationConfig::compatibility`](super::saturate::SaturationConfig::compatibility).
    ///
    /// The three arguments are independent stopping conditions, checked
    /// between rewrite rounds; the run returns as soon as any one of them —
    /// or saturation itself — is reached, whichever comes first:
    ///
    /// - `max_iters` — how many rewrite rounds may run. One round applies
    ///   every rule once and rebuilds once.
    /// - `max_classes` — e-class ceiling. Bounds memory against e-graph
    ///   blowup, and is the only limit also checked *between rules within* a
    ///   round.
    /// - `timeout` — wall-clock ceiling measured from entry. It is the only
    ///   limit pushed down into a single rule's matching, as a deadline, so
    ///   it is what bounds a round that would otherwise run long.
    ///
    /// Returns the [`SaturationStats`] the run actually achieved;
    /// `stats.iterations < max_iters` is how a caller tells convergence from
    /// an exhausted round budget.
    pub fn saturate_with_limits(
        &mut self,
        max_iters: usize,
        max_classes: usize,
        timeout: std::time::Duration,
    ) -> SaturationStats {
        self.saturate_bounded(max_iters, max_classes, None, Some(timeout))
    }

    /// Saturate under deterministic limits only — rounds, classes, and
    /// optionally rule applications — with no clock.
    ///
    /// This is what [`super::optimizer::Optimizer`] drives, and the
    /// difference from [`Self::saturate_with_limits`] is the whole point:
    /// every limit here is a property of the input and the configuration, so
    /// the same term under the same limits produces the same graph on any
    /// machine at any load. `saturate_with_limits`'s `timeout` is not — it
    /// stops the run at a wall-clock boundary and the result then depends on
    /// what else the machine was doing. A wall-clock ceiling is still
    /// available, on the optimizer, where exceeding it panics instead of
    /// quietly changing the answer.
    ///
    /// `SaturationStop::Timeout` is unreachable from here.
    pub fn saturate_budgeted(
        &mut self,
        max_iters: usize,
        max_classes: usize,
        max_applications: Option<u64>,
    ) -> SaturationStats {
        self.saturate_bounded(max_iters, max_classes, max_applications, None)
    }

    /// The one rewrite-until-budget-exhausted loop. Every saturation entry
    /// point in this crate funnels here rather than re-deciding, in a second
    /// copy, when to stop.
    fn saturate_bounded(
        &mut self,
        max_iters: usize,
        max_classes: usize,
        max_applications: Option<u64>,
        timeout: Option<std::time::Duration>,
    ) -> SaturationStats {
        // Growth is decided here and in `apply_rule_at_index_timed`, so the
        // ceiling is applied here: a caller asking for `usize::MAX` classes
        // gets `HARD_CLASS_LIMIT` and a truthful `ClassCap` stop, not a
        // graph that grows until `add` starts refusing to insert. Applied in
        // the shared loop rather than in `saturate_with_limits`, so
        // `saturate_budgeted` — and therefore every production tier — is
        // held to it too.
        let max_classes = max_classes.min(HARD_CLASS_LIMIT);
        let start = std::time::Instant::now();
        // `Instant + Duration::MAX` panics, so the deadline is optional
        // rather than "infinitely far away".
        let deadline = timeout.map(|t| start + t);
        // The cap is enforced deep inside the scan, where an application is
        // about to commit — the only place that can stop mid-round without
        // letting the round decide how far past the budget to go.
        let previous_cap = self.application_cap;
        self.application_cap = max_applications.map(|n| self.applications.saturating_add(n));
        let mut iterations = 0;
        let mut total_unions = 0;
        // Recorded at the point the loop decides to stop — never inferred
        // afterwards from the counters.
        let mut stop = SaturationStop::IterationCeiling;

        for _ in 0..max_iters {
            if let Some(t) = timeout {
                if start.elapsed() >= t {
                    stop = SaturationStop::Timeout;
                    break;
                }
            }
            if self.classes.len() > max_classes {
                stop = SaturationStop::ClassCap;
                break;
            }
            if self
                .application_cap
                .is_some_and(|cap| self.applications >= cap)
            {
                stop = SaturationStop::ApplicationBudget;
                break;
            }
            iterations += 1;

            // Advance the provenance step counter once per saturation
            // iteration — every ApplicationRecord/UnionEvent produced by
            // this iteration's rule applications shares this step.
            self.step += 1;

            // Apply all rules in a single batch — one rebuild per iteration
            let (unions, sweep) = {
                let mut batch = self.batch();
                let n_rules = batch.graph.rules.len();
                let mut total = 0;
                let mut sweep = ScanStop::Completed;
                for rule_idx in 0..n_rules {
                    if batch.node_count() > max_classes {
                        sweep = ScanStop::ClassCap;
                        break;
                    }
                    let result = batch.apply_rule(rule_idx, max_classes, deadline);
                    total += result.changes;
                    match result.scan {
                        ScanStop::Completed => {}
                        // The class budget cut this rule short. A later rule
                        // may still fit inside the cap, so the sweep goes on
                        // — but it is no longer a full sweep, and can never
                        // read as quiescence.
                        ScanStop::ClassCap => sweep = ScanStop::ClassCap,
                        // The wall-clock ceiling is hard: do not start
                        // another rule's scan on the far side of it.
                        ScanStop::Deadline => {
                            sweep = ScanStop::Deadline;
                            break;
                        }
                        // The application budget is spent for the whole run,
                        // not just this rule: no later rule can commit
                        // anything either.
                        ScanStop::ApplicationBudget => {
                            sweep = ScanStop::ApplicationBudget;
                            break;
                        }
                    }
                }
                (total, sweep)
                // rebuild happens here on drop
            };
            total_unions += unions;
            // The union count and the stop reason are independent facts: a
            // sweep that a budget cut short classifies as that budget whether
            // or not it also committed unions. Consulting `truncated` only on
            // the `unions == 0` path missed every truncated-but-productive
            // sweep, and if that was the last allowed iteration the run fell
            // through to this loop's default `IterationCeiling`.
            //
            // `apply_rule` holds `classes.len()` at or under `max_classes` by
            // truncating its own scan, so the cheap check at the top of this
            // loop almost never sees a capped run — the sweep's own report is
            // what makes `ClassCap` observable at all.
            match sweep {
                ScanStop::ClassCap => {
                    stop = SaturationStop::ClassCap;
                    break;
                }
                ScanStop::Deadline => {
                    stop = SaturationStop::Timeout;
                    break;
                }
                ScanStop::ApplicationBudget => {
                    stop = SaturationStop::ApplicationBudget;
                    break;
                }
                // A full sweep that changed nothing is the one diagnostic
                // fixed point this loop can report.
                ScanStop::Completed => {
                    if unions == 0 {
                        stop = SaturationStop::Quiesced;
                        break;
                    }
                }
            }
        }

        self.application_cap = previous_cap;

        SaturationStats {
            iterations,
            total_unions,
            stop,
        }
    }

    /// Apply all rewrite rules once with a node budget.
    ///
    /// Returns the number of changes made. Stops if the graph exceeds
    /// `max_nodes` classes.
    pub fn apply_rules_once(&mut self, max_nodes: usize) -> usize {
        self.apply_rules_budgeted(max_nodes)
    }

    /// Apply a single rule (by index) everywhere it matches, with budget.
    ///
    /// Returns changes made and evaluations consumed. Stops scanning
    /// if the graph exceeds `max_nodes` classes.
    pub fn apply_rule_at_index(&mut self, rule_idx: usize, max_nodes: usize) -> ApplyResult {
        self.apply_rule_at_index_budgeted(rule_idx, max_nodes)
    }

    /// Apply a single rule with a node budget. Stops scanning when the
    /// e-graph exceeds `max_nodes` classes, preventing runaway growth
    /// from a single rule application.
    pub fn apply_rule_at_index_budgeted(
        &mut self,
        rule_idx: usize,
        max_nodes: usize,
    ) -> ApplyResult {
        self.apply_rule_at_index_timed(rule_idx, max_nodes, None)
    }

    /// Apply a single rule with node budget AND optional wall-clock deadline.
    /// Stops if either budget or deadline is exceeded.
    pub fn apply_rule_at_index_timed(
        &mut self,
        rule_idx: usize,
        max_nodes: usize,
        deadline: Option<std::time::Instant>,
    ) -> ApplyResult {
        // See `HARD_CLASS_LIMIT`: the scan below is one of the two places
        // the graph decides to grow, so it is one of the two places the
        // ceiling is applied.
        let max_nodes = max_nodes.min(HARD_CLASS_LIMIT);
        if rule_idx >= self.rules.len() {
            return ApplyResult {
                changes: 0,
                evals: 0,
                scan: ScanStop::Completed,
            };
        }

        // `Instant::now()` costs tens of nanoseconds — the same order as the
        // cheapest rule's `apply` — so polling it once per node would be a
        // measurable tax on every scan. It is polled instead at the three
        // places that make the ceiling hard without that tax: once per
        // e-class (the boundary already clones the class's node vector, so a
        // clock read is lost in the noise), every `DEADLINE_POLL_NODES` nodes
        // within a class (so one enormous e-class cannot run unbounded), and
        // once after the scan and its commit (so a deadline that elapsed
        // inside an opaque `apply` is still seen). The bound that buys is an
        // overrun of at most `DEADLINE_POLL_NODES` rule evaluations plus one
        // `apply`; the timeout is a fail-loud safety ceiling, not a
        // scheduling knob, so shrinking that bound further buys nothing worth
        // a clock read per node.
        const DEADLINE_POLL_NODES: usize = 256;
        let expired = |deadline: Option<std::time::Instant>| match deadline {
            Some(dl) => std::time::Instant::now() > dl,
            None => false,
        };

        let mut unions = 0;
        let mut evals = 0usize;
        let mut updates: Vec<(EClassId, RewriteAction)> = Vec::new();
        let mut estimated_new_nodes: usize = 0;
        let mut scan = ScanStop::Completed;

        // An application budget is spent by *applications*, so the scan stops
        // once it has queued its whole allowance. Checked here and not only
        // between rounds: a round is 62 rules over every e-class, so a
        // between-rounds check alone overshoots a small budget by orders of
        // magnitude — the budget has to bound the work, not describe it
        // afterwards.
        let allowance_spent = |graph: &Self, queued: usize| {
            graph
                .application_cap
                .is_some_and(|cap| graph.applications + queued as u64 >= cap)
        };

        let canonical_ids = self.canonical_class_ids();
        'scan: for canonical in canonical_ids {
            // Budget check: current graph + pending creates must stay under limit
            if self.classes.len() + estimated_new_nodes > max_nodes {
                scan = ScanStop::ClassCap;
                break;
            }
            if expired(deadline) {
                scan = ScanStop::Deadline;
                break;
            }
            if allowance_spent(self, updates.len()) {
                scan = ScanStop::ApplicationBudget;
                break;
            }

            let nodes: Vec<ENode> = self.classes[canonical.index()].nodes.clone();

            for node in &nodes {
                evals += 1;
                if evals % DEADLINE_POLL_NODES == 0 && expired(deadline) {
                    scan = ScanStop::Deadline;
                    break 'scan;
                }
                if let Some(action) = self.rules[rule_idx].apply(self, canonical, node) {
                    // Track how many nodes this action would create
                    let action_cost = match &action {
                        RewriteAction::Union(_) => 0,
                        RewriteAction::Create(_) => 1,
                        // Multi-node actions: conservative upper bound
                        _ => 3,
                    };
                    estimated_new_nodes += action_cost;

                    // If this action would push us over budget, stop scanning
                    if self.classes.len() + estimated_new_nodes > max_nodes {
                        // Don't add this action — discard it and stop
                        scan = ScanStop::ClassCap;
                        break 'scan;
                    }

                    updates.push((canonical, action));
                    *self
                        .match_counts
                        .entry(RuleId::of(self.rules[rule_idx].as_ref()))
                        .or_insert(0) += 1;
                    if allowance_spent(self, updates.len()) {
                        scan = ScanStop::ApplicationBudget;
                        break 'scan;
                    }
                }
            }
        }

        // Commit: all actions in the log are within budget.
        // Do NOT rebuild here — caller is responsible for calling rebuild()
        // after all rules for the epoch are applied (lazy/batched rebuild).
        for (class_id, action) in updates {
            if allowance_spent(self, 0) {
                scan = ScanStop::ApplicationBudget;
                break;
            }
            unions += self.apply_action_from_rule(rule_idx, class_id, action);
        }

        // A deadline that elapsed inside a rule's `apply` or inside the commit
        // above reached none of the checks in the walk. Left as `Completed`,
        // a sweep that blew the hard ceiling would be recorded as quiescence.
        if scan == ScanStop::Completed && expired(deadline) {
            scan = ScanStop::Deadline;
        }

        ApplyResult {
            changes: unions,
            evals,
            scan,
        }
    }

    /// Apply a rewrite action on behalf of a specific rule, attributing
    /// every e-node created and every union performed while executing it to
    /// one [`super::provenance::ApplicationId`].
    ///
    /// This is the sole entry point into `apply_action` from rule-driven
    /// call sites (`apply_single_rule`, `apply_rule_at_index_timed`,
    /// `apply_rules_budgeted`) — it exists so provenance attribution can't
    /// drift out of sync with the actual rewrite dispatch: every caller that
    /// knows a `rule_idx` funnels through here instead of calling
    /// `apply_action` directly.
    ///
    /// Records one [`ApplicationRecord`] up front (even if the action turns
    /// out to produce no net change, e.g. a `Union` with an already-equal
    /// target) — the record's cost is a single `Vec::push`, and recording
    /// unconditionally keeps `step` bookkeeping simple. `match_root` is the
    /// class the rule matched against, i.e. `class_id` as passed in (cheap:
    /// already in hand at the call site).
    fn apply_action_from_rule(
        &mut self,
        rule_idx: usize,
        class_id: EClassId,
        action: RewriteAction,
    ) -> usize {
        // Counted unconditionally: the budget must not depend on whether
        // anyone is watching.
        self.applications += 1;

        if !self.record_provenance {
            return self.apply_action(class_id, action);
        }

        let minted_from = self.next_enode_id;
        let application_id = self.provenance.record_application(ApplicationRecord {
            rule: self.rule_ids.get(rule_idx).copied(),
            rule_idx,
            step: self.step,
            match_root: class_id,
            minted: minted_from..minted_from,
            unions: 0,
        });
        let previous = self.active_application.replace(ActiveApplication {
            rule_idx,
            application_id,
        });
        let result = self.apply_action(class_id, action);
        self.active_application = previous;
        // The record is opened before the action runs (so `add`/`union` can
        // attribute to it) and closed after, which is the only order in
        // which "what did this application mint" is answerable.
        self.provenance.complete_application(
            application_id,
            minted_from..self.next_enode_id,
            result,
        );
        result
    }

    /// Apply a rewrite action and return 1 if a union was made, 0 otherwise.
    ///
    /// Internal executor. Rule-driven callers must go through
    /// `apply_action_from_rule` so provenance attribution stays correct;
    /// this function itself has no notion of "which rule" — it only knows
    /// how to execute the `RewriteAction` variants.
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

    fn apply_action(&mut self, class_id: EClassId, action: RewriteAction) -> usize {
        match action {
            RewriteAction::Union(target_id) => self.union_counted(class_id, target_id),
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
                let ab_node = ENode::Op {
                    op: outer,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab_node);
                let ac_node = ENode::Op {
                    op: outer,
                    children: vec![a, c],
                };
                let ac_id = self.add(ac_node);
                let result_node = ENode::Op {
                    op: inner,
                    children: vec![ab_id, ac_id],
                };
                let result_id = self.add(result_node);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => {
                let sum_node = ENode::Op {
                    op: outer,
                    children: vec![unique_l, unique_r],
                };
                let sum_id = self.add(sum_node);
                let result_node = ENode::Op {
                    op: inner,
                    children: vec![common, sum_id],
                };
                let result_id = self.add(result_node);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => {
                let inv_node = ENode::Op {
                    op: inverse,
                    children: vec![b],
                };
                let inv_id = self.add(inv_node);
                let target_node = ENode::Op {
                    op: target,
                    children: vec![a, inv_id],
                };
                let target_id = self.add(target_node);
                self.union_counted(class_id, target_id)
            }
            RewriteAction::Associate { op, a, b, c } => {
                let bc_node = ENode::Op {
                    op,
                    children: vec![b, c],
                };
                let bc_id = self.add(bc_node);
                let result_node = ENode::Op {
                    op,
                    children: vec![a, bc_id],
                };
                let result_id = self.add(result_node);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ReverseAssociate { op, a, b, c } => {
                // a op (b op c) → (a op b) op c
                let ab_node = ENode::Op {
                    op,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab_node);
                let result_node = ENode::Op {
                    op,
                    children: vec![ab_id, c],
                };
                let result_id = self.add(result_node);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::OddParity { func, inner } => {
                // For odd functions: Op(neg(x)) → neg(Op(x))
                // Create func(inner), then wrap in neg
                let func_node = ENode::Op {
                    op: func,
                    children: vec![inner],
                };
                let func_id = self.add(func_node);
                let neg_node = ENode::Op {
                    op: &ops::Neg,
                    children: vec![func_id],
                };
                let neg_id = self.add(neg_node);
                self.union_counted(class_id, neg_id)
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
                // sin(a+b) → sin(a)cos(b) + cos(a)sin(b)
                // cos(a+b) → cos(a)cos(b) - sin(a)sin(b)
                //
                // Create: term1_op1(a)*term1_op2(b) +/- term2_op1(a)*term2_op2(b)

                // term1_op1(a)
                let t1_left = ENode::Op {
                    op: term1_op1,
                    children: vec![a],
                };
                let t1_left_id = self.add(t1_left);

                // term1_op2(b)
                let t1_right = ENode::Op {
                    op: term1_op2,
                    children: vec![b],
                };
                let t1_right_id = self.add(t1_right);

                // term1_op1(a) * term1_op2(b)
                let term1 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![t1_left_id, t1_right_id],
                };
                let term1_id = self.add(term1);

                // term2_op1(a)
                let t2_left = ENode::Op {
                    op: term2_op1,
                    children: vec![a],
                };
                let t2_left_id = self.add(t2_left);

                // term2_op2(b)
                let t2_right = ENode::Op {
                    op: term2_op2,
                    children: vec![b],
                };
                let t2_right_id = self.add(t2_right);

                // term2_op1(a) * term2_op2(b)
                let term2 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![t2_left_id, t2_right_id],
                };
                let term2_id = self.add(term2);

                // Combine based on sign
                use crate::math::trig::Sign;
                let result_id = match term2_sign {
                    Sign::Plus => {
                        let result = ENode::Op {
                            op: &ops::Add,
                            children: vec![term1_id, term2_id],
                        };
                        self.add(result)
                    }
                    Sign::Minus => {
                        let result = ENode::Op {
                            op: &ops::Sub,
                            children: vec![term1_id, term2_id],
                        };
                        self.add(result)
                    }
                };

                self.union_counted(class_id, result_id)
            }
            RewriteAction::Homomorphism {
                func,
                target_op,
                a,
                b,
            } => {
                // f(a ⊕ b) → f(a) ⊗ f(b)
                // e.g., exp(a + b) → exp(a) * exp(b)

                // func(a)
                let func_a = ENode::Op {
                    op: func,
                    children: vec![a],
                };
                let func_a_id = self.add(func_a);

                // func(b)
                let func_b = ENode::Op {
                    op: func,
                    children: vec![b],
                };
                let func_b_id = self.add(func_b);

                // target_op(func(a), func(b))
                let result = ENode::Op {
                    op: target_op,
                    children: vec![func_a_id, func_b_id],
                };
                let result_id = self.add(result);

                self.union_counted(class_id, result_id)
            }
            RewriteAction::PowerCombine { base, exp_a, exp_b } => {
                // x^a * x^b → x^(a+b)

                // a + b
                let sum = ENode::Op {
                    op: &ops::Add,
                    children: vec![exp_a, exp_b],
                };
                let sum_id = self.add(sum);

                // x^(a+b)
                let result = ENode::Op {
                    op: &ops::Pow,
                    children: vec![base, sum_id],
                };
                let result_id = self.add(result);

                self.union_counted(class_id, result_id)
            }
            RewriteAction::ReverseAngleAddition { trig_op, a, b } => {
                // sin(a)cos(b) + cos(a)sin(b) → sin(a + b)
                // (or cos case)

                // a + b
                let sum = ENode::Op {
                    op: &ops::Add,
                    children: vec![a, b],
                };
                let sum_id = self.add(sum);

                // trig(a + b)
                let result = ENode::Op {
                    op: trig_op,
                    children: vec![sum_id],
                };
                let result_id = self.add(result);

                self.union_counted(class_id, result_id)
            }
            RewriteAction::HalfAngleProduct { x } => {
                // sin(x) * cos(x) → sin(x + x) / 2
                // Derived from: sin(2x) = 2*sin(x)*cos(x)

                // x + x
                let two_x = ENode::Op {
                    op: &ops::Add,
                    children: vec![x, x],
                };
                let two_x_id = self.add(two_x);

                // sin(x + x)
                let sin_2x = ENode::Op {
                    op: &ops::Sin,
                    children: vec![two_x_id],
                };
                let sin_2x_id = self.add(sin_2x);

                // constant 2
                let two = ENode::Const(2.0_f32.to_bits());
                let two_id = self.add(two);

                // sin(x + x) / 2
                let result = ENode::Op {
                    op: &ops::Div,
                    children: vec![sin_2x_id, two_id],
                };
                let result_id = self.add(result);

                self.union_counted(class_id, result_id)
            }
            RewriteAction::Doubling { a } => {
                // a + a → 2 * a
                let two = ENode::Const(2.0_f32.to_bits());
                let two_id = self.add(two);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![two_id, a],
                };
                let result_id = self.add(result);

                self.union_counted(class_id, result_id)
            }
            RewriteAction::Halving { a } => {
                // 2 * a → a + a
                let result = ENode::Op {
                    op: &ops::Add,
                    children: vec![a, a],
                };
                let result_id = self.add(result);

                self.union_counted(class_id, result_id)
            }
            RewriteAction::PowerRecurrence { base, exponent } => {
                let n_minus_1 = ENode::constant((exponent - 1) as f32);
                let n_minus_1_id = self.add(n_minus_1);
                let pow_reduced = ENode::Op {
                    op: &ops::Pow,
                    children: vec![base, n_minus_1_id],
                };
                let pow_id = self.add(pow_reduced);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![base, pow_id],
                };
                let result_id = self.add(result);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::LogPower {
                log_op,
                base,
                exponent,
            } => {
                let log_x = ENode::Op {
                    op: log_op,
                    children: vec![base],
                };
                let log_x_id = self.add(log_x);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![exponent, log_x_id],
                };
                let result_id = self.add(result);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ExpandSquare { a, b } => {
                let a2 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![a, a],
                };
                let a2_id = self.add(a2);
                let b2 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![b, b],
                };
                let b2_id = self.add(b2);
                let ab = ENode::Op {
                    op: &ops::Mul,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab);
                let two = ENode::constant(2.0);
                let two_id = self.add(two);
                let two_ab = ENode::Op {
                    op: &ops::Mul,
                    children: vec![two_id, ab_id],
                };
                let two_ab_id = self.add(two_ab);
                let sum1 = ENode::Op {
                    op: &ops::Add,
                    children: vec![a2_id, two_ab_id],
                };
                let sum1_id = self.add(sum1);
                let result = ENode::Op {
                    op: &ops::Add,
                    children: vec![sum1_id, b2_id],
                };
                let result_id = self.add(result);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::DiffOfSquares { a, b } => {
                let sum = ENode::Op {
                    op: &ops::Add,
                    children: vec![a, b],
                };
                let sum_id = self.add(sum);
                let diff = ENode::Op {
                    op: &ops::Sub,
                    children: vec![a, b],
                };
                let diff_id = self.add(diff);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![sum_id, diff_id],
                };
                let result_id = self.add(result);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Differentiate { inner, var } => {
                let deriv_id = self.build_derivative(&inner, var);
                self.union_counted(class_id, deriv_id)
            }
        }
    }

    /// Build the e-class of the derivative of `inner` with respect to variable
    /// `var`, one chain-rule step deep. Sub-expressions are wrapped in fresh
    /// `Dwrt` nodes so equality saturation continues the expansion; the leaf
    /// cases (`Var`, `Const`) terminate it. Operators whose derivative is not
    /// (yet) known reconstruct the original `Dwrt`, leaving it to survive
    /// saturation as the jet fallback.
    fn build_derivative(&mut self, inner: &ENode, var: u8) -> EClassId {
        let (op, children) = match inner {
            ENode::Const(_) => return self.add(ENode::constant(0.0)),
            ENode::Var(i) => {
                return self.add(ENode::constant(if *i == var { 1.0 } else { 0.0 }));
            }
            // A Buffer leaf is not a value — it only ever appears as Gather's
            // first child, and no rewrite builds `Dwrt(buffer)`. Reaching one
            // here means the graph is malformed; fail loudly. (`Dwrt(gather)`
            // is the Op arm below: Gather has no derivative table entry, so it
            // reconstructs the Dwrt as the jet fallback.)
            ENode::Buffer(decl) => {
                panic!("build_derivative: Dwrt applied to a Buffer leaf ({decl:?})")
            }
            ENode::Op { op, children } => (*op, children.clone()),
        };

        let var_const = self.add(ENode::constant(var as f32));
        // d(child)/dvar as a fresh Dwrt node (saturation expands it later).
        let dwrt = |s: &mut Self, c: EClassId| {
            s.add(ENode::Op {
                op: &ops::Dwrt,
                children: vec![c, var_const],
            })
        };
        let op2 = |s: &mut Self, o: &'static dyn Op, a: EClassId, b: EClassId| {
            s.add(ENode::Op {
                op: o,
                children: vec![a, b],
            })
        };
        let un = |s: &mut Self, o: &'static dyn Op, a: EClassId| {
            s.add(ENode::Op {
                op: o,
                children: vec![a],
            })
        };
        let cst = |s: &mut Self, v: f32| s.add(ENode::constant(v));

        match op.kind() {
            // Linearity: D(a + b) = D(a) + D(b); D(a - b) = D(a) - D(b).
            OpKind::Add | OpKind::Sub => {
                let da = dwrt(self, children[0]);
                let db = dwrt(self, children[1]);
                let same = ops::op_from_kind(op.kind()).expect("add/sub op");
                op2(self, same, da, db)
            }
            OpKind::Neg => {
                let da = dwrt(self, children[0]);
                un(self, &ops::Neg, da)
            }
            // Product rule: D(a*b) = D(a)*b + a*D(b).
            OpKind::Mul => {
                let (a, b) = (children[0], children[1]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let t1 = op2(self, &ops::Mul, da, b);
                let t2 = op2(self, &ops::Mul, a, db);
                op2(self, &ops::Add, t1, t2)
            }
            // Fused multiply-add a*b + c: D = D(a)*b + a*D(b) + D(c).
            OpKind::MulAdd => {
                let (a, b, c) = (children[0], children[1], children[2]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let dc = dwrt(self, c);
                let t1 = op2(self, &ops::Mul, da, b);
                let t2 = op2(self, &ops::Mul, a, db);
                let prod = op2(self, &ops::Add, t1, t2);
                op2(self, &ops::Add, prod, dc)
            }
            // Quotient rule: D(a/b) = (D(a)*b - a*D(b)) / (b*b).
            OpKind::Div => {
                let (a, b) = (children[0], children[1]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let t1 = op2(self, &ops::Mul, da, b);
                let t2 = op2(self, &ops::Mul, a, db);
                let num = op2(self, &ops::Sub, t1, t2);
                let den = op2(self, &ops::Mul, b, b);
                op2(self, &ops::Div, num, den)
            }
            // d(sqrt u) = 0.5 * rsqrt(u) * u'.
            OpKind::Sqrt => {
                let u = children[0];
                let du = dwrt(self, u);
                let half = cst(self, 0.5);
                let rs = un(self, &ops::Rsqrt, u);
                let factor = op2(self, &ops::Mul, half, rs);
                op2(self, &ops::Mul, factor, du)
            }
            // d(recip u) = -u' / (u*u).
            OpKind::Recip => {
                let u = children[0];
                let du = dwrt(self, u);
                let ndu = un(self, &ops::Neg, du);
                let u2 = op2(self, &ops::Mul, u, u);
                op2(self, &ops::Div, ndu, u2)
            }
            // d(|u|) = (u / |u|) * u'.
            OpKind::Abs => {
                let u = children[0];
                let du = dwrt(self, u);
                let au = un(self, &ops::Abs, u);
                let sign = op2(self, &ops::Div, u, au);
                op2(self, &ops::Mul, sign, du)
            }
            // d(rsqrt u) = -0.5 * rsqrt(u) * recip(u) * u'.
            OpKind::Rsqrt => {
                let u = children[0];
                let du = dwrt(self, u);
                let neg_half = cst(self, -0.5);
                let rs = un(self, &ops::Rsqrt, u);
                let rc = un(self, &ops::Recip, u);
                let t = op2(self, &ops::Mul, rs, rc);
                let factor = op2(self, &ops::Mul, neg_half, t);
                op2(self, &ops::Mul, factor, du)
            }
            // Piecewise: derivative of the branch the primal takes. Masks and
            // tie behavior mirror Jet2 (and the runtime `lower_dwrt` pass).
            OpKind::Min => {
                let (a, b) = (children[0], children[1]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let mask = op2(self, &ops::Lt, a, b);
                self.add(ENode::Op {
                    op: &ops::Select,
                    children: vec![mask, da, db],
                })
            }
            OpKind::Max => {
                let (a, b) = (children[0], children[1]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let mask = op2(self, &ops::Gt, a, b);
                self.add(ENode::Op {
                    op: &ops::Select,
                    children: vec![mask, da, db],
                })
            }
            // Blend the branch derivatives on the primal mask; the mask itself
            // is not differentiated.
            OpKind::Select => {
                let (m, t, f) = (children[0], children[1], children[2]);
                let dt = dwrt(self, t);
                let df = dwrt(self, f);
                self.add(ENode::Op {
                    op: &ops::Select,
                    children: vec![m, dt, df],
                })
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
            | OpKind::Round => self.add(ENode::constant(0.0)),
            // d(sin u) = cos(u) * u'.
            OpKind::Sin => {
                let u = children[0];
                let du = dwrt(self, u);
                let c = un(self, &ops::Cos, u);
                op2(self, &ops::Mul, c, du)
            }
            // d(cos u) = -sin(u) * u'.
            OpKind::Cos => {
                let u = children[0];
                let du = dwrt(self, u);
                let s = un(self, &ops::Sin, u);
                let ns = un(self, &ops::Neg, s);
                op2(self, &ops::Mul, ns, du)
            }
            // d(tan u) = u' / cos(u)^2.
            OpKind::Tan => {
                let u = children[0];
                let du = dwrt(self, u);
                let c = un(self, &ops::Cos, u);
                let c2 = op2(self, &ops::Mul, c, c);
                op2(self, &ops::Div, du, c2)
            }
            // d(atan u) = u' / (1 + u*u).
            OpKind::Atan => {
                let u = children[0];
                let du = dwrt(self, u);
                let one = cst(self, 1.0);
                let u2 = op2(self, &ops::Mul, u, u);
                let den = op2(self, &ops::Add, one, u2);
                op2(self, &ops::Div, du, den)
            }
            // d(asin u) = u' / sqrt(1 - u*u).
            OpKind::Asin => {
                let u = children[0];
                let du = dwrt(self, u);
                let one = cst(self, 1.0);
                let u2 = op2(self, &ops::Mul, u, u);
                let diff = op2(self, &ops::Sub, one, u2);
                let s = un(self, &ops::Sqrt, diff);
                op2(self, &ops::Div, du, s)
            }
            // d(acos u) = -u' / sqrt(1 - u*u).
            OpKind::Acos => {
                let u = children[0];
                let du = dwrt(self, u);
                let one = cst(self, 1.0);
                let u2 = op2(self, &ops::Mul, u, u);
                let diff = op2(self, &ops::Sub, one, u2);
                let s = un(self, &ops::Sqrt, diff);
                let q = op2(self, &ops::Div, du, s);
                un(self, &ops::Neg, q)
            }
            // d(exp u) = exp(u) * u'.
            OpKind::Exp => {
                let u = children[0];
                let du = dwrt(self, u);
                let e = un(self, &ops::Exp, u);
                op2(self, &ops::Mul, e, du)
            }
            // d(2^u) = 2^u * ln2 * u'.
            OpKind::Exp2 => {
                let u = children[0];
                let du = dwrt(self, u);
                let e = un(self, &ops::Exp2, u);
                let ln2 = cst(self, core::f32::consts::LN_2);
                let factor = op2(self, &ops::Mul, e, ln2);
                op2(self, &ops::Mul, factor, du)
            }
            // d(ln u) = u' / u.
            OpKind::Ln => {
                let u = children[0];
                let du = dwrt(self, u);
                op2(self, &ops::Div, du, u)
            }
            // d(log2 u) = u' / (u * ln2).
            OpKind::Log2 => {
                let u = children[0];
                let du = dwrt(self, u);
                let ln2 = cst(self, core::f32::consts::LN_2);
                let den = op2(self, &ops::Mul, u, ln2);
                op2(self, &ops::Div, du, den)
            }
            // d(log10 u) = u' / (u * ln10).
            OpKind::Log10 => {
                let u = children[0];
                let du = dwrt(self, u);
                let ln10 = cst(self, core::f32::consts::LN_10);
                let den = op2(self, &ops::Mul, u, ln10);
                op2(self, &ops::Div, du, den)
            }
            // d(atan2(y, x)) = (x*y' - y*x') / (x² + y²).
            OpKind::Atan2 => {
                let (y, x) = (children[0], children[1]);
                let dy = dwrt(self, y);
                let dx = dwrt(self, x);
                let t1 = op2(self, &ops::Mul, x, dy);
                let t2 = op2(self, &ops::Mul, y, dx);
                let num = op2(self, &ops::Sub, t1, t2);
                let x2 = op2(self, &ops::Mul, x, x);
                let y2 = op2(self, &ops::Mul, y, y);
                let den = op2(self, &ops::Add, x2, y2);
                op2(self, &ops::Div, num, den)
            }
            // d(f^g) = f^g * (g'*ln f + g*f'/f)  (Jet2's rule).
            OpKind::Pow => {
                let (f, g) = (children[0], children[1]);
                let df = dwrt(self, f);
                let dg = dwrt(self, g);
                let lnf = un(self, &ops::Ln, f);
                let t1 = op2(self, &ops::Mul, dg, lnf);
                let g_over_f = op2(self, &ops::Div, g, f);
                let t2 = op2(self, &ops::Mul, g_over_f, df);
                let inner = op2(self, &ops::Add, t1, t2);
                let p = op2(self, &ops::Pow, f, g);
                op2(self, &ops::Mul, p, inner)
            }
            // Unknown derivative: reconstruct the Dwrt and let it survive.
            _ => {
                let reconstructed = self.add(inner.clone());
                dwrt(self, reconstructed)
            }
        }
    }

    fn apply_rules_budgeted(&mut self, max_nodes: usize) -> usize {
        // See `HARD_CLASS_LIMIT`.
        let max_nodes = max_nodes.min(HARD_CLASS_LIMIT);
        // One call = one "apply all rules once" pass, the same granularity
        // saturate_with_limits uses for its step counter.
        self.step += 1;

        let mut unions = 0;
        let mut updates: Vec<(usize, EClassId, RewriteAction)> = Vec::new();

        let canonical_ids = self.canonical_class_ids();
        for canonical in canonical_ids {
            if self.classes.len() > max_nodes {
                break;
            }
            let nodes: Vec<ENode> = self.classes[canonical.index()].nodes.clone();

            for node in &nodes {
                for (rule_idx, rule) in self.rules.iter().enumerate() {
                    if let Some(action) = rule.apply(self, canonical, node) {
                        updates.push((rule_idx, canonical, action));
                        *self
                            .match_counts
                            .entry(RuleId::of(rule.as_ref()))
                            .or_insert(0) += 1;
                    }
                }
            }
        }

        for (rule_idx, class_id, action) in updates {
            unions += self.apply_action_from_rule(rule_idx, class_id, action);
            if self.classes.len() > max_nodes {
                break;
            }
        }

        // Lazy rebuild: caller should call rebuild() after all rules applied.
        // saturate_with_limits handles this.
        unions
    }

    pub fn extract_with_costs(&self, root: EClassId, costs: &CostModel) -> ENode {
        let root = self.find(root);
        let mut cost_table: HashMap<EClassId, (usize, ENode)> = HashMap::new();
        let canonical_ids: Vec<EClassId> = self.class_ids().collect();
        // Fixed-point iteration: at most one pass per canonical class.
        for _ in 0..canonical_ids.len() {
            let mut changed = false;
            for &id in &canonical_ids {
                for node in &self.classes[id.index()].nodes {
                    let cost = self.node_cost_with_model(node, &cost_table, costs);
                    let current = cost_table.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX);
                    if cost < current {
                        cost_table.insert(id, (cost, node.clone()));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // `ENode::Const(0)` used to stand in here — the number 0.0 wearing
        // an extracted term's type, returned for a root the fixpoint never
        // priced. The fixpoint visits every canonical class and every class
        // holds at least one node, so a miss is a bug in this loop, and the
        // caller has no way to tell 0.0-the-answer from 0.0-the-excuse.
        let Some((_, node)) = cost_table.get(&root) else {
            panic!(
                "extract_with_costs: root e-class {} was never priced by the \
                 fixpoint — every canonical class holds at least one node, so \
                 this is a bug in the loop above, not an empty graph",
                root.0
            )
        };
        node.clone()
    }

    fn node_cost_with_model(
        &self,
        node: &ENode,
        cost_table: &HashMap<EClassId, (usize, ENode)>,
        costs: &CostModel,
    ) -> usize {
        let get_child_cost = |id: EClassId| {
            let id = self.find(id);
            cost_table
                .get(&id)
                .map(|(c, _)| *c)
                .unwrap_or(usize::MAX / 4)
        };
        let op_cost = costs.node_op_cost(node);
        let child_cost = node
            .children()
            .iter()
            .fold(0usize, |acc, &c| acc.saturating_add(get_child_cost(c)));
        child_cost.saturating_add(op_cost)
    }

    /// Extract the minimum-cost expression from an e-class.
    pub fn extract_expr_with_costs(
        &self,
        root: EClassId,
        costs: &CostModel,
    ) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
        let (arena, arena_root, _cost) = super::extract::extract(self, root, costs);
        (arena, arena_root)
    }

    /// Extract the best expression and its cost.
    ///
    /// The cost function can be any `CostFunction` implementor —
    /// `CostModel` is the hardcoded latency prior.
    pub fn extract_best<C: CostFunction>(
        &self,
        root: EClassId,
        costs: &C,
    ) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, usize) {
        super::extract::extract(self, root, costs)
    }

    /// Extract a DAG with sharing information from an e-class.
    ///
    /// Unlike `extract_expr_with_costs`, this tracks which e-classes are used
    /// multiple times, enabling codegen to emit let-bindings for shared subexprs.
    ///
    /// # Example
    ///
    /// For `sin(X) * sin(X) + sin(X)`:
    /// - Tree extraction would compute sin(X) three times
    /// - DAG extraction marks sin(X) as shared, enabling: `let __0 = X.sin(); __0 * __0 + __0`
    pub fn extract_dag_with_costs(
        &self,
        root: EClassId,
        costs: &CostModel,
    ) -> super::extract::ExtractedDAG {
        super::extract::extract_dag(self, root, costs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::ops;
    use crate::egraph::provenance::ApplicationId;
    use crate::egraph::saturate::SaturationConfig;
    use crate::math::algebra::{
        AddNeg, Annihilator, Cancellation, Canonicalize, Commutative, Distributive, Identity,
        InverseAnnihilation, Involution, MulRecip,
    };

    /// Regression test for a `rebuild_budgeted` bug: when canonicalizing the
    /// worklist item `id`'s nodes triggers a union whose surviving parent is
    /// `id` itself, `union()`'s `extend()` appends the merged-in class's
    /// nodes directly onto `classes[id.index()].nodes` (which `rebuild_budgeted`
    /// had just emptied via `mem::take`). The old code then did
    /// `classes[id.index()].nodes = new_nodes`, an outright *assignment* that
    /// clobbered whatever `union()` had just appended — silently dropping
    /// nodes. The fix extends instead of assigning.
    ///
    /// Trigger recipe:
    /// - class 3 (`nc = Neg(c)`, c = class 2) is the worklist item being
    ///   rebuilt (`id = 3`).
    /// - Before nc's rebuild, `union(b, c)` merges class 2 into class 1
    ///   (1 < 2), so canonicalizing `Neg(c)` during nc's rebuild turns it
    ///   into `Neg(b)`.
    /// - `Neg(b)` is already memoized as class 4 (`nb = Neg(b)`, created
    ///   after nc so it has a strictly higher id: 4 > 3).
    /// - Because `id (3) < existing (4)`, `union(3, 4)` picks **3** as the
    ///   surviving parent — reproducing the exact "current worklist item
    ///   survives the merge" case that dropped nodes.
    /// - Class 4 (`nb`) must still hold its node at the moment of that
    ///   inner union, which it does (only class 3's `.nodes` was drained).
    #[test]
    fn rebuild_budgeted_does_not_drop_nodes_when_current_class_survives_union() {
        let mut eg = EGraph::new();

        let _a = eg.add(ENode::Var(0)); // class 0 (unused, keeps ids spaced out)
        let b = eg.add(ENode::Var(1)); // class 1
        let c = eg.add(ENode::Var(2)); // class 2
        let nc = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![c],
        }); // class 3: memo Neg([2]) -> 3
        let nb = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![b],
        }); // class 4: memo Neg([1]) -> 4
        let d = eg.add(ENode::Var(3)); // class 5, dummy used only to enqueue `nc`
        let marker = eg.add(ENode::Var(9)); // class 6: a node unique to nb's class

        assert_eq!(nc.index(), 3, "test assumes nc is class 3");
        assert_eq!(nb.index(), 4, "test assumes nb is class 4");

        // Enqueue class 3 (nc) on the worklist without disturbing its node
        // list content: nc (3) < d (5), so nc survives as parent and simply
        // gains an extra Var(3) node — it does not lose anything.
        eg.union(nc, d);
        assert_eq!(eg.pending_rebuilds(), 1);

        // Give nb's class a node with no structural twin anywhere else
        // (`Var(9)`), so its loss is directly observable. nb (4) < marker
        // (6), so nb survives as parent and gains `Var(9)`.
        eg.union(nb, marker);

        // Now merge c into b. b (1) < c (2), so b survives; c's node list
        // (just `Var(2)`) is merged in. This does NOT yet touch nc/nb.
        eg.union(b, c);

        // Rebuild exactly one worklist item. LIFO worklist: the most
        // recently pushed parent (class 1, from union(b,c)) pops first.
        // Process it, then process class 3 (nc) next so its rebuild is the
        // one that triggers the id-survives-merge collision with nb (4).
        //
        // Drain the whole worklist via rebuild() (equivalent to
        // rebuild_budgeted(usize::MAX)) so ordering doesn't need to be
        // hand-tracked — the bug reproduces regardless of pop order, as
        // long as nc's rebuild eventually runs after b/c are merged.
        eg.rebuild();

        // The critical assertion: nb's class (now merged into nc's class,
        // since nc=3 < nb=4 survives) must still contain BOTH nodes that
        // were live before the merge: nc's own `Neg([canonical c])` and
        // nb's `Neg([b])` (same canonical shape, but a distinct ENode
        // instance in the vec before dedup would also be acceptable — the
        // point is the vec must not have been clobbered to something
        // that lost nb's contribution entirely).
        let surviving = eg.find(nc);
        assert_eq!(
            surviving,
            eg.find(nb),
            "nc and nb should have been unioned via the canonicalization collision"
        );

        let nodes = eg.nodes(surviving);
        assert!(
            !nodes.is_empty(),
            "rebuild_budgeted must not silently drop all nodes from the surviving class"
        );

        // The dummy Var(3) node pushed onto nc's class via `union(nc, d)`
        // earlier must have survived.
        assert!(
            nodes.iter().any(|n| matches!(n, ENode::Var(3))),
            "expected Var(3) (pushed before the id-survives merge) to still be present; \
             rebuild_budgeted's overwrite bug would have dropped it"
        );

        // The critical, structurally-unique marker node from nb's class
        // (`Var(9)`) must have survived being merged into nc's class via
        // union()'s extend(). This is exactly the data the overwrite bug
        // silently discarded: union() appended it to
        // `classes[id.index()].nodes` mid-loop, and the old
        // `self.classes[id.index()].nodes = new_nodes` assignment clobbered
        // it because `new_nodes` was built from nc's pre-union node list,
        // which never contained `Var(9)`.
        assert!(
            nodes.iter().any(|n| matches!(n, ENode::Var(9))),
            "expected Var(9) (nb's unique marker, merged in via union()'s extend) \
             to still be present; rebuild_budgeted's overwrite bug drops nodes appended \
             by a mid-loop union() when the current worklist item survives as parent"
        );
    }

    /// Create an e-graph with standard algebraic rules for testing.
    fn egraph_with_rules() -> EGraph {
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
            Commutative::new(&ops::Add),
            Commutative::new(&ops::Mul),
            Commutative::new(&ops::Min),
            Commutative::new(&ops::Max),
            // Distributivity
            Distributive::new(&ops::Mul, &ops::Add),
            Distributive::new(&ops::Mul, &ops::Sub),
            // Identity
            Identity::new(&ops::Add),
            Identity::new(&ops::Mul),
            // Annihilator
            Annihilator::new(&ops::Mul),
        ];
        EGraph::with_rules(rules)
    }

    #[test]
    fn inverse_add() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![x],
        });
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, neg_x],
        });
        SaturationConfig::compatibility(100).run(&mut eg);
        let zero = eg.add(ENode::constant(0.0));
        assert_eq!(eg.find(sum), eg.find(zero));
    }

    #[test]
    fn inverse_mul() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let recip_x = eg.add(ENode::Op {
            op: &ops::Recip,
            children: vec![x],
        });
        let product = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, recip_x],
        });
        SaturationConfig::compatibility(100).run(&mut eg);
        let one = eg.add(ENode::constant(1.0));
        assert_eq!(eg.find(product), eg.find(one));
    }

    #[test]
    fn complex_inverse() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, five],
        });
        let div = eg.add(ENode::Op {
            op: &ops::Div,
            children: vec![prod, x],
        });
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
        let b_minus_c = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![b, c],
        }); // 6 - 2 = 4
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![a, b_minus_c],
        }); // 10 - 4 = 6

        SaturationConfig::compatibility(100).run(&mut eg);

        // Extract and verify structure
        let costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
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

        let d_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![d, d],
        });
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![c_sq, r_sq],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![d_sq, inner_sub],
        });

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
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

        let x_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![y, z],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x_sq, inner_sub],
        });

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
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

        let x_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![y, z],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x_sq, inner_sub],
        });

        SaturationConfig::compatibility(100).run(&mut eg);

        // Use default costs like the kernel! macro does
        let costs = CostModel::new();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
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

        let d_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![d, d],
        });
        let a_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![a, a],
        });
        let b_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![b, b],
        });
        let c_sq = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![a_sq, b_sq],
        });
        let r_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![r, r],
        });
        let inner = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![c_sq, r_sq],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![d_sq, inner],
        });

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::new();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
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
            current = eg.add(ENode::Op {
                op: &ops::Add,
                children: vec![current, one],
            });
        }

        SaturationConfig::compatibility(100).run(&mut eg);

        // Extract with default costs (high threshold)
        let default_costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(current, &default_costs);
        assert!(arena.node_count_subtree(root) > 0);

        // Extract with shallow costs (low threshold)
        let mut shallow_costs = CostModel::new();
        shallow_costs.depth_threshold = 3;
        shallow_costs.depth_penalty = 1000;
        let (arena2, root2) = eg.extract_expr_with_costs(current, &shallow_costs);
        assert!(arena2.node_count_subtree(root2) > 0);
    }

    // ========================================================================
    // Provenance tests
    // ========================================================================

    /// Find the `RewriteTarget` for a given rule name, failing loudly (not
    /// silently) if it isn't present — a missing match means the test setup
    /// is wrong, not that provenance has nothing to check.
    fn find_target(eg: &EGraph, rule_name: &str) -> RewriteTarget {
        eg.find_rewrite_matches()
            .into_iter()
            .find(|t| eg.rule(t.rule_idx).unwrap().name() == rule_name)
            .unwrap_or_else(|| panic!("no rewrite match found for rule {rule_name:?}"))
    }

    /// Like `find_target`, but further restricted to matches against a
    /// specific (already-canonical) class — needed once a class holds
    /// multiple nodes that all match the same rule name.
    fn find_target_in_class(eg: &EGraph, rule_name: &str, class: EClassId) -> RewriteTarget {
        eg.find_rewrite_matches()
            .into_iter()
            .find(|t| eg.rule(t.rule_idx).unwrap().name() == rule_name && t.class_id == class)
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
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        // Hand-derivation: x, y, and sum are all seeds.
        for &(class, _) in &[(x, "x"), (y, "y"), (sum, "sum")] {
            for &tag in eg.tags(class) {
                assert_eq!(
                    eg.provenance().origin(tag),
                    Some(Origin::Seed),
                    "seed node in class {class:?} should be Origin::Seed"
                );
            }
        }
        assert_eq!(eg.provenance().application_count(), 0);
        assert_eq!(eg.provenance().union_count(), 0);

        let target = find_target(&eg, "commutative");
        assert_eq!(target.class_id, eg.find(sum));

        let applied = eg.apply_single_rule(target.rule_idx, target.class_id, target.node_idx);
        assert!(applied, "commutative rule should have applied to x + y");

        // Hand-derivation: exactly one application recorded, for the
        // "commutative" rule, matched against `sum`'s class.
        assert_eq!(eg.provenance().application_count(), 1);
        let record = eg.provenance().application(ApplicationId(0)).unwrap();
        assert_eq!(record.rule_idx, target.rule_idx);
        assert_eq!(record.match_root, eg.find(sum));
        assert_eq!(eg.rule(record.rule_idx).unwrap().name(), "commutative");

        // Hand-derivation: the new commuted node (y + x) has Origin::Rule
        // pointing at that one application; it lives in the same class as
        // `sum` post-union.
        let commuted_class = eg.find(sum);
        let rule_origin_tags: Vec<ENodeId> = eg
            .tags(commuted_class)
            .iter()
            .copied()
            .filter(|&t| matches!(eg.provenance().origin(t), Some(Origin::Rule(_))))
            .collect();
        assert_eq!(
            rule_origin_tags.len(),
            1,
            "expected exactly one rule-created node in the merged class"
        );
        assert_eq!(
            eg.provenance().origin(rule_origin_tags[0]),
            Some(Origin::Rule(ApplicationId(0)))
        );

        // Hand-derivation: exactly one union event, attributed to the
        // commutative rule's index, at the step apply_single_rule ran on.
        assert_eq!(eg.provenance().union_count(), 1);
        let union_event = eg.provenance().union_events()[0];
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
        let inner = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        }); // x + y
        let outer = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![inner, z],
        }); // (x + y) + z

        // Rule A: commute the inner sum (x + y) -> (y + x).
        let target_a = find_target_in_class(&eg, "commutative", eg.find(inner));
        assert!(eg.apply_single_rule(target_a.rule_idx, target_a.class_id, target_a.node_idx));
        let app_a = ApplicationId(0);

        // Rule B: commute the outer sum ((x+y)+z) -> (z + (x+y)). After
        // rule A, inner's class holds both `x+y` and `y+x`, both of which
        // still match "commutative" — so we must specifically target
        // outer's class rather than take the first "commutative" match.
        let target_b = find_target_in_class(&eg, "commutative", eg.find(outer));
        assert!(eg.apply_single_rule(target_b.rule_idx, target_b.class_id, target_b.node_idx));
        let app_b = ApplicationId(1);
        assert_eq!(eg.provenance().application_count(), 2);

        // Find B's produced node: the rule-created node in outer's class
        // whose origin is app_b.
        let outer_class = eg.find(outer);
        let b_product_tag = eg
            .tags(outer_class)
            .iter()
            .copied()
            .find(|&t| eg.provenance().origin(t) == Some(Origin::Rule(app_b)))
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
        let nb = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![b],
        });
        let nc = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![c],
        });
        let marker = eg.add(ENode::Var(9));

        assert_eq!(eg.provenance().union_count(), 0);

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
            eg.provenance()
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
            let op: &'static dyn Op = match i % 3 {
                0 => &ops::Add,
                1 => &ops::Mul,
                _ => &ops::Sub,
            };
            current = eg.add(ENode::Op {
                op,
                children: vec![current, y],
            });
        }

        let start = std::time::Instant::now();
        SaturationConfig::compatibility(100).run(&mut eg);
        let elapsed = start.elapsed();

        eprintln!(
            "provenance overhead: saturation took {:?}; origins={} applications={} unions={} classes={}",
            elapsed,
            eg.provenance().origin_count(),
            eg.provenance().application_count(),
            eg.provenance().union_count(),
            eg.num_classes(),
        );
    }

    // ------------------------------------------------------------------
    // The laws. `docs/plans/2026-09-02-optimizer-api.md` §1 states each one
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
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let diff = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x, y],
        });
        let prod = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![sum, z],
        });
        let scaled = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![diff, z],
        });
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
        while eg.classes.len() < HARD_CLASS_LIMIT - 1 {
            let _ = eg.add(ENode::constant(pad));
            pad += 1.0;
        }

        // `add` is still total at the ceiling: the class it returns holds
        // the node, and it is not class 0's term.
        let fresh = eg.add(ENode::Var(200));
        assert_eq!(eg.classes.len(), HARD_CLASS_LIMIT);
        assert!(
            matches!(eg.nodes(eg.find(fresh)).first(), Some(ENode::Var(200))),
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
        let x_plus_zero = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, zero],
        });
        // (x + 0) * 1 = x, two firings away.
        let scaled = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x_plus_zero, one],
        });
        // x * 0 = 0, via Annihilator.
        let annihilated = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, zero],
        });
        // x + y = y + x, via Commutative.
        let xy = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let yx = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![y, x],
        });

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
        let ladder = [1usize, 2, 8, 64, 512, 4096, HARD_CLASS_LIMIT];
        let mut previous: Option<(usize, Vec<(usize, usize)>)> = None;
        let mut ever_found = 0usize;

        for max_classes in ladder {
            let mut eg = egraph_with_rules();
            let probes = seed_derivable_probes(&mut eg);

            // A hand-asserted equality between two terms no rule in this
            // vocabulary relates: only a retraction could ever break it.
            let a = eg.add(ENode::Var(30));
            let b = eg.add(ENode::Var(31));
            eg.union(a, b);
            eg.rebuild();
            assert_eq!(eg.find(a), eg.find(b), "union did not take");

            eg.saturate_with_limits(20, max_classes, std::time::Duration::from_secs(30));

            assert_eq!(
                eg.find(a),
                eg.find(b),
                "budget {max_classes} retracted a hand-made equality"
            );

            // Refinement: every pair equal at a smaller budget is still
            // equal at this one.
            let pairs = equal_pairs(&eg, &probes);
            if let Some((smaller, before)) = &previous {
                for pair in before {
                    assert!(
                        pairs.contains(pair),
                        "budget {max_classes} lost probe pair {pair:?} that budget \
                         {smaller} had found — saturation is not monotone"
                    );
                }
            }
            ever_found = ever_found.max(pairs.len());
            previous = Some((max_classes, pairs));
        }

        // Guard against the ladder asserting nothing: if no budget ever
        // derived an equality, the refinement check above was vacuous.
        assert!(
            ever_found > 0,
            "no budget derived any probe equality — the monotonicity ladder \
             is vacuous and no longer tests L2"
        );
    }
}

// ============================================================================
// EGraphBatch — RAII batched rule application with lazy rebuild
// ============================================================================

/// RAII batch for rule application with budgeted interleaved rebuild.
///
/// Applies rules without per-rule rebuilds. After each rule, processes
/// a chunk of the rebuild worklist to keep classes deduplicated without
/// doing a full rebuild. On drop, drains the remaining worklist.
///
/// The rebuild budget per rule is proportional to the changes made,
/// keeping total work bounded.
///
/// ```ignore
/// {
///     let mut batch = egraph.batch();
///     for rule in approved_rules {
///         batch.apply_rule(rule, 500, Some(deadline));
///     }
/// } // final rebuild on drop
/// ```
pub struct EGraphBatch<'a> {
    graph: &'a mut EGraph,
    any_changes: bool,
    /// Max worklist items to process per rule application.
    /// Keeps class sizes bounded during the batch.
    rebuild_chunk: usize,
}

impl<'a> EGraphBatch<'a> {
    /// Apply a single rule, then process a chunk of pending rebuilds.
    ///
    /// The interleaved rebuild keeps classes from ballooning between rules.
    /// Each rule application is followed by processing up to `rebuild_chunk`
    /// worklist items, so the graph stays approximately deduplicated.
    pub fn apply_rule(
        &mut self,
        rule_idx: usize,
        max_nodes: usize,
        deadline: Option<std::time::Instant>,
    ) -> ApplyResult {
        let result = self
            .graph
            .apply_rule_at_index_timed(rule_idx, max_nodes, deadline);
        if result.changes > 0 {
            self.any_changes = true;
            // Interleaved partial rebuild: process some worklist items to keep
            // classes small. The chunk size bounds work per rule.
            self.graph.rebuild_budgeted(self.rebuild_chunk);
        }
        result
    }

    /// Current number of e-classes.
    pub fn node_count(&self) -> usize {
        self.graph.classes.len()
    }

    /// Whether any rule in this batch produced changes.
    pub fn has_changes(&self) -> bool {
        self.any_changes
    }

    /// Pending rebuild worklist items.
    pub fn pending_rebuilds(&self) -> usize {
        self.graph.pending_rebuilds()
    }
}

impl Drop for EGraphBatch<'_> {
    fn drop(&mut self) {
        // Drain any remaining worklist items
        if self.any_changes {
            self.graph.rebuild();
        }
    }
}

/// Congruence closure: does `x = y` give `f(x) = f(y)`?
///
/// This is the property [`Congruence`] names, isolated from saturation so a
/// failure points at `union`/`rebuild` and nothing else. The
/// [`Congruence::Downward`] test is not an aspiration written backwards — it
/// pins the gap that exists today, so that flipping
/// [`PRODUCTION_CONGRUENCE`](super::optimizer::PRODUCTION_CONGRUENCE) has to
/// come with a deliberate edit here.
#[cfg(test)]
mod congruence_closure {
    use super::*;
    use crate::egraph::ops;

    /// `Add(x, z)` and `Add(y, z)` as two distinct classes, plus the classes
    /// of `x` and `y`.
    fn users_of_two_leaves(
        congruence: Congruence,
    ) -> (EGraph, EClassId, EClassId, EClassId, EClassId) {
        let mut eg = EGraph::new().with_congruence(congruence);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let fx = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, z],
        });
        let fy = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![y, z],
        });
        assert_ne!(eg.find(fx), eg.find(fy), "distinct before any union");
        (eg, x, y, fx, fy)
    }

    #[test]
    fn upward_merges_the_users_of_a_merged_class() {
        let (mut eg, x, y, fx, fy) = users_of_two_leaves(Congruence::Upward);
        eg.union(x, y);
        eg.rebuild();
        assert_eq!(
            eg.find(fx),
            eg.find(fy),
            "x = y must give Add(x, z) = Add(y, z): that is what makes the \
             e-graph a congruence closure, and no rule fires to establish it"
        );
    }

    /// The defect, pinned. Not a wish — a record of what the shipped default
    /// does, so the A/B has a baseline and the flip is deliberate.
    #[test]
    fn downward_leaves_the_users_unmerged() {
        let (mut eg, x, y, fx, fy) = users_of_two_leaves(Congruence::Downward);
        eg.union(x, y);
        eg.rebuild();
        assert_ne!(
            eg.find(fx),
            eg.find(fy),
            "downward repair walks only the merged class's own nodes, so the \
             users stay split — issue #1106"
        );
    }

    /// Two levels up, and through a cycle-prone shape: the repair must reach
    /// transitively, not just one hop.
    #[test]
    fn upward_repair_reaches_transitively() {
        let mut eg = EGraph::new().with_congruence(Congruence::Upward);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let inner_x = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, z],
        });
        let inner_y = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![y, z],
        });
        let outer_x = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![inner_x, z],
        });
        let outer_y = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![inner_y, z],
        });
        eg.union(x, y);
        eg.rebuild();
        assert_eq!(eg.find(inner_x), eg.find(inner_y), "one hop up");
        assert_eq!(
            eg.find(outer_x),
            eg.find(outer_y),
            "two hops up — the repair must re-enqueue owners of a class that \
             was itself merged by an earlier repair"
        );
    }

    /// A cyclic e-graph (what saturation always produces) must still
    /// terminate. `rebuild` is unbudgeted here on purpose: if the
    /// changed-only enqueue guard were wrong, this hangs rather than fails.
    #[test]
    fn upward_repair_terminates_on_a_cyclic_graph() {
        let mut eg = EGraph::new().with_congruence(Congruence::Upward);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let f = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let g = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![y, x],
        });
        // Close the cycle: the class of `f` contains a node whose child is
        // `f`'s own class after this union.
        eg.union(f, x);
        eg.rebuild();
        eg.union(g, y);
        eg.rebuild();
        assert_eq!(eg.find(f), eg.find(x));
    }

    /// Partial rebuild must leave a consistent graph — the guarantee
    /// `rebuild_budgeted` documents. Upward repair adds worklist items, so
    /// the budget now bounds strictly more work; it must still stop.
    #[test]
    fn upward_repair_respects_the_rebuild_budget() {
        let (mut eg, x, y, _fx, _fy) = users_of_two_leaves(Congruence::Upward);
        eg.union(x, y);
        let remaining = eg.rebuild_budgeted(1);
        // One item processed; whatever is left is still a valid worklist.
        assert_eq!(remaining, eg.pending_rebuilds());
        eg.rebuild();
        assert_eq!(eg.pending_rebuilds(), 0);
    }

    /// Switching direction after the graph is populated cannot silently
    /// half-work.
    #[test]
    #[should_panic(expected = "must be chosen before the first add()")]
    fn congruence_cannot_be_switched_on_a_populated_graph() {
        let mut eg = EGraph::new();
        let _ = eg.add(ENode::Var(0));
        let _ = eg.with_congruence(Congruence::Upward);
    }

    /// The downward arm pays nothing for machinery it does not use.
    #[test]
    fn downward_records_no_upward_edges() {
        let (mut eg, x, y, _, _) = users_of_two_leaves(Congruence::Downward);
        eg.union(x, y);
        eg.rebuild();
        assert_eq!(eg.upward_edge_count(), 0);
        assert_eq!(eg.upward_enqueues(), 0);
    }
}
