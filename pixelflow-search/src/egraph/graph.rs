//! The E-Graph data structure and operations.

use rustc_hash::FxHashMap as HashMap;

use super::cost::{CostFunction, CostModel};
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
#[cfg(feature = "provenance-journal")]
use super::provenance::{ApplicationRecord, Origin, UnionEvent};
use super::provenance::{ENodeId, Provenance};
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
    /// The node the rule should try to match, named by its stable
    /// [`ENodeId`] rather than by a position in the class's node vector.
    ///
    /// A position goes stale: a caller that enumerates a round's matches
    /// and then applies them one at a time can rebuild the class between
    /// enumeration and application, and `rebuild`'s take/canonicalize/extend
    /// cycle renumbers it. Re-fetching a stale *index* silently applies the
    /// rule to whichever node inherited that slot, credited to the original
    /// candidate's score and dedup key. Re-resolving a stale *tag* finds
    /// either the same node or nothing at all.
    pub tag: ENodeId,
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
#[cfg(feature = "provenance-journal")]
#[derive(Clone, Copy, Debug)]
struct ActiveApplication {
    rule_idx: usize,
    application_id: super::provenance::ApplicationId,
}

pub struct EGraph {
    pub(crate) classes: Vec<EClass>,
    pub(crate) parent: Vec<EClassId>,
    memo: HashMap<ENode, EClassId>,
    worklist: Vec<EClassId>,
    /// `queued[id.index()]` is `true` while `id` has an entry on `worklist`
    /// awaiting `rebuild_budgeted`. Guards `union`'s push: several unions
    /// landing on the same parent in one round used to enqueue it once per
    /// union, and `rebuild_budgeted` paid a full canonicalize-and-memo-probe
    /// pass over that class's node vector per redundant entry even though
    /// only the state after the *last* union in the round matters. Parallel
    /// to `classes`/`parent`, grown alongside them in `add`.
    queued: Vec<bool>,
    /// Rules are shared via Arc so EGraph can be cloned for search branching.
    rules: std::sync::Arc<Vec<Box<dyn Rewrite>>>,
    /// Stable id per rule, parallel to `rules`. Kept so the graph can name a
    /// rule by something that survives a reordering; see [`super::rules`].
    rule_ids: std::sync::Arc<Vec<RuleId>>,
    /// Matches found per rule, keyed by identity rather than by family name.
    /// A name-keyed map collapsed all four `Commutative` instances into one
    /// bucket, so every per-rule report built on it was wrong by aggregation.
    pub match_counts: HashMap<RuleId, usize>,
    /// Cumulative rule-match ATTEMPTS across this e-graph's whole lifetime —
    /// every node checked against a rule, matched or not, summed from
    /// [`ApplyResult::evals`] at the one site every rule-application path
    /// funnels through.
    ///
    /// This is "raw matches enumerated"
    /// (docs/plans/2026-09-01-phase3-round2-registration-v2.md §7.1): the
    /// Guide-cost-flatness precondition needs it denominated per
    /// application, and per application needs it cumulative across the
    /// resumed budget calls an anytime curve makes, the same way
    /// [`Self::application_count`] is.
    total_evals: usize,
    /// Global monotonic counter minting `ENodeId`s in `add()`.
    next_enode_id: u64,
    /// Saturation-iteration counter, advanced once per `saturate_with_limits`
    /// loop iteration. Recorded on every `ApplicationRecord`/`UnionEvent`.
    step: usize,
    /// Rule provenance: origins, application log, union journal.
    provenance: Provenance,
    /// Which rewrite application (if any) is currently executing — read by
    /// `add()`/`union()` to attribute newly created nodes/unions.
    #[cfg(feature = "provenance-journal")]
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
    /// the budget off `provenance().recorded_count()` would make
    /// recording a load-bearing part of saturation rather than an
    /// observation of it.
    applications: u64,
    /// Whether to record provenance. Recording had no production consumer
    /// (2026-09-01 integration audit) and a production compile discarded a
    /// median 8 446 records per kernel;
    /// [`super::optimizer::Optimizer`] turns it on exactly when an
    /// [`Observer`](super::optimizer::Observer) is attached. Defaults to
    /// `true`, so every direct `EGraph` caller keeps what it had.
    #[cfg(feature = "provenance-journal")]
    record_provenance: bool,
    /// Application ceiling for the current run, or `None`. Set for the
    /// duration of [`EGraph::saturate_budgeted`].
    application_cap: Option<u64>,
    /// Counterfactual-replay mask (docs/plans/2026-09-01-guide-return-to-go.md
    /// §4.1): when `Some`, the application that would otherwise be assigned
    /// this ordinal is skipped entirely by `apply_action_from_rule` — not
    /// counted, not recorded, not applied — and the ordinal it would have
    /// consumed is left for whatever candidate comes next in the same scan
    /// order. Installed only by
    /// [`Optimizer::mask`](super::optimizer::Optimizer::mask), so every
    /// other path is unaffected.
    replay_mask: Option<ApplicationMask>,
    /// How many applications the most recent masked run actually skipped.
    /// Reset when a mask is installed and read back by
    /// [`EGraph::last_replay_mask_skips`] — a caller reporting a
    /// confluence-aware Δ must be able to distinguish "the mask matched
    /// once" from "the mask matched eleven times", and must never have to
    /// infer that a mask fired at all.
    replay_mask_skips: usize,
    /// Global monotonic counter, bumped once per class-content mutation
    /// (`add`'s memo miss, `union`'s merge) — never reset, never bumped by
    /// anything that only re-canonicalizes existing content (`nodes()`
    /// already resolves through `find`, so rewriting a child's stored id to
    /// its current canonical form during rebuild surfaces nothing a lookup
    /// couldn't already see). See [`Self::class_is_dirty`].
    mutation_counter: u64,
    /// `last_changed[c.index()]` is the [`Self::mutation_counter`] value as
    /// of the most recent time canonical class `c`'s own node list gained
    /// content — set at creation in `add` and bumped in `union` when `c`
    /// survives as the merge's parent. Parallel to `classes`, grown
    /// alongside it.
    last_changed: Vec<u64>,
    /// `rule_last_swept[rule_idx]` is the [`Self::mutation_counter`] value
    /// captured just BEFORE that rule's most recent *complete* scan of
    /// `canonical_class_ids()` began. Parallel to `rules`. See
    /// [`Self::class_is_dirty`] for how it is used and why the value is
    /// taken from the scan's start, not its end.
    rule_last_swept: Vec<u64>,
}

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
/// extraction cost can.
///
/// This is a single, uniform, crate-wide constant rather than a per-rule
/// depth precisely so a future rule cannot silently exceed it the way a
/// per-rule value could go stale unnoticed: every rule is checked to this
/// depth regardless of what it actually needs, so a new rule that matches
/// deeper than this is caught the same way `Pythagorean` would be — by
/// `docs/plans/2026-09-08-egraph-cpu-memory-profile.md`'s methodology
/// (compare extracted cost with dirty-tracking on vs. off), not by silent
/// trust in a number nobody re-audits. Raise it if that audit is ever
/// redone and finds a deeper rule; do not lower it without redoing the
/// audit above.
const DIRTY_TRACKING_MAX_DEPTH: u8 = 2;

/// How far a counterfactual replay's mask reaches: one application, or
/// every application that would re-derive the same thing.
///
/// Leave-one-out ([`MaskScope::Single`]) answers "what did THIS firing
/// contribute", but an e-graph is confluent enough that a masked node is
/// often re-derived a few applications later by the same rule matching the
/// same class content again — so a `Δ = 0` under `Single` can mean either
/// "this application did not matter" or "this application mattered and
/// something else silently restored it". [`MaskScope::AllMatchingCandidate`]
/// separates those two: it masks the seed AND every later application
/// sharing its [`CandidateKey`](super::candidate::CandidateKey) — the same
/// key the guided loop dedups on — so no re-derivation by that route can put
/// the node back. The gap between the two Δs is this program's measurement
/// of leave-one-out's confluence blindness
/// (docs/plans/2026-09-01-guide-return-to-go.md §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskScope {
    /// Skip only the application that would be assigned `skip_ordinal`.
    Single,
    /// Skip that application and every later one with the same
    /// [`CandidateKey`](super::candidate::CandidateKey), for the rest of the
    /// run.
    AllMatchingCandidate,
}

/// Names the application to skip in a counterfactual replay, and how far the
/// skip reaches — see [`Optimizer::mask`](super::optimizer::Optimizer::mask)
/// and docs/plans/2026-09-01-guide-return-to-go.md §4.1.
///
/// `skip_ordinal` counts in [`EGraph::application_count`]'s currency, the
/// unconditional budget denominator — the same numbering
/// [`ApplicationId`](super::provenance::ApplicationId) uses for a run that
/// recorded from the start, so a harness can take an ordinal off the journal
/// and hand it straight back. Replay is deterministic and identical to the
/// original run up to the point of divergence, so replaying the same
/// expression through the same configuration and masking the ordinal that
/// was `a` in the original trajectory reproduces `τ \ a` exactly (§4.1's
/// `Δ_a = R(τ\a,B) − R(τ,B)`).
///
/// Under [`MaskScope::AllMatchingCandidate`] the key to keep masking is not
/// supplied by the caller — it is READ off the live graph at the moment the
/// seed ordinal comes up, which is the only moment it is exactly the key the
/// original trajectory's application `a` matched. A caller-supplied key
/// would be a second computation of the same thing against a different graph
/// state, i.e. the drift this codebase's one-constructor rule exists to
/// prevent.
#[derive(Clone, Debug)]
pub struct ApplicationMask {
    /// Ordinal of the seed application to skip.
    pub skip_ordinal: u64,
    /// How far the skip reaches.
    pub scope: MaskScope,
    /// The seed's `(rule, class content)`, captured when `skip_ordinal` came
    /// up. `None` until then, and always `None` under [`MaskScope::Single`],
    /// which never needs it.
    captured: Option<super::candidate::CandidateKey>,
}

impl ApplicationMask {
    /// §4.1's leave-one-out mask: skip exactly this ordinal.
    #[must_use]
    pub fn leave_one_out(skip_ordinal: u64) -> Self {
        Self {
            skip_ordinal,
            scope: MaskScope::Single,
            captured: None,
        }
    }

    /// The confluence-aware mask: skip this ordinal and every later
    /// application sharing its
    /// [`CandidateKey`](super::candidate::CandidateKey).
    #[must_use]
    pub fn all_matching_candidate(skip_ordinal: u64) -> Self {
        Self {
            skip_ordinal,
            scope: MaskScope::AllMatchingCandidate,
            captured: None,
        }
    }
}

/// What [`EGraph::mask_decision`] concluded about the application about to
/// be committed. Three distinct skips, not one boolean: the leave-one-out
/// mask must be consumed after it fires, the confluence-aware seed must
/// capture its key as it fires, and a later re-derivation must do neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaskDecision {
    Proceed,
    Skip,
    SkipAndCapture,
    SkipAndConsume,
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
            queued: self.queued.clone(),
            rules: self.rules.clone(), // Arc clone - cheap, shares rules
            rule_ids: self.rule_ids.clone(),
            match_counts: self.match_counts.clone(),
            total_evals: self.total_evals,
            next_enode_id: self.next_enode_id,
            step: self.step,
            provenance: self.provenance.clone(),
            #[cfg(feature = "provenance-journal")]
            active_application: self.active_application,
            const_fact: self.const_fact.clone(),
            refused_const_unions: self.refused_const_unions.clone(),
            applications: self.applications,
            #[cfg(feature = "provenance-journal")]
            record_provenance: self.record_provenance,
            application_cap: self.application_cap,
            replay_mask: self.replay_mask.clone(),
            replay_mask_skips: self.replay_mask_skips,
            mutation_counter: self.mutation_counter,
            last_changed: self.last_changed.clone(),
            rule_last_swept: self.rule_last_swept.clone(),
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
            memo: HashMap::default(),
            worklist: Vec::new(),
            queued: Vec::new(),
            rules: std::sync::Arc::new(Vec::new()),
            rule_ids: std::sync::Arc::new(Vec::new()),
            match_counts: HashMap::default(),
            total_evals: 0,
            next_enode_id: 0,
            step: 0,
            provenance: Provenance::new(),
            #[cfg(feature = "provenance-journal")]
            active_application: None,
            const_fact: Vec::new(),
            refused_const_unions: Vec::new(),
            applications: 0,
            #[cfg(feature = "provenance-journal")]
            record_provenance: true,
            application_cap: None,
            replay_mask: None,
            replay_mask_skips: 0,
            mutation_counter: 0,
            last_changed: Vec::new(),
            rule_last_swept: Vec::new(),
        }
    }

    /// Create an e-graph with the given rewrite rules.
    ///
    /// Rules are owned by the e-graph and shared via Arc when cloned.
    pub fn with_rules(rules: Vec<Box<dyn Rewrite>>) -> Self {
        let ids: Vec<RuleId> = rules.iter().map(|r| RuleId::of(r.as_ref())).collect();
        let rule_last_swept = vec![0u64; rules.len()];
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::default(),
            worklist: Vec::new(),
            queued: Vec::new(),
            rules: std::sync::Arc::new(rules),
            rule_ids: std::sync::Arc::new(ids),
            match_counts: HashMap::default(),
            total_evals: 0,
            next_enode_id: 0,
            step: 0,
            provenance: Provenance::new(),
            #[cfg(feature = "provenance-journal")]
            active_application: None,
            const_fact: Vec::new(),
            refused_const_unions: Vec::new(),
            applications: 0,
            #[cfg(feature = "provenance-journal")]
            record_provenance: true,
            application_cap: None,
            replay_mask: None,
            replay_mask_skips: 0,
            mutation_counter: 0,
            last_changed: Vec::new(),
            rule_last_swept,
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
        self.rule_last_swept.push(0);
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
        eg.rule_last_swept = vec![0; rules.len()];
        eg.rules = rules;
        eg.rule_ids = rule_ids;
        eg
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
    #[cfg(feature = "provenance-journal")]
    pub fn set_provenance_recording(&mut self, on: bool) {
        self.record_provenance = on;
    }

    /// Whether provenance is being recorded.
    #[cfg(feature = "provenance-journal")]
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
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_) => {}
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
        #[cfg(feature = "provenance-journal")]
        {
            let origin = match self.active_application {
                Some(active) => Origin::Rule(active.application_id),
                None => Origin::Seed,
            };
            self.provenance.record_origin(enode_id, origin);
        }
        self.const_fact.push(node.as_f32().map(f32::to_bits));
        self.classes.push(EClass {
            nodes: vec![node.clone()],
            tags: vec![enode_id],
        });
        self.parent.push(id);
        self.queued.push(false);
        self.mutation_counter += 1;
        self.last_changed.push(self.mutation_counter);
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
        let child_nodes = std::mem::take(&mut self.classes[child.index()].nodes);
        let child_tags = std::mem::take(&mut self.classes[child.index()].tags);
        self.classes[parent.index()].nodes.extend(child_nodes);
        self.classes[parent.index()].tags.extend(child_tags);
        // The guard above proved the two facts agree (or that at most one
        // exists), so `or` is a merge, not a choice.
        if self.const_fact[parent.index()].is_none() {
            self.const_fact[parent.index()] = self.const_fact[child.index()];
        }
        // `parent`'s own node list just changed (gained `child`'s nodes) —
        // see `class_is_dirty`/`DIRTY_TRACKING_MAX_DEPTH`.
        self.mutation_counter += 1;
        self.last_changed[parent.index()] = self.mutation_counter;
        if !self.queued[parent.index()] {
            self.queued[parent.index()] = true;
            self.worklist.push(parent);
        }
        #[cfg(feature = "provenance-journal")]
        self.provenance.record_union(UnionEvent {
            rule_idx: self.active_application.map(|a| a.rule_idx),
            application_id: self.active_application.map(|a| a.application_id),
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
            // Cleared by the slot pushed at (not the class it now
            // canonicalizes to) — matches `union`'s guard, which sets it by
            // the same slot.
            self.queued[id.index()] = false;
            let id = self.find(id);
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
                        // `union` keeps `min(id, existing)`, so this call
                        // can go either way, and the write-back at the
                        // bottom of the loop has to survive both:
                        //   - `id` survives: `union`'s extend() appends
                        //     `existing`'s nodes (and tags) directly onto
                        //     `self.classes[id.index()]`, which mem::take
                        //     just emptied — so the write-back must EXTEND,
                        //     not overwrite, or those are dropped.
                        //   - `existing` survives: `id` is merged away and
                        //     `classes[id.index()]` stops being reachable
                        //     through `find` — so the write-back must target
                        //     `self.find(id)`, or everything still in
                        //     `new_nodes` is orphaned.
                        // This is a rebuild-time
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
            // Write back to the class `id` CANONICALIZES TO, not to `id`.
            // A mid-loop union() above picks `min(id, existing)` as the
            // surviving parent, so when `existing.0 < id.0` it is `id` that
            // was merged away — and `classes[id.index()]` is then a slot
            // `find` no longer routes to. Every node written there is
            // orphaned: `nodes()`/`tags()` canonicalize first, so nothing
            // can read them again. That is lost extraction alternatives
            // (under-merging: sound, since the surviving class still holds
            // only provably-equal terms) but lost *silently*, which this
            // project forbids.
            //
            // And extend, never assign: in the mirror case, where `id`
            // survives, union()'s extend() has already appended `existing`'s
            // nodes and tags onto this very vector (which mem::take emptied
            // above), and an assignment would clobber them. `find(id) == id`
            // there, so one write-back handles both directions.
            let dest = self.find(id);
            self.classes[dest.index()].nodes.extend(new_nodes);
            self.classes[dest.index()].tags.extend(new_tags);
        }
        self.worklist.len()
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
    #[cfg(feature = "provenance-journal")]
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
        let tags_of = |class: EClassId| -> Vec<ENodeId> { self.tags(class).to_vec() };
        let children_of = |tag: ENodeId| -> Vec<EClassId> {
            for class in self.classes.iter() {
                if let Some(idx) = class.tags.iter().position(|&t| t == tag) {
                    return class.nodes[idx].children();
                }
            }
            Vec::new()
        };
        let canonical_of = |class: EClassId| -> EClassId { self.find(class) };
        super::provenance::derivation_ancestors_tight(
            &tags_of,
            &children_of,
            &canonical_of,
            &self.provenance,
            seeds,
            chosen_nodes,
        )
    }

    /// Render a human-readable derivation trace for the given ancestry set
    /// (from [`EGraph::derivation_ancestors`]), resolving rule names via
    /// this e-graph's rule list.
    #[cfg(feature = "provenance-journal")]
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
            ENode::Uniform(_) => pixelflow_ir::OpKind::Uniform,
            ENode::Param(_) => pixelflow_ir::OpKind::Param,
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

                let tags = &self.classes[class_id.index()].tags;
                for (node_idx, node) in nodes.iter().enumerate() {
                    // Check if rule matches this node
                    if rule.apply(self, class_id, node).is_some() {
                        matches.push(RewriteTarget {
                            rule_idx,
                            class_id,
                            tag: tags[node_idx],
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
    pub fn apply_single_rule(&mut self, rule_idx: usize, class_id: EClassId, tag: ENodeId) -> bool {
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
        // Resolve the stable tag against the class as it stands NOW — a
        // rebuild since the caller enumerated this target either left the
        // node in place or removed it, and `None` is the honest answer for
        // the second case. See `RewriteTarget::tag`.
        let Some(node) = self.node_for_tag(class_id, tag).cloned() else {
            return false;
        };

        let Some(action) = rule.apply(self, class_id, &node) else {
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

    /// Whether `canonical`'s forward neighborhood — itself, its direct
    /// children, and their children (see [`DIRTY_TRACKING_MAX_DEPTH`]) —
    /// contains anything that changed since `swept`, a
    /// [`Self::mutation_counter`] value from some rule's last complete scan.
    ///
    /// Reads only `EClassId`s and `last_changed` entries, never clones a
    /// node vector or calls a rule's `apply` — this is the cheap check
    /// [`Self::apply_rule_at_index_timed`] uses to decide whether the
    /// expensive path (clone the class's nodes, run every rule against
    /// every one) is worth paying for at all. Canonicalizes every id it
    /// follows: `parent`/`child` fields on a stored node can lag a union
    /// that happened after the node was written, and `last_changed` is only
    /// ever indexed by canonical class slots.
    fn class_is_dirty(&self, canonical: EClassId, swept: u64) -> bool {
        let canonical = self.find(canonical);
        if self.last_changed[canonical.index()] > swept {
            return true;
        }
        for node in &self.classes[canonical.index()].nodes {
            let ENode::Op { children, .. } = node else {
                continue;
            };
            for &child in children {
                let child = self.find(child);
                if self.last_changed[child.index()] > swept {
                    return true;
                }
                if DIRTY_TRACKING_MAX_DEPTH < 2 {
                    continue;
                }
                for gnode in &self.classes[child.index()].nodes {
                    let ENode::Op {
                        children: gchildren,
                        ..
                    } = gnode
                    else {
                        continue;
                    };
                    for &gchild in gchildren {
                        if self.last_changed[self.find(gchild).index()] > swept {
                            return true;
                        }
                    }
                }
            }
        }
        false
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

        // Captured before the scan touches anything: `class_is_dirty` below
        // compares every candidate class against `swept` (this rule's OWN
        // last complete-scan baseline, from a previous call), and — only if
        // this scan itself runs to completion — `rule_last_swept[rule_idx]`
        // is advanced to `sweep_start` at the end. Taking the *start* value
        // rather than the counter's value after the scan means a mutation
        // this very call makes (via the commit loop's unions below) is
        // correctly seen as new next time this rule runs, without needing
        // to re-examine it against itself right now.
        let sweep_start = self.mutation_counter;
        let swept = self.rule_last_swept[rule_idx];

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
            // Nothing within DIRTY_TRACKING_MAX_DEPTH hops of `canonical`
            // has changed since this rule last swept the whole graph, so
            // this rule's answer on `canonical` cannot have changed either
            // — skip straight past the clone and every `apply` call below.
            if !self.class_is_dirty(canonical, swept) {
                continue;
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

        // Only a scan that reached every canonical id without a budget/
        // deadline break actually looked at everything — advancing the
        // baseline after a cut-short scan would let a class this call never
        // got to sit unswept-but-marked-swept, which is exactly the
        // under-saturation `class_is_dirty` exists to never cause. A
        // cut-short scan leaves `rule_last_swept[rule_idx]` at its old
        // value, so next call simply sees more as dirty than strictly
        // necessary — safe, just less tight.
        if scan == ScanStop::Completed {
            self.rule_last_swept[rule_idx] = sweep_start;
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

        self.total_evals += evals;
        ApplyResult {
            changes: unions,
            evals,
            scan,
        }
    }

    /// Cumulative rule-match attempts across this e-graph's whole lifetime
    /// (see the `total_evals` field doc) — the raw-matches-enumerated
    /// counter §7.1's Guide-overhead-flatness measurement denominates.
    #[must_use]
    pub const fn total_evals(&self) -> usize {
        self.total_evals
    }

    /// How many applications the most recent installed mask skipped. `0`
    /// when no mask was installed, or when the mask's ordinal was never
    /// reached — a caller reporting Δ reads this rather than assuming.
    #[must_use]
    pub const fn last_replay_mask_skips(&self) -> usize {
        self.replay_mask_skips
    }

    /// Install (or clear) the counterfactual-replay mask, resetting the skip
    /// counter. Crate-private on purpose: the mask is a *policy*, in exactly
    /// the sense a `Reranker` and an `Observer` are, and it reaches the graph
    /// through [`Optimizer::mask`](super::optimizer::Optimizer::mask) rather
    /// than through a second public saturation entry point.
    pub(crate) fn set_replay_mask(&mut self, mask: Option<ApplicationMask>) {
        self.replay_mask = mask;
        self.replay_mask_skips = 0;
    }

    /// Decide what the installed replay mask (if any) says about the
    /// application about to be committed. Split out of
    /// `apply_action_from_rule` because the `AllMatchingCandidate` arm needs
    /// to READ the graph (`ClassContentKey::of`) while the mask is borrowed,
    /// and then WRITE the mask — two borrows that cannot overlap.
    fn mask_decision(&self, rule_idx: usize, class_id: EClassId) -> MaskDecision {
        let Some(mask) = &self.replay_mask else {
            return MaskDecision::Proceed;
        };
        // The counter is incremented right after this returns `Proceed`, so
        // its value right now IS the ordinal about to be consumed.
        let at_seed = mask.skip_ordinal == self.applications;
        match (mask.scope, &mask.captured) {
            // Leave-one-out: fire exactly once, then consume the mask.
            (MaskScope::Single, _) => {
                if at_seed {
                    MaskDecision::SkipAndConsume
                } else {
                    MaskDecision::Proceed
                }
            }
            // Confluence-aware, before the seed: nothing to compare against
            // yet, so only the seed ordinal itself is masked — and masking
            // it is what captures the key.
            (MaskScope::AllMatchingCandidate, None) => {
                if at_seed {
                    MaskDecision::SkipAndCapture
                } else {
                    MaskDecision::Proceed
                }
            }
            // Confluence-aware, after the seed: mask every re-derivation by
            // the same (rule, canonical matched-class content).
            (MaskScope::AllMatchingCandidate, Some(key)) => {
                let same_rule = self.rule_ids.get(rule_idx).copied() == Some(key.rule);
                if same_rule
                    && key.content
                        == super::candidate::ClassContentKey::of(self, self.find(class_id))
                {
                    MaskDecision::Skip
                } else {
                    MaskDecision::Proceed
                }
            }
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
        // Counterfactual replay (docs/plans/2026-09-01-guide-return-to-go.md
        // §4.1): a masked application is not counted, not recorded and not
        // applied, so everything downstream — the rest of this scan, the
        // rest of this sweep, later sweeps — proceeds exactly as it would
        // against a graph that simply never received it, and the ordinal it
        // would have consumed goes to the next candidate. Checked BEFORE the
        // counter, which is what makes that last clause true. Which
        // applications are masked is `mask_decision`'s call; see
        // [`MaskScope`].
        match self.mask_decision(rule_idx, class_id) {
            MaskDecision::Proceed => {}
            MaskDecision::Skip => {
                self.replay_mask_skips += 1;
                return 0;
            }
            MaskDecision::SkipAndCapture => {
                let key = super::candidate::CandidateKey {
                    rule: self.rule_ids.get(rule_idx).copied().unwrap_or_else(|| {
                        panic!(
                            "replay mask: rule_idx {rule_idx} has no id in this graph's rule \
                             table — a mask names an application, and an application names a rule"
                        )
                    }),
                    content: super::candidate::ClassContentKey::of(self, self.find(class_id)),
                };
                let mask = self
                    .replay_mask
                    .as_mut()
                    .expect("mask_decision returned SkipAndCapture with no mask installed");
                mask.captured = Some(key);
                self.replay_mask_skips += 1;
                return 0;
            }
            MaskDecision::SkipAndConsume => {
                self.replay_mask = None;
                self.replay_mask_skips += 1;
                return 0;
            }
        }

        // Counted unconditionally: the budget must not depend on whether
        // anyone is watching.
        self.applications += 1;

        #[cfg(feature = "provenance-journal")]
        {
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
            // The record is opened before the action runs (so `add`/`union`
            // can attribute to it) and closed after, which is the only order
            // in which "what did this application mint" is answerable.
            self.provenance.complete_application(
                application_id,
                minted_from..self.next_enode_id,
                result,
            );
            return result;
        }

        #[cfg(not(feature = "provenance-journal"))]
        {
            // `rule_idx` is only needed to attribute a journal record —
            // with the journal compiled out there is nothing to attribute.
            let _ = rule_idx;
            self.apply_action(class_id, action)
        }
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

    /// Materialize `template`'s subtree at `id` into this e-graph, bottom-up,
    /// reading each metavariable leaf from `bindings` rather than creating a
    /// node for it — the same convention every hand-written multi-node
    /// [`RewriteAction`] uses (`Distribute`, `Canonicalize`, …), just driven
    /// by a runtime pattern instead of a fixed shape. `self.add` is what
    /// attributes every node created here to the active application's
    /// provenance, so this must only ever be called from inside
    /// `apply_action` (i.e. under `apply_action_from_rule`).
    ///
    /// # Panics
    ///
    /// On a `Param`/`Buffer` node (a rewrite RHS template must never contain
    /// either), on an `OpKind` with no static [`Op`] (an arena/op-table
    /// drift, not a runtime condition), or on a metavariable with no
    /// binding.
    fn instantiate_template(
        &mut self,
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
            ExprData::Const(v) => self.add(ENode::constant(f32::from_bits(v))),
            ExprData::Param(p) => {
                panic!("instantiate_template: Param({p}) in a rewrite RHS template")
            }
            ExprData::Buffer(b) => {
                panic!("instantiate_template: Buffer({b:?}) in a rewrite RHS template")
            }
            ExprData::Uniform(u) => {
                panic!("instantiate_template: Uniform({u:?}) in a rewrite RHS template")
            }
            ExprData::Op(kind) => {
                let static_op = ops::op_from_kind(kind).unwrap_or_else(|| {
                    panic!("instantiate_template: no static Op for OpKind {kind:?}")
                });
                let children: Vec<EClassId> = node
                    .children()
                    .map(|c| self.instantiate_template(c, bindings))
                    .collect();
                self.add(ENode::Op {
                    op: static_op,
                    children,
                })
            }
        }
    }

    fn apply_action(&mut self, class_id: EClassId, action: RewriteAction) -> usize {
        match action {
            RewriteAction::Union(target_id) => self.union_counted(class_id, target_id),
            RewriteAction::Instantiate {
                template,
                entry,
                bindings,
            } => {
                let node = template.0.entry_at(entry);
                let result_id = self.instantiate_template(node, &bindings);
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
            // Invariant across the lattice: ∂u/∂x = 0, as for a constant.
            ENode::Uniform(_) => return self.add(ENode::constant(0.0)),
            // Likewise ∂p/∂x = 0: a builder's scalar is one number for the
            // whole lattice, whichever number it turns out to be.
            ENode::Param(_) => return self.add(ENode::constant(0.0)),
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
        let mut cost_table: HashMap<EClassId, (usize, ENode)> = HashMap::default();
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
            .children_slice()
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

    /// Soundness regression for saturation's dirty-class tracking
    /// (`class_is_dirty`, `DIRTY_TRACKING_MAX_DEPTH`). `Pythagorean` is one
    /// of exactly two production rules that inspects a GRANDCHILD e-class
    /// (`math/trig.rs`'s `extract_trig_arg`, reached through
    /// `extract_squared_trig`): `sin²(x) + cos²(x) → 1` only matches once
    /// the argument of `Mul(p, p)` is *itself* known to be `Sin(x)`, two
    /// hops below the `Add` node the rule is handed.
    ///
    /// Builds `Add(Mul(p, p), Mul(Cos(x), Cos(x)))` where `p` starts as an
    /// unrelated `Neg(x)` node in its own class — nothing marks it as
    /// `Sin(x)` yet, so `Pythagorean` cannot fire on the first scan. A later
    /// `union(p, sin_x)` (standing in for whatever upstream rewrite would
    /// normally prove `p == sin(x)`) changes `p`'s class content without
    /// touching `sum`'s or `p_sq`'s own node lists — exactly the shape a
    /// 1-hop dirty-tracking scheme would miss, and exactly
    /// `DIRTY_TRACKING_MAX_DEPTH`'s boundary. Asserts a further scan of
    /// `Pythagorean` still finds and fires the identity.
    #[test]
    fn dirty_tracking_still_finds_a_rewrite_two_hops_below_a_changed_grandchild() {
        use crate::math::trig::Pythagorean;

        let mut egraph = EGraph::with_rules(vec![Pythagorean::new()]);
        let op = |kind| ops::op_from_kind(kind).expect("op is modelled");

        let x = egraph.add(ENode::Var(0));
        let sin_x = egraph.add(ENode::Op {
            op: op(pixelflow_ir::OpKind::Sin),
            children: vec![x],
        });
        let cos_x = egraph.add(ENode::Op {
            op: op(pixelflow_ir::OpKind::Cos),
            children: vec![x],
        });
        let cos_sq = egraph.add(ENode::Op {
            op: op(pixelflow_ir::OpKind::Mul),
            children: vec![cos_x, cos_x],
        });

        // `p` starts life indistinguishable from any other opaque unary
        // node — NOT unioned with `sin_x` yet.
        let p = egraph.add(ENode::Op {
            op: op(pixelflow_ir::OpKind::Neg),
            children: vec![x],
        });
        let p_sq = egraph.add(ENode::Op {
            op: op(pixelflow_ir::OpKind::Mul),
            children: vec![p, p],
        });
        let sum = egraph.add(ENode::Op {
            op: op(pixelflow_ir::OpKind::Add),
            children: vec![p_sq, cos_sq],
        });

        // First scan: nothing to find (`p` isn't known to be `sin(x)`).
        let result = egraph.apply_rule_at_index_budgeted(0, 10_000);
        assert_eq!(
            result.changes, 0,
            "the identity cannot match before p == sin(x) is known"
        );
        egraph.rebuild();

        // Now prove p == sin(x) — the grandchild change the rule needs, two
        // hops below `sum`, with `sum`'s and `p_sq`'s own node lists
        // untouched.
        egraph.union(p, sin_x);
        egraph.rebuild();

        // Second scan of the SAME rule must still see `sum` as worth
        // checking, even though neither `sum` nor `p_sq` (only their
        // grandchild `p`) changed.
        let result = egraph.apply_rule_at_index_budgeted(0, 10_000);
        assert_eq!(
            result.changes, 1,
            "dirty-class tracking must not skip `sum` just because its own \
             node list (and its direct child p_sq's) never changed — only \
             the grandchild `p` did, which is exactly Pythagorean's \
             matching depth"
        );
        egraph.rebuild();

        let costs = CostModel::default();
        let (_, _, cost) = crate::egraph::extract::extract(&egraph, egraph.find(sum), &costs);
        assert_eq!(
            cost, 0,
            "sum's class must now also contain Const(1.0), cost 0"
        );
    }

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

    /// The mirror of the test above, and the case its fix left open: when
    /// canonicalizing the worklist item `id`'s nodes triggers a union whose
    /// surviving parent is the OTHER class, `id` itself becomes
    /// non-canonical — and the write-back at the bottom of the loop targets
    /// `classes[id.index()]`, a slot `find` no longer routes to. Every node
    /// `id` still held is orphaned: `EGraph::nodes(id)` canonicalizes first,
    /// so nothing can ever read them again.
    ///
    /// Lost e-nodes are lost extraction alternatives (under-merging, not
    /// unsoundness — every remaining class still holds only provably-equal
    /// terms), but they are lost *silently*, which this project forbids.
    /// The fix writes back to `self.find(id)`, extending rather than
    /// assigning because `union` may already have moved nodes there.
    ///
    /// Trigger recipe — the same shape as the test above with the two `Neg`
    /// classes created in the opposite order, which is all it takes to flip
    /// which side `union`'s `min(a, b)` keeps:
    /// - `nb = Neg(b)` (class 3) is created BEFORE `nc = Neg(c)` (class 4),
    ///   so `memo[Neg([1])] -> 3` and `3 < 4`.
    /// - `union(nc, marker)` gives class 4 a structurally unique `Var(9)`
    ///   and enqueues it for rebuild.
    /// - `union(b, c)` merges class 2 into class 1, so canonicalizing
    ///   `Neg([2])` during class 4's rebuild rewrites it to `Neg([1])`.
    /// - `memo` maps `Neg([1])` to class 3, and `existing (3) < id (4)`, so
    ///   `union(4, 3)` keeps **3** — `id` is merged away mid-loop.
    /// - The write-back then appends `Var(9)` to class 4, which `find` now
    ///   routes past. `nodes(find(nc))` never sees it again.
    #[test]
    fn rebuild_budgeted_does_not_orphan_nodes_when_current_class_is_merged_away() {
        let mut eg = EGraph::new();

        let a = eg.add(ENode::Var(0)); // class 0 (keeps the ids spaced out)
        let b = eg.add(ENode::Var(1)); // class 1
        let c = eg.add(ENode::Var(2)); // class 2
        let nb = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![b],
        }); // class 3: memo Neg([1]) -> 3
        let nc = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![c],
        }); // class 4: memo Neg([2]) -> 4
        let marker = eg.add(ENode::Var(9)); // class 5: a node unique to nc's class

        assert_eq!(nb.index(), 3, "test assumes nb is class 3");
        assert_eq!(nc.index(), 4, "test assumes nc is class 4");

        // Give nc's class a node with no structural twin anywhere else
        // (`Var(9)`), so its loss is directly observable, and enqueue nc for
        // rebuild. nc (4) < marker (5), so nc survives as parent here.
        eg.union(nc, marker);
        assert_eq!(eg.pending_rebuilds(), 1);

        // Merge c into b. b (1) < c (2), so b survives. This is what makes
        // `Neg([2])` canonicalize to `Neg([1])` during nc's rebuild.
        eg.union(b, c);

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

        // The critical assertion: `Var(9)` lived in nc's class when nc was
        // merged away mid-loop, so it must still be reachable through the
        // surviving class. The orphaning bug writes it back to class 4,
        // which `find` routes past — `nodes()` canonicalizes, so it is
        // unreachable from every public entry point from then on.
        let nodes = eg.nodes(surviving);
        assert!(
            nodes.iter().any(|n| matches!(n, ENode::Var(9))),
            "expected Var(9) (nc's unique marker) to still be reachable via \
             nodes(find(nc)); the write-back targeted the merged-away slot \
             classes[{}] instead of classes[{}]. Got: {nodes:?}",
            nc.index(),
            surviving.index()
        );

        // Tags must stay zipped with nodes through the redirected write-back.
        assert_eq!(
            eg.nodes(surviving).len(),
            eg.tags(surviving).len(),
            "EClass.nodes and EClass.tags must never desync"
        );

        // Nothing else got orphaned on the way. Stated as the law the public
        // API actually owes a caller — **every id `add` ever handed out still
        // resolves to a class containing its node** — rather than by reading
        // the merged-away slot directly: a shadow copy is unreachable by
        // definition, so the only honest way to catch one is to check that
        // everything reachable is still there.
        let holds = |eg: &EGraph, id: EClassId, want: &ENode| -> bool {
            eg.nodes(eg.find(id)).iter().any(|n| n == want)
        };
        let neg_b = ENode::Op {
            op: &ops::Neg,
            children: vec![eg.find(b)],
        };
        for (id, want) in [
            (a, ENode::Var(0)),
            (b, ENode::Var(1)),
            (c, ENode::Var(2)),
            (marker, ENode::Var(9)),
            (nb, neg_b.clone()),
            (nc, neg_b.clone()),
        ] {
            assert!(
                holds(&eg, id, &want),
                "id {} no longer resolves to a class holding {want:?}",
                id.index()
            );
        }

        // And the graph must be at rest: a second rebuild is a no-op. The
        // orphaning bug leaves the worklist naming a class whose nodes went
        // somewhere else, so "reachable" was not yet a fixpoint — a state
        // that can look correct until something rebuilds again.
        let before: Vec<ENode> = eg.nodes(surviving).to_vec();
        eg.rebuild();
        assert_eq!(
            eg.find(nc),
            surviving,
            "a second rebuild must not move the surviving class"
        );
        assert_eq!(
            eg.nodes(eg.find(nc)),
            before.as_slice(),
            "rebuild must be idempotent once the worklist is drained"
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
        assert_eq!(eg.provenance().recorded_count(), 0);
        assert_eq!(eg.provenance().union_count(), 0);

        let target = find_target(&eg, "commutative");
        assert_eq!(target.class_id, eg.find(sum));

        let applied = eg.apply_single_rule(target.rule_idx, target.class_id, target.tag);
        assert!(applied, "commutative rule should have applied to x + y");

        // Hand-derivation: exactly one application recorded, for the
        // "commutative" rule, matched against `sum`'s class.
        assert_eq!(eg.provenance().recorded_count(), 1);
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
        assert!(eg.apply_single_rule(target_a.rule_idx, target_a.class_id, target_a.tag));
        let app_a = ApplicationId(0);

        // Rule B: commute the outer sum ((x+y)+z) -> (z + (x+y)). After
        // rule A, inner's class holds both `x+y` and `y+x`, both of which
        // still match "commutative" — so we must specifically target
        // outer's class rather than take the first "commutative" match.
        let target_b = find_target_in_class(&eg, "commutative", eg.find(outer));
        assert!(eg.apply_single_rule(target_b.rule_idx, target_b.class_id, target_b.tag));
        let app_b = ApplicationId(1);
        assert_eq!(eg.provenance().recorded_count(), 2);

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
            eg.provenance().recorded_count(),
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

#[cfg(test)]
mod mask_tests {
    use super::*;
    use crate::egraph::optimizer::{Budget, Optimizer};

    /// A `(x*x + y*y).sqrt()`-shaped arena under the full rule set — a real
    /// sweep where idempotent re-fires (`commutative`/`associative`)
    /// dominate, which is exactly the regime
    /// [`MaskScope::AllMatchingCandidate`] exists to see through.
    fn mask_fixture() -> (Optimizer, EGraph, EClassId) {
        let mut arena = pixelflow_ir::ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let xx = arena.push_binary(pixelflow_ir::OpKind::Mul, x, x);
        let yy = arena.push_binary(pixelflow_ir::OpKind::Mul, y, y);
        let sum = arena.push_binary(pixelflow_ir::OpKind::Add, xx, yy);
        let dist = arena.push_unary(pixelflow_ir::OpKind::Sqrt, sum);
        let doubled = arena.push_binary(pixelflow_ir::OpKind::Add, dist, dist);
        let root = arena.push_binary(pixelflow_ir::OpKind::Sub, doubled, doubled);
        let opt = Optimizer::production()
            .budget(Budget::Applications(MASK_BUDGET))
            .no_ceiling();
        let mut eg = opt.egraph();
        let root_class =
            crate::egraph::insert(&arena, root, &mut eg, crate::egraph::Vocabulary::Templates)
                .expect("insert into e-graph");
        (opt, eg, root_class)
    }

    const MASK_BUDGET: u64 = 60;
    const NODES: usize = 8;

    /// The leave-one-out mask is consumed after it fires: exactly one skip,
    /// no matter how many later applications look like it.
    #[test]
    fn leave_one_out_mask_skips_exactly_one_application() {
        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::leave_one_out(0)));
        opt.run(&mut eg, root, NODES);
        assert_eq!(
            eg.last_replay_mask_skips(),
            1,
            "MaskScope::Single must fire exactly once"
        );
    }

    /// The confluence-aware mask keeps firing: it skips the seed and every
    /// later application of the same (rule, canonical matched-class
    /// content). Under rules that re-match their own installed output, that
    /// is strictly more than one.
    #[test]
    fn all_matching_candidate_mask_skips_every_re_derivation() {
        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::all_matching_candidate(0)));
        opt.run(&mut eg, root, NODES);
        assert!(
            eg.last_replay_mask_skips() > 1,
            "MaskScope::AllMatchingCandidate must keep masking re-derivations of the seed \
             candidate, got {} skip(s)",
            eg.last_replay_mask_skips()
        );
    }

    /// A mask whose ordinal is never reached leaves no trace: zero skips,
    /// and — because nothing was withheld — the same run an unmasked
    /// optimizer produces, down to the extracted term.
    #[test]
    fn a_mask_whose_ordinal_never_fires_changes_nothing() {
        let (mut plain_opt, mut plain_eg, plain_root) = mask_fixture();
        let plain = plain_opt.run(&mut plain_eg, plain_root, NODES);

        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::leave_one_out(u64::MAX)));
        let masked = opt.run(&mut eg, root, NODES);

        assert_eq!(eg.last_replay_mask_skips(), 0);
        assert_eq!(plain.stats.applications, masked.stats.applications);
        assert_eq!(plain.stats.unions, masked.stats.unions);
        assert_eq!(plain.choices, masked.choices);
    }

    /// The whole point of a counterfactual: withholding a real application
    /// must actually withhold it, and the run must be a different run.
    ///
    /// A skipped ordinal is not *burned* — it is left for the next candidate
    /// in the same scan order — so the mask does not by itself shorten the
    /// budget. It can still end the run sooner, and on this fixture it does:
    /// withholding a whole candidate class removes the work that would have
    /// kept the sweep productive, and the masked run quiesces well inside
    /// the 60-application budget while the unmasked one spends all of it.
    /// That is the counterfactual working, so the assertion is the honest
    /// one — the graphs differ — not an equality that happens to hold on a
    /// luckier fixture.
    #[test]
    fn withholding_a_real_application_changes_the_run() {
        let (mut plain_opt, mut plain_eg, plain_root) = mask_fixture();
        let plain = plain_opt.run(&mut plain_eg, plain_root, NODES);

        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::all_matching_candidate(0)));
        let masked = opt.run(&mut eg, root, NODES);

        assert!(eg.last_replay_mask_skips() > 0, "the seed must have fired");
        assert!(
            masked.stats.applications <= plain.stats.applications,
            "a mask can only withhold work, never manufacture it: {} vs {}",
            masked.stats.applications,
            plain.stats.applications
        );
        assert_ne!(
            (plain.stats.unions, plain.stats.classes),
            (masked.stats.unions, masked.stats.classes),
            "withholding a candidate and every re-derivation of it must change what the \
             run built — otherwise Δ is measuring nothing"
        );
    }
}
