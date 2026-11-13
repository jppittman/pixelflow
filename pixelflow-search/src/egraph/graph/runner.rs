//! The runner: a rule set swept over an e-graph, and every number that is
//! denominated in that sweep's history.
//!
//! [`EGraph`] is a store: `add`, `union`, `find`, `rebuild`, the class facts
//! and its own mutation log. It holds no rules and counts no applications.
//! [`Saturation`] pairs one with the [`RuleSet`] that sweeps it and the state
//! the sweep keeps *about that graph*: the per-rule dirty watermark (a
//! `mutation_counter` value, meaningless against any other graph), the
//! application and evaluation counts, the replay mask and the growth
//! telemetry. That state travels with the graph because it is denominated in
//! it. An anytime curve resumes the same pair a checkpoint at a time; a
//! runner rebuilt per call would find every class dirty again, re-fire what
//! it had already fired, and move every count.
//!
//! This module is a child of `graph` so that it reaches the store's private
//! state (the live-class set, the change log, the mutation log) without any
//! of it becoming `pub(crate)`.

use core::ops::{Deref, DerefMut};
use rustc_hash::FxHashMap as HashMap;
use std::time::{Duration, Instant};

use super::EGraph;
use crate::egraph::node::{EClassId, ENode};
use crate::egraph::provenance::ENodeId;
#[cfg(feature = "provenance-journal")]
use crate::egraph::provenance::Opening;
use crate::egraph::rewrite::RewriteAction;
use crate::egraph::rules::{RuleId, RuleIdx, RuleSet};

/// A potential rewrite target: (rule, e-class, node within class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RewriteTarget {
    /// The rule, as a position in the sweep's [`RuleSet`]
    pub rule_idx: RuleIdx,
    /// The e-class to apply the rule to
    pub class_id: EClassId,
    /// The node the rule should try to match, named by its stable
    /// [`ENodeId`] rather than by a position in the class's member order.
    ///
    /// A position goes stale: a caller that enumerates a round's matches
    /// and then applies them one at a time can rebuild the class between
    /// enumeration and application, and `rebuild`'s detach and write-back
    /// renumber it. Re-fetching a stale *index* silently applies the
    /// rule to whichever node inherited that slot, credited to the original
    /// candidate's score and dedup key. Re-resolving a stale *tag* finds
    /// either the same node or nothing at all.
    pub tag: ENodeId,
}

/// The count limits of one run, each named for what it counts.
///
/// What [`Budget`](super::super::optimizer::Budget) resolves to for an input,
/// and what the sweep is held to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    /// Rewrite rounds.
    pub iterations: usize,
    /// The most e-class ids the graph may have minted. Ids are never
    /// reclaimed (`union` repoints a slot, it does not free one), so this
    /// bounds memory, and it is what [`EGraph::num_classes`] reports.
    pub minted: usize,
    /// Rule applications, if capped. A **delta** from the runner's
    /// cumulative [`Saturation::application_count`] when the run starts.
    pub applications: Option<u64>,
}

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
/// sharing its [`CandidateKey`](crate::egraph::candidate::CandidateKey) — the same
/// key the guided loop dedups on — so no re-derivation by that route can put
/// the node back. The gap between the two Δs is this program's measurement
/// of leave-one-out's confluence blindness
/// (docs/archive/plans/2026-09-01-guide-return-to-go.md §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskScope {
    /// Skip only the application that would be assigned `skip_ordinal`.
    Single,
    /// Skip that application and every later one with the same
    /// [`CandidateKey`](crate::egraph::candidate::CandidateKey), for the rest of the
    /// run.
    AllMatchingCandidate,
}

/// Names the application to skip in a counterfactual replay, and how far the
/// skip reaches — see [`Optimizer::mask`](crate::egraph::optimizer::Optimizer::mask)
/// and docs/archive/plans/2026-09-01-guide-return-to-go.md §4.1.
///
/// `skip_ordinal` counts in [`Saturation::application_count`]'s currency, the
/// unconditional budget denominator — the same numbering
/// [`ApplicationId`](crate::egraph::provenance::ApplicationId) uses for a run that
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
    captured: Option<crate::egraph::candidate::CandidateKey>,
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
    /// [`CandidateKey`](crate::egraph::candidate::CandidateKey).
    #[must_use]
    pub fn all_matching_candidate(skip_ordinal: u64) -> Self {
        Self {
            skip_ordinal,
            scope: MaskScope::AllMatchingCandidate,
            captured: None,
        }
    }
}

/// What [`Saturation::mask_decision`] concluded about the application about to
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

/// What one rule's scan-and-commit did: the merges it made, and how the scan
/// ended.
pub(crate) struct ApplyResult {
    pub(crate) changes: usize,
    /// How the scan ended. `changes == 0` is quiescence only when this is
    /// [`ScanStop::Completed`]; under either budget the rest of the graph
    /// was never looked at, so nothing is known about it.
    pub(crate) scan: ScanStop,
}

/// How a single rule's scan over the e-graph ended.
///
/// Two budgets can cut a scan short and they are *different facts*, so this
/// is a three-valued type rather than a `truncated: bool` the caller has to
/// disambiguate afterwards by asking which budget looks closer — that
/// inference is exactly what a stop reason exists to replace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScanStop {
    /// Every e-class was visited and every match committed.
    Completed,
    /// The minted-id budget stopped the scan: the ids the graph has minted
    /// (plus the ones the pending actions would mint) reached the cap.
    MintedCap,
    /// The wall-clock deadline elapsed before the scan finished — either it
    /// cut the walk short, or it passed while a rule's `apply` (or the
    /// commit that follows) was running. Both mean the same thing: this
    /// scan did not finish inside the ceiling.
    Deadline,
    /// The rule-application budget was reached mid-scan.
    ApplicationBudget,
}

/// Why a [`Saturation::saturate_with_limits`] call stopped — an explicit stop
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
    /// The minted-id budget ([`Limits::minted`]) stopped the run (memory
    /// protection) — either the count exceeded it outright, or a sweep
    /// produced zero unions only because every remaining action was
    /// discarded for budget.
    MintedCap,
    /// `max_iters` sweeps completed without any other condition firing.
    IterationCeiling,
    /// The wall-clock safety ceiling elapsed. Offline measurement callers
    /// should treat this as a hard error (fail loud), never as data.
    ///
    /// [`Saturation::saturate_budgeted`] can never report this: it takes no
    /// clock, which is what makes it deterministic.
    Timeout,
    /// The rule-application budget ran out. The one budget dimension that
    /// means the same thing to every ordering policy, and so the one a
    /// research arm can hold two policies to.
    ApplicationBudget,
}

/// Result of one [`Saturation::saturate_with_limits`] run: how many rounds it
/// took and how many rule applications fired in total, whichever limit
/// (iteration count, minted ids, timeout, or convergence) ended the run.
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
    /// (which conflates a timeout or the minted cap with quiescence).
    pub stop: SaturationStop,
}

/// The ceiling on every minted-id budget: no saturation driver grows the
/// graph past this many e-class ids, whatever cap its caller asks for.
///
/// What it counts is ids ever minted ([`EGraph::num_classes`]), not live
/// classes: `add` is the only thing that grows the count and a union never
/// gives an id back, so it is a bound on memory and it only ever rises.
///
/// This is memory protection for call sites that pass no meaningful cap
/// (`usize::MAX`, or a number derived from an unbounded input). It lives
/// here — at the growth decision — and NOT inside [`EGraph::add`], because
/// `add` is the homomorphism onto the semantic quotient (law L1,
/// `docs/archive/plans/2026-09-02-optimizer-api.md` §1): a budget that stops `add`
/// must hand the caller something that is not the class of the node it
/// asked for, and there is no such thing. A budget that stops the *driver*
/// leaves every class the graph does hold correct, which is what
/// truncation has always meant here.
///
/// Twice the classical cap's ceiling (`CLASSICAL_CLASS_CEILING`, 50,000),
/// and well above the floors every shipped kernel keeps (500/2,000/5,000, see
/// `SaturationConfig`), so no shipping configuration meets it.
///
/// It bounds the graph *approximately*: a sweep predicts the ids each pending
/// action will mint against the graph as it stands (`EGraph::predicted_growth`;
/// exact for one action, and for several neither an upper nor a lower bound:
/// shared nodes over-count, and an earlier commit's union can turn a
/// predicted hit into a miss) and
/// then commits the batch it accepted, so the bound holds to the prediction's
/// slack. That is precisely why the limit can live at the driver — it guards
/// memory, not meaning, and a bounded overshoot of a 100 000-id ceiling costs
/// memory nobody was going to miss. The old in-`add` limit bought exactness by returning
/// `EClassId(0)`, a false class id that the `Create` handler then unioned
/// against the match — asserting an equality that does not hold (#1105).
pub const HARD_MINTED_LIMIT: usize = 100_000;

/// How many worklist items to rebuild after a rule that merged something.
///
/// Higher = more deduplication, slower per rule. Lower = less deduplication,
/// faster per rule but classes grow. 256 balances the two.
const PARTIAL_REBUILD_ITEMS: usize = 256;

/// The limits one run is held to, resolved against the runner's counters
/// when the run starts. Passed down the scan rather than parked on the
/// graph: it is a property of the run, not of the graph.
#[derive(Clone, Copy)]
struct Ceilings {
    /// Already clamped to [`HARD_MINTED_LIMIT`].
    minted: usize,
    /// The cumulative application count at which the run stops.
    applications: Option<u64>,
    deadline: Option<Instant>,
}

impl Ceilings {
    /// Only the id ceiling, for a scan driven outside a run.
    #[cfg(test)]
    fn minted(minted: usize) -> Self {
        Self {
            minted: minted.min(HARD_MINTED_LIMIT),
            applications: None,
            deadline: None,
        }
    }

    fn expired(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() > deadline)
    }
}

/// An e-graph being saturated: the store, the rule set that sweeps it, and
/// the runner state denominated in this graph's history.
///
/// Dereferences to the [`EGraph`] it wraps, so everything the store offers —
/// `add`, `find`, `nodes`, extraction over `&EGraph` — is available on the
/// pair; what the runner owns is reachable only here. Replacing the store
/// through `DerefMut` (`*saturation = other`, or restoring a clone) is
/// unsupported: the runner's watermarks are denominated in the graph they
/// were taken against.
pub struct Saturation {
    graph: EGraph,
    rules: RuleSet,
    /// `swept[rule]` is the [`EGraph::mutation_counter`] value captured just
    /// BEFORE that rule's most recent *complete* scan of the live classes
    /// began. See [`EGraph::class_is_dirty`] for how it is used and why the
    /// value is taken from the scan's start, not its end.
    swept: Vec<u64>,
    /// Rule applications performed, counted unconditionally.
    ///
    /// Independent of the provenance journal on purpose: an application budget
    /// must be enforceable whether or not anyone is observing, and reading
    /// the budget off `provenance().recorded_count()` would make recording a
    /// load-bearing part of saturation rather than an observation of it.
    applications: u64,
    /// Cumulative rule-match ATTEMPTS across this graph's whole lifetime —
    /// every node a rule's [`Rewrite::apply`](crate::egraph::Rewrite::apply) was called on, matched or not
    /// (a node outside the rule's [`Rewrite::root_shapes`](crate::egraph::Rewrite::root_shapes) is not an attempt:
    /// the rule is never asked), summed at the one site every scan
    /// funnels through.
    ///
    /// This is the matches enumerated
    /// (docs/archive/plans/2026-09-01-phase3-round2-registration-v2.md §7.1), counted
    /// as `apply` calls. The Guide-cost-flatness precondition needs it
    /// denominated per application, and per application needs it cumulative
    /// across the resumed budget calls an anytime curve makes, the same way
    /// [`Self::application_count`] is.
    total_evals: usize,
    /// Matches found per rule, keyed by identity rather than by family name.
    /// A name-keyed map collapsed all four `Commutative` instances into one
    /// bucket, so every per-rule report built on it was wrong by aggregation.
    match_counts: HashMap<RuleId, usize>,
    /// Counterfactual-replay mask (docs/archive/plans/2026-09-01-guide-return-to-go.md
    /// §4.1): when `Some`, the application that would otherwise be assigned
    /// this ordinal is skipped entirely by [`Self::commit`] — not counted,
    /// not recorded, not applied — and the ordinal it would have consumed is
    /// left for whatever candidate comes next in the same scan order.
    /// Installed only by [`Optimizer::mask`](super::super::optimizer::Optimizer::mask),
    /// so every other path is unaffected.
    mask: Option<ApplicationMask>,
    /// How many applications the most recent masked run actually skipped.
    /// Reset when a mask is installed and read back by
    /// [`Self::last_replay_mask_skips`] — a caller reporting a
    /// confluence-aware Δ must be able to distinguish "the mask matched
    /// once" from "the mask matched eleven times", and must never have to
    /// infer that a mask fired at all.
    mask_skips: usize,
    /// The growth predictor's table of nodes a prediction has minted so far
    /// ([`EGraph::predicted_growth`] empties it before each use). Kept here so
    /// its allocation serves every prediction of the run.
    predictor_pending: HashMap<ENode, EClassId>,
    /// The actions a scan has accepted and not yet committed. Emptied by
    /// the commit, so its allocation serves every sweep of the run.
    updates: Vec<(EClassId, RewriteAction)>,
    /// Per-application growth telemetry (`docs/archive/plans/2026-08-31-guide-design-revision.md`
    /// §4.1): `None` until [`Self::enable_growth_telemetry`] turns it on.
    /// Compiled in only under `saturation-telemetry`, and checked with a bare
    /// `is_some()` on every committed application even then — see
    /// [`crate::egraph::growth`]'s module doc for the "costs nothing when
    /// off" contract this field exists to keep.
    #[cfg(feature = "saturation-telemetry")]
    growth: Option<crate::egraph::growth::GrowthTelemetry>,
}

impl Deref for Saturation {
    type Target = EGraph;

    fn deref(&self) -> &EGraph {
        &self.graph
    }
}

impl DerefMut for Saturation {
    fn deref_mut(&mut self) -> &mut EGraph {
        &mut self.graph
    }
}

impl Saturation {
    /// An empty graph, to be swept by `rules`.
    ///
    /// The mutation log is kept (with `provenance-journal`), so every direct
    /// caller keeps what it had; a caller that wants none says so with
    /// [`Self::without_journal`] before inserting anything.
    #[must_use]
    pub fn over(rules: RuleSet) -> Self {
        Self {
            graph: EGraph::new(),
            swept: vec![0; rules.len()],
            rules,
            applications: 0,
            total_evals: 0,
            match_counts: HashMap::default(),
            mask: None,
            mask_skips: 0,
            predictor_pending: HashMap::default(),
            updates: Vec::new(),
            #[cfg(feature = "saturation-telemetry")]
            growth: None,
        }
    }

    /// The same, with the graph's mutation log turned off.
    ///
    /// Chosen when the graph is built, because the log is not an annotation
    /// that can be switched mid-run: an origin or a union recorded without
    /// the application that caused it reads as a seed or a congruence merge,
    /// which is a wrong answer rather than a missing one. Off, nothing is
    /// recorded and there is no journal to query
    /// ([`EGraph::provenance`] is `None`).
    #[must_use]
    pub fn without_journal(mut self) -> Self {
        self.graph = self.graph.without_journal();
        self
    }

    /// The store, dropping the runner. A cache of saturated graphs keeps
    /// `into_graph().freeze()`, which drops the store's saturation-only state too.
    #[must_use]
    pub fn into_graph(self) -> EGraph {
        self.graph
    }

    /// The rule set sweeping this graph.
    #[must_use]
    pub fn rules(&self) -> &RuleSet {
        &self.rules
    }

    /// Rule applications performed since this graph was created.
    ///
    /// Counted whether or not the journal is kept — the budget and the
    /// observation are separate concerns, and conflating them is what made
    /// recording impossible to turn off.
    #[must_use]
    pub fn application_count(&self) -> u64 {
        self.applications
    }

    /// Cumulative rule-match attempts across this graph's whole lifetime
    /// (see the `total_evals` field doc) — the `apply`-calls counter §7.1's
    /// Guide-overhead-flatness measurement denominates. It excludes nodes
    /// outside a rule's [`Rewrite::root_shapes`](crate::egraph::Rewrite::root_shapes), so curves recorded before
    /// root shapes existed do not compare with it directly.
    #[must_use]
    pub const fn total_evals(&self) -> usize {
        self.total_evals
    }

    /// Matches found per rule since the last call, then reset.
    ///
    /// Taking rather than borrowing: the counts are a per-run reading, and a
    /// reader that wants a delta must be able to zero them without reaching
    /// into the runner.
    #[must_use]
    pub fn take_match_counts(&mut self) -> HashMap<RuleId, usize> {
        std::mem::take(&mut self.match_counts)
    }

    /// How many applications the most recent installed mask skipped. `0`
    /// when no mask was installed, or when the mask's ordinal was never
    /// reached — a caller reporting Δ reads this rather than assuming.
    #[must_use]
    pub const fn last_replay_mask_skips(&self) -> usize {
        self.mask_skips
    }

    /// Install (or clear) the counterfactual-replay mask, resetting the skip
    /// counter. Crate-private on purpose: the mask is a *policy*, in exactly
    /// the sense a `Reranker` and an `Observer` are, and it reaches the
    /// runner through [`Optimizer::mask`](super::super::optimizer::Optimizer::mask)
    /// rather than through a second public saturation entry point.
    pub(crate) fn set_replay_mask(&mut self, mask: Option<ApplicationMask>) {
        self.mask = mask;
        self.mask_skips = 0;
    }

    /// Turn on per-application growth telemetry: how many e-nodes/e-classes
    /// each committed rewrite application actually adds, per rule (see
    /// `crate::egraph::growth`). Off by default and free to leave off — this
    /// is the one call that starts paying for it.
    #[cfg(feature = "saturation-telemetry")]
    pub fn enable_growth_telemetry(&mut self) {
        self.growth = Some(crate::egraph::growth::GrowthTelemetry::default());
    }

    /// The growth telemetry collected so far, if
    /// [`Self::enable_growth_telemetry`] was called on this graph.
    #[cfg(feature = "saturation-telemetry")]
    #[must_use]
    pub fn growth_telemetry(&self) -> Option<&crate::egraph::growth::GrowthTelemetry> {
        self.growth.as_ref()
    }

    /// Render a human-readable derivation trace for the given ancestry set
    /// (from [`EGraph::derivation_ancestors`]), resolving rule names through
    /// the rule set that ran.
    #[cfg(feature = "provenance-journal")]
    #[must_use]
    pub fn format_derivation_trace(
        &self,
        ancestors: &std::collections::BTreeSet<crate::egraph::provenance::ApplicationId>,
    ) -> String {
        let rule_name =
            |idx: RuleIdx| -> Option<String> { Some(self.rules.rule(idx).name().to_string()) };
        crate::egraph::provenance::format_derivation_trace(
            self.graph.journal(),
            ancestors,
            &rule_name,
        )
    }

    /// Find all actual rewrite matches in the E-graph.
    ///
    /// Returns only targets where the rule actually matches (produces an action).
    /// Much more efficient than enumerating all combinations - only scores real matches.
    #[must_use]
    pub fn find_rewrite_matches(&self) -> Vec<RewriteTarget> {
        let mut matches = Vec::new();

        for rule_idx in self.rules.indices() {
            let rule = self.rules.rule(rule_idx);
            for class_id in self.graph.class_ids() {
                for (node, tag) in self.graph.nodes(class_id).zip(self.graph.tags(class_id)) {
                    // Check if rule matches this node
                    if rule.apply(&self.graph, class_id, node).is_some() {
                        matches.push(RewriteTarget {
                            rule_idx,
                            class_id,
                            tag,
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
    /// Unlike the sweeps, this takes no minted-id budget: one call applies at
    /// most one action, so it cannot itself grow the graph without bound —
    /// the loop that calls it can, and that loop owns the budget. Reaching
    /// [`HARD_MINTED_LIMIT`] here therefore means a caller is driving
    /// rewrites with no budget at all, and it **panics** rather than
    /// returning `false`: `false` already means "the rule did not fire",
    /// and a budget exhaustion that is indistinguishable from a non-match
    /// is the silent failure this ceiling exists to prevent. No production
    /// path calls this, and the production caps stay below the limit (the
    /// classical ceiling is half of it).
    pub fn apply_single_rule(
        &mut self,
        rule_idx: RuleIdx,
        class_id: EClassId,
        tag: ENodeId,
    ) -> bool {
        assert!(
            self.graph.num_classes() < HARD_MINTED_LIMIT,
            "apply_single_rule: e-graph is at the hard minted-id limit \
             ({} ids, limit {HARD_MINTED_LIMIT}) — this entry point applies \
             one action per call and carries no budget, so the caller's \
             rewrite loop must impose one (see `saturate_with_limits`)",
            self.graph.num_classes(),
        );

        let class_id = self.graph.find(class_id);
        // Resolve the stable tag against the class as it stands NOW — a
        // rebuild since the caller enumerated this target either left the
        // node in place or removed it, and `None` is the honest answer for
        // the second case. See `RewriteTarget::tag`.
        let Some(node) = self.graph.node_for_tag(class_id, tag).cloned() else {
            return false;
        };

        let Some(action) = self
            .rules
            .rule(rule_idx)
            .apply(&self.graph, class_id, &node)
        else {
            return false;
        };

        // Guided search calls this once per discrete rewrite decision — the
        // step's granularity here is "one apply_single_rule call", mirroring
        // "one saturate_with_limits iteration" for the batched path.
        self.graph.begin_step();

        // The batched path already knows how to execute every action; reuse
        // it so the single-step path can't drift out of sync.
        let changed = self.commit(rule_idx, class_id, action) > 0;

        if changed {
            self.graph.rebuild();
        }
        changed
    }

    /// The one rewrite-until-budget-exhausted loop, under a wall-clock
    /// deadline.
    ///
    /// Every saturation in the workspace bottoms out in the private `saturate`,
    /// and nothing else re-decides when a run stops. Callers reach it through
    /// one of two entry points:
    ///
    /// - [`Optimizer::run`](super::super::optimizer::Optimizer::run) — the
    ///   production path, through [`Self::saturate_budgeted`]. It picks its
    ///   limits from a [`Budget`](super::super::optimizer::Budget), so the
    ///   budget scales with the input and carries no clock. All three
    ///   production tiers use it.
    /// - This method, called directly only where a budget must be pinned
    ///   rather than inherited — tests, the hindsight labeler, and
    ///   measurement harnesses, which spell it
    ///   [`SaturationConfig::compatibility`](super::super::saturate::SaturationConfig::compatibility).
    ///
    /// The arguments are independent stopping conditions, checked between
    /// rewrite rounds; the run returns as soon as any one of them — or
    /// saturation itself — is reached, whichever comes first:
    ///
    /// - `max_iters` — how many rewrite rounds may run. One round applies
    ///   every rule once and rebuilds once.
    /// - `max_minted` — e-class id ceiling. Bounds memory against e-graph
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
        max_minted: usize,
        timeout: Duration,
    ) -> SaturationStats {
        let limits = Limits {
            iterations: max_iters,
            minted: max_minted,
            applications: None,
        };
        self.saturate(limits, Some(timeout))
    }

    /// Saturate under deterministic limits only — rounds, minted ids, and
    /// optionally rule applications — with no clock.
    ///
    /// This is what [`super::super::optimizer::Optimizer`] drives, and the
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
    pub fn saturate_budgeted(&mut self, limits: Limits) -> SaturationStats {
        self.saturate(limits, None)
    }

    /// The one rewrite-until-budget-exhausted loop. Every saturation entry
    /// point in this crate funnels here rather than re-deciding, in a second
    /// copy, when to stop.
    fn saturate(&mut self, limits: Limits, timeout: Option<Duration>) -> SaturationStats {
        // Growth is decided here and in `sweep_rule`, so the ceiling is
        // applied here: a caller asking for `usize::MAX` ids gets
        // `HARD_MINTED_LIMIT` and a truthful `MintedCap` stop, not a graph
        // that grows until `add` starts refusing to insert. Applied in the
        // shared loop rather than in `saturate_with_limits`, so
        // `saturate_budgeted` — and therefore every production tier — is held
        // to it too.
        let max_minted = limits.minted.min(HARD_MINTED_LIMIT);
        let start = Instant::now();
        // The cap is enforced deep inside the scan, where an application is
        // about to commit — the only place that can stop mid-round without
        // letting the round decide how far past the budget to go.
        let ceilings = Ceilings {
            minted: max_minted,
            applications: limits
                .applications
                .map(|n| self.applications.saturating_add(n)),
            // `Instant + Duration::MAX` panics, so the deadline is optional
            // rather than "infinitely far away".
            deadline: timeout.map(|t| start + t),
        };
        let mut iterations = 0;
        let mut total_unions = 0;
        // Recorded at the point the loop decides to stop — never inferred
        // afterwards from the counters.
        let mut stop = SaturationStop::IterationCeiling;

        for _ in 0..limits.iterations {
            if let Some(t) = timeout
                && start.elapsed() >= t
            {
                stop = SaturationStop::Timeout;
                break;
            }
            if self.graph.num_classes() > max_minted {
                stop = SaturationStop::MintedCap;
                break;
            }
            if ceilings
                .applications
                .is_some_and(|cap| self.applications >= cap)
            {
                stop = SaturationStop::ApplicationBudget;
                break;
            }
            iterations += 1;

            // Advance the journal's step once per saturation iteration —
            // every record produced by this iteration's rule applications
            // shares this step.
            self.graph.begin_step();

            // Every rule once, in the set's order, then one rebuild: a rule's
            // commit is followed by a partial rebuild of the unions it made
            // (so classes stay approximately deduplicated for the next rule's
            // scan), and the sweep ends with a full one.
            let (unions, sweep) = self.sweep(&ceilings);
            total_unions += unions;
            // The union count and the stop reason are independent facts: a
            // sweep that a budget cut short classifies as that budget whether
            // or not it also committed unions. Consulting `truncated` only on
            // the `unions == 0` path missed every truncated-but-productive
            // sweep, and if that was the last allowed iteration the run fell
            // through to this loop's default `IterationCeiling`.
            //
            // `sweep_rule` holds the minted count at or under `minted` by
            // truncating its own scan, so the cheap check at the top of this
            // loop almost never sees a capped run — the sweep's own report is
            // what makes `MintedCap` observable at all.
            match sweep {
                ScanStop::MintedCap => {
                    stop = SaturationStop::MintedCap;
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

        SaturationStats {
            iterations,
            total_unions,
            stop,
        }
    }

    /// One round: every rule scanned and committed in turn, rule-major.
    fn sweep(&mut self, ceilings: &Ceilings) -> (usize, ScanStop) {
        let mut any_changes = false;
        let mut total = 0;
        let mut sweep = ScanStop::Completed;
        for rule_idx in self.rules.indices() {
            if self.graph.num_classes() > ceilings.minted {
                sweep = ScanStop::MintedCap;
                break;
            }
            let result = self.sweep_rule(rule_idx, ceilings);
            if result.changes > 0 {
                any_changes = true;
                // Interleaved partial rebuild: process some worklist items to
                // keep classes small. The chunk size bounds work per rule.
                self.graph.rebuild_budgeted(PARTIAL_REBUILD_ITEMS);
            }
            total += result.changes;
            match result.scan {
                ScanStop::Completed => {}
                // The id budget cut this rule short. A later rule may still
                // fit inside the cap, so the sweep goes on — but it is no
                // longer a full sweep, and can never read as quiescence.
                ScanStop::MintedCap => sweep = ScanStop::MintedCap,
                // The wall-clock ceiling is hard: do not start another
                // rule's scan on the far side of it.
                ScanStop::Deadline => {
                    sweep = ScanStop::Deadline;
                    break;
                }
                // The application budget is spent for the whole run, not
                // just this rule: no later rule can commit anything either.
                ScanStop::ApplicationBudget => {
                    sweep = ScanStop::ApplicationBudget;
                    break;
                }
            }
        }
        if any_changes {
            self.graph.rebuild();
        }
        (total, sweep)
    }

    /// Whether a run's application allowance is spent, counting `queued`
    /// applications a scan has accepted but not yet committed.
    fn allowance_spent(&self, ceilings: &Ceilings, queued: usize) -> bool {
        ceilings
            .applications
            .is_some_and(|cap| self.applications + queued as u64 >= cap)
    }

    /// The stop a scan is already at before it visits a class: the id budget
    /// or the application allowance spent. Neither input changes until the
    /// scan queues an action, so one look at entry stands for every class.
    /// A graph with no live class has nothing to scan, so it completes.
    fn scan_entry_stop(&self, ceilings: &Ceilings) -> ScanStop {
        if self.graph.class_ids().next().is_none() {
            return ScanStop::Completed;
        }
        if self.graph.num_classes() > ceilings.minted {
            return ScanStop::MintedCap;
        }
        if self.allowance_spent(ceilings, 0) {
            return ScanStop::ApplicationBudget;
        }
        ScanStop::Completed
    }

    /// Scan the live classes that can root one rule
    /// ([`EGraph::candidate_classes`]), then commit what it found.
    ///
    /// Stops if the id budget, the application allowance or the deadline is
    /// reached, and says which in [`ApplyResult::scan`].
    fn sweep_rule(&mut self, rule_idx: RuleIdx, ceilings: &Ceilings) -> ApplyResult {
        let max_nodes = ceilings.minted;

        // `Instant::now()` costs tens of nanoseconds — the same order as the
        // cheapest rule's `apply` — so polling it once per node would be a
        // measurable tax on every scan. It is polled instead at the three
        // places that make the ceiling hard without that tax: once per
        // visited e-class (a clock read is lost in a class's worth of rule
        // evaluations), every `DEADLINE_POLL_NODES` nodes
        // within a class (so one enormous e-class cannot run unbounded), and
        // once after the scan and its commit (so a deadline that elapsed
        // inside an opaque `apply` is still seen). The bound that buys is an
        // overrun of at most `DEADLINE_POLL_NODES` rule evaluations plus one
        // `apply`; the timeout is a fail-loud safety ceiling, not a
        // scheduling knob, so shrinking that bound further buys nothing worth
        // a clock read per node.
        const DEADLINE_POLL_NODES: usize = 256;

        let mut unions = 0;
        let mut evals = 0usize;
        // Taken, not borrowed: the commit below needs `&mut self`.
        let mut updates = std::mem::take(&mut self.updates);
        let mut estimated_new_nodes: usize = 0;
        let mut scan = self.scan_entry_stop(ceilings);

        // Captured before the scan touches anything: `class_is_dirty` below
        // compares every candidate class against `swept` (this rule's OWN
        // last complete-scan baseline, from a previous call), and — only if
        // this scan itself runs to completion — `swept[rule_idx]` is advanced
        // to `sweep_start` at the end. Taking the *start* value rather than
        // the counter's value after the scan means a mutation this very call
        // makes (via the commit loop's unions below) is correctly seen as new
        // next time this rule runs, without needing to re-examine it against
        // itself right now.
        let sweep_start = self.graph.mutation_counter;
        let swept = self.swept[rule_idx.get()];
        // A watermark is a `mutation_counter` of this graph, so it cannot
        // lead the graph's own counter. It would if the store were swapped
        // out from under the runner through `DerefMut` (`*saturation = ..`,
        // or restoring a clone), after which `class_is_dirty` would read
        // changed classes as clean and saturation would stop short.
        debug_assert!(
            swept <= sweep_start,
            "a sweep watermark ({swept}) is ahead of the graph's mutation counter \
             ({sweep_start}): the store was replaced under the runner"
        );

        // What the rule can match at a root, asked once: the node loop below
        // never hands it a node it would have declined, and the class loop
        // never pays the dirtiness walk for a class holding no such node.
        let rule = self.rules.rule(rule_idx);
        let roots = rule.root_shapes();

        // Only a class that has held a node in the rule's root shapes is
        // visited at all (`candidate_classes`). The id-budget and allowance
        // checks run once at entry (`scan_entry_stop`) and after each action
        // below, the only places their inputs change; the deadline can elapse
        // anywhere, so it is checked per visited class.
        let classes = match scan {
            ScanStop::Completed => Some(self.graph.candidate_classes(roots)),
            _ => None,
        };
        'scan: for canonical in classes.into_iter().flatten() {
            if ceilings.expired() {
                scan = ScanStop::Deadline;
                break;
            }
            // Nothing within DIRTY_TRACKING_MAX_DEPTH hops of `canonical`
            // has changed since this rule last swept the whole graph, so
            // this rule's answer on `canonical` cannot have changed either
            // — skip straight past the clone and every `apply` call below.
            if !self.graph.class_is_dirty(canonical, swept) {
                continue;
            }

            // Borrowed, not cloned: the scan only reads the graph (a rule's
            // `apply` takes `&EGraph`, and every write waits for the commit
            // below), so nothing can move the class's nodes under the loop.
            // The kind index is a superset, so a stale bit costs a visit to a
            // class whose nodes `admits` all decline here.
            for node in self.graph.nodes_of_canonical(canonical) {
                if !roots.admits(node) {
                    continue;
                }
                evals += 1;
                if evals.is_multiple_of(DEADLINE_POLL_NODES) && ceilings.expired() {
                    scan = ScanStop::Deadline;
                    break 'scan;
                }
                if let Some(action) = rule.apply(&self.graph, canonical, node) {
                    // Exactly how many e-classes this action would add,
                    // against the graph as it stands right now — see
                    // `predicted_growth`, whose exactness is asserted on
                    // every application under growth telemetry. This
                    // replaced a flat 0/1/3 guess that was so conservative
                    // saturation stopped ~40% short of its own class budget
                    // (circle_sdf 1218 of 2000, redundant 428 of 500).
                    //
                    // Spending the whole budget moved extraction *both* ways
                    // on the shipped corpus: four kernels better, four worse
                    // (mandelbrot_distance +7.3%, smooth_min_scene +6.0%).
                    // That is not a reason to saturate less. A richer
                    // equivalence class cannot make the true optimum worse;
                    // only an extractor that is not monotone in what it is
                    // given can be led to a worse answer by more choices.
                    // So the estimator tells the truth and the extractor
                    // owns the regression — decision JP, 2026-09-09; see
                    // docs/archive/plans/2026-08-31-guide-design-revision.md §4.2.
                    //
                    // Exact for this action against the graph as it stands.
                    // The running sum is not a bound on the scan's real
                    // growth in either direction (see `predicted_growth`), so
                    // the cap check below holds the batch to the prediction's
                    // slack, which is what `HARD_MINTED_LIMIT` allows.
                    let action_cost = self
                        .graph
                        .predicted_growth(&action, &mut self.predictor_pending);
                    estimated_new_nodes += action_cost;

                    // If this action would push us over budget, stop scanning
                    if self.graph.num_classes() + estimated_new_nodes > max_nodes {
                        // Don't add this action — discard it and stop
                        scan = ScanStop::MintedCap;
                        break 'scan;
                    }

                    updates.push((canonical, action));
                    if self.allowance_spent(ceilings, updates.len()) {
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
        // cut-short scan leaves `swept[rule_idx]` at its old value, so next
        // call simply sees more as dirty than strictly necessary — safe, just
        // less tight.
        if scan == ScanStop::Completed {
            self.swept[rule_idx.get()] = sweep_start;
        }

        // Every queued action is one match, so this scan's matches are
        // `updates`' length — counted once, after the scan, which is what
        // lets the scan borrow the graph rather than clone each class.
        if !updates.is_empty() {
            *self
                .match_counts
                .entry(self.rules.id(rule_idx))
                .or_insert(0) += updates.len();
        }

        // Commit: all actions in the log are within budget.
        // Do NOT rebuild here — the caller rebuilds after the rule (a partial
        // one) and after the round (the full one).
        for (class_id, action) in updates.drain(..) {
            if self.allowance_spent(ceilings, 0) {
                scan = ScanStop::ApplicationBudget;
                break;
            }
            unions += self.commit(rule_idx, class_id, action);
        }
        self.updates = updates;

        // A deadline that elapsed inside a rule's `apply` or inside the commit
        // above reached none of the checks in the walk. Left as `Completed`,
        // a sweep that blew the hard ceiling would be recorded as quiescence.
        if scan == ScanStop::Completed && ceilings.expired() {
            scan = ScanStop::Deadline;
        }

        self.total_evals += evals;
        ApplyResult {
            changes: unions,
            scan,
        }
    }

    /// Decide what the installed replay mask (if any) says about the
    /// application about to be committed. Split out of [`Self::commit`]
    /// because the `AllMatchingCandidate` arm needs to READ the graph
    /// (`ClassContentKey::of`) while the mask is borrowed, and then WRITE the
    /// mask — two borrows that cannot overlap.
    fn mask_decision(&self, rule_idx: RuleIdx, class_id: EClassId) -> MaskDecision {
        let Some(mask) = &self.mask else {
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
                let same_rule = self.rules.id(rule_idx) == key.rule;
                if same_rule
                    && key.content
                        == crate::egraph::candidate::ClassContentKey::of(
                            self.graph.as_frozen(),
                            self.graph.find(class_id),
                        )
                {
                    MaskDecision::Skip
                } else {
                    MaskDecision::Proceed
                }
            }
        }
    }

    /// Commit a rewrite action on behalf of a specific rule, attributing
    /// every e-node created and every union performed while executing it to
    /// one [`ApplicationId`](crate::egraph::provenance::ApplicationId) when
    /// the graph keeps a journal.
    ///
    /// The sole entry point into `apply_action` from rule-driven call sites
    /// ([`Self::apply_single_rule`], [`Self::sweep_rule`]) — it exists so
    /// attribution can't drift out of sync with the actual rewrite dispatch:
    /// every caller that knows a rule funnels through here instead of calling
    /// `apply_action` directly.
    ///
    /// Records one application up front (even if the action turns out to
    /// produce no net change, e.g. a `Union` with an already-equal target) —
    /// the record's cost is a single `Vec::push`, and recording
    /// unconditionally keeps `step` bookkeeping simple. The match root is the
    /// class the rule matched against, i.e. `class_id` as passed in (cheap:
    /// already in hand at the call site).
    fn commit(&mut self, rule_idx: RuleIdx, class_id: EClassId, action: RewriteAction) -> usize {
        // Counterfactual replay (docs/archive/plans/2026-09-01-guide-return-to-go.md
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
                self.mask_skips += 1;
                return 0;
            }
            MaskDecision::SkipAndCapture => {
                let key = crate::egraph::candidate::CandidateKey {
                    rule: self.rules.id(rule_idx),
                    content: crate::egraph::candidate::ClassContentKey::of(
                        self.graph.as_frozen(),
                        self.graph.find(class_id),
                    ),
                };
                let mask = self
                    .mask
                    .as_mut()
                    .expect("mask_decision returned SkipAndCapture with no mask installed");
                mask.captured = Some(key);
                self.mask_skips += 1;
                return 0;
            }
            MaskDecision::SkipAndConsume => {
                self.mask = None;
                self.mask_skips += 1;
                return 0;
            }
        }

        // Counted unconditionally: the budget must not depend on whether
        // anyone is watching.
        self.applications += 1;

        #[cfg(feature = "provenance-journal")]
        {
            let minted_from = self.graph.num_classes() as u64;
            let opening = Opening {
                rule: self.rules.id(rule_idx),
                rule_idx,
                match_root: class_id,
                minted_from,
            };
            let open = self
                .graph
                .journal
                .as_mut()
                .map(|journal| journal.open_application(opening));
            let result = self.apply_measured(rule_idx, class_id, action);
            let minted = minted_from..(self.graph.num_classes() as u64);
            if let (Some(open), Some(journal)) = (open, self.graph.journal.as_mut()) {
                journal.close_application(open, minted, result);
            }
            result
        }

        #[cfg(not(feature = "provenance-journal"))]
        {
            self.apply_measured(rule_idx, class_id, action)
        }
    }

    /// Apply a rewrite action on behalf of `rule_idx`, threading through
    /// per-application growth telemetry when it is compiled in and switched
    /// on ([`Self::enable_growth_telemetry`]). Behaviorally identical to a
    /// bare `self.graph.apply_action(class_id, action)` call in every other
    /// case — with `saturation-telemetry` off this compiles to exactly that
    /// call, and with it on but unused the only added cost is one
    /// `Option::is_some()` check. See [`crate::egraph::growth`]'s module doc.
    #[cfg_attr(not(feature = "saturation-telemetry"), allow(unused_variables))]
    fn apply_measured(
        &mut self,
        rule_idx: RuleIdx,
        class_id: EClassId,
        action: RewriteAction,
    ) -> usize {
        #[cfg(feature = "saturation-telemetry")]
        if let Some(growth) = &mut self.growth {
            // The exactness oracle for `predicted_growth`: computed against
            // the graph exactly as it stands right now, immediately before
            // the same action executes for real, so there is no window in
            // which some other commit could move the memo out from under
            // the comparison. An always-on `assert_eq!`, not `debug_assert!`
            // — this is a measurement path (telemetry is opt-in), not a hot
            // one, and a divergence here is exactly the "most important
            // finding" a mispredicted budget estimator could produce.
            let predicted = self
                .graph
                .predicted_growth(&action, &mut self.predictor_pending);
            let classes_before = self.graph.num_classes();
            let unions = self.graph.apply_action(class_id, action);
            // `add` pushes one node per memo miss and returns its index as
            // the new class: nodes added and classes added are one count.
            let nodes_added = self.graph.num_classes() - classes_before;
            let rule = self.rules.id(rule_idx);
            assert_eq!(
                predicted,
                nodes_added,
                "predicted_growth mismatch on rule {} ({rule:?}): \
                 predicted {predicted} new e-classes, the commit actually \
                 added {nodes_added} — predicted_growth must walk exactly \
                 the same shape apply_action does",
                rule_idx.get(),
            );
            growth.record(rule, nodes_added, unions > 0 || nodes_added > 0);
            return unions;
        }
        self.graph.apply_action(class_id, action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::cost::CostModel;
    use crate::egraph::extract::extract_dag;
    use crate::egraph::optimizer::{Budget, Optimizer};

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

        let mut egraph = Saturation::over(RuleSet::new(vec![Pythagorean::new()]));
        let rule = egraph.rules().indices().next().expect("one rule");

        let x = egraph.add(ENode::Var(0));
        let sin_x = egraph.add(ENode::apply(pixelflow_ir::OpKind::Sin, &[x]));
        let cos_x = egraph.add(ENode::apply(pixelflow_ir::OpKind::Cos, &[x]));
        let cos_sq = egraph.add(ENode::apply(pixelflow_ir::OpKind::Mul, &[cos_x, cos_x]));

        // `p` starts life indistinguishable from any other opaque unary
        // node — NOT unioned with `sin_x` yet.
        let p = egraph.add(ENode::apply(pixelflow_ir::OpKind::Neg, &[x]));
        let p_sq = egraph.add(ENode::apply(pixelflow_ir::OpKind::Mul, &[p, p]));
        let sum = egraph.add(ENode::apply(pixelflow_ir::OpKind::Add, &[p_sq, cos_sq]));

        // First scan: nothing to find (`p` isn't known to be `sin(x)`).
        let result = egraph.sweep_rule(rule, &Ceilings::minted(10_000));
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
        let result = egraph.sweep_rule(rule, &Ceilings::minted(10_000));
        assert_eq!(
            result.changes, 1,
            "dirty-class tracking must not skip `sum` just because its own \
             node list (and its direct child p_sq's) never changed — only \
             the grandchild `p` did, which is exactly Pythagorean's \
             matching depth"
        );
        egraph.rebuild();

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, egraph.find(sum), &costs);
        let cost = dag.total_cost;
        assert_eq!(
            cost, 0,
            "sum's class must now also contain Const(1.0), cost 0"
        );
    }

    /// A `(x*x + y*y).sqrt()`-shaped arena: a real sweep under the full rule
    /// set where idempotent re-fires (`commutative`/`associative`) dominate,
    /// which is exactly the regime [`MaskScope::AllMatchingCandidate`] exists
    /// to see through.
    fn fixture_arena() -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
        let mut arena = pixelflow_ir::ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let xx = arena.push_binary(pixelflow_ir::OpKind::Mul, x, x);
        let yy = arena.push_binary(pixelflow_ir::OpKind::Mul, y, y);
        let sum = arena.push_binary(pixelflow_ir::OpKind::Add, xx, yy);
        let dist = arena.push_unary(pixelflow_ir::OpKind::Sqrt, sum);
        let doubled = arena.push_binary(pixelflow_ir::OpKind::Add, dist, dist);
        let root = arena.push_binary(pixelflow_ir::OpKind::Sub, doubled, doubled);
        (arena, root)
    }

    /// The fixture inserted into a fresh graph built by `opt`.
    fn fixture_under(opt: Optimizer) -> (Optimizer, Saturation, EClassId) {
        let (arena, root) = fixture_arena();
        let mut eg = opt.egraph();
        let root_class =
            crate::egraph::insert(&arena, root, &mut eg, crate::egraph::Vocabulary::Templates)
                .expect("insert into e-graph");
        (opt, eg, root_class)
    }

    fn mask_fixture() -> (Optimizer, Saturation, EClassId) {
        fixture_under(
            Optimizer::production()
                .budget(Budget::Applications(MASK_BUDGET))
                .no_ceiling(),
        )
    }

    const MASK_BUDGET: u64 = 60;

    /// The leave-one-out mask is consumed after it fires: exactly one skip,
    /// no matter how many later applications look like it.
    #[test]
    fn leave_one_out_mask_skips_exactly_one_application() {
        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::leave_one_out(0)));
        opt.run(&mut eg, root);
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
        opt.run(&mut eg, root);
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
        let plain = plain_opt.run(&mut plain_eg, plain_root);

        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::leave_one_out(u64::MAX)));
        let masked = opt.run(&mut eg, root);

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
        let plain = plain_opt.run(&mut plain_eg, plain_root);

        let (opt, mut eg, root) = mask_fixture();
        let mut opt = opt.mask(Some(ApplicationMask::all_matching_candidate(0)));
        let masked = opt.run(&mut eg, root);

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

    /// The journal is chosen when the graph is built, and the budget is not
    /// the journal's: an unobserved graph keeps none, so there is nothing to
    /// query rather than a log whose rule-minted nodes read as seeds, and the
    /// run it makes is the run a journaled graph makes.
    #[cfg(feature = "provenance-journal")]
    #[test]
    fn an_unobserved_graph_keeps_no_journal_and_runs_the_same() {
        let budget = Budget::Applications(MASK_BUDGET);
        let (mut plain, mut plain_eg, plain_root) =
            fixture_under(Optimizer::production().budget(budget).no_ceiling());
        let (mut kept, mut kept_eg, kept_root) = fixture_under(
            Optimizer::production()
                .budget(budget)
                .no_ceiling()
                .observe(Some(Box::new(crate::egraph::KeepJournal))),
        );

        let plain_out = plain.run(&mut plain_eg, plain_root);
        let kept_out = kept.run(&mut kept_eg, kept_root);

        assert!(plain_eg.provenance().is_none(), "no observer, no journal");
        assert!(kept_eg.provenance().is_some(), "an observer keeps one");
        assert_eq!(plain_out.stats.stop, kept_out.stats.stop);
        assert_eq!(plain_out.stats.applications, kept_out.stats.applications);
        assert_eq!(plain_out.stats.unions, kept_out.stats.unions);
        assert_eq!(plain_out.stats.classes, kept_out.stats.classes);
        assert_eq!(plain_out.cost, kept_out.cost);
        assert!(plain_out.stats.applications > 0, "the run must do work");
    }

    /// Everything an application minted or merged is attributed to it. This is
    /// the property a journal that was switched on or off per call lost:
    /// outside the application bracket `add` records a seed and `union` a
    /// congruence merge, so a bracket that did not cover a rule's whole
    /// action would label the rule's own products as inputs.
    #[cfg(feature = "provenance-journal")]
    #[test]
    fn everything_an_application_minted_or_merged_is_attributed_to_it() {
        use crate::egraph::provenance::{ENodeId, Origin};

        let (mut opt, mut eg, root) = fixture_under(
            Optimizer::production()
                .budget(Budget::Applications(MASK_BUDGET))
                .no_ceiling()
                .observe(Some(Box::new(crate::egraph::KeepJournal))),
        );
        opt.run(&mut eg, root);

        let journal = eg.journal();
        let mut minted = 0;
        for (app, record) in journal.applications() {
            for id in record.minted.clone() {
                minted += 1;
                assert_eq!(
                    journal.origin(ENodeId(id)),
                    Some(Origin::Rule(app)),
                    "node {id} was minted inside application {app:?} and must say so"
                );
            }
        }
        assert!(
            minted > 0,
            "the fixture must mint nodes for this to mean anything"
        );

        let mut attributed = 0;
        for event in journal.union_events() {
            assert_eq!(
                event.rule_idx.is_some(),
                event.application_id.is_some(),
                "a union names its rule and its application together, or neither: {event:?}"
            );
            attributed += usize::from(event.application_id.is_some());
        }
        assert!(attributed > 0, "some union must have been a rule's");
    }
}
