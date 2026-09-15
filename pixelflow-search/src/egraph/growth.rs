//! Per-application growth telemetry: how many e-nodes (and, in this
//! representation, e-classes — see below) a rewrite application actually
//! added, not how many its RHS template could add in the worst case.
//!
//! Compiled in only under the `saturation-telemetry` feature, exactly like
//! [`crate::telemetry`] — with the feature off, this module does not exist
//! (see the `#[cfg]` on its declaration in `super::mod`) and there is
//! nothing in the binary that could pay for it. Even with the feature on, an
//! [`EGraph`](super::graph::EGraph) records nothing until
//! [`EGraph::enable_growth_telemetry`](super::graph::EGraph::enable_growth_telemetry)
//! is called: the per-application hook checks `self.growth.is_some()` before
//! doing any work — the same `is_on()`-gated shape
//! `pixelflow-codegen/src/emit/guards.rs`'s `PIXELFLOW_GUARD_TELEMETRY`
//! telemetry uses for a per-node measurement. **It must cost nothing when
//! off**, and this module's only paid work when off is that one
//! `Option::is_some()` check per committed rewrite application.
//!
//! # Why this exists
//!
//! `docs/plans/2026-08-31-guide-design-revision.md` §4.1: before any policy
//! gates a rewrite on "will this grow the graph and give me nothing, or grow
//! it a little and give me something useful", growth has to be *measured*,
//! not guessed. A rewrite's RHS is a template of known size and the e-graph
//! is hash-consed, so the honest, exact measurement is a before/after delta
//! taken around the committed action — not a prediction from the template,
//! which would be a second, unvalidated computation of the same fact.
//!
//! That measured delta is now also the oracle for a *budget-time* estimator:
//! [`EGraph::predicted_growth`](super::graph::EGraph::predicted_growth) walks
//! the same node shape [`EGraph::apply_action`](super::graph::EGraph::apply_action)
//! builds — literally the same code, via the crate-private `NodeSink`
//! abstraction in `graph.rs` — against the memo instead of against a real
//! insert, so it never becomes the second, drifting spelling this doc used to
//! warn against. `EGraph::apply_action_measured` asserts the two agree,
//! unconditionally, on every application recorded while telemetry is on —
//! not `debug_assert!`, because a divergence here is exactly the finding this
//! module exists to surface.
//!
//! # Nodes added and classes added are the same number here
//!
//! [`EGraph::add`](super::graph::EGraph::add) mints exactly one new e-class
//! (a singleton holding the one node just created) alongside exactly one new
//! [`ENodeId`](super::provenance::ENodeId) on every memo miss, and
//! [`EGraph::union`](super::graph::EGraph::union) never allocates a class
//! slot — it only ever repoints a `parent` entry, merging two *existing*
//! slots. So in this representation the e-graph's raw class-slot count and
//! its raw e-node count grow in lockstep, always: a before/after delta
//! recovers `nodes_added` and `classes_added` as the same number by
//! construction, not by coincidence of this measurement. [`RuleGrowth`]
//! therefore keeps one histogram and exposes it under both names; the
//! recording site (`EGraph::apply_action_measured`) pins the invariant with
//! a `debug_assert_eq!`, so a representation change that broke it would fail
//! loudly there before it silently mis-scored anything built on this module.
//!
//! This is itself part of the answer to "how many e-nodes and e-classes did
//! the application add": at the granularity this architecture actually
//! tracks, the two questions collapse into one.

use std::collections::BTreeMap;

use super::rules::RuleId;

/// One rule's growth distribution, accumulated as a histogram rather than a
/// raw sample list — bounded memory (the number of *distinct* per-application
/// node counts is small; every hand-written
/// [`RewriteAction`](super::rewrite::RewriteAction) adds at most a handful of
/// nodes) no matter how many times the rule fires, which matters here: one
/// kernel can drive hundreds of thousands of applications
/// (`docs/plans/2026-08-31-guide-design-revision.md` §2.1's heavy tail).
#[derive(Clone, Debug, Default)]
pub struct RuleGrowth {
    /// Applications recorded for this rule — committed applications, not
    /// match attempts (see [`super::graph::ApplyResult::evals`] for the
    /// attempt count).
    applications: u64,
    /// Applications that changed nothing at all: zero nodes created AND the
    /// closing union did not merge two previously-distinct classes. This is
    /// the idempotent-refire case
    /// `docs/plans/2026-08-31-guide-design-revision.md` §2.2 measures at
    /// 91.1% of all applications.
    no_op: u64,
    /// `nodes_added -> application count`. Doubles as the `classes_added`
    /// histogram — see the module doc for why those coincide here.
    growth_hist: BTreeMap<usize, u64>,
}

impl RuleGrowth {
    fn record(&mut self, nodes_added: usize, changed: bool) {
        self.applications += 1;
        if !changed {
            self.no_op += 1;
        }
        *self.growth_hist.entry(nodes_added).or_insert(0) += 1;
    }

    /// Applications recorded for this rule.
    #[must_use]
    pub fn applications(&self) -> u64 {
        self.applications
    }

    /// Fraction of applications that changed nothing at all (zero nodes
    /// created, no class merge) — the strict "idempotent re-fire" share. See
    /// [`Self::zero_growth_fraction`] for the looser "added zero nodes"
    /// share, which a merge-only application still clears.
    #[must_use]
    pub fn no_op_fraction(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            self.no_op as f64 / self.applications as f64
        }
    }

    /// Fraction of applications that added zero e-nodes. A merge into an
    /// already-fully-present RHS counts as "added zero" here even when the
    /// merge itself changed the graph — [`Self::no_op_fraction`] is the
    /// stricter "changed nothing at all" count.
    #[must_use]
    pub fn zero_growth_fraction(&self) -> f64 {
        if self.applications == 0 {
            return 0.0;
        }
        let zero = self.growth_hist.get(&0).copied().unwrap_or(0);
        zero as f64 / self.applications as f64
    }

    /// Median e-nodes added per application (equivalently, median e-classes
    /// added — see the module doc).
    #[must_use]
    pub fn median_nodes_added(&self) -> usize {
        self.quantile_nodes_added(0.5)
    }

    /// Same distribution as [`Self::median_nodes_added`]; named separately
    /// because the measurement this module answers asks for e-classes
    /// explicitly, and the module doc explains why the two numbers coincide
    /// rather than leaving that as an unstated assumption at the call site.
    #[must_use]
    pub fn median_classes_added(&self) -> usize {
        self.median_nodes_added()
    }

    /// Maximum e-nodes added by a single application of this rule.
    #[must_use]
    pub fn max_nodes_added(&self) -> usize {
        self.growth_hist.keys().next_back().copied().unwrap_or(0)
    }

    /// See [`Self::median_classes_added`].
    #[must_use]
    pub fn max_classes_added(&self) -> usize {
        self.max_nodes_added()
    }

    /// Fold `other`'s recordings into `self` — the sum of the two runs'
    /// histograms. Exact: a histogram sums exactly, unlike averaging two
    /// already-reduced medians would. For a caller combining growth
    /// telemetry from several separately-run kernels into one distribution
    /// (e.g. a corpus sweep, one [`super::graph::EGraph`] per expression).
    pub fn merge(&mut self, other: &Self) {
        self.applications += other.applications;
        self.no_op += other.no_op;
        for (&value, &count) in &other.growth_hist {
            *self.growth_hist.entry(value).or_insert(0) += count;
        }
    }

    /// The value at quantile `q` (`0.5` for the median) of the
    /// `nodes_added` distribution, read off the histogram by cumulative
    /// count — exact, since the histogram holds every recorded application,
    /// never a sample of them.
    fn quantile_nodes_added(&self, q: f64) -> usize {
        if self.applications == 0 {
            return 0;
        }
        // 1-indexed rank, ceiling: the smallest value whose cumulative share
        // is >= q. `q=0.5, n=1` picks rank 1 (the only sample); `q=0.5, n=4`
        // picks rank 2 (the lower of the two middle samples — the
        // conventional discrete median).
        let rank = ((self.applications as f64) * q).ceil().max(1.0) as u64;
        let mut cumulative = 0u64;
        for (&value, &count) in &self.growth_hist {
            cumulative += count;
            if cumulative >= rank {
                return value;
            }
        }
        // Unreachable: `growth_hist`'s counts sum to `applications` by
        // construction (every `record` call increments both together), so
        // the loop above always returns before falling off the end. Kept as
        // a defined fallback rather than `unreachable!()` — this is a
        // measurement path, not a correctness-critical one, and a wrong
        // number here should never be a panic that takes a research harness
        // down with it.
        self.growth_hist.keys().next_back().copied().unwrap_or(0)
    }
}

/// Per-rule growth, accumulated across one [`EGraph`](super::graph::EGraph)'s
/// applications. See the module doc for what "on" means and what it costs.
#[derive(Clone, Debug, Default)]
pub struct GrowthTelemetry {
    by_rule: BTreeMap<RuleId, RuleGrowth>,
    /// Applications whose `rule_idx` had no [`RuleId`] in the graph's rule
    /// table. Should not happen against a graph built by
    /// [`EGraph::with_rules`](super::graph::EGraph::with_rules) or
    /// [`super::rules::RuleSet`] (every rule gets an id at construction) —
    /// kept anyway so a growth run degrades to "one bucket short a name"
    /// rather than panicking or silently dropping the record.
    unlabeled: RuleGrowth,
}

impl GrowthTelemetry {
    /// Record one committed application. `rule` is `None` only when the
    /// firing rule has no id in the graph's table — see [`Self::unlabeled`].
    pub(super) fn record(&mut self, rule: Option<RuleId>, nodes_added: usize, changed: bool) {
        let bucket = match rule {
            Some(id) => self.by_rule.entry(id).or_default(),
            None => &mut self.unlabeled,
        };
        bucket.record(nodes_added, changed);
    }

    /// Per-rule growth, in [`RuleId`] order (stable across runs, though not
    /// meaningful as a ranking — a caller wanting a ranked report sorts this
    /// itself, e.g. by [`RuleGrowth::applications`]).
    pub fn by_rule(&self) -> impl Iterator<Item = (RuleId, &RuleGrowth)> {
        self.by_rule.iter().map(|(&id, g)| (id, g))
    }

    /// Applications whose rule had no [`RuleId`] on the graph — see
    /// [`Self`]'s `unlabeled` field doc. Zero on every production graph.
    #[must_use]
    pub fn unlabeled(&self) -> &RuleGrowth {
        &self.unlabeled
    }

    /// Fold `other`'s per-rule recordings into `self`, rule by rule (see
    /// [`RuleGrowth::merge`]) — combines growth telemetry collected on
    /// separate [`EGraph`](super::graph::EGraph)s (e.g. one per expression
    /// in a corpus sweep) into one distribution per rule.
    pub fn merge(&mut self, other: &Self) {
        for (id, g) in &other.by_rule {
            self.by_rule.entry(*id).or_default().merge(g);
        }
        self.unlabeled.merge(&other.unlabeled);
    }
}
