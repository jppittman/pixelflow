//! Deployment of the cold-start linear Guide, and its
//! zero-candidate-local-information control arm
//! (`docs/plans/2026-08-31-guide-design-revision.md` §5, task 2).
//!
//! # Why this module exists: the train/deploy skew it closes
//!
//! `pixelflow-pipeline`'s `train_guide` binary trains a transparent linear
//! model directly against
//! [`crate::egraph::candidate::CandidateFeatures`]'s own field set, *not*
//! through [`super::SaturationGuide`] — a deliberate scope choice, but one
//! that leaves the trained weights with no path into a live guided
//! saturation loop unless something implements the trait identically to what
//! was trained. [`LinearCandidateGuide`] is that something: it reproduces
//! `Model::logit` field for field. The mandatory skew test
//! (`skew_test_linear_guide`) checks this module's score against the
//! trainer's own forward pass on held-out DEV records and requires
//! agreement to 1e-6 — the same discipline the extraction-head program
//! applied to its own train/deploy boundary, applied here before a weight
//! ever reaches a real saturation loop.
//!
//! # Weights in, JSON out of scope
//!
//! This module takes [`LinearWeights`], a parsed value. It does **not**
//! read a file. The branch this is ported from added `serde_json` to
//! `pixelflow-search` for a loader that only the training crate ever calls,
//! and `pixelflow-pipeline` already depends on `serde_json` — so parsing
//! lives there and the *refusal* lives here, at the constructor, where every
//! path into a deployed guide has to pass it (including a hand-built one in
//! a test). Subtracting the dependency also subtracted a whole class of
//! "which crate validated this?" ambiguity.
//!
//! # Rule identity, not rule position
//!
//! `w_rule` is keyed by [`RuleId`]. The branch keyed it by `rule_idx` with a
//! length check against the live rule table and a name comparison — and
//! `Rewrite::name` is a *family* name, so all four `Commutative` instances
//! answered to `"commutative"` and the comparison passed for a permutation
//! that moved them. A same-length reorder repoints every weight and nothing
//! anywhere is the wrong length. [`LinearWeights::fingerprint`] closes the
//! remaining hole: the whole rule vocabulary, content and order, digested —
//! and [`LinearCandidateGuide::new`] refuses a mismatch rather than scoring
//! with weights that name other rules.
//!
//! # The control arm: [`PerRuleRateGuide`]
//!
//! The Phase 3 registration's pre-flight (§8) warns that rule-granularity
//! oracle filtering barely moves the classical-band curve (94.9% → 85.4%
//! median regret at B=100) — almost all of the theoretical headroom is
//! candidate-local, not rule-local. [`PerRuleRateGuide`] makes that
//! distinction measurable for a *trained* Guide rather than an oracle one:
//! it scores a candidate using only its rule, against that rule's
//! TRAIN-measured strict-positive rate — no neighborhood ops, no budget
//! state, no matched-class size. If [`LinearCandidateGuide`]'s held-out
//! ranking quality is close to this control's, the linear model learned
//! little beyond a per-rule base rate; a real gap is evidence the
//! candidate-local features earned their place.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use pixelflow_ir::OpKind;

use super::{CandidateSummary, SaturationGuide};
use crate::egraph::rules::{Fingerprint, RuleId, RuleSet};

/// A deployed guide refused to be built. Never a default, never a warning:
/// a Guide silently scoring with a weight that names a different rule
/// corrupts move ordering without saying so.
#[derive(Debug)]
pub struct GuideError(String);

impl GuideError {
    /// Build a refusal from an already-formatted explanation.
    ///
    /// `pub(crate)` and constructor-only: every deployed-guide module in
    /// this crate refuses through the same type (see
    /// [`super::bilinear::BilinearCandidateGuide::new`]), but nothing
    /// outside the crate should be able to mint one — an error value a
    /// caller can fabricate is an error value a caller can fake past.
    pub(crate) fn new(message: String) -> Self {
        Self(message)
    }
}

impl core::fmt::Display for GuideError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for GuideError {}

/// The parsed contents of a `train_guide` checkpoint: the linear model's
/// weights plus the identity of the vocabulary they were trained against.
///
/// Both tables are keyed by name, never by position —
/// [`RuleId`] for rules (see the module doc), and `OpKind`'s `Debug` name
/// for ops, which is the string `gen_strict_labels` writes as a
/// neighborhood-histogram key and the convention the NNUE checkpoint format
/// already uses.
#[derive(Clone, Debug)]
pub struct LinearWeights {
    /// Intercept.
    pub bias: f32,
    /// Per-rule weight, by stable rule identity.
    pub w_rule: BTreeMap<RuleId, f32>,
    /// Per-op weight, by `OpKind`'s `Debug` name.
    pub w_op: BTreeMap<String, f32>,
    /// Coefficient on `budget_fraction`.
    pub w_budget: f32,
    /// Coefficient on `log1p(match_class_node_count)`.
    pub w_match_class: f32,
    /// Coefficient on `log1p(neighborhood_op_count)`.
    pub w_neighborhood: f32,
    /// Coefficient on `log1p(expr_node_count)`.
    pub w_expr_size: f32,
    /// The [`RuleSet::fingerprint`] of the vocabulary these weights were
    /// trained against. A checkpoint is a file — it can be stale,
    /// hand-edited, half-copied, or trained against a rule set since
    /// reordered, and none of those announce themselves.
    pub fingerprint: Fingerprint,
}

impl LinearWeights {
    /// Refuse these weights against `rules` if they were trained against a
    /// different vocabulary.
    ///
    /// One check, shared by both heads built on this layout
    /// ([`LinearCandidateGuide`] and [`LinearReturnGuide`]): a second copy
    /// is a second thing to forget to update.
    ///
    /// # Errors
    ///
    /// If [`Self::fingerprint`] disagrees with `rules.fingerprint()`, or if
    /// `rules` contains a rule these weights have no entry for. Both are the
    /// same failure seen from two sides, and both are refused before a
    /// single candidate is scored — the alternative is a guide that runs
    /// happily with `w_rule` naming the wrong rules.
    pub fn check_vocabulary(&self, rules: &RuleSet) -> Result<(), GuideError> {
        let live = rules.fingerprint();
        if self.fingerprint != live {
            return Err(GuideError(alloc::format!(
                "linear guide: weights were trained against rule set {} but are being \
                 deployed against {} — the vocabulary changed since training, so every \
                 w_rule entry may name a different rule. Retrain rather than deploy.",
                self.fingerprint,
                live
            )));
        }
        // The fingerprint covers the vocabulary; this covers the table. A
        // checkpoint that matches the fingerprint but is missing a weight
        // was written by something that did not write one per rule, and the
        // gap would surface as a mid-saturation panic instead of a load
        // failure.
        let missing: Vec<String> = (0..rules.len())
            .filter(|&i| {
                rules
                    .id_of(i)
                    .is_none_or(|id| !self.w_rule.contains_key(&id))
            })
            .filter_map(|i| rules.label_of(i))
            .collect();
        if !missing.is_empty() {
            return Err(GuideError(alloc::format!(
                "linear guide: the rule set's fingerprint matches but {} rule(s) have no \
                 weight: {missing:?}",
                missing.len()
            )));
        }
        Ok(())
    }
}

/// Deploys `train_guide`'s cold-start linear model as a
/// [`SaturationGuide`] — see the module doc for why this must reproduce
/// `Model::logit` exactly rather than approximate it.
///
/// Scores are raw logits, no sigmoid: [`SaturationGuide::score_candidates`]
/// needs a move-ordering rank, sigmoid is monotonic and so preserves it, and
/// skipping it avoids a transcendental per candidate. The skew test compares
/// this logit against the trainer's, not against its reported (post-sigmoid)
/// DEV probabilities.
#[derive(Clone, Debug)]
pub struct LinearCandidateGuide {
    weights: LinearWeights,
}

impl LinearCandidateGuide {
    /// Deploy `weights` against `rules`, refusing them if they were trained
    /// against a different vocabulary.
    ///
    /// # Errors
    ///
    /// If `weights.fingerprint` disagrees with `rules.fingerprint()`, or if
    /// `rules` contains a rule the weights have no entry for. Both are the
    /// same failure seen from two sides, and both are refused before a
    /// single candidate is scored — the alternative is a guide that runs
    /// happily with `w_rule` naming the wrong rules.
    pub fn new(weights: LinearWeights, rules: &RuleSet) -> Result<Self, GuideError> {
        weights.check_vocabulary(rules)?;
        Ok(Self { weights })
    }

    /// The vocabulary these weights were trained against.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.weights.fingerprint
    }

    /// One candidate's logit — `train_guide::Model::logit`'s exact formula,
    /// the single computation the mandatory skew test checks against.
    ///
    /// # Why every step goes through `black_box`
    ///
    /// This function and the trainer's `logit` are two independent
    /// implementations of the identical formula, by design — a shared
    /// helper would make the skew test check nothing. Written the obvious
    /// way, both agree bit for bit in an unoptimized build but can disagree
    /// by a handful of ULPs under release optimization: LLVM is free to fuse
    /// a `w*x + acc` pattern into one rounding (an FMA) in one function's
    /// compiled form and not the other's, purely as a function of each call
    /// site's own inlining and register allocation. That is the same
    /// one-versus-two-rounding `MulAdd` divergence this codebase's
    /// floating-point doctrine documents, surfacing between two host
    /// functions instead of between two ISA targets. A `black_box` after
    /// every partial sum forces a fully-rounded `f32` before the next term,
    /// which removes the contraction opportunity — restoring in release
    /// builds the two-rounding behavior an unoptimized build gives for free.
    fn logit(&self, c: &CandidateSummary) -> f32 {
        self.weights.logit(c)
    }
}

impl LinearWeights {
    /// The shared forward pass — see [`LinearCandidateGuide::logit`] for the
    /// `black_box` discipline and why it is load-bearing.
    fn logit(&self, c: &CandidateSummary) -> f32 {
        let w = self;
        let w_rule = *w.w_rule.get(&c.rule).unwrap_or_else(|| {
            panic!(
                "LinearCandidateGuide: rule {} has no entry in this checkpoint's w_rule \
                 table ({} entries) — the candidate came from a rule set this guide was \
                 not built against, which `LinearCandidateGuide::new` should have refused",
                c.rule,
                w.w_rule.len()
            )
        });
        let log_match_class = (c.match_class_node_count as f32 + 1.0).ln();
        let log_neighborhood = (c.neighborhood_ops.len() as f32 + 1.0).ln();
        let log_expr_size = (c.expr_node_count as f32 + 1.0).ln();

        let mut z = core::hint::black_box(w.bias);
        z = core::hint::black_box(z + w_rule);
        z = core::hint::black_box(z + w.w_budget * c.budget_fraction);
        z = core::hint::black_box(z + w.w_match_class * log_match_class);
        z = core::hint::black_box(z + w.w_neighborhood * log_neighborhood);
        z = core::hint::black_box(z + w.w_expr_size * log_expr_size);
        for (name, count) in op_features(&c.neighborhood_ops) {
            let w_op = *w.w_op.get(&name).unwrap_or_else(|| {
                panic!(
                    "LinearCandidateGuide: neighborhood op {name:?} has no entry in this \
                     checkpoint's w_op table — the checkpoint was trained against a \
                     different OpKind table than this binary was built with; retrain or \
                     rebuild against a matching pixelflow-ir revision"
                )
            });
            z = core::hint::black_box(z + w_op * count);
        }
        z
    }
}

/// One distinct op's `(Debug name, log1p(occurrence count))` pair, computed
/// exactly as `train_guide`'s `to_sample` builds `Sample::op_feats` from
/// `neighborhood_op_hist`: group by [`OpKind`]'s own `Debug` name (the same
/// string `gen_strict_labels` wrote as a histogram key), count, then
/// `(count + 1.0).ln()` — the trainer's literal formula, matched exactly for
/// the skew test's bar. Deterministic order (`BTreeMap`), because floating
/// point addition is not associative and the skew test compares sums.
fn op_features(neighborhood_ops: &[OpKind]) -> Vec<(String, f32)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for op in neighborhood_ops {
        *counts.entry(alloc::format!("{op:?}")).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(name, count)| (name, (count as f32 + 1.0).ln()))
        .collect()
}

impl SaturationGuide for LinearCandidateGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates.iter().map(|c| self.logit(c)).collect()
    }
}

/// Which loss trained a [`LinearReturnGuide`]'s weights.
///
/// A type rather than the checkpoint's raw string: "is this a return head at
/// all, and if so which one" is a three-way question a `String` answers by
/// convention and an enum answers by construction. The pipeline's loader
/// maps `"return-mse"` / `"return-rank"` onto this and refuses everything
/// else, so an unrecognized objective never reaches a scoring loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnObjective {
    /// Squared error against the (expression-centered) return-to-go.
    Mse,
    /// Pairwise rank loss within an expression's candidate set.
    Rank,
}

impl core::fmt::Display for ReturnObjective {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Mse => write!(f, "return-mse"),
            Self::Rank => write!(f, "return-rank"),
        }
    }
}

/// Deploys `train_guide_r2g`'s linear return-to-go regressor as a
/// [`SaturationGuide`] (docs/plans/2026-09-01-guide-return-to-go.md §3.3) —
/// the same [`LinearWeights`] formula [`LinearCandidateGuide`] uses, trained
/// against a different target (expression-centered log-regret rather than
/// the strict load-bearing bit), so its forward pass is a **predicted
/// cost**, not a probability.
///
/// # The sign, spelled out
///
/// [`SaturationGuide::score_candidates`] is a move-ordering rank: the guided
/// loop applies candidates in *descending* score order. A predicted return
/// is a predicted **regret** (§1.2: `R = ln(cost/best)`, 0 = optimal, more
/// positive = worse) — the candidate the loop should fire *first* is the one
/// whose predicted return is **lowest**, not highest. `score_candidates`
/// therefore returns `-predicted_return`, so "highest score" and "lowest
/// predicted regret" agree: the one deliberate difference from
/// [`LinearCandidateGuide`], whose logit is already in "bigger is better"
/// orientation and needs no such flip.
#[derive(Clone, Debug)]
pub struct LinearReturnGuide {
    weights: LinearWeights,
    objective: ReturnObjective,
}

impl LinearReturnGuide {
    /// Deploy `weights` as a return head against `rules`, refusing them if
    /// they were trained against a different vocabulary.
    ///
    /// # Errors
    ///
    /// The same two refusals as [`LinearCandidateGuide::new`] — see
    /// [`LinearWeights::check_vocabulary`].
    pub fn new(
        weights: LinearWeights,
        objective: ReturnObjective,
        rules: &RuleSet,
    ) -> Result<Self, GuideError> {
        weights.check_vocabulary(rules)?;
        Ok(Self { weights, objective })
    }

    /// Which loss trained these weights.
    #[must_use]
    pub fn objective(&self) -> ReturnObjective {
        self.objective
    }

    /// The vocabulary these weights were trained against.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.weights.fingerprint
    }

    /// One candidate's predicted return-to-go (§1.2's `R`, expression-
    /// centered if the checkpoint was trained on the centered target) —
    /// [`LinearWeights`]'s forward pass under a name that matches what this
    /// head predicts. Exposed (not just `score_candidates`'s negated form)
    /// so the mandatory skew test can compare this value directly against
    /// `train_guide_r2g`'s own forward pass, the same discipline
    /// [`LinearCandidateGuide`]'s skew test applies to its logit.
    #[must_use]
    pub fn predicted_return(&self, c: &CandidateSummary) -> f32 {
        self.weights.logit(c)
    }
}

impl SaturationGuide for LinearReturnGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        // Lower predicted return = higher priority: negate so "descending
        // score" (what the guided loop sorts by) means "ascending predicted
        // regret" — see this type's doc, "The sign, spelled out".
        candidates
            .iter()
            .map(|c| -self.predicted_return(c))
            .collect()
    }
}

/// The zero-candidate-local-information control arm — see the module doc.
/// Scores every candidate of a given rule with that rule's TRAIN-measured
/// strict-positive rate, ignoring everything else about the candidate.
#[derive(Clone, Debug)]
pub struct PerRuleRateGuide {
    rate: BTreeMap<RuleId, f32>,
}

impl PerRuleRateGuide {
    /// Build from a `rule -> TRAIN-measured rate` table.
    ///
    /// Keyed by identity like [`LinearWeights::w_rule`], for the same
    /// reason: this table is read out of a JSON report written by another
    /// run, and a report is a file.
    #[must_use]
    pub fn new(rate: BTreeMap<RuleId, f32>) -> Self {
        Self { rate }
    }

    /// The same table keyed by canonical label, which is the form a report
    /// on disk carries. A label that is not a rule in any vocabulary still
    /// hashes to a `RuleId`, and simply never matches a candidate — which is
    /// the honest outcome, since a control arm's table naming a rule this
    /// build does not have is not an error, just an unused row.
    #[must_use]
    pub fn from_labels(rows: &[(String, f32)]) -> Self {
        Self::new(
            rows.iter()
                .map(|(label, rate)| (RuleId::from_label(label), *rate))
                .collect(),
        )
    }
}

impl SaturationGuide for PerRuleRateGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates
            .iter()
            .map(|c| {
                // A rule with no measured rate never fired in TRAIN, which
                // is a real and informative answer (rate zero), not a
                // missing one — unlike `LinearCandidateGuide`, whose weight
                // table is supposed to be complete and panics if it is not.
                self.rate.get(&c.rule).copied().unwrap_or(0.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::factored::EMBED_DIM;

    fn label_of(idx: usize) -> String {
        RuleSet::production()
            .label_of(idx)
            .expect("index within the production set")
    }

    /// Weights over the production vocabulary: every rule gets a weight, so
    /// `new` accepts them, and the first three carry hand-chosen values the
    /// logit test can predict.
    fn weights() -> LinearWeights {
        let rules = RuleSet::production();
        let mut w_rule: BTreeMap<RuleId, f32> = (0..rules.len())
            .map(|i| (rules.id_of(i).expect("id"), 0.0f32))
            .collect();
        w_rule.insert(RuleId::from_label(&label_of(0)), 1.0);
        w_rule.insert(RuleId::from_label(&label_of(1)), -1.0);
        LinearWeights {
            bias: 0.1,
            w_rule,
            w_op: [("Add".to_string(), 0.5f32), ("Mul".to_string(), -0.25)]
                .into_iter()
                .collect(),
            w_budget: 2.0,
            w_match_class: 0.3,
            w_neighborhood: -0.2,
            w_expr_size: 0.05,
            fingerprint: rules.fingerprint(),
        }
    }

    fn candidate(rule: RuleId, ops: Vec<OpKind>) -> CandidateSummary {
        CandidateSummary {
            rule_embed: [0.0; EMBED_DIM],
            neighborhood_ops: ops,
            budget_fraction: 0.4,
            rule,
            match_class_node_count: 3,
            expr_node_count: 10,
        }
    }

    #[test]
    fn scores_a_hand_computed_logit() {
        let guide = LinearCandidateGuide::new(weights(), &RuleSet::production())
            .expect("weights cover the production set");
        let c = candidate(
            RuleId::from_label(&label_of(0)),
            alloc::vec![OpKind::Add, OpKind::Add, OpKind::Mul],
        );
        let scores = guide.score_candidates(&[c]);

        // bias + w_rule + w_budget*0.4 + w_match_class*ln(4)
        //      + w_neighborhood*ln(4) + w_expr_size*ln(11)
        //      + w_op[Add]*ln(3) + w_op[Mul]*ln(2)
        let expected = 0.1
            + 1.0
            + 2.0 * 0.4
            + 0.3 * (4.0f32).ln()
            + (-0.2) * (4.0f32).ln()
            + 0.05 * (11.0f32).ln()
            + 0.5 * (3.0f32).ln()
            + (-0.25) * (2.0f32).ln();
        assert!(
            (scores[0] - expected).abs() < 1e-5,
            "got {}, expected {expected}",
            scores[0]
        );
    }

    /// The G5 failure this module exists to prevent: a same-length REORDER
    /// of the rule vocabulary. Every length check still passes, every family
    /// name still matches, and every weight now names a different rule. The
    /// fingerprint is the only thing that sees it.
    #[test]
    fn a_same_length_reorder_is_refused() {
        let mut reversed = crate::egraph::all_rules();
        reversed.reverse();
        let other = RuleSet::new(reversed);
        assert_eq!(
            other.len(),
            RuleSet::production().len(),
            "precondition: the reorder is same-length, so no length check can catch it"
        );
        let err = LinearCandidateGuide::new(weights(), &other)
            .expect_err("a reordered vocabulary must be refused");
        assert!(
            err.to_string().contains("trained against rule set"),
            "the refusal must name the fingerprint mismatch: {err}"
        );
    }

    #[test]
    fn weights_missing_a_rule_are_refused_before_any_candidate_is_scored() {
        let rules = RuleSet::production();
        let mut w = weights();
        w.w_rule.remove(&rules.id_of(2).expect("id"));
        let err = LinearCandidateGuide::new(w, &rules)
            .expect_err("an incomplete weight table must be refused");
        assert!(err.to_string().contains("no weight"), "{err}");
    }

    #[test]
    fn per_rule_rate_guide_ignores_everything_but_the_rule() {
        let a_label = label_of(0);
        let b_label = label_of(1);
        let guide =
            PerRuleRateGuide::from_labels(&[(a_label.clone(), 0.1), (b_label.clone(), 0.9)]);
        let b = RuleId::from_label(&b_label);
        let scores = guide.score_candidates(&[
            candidate(b, alloc::vec![OpKind::Add, OpKind::Add, OpKind::Add]),
            candidate(b, alloc::vec![OpKind::Sqrt]),
        ]);
        assert!((scores[0] - 0.9).abs() < 1e-9);
        assert!(
            (scores[0] - scores[1]).abs() < 1e-9,
            "the rule alone determines the score"
        );
        // A rule the table never measured scores zero — it never fired in
        // TRAIN, which is a rate, not a missing entry.
        let unseen = guide.score_candidates(&[candidate(RuleId::from_label("not-a-rule"), vec![])]);
        assert!((unseen[0] - 0.0).abs() < 1e-9);
    }

    /// The control arm and the trained model are both `SaturationGuide`s, so
    /// the guided loop takes either without knowing which — the property the
    /// comparison in §8 depends on.
    #[test]
    fn both_arms_are_saturation_guides() {
        let guides: Vec<alloc::boxed::Box<dyn SaturationGuide>> = alloc::vec![
            alloc::boxed::Box::new(
                LinearCandidateGuide::new(weights(), &RuleSet::production()).expect("weights")
            ),
            alloc::boxed::Box::new(PerRuleRateGuide::from_labels(&[(label_of(0), 0.5)])),
        ];
        let c = candidate(RuleId::from_label(&label_of(0)), alloc::vec![OpKind::Add]);
        for g in &guides {
            let scores = g.score_candidates(core::slice::from_ref(&c));
            assert_eq!(scores.len(), 1);
            assert!(scores[0].is_finite());
        }
    }
}
