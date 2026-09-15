//! The representational claim these two model classes differ on, pinned as
//! tests rather than left in a plan document.
//!
//! # The claim
//!
//! [`super::super::linear::LinearCandidateGuide`]'s score is, reading
//! `LinearWeights::logit` term by term:
//!
//! ```text
//! s(r, x) = bias
//!         + w_rule[r]
//!         + w_budget       * budget_fraction
//!         + w_match_class  * ln(1 + match_class_node_count)
//!         + w_neighborhood * ln(1 + |neighborhood_ops|)
//!         + w_expr_size    * ln(1 + expr_node_count)
//!         + Σ_op w_op[op]  * ln(1 + count_op(neighborhood_ops))
//! ```
//!
//! Exactly one term names the rule, and it names nothing else; every other
//! term is a function of the candidate's context `x` alone. So the score is
//! **additively separable**, `s(r, x) = w_rule[r] + g(x)`, and the
//! rule-by-context interaction — the second difference — is identically
//! zero:
//!
//! ```text
//! [s(r1, x) − s(r2, x)] − [s(r1, x') − s(r2, x')] = 0   for all r1, r2, x, x'
//! ```
//!
//! Two readings of that, and the difference between them matters:
//!
//! - **What it forbids.** No context can change how the model ranks two
//!   rules *against each other*. `w_rule[r1] − w_rule[r2]` is the whole
//!   answer, everywhere. The model can say "this match site looks
//!   promising" (`g(x)` is free to vary), but it applies that judgement
//!   identically to every rule, so it can never say "the Pythagorean
//!   identity is worth firing *here* and not *there*" —
//!   `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` §0's
//!   question — while leaving every other rule where it was.
//! - **What it still permits.** Two candidates in one batch generally have
//!   *different* contexts (different match sites), so the additive model can
//!   and does reorder candidates. It is candidate ordering that stays free;
//!   rule ordering *at a fixed context* is one global list. The tests below
//!   pin the fixed-context statement and the zero-interaction statement,
//!   which together are the precise version of "one global rule ranking".
//!
//! ## What the budget term can and cannot do
//!
//! `w_budget * budget_fraction` is the one term that looks like it carries
//! search state, and under the live consumer it carries none. The guided
//! loop reads the application ordinal **once per round**, before it walks
//! the round's matches (`egraph::guided`, `let ordinal =
//! egraph.application_count();`), so every [`CandidateSummary`] in one
//! `score_candidates` batch has the same `budget_fraction`; `expr_node_count`
//! is likewise an episode constant. A term that is constant across a batch
//! adds the same number to every score, and the consumer sorts descending
//! and otherwise ignores the magnitudes — no threshold anywhere. So the
//! budget term can move the absolute score and can never move a decision.
//! It could only matter to a consumer that compared a score against a fixed
//! cutoff, and there is none.
//!
//! # The contrast
//!
//! [`SaturationHead::score_candidate`] ends in
//! [`SaturationHead::bilinear_score`], which is `m(x)ᵀ W r + bᵀr` — linear in
//! the rule embedding `r` with **coefficients that are a function of the
//! context** `m(x)`. Its rule-pair difference is `(m(x)ᵀW + bᵀ)(r1 − r2)`,
//! which depends on `x`. That is a real interaction, and the last test here
//! exhibits a hand-set weight matrix under which two rules swap places
//! between a trig-shaped neighborhood and a polynomial-shaped one.
//!
//! This is a statement about **model class, not about training**: the
//! weights below are chosen, not learned. It says the additive model cannot
//! represent the hypothesis Round 1b tested, not that the bilinear one
//! learns it.

use alloc::string::String;

use super::*;
use crate::egraph::rules::{RuleId, RuleSet};
use crate::nnue::guide::linear::{LinearCandidateGuide, LinearWeights};
use crate::nnue::guide::{CandidateSummary, Guide, SaturationGuide};

/// A neighborhood shaped like the inside of a spherical-harmonic kernel.
const SH_LIKE: OpKind = OpKind::Sin;
/// A neighborhood shaped like the inside of a Bézier evaluation.
const BEZIER_LIKE: OpKind = OpKind::Add;

fn label_of(idx: usize) -> String {
    RuleSet::production()
        .label_of(idx)
        .expect("index within the production set")
}

/// Two rules from the live vocabulary, distinct, with hand-set additive
/// weights that put `hi` strictly above `lo`.
fn two_rules() -> (RuleId, RuleId) {
    (
        RuleId::from_label(&label_of(0)),
        RuleId::from_label(&label_of(1)),
    )
}

/// Additive weights over the whole production vocabulary (so `new` accepts
/// them), with a nonzero weight on each of the two neighborhood ops the
/// contexts differ in — so the two contexts genuinely score differently and
/// the order tests are not vacuous.
fn additive_weights() -> LinearWeights {
    let rules = RuleSet::production();
    let (hi, lo) = two_rules();
    let mut w_rule: alloc::collections::BTreeMap<RuleId, f32> = (0..rules.len())
        .map(|i| (rules.id_of(i).expect("id"), 0.0f32))
        .collect();
    w_rule.insert(hi, 1.0);
    w_rule.insert(lo, -1.0);
    LinearWeights {
        bias: 0.1,
        w_rule,
        w_op: [
            (alloc::format!("{SH_LIKE:?}"), 3.0f32),
            (alloc::format!("{BEZIER_LIKE:?}"), -3.0f32),
        ]
        .into_iter()
        .collect(),
        w_budget: 2.0,
        w_match_class: 0.3,
        w_neighborhood: -0.2,
        w_expr_size: 0.05,
        fingerprint: rules.fingerprint(),
    }
}

/// One candidate: a rule at a context. The two contexts differ **only** in
/// the neighborhood op (same length, so even `w_neighborhood`'s
/// `ln(1 + |ops|)` term is identical between them) — the single degree of
/// freedom the round-1b registration named as the only thing in either arm
/// that can see a trig-dominant neighborhood.
fn candidate(rule: RuleId, context: OpKind, budget_fraction: f32) -> CandidateSummary {
    CandidateSummary {
        rule_embed: rule_embed_for(rule),
        neighborhood_ops: alloc::vec![context],
        budget_fraction,
        rule,
        match_class_node_count: 3,
        expr_node_count: 10,
    }
}

/// One-hot rule embeddings for the bilinear arm: `hi` on lane 0, `lo` on
/// lane 1. The additive arm ignores this field entirely (it keys on
/// `CandidateSummary::rule`), so both arms can score the very same
/// candidates.
fn rule_embed_for(rule: RuleId) -> [f32; EMBED_DIM] {
    let (hi, _lo) = two_rules();
    let mut e = [0.0f32; EMBED_DIM];
    if rule == hi {
        e[0] = 1.0
    } else {
        e[1] = 1.0
    }
    e
}

/// Rules in descending-score order — the exact ordering the guided loop
/// derives from `score_candidates` (`egraph::guided`: `sort_by` on the
/// scores, descending, stable).
fn descending_rule_order(guide: &dyn SaturationGuide, cs: &[CandidateSummary]) -> Vec<RuleId> {
    let scores = guide.score_candidates(cs);
    let mut order: Vec<usize> = (0..cs.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .expect("scores must be finite")
    });
    order.into_iter().map(|i| cs[i].rule).collect()
}

fn additive_guide() -> LinearCandidateGuide {
    LinearCandidateGuide::new(additive_weights(), &RuleSet::production())
        .expect("weights cover the production set")
}

/// The two candidate sets the claim is stated over: the same two rules,
/// once in each context.
fn both_contexts(budget_fraction: f32) -> (Vec<CandidateSummary>, Vec<CandidateSummary>) {
    let (hi, lo) = two_rules();
    (
        alloc::vec![
            candidate(hi, SH_LIKE, budget_fraction),
            candidate(lo, SH_LIKE, budget_fraction),
        ],
        alloc::vec![
            candidate(hi, BEZIER_LIKE, budget_fraction),
            candidate(lo, BEZIER_LIKE, budget_fraction),
        ],
    )
}

// ============================================================================
// The additive model: one global rule ranking
// ============================================================================

#[test]
fn additive_guide_should_induce_the_same_rule_order_in_every_context() {
    let guide = additive_guide();
    let (sh, bezier) = both_contexts(0.4);

    // Non-vacuity first: the two contexts must actually reach the model, or
    // "the order did not change" would be a statement about the fixture.
    let sh_scores = guide.score_candidates(&sh);
    let bezier_scores = guide.score_candidates(&bezier);
    assert!(
        (sh_scores[0] - bezier_scores[0]).abs() > 1e-3,
        "fixture must vary the context the model can see: {sh_scores:?} vs {bezier_scores:?}"
    );

    let (hi, lo) = two_rules();
    assert_eq!(
        descending_rule_order(&guide, &sh),
        alloc::vec![hi, lo],
        "the rule with the larger w_rule ranks first"
    );
    assert_eq!(
        descending_rule_order(&guide, &sh),
        descending_rule_order(&guide, &bezier),
        "an additively separable score imposes one global rule order: no context can \
         reorder two rules, because the context term is the same for both"
    );
}

#[test]
fn additive_guide_should_have_an_identically_zero_rule_by_context_interaction() {
    let guide = additive_guide();
    let (sh, bezier) = both_contexts(0.4);
    let sh_scores = guide.score_candidates(&sh);
    let bezier_scores = guide.score_candidates(&bezier);

    // The sharp form of the claim: the rule-pair gap is the SAME NUMBER in
    // both contexts, not merely the same sign. That number is
    // `w_rule[hi] - w_rule[lo]`, which the fixture set to 1.0 - (-1.0).
    let sh_gap = sh_scores[0] - sh_scores[1];
    let bezier_gap = bezier_scores[0] - bezier_scores[1];
    assert!(
        (sh_gap - bezier_gap).abs() < 1e-5,
        "the rule-by-context second difference must be zero: {sh_gap} vs {bezier_gap}"
    );
    assert!(
        (sh_gap - 2.0).abs() < 1e-5,
        "and it must equal w_rule[hi] - w_rule[lo] = 2.0, got {sh_gap}"
    );
}

#[test]
fn additive_guide_budget_term_should_shift_every_score_equally_and_reorder_nothing() {
    // The guided loop gives every candidate in one batch the same
    // `budget_fraction` (`egraph::guided` reads the ordinal once per round),
    // so the budget term is a per-batch constant. Raising it moves every
    // score by exactly `w_budget * delta` and moves no decision.
    let guide = additive_guide();
    let (early, _) = both_contexts(0.1);
    let (late, _) = both_contexts(0.9);

    let early_scores = guide.score_candidates(&early);
    let late_scores = guide.score_candidates(&late);

    let expected_shift = 2.0 * (0.9 - 0.1); // w_budget * delta
    for (i, (&e, &l)) in early_scores.iter().zip(late_scores.iter()).enumerate() {
        assert!(
            (l - e - expected_shift).abs() < 1e-5,
            "candidate {i}: budget must shift the score by {expected_shift}, got {}",
            l - e
        );
    }
    assert_eq!(
        descending_rule_order(&guide, &early),
        descending_rule_order(&guide, &late),
        "a per-batch constant cannot reorder a batch"
    );
}

// ============================================================================
// The bilinear head: the interaction the additive model does not have
// ============================================================================

/// A hand-set head that routes lane 0 of the candidate tower straight
/// through to lane 0 of the bilinear's context vector, and lane 1 to lane 1,
/// with everything else zero. Then `bilinear_score(m(x), r) = m(x)·r`, and a
/// one-hot context picks out one lane of the rule embedding.
///
/// Every stage is set to a selector rather than to noise so the resulting
/// score is readable by hand: this test is a statement about what the
/// functional form *can* express, and a random head would only show that
/// some numbers came out different.
fn selector_head() -> SaturationHead {
    let mut head = SaturationHead::new();
    // candidate tower: op-embedding lane i -> hidden column i, for i in {0,1}.
    head.candidate_w1[0][0] = 1.0;
    head.candidate_w1[1][1] = 1.0;
    // trunk: identity on the two live columns (the rest stay zero, and the
    // ReLU passes non-negative values through unchanged).
    head.trunk_w[0][0] = 1.0;
    head.trunk_w[1][1] = 1.0;
    // hidden column i -> candidate embedding lane i.
    head.candidate_proj_w[0][0] = 1.0;
    head.candidate_proj_w[1][1] = 1.0;
    // mask MLP: embedding lane i -> mask feature lane i.
    head.mask_mlp_w1[0][0] = 1.0;
    head.mask_mlp_w1[1][1] = 1.0;
    head.mask_mlp_w2[0][0] = 1.0;
    head.mask_mlp_w2[1][1] = 1.0;
    // interaction: identity on the two live lanes, so the score is the dot
    // product of the context's one-hot with the rule embedding.
    head.interaction[0][0] = 1.0;
    head.interaction[1][1] = 1.0;
    head
}

/// Op embeddings that put the SH-shaped op on lane 0 and the Bézier-shaped
/// op on lane 1.
fn one_hot_embeddings() -> OpEmbeddings {
    let mut emb = OpEmbeddings::new();
    let mut sh = [0.0f32; K];
    sh[0] = 1.0;
    let mut bez = [0.0f32; K];
    bez[1] = 1.0;
    emb.e[SH_LIKE] = sh;
    emb.e[BEZIER_LIKE] = bez;
    emb
}

fn bilinear_guide() -> Guide {
    Guide {
        embeddings: one_hot_embeddings(),
        head: selector_head(),
        rule_embeds: alloc::collections::BTreeMap::new(),
    }
}

#[test]
fn bilinear_head_should_reorder_two_rules_between_two_contexts() {
    let guide = bilinear_guide();
    let (sh, bezier) = both_contexts(0.4);
    let (hi, lo) = two_rules();

    // Same two rules, same two candidate sets the additive tests above use.
    // `hi` embeds on lane 0 (the SH lane), `lo` on lane 1 (the Bézier lane),
    // so each context promotes its own rule — the reordering the additive
    // model's zero second difference forbids.
    assert_eq!(
        descending_rule_order(&guide, &sh),
        alloc::vec![hi, lo],
        "the SH-shaped neighborhood must rank the SH-aligned rule first"
    );
    assert_eq!(
        descending_rule_order(&guide, &bezier),
        alloc::vec![lo, hi],
        "the same two rules must swap under the Bézier-shaped neighborhood — this is the \
         rule-by-context interaction `m(x)ᵀW(r1 − r2)` that the additive score's \
         `w_rule[r1] − w_rule[r2]` cannot express"
    );

    // And the second difference the additive model pins at zero is nonzero
    // here — the same statistic, computed the same way, for the contrast.
    let sh_scores = guide.score_candidates(&sh);
    let bezier_scores = guide.score_candidates(&bezier);
    let interaction = (sh_scores[0] - sh_scores[1]) - (bezier_scores[0] - bezier_scores[1]);
    assert!(
        interaction.abs() > 1e-5,
        "the bilinear rule-by-context second difference must be nonzero, got {interaction}"
    );
}

/// Both arms implement the same trait over the same candidates — the
/// property the registered comparison depends on, and the reason the
/// contrast above is between model classes and nothing else.
#[test]
fn both_model_classes_score_the_same_candidate_sets_through_one_trait() {
    let (sh, _) = both_contexts(0.4);
    let arms: Vec<alloc::boxed::Box<dyn SaturationGuide>> = alloc::vec![
        alloc::boxed::Box::new(additive_guide()),
        alloc::boxed::Box::new(bilinear_guide()),
    ];
    for arm in &arms {
        let scores = arm.score_candidates(&sh);
        assert_eq!(scores.len(), sh.len());
        assert!(scores.iter().all(|s| s.is_finite()));
    }
}
