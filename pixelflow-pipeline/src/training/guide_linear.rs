//! Shared feature encoding + forward pass for the cold-start linear
//! saturation Guide (`docs/plans/2026-08-31-guide-design-revision.md` §5).
//!
//! Split out of `train_guide` (2026-09-02, design doc §5 task 2) so the
//! trainer and the mandatory train/deploy skew test
//! (`bin/skew_test_linear_guide.rs`) share exactly one definition of "what
//! a strict-label JSONL row means as a feature vector" and "what the linear
//! model's forward pass computes" — the same anti-drift discipline
//! `pixelflow_search::egraph::candidate`'s own module doc states ("there is
//! exactly one place a caller can get this wrong"). Two independent
//! re-derivations of this encoding were exactly the failure mode the skew
//! test exists to catch; keeping both binaries importing the same `Sample`/
//! `Model` means the only thing left free to diverge between "trainer
//! forward" and "deployed impl" is the actual code path each one calls
//! through — `Model::logit` here vs
//! `pixelflow_search::nnue::guide::linear::LinearCandidateGuide` — which is
//! the divergence the test is supposed to measure.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{Fingerprint, RuleId, RuleSet};
use pixelflow_search::nnue::guide::linear::{
    LinearCandidateGuide, LinearWeights, PerRuleRateGuide,
};
use serde::Deserialize;
use serde_json::Value;

/// One strict-label JSONL row, field-for-field what `gen_strict_labels`
/// writes (`pixelflow-pipeline/src/bin/gen_strict_labels.rs`).
#[derive(Deserialize)]
pub struct Record {
    pub expr_name: String,
    pub family_band: u32,
    pub family_seed: u64,
    pub expr_node_count: usize,
    /// The firing rule's stable identity, as `gen_strict_labels` wrote it
    /// (`RuleId::get()`). Read back through [`RuleId`], never used as a
    /// position: a same-length reorder of `all_rules()` repoints every
    /// positional key silently
    /// (docs/plans/2026-09-02-phase3-forward-port.md §2.2).
    pub rule_id: u64,
    /// The rule's canonical label — the human-readable half of the same
    /// identity, and what a checkpoint's `w_rule` table is keyed by.
    pub rule_name: String,
    pub budget_fraction: f32,
    pub match_class_node_count: usize,
    pub neighborhood_op_count: usize,
    pub neighborhood_op_hist: std::collections::BTreeMap<String, usize>,
    /// `true` when this application re-fired a `CandidateKey` an earlier
    /// application in the same episode already resolved.
    ///
    /// The live `GuidedSaturation` loop removes seen keys BEFORE scoring, so
    /// a repeat row is a candidate no deployed guide will ever be asked to
    /// rank. Consumers that train on or evaluate against these rows are
    /// measuring a population the deployment never sees — 93.4% of the
    /// committed TRAIN split, overwhelmingly negative. Kept in the schema
    /// (rather than filtered at mint time) so the dedup rate stays
    /// measurable; every consumer must decide about it explicitly.
    pub dedup_repeat: bool,
    pub label_positive: bool,
}

/// One training/eval sample's encoded features plus its label. Sparse in the
/// op dimension (`op_feats`) since a matched class's neighborhood rarely
/// touches more than a handful of distinct `OpKind`s even though the full op
/// table has ~50 entries.
pub struct Sample {
    /// Dense position of this candidate's rule **within this run's own
    /// vocabulary** ([`rule_index_table`]) — never serialized, never
    /// compared across runs. The identity that crosses a file boundary is
    /// [`Record::rule_name`]; this is only the slot the weight vector is
    /// laid out in, resolved from that label once per row.
    pub rule_slot: usize,
    /// `(op_index, log1p(count))` pairs, one per distinct op present.
    pub op_feats: Vec<(usize, f32)>,
    pub budget_fraction: f32,
    pub log_match_class: f32,
    pub log_neighborhood: f32,
    pub log_expr_size: f32,
    pub label: f32,
}

/// Build the `OpKind` name -> dense-index table, in `OpKind::all()`'s own
/// order. `gen_strict_labels` wrote histogram keys as `format!("{op:?}")`
/// (a bare variant name, e.g. `"Abs"`), so `format!("{:?}", op)` here reads
/// back exactly the same strings — one definition of the op-name spelling
/// (`OpKind`'s own `Debug` impl), not restated. That spelling is persisted:
/// strict-label and r2g datasets key `neighborhood_op_hist` by it and Guide
/// checkpoints record it as `op_names`, so renaming a variant (as `Select`
/// became `If`) means regenerating them; every reader refuses an unknown name
/// loudly rather than misreading one.
#[must_use]
pub fn op_index_table() -> (Vec<String>, HashMap<String, usize>) {
    let names: Vec<String> = OpKind::all().map(|op| format!("{op:?}")).collect();
    let index: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    (names, index)
}

/// Build the canonical-rule-label -> dense-slot table for THIS build's
/// production rule set, alongside the labels in slot order.
///
/// The labels are [`pixelflow_search::egraph::rule_label`]'s own spelling
/// (via [`RuleSet::label_of`]), so a checkpoint written keyed by label reads
/// back into the same slots on any build whose rule set still contains those
/// rules — and loudly fails to on one that does not.
#[must_use]
pub fn rule_index_table(rules: &RuleSet) -> (Vec<String>, HashMap<String, usize>) {
    let labels: Vec<String> = (0..rules.len())
        .map(|i| {
            rules
                .label_of(i)
                .expect("index within the rule set has a label")
        })
        .collect();
    let index: HashMap<String, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.clone(), i))
        .collect();
    (labels, index)
}

/// Encode one JSONL [`Record`] into a [`Sample`] against a fixed op-name
/// index (see [`op_index_table`]) and rule-label index (see
/// [`rule_index_table`]).
#[must_use]
pub fn to_sample(
    record: &Record,
    op_index: &HashMap<String, usize>,
    rule_index: &HashMap<String, usize>,
) -> Sample {
    let mut op_feats: Vec<(usize, f32)> = record
        .neighborhood_op_hist
        .iter()
        .map(|(name, &count)| {
            let idx = *op_index.get(name).unwrap_or_else(|| {
                panic!(
                    "guide_linear::to_sample: neighborhood_op_hist names op {name:?}, which is \
                     not in OpKind::all() — the strict-label dataset was minted against a \
                     different OpKind table than this binary was built with; regenerate the \
                     dataset"
                )
            });
            (idx, (count as f32 + 1.0).ln())
        })
        .collect();
    op_feats.sort_by_key(|&(i, _)| i);
    // The label is the key; the id in the row is checked against it rather
    // than trusted, so a dataset whose two halves of the identity disagree
    // stops here instead of training a weight onto the wrong rule.
    assert_eq!(
        RuleId::from_label(&record.rule_name).get(),
        record.rule_id,
        "guide_linear::to_sample: row names rule {:?} but carries rule_id {} — the label and \
         the id in one strict-label row must be the same identity",
        record.rule_name,
        record.rule_id,
    );
    let rule_slot = *rule_index.get(&record.rule_name).unwrap_or_else(|| {
        panic!(
            "guide_linear::to_sample: the strict-label dataset names rule {:?}, which is not \
             in this build's rule set — the dataset was minted against a different rule \
             vocabulary; regenerate it or build against the rule set it was minted on",
            record.rule_name
        )
    });
    Sample {
        rule_slot,
        op_feats,
        budget_fraction: record.budget_fraction,
        log_match_class: (record.match_class_node_count as f32 + 1.0).ln(),
        log_neighborhood: (record.neighborhood_op_count as f32 + 1.0).ln(),
        log_expr_size: (record.expr_node_count as f32 + 1.0).ln(),
        label: if record.label_positive { 1.0 } else { 0.0 },
    }
}

/// Model: per-rule bias + bag-of-ops linear term + scalar features.
///
/// `logit = bias + w_rule[rule_slot] + sum_op(hist[op] * w_op[op]) +
/// w_budget*budget_fraction + w_match*log_match_class +
/// w_neigh*log_neighborhood + w_size*log_expr_size`.
///
/// This is the exact formula
/// [`pixelflow_search::nnue::guide::linear::LinearCandidateGuide`] must
/// reproduce; the skew test compares [`Model::logit`] against that type's
/// `score_candidates` on the same checkpoint weights and the same DEV rows.
pub struct Model {
    pub bias: f32,
    pub w_rule: Vec<f32>,
    pub w_op: Vec<f32>,
    pub w_budget: f32,
    pub w_match_class: f32,
    pub w_neighborhood: f32,
    pub w_expr_size: f32,
}

impl Model {
    /// Cold start: every weight at zero (design doc §5, "no warm-start from
    /// any prior guide/mask-head weights").
    #[must_use]
    pub fn zero(num_rules: usize, num_ops: usize) -> Self {
        Self {
            bias: 0.0,
            w_rule: vec![0.0; num_rules],
            w_op: vec![0.0; num_ops],
            w_budget: 0.0,
            w_match_class: 0.0,
            w_neighborhood: 0.0,
            w_expr_size: 0.0,
        }
    }

    /// See `pixelflow_search::nnue::guide::linear::LinearCandidateGuide::logit`'s
    /// doc for why every partial sum here goes through `black_box`: this
    /// function and that one are the two independent sides of the mandatory
    /// train/deploy skew test, and the barrier is what keeps their release-
    /// build FMA-contraction choices from silently diverging by a few ULPs.
    #[must_use]
    pub fn logit(&self, s: &Sample) -> f32 {
        let mut z = std::hint::black_box(self.bias);
        z = std::hint::black_box(z + self.w_rule[s.rule_slot]);
        z = std::hint::black_box(z + self.w_budget * s.budget_fraction);
        z = std::hint::black_box(z + self.w_match_class * s.log_match_class);
        z = std::hint::black_box(z + self.w_neighborhood * s.log_neighborhood);
        z = std::hint::black_box(z + self.w_expr_size * s.log_expr_size);
        for &(idx, count) in &s.op_feats {
            z = std::hint::black_box(z + self.w_op[idx] * count);
        }
        z
    }

    /// One online-SGD step for sample `s`, given the loss gradient w.r.t.
    /// the logit (`grad_z`, already clipped by the caller).
    ///
    /// The data term is sparse — only the features this sample carries have
    /// a gradient. The L2 penalty is NOT: its gradient is `l2 * w` for every
    /// weight on every step. Decaying only the weights a sample happened to
    /// touch made the effective penalty a function of feature frequency, so
    /// a rule firing once per epoch was regularized orders of magnitude less
    /// than `commutative` — which is a distortion of exactly the learned
    /// per-rule priorities this model exists to produce. The tables are ~60
    /// rules and ~50 ops, so decaying all of them per step is cheaper than
    /// the bookkeeping a lazy scheme would need (and exact, which a lazy
    /// scheme is not under this trainer's per-epoch learning-rate decay).
    ///
    /// The bias is exempt by convention: a bias term has no "large weight"
    /// failure mode L2 exists to guard against.
    pub fn sgd_step(&mut self, s: &Sample, grad_z: f32, lr: f32, l2: f32) {
        // Decay first, then the data gradient: `w * (1 - lr*l2) - lr*g` is
        // the same step as `w - lr*(g + l2*w)`, just written so the decay
        // can cover every weight uniformly.
        let keep = 1.0 - lr * l2;
        for w in self.w_rule.iter_mut().chain(self.w_op.iter_mut()) {
            *w *= keep;
        }
        self.w_budget *= keep;
        self.w_match_class *= keep;
        self.w_neighborhood *= keep;
        self.w_expr_size *= keep;

        self.bias -= lr * grad_z;
        self.w_rule[s.rule_slot] -= lr * grad_z;
        for &(idx, count) in &s.op_feats {
            self.w_op[idx] -= lr * grad_z * count;
        }
        self.w_budget -= lr * grad_z * s.budget_fraction;
        self.w_match_class -= lr * grad_z * s.log_match_class;
        self.w_neighborhood -= lr * grad_z * s.log_neighborhood;
        self.w_expr_size -= lr * grad_z * s.log_expr_size;
    }
}

// ── Ranking metrics ──────────────────────────────────────────────────────────
//
// Shared by `train_guide` (reports the linear model's DEV AUC/PR-AUC) and
// `eval_control_guides` (reports the same two metrics for the
// [`crate::training::guide_linear`]-free `PerRuleRateGuide` control) — one
// definition of each metric, so a side-by-side comparison between the two
// guides is comparing like with like.

/// Rank-based AUC-ROC (Mann-Whitney U, average-rank tie handling). `None` if
/// either class is absent — undefined, not zero.
#[must_use]
pub fn auc_roc(scores: &[f32], labels: &[f32]) -> Option<f64> {
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&y| y > 0.5).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or_else(|| panic!("auc_roc: NaN score"))
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let sum_ranks_pos: f64 = (0..n).filter(|&i| labels[i] > 0.5).map(|i| ranks[i]).sum();
    Some(
        (sum_ranks_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64),
    )
}

/// One AUC per decision set, macro-averaged over the sets that contain both
/// classes — the ranking question a deployed guide is actually asked.
///
/// Returns `(mean_auc, groups_scored)`, or `None` if no group has both
/// classes.
///
/// A pooled AUC over every record lets a model gain credit from differences
/// BETWEEN expressions and saturation rounds. Two of the model's features
/// cannot possibly produce that credit at deploy time: `expr_node_count` is
/// constant for a whole episode and `budget_fraction` is shared by every
/// candidate in one live scoring round, so neither can reorder a single
/// round's candidates. Whenever positive prevalence varies with expression
/// size or progress, the pooled number reports ranking signal the guide
/// cannot use for move ordering; grouping removes the between-group
/// variation before the metric sees it.
#[must_use]
pub fn auc_roc_within_groups<K: Ord + Clone>(
    scores: &[f32],
    labels: &[f32],
    groups: &[K],
) -> Option<(f64, usize)> {
    assert_eq!(
        scores.len(),
        labels.len(),
        "auc_roc_within_groups: scores/labels length mismatch"
    );
    assert_eq!(
        scores.len(),
        groups.len(),
        "auc_roc_within_groups: scores/groups length mismatch"
    );
    let mut by_group: std::collections::BTreeMap<K, (Vec<f32>, Vec<f32>)> =
        std::collections::BTreeMap::new();
    for i in 0..scores.len() {
        let e = by_group
            .entry(groups[i].clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        e.0.push(scores[i]);
        e.1.push(labels[i]);
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (s, l) in by_group.values() {
        if let Some(a) = auc_roc(s, l) {
            sum += a;
            n += 1;
        }
    }
    (n > 0).then(|| (sum / n as f64, n))
}

/// Average precision (area under the precision-recall step function,
/// sklearn's `average_precision_score` convention). `None` if there are no
/// positives.
#[must_use]
pub fn average_precision(scores: &[f32], labels: &[f32]) -> Option<f64> {
    let n_pos = labels.iter().filter(|&&y| y > 0.5).count();
    if n_pos == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or_else(|| panic!("average_precision: NaN score"))
    });
    // sklearn's convention evaluates one threshold per DISTINCT score, so
    // every example tied at that score is on the positive side of it
    // together. Walking tied examples one at a time instead makes the result
    // depend on the input's record order — and the per-rule control guide,
    // which assigns one score per rule, is nothing but large tie groups.
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut ap = 0.0f64;
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let mut group_pos = 0usize;
        for &k in &idx[i..=j] {
            if labels[k] > 0.5 {
                group_pos += 1;
            } else {
                fp += 1;
            }
        }
        tp += group_pos;
        if group_pos > 0 {
            let precision = tp as f64 / (tp + fp) as f64;
            ap += precision * (group_pos as f64 / n_pos as f64);
        }
        i = j + 1;
    }
    Some(ap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_decays_every_weight_not_only_the_ones_a_sample_touched() {
        let mut m = Model::zero(3, 2);
        m.w_rule = vec![1.0, 1.0, 1.0];
        m.w_op = vec![1.0, 1.0];
        let s = Sample {
            rule_slot: 0,
            op_feats: vec![(0, 1.0)],
            budget_fraction: 0.0,
            log_match_class: 0.0,
            log_neighborhood: 0.0,
            log_expr_size: 0.0,
            label: 1.0,
        };
        // Zero gradient: the only thing that can move a weight is the L2 term.
        m.sgd_step(&s, 0.0, 0.1, 0.5);
        let keep = 1.0 - 0.1 * 0.5;
        for (i, w) in m.w_rule.iter().enumerate() {
            assert!(
                (w - keep).abs() < 1e-6,
                "w_rule[{i}] = {w}, expected every rule weight decayed to {keep} — a rule \
                 that did not fire this step must still pay the penalty"
            );
        }
        for (i, w) in m.w_op.iter().enumerate() {
            assert!((w - keep).abs() < 1e-6, "w_op[{i}] = {w}, expected {keep}");
        }
    }

    #[test]
    fn average_precision_groups_tied_scores_at_one_threshold() {
        // Two positives and two negatives, all tied at one score: sklearn's
        // average_precision_score is 0.5 (the single threshold's precision).
        let scores = vec![1.0f32; 4];
        let labels = vec![1.0f32, 0.0, 1.0, 0.0];
        let ap = average_precision(&scores, &labels).unwrap();
        assert!((ap - 0.5).abs() < 1e-9, "ap={ap}");
    }

    #[test]
    fn average_precision_of_a_tied_block_is_independent_of_record_order() {
        let scores = vec![1.0f32; 6];
        let a = average_precision(&scores, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let b = average_precision(&scores, &[0.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(
            (a - b).abs() < 1e-12,
            "a tie group must not be ranked by JSONL order: {a} vs {b}"
        );
    }

    #[test]
    fn within_group_auc_ignores_between_group_score_offsets() {
        // Inside each group the ranking is perfect; the groups sit at
        // completely different score levels. Pooled AUC is dragged down by
        // the between-group offset, the grouped one is not.
        let scores = vec![0.1f32, 0.2, 0.8, 0.9];
        let labels = vec![0.0f32, 1.0, 0.0, 1.0];
        let groups = ["a", "a", "b", "b"];
        let (mean, n) = auc_roc_within_groups(&scores, &labels, &groups).unwrap();
        assert_eq!(n, 2);
        assert!((mean - 1.0).abs() < 1e-9, "mean={mean}");
    }

    #[test]
    fn within_group_auc_is_none_when_no_group_holds_both_classes() {
        let scores = vec![0.1f32, 0.2];
        let labels = vec![0.0f32, 0.0];
        let groups = ["a", "b"];
        assert!(auc_roc_within_groups(&scores, &labels, &groups).is_none());
    }

    #[test]
    fn auc_roc_is_one_for_perfectly_separated_scores() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!((auc_roc(&scores, &labels).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_zero_for_perfectly_inverted_scores() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!(auc_roc(&scores, &labels).unwrap().abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_none_when_a_class_is_absent() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![0.0, 0.0, 0.0];
        assert!(auc_roc(&scores, &labels).is_none());
    }

    #[test]
    fn average_precision_is_one_for_perfectly_separated_scores() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!((average_precision(&scores, &labels).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn average_precision_is_none_with_no_positives() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![0.0, 0.0, 0.0];
        assert!(average_precision(&scores, &labels).is_none());
    }

    #[test]
    fn op_index_table_has_one_entry_per_opkind_and_no_duplicate_names() {
        let (names, index) = op_index_table();
        assert_eq!(names.len(), index.len(), "duplicate OpKind Debug names");
        assert!(!names.is_empty());
        for (i, n) in names.iter().enumerate() {
            assert_eq!(index[n], i);
        }
    }

    #[test]
    fn model_zero_start_scores_every_sample_at_the_bias_only() {
        let model = Model::zero(5, 3);
        let s = Sample {
            rule_slot: 2,
            op_feats: vec![(0, 1.5), (2, 0.5)],
            budget_fraction: 0.3,
            log_match_class: 1.0,
            log_neighborhood: 1.0,
            log_expr_size: 1.0,
            label: 1.0,
        };
        assert_eq!(
            model.logit(&s),
            0.0,
            "a zero-initialized model must score zero everywhere"
        );
    }

    #[test]
    fn sgd_step_moves_a_positive_samples_logit_upward() {
        let mut model = Model::zero(3, 2);
        let s = Sample {
            rule_slot: 1,
            op_feats: vec![(0, 1.0)],
            budget_fraction: 0.5,
            log_match_class: 0.5,
            log_neighborhood: 0.5,
            log_expr_size: 0.5,
            label: 1.0,
        };
        let before = model.logit(&s);
        // A positive gradient step (as if pos_weight*(p-1) with p<1, i.e.
        // negative grad_z) should raise the logit; use a concrete negative
        // grad_z directly since sigmoid/loss live in the trainer binary.
        model.sgd_step(&s, -1.0, 0.1, 0.0);
        let after = model.logit(&s);
        assert!(after > before, "before={before} after={after}");
    }
}

// ── Checkpoint: one definition, shared by the trainer and every consumer ─────

/// The saturation Guide's cold-start checkpoint: weights plus everything a
/// consumer needs to know about what they mean, mirroring
/// [`crate::training::mint::MintMetadata`]'s pattern (content-hash schema
/// identity, a named label/objective source, a weights content hash) without
/// reusing that type directly — `MintMetadata` is shaped around bench-timing
/// provenance (`bench_mode`, `sentinel_calibration_ns`, `normalization`) the
/// Guide has none of, since every number in this program is a deterministic
/// `CostModel::latency_prior()` count, never a timing (design doc §0's
/// binding framing). `label_source` is the field
/// `docs/plans/2026-08-31-guide-design-revision.md` §0/§3 calls out by name:
/// recording `"strict-v1"` here is what lets a future label-source swap (to
/// a tightened-labeler variant, §3 option 3's second stage) be validated by
/// comparing checkpoints rather than by assuming which label a given file
/// was trained on.
///
/// It lives here rather than in `train_guide` because three binaries read it
/// (`skew_test_linear_guide`, `phase3_at_budget_eval`, `guide_coverage_table`)
/// and each re-deriving the JSON shape is the drift this module exists to
/// prevent.
///
/// # Rule identity
///
/// `rule_names` holds [`pixelflow_search::egraph::rule_label`]'s canonical
/// labels in `w_rule` slot order, and `rule_fingerprint` is the
/// [`RuleSet::fingerprint`] of the vocabulary they came from. Together they
/// are what makes `w_rule` safe to read on another build: a same-length
/// reorder of `all_rules()` changes the fingerprint, and
/// [`Self::to_weights`] re-keys every weight by [`RuleId`] on the way out,
/// so no consumer ever indexes a weight vector by a position it did not
/// itself compute (docs/plans/2026-09-02-phase3-forward-port.md §2.2).
#[derive(serde::Serialize, Deserialize)]
pub struct GuideCheckpoint {
    pub schema_identity: String,
    pub label_source: String,
    pub trainer: String,
    pub written_at_unix_s: u64,

    pub seed: u64,
    pub epochs: usize,
    pub lr_initial: f32,
    pub lr_decay: f32,
    pub l2: f32,
    pub grad_clip: f32,
    pub pos_weight: f32,

    pub num_rules: usize,
    pub num_ops: usize,
    /// `slot -> canonical rule label`, dense over the vocabulary this run
    /// trained against — the identity half of `w_rule`.
    pub rule_names: Vec<String>,
    /// `RuleSet::fingerprint()` of that vocabulary, as 16 hex digits.
    pub rule_fingerprint: String,
    /// `OpKind::all()` order, i.e. the order `w_op` is indexed in.
    pub op_names: Vec<String>,

    pub bias: f32,
    pub w_rule: Vec<f32>,
    pub w_op: Vec<f32>,
    pub w_budget: f32,
    pub w_match_class: f32,
    pub w_neighborhood: f32,
    pub w_expr_size: f32,

    pub train_samples: usize,
    pub train_families: usize,
    pub train_positive_rate: f64,
    pub dev_samples: usize,
    pub dev_families: usize,
    pub dev_auc: f64,
    pub dev_pr_auc: f64,

    /// Content hash of every field above except this one and
    /// `schema_identity` — binds this JSON file's weights to its own
    /// metadata the way `MintMetadata::weights_fnv64` binds a metadata
    /// sidecar to a separate weights file. Kept in the same file (unlike
    /// `MintMetadata`, which describes an external weights file) because
    /// the Guide's weights are small enough that a separate binary blob buys
    /// nothing; the hash still catches hand-edited or partially-copied
    /// checkpoint files.
    pub weights_fnv64: String,
}

impl crate::schema::SchemaIdentity for GuideCheckpoint {
    const MAGIC: &'static str = "PXGC";
    const SCHEMA: &'static str = "\
        label_source: which hindsight-label variant produced the training targets \
        (e.g. \"strict-v1\" = EpisodeLabels::compute_strict, no ancestry \
        over-approximation) — the field the label-source-swap re-validation in \
        design-doc-2026-08-31 depends on; \
        trainer: which binary wrote these weights; \
        seed/epochs/lr_initial/lr_decay/l2/grad_clip: the SGD run's hyperparameters; \
        pos_weight: inverse-class-frequency weight applied to the LoadBearing class \
        in the training loss; \
        num_rules/num_ops: dense-array lengths for rule_names/w_rule and \
        op_names/w_op; \
        rule_names: w_rule slot -> canonical rule label (rule_label spelling); \
        rule_fingerprint: RuleSet::fingerprint() of the vocabulary trained against, \
        16 hex digits — a loader refuses weights whose fingerprint is not the live \
        rule set's; \
        op_names: OpKind::all() order, i.e. the index space w_op is keyed by; \
        bias/w_rule/w_op/w_budget/w_match_class/w_neighborhood/w_expr_size: the \
        linear model's weights — logit = bias + w_rule[slot of rule_label] + \
        sum_op(hist[op]*w_op[op]) + w_budget*budget_fraction + \
        w_match_class*log1p(match_class_node_count) + \
        w_neighborhood*log1p(neighborhood_op_count) + \
        w_expr_size*log1p(expr_node_count); \
        train_samples/train_families/train_positive_rate: TRAIN-split provenance; \
        dev_samples/dev_families/dev_auc/dev_pr_auc: held-out DEV-family evaluation \
        recorded at write time; \
        weights_fnv64: FNV-1a 64 hex content hash binding this file's metadata to \
        its own weight values";
}

impl GuideCheckpoint {
    #[must_use]
    pub fn current_schema_identity() -> String {
        format!(
            "{:016x}",
            <Self as crate::schema::SchemaIdentity>::SCHEMA_IDENTITY
        )
    }

    /// FNV-1a 64 over the weight-bearing fields only (not the two identity
    /// fields themselves), formatted deterministically. Order and precision
    /// are fixed by this function, not by float `Display`'s platform
    /// behavior, so the same weights always hash the same way.
    #[must_use]
    pub fn weights_fingerprint(&self) -> String {
        let mut buf = String::new();
        buf.push_str(&format!("{:.9}\n", self.bias));
        for w in &self.w_rule {
            buf.push_str(&format!("{w:.9}\n"));
        }
        for w in &self.w_op {
            buf.push_str(&format!("{w:.9}\n"));
        }
        buf.push_str(&format!(
            "{:.9}\n{:.9}\n{:.9}\n{:.9}\n",
            self.w_budget, self.w_match_class, self.w_neighborhood, self.w_expr_size
        ));
        crate::schema::fnv1a64_hex(buf.as_bytes())
    }

    /// Write, stamping the schema identity and the weights hash.
    ///
    /// # Errors
    ///
    /// Serialization or I/O failure.
    pub fn write(&mut self, path: &Path) -> std::io::Result<()> {
        self.schema_identity = Self::current_schema_identity();
        self.weights_fnv64 = self.weights_fingerprint();
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Read and validate a checkpoint written by [`Self::write`].
    ///
    /// # Errors
    ///
    /// I/O failure, a schema-identity mismatch, or a weights hash that does
    /// not match the weights in the file.
    pub fn read(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let meta: Self = serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let expected = Self::current_schema_identity();
        if meta.schema_identity != expected {
            let stored = u64::from_str_radix(&meta.schema_identity, 16).unwrap_or(0);
            return Err(crate::schema::identity_mismatch(
                "guide checkpoint",
                stored,
                <Self as crate::schema::SchemaIdentity>::SCHEMA_IDENTITY,
                "cargo run --release -p pixelflow-pipeline --features training --bin train_guide",
            ));
        }
        let actual = meta.weights_fingerprint();
        if meta.weights_fnv64 != actual {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "guide checkpoint {} metadata/weights mismatch: recorded hash {}, actual {}",
                    path.display(),
                    meta.weights_fnv64,
                    actual
                ),
            ));
        }
        Ok(meta)
    }

    /// The trainer-side model these weights came from — the side of the skew
    /// test that runs `Model::logit`.
    ///
    /// # Panics
    ///
    /// If `rule_names`/`op_names` disagree in length with `w_rule`/`w_op`.
    #[must_use]
    pub fn to_model(&self) -> Model {
        assert_eq!(
            self.rule_names.len(),
            self.w_rule.len(),
            "guide checkpoint: rule_names and w_rule disagree in length"
        );
        assert_eq!(
            self.op_names.len(),
            self.w_op.len(),
            "guide checkpoint: op_names and w_op disagree in length"
        );
        Model {
            bias: self.bias,
            w_rule: self.w_rule.clone(),
            w_op: self.w_op.clone(),
            w_budget: self.w_budget,
            w_match_class: self.w_match_class,
            w_neighborhood: self.w_neighborhood,
            w_expr_size: self.w_expr_size,
        }
    }

    /// The deployed-side weights: `w_rule` re-keyed by [`RuleId`] and `w_op`
    /// by op name, with the trained-against vocabulary's fingerprint
    /// attached so [`LinearCandidateGuide::new`] can refuse a mismatch.
    ///
    /// # Panics
    ///
    /// If the file's `rule_fingerprint` is not 16 hex digits, or the dense
    /// arrays disagree with their name tables.
    #[must_use]
    pub fn to_weights(&self) -> LinearWeights {
        assert_eq!(
            self.rule_names.len(),
            self.w_rule.len(),
            "guide checkpoint: rule_names and w_rule disagree in length"
        );
        assert_eq!(
            self.op_names.len(),
            self.w_op.len(),
            "guide checkpoint: op_names and w_op disagree in length"
        );
        let digest = u64::from_str_radix(&self.rule_fingerprint, 16).unwrap_or_else(|e| {
            panic!(
                "guide checkpoint: rule_fingerprint {:?} is not 16 hex digits ({e}) — a \
                 checkpoint without a readable vocabulary fingerprint cannot be deployed, \
                 because nothing else says which rules its w_rule table names",
                self.rule_fingerprint
            )
        });
        LinearWeights {
            bias: self.bias,
            w_rule: self
                .rule_names
                .iter()
                .zip(&self.w_rule)
                .map(|(label, &w)| (RuleId::from_label(label), w))
                .collect::<BTreeMap<_, _>>(),
            w_op: self
                .op_names
                .iter()
                .cloned()
                .zip(self.w_op.iter().copied())
                .collect::<BTreeMap<_, _>>(),
            w_budget: self.w_budget,
            w_match_class: self.w_match_class,
            w_neighborhood: self.w_neighborhood,
            w_expr_size: self.w_expr_size,
            fingerprint: Fingerprint::from_raw(digest),
        }
    }
}

/// Read a checkpoint and deploy it against `rules`, refusing a vocabulary
/// mismatch loudly.
///
/// # Errors
///
/// I/O or validation failure on the file, or a rule-set fingerprint that is
/// not the live one.
pub fn load_linear_guide(path: &Path, rules: &RuleSet) -> Result<LinearCandidateGuide, String> {
    let ck = GuideCheckpoint::read(path)
        .map_err(|e| format!("guide checkpoint {}: {e}", path.display()))?;
    LinearCandidateGuide::new(ck.to_weights(), rules)
        .map_err(|e| format!("guide checkpoint {}: {e}", path.display()))
}

/// Build the zero-candidate-local-information control arm from a
/// `train_guide` report's `per_rule` table.
///
/// Keyed by the report's `rule` label, hashed straight into a [`RuleId`] —
/// the report's `rule_id` field, when present, is checked against it rather
/// than trusted, so a report whose two halves of the identity disagree stops
/// here.
///
/// # Errors
///
/// I/O failure, malformed JSON, a missing `per_rule` array, or a row without
/// a `rule` / `train_positive_rate` field.
pub fn per_rule_rate_guide_from_report(path: &Path) -> Result<PerRuleRateGuide, String> {
    let p = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| format!("train_guide report {p}: {e}"))?;
    let v: Value =
        serde_json::from_str(&text).map_err(|e| format!("train_guide report {p}: {e}"))?;
    let rows = v
        .get("per_rule")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("train_guide report {p}: no \"per_rule\" array"))?;

    let mut rates: Vec<(String, f32)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        // `train_guide`'s report spells the label `rule_name`;
        // `gen_strict_labels`' spells it `rule`. Both are the same canonical
        // label, so accept either rather than making the two writers agree
        // on a field name they never agreed on.
        let label = row
            .get("rule_name")
            .or_else(|| row.get("rule"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!("train_guide report {p}: per_rule[{i}] has no \"rule_name\"/\"rule\"")
            })?;
        let rate = row
            .get("train_positive_rate")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!("train_guide report {p}: per_rule[{i}] has no \"train_positive_rate\"")
            })? as f32;
        if let Some(id) = row.get("rule_id").and_then(Value::as_u64)
            && RuleId::from_label(label).get() != id
        {
            return Err(format!(
                "train_guide report {p}: per_rule[{i}] names rule {label:?} but carries \
                 rule_id {id} — the label and the id in one row must be the same identity"
            ));
        }
        rates.push((label.to_string(), rate));
    }
    Ok(PerRuleRateGuide::from_labels(&rates))
}

// ── Checkpoint round trip ───────────────────────────────────────────────────

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    fn ckpt_for(rules: &RuleSet) -> GuideCheckpoint {
        let (rule_names, _) = rule_index_table(rules);
        let (op_names, _) = op_index_table();
        GuideCheckpoint {
            schema_identity: String::new(),
            label_source: "strict-v1".to_string(),
            trainer: "train_guide".to_string(),
            written_at_unix_s: 1,
            seed: 1,
            epochs: 1,
            lr_initial: 0.01,
            lr_decay: 0.0,
            l2: 0.0,
            grad_clip: 20.0,
            pos_weight: 1.0,
            num_rules: rule_names.len(),
            num_ops: op_names.len(),
            w_rule: vec![0.25; rule_names.len()],
            w_op: vec![-0.125; op_names.len()],
            rule_names,
            rule_fingerprint: format!("{}", rules.fingerprint()),
            op_names,
            bias: 0.5,
            w_budget: 0.125,
            w_match_class: 0.0625,
            w_neighborhood: -0.031_25,
            w_expr_size: 0.015_625,
            train_samples: 1,
            train_families: 1,
            train_positive_rate: 0.0,
            dev_samples: 1,
            dev_families: 1,
            dev_auc: 0.5,
            dev_pr_auc: 0.5,
            weights_fnv64: String::new(),
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("guide_ckpt_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn a_written_checkpoint_round_trips_and_deploys_against_the_vocabulary_it_names() {
        let dir = scratch("roundtrip");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        ckpt.write(&path).expect("write");

        let loaded = GuideCheckpoint::read(&path).expect("read");
        assert_eq!(loaded.label_source, "strict-v1");
        assert_eq!(
            loaded.schema_identity,
            GuideCheckpoint::current_schema_identity()
        );
        load_linear_guide(&path, &rules)
            .expect("a checkpoint just written must deploy against the rule set it names");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The G5 property: a weight table is bound to the vocabulary it was
    /// trained on by a fingerprint, not by array length. A same-length
    /// reorder is exactly the case a length check cannot see.
    #[test]
    fn a_checkpoint_from_another_vocabulary_is_refused_at_deploy_time() {
        let dir = scratch("wrong_vocab");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        // Same rules, reversed order: same length, same names, different
        // fingerprint.
        ckpt.rule_fingerprint = {
            let mut reversed = pixelflow_search::egraph::all_rules();
            reversed.reverse();
            format!("{}", RuleSet::new(reversed).fingerprint())
        };
        ckpt.write(&path).expect("write");

        let err = load_linear_guide(&path, &rules)
            .expect_err("weights from a reordered vocabulary must be refused");
        assert!(
            err.contains("vocabulary changed") || err.contains("trained against"),
            "the refusal should say the vocabulary disagrees: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_hand_edited_weight_is_refused_by_its_own_content_hash() {
        let dir = scratch("tamper");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        ckpt.write(&path).expect("write");

        let text = std::fs::read_to_string(&path).expect("read back");
        let tampered = text.replace("\"bias\": 0.5", "\"bias\": 99.0");
        assert_ne!(
            text, tampered,
            "test setup: replacement should have matched"
        );
        std::fs::write(&path, tampered).expect("write tampered");

        let err = load_linear_guide(&path, &rules)
            .expect_err("a hand-edited weight must be refused, not silently loaded");
        assert!(
            err.contains("metadata/weights mismatch"),
            "the refusal should name the content hash: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_checkpoint_without_a_readable_fingerprint_is_refused_rather_than_defaulted() {
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        ckpt.rule_fingerprint = "not-hex".to_string();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ckpt.to_weights()));
        assert!(
            caught.is_err(),
            "an unreadable rule fingerprint must panic, not deploy as fingerprint 0"
        );
    }
}
