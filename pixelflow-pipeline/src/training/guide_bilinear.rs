//! Checkpoint format and record→candidate encoding for the **bilinear**
//! saturation Guide (`docs/archive/plans/2026-09-02-bilinear-guide-registration.md`).
//!
//! Shared by `train_guide_bilinear` (writes) and `skew_test_bilinear_guide`
//! (reads), for the same reason `guide_linear` is shared by the additive
//! trainer and its skew test: a skew test that used its own copy of "what a
//! JSONL row means" or "what a checkpoint field is" would check nothing —
//! both sides would be wrong together.
//!
//! Parsing lives here rather than in `pixelflow-search` (which owns
//! [`BilinearWeights`] as a parsed value and owns the *refusal*), exactly as
//! `guide_linear` does for the additive head: `pixelflow-pipeline` already
//! depends on `serde_json`, and adding it to `pixelflow-search` for a loader
//! only the training crate calls would widen that crate's dependency surface
//! for nobody's benefit.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{RuleId, RuleSet};
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::CandidateSummary;
use pixelflow_search::nnue::guide::bilinear::BilinearWeights;

use crate::schema::{SchemaIdentity, fnv1a64_hex};
use crate::training::guide_linear::Record;

/// Rebuild a `Vec<OpKind>` whose per-op occurrence counts match a JSONL
/// row's `neighborhood_op_hist` exactly.
///
/// Order is irrelevant to every consumer — the additive model re-groups and
/// counts, the bilinear tower sums a bag of embeddings — but the *counts*
/// must round-trip, so this repeats each op its histogram count.
///
/// # Panics
///
/// If a histogram key is not an `OpKind::all()` variant name: the dataset
/// was minted against a different `OpKind` table than this binary was built
/// with, and every op-keyed feature would be silently shifted.
#[must_use]
pub fn ops_from_histogram(hist: &BTreeMap<String, usize>) -> Vec<OpKind> {
    let by_name: std::collections::HashMap<String, OpKind> =
        OpKind::all().map(|op| (format!("{op:?}"), op)).collect();
    let mut ops = Vec::new();
    for (name, &count) in hist {
        let op = *by_name.get(name).unwrap_or_else(|| {
            panic!(
                "guide_bilinear: neighborhood_op_hist names op {name:?}, which is not in \
                 OpKind::all() — the dataset was minted against a different OpKind table \
                 than this binary was built with; regenerate the dataset"
            )
        });
        ops.extend(std::iter::repeat_n(op, count));
    }
    ops
}

/// One strict-label row as the bilinear head consumes it: a
/// [`CandidateSummary`] missing only its `rule_embed` (which depends on the
/// model's current weights and so cannot be baked into the dataset), plus
/// the label.
pub struct BilinearSample {
    pub rule: RuleId,
    pub neighborhood_ops: Vec<OpKind>,
    pub budget_fraction: f32,
    pub match_class_node_count: usize,
    pub expr_node_count: usize,
    pub label: f32,
}

impl BilinearSample {
    /// Encode one JSONL [`Record`].
    ///
    /// The rule is read back through [`RuleId::from_label`] and checked
    /// against the row's own `rule_id`, the same cross-check
    /// `guide_linear::to_sample` performs: a row whose two halves of the
    /// identity disagree stops here rather than training a weight onto the
    /// wrong rule.
    #[must_use]
    pub fn from_record(record: &Record) -> Self {
        let rule = RuleId::from_label(&record.rule_name);
        assert_eq!(
            rule.get(),
            record.rule_id,
            "guide_bilinear: row names rule {:?} but carries rule_id {} — the label and \
             the id in one strict-label row must be the same identity",
            record.rule_name,
            record.rule_id,
        );
        Self {
            rule,
            neighborhood_ops: ops_from_histogram(&record.neighborhood_op_hist),
            budget_fraction: record.budget_fraction,
            match_class_node_count: record.match_class_node_count,
            expr_node_count: record.expr_node_count,
            label: if record.label_positive { 1.0 } else { 0.0 },
        }
    }

    /// The [`CandidateSummary`] a guide scores, with `rule_embed` supplied
    /// by the model.
    #[must_use]
    pub fn to_summary(&self, rule_embed: [f32; EMBED_DIM]) -> CandidateSummary {
        CandidateSummary {
            rule_embed,
            neighborhood_ops: self.neighborhood_ops.clone(),
            budget_fraction: self.budget_fraction,
            rule: self.rule,
            match_class_node_count: self.match_class_node_count,
            expr_node_count: self.expr_node_count,
        }
    }
}

/// A trained bilinear Guide on disk.
///
/// Carries the trained parameters, the frozen op embeddings, and the
/// vocabulary's identity — and, deliberately, **no per-rule embeddings**:
/// those are `rule_proj(concat(templates))`, derived identically by the
/// trainer and by `BilinearCandidateGuide::new`. A derived value in a
/// checkpoint is a second copy that can disagree with the first.
#[derive(Serialize, Deserialize)]
pub struct BilinearGuideCheckpoint {
    pub schema_identity: String,
    pub label_source: String,
    pub trainer: String,
    pub written_at_unix_s: u64,

    pub seed: u64,
    pub epochs: usize,
    pub lr_initial: f32,
    pub lr_decay: f32,
    pub l2: f32,
    pub max_grad_norm: f32,
    pub pos_weight: f32,

    /// `RuleSet::fingerprint()` of the vocabulary trained against, as 16 hex
    /// digits — a loader refuses parameters whose fingerprint is not the
    /// live rule set's.
    pub rule_fingerprint: String,
    pub num_rules: usize,
    /// `OpKind::all()` order, i.e. the row order of `op_embeddings`.
    pub op_names: Vec<String>,

    /// `SaturationHead`'s trained parameters, flat, in the head's own
    /// canonical order (`BilinearWeights::parameter_count()` floats).
    pub parameters: Vec<f32>,
    /// The frozen op embeddings, `op_names.len() * K` floats, row-major.
    pub op_embeddings: Vec<f32>,

    pub train_samples: usize,
    pub train_families: usize,
    pub train_positive_rate: f64,
    pub dev_samples: usize,
    pub dev_families: usize,
    pub dev_auc: f64,
    pub dev_pr_auc: f64,

    /// FNV-1a 64 over the two weight blocks — catches a hand-edited or
    /// partially-copied checkpoint that still parses.
    pub weights_fnv64: String,
}

impl SchemaIdentity for BilinearGuideCheckpoint {
    const MAGIC: &'static str = "PXBG";
    const SCHEMA: &'static str = "\
        label_source: which hindsight-label variant produced the training targets; \
        trainer: which binary wrote these weights; \
        seed/epochs/lr_initial/lr_decay/l2/max_grad_norm: the SGD run's \
        hyperparameters — max_grad_norm bounds the Euclidean norm of the whole \
        accumulated gradient (the deep-net analogue of the additive trainer's \
        grad_clip on dLoss/dz); \
        pos_weight: inverse-class-frequency weight applied to the LoadBearing class \
        in the training loss, identical in definition to the additive trainer's; \
        rule_fingerprint: RuleSet::fingerprint() of the vocabulary trained against, \
        16 hex digits — a loader refuses parameters whose fingerprint is not the \
        live rule set's; \
        num_rules: how many rules that vocabulary held; \
        op_names: OpKind::all() order, i.e. the row order of op_embeddings; \
        parameters: SaturationHead's trained parameters, flat, in the head's own \
        canonical visitor order (candidate tower, trunk, candidate projection, mask \
        MLP, bilinear interaction and bias lane, rule projection); \
        op_embeddings: the FROZEN op embeddings the neighborhood and the rule \
        templates are pooled in, op_names.len() * K floats row-major — carried \
        rather than reconstructed from the seed so a later edit to the shared \
        latency-prior table cannot change a deployed checkpoint's behaviour \
        without changing the checkpoint; \
        train_samples/train_families/train_positive_rate: TRAIN-split provenance; \
        dev_samples/dev_families/dev_auc/dev_pr_auc: held-out DEV-family evaluation \
        recorded at write time; \
        weights_fnv64: FNV-1a 64 hex content hash over parameters and op_embeddings";
}

impl BilinearGuideCheckpoint {
    #[must_use]
    pub fn current_schema_identity() -> String {
        format!("{:016x}", <Self as SchemaIdentity>::SCHEMA_IDENTITY)
    }

    /// FNV-1a 64 over the two weight blocks, at a fixed precision and order
    /// so the same weights always hash the same way.
    #[must_use]
    pub fn weights_fingerprint(&self) -> String {
        let mut buf = String::new();
        for w in &self.parameters {
            buf.push_str(&format!("{w:.9}\n"));
        }
        for w in &self.op_embeddings {
            buf.push_str(&format!("{w:.9}\n"));
        }
        fnv1a64_hex(buf.as_bytes())
    }

    /// The parsed value `pixelflow-search` deploys, with this file's own
    /// integrity checked first.
    ///
    /// # Panics
    ///
    /// If the schema identity, the weight hash, or the rule fingerprint's
    /// spelling does not check out. Every one of those is a corrupted or
    /// mismatched checkpoint, and a Guide that scored with one would corrupt
    /// move ordering without saying so.
    #[must_use]
    pub fn to_weights(&self, rules: &RuleSet) -> BilinearWeights {
        let want_schema = Self::current_schema_identity();
        assert_eq!(
            self.schema_identity, want_schema,
            "guide_bilinear: checkpoint schema identity {} is not this build's {want_schema} \
             — the checkpoint format changed; retrain rather than reinterpret its fields",
            self.schema_identity
        );
        let want_hash = self.weights_fingerprint();
        assert_eq!(
            self.weights_fnv64, want_hash,
            "guide_bilinear: checkpoint weights hash {} does not match its own contents \
             ({want_hash}) — the file was edited or partially copied",
            self.weights_fnv64
        );
        // The fingerprint is compared here as text and again inside
        // `BilinearCandidateGuide::new` as a value; this one catches a
        // checkpoint written by a build whose `Fingerprint` display width
        // differs, which would otherwise parse to a different number.
        let live = format!("{}", rules.fingerprint());
        assert_eq!(
            self.rule_fingerprint.len(),
            live.len(),
            "guide_bilinear: checkpoint rule fingerprint {:?} is not the same shape as this \
             build's {live:?}",
            self.rule_fingerprint
        );
        BilinearWeights {
            parameters: self.parameters.clone(),
            op_embeddings: self.op_embeddings.clone(),
            fingerprint: parse_fingerprint(&self.rule_fingerprint),
        }
    }
}

/// Load a trained bilinear checkpoint from `path` and deploy it against
/// `rules`.
///
/// The counterpart of [`crate::training::guide_linear::load_linear_guide`]
/// for the bilinear arm, and the only path any evaluation harness uses: the
/// refusals live in [`BilinearCandidateGuide::new`] and in
/// [`BilinearGuideCheckpoint::to_weights`], so a second hand-rolled reader
/// in a binary would be a second place for one of them to be skipped.
///
/// # Errors
///
/// Returns the reason, naming `path`, when the file is unreadable, is not a
/// checkpoint, or carries weights this build refuses to deploy (vocabulary
/// fingerprint, parameter count, or op-embedding count).
pub fn load_bilinear_guide(
    path: &std::path::Path,
    rules: &RuleSet,
) -> Result<pixelflow_search::nnue::guide::bilinear::BilinearCandidateGuide, String> {
    let p = path.display().to_string();
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("bilinear checkpoint {p}: {e}"))?;
    let checkpoint: BilinearGuideCheckpoint =
        serde_json::from_str(&text).map_err(|e| format!("bilinear checkpoint {p}: {e}"))?;
    let weights = checkpoint.to_weights(rules);
    pixelflow_search::nnue::guide::bilinear::BilinearCandidateGuide::new(&weights, rules)
        .map_err(|e| format!("bilinear checkpoint {p}: {e}"))
}

/// Read a `RuleSet::fingerprint()` back from its hex spelling, by finding
/// the live vocabulary whose fingerprint prints the same way.
///
/// `Fingerprint` has no public constructor from a `u64` — deliberately, so
/// nothing can mint one — so a checkpoint's fingerprint is compared by
/// *spelling* against the live rule set's, and the live value is what is
/// carried forward. That keeps `BilinearCandidateGuide::new`'s refusal
/// meaningful: a checkpoint from another vocabulary produces a fingerprint
/// this function refuses to invent, and the load fails here.
fn parse_fingerprint(text: &str) -> pixelflow_search::egraph::Fingerprint {
    let live = RuleSet::production();
    let live_fp = live.fingerprint();
    assert_eq!(
        format!("{live_fp}"),
        text,
        "guide_bilinear: checkpoint was trained against rule set {text} but this build's \
         production vocabulary is {live_fp} — the vocabulary changed since training, so \
         every derived rule embedding would name a different rule. Retrain rather than \
         deploy."
    );
    live_fp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_from_histogram_round_trips_per_op_counts() {
        let mut hist = BTreeMap::new();
        hist.insert("Add".to_string(), 3usize);
        hist.insert("Mul".to_string(), 1usize);
        let ops = ops_from_histogram(&hist);
        assert_eq!(ops.iter().filter(|&&o| o == OpKind::Add).count(), 3);
        assert_eq!(ops.iter().filter(|&&o| o == OpKind::Mul).count(), 1);
        assert_eq!(ops.len(), 4);
    }

    #[test]
    #[should_panic(expected = "not in OpKind::all()")]
    fn ops_from_histogram_panics_loudly_on_an_unknown_op_name() {
        let mut hist = BTreeMap::new();
        hist.insert("NotARealOp".to_string(), 1usize);
        let _refused = ops_from_histogram(&hist);
    }

    fn checkpoint() -> BilinearGuideCheckpoint {
        let rules = RuleSet::production();
        let trainer = pixelflow_search::nnue::guide::bilinear::BilinearTrainer::new_cold(&rules, 4);
        let w = trainer.weights();
        let mut c = BilinearGuideCheckpoint {
            schema_identity: BilinearGuideCheckpoint::current_schema_identity(),
            label_source: "strict-v1".into(),
            trainer: "test".into(),
            written_at_unix_s: 0,
            seed: 4,
            epochs: 0,
            lr_initial: 0.0,
            lr_decay: 0.0,
            l2: 0.0,
            max_grad_norm: 1.0,
            pos_weight: 1.0,
            rule_fingerprint: format!("{}", rules.fingerprint()),
            num_rules: rules.len(),
            op_names: OpKind::all().map(|op| format!("{op:?}")).collect(),
            parameters: w.parameters,
            op_embeddings: w.op_embeddings,
            train_samples: 0,
            train_families: 0,
            train_positive_rate: 0.0,
            dev_samples: 0,
            dev_families: 0,
            dev_auc: 0.0,
            dev_pr_auc: 0.0,
            weights_fnv64: String::new(),
        };
        c.weights_fnv64 = c.weights_fingerprint();
        c
    }

    #[test]
    fn a_checkpoint_should_round_trip_through_json_bit_exactly() {
        // Float formatting is the whole risk here: a checkpoint whose
        // weights come back one ULP different fails the mandatory skew test
        // for a reason that has nothing to do with the model.
        let c = checkpoint();
        let text = serde_json::to_string(&c).expect("serialize");
        let back: BilinearGuideCheckpoint = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.parameters, c.parameters);
        assert_eq!(back.op_embeddings, c.op_embeddings);
        assert_eq!(back.weights_fingerprint(), c.weights_fnv64);
    }

    #[test]
    #[should_panic(expected = "does not match its own contents")]
    fn to_weights_should_refuse_a_checkpoint_whose_weights_were_edited() {
        let mut c = checkpoint();
        c.parameters[7] += 1.0;
        let _refused = c.to_weights(&RuleSet::production());
    }

    #[test]
    #[should_panic(expected = "is not this build's")]
    fn to_weights_should_refuse_a_checkpoint_from_another_schema() {
        let mut c = checkpoint();
        c.schema_identity = "0000000000000000".into();
        let _refused = c.to_weights(&RuleSet::production());
    }

    #[test]
    #[should_panic(expected = "vocabulary changed since training")]
    fn to_weights_should_refuse_a_checkpoint_from_another_vocabulary() {
        let mut c = checkpoint();
        // Same width, different value — the case a length check alone would
        // wave through.
        let mut chars: Vec<char> = c.rule_fingerprint.chars().collect();
        chars[0] = if chars[0] == 'a' { 'b' } else { 'a' };
        c.rule_fingerprint = chars.into_iter().collect();
        let _refused = c.to_weights(&RuleSet::production());
    }
}
