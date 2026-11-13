//! Cold-start supervised training of the **bilinear** saturation Guide, on
//! the same strict-v1 labels and the same recipe `train_guide` used for the
//! additive one (`docs/archive/plans/2026-09-02-bilinear-guide-registration.md` §9).
//!
//! # The whole point: one difference, and it is the functional form
//!
//! This binary exists so the comparison in the registration is
//! functional-form-vs-functional-form. Everything it can hold equal with
//! `train_guide`, it holds equal:
//!
//! | held equal | how |
//! |---|---|
//! | label source | the same `strict_labels_train.jsonl` / `_dev.jsonl` |
//! | candidate features | the same five context quantities off the same `CandidateSummary` — the neighborhood op multiset (pooled here, counted there) plus `budget_fraction`, `log1p(match_class_node_count)`, `log1p(neighborhood_op_count)`, `log1p(expr_node_count)` |
//! | dedup filtering | `dedup_repeat` rows excluded, reported, never silent |
//! | class weighting | `pos_weight = negatives / positives` measured on the TRAIN split actually used |
//! | loss | weighted BCE-with-logits, the same stable form |
//! | cold start | no warm start from any prior checkpoint |
//! | TRAIN-only | DEV never touches a gradient, never selects a hyperparameter |
//!
//! Two things could **not** be held equal, and both are recorded here and in
//! the results document rather than buried:
//!
//! 1. **Gradient clipping is on the step, not on `dLoss/dz`.** `train_guide`
//!    clips `dLoss/dz` to ±20, the right primitive for a one-layer convex
//!    model whose update is `lr · dz · x`. This head is four layers with a
//!    bilinear top, where the same `dz` produces wildly different
//!    weight-space steps depending on the activations. `--max-grad-norm`
//!    bounds the Euclidean norm of the whole accumulated gradient instead.
//!    The *objective* is byte-identical; only the optimiser is bounded
//!    differently.
//! 2. **The learning rate is selected on a TRAIN-internal holdout.** The
//!    additive model's `--lr 0.01` was not tuned against this architecture
//!    and there is no reason it should transfer. `--lr-grid` fits one model
//!    per candidate rate on TRAIN-minus-holdout and keeps the best holdout
//!    PR-AUC — registration §9: "Model selection happens on a TRAIN-internal
//!    holdout or it does not happen." DEV is not read until the run is over.
//!
//! # Zero-predictor floors are printed next to every metric
//!
//! The strict-positive rate is ~0.6%, so "PR-AUC 0.46" means nothing without
//! "a constant predictor scores 0.0059". Both numbers appear side by side in
//! every report this binary writes, for both arms.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin train_guide_bilinear -- \
//!     --train pixelflow-pipeline/data/strict_labels_train.jsonl \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --out-checkpoint pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json \
//!     --report-json docs/results/2026-09-02-train-guide-bilinear-report.json \
//!     --report-md docs/results/2026-09-02-train-guide-bilinear-report.md
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use clap::Parser;
use serde::Serialize;

use pixelflow_ir::OpKind;
use pixelflow_pipeline::schema::{fnv1a64_hex, unix_now_s};
use pixelflow_pipeline::training::guide_bilinear::{BilinearGuideCheckpoint, BilinearSample};
use pixelflow_pipeline::training::guide_linear::{
    Record, auc_roc, auc_roc_within_groups, average_precision,
};
use pixelflow_search::egraph::{RuleId, RuleSet};
use pixelflow_search::nnue::guide::bilinear::{
    BilinearCandidateGuide, BilinearTrainer, BilinearWeights, SgdStep,
};
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

/// DEV records the write-side skew check replays. The registration's bar is
/// >= 1000; this is the same population `skew_test_bilinear_guide` uses.
const SKEW_RECORDS: usize = 5000;
/// The registration's skew tolerance (§9).
const SKEW_TOL: f64 = 1e-6;

#[derive(Parser)]
#[command(name = "train_guide_bilinear")]
#[command(about = "Cold-start supervised training of the bilinear saturation Guide")]
struct Args {
    /// TRAIN-family strict-label JSONL (`gen_strict_labels`'s `--out-train`).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_train.jsonl"
    )]
    train: String,

    /// DEV-family strict-label JSONL — held out, evaluated only.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    dev: String,

    /// Training epochs (full passes over the shuffled TRAIN set).
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Candidate learning rates, comma-separated. More than one triggers the
    /// TRAIN-internal holdout selection described in this file's module doc.
    #[arg(long, default_value = "0.03,0.01,0.003,0.001")]
    lr_grid: String,

    /// Inverse-time learning-rate decay: `lr_t = lr0 / (1 + lr_decay * t)`.
    #[arg(long, default_value_t = 0.05)]
    lr_decay: f32,

    /// L2 weight decay applied to every weight (not the biases) each step.
    #[arg(long, default_value_t = 1e-4)]
    l2: f32,

    /// Seed for the cold-start head init and the per-epoch shuffle.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Bound on the accumulated gradient's Euclidean norm — see the module
    /// doc for why this replaces `train_guide`'s `--grad-clip`.
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f32,

    /// Fraction of TRAIN *families* reserved for learning-rate selection.
    /// Never trained on during selection; folded back in for the final fit.
    #[arg(long, default_value_t = 0.2)]
    holdout_fraction: f64,

    /// Epochs used per candidate learning rate during selection.
    ///
    /// Defaults to `--epochs` when left at 0, and that is the intended
    /// setting: a shorter selection run ranks a *different* model from the
    /// one the final fit produces, and with an inverse-time decayed rate the
    /// ranking after 5 epochs and after 30 are genuinely different questions
    /// (measured: a 5-epoch selection preferred a rate that was worse at 30).
    /// Selection must rank the model that will be built.
    #[arg(long, default_value_t = 0)]
    select_epochs: usize,

    /// Evaluate held-out DEV every this many epochs of the final fit.
    #[arg(long, default_value_t = 2)]
    eval_every: usize,

    /// Where to write the trained checkpoint.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json"
    )]
    out_checkpoint: String,

    /// Optional JSON report.
    #[arg(long)]
    report_json: Option<String>,

    /// Optional Markdown report.
    #[arg(long)]
    report_md: Option<String>,
}

// ── Data ─────────────────────────────────────────────────────────────────────

struct LoadedSplit {
    samples: Vec<BilinearSample>,
    /// Parallel to `samples`: which expression each came from — the decision
    /// set a live guide actually ranks within.
    groups: Vec<String>,
    /// Parallel to `samples`: which generator family, for the holdout split.
    families: Vec<(u32, u64)>,
    distinct_families: HashSet<(u32, u64)>,
    /// rule -> (fired, positive), measured directly from this split.
    per_rule: BTreeMap<RuleId, (usize, usize)>,
    positives: usize,
    repeats_excluded: usize,
}

fn load_jsonl(path: &Path) -> LoadedSplit {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("train_guide_bilinear: cannot open {}: {e}", path.display()));
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    let mut groups = Vec::new();
    let mut families = Vec::new();
    let mut distinct_families = HashSet::new();
    let mut per_rule: BTreeMap<RuleId, (usize, usize)> = BTreeMap::new();
    let mut positives = 0usize;
    let mut repeats_excluded = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            panic!(
                "train_guide_bilinear: read error on {} line {}: {e}",
                path.display(),
                line_no + 1
            )
        });
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line).unwrap_or_else(|e| {
            panic!(
                "train_guide_bilinear: malformed JSONL at {} line {}: {e}",
                path.display(),
                line_no + 1
            )
        });
        // The live guided loop removes seen keys BEFORE scoring, so a repeat
        // row is a candidate no deployed guide will ever be asked to rank.
        // Excluded here exactly as `train_guide` excludes it — the two arms
        // must see the same population.
        if record.dedup_repeat {
            repeats_excluded += 1;
            continue;
        }
        let family = (record.family_band, record.family_seed);
        distinct_families.insert(family);
        let sample = BilinearSample::from_record(&record);
        let entry = per_rule.entry(sample.rule).or_insert((0, 0));
        entry.0 += 1;
        if record.label_positive {
            entry.1 += 1;
            positives += 1;
        }
        groups.push(record.expr_name.clone());
        families.push(family);
        samples.push(sample);
    }

    eprintln!(
        "train_guide_bilinear: {} — {} first-seen candidates kept, {repeats_excluded} \
         dedup_repeat rows excluded (a deployed guide never scores them)",
        path.display(),
        samples.len()
    );
    assert!(
        !samples.is_empty(),
        "train_guide_bilinear: {} contained zero first-seen candidate records — refusing \
         to train/evaluate on an empty split",
        path.display()
    );

    LoadedSplit {
        samples,
        groups,
        families,
        distinct_families,
        per_rule,
        positives,
        repeats_excluded,
    }
}

// ── Loss (identical in definition to `train_guide`'s) ────────────────────────

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Weighted BCE-with-logits, PyTorch's stable form — the same expression
/// `train_guide` reports its loss with.
fn weighted_bce_loss(z: f32, y: f32, pos_weight: f32) -> f32 {
    let log_weight = 1.0 + (pos_weight - 1.0) * y;
    (1.0 - y) * z + log_weight * ((-z.abs()).exp().ln_1p() + (-z).max(0.0))
}

/// `dLoss/dz`: `pos_weight*(p-1)` for a positive label, `p` for a negative.
fn weighted_bce_grad(p: f32, y: f32, pos_weight: f32) -> f32 {
    if y > 0.5 { pos_weight * (p - 1.0) } else { p }
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        let r = (lcg_next(&mut state) >> 33) as usize % (i + 1);
        idx.swap(i, r);
    }
    idx
}

// ── Training ─────────────────────────────────────────────────────────────────

/// One epoch over `order`, returning the mean weighted loss and the fraction
/// of steps whose gradient norm was clipped.
fn train_epoch(
    trainer: &mut BilinearTrainer,
    split: &LoadedSplit,
    order: &[usize],
    fit: FitParams,
) -> (f64, f64) {
    let mut loss_sum = 0.0f64;
    let mut clipped = 0usize;
    for &i in order {
        let s = &split.samples[i];
        let summary = s.to_summary(trainer.rule_embed(s.rule));
        let forward = trainer.forward(&summary);
        let z = forward.score();
        loss_sum += f64::from(weighted_bce_loss(z, s.label, fit.pos_weight));
        let d_score = weighted_bce_grad(sigmoid(z), s.label, fit.pos_weight);
        trainer.accumulate(&forward, d_score);
        let norm = trainer.apply(fit.step);
        if norm > fit.step.max_grad_norm {
            clipped += 1;
        }
    }
    (
        loss_sum / order.len() as f64,
        clipped as f64 / order.len() as f64,
    )
}

/// The parameters one epoch of fitting needs, grouped so `train_epoch` stays
/// at four arguments.
#[derive(Clone, Copy)]
struct FitParams {
    step: SgdStep,
    pos_weight: f32,
}

/// Score every sample in `order` with the current model.
fn score_all(
    trainer: &BilinearTrainer,
    split: &LoadedSplit,
    order: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut scores = Vec::with_capacity(order.len());
    let mut labels = Vec::with_capacity(order.len());
    for &i in order {
        let s = &split.samples[i];
        let summary = s.to_summary(trainer.rule_embed(s.rule));
        scores.push(trainer.score(&summary));
        labels.push(s.label);
    }
    (scores, labels)
}

/// Split TRAIN's indices into (fit, holdout) at **family** granularity — the
/// same fence granularity the corpus split manifest uses, so a holdout
/// expression never shares a generator family with a fitted one.
fn family_holdout(split: &LoadedSplit, fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut families: Vec<(u32, u64)> = split.distinct_families.iter().copied().collect();
    families.sort_unstable();
    let order = shuffled_indices(families.len(), seed);
    let n_holdout = ((families.len() as f64 * fraction).round() as usize).max(1);
    let held: HashSet<(u32, u64)> = order[..n_holdout].iter().map(|&i| families[i]).collect();

    let mut fit = Vec::new();
    let mut holdout = Vec::new();
    for (i, family) in split.families.iter().enumerate() {
        if held.contains(family) {
            holdout.push(i);
        } else {
            fit.push(i);
        }
    }
    (fit, holdout)
}

// ── Report ───────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EpochPoint {
    epoch: usize,
    lr: f32,
    train_mean_weighted_loss: f64,
    clip_rate: f64,
    dev_auc: Option<f64>,
    dev_pr_auc: Option<f64>,
}

#[derive(Serialize)]
struct LrTrial {
    lr: f32,
    holdout_auc: Option<f64>,
    holdout_pr_auc: Option<f64>,
}

#[derive(Serialize)]
struct RuleRow {
    rule_id: String,
    rule_name: String,
    train_fired: usize,
    train_positive: usize,
    train_positive_rate: f64,
    dev_fired: usize,
    dev_positive: usize,
    dev_positive_rate: f64,
    /// Mean `sigmoid(score)` over this rule's DEV candidates.
    ///
    /// The sigmoid is applied **only here**, so this column is directly
    /// comparable with `train_guide`'s table of the same name: the model is
    /// fit with BCE-with-logits, so its raw score is a logit and the sigmoid
    /// is the probability it was calibrated to. Ranking metrics above are
    /// computed on the raw score, which is the same ordering (the sigmoid is
    /// monotone) and is what a deployed guide sorts by.
    dev_mean_predicted: f64,
}

#[derive(Serialize)]
struct Report {
    label_source: String,
    model_class: String,
    dev_path: String,
    dev_fnv64: String,
    train_path: String,
    train_fnv64: String,

    pos_weight: f32,
    pos_weight_rationale: String,

    train_samples: usize,
    train_families: usize,
    train_positive_rate: f64,
    train_dedup_repeats_excluded: usize,
    dev_samples: usize,
    dev_families: usize,
    dev_positive_rate: f64,
    dev_dedup_repeats_excluded: usize,

    dev_auc: f64,
    dev_pr_auc: f64,
    /// The constant predictor's scores: AUC is exactly 0.5 for any constant
    /// (no pair is ordered), and average precision is the positive rate.
    zero_predictor_auc: f64,
    zero_predictor_pr_auc: f64,
    dev_auc_within_expression: Option<f64>,
    dev_within_expression_groups: usize,

    /// Selection, on TRAIN only.
    lr_selected: f32,
    lr_trials: Vec<LrTrial>,
    holdout_fraction: f64,
    holdout_families: usize,

    epochs: usize,
    lr_decay: f32,
    l2: f32,
    max_grad_norm: f32,
    seed: u64,
    training_curve: Vec<EpochPoint>,

    /// Rule-encoder diagnostics — see the recorded-confound discussion in
    /// `nnue::guide::bilinear`'s module doc.
    rule_encoding_rank: usize,
    rule_count: usize,
    templateless_rules: Vec<String>,
    indistinguishable_rule_pairs: Vec<(String, String)>,

    /// The registered quantity the additive arm pins at exactly zero: the
    /// rule-by-context second difference, measured on the trained model over
    /// a trig-shaped and a polynomial-shaped neighborhood. Zero here would
    /// mean training collapsed the head into an additively separable one.
    trained_second_difference: f32,

    /// The write-side train/deploy skew this run measured: the trained head
    /// in memory against the head loaded back from the checkpoint it just
    /// wrote, over `write_side_skew_records` DEV records. Reported here
    /// because `skew_test_bilinear_guide` — the committed artifact — reads
    /// the file on both sides and so cannot see a flatten-order bug.
    write_side_skew_records: usize,
    write_side_skew_max_abs_diff: f64,

    per_rule: Vec<RuleRow>,
    checkpoint_path: String,
    checkpoint_md5_note: String,
}

fn write_json_report(path: &str, report: &Report) {
    let json = serde_json::to_string_pretty(report)
        .unwrap_or_else(|e| panic!("train_guide_bilinear: report serialization failed: {e}"));
    std::fs::write(path, json)
        .unwrap_or_else(|e| panic!("train_guide_bilinear: cannot write {path}: {e}"));
    eprintln!("train_guide_bilinear: wrote {path}");
}

fn write_md_report(path: &str, report: &Report) {
    use std::fmt::Write as _;
    let md = (|| -> Result<String, std::fmt::Error> {
        let mut md = String::new();
        md.push_str("# Bilinear Guide cold-start training report (strict-v1 labels)\n\n");
        writeln!(
            md,
            "Model class: **{}**. Registration: \
             `docs/archive/plans/2026-09-02-bilinear-guide-registration.md`.\n",
            report.model_class
        )?;
        writeln!(
            md,
            "TRAIN: {} samples, {} families, positive rate {:.4}% \
             ({} dedup_repeat rows excluded).\n",
            report.train_samples,
            report.train_families,
            report.train_positive_rate * 100.0,
            report.train_dedup_repeats_excluded
        )?;
        writeln!(
            md,
            "DEV (held out, never trained on, never used to select a hyperparameter): \
             {} samples, {} families, positive rate {:.4}% ({} dedup_repeat rows excluded).\n",
            report.dev_samples,
            report.dev_families,
            report.dev_positive_rate * 100.0,
            report.dev_dedup_repeats_excluded
        )?;
        writeln!(
            md,
            "**Loss weighting**: pos_weight = {:.3} ({}).\n",
            report.pos_weight, report.pos_weight_rationale
        )?;
        md.push_str("## Held-out DEV ranking quality, against the zero-predictor floor\n\n");
        md.push_str("| metric | bilinear | constant-predictor floor |\n|---|---:|---:|\n");
        writeln!(
            md,
            "| AUC-ROC | {:.4} | {:.4} |\n| PR-AUC (average precision) | {:.4} | {:.4} |",
            report.dev_auc,
            report.zero_predictor_auc,
            report.dev_pr_auc,
            report.zero_predictor_pr_auc
        )?;
        if let Some(a) = report.dev_auc_within_expression {
            writeln!(
                md,
                "\nWithin-expression AUC (the move-ordering question a deployed guide is \
                 actually asked), macro-averaged over the {} expressions whose candidate set \
                 holds both classes: **{a:.4}**.\n",
                report.dev_within_expression_groups
            )?;
        }

        md.push_str("\n## Learning-rate selection (TRAIN-internal holdout only)\n\n");
        writeln!(
            md,
            "{:.0}% of TRAIN families ({} of them) held out; DEV was not read.\n",
            report.holdout_fraction * 100.0,
            report.holdout_families
        )?;
        md.push_str("| lr | holdout AUC | holdout PR-AUC |\n|---:|---:|---:|\n");
        for t in &report.lr_trials {
            writeln!(
                md,
                "| {:.4} | {} | {} |",
                t.lr,
                t.holdout_auc.map_or("-".into(), |v| format!("{v:.4}")),
                t.holdout_pr_auc.map_or("-".into(), |v| format!("{v:.4}"))
            )?;
        }
        writeln!(md, "\nSelected lr = {:.4}.\n", report.lr_selected)?;

        md.push_str("\n## Training curve (final fit, all TRAIN families)\n\n");
        md.push_str("| epoch | lr | train weighted loss | clip rate | DEV AUC | DEV PR-AUC |\n");
        md.push_str("|---:|---:|---:|---:|---:|---:|\n");
        for p in &report.training_curve {
            writeln!(
                md,
                "| {} | {:.5} | {:.6} | {:.3} | {} | {} |",
                p.epoch,
                p.lr,
                p.train_mean_weighted_loss,
                p.clip_rate,
                p.dev_auc.map_or(String::new(), |v| format!("{v:.4}")),
                p.dev_pr_auc.map_or(String::new(), |v| format!("{v:.4}")),
            )?;
        }

        md.push_str("\n## Rule encoder\n\n");
        writeln!(
            md,
            "The rule embedding is trained through `encode_rule`'s \
             `[LHS | RHS | LHS-RHS | LHS*RHS]` projection (registration §4), over \
             depth-bound poolings of each rule's own templates. Rank of the \
             concatenation matrix: **{} of {} rules**. A rank equal to the rule count \
             means the projection can realize any per-rule embedding assignment — i.e. \
             this arm is not handicapped against the additive arm's free per-rule \
             scalar.\n",
            report.rule_encoding_rank, report.rule_count
        )?;
        if report.indistinguishable_rule_pairs.is_empty() {
            md.push_str("No two rules share an encoding.\n");
        } else {
            writeln!(
                md,
                "**Recorded limitation.** {} rule pair(s) encode identically and therefore \
                 score identically in every context, at any weights: {:?}. All of them are \
                 rules that define no template on either side ({:?}), so the registered \
                 encoding has nothing to encode for them.\n",
                report.indistinguishable_rule_pairs.len(),
                report.indistinguishable_rule_pairs,
                report.templateless_rules
            )?;
        }
        writeln!(
            md,
            "\nWrite-side train/deploy skew: max |trainer-in-memory - deployed-from-file| = \
             {:.3e} over {} DEV records (bar 1e-6).\n",
            report.write_side_skew_max_abs_diff, report.write_side_skew_records
        )?;
        writeln!(
            md,
            "\nTrained rule-by-context second difference (trig-shaped vs \
             polynomial-shaped neighborhood, two rules): **{:.6}**. The additive arm pins \
             this at exactly zero by construction; a value of zero here would mean \
             training collapsed this head into that model class.\n",
            report.trained_second_difference
        )?;

        md.push_str(
            "\n## Per-rule: learned priority (DEV mean predicted score) vs measured \
             strict-bound rate\n\n",
        );
        md.push_str(
            "| rule | id | train fired | train rate | DEV fired | DEV measured rate | \
             DEV mean predicted P |\n",
        );
        md.push_str("|---|---|---:|---:|---:|---:|---:|\n");
        for r in &report.per_rule {
            writeln!(
                md,
                "| {} | {} | {} | {:.5} | {} | {:.5} | {:.5} |",
                r.rule_name,
                r.rule_id,
                r.train_fired,
                r.train_positive_rate,
                r.dev_fired,
                r.dev_positive_rate,
                r.dev_mean_predicted
            )?;
        }

        writeln!(md, "\nCheckpoint written to `{}`.", report.checkpoint_path)?;
        Ok(md)
    })()
    .expect("formatting into a String cannot fail");
    std::fs::write(path, md)
        .unwrap_or_else(|e| panic!("train_guide_bilinear: cannot write {path}: {e}"));
    eprintln!("train_guide_bilinear: wrote {path}");
}

// ── The registered second difference, measured on the trained model ──────────

/// `[s(r1, trig) − s(r2, trig)] − [s(r1, poly) − s(r2, poly)]` for two rules
/// and two neighborhoods that differ only in their op.
///
/// The additive model pins this at exactly zero (`scoring::representation`);
/// measuring it on the trained bilinear is the cheapest check that training
/// did not collapse the head into that model class.
fn trained_second_difference(guide: &BilinearCandidateGuide, rules: &RuleSet) -> f32 {
    let (r1, r2) = (
        rules.id_of(0).expect("a first rule"),
        rules.id_of(1).expect("a second rule"),
    );
    let at = |rule: RuleId, op: OpKind| CandidateSummary {
        rule_embed: guide.rule_embed(rule),
        neighborhood_ops: vec![op, op],
        budget_fraction: 0.4,
        rule,
        match_class_node_count: 3,
        expr_node_count: 10,
    };
    let s = guide.score_candidates(&[
        at(r1, OpKind::Sin),
        at(r2, OpKind::Sin),
        at(r1, OpKind::Add),
        at(r2, OpKind::Add),
    ]);
    (s[0] - s[1]) - (s[2] - s[3])
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let rules = RuleSet::production();

    eprintln!("train_guide_bilinear: loading TRAIN from {}", args.train);
    let train = load_jsonl(&PathBuf::from(&args.train));
    eprintln!("train_guide_bilinear: loading DEV from {}", args.dev);
    let dev = load_jsonl(&PathBuf::from(&args.dev));

    let overlap: Vec<(u32, u64)> = train
        .distinct_families
        .intersection(&dev.distinct_families)
        .copied()
        .collect();
    assert!(
        overlap.is_empty(),
        "train_guide_bilinear: TRAIN and DEV share {} generator families (e.g. {:?}) — the \
         reported DEV metrics would not be held out",
        overlap.len(),
        &overlap[..overlap.len().min(3)]
    );

    let train_positive_rate = train.positives as f64 / train.samples.len() as f64;
    let neg = train.samples.len() - train.positives;
    assert!(
        train.positives > 0 && neg > 0,
        "train_guide_bilinear: TRAIN has {} positive and {neg} negative samples — binary \
         supervision needs both",
        train.positives
    );
    let pos_weight = neg as f32 / train.positives as f32;
    let pos_weight_rationale = format!(
        "inverse class frequency (negatives/positives = {neg}/{} measured on this TRAIN \
         split) — identical in definition to `train_guide`'s, so the two arms' losses \
         differ in no term",
        train.positives
    );
    eprintln!(
        "train_guide_bilinear: TRAIN {} samples / {} families, positive rate {:.4}%, \
         pos_weight = {:.3}",
        train.samples.len(),
        train.distinct_families.len(),
        train_positive_rate * 100.0,
        pos_weight
    );

    // ── Learning-rate selection, TRAIN-internal only ─────────────────────
    let lr_grid: Vec<f32> = args
        .lr_grid
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .unwrap_or_else(|e| panic!("train_guide_bilinear: --lr-grid {s:?}: {e}"))
        })
        .collect();
    assert!(
        !lr_grid.is_empty(),
        "train_guide_bilinear: --lr-grid must name at least one learning rate"
    );

    let select_epochs = if args.select_epochs == 0 {
        args.epochs
    } else {
        args.select_epochs
    };
    let (fit_idx, holdout_idx) = family_holdout(&train, args.holdout_fraction, args.seed);
    let holdout_families = train.distinct_families.len()
        - fit_idx
            .iter()
            .map(|&i| train.families[i])
            .collect::<HashSet<_>>()
            .len();
    let mut lr_trials = Vec::new();
    let mut best: Option<(f32, f64)> = None;

    if lr_grid.len() == 1 {
        eprintln!(
            "train_guide_bilinear: one learning rate given ({}), skipping selection",
            lr_grid[0]
        );
        best = Some((lr_grid[0], f64::NAN));
    } else {
        assert!(
            !holdout_idx.is_empty(),
            "train_guide_bilinear: the TRAIN-internal holdout is empty — raise \
             --holdout-fraction; selecting on DEV is not an option"
        );
        for &lr in &lr_grid {
            let mut trainer = BilinearTrainer::new_cold(&rules, args.seed);
            for epoch in 0..select_epochs {
                let e_lr = lr / (1.0 + args.lr_decay * epoch as f32);
                let order = {
                    let perm =
                        shuffled_indices(fit_idx.len(), args.seed.wrapping_add(epoch as u64));
                    perm.into_iter().map(|i| fit_idx[i]).collect::<Vec<_>>()
                };
                let (loss, clip) = train_epoch(
                    &mut trainer,
                    &train,
                    &order,
                    FitParams {
                        step: SgdStep {
                            lr: e_lr,
                            l2: args.l2,
                            max_grad_norm: args.max_grad_norm,
                        },
                        pos_weight,
                    },
                );
                eprintln!(
                    "train_guide_bilinear: [select lr={lr}] epoch {epoch} loss={loss:.6} \
                     clip_rate={clip:.3}"
                );
            }
            let (scores, labels) = score_all(&trainer, &train, &holdout_idx);
            let auc = auc_roc(&scores, &labels);
            let pr = average_precision(&scores, &labels);
            eprintln!("train_guide_bilinear: [select lr={lr}] holdout auc={auc:?} pr_auc={pr:?}");
            lr_trials.push(LrTrial {
                lr,
                holdout_auc: auc,
                holdout_pr_auc: pr,
            });
            if let Some(pr) = pr
                && best.is_none_or(|(_, b)| pr > b)
            {
                best = Some((lr, pr));
            }
        }
    }

    let (lr_selected, _) = best.unwrap_or_else(|| {
        panic!(
            "train_guide_bilinear: no learning rate produced a defined holdout PR-AUC — \
             the holdout split has no positives, so selection is impossible; raise \
             --holdout-fraction"
        )
    });
    eprintln!("train_guide_bilinear: selected lr = {lr_selected}");

    // ── Final fit, all TRAIN families ────────────────────────────────────
    let mut trainer = BilinearTrainer::new_cold(&rules, args.seed);
    eprintln!(
        "train_guide_bilinear: rule encoding rank {} of {} rules; templateless: {:?}; \
         indistinguishable pairs: {:?}",
        trainer.rule_encoding_rank(),
        trainer.rule_count(),
        trainer.templateless_rules(),
        trainer.indistinguishable_rules(&rules),
    );

    let mut training_curve = Vec::new();
    let dev_all: Vec<usize> = (0..dev.samples.len()).collect();
    for epoch in 0..args.epochs {
        let lr = lr_selected / (1.0 + args.lr_decay * epoch as f32);
        let order = shuffled_indices(train.samples.len(), args.seed.wrapping_add(epoch as u64));
        let (loss, clip_rate) = train_epoch(
            &mut trainer,
            &train,
            &order,
            FitParams {
                step: SgdStep {
                    lr,
                    l2: args.l2,
                    max_grad_norm: args.max_grad_norm,
                },
                pos_weight,
            },
        );
        let is_last = epoch + 1 == args.epochs;
        let (dev_auc, dev_pr_auc) =
            if args.eval_every > 0 && (epoch % args.eval_every == 0 || is_last) {
                let (scores, labels) = score_all(&trainer, &dev, &dev_all);
                (
                    auc_roc(&scores, &labels),
                    average_precision(&scores, &labels),
                )
            } else {
                (None, None)
            };
        eprintln!(
            "train_guide_bilinear: epoch {epoch:>3} lr={lr:.5} loss={loss:.6} \
             clip_rate={clip_rate:.3}{}",
            match (dev_auc, dev_pr_auc) {
                (Some(a), Some(p)) => format!("  dev_auc={a:.4} dev_pr_auc={p:.4}"),
                _ => String::new(),
            }
        );
        training_curve.push(EpochPoint {
            epoch,
            lr,
            train_mean_weighted_loss: loss,
            clip_rate,
            dev_auc,
            dev_pr_auc,
        });
    }

    // ── Held-out evaluation ──────────────────────────────────────────────
    let (dev_scores, dev_labels) = score_all(&trainer, &dev, &dev_all);
    let dev_auc = auc_roc(&dev_scores, &dev_labels).unwrap_or_else(|| {
        panic!("train_guide_bilinear: DEV AUC undefined — one class is absent from DEV")
    });
    let dev_pr_auc = average_precision(&dev_scores, &dev_labels).unwrap_or_else(|| {
        panic!("train_guide_bilinear: DEV PR-AUC undefined — DEV has zero positives")
    });
    let dev_positive_rate = dev.positives as f64 / dev.samples.len() as f64;
    let (dev_auc_within_expression, dev_within_expression_groups) =
        match auc_roc_within_groups(&dev_scores, &dev_labels, &dev.groups) {
            Some((a, n)) => (Some(a), n),
            None => (None, 0),
        };
    println!(
        "=== held out DEV: AUC {dev_auc:.4} (floor 0.5), PR-AUC {dev_pr_auc:.4} \
         (floor = positive rate {dev_positive_rate:.6}) ==="
    );

    // ── Checkpoint ───────────────────────────────────────────────────────
    let weights: BilinearWeights = trainer.weights();
    let mut checkpoint = BilinearGuideCheckpoint {
        schema_identity: BilinearGuideCheckpoint::current_schema_identity(),
        label_source: "strict-v1".into(),
        trainer: "train_guide_bilinear".into(),
        written_at_unix_s: unix_now_s(),
        seed: args.seed,
        epochs: args.epochs,
        lr_initial: lr_selected,
        lr_decay: args.lr_decay,
        l2: args.l2,
        max_grad_norm: args.max_grad_norm,
        pos_weight,
        rule_fingerprint: format!("{}", rules.fingerprint()),
        num_rules: rules.len(),
        op_names: OpKind::all().map(|op| format!("{op:?}")).collect(),
        parameters: weights.parameters.clone(),
        op_embeddings: weights.op_embeddings.clone(),
        train_samples: train.samples.len(),
        train_families: train.distinct_families.len(),
        train_positive_rate,
        dev_samples: dev.samples.len(),
        dev_families: dev.distinct_families.len(),
        dev_auc,
        dev_pr_auc,
        weights_fnv64: String::new(),
    };
    checkpoint.weights_fnv64 = checkpoint.weights_fingerprint();
    let json = serde_json::to_string_pretty(&checkpoint)
        .unwrap_or_else(|e| panic!("train_guide_bilinear: checkpoint serialization failed: {e}"));
    std::fs::write(&args.out_checkpoint, json).unwrap_or_else(|e| {
        panic!(
            "train_guide_bilinear: cannot write {}: {e}",
            args.out_checkpoint
        )
    });
    eprintln!("train_guide_bilinear: wrote {}", args.out_checkpoint);

    // Deploy the checkpoint we just wrote, and check two things about it
    // before anything downstream is allowed to believe this run.
    let guide = BilinearCandidateGuide::new(&checkpoint.to_weights(&rules), &rules)
        .unwrap_or_else(|e| panic!("train_guide_bilinear: the checkpoint it just wrote: {e}"));

    // (1) The **writing** half of the train/deploy skew check, and the only
    // place it can be made: this is the sole moment an in-memory trained
    // head and a from-file deployed one both exist. `skew_test_bilinear_guide`
    // is the committed artifact but reads the file on both sides, so a
    // tensor missing from the flatten order would be missing from both its
    // reads identically and pass. Here it cannot.
    let skew_records = SKEW_RECORDS.min(dev.samples.len());
    assert!(
        skew_records >= 1000,
        "train_guide_bilinear: DEV holds only {} scorable records; the skew bar is >= 1000",
        dev.samples.len()
    );
    let mut write_skew_max = 0.0f64;
    let mut write_skew_over = 0usize;
    for s in dev.samples.iter().take(skew_records) {
        let summary = s.to_summary(trainer.rule_embed(s.rule));
        let in_memory = trainer.score(&summary);
        let from_file = guide.score_candidates(std::slice::from_ref(&summary))[0];
        let diff = f64::from((in_memory - from_file).abs());
        write_skew_max = write_skew_max.max(diff);
        if diff > SKEW_TOL {
            write_skew_over += 1;
        }
    }
    assert!(
        write_skew_over == 0,
        "train_guide_bilinear: {write_skew_over} of {skew_records} DEV records differ by \
         more than {SKEW_TOL} between the trained head in memory and the head loaded back \
         from {} (max |diff| {write_skew_max:.3e}) — the checkpoint does not round-trip \
         the model that was trained",
        args.out_checkpoint
    );
    println!(
        "=== write-side skew: {skew_records} DEV records, max |trainer_in_memory - \
         deployed_from_file| = {write_skew_max:.3e} (bar {SKEW_TOL}) ==="
    );

    // (2) The head did not collapse into the additive model class.
    let second_difference = trained_second_difference(&guide, &rules);
    assert!(
        second_difference.abs() > 0.0,
        "train_guide_bilinear: the trained head's rule-by-context second difference is \
         exactly zero — training collapsed it into an additively separable model, which is \
         the model class this arm exists to be distinguished from. Refusing to report this \
         as a bilinear arm."
    );
    println!("=== trained rule-by-context second difference: {second_difference} ===");

    // ── Report ───────────────────────────────────────────────────────────
    let mut dev_mean_pred_by_rule: BTreeMap<RuleId, (f64, usize)> = BTreeMap::new();
    for (s, &z) in dev.samples.iter().zip(dev_scores.iter()) {
        let e = dev_mean_pred_by_rule.entry(s.rule).or_insert((0.0, 0));
        e.0 += f64::from(sigmoid(z));
        e.1 += 1;
    }
    let mut all_rules: Vec<RuleId> = train
        .per_rule
        .keys()
        .chain(dev.per_rule.keys())
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    all_rules.sort_unstable_by_key(|id| id.get());

    let label_of = |id: RuleId| {
        rules
            .index_of(id)
            .and_then(|i| rules.label_of(i))
            .unwrap_or_else(|| format!("<rule {id}>"))
    };
    let per_rule: Vec<RuleRow> = all_rules
        .iter()
        .map(|&id| {
            let (train_fired, train_positive) = train.per_rule.get(&id).copied().unwrap_or((0, 0));
            let (dev_fired, dev_positive) = dev.per_rule.get(&id).copied().unwrap_or((0, 0));
            let (pred_sum, pred_n) = dev_mean_pred_by_rule.get(&id).copied().unwrap_or((0.0, 0));
            RuleRow {
                rule_id: format!("{id}"),
                rule_name: label_of(id),
                train_fired,
                train_positive,
                train_positive_rate: if train_fired == 0 {
                    0.0
                } else {
                    train_positive as f64 / train_fired as f64
                },
                dev_fired,
                dev_positive,
                dev_positive_rate: if dev_fired == 0 {
                    0.0
                } else {
                    dev_positive as f64 / dev_fired as f64
                },
                dev_mean_predicted: if pred_n == 0 {
                    0.0
                } else {
                    pred_sum / pred_n as f64
                },
            }
        })
        .collect();

    let fnv_of = |path: &str| {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("train_guide_bilinear: cannot re-read {path}: {e}"));
        fnv1a64_hex(&bytes)
    };

    let report = Report {
        label_source: "strict-v1".into(),
        model_class: "bilinear (m(x)^T W r + b^T r)".into(),
        dev_path: args.dev.clone(),
        dev_fnv64: fnv_of(&args.dev),
        train_path: args.train.clone(),
        train_fnv64: fnv_of(&args.train),
        pos_weight,
        pos_weight_rationale,
        train_samples: train.samples.len(),
        train_families: train.distinct_families.len(),
        train_positive_rate,
        train_dedup_repeats_excluded: train.repeats_excluded,
        dev_samples: dev.samples.len(),
        dev_families: dev.distinct_families.len(),
        dev_positive_rate,
        dev_dedup_repeats_excluded: dev.repeats_excluded,
        dev_auc,
        dev_pr_auc,
        zero_predictor_auc: 0.5,
        zero_predictor_pr_auc: dev_positive_rate,
        dev_auc_within_expression,
        dev_within_expression_groups,
        lr_selected,
        lr_trials,
        holdout_fraction: args.holdout_fraction,
        holdout_families,
        epochs: args.epochs,
        lr_decay: args.lr_decay,
        l2: args.l2,
        max_grad_norm: args.max_grad_norm,
        seed: args.seed,
        training_curve,
        rule_encoding_rank: trainer.rule_encoding_rank(),
        rule_count: trainer.rule_count(),
        templateless_rules: trainer.templateless_rules().to_vec(),
        indistinguishable_rule_pairs: trainer.indistinguishable_rules(&rules),
        trained_second_difference: second_difference,
        write_side_skew_records: skew_records,
        write_side_skew_max_abs_diff: write_skew_max,
        per_rule,
        checkpoint_path: args.out_checkpoint.clone(),
        checkpoint_md5_note: "md5 recorded in the results document, not here (this file is \
                              written before the checkpoint can be hashed as a whole)"
            .into(),
    };

    if let Some(path) = &args.report_json {
        write_json_report(path, &report);
    }
    if let Some(path) = &args.report_md {
        write_md_report(path, &report);
    }
}
