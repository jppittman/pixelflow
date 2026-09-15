//! The at-budget evaluation of the Phase 3 registered claim, on DEV.
//!
//! Implements `docs/plans/2026-08-31-guide-design-revision.md` §5 against the
//! committed registration `docs/plans/2026-09-01-phase3-registration.md`
//! (nothing in that document is revised here). Every arm of the ablation
//! ladder goes through the ONE anytime-curve definition
//! ([`pixelflow_search::egraph::anytime`]); the arms differ only in how the
//! e-graph is advanced between checkpoints:
//!
//! | arm | stepper |
//! |---|---|
//! | `unguided` | [`run_anytime_curve`] (fixed rule-then-class sweeps) — supplies both unguided-at-B and unguided-at-4B |
//! | `control` | [`GuidedSaturation`] over [`PerRuleRateGuide`] (per-rule TRAIN strict-positive rate, no candidate-local information) |
//! | `linear` | [`GuidedSaturation`] over [`LinearCandidateGuide`] (the trained cold-start Guide — the claim) |
//!
//! A strict-oracle arm (order candidates by their true strict label) is NOT
//! run: the strict label is defined on the `ApplicationId`s of one specific
//! run, and transporting it onto a differently-ordered guided run's candidates
//! needs a `(rule, class content at firing time) -> label` map the provenance
//! log does not record (`ApplicationRecord::match_root` is a class id that
//! union/rebuild renumbers). Building that map means instrumenting the
//! unguided sweep itself — not a cheap replay, so per the task spec it is
//! skipped and said so.
//!
//! # Metric (registration §2/§6, verbatim semantics)
//!
//! Costs are `CostModel::latency_prior()` `extract_dag` costs — deterministic,
//! no wall-clock anywhere in any reported number. Cost at budget B is read at
//! the grid checkpoint for B (the first between-rounds point with cumulative
//! recorded applications ≥ B; both curves report `app_actual`). Regret uses
//! the empirical best any arm reaches at any checkpoint of that expression.
//! Zero-cost references follow the established convention: a positive cost
//! against a zero-cost reference is infinite loss, never 0%.
//!
//! # Production-units context (integration audit, PR #1079)
//!
//! Production saturates with `config_for_node_count` (classical: 100 rounds /
//! 5,000 classes / 200 ms) and has no application counter or cap, so the
//! registered application budget is a proxy for a machine production does not
//! literally run. For every evaluated expression this binary ALSO runs the
//! exact production saturation step
//! ([`pixelflow_search::runtime::production_saturation_probe`] — the same
//! function body `optimize_runtime_arena` runs, not a copy) and records its
//! stop reason (read, not inferred), provenance application count at stop, and
//! latency-prior cost, then locates that cost on the unguided anytime curve.
//! Wall-clock is a stop condition of that call only; `Timeout` stops are
//! flagged machine-dependent and the host load average is recorded as
//! context, never as a metric.
//!
//! # Corpus discipline
//!
//! Reads `corpus_dev.bin` by default, or a `--corpus` file of DEV-only
//! out-of-distribution families (round 1b:
//! `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md`,
//! `corpus_dev_ood.bin`, one family selected with `--name-prefix`). Whatever
//! is loaded is fence-checked against `corpus_train.bin` (a structural
//! collision with TRAIN is a hard error); `corpus_final.bin` is never opened
//! — FINAL stays untouched until a publication run.
//!
//! # Round 1b additions
//!
//! `--stratify-by-ops` labels every classical row with the registration's
//! op-composition stratum (rows written without one are labelled from the
//! corpus in aggregate), each arm records per-rule-INDEX firings and
//! strict-positives at B and over the whole run (`by_rule_idx`, `full_run`),
//! and every classical-sized set reports `D_A = m_A^S − m_A^DEV` against the
//! registered margin `M_B` with the pre-committed H_shift / H_null / H_inv
//! verdict.
//!
//! # Resumability
//!
//! Per-expression results are appended to `--out-jsonl` as each expression
//! finishes; a re-run skips names already present. The aggregate report is
//! computed from the JSONL (`--aggregate-only` recomputes it without running
//! anything), so a crash mid-run loses one expression, not the run.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin phase3_at_budget_eval -- \
//!     --out-jsonl docs/results/2026-09-01-phase3-at-budget-eval.jsonl \
//!     --out-json docs/results/2026-09-01-phase3-at-budget-eval.json \
//!     --out-md docs/results/2026-09-01-phase3-at-budget-eval-report.md
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};
use pixelflow_pipeline::journal::append_record;
use pixelflow_pipeline::schema::fnv1a64_hex;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::guide_linear::{
    load_linear_guide, per_rule_rate_guide_from_report,
};
use pixelflow_pipeline::training::r2g::load_return_guide;
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, ApplicationId, Budget, CostModel, EClassId, EGraph,
    ENode, ENodeId, EpisodeLabels, KeepJournal, Optimizer, Origin, RuleId, RuleSet, SaturationStop,
    config_for_node_count, run_anytime_curve,
};
use pixelflow_search::nnue::guide::linear::{
    LinearCandidateGuide, LinearReturnGuide, PerRuleRateGuide,
};
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

#[derive(Parser)]
#[command(name = "phase3_at_budget_eval")]
#[command(about = "Phase 3 at-budget ablation ladder on DEV against the registered claim")]
struct Args {
    /// Directory holding `corpus_dev.bin` (default corpus) and
    /// `corpus_train.bin` (fence source — read whenever any corpus is
    /// loaded, per docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md §3).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// DEV-side corpus file to evaluate, overriding `<corpus_dir>/corpus_dev.bin`
    /// (for OOD families per the round-1b registration §3). Every entry is
    /// still fence-checked against `<corpus_dir>/corpus_train.bin` — a
    /// collision is a hard error regardless of which corpus is loaded.
    #[arg(long)]
    corpus: Option<String>,

    /// Evaluate only entries whose name starts with this prefix — selects
    /// one named family out of a multi-family file (`corpus_dev_ood.bin`
    /// holds `dev_sh_*` and `dev_bezier_*`). The TRAIN fence is still
    /// checked over the whole file. Empty = every entry.
    #[arg(long, default_value = "")]
    name_prefix: String,

    /// Emit the registered op-composition stratum
    /// (docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md §2)
    /// per expression in the JSONL rows, and per-stratum summary tables
    /// (population counts over the full classical corpus, plus per-stratum
    /// D_A statistics for whatever classical rows this run evaluated).
    #[arg(long, default_value_t = false)]
    stratify_by_ops: bool,

    /// `train_guide` checkpoint for the linear Guide (the claim arm).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    checkpoint: String,

    /// `train_guide_r2g` checkpoint (`docs/plans/2026-09-01-guide-return-to-go.md`
    /// §3.3). When set, the claim arm is [`LinearReturnGuide`] loaded from
    /// this path and `--checkpoint` is not opened. The arm keeps the `linear`
    /// key in every row and aggregate (schema stability for `read_rows` /
    /// `--aggregate-only`); `context.claim_guide` in the report JSON and the
    /// markdown labels say which Guide actually ran, so a row file is never
    /// silently ambiguous — combine runs by that field, never by filename.
    #[arg(long)]
    r2g_checkpoint: Option<String>,

    /// `train_guide` report JSON — supplies the per-rule TRAIN rates the
    /// control arm is built from.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-train-guide-report.json"
    )]
    train_guide_report: String,

    /// DEV classical expressions to evaluate, size-stratified over all DEV
    /// classical expressions. 0 = all of them.
    #[arg(long, default_value_t = 0)]
    classical_samples: usize,

    /// Lower bound (inclusive) on an expression's arena node count; `0` =
    /// unbounded. Applied to BOTH the selection and the aggregation, so a
    /// run and a `--aggregate-only` re-read of its rows report the same
    /// population. Round 3's registered training regime
    /// (docs/plans/2026-09-01-guide-return-to-go.md §2b.3) is node band
    /// 101-1000: `--min-expr-nodes 101 --max-expr-nodes 1000`.
    #[arg(long, default_value_t = 0)]
    min_expr_nodes: usize,

    /// Upper bound (inclusive) on an expression's arena node count; `0` =
    /// unbounded. See `--min-expr-nodes`.
    #[arg(long, default_value_t = 0)]
    max_expr_nodes: usize,

    /// DEV blitz and rapid expressions per band, reported for completeness
    /// only (no claim is registered on them). 0 = none.
    #[arg(long, default_value_t = 30)]
    other_samples: usize,

    /// Per-expression results, appended as produced (resume skips names
    /// already present).
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-at-budget-eval.jsonl"
    )]
    out_jsonl: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-at-budget-eval.json"
    )]
    out_json: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-at-budget-eval-report.md"
    )]
    out_md: String,

    /// Recompute the aggregate report from the JSONL without evaluating.
    #[arg(long, default_value_t = false)]
    aggregate_only: bool,

    /// Comma-separated expression names to skip (documented in the report).
    #[arg(long, default_value = "")]
    skip_names: String,

    /// Journal to append the run record to.
    #[arg(long, default_value = "docs/results/journal.jsonl")]
    journal: String,

    /// Override the guided arms' checkpoint grid (comma-separated, strictly
    /// increasing, must contain every registered B and 4B). Sensitivity
    /// check only: a guided round's remaining scored survivors are dropped
    /// when a checkpoint's application target is reached, so a denser grid
    /// can only cost a guided arm, never help it.
    #[arg(long, default_value = "")]
    guided_grid: String,
}

// ---------------------------------------------------------------------------
// Registered constants (docs/plans/2026-09-01-phase3-registration.md §4–§6).
// Not adjustable after the fact; restated here so the report is self-checking.
// ---------------------------------------------------------------------------

/// Registered classical tiers: (B, Y as a fraction, median unguided
/// truncation loss L in percent).
const REGISTERED_TIERS: [(usize, f64, f64); 2] = [(100, 0.163, 48.47), (200, 0.090, 21.92)];

/// Guided arms are sampled through 4B of the secondary tier — every
/// registered quantity (B, 4B for both tiers) is on this grid, and running a
/// guided arm to quiescence would answer no registered question.
const GUIDED_GRID: &[usize] = &[25, 50, 100, 200, 400, 800];

/// Per-curve wall-clock safety ceiling — PANICS inside the shared curve runner
/// if it binds, never truncates, never appears in a number.
const SAFETY_TIMEOUT: Duration = Duration::from_secs(1800);
const SWEEP_SAFETY_CEILING: usize = 10_000;

/// The structural/congruence rule class the design revision (§2.1) names:
/// rules whose job is enabling congruence closure, scored 63–84% load-bearing
/// by the labeler bound and ~0% by the strict bound.
const STRUCTURAL_RULES: [&str; 6] = [
    "commutative",
    "associative",
    "reverse-associative",
    "distribute",
    "fma-fusion",
    "identity",
];

const ARM_NAMES: [&str; 3] = ["unguided", "control", "linear"];

/// Everything that changes what a row MEANS, as one JSON blob written
/// beside the resumable JSONL. Paths alone would not do: a checkpoint file
/// can be retrained in place under the same name, so the guide artifacts are
/// identified by content hash.
fn config_fingerprint(args: &Args, guided_grid: &[usize], dev_corpus: &Path) -> String {
    let hash_of = |path: &Path| -> String {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("cannot read {} to fingerprint it: {e}", path.display()));
        fnv1a64_hex(&bytes)
    };
    format!(
        "{{\n  \"guided_grid\": {guided_grid:?},\n  \"checkpoint\": {:?},\n  \
         \"checkpoint_fnv64\": {:?},\n  \"train_guide_report\": {:?},\n  \
         \"train_guide_report_fnv64\": {:?},\n  \"dev_corpus\": {:?},\n  \
         \"dev_corpus_fnv64\": {:?},\n  \"classical_samples\": {},\n  \
         \"other_samples\": {}\n}}\n",
        args.checkpoint,
        hash_of(Path::new(&args.checkpoint)),
        args.train_guide_report,
        hash_of(Path::new(&args.train_guide_report)),
        dev_corpus.display().to_string(),
        hash_of(dev_corpus),
        args.classical_samples,
        args.other_samples,
    )
}

// ---------------------------------------------------------------------------
// Round-1b registered constants (docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md
// §1). `D_A(S, B) = m_A^S(B) - m_A^DEV(B)`; `m_A^DEV(B)` below is quoted
// verbatim from Round 1's own DEV-overall classical result (PR #1084,
// `2026-09-01-phase3-at-budget-eval.json`, `tiers[].arms[].ratio_vs_unguided_at_b.median`)
// and is NOT recomputed from this run — recomputing it here would let a
// differently-sampled run silently redefine its own baseline.
// ---------------------------------------------------------------------------

/// `m_A^DEV(B)`: (B, arm) -> Round-1 DEV-overall classical median ratio.
const DEV_REFERENCE_MEDIAN: [(usize, &str, f64); 4] = [
    (100, "control", 0.5655),
    (100, "linear", 0.5366),
    (200, "control", 0.6991),
    (200, "linear", 0.6959),
];

fn dev_reference_median_for(b: usize, arm: &str) -> f64 {
    DEV_REFERENCE_MEDIAN
        .iter()
        .find(|(rb, ra, _)| *rb == b && *ra == arm)
        .unwrap_or_else(|| panic!("no Round-1 DEV reference median for arm {arm:?} at B={b}"))
        .2
}

/// `M_B`: the largest `|D_control - D_linear|` Round 1 already produced
/// across its 8 no-shift DEV families (§1.1) — the null scale a shift
/// effect must exceed to be distinguishable from ordinary between-family
/// noise.
fn registered_margin(b: usize) -> f64 {
    match b {
        100 => 0.06,
        200 => 0.07,
        other => panic!("no registered margin M_B for B={other} (only 100/200 are registered)"),
    }
}

/// A verdict is claimed only on a set with n >= 30 classical expressions
/// (§1, "Primary test").
const MIN_N_FOR_VERDICT: usize = 30;

/// The rules the global prior suppresses (round-1b registration §4), by
/// `all_rules()` index: the enabler `doubling` (20), the parity rules on
/// Sin/Tan/Asin/Atan/Cos (30–34), and the trig identities (36–40). Reported
/// per arm and per set as firings within the first B applications and over
/// the whole run, each with its strict-positive count (§1.3).
const TRIG_RULE_IDX: [usize; 11] = [20, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40];

/// The same eleven rules by canonical label, in the same order — the
/// registration's own prose ("the enabler `doubling`, the parity rules on
/// Sin/Tan/Asin/Atan/Cos, and the trig identities") written as data.
///
/// [`TRIG_RULE_IDX`] is a **frozen registered constant** and is not edited;
/// this is its companion, and [`trig_rule_ids`] asserts the two still name
/// the same rules. A same-length reorder of `all_rules()` would repoint
/// every index silently and there is nothing else in the run that would
/// notice — which is exactly the failure
/// docs/plans/2026-09-02-phase3-forward-port.md §2.2 is about.
const TRIG_RULE_LABELS: [&str; 11] = [
    "doubling",
    "odd-negation(Sin)",
    "odd-negation(Tan)",
    "odd-negation(Asin)",
    "odd-negation(Atan)",
    "even-negation(Cos)",
    "sin-angle-addition",
    "cos-angle-addition",
    "reverse-angle-addition",
    "half-angle-product",
    "pythagorean",
];

/// The registered trig-rule set as stable identities, checked against the
/// registered indices.
///
/// # Panics
///
/// If an index is outside the live rule set, or if the rule at a registered
/// index is no longer the rule the registration names.
fn trig_rule_ids(rules: &RuleSet) -> Vec<(RuleId, &'static str)> {
    TRIG_RULE_IDX
        .iter()
        .zip(TRIG_RULE_LABELS)
        .map(|(&idx, label)| {
            let live = rules.label_of(idx).unwrap_or_else(|| {
                panic!(
                    "registered trig rule idx {idx} is outside all_rules() ({} rules)",
                    rules.len()
                )
            });
            assert_eq!(
                live, label,
                "registered trig rule idx {idx} is {live:?} in this build but the round-1b \
                 registration names it {label:?} — all_rules() was reordered since the \
                 registration, so every positional key in it now points at a different rule. \
                 Re-register rather than silently re-point."
            );
            (RuleId::from_label(label), label)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DEV stratification by op composition (round-1b registration §2). First
// matching row over the arena's non-leaf ops (`Var`/`Const`/`Param`/`Buffer`
// leaves excluded).
// ---------------------------------------------------------------------------

const TRIG_OPS: &[OpKind] = &[
    OpKind::Sin,
    OpKind::Cos,
    OpKind::Tan,
    OpKind::Asin,
    OpKind::Acos,
    OpKind::Atan,
    OpKind::Atan2,
];
const TRANS_OPS: &[OpKind] = &[
    OpKind::Exp,
    OpKind::Exp2,
    OpKind::Ln,
    OpKind::Log2,
    OpKind::Log10,
    OpKind::Pow,
];
const ROOT_OPS: &[OpKind] = &[OpKind::Sqrt, OpKind::Rsqrt, OpKind::Recip];
const POLY_OPS: &[OpKind] = &[
    OpKind::Add,
    OpKind::Sub,
    OpKind::Mul,
    OpKind::MulAdd,
    OpKind::Neg,
];

/// The op of a non-leaf `ExprNode`, or `None` for a leaf
/// (`Var`/`Const`/`Param`/`Buffer`/`Uniform`/`Ref` — excluded from the
/// stratification rule).
fn non_leaf_op(node: &ExprNode) -> Option<OpKind> {
    match node {
        ExprNode::Var(_)
        | ExprNode::Const(_)
        | ExprNode::Param(_)
        | ExprNode::Buffer(_)
        | ExprNode::Uniform(_)
        | ExprNode::Ref(_) => None,
        ExprNode::Unary(op, _)
        | ExprNode::Binary(op, _, _)
        | ExprNode::Ternary(op, _, _, _)
        | ExprNode::Nary(op, _, _) => Some(*op),
        ExprNode::Reduce { .. } => Some(OpKind::Reduce),
    }
}

/// The registered op-composition stratum of an arena (§2): first matching
/// row over `polynomial-only` / `trig-heavy` / `transcendental-heavy` /
/// `sqrt-recip-heavy` / `mixed`. `polynomial-only` is vacuously true for an
/// arena with no non-leaf nodes at all (a bare `Var`/`Const`), matching the
/// registered "every op ∈ POLY" wording.
fn ops_stratum(arena: &ExprArena) -> &'static str {
    let mut trig = 0usize;
    let mut trans = 0usize;
    let mut root = 0usize;
    let mut poly_only = true;
    for node in arena.nodes_raw() {
        let Some(op) = non_leaf_op(node) else {
            continue;
        };
        if TRIG_OPS.contains(&op) {
            trig += 1;
        }
        if TRANS_OPS.contains(&op) {
            trans += 1;
        }
        if ROOT_OPS.contains(&op) {
            root += 1;
        }
        if !POLY_OPS.contains(&op) {
            poly_only = false;
        }
    }
    if poly_only {
        "polynomial-only"
    } else if trig >= 3 {
        "trig-heavy"
    } else if trans >= 3 {
        "transcendental-heavy"
    } else if root >= 3 {
        "sqrt-recip-heavy"
    } else {
        "mixed"
    }
}

/// The registered trig-rule set is checked against the live rule table, so
/// a reorder of `all_rules()` since the round-1b registration is a loud
/// failure rather than eleven silently repointed positional keys.
#[cfg(test)]
mod registered_rule_tests {
    use super::*;

    #[test]
    fn registered_trig_indices_still_name_the_rules_the_registration_names() {
        let rules = RuleSet::production();
        let ids = trig_rule_ids(&rules);
        assert_eq!(ids.len(), TRIG_RULE_IDX.len());
        for ((id, label), &idx) in ids.iter().zip(TRIG_RULE_IDX.iter()) {
            assert_eq!(
                rules.id_of(idx),
                Some(*id),
                "registered index {idx} and label {label:?} must name the same rule"
            );
        }
    }
}

#[cfg(test)]
mod ops_stratum_tests {
    use super::*;

    #[test]
    fn polynomial_only_is_vacuous_on_a_bare_leaf() {
        let mut a = ExprArena::new();
        let _x = a.push_var(0);
        assert_eq!(ops_stratum(&a), "polynomial-only");
    }

    #[test]
    fn polynomial_only_over_add_sub_mul_muladd_neg() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let c = a.push_const(2.0);
        let add = a.push_binary(OpKind::Add, x, c);
        let sub = a.push_binary(OpKind::Sub, add, x);
        let mul = a.push_binary(OpKind::Mul, sub, c);
        let neg = a.push_unary(OpKind::Neg, mul);
        let _ma = a.push_ternary(OpKind::MulAdd, neg, x, c);
        assert_eq!(ops_stratum(&a), "polynomial-only");
    }

    #[test]
    fn one_div_breaks_polynomial_only_into_mixed() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let c = a.push_const(2.0);
        let _ = a.push_binary(OpKind::Div, x, c);
        assert_eq!(ops_stratum(&a), "mixed");
    }

    #[test]
    fn trig_heavy_needs_three_trig_ops() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sin, x);
        let c = a.push_unary(OpKind::Cos, x);
        let two = a.push_binary(OpKind::Mul, s, c);
        // Only two trig ops so far — should NOT be trig-heavy yet.
        assert_ne!(ops_stratum(&a), "trig-heavy");
        let _t = a.push_unary(OpKind::Tan, two);
        assert_eq!(ops_stratum(&a), "trig-heavy");
    }

    #[test]
    fn transcendental_heavy_needs_three_trans_ops() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e1 = a.push_unary(OpKind::Exp, x);
        let e2 = a.push_unary(OpKind::Ln, e1);
        let _e3 = a.push_unary(OpKind::Log2, e2);
        assert_eq!(ops_stratum(&a), "transcendental-heavy");
    }

    #[test]
    fn sqrt_recip_heavy_needs_three_root_ops() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let r1 = a.push_unary(OpKind::Sqrt, x);
        let r2 = a.push_unary(OpKind::Rsqrt, r1);
        let _r3 = a.push_unary(OpKind::Recip, r2);
        assert_eq!(ops_stratum(&a), "sqrt-recip-heavy");
    }

    #[test]
    fn trig_takes_priority_over_transcendental_and_root() {
        // 3 trig + 3 trans + 3 root ops present: trig-heavy wins (first
        // matching row after polynomial-only).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let mut n = a.push_unary(OpKind::Sin, x);
        n = a.push_unary(OpKind::Cos, n);
        n = a.push_unary(OpKind::Tan, n);
        n = a.push_unary(OpKind::Exp, n);
        n = a.push_unary(OpKind::Ln, n);
        n = a.push_unary(OpKind::Log2, n);
        n = a.push_unary(OpKind::Sqrt, n);
        n = a.push_unary(OpKind::Rsqrt, n);
        let _ = a.push_unary(OpKind::Recip, n);
        assert_eq!(ops_stratum(&a), "trig-heavy");
    }

    #[test]
    fn mixed_is_the_fallback() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sin, x);
        let e = a.push_unary(OpKind::Exp, s);
        let _ = a.push_unary(OpKind::Sqrt, e);
        // One of each group, none reaching 3 — falls through to mixed.
        assert_eq!(ops_stratum(&a), "mixed");
    }
}

fn tier_name(node_count: usize) -> &'static str {
    match node_count {
        0..=10 => "blitz",
        11..=50 => "rapid",
        _ => "classical",
    }
}

fn stop_name(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        SaturationStop::ApplicationBudget => "app_budget",
        SaturationStop::ClassCap => "class_cap",
        SaturationStop::IterationCeiling => "iteration_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

fn is_structural(name: &str) -> bool {
    STRUCTURAL_RULES.contains(&name)
}

// ---------------------------------------------------------------------------
// Per-expression record (one JSONL line).
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct ArmCurve {
    grid: Vec<usize>,
    app_actual: Vec<usize>,
    cost: Vec<usize>,
    rounds: Vec<usize>,
    classes: Vec<usize>,
    clamped: Vec<bool>,
    ended: String,
    ended_at_apps: usize,
    /// Guided arms only: distinct candidate keys scored over the whole run
    /// (dedup coverage diagnostic).
    seen_keys: Option<usize>,
}

impl ArmCurve {
    fn idx(&self, b: usize) -> usize {
        self.grid
            .iter()
            .position(|&g| g == b)
            .unwrap_or_else(|| panic!("budget {b} is not on this arm's grid {:?}", self.grid))
    }
    fn cost_at(&self, b: usize) -> usize {
        self.cost[self.idx(b)]
    }
    fn apps_at(&self, b: usize) -> usize {
        self.app_actual[self.idx(b)]
    }
    fn rounds_at(&self, b: usize) -> usize {
        self.rounds[self.idx(b)]
    }
    fn best_cost(&self) -> usize {
        *self.cost.iter().min().expect("non-empty curve")
    }
    /// Best cost reached at any checkpoint whose grid TARGET is `<= limit`.
    ///
    /// The registered regret reference is the best cost either arm reaches
    /// over the grid — but the arms do not run the same grid: unguided
    /// samples `APP_CHECKPOINT_GRID` to 204800, guided arms stop at the last
    /// registered quantity (4B of the secondary tier). Comparing a guided
    /// arm's regret against a reference only the unguided arm was given the
    /// budget to reach measures the grid, not the guide.
    fn best_cost_up_to(&self, limit: usize) -> usize {
        self.grid
            .iter()
            .zip(&self.cost)
            .filter(|(g, _)| **g <= limit)
            .map(|(_, c)| *c)
            .min()
            .expect("every arm's grid contains at least the first checkpoint")
    }
    /// Cost at the first checkpoint whose ACTUAL application count reaches
    /// `apps`, or the curve's final (frozen) cost if it never does.
    ///
    /// Unguided saturation checks its budget between whole rule sweeps and so
    /// overshoots a grid target; guided saturation stops after an individual
    /// candidate and lands on it. Reading both arms at the same grid TARGET
    /// therefore hands the unguided arm more work than the guided one — for
    /// the committed classical baseline, a median 113 applications against
    /// 100. This reads an arm at a common actual-application count instead.
    fn cost_at_apps(&self, apps: usize) -> usize {
        for (i, &a) in self.app_actual.iter().enumerate() {
            if a >= apps {
                return self.cost[i];
            }
        }
        *self.cost.last().expect("non-empty curve")
    }
}

/// What the first `applications` recorded applications of one arm's run
/// looked like — read off the arm's final provenance log, which is
/// append-only and ordered by `ApplicationId`.
#[derive(Serialize, Deserialize, Clone)]
struct AtBudgetDiag {
    applications: usize,
    structural: usize,
    /// Applications among the first `applications` whose output node is on
    /// the arm's OWN final extracted path (strict hindsight label).
    strict_positive: usize,
    rounds: usize,
    rule_hist: BTreeMap<String, usize>,
    /// Round 1b: the same prefix keyed by rule INDEX (`all_rules()` order),
    /// with each rule's strict-positive count. `name()` is not a key here
    /// because it merges the six parity rules under two names
    /// (`odd-negation` for idx 30–33, `even-negation` for 34–35) and the
    /// registration asks about idx 30 (Sin) separately from idx 31 (Tan).
    /// `None` on rows written before this field existed (Round 1) — the
    /// aggregate reports how many rows carry it rather than counting a
    /// missing histogram as zero firings.
    #[serde(default)]
    /// Per-rule firings, keyed by CANONICAL RULE LABEL — never by a
    /// position in `all_rules()`, which a same-length reorder repoints
    /// silently (docs/plans/2026-09-02-phase3-forward-port.md §2.2). `None`
    /// on a row written before this key existed, which every consumer here
    /// refuses loudly rather than reading as "no firings".
    by_rule: Option<BTreeMap<String, RuleFiring>>,
}

/// Firings of one rule within a prefix of one arm's run, and how many of
/// them were strict-positive (output node on that arm's own final
/// extracted path).
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
struct RuleFiring {
    fired: usize,
    strict_positive: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct ProductionRow {
    node_count_reachable: usize,
    config_tier: String,
    max_iterations: usize,
    max_classes: usize,
    /// The production budget's application cap (#1118). Replaces the old
    /// `hard_timeout_ms`: wall clock is a panicking safety ceiling now, not
    /// a stop reason, so it is not a budget dimension to report.
    max_applications: Option<u64>,
    stop: String,
    iterations: usize,
    total_unions: usize,
    applications: usize,
    classes_after: usize,
    cost: usize,
}

/// Enabler-starvation measurement (task 2): did the guided arms ever build the
/// classes that unguided's numeric strict-positives needed structural rules
/// to reach?
#[derive(Serialize, Deserialize, Clone, Default)]
struct EnablerDiag {
    unguided_strict_positive: usize,
    unguided_numeric_strict_positive: usize,
    /// Numeric strict-positives whose output node's tight derivation
    /// ancestry (excluding itself) contains a structural application.
    numeric_structurally_enabled: usize,
    /// Numeric strict-positives with a direct child class whose chosen node
    /// was created by a structural application.
    numeric_direct_child_structural: usize,
    /// Of `unguided_numeric_strict_positive`, how many output terms exist in
    /// each guided arm's final e-graph.
    numeric_present_in: BTreeMap<String, usize>,
    /// Of `numeric_structurally_enabled`, how many output terms exist in
    /// each guided arm's final e-graph.
    enabled_present_in: BTreeMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ExprRow {
    name: String,
    /// Identity of the evaluation configuration that produced this row —
    /// see [`run_config_identity`]. Resumption reuses a row only when this
    /// matches the current run, because a name alone says nothing about
    /// which checkpoint, guided grid, corpus, or revision produced the
    /// numbers under it. `None` on rows written before this field existed
    /// (aggregation still reads them; resumption refuses to extend them).
    #[serde(default)]
    run_config: Option<String>,
    tier: String,
    node_count: usize,
    class_cap: usize,
    /// Round-1b op-composition stratum (registration §2), present only when
    /// `--stratify-by-ops` was set for the run that produced this row.
    #[serde(default)]
    stratum: Option<String>,
    arms: BTreeMap<String, ArmCurve>,
    /// arm -> B -> diagnostics at B.
    at_budget: BTreeMap<String, BTreeMap<usize, AtBudgetDiag>>,
    /// Round 1b: arm -> the same diagnostics over EVERY recorded application
    /// of the run (registration §1.3: "applied at all by any arm at any
    /// checkpoint"). `None` on Round-1 rows.
    #[serde(default)]
    full_run: Option<BTreeMap<String, AtBudgetDiag>>,
    enabler: EnablerDiag,
    production: Option<ProductionRow>,
}

// ---------------------------------------------------------------------------
// Running one expression.
// ---------------------------------------------------------------------------

fn curve_to_arm(out: &AnytimeCurveOutput, seen_keys: Option<usize>) -> ArmCurve {
    let c = &out.curve;
    ArmCurve {
        grid: c.checkpoints.iter().map(|k| k.app_target).collect(),
        app_actual: c.checkpoints.iter().map(|k| k.app_actual).collect(),
        // DAG cost — what the emitted kernel pays (#1117).
        cost: c.checkpoints.iter().map(|k| k.cost.dag).collect(),
        rounds: c.checkpoints.iter().map(|k| k.sweeps).collect(),
        classes: c.checkpoints.iter().map(|k| k.classes).collect(),
        clamped: c.checkpoints.iter().map(|k| k.clamped).collect(),
        ended: stop_name(c.ended).to_string(),
        ended_at_apps: c.ended_at_apps,
        seen_keys,
    }
}

/// The canonical label of a recorded application's rule.
fn rule_label_of(rules: &RuleSet, id: RuleId) -> String {
    rules
        .index_of(id)
        .and_then(|i| rules.label_of(i))
        .unwrap_or_else(|| format!("<rule {}>", id.get()))
}

/// The [`RuleId`] a recorded application carries, or a loud failure.
fn rule_of(rec: &pixelflow_search::egraph::ApplicationRecord) -> RuleId {
    rec.rule.unwrap_or_else(|| {
        panic!(
            "phase3_at_budget_eval: an application carries no RuleId — the graph was built \
             without rule ids, and every table here is keyed by identity"
        )
    })
}

/// Strict hindsight labels for a finished curve (its own final extraction).
fn strict_positives(out: &AnytimeCurveOutput) -> BTreeSet<ApplicationId> {
    EpisodeLabels::compute_strict(&out.egraph, out.root, &out.extraction.choices).load_bearing
}

fn at_budget_diag(
    out: &AnytimeCurveOutput,
    arm: &ArmCurve,
    positives: &BTreeSet<ApplicationId>,
    rules: &RuleSet,
    b: usize,
) -> AtBudgetDiag {
    prefix_diag(out, arm.apps_at(b), arm.rounds_at(b), positives, rules)
}

/// Diagnostics over the whole run: every recorded application, and the
/// sweep count at the curve's last checkpoint.
fn full_run_diag(
    out: &AnytimeCurveOutput,
    arm: &ArmCurve,
    positives: &BTreeSet<ApplicationId>,
    rules: &RuleSet,
) -> AtBudgetDiag {
    let total = out.egraph.provenance().applications().count();
    let rounds = *arm.rounds.last().expect("non-empty curve");
    prefix_diag(out, total, rounds, positives, rules)
}

/// What the first `applications` recorded applications of one arm's run
/// looked like.
fn prefix_diag(
    out: &AnytimeCurveOutput,
    applications: usize,
    rounds: usize,
    positives: &BTreeSet<ApplicationId>,
    rules: &RuleSet,
) -> AtBudgetDiag {
    let mut structural = 0usize;
    let mut strict_positive = 0usize;
    let mut rule_hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_rule: BTreeMap<String, RuleFiring> = BTreeMap::new();
    for (id, rec) in out.egraph.provenance().applications() {
        if id.as_u64() as usize >= applications {
            continue;
        }
        let name = rule_label_of(rules, rule_of(rec));
        if is_structural(&name) {
            structural += 1;
        }
        let positive = positives.contains(&id);
        if positive {
            strict_positive += 1;
        }
        *rule_hist.entry(name.clone()).or_default() += 1;
        let f = by_rule.entry(name).or_default();
        f.fired += 1;
        if positive {
            f.strict_positive += 1;
        }
    }
    AtBudgetDiag {
        applications,
        structural,
        strict_positive,
        rounds,
        rule_hist,
        by_rule: Some(by_rule),
    }
}

/// A concrete term read off one e-graph's chosen extraction, so it can be
/// looked up in ANOTHER e-graph (class ids do not transfer; structure does).
enum Term {
    Leaf(ENode),
    Op {
        op: &'static dyn pixelflow_search::egraph::Op,
        children: Vec<Term>,
    },
}

fn term_of_chosen(
    egraph: &EGraph,
    choices: &[Option<usize>],
    class: EClassId,
    node_idx: usize,
) -> Term {
    let node = &egraph.nodes(class)[node_idx];
    match node {
        ENode::Op { op, children } => Term::Op {
            op: *op,
            children: children
                .iter()
                .map(|&child| {
                    let c = egraph.find(child);
                    let idx = choices.get(c.index()).and_then(|o| *o).unwrap_or_else(|| {
                        panic!(
                            "term_of_chosen: child class {} of a chosen node has no extraction choice",
                            c.index()
                        )
                    });
                    term_of_chosen(egraph, choices, c, idx)
                })
                .collect(),
        },
        leaf => Term::Leaf(leaf.clone()),
    }
}

/// All `(class, node)` pairs of an e-graph, canonical classes only.
struct NodeIndex<'a> {
    egraph: &'a EGraph,
    entries: Vec<(EClassId, &'a ENode)>,
}

impl<'a> NodeIndex<'a> {
    fn new(egraph: &'a EGraph) -> Self {
        let mut entries = Vec::new();
        for c in egraph.class_ids() {
            for n in egraph.nodes(c) {
                entries.push((c, n));
            }
        }
        Self { egraph, entries }
    }

    /// The canonical class containing `term`, if this e-graph has built it.
    fn lookup(&self, term: &Term) -> Option<EClassId> {
        match term {
            Term::Leaf(leaf) => self
                .entries
                .iter()
                .find(|(_, n)| *n == leaf)
                .map(|(c, _)| *c),
            Term::Op { op, children } => {
                let mut child_classes = Vec::with_capacity(children.len());
                for ch in children {
                    child_classes.push(self.lookup(ch)?);
                }
                let kind = op.kind();
                self.entries
                    .iter()
                    .find(|(_, n)| match n {
                        ENode::Op {
                            op: o,
                            children: cs,
                        } => {
                            o.kind() == kind
                                && cs.len() == child_classes.len()
                                && cs
                                    .iter()
                                    .zip(&child_classes)
                                    .all(|(a, b)| self.egraph.find(*a) == *b)
                        }
                        _ => false,
                    })
                    .map(|(c, _)| *c)
            }
        }
    }
}

fn enabler_diag(
    unguided: &AnytimeCurveOutput,
    unguided_positives: &BTreeSet<ApplicationId>,
    guided: &[(&str, &AnytimeCurveOutput)],
    rules: &RuleSet,
) -> EnablerDiag {
    let eg = &unguided.egraph;
    let prov = eg.provenance();
    let choices = &unguided.extraction.choices;

    // Output nodes of every application (a rule RHS may create several),
    // each tag's class, and the set of tags the extraction chose. A strict
    // positive is an application at least one of whose outputs is chosen;
    // the enabler analysis looks at exactly those chosen outputs.
    let mut outputs_of: HashMap<ApplicationId, Vec<ENodeId>> = HashMap::new();
    for (tag, origin) in prov.origins() {
        if let Origin::Rule(app) = origin {
            outputs_of.entry(app).or_default().push(tag);
        }
    }
    let mut class_of_tag: HashMap<ENodeId, (EClassId, usize)> = HashMap::new();
    for c in eg.class_ids() {
        for (i, tag) in eg.tags(c).iter().enumerate() {
            class_of_tag.insert(*tag, (c, i));
        }
    }
    // The extraction's complete class -> chosen-node map, and the set of
    // chosen tags. `derivation_ancestors_tight` prunes with the map: handed
    // only the one output node whose ancestry is being asked about, every
    // class the walk descends into is absent from it and falls back to "all
    // tags in the class", letting unrelated structural alternatives mark the
    // application structurally enabled — the loose bound inflating a
    // diagnostic that reports itself as the tight one.
    let chosen_map: Vec<(EClassId, ENodeId)> = {
        let mut seen: HashSet<EClassId> = HashSet::new();
        let mut stack = vec![unguided.root];
        let mut out = Vec::new();
        while let Some(c) = stack.pop() {
            let c = eg.find(c);
            if !seen.insert(c) {
                continue;
            }
            let idx = choices[c.index()].unwrap_or_else(|| {
                panic!(
                    "class {} reachable via the chosen extraction has no choice",
                    c.index()
                )
            });
            out.push((c, eg.tags(c)[idx]));
            stack.extend(eg.nodes(c)[idx].children());
        }
        out
    };
    let chosen: HashSet<ENodeId> = chosen_map.iter().map(|&(_, t)| t).collect();
    let structural_app = |app: ApplicationId| -> bool {
        let rec = prov
            .application(app)
            .unwrap_or_else(|| panic!("no record for {app:?}"));
        is_structural(&rule_label_of(rules, rule_of(rec)))
    };

    let indexes: Vec<(&str, NodeIndex<'_>)> = guided
        .iter()
        .map(|(name, out)| (*name, NodeIndex::new(&out.egraph)))
        .collect();

    let mut d = EnablerDiag::default();
    for (name, _) in guided {
        d.numeric_present_in.insert(name.to_string(), 0);
        d.enabled_present_in.insert(name.to_string(), 0);
    }
    d.unguided_strict_positive = unguided_positives.len();

    for &app in unguided_positives {
        if structural_app(app) {
            continue;
        }
        d.unguided_numeric_strict_positive += 1;
        let chosen_outputs: Vec<ENodeId> = outputs_of
            .get(&app)
            .unwrap_or_else(|| panic!("strict-positive {app:?} has no output node"))
            .iter()
            .copied()
            .filter(|t| chosen.contains(t))
            .collect();
        assert!(
            !chosen_outputs.is_empty(),
            "strict-positive {app:?} has no chosen output node — label/extraction disagree"
        );
        // Credit the application once; "enabled" / "present" if ANY of its
        // chosen outputs is.
        let mut enabled = false;
        let mut direct = false;
        let mut terms = Vec::new();
        for tag in chosen_outputs {
            let (class, node_idx) = *class_of_tag.get(&tag).unwrap_or_else(|| {
                panic!("output node {tag:?} of {app:?} is in no canonical class")
            });
            let mut ancestry = eg.derivation_ancestors_tight_from(&[(class, tag)], &chosen_map);
            ancestry.remove(&app);
            enabled |= ancestry.iter().any(|&a| structural_app(a));
            let node = &eg.nodes(class)[node_idx];
            direct |= node.children().iter().any(|&child| {
                let c = eg.find(child);
                let idx = choices[c.index()].expect("chosen child has a choice");
                matches!(prov.origin(eg.tags(c)[idx]), Some(Origin::Rule(a)) if structural_app(a))
            });
            terms.push(term_of_chosen(eg, choices, class, node_idx));
        }
        if enabled {
            d.numeric_structurally_enabled += 1;
        }
        if direct {
            d.numeric_direct_child_structural += 1;
        }
        for (name, index) in &indexes {
            if terms.iter().any(|t| index.lookup(t).is_some()) {
                *d.numeric_present_in.get_mut(*name).expect("arm registered") += 1;
                if enabled {
                    *d.enabled_present_in.get_mut(*name).expect("arm registered") += 1;
                }
            }
        }
    }
    d
}

/// The claim arm's scorer: the Round-1 strict-bit head, or the R2G return
/// regressor (`--r2g-checkpoint`). An enum rather than `Box<dyn
/// SaturationGuide>` so the report can name which one ran without a second
/// out-of-band flag.
enum ClaimGuide {
    StrictBit(LinearCandidateGuide),
    Return(LinearReturnGuide),
}

impl ClaimGuide {
    fn label(&self) -> String {
        match self {
            ClaimGuide::StrictBit(_) => "LinearCandidateGuide".to_string(),
            ClaimGuide::Return(g) => format!("LinearReturnGuide[{}]", g.objective()),
        }
    }
}

impl Clone for ClaimGuide {
    fn clone(&self) -> Self {
        match self {
            ClaimGuide::StrictBit(g) => ClaimGuide::StrictBit(g.clone()),
            ClaimGuide::Return(g) => ClaimGuide::Return(g.clone()),
        }
    }
}

impl SaturationGuide for ClaimGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        match self {
            ClaimGuide::StrictBit(g) => g.score_candidates(candidates),
            ClaimGuide::Return(g) => g.score_candidates(candidates),
        }
    }
}

struct Guides {
    control: PerRuleRateGuide,
    linear: ClaimGuide,
}

/// One expression's fixed curve environment, shared by every arm.
#[derive(Clone, Copy)]
struct CurveInput<'a> {
    arena: &'a ExprArena,
    root: ExprId,
    class_cap: usize,
    costs: &'a CostModel,
    guided_grid: &'a [usize],
}

/// The curve environment every arm shares, differing only in whether a
/// [`SaturationGuide`] is attached — which is the whole lever, and after
/// #1108 it is a field rather than a second entry point.
fn arm_optimizer(input: &CurveInput<'_>, guide: Option<Box<dyn SaturationGuide>>) -> Optimizer {
    Optimizer::production()
        .cost(input.costs.clone())
        .guide(guide)
        // The enabler and strict-label diagnostics read the journal off the
        // graph after the run, so recording has to be asked for (#1118).
        .observe(Some(Box::new(KeepJournal)))
        .budget(Budget::Explicit {
            iterations: SWEEP_SAFETY_CEILING,
            classes: input.class_cap,
            applications: None,
        })
        .hard_ceiling(SAFETY_TIMEOUT)
}

fn run_guided(
    guide: Box<dyn SaturationGuide>,
    input: &CurveInput<'_>,
) -> (AnytimeCurveOutput, usize) {
    let mut optimizer = arm_optimizer(input, Some(guide));
    let out = run_anytime_curve(&mut optimizer, input.arena, input.root, input.guided_grid);
    let seen = optimizer
        .guided_keys_seen()
        .expect("a guided optimizer carries an episode");
    (out, seen)
}

/// Run the production saturation step offline for one expression and report
/// what it did.
///
/// `Optimizer::production()` **is** the production configuration — there is
/// no second copy of it to drift (#1108 removed the one there was), so this
/// adds only the reporting: production discards its stats, this keeps them.
fn production_probe(arena: &ExprArena, root: ExprId, costs: &CostModel) -> ProductionRow {
    let node_count = arena.nodes_raw().len();
    let mut optimizer = Optimizer::production().cost(costs.clone());
    let mut egraph = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert(
        arena,
        root,
        &mut egraph,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let limits = optimizer.limits_for(pixelflow_search::egraph::InputSize {
        nodes: node_count,
        classes: egraph.num_classes(),
    });
    let out = optimizer.run(&mut egraph, root_class, node_count);
    ProductionRow {
        node_count_reachable: node_count,
        config_tier: match limits.iterations {
            20 => "blitz",
            50 => "rapid",
            100 => "classical",
            other => panic!("unknown production config tier: {other} iterations"),
        }
        .to_string(),
        max_iterations: limits.iterations,
        max_classes: limits.classes,
        // The production budget's application dimension (#1118). The
        // wall-clock `hard_timeout` this field used to carry is gone: it is
        // a panicking safety ceiling now, never a stop reason, so reporting
        // it as one would be reporting a metric that cannot fire.
        max_applications: limits.applications,
        stop: stop_name(out.stats.stop).to_string(),
        iterations: out.stats.iterations,
        total_unions: out.stats.unions,
        applications: out.stats.applications as usize,
        classes_after: out.stats.classes,
        // DAG cost — what the emitted kernel pays (#1117).
        cost: out.cost.dag,
    }
}

fn evaluate_expression(
    name: &str,
    input: &CurveInput<'_>,
    guides: &Guides,
    rules: &RuleSet,
    stratify_by_ops: bool,
) -> ExprRow {
    let CurveInput {
        arena,
        root,
        class_cap,
        costs,
        ..
    } = *input;
    let node_count = arena.nodes_raw().len();

    let mut unguided_opt = arm_optimizer(input, None);
    let unguided = run_anytime_curve(&mut unguided_opt, arena, root, APP_CHECKPOINT_GRID);
    let (control, control_seen) = run_guided(Box::new(guides.control.clone()), input);
    let (linear, linear_seen) = run_guided(Box::new(guides.linear.clone()), input);

    let outs: [(&str, &AnytimeCurveOutput, Option<usize>); 3] = [
        ("unguided", &unguided, None),
        ("control", &control, Some(control_seen)),
        ("linear", &linear, Some(linear_seen)),
    ];

    let mut arms = BTreeMap::new();
    let mut at_budget = BTreeMap::new();
    let mut full_run = BTreeMap::new();
    let mut unguided_positives = BTreeSet::new();
    for (arm_name, out, seen) in outs {
        let arm = curve_to_arm(out, seen);
        let positives = strict_positives(out);
        let mut per_b = BTreeMap::new();
        for (b, _, _) in REGISTERED_TIERS {
            per_b.insert(b, at_budget_diag(out, &arm, &positives, rules, b));
        }
        full_run.insert(
            arm_name.to_string(),
            full_run_diag(out, &arm, &positives, rules),
        );
        if arm_name == "unguided" {
            unguided_positives = positives;
        }
        at_budget.insert(arm_name.to_string(), per_b);
        arms.insert(arm_name.to_string(), arm);
    }

    let enabler = enabler_diag(
        &unguided,
        &unguided_positives,
        &[("control", &control), ("linear", &linear)],
        rules,
    );

    let production = Some(production_probe(arena, root, costs));

    ExprRow {
        name: name.to_string(),
        // Stamped by the caller, which owns the run's configuration.
        run_config: None,
        tier: tier_name(node_count).to_string(),
        node_count,
        class_cap,
        stratum: stratify_by_ops.then(|| ops_stratum(arena).to_string()),
        arms,
        at_budget,
        full_run: Some(full_run),
        enabler,
        production,
    }
}

// ---------------------------------------------------------------------------
// Aggregation.
// ---------------------------------------------------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty(), "percentile of empty slice");
    let pos = p * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[derive(Serialize, Clone)]
struct Dist {
    n: usize,
    q1: f64,
    median: f64,
    q3: f64,
    p90: f64,
    max: f64,
    inf_count: usize,
}

fn dist(values: &[f64]) -> Dist {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("NaN in distribution"));
    if v.is_empty() {
        return Dist {
            n: 0,
            q1: f64::NAN,
            median: f64::NAN,
            q3: f64::NAN,
            p90: f64::NAN,
            max: f64::NAN,
            inf_count: 0,
        };
    }
    Dist {
        n: v.len(),
        q1: percentile(&v, 0.25),
        median: percentile(&v, 0.5),
        q3: percentile(&v, 0.75),
        p90: percentile(&v, 0.9),
        max: *v.last().expect("non-empty"),
        inf_count: v.iter().filter(|x| x.is_infinite()).count(),
    }
}

/// `a / b` with the zero-reference convention.
fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        if a == 0 { 1.0 } else { f64::INFINITY }
    } else {
        a as f64 / b as f64
    }
}

/// `(a - r) / r * 100` with the zero-reference convention.
fn pct_over(a: usize, r: usize) -> f64 {
    if r == 0 {
        if a == 0 { 0.0 } else { f64::INFINITY }
    } else {
        (a as f64 - r as f64) / r as f64 * 100.0
    }
}

#[derive(Serialize, Clone)]
struct ArmAtB {
    arm: String,
    /// `arm_cost@B / unguided_cost@B`, per expression.
    ratio_vs_unguided_at_b: Dist,
    improved: usize,
    unchanged: usize,
    worse: usize,
    /// `(arm_cost@B - unguided_cost@4B) / unguided_cost@4B` in percent.
    gap_vs_unguided_at_4b_pct: Dist,
    /// `(arm_cost@B - best) / best` in percent, best = empirical best of all
    /// arms at any checkpoint.
    regret_pct: Dist,
    /// Share of the unguided truncation gap `(u@B - u@4B)` the arm closed, on
    /// expressions with a positive gap.
    gap_closed_frac: Dist,
    /// Diagnostics at B: pooled structural share of applications, pooled
    /// strict-positive share (precision), median rounds to reach B.
    structural_share_pooled: f64,
    strict_precision_pooled: f64,
    rounds_to_b: Dist,
    app_actual_at_b: Dist,
    /// The unguided arm's ACTUAL application count at the same grid target —
    /// the number `ratio_vs_unguided_at_b` implicitly gives the unguided arm
    /// and this arm does not get. Reported so the headline ratio is never
    /// read as an equal-work comparison without the reader seeing the gap.
    unguided_app_actual_at_b: Dist,
    /// `arm_cost / unguided_cost@B` with the arm read at the first checkpoint
    /// whose ACTUAL application count reaches the unguided arm's actual count
    /// at B — the equal-work counterpart to `ratio_vs_unguided_at_b`.
    ratio_vs_unguided_at_matched_apps: Dist,
    /// `regret_pct` against the best cost any arm reaches on the grid
    /// prefix every arm actually ran (see `ArmCurve::best_cost_up_to`).
    regret_on_common_grid_pct: Dist,
    /// Expressions where this arm reaches that common-grid best.
    reaches_common_grid_best: usize,
    /// Guided runs that had already quiesced (dedup exhaustion) before
    /// reaching B applications — for them "at B" is "at quiescence".
    ended_before_b: usize,
    /// Expressions where this arm's curve reaches the empirical best cost
    /// somewhere along it.
    reaches_empirical_best: usize,
    /// This arm's cost@B against the production call's cost.
    vs_production_better: usize,
    vs_production_equal: usize,
    vs_production_worse: usize,
    /// Applications when the guided run ended (quiesced or grid end).
    ended_at_apps: Dist,
    /// `(seen_keys - recorded applications) / seen_keys` on quiesced runs:
    /// candidate keys the loop marked seen but whose `apply_single_rule`
    /// recorded nothing (stale `node_idx` after an earlier application in
    /// the same round rebuilt the class) — burned, never retried.
    burned_key_share: Dist,
}

/// Round-1b domain-shift statistic (registration §1): `D_A(S, B) =
/// m_A^S(B) minus m_A^DEV(B)` per arm against Round 1's fixed DEV-overall
/// classical reference, the registered margin `M_B`, and the resulting
/// verdict.
/// Populated only for the classical band — `m_A^DEV(B)` is specifically the
/// classical-band Round-1 reference and is not meaningful against a
/// rapid/blitz set.
#[derive(Serialize, Clone)]
struct Round1bStat {
    /// `m_A^DEV(B)`, per arm — quoted from Round 1, not recomputed here.
    dev_reference_median: BTreeMap<String, f64>,
    /// `D_A(S, B)`, per arm, for THIS run's evaluated set `S`.
    d_arm: BTreeMap<String, f64>,
    /// `M_B` — the largest within-DEV `|D_control - D_linear|` Round 1
    /// already produced across its 8 families (§1.1).
    margin_m: f64,
    /// `D_control - D_linear`.
    d_diff_control_minus_linear: f64,
    /// Registration §1.2 (the Bézier / polynomial-only prediction): point
    /// prediction `D_A <= 0` for both arms, and the binding form
    /// `D_A <= +M_B` for both arms. Reported on every set; only meaningful
    /// as a prediction on `bezier` and the `polynomial-only` stratum.
    both_arms_d_le_zero: bool,
    both_arms_d_le_margin: bool,
    /// `H_shift` / `H_null` / `H_inv` / `underpowered (n < 30)`, per §1's
    /// pre-committed decision rule. `unclassified` covers the residual case
    /// `D_control - D_linear > M` with `D_control <= 0` — outside all three
    /// named hypotheses, reported rather than silently forced into one.
    verdict: String,
}

#[derive(Serialize, Clone)]
struct TierResult {
    band: String,
    b: usize,
    n: usize,
    y_registered: f64,
    l_registered_pct: f64,
    /// Registered Y-clause threshold on the median ratio.
    ratio_threshold: f64,
    /// Registered 4B-approach threshold on the median gap (L/2, percent).
    gap_threshold_pct: f64,
    unguided_regret_at_b_pct: Dist,
    unguided_regret_at_4b_pct: Dist,
    unguided_truncation_loss_pct: Dist,
    arms: Vec<ArmAtB>,
    /// Per-arm verdicts against the registered claim (classical only is
    /// binding; other bands carry the same fields with no claim).
    y_clause_holds: BTreeMap<String, bool>,
    approach_clause_holds: BTreeMap<String, bool>,
    beats_unguided_at_all: BTreeMap<String, bool>,
    linear_beats_control: bool,
    unguided_structural_share_pooled: f64,
    unguided_strict_precision_pooled: f64,
    /// Head-to-head at B: expressions where linear's cost is below / equal
    /// to / above control's.
    linear_lt_control: usize,
    linear_eq_control: usize,
    linear_gt_control: usize,
    /// Round-1b domain-shift statistic (classical band only; `None` for
    /// rapid/blitz).
    round1b: Option<Round1bStat>,
}

#[derive(Serialize, Clone)]
struct ProductionSummary {
    band: String,
    n: usize,
    n_probe_none: usize,
    stops: BTreeMap<String, usize>,
    effective_b: Dist,
    share_apps_ge: BTreeMap<usize, f64>,
    /// Production cost relative to the unguided curve.
    ratio_vs_unguided_at_100: Dist,
    ratio_vs_unguided_at_200: Dist,
    ratio_vs_unguided_at_4b_800: Dist,
    regret_vs_best_pct: Dist,
    /// Smallest unguided grid checkpoint whose cost <= production's cost
    /// (`0` = production is worse than every unguided checkpoint; recorded
    /// as the grid target).
    equivalent_checkpoint_hist: BTreeMap<usize, usize>,
    /// Unguided grid checkpoint whose app_actual first reaches production's
    /// application count.
    app_bracket_hist: BTreeMap<usize, usize>,
    /// Rounds production ran vs its own cap.
    iterations: Dist,
    classes_after: Dist,
}

#[derive(Serialize, Clone)]
struct EnablerSummary {
    band: String,
    n: usize,
    unguided_strict_positive_total: usize,
    numeric_total: usize,
    numeric_structurally_enabled: usize,
    numeric_direct_child_structural: usize,
    numeric_present_in: BTreeMap<String, usize>,
    enabled_present_in: BTreeMap<String, usize>,
    /// Per-arm pooled structural share at B=100 and B=200.
    structural_share_at_b: BTreeMap<String, BTreeMap<usize, f64>>,
    /// Per-arm pooled top rules at B=100.
    top_rules_at_100: BTreeMap<String, Vec<(String, usize)>>,
    /// Guided arms: seen candidate keys / applications at end of run.
    seen_keys_per_application: BTreeMap<String, Dist>,
}

#[derive(Serialize)]
struct Report {
    n_rows: usize,
    n_by_band: BTreeMap<String, usize>,
    skipped_names: Vec<String>,
    tiers: Vec<TierResult>,
    production: Vec<ProductionSummary>,
    enabler: Vec<EnablerSummary>,
    /// Round-1b op-composition stratification (registration §2); `None`
    /// unless `--stratify-by-ops` was set.
    strata: Option<StrataReport>,
    /// Round-1b §1.3 trig-rule firings: one entry for the classical band and
    /// one per op-composition stratum evaluated.
    rule_firing: Vec<RuleFiringSummary>,
    context: BTreeMap<String, String>,
}

/// Round-1b op-composition stratification output (registration §2/§1).
#[derive(Serialize, Clone)]
struct StrataReport {
    /// Stratum -> count over the FULL classical DEV population read from
    /// the corpus this run loaded — independent of how many of them were
    /// actually put through the ablation ladder (`--classical-samples`).
    population_counts: BTreeMap<String, usize>,
    /// One `TierResult` per (stratum, registered B) over the classical rows
    /// this run actually evaluated that carry a stratum label — `band` on
    /// each is the stratum name; `round1b` is always populated.
    evaluated: Vec<TierResult>,
}

/// The registered-tier parameters `tier_result` needs beyond `rows`/`band` —
/// grouped so the function stays under the workspace's argument-count lint
/// (CLAUDE.md: "Functions < 4 arguments (group into structs)").
struct TierBudget {
    b: usize,
    y: f64,
    l: f64,
    /// Whether to compute the round-1b `D_A` statistic (§1) for this set —
    /// true for the whole-DEV classical band and for each op-composition
    /// stratum of it, false for rapid/blitz.
    compute_round1b: bool,
}

fn tier_result(rows: &[&ExprRow], band: &str, budget: TierBudget) -> TierResult {
    let TierBudget {
        b,
        y,
        l,
        compute_round1b,
    } = budget;
    let b4 = 4 * b;
    fn ung(r: &ExprRow) -> &ArmCurve {
        &r.arms["unguided"]
    }
    let best = |r: &ExprRow| {
        r.arms
            .values()
            .map(ArmCurve::best_cost)
            .min()
            .expect("arms")
    };
    // The largest grid target EVERY arm ran — the reference the arms can be
    // compared on without crediting one for checkpoints another was never
    // budgeted to reach.
    let common_limit = |r: &ExprRow| -> usize {
        r.arms
            .values()
            .map(|a| *a.grid.last().expect("non-empty curve"))
            .min()
            .expect("arms")
    };
    let best_common = |r: &ExprRow| {
        let limit = common_limit(r);
        r.arms
            .values()
            .map(|a| a.best_cost_up_to(limit))
            .min()
            .expect("arms")
    };

    let unguided_regret_at_b: Vec<f64> = rows
        .iter()
        .map(|r| pct_over(ung(r).cost_at(b), best(r)))
        .collect();
    let unguided_regret_at_4b: Vec<f64> = rows
        .iter()
        .map(|r| pct_over(ung(r).cost_at(b4), best(r)))
        .collect();
    let trunc: Vec<f64> = rows
        .iter()
        .map(|r| pct_over(ung(r).cost_at(b), ung(r).cost_at(b4)))
        .collect();

    let mut arms = Vec::new();
    for arm_name in ["control", "linear"] {
        let mut ratios = Vec::new();
        let mut gaps = Vec::new();
        let mut regrets = Vec::new();
        let mut closed = Vec::new();
        let (mut imp, mut unch, mut worse) = (0, 0, 0);
        let (mut apps, mut structural, mut positive) = (0usize, 0usize, 0usize);
        let mut rounds = Vec::new();
        let mut app_actual = Vec::new();
        let mut ended_before_b = 0usize;
        let mut reaches_best = 0usize;
        let (mut pb, mut pe, mut pw) = (0usize, 0usize, 0usize);
        let mut ended_at = Vec::new();
        let mut burned = Vec::new();
        let mut unguided_app_actual = Vec::new();
        let mut matched_ratios = Vec::new();
        let mut common_regrets = Vec::new();
        let mut reaches_common = 0usize;
        for r in rows {
            let a = &r.arms[arm_name];
            if a.ended != "app_budget" && a.ended_at_apps < b {
                ended_before_b += 1;
            }
            if a.best_cost() == best(r) {
                reaches_best += 1;
            }
            if let Some(p) = &r.production {
                match a.cost_at(b).cmp(&p.cost) {
                    std::cmp::Ordering::Less => pb += 1,
                    std::cmp::Ordering::Equal => pe += 1,
                    std::cmp::Ordering::Greater => pw += 1,
                }
            }
            ended_at.push(a.ended_at_apps as f64);
            if a.ended == "quiesced" {
                let seen = a.seen_keys.expect("guided arm records seen_keys");
                assert!(
                    seen >= a.ended_at_apps,
                    "{}: {} recorded more applications ({}) than candidate keys scored ({seen})",
                    r.name,
                    arm_name,
                    a.ended_at_apps
                );
                burned.push((seen - a.ended_at_apps) as f64 / seen.max(1) as f64);
            }
            let ca = a.cost_at(b);
            let cu = ung(r).cost_at(b);
            let cu4 = ung(r).cost_at(b4);
            let rt = ratio(ca, cu);
            ratios.push(rt);
            match ca.cmp(&cu) {
                std::cmp::Ordering::Less => imp += 1,
                std::cmp::Ordering::Equal => unch += 1,
                std::cmp::Ordering::Greater => worse += 1,
            }
            gaps.push(pct_over(ca, cu4));
            regrets.push(pct_over(ca, best(r)));
            if cu > cu4 {
                closed.push((cu as f64 - ca as f64) / (cu as f64 - cu4 as f64));
            }
            let u_apps = ung(r).apps_at(b);
            unguided_app_actual.push(u_apps as f64);
            matched_ratios.push(ratio(a.cost_at_apps(u_apps), cu));
            let bc = best_common(r);
            common_regrets.push(pct_over(ca, bc));
            if a.best_cost_up_to(common_limit(r)) == bc {
                reaches_common += 1;
            }

            let d = &r.at_budget[arm_name][&b];
            apps += d.applications;
            structural += d.structural;
            positive += d.strict_positive;
            rounds.push(d.rounds as f64);
            app_actual.push(a.apps_at(b) as f64);
        }
        arms.push(ArmAtB {
            arm: arm_name.to_string(),
            ratio_vs_unguided_at_b: dist(&ratios),
            unguided_app_actual_at_b: dist(&unguided_app_actual),
            ratio_vs_unguided_at_matched_apps: dist(&matched_ratios),
            regret_on_common_grid_pct: dist(&common_regrets),
            reaches_common_grid_best: reaches_common,
            improved: imp,
            unchanged: unch,
            worse,
            gap_vs_unguided_at_4b_pct: dist(&gaps),
            regret_pct: dist(&regrets),
            gap_closed_frac: dist(&closed),
            structural_share_pooled: if apps == 0 {
                0.0
            } else {
                structural as f64 / apps as f64
            },
            strict_precision_pooled: if apps == 0 {
                0.0
            } else {
                positive as f64 / apps as f64
            },
            rounds_to_b: dist(&rounds),
            app_actual_at_b: dist(&app_actual),
            ended_before_b,
            reaches_empirical_best: reaches_best,
            vs_production_better: pb,
            vs_production_equal: pe,
            vs_production_worse: pw,
            ended_at_apps: dist(&ended_at),
            burned_key_share: dist(&burned),
        });
    }

    let ratio_threshold = 1.0 - y;
    let gap_threshold_pct = l / 2.0;
    let mut y_clause = BTreeMap::new();
    let mut approach = BTreeMap::new();
    let mut beats = BTreeMap::new();
    for a in &arms {
        y_clause.insert(
            a.arm.clone(),
            a.ratio_vs_unguided_at_b.median <= ratio_threshold,
        );
        approach.insert(
            a.arm.clone(),
            a.gap_vs_unguided_at_4b_pct.median <= gap_threshold_pct,
        );
        beats.insert(a.arm.clone(), a.ratio_vs_unguided_at_b.median < 1.0);
    }
    let linear_beats_control = arms[1].regret_pct.median < arms[0].regret_pct.median;
    let (mut u_apps, mut u_struct, mut u_pos) = (0usize, 0usize, 0usize);
    let (mut lt, mut eq, mut gt) = (0usize, 0usize, 0usize);
    for r in rows {
        let d = &r.at_budget["unguided"][&b];
        u_apps += d.applications;
        u_struct += d.structural;
        u_pos += d.strict_positive;
        match r.arms["linear"]
            .cost_at(b)
            .cmp(&r.arms["control"].cost_at(b))
        {
            std::cmp::Ordering::Less => lt += 1,
            std::cmp::Ordering::Equal => eq += 1,
            std::cmp::Ordering::Greater => gt += 1,
        }
    }

    // Round-1b domain-shift statistic (registration §1) — only for a
    // classical-sized set, since `m_A^DEV(B)` is the classical-band Round-1
    // reference (the caller decides: the whole-DEV classical band, or a
    // classical-band op-composition stratum; never rapid/blitz).
    let round1b = compute_round1b.then(|| {
        let mut dev_reference_median = BTreeMap::new();
        let mut d_arm = BTreeMap::new();
        for a in &arms {
            let refm = dev_reference_median_for(b, &a.arm);
            dev_reference_median.insert(a.arm.clone(), refm);
            d_arm.insert(a.arm.clone(), a.ratio_vs_unguided_at_b.median - refm);
        }
        let margin_m = registered_margin(b);
        let d_control = d_arm["control"];
        let d_linear = d_arm["linear"];
        let d_diff_control_minus_linear = d_control - d_linear;
        let verdict = if rows.len() < MIN_N_FOR_VERDICT {
            format!("underpowered (n={} < {MIN_N_FOR_VERDICT})", rows.len())
        } else if d_diff_control_minus_linear.abs() <= margin_m {
            "H_null".to_string()
        } else if d_diff_control_minus_linear > margin_m {
            if d_control > 0.0 {
                "H_shift".to_string()
            } else {
                format!(
                    "unclassified (D_control-D_linear={d_diff_control_minus_linear:.4} > M={margin_m:.2} but D_control={d_control:.4} <= 0)"
                )
            }
        } else {
            "H_inv".to_string()
        };
        Round1bStat {
            dev_reference_median,
            d_arm,
            margin_m,
            d_diff_control_minus_linear,
            both_arms_d_le_zero: d_control <= 0.0 && d_linear <= 0.0,
            both_arms_d_le_margin: d_control <= margin_m && d_linear <= margin_m,
            verdict,
        }
    });

    TierResult {
        band: band.to_string(),
        b,
        n: rows.len(),
        y_registered: y,
        l_registered_pct: l,
        ratio_threshold,
        gap_threshold_pct,
        unguided_regret_at_b_pct: dist(&unguided_regret_at_b),
        unguided_regret_at_4b_pct: dist(&unguided_regret_at_4b),
        unguided_truncation_loss_pct: dist(&trunc),
        arms,
        y_clause_holds: y_clause,
        approach_clause_holds: approach,
        beats_unguided_at_all: beats,
        linear_beats_control,
        unguided_structural_share_pooled: if u_apps == 0 {
            0.0
        } else {
            u_struct as f64 / u_apps as f64
        },
        unguided_strict_precision_pooled: if u_apps == 0 {
            0.0
        } else {
            u_pos as f64 / u_apps as f64
        },
        linear_lt_control: lt,
        linear_eq_control: eq,
        linear_gt_control: gt,
        round1b,
    }
}

fn production_summary(rows: &[&ExprRow], band: &str) -> ProductionSummary {
    let with: Vec<(&ExprRow, &ProductionRow)> = rows
        .iter()
        .filter_map(|r| r.production.as_ref().map(|p| (*r, p)))
        .collect();
    let mut stops: BTreeMap<String, usize> = BTreeMap::new();
    let mut eff = Vec::new();
    let mut r100 = Vec::new();
    let mut r200 = Vec::new();
    let mut r800 = Vec::new();
    let mut regret = Vec::new();
    let mut eq_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut br_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut iters = Vec::new();
    let mut classes = Vec::new();
    for (r, p) in &with {
        *stops.entry(p.stop.clone()).or_default() += 1;
        eff.push(p.applications as f64);
        let u = &r.arms["unguided"];
        r100.push(ratio(p.cost, u.cost_at(100)));
        r200.push(ratio(p.cost, u.cost_at(200)));
        r800.push(ratio(p.cost, u.cost_at(800)));
        let best = r
            .arms
            .values()
            .map(ArmCurve::best_cost)
            .min()
            .expect("arms");
        regret.push(pct_over(p.cost, best.min(p.cost)));
        let eq = u
            .grid
            .iter()
            .zip(&u.cost)
            .find(|(_, c)| **c <= p.cost)
            .map(|(&g, _)| g)
            .unwrap_or(0);
        *eq_hist.entry(eq).or_default() += 1;
        let br = u
            .grid
            .iter()
            .zip(&u.app_actual)
            .find(|(_, a)| **a >= p.applications)
            .map(|(&g, _)| g)
            .unwrap_or(usize::MAX);
        *br_hist.entry(br).or_default() += 1;
        iters.push(p.iterations as f64);
        classes.push(p.classes_after as f64);
    }
    let mut share = BTreeMap::new();
    for t in [100usize, 200, 400, 800, 1600] {
        let c = with.iter().filter(|(_, p)| p.applications >= t).count();
        share.insert(
            t,
            if with.is_empty() {
                0.0
            } else {
                c as f64 / with.len() as f64
            },
        );
    }
    ProductionSummary {
        band: band.to_string(),
        n: with.len(),
        n_probe_none: rows.len() - with.len(),
        stops,
        effective_b: dist(&eff),
        share_apps_ge: share,
        ratio_vs_unguided_at_100: dist(&r100),
        ratio_vs_unguided_at_200: dist(&r200),
        ratio_vs_unguided_at_4b_800: dist(&r800),
        regret_vs_best_pct: dist(&regret),
        equivalent_checkpoint_hist: eq_hist,
        app_bracket_hist: br_hist,
        iterations: dist(&iters),
        classes_after: dist(&classes),
    }
}

fn enabler_summary(rows: &[&ExprRow], band: &str) -> EnablerSummary {
    let mut s = EnablerSummary {
        band: band.to_string(),
        n: rows.len(),
        unguided_strict_positive_total: 0,
        numeric_total: 0,
        numeric_structurally_enabled: 0,
        numeric_direct_child_structural: 0,
        numeric_present_in: BTreeMap::new(),
        enabled_present_in: BTreeMap::new(),
        structural_share_at_b: BTreeMap::new(),
        top_rules_at_100: BTreeMap::new(),
        seen_keys_per_application: BTreeMap::new(),
    };
    let mut hist: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut pooled: BTreeMap<String, BTreeMap<usize, (usize, usize)>> = BTreeMap::new();
    let mut seen: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in rows {
        s.unguided_strict_positive_total += r.enabler.unguided_strict_positive;
        s.numeric_total += r.enabler.unguided_numeric_strict_positive;
        s.numeric_structurally_enabled += r.enabler.numeric_structurally_enabled;
        s.numeric_direct_child_structural += r.enabler.numeric_direct_child_structural;
        for (k, v) in &r.enabler.numeric_present_in {
            *s.numeric_present_in.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &r.enabler.enabled_present_in {
            *s.enabled_present_in.entry(k.clone()).or_default() += v;
        }
        for (arm, per_b) in &r.at_budget {
            for (b, d) in per_b {
                let e = pooled
                    .entry(arm.clone())
                    .or_default()
                    .entry(*b)
                    .or_default();
                e.0 += d.structural;
                e.1 += d.applications;
                if *b == 100 {
                    let h = hist.entry(arm.clone()).or_default();
                    for (rule, n) in &d.rule_hist {
                        *h.entry(rule.clone()).or_default() += n;
                    }
                }
            }
        }
        for (arm, a) in &r.arms {
            if let Some(k) = a.seen_keys {
                seen.entry(arm.clone())
                    .or_default()
                    .push(k as f64 / a.ended_at_apps.max(1) as f64);
            }
        }
    }
    for (arm, per_b) in pooled {
        let m: BTreeMap<usize, f64> = per_b
            .into_iter()
            .map(|(b, (st, ap))| (b, if ap == 0 { 0.0 } else { st as f64 / ap as f64 }))
            .collect();
        s.structural_share_at_b.insert(arm, m);
    }
    for (arm, h) in hist {
        let mut v: Vec<(String, usize)> = h.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(8);
        s.top_rules_at_100.insert(arm, v);
    }
    for (arm, v) in seen {
        s.seen_keys_per_application.insert(arm, dist(&v));
    }
    s
}

/// One registered trig rule's pooled firings in one set, per arm
/// (round-1b registration §1.3).
#[derive(Serialize, Clone, Default)]
struct RuleFiringAgg {
    rule_name: String,
    /// Pooled over the set: firings within the first B applications and
    /// how many of them were strict-positive, B = 100 / 200.
    fired_at_100: usize,
    strict_at_100: usize,
    fired_at_200: usize,
    strict_at_200: usize,
    /// Pooled over every recorded application of the run.
    fired_full: usize,
    strict_full: usize,
    /// Expressions on which the rule fired at all, at any point of the run.
    exprs_with_firing_full: usize,
}

#[derive(Serialize, Clone)]
struct RuleFiringSummary {
    set: String,
    n: usize,
    /// Rows that carry the per-rule-index histogram (Round-1 rows do not);
    /// every count below is pooled over exactly these rows.
    n_with_rule_index: usize,
    /// arm -> canonical rule label -> pooled firings.
    arms: BTreeMap<String, BTreeMap<String, RuleFiringAgg>>,
}

fn rule_firing_summary(rows: &[&ExprRow], set: &str, rules: &RuleSet) -> RuleFiringSummary {
    let trig = trig_rule_ids(rules);
    let mut arms: BTreeMap<String, BTreeMap<String, RuleFiringAgg>> = BTreeMap::new();
    for arm in ARM_NAMES {
        let mut per_rule = BTreeMap::new();
        for (_, label) in &trig {
            per_rule.insert(
                (*label).to_string(),
                RuleFiringAgg {
                    rule_name: (*label).to_string(),
                    ..RuleFiringAgg::default()
                },
            );
        }
        arms.insert(arm.to_string(), per_rule);
    }
    let mut n_with_rule_index = 0usize;
    for r in rows {
        let Some(full) = &r.full_run else {
            continue;
        };
        n_with_rule_index += 1;
        for arm in ARM_NAMES {
            let per_rule = arms.get_mut(arm).expect("arm registered");
            let at = |b: usize| -> &BTreeMap<String, RuleFiring> {
                r.at_budget[arm][&b]
                    .by_rule
                    .as_ref()
                    .unwrap_or_else(|| panic!("{}: {arm}@{b} has full_run but no by_rule — a row written by two different harness revisions", r.name))
            };
            let (h100, h200) = (at(100), at(200));
            let hfull = full[arm]
                .by_rule
                .as_ref()
                .unwrap_or_else(|| panic!("{}: {arm} full_run has no by_rule", r.name));
            for (label, agg) in per_rule.iter_mut() {
                let f100 = h100.get(label).copied().unwrap_or_default();
                let f200 = h200.get(label).copied().unwrap_or_default();
                let ffull = hfull.get(label).copied().unwrap_or_default();
                agg.fired_at_100 += f100.fired;
                agg.strict_at_100 += f100.strict_positive;
                agg.fired_at_200 += f200.fired;
                agg.strict_at_200 += f200.strict_positive;
                agg.fired_full += ffull.fired;
                agg.strict_full += ffull.strict_positive;
                if ffull.fired > 0 {
                    agg.exprs_with_firing_full += 1;
                }
            }
        }
    }
    RuleFiringSummary {
        set: set.to_string(),
        n: rows.len(),
        n_with_rule_index,
        arms,
    }
}

fn fmt_rate(fired: usize, strict: usize) -> String {
    if fired == 0 {
        "0".to_string()
    } else {
        format!(
            "{fired} ({strict}, {:.1}%)",
            100.0 * strict as f64 / fired as f64
        )
    }
}

/// Round-1b §1.3 trig-rule firing section: per set, per arm, per registered
/// rule index — firings (strict-positives, strict rate) at B=100, B=200 and
/// over the full run, plus the number of expressions the rule fired on.
fn write_rule_firing(md: &mut String, summaries: &[RuleFiringSummary]) {
    md.push_str("## Trig-rule firings (round-1b registration §1.3 / §4)\n\n");
    md.push_str("Cells are `fired (strict-positive, strict rate)` pooled over the set's rows that carry the per-rule-index histogram; `exprs` = expressions on which the arm fired the rule at any point of its run. A rule with 0 firings in every arm on a set has no live match there — a precondition to state before reading D on that set.\n\n");
    for s in summaries {
        md.push_str(&format!(
            "### set = {} (n = {}, rows with rule-index histogram = {})\n\n",
            s.set, s.n, s.n_with_rule_index
        ));
        if s.n_with_rule_index == 0 {
            md.push_str("No row in this set carries the per-rule-index histogram (rows predate it); nothing is counted here — NOT zero firings.\n\n");
            continue;
        }
        md.push_str("| idx | rule | arm | @100 | @200 | full run | exprs |\n|---:|---|---|---|---|---|---:|\n");
        for (&idx, label) in TRIG_RULE_IDX.iter().zip(TRIG_RULE_LABELS) {
            for arm in ARM_NAMES {
                let a = &s.arms[arm][label];
                md.push_str(&format!(
                    "| {idx} | {} | {arm} | {} | {} | {} | {} |\n",
                    a.rule_name,
                    fmt_rate(a.fired_at_100, a.strict_at_100),
                    fmt_rate(a.fired_at_200, a.strict_at_200),
                    fmt_rate(a.fired_full, a.strict_full),
                    a.exprs_with_firing_full,
                ));
            }
        }
        md.push('\n');
    }
}

fn fmt_d(d: &Dist) -> String {
    if d.n == 0 {
        return "n=0".to_string();
    }
    format!(
        "{:.3} / {:.3} / {:.3} (p90 {:.3}{})",
        d.q1,
        d.median,
        d.q3,
        d.p90,
        if d.inf_count > 0 {
            format!(", {} inf", d.inf_count)
        } else {
            String::new()
        }
    )
}

fn fmt_pct(d: &Dist) -> String {
    if d.n == 0 {
        return "n=0".to_string();
    }
    format!(
        "{:.2}% / {:.2}% / {:.2}% (p90 {:.1}%{})",
        d.q1,
        d.median,
        d.q3,
        d.p90,
        if d.inf_count > 0 {
            format!(", {} inf", d.inf_count)
        } else {
            String::new()
        }
    )
}

/// Round-1b domain-shift statistic (registration §1) for one classical-sized
/// set `S` (the whole-DEV classical band, or one op-composition stratum of
/// it).
fn write_round1b(md: &mut String, set_name: &str, r: &Round1bStat) {
    md.push_str(&format!(
        "**Round-1b domain-shift statistic (S = {set_name}):** M_B = {:.2}. ",
        r.margin_m
    ));
    for arm in ["control", "linear"] {
        md.push_str(&format!(
            "D_{arm} = {:.4} (m_{arm}^S = {:.4}, m_{arm}^DEV = {:.4}). ",
            r.d_arm[arm],
            r.d_arm[arm] + r.dev_reference_median[arm],
            r.dev_reference_median[arm],
        ));
    }
    md.push_str(&format!(
        "D_control − D_linear = {:.4} → **{}**. §1.2 polynomial prediction: both arms D ≤ 0: {}; both arms D ≤ M_B: {}.\n\n",
        r.d_diff_control_minus_linear,
        r.verdict,
        if r.both_arms_d_le_zero { "holds" } else { "FAILS" },
        if r.both_arms_d_le_margin { "holds" } else { "FAILS" },
    ));
}

/// Round-1b op-composition stratification section (registration §2).
fn write_strata(md: &mut String, s: &StrataReport) {
    md.push_str("## DEV op-composition stratification (round-1b registration §2)\n\n");
    let total: usize = s.population_counts.values().sum();
    md.push_str(&format!(
        "Population counts over the FULL classical DEV corpus loaded this run (n = {total}), by first-matching stratum:\n\n"
    ));
    md.push_str("| stratum | n | share |\n|---|---:|---:|\n");
    for (stratum, n) in &s.population_counts {
        md.push_str(&format!(
            "| {stratum} | {n} | {:.1}% |\n",
            if total == 0 {
                0.0
            } else {
                100.0 * *n as f64 / total as f64
            }
        ));
    }
    md.push('\n');
    if s.evaluated.is_empty() {
        md.push_str("No evaluated classical row carries a stratum label (nothing was put through the ablation ladder with `--stratify-by-ops` set).\n\n");
        return;
    }
    md.push_str("Per-stratum results, over classical rows THIS run actually put through the ablation ladder (may be a small sample — see population counts above for the true stratum sizes):\n\n");
    md.push_str("| stratum | B | n | control ratio med | linear ratio med | D_control | D_linear | D_control − D_linear | M_B | verdict |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for t in &s.evaluated {
        let Some(r1b) = &t.round1b else {
            continue;
        };
        let ctl = &t.arms[0];
        let lin = &t.arms[1];
        md.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.4} | {:.4} | {:.4} | {:.2} | {} |\n",
            t.band,
            t.b,
            t.n,
            ctl.ratio_vs_unguided_at_b.median,
            lin.ratio_vs_unguided_at_b.median,
            r1b.d_arm["control"],
            r1b.d_arm["linear"],
            r1b.d_diff_control_minus_linear,
            r1b.margin_m,
            r1b.verdict,
        ));
    }
    md.push('\n');
}

fn write_markdown(report: &Report, path: &Path, jsonl: &str) {
    let mut md = String::new();
    md.push_str(
        "# Phase 3 at-budget evaluation on DEV (ablation ladder against the registered claim)\n\n",
    );
    md.push_str("**Date:** 2026-09-01 · **Registration:** `docs/plans/2026-09-01-phase3-registration.md` (unrevised) · **Authority:** `docs/plans/2026-08-31-guide-design-revision.md` §5\n\n");
    md.push_str(&format!(
        "Per-expression rows: `{jsonl}` ({} expressions: {}). Skipped: {}.\n\n",
        report.n_rows,
        report
            .n_by_band
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(", "),
        if report.skipped_names.is_empty() {
            "none".to_string()
        } else {
            report.skipped_names.join(", ")
        }
    ));
    md.push_str("Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. \
All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). \
Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. \
Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).\n\n");

    for t in &report.tiers {
        let binding = t.band == "classical";
        md.push_str(&format!(
            "## {} band, B = {} ({})\n\n",
            t.band,
            t.b,
            if binding {
                format!(
                    "REGISTERED: Y = {:.1}% → median ratio ≤ {:.3}; 4B-approach: median gap ≤ {:.1}%",
                    t.y_registered * 100.0,
                    t.ratio_threshold,
                    t.gap_threshold_pct
                )
            } else {
                "reported for completeness, NO claim registered".to_string()
            }
        ));
        md.push_str(&format!("n = {}.\n\n", t.n));
        md.push_str("| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |\n|---|---|---|---|---|---|---|---|---|\n");
        md.push_str(&format!(
            "| (a) unguided @B | 1.000 (by definition) | — | {} | {} | 0 | {:.3} | {:.4} | — |\n",
            fmt_pct(&t.unguided_truncation_loss_pct),
            fmt_pct(&t.unguided_regret_at_b_pct),
            t.unguided_structural_share_pooled,
            t.unguided_strict_precision_pooled,
        ));
        md.push_str(&format!(
            "| (b) unguided @4B | — | — | 0 | {} | 1 | — | — | — |\n",
            fmt_pct(&t.unguided_regret_at_4b_pct)
        ));
        let claim_label = format!("(d) {} @B [claim]", report.context["claim_guide"]);
        for a in &t.arms {
            let label = match a.arm.as_str() {
                "control" => "(c) PerRuleRateGuide @B [control]",
                "linear" => claim_label.as_str(),
                other => other,
            };
            md.push_str(&format!(
                "| {label} | {} | {} / {} / {} | {} | {} | {} (n={}) | {:.3} | {:.4} | {:.0} |\n",
                fmt_d(&a.ratio_vs_unguided_at_b),
                a.improved,
                a.unchanged,
                a.worse,
                fmt_pct(&a.gap_vs_unguided_at_4b_pct),
                fmt_pct(&a.regret_pct),
                fmt_d(&a.gap_closed_frac),
                a.gap_closed_frac.n,
                a.structural_share_pooled,
                a.strict_precision_pooled,
                a.rounds_to_b.median,
            ));
        }
        md.push('\n');
        md.push_str(
            "Equal-work and common-grid restatements (the headline row above reads both \
             arms at the same grid TARGET, and its regret reference is the best either arm \
             reached anywhere on its own grid — neither of which is symmetric between a \
             sweep-granular unguided stepper and a per-candidate guided one):\n\n",
        );
        for a in &t.arms {
            md.push_str(&format!(
                "- {}: unguided actual applications at this target {} (this arm lands on B); \
                 ratio at MATCHED actual applications {}; regret on the grid prefix both \
                 arms ran {}; reaches that common-grid best on {} / {}.\n",
                a.arm,
                fmt_d(&a.unguided_app_actual_at_b),
                fmt_d(&a.ratio_vs_unguided_at_matched_apps),
                fmt_pct(&a.regret_on_common_grid_pct),
                a.reaches_common_grid_best,
                t.n,
            ));
        }
        md.push('\n');
        md.push_str(&format!(
            "Ladder diagnostics: linear < control on {} expressions, equal on {}, linear > control on {}. ",
            t.linear_lt_control, t.linear_eq_control, t.linear_gt_control
        ));
        for a in &t.arms {
            md.push_str(&format!(
                "{}: quiesced before B on {} / {} (ended at {} applications; burned-key share {}); reaches the empirical best on {}; vs production cost better / equal / worse = {} / {} / {}. ",
                a.arm,
                a.ended_before_b,
                t.n,
                fmt_d(&a.ended_at_apps),
                fmt_d(&a.burned_key_share),
                a.reaches_empirical_best,
                a.vs_production_better,
                a.vs_production_equal,
                a.vs_production_worse,
            ));
        }
        md.push_str("\n\n");
        if binding {
            let lin = &t.arms[1];
            let ctl = &t.arms[0];
            md.push_str(&format!(
                "**Verdict (B={}):** (d) {} median ratio vs unguided-at-B = **{:.3}** against the registered threshold ≤ {:.3} (Y = {:.1}%) — the Y-clause **{}**; median gap vs unguided-at-4B = {:.2}% against ≤ {:.1}% — the 4B-approach clause **{}**; (d) {} unguided-at-B at all (median ratio {} 1.0, kill-gate view). Ladder: (c) control median ratio {:.3}, regret {:.2}% vs (d) regret {:.2}% — (d) **{}** (c).\n\n",
                t.b,
                report.context["claim_guide"],
                lin.ratio_vs_unguided_at_b.median,
                t.ratio_threshold,
                t.y_registered * 100.0,
                if t.y_clause_holds["linear"] { "HOLDS" } else { "FAILS" },
                lin.gap_vs_unguided_at_4b_pct.median,
                t.gap_threshold_pct,
                if t.approach_clause_holds["linear"] { "HOLDS" } else { "FAILS" },
                if t.beats_unguided_at_all["linear"] { "beats" } else { "does not beat" },
                if t.beats_unguided_at_all["linear"] { "<" } else { "≥" },
                ctl.ratio_vs_unguided_at_b.median,
                ctl.regret_pct.median,
                lin.regret_pct.median,
                if t.linear_beats_control { "beats" } else { "does not beat" },
            ));
        }
        if let Some(r1b) = &t.round1b {
            write_round1b(&mut md, &t.band, r1b);
        }
    }

    if let Some(strata) = &report.strata {
        write_strata(&mut md, strata);
    }

    write_rule_firing(&mut md, &report.rule_firing);

    md.push_str("## Enabler-starvation diagnostics\n\n");
    for e in &report.enabler {
        md.push_str(&format!("### {} (n = {})\n\n", e.band, e.n));
        md.push_str(&format!(
            "Unguided strict-positive applications: {} total, {} numeric (non-structural). Of the numeric ones, **{}** ({:.1}%) have a structural application in their tight derivation ancestry (\"structurally enabled\"), {} have a direct child whose chosen node a structural rule created.\n\n",
            e.unguided_strict_positive_total,
            e.numeric_total,
            e.numeric_structurally_enabled,
            if e.numeric_total == 0 { 0.0 } else { 100.0 * e.numeric_structurally_enabled as f64 / e.numeric_total as f64 },
            e.numeric_direct_child_structural,
        ));
        md.push_str("| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |\n|---|---|---|\n");
        for (arm, n) in &e.numeric_present_in {
            let en = e.enabled_present_in.get(arm).copied().unwrap_or(0);
            md.push_str(&format!(
                "| {arm} | {n} / {} ({:.1}%) | {en} / {} ({:.1}%) |\n",
                e.numeric_total,
                if e.numeric_total == 0 {
                    0.0
                } else {
                    100.0 * *n as f64 / e.numeric_total as f64
                },
                e.numeric_structurally_enabled,
                if e.numeric_structurally_enabled == 0 {
                    0.0
                } else {
                    100.0 * en as f64 / e.numeric_structurally_enabled as f64
                },
            ));
        }
        md.push_str("\n| arm | structural share of applications @100 | @200 | top rules @100 |\n|---|---|---|---|\n");
        for (arm, m) in &e.structural_share_at_b {
            md.push_str(&format!(
                "| {arm} | {:.3} | {:.3} | {} |\n",
                m.get(&100).copied().unwrap_or(f64::NAN),
                m.get(&200).copied().unwrap_or(f64::NAN),
                e.top_rules_at_100
                    .get(arm)
                    .map(|v| v
                        .iter()
                        .map(|(r, n)| format!("{r} {n}"))
                        .collect::<Vec<_>>()
                        .join(", "))
                    .unwrap_or_default()
            ));
        }
        md.push('\n');
        for (arm, d) in &e.seen_keys_per_application {
            md.push_str(&format!(
                "- {arm}: distinct candidate keys scored per recorded application over the run (dedup coverage): {}\n",
                fmt_d(d)
            ));
        }
        md.push('\n');
    }

    md.push_str(
        "## Production-units context (exact production saturation call per expression)\n\n",
    );
    md.push_str("`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.\n\n");
    for p in &report.production {
        md.push_str(&format!(
            "### {} (n = {}, probe returned None for {})\n\n",
            p.band, p.n, p.n_probe_none
        ));
        md.push_str(&format!(
            "- stop reasons: {}\n- effective B (applications at stop): {} (max {:.0})\n- share with applications ≥ 100 / 200 / 400 / 800 / 1600: {}\n- rounds run: {}; classes at stop: {}\n- production cost / unguided cost@100: {}\n- production cost / unguided cost@200: {}\n- production cost / unguided cost@800: {}\n- production regret vs empirical best: {}\n- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): {}\n- unguided checkpoint whose app_actual first reaches production's application count: {}\n\n",
            p.stops.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "),
            fmt_d(&p.effective_b),
            p.effective_b.max,
            p.share_apps_ge.iter().map(|(k, v)| format!("≥{k}: {:.1}%", v * 100.0)).collect::<Vec<_>>().join(", "),
            fmt_d(&p.iterations),
            fmt_d(&p.classes_after),
            fmt_d(&p.ratio_vs_unguided_at_100),
            fmt_d(&p.ratio_vs_unguided_at_200),
            fmt_d(&p.ratio_vs_unguided_at_4b_800),
            fmt_pct(&p.regret_vs_best_pct),
            p.equivalent_checkpoint_hist.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", "),
            p.app_bracket_hist.iter().map(|(k, v)| format!("{}: {v}", if *k == usize::MAX { "beyond grid".to_string() } else { k.to_string() })).collect::<Vec<_>>().join(", "),
        ));
    }

    md.push_str("## Context (not metrics)\n\n");
    for (k, v) in &report.context {
        md.push_str(&format!("- {k}: {v}\n"));
    }

    std::fs::write(path, md).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

fn read_rows(path: &Path) -> Vec<ExprRow> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("{}: I/O error: {e}", path.display()));
        if line.trim().is_empty() {
            continue;
        }
        let row: ExprRow = serde_json::from_str(&line).unwrap_or_else(|e| {
            panic!(
                "{}:{}: malformed row (delete the line to re-run that expression): {e}",
                path.display(),
                i + 1
            )
        });
        rows.push(row);
    }
    rows
}

fn uptime() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("uptime unavailable: {e}"))
}

/// FNV-1a 64 hex over the inputs that decide what a row *means*: the source
/// revision, the two guide artifacts by content (not by path — a path is
/// stable across a retrain), the corpus by content, the guided grid, and the
/// arm/rule sets. Two runs that agree on all of these produce comparable
/// rows; two that do not must not be aggregated together, and a name-only
/// resume check cannot tell them apart.
fn run_config_identity(
    checkpoint: &str,
    train_guide_report: &str,
    corpus: &Path,
    guided_grid: &[usize],
    stratify_by_ops: bool,
) -> String {
    let content = |label: &str, path: &str| match std::fs::read(path) {
        Ok(bytes) => format!("{label}={}\n", fnv1a64_hex(&bytes)),
        Err(e) => {
            panic!("phase3_at_budget_eval: cannot read {label} {path} to identify this run: {e}")
        }
    };
    let mut buf = String::new();
    buf.push_str(&format!("source_rev={}\n", git_rev()));
    buf.push_str(&content("checkpoint", checkpoint));
    buf.push_str(&content("train_guide_report", train_guide_report));
    buf.push_str(&content("corpus", &corpus.display().to_string()));
    buf.push_str(&format!("guided_grid={guided_grid:?}\n"));
    buf.push_str(&format!("unguided_grid={APP_CHECKPOINT_GRID:?}\n"));
    buf.push_str(&format!("arms={ARM_NAMES:?}\n"));
    buf.push_str(&format!("structural_rules={STRUCTURAL_RULES:?}\n"));
    buf.push_str(&format!("stratify_by_ops={stratify_by_ops}\n"));
    fnv1a64_hex(buf.as_bytes())
}

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("git unavailable: {e}"))
}

/// Size-stratified stride sample of `n` items (`items` sorted by size);
/// `None` = every item.
fn stride_sample<T>(mut items: Vec<T>, n: Option<usize>) -> Vec<T> {
    let Some(n) = n else {
        return items;
    };
    if n >= items.len() {
        return items;
    }
    let stride = items.len() as f64 / n as f64;
    let mut keep: Vec<usize> = (0..n).map(|i| ((i as f64) * stride) as usize).collect();
    keep.dedup();
    let keep: HashSet<usize> = keep.into_iter().collect();
    let mut out = Vec::with_capacity(keep.len());
    for (i, item) in items.drain(..).enumerate() {
        if keep.contains(&i) {
            out.push(item);
        }
    }
    out
}

/// Op-composition stratum counts (registration §2) over a set of entries.
fn strata_counts(entries: &[(String, ExprArena, ExprId)]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_, arena, _) in entries {
        *counts.entry(ops_stratum(arena).to_string()).or_default() += 1;
    }
    counts
}

/// Every `FenceKey` in `corpus_dir/corpus_train.bin` — the TRAIN side of the
/// round-1b OOD fence check (registration §3: "every OOD FenceKey probed
/// against corpus_train.bin, collision = hard error").
///
/// # Panics
///
/// Panics if `corpus_train.bin` is missing — a fence that silently checked
/// against nothing would validate nothing while looking like it validated
/// everything (same discipline as `split::Fence::build`).
fn train_fence_keys(corpus_dir: &Path) -> HashSet<FenceKey> {
    let path = corpus_dir.join("corpus_train.bin");
    assert!(
        path.exists(),
        "TRAIN corpus not found at {} — the fence check cannot run without it (regenerate the \
         tiered corpus with gen_bench_corpus, --output {})",
        path.display(),
        corpus_dir.display()
    );
    let entries =
        read_corpus(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    entries
        .iter()
        .map(|(_, arena, root)| FenceKey::of(arena, *root))
        .collect()
}

/// Hard-fails (registration §3) if any entry in `entries` (loaded from
/// `corpus_path`) shares a feature-quotient structure with TRAIN — enforced
/// regardless of which corpus was loaded (`corpus_dev.bin` or a
/// `--corpus`-supplied OOD file), so a caller cannot silently skip the check
/// by pointing `--corpus` at the default.
fn enforce_train_fence(
    corpus_path: &Path,
    corpus_dir: &Path,
    entries: &[(String, ExprArena, ExprId)],
) {
    let train_keys = train_fence_keys(corpus_dir);
    let collisions: Vec<&str> = entries
        .iter()
        .filter(|(_, arena, root)| train_keys.contains(&FenceKey::of(arena, *root)))
        .map(|(name, _, _)| name.as_str())
        .collect();
    assert!(
        collisions.is_empty(),
        "TRAIN-fence violation: {} of {} entries in {} share a feature-quotient structure with \
         {}/corpus_train.bin (a leak, not a hygiene event — round-1b registration §3): {:?}{}",
        collisions.len(),
        entries.len(),
        corpus_path.display(),
        corpus_dir.display(),
        &collisions[..collisions.len().min(10)],
        if collisions.len() > 10 {
            ", ... (truncated)"
        } else {
            ""
        },
    );
    eprintln!(
        "phase3_at_budget_eval: TRAIN fence check OK — {} entries in {} probed against {} TRAIN structures, 0 collisions",
        entries.len(),
        corpus_path.display(),
        train_keys.len(),
    );
}

fn main() {
    let args = Args::parse();
    // One rule vocabulary for the whole run: every guide is deployed
    // against it, every per-rule table is keyed by an identity drawn from
    // it, and `trig_rule_ids` checks the registered indices still name the
    // rules the registration says they do.
    let rules = RuleSet::production();
    let jsonl_path = PathBuf::from(&args.out_jsonl);
    let skipped: Vec<String> = args
        .skip_names
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let load_start = uptime();
    // Populated only when `--stratify-by-ops` runs the corpus-population
    // scan below (skipped in `--aggregate-only`, which reads no corpus).
    let mut strata_population_out: Option<BTreeMap<String, usize>> = None;
    // `--stratify-by-ops`: name -> stratum over the loaded corpus, so rows
    // written without a stratum label (Round 1's) can be stratified in
    // aggregate from the corpus alone — registration §2: "no re-evaluation
    // is needed for the stratified numbers, only the per-expression op
    // counts, read from corpus_dev.bin".
    let mut stratum_by_name: Option<BTreeMap<String, String>> = None;
    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let dev_path: PathBuf = args
        .corpus
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| corpus_dir.join("corpus_dev.bin"));

    if args.aggregate_only && args.stratify_by_ops {
        let entries = read_corpus(&dev_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", dev_path.display()));
        enforce_train_fence(&dev_path, &corpus_dir, &entries);
        let classical: Vec<(String, ExprArena, ExprId)> = entries
            .into_iter()
            .filter(|(name, arena, _)| {
                name.starts_with(&args.name_prefix)
                    && tier_name(arena.nodes_raw().len()) == "classical"
            })
            .collect();
        strata_population_out = Some(strata_counts(&classical));
        stratum_by_name = Some(
            classical
                .iter()
                .map(|(name, arena, _)| (name.clone(), ops_stratum(arena).to_string()))
                .collect(),
        );
    }

    if !args.aggregate_only {
        let costs = CostModel::latency_prior();
        let guided_grid: Vec<usize> = if args.guided_grid.trim().is_empty() {
            GUIDED_GRID.to_vec()
        } else {
            args.guided_grid
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse::<usize>()
                        .unwrap_or_else(|e| panic!("--guided-grid: bad entry {t:?}: {e}"))
                })
                .collect()
        };
        for (b, _, _) in REGISTERED_TIERS {
            assert!(
                guided_grid.contains(&b) && guided_grid.contains(&(4 * b)),
                "--guided-grid {guided_grid:?} must contain registered B={b} and 4B={}",
                4 * b
            );
        }
        let mut entries = read_corpus(&dev_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", dev_path.display()));
        enforce_train_fence(&dev_path, &corpus_dir, &entries);
        if !args.name_prefix.is_empty() {
            let before = entries.len();
            entries.retain(|(name, _, _)| name.starts_with(&args.name_prefix));
            assert!(
                !entries.is_empty(),
                "--name-prefix {:?} matches none of the {before} entries in {}",
                args.name_prefix,
                dev_path.display()
            );
            eprintln!(
                "phase3_at_budget_eval: --name-prefix {:?} kept {} of {before} entries",
                args.name_prefix,
                entries.len()
            );
        }
        entries.sort_by(|a, b| {
            a.1.nodes_raw()
                .len()
                .cmp(&b.1.nodes_raw().len())
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut by_band: BTreeMap<&str, Vec<(String, ExprArena, ExprId)>> = BTreeMap::new();
        for (name, arena, root) in entries {
            by_band
                .entry(tier_name(arena.nodes_raw().len()))
                .or_default()
                .push((name, arena, root));
        }
        let counts: BTreeMap<&str, usize> = by_band.iter().map(|(k, v)| (*k, v.len())).collect();
        eprintln!("phase3_at_budget_eval: DEV population by band: {counts:?}");

        if args.stratify_by_ops {
            let classical_strata = by_band
                .get("classical")
                .map(|v| strata_counts(v))
                .unwrap_or_default();
            eprintln!(
                "phase3_at_budget_eval: classical population by op-composition stratum (n={}): {classical_strata:?}",
                by_band.get("classical").map_or(0, Vec::len),
            );
            strata_population_out = Some(classical_strata);
            stratum_by_name = by_band.get("classical").map(|v| {
                v.iter()
                    .map(|(name, arena, _)| (name.clone(), ops_stratum(arena).to_string()))
                    .collect()
            });
        }

        let in_node_band = |n: usize| {
            (args.min_expr_nodes == 0 || n >= args.min_expr_nodes)
                && (args.max_expr_nodes == 0 || n <= args.max_expr_nodes)
        };
        let mut selected: Vec<(String, ExprArena, ExprId)> = Vec::new();
        selected.extend(stride_sample(
            by_band
                .remove("classical")
                .unwrap_or_default()
                .into_iter()
                .filter(|(_, a, _)| in_node_band(a.nodes_raw().len()))
                .collect(),
            (args.classical_samples > 0).then_some(args.classical_samples),
        ));
        for band in ["rapid", "blitz"] {
            selected.extend(stride_sample(
                by_band
                    .remove(band)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|(_, a, _)| in_node_band(a.nodes_raw().len()))
                    .collect(),
                Some(args.other_samples),
            ));
        }
        if args.min_expr_nodes > 0 || args.max_expr_nodes > 0 {
            eprintln!(
                "phase3_at_budget_eval: node-count band filter [{}, {}] leaves {} expressions",
                args.min_expr_nodes,
                if args.max_expr_nodes == 0 {
                    "inf".to_string()
                } else {
                    args.max_expr_nodes.to_string()
                },
                selected.len()
            );
        }

        let run_config = run_config_identity(
            &args.checkpoint,
            &args.train_guide_report,
            &dev_path,
            &guided_grid,
            args.stratify_by_ops,
        );
        // Resumption reuses a row only if it was produced by THIS
        // configuration. A name is not proof of reusability: the same
        // expression evaluated against a different checkpoint, guided grid,
        // corpus, or source revision is a different measurement, and mixing
        // the two would report a heterogeneous aggregate under a context
        // block that names only the current run.
        let existing: HashSet<String> = read_rows(&jsonl_path)
            .into_iter()
            .map(|r| {
                assert_eq!(
                    r.run_config.as_deref(),
                    Some(run_config.as_str()),
                    "{}: row {:?} was produced by run configuration {:?}, but this run is \
                     {run_config:?} — appending would mix incomparable measurements under one \
                     report. Write to a new --out-jsonl path, or delete the existing file to \
                     re-evaluate every expression under the current configuration.",
                    jsonl_path.display(),
                    r.name,
                    r.run_config
                        .as_deref()
                        .unwrap_or("<absent: pre-identity row>"),
                );
                r.name
            })
            .collect();

        // Resuming skips an expression by NAME. Every skipped row was
        // produced under whatever configuration the earlier run used, so
        // resuming into a file written under a different guided grid,
        // checkpoint, control report or sampling setting silently rebuilds
        // the aggregate from incompatible rows — and the documented
        // guided-grid sensitivity check would have no effect at all once the
        // checked-in run is complete. The configuration is fingerprinted
        // beside the JSONL and compared before any row is reused.
        let fingerprint = config_fingerprint(&args, &guided_grid, &dev_path);
        let fp_path = jsonl_path.with_extension("config.json");
        let existing_rows = read_rows(&jsonl_path);
        if existing_rows.is_empty() {
            std::fs::write(&fp_path, &fingerprint)
                .unwrap_or_else(|e| panic!("cannot write {}: {e}", fp_path.display()));
        } else {
            let recorded = std::fs::read_to_string(&fp_path).unwrap_or_else(|e| {
                panic!(
                    "{} holds {} completed rows but its configuration fingerprint {} is \
                     unreadable ({e}) — those rows were written under an unknown \
                     configuration and cannot be safely resumed into. Delete the JSONL to \
                     start fresh, or re-run with --aggregate-only to read it as-is.",
                    jsonl_path.display(),
                    existing_rows.len(),
                    fp_path.display()
                )
            });
            assert_eq!(
                recorded.trim(),
                fingerprint.trim(),
                "{} was written under a different configuration than this run \
                 (fingerprint {}) — resuming would mix incompatible rows into one \
                 aggregate. Write to a fresh --out-jsonl, or delete the existing file.",
                jsonl_path.display(),
                fp_path.display()
            );
        }
        eprintln!(
            "phase3_at_budget_eval: {} selected, {} already done, {} skipped by flag \
             (run_config {run_config})",
            selected.len(),
            existing.len(),
            skipped.len()
        );

        let guides = Guides {
            control: per_rule_rate_guide_from_report(Path::new(&args.train_guide_report))
                .unwrap_or_else(|e| panic!("control guide: {e}")),
            linear: match &args.r2g_checkpoint {
                Some(path) => ClaimGuide::Return(
                    load_return_guide(Path::new(path), &rules)
                        .unwrap_or_else(|e| panic!("r2g claim guide: {e}")),
                ),
                None => ClaimGuide::StrictBit(
                    load_linear_guide(Path::new(&args.checkpoint), &rules)
                        .unwrap_or_else(|e| panic!("linear guide: {e}")),
                ),
            },
        };

        if let Some(parent) = jsonl_path.parent() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl_path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", jsonl_path.display()));

        let total = selected.len();
        for (i, (name, arena, root)) in selected.iter().enumerate() {
            if existing.contains(name) || skipped.contains(name) {
                continue;
            }
            eprintln!(
                "phase3_at_budget_eval: [{}/{}] {} ({} nodes, {})",
                i + 1,
                total,
                name,
                arena.nodes_raw().len(),
                tier_name(arena.nodes_raw().len())
            );
            let input = CurveInput {
                arena,
                root: *root,
                class_cap: config_for_node_count(arena.nodes_raw().len()).max_classes,
                costs: &costs,
                guided_grid: &guided_grid,
            };
            let mut row = evaluate_expression(name, &input, &guides, &rules, args.stratify_by_ops);
            row.run_config = Some(run_config.clone());
            let line = serde_json::to_string(&row).expect("serialize row");
            writeln!(out, "{line}")
                .unwrap_or_else(|e| panic!("write {}: {e}", jsonl_path.display()));
            out.flush().expect("flush");
        }
    }

    // ------------------------------------------------------------------
    // Aggregate.
    // ------------------------------------------------------------------
    // `--skip-names` documents an expression as excluded from the run; a row
    // for it left in the resumable JSONL (from an earlier run, or read back
    // under `--aggregate-only`) would still enter every count, median and
    // registered verdict. Filter before aggregating, not only before
    // evaluating.
    let all_rows = read_rows(&jsonl_path);
    let mut rows: Vec<ExprRow> = all_rows
        .into_iter()
        .filter(|r| !skipped.contains(&r.name))
        .collect();
    assert!(
        !rows.is_empty(),
        "no rows in {} — nothing to aggregate",
        jsonl_path.display()
    );
    if !skipped.is_empty() {
        eprintln!(
            "phase3_at_budget_eval: aggregating {} rows; {} name(s) excluded by \
             --skip-names: {skipped:?}",
            rows.len(),
            skipped.len()
        );
    }
    if args.min_expr_nodes > 0 || args.max_expr_nodes > 0 {
        let before = rows.len();
        rows.retain(|r| {
            (args.min_expr_nodes == 0 || r.node_count >= args.min_expr_nodes)
                && (args.max_expr_nodes == 0 || r.node_count <= args.max_expr_nodes)
        });
        eprintln!(
            "phase3_at_budget_eval: aggregating {} of {before} rows inside node band [{}, {}]",
            rows.len(),
            args.min_expr_nodes,
            if args.max_expr_nodes == 0 {
                "inf".to_string()
            } else {
                args.max_expr_nodes.to_string()
            }
        );
        assert!(
            !rows.is_empty(),
            "no rows in {} fall inside the requested node band [{}, {}] — nothing to aggregate",
            jsonl_path.display(),
            args.min_expr_nodes,
            args.max_expr_nodes
        );
    }
    if let Some(map) = &stratum_by_name {
        for r in rows.iter_mut().filter(|r| r.tier == "classical") {
            let from_corpus = map.get(&r.name).unwrap_or_else(|| {
                panic!(
                    "{}: classical row {} is not in the loaded corpus {} — cannot stratify it",
                    jsonl_path.display(),
                    r.name,
                    dev_path.display()
                )
            });
            match &r.stratum {
                Some(s) => assert_eq!(
                    s,
                    from_corpus,
                    "{}: row {} carries stratum {s} but the corpus says {from_corpus}",
                    jsonl_path.display(),
                    r.name
                ),
                None => r.stratum = Some(from_corpus.clone()),
            }
        }
    }
    let mut n_by_band: BTreeMap<String, usize> = BTreeMap::new();
    for r in &rows {
        *n_by_band.entry(r.tier.clone()).or_default() += 1;
    }
    let mut tiers = Vec::new();
    let mut production = Vec::new();
    let mut enabler = Vec::new();
    for band in ["classical", "rapid", "blitz"] {
        let sub: Vec<&ExprRow> = rows.iter().filter(|r| r.tier == band).collect();
        if sub.is_empty() {
            continue;
        }
        for (b, y, l) in REGISTERED_TIERS {
            tiers.push(tier_result(
                &sub,
                band,
                TierBudget {
                    b,
                    y,
                    l,
                    compute_round1b: band == "classical",
                },
            ));
        }
        production.push(production_summary(&sub, band));
        enabler.push(enabler_summary(&sub, band));
    }

    // Round-1b op-composition stratification (registration §2), only when
    // any row carries a stratum label.
    let strata = rows.iter().any(|r| r.stratum.is_some()).then(|| {
        let mut by_stratum: BTreeMap<String, Vec<&ExprRow>> = BTreeMap::new();
        for r in rows.iter().filter(|r| r.tier == "classical") {
            if let Some(s) = &r.stratum {
                by_stratum.entry(s.clone()).or_default().push(r);
            }
        }
        let evaluated: Vec<TierResult> = by_stratum
            .iter()
            .flat_map(|(stratum, sub)| {
                REGISTERED_TIERS.iter().map(move |&(b, y, l)| {
                    tier_result(
                        sub,
                        stratum,
                        TierBudget {
                            b,
                            y,
                            l,
                            compute_round1b: true,
                        },
                    )
                })
            })
            .collect();
        StrataReport {
            population_counts: strata_population_out.clone().unwrap_or_default(),
            evaluated,
        }
    });

    let mut rule_firing = Vec::new();
    {
        let classical: Vec<&ExprRow> = rows.iter().filter(|r| r.tier == "classical").collect();
        if !classical.is_empty() {
            rule_firing.push(rule_firing_summary(&classical, "classical", &rules));
        }
        let mut by_stratum: BTreeMap<String, Vec<&ExprRow>> = BTreeMap::new();
        for r in &classical {
            if let Some(s) = &r.stratum {
                by_stratum.entry(s.clone()).or_default().push(r);
            }
        }
        for (stratum, sub) in &by_stratum {
            rule_firing.push(rule_firing_summary(sub, stratum, &rules));
        }
    }

    let mut context = BTreeMap::new();
    context.insert("source_rev".to_string(), git_rev());
    context.insert(
        "run_config".to_string(),
        rows.first()
            .and_then(|r| r.run_config.clone())
            .unwrap_or_else(|| "<absent: rows predate run-config identity>".to_string()),
    );
    context.insert("corpus".to_string(), dev_path.display().to_string());
    context.insert("load_at_start".to_string(), load_start);
    context.insert("load_at_end".to_string(), uptime());
    let (claim_checkpoint, claim_guide) = match &args.r2g_checkpoint {
        Some(path) => (
            path.clone(),
            load_return_guide(Path::new(path), &RuleSet::production())
                .map(|g| ClaimGuide::Return(g).label())
                .unwrap_or_else(|e| panic!("r2g claim guide: {e}")),
        ),
        None => (args.checkpoint.clone(), "LinearCandidateGuide".to_string()),
    };
    context.insert("checkpoint".to_string(), claim_checkpoint.clone());
    context.insert("claim_guide".to_string(), claim_guide.clone());
    context.insert(
        "train_guide_report".to_string(),
        args.train_guide_report.clone(),
    );
    context.insert(
        "guided_grid".to_string(),
        format!(
            "{} (unguided: {APP_CHECKPOINT_GRID:?})",
            if args.guided_grid.trim().is_empty() {
                format!("{GUIDED_GRID:?}")
            } else {
                args.guided_grid.clone()
            }
        ),
    );
    context.insert(
        "structural_rules".to_string(),
        format!("{STRUCTURAL_RULES:?}"),
    );
    context.insert("arms".to_string(), format!("{ARM_NAMES:?}"));

    let report = Report {
        n_rows: rows.len(),
        n_by_band,
        skipped_names: skipped,
        tiers,
        production,
        enabler,
        strata,
        rule_firing,
        context,
    };
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&args.out_json, json)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", args.out_json));
    write_markdown(&report, Path::new(&args.out_md), &args.out_jsonl);

    // Console summary: the registered-claim sentences.
    for t in &report.tiers {
        if t.band != "classical" {
            continue;
        }
        let lin = &t.arms[1];
        let ctl = &t.arms[0];
        println!(
            "classical B={}: n={} | linear median ratio {:.3} (threshold {:.3}) Y-clause {} | gap vs 4B {:.2}% (threshold {:.1}%) {} | control median ratio {:.3} | regret unguided {:.2}% control {:.2}% linear {:.2}% unguided@4B {:.2}%",
            t.b,
            t.n,
            lin.ratio_vs_unguided_at_b.median,
            t.ratio_threshold,
            if t.y_clause_holds["linear"] {
                "HOLDS"
            } else {
                "FAILS"
            },
            lin.gap_vs_unguided_at_4b_pct.median,
            t.gap_threshold_pct,
            if t.approach_clause_holds["linear"] {
                "HOLDS"
            } else {
                "FAILS"
            },
            ctl.ratio_vs_unguided_at_b.median,
            t.unguided_regret_at_b_pct.median,
            ctl.regret_pct.median,
            lin.regret_pct.median,
            t.unguided_regret_at_4b_pct.median,
        );
    }
    let round1b_sets = report
        .tiers
        .iter()
        .chain(report.strata.iter().flat_map(|s| s.evaluated.iter()));
    for t in round1b_sets {
        let Some(r) = &t.round1b else {
            continue;
        };
        println!(
            "round1b S={} B={}: n={} | D_control {:+.4} D_linear {:+.4} | D_control-D_linear {:+.4} vs M={:.2} -> {} | poly prediction (D<=0 both / D<=M both): {} / {}",
            t.band,
            t.b,
            t.n,
            r.d_arm["control"],
            r.d_arm["linear"],
            r.d_diff_control_minus_linear,
            r.margin_m,
            r.verdict,
            r.both_arms_d_le_zero,
            r.both_arms_d_le_margin,
        );
    }
    for p in &report.production {
        println!(
            "production {}: n={} stops {:?} effective-B median {:.0} (q1 {:.0}, q3 {:.0}) share>=100 {:.1}% >=200 {:.1}%",
            p.band,
            p.n,
            p.stops,
            p.effective_b.median,
            p.effective_b.q1,
            p.effective_b.q3,
            p.share_apps_ge[&100] * 100.0,
            p.share_apps_ge[&200] * 100.0
        );
    }

    // Journal (deterministic-metric record; no timing).
    let journal_record = serde_json::json!({
        "record": "phase3_at_budget_eval",
        "ts_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_secs(),
        "config": {
            "source_rev": report.context["source_rev"],
            "corpus": format!("{} (FINAL untouched)", dev_path.display()),
            "protocol": format!("arms=unguided,control,linear;guided_grid={GUIDED_GRID:?};cost=latency_prior;class_cap=production_tier;registered_tiers={REGISTERED_TIERS:?}"),
            "checkpoint": claim_checkpoint,
            "claim_guide": claim_guide,
        },
        "n_by_band": report.n_by_band,
        "classical": report.tiers.iter().filter(|t| t.band == "classical").map(|t| serde_json::json!({
            "B": t.b,
            "n": t.n,
            "linear_median_ratio": t.arms[1].ratio_vs_unguided_at_b.median,
            "control_median_ratio": t.arms[0].ratio_vs_unguided_at_b.median,
            "ratio_threshold": t.ratio_threshold,
            "y_clause_linear": t.y_clause_holds["linear"],
            "approach_clause_linear": t.approach_clause_holds["linear"],
            "linear_gap_vs_4b_median_pct": t.arms[1].gap_vs_unguided_at_4b_pct.median,
            "regret_median_pct": {"unguided_at_b": t.unguided_regret_at_b_pct.median, "control": t.arms[0].regret_pct.median, "linear": t.arms[1].regret_pct.median, "unguided_at_4b": t.unguided_regret_at_4b_pct.median},
        })).collect::<Vec<_>>(),
        "production_classical": report.production.iter().find(|p| p.band == "classical").map(|p| serde_json::json!({
            "stops": p.stops, "effective_b_median": p.effective_b.median, "effective_b_q1": p.effective_b.q1, "effective_b_q3": p.effective_b.q3,
        })),
        "outputs": [args.out_jsonl, args.out_json, args.out_md],
    });
    append_record(Path::new(&args.journal), &journal_record);
    eprintln!(
        "phase3_at_budget_eval: wrote {} and {}",
        args.out_json, args.out_md
    );
}
