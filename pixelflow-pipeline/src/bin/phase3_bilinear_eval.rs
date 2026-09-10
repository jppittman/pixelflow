//! The four-arm at-budget evaluation of `docs/plans/2026-09-02-bilinear-guide-registration.md`.
//!
//! Answers exactly one registered question: does giving the Guide a
//! rule-by-context **interaction** — `m(x)ᵀ W r + bᵀ r`, whose rule-pair
//! difference depends on the context — buy a *domain-conditional* advantage
//! that the additively separable `LinearCandidateGuide` provably cannot
//! represent (registration §1)?
//!
//! | arm | stepper | status |
//! |---|---|---|
//! | `unguided` | [`run_anytime_curve`] with `guide: None` — the denominator of every ratio | reference |
//! | `control` | [`PerRuleRateGuide`]: each rule's TRAIN strict-positive rate, no candidate-local information | frozen |
//! | `linear` | [`LinearCandidateGuide`] from the frozen strict-v1 checkpoint | frozen |
//! | `bilinear` | [`BilinearCandidateGuide`] — the thing under test | new |
//!
//! # What is measured (registration §3.1, verbatim)
//!
//! Every cost is `ExtractedDAG::dag_cost` (`ChoiceCost::dag`, #1117) under
//! `CostModel::latency_prior()`. **No wall clock enters any reported
//! number.** Budgets are recorded rule applications; the registered tiers are
//! `B = 100` (primary) and `B = 200` (secondary).
//!
//! `m_A^S(B)` is the median over set `S` of `dag_cost_A@B / dag_cost_unguided@B`,
//! and `D_A(S, B) = m_A^S(B) − m_A^DEV(B)`. Unlike round 1b, **`m_A^DEV` is
//! measured in this run** for every arm: the round-1b reference medians were
//! taken on tree cost and are not reproducible under this instrument
//! (registration, "Instrument (changed since both, binding here)").
//!
//! # One grid for every arm
//!
//! All four arms run [`GUIDED_GRID`]. Round 1 gave the unguided arm
//! `APP_CHECKPOINT_GRID` (to 204,800) and the guided arms 800, then had to
//! restrict the regret reference to the common prefix anyway. Running one
//! grid makes "the best cost any arm reaches at any checkpoint" a statement
//! about the arms rather than about who was given more budget, and the
//! unguided curve's prefix is unchanged by where it stops — the sweep is
//! deterministic and the checkpoints only decide where it is read.
//!
//! # Diagnostics that make either verdict mean something
//!
//! 1. **Trig-rule firings** (round-1b registration §4's eleven rules, by
//!    canonical label) per arm, per set, within the first `B` applications
//!    and over the whole run — does any arm actually fire the identities on
//!    `sh`, and does it fire them there more than on DEV?
//! 2. **Induced rule order**, the direct test of the representational
//!    claim. A [`RecordingGuide`] captures the bilinear arm's first scored
//!    batch on every expression; its first candidate is that expression's
//!    *real* context `x`. Holding `x` fixed and substituting each rule's
//!    embedding gives the rule ranking that model induces **at that
//!    context** — the object registration §1 says is constant for the
//!    additive class and free to vary for the bilinear one. The report
//!    states how many distinct orders each model produced, and where the
//!    trig identities rank on `sh` versus on DEV under each.
//!
//! Usage (one invocation per set, appending to one JSONL, then aggregate):
//! ```bash
//! phase3_bilinear_eval --set dev --corpus pixelflow-pipeline/data/corpus_dev.bin
//! phase3_bilinear_eval --set sh  --corpus pixelflow-pipeline/data/corpus_dev_ood.bin --name-prefix dev_sh_
//! phase3_bilinear_eval --aggregate-only
//! ```

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_ir::{Environment, ExprData, OpKind, Rooted, Term};
use pixelflow_pipeline::journal::append_record;
use pixelflow_pipeline::schema::fnv1a64_hex;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::guide_bilinear::load_bilinear_guide;
use pixelflow_pipeline::training::guide_linear::{
    load_linear_guide, per_rule_rate_guide_from_report,
};
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::{
    AnytimeCurveOutput, ApplicationId, Budget, CostModel, EpisodeLabels, KeepJournal, Optimizer,
    RuleId, RuleSet, SaturationStop, config_for_node_count, run_anytime_curve,
};
use pixelflow_search::nnue::guide::linear::{LinearCandidateGuide, PerRuleRateGuide};
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

// ---------------------------------------------------------------------------
// Registered constants. Frozen by
// docs/plans/2026-09-02-bilinear-guide-registration.md §3.2/§6 and by the
// round-1b registration it inherits from; restated here so the report is
// self-checking rather than trusting a flag.
// ---------------------------------------------------------------------------

/// Registered budget tiers, in rule applications: primary, secondary.
const REGISTERED_TIERS: [usize; 2] = [100, 200];

/// The checkpoint grid every arm runs.
const GUIDED_GRID: &[usize] = &[25, 50, 100, 200, 400, 800];

/// The regret reference: the best cost any arm reaches at any checkpoint of
/// this grid, i.e. its last entry.
const REGRET_REFERENCE_LIMIT: usize = 800;

const ARM_NAMES: [&str; 4] = ["unguided", "control", "linear", "bilinear"];

/// The two guided arms whose difference the registered statistic is about.
const CLAIM_ARM: &str = "bilinear";
const FROZEN_ARM: &str = "linear";

/// Per-curve wall-clock safety ceiling — PANICS inside the shared curve
/// runner if it binds. Never truncates, never enters a number.
const SAFETY_TIMEOUT: Duration = Duration::from_secs(1800);
const SWEEP_SAFETY_CEILING: usize = 10_000;

/// `M_B`, inherited frozen from the round-1b registration §1.1 and
/// explicitly **not** recalibrated by this round (registration §3.2).
fn registered_margin(b: usize) -> f64 {
    match b {
        100 => 0.06,
        200 => 0.07,
        other => panic!("no registered margin M_B for B={other} (only 100/200 are registered)"),
    }
}

/// Round-1b registration §4's eleven trig-related rules by registered index.
const TRIG_RULE_IDX: [usize; 11] = [20, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40];

/// The same eleven by canonical label, in the same order. A same-length
/// reorder of `all_rules()` repoints every index silently; [`trig_rules`]
/// asserts the two still name the same rules.
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

/// The three identities JP's question is literally about.
const HEADLINE_TRIG: [&str; 4] = [
    "sin-angle-addition",
    "cos-angle-addition",
    "half-angle-product",
    "pythagorean",
];

fn trig_rules(rules: &RuleSet) -> Vec<&'static str> {
    for (&idx, label) in TRIG_RULE_IDX.iter().zip(TRIG_RULE_LABELS) {
        let live = rules.label_of(idx).unwrap_or_else(|| {
            panic!(
                "registered trig rule idx {idx} is outside all_rules() ({} rules)",
                rules.len()
            )
        });
        assert_eq!(
            live, label,
            "registered trig rule idx {idx} is {live:?} in this build but the registration \
             names it {label:?} — all_rules() was reordered since the registration, so every \
             positional key in it now points at a different rule. Re-register rather than \
             silently re-point."
        );
    }
    TRIG_RULE_LABELS.to_vec()
}

#[derive(Parser)]
#[command(name = "phase3_bilinear_eval")]
#[command(about = "Four-arm at-budget evaluation of the bilinear Guide registration")]
struct Args {
    /// Evaluation set label: `dev`, `sh` or `bezier`. Written into every row.
    #[arg(long, default_value = "dev")]
    set: String,

    /// Corpus file to evaluate.
    #[arg(long, default_value = "pixelflow-pipeline/data/corpus_dev.bin")]
    corpus: String,

    /// Directory holding `corpus_train.bin` (the fence source — read
    /// whatever corpus is loaded).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Evaluate only entries whose name starts with this prefix.
    #[arg(long, default_value = "")]
    name_prefix: String,

    /// The additive arm's checkpoint.
    ///
    /// The registration §4 names `guide_checkpoint_strict_v1.json` "exactly
    /// as round 1 trained it". That file is **off-vocabulary on this
    /// branch** — it names 61 rules against this build's 62 — and
    /// `LinearWeights::check_vocabulary` refuses it, correctly. The refusal
    /// is probed and recorded verbatim in the report (see
    /// `--frozen-linear-checkpoint`) rather than worked around; the arm that
    /// actually runs is the same-recipe additive model trained on the same
    /// re-minted dataset as the bilinear arm, which is the stricter reading
    /// of §4's "the only licensed difference is the functional form".
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_remint.json"
    )]
    linear_checkpoint: String,

    /// The registration's frozen additive checkpoint. Not deployed — loaded
    /// only so the report can state, in the loader's own words, why it
    /// could not be.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    frozen_linear_checkpoint: String,

    /// The bilinear checkpoint under test.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json"
    )]
    bilinear_checkpoint: String,

    /// `train_guide` report supplying the control arm's per-rule TRAIN rates.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-train-guide-report.json"
    )]
    train_guide_report: String,

    /// Per-expression rows, appended as produced (resume skips names already present).
    #[arg(long, default_value = "docs/results/2026-09-02-bilinear-guide.jsonl")]
    out_jsonl: String,

    #[arg(long, default_value = "docs/results/2026-09-02-bilinear-guide.json")]
    out_json: String,

    #[arg(long, default_value = "docs/results/2026-09-02-bilinear-guide.md")]
    out_md: String,

    #[arg(long, default_value = "docs/results/2026-09-02-bilinear-guide.csv")]
    out_csv: String,

    /// Recompute the report from the JSONL without evaluating anything.
    #[arg(long, default_value_t = false)]
    aggregate_only: bool,

    /// Journal to append the run record to.
    #[arg(long, default_value = "docs/results/journal.jsonl")]
    journal: String,
}

// ---------------------------------------------------------------------------
// Per-expression record.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct ArmCurve {
    grid: Vec<usize>,
    app_actual: Vec<usize>,
    cost: Vec<usize>,
    rounds: Vec<usize>,
    classes: Vec<usize>,
    ended: String,
    ended_at_apps: usize,
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
    fn best_cost_up_to(&self, limit: usize) -> usize {
        self.grid
            .iter()
            .zip(&self.cost)
            .filter(|(g, _)| **g <= limit)
            .map(|(_, c)| *c)
            .min()
            .expect("every arm's grid contains at least one checkpoint")
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
struct RuleFiring {
    fired: usize,
    strict_positive: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct PrefixDiag {
    applications: usize,
    rounds: usize,
    strict_positive: usize,
    /// Per-rule firings, keyed by CANONICAL RULE LABEL — never by a position
    /// in `all_rules()`, which a same-length reorder repoints silently.
    by_rule: BTreeMap<String, RuleFiring>,
}

/// The rule ranking a model induces at one real candidate context.
#[derive(Serialize, Deserialize, Clone)]
struct InducedOrder {
    /// Rule labels, best-scored first.
    order: Vec<String>,
    /// Score spread over the vocabulary at this context — a flat model can
    /// produce an "order" that is numerically meaningless, and a rank table
    /// alone would not show it.
    score_min: f32,
    score_max: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct ExprRow {
    set: String,
    name: String,
    /// DEV family token (`f00`..`f07`) for the §3.2 within-DEV swing; `None`
    /// for the OOD families, which have exactly one family each.
    family: Option<String>,
    run_config: Option<String>,
    node_count: usize,
    class_cap: usize,
    arms: BTreeMap<String, ArmCurve>,
    /// arm -> B -> prefix diagnostics.
    at_budget: BTreeMap<String, BTreeMap<usize, PrefixDiag>>,
    /// arm -> diagnostics over every recorded application of the run.
    full_run: BTreeMap<String, PrefixDiag>,
    /// The rule order each model class induces at this expression's first
    /// real scored context — `None` when the bilinear arm scored no
    /// candidate at all (an expression with no live match).
    induced_bilinear: Option<InducedOrder>,
    induced_linear: Option<InducedOrder>,
}

// ---------------------------------------------------------------------------
// Running one expression.
// ---------------------------------------------------------------------------

fn stop_name(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        SaturationStop::ApplicationBudget => "app_budget",
        SaturationStop::ClassCap => "class_cap",
        SaturationStop::IterationCeiling => "iteration_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

fn curve_to_arm(out: &AnytimeCurveOutput, seen_keys: Option<usize>) -> ArmCurve {
    let c = &out.curve;
    ArmCurve {
        grid: c.checkpoints.iter().map(|k| k.app_target).collect(),
        app_actual: c.checkpoints.iter().map(|k| k.app_actual).collect(),
        // DAG cost — what the emitted kernel pays (#1117). `total_cost` is
        // never read here; the registration disqualifies a round that does.
        cost: c.checkpoints.iter().map(|k| k.cost.dag).collect(),
        rounds: c.checkpoints.iter().map(|k| k.sweeps).collect(),
        classes: c.checkpoints.iter().map(|k| k.classes).collect(),
        ended: stop_name(c.ended).to_string(),
        ended_at_apps: c.ended_at_apps,
        seen_keys,
    }
}

fn rule_label_of(rules: &RuleSet, id: RuleId) -> String {
    rules
        .index_of(id)
        .and_then(|i| rules.label_of(i))
        .unwrap_or_else(|| format!("<rule {}>", id.get()))
}

fn strict_positives(out: &AnytimeCurveOutput) -> BTreeSet<ApplicationId> {
    EpisodeLabels::compute_strict(&out.egraph, out.root, &out.extraction.choices).load_bearing
}

fn prefix_diag(
    out: &AnytimeCurveOutput,
    applications: usize,
    rounds: usize,
    positives: &BTreeSet<ApplicationId>,
    rules: &RuleSet,
) -> PrefixDiag {
    let mut strict_positive = 0usize;
    let mut by_rule: BTreeMap<String, RuleFiring> = BTreeMap::new();
    for (id, rec) in out.egraph.provenance().applications() {
        if id.as_u64() as usize >= applications {
            continue;
        }
        let rule = rec.rule.unwrap_or_else(|| {
            panic!(
                "phase3_bilinear_eval: an application carries no RuleId — the graph was built \
                 without rule ids, and every table here is keyed by identity"
            )
        });
        let positive = positives.contains(&id);
        if positive {
            strict_positive += 1;
        }
        let f = by_rule.entry(rule_label_of(rules, rule)).or_default();
        f.fired += 1;
        if positive {
            f.strict_positive += 1;
        }
    }
    PrefixDiag {
        applications,
        rounds,
        strict_positive,
        by_rule,
    }
}

/// A [`SaturationGuide`] that delegates every score and keeps the FIRST
/// non-empty batch it was asked to score.
///
/// The scores it returns are the inner guide's, unchanged, so attaching it
/// cannot alter the arm's trajectory — it exists only so the report can name
/// a *real* candidate context rather than a synthetic one. The capture slot
/// is shared with the caller through an [`Rc`], because
/// [`Optimizer::guide`] takes ownership of the box and there is no way to
/// look inside it afterwards.
struct RecordingGuide {
    inner: Box<dyn SaturationGuide>,
    first_batch: Capture,
}

/// Everything about one real candidate except which rule it is — the
/// context `x` of registration §1's second difference.
///
/// A snapshot rather than the [`CandidateSummary`] itself: that type is not
/// `Clone` and does not need to be, and what the diagnostic wants is
/// precisely the rule-free part.
#[derive(Clone)]
struct ContextSnapshot {
    neighborhood_ops: Vec<OpKind>,
    budget_fraction: f32,
    match_class_node_count: usize,
    expr_node_count: usize,
}

impl ContextSnapshot {
    fn of(c: &CandidateSummary) -> Self {
        Self {
            neighborhood_ops: c.neighborhood_ops.clone(),
            budget_fraction: c.budget_fraction,
            match_class_node_count: c.match_class_node_count,
            expr_node_count: c.expr_node_count,
        }
    }

    /// The same context carrying rule `rule`, encoded the way `guide` encodes
    /// rules.
    fn with_rule(&self, rule: RuleId, guide: &dyn SaturationGuide) -> CandidateSummary {
        CandidateSummary {
            rule_embed: guide.rule_embed(rule),
            neighborhood_ops: self.neighborhood_ops.clone(),
            budget_fraction: self.budget_fraction,
            rule,
            match_class_node_count: self.match_class_node_count,
            expr_node_count: self.expr_node_count,
        }
    }
}

type Capture = Rc<RefCell<Option<ContextSnapshot>>>;

impl RecordingGuide {
    fn new(inner: Box<dyn SaturationGuide>) -> (Self, Capture) {
        let slot: Capture = Rc::new(RefCell::new(None));
        (
            Self {
                inner,
                first_batch: Rc::clone(&slot),
            },
            slot,
        )
    }
}

impl SaturationGuide for RecordingGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        if !candidates.is_empty() {
            let mut slot = self.first_batch.borrow_mut();
            if slot.is_none() {
                *slot = Some(ContextSnapshot::of(&candidates[0]));
            }
        }
        self.inner.score_candidates(candidates)
    }

    fn rule_embed(&self, rule: RuleId) -> [f32; pixelflow_search::nnue::EMBED_DIM] {
        self.inner.rule_embed(rule)
    }
}

/// The rule ranking `guide` induces at the fixed context `context`.
///
/// Holds every context field constant and substitutes each rule's identity
/// and this guide's own encoding of it — which is exactly the object
/// registration §1's second difference is about. For the additive class this
/// is `w_rule[r] + g(x)`, so the order is the same list at every context; for
/// the bilinear class it is `(m(x)ᵀW + bᵀ)r`, so it need not be.
fn induced_order(
    guide: &dyn SaturationGuide,
    context: &ContextSnapshot,
    rules: &RuleSet,
) -> InducedOrder {
    let mut probes = Vec::with_capacity(rules.len());
    let mut labels = Vec::with_capacity(rules.len());
    for i in 0..rules.len() {
        let id = rules.id_of(i).expect("index within the rule set");
        probes.push(context.with_rule(id, guide));
        labels.push(rules.label_of(i).unwrap_or_else(|| format!("{id}")));
    }
    let scores = guide.score_candidates(&probes);
    assert_eq!(scores.len(), labels.len(), "one score per probe");
    let mut idx: Vec<usize> = (0..labels.len()).collect();
    // Descending by score; ties broken by label so the order is a function
    // of the model and the context alone, never of vocabulary position.
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .expect("guide scores are finite")
            .then_with(|| labels[a].cmp(&labels[b]))
    });
    InducedOrder {
        order: idx.iter().map(|&i| labels[i].clone()).collect(),
        score_min: scores.iter().copied().fold(f32::INFINITY, f32::min),
        score_max: scores.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    }
}

struct CurveInput<'a> {
    term: Term<'a>,
    class_cap: usize,
    costs: &'a CostModel,
}

fn arm_optimizer(input: &CurveInput<'_>, guide: Option<Box<dyn SaturationGuide>>) -> Optimizer {
    Optimizer::production()
        .cost(input.costs.clone())
        .guide(guide)
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
    let out = run_anytime_curve(&mut optimizer, input.term, GUIDED_GRID);
    let seen = optimizer
        .guided_keys_seen()
        .expect("a guided optimizer carries an episode");
    (out, seen)
}

struct Guides {
    control: PerRuleRateGuide,
    linear: LinearCandidateGuide,
    bilinear: pixelflow_search::nnue::guide::bilinear::BilinearCandidateGuide,
}

fn evaluate_expression(
    set: &str,
    name: &str,
    input: &CurveInput<'_>,
    guides: &Guides,
    rules: &RuleSet,
) -> ExprRow {
    let mut unguided_opt = arm_optimizer(input, None);
    let unguided = run_anytime_curve(&mut unguided_opt, input.term, GUIDED_GRID);
    let (control, control_seen) = run_guided(Box::new(guides.control.clone()), input);
    let (linear, linear_seen) = run_guided(Box::new(guides.linear.clone()), input);

    // The bilinear arm is run through a recorder so the induced-order
    // diagnostic reads a context this expression really produced. The
    // recorder delegates every score to the guide it wraps, so this is the
    // same curve the bare guide would have produced.
    let (recorder, capture) = RecordingGuide::new(Box::new(guides.bilinear.clone()));
    let mut bilinear_opt = arm_optimizer(input, Some(Box::new(recorder)));
    let bilinear = run_anytime_curve(&mut bilinear_opt, input.term, GUIDED_GRID);
    let bilinear_seen = bilinear_opt
        .guided_keys_seen()
        .expect("a guided optimizer carries an episode");
    let context: Option<ContextSnapshot> = capture.borrow().clone();

    let (induced_bilinear, induced_linear) = match context.as_ref() {
        Some(c) => (
            Some(induced_order(&guides.bilinear, c, rules)),
            Some(induced_order(&guides.linear, c, rules)),
        ),
        None => (None, None),
    };

    let outs: [(&str, &AnytimeCurveOutput, Option<usize>); 4] = [
        ("unguided", &unguided, None),
        ("control", &control, Some(control_seen)),
        ("linear", &linear, Some(linear_seen)),
        ("bilinear", &bilinear, Some(bilinear_seen)),
    ];

    let mut arms = BTreeMap::new();
    let mut at_budget = BTreeMap::new();
    let mut full_run = BTreeMap::new();
    for (arm_name, out, seen) in outs {
        let arm = curve_to_arm(out, seen);
        let positives = strict_positives(out);
        let mut per_b = BTreeMap::new();
        for b in REGISTERED_TIERS {
            per_b.insert(
                b,
                prefix_diag(out, arm.apps_at(b), arm.rounds_at(b), &positives, rules),
            );
        }
        let total = out.egraph.provenance().applications().count();
        let last_rounds = *arm.rounds.last().expect("non-empty curve");
        full_run.insert(
            arm_name.to_string(),
            prefix_diag(out, total, last_rounds, &positives, rules),
        );
        at_budget.insert(arm_name.to_string(), per_b);
        arms.insert(arm_name.to_string(), arm);
    }

    ExprRow {
        set: set.to_string(),
        name: name.to_string(),
        family: dev_family(name),
        run_config: None,
        node_count: input.term.root().node_count(),
        class_cap: input.class_cap,
        arms,
        at_budget,
        full_run,
        induced_bilinear,
        induced_linear,
    }
}

/// The DEV family token of `dev_bNN_fMM_KKKKK`, or `None` for any other name.
fn dev_family(name: &str) -> Option<String> {
    let mut parts = name.split('_');
    if parts.next()? != "dev" {
        return None;
    }
    let _band = parts.next()?;
    let family = parts.next()?;
    let rest = parts.next()?;
    if !family.starts_with('f') || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(family.to_string())
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
    min: f64,
    q1: f64,
    median: f64,
    q3: f64,
    p90: f64,
    max: f64,
    mean: f64,
}

fn dist(values: &[f64]) -> Dist {
    assert!(!values.is_empty(), "dist of empty sample");
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Dist {
        n: v.len(),
        min: v[0],
        q1: percentile(&v, 0.25),
        median: percentile(&v, 0.5),
        q3: percentile(&v, 0.75),
        p90: percentile(&v, 0.9),
        max: v[v.len() - 1],
        mean: v.iter().sum::<f64>() / v.len() as f64,
    }
}

/// `a / b`, with the established zero-reference convention: a positive cost
/// against a zero-cost reference is infinite, never 0%.
fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        if a == 0 { 1.0 } else { f64::INFINITY }
    } else {
        a as f64 / b as f64
    }
}

fn pct_over(a: usize, reference: usize) -> f64 {
    if reference == 0 {
        if a == 0 { 0.0 } else { f64::INFINITY }
    } else {
        (a as f64 - reference as f64) / reference as f64 * 100.0
    }
}

#[derive(Serialize, Clone)]
struct ArmAtB {
    arm: String,
    ratio_vs_unguided_at_b: Dist,
    /// `(arm@B − best)/best` in percent, best = the lowest cost any arm
    /// reaches at any checkpoint of the shared grid.
    regret_pct: Dist,
    reaches_best: usize,
    improved: usize,
    unchanged: usize,
    worse: usize,
}

#[derive(Serialize, Clone)]
struct HeadToHead {
    bilinear_lt_linear: usize,
    bilinear_eq_linear: usize,
    bilinear_gt_linear: usize,
}

#[derive(Serialize, Clone)]
struct SetTier {
    set: String,
    b: usize,
    n: usize,
    arms: Vec<ArmAtB>,
    head_to_head: HeadToHead,
    /// `m_A^S(B)` per arm — the median of `ratio_vs_unguided_at_b`.
    m_arm: BTreeMap<String, f64>,
    /// `D_A(S, B) = m_A^S − m_A^DEV`, with `m_A^DEV` measured in this run.
    d_arm: BTreeMap<String, f64>,
    m_dev_reference: BTreeMap<String, f64>,
    margin_m: f64,
    /// `D_linear − D_bilinear` — positive means the bilinear arm degrades less.
    d_linear_minus_d_bilinear: f64,
}

#[derive(Serialize, Clone)]
struct SwingCheck {
    b: usize,
    /// max over DEV families of `|D_linear^f − D_bilinear^f|`.
    max_family_swing: f64,
    per_family: BTreeMap<String, f64>,
    margin_m: f64,
    /// Registration §3.2: exceeding the inherited margin means the round is
    /// underpowered at this tier and no verdict is claimed.
    powered: bool,
}

#[derive(Serialize, Clone)]
struct Verdict {
    b: usize,
    /// §3.3 conjunct 1.
    sh_gap: f64,
    c1_sh_gap_exceeds_m: bool,
    /// §3.3 conjunct 2.
    bezier_gap: f64,
    c2_bezier_within_m: bool,
    /// §3.3 conjunct 3.
    c3_dev_not_wrecked: bool,
    m_bilinear_dev: f64,
    m_linear_dev: f64,
    /// §3.3 conjunct 4.
    c4_powered_and_n30: bool,
    verdict: String,
    sentence: String,
}

/// Where the headline trig identities rank in the order a model induces at a
/// real context, per set.
#[derive(Serialize, Clone)]
struct RankTable {
    set: String,
    model: String,
    n_contexts: usize,
    distinct_orders: usize,
    /// rule label -> mean 1-based rank over the set's contexts (1 = scored
    /// highest of the whole vocabulary).
    mean_rank: BTreeMap<String, f64>,
    /// rule label -> best (lowest) rank seen on any context of this set.
    best_rank: BTreeMap<String, usize>,
    mean_score_spread: f64,
}

#[derive(Serialize, Clone)]
struct TrigFiring {
    set: String,
    arm: String,
    b: usize,
    /// rule label -> (expressions where it fired at all, total firings, strict positives)
    fired_exprs: BTreeMap<String, usize>,
    fired_total: BTreeMap<String, usize>,
    strict_positive: BTreeMap<String, usize>,
    n_exprs: usize,
}

#[derive(Serialize)]
struct Report {
    context: serde_json::Value,
    set_counts: BTreeMap<String, usize>,
    tiers: Vec<SetTier>,
    swing: Vec<SwingCheck>,
    verdicts: Vec<Verdict>,
    rank_tables: Vec<RankTable>,
    trig_firing: Vec<TrigFiring>,
}

fn arm_stats(rows: &[&ExprRow], arm: &str, b: usize) -> ArmAtB {
    let mut ratios = Vec::new();
    let mut regrets = Vec::new();
    let (mut reaches, mut improved, mut unchanged, mut worse) = (0, 0, 0, 0);
    for r in rows {
        let u = r.arms["unguided"].cost_at(b);
        let a = r.arms[arm].cost_at(b);
        ratios.push(ratio(a, u));
        match a.cmp(&u) {
            std::cmp::Ordering::Less => improved += 1,
            std::cmp::Ordering::Equal => unchanged += 1,
            std::cmp::Ordering::Greater => worse += 1,
        }
        let best = ARM_NAMES
            .iter()
            .map(|n| r.arms[*n].best_cost_up_to(REGRET_REFERENCE_LIMIT))
            .min()
            .expect("four arms");
        regrets.push(pct_over(a, best));
        if a <= best {
            reaches += 1;
        }
    }
    ArmAtB {
        arm: arm.to_string(),
        ratio_vs_unguided_at_b: dist(&ratios),
        regret_pct: dist(&regrets),
        reaches_best: reaches,
        improved,
        unchanged,
        worse,
    }
}

fn median_ratio(rows: &[&ExprRow], arm: &str, b: usize) -> f64 {
    let ratios: Vec<f64> = rows
        .iter()
        .map(|r| ratio(r.arms[arm].cost_at(b), r.arms["unguided"].cost_at(b)))
        .collect();
    dist(&ratios).median
}

fn set_tier(
    rows: &[&ExprRow],
    set: &str,
    b: usize,
    dev_reference: &BTreeMap<String, f64>,
) -> SetTier {
    let arms: Vec<ArmAtB> = ["control", "linear", "bilinear"]
        .iter()
        .map(|a| arm_stats(rows, a, b))
        .collect();
    let mut m_arm = BTreeMap::new();
    let mut d_arm = BTreeMap::new();
    for a in &arms {
        let m = a.ratio_vs_unguided_at_b.median;
        let refm = *dev_reference
            .get(&a.arm)
            .unwrap_or_else(|| panic!("no DEV reference median for arm {:?} at B={b}", a.arm));
        m_arm.insert(a.arm.clone(), m);
        d_arm.insert(a.arm.clone(), m - refm);
    }
    let (mut lt, mut eq, mut gt) = (0, 0, 0);
    for r in rows {
        match r.arms[CLAIM_ARM]
            .cost_at(b)
            .cmp(&r.arms[FROZEN_ARM].cost_at(b))
        {
            std::cmp::Ordering::Less => lt += 1,
            std::cmp::Ordering::Equal => eq += 1,
            std::cmp::Ordering::Greater => gt += 1,
        }
    }
    SetTier {
        set: set.to_string(),
        b,
        n: rows.len(),
        d_linear_minus_d_bilinear: d_arm[FROZEN_ARM] - d_arm[CLAIM_ARM],
        arms,
        head_to_head: HeadToHead {
            bilinear_lt_linear: lt,
            bilinear_eq_linear: eq,
            bilinear_gt_linear: gt,
        },
        m_arm,
        d_arm,
        m_dev_reference: dev_reference.clone(),
        margin_m: registered_margin(b),
    }
}

fn rank_table(rows: &[&ExprRow], set: &str, model: &str, headline: &[&str]) -> RankTable {
    let orders: Vec<&InducedOrder> = rows
        .iter()
        .filter_map(|r| match model {
            "bilinear" => r.induced_bilinear.as_ref(),
            "linear" => r.induced_linear.as_ref(),
            other => panic!("unknown model {other:?}"),
        })
        .collect();
    let distinct: HashSet<&Vec<String>> = orders.iter().map(|o| &o.order).collect();
    let mut mean_rank = BTreeMap::new();
    let mut best_rank = BTreeMap::new();
    for label in headline {
        let mut ranks = Vec::new();
        for o in &orders {
            let pos = o
                .order
                .iter()
                .position(|l| l == label)
                .unwrap_or_else(|| panic!("rule {label:?} absent from an induced order"));
            ranks.push(pos + 1);
        }
        if !ranks.is_empty() {
            mean_rank.insert(
                (*label).to_string(),
                ranks.iter().sum::<usize>() as f64 / ranks.len() as f64,
            );
            best_rank.insert(
                (*label).to_string(),
                *ranks.iter().min().expect("non-empty"),
            );
        }
    }
    let spread = if orders.is_empty() {
        0.0
    } else {
        orders
            .iter()
            .map(|o| f64::from(o.score_max - o.score_min))
            .sum::<f64>()
            / orders.len() as f64
    };
    RankTable {
        set: set.to_string(),
        model: model.to_string(),
        n_contexts: orders.len(),
        distinct_orders: distinct.len(),
        mean_rank,
        best_rank,
        mean_score_spread: spread,
    }
}

fn trig_firing(rows: &[&ExprRow], set: &str, arm: &str, b: usize, trig: &[&str]) -> TrigFiring {
    let mut fired_exprs = BTreeMap::new();
    let mut fired_total = BTreeMap::new();
    let mut strict = BTreeMap::new();
    for label in trig {
        let mut e = 0usize;
        let mut t = 0usize;
        let mut s = 0usize;
        for r in rows {
            let d = &r.at_budget[arm][&b];
            if let Some(f) = d.by_rule.get(*label) {
                if f.fired > 0 {
                    e += 1;
                }
                t += f.fired;
                s += f.strict_positive;
            }
        }
        fired_exprs.insert((*label).to_string(), e);
        fired_total.insert((*label).to_string(), t);
        strict.insert((*label).to_string(), s);
    }
    TrigFiring {
        set: set.to_string(),
        arm: arm.to_string(),
        b,
        fired_exprs,
        fired_total,
        strict_positive: strict,
        n_exprs: rows.len(),
    }
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

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("git unavailable: {e}"))
}

fn uptime() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("uptime unavailable: {e}"))
}

fn run_config_identity(args: &Args) -> String {
    let content = |label: &str, path: &str| match std::fs::read(path) {
        Ok(bytes) => format!("{label}={}\n", fnv1a64_hex(&bytes)),
        Err(e) => {
            panic!("phase3_bilinear_eval: cannot read {label} {path} to identify this run: {e}")
        }
    };
    let mut buf = String::new();
    buf.push_str(&format!("source_rev={}\n", git_rev()));
    buf.push_str(&content("linear_checkpoint", &args.linear_checkpoint));
    buf.push_str(&content("bilinear_checkpoint", &args.bilinear_checkpoint));
    buf.push_str(&content("train_guide_report", &args.train_guide_report));
    buf.push_str(&format!("grid={GUIDED_GRID:?}\n"));
    buf.push_str(&format!("arms={ARM_NAMES:?}\n"));
    fnv1a64_hex(buf.as_bytes())
}

fn train_fence_keys(corpus_dir: &Path) -> HashSet<FenceKey> {
    let path = corpus_dir.join("corpus_train.bin");
    assert!(
        path.exists(),
        "TRAIN corpus not found at {} — the fence check cannot run without it",
        path.display()
    );
    let entries =
        read_corpus(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    entries
        .iter()
        .map(|(_, expr)| FenceKey::of(expr.entry()))
        .collect()
}

fn enforce_train_fence(
    corpus_path: &Path,
    corpus_dir: &Path,
    entries: &[(String, Rooted<ExprData>)],
) {
    let train_keys = train_fence_keys(corpus_dir);
    let collisions: Vec<&str> = entries
        .iter()
        .filter(|(_, expr)| train_keys.contains(&FenceKey::of(expr.entry())))
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        collisions.is_empty(),
        "TRAIN-fence violation: {} of {} entries in {} share a feature-quotient structure with \
         {}/corpus_train.bin (a leak, not a hygiene event): {:?}",
        collisions.len(),
        entries.len(),
        corpus_path.display(),
        corpus_dir.display(),
        &collisions[..collisions.len().min(10)],
    );
    eprintln!(
        "phase3_bilinear_eval: TRAIN fence OK — {} entries probed against {} TRAIN structures, 0 collisions",
        entries.len(),
        train_keys.len(),
    );
}

fn main() {
    let args = Args::parse();
    let rules = RuleSet::production();
    let trig = trig_rules(&rules);
    let jsonl_path = PathBuf::from(&args.out_jsonl);
    let started = uptime();

    // Registration §4 names a frozen additive checkpoint. Probe it and
    // record what happens, in the loader's own words: a research round that
    // silently substituted a different file for a registered arm would be
    // unreviewable, and a round that hand-edited the frozen file to make it
    // load would be disqualified outright (§7).
    let frozen_arm_status =
        match load_linear_guide(Path::new(&args.frozen_linear_checkpoint), &rules) {
            Ok(_) => format!(
                "loaded — {} is deployable on this build",
                args.frozen_linear_checkpoint
            ),
            Err(e) => format!("REFUSED, not deployed: {e}"),
        };
    eprintln!("phase3_bilinear_eval: frozen additive arm: {frozen_arm_status}");

    if !args.aggregate_only {
        let costs = CostModel::latency_prior();
        let corpus_path = PathBuf::from(&args.corpus);
        let corpus_dir = PathBuf::from(&args.corpus_dir);
        let mut entries = read_corpus(&corpus_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", corpus_path.display()));
        enforce_train_fence(&corpus_path, &corpus_dir, &entries);
        if !args.name_prefix.is_empty() {
            entries.retain(|(name, _)| name.starts_with(&args.name_prefix));
            assert!(
                !entries.is_empty(),
                "--name-prefix {:?} matches no entry in {}",
                args.name_prefix,
                corpus_path.display()
            );
        }
        // The registered claim is on the classical band only (node count > 50).
        entries.retain(|(_, expr)| expr.len() > 50);
        entries.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| a.0.cmp(&b.0)));
        eprintln!(
            "phase3_bilinear_eval: set {:?} — {} classical expressions",
            args.set,
            entries.len()
        );

        let run_config = run_config_identity(&args);
        let existing: HashSet<String> = read_rows(&jsonl_path)
            .into_iter()
            .map(|r| {
                assert_eq!(
                    r.run_config.as_deref(),
                    Some(run_config.as_str()),
                    "{}: row {:?} was produced by run configuration {:?}, but this run is \
                     {run_config:?} — appending would mix incomparable measurements. Delete \
                     the file to re-evaluate under the current configuration.",
                    jsonl_path.display(),
                    r.name,
                    r.run_config.as_deref().unwrap_or("<absent>"),
                );
                r.name
            })
            .collect();

        let guides = Guides {
            control: per_rule_rate_guide_from_report(Path::new(&args.train_guide_report))
                .unwrap_or_else(|e| panic!("control guide: {e}")),
            linear: load_linear_guide(Path::new(&args.linear_checkpoint), &rules)
                .unwrap_or_else(|e| panic!("linear guide: {e}")),
            bilinear: load_bilinear_guide(Path::new(&args.bilinear_checkpoint), &rules)
                .unwrap_or_else(|e| panic!("bilinear guide: {e}")),
        };

        if let Some(parent) = jsonl_path.parent() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl_path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", jsonl_path.display()));

        let total = entries.len();
        // A corpus entry declares no buffers or uniforms — the format refuses
        // to write one down — so one empty environment serves every term.
        let env = Environment::new();
        for (i, (name, expr)) in entries.iter().enumerate() {
            if existing.contains(name) {
                continue;
            }
            eprintln!(
                "phase3_bilinear_eval: [{}/{total}] {name} ({} nodes)",
                i + 1,
                expr.len()
            );
            let input = CurveInput {
                term: Term::new(expr.entry(), &env),
                class_cap: config_for_node_count(expr.len()).max_classes,
                costs: &costs,
            };
            let mut row = evaluate_expression(&args.set, name, &input, &guides, &rules);
            row.run_config = Some(run_config.clone());
            let line = serde_json::to_string(&row).expect("serialize row");
            writeln!(out, "{line}")
                .unwrap_or_else(|e| panic!("write {}: {e}", jsonl_path.display()));
            out.flush().expect("flush");
        }
    }

    // ---------------------------------------------------------------- aggregate
    let rows = read_rows(&jsonl_path);
    assert!(
        !rows.is_empty(),
        "no rows in {} — nothing to aggregate",
        jsonl_path.display()
    );
    let by_set = |s: &str| -> Vec<&ExprRow> { rows.iter().filter(|r| r.set == s).collect() };
    let dev = by_set("dev");
    let sh = by_set("sh");
    let bezier = by_set("bezier");
    assert!(
        !dev.is_empty(),
        "no DEV rows in {} — every D is taken against a DEV reference measured in this run, so \
         the DEV set must be evaluated before any verdict",
        jsonl_path.display()
    );

    let mut set_counts = BTreeMap::new();
    for (label, r) in [("dev", &dev), ("sh", &sh), ("bezier", &bezier)] {
        set_counts.insert(label.to_string(), r.len());
    }

    let mut tiers = Vec::new();
    let mut swing = Vec::new();
    let mut verdicts = Vec::new();
    let mut dev_refs: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();
    for b in REGISTERED_TIERS {
        let mut refm = BTreeMap::new();
        for arm in ["control", "linear", "bilinear"] {
            refm.insert(arm.to_string(), median_ratio(&dev, arm, b));
        }
        dev_refs.insert(b, refm);
    }

    for b in REGISTERED_TIERS {
        let refm = &dev_refs[&b];
        let m = registered_margin(b);
        for (label, set_rows) in [("dev", &dev), ("sh", &sh), ("bezier", &bezier)] {
            if set_rows.is_empty() {
                continue;
            }
            tiers.push(set_tier(set_rows, label, b, refm));
        }

        // §3.2: recompute the within-DEV family swing on THIS instrument.
        let mut families: BTreeMap<String, Vec<&ExprRow>> = BTreeMap::new();
        for r in &dev {
            if let Some(f) = &r.family {
                families.entry(f.clone()).or_default().push(r);
            }
        }
        let mut per_family = BTreeMap::new();
        let mut max_swing: f64 = 0.0;
        for (f, frows) in &families {
            let d_lin = median_ratio(frows, "linear", b) - refm["linear"];
            let d_bil = median_ratio(frows, "bilinear", b) - refm["bilinear"];
            let s = (d_lin - d_bil).abs();
            per_family.insert(f.clone(), s);
            max_swing = max_swing.max(s);
        }
        swing.push(SwingCheck {
            b,
            max_family_swing: max_swing,
            per_family,
            margin_m: m,
            powered: max_swing <= m,
        });

        // §3.3 verdict.
        let get =
            |label: &str| -> Option<&SetTier> { tiers.iter().find(|t| t.set == label && t.b == b) };
        if let (Some(sh_t), Some(bz_t)) = (get("sh"), get("bezier")) {
            let sh_gap = sh_t.d_linear_minus_d_bilinear;
            let bezier_gap = bz_t.d_linear_minus_d_bilinear;
            let c1 = sh_gap > m;
            let c2 = bezier_gap.abs() <= m;
            let m_bil_dev = refm["bilinear"];
            let m_lin_dev = refm["linear"];
            let c3 = m_bil_dev <= m_lin_dev + m;
            let powered = max_swing <= m;
            let c4 = powered && sh_t.n >= 30 && bz_t.n >= 30;
            let verdict = if !c4 {
                "UNDERPOWERED — no verdict claimed (registration §3.2/§3.3.4)".to_string()
            } else if c1 && bezier_gap > m {
                "H_capacity".to_string()
            } else if c1 && c2 && c3 {
                "H_repr".to_string()
            } else if -sh_gap > m {
                "H_worse".to_string()
            } else if sh_gap.abs() <= m {
                "H_form-null".to_string()
            } else {
                format!(
                    "unclassified (sh gap {sh_gap:+.4} vs M={m:.2}, conjuncts c1={c1} c2={c2} c3={c3})"
                )
            };
            let sentence = match verdict.as_str() {
                "H_repr" => format!(
                    "B={b}: H_repr ACCEPTED — the bilinear arm degrades {sh_gap:+.4} less than \
                     the frozen additive arm under shift to `sh`, past M_{b}={m:.2}, while \
                     staying within {:.4} of it on `bezier`.",
                    bezier_gap.abs()
                ),
                "H_form-null" => format!(
                    "B={b}: H_form-null — D_linear − D_bilinear on `sh` is {sh_gap:+.4}, inside \
                     M_{b}={m:.2}; the rule-by-context interaction buys nothing under domain \
                     shift, so the functional form was not the bottleneck.",
                ),
                "H_capacity" => format!(
                    "B={b}: H_capacity — the bilinear arm gains on BOTH `sh` ({sh_gap:+.4}) and \
                     `bezier` ({bezier_gap:+.4}) past M_{b}={m:.2}: added capacity helps \
                     everywhere, which is not the domain-conditional claim.",
                ),
                "H_worse" => format!(
                    "B={b}: H_worse — the interaction degrades MORE than the additive arm under \
                     shift ({sh_gap:+.4} against M_{b}={m:.2}).",
                ),
                other => format!("B={b}: {other}"),
            };
            verdicts.push(Verdict {
                b,
                sh_gap,
                c1_sh_gap_exceeds_m: c1,
                bezier_gap,
                c2_bezier_within_m: c2,
                c3_dev_not_wrecked: c3,
                m_bilinear_dev: m_bil_dev,
                m_linear_dev: m_lin_dev,
                c4_powered_and_n30: c4,
                verdict,
                sentence,
            });
        }
    }

    let mut rank_tables = Vec::new();
    let mut trig_firing_rows = Vec::new();
    for (label, set_rows) in [("dev", &dev), ("sh", &sh), ("bezier", &bezier)] {
        if set_rows.is_empty() {
            continue;
        }
        for model in ["bilinear", "linear"] {
            rank_tables.push(rank_table(set_rows, label, model, &HEADLINE_TRIG));
        }
        for arm in ARM_NAMES {
            for b in REGISTERED_TIERS {
                trig_firing_rows.push(trig_firing(set_rows, label, arm, b, &trig));
            }
        }
    }

    let report = Report {
        context: serde_json::json!({
            "registration": "docs/plans/2026-09-02-bilinear-guide-registration.md",
            "inherited": [
                "docs/plans/2026-09-01-phase3-registration.md",
                "docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md",
            ],
            "source_rev": git_rev(),
            "cost": "ExtractedDAG::dag_cost (ChoiceCost::dag, #1117) under CostModel::latency_prior()",
            "budget": "recorded rule applications; no wall clock in any metric",
            "grid": GUIDED_GRID,
            "arms": ARM_NAMES,
            "linear_checkpoint": args.linear_checkpoint,
            "frozen_linear_checkpoint": args.frozen_linear_checkpoint,
            "frozen_linear_arm_status": frozen_arm_status,
            "deviation_from_registration_4": "The registration's frozen `LinearCandidateGuide` arm \
        (guide_checkpoint_strict_v1.json) is off-vocabulary on this branch and is refused by \
        LinearWeights::check_vocabulary; the additive arm that ran is the same-recipe model retrained on \
        the same re-minted dataset as the bilinear arm. Recorded, not worked around.",
            "bilinear_checkpoint": args.bilinear_checkpoint,
            "train_guide_report": args.train_guide_report,
            "uptime_at_start": started,
            "uptime_at_end": uptime(),
        }),
        set_counts,
        tiers,
        swing,
        verdicts,
        rank_tables,
        trig_firing: trig_firing_rows,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&args.out_json, &json)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", args.out_json));

    write_csv(&rows, Path::new(&args.out_csv));
    write_markdown(&report, Path::new(&args.out_md), &args.out_jsonl);

    append_record(
        Path::new(&args.journal),
        &serde_json::json!({
            "date": "2026-09-02",
            "kind": "phase3_bilinear_eval",
            "registration": "docs/plans/2026-09-02-bilinear-guide-registration.md",
            "source_rev": git_rev(),
            "sets": report.set_counts,
            "verdicts": report.verdicts.iter().map(|v| (v.b, v.verdict.clone())).collect::<Vec<_>>(),
            "outputs": [args.out_md, args.out_json, args.out_csv, args.out_jsonl],
            "uptime": uptime(),
        }),
    );

    for v in &report.verdicts {
        println!("{}", v.sentence);
    }
}

fn write_csv(rows: &[ExprRow], path: &Path) {
    let mut csv = String::from(
        "set,name,node_count,budget,unguided_dag_cost,control_dag_cost,linear_dag_cost,\
         bilinear_dag_cost,control_ratio,linear_ratio,bilinear_ratio,best_any_arm\n",
    );
    for r in rows {
        for b in REGISTERED_TIERS {
            let u = r.arms["unguided"].cost_at(b);
            let c = r.arms["control"].cost_at(b);
            let l = r.arms["linear"].cost_at(b);
            let bl = r.arms["bilinear"].cost_at(b);
            let best = ARM_NAMES
                .iter()
                .map(|n| r.arms[*n].best_cost_up_to(REGRET_REFERENCE_LIMIT))
                .min()
                .expect("four arms");
            csv.push_str(&format!(
                "{},{},{},{b},{u},{c},{l},{bl},{:.6},{:.6},{:.6},{best}\n",
                r.set,
                r.name,
                r.node_count,
                ratio(c, u),
                ratio(l, u),
                ratio(bl, u),
            ));
        }
    }
    std::fs::write(path, csv).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

fn fmt_dist(d: &Dist) -> String {
    format!(
        "{:.4} | {:.4} | **{:.4}** | {:.4} | {:.4}",
        d.min, d.q1, d.median, d.q3, d.p90
    )
}

fn write_markdown(report: &Report, path: &Path, jsonl: &str) {
    let mut md = String::new();
    md.push_str("# The bilinear Guide: does a rule-by-context interaction buy a domain-conditional advantage?\n\n");
    md.push_str("**Date:** 2026-09-02  \n");
    md.push_str("**Registration:** `docs/plans/2026-09-02-bilinear-guide-registration.md` (frozen; §11 deliberately left unedited — this document is the appendix it names)  \n");
    md.push_str(&format!("**Rows:** `{jsonl}`  \n"));
    md.push_str(&format!(
        "**Cost:** `ExtractedDAG::dag_cost` under `CostModel::latency_prior()`. No wall clock in any number.  \n**Grid (every arm):** {GUIDED_GRID:?} recorded rule applications.\n\n"
    ));
    // The registration names a frozen additive arm. Whether that file could be
    // deployed belongs in the header, not in a JSON field a reader has to go
    // looking for.
    md.push_str(&format!(
        "**Additive arm:** `{}`.  \n**Registration §4's frozen additive checkpoint** (`{}`): {}\n\n",
        report.context["linear_checkpoint"].as_str().unwrap_or("?"),
        report.context["frozen_linear_checkpoint"].as_str().unwrap_or("?"),
        report.context["frozen_linear_arm_status"].as_str().unwrap_or("?"),
    ));

    md.push_str("## Verdict\n\n");
    for v in &report.verdicts {
        md.push_str(&format!("> {}\n>\n", v.sentence));
    }
    md.push('\n');

    md.push_str("| B | D_linear − D_bilinear on `sh` | M_B | `bezier` gap | m_bilinear^DEV | m_linear^DEV | powered | verdict |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|:--|:--|\n");
    for v in &report.verdicts {
        md.push_str(&format!(
            "| {} | **{:+.4}** | {:.2} | {:+.4} | {:.4} | {:.4} | {} | **{}** |\n",
            v.b,
            v.sh_gap,
            registered_margin(v.b),
            v.bezier_gap,
            v.m_bilinear_dev,
            v.m_linear_dev,
            if v.c4_powered_and_n30 { "yes" } else { "NO" },
            v.verdict
        ));
    }

    md.push_str("\n## Per set and tier: cost ratio vs unguided-at-B\n\n");
    md.push_str("`dag_cost_arm@B / dag_cost_unguided@B`, per expression.\n\n");
    md.push_str("| set | n | B | arm | min | q1 | median | q3 | p90 | improved | = | worse | regret% (median) | reaches best |\n");
    md.push_str("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for t in &report.tiers {
        for a in &t.arms {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} |\n",
                t.set,
                t.n,
                t.b,
                a.arm,
                fmt_dist(&a.ratio_vs_unguided_at_b),
                a.improved,
                a.unchanged,
                a.worse,
                a.regret_pct.median,
                a.reaches_best,
            ));
        }
    }

    md.push_str("\n## The registered statistic `D_A(S, B) = m_A^S(B) − m_A^DEV(B)`\n\n");
    md.push_str("`m_A^DEV` is measured in THIS run on `dag_cost` for every arm (registration's instrument note); no round-1 constant is reused.\n\n");
    md.push_str("| set | B | m_control | m_linear | m_bilinear | D_control | D_linear | D_bilinear | D_linear − D_bilinear | M_B |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for t in &report.tiers {
        md.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:+.4} | {:+.4} | {:+.4} | **{:+.4}** | {:.2} |\n",
            t.set,
            t.b,
            t.m_arm["control"],
            t.m_arm["linear"],
            t.m_arm["bilinear"],
            t.d_arm["control"],
            t.d_arm["linear"],
            t.d_arm["bilinear"],
            t.d_linear_minus_d_bilinear,
            t.margin_m,
        ));
    }

    md.push_str("\n### Head to head: bilinear vs linear, per expression at B\n\n");
    md.push_str(
        "| set | B | bilinear < linear | = | bilinear > linear |\n|---|---:|---:|---:|---:|\n",
    );
    for t in &report.tiers {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            t.set,
            t.b,
            t.head_to_head.bilinear_lt_linear,
            t.head_to_head.bilinear_eq_linear,
            t.head_to_head.bilinear_gt_linear
        ));
    }

    md.push_str("\n### §3.2 power check: within-DEV family swing on this instrument\n\n");
    md.push_str("Max over the 8 DEV families of `|D_linear^f − D_bilinear^f|`. Exceeding the inherited `M_B` means the round is underpowered at that tier and no verdict is claimed; the inherited margin is never raised to match.\n\n");
    md.push_str("| B | max family swing | M_B | powered |\n|---:|---:|---:|:--|\n");
    for s in &report.swing {
        md.push_str(&format!(
            "| {} | {:.4} | {:.2} | {} |\n",
            s.b,
            s.max_family_swing,
            s.margin_m,
            if s.powered { "yes" } else { "**NO**" }
        ));
    }

    md.push_str("\n## Diagnostic 1 — do the arms fire the trig identities on `sh`?\n\n");
    md.push_str("Firings within the first B applications, pooled over the set. `exprs` = expressions where the rule fired at all; `fired` = total applications; `sp` = strict-positive.\n\n");
    for b in REGISTERED_TIERS {
        md.push_str(&format!("\n**B = {b}**\n\n"));
        md.push_str("| set | arm | ");
        for label in HEADLINE_TRIG {
            md.push_str(&format!("{label} | "));
        }
        md.push_str("\n|---|---|");
        for _ in HEADLINE_TRIG {
            md.push_str("---:|");
        }
        md.push('\n');
        for row in report.trig_firing.iter().filter(|r| r.b == b) {
            md.push_str(&format!("| {} | {} | ", row.set, row.arm));
            for label in HEADLINE_TRIG {
                let e = row.fired_exprs.get(label).copied().unwrap_or(0);
                let f = row.fired_total.get(label).copied().unwrap_or(0);
                let s = row.strict_positive.get(label).copied().unwrap_or(0);
                md.push_str(&format!("{e} exprs / {f} fired / {s} sp | "));
            }
            md.push('\n');
        }
    }

    md.push_str(
        "\n## Diagnostic 2 — does the bilinear model's induced rule ORDER differ by context?\n\n",
    );
    md.push_str(
        "Each expression's first really-scored candidate is held fixed as the context `x`; every rule's \
         identity and embedding is substituted in turn, and the model's ranking of the whole vocabulary \
         at that context is read off. Registration §1: for the additive class this order is the same list \
         at every context, by construction; for the bilinear class it is free to vary. `rank` is 1-based \
         over the whole rule vocabulary (1 = scored highest).\n\n",
    );
    md.push_str("| set | model | contexts | distinct orders | mean score spread | ");
    for label in HEADLINE_TRIG {
        md.push_str(&format!("{label} mean rank (best) | "));
    }
    md.push_str("\n|---|---|---:|---:|---:|");
    for _ in HEADLINE_TRIG {
        md.push_str("---:|");
    }
    md.push('\n');
    for t in &report.rank_tables {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.4} | ",
            t.set, t.model, t.n_contexts, t.distinct_orders, t.mean_score_spread
        ));
        for label in HEADLINE_TRIG {
            match (t.mean_rank.get(label), t.best_rank.get(label)) {
                (Some(m), Some(b)) => md.push_str(&format!("{m:.1} ({b}) | ")),
                _ => md.push_str("— | "),
            }
        }
        md.push('\n');
    }

    md.push_str("\n## Context\n\n```json\n");
    md.push_str(&serde_json::to_string_pretty(&report.context).expect("context"));
    md.push_str("\n```\n");

    std::fs::write(path, md).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}
