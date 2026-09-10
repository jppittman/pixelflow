//! Counterfactual replay: does removing one application actually help or
//! hurt the extraction at budget B, and do the hand-drawn hindsight bounds
//! (loose / tight / strict / strict-by-output-class) agree with that ground
//! truth? (docs/plans/2026-09-01-guide-return-to-go.md §4, the validation
//! half of the R2G redirect — this binary does not touch §1-§3's training
//! target or model; it only scores the bounds against a measured
//! counterfactual.)
//!
//! # The counterfactual
//!
//! For an expression's UNGUIDED trajectory to budget `B`, and a sampled
//! recorded application `a` at ordinal `t_a`, `τ \ a` is the deterministic
//! masked replay of the identical trajectory with `a` skipped — not applied,
//! not recorded, so the budget slot it would have consumed is spent on
//! whatever candidate the same rule-major scan finds next
//! ([`EGraph::saturate_until_applications_observed`],
//! [`pixelflow_search::egraph::ApplicationMask`]; see that method's doc and
//! `saturate_until_applications_observed_without_mask_matches_original_bit_for_bit`
//! in `pixelflow-search/src/egraph/graph.rs` for the bit-identity pin this
//! relies on).
//!
//! `Δ_a = ln(cost(τ\a, B)) − ln(cost(τ, B))`. This is exactly the plan's
//! `R(τ\a,B) − R(τ,B)` log-regret difference
//! (`R(τ,B) = ln(cost(τ,B) / c*_e)`): the per-expression reference cost
//! `c*_e` cancels in the subtraction, so no cross-trajectory "empirical
//! best" needs to be minted here — the K=12-trajectory diversity dataset
//! (plan §2) is a separate, larger piece of work this binary does not build.
//! `Δ_a > 0` — masked cost exceeds original — means `a` was genuinely
//! helpful; `Δ_a < 0` means `a` was actively wasteful (its slot was worth
//! more to whatever fired instead); `Δ_a = 0` means it didn't matter.
//!
//! # The four bounds
//!
//! `loose`/`tight`/`strict` are [`EpisodeLabels::compute`] /
//! `compute_tight` / `compute_strict` against the ORIGINAL (unmasked)
//! trajectory's extraction at B — the labeler's existing hindsight-ancestry
//! walks, unmodified. `strict_by_output_class` is NEW and computed inline,
//! per the task spec: positive iff the canonical class of `a`'s output — any
//! e-node whose origin is `Origin::Rule(a)`, or either side of a
//! `UnionEvent` `a` caused — is (after `find`) a canonical class the chosen
//! extraction actually visits. This is what credits `pythagorean`-shaped
//! rules whose output unions into an already-existing node (the literal `1`
//! constant): `compute_strict` misses that firing entirely because the
//! chosen node it unioned into was never *created* by `a`, but the class it
//! touched is exactly the one the extraction walks.
//!
//! # The model proxies (plan §4.3, Claim B)
//!
//! With `--r2g-checkpoint` / `--strict-checkpoint` / `--train-guide-report`
//! given, every recorded application `b` with ordinal `< B` on the original
//! trajectory is featurized ONCE against the e-graph at B
//! (`CandidateFeatures::observe` — the same constructor every Guide is fed,
//! the same post-hoc observation the R2G mint used; see
//! `gen_r2g_trajectories`' module doc for why that approximation is stated
//! rather than hidden) and scored by each loaded Guide. For a sampled `a`
//! with `A_t` = the OTHER applications recorded in the same sweep
//! (`ApplicationRecord::step`), the advantage is
//!
//! - R2G: `adv_a = mean_{b ∈ A_t} f(x_b) − f(x_a)`, `f` = predicted return
//!   (`LinearReturnGuide::predicted_return`; positive = `a` predicted
//!   better than its sweep-mates, since a return is a regret);
//! - strict-v1 / per-rule: `adv_a = s(x_a) − mean_{b ∈ A_t} s(x_b)`, `s` =
//!   `score_candidates` (bigger-is-better orientation).
//!
//! An `a` alone in its sweep has no alternatives; its advantage is `None`,
//! it is excluded from the model-proxy correlations, and the exclusion
//! count is reported (never a silent 0). Spearman ρ against Δ is reported
//! pooled and per set with a seeded paired bootstrap CI, plus the CI of
//! each proxy's difference from the R2G model — evidence, not the gate.
//!
//! # Sampling and the wall-clock ceiling
//!
//! Applications are sampled uniformly (seeded, without replacement) from
//! each expression's STATE-CHANGING recorded applications (those that
//! caused at least one `UnionEvent` — the no-op re-fires that make up most
//! of a sweep have `Δ = 0` by construction and would only inflate a tie
//! count, so they are not sampled; their share is reported). This binary is
//! CPU-heavy (one full masked replay per sampled application), so — per this
//! task's own instruction, an explicit exception to this codebase's usual
//! "safety ceilings panic" convention for the SAMPLING loop specifically —
//! `--wall-clock-ceiling-secs` bounds it: hitting the ceiling stops sampling
//! loudly (an `eprintln!` and a `wall_clock_ceiling_hit: true` field in the
//! report) with whatever count was achieved, never a silent partial. Each
//! individual saturation call still has its own generous per-run timeout
//! that PANICS if it binds (`SATURATE_TIMEOUT`) — that is a correctness
//! ceiling on one measurement, unrelated to the outer sampling budget.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin counterfactual_credit -- \
//!     --corpus-dir pixelflow-pipeline/data \
//!     --n-expr-sh 30 --n-expr-dev 30 --n-apps 20 --budget 100 \
//!     --out-jsonl docs/results/2026-09-01-counterfactual-credit.jsonl \
//!     --out-json docs/results/2026-09-01-counterfactual-credit.json \
//!     --out-md docs/results/2026-09-01-counterfactual-credit.md
//! ```

use std::collections::{BTreeSet, HashMap};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;

use pixelflow_ir::{Environment, ExprData, Rooted, Term};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::guide_linear::{
    load_linear_guide, per_rule_rate_guide_from_report,
};
use pixelflow_pipeline::training::r2g::load_return_guide;
use pixelflow_search::egraph::provenance::{ApplicationId, ENodeId, Origin};
use pixelflow_search::egraph::{
    ApplicationMask, Budget, CandidateFeatures, CostModel, EClassId, EGraph, EpisodeLabels, Firing,
    KeepJournal, Label, MaskScope, Optimizer, REGISTERED_PRIMARY_BUDGET_APPLICATIONS, RuleId,
    RuleSet, config_for_node_count, extract_dag,
};
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::linear::{
    LinearCandidateGuide, LinearReturnGuide, PerRuleRateGuide,
};
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

/// Per-run safety ceiling for ONE `saturate_until_applications[_observed]`
/// call. This is a correctness guard (offline measurement must fail loud
/// rather than silently truncate one replay), not the outer sampling
/// ceiling below — see the module doc.
const SATURATE_TIMEOUT: Duration = Duration::from_secs(30);
/// Sweep-count ceiling for one run to a B ≤ a few hundred applications —
/// generous headroom over the handful of sweeps that actually takes.
const SATURATE_MAX_ITERS: usize = 500;

#[derive(Parser)]
#[command(name = "counterfactual_credit")]
#[command(about = "Score the hindsight bounds against a measured leave-one-out counterfactual")]
struct Args {
    /// Directory holding `corpus_dev.bin` / `corpus_dev_ood.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Minimum `sh` (OOD) expressions to sample (`corpus_dev_ood.bin`,
    /// `dev_sh_*` entries).
    #[arg(long, default_value_t = 30)]
    n_expr_sh: usize,

    /// Minimum DEV classical expressions to sample (`corpus_dev.bin`).
    #[arg(long, default_value_t = 30)]
    n_expr_dev: usize,

    /// Lower bound (inclusive) on an expression's arena node count; `0` =
    /// unbounded. Round 3's registered regime
    /// (docs/plans/2026-09-01-guide-return-to-go.md §2b.3) is the band
    /// 101-1000 where guided orderings measurably disagree — the credit
    /// question is only meaningful where the orderings differ at all.
    #[arg(long, default_value_t = 0)]
    min_expr_nodes: usize,

    /// Upper bound (inclusive) on an expression's arena node count; `0` =
    /// unbounded. See `--min-expr-nodes`.
    #[arg(long, default_value_t = 0)]
    max_expr_nodes: usize,

    /// Target sampled (state-changing) applications per expression.
    #[arg(long, default_value_t = 20)]
    n_apps: usize,

    /// Budget B (recorded applications) both the original and every masked
    /// replay run to.
    #[arg(long, default_value_t = 100)]
    budget: usize,

    /// Seed for expression selection and per-expression application
    /// sampling — deterministic and reproducible from this one value.
    #[arg(long, default_value_t = 0x5EED_C0DE_C0FF_EE01)]
    seed: u64,

    /// Wall-clock ceiling on the WHOLE sampling loop (across every
    /// expression and application) — see the module doc. Not a per-run
    /// correctness ceiling.
    #[arg(long, default_value_t = 1_200)]
    wall_clock_ceiling_secs: u64,

    /// `train_guide_r2g` checkpoint — adds the R2G model-advantage proxy
    /// (plan §4.3, the claim column).
    #[arg(long)]
    r2g_checkpoint: Option<String>,

    /// Round-1 strict-bit `train_guide` checkpoint — adds the strict-v1
    /// linear Guide's advantage proxy.
    #[arg(long)]
    strict_checkpoint: Option<String>,

    /// `train_guide` report JSON — adds the `PerRuleRateGuide` advantage
    /// proxy (the Round-1 control).
    #[arg(long)]
    train_guide_report: Option<String>,

    /// Paired-bootstrap resamples for the Spearman CIs (seeded from
    /// `--seed`).
    #[arg(long, default_value_t = 1000)]
    bootstrap_resamples: usize,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-counterfactual-credit.jsonl"
    )]
    out_jsonl: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-counterfactual-credit.json"
    )]
    out_json: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-counterfactual-credit.md"
    )]
    out_md: String,
}

// ---------------------------------------------------------------------------
// Seeded RNG — same xorshift64* idiom this crate's other binaries use
// locally (e.g. `bench_extraction_3way.rs`'s `SeededRng`); determinism only,
// not cryptographic quality, and not worth a shared dependency.
// ---------------------------------------------------------------------------

struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        // SplitMix64-style finalizer so a low-entropy seed (including 0)
        // doesn't produce a degenerate short cycle.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self(z ^ (z >> 31))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Fisher-Yates shuffle.
    fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            s.swap(i, j);
        }
    }
}

/// Seeded selection of at least `min_n` items out of `entries` (all of them
/// if fewer are available): shuffle then take a prefix, so the same seed
/// reproduces the same sample and every entry has equal a-priori chance of
/// selection.
fn seeded_select<T: Clone>(mut entries: Vec<T>, min_n: usize, rng: &mut SeededRng) -> Vec<T> {
    rng.shuffle(&mut entries);
    if min_n >= entries.len() {
        entries
    } else {
        entries.truncate(min_n);
        entries
    }
}

fn uptime() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("uptime unavailable: {e}"))
}

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("git unavailable: {e}"))
}

// ---------------------------------------------------------------------------
// The model proxies (module doc): whichever Guides the caller loaded.
// ---------------------------------------------------------------------------

struct Proxies {
    r2g: Option<LinearReturnGuide>,
    strict_v1: Option<LinearCandidateGuide>,
    per_rule: Option<PerRuleRateGuide>,
}

/// One proxy's per-sweep sums, so `mean over the OTHER applications in the
/// sweep` is `(sum − own) / (count − 1)` without a per-`a` rescan.
struct SweepScores {
    /// `ApplicationId -> this proxy's raw score for that application`.
    own: HashMap<ApplicationId, f64>,
    /// `step -> (sum of scores, count)` over the recorded applications with
    /// ordinal `< B` in that sweep.
    per_step: HashMap<usize, (f64, usize)>,
}

impl SweepScores {
    /// `Some((own score, mean of the other same-sweep scores))`, `None` when
    /// `a` has no same-sweep alternative (the exclusion the module doc
    /// reports).
    fn own_and_others(&self, app_id: ApplicationId, step: usize) -> Option<(f64, f64)> {
        let own = *self.own.get(&app_id).unwrap_or_else(|| {
            panic!(
                "counterfactual_credit: application {} was sampled but never scored",
                app_id.as_u64()
            )
        });
        let &(sum, count) = self.per_step.get(&step).unwrap_or_else(|| {
            panic!("counterfactual_credit: sweep {step} has no score aggregate")
        });
        assert!(
            count >= 1,
            "counterfactual_credit: sweep {step} aggregate is empty"
        );
        if count == 1 {
            return None;
        }
        Some((own, (sum - own) / (count - 1) as f64))
    }
}

/// Score every recorded application with ordinal `< bound` by `guide`,
/// batched once (the same one-batch-per-round discipline the live loop has),
/// and aggregate per sweep.
fn sweep_scores<G: SaturationGuide>(
    guide: &G,
    keys: &[(ApplicationId, usize)],
    summaries: &[CandidateSummary],
) -> SweepScores {
    assert_eq!(keys.len(), summaries.len());
    let scores = guide.score_candidates(summaries);
    assert_eq!(
        scores.len(),
        summaries.len(),
        "counterfactual_credit: guide returned a ragged score batch"
    );
    let mut own = HashMap::with_capacity(summaries.len());
    let mut per_step: HashMap<usize, (f64, usize)> = HashMap::new();
    for ((app_id, step), score) in keys.iter().zip(scores) {
        assert!(
            score.is_finite(),
            "counterfactual_credit: non-finite guide score for application {}",
            app_id.as_u64()
        );
        own.insert(*app_id, score as f64);
        let e = per_step.entry(*step).or_insert((0.0, 0));
        e.0 += score as f64;
        e.1 += 1;
    }
    SweepScores { own, per_step }
}

// ---------------------------------------------------------------------------
// Per-expression replay context: everything computed ONCE against the
// original (unmasked) trajectory, reused across every sampled application.
// ---------------------------------------------------------------------------

struct ExprContext {
    name: String,
    /// The expression, owned, plus the tables its leaves index — the pair
    /// [`ExprContext::term`] hands on.
    expr: Rooted<ExprData>,
    env: Environment,
    max_classes: usize,
    egraph: EGraph,
    cost_original: usize,
    loose: EpisodeLabels,
    tight: EpisodeLabels,
    strict: EpisodeLabels,
    /// `Origin::Rule(a)`-tagged nodes this application created, keyed by
    /// `a` — built once per expression from `provenance().origins()`
    /// instead of re-scanning it per sampled application.
    created_by: HashMap<ApplicationId, Vec<ENodeId>>,
    /// Class pairs unioned as a direct result of application `a`.
    unions_by: HashMap<ApplicationId, Vec<(EClassId, EClassId)>>,
    /// `ENodeId -> its current canonical EClassId`.
    node_to_class: HashMap<ENodeId, EClassId>,
    /// Canonical classes the chosen extraction actually visits.
    reachable: BTreeSet<EClassId>,
    /// Recorded, state-changing application ids with ordinal `< budget`, in
    /// ordinal order — the pool `--n-apps` samples from.
    candidates: Vec<(ApplicationId, RuleId)>, // (id, rule)
    /// `ApplicationId -> ApplicationRecord::step` (the sweep), for every
    /// recorded application with ordinal `< budget`.
    step_of: HashMap<ApplicationId, usize>,
    /// Per-proxy scores over the same applications (module doc); `None` for
    /// a proxy the caller did not load.
    scores_r2g: Option<SweepScores>,
    scores_strict_v1: Option<SweepScores>,
    scores_per_rule: Option<SweepScores>,
}

/// Walk the chosen extraction from `root`, collecting every canonical class
/// it visits. Mirrors `labeler::chosen_tagged_nodes`'s class-level walk
/// (classes only — this binary needs no node-level tag tracking), rebuilt
/// here against the public API per this task's "computed inline"
/// instruction rather than reaching into a private helper.
fn reachable_canonical_classes(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> BTreeSet<EClassId> {
    let mut visited = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "counterfactual_credit: e-class {idx} is reachable from the chosen extraction \
                 but has no recorded choice — extractor/labeler invariant violated"
            )
        });
        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "counterfactual_credit: node_idx {node_idx} out of bounds ({}) for class {idx}",
            nodes.len()
        );
        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }
    visited
}

/// Build the replay context for one expression: run the UNGUIDED trajectory
/// to `budget`, extract, compute the three existing bounds, and index
/// everything the per-application scoring loop needs.
///
/// # Panics
///
impl ExprContext {
    /// The expression paired with the tables its leaves index.
    fn term(&self) -> Term<'_> {
        Term::new(self.expr.entry(), &self.env)
    }
}

/// Panics if the saturation call hits its own wall-clock safety ceiling
/// (`SATURATE_TIMEOUT`) — a correctness guard, not the outer sampling
/// ceiling (see module doc).
fn build_expr_context(
    name: String,
    expr: Rooted<ExprData>,
    budget: usize,
    proxies: &Proxies,
) -> ExprContext {
    // A corpus entry declares no buffers or uniforms — the format refuses to
    // write one down — so an empty environment is the right one.
    let env = Environment::new();
    let term = Term::new(expr.entry(), &env);
    let max_classes = config_for_node_count(expr.len()).max_classes;
    let costs = CostModel::latency_prior();
    // The replay reads the journal afterwards, so recording is asked for
    // with an observer (#1118). `hard_ceiling` panics on the ceiling rather
    // than reporting a stop reason, which is the fail-loud this used to get
    // from asserting on `SaturationStop::Timeout`.
    let mut optimizer = Optimizer::production()
        .cost(costs.clone())
        .observe(Some(Box::new(KeepJournal)))
        .budget(Budget::Explicit {
            iterations: SATURATE_MAX_ITERS,
            classes: max_classes,
            applications: Some(budget as u64),
        })
        .hard_ceiling(SATURATE_TIMEOUT);
    let mut egraph = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term,
        &mut egraph,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let _ = optimizer.run(&mut egraph, root_class, expr.len());

    let extraction = extract_dag(&egraph, root_class, &costs);
    // DAG cost — what the emitted kernel pays (#1117).
    let cost_original = extraction.dag_cost;

    let loose = EpisodeLabels::compute(&egraph, extraction.root, &extraction.choices);
    let tight = EpisodeLabels::compute_tight(&egraph, extraction.root, &extraction.choices);
    let strict = EpisodeLabels::compute_strict(&egraph, extraction.root, &extraction.choices);

    let mut created_by: HashMap<ApplicationId, Vec<ENodeId>> = HashMap::new();
    for (node, origin) in egraph.provenance().origins() {
        if let Origin::Rule(app_id) = origin {
            created_by.entry(app_id).or_default().push(node);
        }
    }
    let mut unions_by: HashMap<ApplicationId, Vec<(EClassId, EClassId)>> = HashMap::new();
    for event in egraph.provenance().union_events() {
        if let Some(app_id) = event.application_id {
            unions_by
                .entry(app_id)
                .or_default()
                .push((event.class_a, event.class_b));
        }
    }

    let mut node_to_class: HashMap<ENodeId, EClassId> = HashMap::new();
    for class in egraph.class_ids() {
        for &tag in egraph.tags(class) {
            node_to_class.insert(tag, class);
        }
    }

    let reachable = reachable_canonical_classes(&egraph, extraction.root, &extraction.choices);

    // State-changing recorded applications with ordinal < budget: the
    // no-op re-fires (idempotent matches) that dominate a sweep have
    // Δ = 0 by construction and are excluded from the sample per the
    // module doc.
    let mut candidates: Vec<(ApplicationId, RuleId)> = Vec::new();
    let bound = budget.min(egraph.application_count() as usize) as u64;
    let expr_node_count = expr.len();
    let mut step_of: HashMap<ApplicationId, usize> = HashMap::new();
    let mut keys: Vec<(ApplicationId, usize)> = Vec::new();
    let mut summaries: Vec<CandidateSummary> = Vec::new();
    for (app_id, record) in egraph.provenance().applications() {
        if app_id.as_u64() >= bound {
            continue;
        }
        let rule = record.rule.unwrap_or_else(|| {
            panic!(
                "counterfactual_credit: application {app_id:?} carries no RuleId — the graph \
                 was built without rule ids, and every table here is keyed by identity"
            )
        });
        if unions_by.contains_key(&app_id) {
            candidates.push((app_id, rule));
        }
        step_of.insert(app_id, record.step);
        let firing = Firing {
            rule,
            match_root: record.match_root,
            application_ordinal: app_id.as_u64(),
            registered_budget: REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
        };
        let features = CandidateFeatures::observe(&egraph, &firing);
        keys.push((app_id, record.step));
        summaries.push(CandidateSummary::new(
            &features,
            [0.0f32; EMBED_DIM],
            expr_node_count,
        ));
    }
    let scores_r2g = proxies.r2g.as_ref().map(|g| {
        // `score_candidates` is `-predicted_return`; the proxy wants `f`
        // itself (module doc), so score with the un-negated forward pass.
        struct Predicted<'a>(&'a LinearReturnGuide);
        impl SaturationGuide for Predicted<'_> {
            fn score_candidates(&self, c: &[CandidateSummary]) -> Vec<f32> {
                c.iter().map(|x| self.0.predicted_return(x)).collect()
            }
        }
        sweep_scores(&Predicted(g), &keys, &summaries)
    });
    let scores_strict_v1 = proxies
        .strict_v1
        .as_ref()
        .map(|g| sweep_scores(g, &keys, &summaries));
    let scores_per_rule = proxies
        .per_rule
        .as_ref()
        .map(|g| sweep_scores(g, &keys, &summaries));

    ExprContext {
        name,
        expr,
        env,
        max_classes,
        egraph,
        cost_original,
        loose,
        tight,
        strict,
        created_by,
        unions_by,
        node_to_class,
        reachable,
        candidates,
        step_of,
        scores_r2g,
        scores_strict_v1,
        scores_per_rule,
    }
}

/// The strict-by-output-class bound (task spec, verbatim): positive iff the
/// canonical class of `a`'s output — any node `a` created, or either side
/// of a union `a` caused — is a class the chosen extraction actually
/// visits. Credits e.g. `pythagorean`'s union into an already-existing `1`,
/// which `EpisodeLabels::compute_strict` cannot see (that node was never
/// *created* by `a`).
fn strict_by_output_class(ctx: &ExprContext, app_id: ApplicationId) -> bool {
    if let Some(nodes) = ctx.created_by.get(&app_id) {
        for &node in nodes {
            if let Some(&class) = ctx.node_to_class.get(&node)
                && ctx.reachable.contains(&ctx.egraph.find(class))
            {
                return true;
            }
        }
    }
    if let Some(pairs) = ctx.unions_by.get(&app_id) {
        for &(a, b) in pairs {
            if ctx.reachable.contains(&ctx.egraph.find(a))
                || ctx.reachable.contains(&ctx.egraph.find(b))
            {
                return true;
            }
        }
    }
    false
}

/// One masked replay's outcome: the extraction cost at `budget`, and how
/// many applications the mask actually skipped.
///
/// The skip count is returned rather than assumed because it is the whole
/// point of the second mask mode: under [`MaskScope::AllMatchingCandidate`]
/// a skip count of 1 means "nothing re-derived this candidate" and the two
/// modes must agree, while a count > 1 names exactly how many
/// re-derivations leave-one-out was silently allowing.
struct MaskedReplay {
    cost: usize,
    skips: usize,
}

/// Re-run the identical trajectory with `app_id` masked under `scope` and
/// return its extraction cost at `budget` — `cost(τ\a, B)` under
/// [`MaskScope::Single`], `cost(τ\[a], B)` (every re-derivation of `a`'s
/// candidate masked too) under [`MaskScope::AllMatchingCandidate`].
///
/// # Panics
///
/// Same per-run safety-ceiling contract as [`build_expr_context`].
fn masked_replay(
    ctx: &ExprContext,
    app_id: ApplicationId,
    budget: usize,
    scope: MaskScope,
) -> MaskedReplay {
    let mask = match scope {
        MaskScope::Single => ApplicationMask::leave_one_out(app_id.as_u64()),
        MaskScope::AllMatchingCandidate => ApplicationMask::all_matching_candidate(app_id.as_u64()),
    };
    // The mask is an optimizer field, not a second saturation entry point:
    // one loop, one budget denominator, one stop vocabulary.
    let costs = CostModel::latency_prior();
    let mut optimizer = Optimizer::production()
        .cost(costs.clone())
        .mask(Some(mask))
        .budget(Budget::Explicit {
            iterations: SATURATE_MAX_ITERS,
            classes: ctx.max_classes,
            applications: Some(budget as u64),
        })
        .hard_ceiling(SATURATE_TIMEOUT);
    let mut egraph = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        ctx.term(),
        &mut egraph,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let _ = optimizer.run(&mut egraph, root_class, ctx.expr.len());
    let skips = egraph.last_replay_mask_skips();
    assert!(
        skips >= 1,
        "counterfactual_credit: mask for '{}' ordinal {} (scope {scope:?}) never fired — the \
         replay diverged from the original trajectory before the seed ordinal, so its Δ would \
         be meaningless",
        ctx.name,
        app_id.as_u64()
    );
    MaskedReplay {
        // DAG cost — what the emitted kernel pays (#1117).
        cost: extract_dag(&egraph, root_class, &costs).dag_cost,
        skips,
    }
}

// ---------------------------------------------------------------------------
// One sampled application's row.
// ---------------------------------------------------------------------------

struct Row {
    expr_name: String,
    set: &'static str,
    application_ordinal: u64,
    /// The rule's stable identity ([`RuleId::get`]) — never a position.
    rule_id: u64,
    rule_name: String,
    cost_original: usize,
    cost_masked: usize,
    delta: f64,
    /// `cost(τ\[a], B)` under [`MaskScope::AllMatchingCandidate`]: the seed
    /// AND every re-derivation of its `(rule, canonical match)` masked.
    cost_masked_multi: usize,
    /// `ln(cost_masked_multi) − ln(cost_original)` — the confluence-aware Δ.
    delta_multi: f64,
    /// How many applications the multi-mask actually skipped (`1` means
    /// nothing re-derived this candidate, so the two Δs must agree).
    multi_mask_skips: usize,
    bit_loose: bool,
    bit_tight: bool,
    bit_strict: bool,
    bit_strict_class: bool,
    /// Same-sweep alternatives `|A_t|` (module doc); `0` means every model
    /// proxy below is `None` for this row.
    n_alternatives: usize,
    /// R2G predicted return `f(x_a)` (raw, for the per-rule table).
    f_r2g: Option<f64>,
    /// `mean_{b ∈ A_t} f(x_b) − f(x_a)`.
    adv_r2g: Option<f64>,
    /// `logit(x_a) − mean_{b ∈ A_t} logit(x_b)`.
    adv_strict_v1: Option<f64>,
    /// `rate(rule_a) − mean_{b ∈ A_t} rate(rule_b)`.
    adv_per_rule: Option<f64>,
}

// ---------------------------------------------------------------------------
// Correlation statistics.
// ---------------------------------------------------------------------------

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Pearson correlation coefficient. For a binary `xs` (0.0/1.0, our four
/// bounds), this IS the point-biserial correlation with `ys` by
/// construction — no separate formula needed. `NaN` when either series has
/// zero variance (reported as such, never silently coerced to 0).
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mx = mean(xs);
    let my = mean(ys);
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Average-rank transform (ties share the mean of their would-be ranks) —
/// the standard Spearman tie convention.
fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .expect("counterfactual_credit: NaN in rank")
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[order[j + 1]] == xs[order[i]] {
            j += 1;
        }
        // Ranks are 1-based; average the tied block's would-be ranks.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    pearson(&average_ranks(xs), &average_ranks(ys))
}

#[derive(Default)]
struct BoundStats {
    pearson_pooled: f64,
    spearman_pooled: f64,
    /// Seeded bootstrap 95% CI of `spearman_pooled` (module doc).
    spearman_pooled_ci: (f64, f64),
    /// Paired-bootstrap 95% CI of `spearman_pooled(this) −
    /// spearman_pooled(r2g)`; `NaN`s when no R2G proxy was loaded.
    spearman_diff_vs_r2g_ci: (f64, f64),
    pearson_sh: f64,
    spearman_sh: f64,
    pearson_dev: f64,
    spearman_dev: f64,
    n_pooled: usize,
    n_sh: usize,
    n_dev: usize,
    /// Rows this proxy could not score (`None`: no same-sweep alternative)
    /// — excluded from every statistic above, reported here.
    n_excluded: usize,
}

/// A proxy's value per row, or `None` where it is undefined for that row.
type ProxyOf<'a> = &'a dyn Fn(&Row) -> Option<f64>;

/// The same thing as a plain function pointer, for the hindsight-bound
/// proxies — which are field reads, not closures over anything.
type BoundProxy = fn(&Row) -> Option<f64>;

/// WHICH counterfactual is the ground truth for a correlation: the
/// leave-one-out Δ, or the confluence-aware multi-mask Δ. Passed explicitly
/// rather than defaulted, so no table can silently report one truth's
/// numbers under the other's heading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Truth {
    LeaveOneOut,
    MultiMask,
}

impl Truth {
    fn of(self, row: &Row) -> f64 {
        match self {
            Truth::LeaveOneOut => row.delta,
            Truth::MultiMask => row.delta_multi,
        }
    }
}

/// Spearman of `proxy` vs `truth`'s Δ over the rows (in `idx`) where the
/// proxy is defined.
fn spearman_over(rows: &[&Row], idx: &[usize], proxy: ProxyOf<'_>, truth: Truth) -> f64 {
    let mut xs = Vec::with_capacity(idx.len());
    let mut ys = Vec::with_capacity(idx.len());
    for &i in idx {
        if let Some(x) = proxy(rows[i]) {
            xs.push(x);
            ys.push(truth.of(rows[i]));
        }
    }
    spearman(&xs, &ys)
}

fn pearson_over(rows: &[&Row], idx: &[usize], proxy: ProxyOf<'_>, truth: Truth) -> f64 {
    let mut xs = Vec::with_capacity(idx.len());
    let mut ys = Vec::with_capacity(idx.len());
    for &i in idx {
        if let Some(x) = proxy(rows[i]) {
            xs.push(x);
            ys.push(truth.of(rows[i]));
        }
    }
    pearson(&xs, &ys)
}

/// 2.5 / 97.5 percentiles of a bootstrap sample, `NaN`s ignored (a resample
/// with zero proxy variance yields `NaN`; if every resample does, the CI
/// is `(NaN, NaN)` — reported as `null`, never coerced).
fn ci95(mut xs: Vec<f64>) -> (f64, f64) {
    xs.retain(|x| !x.is_nan());
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("ci95: NaN survived retain"));
    let at = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    (at(0.025), at(0.975))
}

/// Score one proxy against Δ: pooled / per-set correlations, plus the
/// paired-bootstrap CIs. `resamples` is the shared list of bootstrap index
/// draws (the SAME draws for every proxy — that is what makes the
/// difference CI paired); `r2g_boot` is the R2G proxy's Spearman on each
/// draw, `None` when no R2G proxy was loaded.
fn score_proxy(
    rows: &[&Row],
    proxy: ProxyOf<'_>,
    resamples: &[Vec<usize>],
    r2g_boot: Option<&[f64]>,
    truth: Truth,
) -> BoundStats {
    let all: Vec<usize> = (0..rows.len()).collect();
    let sh_idx: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&i| rows[i].set == "sh")
        .collect();
    let dev_idx: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&i| rows[i].set == "dev")
        .collect();
    let defined = |idx: &[usize]| idx.iter().filter(|&&i| proxy(rows[i]).is_some()).count();

    let boot: Vec<f64> = resamples
        .iter()
        .map(|idx| spearman_over(rows, idx, proxy, truth))
        .collect();
    let diff_ci = match r2g_boot {
        Some(r) => {
            assert_eq!(r.len(), boot.len());
            ci95(boot.iter().zip(r).map(|(b, r)| b - r).collect())
        }
        None => (f64::NAN, f64::NAN),
    };

    BoundStats {
        pearson_pooled: pearson_over(rows, &all, proxy, truth),
        spearman_pooled: spearman_over(rows, &all, proxy, truth),
        spearman_pooled_ci: ci95(boot),
        spearman_diff_vs_r2g_ci: diff_ci,
        pearson_sh: pearson_over(rows, &sh_idx, proxy, truth),
        spearman_sh: spearman_over(rows, &sh_idx, proxy, truth),
        pearson_dev: pearson_over(rows, &dev_idx, proxy, truth),
        spearman_dev: spearman_over(rows, &dev_idx, proxy, truth),
        n_pooled: defined(&all),
        n_sh: defined(&sh_idx),
        n_dev: defined(&dev_idx),
        n_excluded: rows.len() - defined(&all),
    }
}

fn bit(b: bool) -> Option<f64> {
    Some(if b { 1.0 } else { 0.0 })
}

/// Per-rule view of the sampled applications: does the rule pay by Δ, and
/// what does the R2G model think of it? (The `pythagorean` question.)
struct RuleDelta {
    /// The rule's stable identity ([`RuleId::get`]) — never a position.
    rule_id: u64,
    rule_name: String,
    n: usize,
    n_sh: usize,
    delta_mean: f64,
    n_delta_positive: usize,
    n_delta_negative: usize,
    f_r2g_mean: f64,
    adv_r2g_mean: f64,
}

fn per_rule_delta(rows: &[Row]) -> Vec<RuleDelta> {
    let mut by_rule: std::collections::BTreeMap<u64, Vec<&Row>> = std::collections::BTreeMap::new();
    for r in rows {
        by_rule.entry(r.rule_id).or_default().push(r);
    }
    let mean_opt = |xs: Vec<f64>| if xs.is_empty() { f64::NAN } else { mean(&xs) };
    by_rule
        .into_iter()
        .map(|(rule_id, rs)| RuleDelta {
            rule_id,
            rule_name: rs[0].rule_name.clone(),
            n: rs.len(),
            n_sh: rs.iter().filter(|r| r.set == "sh").count(),
            delta_mean: mean(&rs.iter().map(|r| r.delta).collect::<Vec<_>>()),
            n_delta_positive: rs.iter().filter(|r| r.delta > 0.0).count(),
            n_delta_negative: rs.iter().filter(|r| r.delta < 0.0).count(),
            f_r2g_mean: mean_opt(rs.iter().filter_map(|r| r.f_r2g).collect()),
            adv_r2g_mean: mean_opt(rs.iter().filter_map(|r| r.adv_r2g).collect()),
        })
        .collect()
}

fn json_num(x: f64) -> String {
    if x.is_nan() {
        "null".to_string()
    } else {
        format!("{x:.6}")
    }
}

fn write_bound_json(out: &mut String, name: &str, s: &BoundStats, trailing_comma: bool) {
    out.push_str(&format!(
        "    {name:?}: {{\"pearson_pooled\": {}, \"spearman_pooled\": {}, \
         \"spearman_pooled_ci95\": [{}, {}], \"spearman_diff_vs_r2g_ci95\": [{}, {}], \
         \"pearson_sh\": {}, \"spearman_sh\": {}, \"pearson_dev\": {}, \"spearman_dev\": {}, \
         \"n_pooled\": {}, \"n_sh\": {}, \"n_dev\": {}, \"n_excluded\": {}}}{}\n",
        json_num(s.pearson_pooled),
        json_num(s.spearman_pooled),
        json_num(s.spearman_pooled_ci.0),
        json_num(s.spearman_pooled_ci.1),
        json_num(s.spearman_diff_vs_r2g_ci.0),
        json_num(s.spearman_diff_vs_r2g_ci.1),
        json_num(s.pearson_sh),
        json_num(s.spearman_sh),
        json_num(s.pearson_dev),
        json_num(s.spearman_dev),
        s.n_pooled,
        s.n_sh,
        s.n_dev,
        s.n_excluded,
        if trailing_comma { "," } else { "" },
    ));
}

/// A JSON string or `null` — `{:?}` on an `Option<String>` would print
/// `Some("…")`, which is not JSON.
fn opt_str_json(x: &Option<String>) -> String {
    match x {
        Some(v) => format!("{v:?}"),
        None => "null".to_string(),
    }
}

fn opt_json(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{v:.6}"),
        None => "null".to_string(),
    }
}

fn main() {
    let args = Args::parse();
    let start = Instant::now();
    let ceiling = Duration::from_secs(args.wall_clock_ceiling_secs);
    let load_start = uptime();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let dev_path = corpus_dir.join("corpus_dev.bin");
    let ood_path = corpus_dir.join("corpus_dev_ood.bin");

    let dev_entries = read_corpus(&dev_path).unwrap_or_else(|e| {
        panic!(
            "counterfactual_credit: failed to read {}: {e}",
            dev_path.display()
        )
    });
    let ood_entries = read_corpus(&ood_path).unwrap_or_else(|e| {
        panic!(
            "counterfactual_credit: failed to read {}: {e}",
            ood_path.display()
        )
    });
    let sh_entries: Vec<(String, Rooted<ExprData>)> = ood_entries
        .into_iter()
        .filter(|(name, _)| name.starts_with("dev_sh_"))
        .collect();

    let rules = RuleSet::production();
    let proxies = Proxies {
        r2g: args.r2g_checkpoint.as_ref().map(|p| {
            load_return_guide(Path::new(p), &rules)
                .unwrap_or_else(|e| panic!("counterfactual_credit: --r2g-checkpoint: {e}"))
        }),
        strict_v1: args.strict_checkpoint.as_ref().map(|p| {
            load_linear_guide(Path::new(p), &rules)
                .unwrap_or_else(|e| panic!("counterfactual_credit: --strict-checkpoint: {e}"))
        }),
        per_rule: args.train_guide_report.as_ref().map(|p| {
            per_rule_rate_guide_from_report(Path::new(p))
                .unwrap_or_else(|e| panic!("counterfactual_credit: --train-guide-report: {e}"))
        }),
    };

    let in_band = |e: &Rooted<ExprData>| {
        let n = e.len();
        (args.min_expr_nodes == 0 || n >= args.min_expr_nodes)
            && (args.max_expr_nodes == 0 || n <= args.max_expr_nodes)
    };
    let dev_in_band: Vec<_> = dev_entries
        .into_iter()
        .filter(|(_, e)| in_band(e))
        .collect();
    let sh_in_band: Vec<_> = sh_entries.into_iter().filter(|(_, e)| in_band(e)).collect();
    let dev_available = dev_in_band.len();
    let sh_available = sh_in_band.len();

    let mut select_rng = SeededRng::new(args.seed);
    let dev_sample = seeded_select(dev_in_band, args.n_expr_dev, &mut select_rng);
    let sh_sample = seeded_select(sh_in_band, args.n_expr_sh, &mut select_rng);

    assert!(
        dev_sample.len() >= args.n_expr_dev.min(dev_available),
        "counterfactual_credit: corpus_dev.bin has {dev_available} entries in the requested \
         node band, fewer than the {} requested",
        args.n_expr_dev
    );
    assert!(
        !sh_sample.is_empty(),
        "counterfactual_credit: no dev_sh_* entries found in corpus_dev_ood.bin"
    );

    eprintln!(
        "counterfactual_credit: node band [{}, {}] leaves {sh_available} sh / {dev_available} \
         DEV entries; sampling {} sh + {} DEV classical expressions, \
         target {} applications each, budget B={}, wall-clock ceiling {:?}",
        args.min_expr_nodes,
        args.max_expr_nodes,
        sh_sample.len(),
        dev_sample.len(),
        args.n_apps,
        args.budget,
        ceiling
    );

    let out_jsonl_path = PathBuf::from(&args.out_jsonl);
    if let Some(parent) = out_jsonl_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&out_jsonl_path).unwrap_or_else(|e| {
            panic!(
                "counterfactual_credit: cannot create {}: {e}",
                out_jsonl_path.display()
            )
        }));

    let mut apply_rng = SeededRng::new(args.seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut rows: Vec<Row> = Vec::new();
    let mut expressions_processed = 0usize;
    let mut insufficient_candidates: Vec<String> = Vec::new();
    let mut ceiling_hit = false;

    'expressions: for (set, sample) in [("sh", &sh_sample), ("dev", &dev_sample)] {
        for (name, expr) in sample.iter() {
            if start.elapsed() >= ceiling {
                ceiling_hit = true;
                eprintln!(
                    "counterfactual_credit: WALL-CLOCK CEILING ({ceiling:?}) HIT after {} \
                     expressions / {} applications — stopping the sampling loop now, loudly, \
                     with the count achieved (not a silent partial)",
                    expressions_processed,
                    rows.len()
                );
                break 'expressions;
            }

            let ctx = build_expr_context(name.clone(), expr.clone(), args.budget, &proxies);
            expressions_processed += 1;

            if ctx.candidates.len() < args.n_apps {
                insufficient_candidates.push(format!(
                    "{name} ({} state-changing applications < {} target)",
                    ctx.candidates.len(),
                    args.n_apps
                ));
            }

            let mut pool = ctx.candidates.clone();
            let sampled: Vec<(ApplicationId, RuleId)> = if pool.len() <= args.n_apps {
                pool
            } else {
                apply_rng.shuffle(&mut pool);
                pool.truncate(args.n_apps);
                pool
            };

            for (app_id, rule) in sampled {
                if start.elapsed() >= ceiling {
                    ceiling_hit = true;
                    eprintln!(
                        "counterfactual_credit: WALL-CLOCK CEILING ({ceiling:?}) HIT mid-expression \
                         '{name}' after {} applications total — stopping now, loudly, with the \
                         count achieved (not a silent partial)",
                        rows.len()
                    );
                    break 'expressions;
                }

                let single = masked_replay(&ctx, app_id, args.budget, MaskScope::Single);
                let multi =
                    masked_replay(&ctx, app_id, args.budget, MaskScope::AllMatchingCandidate);
                let cost_masked = single.cost;
                let cost_masked_multi = multi.cost;
                let log_delta = |masked: usize, label: &str| -> f64 {
                    if ctx.cost_original == 0 || masked == 0 {
                        eprintln!(
                            "counterfactual_credit: WARNING zero extraction cost for '{name}' \
                             (original={}, {label}={masked}) at ordinal {} — recording delta=0.0 \
                             rather than an undefined ln(0)",
                            ctx.cost_original,
                            app_id.as_u64()
                        );
                        0.0
                    } else {
                        (masked as f64).ln() - (ctx.cost_original as f64).ln()
                    }
                };
                let delta = log_delta(cost_masked, "masked");
                let delta_multi = log_delta(cost_masked_multi, "masked_multi");
                // A multi-mask that skipped exactly the seed masked exactly
                // what leave-one-out masked, so the two replays are the same
                // run and must agree bit-for-bit. Disagreement there would
                // mean the two mask paths diverge for reasons unrelated to
                // confluence — a bug, not a finding.
                assert!(
                    multi.skips > 1 || cost_masked_multi == cost_masked,
                    "counterfactual_credit: '{name}' ordinal {} — the multi-mask skipped only \
                     the seed ({} skips) yet produced cost {cost_masked_multi} against \
                     leave-one-out's {cost_masked}; identical masks must give identical replays",
                    app_id.as_u64(),
                    multi.skips
                );

                let bit_loose = ctx.loose.labels.get(&app_id) == Some(&Label::LoadBearing);
                let bit_tight = ctx.tight.labels.get(&app_id) == Some(&Label::LoadBearing);
                let bit_strict = ctx.strict.labels.get(&app_id) == Some(&Label::LoadBearing);
                let bit_strict_class = strict_by_output_class(&ctx, app_id);

                let step = *ctx.step_of.get(&app_id).unwrap_or_else(|| {
                    panic!(
                        "counterfactual_credit: sampled application {} has no recorded sweep",
                        app_id.as_u64()
                    )
                });
                // Every proxy shares the same alternative set, so
                // `n_alternatives` is one number per row (taken from
                // whichever proxy is loaded; all agree by construction).
                let n_alternatives = ctx
                    .scores_r2g
                    .as_ref()
                    .or(ctx.scores_strict_v1.as_ref())
                    .or(ctx.scores_per_rule.as_ref())
                    .map_or(0, |s| s.per_step[&step].1 - 1);
                let (f_r2g, adv_r2g) = match ctx.scores_r2g.as_ref() {
                    Some(s) => match s.own_and_others(app_id, step) {
                        Some((own, others)) => (Some(own), Some(others - own)),
                        None => (Some(s.own[&app_id]), None),
                    },
                    None => (None, None),
                };
                let adv_strict_v1 = ctx
                    .scores_strict_v1
                    .as_ref()
                    .and_then(|s| s.own_and_others(app_id, step))
                    .map(|(own, others)| own - others);
                let adv_per_rule = ctx
                    .scores_per_rule
                    .as_ref()
                    .and_then(|s| s.own_and_others(app_id, step))
                    .map(|(own, others)| own - others);

                let row = Row {
                    expr_name: name.clone(),
                    set,
                    application_ordinal: app_id.as_u64(),
                    rule_id: rule.get(),
                    rule_name: rules
                        .index_of(rule)
                        .and_then(|i| rules.label_of(i))
                        .unwrap_or_else(|| format!("<rule {}>", rule.get())),
                    cost_original: ctx.cost_original,
                    cost_masked,
                    delta,
                    cost_masked_multi,
                    delta_multi,
                    multi_mask_skips: multi.skips,
                    bit_loose,
                    bit_tight,
                    bit_strict,
                    bit_strict_class,
                    n_alternatives,
                    f_r2g,
                    adv_r2g,
                    adv_strict_v1,
                    adv_per_rule,
                };

                writeln!(
                    out,
                    "{{\"expr_name\":{:?},\"set\":{:?},\"application_ordinal\":{},\
                     \"rule_id\":{},\"rule_name\":{:?},\"cost_original\":{},\
                     \"cost_masked\":{},\"delta\":{:.6},\"cost_masked_multi\":{},\
                     \"delta_multi\":{:.6},\"multi_mask_skips\":{},\"bit_loose\":{},\"bit_tight\":{},\
                     \"bit_strict\":{},\"bit_strict_by_output_class\":{},\
                     \"n_alternatives\":{},\"f_r2g\":{},\"adv_r2g\":{},\"adv_strict_v1\":{},\
                     \"adv_per_rule\":{}}}",
                    row.expr_name,
                    row.set,
                    row.application_ordinal,
                    row.rule_id,
                    row.rule_name,
                    row.cost_original,
                    row.cost_masked,
                    row.delta,
                    row.cost_masked_multi,
                    row.delta_multi,
                    row.multi_mask_skips,
                    row.bit_loose,
                    row.bit_tight,
                    row.bit_strict,
                    row.bit_strict_class,
                    row.n_alternatives,
                    opt_json(row.f_r2g),
                    opt_json(row.adv_r2g),
                    opt_json(row.adv_strict_v1),
                    opt_json(row.adv_per_rule),
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "counterfactual_credit: write to {}: {e}",
                        out_jsonl_path.display()
                    )
                });

                rows.push(row);
            }

            if expressions_processed.is_multiple_of(10) {
                eprintln!(
                    "counterfactual_credit: {expressions_processed} expressions processed, \
                     {} applications sampled so far ({:?} elapsed)",
                    rows.len(),
                    start.elapsed()
                );
            }
        }
    }
    out.flush().unwrap_or_else(|e| {
        panic!(
            "counterfactual_credit: flushing {}: {e}",
            out_jsonl_path.display()
        )
    });

    assert!(
        !rows.is_empty(),
        "counterfactual_credit: sampled zero applications — nothing to score \
         (wall-clock ceiling hit immediately, or every expression had zero \
         state-changing applications)"
    );

    // ---- Δ distribution ----
    let n_zero = rows.iter().filter(|r| r.delta == 0.0).count();
    let n_positive = rows.iter().filter(|r| r.delta > 0.0).count();
    let n_negative = rows.iter().filter(|r| r.delta < 0.0).count();
    let n = rows.len();

    // ---- per-proxy correlations (paired bootstrap: one shared set of draws) ----
    let row_refs: Vec<&Row> = rows.iter().collect();
    let mut boot_rng = SeededRng::new(args.seed ^ 0xB007_B007_B007_B007);
    let resamples: Vec<Vec<usize>> = (0..args.bootstrap_resamples)
        .map(|_| {
            (0..n)
                .map(|_| (boot_rng.next_u64() % n as u64) as usize)
                .collect()
        })
        .collect();
    let r2g_proxy: ProxyOf<'_> = &|r: &Row| r.adv_r2g;
    // One table per ground truth (§4: leave-one-out, and the
    // confluence-aware multi-mask). Built by the same closure so the two
    // tables cannot drift into different proxy sets or different bootstrap
    // draws — only `truth` differs.
    let build_table = |truth: Truth| -> Vec<(&'static str, BoundStats)> {
        let r2g_boot: Option<Vec<f64>> = proxies.r2g.as_ref().map(|_| {
            resamples
                .iter()
                .map(|idx| spearman_over(&row_refs, idx, r2g_proxy, truth))
                .collect()
        });
        let boot = r2g_boot.as_deref();
        let mut table: Vec<(&'static str, BoundStats)> = Vec::new();
        if proxies.r2g.is_some() {
            table.push((
                "r2g_linear",
                score_proxy(&row_refs, r2g_proxy, &resamples, boot, truth),
            ));
        }
        if proxies.strict_v1.is_some() {
            table.push((
                "strict_v1_linear",
                score_proxy(
                    &row_refs,
                    &|r: &Row| r.adv_strict_v1,
                    &resamples,
                    boot,
                    truth,
                ),
            ));
        }
        if proxies.per_rule.is_some() {
            table.push((
                "per_rule_rate",
                score_proxy(
                    &row_refs,
                    &|r: &Row| r.adv_per_rule,
                    &resamples,
                    boot,
                    truth,
                ),
            ));
        }
        let bounds: [(&'static str, BoundProxy); 4] = [
            ("loose", |r| bit(r.bit_loose)),
            ("tight", |r| bit(r.bit_tight)),
            ("strict", |r| bit(r.bit_strict)),
            ("strict_by_output_class", |r| bit(r.bit_strict_class)),
        ];
        for (name, f) in bounds {
            table.push((name, score_proxy(&row_refs, &f, &resamples, boot, truth)));
        }
        table
    };
    let proxy_table = build_table(Truth::LeaveOneOut);
    let proxy_table_multi = build_table(Truth::MultiMask);

    // ---- confluence blindness: what the multi-mask sees that LOO does not ----
    let n_zero_multi = rows.iter().filter(|r| r.delta_multi == 0.0).count();
    let n_positive_multi = rows.iter().filter(|r| r.delta_multi > 0.0).count();
    let n_negative_multi = rows.iter().filter(|r| r.delta_multi < 0.0).count();
    // THE number: applications leave-one-out called irrelevant that the
    // confluence-aware mask shows were load-bearing all along.
    let n_zero_to_positive = rows
        .iter()
        .filter(|r| r.delta == 0.0 && r.delta_multi > 0.0)
        .count();
    let n_zero_to_negative = rows
        .iter()
        .filter(|r| r.delta == 0.0 && r.delta_multi < 0.0)
        .count();
    let n_multi_skips_gt1 = rows.iter().filter(|r| r.multi_mask_skips > 1).count();
    let mean_multi_skips = mean(
        &rows
            .iter()
            .map(|r| r.multi_mask_skips as f64)
            .collect::<Vec<_>>(),
    );
    let max_multi_skips = rows.iter().map(|r| r.multi_mask_skips).max().unwrap_or(0);

    let rule_deltas = per_rule_delta(&rows);

    println!(
        "=== counterfactual_credit: {n} applications sampled ({expressions_processed} \
         expressions, wall_clock_ceiling_hit={ceiling_hit}) ==="
    );
    println!(
        "  delta distribution: zero {n_zero}/{n} ({:.1}%), positive {n_positive}/{n} ({:.1}%), \
         negative {n_negative}/{n} ({:.1}%)",
        100.0 * n_zero as f64 / n as f64,
        100.0 * n_positive as f64 / n as f64,
        100.0 * n_negative as f64 / n as f64,
    );
    println!(
        "  {:<24} {:>10} {:>10} {:>18} {:>8} {:>8} {:>8} {:>6}",
        "proxy", "pearson", "spearman", "spearman ci95", "n_pool", "n_sh", "n_dev", "excl"
    );
    for (name, s) in &proxy_table {
        println!(
            "  {name:<24} {:>10.4} {:>10.4} [{:>7.4}, {:>7.4}] {:>8} {:>8} {:>8} {:>6}",
            s.pearson_pooled,
            s.spearman_pooled,
            s.spearman_pooled_ci.0,
            s.spearman_pooled_ci.1,
            s.n_pooled,
            s.n_sh,
            s.n_dev,
            s.n_excluded
        );
    }
    println!(
        "  MULTI-MASK delta distribution: zero {n_zero_multi}/{n} ({:.1}%), positive \
         {n_positive_multi}/{n} ({:.1}%), negative {n_negative_multi}/{n} ({:.1}%)",
        100.0 * n_zero_multi as f64 / n as f64,
        100.0 * n_positive_multi as f64 / n as f64,
        100.0 * n_negative_multi as f64 / n as f64,
    );
    println!(
        "  confluence blindness: {n_zero_to_positive}/{n_zero} of leave-one-out's zero-delta \
         applications become delta>0 under the multi-mask ({n_zero_to_negative} become \
         delta<0); {n_multi_skips_gt1}/{n} multi-masks skipped >1 application (mean \
         {mean_multi_skips:.2}, max {max_multi_skips})"
    );
    println!(
        "  {:<24} {:>10} {:>10} {:>18} {:>8} {:>8} {:>8} {:>6}   [vs MULTI-MASK delta]",
        "proxy", "pearson", "spearman", "spearman ci95", "n_pool", "n_sh", "n_dev", "excl"
    );
    for (name, s) in &proxy_table_multi {
        println!(
            "  {name:<24} {:>10.4} {:>10.4} [{:>7.4}, {:>7.4}] {:>8} {:>8} {:>8} {:>6}",
            s.pearson_pooled,
            s.spearman_pooled,
            s.spearman_pooled_ci.0,
            s.spearman_pooled_ci.1,
            s.n_pooled,
            s.n_sh,
            s.n_dev,
            s.n_excluded
        );
    }
    if !insufficient_candidates.is_empty() {
        println!(
            "  {} expressions had fewer than {} state-changing applications (used all available):",
            insufficient_candidates.len(),
            args.n_apps
        );
        for line in &insufficient_candidates {
            println!("    {line}");
        }
    }

    // ---- JSON report ----
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"seed\": {},\n", args.seed));
    json.push_str(&format!("  \"budget\": {},\n", args.budget));
    json.push_str(&format!("  \"n_apps_target\": {},\n", args.n_apps));
    json.push_str(&format!("  \"min_expr_nodes\": {},\n", args.min_expr_nodes));
    json.push_str(&format!("  \"max_expr_nodes\": {},\n", args.max_expr_nodes));
    json.push_str(&format!("  \"sh_entries_in_band\": {sh_available},\n"));
    json.push_str(&format!("  \"dev_entries_in_band\": {dev_available},\n"));
    json.push_str(&format!("  \"n_expr_sh\": {},\n", sh_sample.len()));
    json.push_str(&format!("  \"n_expr_dev\": {},\n", dev_sample.len()));
    json.push_str(&format!(
        "  \"expressions_processed\": {expressions_processed},\n"
    ));
    json.push_str(&format!("  \"applications_sampled\": {n},\n"));
    json.push_str(&format!(
        "  \"wall_clock_ceiling_secs\": {},\n",
        args.wall_clock_ceiling_secs
    ));
    json.push_str(&format!("  \"wall_clock_ceiling_hit\": {ceiling_hit},\n"));
    json.push_str(&format!(
        "  \"elapsed_secs\": {:.1},\n",
        start.elapsed().as_secs_f64()
    ));
    json.push_str(&format!(
        "  \"insufficient_candidates_count\": {},\n",
        insufficient_candidates.len()
    ));
    json.push_str("  \"insufficient_candidates\": [\n");
    for (i, line) in insufficient_candidates.iter().enumerate() {
        json.push_str(&format!(
            "    {line:?}{}\n",
            if i + 1 < insufficient_candidates.len() {
                ","
            } else {
                ""
            }
        ));
    }
    json.push_str("  ],\n");
    json.push_str("  \"delta_distribution\": {\n");
    json.push_str(&format!("    \"n\": {n},\n"));
    json.push_str(&format!(
        "    \"zero\": {n_zero}, \"positive\": {n_positive}, \"negative\": {n_negative},\n"
    ));
    json.push_str(&format!(
        "    \"share_zero\": {:.6}, \"share_positive\": {:.6}, \"share_negative\": {:.6}\n",
        n_zero as f64 / n as f64,
        n_positive as f64 / n as f64,
        n_negative as f64 / n as f64,
    ));
    json.push_str("  },\n");
    json.push_str("  \"delta_multi_distribution\": {\n");
    json.push_str(&format!("    \"n\": {n},\n"));
    json.push_str(&format!(
        "    \"zero\": {n_zero_multi}, \"positive\": {n_positive_multi}, \"negative\": {n_negative_multi},\n"
    ));
    json.push_str(&format!(
        "    \"share_zero\": {:.6}, \"share_positive\": {:.6}, \"share_negative\": {:.6}\n",
        n_zero_multi as f64 / n as f64,
        n_positive_multi as f64 / n as f64,
        n_negative_multi as f64 / n as f64,
    ));
    json.push_str("  },\n");
    json.push_str("  \"confluence_blindness\": {\n");
    json.push_str(&format!("    \"n_leave_one_out_zero\": {n_zero},\n"));
    json.push_str(&format!(
        "    \"n_zero_to_positive\": {n_zero_to_positive},\n"
    ));
    json.push_str(&format!(
        "    \"n_zero_to_negative\": {n_zero_to_negative},\n"
    ));
    json.push_str(&format!(
        "    \"share_of_zero_that_becomes_positive\": {:.6},\n",
        if n_zero == 0 {
            0.0
        } else {
            n_zero_to_positive as f64 / n_zero as f64
        }
    ));
    json.push_str(&format!(
        "    \"n_multi_mask_skips_gt_1\": {n_multi_skips_gt1},\n"
    ));
    json.push_str(&format!(
        "    \"mean_multi_mask_skips\": {mean_multi_skips:.4},\n"
    ));
    json.push_str(&format!(
        "    \"max_multi_mask_skips\": {max_multi_skips}\n"
    ));
    json.push_str("  },\n");
    json.push_str(&format!(
        "  \"proxies_loaded\": {{\"r2g_checkpoint\": {}, \"strict_checkpoint\": {}, \"train_guide_report\": {}}},\n",
        opt_str_json(&args.r2g_checkpoint),
        opt_str_json(&args.strict_checkpoint),
        opt_str_json(&args.train_guide_report)
    ));
    json.push_str(&format!(
        "  \"bootstrap_resamples\": {},\n",
        args.bootstrap_resamples
    ));
    json.push_str("  \"bounds\": {\n");
    for (i, (name, s)) in proxy_table.iter().enumerate() {
        write_bound_json(&mut json, name, s, i + 1 < proxy_table.len());
    }
    json.push_str("  },\n");
    json.push_str("  \"bounds_vs_multi_mask\": {\n");
    for (i, (name, s)) in proxy_table_multi.iter().enumerate() {
        write_bound_json(&mut json, name, s, i + 1 < proxy_table_multi.len());
    }
    json.push_str("  },\n");
    json.push_str("  \"per_rule_delta\": [\n");
    for (i, rd) in rule_deltas.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"rule_id\": {}, \"rule_name\": {:?}, \"n\": {}, \"n_sh\": {}, \"delta_mean\": {}, \"n_delta_positive\": {}, \"n_delta_negative\": {}, \"f_r2g_mean\": {}, \"adv_r2g_mean\": {}}}{}\n",
            rd.rule_id, rd.rule_name, rd.n, rd.n_sh, json_num(rd.delta_mean), rd.n_delta_positive,
            rd.n_delta_negative, json_num(rd.f_r2g_mean), json_num(rd.adv_r2g_mean),
            if i + 1 < rule_deltas.len() { "," } else { "" }
        ));
    }
    json.push_str("  ],\n");
    json.push_str(&format!("  \"git_rev\": {:?},\n", git_rev()));
    json.push_str(&format!("  \"load_at_start\": {:?},\n", load_start));
    json.push_str(&format!("  \"load_at_end\": {:?}\n", uptime()));
    json.push_str("}\n");

    let out_json_path = PathBuf::from(&args.out_json);
    if let Some(parent) = out_json_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out_json_path, &json).unwrap_or_else(|e| {
        panic!(
            "counterfactual_credit: cannot write {}: {e}",
            out_json_path.display()
        )
    });
    eprintln!("counterfactual_credit: wrote {}", out_json_path.display());

    // ---- Markdown summary ----
    let mut md = String::new();
    md.push_str("# Counterfactual credit: hindsight bounds vs measured Δ (leave-one-out and confluence-aware)\n\n");
    md.push_str(&format!(
        "Sample: {} `sh` + {} DEV classical expressions, {} applications sampled \
         ({} at ordinal < B={}, seed {:#x}). wall_clock_ceiling_hit = {ceiling_hit}. \
         git {}.\n\n",
        sh_sample.len(),
        dev_sample.len(),
        n,
        n,
        args.budget,
        args.seed,
        git_rev(),
    ));
    md.push_str(&format!(
        "Δ distribution: zero {n_zero}/{n} ({:.1}%), positive {n_positive}/{n} ({:.1}%), \
         negative {n_negative}/{n} ({:.1}%).\n\n",
        100.0 * n_zero as f64 / n as f64,
        100.0 * n_positive as f64 / n as f64,
        100.0 * n_negative as f64 / n as f64,
    ));
    md.push_str(&format!(
        "Proxies loaded: r2g = {}, strict-v1 = {}, per-rule = {}. Bootstrap: {} paired resamples (seeded).\n\n",
        opt_str_json(&args.r2g_checkpoint),
        opt_str_json(&args.strict_checkpoint),
        opt_str_json(&args.train_guide_report),
        args.bootstrap_resamples
    ));
    md.push_str("| proxy | Spearman (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for (name, s) in &proxy_table {
        md.push_str(&format!(
            "| {name} | {} [{}, {}] | [{}, {}] | {} | {} | {} | {} ({}) |\n",
            json_num(s.spearman_pooled),
            json_num(s.spearman_pooled_ci.0),
            json_num(s.spearman_pooled_ci.1),
            json_num(s.spearman_diff_vs_r2g_ci.0),
            json_num(s.spearman_diff_vs_r2g_ci.1),
            json_num(s.spearman_sh),
            json_num(s.spearman_dev),
            json_num(s.pearson_pooled),
            s.n_pooled,
            s.n_excluded,
        ));
    }
    md.push_str("\n## Confluence-aware credit (multi-mask)\n\n");
    md.push_str(&format!(
        "Second mask mode: the seed application AND every later application sharing its \
         `(rule_idx, canonical matched-class content)` are skipped, so an alternative \
         re-derivation cannot silently restore the node leave-one-out removed.\n\n\
         Multi-mask Δ distribution: zero {n_zero_multi}/{n} ({:.1}%), positive \
         {n_positive_multi}/{n} ({:.1}%), negative {n_negative_multi}/{n} ({:.1}%).\n\n\
         **Confluence blindness of leave-one-out: {n_zero_to_positive} of the {n_zero} \
         applications leave-one-out scored Δ = 0 become Δ > 0 under the multi-mask ({:.1}% of \
         them)**; {n_zero_to_negative} become Δ < 0. Multi-masks that skipped more than the \
         seed: {n_multi_skips_gt1}/{n} (mean skips {mean_multi_skips:.2}, max \
         {max_multi_skips}).\n\n",
        100.0 * n_zero_multi as f64 / n as f64,
        100.0 * n_positive_multi as f64 / n as f64,
        100.0 * n_negative_multi as f64 / n as f64,
        if n_zero == 0 {
            0.0
        } else {
            100.0 * n_zero_to_positive as f64 / n_zero as f64
        },
    ));
    md.push_str("| proxy | Spearman vs multi-mask Δ (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for (name, s) in &proxy_table_multi {
        md.push_str(&format!(
            "| {name} | {} [{}, {}] | [{}, {}] | {} | {} | {} | {} ({}) |\n",
            json_num(s.spearman_pooled),
            json_num(s.spearman_pooled_ci.0),
            json_num(s.spearman_pooled_ci.1),
            json_num(s.spearman_diff_vs_r2g_ci.0),
            json_num(s.spearman_diff_vs_r2g_ci.1),
            json_num(s.spearman_sh),
            json_num(s.spearman_dev),
            json_num(s.pearson_pooled),
            s.n_pooled,
            s.n_excluded,
        ));
    }
    md.push_str("\n## Per-rule Δ over the sampled applications\n\n");
    md.push_str("| idx | rule | n (sh) | mean Δ | Δ>0 | Δ<0 | mean f_r2g | mean adv_r2g |\n|---:|---|---:|---:|---:|---:|---:|---:|\n");
    for rd in &rule_deltas {
        md.push_str(&format!(
            "| {} | {} | {} ({}) | {} | {} | {} | {} | {} |\n",
            rd.rule_id,
            rd.rule_name,
            rd.n,
            rd.n_sh,
            json_num(rd.delta_mean),
            rd.n_delta_positive,
            rd.n_delta_negative,
            json_num(rd.f_r2g_mean),
            json_num(rd.adv_r2g_mean)
        ));
    }
    if !insufficient_candidates.is_empty() {
        md.push_str(&format!(
            "\n{} expressions had fewer than {} state-changing applications and contributed \
             all available instead:\n\n",
            insufficient_candidates.len(),
            args.n_apps
        ));
        for line in &insufficient_candidates {
            md.push_str(&format!("- {line}\n"));
        }
    }

    let out_md_path = PathBuf::from(&args.out_md);
    if let Some(parent) = out_md_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out_md_path, &md).unwrap_or_else(|e| {
        panic!(
            "counterfactual_credit: cannot write {}: {e}",
            out_md_path.display()
        )
    });
    eprintln!("counterfactual_credit: wrote {}", out_md_path.display());
    eprintln!("counterfactual_credit: wrote {}", out_jsonl_path.display());
}
