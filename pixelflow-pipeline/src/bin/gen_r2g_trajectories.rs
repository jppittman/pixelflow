//! Trajectory minting for the Guide's return-to-go objective
//! (`docs/plans/2026-09-01-guide-return-to-go.md` §2/§5): for every corpus
//! expression, run `K` differently-ordered saturation trajectories to a
//! generous ceiling budget, and emit one JSONL record per recorded
//! rewrite-rule application carrying its trajectory's hindsight return at
//! the two registered checkpoints (B=100 primary, B=200 secondary) relative
//! to the expression's best cost seen across every trajectory.
//!
//! # Ordering diversity ([`OrderingPolicy`], §2)
//!
//! Twelve trajectories per expression: `unguided` (the production rule-major
//! sweep — see the equivalence test below), `per-rule`
//! ([`PerRuleRateGuide`], the zero-candidate-local-information control),
//! `strict-v1` ([`LinearCandidateGuide`] over the FROZEN Round-1 checkpoint),
//! six `random:<seed>` ([`UniformRandomGuide`]), and three rank-mixes of
//! `strict-v1` with a uniform term (`mix:strict-v1:1/4:<seed>` ×2,
//! `mix:strict-v1:1/2:<seed>` ×1) — §2's exact table, reused here rather
//! than re-derived (K and the seed count are justified there from
//! per-expression application volume and mint cost, not re-argued here).
//!
//! # Checkpoints: the budget ladder ([`BUDGET_LADDER`], round 3)
//!
//! Every trajectory runs through every rung of `--budgets` (default
//! [`BUDGET_LADDER`] = `100,200,400,800,1600,3200`), extracting at each —
//! round 3 (`docs/plans/2026-09-01-guide-return-to-go.md` §2b) generalized
//! this from round 1/2's fixed B=100/B=200/ceiling-at-4B trio specifically
//! to measure where, if anywhere, return spread survives into the budget
//! regime production actually runs classical kernels in (median ≈ 1,671
//! applications, PR #1087) — round 2's finding was that the ORIGINAL
//! `--max-expr-nodes 250`/`--max-classes 2000` mint made B=100 near-
//! quiescence for most surviving expressions, collapsing every guided
//! ordering onto the same committed set. The first two ladder rungs MUST
//! stay 100/200 ([`parse_budgets`] asserts this): every existing
//! `TrajectoryRow::app_actual_b100`/`cost_b100`/`return_b100` (and the
//! `_b200` twins) consumer — `train_guide_r2g`'s [`R2gRecord`],
//! `phase3_at_budget_eval`'s planned `--r2g-checkpoint` arm — reads those
//! two fields positionally, unchanged by this generalization.
//! `EpisodeLabels::compute_strict`'s `label_positive` is computed against
//! the LAST (highest-budget) extraction, same role round 1/2's ceiling
//! extraction played.
//!
//! # Feature observation is post-hoc, not in-loop (a stated approximation)
//!
//! [`CandidateFeatures::observe`] is called once per trajectory, after that
//! trajectory has run to its ceiling, against the FINAL e-graph — exactly
//! `gen_strict_labels`' own precedent (that binary's module doc: this
//! module built on top of `EpisodeLabels`/`CandidateFeatures` from day one).
//! `candidate.rs`'s own module doc names this precisely as an accepted
//! over-approximation ("acceptable for a cold-start dataset and not
//! silent"): a matched e-class's content and neighborhood can include nodes
//! added to it by a LATER application in the same trajectory. The
//! alternative — an in-loop observer hook threaded through
//! `GuidedSaturation` and a parallel one through the unguided sweep — is
//! `docs/plans/2026-09-01-guide-return-to-go.md` §1.5's skew-free design;
//! it is not built here because doing so for `Unguided` requires
//! instrumenting `EGraph`'s production sweep loop, and the smallest thing
//! that answers this task's question (does trajectory-level ordering
//! diversity produce a hindsight-return label with real spread?) does not
//! need it. `application_ordinal`/`budget_fraction` themselves are NOT
//! subject to this approximation (`candidate.rs`'s own doc): they are a
//! pure function of the recorded `ApplicationId`, exact regardless of when
//! `observe` is called.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_r2g_trajectories -- \
//!     --corpus-dir pixelflow-pipeline/data \
//!     --manifest pixelflow-pipeline/corpus_split.toml \
//!     --strict-checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json \
//!     --train-guide-report docs/results/2026-09-01-train-guide-report.json \
//!     --train-limit 800 \
//!     --report-json docs/results/2026-09-01-r2g-dataset.json \
//!     --report-md docs/results/2026-09-01-r2g-dataset.md
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;

use pixelflow_ir::{Environment, ExprData, Rooted, Term};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::guide_linear::{
    load_linear_guide, per_rule_rate_guide_from_report,
};
use pixelflow_pipeline::training::r2g::{
    BUDGET_LADDER, CheckpointRow, OrderingPolicy, TrajectoryRow, log_regret, spread_report,
};
use pixelflow_pipeline::training::split::{Family, SplitManifest, Tier};
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::{
    ApplicationId, Budget, CandidateFeatures, CostModel, EGraph, EpisodeLabels, Firing,
    KeepJournal, Label, Optimizer, Origin, REGISTERED_PRIMARY_BUDGET_APPLICATIONS, RuleSet,
    SaturationStop, config_for_node_count, extract_dag,
};
use pixelflow_search::nnue::guide::diversity::{UniformRandomGuide, uniform_unit_score};
use pixelflow_search::nnue::guide::linear::{LinearCandidateGuide, PerRuleRateGuide};
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

/// Registered primary tier, `B` — `BUDGET_LADDER[0]` MUST equal this (a test
/// below pins the equality) since [`unguided_trajectory_reproduces_run_
/// anytime_curve_cost_at_b`] compares against it directly.
const B_PRIMARY: usize = REGISTERED_PRIMARY_BUDGET_APPLICATIONS;
/// Same iteration ceiling `gen_strict_labels`/`guide_headroom` use — a
/// generous safety cap, not expected to bind under an application budget
/// far below it.
const SATURATE_MAX_ITERS: usize = 2_000;
/// Per-checkpoint wall-clock safety ceiling — this is an offline batch
/// measurement, so production's sub-second deadline does not apply; each
/// of a trajectory's [`BUDGET_LADDER`] stages shares this budget.
const SATURATE_TIMEOUT: Duration = Duration::from_secs(60);
/// Round-3 task's per-expression wall-clock safety ceiling
/// (`docs/plans/2026-09-01-guide-return-to-go.md` §2b): minting one
/// expression (every ordering policy, run out to the full
/// [`BUDGET_LADDER`]) that has not finished within this long is ABANDONED —
/// the outer loop stops launching further policies for that expression,
/// reports it loudly (stderr + a counted report field), and moves on. This
/// is checked between whole policy-trajectory runs (the finest granularity
/// available without threading a preemption hook through `EGraph`'s
/// saturation loop — see `run_trajectory`'s own per-checkpoint
/// `SATURATE_TIMEOUT` for the finer-grained net under that), so a single
/// pathological policy run can still overshoot this ceiling before the
/// check fires; it is a safety net, not a hard real-time bound.
const PER_EXPR_WALLCLOCK_CEILING: Duration = Duration::from_secs(600);

#[derive(Parser)]
#[command(name = "gen_r2g_trajectories")]
#[command(
    about = "Mint the R2G trajectory dataset: K ordering policies per expression, \
                    hindsight return labels at B=100/B=200"
)]
struct Args {
    /// Directory holding `corpus_{train,dev}.bin` and `corpus_dev_ood.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Path to the checked-in family split manifest.
    #[arg(long, default_value = "pixelflow-pipeline/corpus_split.toml")]
    manifest: String,

    /// The frozen Round-1 `LinearCandidateGuide` checkpoint (`strict-v1`),
    /// used by the `FrozenLinear`/`EpsilonMix` policies.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    strict_checkpoint: String,

    /// `train_guide`'s `--report-json` output, supplying the TRAIN-measured
    /// per-rule rates the `per-rule` control policy scores from.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-train-guide-report.json"
    )]
    train_guide_report: String,

    /// Which split(s) to mint: `train`, `dev`, `sh`, `bezier`, or `all`.
    #[arg(long, default_value = "all")]
    tier: String,

    /// Cap on TRAIN expressions processed (0 = all), stride-sampled exactly
    /// as `gen_strict_labels --train-limit` samples — identical semantics,
    /// so the two mints are comparable at the same limit.
    #[arg(long, default_value_t = 800)]
    train_limit: usize,

    /// Cap on DEV expressions processed (0 = all).
    #[arg(long, default_value_t = 0)]
    dev_limit: usize,

    /// Number of `random:<seed>` trajectories per expression (seeds
    /// `1..=n_rand`) — §2's `n_rand = 6` justified there from per-round
    /// survivor-set coverage, not re-derived here.
    #[arg(long, default_value_t = 6)]
    n_rand: u64,

    /// Comma-separated `numerator/denominator` ratios for the
    /// `mix:strict-v1:<n>/<d>:<seed>` policies — §2's table uses
    /// `1/4,1/4,1/2` (three mixes, reusing seeds 1 and 2 from the random
    /// pool: `1/4:1`, `1/4:2`, `1/2:1`).
    #[arg(long, default_value = "1/4,1/4,1/2")]
    mix: String,

    /// Skip corpus entries whose arena node count exceeds this — identical
    /// flag and identical default to `gen_strict_labels --max-expr-nodes`,
    /// for direct comparability (this binary's own module doc and mint
    /// report restate the value used, per this task's instruction).
    /// `0` LIFTS the filter entirely (no expression is skipped for size) —
    /// round 3's "measure spread where the budget binds" task needs the
    /// full TRAIN/DEV population, not the round-1/2
    /// `gen_strict_labels`-inherited `--max-expr-nodes 250` tractability cut
    /// that round 2 found was itself gating out the budget-bound regime.
    #[arg(long, default_value_t = 250)]
    max_expr_nodes: usize,

    /// Per-expression e-class cap passed to every saturation call. Unset
    /// (the default) resolves to production's OWN classical-tier cap, read
    /// from `pixelflow_search::egraph::config_for_node_count` rather than
    /// hardcoded — round 3's instruction is explicit that the cap must
    /// track that function, not restate its current value (5,000) as a
    /// literal that could silently drift out of step. Pass a value
    /// explicitly to override (e.g. to reproduce round 1/2's
    /// `--max-classes 2000`).
    #[arg(long)]
    max_classes: Option<usize>,

    /// Comma-separated application-count checkpoints to extract and label
    /// at, in ascending order — round 3's task: `100,200,400,800,1600,3200`.
    /// The mint's mandatory equivalence test
    /// (`unguided_trajectory_reproduces_run_anytime_curve_cost_at_b`) and
    /// every consumer of the ORIGINAL two-tier `TrajectoryRow` fields
    /// (`app_actual_b100`/`cost_b100`/`return_b100` and the `_b200` twins)
    /// require the first two entries to be exactly 100 and 200 — anything
    /// else panics loudly at startup rather than silently mislabeling those
    /// fields (see `parse_budgets`).
    #[arg(long, default_value = "100,200,400,800,1600,3200")]
    budgets: String,

    /// Output directory for the four `r2g_{train,dev,sh,bezier}.jsonl` /
    /// `r2g_trajectories_{train,dev,sh,bezier}.jsonl` file pairs.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    out_dir: String,

    #[arg(long)]
    report_json: Option<String>,

    #[arg(long)]
    report_md: Option<String>,
}

// ============================================================================
// Guide construction
// ============================================================================

/// The two loaded, checkpoint-backed guides every guided policy but
/// `Random`/`Unguided` needs, loaded once per mint run (not once per
/// expression/trajectory — both are read-only and reused across every
/// trajectory). `Option` (rather than requiring both unconditionally) so a
/// caller — including this binary's own equivalence test, which only
/// exercises `Unguided` — can build an empty cache without touching disk;
/// [`build_guide`] panics loudly (NO SILENT FAILURES) if a policy needs an
/// entry the cache doesn't have.
struct GuideCache {
    per_rule: Option<PerRuleRateGuide>,
    frozen_linear: Option<LinearCandidateGuide>,
}

impl GuideCache {
    /// Used only by this binary's own equivalence test (`Unguided` needs no
    /// checkpoint-backed guide at all) — dead outside `#[cfg(test)]`.
    #[allow(dead_code)]
    fn empty() -> Self {
        Self {
            per_rule: None,
            frozen_linear: None,
        }
    }

    fn load(strict_checkpoint: &Path, train_guide_report: &Path) -> Self {
        let rules = RuleSet::production();
        let frozen_linear = load_linear_guide(strict_checkpoint, &rules).unwrap_or_else(|e| {
            panic!("gen_r2g_trajectories: failed to load strict checkpoint: {e}")
        });
        let per_rule = per_rule_rate_guide_from_report(train_guide_report).unwrap_or_else(|e| {
            panic!("gen_r2g_trajectories: failed to load train_guide report: {e}")
        });
        Self {
            per_rule: Some(per_rule),
            frozen_linear: Some(frozen_linear),
        }
    }
}

/// A fully-built, OWNED ordering scorer for one [`OrderingPolicy`] (every
/// variant except `Unguided`, which never builds one — see
/// [`run_trajectory`]).
///
/// Owned rather than borrowed from the [`GuideCache`]: `Optimizer::guide`
/// takes a `Box<dyn SaturationGuide>`, so the scorer has to outlive the
/// cache's borrow. The guides are a few hundred floats each and one is
/// built per trajectory, so cloning them is cheaper than the lifetime.
enum BuiltGuide {
    PerRule(PerRuleRateGuide),
    Linear(LinearCandidateGuide),
    Random(UniformRandomGuide),
    /// A rank-mix of `base`'s score with a per-candidate uniform term —
    /// `docs/plans/2026-09-01-guide-return-to-go.md` §2's `RankMixGuide`
    /// row, computed inline here rather than as a separate
    /// `pixelflow-search` type (see this binary's module doc: the smallest
    /// thing that answers the question).
    Mix {
        base: Box<BuiltGuide>,
        numerator: u16,
        denominator: u16,
        seed: u64,
    },
}

impl SaturationGuide for BuiltGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        match self {
            BuiltGuide::PerRule(g) => g.score_candidates(candidates),
            BuiltGuide::Linear(g) => g.score_candidates(candidates),
            BuiltGuide::Random(g) => g.score_candidates(candidates),
            BuiltGuide::Mix {
                base,
                numerator,
                denominator,
                seed,
            } => {
                let base_scores = base.score_candidates(candidates);
                let n = base_scores.len();
                if n == 0 {
                    return Vec::new();
                }
                let eps = f32::from(*numerator) / f32::from(*denominator);
                // Rank-normalize `base_scores` to `[0,1]` (highest score ->
                // 1.0) — mixing on ranks, not raw logits, so `eps` means the
                // same thing regardless of the base guide's score scale.
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| {
                    base_scores[b]
                        .partial_cmp(&base_scores[a])
                        .unwrap_or_else(|| {
                            panic!(
                                "BuiltGuide::Mix: base guide produced a non-finite score \
                             ({} or {})",
                                base_scores[a], base_scores[b]
                            )
                        })
                });
                let mut rank_norm = vec![0.0f32; n];
                for (rank, &idx) in order.iter().enumerate() {
                    rank_norm[idx] = if n > 1 {
                        1.0 - (rank as f32 / (n as f32 - 1.0))
                    } else {
                        1.0
                    };
                }
                candidates
                    .iter()
                    .zip(rank_norm.iter())
                    .map(|(c, &rn)| {
                        let u = uniform_unit_score(*seed, c);
                        (1.0 - eps) * rn + eps * u
                    })
                    .collect()
            }
        }
    }
}

/// Build the [`BuiltGuide`] a non-`Unguided` [`OrderingPolicy`] names.
///
/// # Panics
///
/// - `Unguided` has no guide — callers must special-case it (see
///   [`run_trajectory`]); calling this with it is a caller bug.
/// - A `PerRuleRate`/`FrozenLinear` policy whose corresponding `cache` entry
///   is `None` — a caller that built an empty [`GuideCache`] (e.g. this
///   binary's own equivalence test) but then asked for a checkpoint-backed
///   policy anyway.
/// - A nested `EpsilonMix` (an `EpsilonMix` whose `guide` is itself an
///   `EpsilonMix`) — not a meaning this binary defines; §2's table never
///   nests one.
fn build_guide(policy: &OrderingPolicy, cache: &GuideCache) -> BuiltGuide {
    match policy {
        OrderingPolicy::Unguided => {
            panic!("build_guide: OrderingPolicy::Unguided has no SaturationGuide")
        }
        OrderingPolicy::PerRuleRate => BuiltGuide::PerRule(
            cache
                .per_rule
                .as_ref()
                .unwrap_or_else(|| {
                    panic!(
                        "build_guide: PerRuleRate policy requested but GuideCache has no \
                         per_rule guide loaded"
                    )
                })
                .clone(),
        ),
        OrderingPolicy::FrozenLinear(_) => BuiltGuide::Linear(
            cache
                .frozen_linear
                .as_ref()
                .unwrap_or_else(|| {
                    panic!(
                        "build_guide: FrozenLinear policy requested but GuideCache has no \
                             frozen_linear guide loaded"
                    )
                })
                .clone(),
        ),
        OrderingPolicy::Random(seed) => BuiltGuide::Random(UniformRandomGuide { seed: *seed }),
        OrderingPolicy::EpsilonMix {
            guide,
            numerator,
            denominator,
            seed,
        } => {
            assert!(
                !matches!(guide.as_ref(), OrderingPolicy::EpsilonMix { .. }),
                "build_guide: nested EpsilonMix is not supported — {guide:?} is itself an \
                 EpsilonMix"
            );
            BuiltGuide::Mix {
                base: Box::new(build_guide(guide, cache)),
                numerator: *numerator,
                denominator: *denominator,
                seed: *seed,
            }
        }
    }
}

/// The K=12 ordering policies §2's table names, built from CLI-parameterized
/// seed counts. `strict-v1`'s label is fixed to that name regardless of the
/// `--strict-checkpoint` path — see [`OrderingPolicy::FrozenLinear`]'s doc.
fn build_policies(n_rand: u64, mix_ratios: &[(u16, u16)]) -> Vec<OrderingPolicy> {
    let mut policies = vec![
        OrderingPolicy::Unguided,
        OrderingPolicy::PerRuleRate,
        OrderingPolicy::FrozenLinear("strict-v1".to_string()),
    ];
    for seed in 1..=n_rand {
        policies.push(OrderingPolicy::Random(seed));
    }
    // §2's table reuses seeds 1 and 2 from the random pool for its three
    // mixes (`mix:strict-v1:1/4:1`, `mix:strict-v1:1/4:2`,
    // `mix:strict-v1:1/2:1`) rather than minting fresh ones — deliberate,
    // so a mix trajectory shares its randomness source with an existing
    // named `random:<seed>` trajectory for direct comparison.
    let mix_seeds = [1u64, 2u64, 1u64];
    for (i, &(num, den)) in mix_ratios.iter().enumerate() {
        let seed = mix_seeds[i % mix_seeds.len()];
        policies.push(OrderingPolicy::EpsilonMix {
            guide: Box::new(OrderingPolicy::FrozenLinear("strict-v1".to_string())),
            numerator: num,
            denominator: den,
            seed,
        });
    }
    policies
}

/// Parse `--budgets` into an ascending `Vec<usize>`, asserting the first two
/// entries are exactly 100/200 (see the flag's own doc for why: every
/// existing `TrajectoryRow::*_b100`/`*_b200` consumer, and this binary's own
/// equivalence test, depend on that positionally).
fn parse_budgets(s: &str) -> Vec<usize> {
    let budgets: Vec<usize> = s
        .split(',')
        .map(|tok| {
            let tok = tok.trim();
            tok.parse::<usize>().unwrap_or_else(|e| {
                panic!("gen_r2g_trajectories: --budgets entry {tok:?} is not a number: {e}")
            })
        })
        .collect();
    assert!(
        !budgets.is_empty(),
        "gen_r2g_trajectories: --budgets must name at least one checkpoint"
    );
    for w in budgets.windows(2) {
        assert!(
            w[0] < w[1],
            "gen_r2g_trajectories: --budgets must be strictly ascending, got {budgets:?}"
        );
    }
    assert_eq!(
        (budgets[0], *budgets.get(1).unwrap_or(&0)),
        (100, 200),
        "gen_r2g_trajectories: --budgets' first two entries must be 100,200 — every existing \
         TrajectoryRow::app_actual_b100/cost_b100/return_b100 (and the _b200 twins) consumer \
         reads those positionally; got {budgets:?}"
    );
    budgets
}

fn parse_mix_ratios(s: &str) -> Vec<(u16, u16)> {
    s.split(',')
        .map(|tok| {
            let tok = tok.trim();
            let (num, den) = tok.split_once('/').unwrap_or_else(|| {
                panic!("gen_r2g_trajectories: --mix entry {tok:?} is not \"num/den\"")
            });
            let num: u16 = num
                .parse()
                .unwrap_or_else(|e| panic!("gen_r2g_trajectories: --mix numerator {num:?}: {e}"));
            let den: u16 = den
                .parse()
                .unwrap_or_else(|e| panic!("gen_r2g_trajectories: --mix denominator {den:?}: {e}"));
            (num, den)
        })
        .collect()
}

// ============================================================================
// Running one trajectory
// ============================================================================

/// One checkpoint's raw outcome within a trajectory — one per
/// [`BUDGET_LADDER`] entry.
struct RawCheckpoint {
    budget: usize,
    app_actual: u64,
    cost: u64,
    stop: SaturationStop,
}

/// One trajectory's raw outcome: the e-graph it produced (kept, so the
/// per-application feature/label pass below can walk its final provenance
/// log — see the module doc's "Feature observation is post-hoc" section),
/// its per-[`BUDGET_LADDER`]-checkpoint costs, and the hindsight strict
/// labels computed against the LAST (ceiling) extraction.
struct RawTrajectory {
    egraph: EGraph,
    checkpoints: Vec<RawCheckpoint>,
    /// `true` iff saturation had already stopped for a reason other than
    /// the application budget (quiesced / class-capped / iteration-
    /// ceilinged) by the time the LAST checkpoint returned.
    ended: bool,
    labels: EpisodeLabels,
}

/// Run one `(expression, policy)` trajectory through every stage of
/// `budgets` (ascending, e.g. [`BUDGET_LADDER`]), extracting at each. The
/// `Unguided` and guided branches are written out separately, rather than
/// behind one shared closure/helper, because a closure capturing `&mut
/// egraph` would hold that borrow for its own entire lifetime — blocking
/// the `extract_dag(&egraph, ...)` calls this function needs to interleave
/// between checkpoint stages; two direct loops avoid the borrow entirely
/// and read no worse for the duplication.
#[allow(clippy::too_many_arguments)]
fn run_trajectory(
    policy: &OrderingPolicy,
    cache: &GuideCache,
    term: Term<'_>,
    max_classes: usize,
    costs: &CostModel,
    budgets: &[usize],
) -> RawTrajectory {
    // ONE `Optimizer` across every checkpoint of a trajectory. The guided
    // arm's episode state (its resolved-candidate-key set) lives on the
    // optimizer, so rebuilding one per checkpoint would re-score and re-fire
    // every already-resolved key — "silently handicapped at every checkpoint
    // after the first". The unguided arm carries no such state and is the
    // same construction with `guide` left `None`, which is the whole point
    // of the guide being a field.
    let guide: Option<Box<dyn SaturationGuide>> = if matches!(policy, OrderingPolicy::Unguided) {
        None
    } else {
        Some(Box::new(build_guide(policy, cache)))
    };
    let mut optimizer = Optimizer::production()
        .cost(costs.clone())
        .guide(guide)
        // The trajectory's labels are read off the journal after the run, so
        // recording has to be asked for: `Optimizer` records only for an
        // observer (#1118), and production sets none.
        .observe(Some(Box::new(KeepJournal)))
        .hard_ceiling(SATURATE_TIMEOUT);
    let mut egraph = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term,
        &mut egraph,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let node_count = term.root().node_count();
    let mut checkpoints = Vec::with_capacity(budgets.len());
    let mut last_stop = SaturationStop::ApplicationBudget;
    let mut last_ext = None;

    for &budget in budgets {
        // THE DELTA. `Budget`'s application dimension is a gap from the
        // current count, not an absolute target; passing `budget` itself
        // would make the x-axis a prefix sum of itself.
        let already = egraph.application_count();
        let delta = (budget as u64).saturating_sub(already);
        optimizer = optimizer.budget(Budget::Explicit {
            iterations: SATURATE_MAX_ITERS,
            classes: max_classes,
            applications: Some(delta),
        });
        let out = optimizer.run(&mut egraph, root_class, node_count);
        let ext = extract_dag(&egraph, root_class, costs);
        checkpoints.push(RawCheckpoint {
            budget,
            app_actual: egraph.application_count(),
            // DAG cost — what the emitted kernel pays (#1117).
            cost: ext.dag_cost as u64,
            stop: out.stats.stop,
        });
        last_stop = out.stats.stop;
        last_ext = Some(ext);
    }

    let ext_last = last_ext
        .unwrap_or_else(|| panic!("run_trajectory: budgets slice was empty — no checkpoint ran"));
    let labels = EpisodeLabels::compute_strict(&egraph, ext_last.root, &ext_last.choices);
    let ended = last_stop != SaturationStop::ApplicationBudget;

    RawTrajectory {
        egraph,
        checkpoints,
        ended,
        labels,
    }
}

/// Which recorded applications actually changed the e-graph (committed a
/// node or a union), read post-hoc from `egraph`'s own provenance rather
/// than threaded through the saturation call: an application "changed"
/// iff it created at least one e-node (`Origin::Rule(app_id)` appears on
/// some node) or caused at least one recorded union
/// (`UnionEvent::application_id == Some(app_id)`) — the same two ways
/// `apply_action_from_rule`'s executor can report `> 0`.
fn changed_application_ids(egraph: &EGraph) -> HashSet<ApplicationId> {
    let mut ids: HashSet<ApplicationId> = egraph
        .provenance()
        .origins()
        .filter_map(|(_, origin)| match origin {
            Origin::Rule(id) => Some(id),
            Origin::Seed => None,
        })
        .collect();
    for event in egraph.provenance().union_events() {
        if let Some(id) = event.application_id {
            ids.insert(id);
        }
    }
    ids
}

fn opt_f32_json(x: Option<f32>) -> String {
    match x {
        Some(v) => format!("{v:.6}"),
        None => "null".to_string(),
    }
}

fn mean_of_some(xs: &[Option<f32>]) -> Option<f32> {
    let vals: Vec<f32> = xs.iter().filter_map(|&x| x).collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f32>() / vals.len() as f32)
    }
}

// ============================================================================
// Minting one expression (K trajectories -> R2gRecord rows + TrajectoryRows)
// ============================================================================

struct ExprMintOutcome {
    applications_written: usize,
    trajectory_rows: Vec<TrajectoryRow>,
    /// `true` iff every trajectory's checkpoint costs bottomed out at zero
    /// (the registration's zero-best convention: a positive cost against a
    /// zero reference is excluded rather than reported as infinite regret)
    /// — this expression contributed no records to the output file.
    zero_best_excluded: bool,
}

#[allow(clippy::too_many_arguments)]
fn mint_expression(
    name: &str,
    term: Term<'_>,
    family_band: u32,
    family_seed: u64,
    tier_label: &str,
    policies: &[OrderingPolicy],
    cache: &GuideCache,
    max_classes: usize,
    costs: &CostModel,
    rules: &RuleSet,
    budgets: &[usize],
    out: &mut impl Write,
) -> ExprMintOutcome {
    let expr_node_count = term.root().node_count();
    let raw: Vec<RawTrajectory> = policies
        .iter()
        .map(|p| run_trajectory(p, cache, term, max_classes, costs, budgets))
        .collect();

    // Empirical best (§1.2's `c*_e`): the minimum extraction cost seen at
    // ANY checkpoint (any rung of `budgets`) of ANY trajectory — matching
    // "the unguided trajectory contributes its full grid" in spirit without
    // needing the full anytime grid this mint does not sample.
    let best: u64 = raw
        .iter()
        .flat_map(|t| t.checkpoints.iter().map(|c| c.cost))
        .min()
        .unwrap_or(0);

    if best == 0 {
        return ExprMintOutcome {
            applications_written: 0,
            trajectory_rows: Vec::new(),
            zero_best_excluded: true,
        };
    }

    // Returns per (trajectory, budget-ladder-index), plus the per-budget
    // cross-trajectory mean used for centering — generalizes the original
    // b100/b200-only computation to every rung of `budgets`.
    let n_budgets = budgets.len();
    let returns: Vec<Vec<Option<f32>>> = raw
        .iter()
        .map(|t| {
            t.checkpoints
                .iter()
                .map(|c| log_regret(c.cost, best))
                .collect()
        })
        .collect();
    let means: Vec<Option<f32>> = (0..n_budgets)
        .map(|bi| {
            mean_of_some(
                &raw.iter()
                    .enumerate()
                    .map(|(ti, _)| returns[ti][bi])
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    let mut trajectory_rows = Vec::with_capacity(raw.len());
    let mut applications_written = 0usize;

    for (i, (traj, policy)) in raw.iter().zip(policies.iter()).enumerate() {
        let trajectory_id = i as u32;
        let policy_label = policy.label();
        let return_b100 = returns[i][0];
        let return_b200 = returns[i][1];

        let checkpoint_rows: Vec<CheckpointRow> = traj
            .checkpoints
            .iter()
            .zip(returns[i].iter())
            .map(|(c, &r)| CheckpointRow {
                budget: c.budget,
                app_actual: c.app_actual,
                cost: c.cost,
                stop: format!("{:?}", c.stop),
                return_val: r,
            })
            .collect();
        let last_checkpoint = traj.checkpoints.last().unwrap_or_else(|| {
            panic!("mint_expression: trajectory {trajectory_id} ran zero checkpoints")
        });

        trajectory_rows.push(TrajectoryRow {
            expr_name: name.to_string(),
            tier: tier_label.to_string(),
            trajectory_id,
            policy: policy_label.clone(),
            expr_node_count,
            app_actual_b100: traj.checkpoints[0].app_actual,
            cost_b100: traj.checkpoints[0].cost,
            app_actual_b200: traj.checkpoints[1].app_actual,
            cost_b200: traj.checkpoints[1].cost,
            ended: traj.ended,
            ended_at_apps: last_checkpoint.app_actual,
            return_b100,
            return_b200,
            checkpoints: checkpoint_rows,
        });

        let centered_b100 = return_b100.zip(means[0]).map(|(r, m)| r - m);
        let centered_b200 = return_b200.zip(means[1]).map(|(r, m)| r - m);

        let changed_ids = changed_application_ids(&traj.egraph);
        // Same dedup bookkeeping `gen_strict_labels` does, and for the same
        // reason: the guided loop resolves a `CandidateKey` once and never
        // scores it again, so a repeat row is a candidate no deployed guide
        // is ever asked to rank. Recorded rather than filtered, so the rate
        // stays measurable and every consumer decides explicitly — the two
        // datasets share one schema, or `R2gRecord`'s flattened `Record`
        // stops being the same record.
        let mut seen_keys: std::collections::HashSet<pixelflow_search::egraph::CandidateKey> =
            std::collections::HashSet::new();
        for (app_id, record) in traj.egraph.provenance().applications() {
            let rule = record.rule.unwrap_or_else(|| {
                panic!(
                    "gen_r2g_trajectories: application {app_id:?} carries no RuleId — the \
                     graph was built without rule ids, and every table here is keyed by \
                     identity"
                )
            });
            let firing = Firing {
                rule,
                match_root: record.match_root,
                application_ordinal: app_id.as_u64(),
                registered_budget: REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
            };
            let features = CandidateFeatures::observe(&traj.egraph, &firing);
            let dedup_repeat = !seen_keys.insert(features.key.clone());
            let label_positive =
                traj.labels.labels.get(&app_id).copied() == Some(Label::LoadBearing);
            let rule_name = rules
                .index_of(rule)
                .and_then(|i| rules.label_of(i))
                .unwrap_or_else(|| format!("<rule {}>", rule.get()));
            let changed = changed_ids.contains(&app_id);

            // A histogram, not the raw op list — see `gen_strict_labels`'s
            // identical comment (a matched class deep into saturation can
            // hold thousands of near-duplicate node shapes).
            let mut op_hist: BTreeMap<String, usize> = BTreeMap::new();
            for op in &features.neighborhood_ops {
                *op_hist.entry(format!("{op:?}")).or_insert(0) += 1;
            }
            let hist_json: String = {
                let mut s = String::from("{");
                for (i, (op, count)) in op_hist.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&format!("{op:?}:{count}"));
                }
                s.push('}');
                s
            };

            writeln!(
                out,
                "{{\"expr_name\":{:?},\"family_band\":{},\"family_seed\":{},\
                 \"expr_node_count\":{},\"rule_id\":{},\"rule_name\":{:?},\
                 \"budget_fraction\":{:.6},\"match_class_node_count\":{},\
                 \"neighborhood_op_count\":{},\"neighborhood_op_hist\":{},\
                 \"dedup_repeat\":{},\"label_positive\":{},\"trajectory_id\":{},\"policy\":{:?},\
                 \"round_ordinal\":{},\"application_ordinal\":{},\"changed\":{},\
                 \"cost_b100\":{},\"cost_b200\":{},\"expr_best_cost\":{},\
                 \"return_b100\":{},\"return_b200\":{},\"centered_b100\":{},\
                 \"centered_b200\":{}}}",
                name,
                family_band,
                family_seed,
                expr_node_count,
                rule.get(),
                rule_name,
                features.budget_fraction(),
                features.key.content.node_count(),
                features.neighborhood_ops.len(),
                hist_json,
                dedup_repeat,
                label_positive,
                trajectory_id,
                policy_label,
                record.step,
                app_id.as_u64(),
                changed,
                traj.checkpoints[0].cost,
                traj.checkpoints[1].cost,
                best,
                opt_f32_json(return_b100),
                opt_f32_json(return_b200),
                opt_f32_json(centered_b100),
                opt_f32_json(centered_b200),
            )
            .unwrap_or_else(|e| panic!("gen_r2g_trajectories: write failed: {e}"));

            applications_written += 1;
        }
    }

    ExprMintOutcome {
        applications_written,
        trajectory_rows,
        zero_best_excluded: false,
    }
}

// ============================================================================
// Corpus loading, fence checks, and per-split mint
// ============================================================================

/// Parse `gen_bench_corpus`'s `{tier}_b{band:02}_f{seed:02}_{idx:05}` name
/// format back into a [`Family`] — identical logic to `gen_strict_labels`'s
/// own (private, duplicated per this codebase's own convention: every
/// label-minting binary that needs this parses it itself rather than
/// sharing a helper — see that binary's copy).
fn parse_family(name: &str) -> Family {
    let parts: Vec<&str> = name.split('_').collect();
    assert!(
        parts.len() >= 3,
        "gen_r2g_trajectories: corpus entry name '{name}' does not match the expected \
         `{{tier}}_b{{band}}_f{{seed}}_{{idx}}` format"
    );
    let band_tok = parts[1].strip_prefix('b').unwrap_or_else(|| {
        panic!("gen_r2g_trajectories: corpus entry name '{name}' has no 'b<NN>' band token")
    });
    let seed_tok = parts[2].strip_prefix('f').unwrap_or_else(|| {
        panic!("gen_r2g_trajectories: corpus entry name '{name}' has no 'f<NN>' seed token")
    });
    let band: usize = band_tok.parse().unwrap_or_else(|e| {
        panic!("gen_r2g_trajectories: '{name}' band token '{band_tok}' is not a number: {e}")
    });
    let seed: u64 = seed_tok.parse().unwrap_or_else(|e| {
        panic!("gen_r2g_trajectories: '{name}' seed token '{seed_tok}' is not a number: {e}")
    });
    Family::new(band, seed)
}

/// Stride-sample `entries` down to at most `limit`, evenly across the whole
/// population — identical to `gen_strict_labels`'s own `stride_sample`.
fn stride_sample<T: Clone>(entries: Vec<T>, limit: usize) -> Vec<T> {
    if limit == 0 || limit >= entries.len() {
        return entries;
    }
    let stride = entries.len() as f64 / limit as f64;
    let mut sampled = Vec::with_capacity(limit);
    for i in 0..limit {
        let idx = ((i as f64) * stride) as usize;
        sampled.push(entries[idx.min(entries.len() - 1)].clone());
    }
    sampled
}

/// Fail loud (NO SILENT FAILURES) if any classical entry's parsed family is
/// not assigned to `expected` by the checked-in split manifest — the same
/// fence `gen_strict_labels` enforces for TRAIN/DEV.
fn assert_family_fence(
    entries: &[(String, Rooted<ExprData>)],
    manifest: &SplitManifest,
    expected: Tier,
) {
    for (name, _) in entries {
        let family = parse_family(name);
        let assigned = manifest.tier_of(family);
        assert_eq!(
            assigned,
            Some(expected),
            "gen_r2g_trajectories: fence check failed — corpus entry '{name}' parses to \
             {family}, but corpus_split.toml assigns that family to {assigned:?}, not \
             {expected:?}"
        );
    }
}

/// Every `FenceKey` in `corpus_train.bin` — the TRAIN-fence source for the
/// OOD (`sh`/`bezier`) splits, which carry no `{band,seed}` family and so
/// cannot be fence-checked by [`assert_family_fence`].
fn train_fence_keys(corpus_dir: &Path) -> HashSet<FenceKey> {
    let path = corpus_dir.join("corpus_train.bin");
    let entries = read_corpus(&path).unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: failed to read {}: {e}",
            path.display()
        )
    });
    entries
        .iter()
        .map(|(_, expr)| FenceKey::of(expr.entry()))
        .collect()
}

/// Hard-fail if any OOD entry shares a feature-quotient structure with
/// TRAIN — the same discipline `phase3_at_budget_eval`'s
/// `enforce_train_fence` applies (round-1b registration §3).
fn assert_train_fence(
    entries: &[(String, Rooted<ExprData>)],
    train_keys: &HashSet<FenceKey>,
    source: &str,
) {
    let collisions: Vec<&str> = entries
        .iter()
        .filter(|(_, expr)| train_keys.contains(&FenceKey::of(expr.entry())))
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        collisions.is_empty(),
        "gen_r2g_trajectories: TRAIN-fence violation in {source}: {} of {} entries share a \
         feature-quotient structure with corpus_train.bin: {:?}{}",
        collisions.len(),
        entries.len(),
        &collisions[..collisions.len().min(10)],
        if collisions.len() > 10 {
            ", ... (truncated)"
        } else {
            ""
        },
    );
}

/// Everything one split's mint measured — feeds the combined report.
struct SplitOutcome {
    tier: String,
    expressions: usize,
    zero_best_excluded: usize,
    skipped_oversized: usize,
    /// Expressions whose FULL mint (every policy, every `budgets` rung)
    /// exceeded [`PER_EXPR_WALLCLOCK_CEILING`] — abandoned and excluded from
    /// output rather than written partially, and reported loudly (stderr,
    /// plus this counted field) per the round-3 task's "reported loudly as
    /// skipped, never silent" requirement. This check is necessarily
    /// post-hoc (timed after `mint_expression` returns), not preemptive —
    /// see [`PER_EXPR_WALLCLOCK_CEILING`]'s own doc for why a synchronous,
    /// single-threaded mint loop cannot interrupt mid-saturation without a
    /// preemption hook this task's scope does not require building.
    skipped_wallclock: usize,
    skipped_wallclock_names: Vec<String>,
    trajectories: usize,
    applications: usize,
    trajectory_rows: Vec<TrajectoryRow>,
}

#[allow(clippy::too_many_arguments)]
fn mint_split(
    tier_label: &str,
    entries: &[(String, Rooted<ExprData>)],
    policies: &[OrderingPolicy],
    cache: &GuideCache,
    max_expr_nodes: usize,
    max_classes: usize,
    rules: &RuleSet,
    budgets: &[usize],
    family_of: impl Fn(&str) -> (u32, u64),
    out_records_path: &Path,
    out_trajectories_path: &Path,
) -> SplitOutcome {
    let costs = CostModel::latency_prior();
    let mut records_out = BufWriter::new(File::create(out_records_path).unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: cannot create {}: {e}",
            out_records_path.display()
        )
    }));
    let mut trajectories_out =
        BufWriter::new(File::create(out_trajectories_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: cannot create {}: {e}",
                out_trajectories_path.display()
            )
        }));

    let mut outcome = SplitOutcome {
        tier: tier_label.to_string(),
        expressions: 0,
        zero_best_excluded: 0,
        skipped_oversized: 0,
        skipped_wallclock: 0,
        skipped_wallclock_names: Vec::new(),
        trajectories: 0,
        applications: 0,
        trajectory_rows: Vec::new(),
    };

    let n = entries.len();
    // A corpus entry declares no buffers or uniforms — the format refuses to
    // write one down — so one empty environment serves every term.
    let env = Environment::new();
    for (i, (name, expr)) in entries.iter().enumerate() {
        let term = Term::new(expr.entry(), &env);
        let node_count = expr.len();
        // `max_expr_nodes == 0` LIFTS the filter — see the flag's own doc.
        if max_expr_nodes != 0 && node_count > max_expr_nodes {
            outcome.skipped_oversized += 1;
            continue;
        }
        let (family_band, family_seed) = family_of(name);

        // Mint into a scratch buffer first, timed, so an over-ceiling
        // expression's partial output never reaches the real file — see
        // `SplitOutcome::skipped_wallclock`'s doc.
        let mint_start = std::time::Instant::now();
        let mut scratch = Vec::new();
        let result = mint_expression(
            name,
            term,
            family_band,
            family_seed,
            tier_label,
            policies,
            cache,
            max_classes,
            &costs,
            rules,
            budgets,
            &mut scratch,
        );
        let mint_elapsed = mint_start.elapsed();
        if mint_elapsed > PER_EXPR_WALLCLOCK_CEILING {
            eprintln!(
                "gen_r2g_trajectories[{tier_label}]: SKIPPING '{name}' ({node_count} nodes) — \
                 mint took {:.1}s, over the {:.0}s per-expression wall-clock ceiling",
                mint_elapsed.as_secs_f64(),
                PER_EXPR_WALLCLOCK_CEILING.as_secs_f64()
            );
            outcome.skipped_wallclock += 1;
            outcome.skipped_wallclock_names.push(name.clone());
            continue;
        }
        records_out.write_all(&scratch).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: write to {}: {e}",
                out_records_path.display()
            )
        });
        if result.zero_best_excluded {
            outcome.zero_best_excluded += 1;
        } else {
            outcome.expressions += 1;
            outcome.trajectories += result.trajectory_rows.len();
            outcome.applications += result.applications_written;
            for row in &result.trajectory_rows {
                let json = serde_json::to_string(row).unwrap_or_else(|e| {
                    panic!("gen_r2g_trajectories: TrajectoryRow serialization failed: {e}")
                });
                writeln!(trajectories_out, "{json}").unwrap_or_else(|e| {
                    panic!(
                        "gen_r2g_trajectories: write to {}: {e}",
                        out_trajectories_path.display()
                    )
                });
            }
            outcome.trajectory_rows.extend(result.trajectory_rows);
        }
        if (i + 1) % 25 == 0 || i + 1 == n {
            eprintln!(
                "gen_r2g_trajectories[{tier_label}]: {}/{n} expressions minted ({} \
                 applications so far)",
                i + 1,
                outcome.applications
            );
        }
    }

    records_out.flush().unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: flushing {}: {e}",
            out_records_path.display()
        )
    });
    trajectories_out.flush().unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: flushing {}: {e}",
            out_trajectories_path.display()
        )
    });

    outcome
}

// ============================================================================
// Report
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn write_report(
    path_json: Option<&str>,
    path_md: Option<&str>,
    args: &Args,
    resolved_max_classes: usize,
    budgets: &[usize],
    outcomes: &[SplitOutcome],
    mint_seconds: f64,
) {
    let mut per_split_json = String::new();
    let mut per_split_md = String::new();
    for (i, outcome) in outcomes.iter().enumerate() {
        let spread = spread_report(&outcome.trajectory_rows);

        // Per-policy median regret at B=100 — the report's headline
        // per-policy statistic.
        let mut by_policy: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for row in &outcome.trajectory_rows {
            if let Some(r) = row.return_b100 {
                by_policy.entry(row.policy.as_str()).or_default().push(r);
            }
        }
        let mut policy_medians: Vec<(String, f32, usize)> = by_policy
            .into_iter()
            .map(|(policy, mut rets)| {
                rets.sort_by(|a, b| a.partial_cmp(b).expect("finite return"));
                let n = rets.len();
                let median = if n == 0 {
                    0.0
                } else if n % 2 == 1 {
                    rets[n / 2]
                } else {
                    (rets[n / 2 - 1] + rets[n / 2]) / 2.0
                };
                (policy.to_string(), median, n)
            })
            .collect();
        policy_medians.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite median"));

        if i > 0 {
            per_split_json.push_str(",\n");
            per_split_md.push('\n');
        }
        per_split_json.push_str(&format!(
            "  {{\n    \"tier\": \"{}\",\n    \"expressions\": {},\n    \
             \"zero_best_excluded\": {},\n    \"skipped_oversized\": {},\n    \
             \"skipped_wallclock\": {},\n    \"skipped_wallclock_names\": {:?},\n    \
             \"trajectories\": {},\n    \"applications\": {},\n    \
             \"zero_spread_b100\": {},\n    \"zero_spread_b100_share\": {:.4},\n    \
             \"zero_spread_b100_record_share\": {:.4},\n    \
             \"spread_b100_quartiles\": [{:.4}, {:.4}, {:.4}],\n    \
             \"dataset_gate_fired\": {},\n    \"per_policy_median_return_b100\": [\n{}\n    ]\n  }}",
            outcome.tier,
            outcome.expressions,
            outcome.zero_best_excluded,
            outcome.skipped_oversized,
            outcome.skipped_wallclock,
            outcome.skipped_wallclock_names,
            outcome.trajectories,
            outcome.applications,
            spread.zero_spread_b100,
            if spread.expressions == 0 {
                0.0
            } else {
                spread.zero_spread_b100 as f64 / spread.expressions as f64
            },
            spread.zero_spread_b100_record_share,
            spread.spread_b100_quartiles.0,
            spread.spread_b100_quartiles.1,
            spread.spread_b100_quartiles.2,
            spread.dataset_gate_fired,
            policy_medians
                .iter()
                .map(|(policy, median, n)| format!(
                    "      {{\"policy\": {policy:?}, \"median_return_b100\": {median:.4}, \"n\": {n}}}"
                ))
                .collect::<Vec<_>>()
                .join(",\n"),
        ));

        per_split_md.push_str(&format!(
            "## {}\n\n\
             - expressions: {} ({} zero-best excluded, {} oversized-skipped, {} \
             wallclock-skipped{})\n\
             - trajectories: {}\n\
             - applications (JSONL rows): {}\n\
             - return spread at B=100: Q1 {:.4}  median {:.4}  Q3 {:.4}\n\
             - zero-spread expressions: {} / {} ({:.1}%), {:.1}% of records\n\
             - dataset gate (>50% zero-spread): {}\n\n\
             | policy | median return_b100 | n |\n|---|---:|---:|\n{}\n",
            outcome.tier,
            outcome.expressions,
            outcome.zero_best_excluded,
            outcome.skipped_oversized,
            outcome.skipped_wallclock,
            if outcome.skipped_wallclock_names.is_empty() {
                String::new()
            } else {
                format!(": {:?}", outcome.skipped_wallclock_names)
            },
            outcome.trajectories,
            outcome.applications,
            spread.spread_b100_quartiles.0,
            spread.spread_b100_quartiles.1,
            spread.spread_b100_quartiles.2,
            spread.zero_spread_b100,
            spread.expressions,
            if spread.expressions == 0 {
                0.0
            } else {
                100.0 * spread.zero_spread_b100 as f64 / spread.expressions as f64
            },
            100.0 * spread.zero_spread_b100_record_share as f64,
            if spread.dataset_gate_fired {
                "FIRED"
            } else {
                "not fired"
            },
            policy_medians
                .iter()
                .map(|(policy, median, n)| format!("| {policy} | {median:.4} | {n} |"))
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    if let Some(path) = path_json {
        let json = format!(
            "{{\n  \"registered_budget_b\": {B_PRIMARY},\n  \"budget_ladder\": {:?},\n  \
             \"max_expr_nodes\": {},\n  \
             \"max_classes\": {resolved_max_classes},\n  \"train_limit\": {},\n  \
             \"dev_limit\": {},\n  \
             \"n_rand\": {},\n  \"mix\": {:?},\n  \"mint_seconds\": {mint_seconds:.2},\n  \
             \"splits\": [\n{per_split_json}\n  ]\n}}\n",
            budgets, args.max_expr_nodes, args.train_limit, args.dev_limit, args.n_rand, args.mix,
        );
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("gen_r2g_trajectories: cannot write {path}: {e}"));
        eprintln!("gen_r2g_trajectories: wrote {path}");
    }
    if let Some(path) = path_md {
        let md = format!(
            "# R2G trajectory mint\n\n\
             Registered budget B={B_PRIMARY} (primary), budget ladder {:?}. \
             `--max-expr-nodes {}` (`0` = no filter) / `--max-classes {resolved_max_classes}` \
             (resolved from `config_for_node_count` when `--max-classes` is unset). \
             `--train-limit {}` \
             `--dev-limit {}` `--n-rand {}` `--mix {:?}`. Mint wall-clock: {mint_seconds:.1}s.\n\n\
             {per_split_md}",
            budgets, args.max_expr_nodes, args.train_limit, args.dev_limit, args.n_rand, args.mix,
        );
        std::fs::write(path, md)
            .unwrap_or_else(|e| panic!("gen_r2g_trajectories: cannot write {path}: {e}"));
        eprintln!("gen_r2g_trajectories: wrote {path}");
    }
}

// ============================================================================
// main
// ============================================================================

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();

    let budgets = parse_budgets(&args.budgets);
    if args.budgets == "100,200,400,800,1600,3200" {
        assert_eq!(
            budgets,
            BUDGET_LADDER.to_vec(),
            "gen_r2g_trajectories: --budgets' own default string and BUDGET_LADDER have drifted \
             apart from each other"
        );
    }
    // Production's classical tier (§2b's task: "read the constant from
    // saturate.rs config_for_node_count rather than hardcoding") — resolved
    // at a node count large enough to be unambiguously in the `classical()`
    // branch, so a future rebalancing of `config_for_node_count`'s
    // thresholds still resolves correctly without touching this file.
    let resolved_max_classes = args
        .max_classes
        .unwrap_or_else(|| config_for_node_count(usize::MAX).max_classes);
    eprintln!(
        "gen_r2g_trajectories: budgets={budgets:?} max_expr_nodes={} ({}) max_classes={resolved_max_classes}{}",
        args.max_expr_nodes,
        if args.max_expr_nodes == 0 {
            "no filter"
        } else {
            "filtered"
        },
        if args.max_classes.is_none() {
            " (resolved from config_for_node_count)"
        } else {
            " (explicit --max-classes)"
        },
    );

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let out_dir = PathBuf::from(&args.out_dir);
    let manifest = SplitManifest::load(&PathBuf::from(&args.manifest)).unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: failed to load split manifest {}: {e}",
            args.manifest
        )
    });

    let cache = GuideCache::load(
        Path::new(&args.strict_checkpoint),
        Path::new(&args.train_guide_report),
    );
    let mix_ratios = parse_mix_ratios(&args.mix);
    let policies = build_policies(args.n_rand, &mix_ratios);
    eprintln!(
        "gen_r2g_trajectories: {} ordering policies per expression: {:?}",
        policies.len(),
        policies
            .iter()
            .map(OrderingPolicy::label)
            .collect::<Vec<_>>()
    );

    let rules = RuleSet::production();

    let run_all = args.tier == "all";
    let mut outcomes: Vec<SplitOutcome> = Vec::new();

    if run_all || args.tier == "train" {
        let train_path = corpus_dir.join("corpus_train.bin");
        let entries = read_corpus(&train_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to read {}: {e}",
                train_path.display()
            )
        });
        let entries = stride_sample(entries, args.train_limit);
        assert_family_fence(&entries, &manifest, Tier::Train);
        eprintln!(
            "gen_r2g_trajectories: minting TRAIN ({} expressions)",
            entries.len()
        );
        outcomes.push(mint_split(
            "train",
            &entries,
            &policies,
            &cache,
            args.max_expr_nodes,
            resolved_max_classes,
            &rules,
            &budgets,
            |name| {
                let f = parse_family(name);
                (f.band as u32, f.seed)
            },
            &out_dir.join("r2g_train.jsonl"),
            &out_dir.join("r2g_trajectories_train.jsonl"),
        ));
    }

    if run_all || args.tier == "dev" {
        let dev_path = corpus_dir.join("corpus_dev.bin");
        let entries = read_corpus(&dev_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to read {}: {e}",
                dev_path.display()
            )
        });
        let entries = stride_sample(entries, args.dev_limit);
        assert_family_fence(&entries, &manifest, Tier::Dev);
        eprintln!(
            "gen_r2g_trajectories: minting DEV ({} expressions)",
            entries.len()
        );
        outcomes.push(mint_split(
            "dev",
            &entries,
            &policies,
            &cache,
            args.max_expr_nodes,
            resolved_max_classes,
            &rules,
            &budgets,
            |name| {
                let f = parse_family(name);
                (f.band as u32, f.seed)
            },
            &out_dir.join("r2g_dev.jsonl"),
            &out_dir.join("r2g_trajectories_dev.jsonl"),
        ));
    }

    if run_all || args.tier == "sh" || args.tier == "bezier" {
        let ood_path = corpus_dir.join("corpus_dev_ood.bin");
        let ood_entries = read_corpus(&ood_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to read {}: {e}",
                ood_path.display()
            )
        });
        let train_keys = train_fence_keys(&corpus_dir);

        if run_all || args.tier == "sh" {
            let entries: Vec<_> = ood_entries
                .iter()
                .filter(|(name, _)| name.starts_with("dev_sh_"))
                .cloned()
                .collect();
            assert!(
                !entries.is_empty(),
                "gen_r2g_trajectories: no dev_sh_* entries in {}",
                ood_path.display()
            );
            assert_train_fence(&entries, &train_keys, "corpus_dev_ood.bin (dev_sh_*)");
            eprintln!(
                "gen_r2g_trajectories: minting sh ({} expressions)",
                entries.len()
            );
            outcomes.push(mint_split(
                "sh",
                &entries,
                &policies,
                &cache,
                args.max_expr_nodes,
                resolved_max_classes,
                &rules,
                &budgets,
                |_name| (0u32, 0u64),
                &out_dir.join("r2g_sh.jsonl"),
                &out_dir.join("r2g_trajectories_sh.jsonl"),
            ));
        }

        if run_all || args.tier == "bezier" {
            let entries: Vec<_> = ood_entries
                .iter()
                .filter(|(name, _)| name.starts_with("dev_bezier_"))
                .cloned()
                .collect();
            assert!(
                !entries.is_empty(),
                "gen_r2g_trajectories: no dev_bezier_* entries in {}",
                ood_path.display()
            );
            assert_train_fence(&entries, &train_keys, "corpus_dev_ood.bin (dev_bezier_*)");
            eprintln!(
                "gen_r2g_trajectories: minting bezier ({} expressions)",
                entries.len()
            );
            outcomes.push(mint_split(
                "bezier",
                &entries,
                &policies,
                &cache,
                args.max_expr_nodes,
                resolved_max_classes,
                &rules,
                &budgets,
                |_name| (0u32, 0u64),
                &out_dir.join("r2g_bezier.jsonl"),
                &out_dir.join("r2g_trajectories_bezier.jsonl"),
            ));
        }
    }

    let mint_seconds = start.elapsed().as_secs_f64();

    for outcome in &outcomes {
        println!(
            "=== {} split: {} expressions ({} zero-best excluded, {} oversized-skipped), \
             {} trajectories, {} applications ===",
            outcome.tier,
            outcome.expressions,
            outcome.zero_best_excluded,
            outcome.skipped_oversized,
            outcome.trajectories,
            outcome.applications,
        );
    }
    println!("gen_r2g_trajectories: mint took {mint_seconds:.1}s total");

    // Run-level wall-clock safety ceiling, scaled by |R| (the total
    // trajectories actually minted across every split) — a generous
    // per-trajectory allowance (5s), so binding it means the run as a whole
    // ran far slower than any individual saturation call should, which
    // usually means something is stuck rather than merely slow. This is
    // distinct from `skipped_wallclock` above (which skips one over-budget
    // EXPRESSION and continues loudly): this one is checked once, after
    // every split has finished, and PANICS if it binds — NO SILENT
    // FAILURES, a suspect aggregate measurement must not be reported as if
    // it were clean.
    let total_trajectories: usize = outcomes.iter().map(|o| o.trajectories).sum();
    let run_wallclock_ceiling_secs = (total_trajectories.max(1) as f64) * 5.0;
    assert!(
        mint_seconds <= run_wallclock_ceiling_secs,
        "gen_r2g_trajectories: mint took {mint_seconds:.1}s across {total_trajectories} \
         trajectories — over the |R|-scaled wall-clock safety ceiling \
         ({run_wallclock_ceiling_secs:.1}s at 5s/trajectory). This is the RUN-LEVEL ceiling, \
         distinct from the per-expression skip reported above; binding it means the aggregate \
         run ran far slower than any per-expression skip alone explains."
    );

    write_report(
        args.report_json.as_deref(),
        args.report_md.as_deref(),
        &args,
        resolved_max_classes,
        &budgets,
        &outcomes,
        mint_seconds,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_search::egraph::RuleId;
    use pixelflow_search::egraph::{APP_CHECKPOINT_GRID, run_anytime_curve};
    use pixelflow_search::nnue::factored::EMBED_DIM;

    fn small_expr() -> (Rooted<ExprData>, Environment) {
        // (x + y) * (x + y) + 2 * (x + y) — the same fixture
        // `egraph::anytime`'s own tests use.
        let mut a = pixelflow_ir::ExprBuilder::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let s = a.push_binary(pixelflow_ir::OpKind::Add, x, y);
        let s2 = a.push_binary(pixelflow_ir::OpKind::Mul, s, s);
        let two = a.push_const(2.0);
        let ts = a.push_binary(pixelflow_ir::OpKind::Mul, two, s);
        let out = a.push_binary(pixelflow_ir::OpKind::Add, s2, ts);
        a.finish(&[out])
    }

    /// The task's required equivalence check: the `Unguided` trajectory's
    /// cost at the registered primary budget B must equal
    /// `egraph::run_anytime_curve`'s cost at the same target — both
    /// ultimately call `EGraph::saturate_until_applications` with the same
    /// arguments over the same rule set, so this is a definitional
    /// equality, pinned as a regression so a future change to either path
    /// cannot silently diverge them.
    #[test]
    fn unguided_trajectory_reproduces_run_anytime_curve_cost_at_b() {
        let (expr, env) = small_expr();
        let term = Term::new(expr.entry(), &env);
        let costs = CostModel::latency_prior();
        let cache = GuideCache::empty();

        let traj = run_trajectory(
            &OrderingPolicy::Unguided,
            &cache,
            term,
            2_000,
            &costs,
            &BUDGET_LADDER,
        );

        let grid: Vec<usize> = APP_CHECKPOINT_GRID
            .iter()
            .copied()
            .take_while(|&t| t <= B_PRIMARY)
            .collect();
        assert_eq!(
            grid.last().copied(),
            Some(B_PRIMARY),
            "B_PRIMARY must be a literal member of APP_CHECKPOINT_GRID"
        );
        let mut curve_opt = Optimizer::production()
            .cost(costs.clone())
            .budget(Budget::Explicit {
                iterations: 10_000,
                classes: 2_000,
                applications: None,
            })
            .hard_ceiling(Duration::from_secs(30));
        let curve_out = run_anytime_curve(&mut curve_opt, term, &grid);
        let cost_at_b = curve_out
            .curve
            .checkpoints
            .iter()
            .find(|cp| cp.app_target == B_PRIMARY)
            .expect("B_PRIMARY is in the grid")
            .cost
            .dag;

        assert_eq!(
            traj.checkpoints[0].cost, cost_at_b as u64,
            "the minter's Unguided trajectory must extract to the same cost at B=100 as \
             run_anytime_curve's own sample at the same target"
        );
    }

    #[test]
    fn mixed_guide_reduces_to_the_base_guide_when_epsilon_is_zero() {
        // eps = 0/1 -> Mix must reproduce PerRule's own order exactly.
        let per_rule =
            PerRuleRateGuide::from_labels(&[("a".to_string(), 0.9), ("b".to_string(), 0.1)]);
        let base = BuiltGuide::PerRule(per_rule.clone());
        let mixed = BuiltGuide::Mix {
            base: Box::new(BuiltGuide::PerRule(per_rule.clone())),
            numerator: 0,
            denominator: 1,
            seed: 42,
        };
        let candidates = vec![
            CandidateSummary {
                rule_embed: [0.0; EMBED_DIM],
                neighborhood_ops: vec![],
                budget_fraction: 0.1,
                rule: RuleId::from_label("a"),
                match_class_node_count: 1,
                expr_node_count: 1,
            },
            CandidateSummary {
                rule_embed: [0.0; EMBED_DIM],
                neighborhood_ops: vec![],
                budget_fraction: 0.1,
                rule: RuleId::from_label("b"),
                match_class_node_count: 1,
                expr_node_count: 1,
            },
        ];
        let base_scores = base.score_candidates(&candidates);
        let mixed_scores = mixed.score_candidates(&candidates);
        // Both must rank candidate 0 above candidate 1 (0.9 > 0.1), and at
        // eps=0 the mixed score is a pure (monotone) rank transform of the
        // base score, so the ORDER must match even though the raw values
        // differ (rank-normalized vs raw rate).
        assert!(base_scores[0] > base_scores[1]);
        assert!(mixed_scores[0] > mixed_scores[1]);
    }

    #[test]
    #[should_panic(expected = "nested EpsilonMix is not supported")]
    fn build_guide_refuses_a_nested_epsilon_mix() {
        // An EMPTY cache: the nested-mix refusal fires before `build_guide`
        // reaches any cache entry, so loading real checkpoint artifacts here
        // would make the test depend on generated files that are not in the
        // repository — and it would then fail on the *load*, with a
        // `should_panic` string that happens not to match, reporting a
        // missing file as a passing refusal test.
        let cache = GuideCache::empty();
        let policy = OrderingPolicy::EpsilonMix {
            guide: Box::new(OrderingPolicy::EpsilonMix {
                guide: Box::new(OrderingPolicy::PerRuleRate),
                numerator: 1,
                denominator: 4,
                seed: 1,
            }),
            numerator: 1,
            denominator: 2,
            seed: 2,
        };
        let _ = build_guide(&policy, &cache);
    }

    #[test]
    fn build_policies_produces_the_twelve_named_arms() {
        let policies = build_policies(6, &[(1, 4), (1, 4), (1, 2)]);
        let labels: Vec<String> = policies.iter().map(OrderingPolicy::label).collect();
        assert_eq!(labels.len(), 12);
        assert_eq!(labels[0], "unguided");
        assert_eq!(labels[1], "per-rule");
        assert_eq!(labels[2], "strict-v1");
        for seed in 1..=6u64 {
            assert!(labels.contains(&format!("random:{seed}")));
        }
        assert!(labels.contains(&"mix:strict-v1:1/4:1".to_string()));
        assert!(labels.contains(&"mix:strict-v1:1/4:2".to_string()));
        assert!(labels.contains(&"mix:strict-v1:1/2:1".to_string()));
    }

    /// Schema round-trip: every line this binary writes to `r2g_*.jsonl`
    /// must deserialize as `pixelflow_pipeline::training::r2g::R2gRecord`
    /// (the schema `train_guide_r2g` already consumes) and every line
    /// written to `r2g_trajectories_*.jsonl` as `TrajectoryRow` — caught
    /// here rather than only at the first real trainer run against a mint.
    #[test]
    fn minted_records_deserialize_as_r2g_record_and_trajectory_row() {
        use pixelflow_pipeline::training::r2g::R2gRecord;

        let (expr, env) = small_expr();
        let term = Term::new(expr.entry(), &env);
        let policies = vec![
            OrderingPolicy::Unguided,
            OrderingPolicy::Random(1),
            OrderingPolicy::Random(2),
        ];
        let cache = GuideCache::empty();
        let costs = CostModel::latency_prior();
        let rules = RuleSet::production();

        let mut buf: Vec<u8> = Vec::new();
        let outcome = mint_expression(
            "fixture_00000",
            term,
            0,
            0,
            "test",
            &policies,
            &cache,
            2_000,
            &costs,
            &rules,
            &BUDGET_LADDER,
            &mut buf,
        );
        assert!(!outcome.zero_best_excluded);
        assert!(outcome.applications_written > 0);
        assert_eq!(outcome.trajectory_rows.len(), policies.len());

        let text = String::from_utf8(buf).expect("valid UTF-8 JSONL");
        let mut checked = 0usize;
        for line in text.lines() {
            let record: R2gRecord = serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("line failed to deserialize as R2gRecord: {e}\nline: {line}")
            });
            assert_eq!(record.base.expr_name, "fixture_00000");
            checked += 1;
        }
        assert_eq!(checked, outcome.applications_written);

        for row in &outcome.trajectory_rows {
            let json = serde_json::to_string(row).unwrap();
            let round_tripped: TrajectoryRow = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("TrajectoryRow failed to round-trip: {e}"));
            assert_eq!(round_tripped.expr_name, row.expr_name);
        }
    }
}
