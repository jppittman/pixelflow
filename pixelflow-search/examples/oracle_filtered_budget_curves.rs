//! Anytime budget curves for e-graph saturation: unguided vs oracle-filtered.
//!
//! Guided-saturation SCOPING measurement (Phase 3 prep,
//! docs/plans/2026-07-07-guided-saturation-redesign.md /
//! docs/plans/2026-08-05-egraph-nnue-research-workflow.md). Answers: **if a
//! perfect Guide replayed only the load-bearing rewrite applications, what
//! happens to e-graph size, saturation work, and extraction quality?** This
//! bounds what ANY learned Guide can deliver.
//!
//! # Why "anytime curves", not "full vs truncated vs oracle" regimes
//!
//! The kernel language is strongly normalizing (bounded `Reduce`, no
//! recursion) and the optimizer inherits that stance: saturation runs under a
//! budget tier and spends it, full stop — it does not detect or rely on
//! reaching a fixed point, and a fixed point is not a certified state
//! anywhere in this pipeline. So there is no privileged "C_full" to call the
//! reference and everything else "regret against". This harness instead runs
//! ONE thing per rule set: a single incremental saturation from empty,
//! sampling extraction cost at standardized work checkpoints along the way.
//! That is the anytime curve. Two curves are run per expression —
//! **unguided** (the full 62-rule library,
//! `pixelflow_search::math::all_rules()`) and **oracle-filtered** (only the
//! rules a hindsight pass over the unguided run's best-found extraction
//! marked load-bearing) — and the reference for regret at any work level is
//! the lowest cost either curve ever reaches, empirically, never a claimed
//! optimum.
//!
//! # Checkpoint grid: applications, not sweep fractions (2026-09-01 recalibration)
//!
//! The 2026-08-30 run of this harness came back null: its grid (fractions
//! 0.25x..3x of the per-tier nominal *iteration* budget) started past the
//! point where 97.8% of expressions had already quiesced or hit their class
//! cap, so every curve was flat before the first sample
//! (docs/results/2026-08-30-oracle-filtered-budget-curves.md). Checkpoints
//! are now denominated in **cumulative rule applications** — the same unit
//! the Phase 3 registration denominates budgets in — on a geometric grid
//! ([`pixelflow_search::egraph::APP_CHECKPOINT_GRID`]) that resolves both
//! the median-~195-applications regime and the heavy tail. The curve loop
//! itself lives in `pixelflow_search::egraph::anytime` (one definition,
//! imported — the pipeline's baseline binary uses the identical loop), and
//! the class cap is the production tier's fixed memory-protection cap: under
//! application denomination the x-axis measures work actually done, so the
//! per-checkpoint class-cap scaling the PR #1067 review demanded of the old
//! fraction grid is superseded, not dropped — see the module doc of
//! `anytime.rs` for the argument.
//!
//! # Oracle replay: the approximation actually taken
//!
//! Exact per-application replay is structurally hard here and the task this
//! harness answers anticipates that: "if exact replay is hard, approximate
//! honestly and say how." `EClassId`s are allocated sequentially as
//! `EGraph::add` is called; a recorded `ApplicationRecord::match_root` names
//! a class in the ORIGINAL run's id space, and skipping even one
//! non-load-bearing application during a replay shifts every later
//! `EClassId` by one, so "replay only these `ApplicationId`s on a fresh
//! graph" has no stable target to aim `apply_single_rule` at without a
//! content-addressed match key this architecture does not have (adding one
//! would be a `provenance.rs` redesign, out of scope this round).
//!
//! So this harness filters at RULE granularity instead of match granularity:
//! `R* = { rule_idx : some load-bearing application in the unguided run's
//! best-found state fired this rule }`, then a fresh e-graph's rule LIST is
//! restricted to `R*` for the whole oracle curve. This sidesteps id stability
//! entirely (no cross-run `EClassId`/`ApplicationId` needs to survive), and
//! it makes the "transitive closure of match dependencies" concern the task
//! flags moot: filtering keeps or drops a whole rule, so a chain "rule A's
//! product enables rule B's match" survives automatically whenever both A and
//! B are in `R*` — no explicit dependency-closure computation needed. It is
//! coarser than the task's literal ask in exactly one, safe direction: it
//! retains EVERY firing of an oracle-approved rule, including firings that
//! individually would have been wasted, so oracle's work numbers are a
//! pessimistic (upper-bound) estimate of what a perfect application-level
//! oracle would need. This also matches how the Guide is motivated in the
//! redesign doc ("Can rule filtering keep the e-graph within budget...") —
//! global rule masking is the mechanism the thesis names, not a
//! simplification of it.
//!
//! # Non-direct-creator share (redesign.md lines 88-90 follow-up)
//!
//! Additive, read-only measurement, no change to `derivation_ancestors`'s
//! logic: for every application the labeler marks `LoadBearing` (against the
//! unguided run's best-found extraction), this harness checks whether the
//! e-node(s) that application actually CREATED are among the literal
//! chosen-extraction tags (walked the same way `labeler::chosen_tagged_nodes`
//! does, via the public `find`/`nodes`/`tags` API). The fraction that are
//! NOT is reported as `credited_non_direct_ratio`.
//!
//! **This does not isolate any single over-approximation axis** (a PR
//! review caught an earlier version of this doc/code claiming it did).
//! `derivation_ancestors` documents three (`provenance.rs`): (1) crediting
//! every node in a class, not just the chosen one, (2) pulling in union
//! events by class membership, (3) no fixed-point pruning. A load-bearing
//! application whose created node isn't on the literal chosen path could be
//! there via any of the three, OR because it is a genuine transitive
//! enabler the labeler is *correctly* crediting (real "A enables B" credit
//! chess-style provenance can't observe any other way — see the redesign
//! plan's "observed credit" section). This measurement cannot tell those
//! apart; it answers "what fraction of load-bearing applications are not the
//! literal creator of a chosen node," not "what fraction is over-crediting
//! via class membership specifically." Distinguishing genuine transitive
//! credit from each named over-approximation axis would need per-application
//! provenance of *why* it entered the ancestry closure, which
//! `derivation_ancestors` does not currently record and this round does not
//! add. The one new API surface, `Provenance::origins()`, is a read-only
//! iterator mirroring the existing `applications()`/`union_events()`
//! accessors — no semantic change.
//!
//! # "Quiesced before cap" — a diagnostic, not an organizing axis
//!
//! For each curve this harness also records whether `saturate_with_limits`
//! stopped producing new nodes (`total_unions == 0`) strictly before the
//! expression's own nominal tier budget was spent. This is reported purely
//! as an emergent fact (how often does a production-sized budget have slack
//! left over, and does that correlate with expression size) — it never
//! partitions the corpus or gates which numbers get analyzed.
//!
//! # Shaders in-sample
//!
//! Five named, hand-written realistic kernels (`swirl`, `circle_sdf`, `poly`,
//! `redundant`, `normalize` — the same five `pixelflow-search/examples/
//! rule_report.rs` uses) are run through the identical pipeline alongside the
//! synthetic corpus, and reported against the synthetic corpus's per-tier
//! distribution, so the synthetic-corpus conclusions aren't only validated on
//! synthetic shapes.
//!
//! All measurements are deterministic counts and static-latency-prior cost
//! units (`CostModel::latency_prior()`). The only `Duration` in play is a
//! 300s per-curve safety ceiling that PANICS if it ever binds (inside
//! `run_anytime_curve`) — wall-clock plays no role in any reported number.
//!
//! Run: `cargo run --release -p pixelflow-search --example oracle_filtered_budget_curves`
//! Output: prints a summary to stdout and writes a full per-expression,
//! per-curve, per-checkpoint CSV to `--out` (default
//! `docs/results/2026-09-01-oracle-filtered-budget-curves.csv`).

use std::collections::{BTreeSet, HashMap};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use pixelflow_ir::arena::ExprNode;
use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, CostModel, EClassId, EGraph, ENodeId, EpisodeLabels,
    Origin, Rewrite, SaturationStop, all_rules, collect_rule_templates, config_for_node_count,
    run_anytime_curve,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

/// Safety ceiling only — applies to one expression's WHOLE curve (the
/// `anytime` module computes one deadline per curve and passes the remaining
/// duration to each saturation call, then PANICS if it ever binds — offline
/// measurement fails loud, never silently truncates).
const SAFETY_TIMEOUT: Duration = Duration::from_secs(300);

/// Generous safety ceiling on total sweeps per curve. With the budget
/// denominated in applications, sweeps are bounded by quiescence/class-cap in
/// practice; this only exists so a pathological expression cannot loop
/// unboundedly, and hitting it is reported as a distinct stop reason.
const SWEEP_SAFETY_CEILING: usize = 10_000;

// ============================================================================
// Corpus generation
// ============================================================================

struct Band {
    max_depth: usize,
    leaf_prob: f32,
    num_vars: usize,
}

/// Mirrors the shape (not the exact list) of `pixelflow-pipeline/src/bin/
/// gen_bench_corpus.rs`'s `BANDS` — depth/leaf/var knobs spanning tiny to
/// very-deep expressions, so the corpus naturally spans all three production
/// time-control tiers (blitz/rapid/classical).
const BANDS: &[Band] = &[
    Band {
        max_depth: 2,
        leaf_prob: 0.55,
        num_vars: 2,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.50,
        num_vars: 2,
    },
    Band {
        max_depth: 4,
        leaf_prob: 0.45,
        num_vars: 3,
    },
    Band {
        max_depth: 5,
        leaf_prob: 0.40,
        num_vars: 4,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.35,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.30,
        num_vars: 4,
    },
    Band {
        max_depth: 10,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 12,
        leaf_prob: 0.20,
        num_vars: 4,
    },
    Band {
        max_depth: 15,
        leaf_prob: 0.18,
        num_vars: 4,
    },
    Band {
        max_depth: 18,
        leaf_prob: 0.15,
        num_vars: 4,
    },
];

/// 10 bands x 22 = 220 >= the task's 200 floor.
const SAMPLES_PER_BAND: usize = 22;

/// Fixed seed: the whole synthetic corpus (and therefore every measurement
/// below) is reproducible byte-for-byte from this one constant.
const BASE_SEED: u64 = 20260830;

/// Compact the reachable subtree of `root` in `src` into a fresh, minimal
/// arena, remapping `ExprId`s along the way.
///
/// `EGraph::add_arena` walks `0..arena.len()` unconditionally (it requires
/// topological order from 0, not just reachability from `root`).
/// `BwdGenerator::generate_arena` packs BOTH the optimized and unoptimized
/// subtree of one `BwdTrainingPairArena` side by side in a single arena, so
/// passing that raw arena straight to `add_arena` would silently add the
/// disconnected sibling subtree's nodes into the e-graph too, polluting every
/// work/cost measurement with rule matches against an expression nobody
/// asked to saturate (the same B3 bug `pixelflow-pipeline`'s corpus writer
/// documents closing).
fn compact_subtree(
    src: &ExprArena,
    root: ExprId,
    dst: &mut ExprArena,
    memo: &mut HashMap<u32, ExprId>,
) -> ExprId {
    if let Some(&id) = memo.get(&root.0) {
        return id;
    }
    let node = src.node(root).clone();
    let new_id = match node {
        ExprNode::Var(v) => dst.push_var(v),
        ExprNode::Const(v) => dst.push_const(v),
        ExprNode::Unary(op, c) => {
            let nc = compact_subtree(src, c, dst, memo);
            dst.push_unary(op, nc)
        }
        ExprNode::Binary(op, a, b) => {
            let na = compact_subtree(src, a, dst, memo);
            let nb = compact_subtree(src, b, dst, memo);
            dst.push_binary(op, na, nb)
        }
        ExprNode::Ternary(op, a, b, c) => {
            let na = compact_subtree(src, a, dst, memo);
            let nb = compact_subtree(src, b, dst, memo);
            let nc = compact_subtree(src, c, dst, memo);
            dst.push_ternary(op, na, nb, nc)
        }
        ExprNode::Nary(op, _, _) => {
            let kids: Vec<ExprId> = src
                .children(root)
                .map(|c| compact_subtree(src, c, dst, memo))
                .collect();
            dst.push_nary(op, &kids)
        }
        ExprNode::Param(i) => panic!(
            "compact_subtree: unexpected Param({i}) in a BwdGenerator expression \
             (the generator never produces Param nodes)"
        ),
        ExprNode::Buffer(_) => panic!(
            "compact_subtree: unexpected Buffer node in a BwdGenerator expression \
             (the generator never produces memory ops)"
        ),
    };
    memo.insert(root.0, new_id);
    new_id
}

struct CorpusItem {
    name: String,
    arena: ExprArena,
    root: ExprId,
    node_count: usize,
    /// `true` for the five named realistic kernels (the "shaders in-sample"
    /// check); `false` for synthetic band-generated expressions.
    is_named_shader: bool,
}

fn build_synthetic_corpus() -> Vec<CorpusItem> {
    let templates = collect_rule_templates();
    let mut generator = BwdGenerator::new(BASE_SEED, BwdGenConfig::default(), templates);
    let mut corpus = Vec::with_capacity(BANDS.len() * SAMPLES_PER_BAND);

    for (band_idx, band) in BANDS.iter().enumerate() {
        generator.config = BwdGenConfig {
            max_depth: band.max_depth,
            leaf_prob: band.leaf_prob,
            num_vars: band.num_vars,
            ..BwdGenConfig::default()
        };
        for sample in 0..SAMPLES_PER_BAND {
            let pair = generator.generate_arena();
            let mut dst = ExprArena::with_capacity(pair.arena.node_count_subtree(pair.unoptimized));
            let mut memo = HashMap::new();
            let new_root = compact_subtree(&pair.arena, pair.unoptimized, &mut dst, &mut memo);
            let node_count = dst.len();
            corpus.push(CorpusItem {
                name: format!("band{band_idx}_depth{}_s{sample}", band.max_depth),
                arena: dst,
                root: new_root,
                node_count,
                is_named_shader: false,
            });
        }
    }
    corpus
}

/// The five named realistic kernels from `rule_report.rs`, for the
/// shaders-in-sample check.
fn named_shader_corpus() -> Vec<CorpusItem> {
    fn swirl() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let yy = a.push_binary(OpKind::Mul, y, y);
        let d = a.push_binary(OpKind::Add, xx, yy);
        let s = a.push_unary(OpKind::Sqrt, d);
        let kf = a.push_const(3.0);
        let sf = a.push_binary(OpKind::Mul, s, kf);
        let sn = a.push_unary(OpKind::Sin, sf);
        let ka = a.push_const(0.5);
        let prod = a.push_binary(OpKind::Mul, sn, ka);
        let kb = a.push_const(0.5);
        let out = a.push_binary(OpKind::Add, prod, kb);
        (a, out)
    }
    fn circle_sdf() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let cx = a.push_const(0.3);
        let cy = a.push_const(-0.2);
        let dx = a.push_binary(OpKind::Sub, x, cx);
        let dy = a.push_binary(OpKind::Sub, y, cy);
        let dx2 = a.push_binary(OpKind::Mul, dx, dx);
        let dy2 = a.push_binary(OpKind::Mul, dy, dy);
        let sum = a.push_binary(OpKind::Add, dx2, dy2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let r = a.push_const(0.5);
        let out = a.push_binary(OpKind::Sub, dist, r);
        (a, out)
    }
    fn poly() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let ka = a.push_const(2.0);
        let kb = a.push_const(-3.0);
        let kc = a.push_const(1.0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let ax2 = a.push_binary(OpKind::Mul, ka, xx);
        let bx = a.push_binary(OpKind::Mul, kb, x);
        let s1 = a.push_binary(OpKind::Add, ax2, bx);
        let out = a.push_binary(OpKind::Add, s1, kc);
        (a, out)
    }
    fn redundant() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let s = a.push_binary(OpKind::Add, x, y);
        let s2 = a.push_binary(OpKind::Mul, s, s);
        let two = a.push_const(2.0);
        let ts = a.push_binary(OpKind::Mul, two, s);
        let out = a.push_binary(OpKind::Add, s2, ts);
        (a, out)
    }
    fn normalize() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let yy = a.push_binary(OpKind::Mul, y, y);
        let d = a.push_binary(OpKind::Add, xx, yy);
        let s = a.push_unary(OpKind::Sqrt, d);
        let out = a.push_binary(OpKind::Div, x, s);
        (a, out)
    }

    type ShaderBuilder = fn() -> (ExprArena, ExprId);
    let cases: Vec<(&str, ShaderBuilder)> = vec![
        ("shader_swirl", swirl),
        ("shader_circle_sdf", circle_sdf),
        ("shader_poly", poly),
        ("shader_redundant", redundant),
        ("shader_normalize", normalize),
    ];
    cases
        .into_iter()
        .map(|(name, build)| {
            let (arena, root) = build();
            let node_count = arena.node_count_subtree(root);
            CorpusItem {
                name: name.to_string(),
                arena,
                root,
                node_count,
                is_named_shader: true,
            }
        })
        .collect()
}

fn tier_name(node_count: usize) -> &'static str {
    match node_count {
        0..=10 => "blitz",
        11..=50 => "rapid",
        _ => "classical",
    }
}

// ============================================================================
// Chosen-tag walk (mirrors `labeler::chosen_tagged_nodes`, public API only)
// ============================================================================

fn chosen_tags(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> BTreeSet<ENodeId> {
    let mut visited: BTreeSet<EClassId> = BTreeSet::new();
    let mut stack = vec![root];
    let mut tags = BTreeSet::new();
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!("chosen_tags: e-class {idx} reachable from root has no recorded choice")
        });
        let nodes = egraph.nodes(canonical);
        let node_tags = egraph.tags(canonical);
        tags.insert(node_tags[node_idx]);
        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }
    tags
}

// ============================================================================
// Per-expression measurement (curve loop imported from `egraph::anytime`)
// ============================================================================

fn stop_name(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        SaturationStop::ApplicationBudget => "app_budget",
        SaturationStop::ClassCap => "class_cap",
        SaturationStop::IterationCeiling => "sweep_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

#[derive(Clone, Debug)]
struct CurveRow {
    expr_name: String,
    tier: &'static str,
    node_count: usize,
    curve: &'static str, // "unguided" | "oracle"
    app_target: usize,
    app_actual: usize,
    sweeps: usize,
    rules_allowed: usize,
    classes: usize,
    nodes: usize,
    cost: usize,
    stop: &'static str,
    clamped: bool,
    regret_pct: f64,
}

struct ExprMeasurement {
    rows: Vec<CurveRow>,
    /// How each curve's run ended (quiesced / class cap / grid exhausted /
    /// sweep ceiling) — a distinct status per the 2026-08-30 report's
    /// design-implication #3, no longer inferred from class-count forensics.
    unguided_ended: SaturationStop,
    oracle_ended: SaturationStop,
    unguided_ended_at_apps: usize,
    /// Number of distinct cost values along the unguided curve — the direct
    /// "did the grid see any curve shape at all" recalibration check (the
    /// 2026-08-30 run had 1 everywhere).
    unguided_distinct_costs: usize,
    /// Relative gap between the first checkpoint's cost and the final cost
    /// on the unguided curve (>0 means the curve has shape).
    first_to_final_gap_pct: f64,
    load_bearing: usize,
    total_applications: usize,
    credited_non_direct: usize,
}

fn measure_expression(item: &CorpusItem) -> ExprMeasurement {
    let tier = tier_name(item.node_count);
    let config = config_for_node_count(item.node_count);
    // Fixed environment cap (production tier value) for the whole curve —
    // see the module doc's recalibration section for why this replaces the
    // old per-checkpoint class-cap scaling.
    let class_cap = config.max_classes;
    let costs = CostModel::latency_prior();

    let full_rules = all_rules();
    let full_rules_count = full_rules.len();
    let unguided: AnytimeCurveOutput = run_anytime_curve(
        &item.arena,
        item.root,
        full_rules,
        APP_CHECKPOINT_GRID,
        class_cap,
        SWEEP_SAFETY_CEILING,
        SAFETY_TIMEOUT,
        &costs,
    );

    let labels = EpisodeLabels::compute(
        &unguided.egraph,
        unguided.extraction.root,
        &unguided.extraction.choices,
    );
    let oracle_rule_idxs: BTreeSet<usize> = labels
        .load_bearing
        .iter()
        .filter_map(|app_id| unguided.egraph.provenance().application(*app_id))
        .map(|record| record.rule_idx)
        .collect();
    let oracle_rules: Vec<Box<dyn Rewrite>> = all_rules()
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| oracle_rule_idxs.contains(idx))
        .map(|(_, r)| r)
        .collect();
    let oracle_rules_len = oracle_rules.len();

    let oracle: AnytimeCurveOutput = run_anytime_curve(
        &item.arena,
        item.root,
        oracle_rules,
        APP_CHECKPOINT_GRID,
        class_cap,
        SWEEP_SAFETY_CEILING,
        SAFETY_TIMEOUT,
        &costs,
    );

    // Over-approximation looseness.
    let chosen = chosen_tags(
        &unguided.egraph,
        unguided.extraction.root,
        &unguided.extraction.choices,
    );
    let mut created_by: HashMap<u64, Vec<ENodeId>> = HashMap::new();
    for (enode_id, origin) in unguided.egraph.provenance().origins() {
        if let Origin::Rule(app_id) = origin {
            created_by
                .entry(app_id.as_u64())
                .or_default()
                .push(enode_id);
        }
    }
    let mut credited_non_direct = 0usize;
    for app_id in &labels.load_bearing {
        let created = created_by.get(&app_id.as_u64());
        let on_chosen_path = created
            .map(|nodes| nodes.iter().any(|n| chosen.contains(n)))
            .unwrap_or(false);
        if !on_chosen_path {
            credited_non_direct += 1;
        }
    }

    let uc = &unguided.curve.checkpoints;
    let oc = &oracle.curve.checkpoints;
    let global_best = uc
        .iter()
        .chain(oc.iter())
        .map(|c| c.cost)
        .min()
        .expect("grid non-empty => at least one checkpoint per curve");

    // `global_best == 0` means SOME checkpoint (either curve, any target)
    // extracted to a free `Const`/`Var`. A checkpoint that matches that
    // (cost == 0 too) is exactly as good -- 0% regret, correct. A checkpoint
    // with positive cost is not "equally good relative to a free reference"
    // -- ANY nonzero cost is unboundedly worse than free, so report INF
    // rather than silently erasing the regret (PR #1067 finding, kept).
    let regret = |cost: usize| -> f64 {
        if global_best == 0 {
            if cost == 0 { 0.0 } else { f64::INFINITY }
        } else {
            (cost as f64 - global_best as f64) / global_best as f64 * 100.0
        }
    };

    let first_cost = uc.first().expect("non-empty grid").cost;
    let final_cost = uc.last().expect("non-empty grid").cost;
    let first_to_final_gap_pct = if first_cost == 0 {
        0.0
    } else {
        (first_cost as f64 - final_cost as f64) / first_cost as f64 * 100.0
    };
    let unguided_distinct_costs = {
        let set: BTreeSet<usize> = uc.iter().map(|c| c.cost).collect();
        set.len()
    };

    let mut rows = Vec::with_capacity(uc.len() + oc.len());
    for (curve_name, rules_allowed, cps) in [
        ("unguided", full_rules_count, uc),
        ("oracle", oracle_rules_len, oc),
    ] {
        for c in cps.iter() {
            rows.push(CurveRow {
                expr_name: item.name.clone(),
                tier,
                node_count: item.node_count,
                curve: curve_name,
                app_target: c.app_target,
                app_actual: c.app_actual,
                sweeps: c.sweeps,
                rules_allowed,
                classes: c.classes,
                nodes: c.nodes,
                cost: c.cost,
                stop: stop_name(c.stop),
                clamped: c.clamped,
                regret_pct: regret(c.cost),
            });
        }
    }

    ExprMeasurement {
        rows,
        unguided_ended: unguided.curve.ended,
        oracle_ended: oracle.curve.ended,
        unguided_ended_at_apps: unguided.curve.ended_at_apps,
        unguided_distinct_costs,
        first_to_final_gap_pct,
        load_bearing: labels.load_bearing.len(),
        total_applications: unguided.egraph.provenance().application_count(),
        credited_non_direct,
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("docs/results/2026-09-01-oracle-filtered-budget-curves.csv")
        });

    let rule_count = all_rules().len();
    println!("rule library size: {rule_count} rules");
    println!("checkpoint grid (applications): {APP_CHECKPOINT_GRID:?}");

    let mut corpus = build_synthetic_corpus();
    let synthetic_count = corpus.len();
    corpus.extend(named_shader_corpus());
    eprintln!(
        "corpus: {} synthetic + {} named shaders = {} expressions across {} bands",
        synthetic_count,
        corpus.len() - synthetic_count,
        corpus.len(),
        BANDS.len()
    );
    assert!(
        synthetic_count >= 200,
        "task requires >=200 synthetic corpus expressions, got {synthetic_count}"
    );

    let mut all_rows: Vec<CurveRow> = Vec::new();
    let mut end_diag: Vec<(String, &'static str, SaturationStop, SaturationStop, usize)> =
        Vec::new();
    let mut shape_diag: Vec<(String, &'static str, usize, f64)> = Vec::new();
    let mut looseness_diag: Vec<(String, &'static str, usize, usize, usize)> = Vec::new();

    for (i, item) in corpus.iter().enumerate() {
        let m = measure_expression(item);
        all_rows.extend(m.rows);
        let tier = tier_name(item.node_count);
        end_diag.push((
            item.name.clone(),
            tier,
            m.unguided_ended,
            m.oracle_ended,
            m.unguided_ended_at_apps,
        ));
        shape_diag.push((
            item.name.clone(),
            tier,
            m.unguided_distinct_costs,
            m.first_to_final_gap_pct,
        ));
        looseness_diag.push((
            item.name.clone(),
            tier,
            m.load_bearing,
            m.total_applications,
            m.credited_non_direct,
        ));
        if (i + 1) % 40 == 0 {
            eprintln!("... {}/{} expressions done", i + 1, corpus.len());
        }
    }

    // ------------------------------------------------------------------
    // CSV
    // ------------------------------------------------------------------
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let mut f = std::fs::File::create(&out_path).expect("create output CSV");
    writeln!(
        f,
        "expr_name,tier,node_count,curve,app_target,app_actual,sweeps,rules_allowed,classes,nodes,cost,stop,clamped,regret_pct"
    )
    .unwrap();
    for r in &all_rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4}",
            r.expr_name,
            r.tier,
            r.node_count,
            r.curve,
            r.app_target,
            r.app_actual,
            r.sweeps,
            r.rules_allowed,
            r.classes,
            r.nodes,
            r.cost,
            r.stop,
            r.clamped,
            r.regret_pct,
        )
        .unwrap();
    }
    println!("wrote {} rows to {}", all_rows.len(), out_path.display());

    // ------------------------------------------------------------------
    // Summary: regret% by curve x application target, overall and per tier.
    // Synthetic corpus only (named shaders reported separately).
    // ------------------------------------------------------------------
    let synthetic_names: BTreeSet<&str> = corpus
        .iter()
        .filter(|c| !c.is_named_shader)
        .map(|c| c.name.as_str())
        .collect();
    let tiers = ["blitz", "rapid", "classical"];

    println!(
        "\n=== anytime curves: regret% vs applications, synthetic corpus (n={synthetic_count}) ==="
    );
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        println!("--- scope: {scope} ---");
        for curve in ["unguided", "oracle"] {
            for &target in APP_CHECKPOINT_GRID {
                let mut regrets: Vec<f64> = Vec::new();
                let mut live = 0usize;
                for r in all_rows.iter().filter(|r| {
                    r.curve == curve
                        && r.app_target == target
                        && synthetic_names.contains(r.expr_name.as_str())
                        && (*scope == "ALL" || r.tier == *scope)
                }) {
                    regrets.push(r.regret_pct);
                    if !r.clamped {
                        live += 1;
                    }
                }
                if regrets.is_empty() {
                    continue;
                }
                regrets.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mean = regrets.iter().sum::<f64>() / regrets.len() as f64;
                let median = regrets[regrets.len() / 2];
                println!(
                    "  {curve:<9} B={target:<7} n={:<4} live={live:<4} regret%: mean={mean:>8.2} median={median:>7.2}",
                    regrets.len(),
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Diagnostic: how runs ended (explicit stop reasons, not forensics).
    // ------------------------------------------------------------------
    println!("\n=== diagnostic: run end status (unguided curve) ===");
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        let rows: Vec<_> = end_diag
            .iter()
            .filter(|(name, t, ..)| {
                synthetic_names.contains(name.as_str()) && (*scope == "ALL" || t == scope)
            })
            .collect();
        if rows.is_empty() {
            continue;
        }
        let n = rows.len();
        let count = |s: SaturationStop| rows.iter().filter(|(_, _, u, _, _)| *u == s).count();
        let mut ended_apps: Vec<f64> = rows.iter().map(|(_, _, _, _, a)| *a as f64).collect();
        ended_apps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {scope:<10} n={n:<4} quiesced={} class_cap={} grid_exhausted={} sweep_ceiling={}  \
             ended-at-apps median={:.0} p90={:.0}",
            count(SaturationStop::Quiesced),
            count(SaturationStop::ClassCap),
            count(SaturationStop::ApplicationBudget),
            count(SaturationStop::IterationCeiling),
            ended_apps[ended_apps.len() / 2],
            ended_apps[(ended_apps.len() * 9) / 10],
        );
    }

    // ------------------------------------------------------------------
    // Diagnostic: did the recalibrated grid see curve shape at all?
    // (The 2026-08-30 fraction grid saw exactly one distinct cost per
    // expression, everywhere — the null result this recalibration fixes.)
    // ------------------------------------------------------------------
    println!("\n=== diagnostic: unguided curve shape (distinct costs across checkpoints) ===");
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        let rows: Vec<_> = shape_diag
            .iter()
            .filter(|(name, t, ..)| {
                synthetic_names.contains(name.as_str()) && (*scope == "ALL" || t == scope)
            })
            .collect();
        if rows.is_empty() {
            continue;
        }
        let n = rows.len();
        let with_shape = rows.iter().filter(|(_, _, d, _)| *d > 1).count();
        let gaps: Vec<f64> = rows.iter().map(|(_, _, _, g)| *g).collect();
        let mut sorted = gaps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {scope:<10} n={n:<4} curves-with-shape: {with_shape}/{n} ({:.1}%)  \
             first-to-final gap: mean={:.2}% median={:.2}%",
            with_shape as f64 / n as f64 * 100.0,
            gaps.iter().sum::<f64>() / n as f64,
            sorted[n / 2],
        );
    }

    // ------------------------------------------------------------------
    // Over-approximation looseness.
    // ------------------------------------------------------------------
    let synth_loose: Vec<_> = looseness_diag
        .iter()
        .filter(|(name, ..)| synthetic_names.contains(name.as_str()))
        .collect();
    let total_lb: usize = synth_loose.iter().map(|(_, _, lb, _, _)| lb).sum();
    let total_credited: usize = synth_loose.iter().map(|(_, _, _, _, c)| c).sum();
    println!("\n=== non-direct-creator share of load-bearing applications (redesign.md 88-90) ===");
    println!(
        "  overall: {total_credited}/{total_lb} ({:.2}%) of load-bearing applications did not \
         literally create a node on the chosen-extraction path (does NOT isolate class-membership \
         over-approximation specifically -- see module doc's \"Non-direct-creator share\" section; \
         genuine transitive-enabler credit and all three named over-approximation axes are lumped \
         together here)",
        total_credited as f64 / total_lb.max(1) as f64 * 100.0,
    );
    for t in tiers {
        let lb: usize = synth_loose
            .iter()
            .filter(|(_, tt, ..)| *tt == t)
            .map(|(_, _, lb, _, _)| lb)
            .sum();
        let c: usize = synth_loose
            .iter()
            .filter(|(_, tt, ..)| *tt == t)
            .map(|(_, _, _, _, c)| c)
            .sum();
        if lb > 0 {
            println!("  {t:<10}: {c}/{lb} ({:.2}%)", c as f64 / lb as f64 * 100.0);
        }
    }

    // ------------------------------------------------------------------
    // Shaders in-sample.
    // ------------------------------------------------------------------
    // Compared at the FINAL grid target: every curve has exactly one row per
    // grid target (clamped rows freeze the end state), so the last target is
    // the "all the budget this harness ever grants" point — the analogue of
    // the old fraction grid's 1.0x row.
    let final_target = *APP_CHECKPOINT_GRID
        .last()
        .expect("checkpoint grid is non-empty");
    println!("\n=== shaders in-sample check (named realistic kernels vs synthetic corpus) ===");
    for item in corpus.iter().filter(|c| c.is_named_shader) {
        let tier = tier_name(item.node_count);
        let synth_regret_at_final: Vec<f64> = all_rows
            .iter()
            .filter(|r| {
                r.tier == tier
                    && r.curve == "unguided"
                    && r.app_target == final_target
                    && synthetic_names.contains(r.expr_name.as_str())
            })
            .map(|r| r.regret_pct)
            .collect();
        let (lo, hi) = if synth_regret_at_final.is_empty() {
            (0.0, 0.0)
        } else {
            let mut s = synth_regret_at_final.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (s[0], s[s.len() - 1])
        };
        let uc_regret = all_rows
            .iter()
            .find(|r| {
                r.expr_name == item.name && r.curve == "unguided" && r.app_target == final_target
            })
            .map(|r| r.regret_pct)
            .unwrap_or(f64::NAN);
        let oc_regret = all_rows
            .iter()
            .find(|r| {
                r.expr_name == item.name && r.curve == "oracle" && r.app_target == final_target
            })
            .map(|r| r.regret_pct)
            .unwrap_or(f64::NAN);
        let in_sample = uc_regret >= lo && uc_regret <= hi;
        println!(
            "  {:<20} nodes={:<4} tier={:<10} unguided_regret@end={:>7.2}%  oracle_regret@end={:>7.2}%  \
             synthetic-{tier}-tier range=[{lo:.2}%,{hi:.2}%] in-sample={}",
            item.name, item.node_count, tier, uc_regret, oc_regret, in_sample,
        );
    }
}
