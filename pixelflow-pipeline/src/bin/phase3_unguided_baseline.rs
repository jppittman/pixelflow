//! Unguided anytime baseline curves for the Phase 3 pre-registration.
//!
//! Produces the UNGUIDED side of the pre-registered Phase 3 experiment
//! (docs/plans/2026-08-31-guide-design-revision.md §5, registration in
//! docs/plans/2026-09-01-phase3-registration.md) — the curves the budget
//! tier(s) B and the improvement threshold Y are fixed FROM, before any
//! Guide training exists. Nothing here scores, trains, or loads a Guide.
//!
//! For each sampled corpus expression this runs ONE unguided saturation
//! (full rule library) via the shared anytime loop
//! ([`pixelflow_search::egraph::anytime`] — the same single definition the
//! scoping harness and any future guided run use), sampling best extraction
//! cost (`CostModel::latency_prior()`, deterministic, no wall-clock in any
//! reported number) at the geometric application-count grid
//! [`pixelflow_search::egraph::APP_CHECKPOINT_GRID`].
//!
//! From those curves it reports, per production tier and per candidate
//! budget B, the **truncation loss**: how much worse unguided-at-B (and
//! unguided-at-B/2) is than unguided-at-4B, as
//! `(cost@B - cost@4B) / cost@4B`. A budget tier where truncation
//! demonstrably loses quality is where a Guide has something to buy back;
//! if truncation loses nothing at any feasible B, that is a
//! stop-the-presses finding for the whole Phase 3 experiment and this
//! binary's summary says so in as many words.
//!
//! Corpus discipline: reads `corpus_train.bin` + `corpus_dev.bin` ONLY.
//! `corpus_final.bin` is never opened here — FINAL stays untouched until a
//! publication run, per the extraction-head split discipline.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin phase3_unguided_baseline -- \
//!     --corpus-dir pixelflow-pipeline/data --samples 400 \
//!     --out-csv docs/results/2026-09-01-phase3-unguided-baseline.csv \
//!     --out-json docs/results/2026-09-01-phase3-unguided-baseline.json
//! ```

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, Budget, CostModel, Optimizer, SaturationStop,
    config_for_node_count, run_anytime_curve,
};

#[derive(Parser)]
#[command(name = "phase3_unguided_baseline")]
#[command(
    about = "Phase 3 pre-registration: unguided anytime curves + truncation loss at candidate budgets"
)]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    /// (`corpus_final.bin` is never read — FINAL stays untouched.)
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Number of expressions to measure, stratified by size across the full
    /// TRAIN+DEV population (must be >= 300 per the task spec; fail loud,
    /// never silently under-measure).
    #[arg(long, default_value_t = 400)]
    samples: usize,

    /// Full per-expression, per-checkpoint curve CSV.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-unguided-baseline.csv"
    )]
    out_csv: String,

    /// Aggregate truncation-loss tables as JSON.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-unguided-baseline.json"
    )]
    out_json: String,
}

/// Floor on `--samples`, from the recalibration task spec.
const MIN_SAMPLES: usize = 300;

/// Per-curve wall-clock safety ceiling — fails loud inside the shared curve
/// runner (panic), never truncates silently, never appears in any number.
const SAFETY_TIMEOUT: Duration = Duration::from_secs(300);

/// Generous sweep safety ceiling per curve; hitting it is a distinct,
/// reported stop reason, not a silent truncation.
const SWEEP_SAFETY_CEILING: usize = 10_000;

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
        SaturationStop::IterationCeiling => "sweep_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

struct ExprCurve {
    name: String,
    origin: &'static str, // "train" | "dev"
    tier: &'static str,
    node_count: usize,
    /// Cost at each grid target, in grid order (clamped rows carry the
    /// frozen final cost, so this is total on the grid).
    cost_at: Vec<usize>,
    /// Actual application count at each grid target's sample.
    apps_at: Vec<usize>,
    ended: SaturationStop,
    ended_at_apps: usize,
    /// Per-checkpoint stop reason, in grid order. A checkpoint is LIVE — its
    /// cost can still move at a later grid target — exactly when it stopped
    /// on [`SaturationStop::ApplicationBudget`], which is what
    /// `AnytimeCheckpoint::stop`'s own doc says. `ended_at_apps > b` is not
    /// the same predicate: the budget is crossed at rule-sweep granularity,
    /// so a run can quiesce while overshooting B (the B=100 checkpoint
    /// finishing quiesced at 150 applications), and counting that finalized
    /// checkpoint as live inflates `live_at_B` and the live-conditioned
    /// loss statistics with rows whose cost cannot change.
    stop_at: Vec<SaturationStop>,
}

/// Truncation loss in percent: how much worse `cost_b` is than `cost_4b`.
/// A positive cost against a free (0-cost) reference is unboundedly worse,
/// not 0% (same convention as the scoping harness's regret, PR #1067 P2).
fn loss_pct(cost_b: usize, cost_4b: usize) -> f64 {
    if cost_4b == 0 {
        if cost_b == 0 { 0.0 } else { f64::INFINITY }
    } else {
        (cost_b as f64 - cost_4b as f64) / cost_4b as f64 * 100.0
    }
}

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

/// One percentage aggregate as JSON. Every one of these can legitimately be
/// infinite under the zero-reference convention (`cost@4B` reached 0 while
/// `cost@B` was positive), and Rust formats an infinite `f64` as a bare
/// `inf`, which is not JSON — a report that parses on some runs and not
/// others is a silent failure of the reporting, so all of them go through
/// one encoder and become `null`. `inf_count` distinguishes "infinite" from
/// "absent".
fn json_pct(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        "null".to_string()
    }
}

/// Aggregate truncation-loss stats for one (scope, B) cell.
struct LossCell {
    n: usize,
    /// Expressions still live at B (run ended strictly after B applications
    /// — the only ones whose cost could still have moved past B).
    live_at_b: usize,
    inf_count: usize,
    positive_count: usize,
    median: f64,
    p90: f64,
    max: f64,
    mean_finite: f64,
    /// Same stats conditioned on live-at-B expressions only.
    live_median: f64,
    live_positive_count: usize,
}

fn loss_cell(rows: &[(f64, bool)]) -> LossCell {
    // rows: (loss_pct, live_at_b)
    let n = rows.len();
    assert!(n > 0, "loss_cell over empty scope");
    let mut all: Vec<f64> = rows.iter().map(|(l, _)| *l).collect();
    all.sort_by(|a, b| a.partial_cmp(b).expect("loss NaN"));
    let finite: Vec<f64> = all.iter().copied().filter(|l| l.is_finite()).collect();
    let live: Vec<f64> = rows.iter().filter(|(_, lv)| *lv).map(|(l, _)| *l).collect();
    let mut live_sorted = live.clone();
    live_sorted.sort_by(|a, b| a.partial_cmp(b).expect("loss NaN"));
    LossCell {
        n,
        live_at_b: live.len(),
        inf_count: all.iter().filter(|l| l.is_infinite()).count(),
        positive_count: all.iter().filter(|l| **l > 1e-9).count(),
        median: percentile(&all, 0.5),
        p90: percentile(&all, 0.9),
        max: *all.last().expect("non-empty"),
        mean_finite: if finite.is_empty() {
            0.0
        } else {
            finite.iter().sum::<f64>() / finite.len() as f64
        },
        live_median: if live_sorted.is_empty() {
            0.0
        } else {
            percentile(&live_sorted, 0.5)
        },
        live_positive_count: live.iter().filter(|l| **l > 1e-9).count(),
    }
}

fn main() {
    let args = Args::parse();
    assert!(
        args.samples >= MIN_SAMPLES,
        "phase3_unguided_baseline: --samples {} < the {MIN_SAMPLES} floor the registration \
         requires — fail loud, never silently under-measure",
        args.samples
    );

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let mut entries: Vec<(&'static str, String, ExprArena, ExprId)> = Vec::new();
    for (origin, file) in [("train", "corpus_train.bin"), ("dev", "corpus_dev.bin")] {
        let path = corpus_dir.join(file);
        let tier_entries = read_corpus(&path).unwrap_or_else(|e| {
            panic!(
                "phase3_unguided_baseline: failed to read {}: {e}",
                path.display()
            )
        });
        for (name, arena, root) in tier_entries {
            entries.push((origin, name, arena, root));
        }
    }
    let total_available = entries.len();
    assert!(
        total_available >= args.samples,
        "phase3_unguided_baseline: corpus has only {total_available} TRAIN+DEV expressions, \
         need >= {} — regenerate a larger corpus",
        args.samples
    );

    // Stratified-by-size sampling: sort the whole TRAIN+DEV population by
    // node count (name as deterministic tiebreak), then stride-sample so
    // every size stratum is represented proportionally. Deterministic:
    // depends only on corpus content.
    entries.sort_by(|a, b| {
        let na = a.2.len();
        let nb = b.2.len();
        na.cmp(&nb).then_with(|| a.1.cmp(&b.1))
    });
    let stride = entries.len() as f64 / args.samples as f64;
    let mut sampled: Vec<&(&'static str, String, ExprArena, ExprId)> =
        Vec::with_capacity(args.samples);
    for i in 0..args.samples {
        let idx = ((i as f64) * stride) as usize;
        sampled.push(&entries[idx.min(entries.len() - 1)]);
    }
    eprintln!(
        "phase3_unguided_baseline: {} of {total_available} TRAIN+DEV expressions, \
         size-stratified (stride {:.2})",
        sampled.len(),
        stride
    );

    let costs = CostModel::latency_prior();
    let grid = APP_CHECKPOINT_GRID;
    let mut curves: Vec<ExprCurve> = Vec::with_capacity(sampled.len());

    for (i, (origin, name, arena, root)) in sampled.iter().enumerate() {
        let node_count = arena.len();
        let class_cap = config_for_node_count(node_count).max_classes;
        // The curve's environment, named outright: this expression's own
        // production class cap, a generous sweep ceiling, and no application
        // cap of the optimizer's own — `run_anytime_curve` supplies the
        // application dimension one checkpoint gap at a time. `hard_ceiling`
        // is the wall clock's only remaining role after #1118: a panicking
        // safety ceiling, never a stop reason.
        let mut optimizer = Optimizer::production()
            .cost(costs.clone())
            .budget(Budget::Explicit {
                iterations: SWEEP_SAFETY_CEILING,
                classes: class_cap,
                applications: None,
            })
            .hard_ceiling(SAFETY_TIMEOUT);
        let AnytimeCurveOutput { curve, .. } =
            run_anytime_curve(&mut optimizer, arena, *root, grid);
        curves.push(ExprCurve {
            name: name.clone(),
            origin,
            tier: tier_name(node_count),
            node_count,
            // DAG cost — what the emitted kernel pays (#1117), not the
            // DP's tree total.
            cost_at: curve.checkpoints.iter().map(|c| c.cost.dag).collect(),
            apps_at: curve.checkpoints.iter().map(|c| c.app_actual).collect(),
            ended: curve.ended,
            ended_at_apps: curve.ended_at_apps,
            stop_at: curve.checkpoints.iter().map(|c| c.stop).collect(),
        });
        if (i + 1) % 50 == 0 || i + 1 == sampled.len() {
            eprintln!(
                "phase3_unguided_baseline: {}/{} curves done",
                i + 1,
                sampled.len()
            );
        }
    }

    // ------------------------------------------------------------------
    // CSV: full curves.
    // ------------------------------------------------------------------
    let out_csv = PathBuf::from(&args.out_csv);
    if let Some(parent) = out_csv.parent() {
        std::fs::create_dir_all(parent).expect("create CSV output directory");
    }
    let mut f = std::fs::File::create(&out_csv)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_csv.display()));
    writeln!(
        f,
        "expr_name,origin,tier,node_count,app_target,app_actual,cost,ended,ended_at_apps"
    )
    .unwrap();
    for c in &curves {
        for (j, &target) in grid.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{}",
                c.name,
                c.origin,
                c.tier,
                c.node_count,
                target,
                c.apps_at[j],
                c.cost_at[j],
                stop_name(c.ended),
                c.ended_at_apps,
            )
            .unwrap();
        }
    }
    println!(
        "wrote {} curve rows to {}",
        curves.len() * grid.len(),
        out_csv.display()
    );

    // ------------------------------------------------------------------
    // Feasibility: how runs ended, per tier + work distribution.
    // ------------------------------------------------------------------
    let tiers = ["blitz", "rapid", "classical"];
    println!("\n=== how unguided runs ended (per tier) ===");
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        let sub: Vec<&ExprCurve> = curves
            .iter()
            .filter(|c| *scope == "ALL" || c.tier == *scope)
            .collect();
        if sub.is_empty() {
            continue;
        }
        let n = sub.len();
        let count = |s: SaturationStop| sub.iter().filter(|c| c.ended == s).count();
        let mut apps: Vec<f64> = sub.iter().map(|c| c.ended_at_apps as f64).collect();
        apps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {scope:<10} n={n:<4} quiesced={} class_cap={} grid_exhausted={} sweep_ceiling={} | \
             ended_at_apps: p25={:.0} median={:.0} p75={:.0} p95={:.0} max={:.0}",
            count(SaturationStop::Quiesced),
            count(SaturationStop::ClassCap),
            count(SaturationStop::ApplicationBudget),
            count(SaturationStop::IterationCeiling),
            percentile(&apps, 0.25),
            percentile(&apps, 0.5),
            percentile(&apps, 0.75),
            percentile(&apps, 0.95),
            percentile(&apps, 1.0),
        );
    }

    // ------------------------------------------------------------------
    // Truncation loss per candidate B: cost@B (and cost@B/2) vs cost@4B.
    // Candidate B are grid targets with both B/2 and 4B on the grid.
    // ------------------------------------------------------------------
    let idx_of: BTreeMap<usize, usize> = grid.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let candidates: Vec<usize> = grid
        .iter()
        .copied()
        .filter(|b| b % 2 == 0 && idx_of.contains_key(&(b / 2)) && idx_of.contains_key(&(b * 4)))
        .collect();

    println!(
        "\n=== truncation loss: unguided-at-B vs unguided-at-4B (loss% = (cost@B - cost@4B)/cost@4B) ==="
    );
    let mut json_cells: Vec<String> = Vec::new();
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        let sub: Vec<&ExprCurve> = curves
            .iter()
            .filter(|c| *scope == "ALL" || c.tier == *scope)
            .collect();
        if sub.is_empty() {
            continue;
        }
        println!("--- scope: {scope} (n={}) ---", sub.len());
        println!(
            "  {:>7} {:>5} {:>7} {:>8} {:>8} {:>8} {:>9} {:>8} | {:>9} {:>10} | {:>12}",
            "B",
            "n",
            "live@B",
            "median%",
            "p90%",
            "max%",
            "mean_fin%",
            ">0 cnt",
            "live_med%",
            "live>0 cnt",
            "halfB_med%"
        );
        for &b in &candidates {
            let bi = idx_of[&b];
            let hi = idx_of[&(b / 2)];
            let fi = idx_of[&(b * 4)];
            let rows: Vec<(f64, bool)> = sub
                .iter()
                .map(|c| {
                    (
                        loss_pct(c.cost_at[bi], c.cost_at[fi]),
                        c.stop_at[bi] == SaturationStop::ApplicationBudget,
                    )
                })
                .collect();
            let half_rows: Vec<(f64, bool)> = sub
                .iter()
                .map(|c| {
                    (
                        loss_pct(c.cost_at[hi], c.cost_at[fi]),
                        c.stop_at[hi] == SaturationStop::ApplicationBudget,
                    )
                })
                .collect();
            let cell = loss_cell(&rows);
            let half_cell = loss_cell(&half_rows);
            println!(
                "  {b:>7} {:>5} {:>7} {:>8.3} {:>8.3} {:>8.1} {:>9.3} {:>8} | {:>9.3} {:>10} | {:>12.3}",
                cell.n,
                cell.live_at_b,
                cell.median,
                cell.p90,
                cell.max,
                cell.mean_finite,
                cell.positive_count,
                cell.live_median,
                cell.live_positive_count,
                half_cell.median,
            );
            json_cells.push(format!(
                "    {{\"scope\": \"{scope}\", \"B\": {b}, \"n\": {}, \"live_at_B\": {}, \
                 \"inf_count\": {}, \"positive_count\": {}, \"median_pct\": {}, \
                 \"p90_pct\": {}, \"max_pct\": {}, \"mean_finite_pct\": {}, \
                 \"live_median_pct\": {}, \"live_positive_count\": {}, \
                 \"half_b_median_pct\": {}, \"half_b_positive_count\": {}}}",
                cell.n,
                cell.live_at_b,
                cell.inf_count,
                cell.positive_count,
                json_pct(cell.median),
                json_pct(cell.p90),
                json_pct(cell.max),
                json_pct(cell.mean_finite),
                json_pct(cell.live_median),
                cell.live_positive_count,
                json_pct(half_cell.median),
                half_cell.positive_count,
            ));
        }
    }
    println!(
        "\n  (live@B = run ended after more than B applications; expressions already \
         quiesced/capped at <= B have truncation loss exactly 0 by construction. \
         inf loss = cost@4B reached 0 while cost@B was positive; excluded from mean_fin, \
         counted in >0 cnt. EVERY non-finite aggregate is written as JSON null — read \
         inf_count to tell an infinite aggregate from a missing one.)"
    );

    // Stop-the-presses check, in as many words (task spec: report loudly).
    let any_loss = curves.iter().any(|c| {
        candidates.iter().any(|&b| {
            let bi = idx_of[&b];
            let fi = idx_of[&(b * 4)];
            loss_pct(c.cost_at[bi], c.cost_at[fi]) > 1e-9
        })
    });
    if !any_loss {
        println!(
            "\n*** STOP THE PRESSES: no expression shows ANY truncation loss at ANY candidate B \
             on this corpus — unguided saturation at every tested budget already reaches its \
             4B-budget quality. The Guide has nothing to buy back here; the Phase 3 experiment \
             as designed has no headroom on this corpus and the registration must say so. ***"
        );
    }

    // ------------------------------------------------------------------
    // JSON aggregates.
    // ------------------------------------------------------------------
    let out_json = PathBuf::from(&args.out_json);
    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent).expect("create JSON output directory");
    }
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"num_expressions\": {},\n", curves.len()));
    json.push_str(&format!(
        "  \"corpus_train_dev_total\": {total_available},\n"
    ));
    json.push_str(&format!("  \"grid\": {grid:?},\n"));
    json.push_str(&format!("  \"any_truncation_loss\": {any_loss},\n"));
    json.push_str("  \"per_expression\": [\n");
    for (i, c) in curves.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"name\": {:?}, \"origin\": \"{}\", \"tier\": \"{}\", \"node_count\": {}, \
             \"ended\": \"{}\", \"ended_at_apps\": {}, \"cost_at\": {:?}}}{}\n",
            c.name,
            c.origin,
            c.tier,
            c.node_count,
            stop_name(c.ended),
            c.ended_at_apps,
            c.cost_at,
            if i + 1 < curves.len() { "," } else { "" }
        ));
    }
    json.push_str("  ],\n");
    json.push_str("  \"truncation_loss\": [\n");
    json.push_str(&json_cells.join(",\n"));
    json.push_str("\n  ]\n}\n");
    let mut jf = std::fs::File::create(&out_json)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_json.display()));
    jf.write_all(json.as_bytes())
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out_json.display()));
    eprintln!("phase3_unguided_baseline: wrote {}", out_json.display());
}
