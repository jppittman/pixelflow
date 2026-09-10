//! Round 2 UNGUIDED anytime curves vs rule count, modes (i)/(ii)
//! (`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2, §5, §9 step 2).
//!
//! This is the pre-registration data: unguided-only curves at every |R|
//! grid point of mode (i) (exact duplicates) and mode (ii) (mechanical
//! compositions), on the Round-1 baseline sample (same 400-expression
//! stratified stride over TRAIN+DEV, reproduced here exactly as
//! `phase3_unguided_baseline.rs` computes it — corpus content is the only
//! input, so the same stride yields the same 400 names). No guided run
//! anywhere in this binary.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
//!     --corpus-dir pixelflow-pipeline/data --samples 400 \
//!     --out-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv \
//!     --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.json
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

use pixelflow_ir::{Environment, ExprData, Rooted, Term};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, Budget, CostModel, EGraph, Optimizer, Rewrite,
    RuleSet, SaturationStop, config_for_node_count, run_anytime_curve,
};
use pixelflow_search::math::inflate::{RuleSetSpec, build_rule_set, rule_set_fingerprint};

/// Applications recorded by exactly one full rule sweep, on a fresh e-graph
/// for `(arena, root)` under `rules` — a throwaway probe, discarded
/// immediately after, never sharing state with the real curve run.
/// `max_iters = 1` guarantees the sweep runs to completion (or to
/// quiescence, which still completes rule 0's pass before checking) rather
/// than being cut off by the application budget: no application cap is
/// passed, so only the iteration ceiling can stop it.
fn apps_per_sweep_probe(term: Term<'_>, rules: Vec<Box<dyn Rewrite>>, max_classes: usize) -> usize {
    let mut egraph = EGraph::with_rules(rules);
    pixelflow_search::egraph::insert_term(
        term,
        &mut egraph,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let _ = egraph.saturate_budgeted(1, max_classes, None);
    egraph.application_count() as usize
}

#[derive(Parser)]
#[command(name = "phase3_round2_unguided_curves")]
#[command(about = "Round 2: UNGUIDED anytime curves at every |R| grid point of modes (i)/(ii)")]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    /// (`corpus_final.bin` is never read — FINAL stays untouched.)
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Number of expressions to measure — must match Round 1's 400 to
    /// reproduce its stride exactly (fail loud, never silently diverge).
    #[arg(long, default_value_t = 400)]
    samples: usize,

    /// Full per-expression, per-checkpoint, per-rule-set curve CSV.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv"
    )]
    out_csv: String,

    /// Aggregate tables as JSON.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.json"
    )]
    out_json: String,

    /// Rule-set specs to run, comma-separated (parsed by
    /// `RuleSetSpec::parse`). Default: the full §3 grid for modes (i)/(ii),
    /// with `base` (|R|=62) run once and shared.
    #[arg(
        long,
        default_value = "base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,comp:186,comp:248",
        value_delimiter = ','
    )]
    rule_sets: Vec<String>,
}

/// Floor on `--samples`, matching Round 1.
const MIN_SAMPLES: usize = 300;

/// Per-curve wall-clock safety ceiling for the |R|=62 baseline — scaled by
/// `|R|/62` per §5, so more rules do not make an honest curve panic. Still
/// PANICS when it binds (offline measurement fails loud, never truncates
/// silently).
const BASE_SAFETY_TIMEOUT_SECS: u64 = 300;

/// Generous sweep safety ceiling per curve; hitting it is a distinct,
/// reported stop reason, not a silent truncation. Scaled the same way.
const BASE_SWEEP_SAFETY_CEILING: usize = 10_000;

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
    origin: &'static str,
    tier: &'static str,
    node_count: usize,
    cost_at: Vec<usize>,
    apps_at: Vec<usize>,
    /// Cumulative completed sweeps at each grid checkpoint (diagnostic,
    /// mirrors `apps_at`) — divides `evals_at` into "raw matches enumerated
    /// per round" per §7.1.
    sweeps_at: Vec<usize>,
    /// Cumulative rule-match attempts (`EGraph::total_evals`) at each grid
    /// checkpoint — "raw matches enumerated" per §7.1; unguided has no
    /// scorer, so `evals_at[j] / apps_at[j]` stands in for a scored-
    /// candidate-per-application count.
    evals_at: Vec<usize>,
    /// Applications recorded in exactly one full sweep of this rule set on
    /// this expression, measured by a separate one-sweep probe run before
    /// the real curve (§9 step 2's "apps_per_sweep" addition) — lets every
    /// `app_actual` be read off in SWEEPS, not just applications, per the
    /// v2 registration's binding rule that every budget is reported both
    /// ways.
    apps_per_sweep: usize,
    ended: SaturationStop,
    ended_at_apps: usize,
}

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

fn main() {
    let args = Args::parse();
    assert!(
        args.samples >= MIN_SAMPLES,
        "phase3_round2_unguided_curves: --samples {} < the {MIN_SAMPLES} floor — fail loud, \
         never silently under-measure",
        args.samples
    );

    // ------------------------------------------------------------------
    // Reproduce Round 1's exact stratified-by-size sample.
    // ------------------------------------------------------------------
    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let mut entries: Vec<(&'static str, String, Rooted<ExprData>)> = Vec::new();
    for (origin, file) in [("train", "corpus_train.bin"), ("dev", "corpus_dev.bin")] {
        let path = corpus_dir.join(file);
        let tier_entries = read_corpus(&path).unwrap_or_else(|e| {
            panic!(
                "phase3_round2_unguided_curves: failed to read {}: {e}",
                path.display()
            )
        });
        for (name, expr) in tier_entries {
            entries.push((origin, name, expr));
        }
    }
    let total_available = entries.len();
    assert!(
        total_available >= args.samples,
        "phase3_round2_unguided_curves: corpus has only {total_available} TRAIN+DEV \
         expressions, need >= {} — regenerate a larger corpus",
        args.samples
    );
    entries.sort_by(|a, b| {
        let na = a.2.len();
        let nb = b.2.len();
        na.cmp(&nb).then_with(|| a.1.cmp(&b.1))
    });
    let stride = entries.len() as f64 / args.samples as f64;
    let mut sampled: Vec<&(&'static str, String, Rooted<ExprData>)> =
        Vec::with_capacity(args.samples);
    for i in 0..args.samples {
        let idx = ((i as f64) * stride) as usize;
        sampled.push(&entries[idx.min(entries.len() - 1)]);
    }
    eprintln!(
        "phase3_round2_unguided_curves: {} of {total_available} TRAIN+DEV expressions, \
         size-stratified (stride {:.2}) — reproduces Round 1's sample",
        sampled.len(),
        stride
    );

    let costs = CostModel::latency_prior();
    let grid = APP_CHECKPOINT_GRID;

    let out_csv = PathBuf::from(&args.out_csv);
    if let Some(parent) = out_csv.parent() {
        std::fs::create_dir_all(parent).expect("create CSV output directory");
    }
    let mut csv_f = std::fs::File::create(&out_csv)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_csv.display()));
    writeln!(
        csv_f,
        "rule_set,num_rules,fingerprint,expr_name,origin,tier,node_count,app_target,app_actual,\
         sweeps_actual,evals_actual,apps_per_sweep,cost,ended,ended_at_apps"
    )
    .unwrap();

    let mut json_rule_sets: Vec<String> = Vec::new();

    for spec_str in &args.rule_sets {
        let spec = RuleSetSpec::parse(spec_str).unwrap_or_else(|e| {
            panic!("phase3_round2_unguided_curves: bad --rule-sets entry {spec_str:?}: {e}")
        });
        let rules = match build_rule_set(&spec) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "phase3_round2_unguided_curves: SKIPPING {spec_str:?} — {e} (never padded)"
                );
                continue;
            }
        };
        let num_rules = rules.len();
        let fingerprint = rule_set_fingerprint(&rules);
        let scale = num_rules as f64 / 62.0;
        let safety_timeout = Duration::from_secs((BASE_SAFETY_TIMEOUT_SECS as f64 * scale) as u64);
        let sweep_ceiling = ((BASE_SWEEP_SAFETY_CEILING as f64) * scale) as usize;

        eprintln!(
            "=== rule_set={spec_str} |R|={num_rules} fingerprint={fingerprint} \
             safety_timeout={safety_timeout:?} sweep_ceiling={sweep_ceiling} ==="
        );

        // A corpus entry declares no buffers or uniforms — the format
        // refuses to write one down — so one empty environment serves every
        // term.
        let env = Environment::new();
        let mut curves: Vec<ExprCurve> = Vec::with_capacity(sampled.len());
        for (i, (origin, name, expr)) in sampled.iter().enumerate() {
            let term = Term::new(expr.entry(), &env);
            let node_count = expr.len();
            let class_cap = config_for_node_count(node_count).max_classes;
            // Every rule set needs its own fresh Vec<Box<dyn Rewrite>> — the
            // e-graph consumes it. Rebuilding per expression, per rule-set,
            // is the cost this binary pays for `build_rule_set`'s honesty
            // (never silently reusing a stale rule set across runs).
            let probe_rules = build_rule_set(&spec)
                .unwrap_or_else(|e| panic!("rule set became unbuildable mid-run: {e}"));
            let apps_per_sweep = apps_per_sweep_probe(term, probe_rules, class_cap);
            let rules_for_this_expr = build_rule_set(&spec)
                .unwrap_or_else(|e| panic!("rule set became unbuildable mid-run: {e}"));
            // This arm's rule set names the arm: `Optimizer::rules` is the
            // one place a non-production vocabulary enters, and the
            // optimizer's own `fingerprint()` then identifies the
            // configuration the curve was measured under.
            let mut optimizer = Optimizer::production()
                .rules(RuleSet::new(rules_for_this_expr))
                .cost(costs.clone())
                .budget(Budget::Explicit {
                    iterations: sweep_ceiling,
                    classes: class_cap,
                    applications: None,
                })
                .hard_ceiling(safety_timeout);
            let AnytimeCurveOutput { curve, .. } = run_anytime_curve(&mut optimizer, term, grid);
            curves.push(ExprCurve {
                name: name.clone(),
                origin,
                tier: tier_name(node_count),
                node_count,
                // DAG cost — what the emitted kernel pays (#1117).
                cost_at: curve.checkpoints.iter().map(|c| c.cost.dag).collect(),
                apps_at: curve.checkpoints.iter().map(|c| c.app_actual).collect(),
                sweeps_at: curve.checkpoints.iter().map(|c| c.sweeps).collect(),
                evals_at: curve.checkpoints.iter().map(|c| c.evals_actual).collect(),
                apps_per_sweep,
                ended: curve.ended,
                ended_at_apps: curve.ended_at_apps,
            });
            if (i + 1) % 25 == 0 || i + 1 == sampled.len() {
                eprintln!(
                    "  rule_set={spec_str}: {}/{} curves done",
                    i + 1,
                    sampled.len()
                );
            }
        }

        for c in &curves {
            for (j, &target) in grid.iter().enumerate() {
                writeln!(
                    csv_f,
                    "{spec_str},{num_rules},{fingerprint},{},{},{},{},{},{},{},{},{},{},{},{}",
                    c.name,
                    c.origin,
                    c.tier,
                    c.node_count,
                    target,
                    c.apps_at[j],
                    c.sweeps_at[j],
                    c.evals_at[j],
                    c.apps_per_sweep,
                    c.cost_at[j],
                    stop_name(c.ended),
                    c.ended_at_apps,
                )
                .unwrap();
            }
        }
        csv_f.flush().unwrap();

        // Per-rule-set summary printed immediately (journal-friendly even
        // if a later rule set panics on its safety ceiling).
        let tiers = ["blitz", "rapid", "classical"];
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
                "  [{spec_str}] {scope:<10} n={n:<4} quiesced={} class_cap={} \
                 grid_exhausted={} sweep_ceiling={} | ended_at_apps: p25={:.0} median={:.0} \
                 p75={:.0} p95={:.0} max={:.0}",
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

        // cost@B (B=100,200) median/quartiles per band, and truncation loss
        // vs 4B where both checkpoints exist on the grid.
        let idx_of: std::collections::BTreeMap<usize, usize> =
            grid.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        for &b in &[100usize, 200usize] {
            let Some(&bi) = idx_of.get(&b) else { continue };
            for scope in ["ALL"].iter().chain(tiers.iter()) {
                let sub: Vec<&ExprCurve> = curves
                    .iter()
                    .filter(|c| *scope == "ALL" || c.tier == *scope)
                    .collect();
                if sub.is_empty() {
                    continue;
                }
                let mut costs_at_b: Vec<f64> = sub.iter().map(|c| c.cost_at[bi] as f64).collect();
                costs_at_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let loss = idx_of.get(&(b * 4)).map(|&fi| {
                    let rows: Vec<f64> = sub
                        .iter()
                        .map(|c| loss_pct(c.cost_at[bi], c.cost_at[fi]))
                        .collect();
                    let mut sorted = rows.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    percentile(&sorted, 0.5)
                });
                println!(
                    "  [{spec_str}] B={b:<4} {scope:<10} n={:<4} cost@B: p25={:.1} \
                     median={:.1} p75={:.1} | truncation_loss_median%={:?}",
                    sub.len(),
                    percentile(&costs_at_b, 0.25),
                    percentile(&costs_at_b, 0.5),
                    percentile(&costs_at_b, 0.75),
                    loss,
                );
            }
        }

        // JSON per-expression rows for this rule set.
        let mut per_expr_json = String::new();
        for (i, c) in curves.iter().enumerate() {
            per_expr_json.push_str(&format!(
                "      {{\"name\": {:?}, \"origin\": \"{}\", \"tier\": \"{}\", \
                 \"node_count\": {}, \"ended\": \"{}\", \"ended_at_apps\": {}, \
                 \"apps_per_sweep\": {}, \"cost_at\": {:?}, \"app_actual_at\": {:?}, \
                 \"sweeps_actual_at\": {:?}, \"evals_actual_at\": {:?}}}{}\n",
                c.name,
                c.origin,
                c.tier,
                c.node_count,
                stop_name(c.ended),
                c.ended_at_apps,
                c.apps_per_sweep,
                c.cost_at,
                c.apps_at,
                c.sweeps_at,
                c.evals_at,
                if i + 1 < curves.len() { "," } else { "" }
            ));
        }
        json_rule_sets.push(format!(
            "  {{\"rule_set\": {spec_str:?}, \"num_rules\": {num_rules}, \"fingerprint\": \
             \"{fingerprint}\", \"per_expression\": [\n{per_expr_json}    ]}}"
        ));
    }

    println!("wrote curve rows to {}", out_csv.display());

    let out_json = PathBuf::from(&args.out_json);
    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent).expect("create JSON output directory");
    }
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"num_expressions\": {},\n", sampled.len()));
    json.push_str(&format!(
        "  \"corpus_train_dev_total\": {total_available},\n"
    ));
    json.push_str(&format!("  \"grid\": {grid:?},\n"));
    json.push_str("  \"rule_sets\": [\n");
    json.push_str(&json_rule_sets.join(",\n"));
    json.push_str("\n  ]\n}\n");
    let mut jf = std::fs::File::create(&out_json)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_json.display()));
    jf.write_all(json.as_bytes())
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out_json.display()));
    eprintln!(
        "phase3_round2_unguided_curves: wrote {}",
        out_json.display()
    );
}
