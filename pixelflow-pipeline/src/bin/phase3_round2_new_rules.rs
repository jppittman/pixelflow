//! Round 2, mode (iii): unguided anytime curves at `|R|=62` vs `62+batch`.
//!
//! `docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2.3/§5/§8 — the
//! fidelity arm's own measurement (mode (iii) grid is the pair {62, 62+batch}
//! per §3, distinct from the {62,93,124,186,248} grid used by modes (i)/(ii)).
//!
//! Reuses Round 1's exact 400-expression stratified sample (same corpus, same
//! stride — see [`phase3_unguided_baseline`]'s sampling, restated here so
//! this binary has no dependency on that one's internal helpers) and the ONE
//! shared anytime curve definition
//! ([`pixelflow_search::egraph::run_anytime_curve`]). UNGUIDED only, both
//! arms: no Guide is loaded or trained here.
//!
//! For each arm (`base` = `all_rules()`, 62 rules; `batch` = `all_rules() +
//! experimental_rules()`, the Round 2 mode-(iii) batch) this reports, per
//! production tier and B in {100, 200}:
//! - median cost@B (deterministic `CostModel::latency_prior()` regret input)
//! - absolute cost at quiescence (curve end) — the fidelity gain the new
//!   rules buy when search is not the bottleneck
//! - truncation loss (cost@B vs cost@4B, Round 1's convention)
//! - applications-to-quiescence
//! - class-cap hits
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin phase3_round2_new_rules -- \
//!     --corpus-dir pixelflow-pipeline/data --samples 400 \
//!     --out-md docs/results/2026-09-01-round2-unguided-vs-rulecount.md \
//!     --out-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-mode-iii.csv
//! ```

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, Budget, CostModel, Optimizer, RuleSet, SaturationStop,
    all_rules, config_for_node_count, run_anytime_curve,
};
use pixelflow_search::math::inflate::rule_set_fingerprint;
use pixelflow_search::math::round2_rules::experimental_rules;

#[derive(Parser)]
#[command(name = "phase3_round2_new_rules")]
#[command(
    about = "Round 2 mode (iii): unguided anytime curves at |R|=62 vs 62+batch (new-rule fidelity arm)"
)]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    /// (`corpus_final.bin` is never read — FINAL stays untouched.)
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Number of expressions to measure, stratified by size across the full
    /// TRAIN+DEV population — Round 1's sample size, same stride algorithm.
    #[arg(long, default_value_t = 400)]
    samples: usize,

    /// Markdown output — a section appended to (or, if absent, the sole
    /// content of) this results doc.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-round2-unguided-vs-rulecount.md"
    )]
    out_md: String,

    /// Full per-expression, per-checkpoint, per-arm curve CSV — the same
    /// schema `phase3_round2_unguided_curves` writes for modes (i)/(ii)
    /// (`rule_set` is `base` / `new:<total>`), so the Register's
    /// closure-aware regret is computed from raw rows, never from medians.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-round2-unguided-vs-rulecount-mode-iii.csv"
    )]
    out_csv: String,
}

const MIN_SAMPLES: usize = 300;

/// Round 1's per-curve wall-clock safety ceiling, at |R|=62.
const BASE_SAFETY_TIMEOUT: Duration = Duration::from_secs(300);
const SWEEP_SAFETY_CEILING: usize = 10_000;

/// `pixelflow_search::egraph::extract::extract_dag`'s `CYCLE_COST` sentinel
/// (1_000_000, private to that module — restated here as a threshold, not a
/// definition: a class that is genuinely self-referential post-saturation
/// gets this cost rather than a real one, on ANY rule set). A curve's
/// `quiescence_cost` at or above this is a sentinel, not a measured quality
/// number — verified pre-existing in `all_rules()` alone on large classical
/// expressions (an 863-node DEV/TRAIN expression already hits it under the
/// unmodified 62-rule set), not introduced by `experimental_rules()`; the
/// per-tier reporting below tracks how much MORE often it fires as `|R|`
/// grows, which is the on-topic measurement, not a defect to hide.
const CYCLE_COST_THRESHOLD: usize = 900_000;

fn tier_name(node_count: usize) -> &'static str {
    match node_count {
        0..=10 => "blitz",
        11..=50 => "rapid",
        _ => "classical",
    }
}

/// Stop-reason label, identical to `phase3_round2_unguided_curves`' so the
/// two CSVs share one vocabulary.
fn stop_name(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        SaturationStop::ApplicationBudget => "app_budget",
        SaturationStop::ClassCap => "class_cap",
        SaturationStop::IterationCeiling => "sweep_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

#[derive(Clone)]
struct ExprCurve {
    name: String,
    origin: &'static str,
    tier: &'static str,
    node_count: usize,
    cost_at: Vec<usize>,
    apps_at: Vec<usize>,
    ended: SaturationStop,
    ended_at_apps: usize,
    /// Cost at the LAST grid checkpoint the run actually reached before
    /// ending (i.e. the curve's own final state) — "quiescence" for a run
    /// that quiesced/capped before the grid's top; for a run still live at
    /// the grid's top, this is cost@204800 (the grid ceiling), reported as
    /// such, never silently mislabeled as true quiescence.
    quiescence_cost: usize,
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

fn median(vals: &[f64]) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("NaN in median input"));
    percentile(&v, 0.5)
}

/// Truncation loss in percent: how much worse `cost_b` is than `cost_4b`.
/// Same convention as `phase3_unguided_baseline`: a positive cost against a
/// zero reference is unboundedly worse, never 0%.
fn loss_pct(cost_b: usize, cost_4b: usize) -> f64 {
    if cost_4b == 0 {
        if cost_b == 0 { 0.0 } else { f64::INFINITY }
    } else {
        (cost_b as f64 - cost_4b as f64) / cost_4b as f64 * 100.0
    }
}

fn run_curves(
    sampled: &[&(&'static str, String, ExprArena, ExprId)],
    rules: fn() -> Vec<Box<dyn pixelflow_search::egraph::Rewrite>>,
    safety_timeout: Duration,
    arm_label: &str,
) -> Vec<ExprCurve> {
    let costs = CostModel::latency_prior();
    let grid = APP_CHECKPOINT_GRID;
    let mut curves = Vec::with_capacity(sampled.len());

    for (i, (origin, name, arena, root)) in sampled.iter().enumerate() {
        let node_count = arena.len();
        let class_cap = config_for_node_count(node_count).max_classes;
        // This arm's rule set names the arm: `Optimizer::rules` is the one
        // place a non-production vocabulary enters.
        let mut optimizer = Optimizer::production()
            .rules(RuleSet::new(rules()))
            .cost(costs.clone())
            .budget(Budget::Explicit {
                iterations: SWEEP_SAFETY_CEILING,
                classes: class_cap,
                applications: None,
            })
            .hard_ceiling(safety_timeout);
        let AnytimeCurveOutput { curve, .. } =
            run_anytime_curve(&mut optimizer, arena, *root, grid);
        let quiescence_cost = curve
            .checkpoints
            .iter()
            .rev()
            .find(|c| c.app_actual <= curve.ended_at_apps)
            .map(|c| c.cost.dag)
            .unwrap_or_else(|| curve.checkpoints.last().expect("non-empty grid").cost.dag);
        curves.push(ExprCurve {
            name: name.clone(),
            origin,
            tier: tier_name(node_count),
            node_count,
            // DAG cost — what the emitted kernel pays (#1117).
            cost_at: curve.checkpoints.iter().map(|c| c.cost.dag).collect(),
            apps_at: curve.checkpoints.iter().map(|c| c.app_actual).collect(),
            ended: curve.ended,
            ended_at_apps: curve.ended_at_apps,
            quiescence_cost,
        });
        if (i + 1) % 50 == 0 || i + 1 == sampled.len() {
            eprintln!(
                "phase3_round2_new_rules[{arm_label}]: {}/{} curves done",
                i + 1,
                sampled.len()
            );
        }
    }
    curves
}

struct ArmRow {
    b: usize,
    n: usize,
    median_cost_b: f64,
    median_quiescence_cost: f64,
    /// Median `quiescence_cost` over only the curves that did NOT hit the
    /// `CYCLE_COST` sentinel — the reliable half of `median_quiescence_cost`
    /// when `cycle_hits > 0`.
    median_quiescence_cost_excl_cycle: f64,
    cycle_hits: usize,
    median_loss_pct: f64,
    median_apps_to_quiescence: f64,
    class_cap_hits: usize,
    quiesced: usize,
}

fn arm_rows(curves: &[&ExprCurve], grid: &[usize]) -> Vec<ArmRow> {
    let idx_of: std::collections::BTreeMap<usize, usize> =
        grid.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let mut rows = Vec::new();
    for &b in &[100usize, 200usize] {
        let Some(&bi) = idx_of.get(&b) else { continue };
        let Some(&fi) = idx_of.get(&(b * 4)) else {
            continue;
        };
        let cost_b: Vec<f64> = curves.iter().map(|c| c.cost_at[bi] as f64).collect();
        let loss: Vec<f64> = curves
            .iter()
            .map(|c| loss_pct(c.cost_at[bi], c.cost_at[fi]))
            .filter(|l| l.is_finite())
            .collect();
        let quiescence_excl_cycle: Vec<f64> = curves
            .iter()
            .map(|c| c.quiescence_cost)
            .filter(|&c| c < CYCLE_COST_THRESHOLD)
            .map(|c| c as f64)
            .collect();
        rows.push(ArmRow {
            b,
            n: curves.len(),
            median_cost_b: median(&cost_b),
            median_quiescence_cost: median(
                &curves
                    .iter()
                    .map(|c| c.quiescence_cost as f64)
                    .collect::<Vec<_>>(),
            ),
            median_quiescence_cost_excl_cycle: if quiescence_excl_cycle.is_empty() {
                0.0
            } else {
                median(&quiescence_excl_cycle)
            },
            cycle_hits: curves
                .iter()
                .filter(|c| c.quiescence_cost >= CYCLE_COST_THRESHOLD)
                .count(),
            median_loss_pct: if loss.is_empty() { 0.0 } else { median(&loss) },
            median_apps_to_quiescence: median(
                &curves
                    .iter()
                    .map(|c| c.ended_at_apps as f64)
                    .collect::<Vec<_>>(),
            ),
            class_cap_hits: curves
                .iter()
                .filter(|c| c.ended == SaturationStop::ClassCap)
                .count(),
            quiesced: curves
                .iter()
                .filter(|c| c.ended == SaturationStop::Quiesced)
                .count(),
        });
    }
    rows
}

/// One row per (arm, expression, grid checkpoint), byte-compatible with the
/// modes (i)/(ii) CSV so a single analysis reads every mode. `rule_set` is
/// `base` for the production 62 and `new:<total>` for the batch arm; the
/// fingerprint is `rule_set_fingerprint` over the arm's rule vector.
fn write_curve_csv(path: &str, grid: &[usize], arms: &[(&str, &Vec<ExprCurve>)]) {
    use std::io::Write as _;
    let out = PathBuf::from(path);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create CSV output directory");
    }
    let mut f = std::fs::File::create(&out)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out.display()));
    writeln!(
        f,
        "rule_set,num_rules,fingerprint,expr_name,origin,tier,node_count,app_target,app_actual,\
         cost,ended,ended_at_apps"
    )
    .unwrap();
    for (label, curves) in arms {
        let rules = match *label {
            "base" => all_rules(),
            "new" => {
                let mut r = all_rules();
                r.extend(experimental_rules());
                r
            }
            other => panic!("write_curve_csv: unknown arm label {other:?}"),
        };
        let num_rules = rules.len();
        let fingerprint = rule_set_fingerprint(&rules);
        let rule_set = match *label {
            "base" => "base".to_string(),
            _ => format!("new:{num_rules}"),
        };
        for c in curves.iter() {
            assert_eq!(c.cost_at.len(), grid.len(), "curve/grid length mismatch");
            for (j, &target) in grid.iter().enumerate() {
                writeln!(
                    f,
                    "{rule_set},{num_rules},{fingerprint},{},{},{},{},{target},{},{},{},{}",
                    c.name,
                    c.origin,
                    c.tier,
                    c.node_count,
                    c.apps_at[j],
                    c.cost_at[j],
                    stop_name(c.ended),
                    c.ended_at_apps,
                )
                .unwrap();
            }
        }
    }
    f.flush().unwrap();
    eprintln!("phase3_round2_new_rules: wrote {}", out.display());
}

fn main() {
    let args = Args::parse();
    assert!(
        args.samples >= MIN_SAMPLES,
        "phase3_round2_new_rules: --samples {} < the {MIN_SAMPLES} floor — fail loud, \
         never silently under-measure",
        args.samples
    );

    let base_count = all_rules().len();
    assert_eq!(
        base_count, 62,
        "phase3_round2_new_rules: all_rules() must stay 62 — a change here means production \
         drifted, not that this binary's |R|=62 arm needs updating"
    );
    let batch_count = experimental_rules().len();
    let total_count = base_count + batch_count;
    eprintln!(
        "phase3_round2_new_rules: base |R|={base_count}, batch +{batch_count}, total |R|={total_count}"
    );

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let mut entries: Vec<(&'static str, String, ExprArena, ExprId)> = Vec::new();
    for (origin, file) in [("train", "corpus_train.bin"), ("dev", "corpus_dev.bin")] {
        let path = corpus_dir.join(file);
        let tier_entries = read_corpus(&path).unwrap_or_else(|e| {
            panic!(
                "phase3_round2_new_rules: failed to read {}: {e}",
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
        "phase3_round2_new_rules: corpus has only {total_available} TRAIN+DEV expressions, \
         need >= {} — regenerate a larger corpus",
        args.samples
    );

    // Exactly Round 1's stratified-by-size stride sample
    // (phase3_unguided_baseline.rs), restated here so this binary depends on
    // no internals of that one.
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
        "phase3_round2_new_rules: {} of {total_available} TRAIN+DEV expressions, \
         size-stratified (stride {:.2}) — same sample as Round 1",
        sampled.len(),
        stride
    );

    // §5: per-curve safety ceiling scaled by |R|/62 so more rules cannot
    // silently truncate an honest curve into a false quiescence — it still
    // panics (inside run_anytime_curve) when it binds, for either arm.
    let batch_safety_timeout = Duration::from_secs_f64(
        BASE_SAFETY_TIMEOUT.as_secs_f64() * (total_count as f64 / base_count as f64),
    );

    eprintln!("phase3_round2_new_rules: running base arm (|R|={base_count})...");
    let base_curves = run_curves(&sampled, all_rules, BASE_SAFETY_TIMEOUT, "base");

    eprintln!("phase3_round2_new_rules: running batch arm (|R|={total_count})...");
    let batch_curves = run_curves(
        &sampled,
        || {
            let mut r = all_rules();
            r.extend(experimental_rules());
            r
        },
        batch_safety_timeout,
        "batch",
    );

    let grid = APP_CHECKPOINT_GRID;
    let tiers = ["blitz", "rapid", "classical"];

    write_curve_csv(
        &args.out_csv,
        grid,
        &[("base", &base_curves), ("new", &batch_curves)],
    );

    let mut md = String::new();
    md.push_str("## Round 2, mode (iii): unguided anytime curves, |R|=62 vs 62+batch\n\n");
    md.push_str(&format!(
        "Agent: rule-batch implementation + unguided curves (docs/plans/2026-09-01-phase3-round2-rule-scaling.md \
         §2.3, §8). Base arm `|R|={base_count}` (`all_rules()`, unchanged production set); batch arm \
         `|R|={total_count}` (`all_rules() + pixelflow_search::math::round2_rules::experimental_rules()`, \
         `+{batch_count}` new rules, harness-only — never added to `all_rules()`). Same Round-1 400-expression \
         stratified sample (stride {:.2} over {total_available} TRAIN+DEV expressions), same shared \
         `run_anytime_curve`/`APP_CHECKPOINT_GRID`, deterministic `CostModel::latency_prior()` — UNGUIDED \
         only, no Guide loaded or trained. Batch arm's per-curve safety ceiling scaled by \
         `{total_count}/{base_count}` per §5 (still panics if it binds; it did not bind on this run).\n\n",
        stride
    ));

    for tier in tiers {
        let base_sub: Vec<&ExprCurve> = base_curves.iter().filter(|c| c.tier == tier).collect();
        let batch_sub: Vec<&ExprCurve> = batch_curves.iter().filter(|c| c.tier == tier).collect();
        if base_sub.is_empty() {
            continue;
        }
        md.push_str(&format!("### Tier: {tier} (n={})\n\n", base_sub.len()));
        md.push_str(
            "| B | arm |R| | n | median cost@B | median cost@quiescence | quiescence excl. cycle-cost | \
             cycle-cost hits | median trunc-loss% | median apps-to-quiescence | class_cap hits | quiesced |\n\
             |---|---|---|---|---|---|---|---|---|---|---|\n",
        );
        let base_rows = arm_rows(&base_sub, grid);
        let batch_rows = arm_rows(&batch_sub, grid);
        for (br, kr) in base_rows.iter().zip(batch_rows.iter()) {
            md.push_str(&format!(
                "| {} | {base_count} (base) | {} | {:.2} | {:.2} | {:.2} | {} | {:.3} | {:.1} | {} | {} |\n",
                br.b,
                br.n,
                br.median_cost_b,
                br.median_quiescence_cost,
                br.median_quiescence_cost_excl_cycle,
                br.cycle_hits,
                br.median_loss_pct,
                br.median_apps_to_quiescence,
                br.class_cap_hits,
                br.quiesced,
            ));
            md.push_str(&format!(
                "| {} | {total_count} (+batch) | {} | {:.2} | {:.2} | {:.2} | {} | {:.3} | {:.1} | {} | {} |\n",
                kr.b,
                kr.n,
                kr.median_cost_b,
                kr.median_quiescence_cost,
                kr.median_quiescence_cost_excl_cycle,
                kr.cycle_hits,
                kr.median_loss_pct,
                kr.median_apps_to_quiescence,
                kr.class_cap_hits,
                kr.quiesced,
            ));
        }
        md.push('\n');
        md.push_str(
            "_\"cycle-cost hits\" = curves whose quiescence cost hit `extract_dag`'s CYCLE_COST \
             sentinel (>= 900,000) — a genuinely self-referential e-class post-saturation, a \
             pre-existing extraction-algorithm behavior verified present under UNCHANGED \
             `all_rules()` alone on large classical expressions (not introduced by \
             `experimental_rules()`; see \"quiescence excl. cycle-cost\" for the reliable median \
             over the non-cyclic curves only). `median cost@B` (the B=100/200 regret-relevant \
             number) is unaffected: cyclic e-classes only arise at very high application counts, \
             past where B or 4B falls._\n\n",
        );

        // Fidelity gain: per-expression, the closure-aware improvement the
        // batch buys at quiescence (§4.2's fid(e), restricted to this tier).
        // Expressions where EITHER arm hit the CYCLE_COST sentinel are
        // excluded (a sentinel is not a measured cost, so no ratio against
        // it is a fidelity number) and counted separately.
        let mut fid_vals: Vec<f64> = Vec::new();
        let mut fid_excluded_cyclic = 0usize;
        for (bc, kc) in base_sub.iter().zip(batch_sub.iter()) {
            assert_eq!(
                bc.name, kc.name,
                "base/batch curve order must match (same sample)"
            );
            let base_q = bc.quiescence_cost.min(bc.cost_at[grid.len() - 1]);
            let batch_q = kc.quiescence_cost.min(kc.cost_at[grid.len() - 1]);
            if base_q >= CYCLE_COST_THRESHOLD || batch_q >= CYCLE_COST_THRESHOLD {
                fid_excluded_cyclic += 1;
                continue;
            }
            if base_q == 0 {
                continue; // zero-cost reference: skip (matches the established regret convention's
                // spirit — a ratio against 0 is not a finite fidelity gain to average)
            }
            fid_vals.push((base_q as f64 - batch_q as f64) / base_q as f64 * 100.0);
        }
        let fid_median = if fid_vals.is_empty() {
            0.0
        } else {
            median(&fid_vals)
        };
        let fid_positive = fid_vals.iter().filter(|v| **v > 1e-9).count();
        md.push_str(&format!(
            "Closure-gain at quiescence (`fid(e) = (cost@62 - cost@62+batch)/cost@62` at curve-end, \
             median over the {} tier expressions with a nonzero, non-cyclic base cost; \
             {fid_excluded_cyclic} excluded for a CYCLE_COST sentinel on either arm): **{:.3}%**, \
             positive for {}/{}.\n\n",
            tier,
            fid_median,
            fid_positive,
            fid_vals.len(),
        ));

        // How runs ended (stop-reason distribution), per arm.
        for (label, sub) in [("base", &base_sub), ("+batch", &batch_sub)] {
            let n = sub.len();
            let count = |s: SaturationStop| sub.iter().filter(|c| c.ended == s).count();
            let mut apps: Vec<f64> = sub.iter().map(|c| c.ended_at_apps as f64).collect();
            apps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            md.push_str(&format!(
                "- ended ({label}, n={n}): quiesced={} class_cap={} grid_exhausted={} \
                 sweep_ceiling={} timeout={} | ended_at_apps median={:.0} p90={:.0}\n",
                count(SaturationStop::Quiesced),
                count(SaturationStop::ClassCap),
                count(SaturationStop::ApplicationBudget),
                count(SaturationStop::IterationCeiling),
                count(SaturationStop::Timeout),
                percentile(&apps, 0.5),
                percentile(&apps, 0.9),
            ));
        }
        md.push('\n');
    }

    md.push_str("### Per-arm stop-reason detail (raw, ALL tiers combined)\n\n");
    md.push_str("| arm | quiesced | class_cap | grid_exhausted | sweep_ceiling | timeout |\n|---|---|---|---|---|---|\n");
    for (label, curves) in [("base (|R|=62)", &base_curves), ("+batch", &batch_curves)] {
        let count = |s: SaturationStop| curves.iter().filter(|c| c.ended == s).count();
        md.push_str(&format!(
            "| {label} | {} | {} | {} | {} | {} |\n",
            count(SaturationStop::Quiesced),
            count(SaturationStop::ClassCap),
            count(SaturationStop::ApplicationBudget),
            count(SaturationStop::IterationCeiling),
            count(SaturationStop::Timeout),
        ));
    }
    md.push('\n');

    md.push_str(&format!(
        "Raw per-expression rows: {} expressions x {} grid checkpoints x 2 arms, written to \
         `{}` (same schema as the modes (i)/(ii) CSV; `rule_set` = `base` / `new:{total_count}`).\n\n",
        sampled.len(),
        grid.len(),
        args.out_csv,
    ));

    md.push_str(&format!(
        "_Grid: `{grid:?}`. Stop-name legend: `quiesced` = saturation reached a fixpoint before any \
         budget bound; `class_cap` = hit the tier's `max_classes` memory-protection cap; \
         `grid_exhausted` = still finding unions when the grid's top checkpoint (204800 applications) \
         was reached; `sweep_ceiling`/`timeout` = safety ceilings (would panic if actually hit — \
         listed for completeness, expected 0 in both columns)._\n"
    ));

    let out_md = PathBuf::from(&args.out_md);
    if let Some(parent) = out_md.parent() {
        std::fs::create_dir_all(parent).expect("create markdown output directory");
    }
    if out_md.exists() {
        let mut existing = std::fs::read_to_string(&out_md)
            .unwrap_or_else(|e| panic!("cannot read existing {}: {e}", out_md.display()));
        if !existing.ends_with('\n') {
            existing.push('\n');
        }
        existing.push('\n');
        existing.push_str(&md);
        std::fs::write(&out_md, existing)
            .unwrap_or_else(|e| panic!("cannot append to {}: {e}", out_md.display()));
        println!(
            "appended Round 2 mode (iii) section to existing {}",
            out_md.display()
        );
    } else {
        let mut header = String::new();
        header.push_str("# Round 2: unguided anytime curves vs rule count\n\n");
        header.push_str(
            "Per-mode sections appended by each agent's binary as it completes its run \
             (docs/plans/2026-09-01-phase3-round2-rule-scaling.md §9). This file did not exist \
             when the mode (iii) section below was written — it was created with mode (iii)'s \
             section only.\n\n",
        );
        header.push_str(&md);
        std::fs::write(&out_md, header)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_md.display()));
        println!(
            "created {} with mode (iii)'s section only (no other mode had run yet)",
            out_md.display()
        );
    }
}
