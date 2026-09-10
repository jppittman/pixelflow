//! Oracle headroom for Phase 3 (the Guide) at corpus scale.
//!
//! Answers the one number that decides whether Phase 3 is worth running —
//! the analogue of the extraction-head program's static/noswap=0.54: what
//! fraction of rule applications fired during full saturation are actually
//! load-bearing for the table-optimal (static latency-prior) extraction?
//!
//! For each corpus expression this runs the existing, unmodified pipeline
//! (`EGraph::saturate` -> `extract_dag` with `CostModel::latency_prior()` ->
//! `EpisodeLabels::compute`, exactly [`pixelflow_search::egraph::run_episode`])
//! and additionally computes a **strict lower bound**: the hindsight labeler
//! (`derivation_ancestors`) is a deliberate, documented over-approximation
//! (see `pixelflow-search/src/egraph/provenance.rs` lines ~214-239 — child
//! *classes* not child *nodes*, union events by class membership, no
//! fixed-point pruning). The strict bound instead asks only "did this
//! application's output node literally appear on the extracted derivation
//! path" — i.e. walk the *chosen* nodes only (the same walk
//! `labeler::chosen_tagged_nodes` does internally, reimplemented here against
//! the crate's public API since this is additive scratch code and
//! `labeler.rs`/`provenance.rs` are explicitly not to be touched this round)
//! and credit only the direct creating application of each chosen node.
//!
//! The gap between the strict ratio and the labeler ratio is the label slack
//! a trained Guide would inherit from the over-approximation.
//!
//! Every number this binary reports is a deterministic count or a
//! `CostModel::latency_prior()` cost — no wall-clock timing gates
//! correctness, so machine contention cannot corrupt the measurement (it can
//! only slow the run down). `EGraph::saturate()`'s production default caps
//! each round at a 500ms wall-clock deadline, but that deadline is a
//! *latency-sensitivity* concern specific to the compiler's interactive
//! path, not this offline, batch measurement — so this binary calls
//! `saturate_with_limits` with a generous, effectively-non-binding
//! [`SATURATE_TIMEOUT`] (60s) instead of reusing the production 500ms value,
//! and separately times each call with `Instant` purely as an *informational*
//! diagnostic (`slow_expressions_over_500ms`, reported but never gating which
//! samples count). This closes a gap an earlier draft of this harness left
//! open (and flagged rather than silently assuming away): with the
//! production 500ms deadline actually wired into the stopping condition, a
//! run that happened to exceed it partway through would have its *partial*,
//! non-converged e-graph counted as the expression's result — indistinguishable,
//! by the iteration/class-count-only proxy below, from a run that truly
//! converged — making the reported ratios depend on host speed for any
//! expression that ran long. With the timeout now generous, that dependency
//! is removed: iteration/class-count caps are the only remaining stopping
//! conditions besides genuine convergence, and both are deterministic counts.
//!
//! # Quiescence diagnostic
//!
//! `saturate_with_limits`'s doc comment names four ways a run can end:
//! iteration cap, class-count cap, timeout, or **"Saturation achieved (no
//! more changes)"** — the loop's own convergence check
//! (`if unions == 0 { break; }` in `pixelflow-search/src/egraph/graph.rs`).
//! So this crate's saturation is *not* pure budget-truncation-only: a run can
//! and does reach quiescence — a diagnostic condition, never a certified
//! fixpoint (this optimizer is budget-only by design) — before its budget is
//! spent, and the existing code already detects that case. This binary
//! reports, per expression, a `quiesced_before_cap` diagnostic derived from
//! `SaturationStats` alone (`iterations < max_iters` and the e-graph did not
//! reach `max_classes`) — both deterministic counts, not timing reads. With
//! [`SATURATE_TIMEOUT`] now generous rather than production's 500ms (see
//! above), the one remaining ambiguity this proxy can't resolve without
//! `SaturationStats` growing an explicit stop-reason field (out of scope —
//! no edits to core e-graph semantics this round beyond the read-only
//! `Instant` wrapping above) is a run hitting the 60s safety ceiling itself —
//! expected never to happen at this corpus's scale, and reported
//! (`hit_safety_timeout`) rather than silently assumed away if it ever does.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin guide_headroom -- \
//!     --corpus-dir pixelflow-pipeline/data --limit 500 \
//!     --out docs/results/2026-08-30-guide-headroom.json
//! ```

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;

use pixelflow_ir::ExprArena;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    CostModel, EGraph, EpisodeLabels, Origin, SaturationConfig, extract_dag,
};
use pixelflow_search::math::all_rules;

#[derive(Parser)]
#[command(name = "guide_headroom")]
#[command(
    about = "Phase 3 oracle-headroom measurement: load-bearing ratio of rule applications at corpus scale"
)]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Minimum number of expressions to measure (paper task: >= 500).
    /// All available train+dev expressions are used if that already exceeds
    /// this; otherwise this is a floor the corpus must clear (fail loud, not
    /// silently under-measure).
    #[arg(long, default_value_t = 500)]
    min_expressions: usize,

    /// Optional cap on the number of expressions measured (0 = no cap),
    /// deterministically the first N in corpus order after concatenating
    /// train then dev.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Write the full structured result (per-expression + per-rule + corpus
    /// summary) as JSON to this path. Always also prints a human summary to
    /// stdout.
    #[arg(long)]
    out: Option<String>,
}

/// Safety ceiling only, per the module doc — deliberately NOT production's
/// 500ms (a latency-sensitivity budget for the interactive compiler path,
/// not this offline measurement). 60s is expected to never bind; if it ever
/// does, `hit_safety_timeout` reports it rather than silently mislabeling a
/// truncated run as converged.
const SATURATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// This binary's saturation budget: the named compatibility budget
/// ([`SaturationConfig::compatibility`], 100 rounds / 10,000 classes) with
/// the wall clock swapped for [`SATURATE_TIMEOUT`].
///
/// Naming it rather than re-spelling the round and class caps is what keeps
/// this binary's headroom numbers comparable with the other non-production
/// harnesses: revising the shared budget reaches here too, instead of
/// leaving a third private copy behind.
fn saturation_budget() -> SaturationConfig {
    SaturationConfig {
        safety_ceiling: SATURATE_TIMEOUT,
        ..SaturationConfig::compatibility(100)
    }
}
/// Informational-only threshold: did this expression's saturation call take
/// long enough that it WOULD have hit production's 500ms deadline, had this
/// binary reused it? Never gates correctness or which samples count — purely
/// context for how close this corpus runs to the production budget.
const PRODUCTION_DEADLINE_FOR_COMPARISON: std::time::Duration =
    std::time::Duration::from_millis(500);

/// Per-expression measurement.
struct ExprMeasurement {
    name: String,
    node_count: usize,
    total_applications: usize,
    labeler_load_bearing: usize,
    strict_load_bearing: usize,
    /// Latency-prior **DAG** cost of the extracted term — each distinct
    /// chosen e-class priced once, which is what the emitted kernel pays.
    extracted_cost: usize,
    /// See module docs' "Quiescence diagnostic" section: `true` means the
    /// run stopped before exhausting either the iteration or class-count
    /// cap — i.e. truly converged, now that `SATURATE_TIMEOUT` is a
    /// non-binding safety ceiling rather than production's 500ms deadline.
    quiesced_before_cap: bool,
    saturation_iterations: usize,
    /// Wall-clock elapsed for this expression's `saturate_with_limits` call —
    /// informational only (see `PRODUCTION_DEADLINE_FOR_COMPARISON`), never
    /// used to decide `quiesced_before_cap` or any reported ratio.
    exceeded_production_deadline: bool,
}

impl ExprMeasurement {
    fn labeler_ratio(&self) -> f64 {
        ratio(self.labeler_load_bearing, self.total_applications)
    }
    fn strict_ratio(&self) -> f64 {
        ratio(self.strict_load_bearing, self.total_applications)
    }
}

/// `0.0` for zero applications (no evidence, not `NaN`) — matches
/// `RuleStats::load_bearing_ratio`'s convention exactly, for the same reason.
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Per-rule aggregate across the whole corpus, under both bounds.
#[derive(Default, Clone, Copy)]
struct RuleAgg {
    fired: usize,
    labeler_load_bearing: usize,
    strict_load_bearing: usize,
}

/// Walk the extraction's *chosen* nodes only — root plus, recursively, each
/// chosen child — mirroring `labeler::chosen_tagged_nodes` exactly (same
/// panics-on-missing-choice contract, since a missing choice for a class
/// reachable via the chosen extraction is an extractor bug, not something to
/// paper over). Returns the `ENodeId` tag of every node actually used by the
/// winning extraction.
fn chosen_node_tags(
    egraph: &EGraph,
    root: pixelflow_search::egraph::EClassId,
    choices: &[Option<usize>],
) -> Vec<pixelflow_search::egraph::ENodeId> {
    use std::collections::BTreeSet;

    let mut visited: BTreeSet<pixelflow_search::egraph::EClassId> = BTreeSet::new();
    let mut stack = vec![root];
    let mut result = Vec::new();

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "guide_headroom: e-class {idx} reachable from root {} via the chosen \
                 extraction has no recorded choice — extractor invariant violated",
                root.index()
            )
        });
        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "guide_headroom: node_idx {node_idx} out of bounds ({}) for e-class {idx}",
            nodes.len()
        );
        let tags = egraph.tags(canonical);
        result.push(tags[node_idx]);
        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }
    result
}

fn quantiles(mut xs: Vec<f64>) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("guide_headroom: NaN ratio"));
    let q = |p: f64| -> f64 {
        let n = xs.len();
        if n == 1 {
            return xs[0];
        }
        let pos = p * (n as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            xs[lo]
        } else {
            let frac = pos - lo as f64;
            xs[lo] * (1.0 - frac) + xs[hi] * frac
        }
    };
    (q(0.25), q(0.5), q(0.75))
}

fn main() {
    let args = Args::parse();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let train_path = corpus_dir.join("corpus_train.bin");
    let dev_path = corpus_dir.join("corpus_dev.bin");

    let mut entries: Vec<(String, ExprArena, pixelflow_ir::ExprId)> = read_corpus(&train_path)
        .unwrap_or_else(|e| {
            panic!(
                "guide_headroom: failed to read {}: {e}",
                train_path.display()
            )
        });
    let dev_entries = read_corpus(&dev_path)
        .unwrap_or_else(|e| panic!("guide_headroom: failed to read {}: {e}", dev_path.display()));
    let train_count = entries.len();
    entries.extend(dev_entries);
    let total_available = entries.len();

    assert!(
        total_available >= args.min_expressions,
        "guide_headroom: corpus has only {total_available} expressions \
         (train {train_count} + dev {}), need >= {} — regenerate a larger corpus, \
         do not silently measure on too few expressions",
        total_available - train_count,
        args.min_expressions
    );

    // Corpus write order is band-by-band (increasing complexity) within each
    // tier, so a plain `truncate` would sample only the smallest/simplest
    // band and silently misrepresent "at scale". Stride-sample evenly across
    // the full train+dev population instead, so `--limit` expressions cover
    // every band/family the corpus generator produced.
    if args.limit > 0 && args.limit < entries.len() {
        let stride = entries.len() as f64 / args.limit as f64;
        let mut sampled = Vec::with_capacity(args.limit);
        for i in 0..args.limit {
            let idx = ((i as f64) * stride) as usize;
            sampled.push(entries[idx.min(entries.len() - 1)].clone());
        }
        entries = sampled;
    }
    let n = entries.len();
    eprintln!(
        "guide_headroom: measuring {n} expressions (train {train_count}, dev {}, corpus total {total_available})",
        total_available - train_count
    );

    let costs = CostModel::latency_prior();
    let rule_names: Vec<String> = {
        let probe = EGraph::with_rules(all_rules());
        (0..all_rules().len())
            .map(|i| {
                probe
                    .rule(i)
                    .map(|r| r.name().to_string())
                    .unwrap_or_else(|| format!("<rule {i}>"))
            })
            .collect()
    };

    let mut measurements: Vec<ExprMeasurement> = Vec::with_capacity(n);
    let mut rule_agg: BTreeMap<usize, RuleAgg> = BTreeMap::new();
    let mut total_apps_all: u64 = 0;
    let mut total_labeler_lb_all: u64 = 0;
    let mut total_strict_lb_all: u64 = 0;

    for (i, (name, arena, root)) in entries.iter().enumerate() {
        let mut egraph = EGraph::with_rules(all_rules());
        let root_class = pixelflow_search::egraph::insert(
            arena,
            *root,
            &mut egraph,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let budget = saturation_budget();
        let saturate_started = std::time::Instant::now();
        let sat_stats = budget.run(&mut egraph);
        let saturate_elapsed = saturate_started.elapsed();
        let hit_iteration_cap = sat_stats.iterations >= budget.max_iterations;
        let hit_class_cap = egraph.num_classes() > budget.max_classes;
        let hit_safety_timeout = saturate_elapsed >= budget.safety_ceiling;
        assert!(
            !hit_safety_timeout,
            "guide_headroom: expression '{name}' ran {saturate_elapsed:?}, hitting the \
             {SATURATE_TIMEOUT:?} safety ceiling — this was expected to never bind at this \
             corpus's scale; fail loud rather than silently report a truncated sample as \
             converged"
        );
        let quiesced_before_cap = !hit_iteration_cap && !hit_class_cap;
        let exceeded_production_deadline = saturate_elapsed >= PRODUCTION_DEADLINE_FOR_COMPARISON;

        let extraction = extract_dag(&egraph, root_class, &costs);
        let labels = EpisodeLabels::compute(&egraph, extraction.root, &extraction.choices);

        let chosen_tags = chosen_node_tags(&egraph, extraction.root, &extraction.choices);
        let mut strict_set: std::collections::BTreeSet<pixelflow_search::egraph::ApplicationId> =
            std::collections::BTreeSet::new();
        for tag in &chosen_tags {
            if let Some(Origin::Rule(app_id)) = egraph.provenance().origin(*tag) {
                strict_set.insert(app_id);
            }
        }

        let total_applications = egraph.provenance().recorded_count();
        let labeler_lb = labels.load_bearing.len();
        let strict_lb = strict_set.len();

        assert!(
            strict_lb <= labeler_lb,
            "guide_headroom: strict bound ({strict_lb}) exceeded labeler bound ({labeler_lb}) \
             for expression '{name}' — the labeler is documented as a superset of the strict \
             walk, a violation means the reimplementation has drifted from \
             `labeler::chosen_tagged_nodes`, fail loud rather than report a bogus gap"
        );

        for (app_id, record) in egraph.provenance().applications() {
            let agg = rule_agg.entry(record.rule_idx).or_default();
            agg.fired += 1;
            if labels.load_bearing.contains(&app_id) {
                agg.labeler_load_bearing += 1;
            }
            if strict_set.contains(&app_id) {
                agg.strict_load_bearing += 1;
            }
        }

        total_apps_all += total_applications as u64;
        total_labeler_lb_all += labeler_lb as u64;
        total_strict_lb_all += strict_lb as u64;

        measurements.push(ExprMeasurement {
            name: name.clone(),
            node_count: arena.len(),
            total_applications,
            labeler_load_bearing: labeler_lb,
            strict_load_bearing: strict_lb,
            // The DAG cost — what the emitted kernel pays — not the DP's
            // tree cost, which prices a shared subterm once per use (#1111).
            extracted_cost: extraction.dag_cost,
            quiesced_before_cap,
            saturation_iterations: sat_stats.iterations,
            exceeded_production_deadline,
        });

        if (i + 1) % 100 == 0 || i + 1 == n {
            eprintln!("guide_headroom: {}/{n} expressions measured", i + 1);
        }
    }

    let labeler_ratios: Vec<f64> = measurements.iter().map(|m| m.labeler_ratio()).collect();
    let strict_ratios: Vec<f64> = measurements.iter().map(|m| m.strict_ratio()).collect();
    let (l_q1, l_med, l_q3) = quantiles(labeler_ratios.clone());
    let (s_q1, s_med, s_q3) = quantiles(strict_ratios.clone());

    let pooled_labeler_ratio = ratio(total_labeler_lb_all as usize, total_apps_all as usize);
    let pooled_strict_ratio = ratio(total_strict_lb_all as usize, total_apps_all as usize);

    println!("=== Phase 3 oracle headroom: {n} expressions ===");
    println!(
        "total rule applications: {total_apps_all}  (labeler load-bearing: {total_labeler_lb_all}, strict load-bearing: {total_strict_lb_all})"
    );
    println!();
    println!("--- labeler bound (derivation_ancestors, over-approximate) ---");
    println!("  pooled ratio (sum LB / sum applications): {pooled_labeler_ratio:.4}");
    println!("  per-expression ratio: median {l_med:.4}  Q1 {l_q1:.4}  Q3 {l_q3:.4}");
    println!(
        "  implied oracle savings (1/median): {:.2}x",
        if l_med > 0.0 {
            1.0 / l_med
        } else {
            f64::INFINITY
        }
    );
    println!();
    println!("--- strict bound (node literally on extracted path) ---");
    println!("  pooled ratio (sum LB / sum applications): {pooled_strict_ratio:.4}");
    println!("  per-expression ratio: median {s_med:.4}  Q1 {s_q1:.4}  Q3 {s_q3:.4}");
    println!(
        "  implied oracle savings (1/median): {:.2}x",
        if s_med > 0.0 {
            1.0 / s_med
        } else {
            f64::INFINITY
        }
    );
    println!();
    println!(
        "--- label slack (labeler ratio - strict ratio), pooled: {:.4} ---",
        pooled_labeler_ratio - pooled_strict_ratio
    );

    // Diagnostic: does load-bearing ratio differ between expressions whose
    // saturation quiesced (hit a real fixed point, or — the undistinguished
    // rare case — the wall-clock deadline) before spending its iteration/
    // class budget, vs. expressions that exhausted the budget outright? See
    // the module doc's "Quiescence diagnostic" section for exactly what this
    // proxy can and can't tell apart. This is diagnostic only — no bearing
    // on the headroom numbers above, which are unconditional over all
    // expressions.
    println!();
    println!("--- diagnostic: quiesced-before-cap vs exhausted-budget (see module docs) ---");
    let print_group = |label: &str, subset: &[&ExprMeasurement]| {
        if subset.is_empty() {
            println!("  {label}: n=0 (no expressions in this bucket)");
            return;
        }
        let l: Vec<f64> = subset.iter().map(|m| m.labeler_ratio()).collect();
        let s: Vec<f64> = subset.iter().map(|m| m.strict_ratio()).collect();
        let (lq1, lmed, lq3) = quantiles(l);
        let (sq1, smed, sq3) = quantiles(s);
        println!(
            "  {label}: n={:<5} labeler median {lmed:.4} (Q1 {lq1:.4} Q3 {lq3:.4})  strict median {smed:.4} (Q1 {sq1:.4} Q3 {sq3:.4})",
            subset.len()
        );
    };
    let quiesced: Vec<&ExprMeasurement> = measurements
        .iter()
        .filter(|m| m.quiesced_before_cap)
        .collect();
    let exhausted: Vec<&ExprMeasurement> = measurements
        .iter()
        .filter(|m| !m.quiesced_before_cap)
        .collect();
    print_group("quiesced before cap", &quiesced);
    print_group("exhausted budget    ", &exhausted);
    let exceeded_production_deadline: Vec<&ExprMeasurement> = measurements
        .iter()
        .filter(|m| m.exceeded_production_deadline)
        .collect();
    println!(
        "  (informational, not a correctness gate: {}/{n} expressions took >= 500ms wall-clock \
         -- i.e. would have hit production `saturate()`'s deadline had this offline harness reused \
         it instead of a generous, non-binding safety ceiling)",
        exceeded_production_deadline.len()
    );

    // Same split by expression size (arena node count), since size is the
    // obvious confound: larger expressions are both more likely to exhaust
    // the budget and mechanically produce lower ratios (more churn before
    // the useful structure is found).
    println!();
    println!("--- diagnostic: load-bearing ratio by expression size (arena node count) ---");
    let mut by_size: Vec<&ExprMeasurement> = measurements.iter().collect();
    by_size.sort_by_key(|m| m.node_count);
    let tercile = by_size.len() / 3;
    if tercile > 0 {
        let (small, rest) = by_size.split_at(tercile);
        let (mid, large) = rest.split_at(tercile.min(rest.len()));
        print_group("small (tercile 1)   ", small);
        print_group("medium (tercile 2)  ", mid);
        print_group("large (tercile 3)   ", large);
    }

    println!();
    println!(
        "{:<28} {:>7} {:>10} {:>7} {:>10} {:>7}",
        "rule", "fired", "lbl-LB", "lbl%", "strict-LB", "strict%"
    );
    let mut rows: Vec<(usize, RuleAgg)> = rule_agg.iter().map(|(&k, &v)| (k, v)).collect();
    rows.sort_by(|a, b| {
        let ra = ratio(a.1.labeler_load_bearing, a.1.fired);
        let rb = ratio(b.1.labeler_load_bearing, b.1.fired);
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    for (idx, agg) in &rows {
        let name = rule_names
            .get(*idx)
            .cloned()
            .unwrap_or_else(|| format!("<rule {idx}>"));
        println!(
            "{:<28} {:>7} {:>10} {:>6.1}% {:>10} {:>6.1}%",
            name,
            agg.fired,
            agg.labeler_load_bearing,
            ratio(agg.labeler_load_bearing, agg.fired) * 100.0,
            agg.strict_load_bearing,
            ratio(agg.strict_load_bearing, agg.fired) * 100.0,
        );
    }
    // Rules in the 62-rule library that never fired on this corpus at all —
    // reported explicitly rather than silently absent from the table above,
    // since "never fired" and "fired but always wasted" are different facts
    // for rule-triage purposes.
    let never_fired: Vec<&str> = rule_names
        .iter()
        .enumerate()
        .filter(|(idx, _)| !rule_agg.contains_key(idx))
        .map(|(_, name)| name.as_str())
        .collect();
    println!();
    println!(
        "rules that never fired on this corpus ({}/{}): {}",
        never_fired.len(),
        rule_names.len(),
        if never_fired.is_empty() {
            "none".to_string()
        } else {
            never_fired.join(", ")
        }
    );

    if let Some(out_path) = &args.out {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"num_expressions\": {n},\n"));
        json.push_str(&format!("  \"total_applications\": {total_apps_all},\n"));
        json.push_str(&format!(
            "  \"pooled_labeler_ratio\": {pooled_labeler_ratio:.6},\n"
        ));
        json.push_str(&format!(
            "  \"pooled_strict_ratio\": {pooled_strict_ratio:.6},\n"
        ));
        json.push_str(&format!(
            "  \"labeler_ratio_quartiles\": [{l_q1:.6}, {l_med:.6}, {l_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"strict_ratio_quartiles\": [{s_q1:.6}, {s_med:.6}, {s_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"implied_oracle_savings_labeler_median\": {:.6},\n",
            if l_med > 0.0 { 1.0 / l_med } else { 0.0 }
        ));
        json.push_str(&format!(
            "  \"implied_oracle_savings_strict_median\": {:.6},\n",
            if s_med > 0.0 { 1.0 / s_med } else { 0.0 }
        ));
        json.push_str("  \"per_rule\": [\n");
        for (i, (idx, agg)) in rows.iter().enumerate() {
            let name = rule_names
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("<rule {idx}>"));
            json.push_str(&format!(
                "    {{\"rule\": \"{name}\", \"rule_idx\": {idx}, \"fired\": {}, \"labeler_load_bearing\": {}, \"strict_load_bearing\": {}}}{}\n",
                agg.fired,
                agg.labeler_load_bearing,
                agg.strict_load_bearing,
                if i + 1 < rows.len() { "," } else { "" }
            ));
        }
        json.push_str("  ],\n");
        json.push_str(&format!("  \"never_fired_rules\": {:?},\n", never_fired));
        json.push_str(&format!(
            "  \"quiesced_before_cap_count\": {},\n",
            quiesced.len()
        ));
        json.push_str(&format!(
            "  \"exhausted_budget_count\": {},\n",
            exhausted.len()
        ));
        let exceeded_production_deadline_count = measurements
            .iter()
            .filter(|m| m.exceeded_production_deadline)
            .count();
        json.push_str(&format!(
            "  \"exceeded_production_deadline_count\": {exceeded_production_deadline_count},\n"
        ));
        json.push_str("  \"per_expression\": [\n");
        for (i, m) in measurements.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"name\": {:?}, \"node_count\": {}, \"total_applications\": {}, \"labeler_load_bearing\": {}, \"strict_load_bearing\": {}, \"labeler_ratio\": {:.6}, \"strict_ratio\": {:.6}, \"extracted_cost\": {}, \"quiesced_before_cap\": {}, \"saturation_iterations\": {}, \"exceeded_production_deadline\": {}}}{}\n",
                m.name,
                m.node_count,
                m.total_applications,
                m.labeler_load_bearing,
                m.strict_load_bearing,
                m.labeler_ratio(),
                m.strict_ratio(),
                m.extracted_cost,
                m.quiesced_before_cap,
                m.saturation_iterations,
                m.exceeded_production_deadline,
                if i + 1 < measurements.len() { "," } else { "" }
            ));
        }
        json.push_str("  ]\n");
        json.push_str("}\n");

        let mut f = std::fs::File::create(out_path)
            .unwrap_or_else(|e| panic!("guide_headroom: cannot create {out_path}: {e}"));
        f.write_all(json.as_bytes())
            .unwrap_or_else(|e| panic!("guide_headroom: cannot write {out_path}: {e}"));
        eprintln!("guide_headroom: wrote {out_path}");
    }
}
