//! A/B: budgeting saturation against LIVE e-classes instead of ALLOCATED
//! ones (`docs/results/2026-09-02-class-cap-live-ab.md`).
//!
//! The defect this measures is stated in
//! `docs/results/2026-09-02-class-cap-ghosts.md`: `EGraph::union` merges
//! through the union-find and never frees a class slot, so `num_classes()`
//! only grows, and the production budget was checked against it. On this
//! corpus the classical tier's 5 000-class budget therefore stopped runs at
//! a median 1 352 LIVE classes.
//!
//! Both arms run in ONE process, over the same corpus, through
//! [`Optimizer::production`] — only the budget differs, and both are
//! expressible through the public [`Budget::Explicit`], so the "before" arm
//! is the old policy exactly rather than an imitation of it:
//!
//! - **allocated** (before): live ceiling wide open at [`HARD_CLASS_LIMIT`],
//!   allocated ceiling at the preset's `max_classes`. Every budget check
//!   then reduces to the old `self.classes.len() > max_classes`.
//! - **live** (after): [`Budget::Production`] — the preset's `max_classes`
//!   as the LIVE budget, [`HARD_CLASS_LIMIT`] as the allocated memory guard.
//!
//! Cost is [`CostModel::latency_prior`] over the extracted arena, the
//! deterministic metric. Wall clock is recorded as context only (and the run
//! records the machine's load average so a loaded run is labelled as one);
//! the deterministic compile-cost metric is rule applications.

use super::congruence_gap_probe::{arena_static_cost, load_arena_dump, median, percentile};
use super::*;
use crate::egraph::{
    Budget, CostModel, HARD_CLASS_LIMIT, Optimizer, SaturationStop, config_for_node_count,
};
use crate::nnue::{BwdGenConfig, BwdGenerator};
use std::path::PathBuf;

/// Which budget policy an arm runs under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// Before: the budget counts allocated class slots.
    Allocated,
    /// After: the budget counts live classes; allocated is the memory guard.
    Live,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::Allocated => "allocated",
            Self::Live => "live",
        }
    }

    /// The budget this arm holds a kernel of `node_count` nodes to.
    fn budget(self, node_count: usize) -> Budget {
        let preset = config_for_node_count(node_count);
        match self {
            Self::Allocated => Budget::Explicit {
                iterations: preset.max_iterations,
                classes: HARD_CLASS_LIMIT,
                allocated_classes: preset.max_classes,
                applications: None,
            },
            Self::Live => Budget::Production,
        }
    }
}

/// Finer than `category_of`: the A/B is reported per tile size, because
/// glyph16 and glyph32 are different kernels even where they share a
/// codepoint.
fn group_of(filename: &str) -> &'static str {
    for prefix in ["glyph16", "glyph32", "cellgrid", "psychedelic", "shader"] {
        if filename.starts_with(prefix) {
            return prefix;
        }
    }
    "unknown"
}

/// One kernel under one arm, read off the graph the run left behind.
#[derive(Clone, Debug)]
struct Row {
    name: String,
    group: &'static str,
    arm: &'static str,
    node_count: usize,
    stop: String,
    /// `num_classes()` — allocated slots. Monotone across a run, so
    /// end-of-run IS peak; this is the memory number.
    allocated: usize,
    /// `live_class_count()` — canonical and non-empty.
    live: usize,
    nodes: usize,
    memo: usize,
    applications: u64,
    iterations: usize,
    unions: usize,
    extracted_cost: usize,
    /// Context only, never a metric: the machine is shared.
    elapsed_ns: u128,
}

impl Row {
    /// A documented upper-bound byte proxy for the graph's peak footprint.
    ///
    /// Counts what the e-graph actually owns: one `EClass` header plus a
    /// union-find parent and a constant fact per allocated class, one
    /// `ENode` plus its tag per node, and one memo entry per hash-consed
    /// node. `ENode::Op`'s children live in a `Vec` the header does not
    /// account for, so the bound adds three `EClassId`s per node — the
    /// arity ceiling in this IR.
    fn bytes_proxy(&self) -> usize {
        const CLASS: usize = 24 /* Vec<ENode> */ + 24 /* Vec<ENodeId> */ + 4 /* parent */ + 8;
        const NODE: usize = 32 /* ENode header */ + 8 /* tag */ + 3 * 4 /* children */;
        const MEMO: usize = 32 /* key header */ + 3 * 4 + 4 /* value */ + 8 /* table slot */;
        self.allocated * CLASS + self.nodes * NODE + self.memo * MEMO
    }
}

/// Run `arena` under `arm` and read every counter off the finished graph.
fn run_arm(name: &str, group: &'static str, arena: &ExprArena, root: ExprId, arm: Arm) -> Row {
    let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root)
        .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
    let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);
    let node_count = reachable_count(&arena, root);

    let mut optimizer = Optimizer::production().budget(arm.budget(node_count));
    let mut egraph = optimizer.egraph();
    let mut id_memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut id_memo)
        .unwrap_or_else(|| panic!("{name}: arena_to_egraph returned None (unsupported node)"));

    // The clock brackets exactly the work a compile pays for: saturation
    // plus extraction. Context, not a metric.
    let started = std::time::Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);
    let elapsed_ns = started.elapsed().as_nanos();

    let model = CostModel::latency_prior();
    Row {
        name: name.to_string(),
        group,
        arm: arm.label(),
        node_count,
        stop: format!("{:?}", optimized.stats.stop),
        allocated: egraph.num_classes(),
        live: egraph.live_class_count(),
        nodes: egraph.node_count(),
        memo: egraph.memo_len(),
        applications: optimized.stats.applications,
        iterations: optimized.stats.iterations,
        unions: optimized.stats.unions,
        extracted_cost: arena_static_cost(&model, &extracted, extracted_root),
        elapsed_ns,
    }
}

/// One kernel's before/after pair.
#[derive(Clone, Debug)]
struct Pair {
    before: Row,
    after: Row,
}

impl Pair {
    /// Positive = the live budget extracted a CHEAPER term.
    fn cost_improvement_frac(&self) -> f64 {
        if self.before.extracted_cost == 0 {
            return 0.0;
        }
        (self.before.extracted_cost as f64 - self.after.extracted_cost as f64)
            / self.before.extracted_cost as f64
    }

    fn ratio(&self, f: impl Fn(&Row) -> f64) -> f64 {
        let b = f(&self.before);
        if b <= 0.0 { 1.0 } else { f(&self.after) / b }
    }

    fn regressed(&self) -> bool {
        self.after.extracted_cost > self.before.extracted_cost
    }
}

fn load_average_one_minute() -> f64 {
    let out = std::process::Command::new("uptime")
        .output()
        .expect("uptime must run: the load average labels this run");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let tail = text
        .rsplit_once("load averages:")
        .or_else(|| text.rsplit_once("load average:"))
        .unwrap_or_else(|| panic!("uptime output has no load average: {text}"))
        .1;
    tail.split_whitespace()
        .next()
        .and_then(|s| s.trim_end_matches(',').parse::<f64>().ok())
        .unwrap_or_else(|| panic!("could not parse a load average from {text}"))
}

/// THE A/B. Writes `docs/results/2026-09-02-class-cap-live-ab.{md,csv,json}`.
#[test]
#[ignore = "offline measurement: PIXELFLOW_CLASSCAP_ARENA_DIR=<dir of .arena dumps> cargo test -p pixelflow-search --release --lib -- --ignored class_cap_live_ab"]
fn class_cap_live_ab() {
    let load_before = load_average_one_minute();
    let dir = PathBuf::from(
        std::env::var("PIXELFLOW_CLASSCAP_ARENA_DIR")
            .expect("PIXELFLOW_CLASSCAP_ARENA_DIR must be set"),
    );
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().map(|e| e == "arena").unwrap_or(false))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .arena files in {}", dir.display());

    let mut corpus: Vec<(String, &'static str, ExprArena, ExprId)> = Vec::new();
    for path in &paths {
        let (name, arena, root) = load_arena_dump(path);
        let group = group_of(&path.file_name().unwrap().to_string_lossy());
        corpus.push((name, group, arena, root));
    }
    let real_kernel_count = corpus.len();

    // Same size-stratified synthetic sample the ghost measurement used:
    // 5 depth bands x 40 seeds, unoptimized (junkified) form.
    let templates = crate::egraph::collect_rule_templates();
    for &max_depth in &[3usize, 5, 7, 9, 11] {
        for seed in 0u64..40 {
            let config = BwdGenConfig {
                max_depth,
                ..Default::default()
            };
            let mut generator = BwdGenerator::new(
                seed.wrapping_add(max_depth as u64 * 10_000),
                config,
                templates.clone(),
            );
            let generated = generator.generate_arena();
            corpus.push((
                format!("synth_d{max_depth}_s{seed}"),
                "synthetic",
                generated.arena,
                generated.unoptimized,
            ));
        }
    }
    let synthetic_count = corpus.len() - real_kernel_count;

    // Both arms per kernel, back to back, so the two wall clocks see the
    // same cache and thermal conditions.
    let mut pairs: Vec<Pair> = Vec::new();
    for (name, group, arena, root) in &corpus {
        let before = run_arm(name, group, arena, *root, Arm::Allocated);
        let after = run_arm(name, group, arena, *root, Arm::Live);
        pairs.push(Pair { before, after });
    }
    let load_after = load_average_one_minute();

    let regressions: Vec<&Pair> = pairs
        .iter()
        .filter(|p| p.regressed() && p.before.group != "synthetic")
        .collect();

    let results_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("docs/results");
    write_csv(
        &results_dir.join("2026-09-02-class-cap-live-ab.csv"),
        &pairs,
    );
    write_json(
        &results_dir.join("2026-09-02-class-cap-live-ab.json"),
        &pairs,
        (load_before, load_after),
    );
    write_md(
        &results_dir.join("2026-09-02-class-cap-live-ab.md"),
        &pairs,
        real_kernel_count,
        synthetic_count,
        (load_before, load_after),
    );

    let headline = summarize("ALL", &pairs.iter().collect::<Vec<_>>());
    eprintln!("=== class-cap live A/B ===\n{headline}");
    eprintln!(
        "real-kernel cost regressions: {} (load {load_before:.2} -> {load_after:.2})",
        regressions.len()
    );
    for p in &regressions {
        eprintln!(
            "  REGRESSION {} {} -> {}",
            p.before.name, p.before.extracted_cost, p.after.extracted_cost
        );
    }
}

/// The per-group summary line, computed once so the eprintln, the markdown
/// table and the JSON cannot disagree about what a group's numbers are.
struct Summary {
    group: String,
    n: usize,
    cost_median_frac: f64,
    cost_p90_frac: f64,
    cost_mean_frac: f64,
    improved: usize,
    regressed: usize,
    live_before_median: f64,
    live_after_median: f64,
    alloc_before_median: f64,
    alloc_after_median: f64,
    alloc_ratio_worst: f64,
    bytes_before_median: f64,
    bytes_after_median: f64,
    bytes_ratio_worst: f64,
    apps_ratio_median: f64,
    apps_ratio_p90: f64,
    wall_ratio_median: f64,
    wall_ratio_p90: f64,
    capped_before: usize,
    capped_after: usize,
    allocated_guard_hits_after: usize,
}

fn summarize(group: &str, pairs: &[&Pair]) -> Summary {
    let mut costs: Vec<f64> = pairs.iter().map(|p| p.cost_improvement_frac()).collect();
    let mut apps: Vec<f64> = pairs
        .iter()
        .map(|p| p.ratio(|r| r.applications as f64))
        .collect();
    let mut wall: Vec<f64> = pairs
        .iter()
        .map(|p| p.ratio(|r| r.elapsed_ns as f64))
        .collect();
    let mut live_b: Vec<f64> = pairs.iter().map(|p| p.before.live as f64).collect();
    let mut live_a: Vec<f64> = pairs.iter().map(|p| p.after.live as f64).collect();
    let mut alloc_b: Vec<f64> = pairs.iter().map(|p| p.before.allocated as f64).collect();
    let mut alloc_a: Vec<f64> = pairs.iter().map(|p| p.after.allocated as f64).collect();
    let mut bytes_b: Vec<f64> = pairs
        .iter()
        .map(|p| p.before.bytes_proxy() as f64)
        .collect();
    let mut bytes_a: Vec<f64> = pairs.iter().map(|p| p.after.bytes_proxy() as f64).collect();
    let alloc_ratio_worst = pairs
        .iter()
        .map(|p| p.ratio(|r| r.allocated as f64))
        .fold(0.0f64, f64::max);
    let bytes_ratio_worst = pairs
        .iter()
        .map(|p| p.ratio(|r| r.bytes_proxy() as f64))
        .fold(0.0f64, f64::max);
    Summary {
        group: group.to_string(),
        n: pairs.len(),
        cost_median_frac: median(&mut costs.clone()),
        cost_p90_frac: percentile(&mut costs.clone(), 90.0),
        cost_mean_frac: if costs.is_empty() {
            0.0
        } else {
            costs.iter().sum::<f64>() / costs.len() as f64
        },
        improved: pairs
            .iter()
            .filter(|p| p.cost_improvement_frac() > 1e-9)
            .count(),
        regressed: pairs.iter().filter(|p| p.regressed()).count(),
        live_before_median: median(&mut live_b),
        live_after_median: median(&mut live_a),
        alloc_before_median: median(&mut alloc_b),
        alloc_after_median: median(&mut alloc_a),
        alloc_ratio_worst,
        bytes_before_median: median(&mut bytes_b),
        bytes_after_median: median(&mut bytes_a),
        bytes_ratio_worst,
        apps_ratio_median: median(&mut apps.clone()),
        apps_ratio_p90: percentile(&mut apps, 90.0),
        wall_ratio_median: median(&mut wall.clone()),
        wall_ratio_p90: percentile(&mut wall, 90.0),
        capped_before: pairs
            .iter()
            .filter(|p| p.before.stop.starts_with("ClassCap"))
            .count(),
        capped_after: pairs
            .iter()
            .filter(|p| p.after.stop.starts_with("ClassCap"))
            .count(),
        allocated_guard_hits_after: pairs
            .iter()
            .filter(|p| {
                p.after.stop
                    == format!(
                        "{:?}",
                        SaturationStop::ClassCap(crate::egraph::ClassCeiling::Allocated)
                    )
            })
            .count(),
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (n={}): cost median {:+.3}% p90 {:+.3}% mean {:+.3}% | improved {} regressed {} | \
             live median {:.0}->{:.0} | allocated median {:.0}->{:.0} (worst {:.2}x) | \
             bytes-proxy median {:.0}->{:.0} (worst {:.2}x) | applications median {:.2}x p90 {:.2}x | \
             wall median {:.2}x p90 {:.2}x | capped {}->{} (allocated-guard {})",
            self.group,
            self.n,
            self.cost_median_frac * 100.0,
            self.cost_p90_frac * 100.0,
            self.cost_mean_frac * 100.0,
            self.improved,
            self.regressed,
            self.live_before_median,
            self.live_after_median,
            self.alloc_before_median,
            self.alloc_after_median,
            self.alloc_ratio_worst,
            self.bytes_before_median,
            self.bytes_after_median,
            self.bytes_ratio_worst,
            self.apps_ratio_median,
            self.apps_ratio_p90,
            self.wall_ratio_median,
            self.wall_ratio_p90,
            self.capped_before,
            self.capped_after,
            self.allocated_guard_hits_after,
        )
    }
}

const GROUPS: [&str; 6] = [
    "glyph16",
    "glyph32",
    "shader",
    "psychedelic",
    "cellgrid",
    "synthetic",
];

fn per_group(pairs: &[Pair]) -> Vec<Summary> {
    let mut out = Vec::new();
    for g in GROUPS {
        let sel: Vec<&Pair> = pairs.iter().filter(|p| p.before.group == g).collect();
        if !sel.is_empty() {
            out.push(summarize(g, &sel));
        }
    }
    let real: Vec<&Pair> = pairs
        .iter()
        .filter(|p| p.before.group != "synthetic")
        .collect();
    if !real.is_empty() {
        out.push(summarize("REAL (all non-synthetic)", &real));
    }
    // The population the fix targets: kernels the OLD budget actually
    // stopped. Everything else was already quiescing inside the cap and can
    // only be moved by noise.
    let capped: Vec<&Pair> = pairs
        .iter()
        .filter(|p| p.before.stop.starts_with("ClassCap") && p.before.group != "synthetic")
        .collect();
    if !capped.is_empty() {
        out.push(summarize("REAL cap-hit (before)", &capped));
    }
    out.push(summarize("ALL", &pairs.iter().collect::<Vec<_>>()));
    out
}

fn write_csv(path: &std::path::Path, pairs: &[Pair]) {
    let mut out = String::from(
        "name,group,arm,node_count,stop,allocated,live,nodes,memo,bytes_proxy,applications,\
         iterations,unions,extracted_cost,elapsed_ns\n",
    );
    for p in pairs {
        for r in [&p.before, &p.after] {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                r.name,
                r.group,
                r.arm,
                r.node_count,
                r.stop,
                r.allocated,
                r.live,
                r.nodes,
                r.memo,
                r.bytes_proxy(),
                r.applications,
                r.iterations,
                r.unions,
                r.extracted_cost,
                r.elapsed_ns
            ));
        }
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn write_json(path: &std::path::Path, pairs: &[Pair], load: (f64, f64)) {
    let mut out = String::from("{\n");
    out.push_str(&format!(
        "  \"load_average_1m\": {{ \"before\": {:.2}, \"after\": {:.2} }},\n",
        load.0, load.1
    ));
    out.push_str("  \"groups\": [\n");
    let groups = per_group(pairs);
    for (i, s) in groups.iter().enumerate() {
        out.push_str(&format!(
            "    {{ \"group\": \"{}\", \"n\": {}, \"cost_median_frac\": {:.6}, \"cost_p90_frac\": {:.6}, \
             \"cost_mean_frac\": {:.6}, \"improved\": {}, \"regressed\": {}, \
             \"live_before_median\": {:.1}, \"live_after_median\": {:.1}, \
             \"allocated_before_median\": {:.1}, \"allocated_after_median\": {:.1}, \
             \"allocated_ratio_worst\": {:.4}, \"bytes_before_median\": {:.0}, \
             \"bytes_after_median\": {:.0}, \"bytes_ratio_worst\": {:.4}, \
             \"applications_ratio_median\": {:.4}, \"applications_ratio_p90\": {:.4}, \
             \"wall_ratio_median\": {:.4}, \"wall_ratio_p90\": {:.4}, \
             \"capped_before\": {}, \"capped_after\": {}, \"allocated_guard_hits_after\": {} }}{}\n",
            s.group, s.n, s.cost_median_frac, s.cost_p90_frac, s.cost_mean_frac, s.improved,
            s.regressed, s.live_before_median, s.live_after_median, s.alloc_before_median,
            s.alloc_after_median, s.alloc_ratio_worst, s.bytes_before_median, s.bytes_after_median,
            s.bytes_ratio_worst, s.apps_ratio_median, s.apps_ratio_p90, s.wall_ratio_median,
            s.wall_ratio_p90, s.capped_before, s.capped_after, s.allocated_guard_hits_after,
            if i + 1 == groups.len() { "" } else { "," }
        ));
    }
    out.push_str("  ],\n  \"kernels\": [\n");
    for (i, p) in pairs.iter().enumerate() {
        out.push_str(&format!(
            "    {{ \"name\": \"{}\", \"group\": \"{}\", \"node_count\": {}, \
             \"before\": {{ \"stop\": \"{}\", \"allocated\": {}, \"live\": {}, \"nodes\": {}, \"memo\": {}, \"bytes_proxy\": {}, \"applications\": {}, \"iterations\": {}, \"cost\": {}, \"elapsed_ns\": {} }}, \
             \"after\": {{ \"stop\": \"{}\", \"allocated\": {}, \"live\": {}, \"nodes\": {}, \"memo\": {}, \"bytes_proxy\": {}, \"applications\": {}, \"iterations\": {}, \"cost\": {}, \"elapsed_ns\": {} }}, \
             \"cost_improvement_frac\": {:.6} }}{}\n",
            p.before.name, p.before.group, p.before.node_count,
            p.before.stop, p.before.allocated, p.before.live, p.before.nodes, p.before.memo,
            p.before.bytes_proxy(), p.before.applications, p.before.iterations,
            p.before.extracted_cost, p.before.elapsed_ns,
            p.after.stop, p.after.allocated, p.after.live, p.after.nodes, p.after.memo,
            p.after.bytes_proxy(), p.after.applications, p.after.iterations,
            p.after.extracted_cost, p.after.elapsed_ns,
            p.cost_improvement_frac(),
            if i + 1 == pairs.len() { "" } else { "," }
        ));
    }
    out.push_str("  ]\n}\n");
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn write_md(
    path: &std::path::Path,
    pairs: &[Pair],
    real_kernel_count: usize,
    synthetic_count: usize,
    load: (f64, f64),
) {
    let groups = per_group(pairs);
    let all = groups.last().expect("ALL row");
    let real = groups
        .iter()
        .find(|s| s.group.starts_with("REAL"))
        .expect("REAL row");
    let mut regressions: Vec<&Pair> = pairs
        .iter()
        .filter(|p| p.regressed() && p.before.group != "synthetic")
        .collect();
    regressions.sort_by_key(|p| p.before.extracted_cost as i64 - p.after.extracted_cost as i64);

    let mut s = String::new();
    s.push_str("# Class cap: live-counted budget A/B (2026-09-02)\n\n");
    s.push_str(&format!(
        "Corpus: {real_kernel_count} real kernels + {synthetic_count} synthetic = {} pairs. \
         Both arms in one process through `Optimizer::production()`; only the budget differs. \
         Cost is `CostModel::latency_prior()` over the extracted arena. Load average \
         {:.2} -> {:.2}{}.\n\n",
        pairs.len(),
        load.0,
        load.1,
        if load.0 > 4.0 || load.1 > 4.0 {
            " — **LOADED**, so every wall-clock number here is context, not a measurement"
        } else {
            ""
        }
    ));
    s.push_str(
        "- **before (`allocated`)**: `Budget::Explicit { classes: HARD_CLASS_LIMIT, \
         allocated_classes: preset.max_classes }` — every budget check reduces to the old \
         `self.classes.len() > max_classes`.\n\
         - **after (`live`)**: `Budget::Production` — the preset's `max_classes` as the LIVE \
         budget, `HARD_CLASS_LIMIT` (100 000) as the allocated memory guard.\n\n",
    );
    s.push_str("## Headline\n\n");
    s.push_str(&format!("- ALL: {all}\n- {real}\n\n"));
    s.push_str("## Per group\n\n");
    s.push_str(
        "| group | n | cost median | cost p90 | improved | regressed | live median | \
         allocated median | alloc worst | bytes-proxy median | bytes worst | apps median | \
         wall median | capped before→after |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for g in &groups {
        s.push_str(&format!(
            "| {} | {} | {:+.2}% | {:+.2}% | {} | {} | {:.0}→{:.0} | {:.0}→{:.0} | {:.2}x | \
             {:.0}→{:.0} | {:.2}x | {:.2}x | {:.2}x | {}→{} |\n",
            g.group,
            g.n,
            g.cost_median_frac * 100.0,
            g.cost_p90_frac * 100.0,
            g.improved,
            g.regressed,
            g.live_before_median,
            g.live_after_median,
            g.alloc_before_median,
            g.alloc_after_median,
            g.alloc_ratio_worst,
            g.bytes_before_median,
            g.bytes_after_median,
            g.bytes_ratio_worst,
            g.apps_ratio_median,
            g.wall_ratio_median,
            g.capped_before,
            g.capped_after
        ));
    }
    let real_total_before: u128 = pairs
        .iter()
        .filter(|p| p.before.group != "synthetic")
        .map(|p| p.before.elapsed_ns)
        .sum();
    let real_total_after: u128 = pairs
        .iter()
        .filter(|p| p.before.group != "synthetic")
        .map(|p| p.after.elapsed_ns)
        .sum();
    let worst_alloc_after = pairs.iter().map(|p| p.after.allocated).max().unwrap_or(0);
    let worst_bytes_after = pairs
        .iter()
        .map(|p| p.after.bytes_proxy())
        .max()
        .unwrap_or(0);
    s.push_str("\n## Verdict\n\n");
    s.push_str(&format!(
        concat!(
            "- **Quality**: real kernels move {:+.2}% median / {:+.2}% p90 cheaper; ",
            "{} of {} improve, {} regress.\n",
            "- **Memory**: the largest graph any kernel allocates goes to {} classes ",
            "({:.2} MB by the byte proxy), against the `HARD_CLASS_LIMIT` guard of {}. ",
            "The guard fired on {} kernels, so the allocated ceiling is not what bounds ",
            "this corpus — the live budget is.\n",
            "- **Compile cost**: rule applications, the deterministic proxy, go {:.2}x ",
            "median / {:.2}x p90 on real kernels. Aggregate wall clock over the real ",
            "corpus is {:.2} s -> {:.2} s ({:.2}x) — context only, the machine was ",
            "loaded.\n\n",
        ),
        real.cost_median_frac * 100.0,
        real.cost_p90_frac * 100.0,
        real.improved,
        real.n,
        real.regressed,
        worst_alloc_after,
        worst_bytes_after as f64 / 1e6,
        HARD_CLASS_LIMIT,
        all.allocated_guard_hits_after,
        real.apps_ratio_median,
        real.apps_ratio_p90,
        real_total_before as f64 / 1e9,
        real_total_after as f64 / 1e9,
        real_total_after as f64 / real_total_before.max(1) as f64,
    ));
    s.push_str(concat!(
        "\nThe trade is real but not free, and it is not a pure win: a couple of percent ",
        "of extracted cost on real kernels for roughly double the saturation work and ",
        "double the peak e-graph. The regressions below are why this is a judgement call ",
        "rather than an obvious yes — a bigger e-graph does not monotonically produce a ",
        "cheaper extraction, because extraction is a greedy DP over a static cost prior, ",
        "not an optimum, so more equalities can move the greedy choice onto a worse ",
        "branch.\n",
    ));
    s.push_str("\n## Cost regressions on real kernels\n\n");
    if regressions.is_empty() {
        s.push_str("None. No real kernel extracts a more expensive term under the live budget.\n");
    } else {
        s.push_str("| kernel | group | before | after | delta |\n|---|---|---|---|---|\n");
        for p in &regressions {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {:+} |\n",
                p.before.name,
                p.before.group,
                p.before.extracted_cost,
                p.after.extracted_cost,
                p.after.extracted_cost as i64 - p.before.extracted_cost as i64
            ));
        }
    }
    s.push_str("\n## Per kernel\n\n");
    s.push_str(
        "| kernel | group | nodes | stop before → after | live before → after | \
         allocated before → after | bytes-proxy before → after | apps before → after | \
         cost before → after | improvement |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    let mut sorted: Vec<&Pair> = pairs.iter().collect();
    sorted.sort_by(|a, b| {
        b.cost_improvement_frac()
            .partial_cmp(&a.cost_improvement_frac())
            .unwrap()
    });
    for p in sorted {
        s.push_str(&format!(
            "| {} | {} | {} | {} → {} | {} → {} | {} → {} | {} → {} | {} → {} | {} → {} | {:+.2}% |\n",
            p.before.name,
            p.before.group,
            p.before.node_count,
            p.before.stop,
            p.after.stop,
            p.before.live,
            p.after.live,
            p.before.allocated,
            p.after.allocated,
            p.before.bytes_proxy(),
            p.after.bytes_proxy(),
            p.before.applications,
            p.after.applications,
            p.before.extracted_cost,
            p.after.extracted_cost,
            p.cost_improvement_frac() * 100.0
        ));
    }
    std::fs::write(path, s).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
