//! Guided-regression bisect harness (main era: `Optimizer::guide`).
//!
//! The same measurement as the retired-branch `phase3_bisect_eval`: the
//! Phase 3 arms on the guided grid, with BOTH cost shapes at every
//! checkpoint (`tree_dp` = `ChoiceCost::tree`, the DP objective every
//! Round-1 number was read from; `dag` = `ChoiceCost::dag`, #1117). On this
//! era both come straight off `run_anytime_curve`, so `tree_walk == tree_dp`
//! by construction and is carried only so the row schema is one schema.
//!
//! Arms:
//! - `unguided`
//! - `control`: per-rule TRAIN rate mapped by `rule_idx` to the live rule at
//!   that index, i.e. what Round 1b's
//!   `PerRuleRateGuide::from_train_guide_report` did.
//! - `control_label`: the same report mapped by `rule_name` label, which is
//!   what `per_rule_rate_guide_from_report` does on this era and what the
//!   2026-09-02 bilinear eval ran. The report's labels are the bare family
//!   names (`associative`, not `associative(Add)`), so on this era every
//!   specialized rule silently gets rate 0.0.
//! - `linear`: the additive strict-label Guide.
//!
//! The optimizer is built exactly as `phase3_at_budget_eval::arm_optimizer`
//! builds it, so the `control_label` / `linear` rows reproduce that harness.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;
use serde_json::Value;

use pixelflow_ir::{Environment, ExprData, Rooted, Term};
use pixelflow_pipeline::schema::fnv1a64_hex;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::guide_linear::{
    load_linear_guide, per_rule_rate_guide_from_report,
};
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::{
    AnytimeCurveOutput, Budget, CostModel, KeepJournal, Optimizer, RuleSet, SaturationStop,
    config_for_node_count, run_anytime_curve,
};
use pixelflow_search::nnue::guide::SaturationGuide;
use pixelflow_search::nnue::guide::linear::PerRuleRateGuide;

/// The registered guided grid (docs/plans/2026-09-01-phase3-registration.md).
const GRID: &[usize] = &[25, 50, 100, 200, 400, 800];
const SWEEP_SAFETY_CEILING: usize = 10_000;
const SAFETY_TIMEOUT: Duration = Duration::from_secs(1800);

#[derive(Parser)]
#[command(name = "phase3_bisect_eval")]
struct Args {
    /// Corpus file to evaluate.
    #[arg(long)]
    corpus: String,
    /// Directory holding `corpus_train.bin` (the TRAIN fence source).
    #[arg(long)]
    corpus_dir: String,
    /// Only entries whose name starts with this prefix. Empty = every entry.
    #[arg(long, default_value = "")]
    name_prefix: String,
    /// `train_guide` checkpoint for the linear arm.
    #[arg(long)]
    checkpoint: String,
    /// `train_guide` report JSON (per-rule TRAIN rates for the control arms).
    #[arg(long)]
    train_guide_report: String,
    /// Per-expression rows, appended; names already present are skipped.
    #[arg(long)]
    out_jsonl: String,
    /// Free-text label for the source revision / lever configuration.
    #[arg(long)]
    era: String,
    /// Evaluate at most this many selected expressions (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(Serialize)]
struct Checkpoint {
    app_target: usize,
    app_actual: usize,
    sweeps: usize,
    classes: usize,
    nodes: usize,
    tree_dp: usize,
    tree_walk: usize,
    dag: usize,
    stop: String,
    clamped: bool,
}

#[derive(Serialize)]
struct Arm {
    checkpoints: Vec<Checkpoint>,
    ended: String,
    ended_at_apps: usize,
    seen_keys: Option<usize>,
}

#[derive(Serialize)]
struct Row {
    name: String,
    era: String,
    corpus_fnv64: String,
    node_count: usize,
    class_cap: usize,
    arms: BTreeMap<String, Arm>,
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

fn tier_is_classical(node_count: usize) -> bool {
    node_count > 50
}

fn arm_optimizer(
    class_cap: usize,
    costs: &CostModel,
    guide: Option<Box<dyn SaturationGuide>>,
) -> Optimizer {
    Optimizer::production()
        .cost(costs.clone())
        .guide(guide)
        .observe(Some(Box::new(KeepJournal)))
        .budget(Budget::Explicit {
            iterations: SWEEP_SAFETY_CEILING,
            classes: class_cap,
            applications: None,
        })
        .hard_ceiling(SAFETY_TIMEOUT)
}

fn curve_to_arm(out: &AnytimeCurveOutput, seen_keys: Option<usize>) -> Arm {
    let c = &out.curve;
    Arm {
        checkpoints: c
            .checkpoints
            .iter()
            .map(|k| Checkpoint {
                app_target: k.app_target,
                app_actual: k.app_actual,
                sweeps: k.sweeps,
                classes: k.classes,
                nodes: k.nodes,
                tree_dp: k.cost.tree,
                tree_walk: k.cost.tree,
                dag: k.cost.dag,
                stop: stop_name(k.stop).to_string(),
                clamped: k.clamped,
            })
            .collect(),
        ended: stop_name(c.ended).to_string(),
        ended_at_apps: c.ended_at_apps,
        seen_keys,
    }
}

fn run_arm(
    guide: Option<Box<dyn SaturationGuide>>,
    term: Term<'_>,
    class_cap: usize,
    costs: &CostModel,
) -> Arm {
    let guided = guide.is_some();
    let mut optimizer = arm_optimizer(class_cap, costs, guide);
    let out = run_anytime_curve(&mut optimizer, term, GRID);
    let seen = if guided {
        Some(
            optimizer
                .guided_keys_seen()
                .expect("a guided optimizer carries an episode"),
        )
    } else {
        None
    };
    curve_to_arm(&out, seen)
}

/// Round 1b's control mapping: `per_rule[i].rule_idx` names the position in
/// the production rule vector; the rate attaches to whatever rule is there.
fn control_by_index(path: &Path, rules: &RuleSet) -> PerRuleRateGuide {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("train_guide report {}: {e}", path.display()));
    let v: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("train_guide report {}: {e}", path.display()));
    let rows = v["per_rule"]
        .as_array()
        .unwrap_or_else(|| panic!("train_guide report {}: no per_rule", path.display()));
    let mut labelled: Vec<(String, f32)> = Vec::with_capacity(rows.len());
    for row in rows {
        let idx = row["rule_idx"]
            .as_u64()
            .unwrap_or_else(|| panic!("per_rule row without rule_idx: {row}"))
            as usize;
        let rate = row["train_positive_rate"]
            .as_f64()
            .unwrap_or_else(|| panic!("per_rule row without train_positive_rate: {row}"))
            as f32;
        let label = rules
            .label_of(idx)
            .unwrap_or_else(|| panic!("rule_idx {idx} is outside the {} live rules", rules.len()));
        labelled.push((label, rate));
    }
    PerRuleRateGuide::from_labels(&labelled)
}

fn enforce_train_fence(corpus_dir: &Path, entries: &[(String, Rooted<ExprData>)]) {
    let path = corpus_dir.join("corpus_train.bin");
    assert!(
        path.exists(),
        "TRAIN corpus not found at {}",
        path.display()
    );
    let train: HashSet<FenceKey> = read_corpus(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .iter()
        .map(|(_, e)| FenceKey::of(e.entry()))
        .collect();
    let collisions: Vec<&str> = entries
        .iter()
        .filter(|(_, e)| train.contains(&FenceKey::of(e.entry())))
        .map(|(n, _)| n.as_str())
        .collect();
    assert!(
        collisions.is_empty(),
        "TRAIN-fence violation: {} entries collide: {:?}",
        collisions.len(),
        &collisions[..collisions.len().min(10)]
    );
    eprintln!(
        "bisect: TRAIN fence OK — {} entries probed against {} TRAIN structures",
        entries.len(),
        train.len()
    );
}

fn existing_names(path: &Path) -> HashSet<String> {
    let Ok(f) = std::fs::File::open(path) else {
        return HashSet::new();
    };
    std::io::BufReader::new(f)
        .lines()
        .map(|l| l.expect("read jsonl line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(&l).expect("parse jsonl row");
            v["name"].as_str().expect("row has a name").to_string()
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    let corpus_path = PathBuf::from(&args.corpus);
    let corpus_bytes = std::fs::read(&corpus_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", corpus_path.display()));
    let corpus_fnv64 = fnv1a64_hex(&corpus_bytes);
    let entries = read_corpus(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", corpus_path.display()));
    enforce_train_fence(Path::new(&args.corpus_dir), &entries);

    let selected: Vec<&(String, Rooted<ExprData>)> = entries
        .iter()
        .filter(|(name, expr)| name.starts_with(&args.name_prefix) && tier_is_classical(expr.len()))
        .collect();
    let selected = if args.limit > 0 {
        selected.into_iter().take(args.limit).collect()
    } else {
        selected
    };

    let costs = CostModel::latency_prior();
    let rules = RuleSet::production();
    let control_index = control_by_index(Path::new(&args.train_guide_report), &rules);
    let control_label = per_rule_rate_guide_from_report(Path::new(&args.train_guide_report))
        .unwrap_or_else(|e| panic!("control guide (label-mapped): {e}"));
    let linear = load_linear_guide(Path::new(&args.checkpoint), &rules)
        .unwrap_or_else(|e| panic!("linear guide: {e}"));

    let out_path = PathBuf::from(&args.out_jsonl);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let done = existing_names(&out_path);
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", out_path.display()));

    eprintln!(
        "bisect[{}]: {} selected from {} ({} already done), corpus fnv64 {corpus_fnv64}, rules {} (fingerprint {})",
        args.era,
        selected.len(),
        corpus_path.display(),
        done.len(),
        rules.len(),
        rules.fingerprint()
    );
    let total = selected.len();
    // A corpus entry declares no buffers or uniforms — the format refuses to
    // write one down — so one empty environment serves every term.
    let env = Environment::new();
    for (i, (name, expr)) in selected.into_iter().enumerate() {
        if done.contains(name) {
            continue;
        }
        let term = Term::new(expr.entry(), &env);
        let node_count = expr.len();
        let class_cap = config_for_node_count(node_count).max_classes;
        let started = Instant::now();

        let mut arms = BTreeMap::new();
        arms.insert(
            "unguided".to_string(),
            run_arm(None, term, class_cap, &costs),
        );
        arms.insert(
            "control".to_string(),
            run_arm(
                Some(Box::new(control_index.clone())),
                term,
                class_cap,
                &costs,
            ),
        );
        arms.insert(
            "control_label".to_string(),
            run_arm(
                Some(Box::new(control_label.clone())),
                term,
                class_cap,
                &costs,
            ),
        );
        arms.insert(
            "linear".to_string(),
            run_arm(Some(Box::new(linear.clone())), term, class_cap, &costs),
        );

        let row = Row {
            name: name.clone(),
            era: args.era.clone(),
            corpus_fnv64: corpus_fnv64.clone(),
            node_count,
            class_cap,
            arms,
        };
        let line = serde_json::to_string(&row).expect("serialize row");
        writeln!(out, "{line}").expect("write row");
        out.flush().expect("flush row");
        eprintln!(
            "bisect[{}]: [{}/{}] {} ({} nodes) {:.1}s",
            args.era,
            i + 1,
            total,
            name,
            node_count,
            started.elapsed().as_secs_f64()
        );
    }
}
