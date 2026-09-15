//! Strict-label credit re-count on one DEV family, by output NODE versus by
//! output CLASS (`docs/results/2026-09-01-strict-label-constant-output-blindspot.md`).
//!
//! **A measurement, not a fix.** Round 1b
//! (`docs/results/2026-09-01-phase3-round1b-domain-shift.md`) found that
//! `pythagorean` (sin²x + cos²x → 1) fired 231 times across 19 `sh`
//! expressions under unguided saturation and was strict-positive 0 / 231
//! times. The strict hindsight label (`EpisodeLabels::compute_strict`) credits
//! an application iff a node *it minted* is on the extracted path. A rewrite
//! whose right-hand side already exists in the e-graph — a constant the seed
//! expression already carried, a child term, a fold target — is a memo hit in
//! `EGraph::add`: no node is minted, no `Origin::Rule` is recorded, and the
//! firing's only effect is a union, which the strict label never reads. This
//! binary re-runs the same unguided arm the Round-1b harness ran and, for
//! every recorded application, classifies:
//!
//! - **effect** — `Minted` (created at least one e-node), `UnionOnly` (its
//!   RHS was a memo hit, but the union merged two distinct classes), or
//!   `NoOp` (memo hit into an already-equal class);
//! - **strict** — output node on the extracted path (the label as minted);
//! - **class_on_path** — the application's output e-class (canonical match
//!   root at extraction time) is one of the extracted path's classes, i.e.
//!   the *value* the rewrite established is what the extraction consumes,
//!   whoever's node happens to represent it;
//! - **tight** / **loose** — `compute_tight` / `compute` on the same episode.
//!
//! The `strict ⊆ minted` and `strict ⊆ tight ⊆ loose` inclusions are asserted
//! on every episode, not assumed.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin strict_label_output_class_recount -- \
//!     --corpus pixelflow-pipeline/data/corpus_dev_ood.bin --name-prefix dev_sh_ \
//!     --out-json docs/results/2026-09-01-strict-label-constant-output-blindspot.json
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;

use pixelflow_ir::{ExprArena, ExprNode};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, ApplicationId, Budget, CostModel, EClassId, EGraph,
    ENode, ENodeId, EpisodeLabels, KeepJournal, Optimizer, Origin, RuleId, RuleSet,
    config_for_node_count, run_anytime_curve,
};

/// Per-curve safety ceilings — identical to `phase3_at_budget_eval`'s, so the
/// unguided arm here IS that harness's unguided arm (same grid, same class
/// cap rule, same rules, same cost model).
const SAFETY_TIMEOUT: Duration = Duration::from_secs(1800);
const SWEEP_SAFETY_CEILING: usize = 10_000;

#[derive(Parser)]
#[command(name = "strict_label_output_class_recount")]
#[command(about = "Re-count strict-label credit by output node vs output class on one DEV family")]
struct Args {
    /// DEV-side corpus file (multi-family; select one with --name-prefix).
    #[arg(long, default_value = "pixelflow-pipeline/data/corpus_dev_ood.bin")]
    corpus: String,
    /// Evaluate only entries whose name starts with this prefix.
    #[arg(long, default_value = "dev_sh_")]
    name_prefix: String,
    /// Where the per-rule tallies and per-expression rows are written.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-strict-label-constant-output-blindspot.json"
    )]
    out_json: String,
    /// Stop after this many expressions (smoke runs).
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Effect {
    /// The firing created at least one e-node (`Origin::Rule(app)` exists).
    Minted,
    /// RHS was a memo hit; the firing's union merged two distinct classes.
    UnionOnly,
    /// RHS was a memo hit into a class already equal to the match root.
    NoOp,
}

#[derive(Clone, Default, Serialize)]
struct RuleTally {
    name: String,
    fired: usize,
    minted: usize,
    union_only: usize,
    no_op: usize,
    /// Output node on the extracted path — `compute_strict`, the label as minted.
    strict: usize,
    /// Output class (canonical match root at extraction time) on the path.
    class_on_path: usize,
    /// `class_on_path` and the firing was not a `NoOp`.
    class_on_path_effective: usize,
    tight: usize,
    loose: usize,
    expressions_fired: usize,
    /// When the output class is on the path: who minted the node the
    /// extraction chose in that class (`"seed"` or a rule name).
    chosen_origin_when_class_on_path: BTreeMap<String, usize>,
}

#[derive(Clone, Serialize)]
struct PythagoreanRow {
    name: String,
    node_count: usize,
    /// The seed arena already carries a literal `1.0` — the RHS pre-exists.
    seed_has_const_one: bool,
    fired: usize,
    minted: usize,
    union_only: usize,
    no_op: usize,
    strict: usize,
    class_on_path: usize,
    tight: usize,
    /// Origin of the node the extraction chose in `pythagorean`'s output
    /// class, for each firing whose class is on the path (deduplicated).
    chosen_origin_in_output_class: BTreeSet<String>,
    /// The extracted expression, rendered structurally.
    extracted: String,
    /// Whether any `Const(1)` survives on the extracted path at all.
    const_one_on_path: bool,
    /// Which rules DID earn strict credit on this expression (rule → count).
    strict_credited_rules: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct Report {
    corpus: String,
    name_prefix: String,
    expressions: usize,
    total_applications: usize,
    /// Keyed by `all_rules()` index (as a string, for JSON), only rules that
    /// fired at least once.
    by_rule_idx: BTreeMap<String, RuleTally>,
    pythagorean: Vec<PythagoreanRow>,
}

/// The extracted expression, rendered structurally: `Add(1, v1)`.
fn render_chosen(eg: &EGraph, choices: &[Option<usize>], class: EClassId) -> String {
    let c = eg.find(class);
    let idx = choices[c.index()].unwrap_or_else(|| panic!("class e{} has no choice", c.index()));
    match &eg.nodes(c)[idx] {
        ENode::Var(v) => format!("v{v}"),
        ENode::Const(bits) => format!("{}", f32::from_bits(*bits)),
        ENode::Buffer(_) => "buf".to_string(),
        ENode::Uniform(_) => "uniform".to_string(),
        ENode::Param(i) => format!("p{i}"),
        ENode::Op { op, children } => {
            let kids: Vec<String> = children
                .iter()
                .map(|&k| render_chosen(eg, choices, k))
                .collect();
            format!("{:?}({})", op.kind(), kids.join(", "))
        }
        ENode::Reduce { fold, body } => format!(
            "Reduce[{}..{}]({})",
            fold.range().start,
            fold.range().end,
            render_chosen(eg, choices, *body)
        ),
    }
}

/// The extracted path: its classes, and the tag the extraction chose in each.
fn chosen_path(out: &AnytimeCurveOutput) -> HashMap<EClassId, ENodeId> {
    let eg = &out.egraph;
    let choices = &out.extraction.choices;
    let mut chosen: HashMap<EClassId, ENodeId> = HashMap::new();
    let mut stack = vec![out.root];
    while let Some(c) = stack.pop() {
        let c = eg.find(c);
        if chosen.contains_key(&c) {
            continue;
        }
        let idx = choices[c.index()].unwrap_or_else(|| {
            panic!(
                "class {} reachable via the chosen extraction has no choice",
                c.index()
            )
        });
        chosen.insert(c, eg.tags(c)[idx]);
        stack.extend(eg.nodes(c)[idx].children());
    }
    chosen
}

/// The canonical label of a recorded application's rule.
///
/// Keyed by [`RuleId`], never by a position in `all_rules()`: a same-length
/// reorder repoints every positional key silently
/// (docs/plans/2026-09-02-phase3-forward-port.md §2.2).
fn rule_name(rules: &RuleSet, id: RuleId) -> String {
    rules
        .index_of(id)
        .and_then(|i| rules.label_of(i))
        .unwrap_or_else(|| format!("<rule {}>", id.get()))
}

/// The [`RuleId`] a recorded application carries, or a loud failure.
fn rule_of(rec: &pixelflow_search::egraph::ApplicationRecord) -> RuleId {
    rec.rule.unwrap_or_else(|| {
        panic!(
            "strict_label_output_class_recount: an application carries no RuleId — the graph \
             was built without rule ids, and every table here is keyed by identity"
        )
    })
}

fn origin_label(eg: &EGraph, rules: &RuleSet, tag: ENodeId) -> String {
    match eg.provenance().origin(tag) {
        Some(Origin::Seed) => "seed".to_string(),
        Some(Origin::Rule(app)) => {
            let rec = eg
                .provenance()
                .application(app)
                .unwrap_or_else(|| panic!("no record for {app:?}"));
            rule_name(rules, rule_of(rec))
        }
        None => panic!("chosen node {tag:?} has no recorded origin"),
    }
}

fn seed_has_const_one(arena: &ExprArena) -> bool {
    arena
        .nodes_raw()
        .iter()
        .any(|n| matches!(n, ExprNode::Const(v) if v.to_bits() == 1.0_f32.to_bits()))
}

struct Labels {
    strict: BTreeSet<ApplicationId>,
    tight: BTreeSet<ApplicationId>,
    loose: BTreeSet<ApplicationId>,
}

fn labels_of(out: &AnytimeCurveOutput) -> Labels {
    let eg = &out.egraph;
    let root = out.root;
    let choices = &out.extraction.choices;
    let strict = EpisodeLabels::compute_strict(eg, root, choices).load_bearing;
    let tight = EpisodeLabels::compute_tight(eg, root, choices).load_bearing;
    let loose = EpisodeLabels::compute(eg, root, choices).load_bearing;
    assert!(
        strict.is_subset(&tight),
        "strict ⊄ tight: {} strict, {} tight",
        strict.len(),
        tight.len()
    );
    assert!(
        tight.is_subset(&loose),
        "tight ⊄ loose: {} tight, {} loose",
        tight.len(),
        loose.len()
    );
    Labels {
        strict,
        tight,
        loose,
    }
}

fn tally_expression(
    name: &str,
    arena: &ExprArena,
    out: &AnytimeCurveOutput,
    rules: &RuleSet,
    by_rule: &mut BTreeMap<RuleId, RuleTally>,
) -> Option<PythagoreanRow> {
    let eg = &out.egraph;
    let prov = eg.provenance();
    let labels = labels_of(out);
    let path = chosen_path(out);

    let minted: HashSet<ApplicationId> = prov
        .origins()
        .filter_map(|(_, o)| match o {
            Origin::Rule(app) => Some(app),
            Origin::Seed => None,
        })
        .collect();
    let merged: HashSet<ApplicationId> = prov
        .union_events()
        .iter()
        .filter_map(|u| u.application_id)
        .collect();

    let mut fired_here: BTreeSet<RuleId> = BTreeSet::new();
    let mut pyth: Option<PythagoreanRow> = None;

    for (app, rec) in prov.applications() {
        let effect = if minted.contains(&app) {
            Effect::Minted
        } else if merged.contains(&app) {
            Effect::UnionOnly
        } else {
            Effect::NoOp
        };
        let is_strict = labels.strict.contains(&app);
        assert!(
            !is_strict || effect == Effect::Minted,
            "{name}: {app:?} is strict-positive but minted no node — the strict label \
             credits only minted nodes by construction"
        );
        let out_class = eg.find(rec.match_root);
        let chosen_here = path.get(&out_class).copied();
        let class_on_path = chosen_here.is_some();
        let is_tight = labels.tight.contains(&app);
        let is_loose = labels.loose.contains(&app);

        let rid = rule_of(rec);
        let rname = rule_name(rules, rid);
        let t = by_rule.entry(rid).or_default();
        t.name = rname.clone();
        t.fired += 1;
        match effect {
            Effect::Minted => t.minted += 1,
            Effect::UnionOnly => t.union_only += 1,
            Effect::NoOp => t.no_op += 1,
        }
        t.strict += usize::from(is_strict);
        t.class_on_path += usize::from(class_on_path);
        t.class_on_path_effective += usize::from(class_on_path && effect != Effect::NoOp);
        t.tight += usize::from(is_tight);
        t.loose += usize::from(is_loose);
        let chosen_origin = chosen_here.map(|tag| origin_label(eg, rules, tag));
        if let Some(o) = &chosen_origin {
            *t.chosen_origin_when_class_on_path
                .entry(o.clone())
                .or_default() += 1;
        }
        fired_here.insert(rid);

        if rname == "pythagorean" {
            let row = pyth.get_or_insert_with(|| PythagoreanRow {
                name: name.to_string(),
                node_count: arena.nodes_raw().len(),
                seed_has_const_one: seed_has_const_one(arena),
                fired: 0,
                minted: 0,
                union_only: 0,
                no_op: 0,
                strict: 0,
                class_on_path: 0,
                tight: 0,
                chosen_origin_in_output_class: BTreeSet::new(),
                extracted: render_chosen(eg, &out.extraction.choices, out.root),
                const_one_on_path: path.iter().any(|(c, tag)| {
                    let i = eg
                        .tags(*c)
                        .iter()
                        .position(|t| t == tag)
                        .expect("tag in class");
                    matches!(eg.nodes(*c)[i], ENode::Const(b) if b == 1.0_f32.to_bits())
                }),
                strict_credited_rules: BTreeMap::new(),
            });
            row.fired += 1;
            match effect {
                Effect::Minted => row.minted += 1,
                Effect::UnionOnly => row.union_only += 1,
                Effect::NoOp => row.no_op += 1,
            }
            row.strict += usize::from(is_strict);
            row.class_on_path += usize::from(class_on_path);
            row.tight += usize::from(is_tight);
            if let Some(o) = chosen_origin {
                row.chosen_origin_in_output_class.insert(o);
            }
        }
    }
    for id in fired_here {
        by_rule
            .get_mut(&id)
            .expect("tallied above")
            .expressions_fired += 1;
    }
    if let Some(row) = pyth.as_mut() {
        for app in &labels.strict {
            let rec = prov.application(*app).expect("record");
            *row.strict_credited_rules
                .entry(rule_name(rules, rule_of(rec)))
                .or_default() += 1;
        }
    }
    pyth
}

fn print_table(by_rule: &BTreeMap<RuleId, RuleTally>) {
    println!(
        "| id | rule | fired | minted | union-only | no-op | strict (node) | class on path | class on path, effective | tight | loose | exprs |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (id, t) in by_rule {
        println!(
            "| {id} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            t.name,
            t.fired,
            t.minted,
            t.union_only,
            t.no_op,
            t.strict,
            t.class_on_path,
            t.class_on_path_effective,
            t.tight,
            t.loose,
            t.expressions_fired
        );
    }
}

fn main() {
    let args = Args::parse();
    let corpus_path = Path::new(&args.corpus);
    let entries = read_corpus(corpus_path)
        .unwrap_or_else(|e| panic!("failed to read corpus {}: {e}", corpus_path.display()));
    let selected: Vec<&(String, ExprArena, pixelflow_ir::ExprId)> = entries
        .iter()
        .filter(|(name, _, _)| name.starts_with(&args.name_prefix))
        .take(args.limit.unwrap_or(usize::MAX))
        .collect();
    assert!(
        !selected.is_empty(),
        "no corpus entries with prefix {:?} in {}",
        args.name_prefix,
        corpus_path.display()
    );
    eprintln!(
        "{} entries selected with prefix {:?} from {}",
        selected.len(),
        args.name_prefix,
        corpus_path.display()
    );

    let costs = CostModel::latency_prior();
    let rules = RuleSet::production();
    let mut by_rule: BTreeMap<RuleId, RuleTally> = BTreeMap::new();
    let mut pythagorean_rows = Vec::new();
    let mut total_applications = 0usize;
    let started = Instant::now();

    for (i, (name, arena, root)) in selected.iter().enumerate() {
        let class_cap = config_for_node_count(arena.nodes_raw().len()).max_classes;
        let t0 = Instant::now();
        let mut optimizer = Optimizer::production()
            .cost(costs.clone())
            // This harness reads the journal after the run, so recording has
            // to be asked for: `Optimizer` records only for an observer
            // (#1118), and production sets none.
            .observe(Some(Box::new(KeepJournal)))
            .budget(Budget::Explicit {
                iterations: SWEEP_SAFETY_CEILING,
                classes: class_cap,
                applications: None,
            })
            .hard_ceiling(SAFETY_TIMEOUT);
        let out = run_anytime_curve(&mut optimizer, arena, *root, APP_CHECKPOINT_GRID);
        // The budget denominator, not the journal's own count.
        let apps = out.egraph.application_count() as usize;
        total_applications += apps;
        let row = tally_expression(name, arena, &out, &rules, &mut by_rule);
        let pyth_note = row
            .as_ref()
            .map(|r| {
                format!(
                    " pythagorean fired={} minted={} strict={} class_on_path={} tight={} seed_one={}",
                    r.fired, r.minted, r.strict, r.class_on_path, r.tight, r.seed_has_const_one
                )
            })
            .unwrap_or_default();
        eprintln!(
            "[{}/{}] {name}: {apps} applications in {:.1?} (elapsed {:.0?}){pyth_note}",
            i + 1,
            selected.len(),
            t0.elapsed(),
            started.elapsed()
        );
        if let Some(r) = row {
            pythagorean_rows.push(r);
        }
    }

    print_table(&by_rule);
    println!();
    for r in &pythagorean_rows {
        println!(
            "- `{}` ({} nodes): extracted `{}`; Const(1) on path: {}; strict credit went to {:?}",
            r.name, r.node_count, r.extracted, r.const_one_on_path, r.strict_credited_rules
        );
    }
    println!();
    println!(
        "pythagorean: {} expressions, fired={} minted={} union_only={} no_op={} strict={} class_on_path={} tight={} seed_has_const_one={}",
        pythagorean_rows.len(),
        pythagorean_rows.iter().map(|r| r.fired).sum::<usize>(),
        pythagorean_rows.iter().map(|r| r.minted).sum::<usize>(),
        pythagorean_rows.iter().map(|r| r.union_only).sum::<usize>(),
        pythagorean_rows.iter().map(|r| r.no_op).sum::<usize>(),
        pythagorean_rows.iter().map(|r| r.strict).sum::<usize>(),
        pythagorean_rows
            .iter()
            .map(|r| r.class_on_path)
            .sum::<usize>(),
        pythagorean_rows.iter().map(|r| r.tight).sum::<usize>(),
        pythagorean_rows
            .iter()
            .filter(|r| r.seed_has_const_one)
            .count(),
    );

    let report = Report {
        corpus: args.corpus.clone(),
        name_prefix: args.name_prefix.clone(),
        expressions: selected.len(),
        total_applications,
        by_rule_idx: by_rule
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        pythagorean: pythagorean_rows,
    };
    let json = serde_json::to_string_pretty(&report).expect("report serializes");
    std::fs::write(&args.out_json, json)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", args.out_json));
    eprintln!("wrote {}", args.out_json);
}
