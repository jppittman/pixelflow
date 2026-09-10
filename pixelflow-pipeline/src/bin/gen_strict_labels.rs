//! Strict-label dataset minting for the saturation Guide's cold start
//! (`docs/plans/2026-08-31-guide-design-revision.md` §3 option 3 stage 1,
//! §4, §5).
//!
//! For every TRAIN-family corpus expression, this runs budgeted saturation
//! with provenance, extracts with the static latency-prior cost model, and
//! emits one training record per recorded rewrite-rule application: the
//! candidate-local feature ([`CandidateFeatures`], `pixelflow-search/src/
//! egraph/candidate.rs` — rule identity, matched e-class canonical content,
//! local neighborhood ops, budget state at firing time) paired with the
//! STRICT hindsight label ([`EpisodeLabels::compute_strict`],
//! `pixelflow-search/src/egraph/labeler.rs` — the application's output node
//! literally on the extracted derivation path, no ancestry over-
//! approximation). DEV-family expressions are minted the same way but kept
//! in a separate output file, for held-out evaluation only — never mixed
//! into the training records.
//!
//! # This is a lift, not a reimplementation
//!
//! `guide_headroom` (`pixelflow-pipeline/src/bin/guide_headroom.rs`)
//! computed this exact strict bound first, but at a point in this program's
//! history when `labeler.rs`/`provenance.rs` were explicitly off-limits, so
//! it reimplemented the chosen-node walk against the crate's public API
//! (see that binary's module doc). `EpisodeLabels::compute_strict` now
//! exists as the one shared computation (labeler.rs's own module doc calls
//! this out explicitly), so this binary calls it directly rather than
//! restating the walk a third time.
//!
//! # Budget denominator (now the registered constant, not a placeholder)
//!
//! [`Firing::registered_budget`] is documented as "the pre-registered
//! rule-application budget this episode's saturation is being measured
//! against" — a live guided loop's budget tier `B`. This offline,
//! full-saturation replay has no live per-call budget to report, so it uses
//! [`REGISTERED_PRIMARY_BUDGET_APPLICATIONS`] — the classical-band primary
//! tier B=100 fixed by `docs/plans/2026-09-01-phase3-registration.md` §4 —
//! exactly the same constant [`saturate_guided_until_applications`] imports
//! for the live guided loop. This dataset's earlier mint (see git history)
//! used a 195-application placeholder (this round's measured per-expression
//! median application count, before B was registered); that was a
//! train/deploy denominator skew — evaluating a Guide trained on one
//! denominator against a loop that divided by a different one would
//! silently change what `budget_fraction` means — caught and fixed before
//! any guided-saturation evaluation ran. `budget_fraction` in this dataset
//! is now "position relative to the registered primary tier B=100," the
//! same units a live guided loop uses.
//!
//! # Fence check
//!
//! Every corpus entry's `(band, seed)` family (parsed from its
//! `{tier}_b{band:02}_f{seed:02}_{idx:05}` name, `gen_bench_corpus`'s own
//! naming convention) is checked against `corpus_split.toml`'s
//! [`SplitManifest`]: a TRAIN corpus entry whose family the manifest does
//! not also call TRAIN (and likewise for DEV) is a corpus/manifest
//! desynchronization and fails loud rather than silently leaking a held-out
//! family into the training file.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_strict_labels -- \
//!     --corpus-dir pixelflow-pipeline/data \
//!     --manifest pixelflow-pipeline/corpus_split.toml \
//!     --train-limit 300 --dev-limit 100 \
//!     --out-train pixelflow-pipeline/data/strict_labels_train.jsonl \
//!     --out-dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --report-json docs/results/2026-09-01-strict-label-dataset.json
//! ```

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;

use pixelflow_ir::ExprArena;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::split::{Family, SplitManifest, Tier};
use pixelflow_search::egraph::{
    CandidateFeatures, CostModel, EGraph, EpisodeLabels, Firing, Label,
    REGISTERED_PRIMARY_BUDGET_APPLICATIONS, RuleId, RuleSet, SaturationStop, extract_dag,
};
use pixelflow_search::math::all_rules;

/// The canonical label of `id` within `rules`, or a digest placeholder for a
/// rule this build does not have.
///
/// Every rule-keyed table in this binary is keyed by [`RuleId`], never by a
/// position in `all_rules()`: a same-length reorder of that vector repoints
/// every positional row silently, and nothing anywhere is the wrong length
/// (docs/plans/2026-09-02-phase3-forward-port.md §2.2). The label is what
/// the JSON carries, because a label is what a downstream loader can hash
/// back into the same `RuleId`.
fn label_of(rules: &RuleSet, id: RuleId) -> String {
    rules
        .index_of(id)
        .and_then(|i| rules.label_of(i))
        .unwrap_or_else(|| format!("<rule {}>", id.get()))
}

#[derive(Parser)]
#[command(name = "gen_strict_labels")]
#[command(about = "Mint the strict-label training dataset for the saturation Guide's cold start")]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Path to the checked-in family split manifest.
    #[arg(long, default_value = "pixelflow-pipeline/corpus_split.toml")]
    manifest: String,

    /// Cap on TRAIN expressions processed (0 = all), stride-sampled across
    /// the whole file so every band/family is represented, not just a
    /// prefix.
    #[arg(long, default_value_t = 300)]
    train_limit: usize,

    /// Cap on DEV expressions processed (0 = all), same stride sampling.
    #[arg(long, default_value_t = 100)]
    dev_limit: usize,

    /// Where to write the TRAIN JSONL dataset (one line per rewrite-rule
    /// application).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_train.jsonl"
    )]
    out_train: String,

    /// Where to write the DEV JSONL dataset (held-out; same schema).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    out_dev: String,

    /// Optional JSON summary report (dataset size, positive rates, dedup
    /// hit rate).
    #[arg(long)]
    report_json: Option<String>,

    /// Skip corpus entries whose arena node count exceeds this — the
    /// per-expression application count is heavy-tailed to hundreds of
    /// thousands (`docs/results/2026-08-30-guide-headroom.md` §2.1), and a
    /// handful of such expressions dominate this binary's own runtime
    /// (confirmed by a dry run: 3 oversized expressions produced >100k
    /// applications and a multi-hundred-MB JSONL file before the rest of a
    /// 20-expression sample even started). This is a deliberate scope
    /// bound for the cold-start dataset, not a claim that large expressions
    /// don't matter — a follow-up scaled run should widen or drop this
    /// filter once the training pipeline itself is validated end to end.
    #[arg(long, default_value_t = 250)]
    max_expr_nodes: usize,

    /// Stop minting a split once its cumulative emitted-application count
    /// reaches this — a hard, deterministic (never wall-clock) ceiling on
    /// this run's own size, independent of `--max-expr-nodes` filtering
    /// working as intended. Applications denominated, per this program's
    /// binding budget-only framing.
    #[arg(long, default_value_t = 500_000)]
    max_total_applications: usize,

    /// Per-expression e-class cap passed to `saturate_with_limits`.
    /// `guide_headroom` uses production's 10,000; this binary defaults
    /// lower (see module doc "Budget denominator") because class-count
    /// growth dominates this binary's own wall-clock cost on the corpus's
    /// heavy tail, and the strict label / dedup-key measurements this
    /// binary reports do not depend on matching guide_headroom's cap
    /// exactly — only on saturating far enough that the extraction and
    /// provenance log are representative. Pass `--max-classes 10000` to
    /// reproduce guide_headroom's exact budget on a smaller expression
    /// sample.
    #[arg(long, default_value_t = 2_000)]
    max_classes: usize,
}

/// Saturation iteration budget — identical to `guide_headroom`'s (and
/// thereby `EGraph::saturate()`'s) default; called out explicitly because
/// this binary needs the `SaturationStats`/provenance `saturate()` discards.
const SATURATE_MAX_ITERS: usize = 100;
/// Safety ceiling only (see `guide_headroom`'s identical constant and its
/// module doc): this is an offline batch measurement, not the interactive
/// compiler path, so production's 500ms deadline does not apply here.
const SATURATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Parse `gen_bench_corpus`'s `{tier}_b{band:02}_f{seed:02}_{idx:05}` name
/// format back into a [`Family`]. Panics on a name that doesn't match — a
/// corpus entry this binary can't attribute to a family can't be
/// fence-checked, and silently skipping it would silently shrink the
/// dataset without saying so.
fn parse_family(name: &str) -> Family {
    let parts: Vec<&str> = name.split('_').collect();
    assert!(
        parts.len() >= 3,
        "gen_strict_labels: corpus entry name '{name}' does not match the expected \
         `{{tier}}_b{{band}}_f{{seed}}_{{idx}}` format (too few '_'-separated parts)"
    );
    let band_tok = parts[1].strip_prefix('b').unwrap_or_else(|| {
        panic!(
            "gen_strict_labels: corpus entry name '{name}' has no 'b<NN>' band token \
             where expected (got '{}')",
            parts[1]
        )
    });
    let seed_tok = parts[2].strip_prefix('f').unwrap_or_else(|| {
        panic!(
            "gen_strict_labels: corpus entry name '{name}' has no 'f<NN>' seed token \
             where expected (got '{}')",
            parts[2]
        )
    });
    let band: usize = band_tok.parse().unwrap_or_else(|e| {
        panic!(
            "gen_strict_labels: corpus entry name '{name}' band token '{band_tok}' \
                is not a number: {e}"
        )
    });
    let seed: u64 = seed_tok.parse().unwrap_or_else(|e| {
        panic!(
            "gen_strict_labels: corpus entry name '{name}' seed token '{seed_tok}' \
                is not a number: {e}"
        )
    });
    Family::new(band, seed)
}

/// Stride-sample `entries` down to at most `limit` items, evenly across the
/// whole population (matches `guide_headroom`'s sampling — corpus write
/// order is band-by-band, so a plain prefix `truncate` would silently
/// under-represent every family after the first).
fn stride_sample<T: Clone>(mut entries: Vec<T>, limit: usize) -> Vec<T> {
    if limit == 0 || limit >= entries.len() {
        return entries;
    }
    let stride = entries.len() as f64 / limit as f64;
    let mut sampled = Vec::with_capacity(limit);
    for i in 0..limit {
        let idx = ((i as f64) * stride) as usize;
        sampled.push(entries[idx.min(entries.len() - 1)].clone());
    }
    entries = sampled;
    entries
}

/// A stable, cross-run-deterministic fingerprint of a [`CandidateKey`] for
/// JSON output — the key's own fields (`rule` + the private
/// `ClassContentKey` node-shape vector) aren't directly serializable, so
/// this hashes the whole `Hash`-deriving key instead of reaching into it.
/// Used only for downstream joinability (e.g. "did this same candidate
/// appear elsewhere in this expression's replay"), never as a training
/// feature itself.
fn key_fingerprint(key: &pixelflow_search::egraph::CandidateKey) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// (Q1, median, Q3) via linear interpolation — identical convention to
/// `guide_headroom`'s own `quantiles()`, kept in step deliberately so the
/// two binaries' per-expression numbers are directly comparable.
fn quantiles(mut xs: Vec<f64>) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("gen_strict_labels: NaN ratio"));
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

/// Per-rule aggregate: fired count and strict-positive count, pooled across
/// every expression in one split (TRAIN or DEV).
#[derive(Default, Clone, Copy)]
struct RuleAgg {
    fired: usize,
    positive: usize,
}

/// Everything minted for one split (TRAIN or DEV): the JSONL records
/// written, plus the aggregates the report needs.
struct SplitStats {
    tier_name: &'static str,
    expressions: usize,
    families: std::collections::BTreeSet<Family>,
    applications: usize,
    positives: usize,
    dedup_hits: usize,
    rule_agg: BTreeMap<RuleId, RuleAgg>,
    /// Entries skipped for exceeding `--max-expr-nodes` (see `Args` doc).
    skipped_oversized: usize,
    /// Replays whose saturation stopped at a safety ceiling instead of
    /// quiescing — excluded from the dataset, reported here.
    skipped_non_quiescent: usize,
    /// `true` if `--max-total-applications` cut this split short before
    /// every sampled entry was processed.
    hit_total_cap: bool,
    /// Per-expression strict positive rate — the design doc's own headline
    /// statistic (`docs/results/2026-08-30-guide-headroom.md` §2.1 leads
    /// with the per-expression median, not the pooled ratio, precisely
    /// because heavy-tailed application counts let a few huge expressions
    /// dominate a pooled sum). Kept alongside the pooled `positive_rate()`
    /// rather than instead of it, so both are reported and the (expected)
    /// gap between them is visible, not silently collapsed to one number.
    per_expr_positive_rates: Vec<f64>,
}

impl SplitStats {
    fn new(tier_name: &'static str) -> Self {
        Self {
            tier_name,
            expressions: 0,
            families: std::collections::BTreeSet::new(),
            applications: 0,
            positives: 0,
            dedup_hits: 0,
            rule_agg: BTreeMap::new(),
            skipped_oversized: 0,
            skipped_non_quiescent: 0,
            hit_total_cap: false,
            per_expr_positive_rates: Vec::new(),
        }
    }

    fn positive_rate(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            self.positives as f64 / self.applications as f64
        }
    }

    fn dedup_hit_rate(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            self.dedup_hits as f64 / self.applications as f64
        }
    }
}

/// Mint one split (TRAIN or DEV): saturate + extract + strict-label every
/// expression, write one JSONL record per rewrite-rule application, fence-
/// check each entry's family against the manifest, and accumulate the
/// aggregates the report needs.
#[allow(clippy::too_many_arguments)]
fn mint_split(
    entries: &[(String, ExprArena, pixelflow_ir::ExprId)],
    tier: Tier,
    manifest: &SplitManifest,
    rules: &RuleSet,
    out_path: &str,
    tier_name: &'static str,
    max_expr_nodes: usize,
    max_total_applications: usize,
    max_classes: usize,
) -> SplitStats {
    let costs = CostModel::latency_prior();
    let mut stats = SplitStats::new(tier_name);
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(out_path)
            .unwrap_or_else(|e| panic!("gen_strict_labels: cannot create {out_path}: {e}")),
    );

    let n = entries.len();
    for (i, (name, arena, root)) in entries.iter().enumerate() {
        if stats.applications >= max_total_applications {
            stats.hit_total_cap = true;
            eprintln!(
                "gen_strict_labels[{tier_name}]: hit --max-total-applications ({max_total_applications}) \
                 after {}/{n} entries — stopping this split early, not silently truncating a \
                 single expression's own replay",
                i
            );
            break;
        }
        let expr_node_count = arena.len();
        if expr_node_count > max_expr_nodes {
            stats.skipped_oversized += 1;
            continue;
        }
        let family = parse_family(name);
        let assigned = manifest.tier_of(family);
        assert_eq!(
            assigned,
            Some(tier),
            "gen_strict_labels: fence check failed — corpus entry '{name}' parses to \
             {family}, but corpus_split.toml assigns that family to {assigned:?}, not \
             {tier} — the {tier} corpus file and the split manifest have desynchronized; \
             regenerate the corpus against the current manifest rather than minting labels \
             from a stale/leaking split"
        );
        stats.families.insert(family);

        let mut egraph = EGraph::with_rules(all_rules());
        let root_class = pixelflow_search::egraph::insert(
            arena,
            *root,
            &mut egraph,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let saturate_started = std::time::Instant::now();
        let sat_stats =
            egraph.saturate_with_limits(SATURATE_MAX_ITERS, max_classes, SATURATE_TIMEOUT);
        let saturate_elapsed = saturate_started.elapsed();
        assert!(
            saturate_elapsed < SATURATE_TIMEOUT,
            "gen_strict_labels: expression '{name}' ran {saturate_elapsed:?}, hitting the \
             {SATURATE_TIMEOUT:?} safety ceiling — fail loud rather than mint labels from a \
             truncated, non-representative replay"
        );
        // The elapsed-time assertion above only catches the wall-clock
        // ceiling. A replay that stopped at the class cap or the iteration
        // ceiling was cut wherever that safety limit happened to land, and
        // its strict labels and final-graph features then describe the graph
        // the cutoff left behind rather than a settled one — contamination
        // that would flow straight into the trained checkpoint. Excluded and
        // counted, never minted in silence.
        if sat_stats.stop != SaturationStop::Quiesced {
            stats.skipped_non_quiescent += 1;
            eprintln!(
                "gen_strict_labels[{tier_name}]: excluding '{name}' — saturation stopped \
                 with {:?}, not Quiesced",
                sat_stats.stop
            );
            continue;
        }

        let extraction = extract_dag(&egraph, root_class, &costs);
        let labels = EpisodeLabels::compute_strict(&egraph, extraction.root, &extraction.choices);

        let total_applications = egraph.provenance().recorded_count();
        let mut seen_keys: std::collections::HashSet<pixelflow_search::egraph::CandidateKey> =
            std::collections::HashSet::new();
        let mut expr_positives = 0usize;

        for (app_id, record) in egraph.provenance().applications() {
            let rule = record.rule.unwrap_or_else(|| {
                panic!(
                    "gen_strict_labels: application {app_id:?} on '{name}' carries no RuleId \
                     — the graph was built without rule ids, and every table here is keyed \
                     by identity"
                )
            });
            let firing = Firing {
                rule,
                match_root: record.match_root,
                application_ordinal: app_id.as_u64(),
                registered_budget: REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
            };
            let features = CandidateFeatures::observe(&egraph, &firing);
            let is_repeat = !seen_keys.insert(features.key.clone());
            if is_repeat {
                stats.dedup_hits += 1;
            }

            let label = labels.labels.get(&app_id).unwrap_or_else(|| {
                panic!(
                    "gen_strict_labels: application {app_id:?} recorded in provenance has no \
                     entry in EpisodeLabels::compute_strict's output for '{name}' — \
                     from_load_bearing's contract is one label per recorded application, \
                     this is a labeler bug, not something to paper over"
                )
            });
            let positive = *label == Label::LoadBearing;

            let rule_name = label_of(rules, rule);

            // A histogram, not the raw list: `neighborhood_ops` walks every
            // node in the matched class times every child's op nodes, and a
            // class deep into saturation can hold thousands of near-
            // duplicate node shapes (exactly the idempotent-refire
            // over-growth §2.2 measured) — the raw per-occurrence list blew
            // one dataset row past 20KB on a handful of pathological
            // expressions during this binary's own dry run. A histogram is
            // bounded by the number of distinct `OpKind` variants (a small,
            // fixed constant) regardless of class size, and is a strictly
            // more useful fixed-width representation for a feature encoder
            // than a variable-length occurrence list anyway.
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
                "{{\"expr_name\":{:?},\"tier\":{:?},\"family_band\":{},\"family_seed\":{},\
                 \"expr_node_count\":{},\"rule_id\":{},\"rule_name\":{:?},\
                 \"application_ordinal\":{},\"budget_fraction\":{:.6},\
                 \"match_class_node_count\":{},\"neighborhood_op_count\":{},\
                 \"neighborhood_op_hist\":{},\
                 \"candidate_key_fingerprint\":{},\"dedup_repeat\":{},\"label_positive\":{}}}",
                name,
                tier_name,
                family.band,
                family.seed,
                arena.len(),
                rule.get(),
                rule_name,
                app_id.as_u64(),
                features.budget_fraction(),
                features.key.content.node_count(),
                features.neighborhood_ops.len(),
                hist_json,
                key_fingerprint(&features.key),
                is_repeat,
                positive,
            )
            .unwrap_or_else(|e| panic!("gen_strict_labels: write to {out_path} failed: {e}"));

            stats.applications += 1;
            if positive {
                stats.positives += 1;
                expr_positives += 1;
            }
            let agg = stats.rule_agg.entry(rule).or_default();
            agg.fired += 1;
            if positive {
                agg.positive += 1;
            }
        }

        assert_eq!(
            total_applications,
            egraph.provenance().recorded_count(),
            "gen_strict_labels: provenance application count changed while iterating '{name}' \
             — should be impossible against an immutable-after-saturation egraph"
        );

        stats.expressions += 1;
        if total_applications > 0 {
            stats
                .per_expr_positive_rates
                .push(expr_positives as f64 / total_applications as f64);
        }
        if (i + 1) % 50 == 0 || i + 1 == n {
            eprintln!(
                "gen_strict_labels[{tier_name}]: {}/{n} expressions minted ({} applications so far)",
                i + 1,
                stats.applications
            );
        }
    }

    out.flush()
        .unwrap_or_else(|e| panic!("gen_strict_labels: flushing {out_path} failed: {e}"));
    stats
}

fn print_split_report(stats: &SplitStats, rules: &RuleSet) {
    println!(
        "=== {} split: {} expressions, {} families, {} applications \
         ({} skipped as oversized{}) ===",
        stats.tier_name,
        stats.expressions,
        stats.families.len(),
        stats.applications,
        stats.skipped_oversized,
        if stats.hit_total_cap {
            ", hit --max-total-applications early"
        } else {
            ""
        }
    );
    println!(
        "  strict positive rate, pooled: {:.4} ({}/{})",
        stats.positive_rate(),
        stats.positives,
        stats.applications
    );
    {
        let (q1, med, q3) = quantiles(stats.per_expr_positive_rates.clone());
        println!(
            "  strict positive rate, per-expression: median {med:.4}  Q1 {q1:.4}  Q3 {q3:.4} \
             (n={} expressions)",
            stats.per_expr_positive_rates.len()
        );
    }
    println!(
        "  dedup-key hit rate (repeat (rule, class-content) pairs): {:.4} ({}/{})",
        stats.dedup_hit_rate(),
        stats.dedup_hits,
        stats.applications
    );
    println!();
    println!(
        "  {:<28} {:>8} {:>10} {:>8}",
        "rule", "fired", "positive", "rate"
    );
    let mut rows: Vec<(RuleId, RuleAgg)> = stats.rule_agg.iter().map(|(&k, &v)| (k, v)).collect();
    rows.sort_by(|a, b| {
        let ra = if a.1.fired > 0 {
            a.1.positive as f64 / a.1.fired as f64
        } else {
            0.0
        };
        let rb = if b.1.fired > 0 {
            b.1.positive as f64 / b.1.fired as f64
        } else {
            0.0
        };
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.get().cmp(&b.0.get()))
    });
    for (id, agg) in &rows {
        let name = label_of(rules, *id);
        let rate = if agg.fired > 0 {
            agg.positive as f64 / agg.fired as f64
        } else {
            0.0
        };
        println!(
            "  {:<28} {:>8} {:>10} {:>7.1}%",
            name,
            agg.fired,
            agg.positive,
            rate * 100.0
        );
    }
    println!();
}

fn write_json_report(path: &str, train: &SplitStats, dev: &SplitStats, rules: &RuleSet) {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!(
        "  \"registered_budget_b\": {REGISTERED_PRIMARY_BUDGET_APPLICATIONS},\n"
    ));

    let write_split = |json: &mut String, s: &SplitStats| {
        json.push_str(&format!("    \"tier\": \"{}\",\n", s.tier_name));
        json.push_str(&format!("    \"expressions\": {},\n", s.expressions));
        json.push_str(&format!("    \"families\": {},\n", s.families.len()));
        json.push_str(&format!(
            "    \"skipped_non_quiescent\": {},\n",
            s.skipped_non_quiescent
        ));
        json.push_str(&format!(
            "    \"skipped_oversized\": {},\n",
            s.skipped_oversized
        ));
        json.push_str(&format!("    \"hit_total_cap\": {},\n", s.hit_total_cap));
        json.push_str(&format!("    \"applications\": {},\n", s.applications));
        json.push_str(&format!("    \"positives\": {},\n", s.positives));
        json.push_str(&format!(
            "    \"positive_rate_pooled\": {:.6},\n",
            s.positive_rate()
        ));
        {
            let (q1, med, q3) = quantiles(s.per_expr_positive_rates.clone());
            json.push_str(&format!(
                "    \"positive_rate_per_expr_quartiles\": [{q1:.6}, {med:.6}, {q3:.6}],\n"
            ));
        }
        json.push_str(&format!("    \"dedup_hits\": {},\n", s.dedup_hits));
        json.push_str(&format!(
            "    \"dedup_hit_rate\": {:.6},\n",
            s.dedup_hit_rate()
        ));
        json.push_str("    \"per_rule\": [\n");
        let mut rows: Vec<(RuleId, RuleAgg)> = s.rule_agg.iter().map(|(&k, &v)| (k, v)).collect();
        rows.sort_by_key(|(id, _)| id.get());
        for (i, (id, agg)) in rows.iter().enumerate() {
            let name = label_of(rules, *id);
            let rate = if agg.fired > 0 {
                agg.positive as f64 / agg.fired as f64
            } else {
                0.0
            };
            json.push_str(&format!(
                "      {{\"rule\": \"{name}\", \"rule_id\": {}, \"fired\": {}, \
                 \"positive\": {}, \"positive_rate\": {rate:.6}}}{}\n",
                id.get(),
                agg.fired,
                agg.positive,
                if i + 1 < rows.len() { "," } else { "" }
            ));
        }
        json.push_str("    ]\n");
    };

    json.push_str("  \"train\": {\n");
    write_split(&mut json, train);
    json.push_str("  },\n");
    json.push_str("  \"dev\": {\n");
    write_split(&mut json, dev);
    json.push_str("  }\n");
    json.push_str("}\n");

    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("gen_strict_labels: cannot create {path}: {e}"));
    f.write_all(json.as_bytes())
        .unwrap_or_else(|e| panic!("gen_strict_labels: cannot write {path}: {e}"));
    eprintln!("gen_strict_labels: wrote {path}");
}

fn main() {
    let args = Args::parse();

    let manifest = SplitManifest::load(&PathBuf::from(&args.manifest)).unwrap_or_else(|e| {
        panic!(
            "gen_strict_labels: failed to load split manifest {}: {e}",
            args.manifest
        )
    });

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let train_path = corpus_dir.join("corpus_train.bin");
    let dev_path = corpus_dir.join("corpus_dev.bin");

    let train_entries = read_corpus(&train_path).unwrap_or_else(|e| {
        panic!(
            "gen_strict_labels: failed to read {}: {e}",
            train_path.display()
        )
    });
    let dev_entries = read_corpus(&dev_path).unwrap_or_else(|e| {
        panic!(
            "gen_strict_labels: failed to read {}: {e}",
            dev_path.display()
        )
    });

    eprintln!(
        "gen_strict_labels: corpus has {} TRAIN, {} DEV expressions available",
        train_entries.len(),
        dev_entries.len()
    );

    let train_entries = stride_sample(train_entries, args.train_limit);
    let dev_entries = stride_sample(dev_entries, args.dev_limit);

    eprintln!(
        "gen_strict_labels: minting {} TRAIN, {} DEV expressions (registered budget B={REGISTERED_PRIMARY_BUDGET_APPLICATIONS})",
        train_entries.len(),
        dev_entries.len()
    );

    let rules = RuleSet::production();

    let train_stats = mint_split(
        &train_entries,
        Tier::Train,
        &manifest,
        &rules,
        &args.out_train,
        "train",
        args.max_expr_nodes,
        args.max_total_applications,
        args.max_classes,
    );
    eprintln!("gen_strict_labels: wrote {}", args.out_train);

    let dev_stats = mint_split(
        &dev_entries,
        Tier::Dev,
        &manifest,
        &rules,
        &args.out_dev,
        "dev",
        args.max_expr_nodes,
        args.max_total_applications,
        args.max_classes,
    );
    eprintln!("gen_strict_labels: wrote {}", args.out_dev);

    print_split_report(&train_stats, &rules);
    print_split_report(&dev_stats, &rules);

    println!(
        "=== class balance ===\n  TRAIN positive rate {:.4} ({:.1}% of applications are the \
         minority LoadBearing class) — a naive trainer needs positive-class loss weighting \
         (e.g. inverse-frequency or focal loss) or it will collapse to predicting Wasted \
         everywhere and still score >{:.0}% raw accuracy.",
        train_stats.positive_rate(),
        train_stats.positive_rate() * 100.0,
        (1.0 - train_stats.positive_rate()) * 100.0,
    );

    if let Some(report_path) = &args.report_json {
        write_json_report(report_path, &train_stats, &dev_stats, &rules);
    }
}
