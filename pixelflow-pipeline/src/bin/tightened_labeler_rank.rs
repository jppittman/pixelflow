//! Tightened-labeler re-measurement for Phase 3 (`docs/plans/2026-08-31-guide-design-revision.md`
//! §3, option 3, stage-2 prep).
//!
//! `guide_headroom` already measures two points on the over-approximation
//! spectrum — the loose labeler (`derivation_ancestors`) and the strict
//! lower bound (node literally on the extracted path) — and found their
//! per-rule ranking correlates only moderately (Spearman ρ ≈ 0.35,
//! `docs/results/2026-08-30-guide-headroom.md`), with structural/congruence
//! rules (commutative, fma-fusion, distribute, reverse-associative,
//! associative, identity) scoring 63-84% under the labeler bound and ~0%
//! under the strict one.
//!
//! `pixelflow-search::egraph::derivation_ancestors_tight` (and
//! `EpisodeLabels::compute_tight`) narrow the labeler's three named
//! over-approximation axes without touching the original — see that
//! function's doc comment. This binary answers the design doc's open
//! question directly: does narrowing those axes move the structural-rule
//! class toward the strict bound (they really are mostly waste, and the
//! labeler was simply loose), or does substantial enabling-credit survive
//! tightening (the labeler bound was pointing at something real, and stage-2
//! refinement will matter for Phase 3 training)?
//!
//! For every measured expression this runs the identical pipeline
//! `guide_headroom` uses (`EGraph::saturate_with_limits` with the same
//! generous, non-binding safety ceiling, `CostModel::latency_prior()`
//! extraction) and computes all three labelers —
//! [`EpisodeLabels::compute`] (loose), [`EpisodeLabels::compute_tight`]
//! (tight), [`EpisodeLabels::compute_strict`] (strict) — on the *same*
//! episode, so their outputs are directly comparable per expression, per
//! rule, and per individual application. Every number is a deterministic
//! count or a `CostModel::latency_prior()` cost — no wall-clock timing gates
//! correctness.
//!
//! # Rank correlation, two granularities
//!
//! - **Per-rule**: pooled (fired, load-bearing) counts per rule *name*
//!   (operator instances of the same rule merged, matching
//!   `guide-headroom.md`'s convention) give one load-bearing ratio per rule
//!   under each bound; Spearman ρ (average-rank tie handling) between each
//!   pair of bounds' ratio vectors.
//! - **Per-application**: every individual recorded application, pooled
//!   across the whole sample, contributes one `(loose_bit, tight_bit,
//!   strict_bit)` triple. Spearman's rho of two binary variables is exactly
//!   their Pearson correlation (rank-mapping a two-valued variable is a
//!   strictly increasing affine transform, which Pearson's r is invariant
//!   under) — i.e. the phi coefficient of the pair's 2x2 contingency table.
//!   Computed via three running contingency tables (O(1) memory per pair,
//!   no full sort of a multi-million-row array needed) rather than the
//!   sort-based rank function used for the (small, continuous) per-rule
//!   vectors.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin tightened_labeler_rank -- \
//!     --corpus-dir pixelflow-pipeline/data --limit 300 \
//!     --out docs/results/2026-09-01-tightened-labeler-rank.json
//! ```

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;

use pixelflow_ir::{Environment, ExprData, Rooted, Term};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    CostModel, EGraph, EpisodeLabels, RuleId, RuleSet, SaturationStop, extract_dag,
};
use pixelflow_search::math::all_rules;

/// The canonical label of `id` within `rules` — see
/// docs/plans/2026-09-02-phase3-forward-port.md §2.2 for why every rule-keyed
/// table in this binary is keyed by [`RuleId`] and never by a position in
/// `all_rules()`.
fn label_of(rules: &RuleSet, id: RuleId) -> String {
    rules
        .index_of(id)
        .and_then(|i| rules.label_of(i))
        .unwrap_or_else(|| format!("<rule {}>", id.get()))
}

/// The rule FAMILY a canonical label names: `"commutative(Mul)"` ->
/// `"commutative"`. `rule_label` is the only renderer of a rule, so this is
/// its exact inverse rather than a second naming convention.
fn family_of(label: &str) -> &str {
    label.split('(').next().unwrap_or(label)
}

#[derive(Parser)]
#[command(name = "tightened_labeler_rank")]
#[command(
    about = "Phase 3 stage-2 prep: rank correlation of tightened-labeler vs strict vs original-labeler"
)]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Minimum number of expressions to measure (task requirement: >= 300).
    #[arg(long, default_value_t = 300)]
    min_expressions: usize,

    /// Optional cap on the number of expressions measured (0 = no cap),
    /// stride-sampled across the full train+dev population (see
    /// `guide_headroom` — corpus write order is band-by-band, so a plain
    /// truncate would misrepresent "at scale").
    #[arg(long, default_value_t = 300)]
    limit: usize,

    /// Write the full structured result as JSON to this path.
    #[arg(long)]
    out: Option<String>,
}

/// Identical to `guide_headroom`'s budget — same corpus, same pipeline, so
/// the two measurements' per-expression rows are directly comparable.
const SATURATE_MAX_ITERS: usize = 100;
const SATURATE_MAX_CLASSES: usize = 10_000;
const SATURATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Rule names §2.1 of the design doc names as the structural/congruence
/// class (labeler 63-84%, strict ~0% in the original headroom measurement).
const STRUCTURAL_RULE_NAMES: &[&str] = &[
    "commutative",
    "fma-fusion",
    "distribute",
    "reverse-associative",
    "associative",
    "identity",
];

/// Rule names §2.1 names as the numeric/transcendental class (the one whose
/// two bounds already track each other closely).
const NUMERIC_RULE_NAMES: &[&str] = &[
    "power-recip",
    "power-sqrt",
    "power-rsqrt",
    "recip-sqrt",
    "even-negation",
];

/// `0.0` for zero applications (no evidence, not `NaN`) — matches
/// `RuleStats::load_bearing_ratio`'s convention.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn quantiles(mut xs: Vec<f64>) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("quantiles: NaN ratio"));
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

/// Average-rank Spearman correlation for small, continuous vectors (the
/// per-rule ratio comparisons — n is at most the rule-library size, ~62).
/// Returns `None` if either input is constant (correlation undefined, not
/// zero-by-convention — the caller decides how to report that).
fn spearman(xs: &[f64], ys: &[f64]) -> Option<f64> {
    assert_eq!(xs.len(), ys.len(), "spearman: mismatched lengths");
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("spearman: NaN input"));
        let mut ranks = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
            for k in idx.iter().take(j + 1).skip(i) {
                ranks[*k] = avg_rank;
            }
            i = j + 1;
        }
        ranks
    };
    pearson(&rank(xs), &rank(ys))
}

fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return None;
    }
    Some(cov / (vx.sqrt() * vy.sqrt()))
}

/// A running 2x2 contingency table between two binary labels, observed one
/// application at a time. `phi()` is the phi coefficient — exactly the
/// Spearman/Pearson correlation of the two binary variables (see module
/// docs' "Rank correlation" section for why).
#[derive(Default, Clone, Copy)]
struct Contingency {
    /// x=1, y=1
    n11: u64,
    /// x=1, y=0
    n10: u64,
    /// x=0, y=1
    n01: u64,
    /// x=0, y=0
    n00: u64,
}

impl Contingency {
    fn observe(&mut self, x: bool, y: bool) {
        match (x, y) {
            (true, true) => self.n11 += 1,
            (true, false) => self.n10 += 1,
            (false, true) => self.n01 += 1,
            (false, false) => self.n00 += 1,
        }
    }

    fn total(&self) -> u64 {
        self.n11 + self.n10 + self.n01 + self.n00
    }

    /// `None` if either marginal is degenerate (all-same on one side),
    /// matching `pearson`'s convention for a constant variable.
    fn phi(&self) -> Option<f64> {
        let (n11, n10, n01, n00) = (
            self.n11 as f64,
            self.n10 as f64,
            self.n01 as f64,
            self.n00 as f64,
        );
        let x_marg = (n11 + n10) * (n01 + n00);
        let y_marg = (n11 + n01) * (n10 + n00);
        if x_marg == 0.0 || y_marg == 0.0 {
            return None;
        }
        Some((n11 * n00 - n10 * n01) / (x_marg * y_marg).sqrt())
    }
}

/// Pooled per-rule counts under all three bounds.
#[derive(Default, Clone, Copy)]
struct RuleAgg {
    fired: u64,
    loose_lb: u64,
    tight_lb: u64,
    strict_lb: u64,
}

/// Per-expression summary row, written to JSON for downstream inspection.
struct ExprRow {
    name: String,
    node_count: usize,
    total_applications: u64,
    loose_lb: u64,
    tight_lb: u64,
    strict_lb: u64,
}

fn main() {
    let args = Args::parse();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let train_path = corpus_dir.join("corpus_train.bin");
    let dev_path = corpus_dir.join("corpus_dev.bin");

    let mut entries: Vec<(String, Rooted<ExprData>)> =
        read_corpus(&train_path).unwrap_or_else(|e| {
            panic!(
                "tightened_labeler_rank: failed to read {}: {e}",
                train_path.display()
            )
        });
    let dev_entries = read_corpus(&dev_path).unwrap_or_else(|e| {
        panic!(
            "tightened_labeler_rank: failed to read {}: {e}",
            dev_path.display()
        )
    });
    let train_count = entries.len();
    entries.extend(dev_entries);
    let total_available = entries.len();

    assert!(
        total_available >= args.min_expressions,
        "tightened_labeler_rank: corpus has only {total_available} expressions \
         (train {train_count} + dev {}), need >= {} — regenerate a larger corpus, \
         do not silently measure on too few expressions",
        total_available - train_count,
        args.min_expressions
    );

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
    assert!(
        n >= args.min_expressions,
        "tightened_labeler_rank: sampled {n} expressions, need >= {} — --limit was set below \
         --min-expressions, fail loud rather than silently under-measure",
        args.min_expressions
    );
    eprintln!(
        "tightened_labeler_rank: measuring {n} expressions (train {train_count}, dev {}, corpus total {total_available})",
        total_available - train_count
    );

    let costs = CostModel::latency_prior();
    let rules = RuleSet::production();

    let mut rows: Vec<ExprRow> = Vec::with_capacity(n);
    let mut rule_agg: BTreeMap<RuleId, RuleAgg> = BTreeMap::new();
    let (mut loose_vs_strict, mut tight_vs_strict, mut loose_vs_tight) = (
        Contingency::default(),
        Contingency::default(),
        Contingency::default(),
    );
    let (mut total_apps, mut total_loose, mut total_tight, mut total_strict) =
        (0u64, 0u64, 0u64, 0u64);
    let mut non_quiescent = 0usize;

    // A corpus entry declares no buffers or uniforms — the format refuses to
    // write one down — so one empty environment serves every term.
    let env = Environment::new();
    for (i, (name, expr)) in entries.iter().enumerate() {
        let term = Term::new(expr.entry(), &env);
        let mut egraph = EGraph::with_rules(all_rules());
        let root_class = pixelflow_search::egraph::insert_term(
            term,
            &mut egraph,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let saturate_started = std::time::Instant::now();
        let sat_stats =
            egraph.saturate_with_limits(SATURATE_MAX_ITERS, SATURATE_MAX_CLASSES, SATURATE_TIMEOUT);
        let saturate_elapsed = saturate_started.elapsed();
        assert!(
            saturate_elapsed < SATURATE_TIMEOUT,
            "tightened_labeler_rank: expression '{name}' hit the {SATURATE_TIMEOUT:?} safety \
             ceiling — expected to never bind at this corpus's scale; fail loud rather than \
             silently report a truncated sample as converged"
        );
        // A run that stopped at the class cap or the iteration ceiling was
        // cut wherever the safety limit happened to land: its available
        // applications, its extracted path and all three label ratios then
        // describe that truncation, not the rule library. Excluded and
        // counted, never pooled in silently.
        if sat_stats.stop != SaturationStop::Quiesced {
            non_quiescent += 1;
            eprintln!(
                "tightened_labeler_rank: excluding '{name}' — saturation stopped with \
                 {:?}, not Quiesced",
                sat_stats.stop
            );
            continue;
        }

        let extraction = extract_dag(&egraph, root_class, &costs);
        let loose = EpisodeLabels::compute(&egraph, extraction.root, &extraction.choices);
        let tight = EpisodeLabels::compute_tight(&egraph, extraction.root, &extraction.choices);
        let strict = EpisodeLabels::compute_strict(&egraph, extraction.root, &extraction.choices);

        assert!(
            tight.load_bearing.is_subset(&loose.load_bearing),
            "tightened_labeler_rank: tight bound ({} applications) is not a subset of the loose \
             bound ({} applications) for expression '{name}' — derivation_ancestors_tight has \
             drifted from its documented safety property, fail loud",
            tight.load_bearing.len(),
            loose.load_bearing.len()
        );
        assert!(
            strict.load_bearing.is_subset(&tight.load_bearing),
            "tightened_labeler_rank: strict bound ({} applications) is not a subset of the tight \
             bound ({} applications) for expression '{name}' — the tight walk's node-credit \
             portion should always dominate the strict walk, fail loud",
            strict.load_bearing.len(),
            tight.load_bearing.len()
        );

        let total_applications = egraph.provenance().recorded_count() as u64;
        for (app_id, record) in egraph.provenance().applications() {
            let is_loose = loose.load_bearing.contains(&app_id);
            let is_tight = tight.load_bearing.contains(&app_id);
            let is_strict = strict.load_bearing.contains(&app_id);

            let rule = record.rule.unwrap_or_else(|| {
                panic!(
                    "tightened_labeler_rank: application {app_id:?} carries no RuleId — the \
                     graph was built without rule ids, and every table here is keyed by \
                     identity"
                )
            });
            let agg = rule_agg.entry(rule).or_default();
            agg.fired += 1;
            agg.loose_lb += u64::from(is_loose);
            agg.tight_lb += u64::from(is_tight);
            agg.strict_lb += u64::from(is_strict);

            loose_vs_strict.observe(is_loose, is_strict);
            tight_vs_strict.observe(is_tight, is_strict);
            loose_vs_tight.observe(is_loose, is_tight);
        }

        total_apps += total_applications;
        total_loose += loose.load_bearing.len() as u64;
        total_tight += tight.load_bearing.len() as u64;
        total_strict += strict.load_bearing.len() as u64;

        rows.push(ExprRow {
            name: name.clone(),
            node_count: expr.len(),
            total_applications,
            loose_lb: loose.load_bearing.len() as u64,
            tight_lb: tight.load_bearing.len() as u64,
            strict_lb: strict.load_bearing.len() as u64,
        });

        if (i + 1) % 50 == 0 || i + 1 == n {
            eprintln!("tightened_labeler_rank: {}/{n} expressions measured", i + 1);
        }
    }

    assert_eq!(
        loose_vs_strict.total(),
        total_apps,
        "tightened_labeler_rank: pooled contingency total must equal total applications"
    );

    // --- Per-expression ratio quartiles, all three bounds ---
    let loose_ratios: Vec<f64> = rows
        .iter()
        .map(|r| ratio(r.loose_lb, r.total_applications))
        .collect();
    let tight_ratios: Vec<f64> = rows
        .iter()
        .map(|r| ratio(r.tight_lb, r.total_applications))
        .collect();
    let strict_ratios: Vec<f64> = rows
        .iter()
        .map(|r| ratio(r.strict_lb, r.total_applications))
        .collect();
    let (l_q1, l_med, l_q3) = quantiles(loose_ratios.clone());
    let (t_q1, t_med, t_q3) = quantiles(tight_ratios.clone());
    let (s_q1, s_med, s_q3) = quantiles(strict_ratios.clone());

    let pooled_loose = ratio(total_loose, total_apps);
    let pooled_tight = ratio(total_tight, total_apps);
    let pooled_strict = ratio(total_strict, total_apps);

    println!("=== Tightened-labeler re-measurement: {n} expressions ===");
    println!("total rule applications: {total_apps}");
    println!();
    println!(
        "{:<10} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "bound", "pooled LB", "pooled %", "median %", "Q1 %", "Q3 %"
    );
    let print_bound = |label: &str, pooled_lb: u64, pooled: f64, q1: f64, med: f64, q3: f64| {
        println!(
            "{:<10} {:>12} {:>9.2}% {:>9.2}% {:>9.2}% {:>9.2}%",
            label,
            pooled_lb,
            pooled * 100.0,
            med * 100.0,
            q1 * 100.0,
            q3 * 100.0
        );
    };
    print_bound("loose", total_loose, pooled_loose, l_q1, l_med, l_q3);
    print_bound("tight", total_tight, pooled_tight, t_q1, t_med, t_q3);
    print_bound("strict", total_strict, pooled_strict, s_q1, s_med, s_q3);

    // --- Per-rule pooled ratios + Spearman between bounds ---
    let mut rule_rows: Vec<(RuleId, RuleAgg)> = rule_agg
        .iter()
        .filter(|(_, agg)| agg.fired > 0)
        .map(|(&k, &v)| (k, v))
        .collect();
    rule_rows.sort_by(|a, b| {
        ratio(b.1.loose_lb, b.1.fired)
            .partial_cmp(&ratio(a.1.loose_lb, a.1.fired))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.get().cmp(&b.0.get()))
    });

    // The rank vectors are pooled by rule NAME, not by rule index. The rule
    // library registers several indexed variants under one semantic name
    // (`commutative`, `identity`, ... one per operator); keying the ranks by
    // index would treat those variants as separate observations, overweight
    // whichever families happen to have more registered operators, and make
    // this correlation incomparable with `guide_headroom`'s pooled-by-name
    // number that it exists to remeasure.
    let mut name_agg: BTreeMap<String, RuleAgg> = BTreeMap::new();
    for (&id, a) in &rule_agg {
        let e = name_agg
            .entry(family_of(&label_of(&rules, id)).to_string())
            .or_default();
        e.fired += a.fired;
        e.loose_lb += a.loose_lb;
        e.tight_lb += a.tight_lb;
        e.strict_lb += a.strict_lb;
    }
    let named_rows: Vec<(&String, &RuleAgg)> =
        name_agg.iter().filter(|(_, a)| a.fired > 0).collect();
    let loose_rule_ratios: Vec<f64> = named_rows
        .iter()
        .map(|(_, a)| ratio(a.loose_lb, a.fired))
        .collect();
    let tight_rule_ratios: Vec<f64> = named_rows
        .iter()
        .map(|(_, a)| ratio(a.tight_lb, a.fired))
        .collect();
    let strict_rule_ratios: Vec<f64> = named_rows
        .iter()
        .map(|(_, a)| ratio(a.strict_lb, a.fired))
        .collect();

    let spearman_loose_strict = spearman(&loose_rule_ratios, &strict_rule_ratios);
    let spearman_tight_strict = spearman(&tight_rule_ratios, &strict_rule_ratios);
    let spearman_loose_tight = spearman(&loose_rule_ratios, &tight_rule_ratios);

    println!();
    println!(
        "--- Per-rule Spearman rank correlation (n={} distinct rule NAMES that fired, \
         pooled across their indexed operator variants) ---",
        named_rows.len()
    );
    println!(
        "  loose vs strict: {}",
        spearman_loose_strict
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined (constant input)".to_string())
    );
    println!(
        "  tight vs strict: {}",
        spearman_tight_strict
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined (constant input)".to_string())
    );
    println!(
        "  loose vs tight:  {}",
        spearman_loose_tight
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined (constant input)".to_string())
    );

    // --- Per-application Spearman (== phi coefficient for binary data) ---
    println!();
    println!("--- Per-application Spearman rank correlation (n={total_apps} applications) ---");
    let print_phi = |label: &str, c: &Contingency| {
        println!(
            "  {label}: {}",
            c.phi()
                .map(|r| format!("{r:.4}"))
                .unwrap_or_else(|| "undefined (constant input)".to_string())
        );
    };
    print_phi("loose vs strict", &loose_vs_strict);
    print_phi("tight vs strict", &tight_vs_strict);
    print_phi("loose vs tight ", &loose_vs_tight);

    // --- Structural/congruence vs numeric rule-class breakdown ---
    let name_of = |id: RuleId| -> String { label_of(&rules, id) };
    let class_pooled = |names: &[&str]| -> RuleAgg {
        let mut agg = RuleAgg::default();
        for (id, a) in &rule_agg {
            if names.contains(&family_of(&name_of(*id))) {
                agg.fired += a.fired;
                agg.loose_lb += a.loose_lb;
                agg.tight_lb += a.tight_lb;
                agg.strict_lb += a.strict_lb;
            }
        }
        agg
    };
    let structural = class_pooled(STRUCTURAL_RULE_NAMES);
    let numeric = class_pooled(NUMERIC_RULE_NAMES);

    println!();
    println!("--- Rule-class breakdown (pooled ratios; §2.1's central open question) ---");
    println!(
        "{:<14} {:>12} {:>10} {:>10} {:>10}",
        "class", "fired", "loose %", "tight %", "strict %"
    );
    let print_class = |label: &str, agg: RuleAgg| {
        println!(
            "{:<14} {:>12} {:>9.2}% {:>9.2}% {:>9.2}%",
            label,
            agg.fired,
            ratio(agg.loose_lb, agg.fired) * 100.0,
            ratio(agg.tight_lb, agg.fired) * 100.0,
            ratio(agg.strict_lb, agg.fired) * 100.0,
        );
    };
    print_class("structural", structural);
    print_class("numeric", numeric);

    // --- Full per-rule table ---
    println!();
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9}",
        "rule", "fired", "loose %", "tight %", "strict %"
    );
    for (id, agg) in &rule_rows {
        println!(
            "{:<28} {:>9} {:>8.1}% {:>8.1}% {:>8.1}%",
            name_of(*id),
            agg.fired,
            ratio(agg.loose_lb, agg.fired) * 100.0,
            ratio(agg.tight_lb, agg.fired) * 100.0,
            ratio(agg.strict_lb, agg.fired) * 100.0,
        );
    }

    if let Some(out_path) = &args.out {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"num_expressions\": {},\n", rows.len()));
        json.push_str(&format!("  \"corpus_entries\": {n},\n"));
        json.push_str(&format!("  \"non_quiescent_excluded\": {non_quiescent},\n"));
        json.push_str(&format!("  \"total_applications\": {total_apps},\n"));
        json.push_str(&format!("  \"pooled_loose_ratio\": {pooled_loose:.6},\n"));
        json.push_str(&format!("  \"pooled_tight_ratio\": {pooled_tight:.6},\n"));
        json.push_str(&format!("  \"pooled_strict_ratio\": {pooled_strict:.6},\n"));
        json.push_str(&format!(
            "  \"loose_ratio_quartiles\": [{l_q1:.6}, {l_med:.6}, {l_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"tight_ratio_quartiles\": [{t_q1:.6}, {t_med:.6}, {t_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"strict_ratio_quartiles\": [{s_q1:.6}, {s_med:.6}, {s_q3:.6}],\n"
        ));
        json.push_str("  \"per_rule_spearman\": {\n");
        json.push_str(&format!("    \"n_rule_names\": {},\n", named_rows.len()));
        json.push_str(&format!(
            "    \"loose_vs_strict\": {},\n",
            spearman_loose_strict
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string())
        ));
        json.push_str(&format!(
            "    \"tight_vs_strict\": {},\n",
            spearman_tight_strict
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string())
        ));
        json.push_str(&format!(
            "    \"loose_vs_tight\": {}\n",
            spearman_loose_tight
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string())
        ));
        json.push_str("  },\n");
        json.push_str("  \"per_application_spearman\": {\n");
        json.push_str(&format!(
            "    \"loose_vs_strict\": {},\n",
            loose_vs_strict
                .phi()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string())
        ));
        json.push_str(&format!(
            "    \"tight_vs_strict\": {},\n",
            tight_vs_strict
                .phi()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string())
        ));
        json.push_str(&format!(
            "    \"loose_vs_tight\": {}\n",
            loose_vs_tight
                .phi()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string())
        ));
        json.push_str("  },\n");
        json.push_str("  \"rule_class_breakdown\": {\n");
        let class_json = |agg: &RuleAgg| -> String {
            format!(
                "{{\"fired\": {}, \"loose_ratio\": {:.6}, \"tight_ratio\": {:.6}, \"strict_ratio\": {:.6}}}",
                agg.fired,
                ratio(agg.loose_lb, agg.fired),
                ratio(agg.tight_lb, agg.fired),
                ratio(agg.strict_lb, agg.fired),
            )
        };
        json.push_str(&format!(
            "    \"structural\": {},\n",
            class_json(&structural)
        ));
        json.push_str(&format!("    \"numeric\": {}\n", class_json(&numeric)));
        json.push_str("  },\n");
        json.push_str("  \"per_rule\": [\n");
        for (i, (id, agg)) in rule_rows.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"rule\": \"{}\", \"rule_id\": {}, \"fired\": {}, \"loose_load_bearing\": {}, \"tight_load_bearing\": {}, \"strict_load_bearing\": {}}}{}\n",
                name_of(*id),
                id.get(),
                agg.fired,
                agg.loose_lb,
                agg.tight_lb,
                agg.strict_lb,
                if i + 1 < rule_rows.len() { "," } else { "" }
            ));
        }
        json.push_str("  ],\n");
        json.push_str("  \"per_expression\": [\n");
        for (i, r) in rows.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"name\": {:?}, \"node_count\": {}, \"total_applications\": {}, \"loose_load_bearing\": {}, \"tight_load_bearing\": {}, \"strict_load_bearing\": {}}}{}\n",
                r.name,
                r.node_count,
                r.total_applications,
                r.loose_lb,
                r.tight_lb,
                r.strict_lb,
                if i + 1 < rows.len() { "," } else { "" }
            ));
        }
        json.push_str("  ]\n");
        json.push_str("}\n");

        let mut f = std::fs::File::create(out_path)
            .unwrap_or_else(|e| panic!("tightened_labeler_rank: cannot create {out_path}: {e}"));
        f.write_all(json.as_bytes())
            .unwrap_or_else(|e| panic!("tightened_labeler_rank: cannot write {out_path}: {e}"));
        eprintln!("tightened_labeler_rank: wrote {out_path}");
    }
}
