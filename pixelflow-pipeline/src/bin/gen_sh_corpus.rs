//! Generate the `sh` round-1b out-of-distribution corpus.
//!
//! Builds real spherical-harmonics expressions
//! ([`pixelflow_pipeline::training::sh_family`]) and writes them to a
//! **separate** DEV-side corpus file, `corpus_dev_ood.bin`, per
//! `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` §3a
//! / §6: never merged into `corpus_dev.bin` — Round 1's file and its MD5
//! (`3026133ebba066eeca10f658da554400`) stay untouched.
//!
//! Every candidate passes three gates, in order:
//!
//! 1. **Size band**: node count in `[MIN_NODES, MAX_NODES]` (the registered
//!    classical band, `> 50`, targeting 51-400). Outside it, the candidate
//!    is discarded and counted — not an error, since the generator
//!    necessarily overshoots and undershoots by design.
//! 2. **Within-family structural dedup**: the same [`FenceKey`]
//!    (docs `training::structural`) admitted twice is dropped.
//! 3. **Train-fence collision**: every admitted candidate's `FenceKey` is
//!    probed against `corpus_train.bin`. A collision is a hard, named
//!    error — per the registration, "a structural duplicate of a TRAIN
//!    expression is a leak, not a hygiene event" for a model already
//!    trained on that file.
//! 4. **Numeric quarantine**: the same JIT-vs-scalar-oracle cross-check
//!    every other corpus tier uses
//!    ([`pixelflow_pipeline::training::quarantine::Quarantine`]), on a
//!    seeded 64-point magnitude sweep bounded well inside
//!    `pixelflow_ir::passes::TRIG_DOMAIN`.
//!
//! Fewer than [`MIN_SURVIVORS`] admitted expressions is a failed generation
//! (registration §3), not a small corpus — the run fails loudly rather than
//! publishing an underpowered family silently.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use clap::Parser;
use pixelflow_ir::{ExprData, Node, OpKind, Term};
use pixelflow_pipeline::training::corpus::{Entry, read_corpus, write_corpus};
use pixelflow_pipeline::training::quarantine::Quarantine;
use pixelflow_pipeline::training::sh_family::{self, Rng};
use pixelflow_pipeline::training::split::SplitManifest;
use pixelflow_pipeline::training::structural::FenceKey;

/// The registered family name — must appear in the manifest's
/// `[dev] families` before this binary will run.
const FAMILY_NAME: &str = "sh";

/// Lower node-count bound: the registered classical band is `> 50`.
const MIN_NODES: usize = 51;
/// Upper node-count bound: the registered target ceiling.
const MAX_NODES: usize = 400;

/// Registered floor: fewer distinct admitted expressions than this is a
/// failed generation.
const MIN_SURVIVORS: usize = 30;
/// Registered target band for the published corpus.
const TARGET_MIN: usize = 60;
const TARGET_MAX: usize = 100;

/// `TRIG_OPS` mirrors the registration's §2 stratification table —
/// duplicated here (rather than imported) because the harness's own copy
/// (`phase3_at_budget_eval.rs`) is a separate, independently evolving
/// binary; both must agree with the registered op list, not with each
/// other's internals.
const TRIG_OPS: &[OpKind] = &[
    OpKind::Sin,
    OpKind::Cos,
    OpKind::Tan,
    OpKind::Asin,
    OpKind::Acos,
    OpKind::Atan,
    OpKind::Atan2,
];

#[derive(Parser)]
#[command(name = "gen_sh_corpus")]
#[command(about = "Generate the round-1b `sh` out-of-distribution DEV corpus")]
struct Args {
    /// Directory holding `corpus_train.bin` (fence source, read-only) and
    /// `corpus_dev_ood.bin` (write target — merged, not overwritten: any
    /// non-`dev_sh_*` entries already there, e.g. the `bezier` family, are
    /// preserved verbatim).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    output: String,

    /// Split manifest path — `sh` must appear in `[dev] families`.
    #[arg(long, default_value = "pixelflow-pipeline/corpus_split.toml")]
    manifest: String,

    /// Random seed.
    #[arg(long, default_value_t = 0x5117_0000)]
    seed: u64,

    /// Stop once this many admitted expressions have been collected (may
    /// stop before `--max-attempts` is reached).
    #[arg(long, default_value_t = TARGET_MAX)]
    target: usize,

    /// Give up after this many draws even if `--target` was not reached.
    #[arg(long, default_value_t = 20_000)]
    max_attempts: usize,

    /// Quarantine exclusion sidecar (JSONL). Defaults to
    /// `<output>/corpus_sh_quarantine.jsonl`.
    #[arg(long)]
    quarantine_log: Option<String>,
}

/// `(non_leaf_op_count, trig_op_count)` reachable from `root` — a plain DAG
/// walk (visited-set, not `node_count_subtree`'s raw count) so a shared
/// subexpression is counted once, matching how the corpus's own dedup key
/// ([`FenceKey`]) sees the expression.
fn op_counts(root: Node<'_, ExprData>) -> (usize, usize) {
    let mut non_leaf = 0usize;
    let mut trig = 0usize;
    for node in root.descendants() {
        let ExprData::Op(kind) = *node else { continue };
        non_leaf += 1;
        if TRIG_OPS.contains(&kind) {
            trig += 1;
        }
    }
    (non_leaf, trig)
}

fn percentile(sorted: &[usize], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

fn main() {
    let args = Args::parse();
    assert!(
        args.target <= TARGET_MAX,
        "--target {} exceeds the registered ceiling of {TARGET_MAX} distinct expressions",
        args.target
    );

    let manifest = SplitManifest::load(Path::new(&args.manifest))
        .unwrap_or_else(|e| panic!("failed to load split manifest {}: {e}", args.manifest));
    assert!(
        manifest.dev_families.iter().any(|f| f == FAMILY_NAME),
        "\"{FAMILY_NAME}\" is not registered under [dev] families in {} — add it before \
         generating its corpus (docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md §6)",
        args.manifest
    );

    std::fs::create_dir_all(&args.output)
        .unwrap_or_else(|e| panic!("failed to create output directory {}: {e}", args.output));

    // ── Train fence: every admitted candidate is probed against it, and a
    // collision is a hard error (registration §3: "any collision is a hard,
    // named error"). Built directly from corpus_train.bin rather than the
    // sealed `Fence<T>` type (`training::split`), which is deliberately
    // unable to name `Tier::Train` — that seal protects a *different*
    // direction of leak (TRAIN duplicating a held-out DEV/FINAL structure).
    // Here the direction is reversed: DEV duplicating a structure the model
    // already trained on. Reading corpus_train.bin and hashing its
    // FenceKeys directly is a distinct, legitimate use, not a workaround of
    // the seal.
    let train_path = PathBuf::from(&args.output).join("corpus_train.bin");
    let train_entries = read_corpus(&train_path).unwrap_or_else(|e| {
        panic!(
            "failed to read TRAIN corpus {} (fence source): {e} — the OOD fence check cannot \
             run without it",
            train_path.display()
        )
    });
    assert!(
        !train_entries.is_empty(),
        "TRAIN corpus {} is empty (zero entries) — a fence built from it would block nothing",
        train_path.display()
    );
    let train_fence: HashSet<FenceKey> = train_entries
        .iter()
        .map(|(_name, expr)| FenceKey::of(expr.entry()))
        .collect();
    println!(
        "Train fence: {} entries, {} distinct structural keys",
        train_entries.len(),
        train_fence.len()
    );

    let quarantine_log_path = args
        .quarantine_log
        .clone()
        .unwrap_or_else(|| format!("{}/corpus_sh_quarantine.jsonl", args.output));
    let mut quarantine = Quarantine::new(&quarantine_log_path);

    let mut rng = Rng::new(args.seed);
    let mut seen_within: HashSet<FenceKey> = HashSet::new();
    let mut admitted: Vec<Entry> = Vec::new();
    let mut node_counts: Vec<usize> = Vec::new();
    let mut trig_fractions: Vec<f64> = Vec::new();
    let mut trig_heavy_count = 0usize;

    let mut attempts = 0usize;
    let mut out_of_band = 0usize;
    let mut duplicate_within = 0usize;
    let mut quarantined = 0usize;

    while admitted.len() < args.target && attempts < args.max_attempts {
        attempts += 1;
        let (expr, env) = sh_family::draw(&mut rng);
        let term = Term::new(expr.entry(), &env);
        // The reachable, unique-node count — what `write_corpus` stores (the
        // encoder is reachable-only) and what `phase3_at_budget_eval` reads
        // back as `len()`. `node_count_subtree` counts per *reference*
        // instead, so a shared node is counted once per parent; SH
        // expressions deliberately share their trigonometric basis nodes,
        // which made that number strictly larger than the one the registered
        // band is measured against and admitted candidates the evaluator then
        // classified below `classical`.
        let n = term.root().node_count();
        if !(MIN_NODES..=MAX_NODES).contains(&n) {
            out_of_band += 1;
            continue;
        }

        let key = FenceKey::of(term.root());
        assert!(
            !train_fence.contains(&key),
            "sh candidate #{attempts} (node_count={n}) structurally duplicates a TRAIN \
             expression — this is a leak into a model already trained on {}, not a hygiene \
             event; see docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md §3",
            train_path.display()
        );
        if !seen_within.insert(key) {
            duplicate_within += 1;
            continue;
        }

        let name = format!("dev_sh_{:05}", admitted.len());
        if !quarantine.check(&name, term) {
            quarantined += 1;
            continue;
        }

        let (non_leaf, trig) = op_counts(term.root());
        trig_fractions.push(if non_leaf == 0 {
            0.0
        } else {
            trig as f64 / non_leaf as f64
        });
        if trig >= 3 {
            trig_heavy_count += 1;
        }
        node_counts.push(n);

        admitted.push((name, expr));
    }

    let (checked, excluded, mismatched) = quarantine.tallies();
    quarantine.finish();

    println!("\n=== sh generation ===");
    println!("Attempts:                {attempts}");
    println!("Admitted:                {}", admitted.len());
    println!("Out of size band [{MIN_NODES}, {MAX_NODES}]: {out_of_band}");
    println!("Duplicate within family: {duplicate_within}");
    println!("Quarantine checked/excluded/mismatched: {checked}/{excluded}/{mismatched}");

    assert!(
        admitted.len() >= MIN_SURVIVORS,
        "sh generation FAILED: only {} survivors (< registered floor {MIN_SURVIVORS}) after \
         {attempts} attempts — attrition: {out_of_band} out of band, {duplicate_within} \
         duplicate, {quarantined} quarantined",
        admitted.len()
    );
    if admitted.len() < TARGET_MIN {
        println!(
            "WARNING: {} admitted is below the registered target band [{TARGET_MIN}, {TARGET_MAX}] \
             (still >= the {MIN_SURVIVORS}-survivor floor, so this is a published but thin corpus)",
            admitted.len()
        );
    }

    // Merge into corpus_dev_ood.bin: preserve any entry that is not a
    // `dev_sh_*` name (e.g. the `bezier` family) untouched, replace the sh
    // entries wholesale with this run's admissions. `corpus_dev_ood.bin` is
    // shared by every registered OOD family, so overwriting it would delete
    // whichever family happened to be generated first — the same merge
    // `gen_bezier_corpus` already performs from its side.
    let out_path = PathBuf::from(&args.output).join("corpus_dev_ood.bin");
    let existing = if out_path.exists() {
        read_corpus(&out_path)
            .unwrap_or_else(|e| panic!("failed to read existing {}: {e}", out_path.display()))
    } else {
        Vec::new()
    };
    let preserved: Vec<Entry> = existing
        .into_iter()
        .filter(|(name, _)| !name.starts_with("dev_sh_"))
        .collect();
    println!(
        "Preserved {} non-`dev_sh_*` entries already in {}",
        preserved.len(),
        out_path.display()
    );
    let mut out_entries = preserved;
    out_entries.extend(admitted.iter().map(|(n, e)| (n.clone(), e.clone())));
    write_corpus(&out_path, &out_entries)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));

    let mut sorted_nodes = node_counts.clone();
    sorted_nodes.sort_unstable();
    let min_n = *sorted_nodes.first().unwrap_or(&0);
    let max_n = *sorted_nodes.last().unwrap_or(&0);
    let mean_n = if sorted_nodes.is_empty() {
        0.0
    } else {
        sorted_nodes.iter().sum::<usize>() as f64 / sorted_nodes.len() as f64
    };
    let median_n = percentile(&sorted_nodes, 0.5);
    let mean_trig_frac = if trig_fractions.is_empty() {
        0.0
    } else {
        trig_fractions.iter().sum::<f64>() / trig_fractions.len() as f64
    };

    println!("\n=== Report ===");
    println!(
        "Wrote {} expressions to {}",
        admitted.len(),
        out_path.display()
    );
    println!(
        "Node count: min={min_n} p50={median_n:.0} mean={mean_n:.1} max={max_n} (target band \
         [{MIN_NODES}, {MAX_NODES}])"
    );
    println!(
        "Trig-op share: mean {:.1}% of non-leaf nodes per expression; {}/{} expressions are \
         trig-heavy (>= 3 trig ops, the registration's §2 rule)",
        mean_trig_frac * 100.0,
        trig_heavy_count,
        admitted.len()
    );
    println!(
        "TRAIN overlap: 0 collisions across {attempts} candidates probed against {} TRAIN keys \
         (a nonzero count would have panicked above, not been reported here)",
        train_fence.len()
    );
}
