//! Generate the tiered expression corpus for JIT benchmarking.
//!
//! Generates unique expressions across varied depth/complexity bands and
//! writes the three-tier corpus (plan 0.2) as binary PXCR files:
//! `corpus_train.bin`, `corpus_dev.bin`, `corpus_final.bin`. Tier membership
//! is decided by the checked-in split manifest
//! (`pixelflow-pipeline/corpus_split.toml`) — families are held out whole,
//! never by random split. Structural dedup runs ACROSS tiers: a duplicate
//! spanning tiers is dropped from the less sacred tier (TRAIN before DEV
//! before FINAL) and counted.
//!
//! Every candidate passes the numeric quarantine
//! ([`pixelflow_pipeline::training::quarantine`], plan 0.4, audit H1/H2)
//! before entering ANY tier: the JIT-compiled kernel is cross-checked against
//! the independent scalar oracle on a seeded 64-point magnitude-sweep grid,
//! each point judged against the error bound THAT POINT composed. This is a
//! SAME-ARENA check — implementation correctness only, never algebraic-rewrite
//! judgment — so any disagreement is a JIT/lowering bug: the expression is
//! excluded from every tier, logged to a JSONL sidecar, and a mismatch rate
//! above 1% fails the run (compiler bugs, not corpus noise). A total exclusion
//! rate above 5% also fails — that is a censoring alarm (audit M2).
//!
//! The quarantine predicate lives in the library, not here, so the trainer's
//! live synthetic stream applies the identical check (smoke finding B8).
//!
//! Admitted expressions whose grid carries divergent, unbounded, or
//! ill-conditioned points get an informational `conditioning_metadata` sidecar
//! line naming those grid indices, so downstream evals can restrict CROSS-form
//! comparisons to well-conditioned points. Absence of a line means the whole
//! grid is clean.
//!
//! # Per-family integrity
//!
//! A tier is not a bag of expressions, it is a set of generator FAMILIES held
//! out whole. A family that admits nothing is a hole in the holdout, and the
//! five named production kernels are enough to keep `corpus_final.bin`
//! nonempty while every synthetic FINAL family has silently vanished. So each
//! family must admit at least [`MIN_FAMILY_ADMISSIONS`] expressions (or its
//! whole quota, when the quota is smaller) and must not lose more than
//! [`MAX_FAMILY_QUARANTINE_RATE`] of the candidates that reached the numeric
//! gate; short or over-censored families are a hard, named failure listing
//! their attrition, and every family's admitted/attempted counts are reported.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus -- \
//!     --target 20000 --output pixelflow-pipeline/data \
//!     --manifest pixelflow-pipeline/corpus_split.toml
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::hash::Hash;
use std::path::{Path, PathBuf};

use clap::Parser;
use pixelflow_ir::{ExprBuilder, Term};
use pixelflow_pipeline::shader_bench::named_kernel;
use pixelflow_pipeline::training::corpus::{Entry, write_corpus};
use pixelflow_pipeline::training::quarantine::Quarantine;
use pixelflow_pipeline::training::split::{Family, SplitManifest, Tier};
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::collect_rule_templates;
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

#[derive(Parser)]
#[command(name = "gen_bench_corpus")]
#[command(about = "Generate the tiered (train/dev/final) expression corpus for JIT benchmarking")]
struct Args {
    /// Output DIRECTORY for the tiered corpus files
    /// (`corpus_train.bin` / `corpus_dev.bin` / `corpus_final.bin`).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    output: String,

    /// Target number of unique expressions across all tiers.
    #[arg(long, default_value_t = 360000)]
    target: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum node count (skip overly huge expressions).
    #[arg(long, default_value_t = 1000)]
    max_nodes: usize,

    /// Split manifest path (plan 0.2).
    #[arg(long, default_value = "pixelflow-pipeline/corpus_split.toml")]
    manifest: String,

    /// Quarantine exclusion sidecar (JSONL). Defaults to
    /// `<output>/corpus_quarantine.jsonl`.
    #[arg(long)]
    quarantine_log: Option<String>,
}

struct Band {
    max_depth: usize,
    leaf_prob: f32,
    num_vars: usize,
}

const BANDS: &[Band] = &[
    // Tiny expressions — 1-2 vars
    Band {
        max_depth: 2,
        leaf_prob: 0.6,
        num_vars: 1,
    },
    Band {
        max_depth: 2,
        leaf_prob: 0.5,
        num_vars: 2,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.6,
        num_vars: 1,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.5,
        num_vars: 2,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.4,
        num_vars: 4,
    },
    // Small
    Band {
        max_depth: 4,
        leaf_prob: 0.5,
        num_vars: 2,
    },
    Band {
        max_depth: 4,
        leaf_prob: 0.4,
        num_vars: 4,
    },
    Band {
        max_depth: 4,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    // Medium
    Band {
        max_depth: 5,
        leaf_prob: 0.4,
        num_vars: 2,
    },
    Band {
        max_depth: 5,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 5,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.35,
        num_vars: 3,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    // Large
    Band {
        max_depth: 7,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 7,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 7,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    // Deep
    Band {
        max_depth: 9,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 9,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 10,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 10,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    // Very deep — complex kernels
    Band {
        max_depth: 11,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 11,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 12,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 12,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 13,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 14,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    // Huge — saturation-scale expressions (100-500+ nodes)
    Band {
        max_depth: 15,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 16,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 16,
        leaf_prob: 0.12,
        num_vars: 4,
    },
    Band {
        max_depth: 18,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 18,
        leaf_prob: 0.12,
        num_vars: 4,
    },
    Band {
        max_depth: 20,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 20,
        leaf_prob: 0.12,
        num_vars: 4,
    },
    Band {
        max_depth: 20,
        leaf_prob: 0.10,
        num_vars: 4,
    },
];

// The feature-quotient key lives in `pixelflow_pipeline::training::structural`
// so every stage that fences by structure agrees on what "the model has
// already seen this" means (P1(d)).

// ============================================================================
// Tier-aware structural dedup (plan 0.2)
// ============================================================================

/// Outcome of probing a candidate's structural key against the ledger.
#[derive(Debug, PartialEq, Eq)]
enum Admission {
    /// Unseen structure; the caller may quarantine-check it and, on pass,
    /// [`DedupLedger::register`] it.
    New,
    /// Same structure already admitted to the same tier.
    DuplicateWithin,
    /// Same structure already owned by a MORE sacred tier; the candidate is
    /// dropped from its own (less sacred) tier — never the other way around.
    CrossTierDrop,
}

/// Cross-tier structural dedup ledger. The corpus build admits tiers in
/// descending sacredness (FINAL, then DEV, then TRAIN), so an owner is
/// always at least as sacred as any later colliding candidate — which is
/// exactly the plan-0.2 rule "a duplicate spanning tiers is dropped from
/// TRAIN, never from DEV/FINAL". A candidate MORE sacred than an existing
/// owner would need to displace it, so `probe` panics on that ordering
/// violation instead of guessing.
struct DedupLedger<K> {
    owners: HashMap<K, Tier>,
    duplicates_within: usize,
    cross_tier_dropped_from_train: usize,
    cross_tier_dropped_from_dev: usize,
}

impl<K: Eq + Hash> DedupLedger<K> {
    fn new() -> Self {
        Self {
            owners: HashMap::new(),
            duplicates_within: 0,
            cross_tier_dropped_from_train: 0,
            cross_tier_dropped_from_dev: 0,
        }
    }

    /// Probe `key` for `candidate` tier, updating drop counters. Only a
    /// [`Admission::New`] result may later be [`register`](Self::register)ed
    /// (after the candidate passes quarantine).
    fn probe(&mut self, key: &K, candidate: Tier) -> Admission {
        let Some(&owner) = self.owners.get(key) else {
            return Admission::New;
        };
        assert!(
            candidate <= owner,
            "DedupLedger: candidate tier {candidate} collides with less sacred owner {owner} — \
             the corpus build must admit FINAL before DEV before TRAIN"
        );
        if owner == candidate {
            self.duplicates_within += 1;
            Admission::DuplicateWithin
        } else {
            match candidate {
                Tier::Train => self.cross_tier_dropped_from_train += 1,
                Tier::Dev => self.cross_tier_dropped_from_dev += 1,
                Tier::Final => unreachable!("Final is the most sacred tier; assert above"),
            }
            Admission::CrossTierDrop
        }
    }

    /// Record `tier` as the owner of `key`. Must follow a `New` probe.
    fn register(&mut self, key: K, tier: Tier) {
        let previous = self.owners.insert(key, tier);
        assert!(
            previous.is_none(),
            "DedupLedger::register called for an already-owned key (owner {:?}) — \
             probe before registering",
            previous
        );
    }
}

// ============================================================================
// Per-family integrity (Codex P1)
// ============================================================================

/// Absolute floor on what one manifest family must contribute. A family that
/// contributes nothing — or almost nothing — is a hole in the by-family
/// holdout the whole split exists to provide, and the five named production
/// kernels are enough to keep `corpus_final.bin` nonempty while every
/// synthetic FINAL family has quietly vanished. A tier being NONEMPTY is
/// therefore not a check; this is.
///
/// Clamped to the family's own quota, so a run whose per-family share is
/// smaller than the floor is judged against the share.
const MIN_FAMILY_ADMISSIONS: usize = 8;

/// Maximum fraction of a family's *quarantined* candidates — candidates that
/// reached the numeric gate at all, so neither too-large nor duplicates count
/// here. Above this, the family's admissions are selected by whatever the
/// quarantine happens to reject, which is exactly the per-family censoring
/// the global rate alarm cannot see (it averages over ~300 families).
const MAX_FAMILY_QUARANTINE_RATE: f64 = 0.20;

/// Everything one `(tier, family)` did with its attempts. Every candidate is
/// accounted for: `attempts == admitted + too_large + duplicate_within +
/// cross_tier_drop + quarantined`.
struct FamilyOutcome {
    tier: Tier,
    family: Family,
    /// Expressions this family was asked for.
    quota: usize,
    attempts: usize,
    admitted: usize,
    too_large: usize,
    duplicate_within: usize,
    cross_tier_drop: usize,
    /// Candidates the numeric quarantine refused, by reason slug.
    quarantined: BTreeMap<&'static str, usize>,
}

impl FamilyOutcome {
    fn new(tier: Tier, family: Family, quota: usize) -> Self {
        Self {
            tier,
            family,
            quota,
            attempts: 0,
            admitted: 0,
            too_large: 0,
            duplicate_within: 0,
            cross_tier_drop: 0,
            quarantined: BTreeMap::new(),
        }
    }

    fn label(&self) -> String {
        format!(
            "{} band {:>2} family {:>2}",
            self.tier, self.family.band, self.family.seed
        )
    }

    fn quarantined_total(&self) -> usize {
        self.quarantined.values().sum()
    }

    /// Candidates that reached the numeric gate: admitted plus refused.
    fn screened(&self) -> usize {
        self.admitted + self.quarantined_total()
    }

    fn quarantine_rate(&self) -> f64 {
        let screened = self.screened();
        if screened == 0 {
            return 0.0;
        }
        self.quarantined_total() as f64 / screened as f64
    }

    /// The floor this family is judged against: [`MIN_FAMILY_ADMISSIONS`], or
    /// its whole quota when the quota is smaller — but never zero, because a
    /// family that contributes nothing is the hole this check exists to find.
    fn required(&self) -> usize {
        self.quota.clamp(1, MIN_FAMILY_ADMISSIONS)
    }

    /// Why this family fails the integrity check, or `None` if it passes.
    fn failure(&self) -> Option<String> {
        let required = self.required();
        if self.admitted < required {
            return Some(format!(
                "admitted {}/{} required (quota {})",
                self.admitted, required, self.quota
            ));
        }
        if self.quarantine_rate() > MAX_FAMILY_QUARANTINE_RATE {
            return Some(format!(
                "quarantine refused {}/{} of the candidates that reached it ({:.1}% > {:.0}%)",
                self.quarantined_total(),
                self.screened(),
                self.quarantine_rate() * 100.0,
                MAX_FAMILY_QUARANTINE_RATE * 100.0
            ));
        }
        None
    }

    /// The attrition breakdown, for the report and for failure messages.
    fn attrition(&self) -> String {
        let mut s = format!(
            "attempts {}, admitted {}, too_large {}, dup_within {}, cross_tier {}",
            self.attempts,
            self.admitted,
            self.too_large,
            self.duplicate_within,
            self.cross_tier_drop
        );
        for (reason, count) in &self.quarantined {
            let _ = write!(s, ", {reason} {count}");
        }
        s
    }
}

/// Check every family's integrity and fail — once, naming all offenders —
/// before any tier file is written.
///
/// # Panics
///
/// Panics naming each short or over-censored family and its attrition. A note
/// on stdout was the previous behavior and it is what let a whole held-out
/// family disappear silently.
fn assert_family_integrity(families: &[FamilyOutcome]) {
    assert!(
        !families.is_empty(),
        "no manifest families were generated at all"
    );
    let failures: Vec<String> = families
        .iter()
        .filter_map(|f| {
            f.failure()
                .map(|why| format!("  {}: {why}\n      {}", f.label(), f.attrition()))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{}/{} manifest families failed the per-family integrity check — the published corpus \
         would no longer contain the families the split manifest claims to hold out, and a \
         NONEMPTY tier does not detect that (the named production kernels alone keep \
         corpus_final.bin nonempty). Offenders:\n{}",
        failures.len(),
        families.len(),
        failures.join("\n")
    );
}

// ============================================================================
// Generation
// ============================================================================

/// Decorrelated per-family RNG seed (SplitMix64 finalizer over a unique
/// (band, family) encoding), so adjacent family indices do not produce
/// correlated generator streams.
fn family_rng_seed(global_seed: u64, family: Family) -> u64 {
    let mut z = global_seed
        .wrapping_add((family.band as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(family.seed.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn tier_index(tier: Tier) -> usize {
    match tier {
        Tier::Train => 0,
        Tier::Dev => 1,
        Tier::Final => 2,
    }
}

/// Final path of a tier's corpus file.
fn tier_path(output_dir: &str, tier: Tier) -> PathBuf {
    PathBuf::from(format!("{output_dir}/corpus_{}.bin", tier.name()))
}

/// Staging path a tier is written to before promotion.
fn tier_staging_path(output_dir: &str, tier: Tier) -> PathBuf {
    PathBuf::from(format!("{output_dir}/corpus_{}.bin.staging", tier.name()))
}

/// Write all three tiers to staging files and promote them only once every
/// write has succeeded (Codex R4).
///
/// `write_corpus` truncates through `File::create`, so publishing tier by tier
/// left a failure mid-loop with a fresh TRAIN beside a stale DEV/FINAL that
/// were never deduplicated against it — and the next stage only checks that
/// the files exist. Staging plus rename makes the three files one artifact.
///
/// # Panics
///
/// Panics on any write or rename failure, after cleaning up the staging files
/// so a later run cannot mistake them for output.
fn publish_tiers(output_dir: &str, tier_entries: &[Vec<Entry>; 3]) {
    // Cleanup REPORTS rather than panics: a failure to remove a staging file is
    // secondary to the failure that aborted the publish, and panicking here
    // would replace the primary diagnosis with a housekeeping error. Nothing is
    // swallowed — every leftover is named in the panic message below.
    let cleanup = || -> String {
        let mut leftovers: Vec<String> = Vec::new();
        for &tier in &Tier::ALL {
            let staging = tier_staging_path(output_dir, tier);
            if staging.exists()
                && let Err(e) = std::fs::remove_file(&staging)
            {
                leftovers.push(format!("{} ({e})", staging.display()));
            }
        }
        if leftovers.is_empty() {
            String::new()
        } else {
            format!(
                " STAGING FILES LEFT BEHIND, remove them by hand before rerunning: {}",
                leftovers.join(", ")
            )
        }
    };

    for &tier in &Tier::ALL {
        let entries = &tier_entries[tier_index(tier)];
        let staging = tier_staging_path(output_dir, tier);
        if let Err(e) = write_corpus(&staging, entries) {
            let leftovers = cleanup();
            panic!(
                "failed to write staging corpus {}: {e} — no tier file was promoted, so the \
                 previous corpus is intact.{leftovers}",
                staging.display()
            );
        }
    }

    // Every tier is on disk and complete; promotion is the only step left.
    for &tier in &Tier::ALL {
        let staging = tier_staging_path(output_dir, tier);
        let path = tier_path(output_dir, tier);
        if let Err(e) = std::fs::rename(&staging, &path) {
            let leftovers = cleanup();
            panic!(
                "failed to promote {} to {}: {e} — the corpus is now PARTIALLY published and \
                 the tiers must be regenerated.{leftovers}",
                staging.display(),
                path.display()
            );
        }
        println!(
            "Wrote {:>6} expressions to {}",
            tier_entries[tier_index(tier)].len(),
            path.display()
        );
    }
}

fn main() {
    let args = Args::parse();
    assert!(
        !args.output.ends_with(".bin"),
        "--output is now the tier output DIRECTORY (the tiered build writes corpus_train.bin, \
         corpus_dev.bin, and corpus_final.bin into it); got a .bin file path: {}",
        args.output
    );

    let manifest = SplitManifest::load(Path::new(&args.manifest))
        .unwrap_or_else(|e| panic!("failed to load split manifest {}: {e}", args.manifest));
    assert!(
        manifest.max_band_index() < BANDS.len(),
        "split manifest references band {} but the generator defines only {} bands (0..={})",
        manifest.max_band_index(),
        BANDS.len(),
        BANDS.len() - 1
    );

    std::fs::create_dir_all(&args.output)
        .unwrap_or_else(|e| panic!("failed to create output directory {}: {e}", args.output));
    let quarantine_log_path = args
        .quarantine_log
        .clone()
        .unwrap_or_else(|| format!("{}/corpus_quarantine.jsonl", args.output));
    let mut quarantine = Quarantine::new(&quarantine_log_path);

    let mut ledger: DedupLedger<FenceKey> = DedupLedger::new();
    let mut tier_entries: [Vec<Entry>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // The named production kernels enter FINAL first, so their structures are
    // fenced off before any synthetic generation can collide with them.
    for name in &manifest.final_kernels {
        let (expr, env) = named_kernel(name).unwrap_or_else(|| {
            panic!(
                "split manifest FINAL kernel \"{name}\" is not a known production kernel \
                 (known: swirl, circle_sdf, poly, redundant, normalize, plus the withheld \
                 ShaderToy set — see pixelflow_pipeline::shader_bench::SHADERTOY_KERNEL_NAMES)"
            )
        });
        let key = FenceKey::of(expr.entry());
        assert_eq!(
            ledger.probe(&key, Tier::Final),
            Admission::New,
            "named FINAL kernels \"{name}\" duplicates another named kernel structurally"
        );
        // A named FINAL kernel failing the JIT-vs-oracle cross-check is not a
        // corpus-hygiene event — it is an emitter or oracle bug invalidating
        // the whole eval tier. Fail immediately.
        assert!(
            quarantine.check(name, Term::new(expr.entry(), &env)),
            "named FINAL kernel \"{name}\" failed the numeric quarantine — JIT emitter or \
             oracle bug; see {quarantine_log_path}"
        );
        ledger.register(key, Tier::Final);
        tier_entries[tier_index(Tier::Final)].push((name.clone(), expr));
    }

    let total_families: usize = Tier::ALL
        .iter()
        .map(|&t| manifest.spec(t).family_count())
        .sum();
    let per_family = args.target.div_ceil(total_families);
    println!(
        "Generating {} families x {} expressions (target {}) from {}",
        total_families, per_family, args.target, args.manifest
    );

    let mut families: Vec<FamilyOutcome> = Vec::with_capacity(total_families);

    // Descending sacredness, so cross-tier duplicates always resolve in favor
    // of the already-admitted, more sacred owner (see DedupLedger).
    for tier in [Tier::Final, Tier::Dev, Tier::Train] {
        let spec = manifest.spec(tier);
        let mut tier_generated = 0usize;
        for family in spec.families() {
            let band = &BANDS[family.band];
            let config = BwdGenConfig {
                max_depth: band.max_depth,
                leaf_prob: band.leaf_prob,
                num_vars: band.num_vars,
                fused_op_prob: 0.0,
                ..BwdGenConfig::default()
            };
            let mut rng = BwdGenerator::new(
                family_rng_seed(args.seed, family),
                config,
                collect_rule_templates(),
            );

            let mut outcome = FamilyOutcome::new(tier, family, per_family);
            let max_attempts = per_family * 10;

            while outcome.admitted < per_family && outcome.attempts < max_attempts {
                outcome.attempts += 1;
                let pair = rng.generate();
                // The generator hands back one graph holding both forms; this
                // corpus stores the junkified one, in a graph of its own.
                let (expr, env) = {
                    let mut b = ExprBuilder::new();
                    let r = b.splice(pair.unoptimized());
                    b.finish(&[r])
                };
                let term = Term::new(expr.entry(), &env);

                if pixelflow_ir::node_count_subtree(term.root()) > args.max_nodes {
                    outcome.too_large += 1;
                    continue;
                }

                let key = FenceKey::of(term.root());
                match ledger.probe(&key, tier) {
                    Admission::New => {}
                    Admission::DuplicateWithin => {
                        outcome.duplicate_within += 1;
                        continue;
                    }
                    Admission::CrossTierDrop => {
                        outcome.cross_tier_drop += 1;
                        continue;
                    }
                }

                let name = format!(
                    "{}_b{:02}_f{:02}_{:05}",
                    tier.name(),
                    family.band,
                    family.seed,
                    tier_generated + outcome.admitted
                );
                let (exclusion, _conditioning) = quarantine.verdict(&name, term);
                if let Some(exclusion) = exclusion {
                    *outcome.quarantined.entry(exclusion.reason()).or_insert(0) += 1;
                    continue;
                }

                ledger.register(key, tier);
                tier_entries[tier_index(tier)].push((name, expr));
                outcome.admitted += 1;
            }

            tier_generated += outcome.admitted;
            families.push(outcome);
        }
        println!(
            "Tier {:>5}: {:>6} synthetic expressions across {} families",
            tier.to_string(),
            tier_generated,
            spec.family_count()
        );
    }

    // Per-family report BEFORE the alarms, so a failing run still shows which
    // families were thin and why.
    println!("\n=== Per-family admissions ===");
    for f in &families {
        let flag = if f.failure().is_some() {
            " <== FAIL"
        } else {
            ""
        };
        println!("  {}: {}{flag}", f.label(), f.attrition());
    }

    // Alarms BEFORE any tier file is written: a run tripping the censoring,
    // mismatch, or per-family alarm dies loudly HERE, leaving no
    // normal-looking corpus on disk — the next pipeline stage gates only on
    // the files existing, and this process's panic is invisible to that one.
    // A previous good run's tier files survive untouched.
    let (checked, excluded, mismatched) = quarantine.tallies();
    quarantine.finish();
    assert_family_integrity(&families);

    for &tier in &Tier::ALL {
        assert!(
            !tier_entries[tier_index(tier)].is_empty(),
            "tier {tier} produced 0 expressions — target too small for the manifest's family \
             count, or quarantine excluded everything; refusing to write an empty corpus"
        );
    }
    publish_tiers(&args.output, &tier_entries);

    println!("\n=== Summary ===");
    let total: usize = tier_entries.iter().map(Vec::len).sum();
    let too_large: usize = families.iter().map(|f| f.too_large).sum();
    println!("Total unique expressions:      {total}");
    println!("Families generated:            {}", families.len());
    println!(
        "Duplicates within tier:        {}",
        ledger.duplicates_within
    );
    println!(
        "Cross-tier drops (from train): {}",
        ledger.cross_tier_dropped_from_train
    );
    println!(
        "Cross-tier drops (from dev):   {}",
        ledger.cross_tier_dropped_from_dev
    );
    println!("Too large skipped:             {too_large}");
    println!("Quarantine checked/excluded/miscompiles: {checked}/{excluded}/{mismatched}");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_pipeline::shader_bench::NAMED_KERNEL_NAMES;
    use std::collections::HashSet;

    // ── Cross-tier dedup ────────────────────────────────────────────────────

    #[test]
    fn dedup_train_duplicate_of_dev_is_dropped_from_train() {
        let mut ledger: DedupLedger<&str> = DedupLedger::new();
        assert_eq!(ledger.probe(&"k", Tier::Dev), Admission::New);
        ledger.register("k", Tier::Dev);

        assert_eq!(ledger.probe(&"k", Tier::Train), Admission::CrossTierDrop);
        assert_eq!(ledger.cross_tier_dropped_from_train, 1);
        assert_eq!(ledger.cross_tier_dropped_from_dev, 0);
        // The DEV owner is untouched.
        assert_eq!(ledger.owners.get("k"), Some(&Tier::Dev));
    }

    #[test]
    fn dedup_dev_duplicate_of_final_is_dropped_from_dev() {
        let mut ledger: DedupLedger<&str> = DedupLedger::new();
        assert_eq!(ledger.probe(&"k", Tier::Final), Admission::New);
        ledger.register("k", Tier::Final);

        assert_eq!(ledger.probe(&"k", Tier::Dev), Admission::CrossTierDrop);
        assert_eq!(ledger.cross_tier_dropped_from_dev, 1);
        assert_eq!(ledger.owners.get("k"), Some(&Tier::Final));
    }

    #[test]
    fn dedup_within_tier_duplicate_is_counted() {
        let mut ledger: DedupLedger<&str> = DedupLedger::new();
        assert_eq!(ledger.probe(&"k", Tier::Train), Admission::New);
        ledger.register("k", Tier::Train);

        assert_eq!(ledger.probe(&"k", Tier::Train), Admission::DuplicateWithin);
        assert_eq!(ledger.duplicates_within, 1);
        assert_eq!(ledger.cross_tier_dropped_from_train, 0);
    }

    #[test]
    #[should_panic(expected = "FINAL before DEV before TRAIN")]
    fn dedup_generation_order_violation_panics() {
        let mut ledger: DedupLedger<&str> = DedupLedger::new();
        assert_eq!(ledger.probe(&"k", Tier::Train), Admission::New);
        ledger.register("k", Tier::Train);
        // A FINAL candidate colliding with a TRAIN owner would require
        // displacing the owner — the build order makes this unreachable, and
        // reaching it means the order was violated.
        let _ = ledger.probe(&"k", Tier::Final);
    }

    #[test]
    #[should_panic(expected = "probe before registering")]
    fn dedup_double_register_panics() {
        let mut ledger: DedupLedger<&str> = DedupLedger::new();
        ledger.register("k", Tier::Train);
        ledger.register("k", Tier::Train);
    }

    // ── Family seeding ──────────────────────────────────────────────────────

    #[test]
    fn family_rng_seeds_are_distinct() {
        let mut seen = HashSet::new();
        for band in 0..BANDS.len() {
            for fseed in 0..8u64 {
                assert!(
                    seen.insert(family_rng_seed(42, Family::new(band, fseed))),
                    "seed collision at band {band}, family {fseed}"
                );
            }
        }
    }

    // ── Per-family integrity (Codex P1) ─────────────────────────────────────

    /// A family that admitted everything it was asked for.
    fn healthy_family() -> FamilyOutcome {
        let mut f = FamilyOutcome::new(Tier::Dev, Family::new(3, 7), 40);
        f.attempts = 40;
        f.admitted = 40;
        f
    }

    #[test]
    fn healthy_family_passes_and_reports_its_attrition() {
        let f = healthy_family();
        assert!(f.failure().is_none());
        assert!(f.attrition().contains("admitted 40"));
        assert_family_integrity(&[f]);
    }

    #[test]
    #[should_panic(expected = "failed the per-family integrity check")]
    fn family_with_zero_admissions_is_a_hard_failure() {
        // The Codex P1 shape: every candidate was too large or a duplicate, so
        // the family contributes NOTHING to its tier. The old code printed a
        // note and continued, and the five named kernels kept corpus_final.bin
        // nonempty — so the "tier is nonempty" check saw nothing wrong while
        // the manifest's held-out family had vanished.
        let mut f = FamilyOutcome::new(Tier::Final, Family::new(5, 2), 40);
        f.attempts = 400;
        f.too_large = 300;
        f.duplicate_within = 100;
        assert_family_integrity(&[f]);
    }

    #[test]
    fn zero_admission_failure_names_the_family_and_its_attrition() {
        let mut f = FamilyOutcome::new(Tier::Final, Family::new(5, 2), 40);
        f.attempts = 400;
        f.too_large = 300;
        f.duplicate_within = 100;
        let why = f.failure().expect("a family admitting nothing must fail");
        assert!(why.contains("admitted 0/8"), "got: {why}");
        assert!(f.label().contains("band  5"), "got: {}", f.label());
        assert!(f.attrition().contains("too_large 300"));
        assert!(f.attrition().contains("dup_within 100"));
    }

    #[test]
    fn family_short_of_the_admission_floor_fails() {
        let mut f = FamilyOutcome::new(Tier::Train, Family::new(0, 1), 40);
        f.attempts = 400;
        f.admitted = MIN_FAMILY_ADMISSIONS - 1;
        f.too_large = 400 - f.admitted;
        assert!(f.failure().is_some(), "{} admissions is short", f.admitted);
        let mut ok = f;
        ok.admitted = MIN_FAMILY_ADMISSIONS;
        assert!(ok.failure().is_none(), "exactly the floor is enough");
    }

    #[test]
    fn a_family_whose_quota_is_below_the_floor_is_judged_against_its_quota() {
        // A run with more families than target expressions asks each family for
        // less than MIN_FAMILY_ADMISSIONS; judging it against the floor would
        // fail every family on an arithmetic impossibility.
        let mut f = FamilyOutcome::new(Tier::Dev, Family::new(1, 1), 3);
        f.attempts = 3;
        f.admitted = 3;
        assert_eq!(f.required(), 3);
        assert!(f.failure().is_none());
    }

    #[test]
    fn family_censored_by_the_quarantine_fails_even_when_it_meets_the_floor() {
        // 20 admitted clears MIN_FAMILY_ADMISSIONS, but half of everything that
        // reached the numeric gate was refused: the survivors are selected by
        // whatever the quarantine happens to reject. The GLOBAL rate alarm
        // averages over hundreds of families and cannot see this.
        let mut f = FamilyOutcome::new(Tier::Train, Family::new(2, 4), 40);
        f.attempts = 40;
        f.admitted = 20;
        f.quarantined.insert("numeric_mismatch", 20);
        assert_eq!(f.screened(), 40);
        assert!((f.quarantine_rate() - 0.5).abs() < 1e-12);
        let why = f.failure().expect("50% quarantine refusal must fail");
        assert!(why.contains("quarantine refused 20/40"), "got: {why}");
    }

    #[test]
    fn quarantine_rate_ignores_candidates_that_never_reached_the_gate() {
        // Too-large and duplicate candidates are not evidence about numerics.
        let mut f = FamilyOutcome::new(Tier::Train, Family::new(2, 4), 40);
        f.attempts = 400;
        f.admitted = 40;
        f.too_large = 300;
        f.duplicate_within = 59;
        f.quarantined.insert("compile_failed", 1);
        assert_eq!(f.screened(), 41);
        assert!(f.quarantine_rate() < MAX_FAMILY_QUARANTINE_RATE);
        assert!(f.failure().is_none());
    }

    #[test]
    #[should_panic(expected = "no manifest families were generated at all")]
    fn integrity_check_refuses_an_empty_family_set() {
        assert_family_integrity(&[]);
    }

    #[test]
    #[should_panic(expected = "failed the per-family integrity check")]
    fn one_bad_family_fails_a_run_of_healthy_ones() {
        let mut bad = FamilyOutcome::new(Tier::Final, Family::new(9, 9), 40);
        bad.attempts = 400;
        bad.too_large = 400;
        assert_family_integrity(&[healthy_family(), bad, healthy_family()]);
    }

    // ── Atomic tier publish (Codex R4) ──────────────────────────────────────

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pxgen_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn one_entry(name: &str) -> Vec<Entry> {
        let (expr, _env) = named_kernel("poly").expect("poly is a named kernel");
        vec![(name.to_string(), expr)]
    }

    #[test]
    fn publish_promotes_every_tier_and_leaves_no_staging_files() {
        let dir = scratch_dir("publish_ok");
        let out = dir.to_string_lossy().to_string();
        let entries = [
            one_entry("train_0"),
            one_entry("dev_0"),
            one_entry("final_0"),
        ];
        publish_tiers(&out, &entries);
        for &tier in &Tier::ALL {
            assert!(
                tier_path(&out, tier).exists(),
                "tier {tier} was not promoted"
            );
            assert!(
                !tier_staging_path(&out, tier).exists(),
                "tier {tier} left a staging file behind"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "no tier file was promoted")]
    fn a_failed_tier_write_promotes_nothing() {
        // A directory where DEV's staging file must go makes `File::create`
        // fail mid-loop — the exact shape that used to leave a fresh TRAIN
        // beside a stale DEV/FINAL that had never been deduplicated with it.
        let dir = scratch_dir("publish_fail");
        let out = dir.to_string_lossy().to_string();
        std::fs::create_dir_all(tier_staging_path(&out, Tier::Dev)).expect("blocking dir");
        let entries = [
            one_entry("train_0"),
            one_entry("dev_0"),
            one_entry("final_0"),
        ];
        publish_tiers(&out, &entries);
    }

    // ── Quarantine wiring ───────────────────────────────────────────────────
    //
    // The gate ITSELF (compositional bound, divergence, mask proximity, grid
    // determinism) is tested in `pixelflow_pipeline::training::quarantine`,
    // where it now lives so the trainer's live synthetic stream runs the same
    // predicate. What belongs here is that THIS binary's inputs pass it.

    /// The five original production kernels plus the withheld ShaderToy set
    /// — every name `named_kernel` must resolve.
    fn all_named_kernels() -> Vec<&'static str> {
        NAMED_KERNEL_NAMES
            .into_iter()
            .chain(pixelflow_pipeline::shader_bench::SHADERTOY_KERNEL_NAMES)
            .collect()
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn named_kernels_pass_the_numeric_quarantine() {
        use pixelflow_pipeline::training::quarantine::{QuarantineGrid, quarantine_verdict};
        let grid = QuarantineGrid::new();
        for name in all_named_kernels() {
            let (expr, env) = named_kernel(name).expect("known kernel");
            let verdict = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
            assert!(
                verdict.exclusion.is_none(),
                "named kernel {name} must pass its own quarantine; got {:?}",
                verdict.exclusion
            );
        }
    }

    #[test]
    fn named_kernels_are_screenable_by_the_oracle() {
        use pixelflow_pipeline::training::quarantine::screen_for_oracle;
        for name in all_named_kernels() {
            let (expr, env) = named_kernel(name).expect("known kernel");
            screen_for_oracle(Term::new(expr.entry(), &env))
                .unwrap_or_else(|e| panic!("named kernel {name} is not quarantinable: {e}"));
        }
    }

    #[test]
    fn named_kernel_lookup_rejects_unknown_names() {
        assert!(named_kernel("swirl").is_some());
        assert!(named_kernel("mandelbrot_distance").is_some());
        assert!(named_kernel("does_not_exist").is_none());
    }
}
