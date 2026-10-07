//! Candidate static predictors, and what they are scored on.
//!
//! Every predictor here is a closed-form function of one row's static
//! features. There is deliberately nothing fitted: the record this harness was
//! built to serve says three allocator policies lost to a hand-written
//! constant because the quantity they optimized was not time, and a model with
//! more parameters than the data supports would repeat that failure with more
//! machinery (the extraction-head workshop paper is the same lesson one layer
//! up; see `docs/plans/2026-09-01-schedule-cost-model-denotation.md`). What
//! is wanted first is whether *any* simple, inspectable
//! quantity tracks the clock on both tiers.
//!
//! Two scores, and the second is the one that matters.
//!
//! - **Spearman ρ** between predicted and measured across kernels, per tier
//!   and allocation. It says whether the predictor orders kernels correctly —
//!   easy to score well on, because kernel size dominates and every candidate
//!   here grows with it.
//! - **Sign agreement** on the delta between two allocations of the *same*
//!   kernel. That is the question an allocator actually asks: given two ways
//!   to allocate this kernel, is B faster than A? A predictor can rank kernels
//!   at ρ = 0.99 and still get every one of these backwards, and the 2026-09-04
//!   measurements are exactly that case.
//!
//! A delta smaller than the two measurements' own pass-to-pass spreads is
//! excluded from the sign score and reported as unresolved, because a coin
//! flip on an unmeasurable difference is not evidence either way. That
//! threshold is per comparison rather than corpus-wide: one threshold for the
//! whole corpus is set by whichever kernel measured worst, and throws away
//! every comparison on the kernels that measured well.

use std::collections::BTreeMap;

use super::row::{Row, Stat};

/// L1 instruction cache, the budget a kernel's code either fits in or does
/// not. 32 KB on every x86 core this repo targets.
const L1I_BYTES: f64 = 32.0 * 1024.0;

/// Relative deltas below this are unresolved however tight the passes looked.
/// A kernel whose passes happened to agree exactly has not proved that a 0.1%
/// difference is real.
const MIN_RESOLVABLE: f64 = 0.01;

/// A named, closed-form function of a row's static features.
pub struct Predictor {
    pub name: &'static str,
    /// What it means, one line, for the report.
    pub gloss: &'static str,
    pub eval: fn(&Row) -> f64,
}

/// Trip-weight a per-scope quantity: the prologue runs once per call, the row
/// prologue once per row, the body once per group per row.
///
/// The scaffold is excluded — it is the same code under every allocation, so
/// it adds the same constant to both sides of every comparison.
fn dyn_of(r: &Row, per_scope: fn(&super::row::ScopeRow) -> u64) -> f64 {
    let s = &r.statics;
    let (rows, groups) = (r.rows as f64, r.groups as f64);
    per_scope(&s.frame) as f64
        + per_scope(&s.row) as f64 * rows
        + per_scope(&s.body) as f64 * rows * groups
}

/// The candidates, simplest first.
#[must_use]
pub fn predictors() -> Vec<Predictor> {
    vec![
        Predictor {
            name: "dyn_mem_ops",
            gloss: "Σ scopes (loads+stores × executions per call) — the quantity 3″ optimized",
            eval: |r| r.statics.dyn_memory_ops as f64,
        },
        Predictor {
            name: "dyn_mem_bytes",
            gloss: "dyn_mem_ops × vector width — a 64-byte spill priced above a 16-byte one \
                    (identical to `dyn_mem_ops` within one tier; the difference is visible only \
                    in the pooled column)",
            eval: |r| r.statics.dyn_memory_ops as f64 * f64::from(r.statics.vector_bytes),
        },
        Predictor {
            name: "dyn_sched_ops",
            gloss: "trip-weighted count of the DAG's scheduled operations — blind to allocation \
                    by construction, since allocating a schedule does not change it",
            eval: |r| r.statics.dyn_instructions as f64,
        },
        Predictor {
            name: "dyn_traffic",
            gloss: "dyn_mem_ops + trip-weighted rematerializations — every value the code has to \
                    fetch or rebuild, whether or not it comes from memory",
            eval: |r| {
                dyn_of(r, |s| {
                    s.loads_transient + s.loads_kept + s.remats + s.stores
                })
            },
        },
        Predictor {
            name: "dyn_emitted_ops",
            gloss: "trip-weighted scheduled operations + every load, store and rematerialization \
                    around them",
            eval: |r| {
                dyn_of(r, |s| {
                    s.instructions + s.loads_transient + s.loads_kept + s.remats + s.stores
                })
            },
        },
        Predictor {
            name: "dyn_bytes",
            gloss: "trip-weighted code bytes",
            eval: |r| r.statics.dyn_bytes as f64,
        },
        Predictor {
            name: "dyn_bytes_l1i",
            gloss: "trip-weighted code bytes × (1 + the function's overflow of a 32 KB L1i)",
            eval: |r| r.statics.dyn_bytes as f64 * l1i_pressure(r),
        },
        Predictor {
            name: "dyn_ops_plus_3mem",
            gloss: "trip-weighted scheduled operations + 3 × dyn_mem_ops (a memory op priced as \
                    three arithmetic ones)",
            eval: |r| r.statics.dyn_instructions as f64 + 3.0 * r.statics.dyn_memory_ops as f64,
        },
        Predictor {
            name: "static_mem_ops",
            gloss: "the rejected quantity: memory ops with no trip weighting at all",
            eval: |r| {
                (r.statics.frame.memory_ops()
                    + r.statics.row.memory_ops()
                    + r.statics.body.memory_ops()) as f64
            },
        },
        Predictor {
            name: "static_bytes",
            gloss: "the other rejected quantity: total emitted code bytes",
            eval: |r| r.statics.bytes_total as f64,
        },
    ]
}

/// 1 when the whole function fits L1i, growing linearly with the overflow.
fn l1i_pressure(r: &Row) -> f64 {
    1.0 + (r.statics.bytes_total as f64 / L1I_BYTES - 1.0).max(0.0)
}

// =============================================================================
// Aggregation
// =============================================================================

/// One kernel, one tier, one allocation: passes collapsed to a median.
#[derive(Clone, Debug)]
pub struct Point {
    pub git_ref: String,
    pub tier: String,
    pub kernel: String,
    pub family: String,
    pub ns: f64,
    /// Interquartile relative spread across this build's passes of this
    /// kernel — the harness's own resolution here.
    pub pass_spread: f64,
    pub row: Row,
}

/// Collapse passes into one point per (ref, tier, kernel).
///
/// # Panics
/// If two rows claim the same (ref, tier, kernel, pass), or if the file mixes
/// schema versions.
#[must_use]
pub fn points(rows: &[Row], stat: Stat) -> Vec<Point> {
    let mut grouped: BTreeMap<(String, String, String), Vec<&Row>> = BTreeMap::new();
    for row in rows {
        assert_eq!(
            row.schema,
            super::row::SCHEMA,
            "{}: row written by a different schema version",
            row.kernel
        );
        grouped
            .entry((row.git_ref.clone(), row.tier.clone(), row.kernel.clone()))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(|((git_ref, tier, kernel), rows)| {
            let ns: Vec<f64> = rows.iter().map(|r| stat.of(&r.measured)).collect();
            let median = percentile(&ns, 50.0);
            // Interquartile, not max-minus-min: a corpus-wide floor taken from
            // the extremes is set by whichever pass the machine hiccuped in,
            // and one such pass in a sweep makes every kernel look unresolved.
            let spread = if ns.len() > 2 {
                (percentile(&ns, 75.0) - percentile(&ns, 25.0)) / median.max(f64::MIN_POSITIVE)
            } else {
                0.0
            };
            Point {
                git_ref,
                tier,
                kernel,
                family: rows[0].family.clone(),
                ns: median,
                pass_spread: spread,
                row: rows[0].clone(),
            }
        })
        .collect()
}

// =============================================================================
// Statistics
// =============================================================================

/// Spearman's ρ, average ranks for ties. NaN for fewer than three points or a
/// constant series (a rank correlation with nothing to rank is not zero).
#[must_use]
pub fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "spearman needs paired series");
    if xs.len() < 3 {
        return f64::NAN;
    }
    let (rx, ry) = (ranks(xs), ranks(ys));
    pearson(&rx, &ry)
}

fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut out = vec![0.0; xs.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && xs[order[j + 1]] == xs[order[i]] {
            j += 1;
        }
        // Average rank across the tie group, 1-based.
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            out[k] = rank;
        }
        i = j + 1;
    }
    out
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}

/// Nearest-rank percentile of an unsorted slice. NaN when empty.
#[must_use]
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Geometric mean of positive ratios. NaN when empty.
#[must_use]
pub fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = xs.iter().map(|x| x.ln()).sum();
    (sum / xs.len() as f64).exp()
}

// =============================================================================
// The report
// =============================================================================

/// How one predictor did on one tier's sign test.
#[derive(Clone, Debug)]
pub struct SignScore {
    pub agreed: usize,
    pub disagreed: usize,
    pub predictor_flat: usize,
}

impl SignScore {
    #[must_use]
    pub fn decided(&self) -> usize {
        self.agreed + self.disagreed
    }
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.decided() == 0 {
            return f64::NAN;
        }
        self.agreed as f64 / self.decided() as f64
    }
}

/// The whole analysis, rendered as markdown so it can be pasted into the plan
/// document that asked for it.
///
/// # Panics
/// If the rows hold no tier or no allocation variant.
#[must_use]
pub fn report(rows: &[Row], stat: Stat) -> String {
    use std::fmt::Write as _;

    let points = points(rows, stat);
    assert!(!points.is_empty(), "no rows to analyze");

    let tiers = distinct(points.iter().map(|p| p.tier.clone()));
    let refs = distinct(points.iter().map(|p| p.git_ref.clone()));
    let mut out = String::new();
    writeln!(
        out,
        "Timing statistic: **{}**, over {} rows, {} allocation(s), {} tier(s).\n",
        stat.name(),
        rows.len(),
        refs.len(),
        tiers.len()
    )
    .expect("fmt");

    // ---- the harness's own noise floor -------------------------------------
    writeln!(out, "## A/A noise floor (pass-to-pass, same build)\n").expect("fmt");
    writeln!(out, "| tier | kernels | median IQ spread | p95 IQ spread |").expect("fmt");
    writeln!(out, "|---|---:|---:|---:|").expect("fmt");
    for tier in &tiers {
        let s: Vec<f64> = points
            .iter()
            .filter(|p| &p.tier == tier)
            .map(|p| p.pass_spread)
            .collect();
        writeln!(
            out,
            "| {tier} | {} | {:.2}% | {:.2}% |",
            s.len(),
            percentile(&s, 50.0) * 100.0,
            percentile(&s, 95.0) * 100.0
        )
        .expect("fmt");
    }
    writeln!(
        out,
        "\nA comparison is *resolved* when the two allocations differ by more than the mean of \
         their own interquartile spreads (floor {:.0}%). Resolution is per comparison: a single \
         corpus-wide threshold is set by the worst kernel in the corpus and would discard every \
         comparison on the well-behaved ones.\n",
        MIN_RESOLVABLE * 100.0
    )
    .expect("fmt");

    // ---- what each allocation actually costs --------------------------------
    let baseline = refs.first().expect("at least one ref").clone();
    writeln!(
        out,
        "## Measured time by allocation (geomean vs `{baseline}`)\n"
    )
    .expect("fmt");
    write!(out, "| tier | family |").expect("fmt");
    for r in &refs {
        write!(out, " {r} |").expect("fmt");
    }
    writeln!(out).expect("fmt");
    write!(out, "|---|---|").expect("fmt");
    for _ in &refs {
        write!(out, "---:|").expect("fmt");
    }
    writeln!(out).expect("fmt");
    let mut families = distinct(points.iter().map(|p| p.family.clone()));
    families.insert(0, "ALL".to_string());
    for tier in &tiers {
        for family in &families {
            write!(out, "| {tier} | {family} |").expect("fmt");
            for r in &refs {
                let ratios: Vec<f64> = points
                    .iter()
                    .filter(|p| {
                        &p.tier == tier
                            && &p.git_ref == r
                            && (family == "ALL" || &p.family == family)
                    })
                    .filter_map(|p| {
                        let base = points.iter().find(|q| {
                            q.git_ref == baseline && q.tier == p.tier && q.kernel == p.kernel
                        })?;
                        Some(p.ns / base.ns)
                    })
                    .collect();
                write!(out, " {:.4} |", geomean(&ratios)).expect("fmt");
            }
            writeln!(out).expect("fmt");
        }
    }
    writeln!(out).expect("fmt");

    // ---- the corpus totals, per allocation ----------------------------------
    writeln!(
        out,
        "## Corpus totals per allocation\n\nWhat the allocations differ by, summed over the \
         corpus and weighted by each scope's trip count. `sched ops` is the DAG's own operation \
         count, which allocation does not change — it is here to show that.\n"
    )
    .expect("fmt");
    writeln!(
        out,
        "| tier | allocation | dyn mem ops | dyn remats | mem + remats | sched ops | code bytes |"
    )
    .expect("fmt");
    writeln!(out, "|---|---|---:|---:|---:|---:|---:|").expect("fmt");
    for tier in &tiers {
        for r in &refs {
            let sel: Vec<&Point> = points
                .iter()
                .filter(|p| &p.tier == tier && &p.git_ref == r)
                .collect();
            let sum = |f: fn(&Row) -> f64| -> f64 { sel.iter().map(|p| f(&p.row)).sum() };
            let mem = sum(|r| r.statics.dyn_memory_ops as f64);
            let remat = sum(|r| dyn_of(r, |s| s.remats));
            writeln!(
                out,
                "| {tier} | {r} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} |",
                mem,
                remat,
                mem + remat,
                sum(|r| r.statics.dyn_instructions as f64),
                sum(|r| r.statics.bytes_total as f64),
            )
            .expect("fmt");
        }
    }
    writeln!(out).expect("fmt");

    // ---- Spearman -----------------------------------------------------------
    writeln!(out, "## Spearman ρ, predicted vs measured across kernels\n").expect("fmt");
    writeln!(
        out,
        "Per tier and pooled over both. The pooled column is the only place a predictor that \
         prices the vector width can differ from one that does not — within a tier the width is \
         a constant, so multiplying by it cannot change a ranking.\n"
    )
    .expect("fmt");
    write!(out, "| predictor |").expect("fmt");
    for tier in &tiers {
        write!(out, " {tier} |").expect("fmt");
    }
    writeln!(out, " pooled |").expect("fmt");
    write!(out, "|---|").expect("fmt");
    for _ in &tiers {
        write!(out, "---:|").expect("fmt");
    }
    writeln!(out, "---:|").expect("fmt");
    for p in predictors() {
        write!(out, "| `{}` |", p.name).expect("fmt");
        for tier in &tiers {
            // Pooled over allocations: each (ref, kernel) is one observation.
            let sel: Vec<&Point> = points.iter().filter(|q| &q.tier == tier).collect();
            let xs: Vec<f64> = sel.iter().map(|q| (p.eval)(&q.row)).collect();
            let ys: Vec<f64> = sel.iter().map(|q| q.ns).collect();
            write!(out, " {:.3} |", spearman(&xs, &ys)).expect("fmt");
        }
        let xs: Vec<f64> = points.iter().map(|q| (p.eval)(&q.row)).collect();
        let ys: Vec<f64> = points.iter().map(|q| q.ns).collect();
        writeln!(out, " {:.3} |", spearman(&xs, &ys)).expect("fmt");
    }
    writeln!(out).expect("fmt");

    // ---- the sign test ------------------------------------------------------
    writeln!(
        out,
        "## Sign of the delta between allocations of the same kernel\n"
    )
    .expect("fmt");
    writeln!(
        out,
        "Every pair of allocations, every kernel whose measured delta is resolved. `n` is how \
         many such comparisons exist; accuracy is how often the predictor got the direction \
         right; `flat` is how often it had no opinion (predicted exactly zero), which is not \
         counted either way.\n"
    )
    .expect("fmt");
    write!(out, "| predictor |").expect("fmt");
    for tier in &tiers {
        write!(out, " {tier} acc | n | flat |").expect("fmt");
    }
    writeln!(out, " both |").expect("fmt");
    write!(out, "|---|").expect("fmt");
    for _ in &tiers {
        write!(out, "---:|---:|---:|").expect("fmt");
    }
    writeln!(out, "---:|").expect("fmt");
    for p in predictors() {
        write!(out, "| `{}` |", p.name).expect("fmt");
        let mut all = SignScore {
            agreed: 0,
            disagreed: 0,
            predictor_flat: 0,
        };
        for tier in &tiers {
            let score = sign_score(&points, &refs, tier, &p);
            all.agreed += score.agreed;
            all.disagreed += score.disagreed;
            all.predictor_flat += score.predictor_flat;
            write!(
                out,
                " {:.1}% | {} | {} |",
                score.accuracy() * 100.0,
                score.decided(),
                score.predictor_flat
            )
            .expect("fmt");
        }
        writeln!(out, " {:.1}% |", all.accuracy() * 100.0).expect("fmt");
    }
    writeln!(out).expect("fmt");

    // ---- the same, broken out by which two allocations are compared ---------
    writeln!(
        out,
        "### By allocation pair\n\nOne easy comparison can carry an aggregate: an allocation that \
         is twice as slow everywhere is predicted correctly by anything monotone in code size. \
         Accuracy per pair says which comparisons a predictor is actually making.\n"
    )
    .expect("fmt");
    let pairs: Vec<(String, String)> = refs
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            refs.iter()
                .skip(i + 1)
                .map(move |b| (a.clone(), b.clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    for tier in &tiers {
        writeln!(out, "**{tier}**\n").expect("fmt");
        write!(out, "| pair | n |").expect("fmt");
        for p in predictors() {
            write!(out, " `{}` |", p.name).expect("fmt");
        }
        writeln!(out).expect("fmt");
        write!(out, "|---|---:|").expect("fmt");
        for _ in predictors() {
            write!(out, "---:|").expect("fmt");
        }
        writeln!(out).expect("fmt");
        for (a, b) in &pairs {
            let cmps = comparisons(&points, tier, a, b);
            write!(out, "| {a} → {b} | {} |", cmps.len()).expect("fmt");
            for p in predictors() {
                let s = score_comparisons(&cmps, &p);
                if s.decided() == 0 {
                    write!(out, " — |").expect("fmt");
                } else {
                    write!(out, " {:.0}% |", s.accuracy() * 100.0).expect("fmt");
                }
            }
            writeln!(out).expect("fmt");
        }
        writeln!(out).expect("fmt");
    }

    // ---- glosses ------------------------------------------------------------
    writeln!(out, "## What each predictor is\n").expect("fmt");
    for p in predictors() {
        writeln!(out, "- `{}` — {}", p.name, p.gloss).expect("fmt");
    }

    out
}

/// One resolved comparison: two allocations of one kernel on one tier, whose
/// measured difference is bigger than either measurement's own dispersion.
struct Comparison<'a> {
    a: &'a Point,
    b: &'a Point,
    /// `b − a`, relative.
    measured: f64,
}

/// Every comparison on `tier` between the given allocations that the
/// measurement resolves.
///
/// Resolution is per comparison, not corpus-wide: the difference has to exceed
/// the mean of the two points' own interquartile spreads (floored at
/// [`MIN_RESOLVABLE`]). A single corpus-wide threshold is set by the worst
/// kernel in the corpus and throws away every comparison on the well-behaved
/// ones.
fn comparisons<'a>(
    points: &'a [Point],
    tier: &str,
    a_ref: &str,
    b_ref: &str,
) -> Vec<Comparison<'a>> {
    let mut out = Vec::new();
    for pa in points
        .iter()
        .filter(|p| p.tier == tier && p.git_ref == a_ref)
    {
        let Some(pb) = points
            .iter()
            .find(|p| p.tier == tier && p.git_ref == b_ref && p.kernel == pa.kernel)
        else {
            continue;
        };
        let scale = (pa.ns + pb.ns) / 2.0;
        if scale <= 0.0 {
            continue;
        }
        let measured = (pb.ns - pa.ns) / scale;
        let resolvable = ((pa.pass_spread + pb.pass_spread) / 2.0).max(MIN_RESOLVABLE);
        if measured.abs() < resolvable {
            continue;
        }
        out.push(Comparison {
            a: pa,
            b: pb,
            measured,
        });
    }
    out
}

fn score_comparisons(cmps: &[Comparison<'_>], predictor: &Predictor) -> SignScore {
    let mut score = SignScore {
        agreed: 0,
        disagreed: 0,
        predictor_flat: 0,
    };
    for c in cmps {
        let predicted = (predictor.eval)(&c.b.row) - (predictor.eval)(&c.a.row);
        if predicted == 0.0 {
            score.predictor_flat += 1;
            continue;
        }
        if predicted.signum() == c.measured.signum() {
            score.agreed += 1;
        } else {
            score.disagreed += 1;
        }
    }
    score
}

fn sign_score(points: &[Point], refs: &[String], tier: &str, predictor: &Predictor) -> SignScore {
    let mut total = SignScore {
        agreed: 0,
        disagreed: 0,
        predictor_flat: 0,
    };
    for (i, a) in refs.iter().enumerate() {
        for b in refs.iter().skip(i + 1) {
            let s = score_comparisons(&comparisons(points, tier, a, b), predictor);
            total.agreed += s.agreed;
            total.disagreed += s.disagreed;
            total.predictor_flat += s.predictor_flat;
        }
    }
    total
}

fn distinct(it: impl Iterator<Item = String>) -> Vec<String> {
    let mut v: Vec<String> = it.collect();
    v.sort();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_is_one_on_a_monotone_series_and_minus_one_reversed() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [10.0, 20.0, 25.0, 40.0, 90.0];
        assert!((spearman(&xs, &ys) - 1.0).abs() < 1e-12);
        let rev: Vec<f64> = ys.iter().rev().copied().collect();
        assert!((spearman(&xs, &rev) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn spearman_averages_tied_ranks() {
        // Ties must not be broken by input order, or the score depends on the
        // order rows happened to be written in.
        let xs = [1.0, 1.0, 2.0, 3.0];
        let ys = [5.0, 5.0, 6.0, 7.0];
        let forward = spearman(&xs, &ys);
        let xs_rev = [1.0, 1.0, 2.0, 3.0];
        let ys_rev = [5.0, 5.0, 6.0, 7.0];
        assert_eq!(forward, spearman(&xs_rev, &ys_rev));
        assert!((forward - 1.0).abs() < 1e-12);
    }

    #[test]
    fn geomean_of_ones_is_one() {
        assert!((geomean(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-12);
        assert!((geomean(&[0.5, 2.0]) - 1.0).abs() < 1e-12);
    }
}
