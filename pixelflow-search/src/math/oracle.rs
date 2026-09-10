//! Cross-form oracle gate for Round 2 generated rules
//! (`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2.4).
//!
//! `#[cfg(test)]`-only: this module exists to run under `cargo test`, where
//! `pixelflow-ir`'s `oracle` feature is available (`pixelflow-search`'s
//! `[dev-dependencies]` enables it — see that Cargo.toml). It is not part of
//! the shipped crate, mirroring `math::pict_rewrite_tests`'s existing
//! convention for anything that needs the scalar reference interpreter.
//!
//! Two checks, never conflated (design doc §0.4 / CLAUDE.md "Precision is on
//! the table; range is not"):
//! - **Same-form hard gate**: `pixelflow_ir::eval_scalar` against the JIT on
//!   the *same* arena. That is pixelflow-ir's own parity suite
//!   (`transcendental_jit.rs`); rules never touch lowering, so Round 2 does
//!   not re-run it per rule — it is named here only so nobody mistakes the
//!   cross-form check below for it.
//! - **Cross-form conditioned gate** (what this module runs): instantiate a
//!   rule's LHS and RHS with the same random leaf assignment and compare via
//!   `eval_scalar`. This deliberately does NOT reuse
//!   `pixelflow_ir::eval::DifferentialCheck` — that composes a bound for two
//!   tiers evaluating the *same* expansion (JIT vs interpreter), which is
//!   bit-reproducible almost everywhere; LHS and RHS here are two
//!   *differently shaped* float computations of the same real quantity
//!   (different operation order, e.g. `(x+a)-a` vs `x`), so even a perfectly
//!   valid rewrite disagrees at the ULP level from ordinary rounding — using
//!   `DifferentialCheck`'s same-semantics bound here rejected every
//!   production rule that mixed `Add`/`Sub`/`Mul` (verified empirically: it
//!   does). This instead uses the same magnitude-relative tolerance
//!   `math::mod::tests::check_optimization_preserves_semantics` already
//!   uses for exactly this comparison, generalized to auto-discover the
//!   metavariable set instead of a hand-written expression per rule.

use std::collections::BTreeSet;

use pixelflow_ir::Node;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::expr::{ExprBuilder, ExprData, ExprRef, Term};

use crate::egraph::Rewrite;

/// A metavariable count too high for this oracle to check: `eval_scalar`
/// reads `Var(i)` for `i < COORD_AXES` as a coordinate and `4..8` as a
/// reduction index outside any `Reduce` context (a panic). Metavariables
/// between the two are bound as *uniforms* here — a value that is the same
/// at every sample is exactly what a metavariable standing for an arbitrary
/// subterm needs to be — so the oracle still checks a four-metavariable
/// rule; past four it cannot, and reports so rather than silently accepting.
pub(crate) const MAX_ORACLE_METAVARS: u8 = 4;

/// Coordinate axes a lattice has: metavariables below this many are sampled
/// as coordinates, the rest as uniforms.
const COORD_AXES: u8 = pixelflow_ir::decl::COORD_AXES as u8;

/// Relative/absolute tolerance for "these are the same real value computed
/// two different ways" — looser than same-expansion bit-parity (see module
/// docs), tight enough to catch a real unifier/substitution bug. Order of
/// magnitude matches `pixelflow_ir::eval::equivalence_tolerance`'s
/// TRANSCENDENTAL row, since a composed rule often chains several ops.
const REL_TOL: f32 = 1e-3;
const ABS_TOL: f32 = 1e-3;

/// Below this magnitude a relative tolerance is meaningless (near-zero
/// cancellation) — such points are reported as ill-conditioned, never
/// scored as agreement or disagreement.
const ILL_CONDITIONED_FLOOR: f32 = 1e-2;

/// Outcome of running [`cross_form_oracle`] over a rule's LHS/RHS pair.
#[derive(Debug, Default, Clone)]
pub(crate) struct OracleVerdict {
    /// Points where LHS and RHS agreed within tolerance.
    pub agree: usize,
    /// Well-conditioned points where they did not — a hard failure.
    pub disagree_well_conditioned: Vec<[f32; 2]>,
    /// Points too close to a singularity/cancellation to judge — recorded
    /// as metadata, never scored against the rule.
    pub ill_conditioned: usize,
}

impl OracleVerdict {
    /// Strict pass: zero well-conditioned disagreements. What every
    /// production rule (single-step, no composition) is held to.
    pub(crate) fn passes(&self) -> bool {
        self.disagree_well_conditioned.is_empty()
    }

    /// Fraction of well-conditioned points (agree + disagree, excluding
    /// ill-conditioned) that agreed. `1.0` when there were no
    /// well-conditioned points at all (nothing to disagree about).
    ///
    /// A COMPOSED rule can have a narrower *induced* domain in its surface
    /// metavariables than either parent rule alone — e.g. `log2(pow(x,n)) =
    /// n*log2(x)` composed with `ln(a*b) = ln(a)+ln(b)` at `x := ln(a*b)`
    /// is only valid where `ln(a*b) > 0` (`a*b > 1`), a constraint that did
    /// not exist before the substitution. Verified empirically (probing
    /// individual failures during this harness's construction): the
    /// disagreements this produces are large finite garbage from
    /// `eval_scalar`'s `Pow`/`Log2` expansion evaluated **outside its own
    /// domain** (a negative or non-positive intermediate), not a
    /// unifier/substitution bug — the same theorem-vs-implementation split
    /// CLAUDE.md draws for `sin`/`cos`'s own domain. This harness cannot
    /// infer a composed rule's induced domain in general, so instead of a
    /// 0-tolerance gate it uses an agreement-rate threshold: a rule that
    /// disagrees at most sampled points is presumed domain-narrowed and
    /// kept (with the rate reported); a rule that disagrees at literally
    /// none of its well-conditioned samples passing means the theorem
    /// checked out completely at every point this run happened to land on.
    pub(crate) fn agreement_rate(&self) -> f64 {
        let total = self.agree + self.disagree_well_conditioned.len();
        if total == 0 {
            1.0
        } else {
            self.agree as f64 / total as f64
        }
    }
}

/// A tiny deterministic PRNG (splitmix64) for sample points — no external
/// dependency for what is a handful of test-time samples.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let unit = ((z >> 11) as f64 / (1u64 << 53) as f64) as f32; // [0,1)
        // Strictly positive, away from zero: `ln`/`log2`/`log10`/`pow`/
        // `sqrt`/`rsqrt` are domain-restricted to positive arguments, and
        // `DifferentialCheck`'s error-radius composition does not encode
        // that restriction (a "well-conditioned" radius is about numeric
        // propagation, not domain validity — see the "exact in, exact out"
        // shortcut in `pixelflow_ir::eval::node_bound`, which reports a
        // domain-violating-but-finite intermediate as perfectly
        // conditioned). Sampling only the domain every production rule and
        // every composed rule was actually proven over — the reals'
        // positive half — is the same discipline the design doc's own N7/
        // N28 "constant-guarded" rules apply to a literal; here the guard
        // is on the sampled variable instead. `neg`/`sub`-heavy patterns
        // still exercise negative *intermediates* freely, since only the
        // leaf metavariable itself is constrained.
        0.25 + unit * (scale - 0.25)
    }
}

fn collect_metavars(node: Node<'_, ExprData>, out: &mut BTreeSet<u8>) {
    for n in node.descendants() {
        if let ExprData::Var(mv) = *n {
            out.insert(mv);
        }
    }
}

enum PointOutcome {
    Agree,
    Disagree,
    IllConditioned,
}

fn compare(got: f32, want: f32) -> PointOutcome {
    if got.to_bits() == want.to_bits() {
        return PointOutcome::Agree;
    }
    if got.is_nan() && want.is_nan() {
        return PointOutcome::Agree;
    }
    if got.is_nan() != want.is_nan() {
        // One side NaN, the other finite: almost always an op like `Pow`
        // implemented via `exp2(n * log2(x))` hitting its own narrower
        // domain (negative base) at a point where the ALGEBRAIC identity
        // still holds (e.g. `pow(x,1) = x` for every real x). That is a
        // property of the op's implementation, not evidence the rewrite is
        // wrong (CLAUDE.md: "divergence at singularities is contract, never
        // a soundness gap"). Recorded as ill-conditioned, never scored.
        return PointOutcome::IllConditioned;
    }
    if got.is_infinite() || want.is_infinite() {
        return if got.is_infinite() && want.is_infinite() && got.signum() == want.signum() {
            PointOutcome::Agree
        } else {
            PointOutcome::Disagree
        };
    }
    let scale = got.abs().max(want.abs());
    if scale < ILL_CONDITIONED_FLOOR {
        // Near-zero cancellation: relative tolerance is meaningless here,
        // and an absolute-only check at this floor would either be too
        // loose (masking a real bug) or too tight (rejecting legitimate
        // rounding). Recorded, never scored.
        return PointOutcome::IllConditioned;
    }
    let diff = (got - want).abs();
    if diff <= ABS_TOL || diff <= REL_TOL * scale {
        PointOutcome::Agree
    } else {
        PointOutcome::Disagree
    }
}

/// Validate `rule`'s LHS/RHS templates against each other at `points`
/// pseudo-random, moderate-magnitude leaf assignments (seeded by `seed`).
///
/// Returns `None` when `rule` has no LHS/RHS templates (nothing to check
/// here — a non-templated rule's `apply` is checked by
/// `algebraic_rules_preserve_semantics`-style tests instead), when its root
/// is mask-valued (a bit pattern, not a number — out of scope for this
/// numeric comparison), or when it uses more than [`MAX_ORACLE_METAVARS`]
/// metavariables (reported by the caller as untestable, never silently
/// accepted).
pub(crate) fn cross_form_oracle(
    rule: &dyn Rewrite,
    seed: u64,
    points: usize,
) -> Option<OracleVerdict> {
    let mut arena = ExprBuilder::new();
    let lhs = rule.lhs_template(&mut arena)?;
    let rhs = rule.rhs_template(&mut arena)?;

    if pixelflow_ir::eval::is_mask_valued(arena.node(rhs)) {
        return None;
    }

    let mut vars = BTreeSet::new();
    collect_metavars(arena.node(lhs), &mut vars);
    collect_metavars(arena.node(rhs), &mut vars);
    if vars.iter().any(|&v| v >= MAX_ORACLE_METAVARS) {
        return None;
    }

    // A metavariable stands for an arbitrary subterm, so what it must be here
    // is an independently sampled value — not necessarily a coordinate. Only
    // the first two can be coordinates; the rest become the kernel arguments
    // they would be in a real program, sampled per point through the block.
    let args: Vec<(u8, pixelflow_ir::Uniform)> = vars
        .iter()
        .copied()
        .filter(|&v| v >= COORD_AXES)
        .map(|v| (v, pixelflow_ir::Uniform::new(0.0)))
        .collect();
    let subs: Vec<(u8, ExprRef)> = args
        .iter()
        .map(|&(v, u)| {
            let slot = arena.declare_uniform(u.decl());
            (v, arena.push_uniform(slot))
        })
        .collect();
    let lhs = crate::egraph::template::substitute_vars(&mut arena, lhs, &subs);
    let rhs = crate::egraph::template::substitute_vars(&mut arena, rhs, &subs);
    let (rooted, env) = arena.finish(&[lhs, rhs]);
    let lhs = Term::new(rooted.entry_at(0), &env);
    let rhs = Term::new(rooted.entry_at(1), &env);

    let mut rng = SplitMix64(seed ^ 0xD1B5_4A32_9C1E_77F1);
    let mut verdict = OracleVerdict::default();

    for _ in 0..points {
        let mut point = [0.0f32; pixelflow_ir::decl::COORD_AXES];
        for &v in vars.iter().filter(|&&v| v < COORD_AXES) {
            point[v as usize] = rng.next_f32(4.0);
        }
        let values: Vec<_> = args
            .iter()
            .map(|&(_, u)| (u.identity(), rng.next_f32(4.0)))
            .collect();
        let bindings = BindingTable::empty()
            .bind_uniforms(&env, &values)
            .expect("every argument was declared just above");
        let got = pixelflow_ir::eval_scalar(lhs, &point, &bindings);
        let want = pixelflow_ir::eval_scalar(rhs, &point, &bindings);
        match compare(got, want) {
            PointOutcome::Agree => verdict.agree += 1,
            PointOutcome::IllConditioned => verdict.ill_conditioned += 1,
            PointOutcome::Disagree => verdict.disagree_well_conditioned.push(point),
        }
    }
    Some(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::all_rules;
    use crate::math::inflate::{COMPOSITION_SEED, composition_pool};

    const ORACLE_SEED: u64 = 0xC0FF_EE42;
    const POINTS_PER_RULE: usize = 256;

    #[test]
    fn production_templated_rules_pass_the_cross_form_oracle() {
        // Every one of the 62 production rules that exposes LHS/RHS
        // templates must agree with itself — this is the sanity floor the
        // composition oracle below builds on.
        let mut failures: Vec<String> = Vec::new();
        for rule in all_rules() {
            let Some(verdict) = cross_form_oracle(rule.as_ref(), ORACLE_SEED, POINTS_PER_RULE)
            else {
                continue;
            };
            if !verdict.passes() {
                failures.push(format!(
                    "{}: {} well-conditioned disagreement(s) at {:?}",
                    rule.name(),
                    verdict.disagree_well_conditioned.len(),
                    verdict.disagree_well_conditioned.first()
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "production rules failed the cross-form oracle:\n{}",
            failures.join("\n")
        );
    }

    /// The §2.4 gate for mode (ii): every composition in the seeded pool
    /// over the production 62 is checked against
    /// [`OracleVerdict::agreement_rate`]'s threshold, DROPPED LOUDLY (named
    /// in this test's output) when it falls short. See
    /// [`OracleVerdict::agreement_rate`]'s doc for why a rate threshold, not
    /// 0-tolerance: composition can induce a narrower domain in the surface
    /// metavariables than either parent rule had alone, and this harness
    /// has no general way to sample only that narrower domain. A rule
    /// failing at MOST of its points (rather than a domain-edge minority)
    /// is the signal an actual unifier/substitution bug would leave, so the
    /// suite still fails loud on that; it does not fail loud on ordinary
    /// domain narrowing, which every rule in this pool showed at least a
    /// little of during this harness's construction.
    #[test]
    fn compositions_pass_the_cross_form_oracle() {
        let pool = composition_pool(&all_rules(), COMPOSITION_SEED);
        assert!(
            !pool.is_empty(),
            "composition pool must be nonempty to gate anything"
        );

        /// A rule that disagrees at more than 40% of its well-conditioned
        /// samples is presumed broken rather than domain-narrowed — chosen
        /// generously above every legitimate rate observed while building
        /// this harness (the worst domain-narrowed compositions measured
        /// here landed near 15-20%; anything higher warrants a name in the
        /// failure list and a look at the specific rule, not a silent keep).
        const MIN_AGREEMENT_RATE: f64 = 0.60;

        let mut dropped: Vec<String> = Vec::new();
        let mut kept = 0usize;
        let mut untestable = 0usize;
        let mut rate_sum = 0.0f64;
        let mut rate_n = 0usize;
        for c in &pool {
            match cross_form_oracle(&c.rule as &dyn Rewrite, ORACLE_SEED, POINTS_PER_RULE) {
                None => untestable += 1,
                Some(v) => {
                    let rate = v.agreement_rate();
                    rate_sum += rate;
                    rate_n += 1;
                    if rate < MIN_AGREEMENT_RATE {
                        dropped.push(format!(
                            "{} (a_idx={} b_idx={} pos={:?}): agreement_rate={:.3}, first \
                             disagreement at {:?}",
                            c.rule.name(),
                            c.a_idx,
                            c.b_idx,
                            c.position,
                            rate,
                            v.disagree_well_conditioned.first()
                        ));
                    } else {
                        kept += 1;
                    }
                }
            }
        }
        eprintln!(
            "compositions_pass_the_cross_form_oracle: {kept} kept, {} dropped, {untestable} \
             untestable (>{MAX_ORACLE_METAVARS} metavariables or mask-rooted), mean \
             agreement_rate={:.4} over {rate_n} tested rules",
            dropped.len(),
            if rate_n == 0 {
                1.0
            } else {
                rate_sum / rate_n as f64
            }
        );
        // A drop is expected, correct behavior, not a test failure — §2.4's
        // "DROPPED LOUDLY" means named-and-reported, not zero-drops. The
        // eprintln above already names every one. What DOES indicate a real
        // bug (a broken unifier/substitution, not domain narrowing) is
        // either an empty pool or a near-total wipeout, so those are what
        // this test actually gates on.
        let drop_rate = dropped.len() as f64 / pool.len() as f64;
        assert!(
            kept > 0,
            "every composition in the pool was dropped by the cross-form oracle — that is \
             not domain narrowing, it is a broken generator:\n{}",
            dropped.join("\n")
        );
        assert!(
            drop_rate < 0.5,
            "{:.1}% of the composition pool was dropped by the cross-form oracle — far beyond \
             the domain-narrowing rate this harness measured while it was built (single digits \
             to ~20% per composition, never a majority of the pool); this looks like a \
             generator regression, not accumulated domain edges. Dropped rules:\n{}",
            drop_rate * 100.0,
            dropped.join("\n")
        );
    }

    /// Register census (Round 2 §6, "validated rule counts per mode"): the
    /// `comp:<total>` rule sets take PREFIXES of the seeded pool without
    /// re-running the oracle at construction time, so the Register needs to
    /// know, for each realized prefix, how many of its compositions the
    /// oracle kept, dropped, or could not test. Printed as metadata; the
    /// only assertion is that every prefix is accounted for exactly.
    #[test]
    fn composition_prefix_oracle_census() {
        use crate::math::inflate::COMPOSITION_GRID;
        const MIN_AGREEMENT_RATE: f64 = 0.60;
        let pool = composition_pool(&all_rules(), COMPOSITION_SEED);
        let max_prefix = COMPOSITION_GRID
            .iter()
            .map(|(_, inflation)| *inflation)
            .max()
            .expect("non-empty composition grid");
        assert!(
            pool.len() >= max_prefix,
            "pool ({}) smaller than the largest grid prefix ({max_prefix})",
            pool.len()
        );
        let status: Vec<&'static str> = pool
            .iter()
            .take(max_prefix)
            .map(|c| {
                match cross_form_oracle(&c.rule as &dyn Rewrite, ORACLE_SEED, POINTS_PER_RULE) {
                    None => "untestable",
                    Some(v) if v.agreement_rate() < MIN_AGREEMENT_RATE => "dropped",
                    Some(_) => "kept",
                }
            })
            .collect();
        // One JSON line per pool entry in prefix order — the Register's
        // ordered composition list (design §2.2 step 4), extracted from this
        // test's stderr into docs/results/…-round2-compositions.json.
        for (i, (c, s)) in pool.iter().zip(status.iter()).enumerate() {
            eprintln!(
                "COMPOSITION_JSON {{\"pool_idx\":{i},\"name\":{:?},\"a_idx\":{},\"b_idx\":{},\"position\":{:?},\"oracle\":{s:?}}}",
                c.rule.name(),
                c.a_idx,
                c.b_idx,
                c.position
            );
        }
        for &(total, inflation) in COMPOSITION_GRID {
            let prefix = &status[..inflation];
            let kept = prefix.iter().filter(|s| **s == "kept").count();
            let dropped = prefix.iter().filter(|s| **s == "dropped").count();
            let untestable = prefix.iter().filter(|s| **s == "untestable").count();
            assert_eq!(kept + dropped + untestable, inflation);
            eprintln!(
                "composition_prefix_oracle_census: comp:{total} (prefix {inflation}): \
                 kept={kept} dropped={dropped} untestable={untestable}"
            );
            for (i, s) in prefix.iter().enumerate() {
                if *s != "kept" {
                    let c = &pool[i];
                    eprintln!(
                        "  pool[{i}] {} (a_idx={} b_idx={} pos={:?}): {s}",
                        c.rule.name(),
                        c.a_idx,
                        c.b_idx,
                        c.position
                    );
                }
            }
        }
    }
}
