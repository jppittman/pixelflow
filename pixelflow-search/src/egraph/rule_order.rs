//! Sweep order for the base-62 production rule set
//! (docs/results/2026-09-01-rule-order-real-kernels.md).
//!
//! `all_rules()` is a `Vec`, and `EGraph::with_rules` sweeps it in order —
//! every saturation round applies rule 0, then rule 1, ..., then rule 61.
//! Which rule fires *first* inside a round, when several match the same
//! e-class, changes what an early-terminated sweep (an iteration cap, a
//! class cap, a wall-clock deadline) has done by the time it stops. Round 2
//! v3 (`docs/plans/2026-09-01-phase3-round2-registration-v3.md` §6b) found
//! this order effect dominant on synthetic classical expressions; this
//! module is the harness-scale piece that lets
//! `docs/results/2026-09-01-rule-order-real-kernels.md` re-measure it on
//! real kernels: production `all_rules()` order, the numeric-first static
//! reorder, and seeded shuffles.
//!
//! **Harness-only, same as `pixelflow-search::runtime`'s
//! `production_telemetry` module below it.** Nothing here changes
//! `all_rules()` itself or any production call — `RuleOrder::Production`
//! returns `all_rules()` verbatim, and every other variant is reached only
//! by a caller that explicitly asks for it.

use super::rewrite::Rewrite;

/// Which order to sweep the base-62 rule set in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleOrder {
    /// `all_rules()` verbatim — what ships today.
    Production,
    /// [`NUMERIC_FIRST_ORDER`]: descending TRAIN strict-positive rate, ties
    /// broken by ascending production index.
    NumericFirst,
    /// A Fisher-Yates shuffle of the base-62, seeded.
    Shuffled(u64),
}

impl core::fmt::Display for RuleOrder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Production => write!(f, "production"),
            Self::NumericFirst => write!(f, "numeric-first"),
            Self::Shuffled(seed) => write!(f, "shuffled({seed})"),
        }
    }
}

/// Build the base-62 rule set in `order`. Always the same 62 rules
/// (`super::all_rules()`'s multiset) — only the sequence changes, so a
/// caller comparing arms never has to worry about a rule being present in
/// one arm and absent in another.
#[must_use]
pub fn build_rule_set(order: RuleOrder) -> Vec<Box<dyn Rewrite>> {
    match order {
        RuleOrder::Production => super::all_rules(),
        RuleOrder::NumericFirst => {
            let mut base: Vec<Option<Box<dyn Rewrite>>> =
                super::all_rules().into_iter().map(Some).collect();
            NUMERIC_FIRST_ORDER
                .iter()
                .map(|&i| {
                    base[i].take().unwrap_or_else(|| {
                        panic!("build_rule_set: NumericFirst index {i} visited twice")
                    })
                })
                .collect()
        }
        RuleOrder::Shuffled(seed) => {
            let mut rules = super::all_rules();
            fisher_yates(&mut rules, seed);
            rules
        }
    }
}

/// Deterministic Fisher-Yates shuffle, seeded by a small xorshift64 PRNG —
/// no external `rand` dependency, and no reliance on any particular crate
/// version's shuffle algorithm (a portability concern for a *pinned*
/// ordering: `docs/plans/2026-09-01-phase3-round2-registration-v3.md` uses
/// the same construction for its `Interleave`/`Shuffled` orders, reproduced
/// here rather than imported since that plan's `inflate.rs` never shipped
/// to `main`).
fn fisher_yates<T>(v: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || {
        // xorshift64*
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..v.len()).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Base-62 production indices (into `super::all_rules()`, matching
/// `docs/results/2026-09-01-train-guide-report.md`'s "idx" column) ordered
/// by descending TRAIN strict-positive rate, ties broken by ascending
/// index. The 10 rules the report never mined a candidate for during label
/// generation (indices 5, 32, 33, 40, 46, 50, 54, 55, 57, 61) are treated as
/// rate 0.0, tied with every measured-zero rule.
///
/// Not recomputed at runtime — the report is a frozen artifact from a
/// specific training run. [`crate::egraph::rule_order::tests::numeric_first_order_is_pinned`]
/// re-derives this from the report's markdown table and asserts equality,
/// so a hand-transcription error is a test failure, not a silent drift.
#[rustfmt::skip]
pub const NUMERIC_FIRST_ORDER: [usize; 62] = [
    53, 60, 52, 51, 48, 35, 34, 47, 8, 0,
    3, 30, 38, 37, 20, 39, 59, 25, 31, 36,
    19, 27, 23, 22, 26, 21, 1, 2, 4, 5,
    6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 24, 28, 29, 32, 33, 40, 41, 42,
    43, 44, 45, 46, 49, 50, 54, 55, 56, 57,
    58, 61,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// `NUMERIC_FIRST_ORDER` must be a permutation of `0..62` — catches a
    /// transcription error (duplicate or missing index) that a purely
    /// arithmetic bug (e.g. a swapped digit) would otherwise let through
    /// silently, since `build_rule_set` panics on a repeat but would happily
    /// run forever short one rule if an index were simply missing.
    #[test]
    fn numeric_first_order_is_a_permutation() {
        let set: HashSet<usize> = NUMERIC_FIRST_ORDER.iter().copied().collect();
        assert_eq!(
            set.len(),
            62,
            "NUMERIC_FIRST_ORDER must be a permutation of every base index exactly once"
        );
        assert_eq!(*set.iter().max().unwrap(), 61);
    }

    /// Re-derive `NUMERIC_FIRST_ORDER` from
    /// `docs/results/2026-09-01-train-guide-report.md`'s per-rule table
    /// (descending "train rate", ties broken by ascending "idx"; a rule
    /// absent from the table gets rate 0.0) and assert it matches the
    /// pinned constant byte-for-byte. This is the reproducibility test the
    /// const's doc comment promises: run it after editing the report and a
    /// stale hand-transcription fails loudly instead of drifting.
    #[test]
    fn numeric_first_order_is_pinned() {
        let report = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../docs/results/2026-09-01-train-guide-report.md"
        ))
        .expect("read train-guide report");

        // Parse every "| rule | idx | train fired | train rate | ... |" row
        // out of the per-rule table. Column order is pinned by the report's
        // own header row, checked below so a reordered table fails loudly
        // rather than silently reading the wrong column.
        let header = "| rule | idx | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted |";
        assert!(
            report.contains(header),
            "train-guide report's per-rule table header changed — update the column \
             indices below"
        );

        // Only the per-rule table's rows, not the training-curve or
        // calibration tables above it (both also start with `|` and have a
        // numeric second column) — scan strictly after the header line.
        let body = report
            .split_once(header)
            .expect("header located above")
            .1
            .trim_start();

        let mut rate: [f64; 62] = [0.0; 62];
        let mut seen: HashSet<usize> = HashSet::new();
        for line in body.lines() {
            if line.trim().is_empty() {
                break;
            }
            if !line.starts_with('|') || line.starts_with("|---") {
                continue;
            }
            let cols: Vec<&str> = line.split('|').map(str::trim).collect();
            // cols[0] is "" (before the leading `|`); rule=1, idx=2,
            // train fired=3, train rate=4.
            if cols.len() < 5 {
                continue;
            }
            let Ok(idx) = cols[2].parse::<usize>() else {
                continue;
            };
            let Ok(train_rate) = cols[4].parse::<f64>() else {
                continue;
            };
            assert!(idx < 62, "report idx {idx} out of base-62 range");
            rate[idx] = train_rate;
            seen.insert(idx);
        }
        assert_eq!(
            seen.len(),
            52,
            "expected 52 rules with a mined row in the report (10 of 62 never fired a \
             candidate and are implicitly rate 0.0)"
        );

        let mut derived: Vec<usize> = (0..62).collect();
        derived.sort_by(|&a, &b| {
            rate[b]
                .partial_cmp(&rate[a])
                .expect("train rate is never NaN")
                .then(a.cmp(&b))
        });

        assert_eq!(
            derived, NUMERIC_FIRST_ORDER,
            "NUMERIC_FIRST_ORDER has drifted from a fresh derivation off the pinned report"
        );
    }

    #[test]
    fn production_order_is_all_rules_verbatim() {
        let production = build_rule_set(RuleOrder::Production);
        let all = super::super::all_rules();
        assert_eq!(production.len(), all.len());
        for (a, b) in production.iter().zip(all.iter()) {
            assert_eq!(a.name(), b.name());
        }
    }

    #[test]
    fn numeric_first_is_a_reordering_of_the_same_62_rules() {
        let numeric = build_rule_set(RuleOrder::NumericFirst);
        let mut numeric_names: Vec<String> = numeric.iter().map(|r| r.name().to_string()).collect();
        let mut all_names: Vec<String> = super::super::all_rules()
            .iter()
            .map(|r| r.name().to_string())
            .collect();
        numeric_names.sort_unstable();
        all_names.sort_unstable();
        assert_eq!(numeric_names, all_names);
    }

    #[test]
    fn shuffled_orders_are_a_reordering_and_differ_by_seed() {
        let s1 = build_rule_set(RuleOrder::Shuffled(1));
        let s2 = build_rule_set(RuleOrder::Shuffled(2));
        let s3 = build_rule_set(RuleOrder::Shuffled(3));
        assert_eq!(s1.len(), 62);
        let names = |rules: &[Box<dyn Rewrite>]| -> Vec<String> {
            rules.iter().map(|r| r.name().to_string()).collect()
        };
        let n1 = names(&s1);
        let n2 = names(&s2);
        let n3 = names(&s3);
        assert_ne!(
            n1, n2,
            "seeds 1 and 2 must not coincidentally produce the same order"
        );
        assert_ne!(
            n2, n3,
            "seeds 2 and 3 must not coincidentally produce the same order"
        );
        let mut sorted1 = n1.clone();
        let mut sorted_all: Vec<String> = super::super::all_rules()
            .iter()
            .map(|r| r.name().to_string())
            .collect();
        sorted1.sort_unstable();
        sorted_all.sort_unstable();
        assert_eq!(sorted1, sorted_all);
    }
}
