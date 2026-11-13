//! Return-to-go record schema and label arithmetic for the Guide's new
//! training target (`docs/archive/plans/2026-09-01-guide-return-to-go.md`).
//!
//! # Scope of this module, and what it does not build
//!
//! The full mint (`gen_r2g_trajectories`, the twelve-policy trajectory
//! diversity of §2, the counterfactual replay of §4) is a separate,
//! larger task and may not have landed in this tree yet — this module
//! builds only what `train_guide_r2g` needs to consume the mint's *output*:
//! the JSONL record schema (§8's `R2gRecord`) and the pure label arithmetic
//! (§1.2's `log_regret`, §1.3's centering, §3.4's cross-trajectory ranking
//! pairs). If the minter has not landed when this lands, [`R2gRecord`] is
//! still exactly the schema §8 specifies, and the trainer is tested against
//! a small hand-built fixture (`train_guide_r2g`'s own test module) rather
//! than a real mint.
//!
//! # Why `R2gRecord` wraps [`Record`] instead of restating its fields
//!
//! §8: *"`R2gRecord` (all `guide_linear::Record` fields + ...)"* — a strict
//! superset, specifically so that [`crate::training::guide_linear::to_sample`]
//! applies to an `R2gRecord` unchanged (`&record.base`), the same anti-drift
//! discipline that module's own doc names: one feature encoding, imported
//! everywhere a candidate becomes a sample, never a second hand-rolled copy.
//! `base.label_positive` (the strict bit) is minted alongside for provenance
//! parity with the strict-label pipeline but is not what this record trains
//! on — the r2g-specific `return_b100`/`return_b200`/`centered_b100`/
//! `centered_b200` fields are (see [`target_value`]).

use std::collections::HashMap;

use pixelflow_search::egraph::{Fingerprint, RuleId, RuleSet};
use pixelflow_search::nnue::guide::linear::{LinearReturnGuide, LinearWeights, ReturnObjective};
use serde::Deserialize;

use super::guide_linear::{Model, Record};

/// One trajectory-labelled application record — everything a
/// [`Record`] carries (feature encoding), plus the return-to-go bookkeeping
/// `train_guide_r2g` needs: which trajectory/policy this application
/// belonged to, where in the trajectory it fell, and its trajectory's
/// return at each registered tier (§1.2/§1.3).
#[derive(Deserialize)]
pub struct R2gRecord {
    #[serde(flatten)]
    pub base: Record,

    /// Groups every record minted from the same `(expr, policy-run)` —
    /// distinct even between two trajectories of the same policy on the
    /// same expression (e.g. two different random seeds), so a consumer
    /// can tell "same trajectory" from "same policy" apart.
    pub trajectory_id: u32,
    /// `Policy::to_string()` — human-readable ordering-policy name (§2's
    /// table: `"unguided"`, `"per-rule"`, `"strict-v1"`, `"random:<seed>"`,
    /// `"mix:strict-v1:<n>/<d>:<seed>"`).
    pub policy: String,
    /// Saturation-round ordinal this application's *round* started at
    /// (guided-family trajectories only meaningfully use this; unguided's
    /// finest granularity is a per-rule sweep batch — §1.5).
    pub round_ordinal: u32,
    /// This application's own ordinal within its trajectory (`t` in §1.1's
    /// `a_t`).
    pub application_ordinal: u64,
    /// Whether this application actually changed the e-graph (committed a
    /// node or a union) — the no-op re-fires §2/§4.2 both call out as the
    /// majority of a rule-major sweep.
    pub changed: bool,

    /// Extraction cost at the B=100 checkpoint, of the trajectory this
    /// record's application belongs to.
    pub cost_b100: u64,
    /// Extraction cost at the B=200 checkpoint, same trajectory.
    pub cost_b200: u64,
    /// `c*_e` (§1.2): the best extraction cost seen across every trajectory
    /// of this expression, at any checkpoint — the per-expression reference
    /// every tier's regret is measured against.
    pub expr_best_cost: u64,

    /// `R(τ, 100)` (§1.2), `None` when `t > 100` or `expr_best_cost == 0`
    /// (§1.2's exclusion convention).
    pub return_b100: Option<f32>,
    /// `R(τ, 200)`, same convention.
    pub return_b200: Option<f32>,
    /// `R(τ,100) − R̄_e(100)` (§1.3) — the primary training target.
    pub centered_b100: Option<f32>,
    /// `R(τ,200) − R̄_e(200)`.
    pub centered_b200: Option<f32>,
}

/// Which registered tier's label to read off an [`R2gRecord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelBudget {
    B100,
    B200,
}

impl LabelBudget {
    #[must_use]
    pub fn as_u32(self) -> u32 {
        match self {
            LabelBudget::B100 => 100,
            LabelBudget::B200 => 200,
        }
    }
}

impl std::str::FromStr for LabelBudget {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "100" => Ok(LabelBudget::B100),
            "200" => Ok(LabelBudget::B200),
            other => Err(format!(
                "LabelBudget: {other:?} is not a registered tier — expected \"100\" or \"200\""
            )),
        }
    }
}

/// Which label column trains the model (§3.4's "raw target" ablation vs the
/// §3.2 primary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegressionTarget {
    /// §3.2 primary: expression-centered log-regret.
    Centered,
    /// §3.4 ablation: the raw (uncentered) log-regret.
    Raw,
}

impl std::str::FromStr for RegressionTarget {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "centered" => Ok(RegressionTarget::Centered),
            "raw" => Ok(RegressionTarget::Raw),
            other => Err(format!(
                "RegressionTarget: {other:?} is not \"centered\" or \"raw\""
            )),
        }
    }
}

/// Read the one label column `--target`/`--label-b` select off a record —
/// the single place that combination is resolved, so the trainer and any
/// future consumer (e.g. a report) can never disagree about which of the
/// four optional columns a given `(target, label_b)` pair means.
#[must_use]
pub fn target_value(
    record: &R2gRecord,
    label_b: LabelBudget,
    target: RegressionTarget,
) -> Option<f32> {
    match (label_b, target) {
        (LabelBudget::B100, RegressionTarget::Raw) => record.return_b100,
        (LabelBudget::B100, RegressionTarget::Centered) => record.centered_b100,
        (LabelBudget::B200, RegressionTarget::Raw) => record.return_b200,
        (LabelBudget::B200, RegressionTarget::Centered) => record.centered_b200,
    }
}

/// The raw trajectory return at `label_b`, regardless of `--target` —
/// [`sample_rank_pairs`] always orders pairs by the *raw* trajectory return
/// (§1.4: "ordered by `R(τ_i) − R(τ_j)`"), never the centered value, since
/// centering is a per-record nuisance-removal step for the regression
/// target and the ranking ablation compares whole trajectories directly.
fn raw_return(record: &R2gRecord, label_b: LabelBudget) -> Option<f32> {
    match label_b {
        LabelBudget::B100 => record.return_b100,
        LabelBudget::B200 => record.return_b200,
    }
}

/// §1.2's `R(τ,B) = ln(c_τ(B) / c*_e)`, with the registration's zero-best
/// convention: a positive cost against a zero reference is excluded
/// (`None`, "infinite loss"), and `cost == best == 0` is defined as `R = 0`
/// rather than excluded (the trajectory reached the — trivial — best
/// exactly). Minting itself decides *expression-level* exclusion
/// (`zero_best_excluded`, §1.2); this function only computes the ratio for
/// one `(cost, best)` pair and is the literal formula both the mint and any
/// consumer (§4's counterfactual `Δ`) share.
#[must_use]
pub fn log_regret(cost: u64, best: u64) -> Option<f32> {
    if best == 0 {
        return if cost == 0 { Some(0.0) } else { None };
    }
    Some((cost as f64 / best as f64).ln() as f32)
}

/// One §3.4 pairwise-ranking training example: two records of the **same
/// expression** at the **same budget-fraction decile**, drawn from
/// **different trajectories**, ordered by which trajectory's return was
/// better.
///
/// §1.4 is explicit that a within-*trajectory* pairwise loss is vacuous
/// (every application in one trajectory carries the same label) — "within a
/// step" only has signal *across* trajectories of the same expression at
/// the same position bucket, which is what this pairs.
pub struct RankPair {
    /// Index into the records slice the pair was sampled from.
    pub a: usize,
    pub b: usize,
    /// `sign(R(τ_b) − R(τ_a))`: `+1.0` if `a`'s trajectory reached the
    /// better (lower-regret) outcome, `-1.0` if `b`'s did. Zero-difference
    /// pairs (`spread_e(B) = 0` for this bucket) are never emitted — they
    /// carry no ranking signal (§2's spread-gate framing).
    pub sign: f32,
}

/// Deterministic splitmix64, matching the small-seeded-generator pattern
/// `train_guide`'s own `lcg_next`/`shuffled_indices` establish for this
/// crate's training binaries (reproducible from `--seed` alone, no external
/// RNG dependency).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Budget-fraction decile bucket, `0..=9` (`budget_fraction` is defined in
/// `[0,1]`; a value of exactly `1.0` falls in bucket 9, not an
/// out-of-range 10th bucket).
fn decile(budget_fraction: f32) -> u32 {
    ((budget_fraction * 10.0).floor() as i64).clamp(0, 9) as u32
}

/// Sample §3.4's cross-trajectory ranking pairs from a slice of records
/// (typically one split, e.g. all of TRAIN). Groups records by
/// `(expr_name, budget-fraction decile)`, and for every record whose bucket
/// holds at least one other trajectory, draws up to `pairs_per_record`
/// partners from a different `trajectory_id` in the same bucket,
/// deterministically from `seed`.
#[must_use]
pub fn sample_rank_pairs(
    records: &[R2gRecord],
    label_b: LabelBudget,
    pairs_per_record: usize,
    seed: u64,
) -> Vec<RankPair> {
    // (expr_name, decile) -> record indices, in encounter order.
    let mut buckets: HashMap<(String, u32), Vec<usize>> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        if raw_return(r, label_b).is_none() {
            continue;
        }
        let key = (r.base.expr_name.clone(), decile(r.base.budget_fraction));
        buckets.entry(key).or_default().push(i);
    }

    let mut pairs = Vec::new();
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;

    for idxs in buckets.values() {
        if idxs.len() < 2 {
            continue;
        }
        for &i in idxs {
            let ri = &records[i];
            let ret_i = raw_return(ri, label_b).expect("filtered above");
            let mut drawn = 0usize;
            let mut attempts = 0usize;
            // Bounded attempts: a bucket where every other record shares
            // `i`'s trajectory_id (single-trajectory bucket after all, just
            // multiple applications of it) has no valid partner — stop
            // rather than loop forever hunting for one.
            while drawn < pairs_per_record && attempts < pairs_per_record * 8 {
                attempts += 1;
                let r = splitmix64(&mut state) as usize % idxs.len();
                let j = idxs[r];
                if j == i {
                    continue;
                }
                let rj = &records[j];
                if rj.trajectory_id == ri.trajectory_id {
                    continue;
                }
                let ret_j = raw_return(rj, label_b).expect("filtered above");
                let diff = ret_j - ret_i;
                if diff == 0.0 {
                    continue;
                }
                pairs.push(RankPair {
                    a: i,
                    b: j,
                    sign: diff.signum(),
                });
                drawn += 1;
            }
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        expr_name: &str,
        trajectory_id: u32,
        budget_fraction: f32,
        return_b100: Option<f32>,
    ) -> R2gRecord {
        R2gRecord {
            base: Record {
                expr_name: expr_name.to_string(),
                family_band: 0,
                family_seed: 0,
                expr_node_count: 10,
                rule_id: pixelflow_search::egraph::RuleId::from_label("r0").get(),
                rule_name: "r0".to_string(),
                budget_fraction,
                match_class_node_count: 3,
                neighborhood_op_count: 2,
                neighborhood_op_hist: std::collections::BTreeMap::new(),
                dedup_repeat: false,
                label_positive: false,
            },
            trajectory_id,
            policy: "unguided".to_string(),
            round_ordinal: 0,
            application_ordinal: 1,
            changed: true,
            cost_b100: 10,
            cost_b200: 10,
            expr_best_cost: 5,
            return_b100,
            return_b200: None,
            centered_b100: return_b100.map(|r| r - 0.1),
            centered_b200: None,
        }
    }

    // ── log_regret ──────────────────────────────────────────────────────

    #[test]
    fn log_regret_is_zero_when_cost_equals_best() {
        assert_eq!(log_regret(7, 7), Some(0.0));
    }

    #[test]
    fn log_regret_is_positive_when_cost_exceeds_best() {
        let r = log_regret(20, 10).unwrap();
        assert!((r - (2.0f32).ln()).abs() < 1e-6);
        assert!(r > 0.0);
    }

    #[test]
    fn log_regret_excludes_a_positive_cost_against_a_zero_best() {
        assert_eq!(log_regret(5, 0), None);
    }

    #[test]
    fn log_regret_is_zero_when_both_cost_and_best_are_zero() {
        assert_eq!(log_regret(0, 0), Some(0.0));
    }

    // ── target_value ────────────────────────────────────────────────────

    #[test]
    fn target_value_reads_the_requested_column() {
        let mut r = record("e0", 0, 0.5, Some(1.5));
        r.return_b200 = Some(2.5);
        r.centered_b200 = Some(2.0);

        assert_eq!(
            target_value(&r, LabelBudget::B100, RegressionTarget::Raw),
            Some(1.5)
        );
        assert_eq!(
            target_value(&r, LabelBudget::B100, RegressionTarget::Centered),
            Some(1.4)
        );
        assert_eq!(
            target_value(&r, LabelBudget::B200, RegressionTarget::Raw),
            Some(2.5)
        );
        assert_eq!(
            target_value(&r, LabelBudget::B200, RegressionTarget::Centered),
            Some(2.0)
        );
    }

    #[test]
    fn target_value_is_none_where_the_mint_recorded_no_label() {
        let r = record("e0", 0, 0.5, None);
        assert_eq!(
            target_value(&r, LabelBudget::B100, RegressionTarget::Raw),
            None
        );
    }

    // ── LabelBudget / RegressionTarget parsing ─────────────────────────

    #[test]
    fn label_budget_parses_the_two_registered_tiers_and_rejects_others() {
        assert_eq!("100".parse::<LabelBudget>().unwrap(), LabelBudget::B100);
        assert_eq!("200".parse::<LabelBudget>().unwrap(), LabelBudget::B200);
        assert!("50".parse::<LabelBudget>().is_err());
    }

    #[test]
    fn regression_target_parses_centered_and_raw_and_rejects_others() {
        assert_eq!(
            "centered".parse::<RegressionTarget>().unwrap(),
            RegressionTarget::Centered
        );
        assert_eq!(
            "raw".parse::<RegressionTarget>().unwrap(),
            RegressionTarget::Raw
        );
        assert!("logit".parse::<RegressionTarget>().is_err());
    }

    // ── R2gRecord deserialization (the minter's schema, §8) ────────────

    #[test]
    fn r2g_record_deserializes_the_flattened_base_plus_trajectory_fields() {
        let json = serde_json::json!({
            "expr_name": "e42",
            "family_band": 1,
            "family_seed": 7,
            "expr_node_count": 12,
            "rule_id": 3,
            "rule_name": "mul_comm",
            "budget_fraction": 0.42,
            "match_class_node_count": 5,
            "neighborhood_op_count": 4,
            "neighborhood_op_hist": {"Add": 2, "Mul": 1},
            "dedup_repeat": false,
            "label_positive": true,
            "trajectory_id": 9,
            "policy": "random:3",
            "round_ordinal": 2,
            "application_ordinal": 17,
            "changed": true,
            "cost_b100": 100,
            "cost_b200": 90,
            "expr_best_cost": 80,
            "return_b100": 0.223,
            "return_b200": 0.117,
            "centered_b100": 0.05,
            "centered_b200": -0.01
        });
        let record: R2gRecord = serde_json::from_value(json).unwrap();
        assert_eq!(record.base.expr_name, "e42");
        assert_eq!(record.base.rule_id, 3);
        assert_eq!(record.base.neighborhood_op_hist.len(), 2);
        assert_eq!(record.trajectory_id, 9);
        assert_eq!(record.policy, "random:3");
        assert_eq!(record.application_ordinal, 17);
        assert!(record.changed);
        assert_eq!(record.cost_b100, 100);
        assert_eq!(record.expr_best_cost, 80);
        assert!((record.return_b100.unwrap() - 0.223).abs() < 1e-6);
        assert!((record.centered_b200.unwrap() - (-0.01)).abs() < 1e-6);
    }

    // ── sample_rank_pairs ───────────────────────────────────────────────

    #[test]
    fn sample_rank_pairs_only_pairs_records_of_the_same_expression_and_decile() {
        let records = vec![
            record("e0", 0, 0.15, Some(0.1)), // decile 1
            record("e0", 1, 0.19, Some(0.9)), // decile 1, different trajectory
            record("e1", 0, 0.15, Some(0.1)), // different expression
        ];
        let pairs = sample_rank_pairs(&records, LabelBudget::B100, 4, 1);
        for p in &pairs {
            assert_eq!(records[p.a].base.expr_name, records[p.b].base.expr_name);
            assert_ne!(records[p.a].trajectory_id, records[p.b].trajectory_id);
        }
        // The e0 pair at decile 1 must be found (only one possible pairing).
        assert!(
            pairs
                .iter()
                .any(|p| records[p.a].base.expr_name == "e0" && records[p.b].base.expr_name == "e0")
        );
    }

    #[test]
    fn sample_rank_pairs_signs_by_which_trajectory_had_lower_regret() {
        // a (traj 0) has the lower (better) return than b (traj 1).
        let records = vec![
            record("e0", 0, 0.5, Some(0.1)),
            record("e0", 1, 0.5, Some(0.9)),
        ];
        let pairs = sample_rank_pairs(&records, LabelBudget::B100, 8, 42);
        assert!(!pairs.is_empty());
        for p in &pairs {
            let (better_idx, worse_idx) = if records[p.a].trajectory_id == 0 {
                (p.a, p.b)
            } else {
                (p.b, p.a)
            };
            // sign is defined as sign(R(other) - R(this pair's `a`)); check
            // the invariant algebraically instead of hard-coding which slot
            // ends up `a` vs `b`.
            let ret_a = records[p.a].return_b100.unwrap();
            let ret_b = records[p.b].return_b100.unwrap();
            assert_eq!(p.sign, (ret_b - ret_a).signum());
            let _ = (better_idx, worse_idx);
        }
    }

    #[test]
    fn sample_rank_pairs_emits_nothing_for_a_single_trajectory_expression() {
        let records = vec![
            record("e0", 0, 0.5, Some(0.1)),
            record("e0", 0, 0.5, Some(0.1)), // same trajectory_id
        ];
        let pairs = sample_rank_pairs(&records, LabelBudget::B100, 8, 1);
        assert!(pairs.is_empty());
    }

    #[test]
    fn sample_rank_pairs_skips_zero_difference_pairs() {
        let records = vec![
            record("e0", 0, 0.5, Some(0.3)),
            record("e0", 1, 0.5, Some(0.3)), // identical return: no signal
        ];
        let pairs = sample_rank_pairs(&records, LabelBudget::B100, 8, 1);
        assert!(pairs.is_empty());
    }

    #[test]
    fn sample_rank_pairs_is_deterministic_given_the_same_seed() {
        let records = vec![
            record("e0", 0, 0.5, Some(0.1)),
            record("e0", 1, 0.5, Some(0.4)),
            record("e0", 2, 0.5, Some(0.7)),
        ];
        let a = sample_rank_pairs(&records, LabelBudget::B100, 4, 99);
        let b = sample_rank_pairs(&records, LabelBudget::B100, 4, 99);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.a, y.a);
            assert_eq!(x.b, y.b);
            assert_eq!(x.sign, y.sign);
        }
    }
}

// ============================================================================
// Trajectory minter support (`bin/gen_r2g_trajectories.rs`) — §2's ordering
// diversity, and the per-expression spread statistic/dataset gate it exists
// to produce. Added alongside, not inside, the schema/label-arithmetic
// section above: this half has no dependency on `R2gRecord`'s consumer
// (`train_guide_r2g`) and none of the above depends on it, so the two halves
// of this module stay independently reviewable.
// ============================================================================

/// Which deterministic ordering rule drives one trajectory
/// (`docs/archive/plans/2026-09-01-guide-return-to-go.md` §2's diversity table:
/// unguided / per-rule / a frozen linear checkpoint / a fixed pseudo-random
/// preference / a rank-mix of a base policy with random noise).
///
/// Purely a labeling/recipe type. Building the actual
/// `pixelflow_search::nnue::guide::SaturationGuide` this recipe names (or,
/// for `Unguided`, skipping a Guide entirely and calling
/// `EGraph::saturate_until_applications` directly, so the unguided
/// trajectory is byte-for-byte the production path — see
/// `bin/gen_r2g_trajectories.rs`'s equivalence test against
/// `egraph::run_anytime_curve`'s cost at B) is the minter binary's job: that
/// is where the NNUE-guide-shaped dependency belongs, and every other type
/// in this module stays free of it.
#[derive(Clone, Debug)]
pub enum OrderingPolicy {
    /// The rule-major sweep, no dedup: `EGraph::saturate_until_applications`
    /// verbatim, unmodified.
    Unguided,
    /// `PerRuleRateGuide` — the zero-candidate-local-information control.
    PerRuleRate,
    /// `LinearCandidateGuide` over a frozen checkpoint. The `String` is a
    /// human-readable checkpoint label (`"strict-v1"` for the Round-1
    /// checkpoint), not a filesystem path — the minter loads the actual
    /// checkpoint once from a CLI flag and threads the loaded guide through
    /// separately; this variant only carries what the checkpoint should be
    /// CALLED in `policy`/`R2gRecord::policy` output.
    FrozenLinear(String),
    /// A fixed pseudo-random ordering
    /// (`pixelflow_search::nnue::guide::diversity::UniformRandomGuide`).
    Random(u64),
    /// A rank-mix of a base policy's score with a uniform-random term:
    /// `score = (1 − eps)·rank_norm(base_score) + eps·uniform(seed)`,
    /// `eps = numerator/denominator`, matching
    /// `docs/archive/plans/2026-09-01-guide-return-to-go.md` §2's `RankMixGuide`
    /// row (mixing on ranks, not raw logits, so `eps` means the same thing
    /// regardless of the base policy's score scale). `guide` names the base
    /// being perturbed — expected to be a non-`Random`, non-`EpsilonMix`
    /// named policy (`PerRuleRate`/`FrozenLinear`); the minter panics loudly
    /// on a nested `EpsilonMix` rather than silently defining what that
    /// would mean.
    EpsilonMix {
        guide: Box<OrderingPolicy>,
        numerator: u16,
        denominator: u16,
        seed: u64,
    },
}

impl OrderingPolicy {
    /// The `"policy"` string minted into every `R2gRecord`/`TrajectoryRow`
    /// this trajectory produces — matches §2's table naming convention:
    /// `"unguided"`, `"per-rule"`, `"strict-v1"`, `"random:<seed>"`,
    /// `"mix:strict-v1:<n>/<d>:<seed>"`.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            OrderingPolicy::Unguided => "unguided".to_string(),
            OrderingPolicy::PerRuleRate => "per-rule".to_string(),
            OrderingPolicy::FrozenLinear(name) => name.clone(),
            OrderingPolicy::Random(seed) => format!("random:{seed}"),
            OrderingPolicy::EpsilonMix {
                guide,
                numerator,
                denominator,
                seed,
            } => format!("mix:{}:{numerator}/{denominator}:{seed}", guide.label()),
        }
    }
}

/// The extended budget ladder for the round-3 "measure spread where the
/// budget binds" task (`docs/archive/plans/2026-09-01-guide-return-to-go.md` §2b):
/// the two originally-registered tiers (B=100/200, unchanged — still what
/// `train_guide_r2g` reads via [`TrajectoryRow::app_actual_b100`] etc.) plus
/// four higher tiers reaching into the regime where production's classical
/// kernels actually run (median ~1,671 applications, PR #1087). `100` and
/// `200` MUST remain the first two entries — `gen_r2g_trajectories`'
/// `mint_expression` reads `checkpoints[0]`/`checkpoints[1]` positionally to
/// populate `TrajectoryRow`'s original `*_b100`/`*_b200` fields (its own
/// `parse_budgets` asserts this at startup), so every existing consumer of
/// those two fields keeps working unchanged.
pub const BUDGET_LADDER: [usize; 6] = [100, 200, 400, 800, 1600, 3200];

/// One trajectory's cost/return at one point on [`BUDGET_LADDER`], plus the
/// TYPED stop reason the underlying `SaturationStop` reported at that
/// checkpoint (`Debug`-formatted — `pixelflow_search::egraph::SaturationStop`
/// is not itself `serde`-derived, and this module must not depend on
/// `pixelflow-search` for a label string) — the round-3 task's "read the
/// typed stop if the branch has it" requirement, not an inferred guess.
#[derive(Clone, Debug, serde::Serialize, Deserialize)]
pub struct CheckpointRow {
    pub budget: usize,
    /// Cumulative recorded applications at this checkpoint (may overshoot
    /// `budget` — see `EGraph::saturate_until_applications`'s doc).
    pub app_actual: u64,
    pub cost: u64,
    /// `format!("{:?}", SaturationStop)` — `"Quiesced"`, `"ApplicationBudget"`,
    /// `"MintedCap"`, `"IterationCeiling"`, or `"Timeout"`.
    pub stop: String,
    /// `R(τ, budget)` (§1.2) — `None` when `expr_best_cost == 0` for this
    /// expression (the registration's zero-best exclusion convention).
    pub return_val: Option<f32>,
}

/// One trajectory's summary row — one per `(expression, trajectory)`, not
/// per application. Written to `r2g_trajectories_*.jsonl` alongside the
/// (much larger) per-application `r2g_*.jsonl` records, so
/// [`spread_report`] and the mint's per-policy median-regret statistic can
/// be computed without re-scanning the per-application file.
#[derive(Clone, Debug, serde::Serialize, Deserialize)]
pub struct TrajectoryRow {
    pub expr_name: String,
    /// `"train"` / `"dev"` / `"sh"` / `"bezier"` — which mint split this
    /// trajectory belongs to (the four separate output files' own names).
    pub tier: String,
    pub trajectory_id: u32,
    pub policy: String,
    /// Arena node count of the source expression — carried on the
    /// trajectory row (not just the per-application file) so the round-3
    /// spread-vs-budget report can bucket by node-count band without
    /// rejoining against the corpus. `0` on rows minted before this field
    /// existed (`#[serde(default)]`).
    #[serde(default)]
    pub expr_node_count: usize,
    /// Cumulative recorded applications at the B=100 checkpoint (may
    /// overshoot 100 for the unguided sweep, which checks the budget
    /// between rules — see `EGraph::saturate_until_applications`'s doc).
    pub app_actual_b100: u64,
    pub cost_b100: u64,
    pub app_actual_b200: u64,
    pub cost_b200: u64,
    /// `true` if this trajectory's saturation had already ended (quiesced /
    /// class-capped / iteration-ceilinged) at or before the ceiling
    /// checkpoint the mint ran it to (§1.2: "the unguided trajectory
    /// contributes its full grid through quiescence/cap").
    pub ended: bool,
    pub ended_at_apps: u64,
    /// `R(τ, 100)` (§1.2) — `None` when `expr_best_cost == 0` for this
    /// expression (the registration's zero-best exclusion convention).
    pub return_b100: Option<f32>,
    pub return_b200: Option<f32>,
    /// One entry per [`BUDGET_LADDER`] tier (superset of `b100`/`b200`
    /// above, which stay for backward compatibility with existing
    /// consumers). Empty on rows minted before round 3
    /// (`#[serde(default)]`).
    #[serde(default)]
    pub checkpoints: Vec<CheckpointRow>,
}

/// Per-expression return-spread statistics — §2's dataset statistic and
/// gate ("if more than 50% of TRAIN classical expressions have
/// `spread_e(100) = 0` ... record it, do not train").
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SpreadReport {
    /// Expressions with at least one trajectory carrying a `return_b100`
    /// label (i.e. `expr_best_cost != 0` for that expression).
    pub expressions: usize,
    pub zero_spread_b100: usize,
    pub zero_spread_b200: usize,
    /// (Q1, median, Q3) of `spread_e(100) = max R(τ,100) − min R(τ,100)`
    /// over trajectories of the same expression.
    pub spread_b100_quartiles: (f32, f32, f32),
    pub spread_b200_quartiles: (f32, f32, f32),
    /// Share of *records* (trajectories, here — one row per trajectory)
    /// belonging to a zero-spread-at-B100 expression — §2: "the share of
    /// records from such expressions," reported alongside the
    /// expression-count share so a large zero-spread expression (many
    /// trajectories, e.g. one that quiesces identically under every policy)
    /// cannot look small just because expression *counts* are unweighted.
    pub zero_spread_b100_record_share: f32,
    /// `true` when more than half of `expressions` have `spread_e(100) = 0`
    /// — the pre-registered dataset gate. The mint does not stop itself on
    /// this (minting the report status IS the deliverable per §2); a
    /// caller decides what to do with it.
    pub dataset_gate_fired: bool,
}

/// (Q1, median, Q3) via linear interpolation — the same convention
/// `bin/gen_strict_labels.rs`'s own `quantiles()` uses, kept in step
/// deliberately (comparable percentile semantics across this program's
/// reports), just over `f32` here since spread is already `f32`.
fn quantiles_f32(mut xs: Vec<f32>) -> (f32, f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("quantiles_f32: NaN spread value"));
    let q = |p: f32| -> f32 {
        let n = xs.len();
        if n == 1 {
            return xs[0];
        }
        let pos = p * (n as f32 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            xs[lo]
        } else {
            let frac = pos - lo as f32;
            xs[lo] * (1.0 - frac) + xs[hi] * frac
        }
    };
    (q(0.25), q(0.5), q(0.75))
}

/// §2's dataset-gate statistic: per-expression return spread at each
/// registered tier, computed from one split's [`TrajectoryRow`]s (grouped
/// by `expr_name`).
#[must_use]
pub fn spread_report(rows: &[TrajectoryRow]) -> SpreadReport {
    use std::collections::BTreeMap;

    let mut by_expr: BTreeMap<&str, Vec<&TrajectoryRow>> = BTreeMap::new();
    for r in rows {
        by_expr.entry(r.expr_name.as_str()).or_default().push(r);
    }

    let mut spreads_100 = Vec::new();
    let mut spreads_200 = Vec::new();
    let mut zero_100 = 0usize;
    let mut zero_200 = 0usize;
    let mut zero_100_records = 0usize;
    let mut total_records_with_b100 = 0usize;
    let mut expressions_with_b100 = 0usize;

    for group in by_expr.values() {
        let r100: Vec<f32> = group.iter().filter_map(|r| r.return_b100).collect();
        let r200: Vec<f32> = group.iter().filter_map(|r| r.return_b200).collect();
        if !r100.is_empty() {
            expressions_with_b100 += 1;
            total_records_with_b100 += group.len();
            let lo = r100.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = r100.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let spread = hi - lo;
            spreads_100.push(spread);
            if spread <= 0.0 {
                zero_100 += 1;
                zero_100_records += group.len();
            }
        }
        if !r200.is_empty() {
            let lo = r200.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = r200.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let spread = hi - lo;
            spreads_200.push(spread);
            if spread <= 0.0 {
                zero_200 += 1;
            }
        }
    }

    let dataset_gate_fired = expressions_with_b100 > 0 && zero_100 * 2 > expressions_with_b100;

    SpreadReport {
        expressions: expressions_with_b100,
        zero_spread_b100: zero_100,
        zero_spread_b200: zero_200,
        spread_b100_quartiles: quantiles_f32(spreads_100),
        spread_b200_quartiles: quantiles_f32(spreads_200),
        zero_spread_b100_record_share: if total_records_with_b100 == 0 {
            0.0
        } else {
            zero_100_records as f32 / total_records_with_b100 as f32
        },
        dataset_gate_fired,
    }
}

#[cfg(test)]
mod minter_tests {
    use super::*;

    fn row(
        expr_name: &str,
        trajectory_id: u32,
        return_b100: Option<f32>,
        return_b200: Option<f32>,
    ) -> TrajectoryRow {
        TrajectoryRow {
            expr_name: expr_name.to_string(),
            tier: "train".to_string(),
            trajectory_id,
            policy: "unguided".to_string(),
            expr_node_count: 10,
            app_actual_b100: 100,
            cost_b100: 50,
            app_actual_b200: 200,
            cost_b200: 40,
            ended: false,
            ended_at_apps: 200,
            return_b100,
            return_b200,
            checkpoints: Vec::new(),
        }
    }

    // ── OrderingPolicy::label ──────────────────────────────────────────

    #[test]
    fn label_matches_the_section_2_naming_convention() {
        assert_eq!(OrderingPolicy::Unguided.label(), "unguided");
        assert_eq!(OrderingPolicy::PerRuleRate.label(), "per-rule");
        assert_eq!(
            OrderingPolicy::FrozenLinear("strict-v1".to_string()).label(),
            "strict-v1"
        );
        assert_eq!(OrderingPolicy::Random(7).label(), "random:7");
        assert_eq!(
            OrderingPolicy::EpsilonMix {
                guide: Box::new(OrderingPolicy::FrozenLinear("strict-v1".to_string())),
                numerator: 1,
                denominator: 4,
                seed: 3,
            }
            .label(),
            "mix:strict-v1:1/4:3"
        );
    }

    // ── spread_report ───────────────────────────────────────────────────

    #[test]
    fn spread_report_computes_max_minus_min_per_expression() {
        let rows = vec![
            row("e0", 0, Some(0.1), Some(0.05)),
            row("e0", 1, Some(0.9), Some(0.05)),
            row("e1", 0, Some(0.3), None),
        ];
        let report = spread_report(&rows);
        assert_eq!(report.expressions, 2);
        // e0's spread is 0.8, e1's is 0.0 (only one trajectory) -> median
        // of [0.0, 0.8] with linear interpolation at n=2 is 0.4.
        assert!((report.spread_b100_quartiles.1 - 0.4).abs() < 1e-5);
    }

    #[test]
    fn spread_report_counts_zero_spread_expressions_and_their_record_share() {
        let rows = vec![
            row("e0", 0, Some(0.2), None), // single trajectory -> spread 0
            row("e1", 0, Some(0.1), None),
            row("e1", 1, Some(0.1), None), // identical -> spread 0
            row("e2", 0, Some(0.1), None),
            row("e2", 1, Some(0.9), None), // real spread
        ];
        let report = spread_report(&rows);
        assert_eq!(report.expressions, 3);
        assert_eq!(report.zero_spread_b100, 2, "e0 and e1 are zero-spread");
        // zero-spread records: e0 (1) + e1 (2) = 3, out of 5 total.
        assert!((report.zero_spread_b100_record_share - 0.6).abs() < 1e-5);
    }

    #[test]
    fn spread_report_gate_fires_past_half_zero_spread() {
        let rows = vec![
            row("e0", 0, Some(0.1), None),
            row("e1", 0, Some(0.1), None),
            row("e2", 0, Some(0.1), None),
            row("e2", 1, Some(0.9), None),
        ];
        let report = spread_report(&rows);
        assert_eq!(report.expressions, 3);
        assert_eq!(report.zero_spread_b100, 2);
        assert!(
            report.dataset_gate_fired,
            "2 of 3 expressions (>50%) are zero-spread — the gate must fire"
        );
    }

    #[test]
    fn spread_report_gate_does_not_fire_at_exactly_half() {
        let rows = vec![
            row("e0", 0, Some(0.1), None), // zero-spread
            row("e1", 0, Some(0.1), None),
            row("e1", 1, Some(0.9), None), // real spread
        ];
        let report = spread_report(&rows);
        assert_eq!(report.zero_spread_b100, 1);
        assert_eq!(report.expressions, 2);
        assert!(
            !report.dataset_gate_fired,
            "exactly 50% zero-spread must not fire a \"more than 50%\" gate"
        );
    }

    #[test]
    fn spread_report_ignores_records_with_no_b100_return() {
        let rows = vec![row("e0", 0, None, None)];
        let report = spread_report(&rows);
        assert_eq!(report.expressions, 0);
        assert_eq!(report.spread_b100_quartiles, (0.0, 0.0, 0.0));
    }
}
// ── Checkpoint ───────────────────────────────────────────────────────────

/// The return-to-go head's checkpoint — same weight layout
/// `pixelflow_search::nnue::guide::linear::LinearWeights` deploys, plus
/// mint-style provenance (`docs/archive/plans/2026-09-01-guide-return-to-go.md`
/// §3.3): which objective/target/tier trained it, which ordering policies
/// contributed TRAIN records, and a content hash of each split's source
/// file (see [`LoadedSplit::source_fnv64`]'s doc for why that stands in for
/// a "corpus MD5"). `objective` is the field
/// `pixelflow_search::nnue::guide::linear::LinearReturnGuide::load` gates
/// on, and its absence is what `LinearCandidateGuide::load` gates on in the
/// other direction — the two loaders' cross-loading refusal (§3.3) depends
/// on this field always being present here and never present in
/// `train_guide`'s checkpoint.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct R2gCheckpoint {
    pub schema_identity: String,
    /// Fixed: the label *family* this checkpoint was trained against
    /// (`docs/archive/plans/2026-09-01-guide-return-to-go.md`), mirroring
    /// `train_guide`'s `label_source` field. Distinct from `regression_target`
    /// (which of the two return-to-go columns, centered or raw) and from
    /// `objective` (which loss trained it) — three separate questions this
    /// module keeps separately named rather than conflating.
    pub label_family: String,
    /// `"centered"` or `"raw"` — which return-to-go column trained this
    /// checkpoint (§3.4's ablation vs §3.2's primary).
    pub regression_target: String,
    /// `100` or `200` — which registered tier's label trained this
    /// checkpoint.
    pub label_b: u32,
    /// `"return-mse"` (§3.2 primary) or `"return-rank"` (§3.4 ablation) —
    /// the field `LinearReturnGuide::load`/`LinearCandidateGuide::load`
    /// gate on.
    pub objective: String,
    pub trainer: String,
    pub written_at_unix_s: u64,

    pub seed: u64,
    pub epochs: usize,
    pub lr_initial: f32,
    pub lr_decay: f32,
    pub l2: f32,
    pub grad_clip: f32,
    pub pairs_per_record: usize,

    pub num_rules: usize,
    pub num_ops: usize,
    /// `w_rule` slot -> canonical rule label — the identity half of
    /// `w_rule`, exactly as `GuideCheckpoint`'s is.
    pub rule_names: Vec<String>,
    /// `RuleSet::fingerprint()` of the vocabulary trained against, as 16 hex
    /// digits. A loader refuses weights whose fingerprint is not the live
    /// rule set's.
    pub rule_fingerprint: String,
    pub op_names: Vec<String>,

    pub bias: f32,
    pub w_rule: Vec<f32>,
    pub w_op: Vec<f32>,
    pub w_budget: f32,
    pub w_match_class: f32,
    pub w_neighborhood: f32,
    pub w_expr_size: f32,

    pub train_records: usize,
    pub train_labelled_samples: usize,
    /// Distinct `Policy` names observed in TRAIN (§2) — the "policies used"
    /// provenance line item.
    pub train_policies: Vec<String>,
    /// Content hash of the TRAIN JSONL's own bytes — see
    /// [`LoadedSplit::source_fnv64`].
    pub train_source_fnv64: String,
    pub dev_records: usize,
    pub dev_labelled_samples: usize,
    pub dev_policies: Vec<String>,
    pub dev_source_fnv64: String,
    pub dev_mse: f64,
    pub dev_spearman: Option<f64>,

    pub weights_fnv64: String,
}

impl crate::schema::SchemaIdentity for R2gCheckpoint {
    const MAGIC: &'static str = "PXGR";
    const SCHEMA: &'static str = "\
        label_family: fixed \"return-to-go\" — the label family this checkpoint was trained \
        against (docs/archive/plans/2026-09-01-guide-return-to-go.md); \
        regression_target: \"centered\" (expression-centered log-regret, §3.2 primary) or \
        \"raw\" (uncentered, §3.4 ablation) — which return-to-go column trained this \
        checkpoint; \
        label_b: 100 or 200 — which registered budget tier's label trained this checkpoint; \
        objective: \"return-mse\" or \"return-rank\" — the loss the trainer minimized; the \
        field LinearReturnGuide::load requires and LinearCandidateGuide::load refuses; \
        trainer: which binary wrote these weights; \
        seed/epochs/lr_initial/lr_decay/l2/grad_clip/pairs_per_record: the SGD run's \
        hyperparameters (pairs_per_record only consumed under objective=return-rank); \
        num_rules/num_ops: dense-array lengths for rule_names/w_rule and op_names/w_op; \
        rule_names: w_rule slot -> canonical rule label (rule_label spelling); \
        rule_fingerprint: RuleSet::fingerprint() of the vocabulary trained against, \
        16 hex digits, refused on mismatch; \
        op_names: OpKind::all() order, the index space w_op is keyed by; \
        bias/w_rule/w_op/w_budget/w_match_class/w_neighborhood/w_expr_size: the linear \
        model's weights — logit = bias + w_rule[slot of rule_label] + sum_op(hist[op]*w_op[op]) + \
        w_budget*budget_fraction + w_match_class*log1p(match_class_node_count) + \
        w_neighborhood*log1p(neighborhood_op_count) + w_expr_size*log1p(expr_node_count) — \
        the exact formula pixelflow_search::nnue::guide::linear::LinearWeights::logit \
        reproduces; \
        train_records/train_labelled_samples/train_policies/train_source_fnv64: TRAIN-split \
        provenance (train_labelled_samples is the subset of train_records this run's \
        (regression_target, label_b) actually trained on; train_source_fnv64 is a content \
        hash of the TRAIN JSONL's own bytes, standing in for a corpus checksum); \
        dev_records/dev_labelled_samples/dev_policies/dev_source_fnv64/dev_mse/dev_spearman: \
        held-out DEV-split provenance and evaluation recorded at write time; \
        weights_fnv64: FNV-1a 64 hex content hash binding this file's metadata to its own \
        weight values";
}

impl R2gCheckpoint {
    #[must_use]
    pub fn current_schema_identity() -> String {
        format!(
            "{:016x}",
            <Self as crate::schema::SchemaIdentity>::SCHEMA_IDENTITY
        )
    }

    #[must_use]
    pub fn weights_fingerprint(&self) -> String {
        let mut buf = String::new();
        buf.push_str(&format!("{:.9}\n", self.bias));
        for w in &self.w_rule {
            buf.push_str(&format!("{w:.9}\n"));
        }
        for w in &self.w_op {
            buf.push_str(&format!("{w:.9}\n"));
        }
        buf.push_str(&format!(
            "{:.9}\n{:.9}\n{:.9}\n{:.9}\n",
            self.w_budget, self.w_match_class, self.w_neighborhood, self.w_expr_size
        ));
        crate::schema::fnv1a64_hex(buf.as_bytes())
    }

    pub fn write(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        self.schema_identity = Self::current_schema_identity();
        self.weights_fnv64 = self.weights_fingerprint();
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    pub fn read(path: &std::path::Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let meta: Self = serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let expected = Self::current_schema_identity();
        if meta.schema_identity != expected {
            let stored = u64::from_str_radix(&meta.schema_identity, 16).unwrap_or(0);
            return Err(crate::schema::identity_mismatch(
                "r2g guide checkpoint",
                stored,
                <Self as crate::schema::SchemaIdentity>::SCHEMA_IDENTITY,
                "cargo run --release -p pixelflow-pipeline --features training --bin \
                 train_guide_r2g",
            ));
        }
        let actual = meta.weights_fingerprint();
        if meta.weights_fnv64 != actual {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "r2g guide checkpoint {}: metadata/weights mismatch: recorded hash {}, \
                     actual {}",
                    path.display(),
                    meta.weights_fnv64,
                    actual
                ),
            ));
        }
        Ok(meta)
    }
}

impl R2gCheckpoint {
    /// Which loss trained these weights, as a type.
    ///
    /// # Errors
    ///
    /// If `objective` is neither `"return-mse"` nor `"return-rank"` — an
    /// unrecognized objective is refused rather than loaded as if it were
    /// one of the two.
    pub fn objective_typed(&self) -> Result<ReturnObjective, String> {
        match self.objective.as_str() {
            "return-mse" => Ok(ReturnObjective::Mse),
            "return-rank" => Ok(ReturnObjective::Rank),
            other => Err(format!(
                "r2g checkpoint: objective {other:?} is neither \"return-mse\" nor \
                 \"return-rank\" — refusing to load an unrecognized objective as a return head"
            )),
        }
    }

    /// The deployed-side weights: `w_rule` re-keyed by [`RuleId`] and `w_op`
    /// by op name, with the trained-against vocabulary's fingerprint
    /// attached so [`LinearReturnGuide::new`] can refuse a mismatch.
    ///
    /// # Panics
    ///
    /// If `rule_fingerprint` is not 16 hex digits, or the dense arrays
    /// disagree with their name tables.
    #[must_use]
    pub fn to_weights(&self) -> LinearWeights {
        assert_eq!(
            self.rule_names.len(),
            self.w_rule.len(),
            "r2g checkpoint: rule_names and w_rule disagree in length"
        );
        assert_eq!(
            self.op_names.len(),
            self.w_op.len(),
            "r2g checkpoint: op_names and w_op disagree in length"
        );
        let digest = u64::from_str_radix(&self.rule_fingerprint, 16).unwrap_or_else(|e| {
            panic!(
                "r2g checkpoint: rule_fingerprint {:?} is not 16 hex digits ({e}) — a \
                 checkpoint without a readable vocabulary fingerprint cannot be deployed",
                self.rule_fingerprint
            )
        });
        LinearWeights {
            bias: self.bias,
            w_rule: self
                .rule_names
                .iter()
                .zip(&self.w_rule)
                .map(|(label, &w)| (RuleId::from_label(label), w))
                .collect(),
            w_op: self
                .op_names
                .iter()
                .cloned()
                .zip(self.w_op.iter().copied())
                .collect(),
            w_budget: self.w_budget,
            w_match_class: self.w_match_class,
            w_neighborhood: self.w_neighborhood,
            w_expr_size: self.w_expr_size,
            fingerprint: Fingerprint::from_raw(digest),
        }
    }

    /// The trainer-side model these weights came from — the side of the skew
    /// test that runs `Model::logit`.
    ///
    /// # Panics
    ///
    /// If the dense arrays disagree with their name tables.
    #[must_use]
    pub fn to_model(&self) -> Model {
        assert_eq!(
            self.rule_names.len(),
            self.w_rule.len(),
            "r2g checkpoint: rule_names and w_rule disagree in length"
        );
        assert_eq!(
            self.op_names.len(),
            self.w_op.len(),
            "r2g checkpoint: op_names and w_op disagree in length"
        );
        Model {
            bias: self.bias,
            w_rule: self.w_rule.clone(),
            w_op: self.w_op.clone(),
            w_budget: self.w_budget,
            w_match_class: self.w_match_class,
            w_neighborhood: self.w_neighborhood,
            w_expr_size: self.w_expr_size,
        }
    }
}

/// Read a return-to-go checkpoint and deploy it against `rules`.
///
/// The cross-loading refusal (§3.3) lives here, where the file is parsed: the
/// two checkpoint shapes carry different `SchemaIdentity` magics, so a
/// strict-bit checkpoint is rejected by [`R2gCheckpoint::read`] before this
/// is reached, and a return checkpoint is rejected by
/// [`crate::training::guide_linear::GuideCheckpoint::read`] in the other
/// direction. Neither can be loaded as the other and silently invert a move
/// ordering.
///
/// # Errors
///
/// I/O or validation failure on the file, an unrecognized objective, or a
/// rule-set fingerprint that is not the live one.
pub fn load_return_guide(
    path: &std::path::Path,
    rules: &RuleSet,
) -> Result<LinearReturnGuide, String> {
    let ck =
        R2gCheckpoint::read(path).map_err(|e| format!("r2g checkpoint {}: {e}", path.display()))?;
    let objective = ck.objective_typed()?;
    LinearReturnGuide::new(ck.to_weights(), objective, rules)
        .map_err(|e| format!("r2g checkpoint {}: {e}", path.display()))
}

// ── R2G checkpoint round trip ───────────────────────────────────────────────

#[cfg(test)]
mod r2g_checkpoint_tests {
    use super::*;

    fn ckpt_for(rules: &RuleSet) -> R2gCheckpoint {
        let (rule_names, _) = crate::training::guide_linear::rule_index_table(rules);
        let (op_names, _) = crate::training::guide_linear::op_index_table();
        R2gCheckpoint {
            schema_identity: String::new(),
            label_family: "return-to-go".to_string(),
            regression_target: "centered".to_string(),
            label_b: 100,
            objective: "return-mse".to_string(),
            trainer: "train_guide_r2g".to_string(),
            written_at_unix_s: 1,
            seed: 1,
            epochs: 1,
            lr_initial: 0.01,
            lr_decay: 0.0,
            l2: 0.0,
            grad_clip: 10.0,
            pairs_per_record: 8,
            num_rules: rule_names.len(),
            num_ops: op_names.len(),
            w_rule: vec![0.25; rule_names.len()],
            w_op: vec![-0.125; op_names.len()],
            rule_names,
            rule_fingerprint: format!("{}", rules.fingerprint()),
            op_names,
            bias: 0.1,
            w_budget: 0.5,
            w_match_class: 0.6,
            w_neighborhood: 0.7,
            w_expr_size: 0.8,
            train_records: 10,
            train_labelled_samples: 8,
            train_policies: vec!["unguided".to_string()],
            train_source_fnv64: "deadbeef".to_string(),
            dev_records: 5,
            dev_labelled_samples: 5,
            dev_policies: vec!["unguided".to_string()],
            dev_source_fnv64: "cafef00d".to_string(),
            dev_mse: 0.05,
            dev_spearman: Some(0.8),
            weights_fnv64: String::new(),
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("r2g_ckpt_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn a_written_checkpoint_round_trips_and_deploys_as_a_return_head() {
        let dir = scratch("roundtrip");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        ckpt.write(&path).expect("write");

        let loaded = R2gCheckpoint::read(&path).expect("read");
        assert_eq!(loaded.objective, "return-mse");
        assert_eq!(loaded.label_family, "return-to-go");
        let guide = load_return_guide(&path, &rules).expect("deploys");
        assert_eq!(guide.objective(), ReturnObjective::Mse);

        // The raw JSON carries the top-level "objective" key — the field
        // that tells a return checkpoint from a strict-bit one.
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read back"))
                .expect("valid json");
        assert_eq!(
            raw.get("objective").and_then(serde_json::Value::as_str),
            Some("return-mse")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unrecognized_objective_is_refused_rather_than_deployed() {
        let dir = scratch("objective");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        ckpt.objective = "return-something-else".to_string();
        ckpt.write(&path).expect("write");

        let err = load_return_guide(&path, &rules)
            .expect_err("an unrecognized objective must not deploy as a return head");
        assert!(err.contains("neither"), "got: {err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The G5 property for the return head, same as the strict head's.
    #[test]
    fn a_checkpoint_from_another_vocabulary_is_refused_at_deploy_time() {
        let dir = scratch("wrong_vocab");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let mut ckpt = ckpt_for(&rules);
        ckpt.rule_fingerprint = {
            let mut reversed = pixelflow_search::egraph::all_rules();
            reversed.reverse();
            format!("{}", RuleSet::new(reversed).fingerprint())
        };
        ckpt.write(&path).expect("write");

        let err = load_return_guide(&path, &rules)
            .expect_err("weights from a reordered vocabulary must be refused");
        assert!(
            err.contains("vocabulary changed") || err.contains("trained against"),
            "got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A strict-bit checkpoint and a return checkpoint carry different
    /// `SchemaIdentity` magics, so neither loader can be handed the other's
    /// file and silently invert a move ordering (§3.3).
    #[test]
    fn a_strict_bit_checkpoint_does_not_load_as_a_return_head() {
        let dir = scratch("crossload");
        let path = dir.join("checkpoint.json");
        let rules = RuleSet::production();
        let (rule_names, _) = crate::training::guide_linear::rule_index_table(&rules);
        let (op_names, _) = crate::training::guide_linear::op_index_table();
        let mut strict = crate::training::guide_linear::GuideCheckpoint {
            schema_identity: String::new(),
            label_source: "strict-v1".to_string(),
            trainer: "train_guide".to_string(),
            written_at_unix_s: 1,
            seed: 1,
            epochs: 1,
            lr_initial: 0.01,
            lr_decay: 0.0,
            l2: 0.0,
            grad_clip: 20.0,
            pos_weight: 1.0,
            num_rules: rule_names.len(),
            num_ops: op_names.len(),
            w_rule: vec![0.25; rule_names.len()],
            w_op: vec![-0.125; op_names.len()],
            rule_names,
            rule_fingerprint: format!("{}", rules.fingerprint()),
            op_names,
            bias: 0.5,
            w_budget: 0.125,
            w_match_class: 0.0625,
            w_neighborhood: -0.031_25,
            w_expr_size: 0.015_625,
            train_samples: 1,
            train_families: 1,
            train_positive_rate: 0.0,
            dev_samples: 1,
            dev_families: 1,
            dev_auc: 0.5,
            dev_pr_auc: 0.5,
            weights_fnv64: String::new(),
        };
        strict.write(&path).expect("write");

        let err = load_return_guide(&path, &rules)
            .expect_err("a strict-bit checkpoint must not deploy as a return head");
        // It fails on the shape, before any weight is read: a strict-bit
        // checkpoint has no `label_family`/`objective` at all.
        assert!(
            err.contains("label_family") || err.contains("identity"),
            "got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
