//! Rule identity: what names a rewrite rule when its position moves.
//!
//! A `Vec<Box<dyn Rewrite>>` gives every rule an index, and an index is the
//! obvious thing to key a per-rule table on — a trained policy's weights, a
//! label table, a JSON report. It is also not an identity. `all_rules()`
//! returns a vector; inserting a rule renumbers everything after it and
//! reordering renumbers everything, and neither change touches the tables
//! that were keyed by the old numbering. The reorder is the dangerous one:
//! a same-length permutation repoints every weight in a checkpoint and
//! nothing anywhere is the wrong length, so nothing complains.
//!
//! [`RuleId`] is the fix, and it is the same fix the NNUE checkpoint format
//! already applies to operators (`op_names` → `op_index`, with a loud
//! disagreement check): key by a name, not by a slot. Rules are the one axis
//! that never got that treatment.
//!
//! The name has to *discriminate*, which family names do not —
//! [`Rewrite::name`] answers `"commutative"` for all four `Commutative`
//! instances. [`Rewrite::specialization`] supplies the other half, and
//! [`rule_label`] is the pair rendered as the canonical, stable string that
//! [`RuleId`] hashes and that reports should print.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use super::rewrite::Rewrite;

/// FNV-1a over bytes — a *stable* hash, unlike `DefaultHasher`, whose output
/// is explicitly not guaranteed across releases. A [`RuleId`] outlives the
/// process that minted it (it is written into checkpoints and reports), so
/// the digest has to be reproducible by a different build.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// The canonical, stable name of one rule *instance*.
///
/// `"commutative(Mul)"` for the `Mul` instance of the `Commutative` family;
/// the bare family name (`"fma-fusion"`) for a family with one instance.
/// This is the string a report should print and the string [`RuleId`]
/// hashes — the two must not drift, so nothing else may render a rule.
#[must_use]
pub fn rule_label(rule: &dyn Rewrite) -> String {
    match rule.specialization() {
        Some(op) => format!("{}({:?})", rule.name(), op),
        None => String::from(rule.name()),
    }
}

/// A stable identity for a rewrite rule, independent of its position in any
/// rule vector.
///
/// Derived from [`rule_label`], so it survives both reordering and
/// insertion. Two builds of the same rule agree; a rule that is renamed does
/// not, which is the correct answer — a renamed rule is a different rule as
/// far as a checkpoint trained against the old one is concerned, and the
/// mismatch should be loud.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleId(u64);

impl RuleId {
    /// The id of `rule`, from its [`rule_label`].
    #[must_use]
    pub fn of(rule: &dyn Rewrite) -> Self {
        Self::from_label(&rule_label(rule))
    }

    /// The id a given canonical label hashes to — the entry point for a
    /// loader reading rule keys back out of a checkpoint or a JSON report.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        Self(fnv1a(label.as_bytes()))
    }

    /// The raw digest, for serialization.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RuleId({:016x})", self.0)
    }
}

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// A digest over a whole optimizer configuration — today, over a
/// [`RuleSet`]'s ordered ids.
///
/// Order is part of it, not just content: saturation under a budget is
/// order-sensitive, so the same rules in a different sequence can extract a
/// different (equally valid, differently priced) term. A cache keyed on the
/// input expression alone would then serve one configuration's code to
/// another.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(u64);

impl Fingerprint {
    /// The raw digest.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }

    /// Reconstruct a fingerprint from a digest read back off disk.
    ///
    /// The inverse of [`Self::get`], and the only way a checkpoint loader
    /// living outside this crate (JSON parsing belongs in
    /// `pixelflow-pipeline`) can say "these weights were trained against
    /// vocabulary X" without inventing a second identity type. It asserts
    /// nothing: a digest that names no rule set this build has is exactly
    /// what [`LinearCandidateGuide::new`](crate::nnue::guide::linear::LinearCandidateGuide::new)
    /// refuses, and refusing it there is what keeps the check in one place.
    #[must_use]
    pub fn from_raw(digest: u64) -> Self {
        Self(digest)
    }

    /// Little-endian bytes, for mixing into a cache key.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fingerprint({:016x})", self.0)
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// An ordered rule vocabulary, plus the ids that name its members.
///
/// Rules are held behind an `Arc` — the same one an
/// [`EGraph`](super::EGraph) stores — so an
/// [`Optimizer`](super::optimizer::Optimizer) can hand the vocabulary to as
/// many graphs as it likes without rebuilding it, and so the ids it computed
/// once are provably the ids of the rules those graphs apply.
pub struct RuleSet {
    rules: Arc<Vec<Box<dyn Rewrite>>>,
    ids: Arc<Vec<RuleId>>,
    fingerprint: Fingerprint,
}

impl RuleSet {
    /// The production vocabulary: [`super::all_rules`], in declaration
    /// order.
    #[must_use]
    pub fn production() -> Self {
        Self::new(super::all_rules())
    }

    /// [`RuleSet::production`] plus the bounded-fold decompositions.
    ///
    /// The runtime tier's set, and only its: `kernel!` has no syntax that
    /// builds a fold, so at the macro tier these two rules can never fire.
    /// Adding them to [`super::all_rules`] instead would leave them inert
    /// there while perturbing the pinned rule-set grids that
    /// `crate::math::inflate`'s inflation study measures against — a changed
    /// baseline for no measurement's sake. Which rules a tier holds is
    /// already a place the tiers differ (so is the vocabulary, and so is
    /// whether extraction is priced against a lattice).
    #[must_use]
    pub fn runtime() -> Self {
        let mut rules = super::all_rules();
        rules.extend(super::fold_rules::fold_rules());
        Self::new(rules)
    }

    /// A rule set over an explicit vector — the research harnesses that
    /// compare orderings, and the tests.
    #[must_use]
    pub fn new(rules: Vec<Box<dyn Rewrite>>) -> Self {
        let ids: Vec<RuleId> = rules.iter().map(|r| RuleId::of(r.as_ref())).collect();
        let mut h = FNV_OFFSET;
        for id in &ids {
            for &b in &id.0.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        Self {
            rules: Arc::new(rules),
            ids: Arc::new(ids),
            fingerprint: Fingerprint(h),
        }
    }

    /// The shared rule vector and its ids, for building an
    /// [`EGraph`](super::EGraph) over this vocabulary.
    #[must_use]
    pub fn shared(&self) -> (Arc<Vec<Box<dyn Rewrite>>>, Arc<Vec<RuleId>>) {
        (Arc::clone(&self.rules), Arc::clone(&self.ids))
    }

    /// Digest over the ordered sequence of [`RuleId`]s — content *and*
    /// order.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    /// How many rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The id at a position — the translation a caller holding a positional
    /// `rule_idx` (an [`ApplicationRecord`](super::provenance::ApplicationRecord),
    /// a candidate) needs to name what it is holding.
    #[must_use]
    pub fn id_of(&self, idx: usize) -> Option<RuleId> {
        self.ids.get(idx).copied()
    }

    /// The position of an id, for a consumer that has to index back into a
    /// positional structure.
    #[must_use]
    pub fn index_of(&self, id: RuleId) -> Option<usize> {
        self.ids.iter().position(|&i| i == id)
    }

    /// The ordered ids.
    #[must_use]
    pub fn ids(&self) -> &[RuleId] {
        &self.ids
    }

    /// The canonical label at a position.
    #[must_use]
    pub fn label_of(&self, idx: usize) -> Option<String> {
        self.rules.get(idx).map(|r| rule_label(r.as_ref()))
    }
}

impl Default for RuleSet {
    fn default() -> Self {
        Self::production()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;

    /// G5, and the guard that keeps the queued 33-rule batch honest: every
    /// rule in the production vocabulary must have its own id.
    ///
    /// This is the property `Rewrite::name` alone does not have — 30 of the
    /// 62 instances answer to 11 shared family names — and the reason
    /// `specialization` exists.
    #[test]
    fn every_production_rule_has_a_distinct_id() {
        let set = RuleSet::production();
        let mut by_id: BTreeMap<RuleId, Vec<String>> = BTreeMap::new();
        for idx in 0..set.len() {
            let id = set.id_of(idx).expect("id_of within len");
            by_id
                .entry(id)
                .or_default()
                .push(set.label_of(idx).expect("label_of within len"));
        }
        let collisions: Vec<_> = by_id.values().filter(|labels| labels.len() > 1).collect();
        assert!(
            collisions.is_empty(),
            "rule ids must be distinct; collided: {collisions:?}"
        );
        assert_eq!(
            by_id.len(),
            set.len(),
            "one id per rule instance in the production set"
        );
    }

    /// The bug this module exists to prevent: family names alias, so a table
    /// keyed by `name()` silently aggregates four rules into one bucket.
    /// Pinned as a *fact about `name()`*, so that if someone later makes
    /// names discriminating this test fails and points here.
    #[test]
    fn family_names_alias_but_labels_do_not() {
        let rules = super::super::all_rules();
        let mut names: BTreeMap<&str, usize> = BTreeMap::new();
        let mut labels: BTreeMap<String, usize> = BTreeMap::new();
        for r in &rules {
            *names.entry(r.name()).or_insert(0) += 1;
            *labels.entry(rule_label(r.as_ref())).or_insert(0) += 1;
        }
        assert!(
            names.len() < rules.len(),
            "precondition: family names are known to alias"
        );
        assert_eq!(
            labels.len(),
            rules.len(),
            "labels must not alias: {:?}",
            labels.iter().filter(|&(_, &n)| n > 1).collect::<Vec<_>>()
        );
    }

    /// The fingerprint covers order, not just content — the whole reason it
    /// is not simply a hash of the rule names as a set.
    #[test]
    fn fingerprint_covers_order() {
        let a = RuleSet::production();
        let mut reversed = super::super::all_rules();
        reversed.reverse();
        let b = RuleSet::new(reversed);
        assert_eq!(a.len(), b.len());
        assert_ne!(
            a.fingerprint(),
            b.fingerprint(),
            "a reordering must change the fingerprint"
        );
    }

    /// Two constructions of the production set agree — the property a
    /// checkpoint written by one build and read by another depends on.
    #[test]
    fn fingerprint_is_reproducible() {
        assert_eq!(
            RuleSet::production().fingerprint(),
            RuleSet::production().fingerprint()
        );
    }

    /// A label round-trips to the same id, which is what lets a loader read
    /// rule keys back out of JSON.
    #[test]
    fn label_round_trips_to_id() {
        let set = RuleSet::production();
        for idx in 0..set.len() {
            let label = set.label_of(idx).expect("label");
            assert_eq!(RuleId::from_label(&label), set.id_of(idx).expect("id"));
        }
    }
}
