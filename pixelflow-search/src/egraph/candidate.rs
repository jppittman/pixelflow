//! Candidate-local features for the saturation Guide
//! (`docs/plans/2026-08-31-guide-design-revision.md` §4, Phase 3 task 1).
//!
//! The strict extracted-path label itself lives in
//! [`super::labeler::EpisodeLabels::compute_strict`] (alongside the loose
//! `compute` and tightened `compute_tight` variants — see that module's doc
//! for the full bound spectrum); this module holds only the other half of
//! the label-minting record: the candidate-local *feature* a Guide would
//! train and score on.
//!
//! # The feature IS the dedup key (design doc §4)
//!
//! §2.2's headline finding is that 90.4% of scored rewrite candidates commit
//! no change at all: `commutative`/`associative`/`reverse-associative`/
//! `even-negation`/`involution` have no "already applied" check and re-match
//! their own already-installed output on every subsequent scan. The dominant
//! per-round cost for any guided saturation loop is recognizing that *before*
//! scoring the candidate, not scoring it cheaply once recognized. [`CandidateKey`]
//! is exactly the natural dedup key for that: `(rule, canonical class
//! content)` — two candidates with the same key are, by construction, the
//! same rule re-matching the same class content, which is exactly the
//! idempotent-refire pattern §2.2 measured. [`CandidateFeatures::observe`] is
//! the single constructor that builds both the key and the rest of the
//! candidate-local feature (local neighborhood ops, budget state) in one
//! pass, so there is no second code path that could compute the dedup key
//! differently from the feature a Guide would actually train and score on.
//!
//! # Approximation, stated plainly
//!
//! [`CandidateFeatures::observe`] reads `egraph` in whatever state the
//! caller hands it — in practice, for the label-minting pipeline this module
//! was built for, that is the FINAL (fully-saturated) e-graph, not a live
//! snapshot taken at the instant the application actually fired. One
//! consequence, acceptable for a cold-start dataset and not silent:
//! [`ClassContentKey`] and the neighborhood ops can include nodes that were
//! added to the match's canonical class *after* this application fired (the
//! class only grows — nodes are never removed, only merged in). This is the
//! same "any tagged node in a visited class" over-approximation
//! [`super::provenance::derivation_ancestors`] already makes for a different
//! purpose, for the same reason: recovering the exact node set present at
//! firing time would need a second, content-diffing instrumentation layer
//! this stage does not build. A consequence: two firings of the same rule
//! against the same class at genuinely different saturation steps can
//! collapse onto the same key if the class's content hadn't changed between
//! them (which is, not coincidentally, exactly the idempotent-refire case
//! this key exists to catch) — but a firing early against a class that later
//! grows and a firing late against the grown class can *also* collapse onto
//! the same key, which is the over-approximation's cost.
//!
//! [`CandidateFeatures::budget_fraction`] is NOT subject to that
//! approximation: it is a pure function of [`Firing::application_ordinal`]
//! and [`Firing::registered_budget`], both supplied by the caller from
//! [`super::provenance::ApplicationId`]/[`super::provenance::ApplicationRecord`]
//! — exact regardless of when `observe` is called. Per the design's binding
//! budget-only framing (`docs/plans/2026-07-07-guided-saturation-redesign.md`
//! §0): denominated in rule-application ordinal only, never an
//! iteration/sweep counter and never wall-clock.

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, UniformDecl};
use pixelflow_ir::fold::Fold;

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rules::RuleId;

/// The pre-registered Phase 3 primary budget tier B for the classical band
/// (`docs/plans/2026-09-01-phase3-registration.md` §4: "classical: B = 100
/// (primary), B = 200 (secondary)"). This is the ONE denominator
/// [`Firing::registered_budget`]/[`CandidateFeatures::budget_fraction`] means
/// when the design doc talks about "budget spent, in units of B": both the
/// offline label-minting replay (`gen_strict_labels`, which has no live
/// per-call budget to report) and the live guided saturation loop
/// (`super::saturate::saturate_guided_until_applications`) import this same
/// constant rather than each supplying their own call's budget argument as
/// the denominator.
///
/// This is deliberately NOT the same thing as a caller's own
/// `max_total_applications`/budget-tier argument: evaluating a trained Guide
/// at the secondary tier (B=200) or at any other budget must not silently
/// change what `budget_fraction` *means* for a feature the Guide was
/// trained on — the feature's units are fixed at mint time (this constant),
/// and a fraction past `1.0` at a larger evaluation budget is the correct,
/// expected reading of "further past the primary tier than training data
/// ever went," not a unit change.
pub const REGISTERED_PRIMARY_BUDGET_APPLICATIONS: usize = 100;

/// One node's shape, canonicalized against the current union-find so two
/// structurally-equal nodes compare equal regardless of which class holds
/// them or where they sit in that class's node vector.
///
/// Deliberately NOT `ENode` itself: `ENode::Op` carries a `&'static dyn Op`
/// pointer, which has no `Hash`/`Ord` impl (`node.rs`'s own `PartialEq`
/// impl already compares by `OpKind` for the same reason — see its comment).
/// `NodeShape` is the same content, compared the same way, made hashable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NodeShape {
    Var(u8),
    Const(u32),
    Buffer(BufferDecl),
    Uniform(UniformDecl),
    Param(u8),
    Op(OpKind, Vec<u32>),
    /// A fold's shape is its metadata plus its body's class — the metadata
    /// is part of the node's identity, so it is part of its shape.
    Reduce(Fold, u32),
}

impl NodeShape {
    fn of(egraph: &EGraph, node: &ENode) -> Self {
        match node {
            ENode::Var(v) => NodeShape::Var(*v),
            ENode::Const(bits) => NodeShape::Const(*bits),
            ENode::Buffer(decl) => NodeShape::Buffer(*decl),
            ENode::Uniform(decl) => NodeShape::Uniform(*decl),
            ENode::Param(i) => NodeShape::Param(*i),
            ENode::Op { op, children } => NodeShape::Op(
                op.kind(),
                children
                    .iter()
                    .map(|&c| egraph.find(c).index() as u32)
                    .collect(),
            ),
            ENode::Reduce { fold, body } => {
                NodeShape::Reduce(*fold, egraph.find(*body).index() as u32)
            }
        }
    }

    /// Deterministic sort key. Any deterministic total order suffices here —
    /// this is used only to canonicalize the order of a content vector that
    /// is otherwise sensitive to `EGraph`'s internal node-vector layout
    /// (`rebuild_budgeted`'s take/canonicalize/extend cycle reorders it), not
    /// to give `NodeShape` a meaningful `Ord`. Hashing avoids requiring
    /// `Ord` on `OpKind`'s `Vec<u32>` payload or on `BufferDecl`, neither of
    /// which derive it.
    fn sort_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// A canonical, order-independent fingerprint of an e-class's content — the
/// "canonical class content" half of [`CandidateKey`].
///
/// Two calls against classes whose current node sets are structurally
/// identical produce equal keys, regardless of node_idx ordering — the same
/// property `EGraph`'s own memo table (`ENode` + `Hash`, hash-consing on
/// `add()`) already relies on for a single node; this extends it to a whole
/// class's node set, which is what a rewrite rule actually matches against.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClassContentKey(Vec<NodeShape>);

impl ClassContentKey {
    /// Number of distinct node shapes recorded in this class. Because
    /// `EGraph::add` hash-conses globally (not per-class), a class can never
    /// hold two nodes with the same `NodeShape` — this is exactly
    /// `egraph.nodes(class).len()` for the class this key was built from.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.0.len()
    }

    /// The content key of `canonical` as the graph stands now.
    ///
    /// Crate-visible so the counterfactual-replay mask can read the key of
    /// the application it is about to skip — it must be THE key that
    /// application matched, read at that instant, not a caller's second
    /// computation against a different graph state.
    pub(crate) fn of(egraph: &EGraph, canonical: EClassId) -> Self {
        let mut shapes: Vec<NodeShape> = egraph
            .nodes(canonical)
            .iter()
            .map(|n| NodeShape::of(egraph, n))
            .collect();
        shapes.sort_by_key(NodeShape::sort_key);
        ClassContentKey(shapes)
    }
}

/// The deduplication key a Guide-driven saturation loop needs: which rule,
/// matched against which class content. See the module doc for why this is
/// deliberately the same object as (part of) the training feature, not a
/// second computation of related information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandidateKey {
    /// Which rewrite rule, by stable identity.
    ///
    /// A [`RuleId`], not the positional `rule_idx` this key originally
    /// carried: a candidate key outlives the vector it was minted against
    /// (it is written into training artifacts and compared across runs),
    /// and a same-length reorder of the rule table silently repoints every
    /// positional key without changing any length — the failure
    /// [`super::rules`] exists to prevent.
    pub rule: RuleId,
    /// Canonical content of the matched e-class at observation time (see the
    /// module doc's "Approximation, stated plainly" section for exactly what
    /// "at observation time" means).
    pub content: ClassContentKey,
}

/// The caller-supplied firing context for one recorded rewrite-rule
/// application — kept as one struct (rather than four loose arguments to
/// [`CandidateFeatures::observe`]) so `registered_budget`, constant across
/// one episode's whole application log, has an obvious home next to the
/// per-application fields a caller reads out of
/// [`super::provenance::Provenance::applications`].
#[derive(Clone, Copy, Debug)]
pub struct Firing {
    /// Which rewrite rule fired, by stable identity
    /// (`ApplicationRecord::rule`, or `EGraph::rule_id(idx)` for a caller
    /// holding a positional index).
    pub rule: RuleId,
    /// The e-class the rule matched against (`ApplicationRecord::match_root`).
    pub match_root: EClassId,
    /// This application's own firing order (`ApplicationId::as_u64()`).
    pub application_ordinal: u64,
    /// The budget-only anytime denominator (design doc §0/§5): the
    /// pre-registered rule-application budget this episode's saturation is
    /// being measured against — NOT the episode's own eventual total
    /// application count, and NOT an iteration/sweep counter. An application
    /// firing past this budget is legitimate (a real anytime curve keeps
    /// sampling past B); it simply yields `budget_fraction > 1.0`.
    pub registered_budget: usize,
}

/// The candidate-local feature for one rewrite-rule application (design doc
/// §4): rule identity + matched e-class canonical content (together,
/// [`CandidateKey`] — see module doc) + local neighborhood ops + budget
/// state at firing time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandidateFeatures {
    /// Rule identity + matched e-class content — also the dedup key.
    pub key: CandidateKey,
    /// One-hop child ops of every node in the matched class: the operations
    /// immediately reachable from this match, sorted for determinism. Not
    /// part of the dedup key (two firings with the same `key` dedup
    /// regardless of neighborhood — the neighborhood is a function of the
    /// class content already inside `key`, kept separate here because it's
    /// the shape a feature encoder consumes, not a set-membership test).
    pub neighborhood_ops: Vec<OpKind>,
    /// `application_ordinal / registered_budget` — see [`Firing::registered_budget`].
    /// Stored as bits so `CandidateFeatures` can derive `Eq`/`Hash` (needed
    /// to dedup features themselves in a training-set builder) without a
    /// hand-rolled `f32` comparison; use [`Self::budget_fraction`] to read it.
    budget_fraction_bits: u32,
}

impl CandidateFeatures {
    /// Build the candidate-local feature for one recorded rewrite-rule
    /// firing — and, in the same pass, its deduplication key (`self.key`).
    /// One constructor: see the module doc for why the feature and the key
    /// must never be computed by two different code paths.
    ///
    /// `firing.match_root` is canonicalized here via `egraph.find`, so a
    /// caller may pass either the class id recorded at firing time or its
    /// current canonical form.
    #[must_use]
    pub fn observe(egraph: &EGraph, firing: &Firing) -> Self {
        let canonical = egraph.find(firing.match_root);
        let content = ClassContentKey::of(egraph, canonical);
        let neighborhood_ops = neighborhood_ops(egraph, canonical);
        let key = CandidateKey {
            rule: firing.rule,
            content,
        };
        let budget_fraction = if firing.registered_budget == 0 {
            0.0
        } else {
            firing.application_ordinal as f32 / firing.registered_budget as f32
        };
        Self {
            key,
            neighborhood_ops,
            budget_fraction_bits: budget_fraction.to_bits(),
        }
    }

    /// `application_ordinal / registered_budget`, in `[0.0, +inf)` — exact,
    /// see module doc. Can exceed `1.0`: an application firing after the
    /// registered budget is a legitimate anytime-curve sample, not an error.
    #[must_use]
    pub fn budget_fraction(&self) -> f32 {
        f32::from_bits(self.budget_fraction_bits)
    }
}

/// One-hop child ops of every node tagged in `canonical`, sorted for
/// deterministic output. Not deduplicated — a repeated op counts once per
/// occurrence, since "how many multiplies feed this match" is itself
/// informative for a move-ordering feature (unlike [`ClassContentKey`],
/// which needs no separate dedup because hash-consing already guarantees
/// distinct shapes).
fn neighborhood_ops(egraph: &EGraph, canonical: EClassId) -> Vec<OpKind> {
    let mut ops = Vec::new();
    for node in egraph.nodes(canonical) {
        for &child in node.children_slice() {
            let child_canonical = egraph.find(child);
            for child_node in egraph.nodes(child_canonical) {
                if let ENode::Op { op, .. } = child_node {
                    ops.push(op.kind());
                }
            }
        }
    }
    ops.sort();
    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::ops;
    use crate::egraph::rewrite::Rewrite;
    use crate::math::algebra::Commutative;

    fn egraph_with_commutative() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = vec![Commutative::new(&ops::Add)];
        EGraph::with_rules(rules)
    }

    /// The id of the rule at `idx` in `eg`'s own table — the translation
    /// every caller holding a positional match makes.
    fn rule_id_at(eg: &EGraph, idx: usize) -> RuleId {
        eg.rule_id(idx)
            .expect("rule index within the graph's table")
    }

    #[test]
    fn class_content_key_is_stable_under_reflexive_recomputation() {
        let mut eg = egraph_with_commutative();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let a = ClassContentKey::of(&eg, eg.find(sum));
        let b = ClassContentKey::of(&eg, eg.find(sum));
        assert_eq!(
            a, b,
            "recomputing the key against an unchanged class must be stable"
        );
        assert_eq!(a.node_count(), 1);
    }

    #[test]
    fn class_content_key_changes_when_a_rule_adds_a_node() {
        let mut eg = egraph_with_commutative();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let before = ClassContentKey::of(&eg, eg.find(sum));

        let target = eg
            .find_rewrite_matches()
            .into_iter()
            .find(|t| t.class_id == eg.find(sum))
            .expect("commutative should match x + y");
        assert!(eg.apply_single_rule(target.rule_idx, target.class_id, target.tag));

        let after = ClassContentKey::of(&eg, eg.find(sum));
        assert_ne!(
            before, after,
            "adding a node to the class must change its content key"
        );
        assert_eq!(after.node_count(), 2);
    }

    #[test]
    fn candidate_features_observe_is_deterministic_and_matches_its_own_key() {
        let mut eg = egraph_with_commutative();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let sum_class = eg.find(sum);
        let rule = RuleId::from_label("some-rule(Add)");
        let firing = Firing {
            rule,
            match_root: sum_class,
            application_ordinal: 5,
            registered_budget: 20,
        };

        let a = CandidateFeatures::observe(&eg, &firing);
        let b = CandidateFeatures::observe(&eg, &firing);
        assert_eq!(a, b);
        assert_eq!(a.key.rule, rule);
        assert_eq!(a.key.content.node_count(), 1);
        assert!((a.budget_fraction() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn observe_zero_registered_budget_yields_zero_fraction_not_nan_or_inf() {
        let mut eg = egraph_with_commutative();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let firing = Firing {
            rule: rule_id_at(&eg, 0),
            match_root: eg.find(sum),
            application_ordinal: 0,
            registered_budget: 0,
        };
        let f = CandidateFeatures::observe(&eg, &firing);
        assert_eq!(f.budget_fraction(), 0.0);
    }

    #[test]
    fn budget_fraction_can_exceed_one_past_the_registered_budget() {
        let mut eg = egraph_with_commutative();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let firing = Firing {
            rule: rule_id_at(&eg, 0),
            match_root: eg.find(sum),
            application_ordinal: 50,
            registered_budget: 20,
        };
        let f = CandidateFeatures::observe(&eg, &firing);
        assert!((f.budget_fraction() - 2.5).abs() < 1e-6);
    }

    /// Full-episode sanity: on a real saturated e-graph, repeated matches of
    /// the same rule against an unchanged class collapse onto the same
    /// `CandidateKey` (the dedup property §2.2/§4 are built on).
    #[cfg(feature = "provenance-journal")]
    #[test]
    fn repeated_idempotent_refires_share_a_candidate_key() {
        use pixelflow_ir::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(pixelflow_ir::OpKind::Add, x, y);
        let root = arena.push_binary(pixelflow_ir::OpKind::Mul, sum, sum);

        let mut egraph = EGraph::with_rules(crate::egraph::all_rules());
        let root_class = crate::egraph::insert(
            &arena,
            root,
            &mut egraph,
            crate::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        egraph.saturate_budgeted(30, 5_000, None);

        // The denominator is the run's own application count, read off the
        // unconditional budget counter rather than the journal — the two
        // agree (both count one per action commit), and only the former is
        // the budget's own currency.
        let registered_budget = egraph.application_count() as usize;
        let mut seen: std::collections::HashSet<CandidateKey> = std::collections::HashSet::new();
        let mut repeats = 0usize;
        let firings: Vec<Firing> = egraph
            .provenance()
            .applications()
            .map(|(app_id, record)| Firing {
                rule: record
                    .rule
                    .expect("a recorded application names the rule that fired"),
                match_root: record.match_root,
                application_ordinal: app_id.as_u64(),
                registered_budget,
            })
            .collect();
        for firing in &firings {
            let features = CandidateFeatures::observe(&egraph, firing);
            if !seen.insert(features.key.clone()) {
                repeats += 1;
            }
        }
        assert!(
            repeats > 0,
            "a real saturation episode with commutative/associative rules in play should \
             produce at least one repeated (rule, class-content) pair — if this starts \
             failing, either the rule set or `all_rules()` changed in a way that removed \
             the idempotent-refire pattern §2.2 measured, or CandidateKey stopped \
             recognizing it"
        );
        let _ = root_class; // kept alive; only provenance is inspected above
    }
}
