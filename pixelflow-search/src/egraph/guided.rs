//! Guided saturation: the same rewrites, in an order a
//! [`SaturationGuide`] chooses.
//!
//! # What a Guide is allowed to do, and why that is the whole safety argument
//!
//! A Guide **orders candidates**. It does not construct a
//! [`RewriteAction`](super::rewrite::RewriteAction), does not call `union`,
//! does not touch `const_fact`, and never sees a mutable e-graph. Every
//! mutation this loop performs goes through
//! [`EGraph::apply_single_rule`], which re-checks the rule against the
//! matched node and dispatches through the same `apply_action_from_rule` the
//! unguided sweep uses — so the guided run applies a *subsequence* of the
//! rewrites an exhaustive run would apply, in a different order, and nothing
//! else.
//!
//! That is exactly the precondition of law **L4** in
//! `docs/plans/2026-09-02-optimizer-api.md`: every rule preserves denotation
//! (L1), saturation only ever adds equalities (L2), so any policy that orders
//! or truncates leaves the graph holding a *subset* of the equalities an
//! exhaustive run would hold — never a different one — and extraction picks a
//! node from the root's class, every member of which denotes the root's
//! function. A Guide therefore changes cost and compile time, never meaning.
//! `Optimizer::run`'s L4 test pins this with a reversing guide.
//!
//! # How it differs from the unguided sweep
//!
//! Unguided saturation ([`EGraph::saturate_budgeted`]) applies every rule
//! against every matching node in a fixed rule-then-class scan order, once
//! per sweep, with no notion of "have I already resolved this exact (rule,
//! class content) pair". This loop instead, once per round:
//!
//! 1. Enumerates every current match ([`EGraph::find_rewrite_matches`]).
//! 2. Builds each match's [`CandidateKey`] and skips any key already seen in
//!    an EARLIER round of this same episode, without ever handing it to the
//!    Guide — the "dedup before scoring" step
//!    `docs/plans/2026-08-31-guide-design-revision.md` §2.2 measured as the
//!    dominant per-round cost (90.4% of raw matches are an idempotent
//!    re-fire of an already-installed rewrite, from rules like
//!    `commutative`/`associative` that carry no "already applied" check).
//! 3. Scores the survivors in ONE batched
//!    [`SaturationGuide::score_candidates`] call (binding rule: batch per
//!    round, no incrementality work).
//! 4. Applies them in descending-score order, budget- and cap-checked per
//!    application — finer-grained than the unguided per-sweep check, since
//!    one guided round can apply many more candidates before its next full
//!    rescan.
//!
//! A round in which every match is an already-seen key is quiescence: without
//! that, guided mode would loop re-enumerating a fully-deduped match set
//! until the round ceiling, and report "ran out of rounds" for a graph that
//! had nothing left to try.
//!
//! # Deterministic by construction
//!
//! No clock. The stop reasons this loop can report are
//! [`SaturationStop::Quiesced`], [`SaturationStop::ApplicationBudget`],
//! [`SaturationStop::ClassCap`] and [`SaturationStop::IterationCeiling`] —
//! never [`SaturationStop::Timeout`], for the same reason
//! [`EGraph::saturate_budgeted`] cannot report it. A wall-clock ceiling
//! still exists on [`Optimizer::hard_ceiling`](super::optimizer::Optimizer::hard_ceiling),
//! where exceeding it panics instead of quietly changing the answer.
//!
//! # Approximation, stated plainly (mirrors `egraph::candidate`'s own note)
//!
//! A match enumerated at the start of a round can be invalidated mid-round:
//! applying an earlier, higher-scored candidate may union or rebuild the
//! e-class a later candidate matched against. [`RewriteTarget`] therefore
//! names the matched node by its stable
//! [`ENodeId`](super::provenance::ENodeId), not by a position in the class's
//! node vector, and `apply_single_rule` re-resolves that identity before
//! re-checking the rule — so a mid-round rebuild either leaves the exact
//! matched node in place (the rule is re-checked against it, as before) or
//! removes it, in which case nothing fires and nothing is recorded. It can
//! never apply a *different* node that happened to inherit the old index
//! while the stale candidate's score and dedup key were credited to it. The
//! missed opportunity, if any, is picked up on the next round's rescan.
//! `budget_fraction`'s `application_ordinal` is similarly a per-round
//! approximation (the cumulative application count at the START of scoring,
//! shared by every candidate scored that round, not each candidate's own
//! eventual firing order) — exact per-candidate ordinals would require
//! scoring one candidate at a time, which the batch-per-round binding rule
//! rules out.

use std::collections::{HashMap, HashSet};

use super::candidate::{
    CandidateFeatures, CandidateKey, Firing, REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
};
use super::graph::{EGraph, RewriteTarget, SaturationStats, SaturationStop};
use super::optimizer::Limits;
use super::rules::RuleId;
use crate::nnue::factored::EMBED_DIM;
use crate::nnue::guide::{CandidateSummary, SaturationGuide};

/// One guided-saturation *episode*: the candidate-key dedup set, the
/// episode-level feature constant, and the per-rule embedding cache, kept
/// together so a caller that advances the same e-graph through several
/// successive application budgets — an anytime curve sampled at 25, 50, 100,
/// … applications ([`super::anytime`]) — sees ONE continuous guided run
/// rather than a fresh episode per checkpoint.
///
/// Why this is state and not a per-call local: with the dedup set recreated
/// on every call, the second checkpoint's first round would re-score and
/// re-fire every already-resolved `(rule, class content)` key from the first
/// — each an idempotent re-fire that still *records* an application, i.e.
/// spends the budget on exactly the ~90% no-op traffic dedup exists to
/// eliminate. Unguided saturation has no per-call state, so its curve was
/// never exposed to this; the guided curve would have been silently
/// handicapped at every checkpoint after the first. Measured on 2026-09-01
/// (DEV classical, 334 expressions): sampling a guided anytime curve at
/// 25/50 before 100 cost the guided arm a median ~50% of its
/// applications-to-quiescence versus sampling at 100 directly.
#[derive(Default)]
pub(crate) struct GuidedEpisode {
    seen_keys: HashSet<CandidateKey>,
    /// The graph's node count when the episode started, before any guided
    /// round fired — `CandidateSummary`'s `expr_node_count`, an
    /// episode-level constant.
    expr_node_count: Option<usize>,
    rule_embeds: HashMap<RuleId, [f32; EMBED_DIM]>,
}

impl GuidedEpisode {
    /// How many distinct candidate keys this episode has resolved
    /// (diagnostic: dedup coverage).
    pub(super) fn seen_key_count(&self) -> usize {
        self.seen_keys.len()
    }

    /// Advance `egraph` under `limits`, applying candidates in the order
    /// `guide` scores them.
    ///
    /// `limits.applications` is a **delta** from the graph's current
    /// cumulative count, resolved here exactly as
    /// `EGraph::saturate_bounded` resolves it — so a caller stepping an
    /// anytime curve passes the gap to the next checkpoint, not the
    /// checkpoint itself.
    ///
    /// # Panics
    ///
    /// If `guide.score_candidates` returns a different number of scores than
    /// candidates given, or any non-finite score. A guide bug must not
    /// silently corrupt move ordering.
    pub(crate) fn advance(
        &mut self,
        egraph: &mut EGraph,
        guide: &dyn SaturationGuide,
        limits: Limits,
    ) -> SaturationStats {
        let max_classes = limits.classes.min(super::graph::HARD_CLASS_LIMIT);
        let application_cap = limits
            .applications
            .map(|n| egraph.application_count().saturating_add(n));
        // The episode's feature constant: the live analogue of the offline
        // label-minting replay's node-count snapshot, taken
        // before saturation runs.
        let expr_node_count = *self
            .expr_node_count
            .get_or_insert_with(|| egraph.node_count());

        let mut iterations = 0usize;
        let mut total_unions = 0usize;

        let stop = 'outer: loop {
            if application_cap.is_some_and(|cap| egraph.application_count() >= cap) {
                break SaturationStop::ApplicationBudget;
            }
            if iterations >= limits.iterations {
                break SaturationStop::IterationCeiling;
            }
            if egraph.num_classes() > max_classes {
                break SaturationStop::ClassCap;
            }
            iterations += 1;

            let matches = egraph.find_rewrite_matches();
            if matches.is_empty() {
                break SaturationStop::Quiesced;
            }

            // Dedup BEFORE scoring (§2.2/§4). The ordinal recorded is a
            // per-round approximation (see the module doc), used only for
            // the `budget_fraction` feature, never for correctness.
            //
            // A key becomes "seen" only once an application has actually
            // been RECORDED for it (below), never here at enumeration time.
            // Marking at enumeration burned every scored survivor the round
            // never got to — the ones after a mid-round budget or class-cap
            // stop, and the ones whose matched node an earlier application
            // in the same round removed so `apply_single_rule` recorded
            // nothing — and the next round's rescan then deduped them away
            // for good. Measured on 2026-09-01 (DEV classical, 334
            // expressions): a median 39–44% of scored keys were burned that
            // way.
            //
            // Within a round, several nodes of one class can match the same
            // rule. They share a `CandidateKey` — the key is `(rule, class
            // content)` and carries no node identity — so they are scored
            // ONCE, as one candidate. They are NOT collapsed to one target:
            // `apply_single_rule` on the first of them can record an
            // application that changes nothing (91% of recorded applications
            // are exact no-ops, per
            // docs/results/2026-08-30-guide-scope-saturation-delta.md), which
            // would mark the shared key resolved while a sibling target that
            // would have unioned a class was never attempted at all — and the
            // loop could then report `Quiesced` with applicable work left.
            // So targets sharing a key are kept together and every one of
            // them is attempted before the key is marked seen.
            let ordinal = egraph.application_count();
            let mut survivors: Vec<(Vec<RewriteTarget>, CandidateFeatures)> = Vec::new();
            let mut round_slot: HashMap<CandidateKey, usize> = HashMap::new();
            for target in matches {
                let rule = egraph.rule_id(target.rule_idx).unwrap_or_else(|| {
                    panic!(
                        "guided saturation: match named rule_idx {} but the graph's rule table \
                         has no id for it",
                        target.rule_idx
                    )
                });
                let firing = Firing {
                    rule,
                    match_root: target.class_id,
                    application_ordinal: ordinal,
                    registered_budget: REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
                };
                let features = CandidateFeatures::observe(egraph, &firing);
                if self.seen_keys.contains(&features.key) {
                    continue;
                }
                match round_slot.get(&features.key) {
                    Some(&slot) => survivors[slot].0.push(target),
                    None => {
                        round_slot.insert(features.key.clone(), survivors.len());
                        survivors.push((vec![target], features));
                    }
                }
            }

            if survivors.is_empty() {
                break SaturationStop::Quiesced;
            }

            // Batch-score (never one candidate at a time — binding rule).
            let summaries: Vec<CandidateSummary> = survivors
                .iter()
                .map(|(_, features)| {
                    let rule = features.key.rule;
                    // One encode per rule per episode, not per candidate:
                    // the embedding is a pure function of the rule.
                    let rule_embed = *self
                        .rule_embeds
                        .entry(rule)
                        .or_insert_with(|| guide.rule_embed(rule));
                    CandidateSummary::new(features, rule_embed, expr_node_count)
                })
                .collect();
            let scores = guide.score_candidates(&summaries);
            assert_eq!(
                scores.len(),
                survivors.len(),
                "SaturationGuide::score_candidates must return exactly one score per candidate \
                 — got {} scores for {} candidates",
                scores.len(),
                survivors.len()
            );
            for &s in &scores {
                assert!(
                    s.is_finite(),
                    "SaturationGuide produced a non-finite score ({s}) — fail loud rather than \
                     let NaN/inf silently corrupt move ordering"
                );
            }

            // Descending by score; ties keep match-enumeration order
            // (`sort_by` is stable), so a fixed guide gives a fixed run.
            let mut order: Vec<usize> = (0..survivors.len()).collect();
            order.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .expect("scores were just asserted finite")
            });

            for idx in order {
                let (targets, features) = &survivors[idx];
                let (targets, key) = (targets.clone(), features.key.clone());
                let mut recorded_any = false;
                for target in &targets {
                    if application_cap.is_some_and(|cap| egraph.application_count() >= cap) {
                        if recorded_any {
                            self.seen_keys.insert(key);
                        }
                        break 'outer SaturationStop::ApplicationBudget;
                    }
                    let before = egraph.application_count();
                    if egraph.apply_single_rule(target.rule_idx, target.class_id, target.tag) {
                        total_unions += 1;
                    }
                    if egraph.application_count() > before {
                        recorded_any = true;
                    }
                    if egraph.num_classes() > max_classes {
                        if recorded_any {
                            self.seen_keys.insert(key);
                        }
                        break 'outer SaturationStop::ClassCap;
                    }
                }
                // At least one target recorded (whether or not it changed
                // anything): this key is resolved. None recorded (every
                // matched node gone, so nothing fired): leave it unseen so
                // the next rescan can retry it against the rebuilt graph.
                if recorded_any {
                    self.seen_keys.insert(key);
                }
            }
        };

        SaturationStats {
            iterations,
            total_unions,
            stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::rewrite::Rewrite;
    use crate::egraph::{ENode, Optimizer, ops};
    use crate::math::algebra::Commutative;
    use crate::nnue::factored::OpEmbeddings;
    use crate::nnue::guide::Guide;
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    /// Scores every candidate zero — ordering therefore falls back to the
    /// stable sort's tie-break, i.e. match-enumeration order.
    struct FlatGuide;
    impl SaturationGuide for FlatGuide {
        fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
            alloc::vec![0.0; candidates.len()]
        }
    }

    /// Reverses match-enumeration order: the adversarial ordering L4 has to
    /// hold under.
    struct ReversingGuide;
    impl SaturationGuide for ReversingGuide {
        fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
            (0..candidates.len()).map(|i| i as f32).collect()
        }
    }

    /// Records how many times it was asked to score, and how many candidates
    /// it was handed each time.
    struct SpyGuide {
        calls: std::cell::RefCell<Vec<usize>>,
    }
    impl SaturationGuide for SpyGuide {
        fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
            self.calls.borrow_mut().push(candidates.len());
            alloc::vec![0.0; candidates.len()]
        }
    }

    fn commutative_only_egraph() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = alloc::vec![Commutative::new(&ops::Add)];
        let mut eg = EGraph::with_rules(rules);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![x, y],
        });
        eg
    }

    /// One e-class holding several `Add` nodes, all matching `Commutative`,
    /// so all of them share a single `CandidateKey`. `x+y` and `y+x` commute
    /// into each other and are therefore no-ops; `z+w` has no commuted form
    /// in the graph yet, so it is the one productive target — and it is last
    /// in enumeration order.
    fn one_class_three_matching_nodes_egraph() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = alloc::vec![Commutative::new(&ops::Add)];
        let mut eg = EGraph::with_rules(rules);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let w = eg.add(ENode::Var(3));
        let xy = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![x, y],
        });
        // Pre-existing commuted form: applying Commutative at `x+y` (or at
        // `y+x`) can only re-derive a node that is already here.
        eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![y, x],
        });
        let zw = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![z, w],
        });
        eg.union(xy, zw);
        eg.rebuild();
        eg
    }

    fn generous_limits() -> Limits {
        Limits {
            iterations: 200,
            classes: 10_000,
            applications: Some(10_000),
        }
    }

    /// `Quiesced` is a claim about the graph, not about the loop's own
    /// bookkeeping: if the guided loop says there is nothing left to apply,
    /// an unguided sweep — which shares no `seen_keys` with it — must find
    /// nothing either.
    ///
    /// This is the property a node-blind candidate key breaks. Several nodes
    /// of one class can match one rule and share a `CandidateKey`;
    /// collapsing them to a single target lets a no-op application mark the
    /// key resolved while a productive sibling is never attempted, and the
    /// next rescan then dedups it away for good.
    #[test]
    fn quiescence_must_mean_an_unguided_sweep_finds_nothing_left() {
        for (label, mut eg) in [
            (
                "one-class-three-nodes",
                one_class_three_matching_nodes_egraph(),
            ),
            ("commutative", commutative_only_egraph()),
        ] {
            let guide = FlatGuide;
            let mut episode = GuidedEpisode::default();
            let guided = episode.advance(&mut eg, &guide, generous_limits());
            assert_eq!(
                guided.stop,
                SaturationStop::Quiesced,
                "{label}: expected the guided loop to quiesce under a generous budget: {guided:?}"
            );

            let after = eg.saturate_budgeted(200, 10_000, None);
            assert_eq!(
                after.total_unions, 0,
                "{label}: the guided loop reported Quiesced, but an unguided sweep over the same \
                 graph still merged {} classes — applicable work was left behind",
                after.total_unions
            );
        }
    }

    #[test]
    fn dedup_keeps_the_guide_from_ever_rescoring_a_resolved_candidate_key() {
        let mut eg = commutative_only_egraph();
        let guide = SpyGuide {
            calls: std::cell::RefCell::new(Vec::new()),
        };
        let mut episode = GuidedEpisode::default();
        let stats = episode.advance(&mut eg, &guide, generous_limits());
        assert_eq!(stats.stop, SaturationStop::Quiesced);
        let calls = guide.calls.into_inner();
        assert!(
            !calls.is_empty(),
            "the guide must have been asked to score at least once"
        );
        assert!(
            calls.iter().sum::<usize>() >= episode.seen_key_count(),
            "every resolved key was scored exactly once: scored {calls:?}, resolved {}",
            episode.seen_key_count()
        );
    }

    #[test]
    fn the_application_budget_is_a_delta_from_the_current_count() {
        let mut eg = commutative_only_egraph();
        let guide = FlatGuide;
        let mut episode = GuidedEpisode::default();
        let limits = Limits {
            iterations: 200,
            classes: 10_000,
            applications: Some(1),
        };
        let first = episode.advance(&mut eg, &guide, limits);
        assert_eq!(first.stop, SaturationStop::ApplicationBudget);
        let after_first = eg.application_count();
        assert_eq!(after_first, 1, "one application, not one *target* count");

        let second = episode.advance(&mut eg, &guide, limits);
        assert_eq!(
            eg.application_count(),
            2,
            "a second budget of 1 buys one more application, not zero — the cap is \
             current + n, never an absolute target: {second:?}"
        );
    }

    /// L4, guide neutrality (`docs/plans/2026-09-02-optimizer-api.md`): a
    /// Guide may change what saturation costs and which term extraction
    /// picks; it may not change what that term *denotes*.
    ///
    /// Denotation, not syntax, and that distinction is the point. The
    /// reversing guide below really does extract a different term than the
    /// flat one — a 4-node `MulAdd` form against a 5-node one — which is the
    /// lever doing its job. What must not vary is the function: L1 says every
    /// rule preserves denotation, L2 says saturation only adds equalities, so
    /// every node of the root's class denotes the root's function whichever
    /// subset of them a policy managed to install. A guide that broke that
    /// would show up here as an unequal value at some sample point, not as a
    /// different shape.
    #[test]
    fn l4_a_guide_changes_the_order_not_the_denotation() {
        use pixelflow_ir::binding::BindingTable;
        use pixelflow_ir::eval_scalar;
        use pixelflow_ir::expr::{ExprBuilder, Term};

        let mut arena = ExprBuilder::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(pixelflow_ir::OpKind::Add, x, y);
        let prod = arena.push_binary(pixelflow_ir::OpKind::Mul, sum, y);
        let recip = arena.push_unary(pixelflow_ir::OpKind::Sqrt, prod);
        let root = arena.push_binary(pixelflow_ir::OpKind::Add, recip, sum);

        let samples: Vec<[f32; 2]> =
            alloc::vec![[0.5, 1.5], [2.0, 3.0], [-1.25, 4.5], [7.0, 0.25],];
        let want: Vec<f32> = samples
            .iter()
            .map(|c| {
                eval_scalar(
                    Term::new(arena.node(root), arena.env()),
                    c,
                    &BindingTable::empty(),
                )
            })
            .collect();

        let guides: Vec<(&str, Box<dyn SaturationGuide>)> = alloc::vec![
            ("flat", Box::new(FlatGuide) as Box<dyn SaturationGuide>),
            ("reversing", Box::new(ReversingGuide)),
            (
                "untrained-nnue",
                Box::new(Guide::new_random(OpEmbeddings::new_random(7), 11)),
            ),
        ];

        let mut shapes: Vec<(&str, usize)> = Vec::new();
        for (label, guide) in guides {
            let mut opt = Optimizer::production()
                .budget(super::super::optimizer::Budget::Applications(4_000))
                .guide(Some(guide))
                .no_ceiling();
            let mut eg = opt.egraph();
            let root_class =
                crate::egraph::insert(&arena, root, &mut eg, crate::egraph::Vocabulary::Templates)
                    .expect("insert into e-graph");
            let out = opt.run(&mut eg, root_class, arena.len());
            let (got, got_env) = out.to_rooted(&eg, root_class);
            for (c, expected) in samples.iter().zip(&want) {
                let value = eval_scalar(
                    Term::new(got.entry(), &got_env),
                    c,
                    &BindingTable::empty(),
                );
                assert!(
                    (value - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "L4: guide {label} changed the denotation at {c:?}: {value} vs {expected}"
                );
            }
            shapes.push((label, got.len()));
        }
        assert_eq!(shapes.len(), 3);
    }

    /// The trait's default `rule_embed` is all-zero, so a guide that does
    /// not encode rules (every per-rule linear model) needs no boilerplate.
    /// An NNUE `Guide` with no attached table says the same thing — an
    /// untrained guide has no encoding — and one with a table says what the
    /// table says, keyed by identity, never by position.
    #[test]
    fn rule_embed_defaults_to_zero_and_is_keyed_by_rule_identity() {
        let rule = RuleId::from_label("commutative(Add)");
        let other = RuleId::from_label("fma-fusion");
        assert_eq!(FlatGuide.rule_embed(rule), [0.0; EMBED_DIM]);

        let untrained = Guide::new_random(OpEmbeddings::new_random(3), 5);
        assert_eq!(
            untrained.rule_embed(rule),
            [0.0; EMBED_DIM],
            "an NNUE guide with no rule table has no encoding to offer"
        );

        let mut table = alloc::collections::BTreeMap::new();
        table.insert(rule, [0.5f32; EMBED_DIM]);
        let trained = Guide::new_random(OpEmbeddings::new_random(3), 5).with_rule_embeds(table);
        assert_eq!(trained.rule_embed(rule), [0.5; EMBED_DIM]);
        assert_eq!(
            trained.rule_embed(other),
            [0.0; EMBED_DIM],
            "a rule absent from the table falls back to the default, never to \
             another rule's vector"
        );
    }
}
