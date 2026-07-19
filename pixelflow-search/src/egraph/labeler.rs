//! Hindsight labeler: turn a finished saturation + extraction episode into
//! supervised training labels for the future Guide (match scorer).
//!
//! This is Phase 1's final piece (see
//! `docs/plans/2026-07-07-guided-saturation-redesign.md`): the e-graph is an
//! audit log, and after an episode the winning extraction's derivation DAG
//! gives every fired rewrite an exact, observed label — *load-bearing* (it
//! contributed to the chosen extraction) or *wasted* (it fired but the
//! extraction never used what it produced). No estimator, no critic: the
//! label is a fact recorded by [`super::provenance::derivation_ancestors`].
//!
//! # Building blocks
//!
//! - [`EpisodeLabels::compute`]: label every recorded [`ApplicationId`] given
//!   an e-graph and a chosen extraction (the `(EClassId) -> node_idx` choice
//!   map the extractor produces — see [`super::extract::ExtractedDAG`] /
//!   [`super::extract::extract_dag`]).
//! - [`run_episode`]: convenience entry point running the whole pipeline
//!   (saturate → extract with the latency-prior cost model → label) on a
//!   single expression. This is the seed of the future episode collector.
//! - [`EpisodeLabels::format_rule_report`]: a sorted, human-readable
//!   "which rules earn their keep" table.
//!
//! # Over-approximation, inherited
//!
//! `load_bearing` is exactly `derivation_ancestors` of the chosen nodes, so
//! it inherits that function's conservative over-approximation (see its doc
//! comment): an application can be marked load-bearing even if, on
//! reflection, its specific contribution wasn't strictly necessary — e.g. it
//! shares a class with the node that was actually chosen. The property that
//! must hold (and does, transitively) is the safety direction: an
//! application that measurably contributed to the extracted expression is
//! never marked wasted. Over-crediting a handful of applications is an
//! acceptable, documented cost of a conservative label; under-crediting
//! would silently corrupt the Guide's training signal, which is not
//! acceptable.

use std::collections::{BTreeMap, BTreeSet};

use super::cost::CostModel;
use super::extract::{self, ExtractedDAG};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::provenance::{ApplicationId, ENodeId};
use super::rewrite::Rewrite;

/// Binary hindsight label for one recorded rewrite application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    /// The application is a (conservative) ancestor of the chosen
    /// extraction — see the module-level over-approximation note.
    LoadBearing,
    /// The application fired but nothing it produced was reachable from the
    /// chosen extraction.
    Wasted,
}

/// Per-rule aggregate counts: how many times a rule fired during this
/// episode vs. how many of those firings turned out load-bearing.
///
/// This is the Guide's first training signal: `load_bearing_ratio()` per
/// rule is a crude but honest "does this rule earn its keep" prior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuleStats {
    /// Number of times this rule fired (recorded an `ApplicationRecord`).
    pub fired: usize,
    /// Number of those firings labeled load-bearing.
    pub load_bearing: usize,
}

impl RuleStats {
    /// Firings that were not load-bearing. Always `fired - load_bearing`;
    /// exposed so callers don't have to re-derive the subtraction.
    pub fn wasted(&self) -> usize {
        self.fired - self.load_bearing
    }

    /// Fraction of firings that were load-bearing, in `[0.0, 1.0]`.
    /// `0.0` for a rule that never fired (rather than `NaN`) — a rule with
    /// zero firings has no evidence either way, and the Guide should treat
    /// "no evidence" and "never paid off" differently upstream, not collapse
    /// them into a propagating `NaN`.
    pub fn load_bearing_ratio(&self) -> f64 {
        if self.fired == 0 {
            0.0
        } else {
            self.load_bearing as f64 / self.fired as f64
        }
    }
}

/// Supervised training labels derived from a finished saturation +
/// extraction episode.
///
/// Conservative, data-shaped by design (no callbacks): a future trainer
/// consumes `labels` and `rule_stats` directly.
#[derive(Clone, Debug)]
pub struct EpisodeLabels {
    /// The load-bearing `ApplicationId`s: `derivation_ancestors` of the
    /// chosen extraction's nodes. See the module doc's over-approximation
    /// note.
    pub load_bearing: BTreeSet<ApplicationId>,
    /// Binary label for every application recorded in the episode's
    /// provenance log (`labels.len() == provenance.application_count()`).
    pub labels: BTreeMap<ApplicationId, Label>,
    /// Per-rule aggregates, keyed by `rule_idx` (index into the e-graph's
    /// rule list at episode time).
    pub rule_stats: BTreeMap<usize, RuleStats>,
}

impl EpisodeLabels {
    /// Label every recorded rewrite application in `egraph`, given a chosen
    /// extraction: `root` and `choices` (a `(canonical EClassId) -> node_idx`
    /// map, indexed by class id — the representation produced by
    /// [`super::extract::extract_dag`] / [`super::extract::ExtractedDAG::choices`]
    /// / `IncrementalExtractor::extract_choices_only`).
    ///
    /// # Panics
    ///
    /// Panics if `choices` is missing an entry (or has an out-of-range node
    /// index) for a class actually reachable from `root` via the chosen
    /// nodes — the same invariant `choices_to_arena` enforces, and for the
    /// same reason: silently defaulting to node 0 here would fabricate a
    /// label for a node that was never really "chosen," corrupting the
    /// training signal instead of surfacing the extractor bug that produced
    /// an inconsistent `choices` map.
    pub fn compute(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> Self {
        let chosen_nodes = chosen_tagged_nodes(egraph, root, choices);
        let load_bearing = egraph.derivation_ancestors(&chosen_nodes);

        let mut labels = BTreeMap::new();
        let mut rule_stats: BTreeMap<usize, RuleStats> = BTreeMap::new();

        for (app_id, record) in egraph.provenance().applications() {
            let label = if load_bearing.contains(&app_id) {
                Label::LoadBearing
            } else {
                Label::Wasted
            };
            labels.insert(app_id, label);

            let stats = rule_stats.entry(record.rule_idx).or_default();
            stats.fired += 1;
            if label == Label::LoadBearing {
                stats.load_bearing += 1;
            }
        }

        Self {
            load_bearing,
            labels,
            rule_stats,
        }
    }

    /// Render a sorted, human-readable "which rules earn their keep" table:
    /// one row per rule that fired at least once, ordered by descending
    /// `load_bearing_ratio` (ties broken by `rule_idx` for determinism).
    ///
    /// Rule names are resolved via `egraph.rule(idx).name()`. If a rule
    /// index has no resolvable name against the given e-graph (e.g. the
    /// rule list changed since the episode ran), the row falls back to
    /// printing the raw index in `<rule N>` form rather than panicking or
    /// silently dropping the row.
    pub fn format_rule_report(&self, egraph: &EGraph) -> String {
        use std::fmt::Write;

        let mut rows: Vec<(usize, String, RuleStats)> = self
            .rule_stats
            .iter()
            .map(|(&rule_idx, &stats)| {
                let name = egraph
                    .rule(rule_idx)
                    .map(|r| r.name().to_string())
                    .unwrap_or_else(|| format!("<rule {rule_idx}>"));
                (rule_idx, name, stats)
            })
            .collect();

        rows.sort_by(|a, b| {
            b.2.load_bearing_ratio()
                .partial_cmp(&a.2.load_bearing_ratio())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let mut out = String::new();
        writeln!(
            &mut out,
            "{:<28} {:>6} {:>13} {:>7}",
            "rule", "fired", "load-bearing", "ratio"
        )
        .expect("format_rule_report: writing to String cannot fail");
        for (_, name, stats) in rows {
            writeln!(
                &mut out,
                "{:<28} {:>6} {:>13} {:>6.1}%",
                name,
                stats.fired,
                stats.load_bearing,
                stats.load_bearing_ratio() * 100.0,
            )
            .expect("format_rule_report: writing to String cannot fail");
        }
        out
    }
}

/// Walk the chosen extraction from `root`, following only the chosen node's
/// children at each class (unlike blindly reading every `Some` entry out of
/// `choices` — the DP that fills `choices` computes costs for every class
/// reachable via *any* candidate node while scoring alternatives, so most
/// `choices` vectors have many `Some` entries for classes never actually
/// used by the winning extraction). Returns the `(EClassId, ENodeId)` pairs
/// [`EGraph::derivation_ancestors`] expects.
fn chosen_tagged_nodes(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Vec<(EClassId, ENodeId)> {
    let mut visited: BTreeSet<EClassId> = BTreeSet::new();
    let mut stack: Vec<EClassId> = vec![root];
    let mut result = Vec::new();

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }

        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "chosen_tagged_nodes: e-class {} is reachable from root {} via the chosen \
                 extraction but has no recorded choice — `choices` must guarantee every \
                 class reachable via chosen nodes has Some(idx)",
                idx,
                root.index()
            )
        });

        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "chosen_tagged_nodes: node_idx {} out of bounds ({}) for e-class {}",
            node_idx,
            nodes.len(),
            idx
        );
        let tags = egraph.tags(canonical);
        result.push((canonical, tags[node_idx]));

        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }

    result
}

/// Result of running the full episode pipeline once: the extraction the
/// e-graph produced, the hindsight labels derived from it, and the e-graph
/// itself (kept around so callers can further inspect provenance — e.g.
/// `format_derivation_trace` on specific nodes of interest — after the fact).
///
/// No `Debug` derive: [`EGraph`] itself does not implement `Debug` (it holds
/// `dyn Rewrite` trait objects), so this type can't either without a manual
/// impl that skips `egraph` — not worth it for a struct callers can already
/// inspect field-by-field.
#[derive(Clone)]
pub struct EpisodeResult {
    /// The saturated, provenance-tracked e-graph.
    pub egraph: EGraph,
    /// The extraction chosen from it (latency-prior cost model).
    pub extraction: ExtractedDAG,
    /// Hindsight labels derived from `extraction` over `egraph`.
    pub labels: EpisodeLabels,
}

/// Run the whole episode pipeline on one expression: saturate → extract
/// (latency-prior [`CostModel`]) → label.
///
/// Kept deliberately simple and synchronous — this is the seed of the
/// future episode collector, not the collector itself. Callers supply the
/// rule set explicitly (mirroring [`EGraph::with_rules`] everywhere else in
/// this crate); use [`super::all_rules`] for the full library.
pub fn run_episode(
    arena: &pixelflow_ir::ExprArena,
    root: pixelflow_ir::ExprId,
    rules: Vec<Box<dyn Rewrite>>,
) -> EpisodeResult {
    let mut egraph = EGraph::with_rules(rules);
    let root_class = egraph.add_arena(arena, root);
    egraph.saturate();

    let costs = CostModel::latency_prior();
    let extraction = extract::extract_dag(&egraph, root_class, &costs);
    let labels = EpisodeLabels::compute(&egraph, extraction.root, &extraction.choices);

    EpisodeResult {
        egraph,
        extraction,
        labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::ops;
    use crate::egraph::provenance::Origin;
    use crate::math::algebra::Commutative;

    /// Minimal rule set for hand-derivable tests: just commutativity of Add,
    /// so every application in these tests is unambiguous.
    fn egraph_with_commutative() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = vec![Commutative::new(&ops::Add)];
        EGraph::with_rules(rules)
    }

    /// Find the tag (within `class`) whose origin is `Origin::Rule(app_id)`.
    fn tag_created_by(egraph: &EGraph, class: EClassId, app_id: ApplicationId) -> ENodeId {
        egraph
            .tags(class)
            .iter()
            .copied()
            .find(|&t| egraph.provenance().origin(t) == Some(Origin::Rule(app_id)))
            .unwrap_or_else(|| panic!("no node in class {class:?} was created by {app_id:?}"))
    }

    /// (a) Hand-derivable case: one rule application is provably
    /// load-bearing (its product is the chosen extraction), a second
    /// application of the *same* rule, fired against a wholly disjoint
    /// expression never reachable from the chosen root, is provably wasted.
    #[test]
    fn hand_derivable_load_bearing_vs_wasted() {
        let mut eg = egraph_with_commutative();

        // Chosen side: x + y, then commute to y + x — this IS the extraction.
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let target = eg
            .find_rewrite_matches()
            .into_iter()
            .find(|t| t.class_id == eg.find(sum))
            .expect("commutative should match x + y");
        assert!(eg.apply_single_rule(target.rule_idx, target.class_id, target.node_idx));
        let app_load_bearing = ApplicationId(0);

        // Disjoint side: z + w, commuted too, but never referenced by the
        // chosen root (`sum`) or any of its (transitive) children.
        let z = eg.add(ENode::Var(2));
        let w = eg.add(ENode::Var(3));
        let other_sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![z, w],
        });
        let other_target = eg
            .find_rewrite_matches()
            .into_iter()
            .find(|t| t.class_id == eg.find(other_sum))
            .expect("commutative should match z + w");
        assert!(eg.apply_single_rule(
            other_target.rule_idx,
            other_target.class_id,
            other_target.node_idx
        ));
        let app_wasted = ApplicationId(1);
        assert_eq!(eg.provenance().application_count(), 2);

        // Build the chosen extraction by hand: root = sum's class, choosing
        // the rule-created (commuted) node for `sum`, and node 0 (the only
        // node) for the leaves. `other_sum`/z/w are left as `None` — they
        // are unreachable from `sum`, so `chosen_tagged_nodes` must never
        // need them.
        let sum_class = eg.find(sum);
        let commuted_tag = tag_created_by(&eg, sum_class, app_load_bearing);
        let commuted_idx = eg
            .tags(sum_class)
            .iter()
            .position(|&t| t == commuted_tag)
            .unwrap();

        let mut choices: Vec<Option<usize>> = vec![None; eg.num_classes()];
        choices[eg.find(x).index()] = Some(0);
        choices[eg.find(y).index()] = Some(0);
        choices[sum_class.index()] = Some(commuted_idx);

        let labels = EpisodeLabels::compute(&eg, sum_class, &choices);

        assert_eq!(labels.load_bearing, BTreeSet::from([app_load_bearing]));
        assert_eq!(labels.labels[&app_load_bearing], Label::LoadBearing);
        assert_eq!(labels.labels[&app_wasted], Label::Wasted);

        let stats = labels.rule_stats[&target.rule_idx];
        assert_eq!(stats.fired, 2);
        assert_eq!(stats.load_bearing, 1);
        assert_eq!(stats.wasted(), 1);

        let report = labels.format_rule_report(&eg);
        assert!(report.contains("commutative"));
    }

    /// (b) Chain case: rule A's product is consumed by rule B's match, and
    /// B's product is the chosen node. Both applications must be labeled
    /// load-bearing.
    #[test]
    fn chain_both_applications_load_bearing() {
        let mut eg = egraph_with_commutative();

        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let inner = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        }); // x + y
        let outer = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![inner, z],
        }); // (x + y) + z

        // Rule A: commute the inner sum.
        let target_a = eg
            .find_rewrite_matches()
            .into_iter()
            .find(|t| t.class_id == eg.find(inner))
            .expect("commutative should match x + y");
        assert!(eg.apply_single_rule(target_a.rule_idx, target_a.class_id, target_a.node_idx));
        let app_a = ApplicationId(0);

        // Rule B: commute the outer sum. `inner`'s class now holds two
        // nodes (both "commutative" matches), so target specifically by
        // outer's class.
        let target_b = eg
            .find_rewrite_matches()
            .into_iter()
            .find(|t| t.class_id == eg.find(outer))
            .expect("commutative should match (x + y) + z");
        assert!(eg.apply_single_rule(target_b.rule_idx, target_b.class_id, target_b.node_idx));
        let app_b = ApplicationId(1);
        assert_eq!(eg.provenance().application_count(), 2);

        let outer_class = eg.find(outer);
        let inner_class = eg.find(inner);
        let b_product_tag = tag_created_by(&eg, outer_class, app_b);
        let b_product_idx = eg
            .tags(outer_class)
            .iter()
            .position(|&t| t == b_product_tag)
            .unwrap();

        // B's product is `z + (x + y)`: its children are z and inner's
        // class. Pick node 0 for z; for inner's class, pick the node A
        // created (so A's application is unambiguously exercised by the
        // chosen path too, not just pulled in via over-approximation).
        let a_product_tag = tag_created_by(&eg, inner_class, app_a);
        let a_product_idx = eg
            .tags(inner_class)
            .iter()
            .position(|&t| t == a_product_tag)
            .unwrap();

        let mut choices: Vec<Option<usize>> = vec![None; eg.num_classes()];
        choices[eg.find(x).index()] = Some(0);
        choices[eg.find(y).index()] = Some(0);
        choices[eg.find(z).index()] = Some(0);
        choices[inner_class.index()] = Some(a_product_idx);
        choices[outer_class.index()] = Some(b_product_idx);

        let labels = EpisodeLabels::compute(&eg, outer_class, &choices);

        assert_eq!(labels.labels[&app_a], Label::LoadBearing);
        assert_eq!(labels.labels[&app_b], Label::LoadBearing);
        assert!(labels.load_bearing.contains(&app_a));
        assert!(labels.load_bearing.contains(&app_b));

        let stats = labels.rule_stats[&target_b.rule_idx];
        assert_eq!(stats.fired, 2);
        assert_eq!(stats.load_bearing, 2);
        assert_eq!(stats.wasted(), 0);
    }

    /// (c) Aggregate counts sum correctly on a real saturation episode:
    /// fired == load_bearing + wasted per rule, and the per-rule totals
    /// reconcile against the flat label map / provenance log.
    #[test]
    fn aggregate_counts_reconcile() {
        use pixelflow_ir::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(pixelflow_ir::OpKind::Add, x, y);
        let doubled = arena.push_binary(pixelflow_ir::OpKind::Mul, sum, sum);
        let root = arena.push_binary(pixelflow_ir::OpKind::Sub, doubled, doubled);

        let result = run_episode(&arena, root, crate::egraph::all_rules());

        let total_fired: usize = result.labels.rule_stats.values().map(|s| s.fired).sum();
        let total_load_bearing: usize = result
            .labels
            .rule_stats
            .values()
            .map(|s| s.load_bearing)
            .sum();

        assert_eq!(total_fired, result.egraph.provenance().application_count());
        assert_eq!(total_fired, result.labels.labels.len());
        assert_eq!(
            total_load_bearing,
            result
                .labels
                .labels
                .values()
                .filter(|&&l| l == Label::LoadBearing)
                .count()
        );
        assert_eq!(
            total_load_bearing,
            result.labels.load_bearing.len().min(total_load_bearing)
        );

        for stats in result.labels.rule_stats.values() {
            assert_eq!(stats.fired, stats.load_bearing + stats.wasted());
        }

        // Report renders without panicking and lists every rule that fired.
        let report = result.labels.format_rule_report(&result.egraph);
        assert_eq!(report.lines().count(), 1 + result.labels.rule_stats.len());
    }
}
