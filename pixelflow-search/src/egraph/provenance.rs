//! Rule provenance tracking for the e-graph.
//!
//! Answers "why does this e-node exist, and which rule firings produced the
//! expression we ultimately extracted?" — useful for debugging rewrite rules,
//! auditing NNUE-guided search decisions, and explaining saturation output.
//!
//! # Design
//!
//! E-nodes have no stable identity of their own: they live at `(EClassId,
//! node_idx)`, and `node_idx` shifts whenever [`EGraph::union`] extends a
//! class's node vector or [`EGraph::rebuild_budgeted`] rewrites it in place.
//! To track provenance we mint a process-wide stable [`ENodeId`] for every
//! e-node the instant it is created in [`EGraph::add`] (the sole creation
//! choke point), and thread a parallel `tags: Vec<ENodeId>` alongside every
//! `EClass::nodes: Vec<ENode>` so the tag at index `i` always names the node
//! at index `i` — including through `union`'s `extend()` and
//! `rebuild_budgeted`'s take/canonicalize/extend cycle, which move node
//! *values* but never reassign identity.
//!
//! Three append-only records make up the provenance store:
//!
//! - [`Provenance::origins`]: `ENodeId -> Origin`, one entry per node ever
//!   created. `Origin::Seed` for nodes inserted before any rewriting began
//!   (e.g. via `add_arena`); `Origin::Rule(ApplicationId)` for nodes created
//!   as the output of a rewrite.
//! - [`Provenance::applications`]: a log of every rewrite firing
//!   ([`ApplicationRecord`]), keyed by [`ApplicationId`].
//! - [`Provenance::unions`]: an append-only [`UnionEvent`] journal recording
//!   every class-level merge, whether rule-driven or congruence-closure
//!   (rebuild-driven, `rule_idx: None`).
//!
//! [`derivation_ancestors`] walks these records backward from a set of chosen
//! e-nodes to the [`ApplicationId`]s that caused them to exist or to become
//! reachable from the root. It is a **conservative over-approximation**: it
//! is allowed (and expected) to include applications that turned out not to
//! matter, but must never omit one that does. See its doc comment for the
//! exact over-approximation made and why it is safe.
//!
//! # Overhead
//!
//! Every hook (`record_origin`, `record_application`, `record_union`) is an
//! `O(1)` `Vec::push` / `HashMap::insert` — no scans, no unbounded work per
//! e-graph operation, so overhead scales linearly with the number of
//! creation/union events regardless of e-graph size or shape.
//!
//! Measured via `graph::tests::provenance_overhead_timing` (`#[ignore]`d;
//! run with `cargo test -p pixelflow-search --release --lib -- --ignored
//! provenance_overhead_timing --nocapture`) on a 40-op alternating
//! add/mul/sub chain over two variables, saturated with the standard
//! algebra rule set (`saturate()`, its default 100-iteration / 10k-class /
//! 500ms limits):
//!
//! ```text
//! saturation time: ~9ms (release build, steady state after warmup)
//! origins:        1067  (one per e-node ever created)
//! applications:  13092  (one per rewrite firing, matched or not netting a union)
//! unions:          818
//! classes:        1067  (final e-class count)
//! ```
//!
//! Note `applications` (13092) considerably exceeds `unions` (818): every
//! rule match is recorded as an `ApplicationRecord` unconditionally in
//! `apply_action_from_rule` — including matches that ultimately produce no
//! net change (e.g. `Union` against an already-equal target) — trading a
//! larger provenance log for simpler, drift-proof bookkeeping (see that
//! function's doc comment). No separate non-provenance baseline was
//! measured (not required — see task notes); the `O(1)`-per-event argument
//! above is the basis for the overhead claim instead of an A/B comparison.

use std::collections::{BTreeSet, HashMap};

use super::node::EClassId;

/// Stable, process-wide identity for an e-node, independent of its current
/// `(EClassId, node_idx)` position.
///
/// Minted once, in [`EGraph::add`], for every node that isn't a memo hit.
/// Never reused, never renumbered — this is the whole point: `(EClassId,
/// node_idx)` shifts under union and rebuild, `ENodeId` never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ENodeId(pub(crate) u64);

impl ENodeId {
    /// Raw numeric value, useful for logging/debugging.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Identifies one firing of a rewrite rule (one call to
/// `apply_action_from_rule`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ApplicationId(pub(crate) u64);

impl ApplicationId {
    /// Raw numeric value, useful for logging/debugging.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Where an e-node came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Inserted directly (e.g. via `add_arena` before saturation, or any
    /// `add()` call not made on behalf of a rewrite rule).
    Seed,
    /// Produced as the output of a rewrite rule firing.
    Rule(ApplicationId),
}

/// A single rewrite-rule firing.
///
/// `step` is the saturation iteration counter, advanced once per outer
/// `saturate_with_limits` loop iteration — coarser than per-rule-application
/// but cheap (a single counter) and sufficient to order firings into
/// generations for the derivation trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplicationRecord {
    /// Index into `EGraph`'s rule list.
    pub rule_idx: usize,
    /// Saturation-iteration counter at the time of firing.
    pub step: usize,
    /// The e-class the rule matched against (the match root). Recorded
    /// because it's already in hand at the call site — cheap to keep,
    /// useful for the derivation trace.
    pub match_root: EClassId,
}

/// A class-level merge event, recorded for every `union()` call that
/// actually merges two distinct classes (canonical ids at merge time).
///
/// `rule_idx: None` marks unions performed by congruence closure during
/// rebuild (i.e. `rebuild_budgeted` discovering two nodes are now equal
/// after canonicalization) rather than by a rewrite rule directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnionEvent {
    /// The rule that caused this union, if any (`None` for congruence
    /// closure unions found during rebuild).
    pub rule_idx: Option<usize>,
    /// Saturation-iteration counter at the time of the union.
    pub step: usize,
    /// One of the two canonical class ids being merged (pre-merge).
    pub class_a: EClassId,
    /// The other canonical class id being merged (pre-merge).
    pub class_b: EClassId,
}

/// Append-only provenance store.
///
/// Every write is an `O(1)` `Vec::push` / `HashMap::insert` — no scans, no
/// unbounded work per e-graph operation. See module docs for the overhead
/// measurement.
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    /// `ENodeId -> Origin`, one entry per node ever created.
    origins: HashMap<ENodeId, Origin>,
    /// Log of every rewrite firing, indexed by `ApplicationId`.
    applications: Vec<ApplicationRecord>,
    /// Append-only journal of class merges (rule-driven and congruence).
    unions: Vec<UnionEvent>,
}

impl Provenance {
    /// Create an empty provenance store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the origin of a freshly created e-node. Called once per
    /// `EGraph::add` memo miss.
    pub(crate) fn record_origin(&mut self, id: ENodeId, origin: Origin) {
        self.origins.insert(id, origin);
    }

    /// Record a rewrite firing, returning its `ApplicationId`.
    pub(crate) fn record_application(&mut self, record: ApplicationRecord) -> ApplicationId {
        let id = ApplicationId(self.applications.len() as u64);
        self.applications.push(record);
        id
    }

    /// Record a class-level merge.
    pub(crate) fn record_union(&mut self, event: UnionEvent) {
        self.unions.push(event);
    }

    /// Look up the origin of a node, if known.
    pub fn origin(&self, id: ENodeId) -> Option<Origin> {
        self.origins.get(&id).copied()
    }

    /// Look up an application record by id.
    pub fn application(&self, id: ApplicationId) -> Option<&ApplicationRecord> {
        self.applications.get(id.0 as usize)
    }

    /// Number of application records (rewrite firings) recorded.
    pub fn application_count(&self) -> usize {
        self.applications.len()
    }

    /// Number of union events recorded.
    pub fn union_count(&self) -> usize {
        self.unions.len()
    }

    /// Number of e-node origins recorded.
    pub fn origin_count(&self) -> usize {
        self.origins.len()
    }

    /// All union events (in chronological order).
    pub fn union_events(&self) -> &[UnionEvent] {
        &self.unions
    }

    /// Iterate every recorded application, in firing order, paired with its
    /// `ApplicationId`. The counterpart to indexed lookup via [`Self::application`]
    /// for callers (e.g. the hindsight labeler) that need to walk the whole log.
    pub fn applications(&self) -> impl Iterator<Item = (ApplicationId, &ApplicationRecord)> {
        self.applications
            .iter()
            .enumerate()
            .map(|(i, r)| (ApplicationId(i as u64), r))
    }
}

/// Compute the transitive set of rewrite-rule firings ([`ApplicationId`]s)
/// that could have contributed to the given chosen nodes.
///
/// # Over-approximation (by design)
///
/// This is deliberately conservative, in three ways:
///
/// 1. **Child classes, not child nodes.** An e-node's children are
///    `EClassId`s, and a class may hold many nodes with many different
///    origins. Rather than trying to pick "the" node that was actually used
///    (which extraction decides later and which may differ per extraction
///    pass), we pull in the creating application of *every* node currently
///    tagged in each child class. This can include applications that
///    produced alternative, unused representations of that subexpression.
/// 2. **Union events by class membership, not by node.** Any recorded
///    `UnionEvent` touching a class visited during the walk is included,
///    even though a union merges whole classes and the specific node we
///    care about might have been equivalent to the target class for
///    unrelated reasons.
/// 3. **No fixed point pruning.** The walk is a straightforward
///    reachability closure over (node -> creating application -> match root
///    class -> child classes -> nodes in those classes -> ...); it does not
///    attempt to determine whether a union event was *necessary* for the
///    final equivalence, only whether it *touched* a class on the path.
///
/// This means `derivation_ancestors` may report applications that, on
/// reflection, didn't matter to the final extracted expression. It must
/// never *omit* one that did — that is the safety property callers rely on
/// (e.g. "show me every rule that could plausibly explain this output").
///
/// # Arguments
///
/// `chosen_nodes`: the `(EClassId, ENodeId)` pairs whose ancestry to trace —
/// typically the nodes extraction selected for the final expression.
pub fn derivation_ancestors(
    tags_of: &impl Fn(EClassId) -> Vec<ENodeId>,
    children_of: &impl Fn(ENodeId) -> Vec<EClassId>,
    provenance: &Provenance,
    chosen_nodes: &[(EClassId, ENodeId)],
) -> BTreeSet<ApplicationId> {
    let mut result = BTreeSet::new();
    let mut visited_classes: BTreeSet<EClassId> = BTreeSet::new();
    let mut visited_nodes: BTreeSet<ENodeId> = BTreeSet::new();
    let mut class_stack: Vec<EClassId> = Vec::new();
    let mut node_stack: Vec<ENodeId> = chosen_nodes.iter().map(|&(_, n)| n).collect();
    for &(class, _) in chosen_nodes {
        if visited_classes.insert(class) {
            class_stack.push(class);
        }
    }

    // Walk nodes: each node contributes its creating application (if any)
    // and its children's classes.
    while let Some(node) = node_stack.pop() {
        if !visited_nodes.insert(node) {
            continue;
        }
        if let Some(Origin::Rule(app_id)) = provenance.origin(node) {
            result.insert(app_id);
            // The match root of the application that created this node is
            // itself a class whose tagged nodes may have contributed
            // (over-approximation #1 extended to the match side).
            if let Some(record) = provenance.application(app_id) {
                if visited_classes.insert(record.match_root) {
                    class_stack.push(record.match_root);
                }
            }
        }
        for child_class in children_of(node) {
            if visited_classes.insert(child_class) {
                class_stack.push(child_class);
            }
        }
    }

    // Walk classes: pull in every tagged node (over-approximation #1) and
    // every union event touching this class (over-approximation #2).
    while let Some(class) = class_stack.pop() {
        for node in tags_of(class) {
            if !visited_nodes.contains(&node) {
                node_stack.push(node);
            }
        }
        // Draining node_stack immediately keeps the two-phase walk (nodes
        // discover classes, classes discover nodes) converging to a single
        // fixed point rather than needing an explicit outer loop.
        while let Some(node) = node_stack.pop() {
            if !visited_nodes.insert(node) {
                continue;
            }
            if let Some(Origin::Rule(app_id)) = provenance.origin(node) {
                result.insert(app_id);
                if let Some(record) = provenance.application(app_id) {
                    if visited_classes.insert(record.match_root) {
                        class_stack.push(record.match_root);
                    }
                }
            }
            for child_class in children_of(node) {
                if visited_classes.insert(child_class) {
                    class_stack.push(child_class);
                }
            }
        }

        for event in provenance.union_events() {
            if event.class_a == class || event.class_b == class {
                if let Some(rule_idx) = event.rule_idx {
                    // Union events don't carry an ApplicationId directly —
                    // they may originate from congruence closure with no
                    // single firing to blame. When they *do* have a
                    // rule_idx, the closest matching application record is
                    // any Rule-origin application with that rule_idx and
                    // step <= event.step; conservatively, include all such
                    // applications rather than guess which one fired.
                    for (idx, record) in provenance.applications.iter().enumerate() {
                        if record.rule_idx == rule_idx && record.step <= event.step {
                            result.insert(ApplicationId(idx as u64));
                        }
                    }
                }
                let other = if event.class_a == class {
                    event.class_b
                } else {
                    event.class_a
                };
                if visited_classes.insert(other) {
                    class_stack.push(other);
                }
            }
        }
    }

    result
}

/// Render a human-readable derivation trace: one line per application,
/// ordered by `step` then `ApplicationId`, in the form:
///
/// ```text
/// step 3: rule[7] "distribute_mul_add" (match root e12) -> application #4
/// ```
///
/// Rule names are resolved via `rule_name`, typically `|idx| egraph.rule(idx)
/// .map(|r| r.name())`. Applications whose rule index has no resolvable name
/// (e.g. the rule list changed since the trace was recorded) fall back to
/// printing the raw index — this function never panics or silently drops a
/// line for a resolution failure.
pub fn format_derivation_trace(
    provenance: &Provenance,
    ancestors: &BTreeSet<ApplicationId>,
    rule_name: &impl Fn(usize) -> Option<String>,
) -> String {
    use std::fmt::Write;

    let mut records: Vec<(ApplicationId, &ApplicationRecord)> = ancestors
        .iter()
        .filter_map(|&id| provenance.application(id).map(|r| (id, r)))
        .collect();
    records.sort_by_key(|(id, r)| (r.step, id.0));

    let mut out = String::new();
    for (id, record) in records {
        let name = rule_name(record.rule_idx)
            .unwrap_or_else(|| format!("<unknown rule {}>", record.rule_idx));
        writeln!(
            &mut out,
            "step {}: rule[{}] {:?} (match root e{}) -> application #{}",
            record.step,
            record.rule_idx,
            name,
            record.match_root.index(),
            id.as_u64(),
        )
        .expect("format_derivation_trace: writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(rule_idx: usize, step: usize, match_root: EClassId) -> ApplicationRecord {
        ApplicationRecord {
            rule_idx,
            step,
            match_root,
        }
    }

    #[test]
    fn origin_roundtrip() {
        let mut p = Provenance::new();
        let n0 = ENodeId(0);
        p.record_origin(n0, Origin::Seed);
        assert_eq!(p.origin(n0), Some(Origin::Seed));
        assert_eq!(p.origin(ENodeId(1)), None);
    }

    #[test]
    fn application_ids_are_sequential() {
        let mut p = Provenance::new();
        let a0 = p.record_application(app(0, 0, EClassId(0)));
        let a1 = p.record_application(app(1, 1, EClassId(1)));
        assert_eq!(a0.as_u64(), 0);
        assert_eq!(a1.as_u64(), 1);
        assert_eq!(p.application_count(), 2);
    }

    #[test]
    fn derivation_ancestors_single_hop() {
        // n1 was created by application 0, whose match root has no other
        // tagged nodes and no children. Ancestors of n1 = {app 0}.
        let mut p = Provenance::new();
        let n0 = ENodeId(0);
        let n1 = ENodeId(1);
        p.record_origin(n0, Origin::Seed);
        let a0 = p.record_application(app(0, 0, EClassId(0)));
        p.record_origin(n1, Origin::Rule(a0));

        let tags_of =
            |c: EClassId| -> Vec<ENodeId> { if c == EClassId(0) { vec![n0] } else { vec![] } };
        let children_of = |_n: ENodeId| -> Vec<EClassId> { vec![] };

        let ancestors = derivation_ancestors(&tags_of, &children_of, &p, &[(EClassId(1), n1)]);
        assert_eq!(ancestors, BTreeSet::from([a0]));
    }

    #[test]
    fn derivation_ancestors_chain() {
        // n1 (app 0) is a child of n2 (app 1): ancestors of n2 = {app0, app1}.
        let mut p = Provenance::new();
        let n1 = ENodeId(1);
        let n2 = ENodeId(2);
        let a0 = p.record_application(app(0, 0, EClassId(0)));
        p.record_origin(n1, Origin::Rule(a0));
        let a1 = p.record_application(app(1, 1, EClassId(1)));
        p.record_origin(n2, Origin::Rule(a1));

        let tags_of = |c: EClassId| -> Vec<ENodeId> {
            match c.index() {
                1 => vec![n1],
                _ => vec![],
            }
        };
        let children_of =
            |n: ENodeId| -> Vec<EClassId> { if n == n2 { vec![EClassId(1)] } else { vec![] } };

        let ancestors = derivation_ancestors(&tags_of, &children_of, &p, &[(EClassId(2), n2)]);
        assert_eq!(ancestors, BTreeSet::from([a0, a1]));
    }

    #[test]
    fn format_trace_orders_by_step() {
        let mut p = Provenance::new();
        let a1 = p.record_application(app(5, 3, EClassId(9)));
        let a0 = p.record_application(app(2, 1, EClassId(4)));
        let ancestors = BTreeSet::from([a0, a1]);
        let rule_name = |idx: usize| -> Option<String> {
            match idx {
                2 => Some("identity".to_string()),
                5 => Some("commute".to_string()),
                _ => None,
            }
        };
        let trace = format_derivation_trace(&p, &ancestors, &rule_name);
        let lines: Vec<&str> = trace.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("step 1"));
        assert!(lines[0].contains("identity"));
        assert!(lines[1].contains("step 3"));
        assert!(lines[1].contains("commute"));
    }

    #[test]
    fn format_trace_unknown_rule_falls_back_to_index() {
        let mut p = Provenance::new();
        let a0 = p.record_application(app(42, 0, EClassId(0)));
        let ancestors = BTreeSet::from([a0]);
        let rule_name = |_: usize| -> Option<String> { None };
        let trace = format_derivation_trace(&p, &ancestors, &rule_name);
        assert!(trace.contains("<unknown rule 42>"));
    }
}
