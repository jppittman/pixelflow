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
//! [`derivation_ancestors_tight`] is a second, narrower walk over the same
//! records (`docs/plans/2026-08-31-guide-design-revision.md` §3, option 3):
//! it exists **alongside** `derivation_ancestors`, not in place of it, so both
//! can be computed on the same episode and compared. It narrows all three
//! named over-approximation axes but does not eliminate them — see its doc
//! comment for exactly what changed and what didn't.
//!
//! # Overhead
//!
//! Every hook (`record_origin`, `record_application`, `record_union`) is an
//! `O(1)` `Vec::push` / `HashMap::insert` — no scans, no unbounded work per
//! e-graph operation, so overhead scales linearly with the number of
//! creation/union events regardless of e-graph size or shape.
//!
//! Every rule match is recorded as an `ApplicationRecord` unconditionally in
//! `apply_action_from_rule` — including matches that ultimately produce no
//! net change (e.g. `Union` against an already-equal target) — trading a
//! larger provenance log for simpler, drift-proof bookkeeping (see that
//! function's doc comment).

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
    /// The exact firing that caused this union, if any — `Some` iff
    /// `rule_idx` is `Some` (both come from the same `active_application`
    /// read at the `EGraph::union` call site; `None` for congruence-closure
    /// unions found during rebuild, which have no active rule application).
    ///
    /// Purely additive: recorded because the information is already in hand
    /// (`EGraph::union` reads `self.active_application`, which carries both
    /// `rule_idx` and `application_id`) at zero extra cost, and exists to let
    /// [`derivation_ancestors_tight`] credit the *one* application that
    /// actually caused a union instead of `derivation_ancestors`'s
    /// `rule_idx` + `step <=` superset match. Does not change what
    /// `rule_idx` means or how the existing (loose) `derivation_ancestors`
    /// uses this event — that function never reads this field.
    pub application_id: Option<ApplicationId>,
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

    /// Iterate every recorded `(ENodeId, Origin)` pair, in arbitrary order
    /// (backed by a `HashMap`). Read-only counterpart to [`Self::applications`]
    /// / [`Self::union_events`] for callers that need the reverse direction —
    /// e.g. "which e-nodes did this `ApplicationId` create?" — without
    /// changing what gets recorded or how ancestry is computed. Added for the
    /// guided-saturation scoping measurements (`derivation_ancestors`'s
    /// over-approximation looseness, docs/plans/2026-07-07-guided-saturation-redesign.md
    /// lines 88-90): purely additive, no existing behavior touched.
    pub fn origins(&self) -> impl Iterator<Item = (ENodeId, Origin)> + '_ {
        self.origins.iter().map(|(&id, &origin)| (id, origin))
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

/// A second, narrower ancestry walk over the same provenance records as
/// [`derivation_ancestors`] — additive, not a replacement. Both functions are
/// meant to be run on the *same* episode so their outputs can be compared
/// directly; this one changes only what gets *credited*, not what
/// `Provenance` records or how `derivation_ancestors` computes its own
/// (unchanged) result.
///
/// # What's tightened, and what isn't
///
/// `derivation_ancestors`'s doc comment names three over-approximation axes.
/// This function narrows all three, but by construction rather than by a
/// true minimality/necessity proof (which would require replaying the
/// episode to ask "does removing application X still leave the chosen node
/// reachable" — out of scope for a "narrow, don't redesign" follow-up):
///
/// 1. **Child *nodes*, not child classes, wherever the choice is known.**
///    `chosen_nodes` is expected to be a *complete* per-class map — exactly
///    what `labeler::chosen_tagged_nodes` already builds by
///    recursively walking every class the winning extraction actually
///    visits, one entry per class. Given that map, whenever this walk
///    reaches a class it already has a known choice for (via
///    `children_of`'s class result, or via a union event's far side — see
///    axis 3 below), it follows *only* that one node instead of pulling
///    every node [`Provenance`] ever tagged into the class. For a class with
///    no known choice (the two hand-derivable unit tests below only supply a
///    partial map, on purpose, to exercise this path) it falls back to the
///    same "every tagged node" behavior `derivation_ancestors` always uses —
///    still safe, just not tightened for that class.
/// 2. **The exact firing, not every same-rule firing.** A [`UnionEvent`]'s
///    `application_id` (recorded for real at `EGraph::union`'s
///    call site, from the same `active_application` state that already
///    supplies `rule_idx` — see the field's doc comment) names the *one*
///    application that caused that merge. This walk credits exactly that
///    application; `derivation_ancestors` still credits every application
///    sharing the event's `rule_idx` with `step <= event.step`, unchanged.
/// 3. **Fixed-point pruning, as a consequence of #1, not a separate proof.**
///    Because a known-choice class only ever contributes its one node (#1),
///    the reachable set this walk explores is a strict subset of what
///    `derivation_ancestors` explores from the same input — it terminates on
///    a tighter fixed point without an explicit "was this merge necessary"
///    analysis. Union events are still walked outward from every visited
///    class exactly as `derivation_ancestors` does (a `Union`-shaped
///    `RewriteAction` creates no node of its
///    own, so union-event crediting is the *only* way such an application
///    can ever be labeled load-bearing at all — dropping it would violate
///    the safety property, not just tighten it).
///
/// # Safety (still an over-approximation, never omits a true ancestor)
///
/// Every application `derivation_ancestors` would credit via a node whose
/// class has a known choice is still credited here — the known-choice node
/// *is* the node that walk would eventually reach through that class's tags,
/// since `chosen_nodes` (as `labeler::chosen_tagged_nodes` builds it) always
/// includes the actually-extracted node. Every application credited via a
/// union event here is credited under a strictly narrower (exact-application)
/// condition than `derivation_ancestors`'s (same-`rule_idx`-and-earlier)
/// condition, which only shrinks the credited set for congruence-driven and
/// unrelated same-rule events, never a true one. Classes with no known choice
/// fall back to the identical unpruned behavior. So this function's result is
/// always a subset of `derivation_ancestors`'s result on the same inputs
/// (enforced at measurement time by an assertion analogous to
/// `guide_headroom`'s existing `strict_lb <= labeler_lb` check) and always a
/// superset of the strict bound
/// (every node directly on the chosen path is, by definition, in a
/// known-choice class, so #1 alone reduces to the strict node-level walk for
/// it; the only additions beyond strict are exact-application union credits,
/// which strict does not attempt at all).
///
/// # Panics
///
/// Panics if `chosen_nodes` names the same `EClassId` twice with two
/// *different* `ENodeId`s — a caller contract violation (the map is supposed
/// to name one chosen node per class), not a condition to paper over by
/// picking one arbitrarily.
pub fn derivation_ancestors_tight(
    tags_of: &impl Fn(EClassId) -> Vec<ENodeId>,
    children_of: &impl Fn(ENodeId) -> Vec<EClassId>,
    provenance: &Provenance,
    chosen_nodes: &[(EClassId, ENodeId)],
) -> BTreeSet<ApplicationId> {
    let mut chosen_map: HashMap<EClassId, ENodeId> = HashMap::new();
    for &(class, node) in chosen_nodes {
        match chosen_map.entry(class) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(node);
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                assert_eq!(
                    *e.get(),
                    node,
                    "derivation_ancestors_tight: class {class:?} was given two different \
                     chosen nodes ({:?} and {node:?}) — `chosen_nodes` must name at most one \
                     node per class",
                    e.get()
                );
            }
        }
    }

    // Axis 1: known-choice short-circuit. Falls back to `tags_of` (identical
    // to `derivation_ancestors`) for a class with no recorded choice.
    let nodes_to_follow = |class: EClassId| -> Vec<ENodeId> {
        match chosen_map.get(&class) {
            Some(&node) => vec![node],
            None => tags_of(class),
        }
    };

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

    // Follow one node's own creation + structural children — identical logic
    // to `derivation_ancestors`'s node walk (axis 1 only changes *which*
    // nodes get pushed onto `node_stack` in the first place, via
    // `nodes_to_follow` below).
    let follow_node = |node: ENodeId,
                       result: &mut BTreeSet<ApplicationId>,
                       visited_nodes: &mut BTreeSet<ENodeId>,
                       visited_classes: &mut BTreeSet<EClassId>,
                       class_stack: &mut Vec<EClassId>| {
        if !visited_nodes.insert(node) {
            return;
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
    };

    while let Some(node) = node_stack.pop() {
        follow_node(
            node,
            &mut result,
            &mut visited_nodes,
            &mut visited_classes,
            &mut class_stack,
        );
    }

    while let Some(class) = class_stack.pop() {
        for node in nodes_to_follow(class) {
            if !visited_nodes.contains(&node) {
                node_stack.push(node);
            }
        }
        while let Some(node) = node_stack.pop() {
            follow_node(
                node,
                &mut result,
                &mut visited_nodes,
                &mut visited_classes,
                &mut class_stack,
            );
        }

        for event in provenance.union_events() {
            if event.class_a == class || event.class_b == class {
                // Axis 2: credit the exact firing, not every same-rule_idx
                // application at or before this event's step.
                if let Some(app_id) = event.application_id {
                    result.insert(app_id);
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

    /// `derivation_ancestors_tight` on the same two hand-derivable inputs
    /// above must agree with `derivation_ancestors` — with only one tagged
    /// node per relevant class, axis 1's short-circuit changes nothing.
    #[test]
    fn tight_matches_loose_when_no_sibling_nodes() {
        let mut p = Provenance::new();
        let n0 = ENodeId(0);
        let n1 = ENodeId(1);
        p.record_origin(n0, Origin::Seed);
        let a0 = p.record_application(app(0, 0, EClassId(0)));
        p.record_origin(n1, Origin::Rule(a0));

        let tags_of =
            |c: EClassId| -> Vec<ENodeId> { if c == EClassId(0) { vec![n0] } else { vec![] } };
        let children_of = |_n: ENodeId| -> Vec<EClassId> { vec![] };

        let loose = derivation_ancestors(&tags_of, &children_of, &p, &[(EClassId(1), n1)]);
        let tight = derivation_ancestors_tight(&tags_of, &children_of, &p, &[(EClassId(1), n1)]);
        assert_eq!(loose, BTreeSet::from([a0]));
        assert_eq!(tight, loose);
    }

    /// Axis 1: a class holding the chosen node *and* an unrelated sibling
    /// node (from a totally disjoint derivation) is over-credited by the
    /// loose walk — which pulls every tag in a visited class — but not by
    /// the tight one, which follows only the class's known choice.
    #[test]
    fn tight_excludes_unrelated_sibling_in_same_class() {
        let mut p = Provenance::new();
        let root = EClassId(0);
        let n_chosen = ENodeId(0);
        let n_sibling = ENodeId(1);

        let a_good = p.record_application(app(1, 0, EClassId(5)));
        p.record_origin(n_chosen, Origin::Rule(a_good));
        let a_bad = p.record_application(app(2, 0, EClassId(6)));
        p.record_origin(n_sibling, Origin::Rule(a_bad));

        // Both nodes live tagged in `root`'s class, but only `n_chosen` is
        // what the extraction actually picked.
        let tags_of = |c: EClassId| -> Vec<ENodeId> {
            if c == root {
                vec![n_chosen, n_sibling]
            } else {
                vec![]
            }
        };
        let children_of = |_n: ENodeId| -> Vec<EClassId> { vec![] };

        let chosen = [(root, n_chosen)];
        let loose = derivation_ancestors(&tags_of, &children_of, &p, &chosen);
        let tight = derivation_ancestors_tight(&tags_of, &children_of, &p, &chosen);

        assert_eq!(
            loose,
            BTreeSet::from([a_good, a_bad]),
            "loose must over-credit the sibling application via axis 1 (all tags of a \
             visited class), exactly as documented"
        );
        assert_eq!(
            tight,
            BTreeSet::from([a_good]),
            "tight must follow only the known-choice node, excluding the unrelated sibling"
        );
    }

    /// Axis 2: two applications of the same rule (different firings)
    /// produce a `UnionEvent` each; only one of them is the actual cause of
    /// the union that connects the chosen class to another. The loose walk
    /// credits both (`rule_idx` match + `step <=`, a superset); the tight
    /// walk credits only the `application_id` recorded on the event.
    #[test]
    fn tight_credits_exact_union_application_not_every_same_rule_firing() {
        let mut p = Provenance::new();
        const RULE: usize = 5;
        let a_earlier = p.record_application(app(RULE, 0, EClassId(0)));
        let a_actual = p.record_application(app(RULE, 1, EClassId(1)));

        let root = EClassId(2);
        let other = EClassId(9);
        let n0 = ENodeId(0);
        p.record_origin(n0, Origin::Seed);
        p.record_union(UnionEvent {
            rule_idx: Some(RULE),
            application_id: Some(a_actual),
            step: 2,
            class_a: root,
            class_b: other,
        });

        let tags_of = |c: EClassId| -> Vec<ENodeId> { if c == root { vec![n0] } else { vec![] } };
        let children_of = |_n: ENodeId| -> Vec<EClassId> { vec![] };

        let chosen = [(root, n0)];
        let loose = derivation_ancestors(&tags_of, &children_of, &p, &chosen);
        let tight = derivation_ancestors_tight(&tags_of, &children_of, &p, &chosen);

        assert_eq!(
            loose,
            BTreeSet::from([a_earlier, a_actual]),
            "loose credits every same-rule_idx application at or before the event's step"
        );
        assert_eq!(
            tight,
            BTreeSet::from([a_actual]),
            "tight credits exactly the application recorded on the union event"
        );
    }

    /// A `Union`-shaped rewrite action creates no node, so the only way it
    /// can ever be labeled load-bearing — under either walk — is via the
    /// union event it produces. Confirms the tight walk still catches this
    /// case (narrowing axis 2 must not accidentally drop it to zero).
    #[test]
    fn tight_still_credits_union_only_application() {
        let mut p = Provenance::new();
        let a_union_only = p.record_application(app(3, 0, EClassId(0)));
        let root = EClassId(1);
        let other = EClassId(2);
        let n0 = ENodeId(0);
        p.record_origin(n0, Origin::Seed);
        p.record_union(UnionEvent {
            rule_idx: Some(3),
            application_id: Some(a_union_only),
            step: 0,
            class_a: root,
            class_b: other,
        });

        let tags_of = |c: EClassId| -> Vec<ENodeId> { if c == root { vec![n0] } else { vec![] } };
        let children_of = |_n: ENodeId| -> Vec<EClassId> { vec![] };
        let chosen = [(root, n0)];

        let tight = derivation_ancestors_tight(&tags_of, &children_of, &p, &chosen);
        assert_eq!(tight, BTreeSet::from([a_union_only]));
    }

    /// `chosen_nodes` naming the same class twice with two different nodes
    /// is a caller contract violation — must panic loudly, not silently pick
    /// one.
    #[test]
    #[should_panic(expected = "two different chosen nodes")]
    fn tight_panics_on_conflicting_chosen_nodes_for_one_class() {
        let p = Provenance::new();
        let c = EClassId(0);
        let n_a = ENodeId(0);
        let n_b = ENodeId(1);
        let tags_of = |_: EClassId| -> Vec<ENodeId> { vec![] };
        let children_of = |_: ENodeId| -> Vec<EClassId> { vec![] };
        let _ = derivation_ancestors_tight(&tags_of, &children_of, &p, &[(c, n_a), (c, n_b)]);
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
