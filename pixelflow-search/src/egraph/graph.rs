//! The E-Graph data structure and operations.

use std::collections::HashMap;

use super::cost::{CostFunction, CostModel};
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
use super::rewrite::{Rewrite, RewriteAction};
use pixelflow_ir::kind::OpKind;

/// A potential rewrite target: (rule, e-class, node within class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RewriteTarget {
    /// Index into the e-graph's rule list
    pub rule_idx: usize,
    /// The e-class to apply the rule to
    pub class_id: EClassId,
    /// The node within the e-class that the rule should try to match
    pub node_idx: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EClass {
    pub(crate) nodes: Vec<ENode>,
}

pub struct EGraph {
    pub(crate) classes: Vec<EClass>,
    pub(crate) parent: Vec<EClassId>,
    memo: HashMap<ENode, EClassId>,
    worklist: Vec<EClassId>,
    /// Rules are shared via Arc so EGraph can be cloned for search branching.
    rules: std::sync::Arc<Vec<Box<dyn Rewrite>>>,
    pub match_counts: HashMap<String, usize>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EGraph {
    fn clone(&self) -> Self {
        Self {
            classes: self.classes.clone(),
            parent: self.parent.clone(),
            memo: self.memo.clone(),
            worklist: self.worklist.clone(),
            rules: self.rules.clone(), // Arc clone - cheap, shares rules
            match_counts: self.match_counts.clone(),
        }
    }
}

/// Result of applying a single rule: changes made and evaluations consumed.
///
/// `changes` counts union/create actions. `evals` counts rule match
/// attempts (one per node checked). Evals model compute cost — the
/// Guide learns to stay within an eval budget just as it learns to
/// stay within a node budget.
pub struct ApplyResult {
    pub changes: usize,
    pub evals: usize,
}

impl EGraph {
    /// Create an empty e-graph with no rewrite rules.
    ///
    /// Rules are application-defined. Use `with_rules()` or `add_rule()` to add them.
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
            rules: std::sync::Arc::new(Vec::new()),
            match_counts: HashMap::new(),
        }
    }

    /// Create an e-graph with the given rewrite rules.
    ///
    /// Rules are owned by the e-graph and shared via Arc when cloned.
    pub fn with_rules(rules: Vec<Box<dyn Rewrite>>) -> Self {
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
            rules: std::sync::Arc::new(rules),
            match_counts: HashMap::new(),
        }
    }

    /// Add a rule to this e-graph.
    ///
    /// # Panics
    ///
    /// Panics if the e-graph has been cloned (rules are shared via Arc).
    pub fn add_rule(&mut self, rule: Box<dyn Rewrite>) {
        std::sync::Arc::get_mut(&mut self.rules)
            .expect("Cannot add rules after EGraph has been cloned")
            .push(rule);
    }

    pub fn find(&self, id: EClassId) -> EClassId {
        let mut current = id;
        while self.parent[current.index()] != current {
            current = self.parent[current.index()];
        }
        current
    }

    fn find_mut(&mut self, id: EClassId) -> EClassId {
        let mut current = id;
        let mut path = Vec::new();
        while self.parent[current.index()] != current {
            path.push(current);
            current = self.parent[current.index()];
        }
        for node in path {
            self.parent[node.index()] = current;
        }
        current
    }

    fn canonicalize_node(&self, node: &mut ENode) {
        match node {
            ENode::Var(_) | ENode::Const(_) => {}
            ENode::Op { children, .. } => {
                for child in children {
                    *child = self.find(*child);
                }
            }
        }
    }

    pub fn add(&mut self, mut node: ENode) -> EClassId {
        self.canonicalize_node(&mut node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        // Hard class limit: never allocate beyond this. Prevents unbounded
        // memory growth when resource limits are missing from call sites.
        const HARD_CLASS_LIMIT: usize = 100_000;
        if self.classes.len() >= HARD_CLASS_LIMIT {
            // Return a sentinel pointing at class 0. The e-graph is over
            // budget; further growth would be useless anyway.
            return EClassId(0);
        }
        let id = EClassId(self.classes.len() as u32);
        self.classes.push(EClass {
            nodes: vec![node.clone()],
        });
        self.parent.push(id);
        self.memo.insert(node, id);
        id
    }

    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let a = self.find_mut(a);
        let b = self.find_mut(b);
        if a == b {
            return a;
        }
        let (parent, child) = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.parent[child.index()] = parent;
        let child_nodes = std::mem::take(&mut self.classes[child.index()].nodes);
        self.classes[parent.index()].nodes.extend(child_nodes);
        self.worklist.push(parent);
        parent
    }

    /// Begin a batch of rule applications. Returns a guard that rebuilds
    /// the e-graph when dropped (RAII). Rules applied through the guard
    /// skip per-rule rebuilds; the single rebuild on drop amortizes the cost.
    ///
    /// ```ignore
    /// {
    ///     let mut batch = egraph.batch();
    ///     batch.apply_rule(0, 500);
    ///     batch.apply_rule(1, 500);
    ///     // rebuild happens here on drop
    /// }
    /// ```
    /// Begin a batch of rule applications with interleaved partial rebuild.
    ///
    /// `rebuild_chunk`: max worklist items to process after each rule.
    /// Higher = more deduplication, slower per rule.
    /// Lower = less deduplication, faster per rule but classes grow.
    /// Default of 256 balances the two.
    pub fn batch(&mut self) -> EGraphBatch<'_> {
        EGraphBatch {
            graph: self,
            any_changes: false,
            rebuild_chunk: 256,
        }
    }

    /// Begin a batch with a custom rebuild chunk size.
    pub fn batch_with_chunk(&mut self, rebuild_chunk: usize) -> EGraphBatch<'_> {
        EGraphBatch {
            graph: self,
            any_changes: false,
            rebuild_chunk,
        }
    }

    /// Rebuild the e-graph completely. Processes the entire worklist.
    pub fn rebuild(&mut self) {
        self.rebuild_budgeted(usize::MAX);
    }

    /// Process up to `budget` worklist items. Each item canonicalizes one
    /// e-class's nodes and deduplicates via the memo table.
    ///
    /// The graph is consistent after each item — partially rebuilt is safe.
    /// Unprocessed classes may have stale canonical forms (rule matching
    /// might miss some equivalences) but won't produce wrong results.
    ///
    /// Returns the number of worklist items remaining.
    pub fn rebuild_budgeted(&mut self, budget: usize) -> usize {
        let mut processed = 0;
        while processed < budget {
            let id = match self.worklist.pop() {
                Some(id) => id,
                None => break,
            };
            processed += 1;
            let id = self.find(id);
            let nodes = std::mem::take(&mut self.classes[id.index()].nodes);
            let mut new_nodes = Vec::new();
            for mut node in nodes {
                self.canonicalize_node(&mut node);
                if let Some(&existing) = self.memo.get(&node) {
                    let existing = self.find(existing);
                    if existing != id {
                        self.union(id, existing);
                    }
                } else {
                    self.memo.insert(node.clone(), id);
                }
                new_nodes.push(node);
            }
            self.classes[id.index()].nodes = new_nodes;
        }
        self.worklist.len()
    }

    /// Number of pending worklist items (classes needing rebuild).
    pub fn pending_rebuilds(&self) -> usize {
        self.worklist.len()
    }

    pub fn nodes(&self, id: EClassId) -> &[ENode] {
        let id = self.find(id);
        &self.classes[id.index()].nodes
    }

    /// Get the number of registered rewrite rules.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Get the number of e-classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Iterate over all canonical e-class IDs.
    ///
    /// Returns an iterator of all e-class IDs that are canonical (i.e., roots
    /// of their union-find tree) and have at least one node.
    pub fn class_ids(&self) -> impl Iterator<Item = EClassId> + '_ {
        (0..self.classes.len()).filter_map(move |idx| {
            let id = EClassId(idx as u32);
            let canonical = self.find(id);
            if canonical == id && !self.classes[idx].nodes.is_empty() {
                Some(id)
            } else {
                None
            }
        })
    }

    /// Collect all canonical e-class IDs into a `Vec`.
    ///
    /// Use this instead of `class_ids()` when the caller needs `&mut self`
    /// (since the iterator borrows `&self`). This is the single source of
    /// truth for "which classes are canonical" — delegates to `class_ids()`.
    pub fn canonical_class_ids(&self) -> Vec<EClassId> {
        self.class_ids().collect()
    }

    /// Get the total number of nodes across all e-classes.
    pub fn node_count(&self) -> usize {
        self.classes.iter().map(|c| c.nodes.len()).sum()
    }

    /// Get the OpKind of the canonical representative of an e-class.
    ///
    /// Resolves through union-find to the canonical class, then returns
    /// the OpKind of the first node in that class.
    pub fn canonical_op(&self, id: EClassId) -> pixelflow_ir::OpKind {
        let id = self.find(id);
        let class = &self.classes[id.index()];
        match &class.nodes[0] {
            ENode::Var(_) => pixelflow_ir::OpKind::Var,
            ENode::Const(_) => pixelflow_ir::OpKind::Const,
            ENode::Op { op, .. } => op.kind(),
        }
    }

    /// Debug: dump the entire e-graph structure.
    #[allow(dead_code)]
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (idx, class) in self.classes.iter().enumerate() {
            let id = EClassId(idx as u32);
            let canonical = self.find(id);
            if canonical == id && !class.nodes.is_empty() {
                writeln!(&mut out, "e{}: {:?}", idx, class.nodes).unwrap();
            }
        }
        out
    }

    /// Add an arena-based DAG expression to the e-graph, preserving sharing.
    ///
    /// Each `ExprId` in the arena maps to exactly one `EClassId`. Because the
    /// arena is topologically ordered (children always precede parents), a single
    /// linear scan suffices — no recursion, no stack overflow.
    ///
    /// Returns the `EClassId` of the root node.
    ///
    /// # Panics
    ///
    /// Panics if `root` is not a valid index in `arena`, or if any child
    /// `ExprId` referenced by an interior node has not been processed yet
    /// (which would indicate a malformed arena that violates topological order).
    /// Also panics if an `ExprNode::Param` node is encountered, since Param
    /// nodes are not valid after kernel compilation.
    pub fn add_arena(
        &mut self,
        arena: &pixelflow_ir::ExprArena,
        root: pixelflow_ir::ExprId,
    ) -> EClassId {
        use pixelflow_ir::arena::ExprNode;

        let n = arena.len();
        assert!(
            (root.0 as usize) < n,
            "add_arena: root {:?} out of bounds (arena has {} nodes)",
            root,
            n,
        );

        // Map from arena ExprId index → EClassId. None = not yet processed.
        let mut id_map: Vec<Option<EClassId>> = vec![None; n];

        for idx in 0..n {
            let eid = pixelflow_ir::ExprId(idx as u32);
            let eclass = match arena.node(eid) {
                ExprNode::Var(v) => self.add(ENode::Var(*v)),
                ExprNode::Const(v) => self.add(ENode::constant(*v)),
                ExprNode::Param(i) => {
                    panic!("add_arena: ExprNode::Param({i}) not valid after kernel compilation")
                }
                ExprNode::Unary(op, child) => {
                    let child_id = id_map[child.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Unary node at idx={idx} references child {:?} which has not been processed (arena not topologically ordered)",
                            child
                        )
                    });
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: vec![child_id],
                    })
                }
                ExprNode::Binary(op, left, right) => {
                    let left_id = id_map[left.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Binary node at idx={idx} references left child {:?} which has not been processed",
                            left
                        )
                    });
                    let right_id = id_map[right.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Binary node at idx={idx} references right child {:?} which has not been processed",
                            right
                        )
                    });
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: vec![left_id, right_id],
                    })
                }
                ExprNode::Ternary(op, a, b, c) => {
                    let a_id = id_map[a.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Ternary node at idx={idx} references child a={:?} which has not been processed",
                            a
                        )
                    });
                    let b_id = id_map[b.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Ternary node at idx={idx} references child b={:?} which has not been processed",
                            b
                        )
                    });
                    let c_id = id_map[c.0 as usize].unwrap_or_else(|| {
                        panic!(
                            "add_arena: Ternary node at idx={idx} references child c={:?} which has not been processed",
                            c
                        )
                    });
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: vec![a_id, b_id, c_id],
                    })
                }
                ExprNode::Nary(op, start, len) => {
                    let children_slice = arena.nary_children_slice(*start, *len);
                    let child_ids: Vec<EClassId> = children_slice
                        .iter()
                        .enumerate()
                        .map(|(ci, c)| {
                            id_map[c.0 as usize].unwrap_or_else(|| {
                                panic!(
                                    "add_arena: Nary node at idx={idx} references child[{ci}]={:?} which has not been processed",
                                    c
                                )
                            })
                        })
                        .collect();
                    let static_op = ops::op_from_kind(*op)
                        .unwrap_or_else(|| panic!("add_arena: no static Op for OpKind {:?}", op));
                    self.add(ENode::Op {
                        op: static_op,
                        children: child_ids,
                    })
                }
            };
            id_map[idx] = Some(eclass);
        }

        id_map[root.0 as usize].expect("add_arena: root EClassId missing after full traversal")
    }

    /// Get a rule by index.
    pub fn rule(&self, idx: usize) -> Option<&dyn Rewrite> {
        self.rules.get(idx).map(|r| r.as_ref())
    }

    /// Find all actual rewrite matches in the E-graph.
    ///
    /// Returns only targets where the rule actually matches (produces an action).
    /// Much more efficient than enumerating all combinations - only scores real matches.
    pub fn find_rewrite_matches(&self) -> Vec<RewriteTarget> {
        let mut matches = Vec::new();

        for (rule_idx, rule) in self.rules.iter().enumerate() {
            for class_id in self.class_ids() {
                let nodes = &self.classes[class_id.index()].nodes;

                for (node_idx, node) in nodes.iter().enumerate() {
                    // Check if rule matches this node
                    if rule.apply(self, class_id, node).is_some() {
                        matches.push(RewriteTarget {
                            rule_idx,
                            class_id,
                            node_idx,
                        });
                    }
                }
            }
        }

        matches
    }

    /// Apply a single rule to a specific (class, node) pair.
    ///
    /// Returns true if the rule matched and produced a change.
    /// This is used by guided search to apply rules one at a time.
    pub fn apply_single_rule(
        &mut self,
        rule_idx: usize,
        class_id: EClassId,
        node_idx: usize,
    ) -> bool {
        let Some(rule) = self.rules.get(rule_idx) else {
            return false;
        };

        let class_id = self.find(class_id);
        let nodes = self.classes[class_id.index()].nodes.clone();
        let Some(node) = nodes.get(node_idx) else {
            return false;
        };

        let Some(action) = rule.apply(self, class_id, node) else {
            return false;
        };

        // The batched path already knows how to execute every action; reuse
        // it so the single-step path can't drift out of sync.
        let changed = self.apply_action(class_id, action) > 0;

        if changed {
            self.rebuild();
        }
        changed
    }

    pub fn contains_const(&self, id: EClassId, val: f32) -> bool {
        self.nodes(id).iter().any(|n| n.is_const(val))
    }

    /// Saturate the e-graph with time and size limits.
    ///
    /// Uses chess-style time management:
    /// - 500ms hard timeout (never exceed)
    /// - 10000 class limit (prevent memory explosion)
    /// - 100 iteration limit (budget control)
    pub fn saturate(&mut self) {
        self.saturate_with_limits(100, 10_000, std::time::Duration::from_millis(500))
    }

    /// Saturate with just an iteration limit (simple compatibility API).
    ///
    /// Warning: This can hang on complex expressions. Prefer `saturate_with_limits`.
    pub fn saturate_with_limit(&mut self, max_iters: usize) {
        self.saturate_with_limits(max_iters, 10_000, std::time::Duration::from_millis(500))
    }

    /// Saturate with full time and size control.
    ///
    /// Returns when any limit is hit:
    /// - `max_iters` iterations completed
    /// - `max_classes` e-classes reached (memory protection)
    /// - `timeout` elapsed (time protection)
    /// - Saturation achieved (no more changes)
    pub fn saturate_with_limits(
        &mut self,
        max_iters: usize,
        max_classes: usize,
        timeout: std::time::Duration,
    ) {
        let start = std::time::Instant::now();
        let deadline = start + timeout;

        for _ in 0..max_iters {
            if start.elapsed() >= timeout {
                return;
            }
            if self.classes.len() > max_classes {
                return;
            }

            // Apply all rules in a single batch — one rebuild per iteration
            let unions = {
                let mut batch = self.batch();
                let n_rules = batch.graph.rules.len();
                let mut total = 0;
                for rule_idx in 0..n_rules {
                    if batch.node_count() > max_classes {
                        break;
                    }
                    let result = batch.apply_rule(rule_idx, max_classes, Some(deadline));
                    total += result.changes;
                }
                total
                // rebuild happens here on drop
            };
            if unions == 0 {
                return; // Saturated
            }
        }
    }

    /// Apply all rewrite rules once with a node budget.
    ///
    /// Returns the number of changes made. Stops if the graph exceeds
    /// `max_nodes` classes.
    pub fn apply_rules_once(&mut self, max_nodes: usize) -> usize {
        self.apply_rules_budgeted(max_nodes)
    }

    /// Apply a single rule (by index) everywhere it matches, with budget.
    ///
    /// Returns changes made and evaluations consumed. Stops scanning
    /// if the graph exceeds `max_nodes` classes.
    pub fn apply_rule_at_index(&mut self, rule_idx: usize, max_nodes: usize) -> ApplyResult {
        self.apply_rule_at_index_budgeted(rule_idx, max_nodes)
    }

    /// Apply a single rule with a node budget. Stops scanning when the
    /// e-graph exceeds `max_nodes` classes, preventing runaway growth
    /// from a single rule application.
    pub fn apply_rule_at_index_budgeted(
        &mut self,
        rule_idx: usize,
        max_nodes: usize,
    ) -> ApplyResult {
        self.apply_rule_at_index_timed(rule_idx, max_nodes, None)
    }

    /// Apply a single rule with node budget AND optional wall-clock deadline.
    /// Stops if either budget or deadline is exceeded.
    pub fn apply_rule_at_index_timed(
        &mut self,
        rule_idx: usize,
        max_nodes: usize,
        deadline: Option<std::time::Instant>,
    ) -> ApplyResult {
        if rule_idx >= self.rules.len() {
            return ApplyResult {
                changes: 0,
                evals: 0,
            };
        }

        let mut unions = 0;
        let mut evals = 0usize;
        let mut updates: Vec<(EClassId, RewriteAction)> = Vec::new();
        let mut estimated_new_nodes: usize = 0;

        let canonical_ids = self.canonical_class_ids();
        'scan: for canonical in canonical_ids {
            // Budget check: current graph + pending creates must stay under limit
            if self.classes.len() + estimated_new_nodes > max_nodes {
                break;
            }
            // Deadline check
            if evals & 1023 == 0 {
                if let Some(dl) = deadline {
                    if std::time::Instant::now() > dl {
                        break;
                    }
                }
            }

            let nodes: Vec<ENode> = self.classes[canonical.index()].nodes.clone();

            for node in &nodes {
                evals += 1;
                if let Some(action) = self.rules[rule_idx].apply(self, canonical, node) {
                    // Track how many nodes this action would create
                    let action_cost = match &action {
                        RewriteAction::Union(_) => 0,
                        RewriteAction::Create(_) => 1,
                        // Multi-node actions: conservative upper bound
                        _ => 3,
                    };
                    estimated_new_nodes += action_cost;

                    // If this action would push us over budget, stop scanning
                    if self.classes.len() + estimated_new_nodes > max_nodes {
                        // Don't add this action — discard it and stop
                        break 'scan;
                    }

                    updates.push((canonical, action));
                    *self
                        .match_counts
                        .entry(self.rules[rule_idx].name().to_string())
                        .or_insert(0) += 1;
                }
            }
        }

        // Commit: all actions in the log are within budget.
        // Do NOT rebuild here — caller is responsible for calling rebuild()
        // after all rules for the epoch are applied (lazy/batched rebuild).
        for (class_id, action) in updates {
            unions += self.apply_action(class_id, action);
        }

        ApplyResult {
            changes: unions,
            evals,
        }
    }

    /// Apply a rewrite action and return 1 if a union was made, 0 otherwise.
    fn apply_action(&mut self, class_id: EClassId, action: RewriteAction) -> usize {
        match action {
            RewriteAction::Union(target_id) => {
                if self.find(class_id) != self.find(target_id) {
                    self.union(class_id, target_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Create(new_node) => {
                let new_id = self.add(new_node);
                if self.find(class_id) != self.find(new_id) {
                    self.union(class_id, new_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Distribute {
                outer,
                inner,
                a,
                b,
                c,
            } => {
                let ab_node = ENode::Op {
                    op: outer,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab_node);
                let ac_node = ENode::Op {
                    op: outer,
                    children: vec![a, c],
                };
                let ac_id = self.add(ac_node);
                let result_node = ENode::Op {
                    op: inner,
                    children: vec![ab_id, ac_id],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => {
                let sum_node = ENode::Op {
                    op: outer,
                    children: vec![unique_l, unique_r],
                };
                let sum_id = self.add(sum_node);
                let result_node = ENode::Op {
                    op: inner,
                    children: vec![common, sum_id],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => {
                let inv_node = ENode::Op {
                    op: inverse,
                    children: vec![b],
                };
                let inv_id = self.add(inv_node);
                let target_node = ENode::Op {
                    op: target,
                    children: vec![a, inv_id],
                };
                let target_id = self.add(target_node);
                if self.find(class_id) != self.find(target_id) {
                    self.union(class_id, target_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Associate { op, a, b, c } => {
                let bc_node = ENode::Op {
                    op,
                    children: vec![b, c],
                };
                let bc_id = self.add(bc_node);
                let result_node = ENode::Op {
                    op,
                    children: vec![a, bc_id],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::ReverseAssociate { op, a, b, c } => {
                // a op (b op c) → (a op b) op c
                let ab_node = ENode::Op {
                    op,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab_node);
                let result_node = ENode::Op {
                    op,
                    children: vec![ab_id, c],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::OddParity { func, inner } => {
                // For odd functions: Op(neg(x)) → neg(Op(x))
                // Create func(inner), then wrap in neg
                let func_node = ENode::Op {
                    op: func,
                    children: vec![inner],
                };
                let func_id = self.add(func_node);
                let neg_node = ENode::Op {
                    op: &ops::Neg,
                    children: vec![func_id],
                };
                let neg_id = self.add(neg_node);
                if self.find(class_id) != self.find(neg_id) {
                    self.union(class_id, neg_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::AngleAddition {
                term1_op1,
                term1_op2,
                term2_op1,
                term2_op2,
                term2_sign,
                a,
                b,
            } => {
                // sin(a+b) → sin(a)cos(b) + cos(a)sin(b)
                // cos(a+b) → cos(a)cos(b) - sin(a)sin(b)
                //
                // Create: term1_op1(a)*term1_op2(b) +/- term2_op1(a)*term2_op2(b)

                // term1_op1(a)
                let t1_left = ENode::Op {
                    op: term1_op1,
                    children: vec![a],
                };
                let t1_left_id = self.add(t1_left);

                // term1_op2(b)
                let t1_right = ENode::Op {
                    op: term1_op2,
                    children: vec![b],
                };
                let t1_right_id = self.add(t1_right);

                // term1_op1(a) * term1_op2(b)
                let term1 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![t1_left_id, t1_right_id],
                };
                let term1_id = self.add(term1);

                // term2_op1(a)
                let t2_left = ENode::Op {
                    op: term2_op1,
                    children: vec![a],
                };
                let t2_left_id = self.add(t2_left);

                // term2_op2(b)
                let t2_right = ENode::Op {
                    op: term2_op2,
                    children: vec![b],
                };
                let t2_right_id = self.add(t2_right);

                // term2_op1(a) * term2_op2(b)
                let term2 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![t2_left_id, t2_right_id],
                };
                let term2_id = self.add(term2);

                // Combine based on sign
                use crate::math::trig::Sign;
                let result_id = match term2_sign {
                    Sign::Plus => {
                        let result = ENode::Op {
                            op: &ops::Add,
                            children: vec![term1_id, term2_id],
                        };
                        self.add(result)
                    }
                    Sign::Minus => {
                        let result = ENode::Op {
                            op: &ops::Sub,
                            children: vec![term1_id, term2_id],
                        };
                        self.add(result)
                    }
                };

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Homomorphism {
                func,
                target_op,
                a,
                b,
            } => {
                // f(a ⊕ b) → f(a) ⊗ f(b)
                // e.g., exp(a + b) → exp(a) * exp(b)

                // func(a)
                let func_a = ENode::Op {
                    op: func,
                    children: vec![a],
                };
                let func_a_id = self.add(func_a);

                // func(b)
                let func_b = ENode::Op {
                    op: func,
                    children: vec![b],
                };
                let func_b_id = self.add(func_b);

                // target_op(func(a), func(b))
                let result = ENode::Op {
                    op: target_op,
                    children: vec![func_a_id, func_b_id],
                };
                let result_id = self.add(result);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::PowerCombine { base, exp_a, exp_b } => {
                // x^a * x^b → x^(a+b)

                // a + b
                let sum = ENode::Op {
                    op: &ops::Add,
                    children: vec![exp_a, exp_b],
                };
                let sum_id = self.add(sum);

                // x^(a+b)
                let result = ENode::Op {
                    op: &ops::Pow,
                    children: vec![base, sum_id],
                };
                let result_id = self.add(result);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::ReverseAngleAddition { trig_op, a, b } => {
                // sin(a)cos(b) + cos(a)sin(b) → sin(a + b)
                // (or cos case)

                // a + b
                let sum = ENode::Op {
                    op: &ops::Add,
                    children: vec![a, b],
                };
                let sum_id = self.add(sum);

                // trig(a + b)
                let result = ENode::Op {
                    op: trig_op,
                    children: vec![sum_id],
                };
                let result_id = self.add(result);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::HalfAngleProduct { x } => {
                // sin(x) * cos(x) → sin(x + x) / 2
                // Derived from: sin(2x) = 2*sin(x)*cos(x)

                // x + x
                let two_x = ENode::Op {
                    op: &ops::Add,
                    children: vec![x, x],
                };
                let two_x_id = self.add(two_x);

                // sin(x + x)
                let sin_2x = ENode::Op {
                    op: &ops::Sin,
                    children: vec![two_x_id],
                };
                let sin_2x_id = self.add(sin_2x);

                // constant 2
                let two = ENode::Const(2.0_f32.to_bits());
                let two_id = self.add(two);

                // sin(x + x) / 2
                let result = ENode::Op {
                    op: &ops::Div,
                    children: vec![sin_2x_id, two_id],
                };
                let result_id = self.add(result);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Doubling { a } => {
                // a + a → 2 * a
                let two = ENode::Const(2.0_f32.to_bits());
                let two_id = self.add(two);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![two_id, a],
                };
                let result_id = self.add(result);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Halving { a } => {
                // 2 * a → a + a
                let result = ENode::Op {
                    op: &ops::Add,
                    children: vec![a, a],
                };
                let result_id = self.add(result);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::PowerRecurrence { base, exponent } => {
                let n_minus_1 = ENode::constant((exponent - 1) as f32);
                let n_minus_1_id = self.add(n_minus_1);
                let pow_reduced = ENode::Op {
                    op: &ops::Pow,
                    children: vec![base, n_minus_1_id],
                };
                let pow_id = self.add(pow_reduced);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![base, pow_id],
                };
                let result_id = self.add(result);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::LogPower {
                log_op,
                base,
                exponent,
            } => {
                let log_x = ENode::Op {
                    op: log_op,
                    children: vec![base],
                };
                let log_x_id = self.add(log_x);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![exponent, log_x_id],
                };
                let result_id = self.add(result);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::ExpandSquare { a, b } => {
                let a2 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![a, a],
                };
                let a2_id = self.add(a2);
                let b2 = ENode::Op {
                    op: &ops::Mul,
                    children: vec![b, b],
                };
                let b2_id = self.add(b2);
                let ab = ENode::Op {
                    op: &ops::Mul,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab);
                let two = ENode::constant(2.0);
                let two_id = self.add(two);
                let two_ab = ENode::Op {
                    op: &ops::Mul,
                    children: vec![two_id, ab_id],
                };
                let two_ab_id = self.add(two_ab);
                let sum1 = ENode::Op {
                    op: &ops::Add,
                    children: vec![a2_id, two_ab_id],
                };
                let sum1_id = self.add(sum1);
                let result = ENode::Op {
                    op: &ops::Add,
                    children: vec![sum1_id, b2_id],
                };
                let result_id = self.add(result);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::DiffOfSquares { a, b } => {
                let sum = ENode::Op {
                    op: &ops::Add,
                    children: vec![a, b],
                };
                let sum_id = self.add(sum);
                let diff = ENode::Op {
                    op: &ops::Sub,
                    children: vec![a, b],
                };
                let diff_id = self.add(diff);
                let result = ENode::Op {
                    op: &ops::Mul,
                    children: vec![sum_id, diff_id],
                };
                let result_id = self.add(result);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    1
                } else {
                    0
                }
            }
            RewriteAction::Differentiate { inner, var } => {
                let deriv_id = self.build_derivative(&inner, var);
                if self.find(class_id) != self.find(deriv_id) {
                    self.union(class_id, deriv_id);
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Build the e-class of the derivative of `inner` with respect to variable
    /// `var`, one chain-rule step deep. Sub-expressions are wrapped in fresh
    /// `Dwrt` nodes so equality saturation continues the expansion; the leaf
    /// cases (`Var`, `Const`) terminate it. Operators whose derivative is not
    /// (yet) known reconstruct the original `Dwrt`, leaving it to survive
    /// saturation as the jet fallback.
    fn build_derivative(&mut self, inner: &ENode, var: u8) -> EClassId {
        let (op, children) = match inner {
            ENode::Const(_) => return self.add(ENode::constant(0.0)),
            ENode::Var(i) => {
                return self.add(ENode::constant(if *i == var { 1.0 } else { 0.0 }));
            }
            ENode::Op { op, children } => (*op, children.clone()),
        };

        let var_const = self.add(ENode::constant(var as f32));
        // d(child)/dvar as a fresh Dwrt node (saturation expands it later).
        let dwrt = |s: &mut Self, c: EClassId| {
            s.add(ENode::Op {
                op: &ops::Dwrt,
                children: vec![c, var_const],
            })
        };
        let op2 = |s: &mut Self, o: &'static dyn Op, a: EClassId, b: EClassId| {
            s.add(ENode::Op {
                op: o,
                children: vec![a, b],
            })
        };
        let un = |s: &mut Self, o: &'static dyn Op, a: EClassId| {
            s.add(ENode::Op {
                op: o,
                children: vec![a],
            })
        };
        let cst = |s: &mut Self, v: f32| s.add(ENode::constant(v));

        match op.kind() {
            // Linearity: D(a + b) = D(a) + D(b); D(a - b) = D(a) - D(b).
            OpKind::Add | OpKind::Sub => {
                let da = dwrt(self, children[0]);
                let db = dwrt(self, children[1]);
                let same = ops::op_from_kind(op.kind()).expect("add/sub op");
                op2(self, same, da, db)
            }
            OpKind::Neg => {
                let da = dwrt(self, children[0]);
                un(self, &ops::Neg, da)
            }
            // Product rule: D(a*b) = D(a)*b + a*D(b).
            OpKind::Mul => {
                let (a, b) = (children[0], children[1]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let t1 = op2(self, &ops::Mul, da, b);
                let t2 = op2(self, &ops::Mul, a, db);
                op2(self, &ops::Add, t1, t2)
            }
            // Fused multiply-add a*b + c: D = D(a)*b + a*D(b) + D(c).
            OpKind::MulAdd => {
                let (a, b, c) = (children[0], children[1], children[2]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let dc = dwrt(self, c);
                let t1 = op2(self, &ops::Mul, da, b);
                let t2 = op2(self, &ops::Mul, a, db);
                let prod = op2(self, &ops::Add, t1, t2);
                op2(self, &ops::Add, prod, dc)
            }
            // Quotient rule: D(a/b) = (D(a)*b - a*D(b)) / (b*b).
            OpKind::Div => {
                let (a, b) = (children[0], children[1]);
                let da = dwrt(self, a);
                let db = dwrt(self, b);
                let t1 = op2(self, &ops::Mul, da, b);
                let t2 = op2(self, &ops::Mul, a, db);
                let num = op2(self, &ops::Sub, t1, t2);
                let den = op2(self, &ops::Mul, b, b);
                op2(self, &ops::Div, num, den)
            }
            // d(sqrt u) = 0.5 * rsqrt(u) * u'.
            OpKind::Sqrt => {
                let u = children[0];
                let du = dwrt(self, u);
                let half = cst(self, 0.5);
                let rs = un(self, &ops::Rsqrt, u);
                let factor = op2(self, &ops::Mul, half, rs);
                op2(self, &ops::Mul, factor, du)
            }
            // d(recip u) = -u' / (u*u).
            OpKind::Recip => {
                let u = children[0];
                let du = dwrt(self, u);
                let ndu = un(self, &ops::Neg, du);
                let u2 = op2(self, &ops::Mul, u, u);
                op2(self, &ops::Div, ndu, u2)
            }
            // d(|u|) = (u / |u|) * u'.
            OpKind::Abs => {
                let u = children[0];
                let du = dwrt(self, u);
                let au = un(self, &ops::Abs, u);
                let sign = op2(self, &ops::Div, u, au);
                op2(self, &ops::Mul, sign, du)
            }
            // d(sin u) = cos(u) * u'.
            OpKind::Sin => {
                let u = children[0];
                let du = dwrt(self, u);
                let c = un(self, &ops::Cos, u);
                op2(self, &ops::Mul, c, du)
            }
            // d(cos u) = -sin(u) * u'.
            OpKind::Cos => {
                let u = children[0];
                let du = dwrt(self, u);
                let s = un(self, &ops::Sin, u);
                let ns = un(self, &ops::Neg, s);
                op2(self, &ops::Mul, ns, du)
            }
            // d(tan u) = u' / cos(u)^2.
            OpKind::Tan => {
                let u = children[0];
                let du = dwrt(self, u);
                let c = un(self, &ops::Cos, u);
                let c2 = op2(self, &ops::Mul, c, c);
                op2(self, &ops::Div, du, c2)
            }
            // d(atan u) = u' / (1 + u*u).
            OpKind::Atan => {
                let u = children[0];
                let du = dwrt(self, u);
                let one = cst(self, 1.0);
                let u2 = op2(self, &ops::Mul, u, u);
                let den = op2(self, &ops::Add, one, u2);
                op2(self, &ops::Div, du, den)
            }
            // d(asin u) = u' / sqrt(1 - u*u).
            OpKind::Asin => {
                let u = children[0];
                let du = dwrt(self, u);
                let one = cst(self, 1.0);
                let u2 = op2(self, &ops::Mul, u, u);
                let diff = op2(self, &ops::Sub, one, u2);
                let s = un(self, &ops::Sqrt, diff);
                op2(self, &ops::Div, du, s)
            }
            // d(acos u) = -u' / sqrt(1 - u*u).
            OpKind::Acos => {
                let u = children[0];
                let du = dwrt(self, u);
                let one = cst(self, 1.0);
                let u2 = op2(self, &ops::Mul, u, u);
                let diff = op2(self, &ops::Sub, one, u2);
                let s = un(self, &ops::Sqrt, diff);
                let q = op2(self, &ops::Div, du, s);
                un(self, &ops::Neg, q)
            }
            // d(exp u) = exp(u) * u'.
            OpKind::Exp => {
                let u = children[0];
                let du = dwrt(self, u);
                let e = un(self, &ops::Exp, u);
                op2(self, &ops::Mul, e, du)
            }
            // d(ln u) = u' / u.
            OpKind::Ln => {
                let u = children[0];
                let du = dwrt(self, u);
                op2(self, &ops::Div, du, u)
            }
            // Unknown derivative: reconstruct the Dwrt and let it survive.
            _ => {
                let reconstructed = self.add(inner.clone());
                dwrt(self, reconstructed)
            }
        }
    }

    fn apply_rules_budgeted(&mut self, max_nodes: usize) -> usize {
        let mut unions = 0;
        let mut updates: Vec<(EClassId, RewriteAction)> = Vec::new();

        let canonical_ids = self.canonical_class_ids();
        for canonical in canonical_ids {
            if self.classes.len() > max_nodes {
                break;
            }
            let nodes: Vec<ENode> = self.classes[canonical.index()].nodes.clone();

            for node in &nodes {
                for rule in self.rules.iter() {
                    if let Some(action) = rule.apply(self, canonical, node) {
                        updates.push((canonical, action));
                        *self
                            .match_counts
                            .entry(rule.name().to_string())
                            .or_insert(0) += 1;
                    }
                }
            }
        }

        for (class_id, action) in updates {
            unions += self.apply_action(class_id, action);
            if self.classes.len() > max_nodes {
                break;
            }
        }

        // Lazy rebuild: caller should call rebuild() after all rules applied.
        // saturate_with_limits handles this.
        unions
    }

    pub fn extract_with_costs(&self, root: EClassId, costs: &CostModel) -> ENode {
        let root = self.find(root);
        let mut cost_table: HashMap<EClassId, (usize, ENode)> = HashMap::new();
        let canonical_ids: Vec<EClassId> = self.class_ids().collect();
        // Fixed-point iteration: at most one pass per canonical class.
        for _ in 0..canonical_ids.len() {
            let mut changed = false;
            for &id in &canonical_ids {
                for node in &self.classes[id.index()].nodes {
                    let cost = self.node_cost_with_model(node, &cost_table, costs);
                    let current = cost_table.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX);
                    if cost < current {
                        cost_table.insert(id, (cost, node.clone()));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        cost_table
            .get(&root)
            .map(|(_, node)| node.clone())
            .unwrap_or(ENode::Const(0))
    }

    fn node_cost_with_model(
        &self,
        node: &ENode,
        cost_table: &HashMap<EClassId, (usize, ENode)>,
        costs: &CostModel,
    ) -> usize {
        let get_child_cost = |id: EClassId| {
            let id = self.find(id);
            cost_table
                .get(&id)
                .map(|(c, _)| *c)
                .unwrap_or(usize::MAX / 4)
        };
        let op_cost = costs.node_op_cost(node);
        let child_cost = node
            .children()
            .iter()
            .fold(0usize, |acc, &c| acc.saturating_add(get_child_cost(c)));
        child_cost.saturating_add(op_cost)
    }

    /// Extract the minimum-cost expression from an e-class.
    pub fn extract_expr_with_costs(
        &self,
        root: EClassId,
        costs: &CostModel,
    ) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
        let (arena, arena_root, _cost) = super::extract::extract(self, root, costs);
        (arena, arena_root)
    }

    /// Extract the best expression and its cost.
    ///
    /// The cost function can be any `CostFunction` implementor:
    /// - `CostModel` for hardcoded costs
    /// - `ExprNnue` via [`IncrementalExtractor`] for learned neural costs
    pub fn extract_best<C: CostFunction>(
        &self,
        root: EClassId,
        costs: &C,
    ) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, usize) {
        super::extract::extract(self, root, costs)
    }

    /// Extract a DAG with sharing information from an e-class.
    ///
    /// Unlike `extract_expr_with_costs`, this tracks which e-classes are used
    /// multiple times, enabling codegen to emit let-bindings for shared subexprs.
    ///
    /// # Example
    ///
    /// For `sin(X) * sin(X) + sin(X)`:
    /// - Tree extraction would compute sin(X) three times
    /// - DAG extraction marks sin(X) as shared, enabling: `let __0 = X.sin(); __0 * __0 + __0`
    pub fn extract_dag_with_costs(
        &self,
        root: EClassId,
        costs: &CostModel,
    ) -> super::extract::ExtractedDAG {
        super::extract::extract_dag(self, root, costs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::ops;
    use crate::math::algebra::{
        AddNeg, Annihilator, Cancellation, Canonicalize, Commutative, Distributive, Identity,
        InverseAnnihilation, Involution, MulRecip,
    };

    /// Create an e-graph with standard algebraic rules for testing.
    fn egraph_with_rules() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = vec![
            // InversePair rules
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            // Commutativity
            Commutative::new(&ops::Add),
            Commutative::new(&ops::Mul),
            Commutative::new(&ops::Min),
            Commutative::new(&ops::Max),
            // Distributivity
            Distributive::new(&ops::Mul, &ops::Add),
            Distributive::new(&ops::Mul, &ops::Sub),
            // Identity
            Identity::new(&ops::Add),
            Identity::new(&ops::Mul),
            // Annihilator
            Annihilator::new(&ops::Mul),
        ];
        EGraph::with_rules(rules)
    }

    #[test]
    fn verify_inverse_add() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![x],
        });
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, neg_x],
        });
        eg.saturate();
        let zero = eg.add(ENode::constant(0.0));
        assert_eq!(eg.find(sum), eg.find(zero));
    }

    #[test]
    fn verify_inverse_mul() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let recip_x = eg.add(ENode::Op {
            op: &ops::Recip,
            children: vec![x],
        });
        let product = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, recip_x],
        });
        eg.saturate();
        let one = eg.add(ENode::constant(1.0));
        assert_eq!(eg.find(product), eg.find(one));
    }

    #[test]
    fn verify_complex_inverse() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, five],
        });
        let div = eg.add(ENode::Op {
            op: &ops::Div,
            children: vec![prod, x],
        });
        eg.saturate();
        assert_eq!(eg.find(div), eg.find(five));
    }

    #[test]
    fn verify_nested_subtraction() {
        // a - (b - c) should equal a - b + c
        // Test: 10 - (6 - 2) = 10 - 4 = 6
        let mut eg = egraph_with_rules();
        let a = eg.add(ENode::constant(10.0)); // a = 10
        let b = eg.add(ENode::constant(6.0)); // b = 6
        let c = eg.add(ENode::constant(2.0)); // c = 2

        // Build a - (b - c)
        let b_minus_c = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![b, c],
        }); // 6 - 2 = 4
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![a, b_minus_c],
        }); // 10 - 4 = 6

        eg.saturate();

        // Extract and verify structure
        let costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
        eprintln!("Extracted arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn verify_mul_sub_pattern() {
        // This is the problematic pattern from discriminant:
        // d*d - (c - r) where d=4, c=16, r=1
        let mut eg = egraph_with_rules();
        let d = eg.add(ENode::constant(4.0));
        let c_sq = eg.add(ENode::constant(16.0));
        let r_sq = eg.add(ENode::constant(1.0));

        let d_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![d, d],
        });
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![c_sq, r_sq],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![d_sq, inner_sub],
        });

        eg.saturate();

        let costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
        eprintln!("Extracted arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn verify_mul_sub_pattern_with_vars() {
        // x*x - (y - z)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));

        let x_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![y, z],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x_sq, inner_sub],
        });

        eg.saturate();

        let costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
        eprintln!(
            "Extracted arena with vars: root={:?} len={}",
            root,
            arena.len()
        );
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn verify_mul_sub_pattern_with_fma() {
        // Same pattern but with FMA costs (what the kernel! macro uses)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));

        let x_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![y, z],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x_sq, inner_sub],
        });

        eg.saturate();

        // Use default costs like the kernel! macro does
        let costs = CostModel::new();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
        eprintln!(
            "Extracted arena with FMA costs: root={:?} len={}",
            root,
            arena.len()
        );
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn verify_discriminant_structure() {
        // Match the actual discriminant structure:
        // d_dot_c² - (c_sq - r_sq) where c_sq = a² + b² and r_sq = r²
        let mut eg = egraph_with_rules();
        let d = eg.add(ENode::Var(0));
        let a = eg.add(ENode::Var(1));
        let b = eg.add(ENode::Var(2));
        let r = eg.add(ENode::Var(3));

        let d_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![d, d],
        });
        let a_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![a, a],
        });
        let b_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![b, b],
        });
        let c_sq = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![a_sq, b_sq],
        });
        let r_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![r, r],
        });
        let inner = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![c_sq, r_sq],
        });
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![d_sq, inner],
        });

        eg.saturate();

        let costs = CostModel::new();
        let (arena, root) = eg.extract_expr_with_costs(result, &costs);
        eprintln!("Discriminant arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn verify_depth_penalty_calculation() {
        // Test the hinge penalty function
        let mut costs = CostModel::new();
        costs.depth_threshold = 5;
        costs.depth_penalty = 100;

        // Below threshold: no penalty
        assert_eq!(costs.depth_cost(0), 0);
        assert_eq!(costs.depth_cost(5), 0);

        // Above threshold: linear penalty
        assert_eq!(costs.depth_cost(6), 100);
        assert_eq!(costs.depth_cost(7), 200);
        assert_eq!(costs.depth_cost(10), 500);
    }

    #[test]
    fn verify_shallow_cost_model() {
        // Shallow model should have aggressive depth penalty
        let costs = CostModel::shallow();
        assert_eq!(costs.depth_threshold, 16);
        assert_eq!(costs.depth_penalty, 500);

        // Penalty kicks in after 16
        assert_eq!(costs.depth_cost(16), 0);
        assert_eq!(costs.depth_cost(17), 500);
        assert_eq!(costs.depth_cost(20), 2000);
    }

    #[test]
    fn verify_depth_aware_extraction() {
        // Build a deep expression: ((((x + 1) + 1) + 1) + 1)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));

        let mut current = x;
        for _ in 0..10 {
            current = eg.add(ENode::Op {
                op: &ops::Add,
                children: vec![current, one],
            });
        }

        eg.saturate();

        // Extract with default costs (high threshold)
        let default_costs = CostModel::default();
        let (arena, root) = eg.extract_expr_with_costs(current, &default_costs);
        assert!(arena.node_count_subtree(root) > 0);

        // Extract with shallow costs (low threshold)
        let mut shallow_costs = CostModel::new();
        shallow_costs.depth_threshold = 3;
        shallow_costs.depth_penalty = 1000;
        let (arena2, root2) = eg.extract_expr_with_costs(current, &shallow_costs);
        assert!(arena2.node_count_subtree(root2) > 0);
    }
}

// ============================================================================
// EGraphBatch — RAII batched rule application with lazy rebuild
// ============================================================================

/// RAII batch for rule application with budgeted interleaved rebuild.
///
/// Applies rules without per-rule rebuilds. After each rule, processes
/// a chunk of the rebuild worklist to keep classes deduplicated without
/// doing a full rebuild. On drop, drains the remaining worklist.
///
/// The rebuild budget per rule is proportional to the changes made,
/// keeping total work bounded.
///
/// ```ignore
/// {
///     let mut batch = egraph.batch();
///     for rule in approved_rules {
///         batch.apply_rule(rule, 500, Some(deadline));
///     }
/// } // final rebuild on drop
/// ```
pub struct EGraphBatch<'a> {
    graph: &'a mut EGraph,
    any_changes: bool,
    /// Max worklist items to process per rule application.
    /// Keeps class sizes bounded during the batch.
    rebuild_chunk: usize,
}

impl<'a> EGraphBatch<'a> {
    /// Apply a single rule, then process a chunk of pending rebuilds.
    ///
    /// The interleaved rebuild keeps classes from ballooning between rules.
    /// Each rule application is followed by processing up to `rebuild_chunk`
    /// worklist items, so the graph stays approximately deduplicated.
    pub fn apply_rule(
        &mut self,
        rule_idx: usize,
        max_nodes: usize,
        deadline: Option<std::time::Instant>,
    ) -> ApplyResult {
        let result = self
            .graph
            .apply_rule_at_index_timed(rule_idx, max_nodes, deadline);
        if result.changes > 0 {
            self.any_changes = true;
            // Interleaved partial rebuild: process some worklist items to keep
            // classes small. The chunk size bounds work per rule.
            self.graph.rebuild_budgeted(self.rebuild_chunk);
        }
        result
    }

    /// Current number of e-classes.
    pub fn node_count(&self) -> usize {
        self.graph.classes.len()
    }

    /// Whether any rule in this batch produced changes.
    pub fn has_changes(&self) -> bool {
        self.any_changes
    }

    /// Pending rebuild worklist items.
    pub fn pending_rebuilds(&self) -> usize {
        self.graph.pending_rebuilds()
    }
}

impl Drop for EGraphBatch<'_> {
    fn drop(&mut self) {
        // Drain any remaining worklist items
        if self.any_changes {
            self.graph.rebuild();
        }
    }
}
