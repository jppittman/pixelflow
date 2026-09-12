//! Generic template rewrite: a rule whose LHS/RHS are data (an [`ExprArena`]
//! pattern) rather than a hand-written combinator.
//!
//! Round 2's mode (ii) (docs/plans/2026-09-01-phase3-round2-rule-scaling.md
//! §2.2, §8) generates rules by composing two existing rules' templates at
//! harness startup — there is no `struct` to write per composition, only a
//! pattern discovered at runtime. [`TemplateRewrite`] is the one executor
//! every such pattern shares: e-match `lhs` against `(class, node)` exactly
//! as every hand-written multi-level rule already does (see `Factor`,
//! `math::algebra`, which loops `egraph.nodes(child)` the same way), then
//! hand the bindings to [`super::rewrite::RewriteAction::Instantiate`].
//!
//! # Matching convention
//!
//! A pattern position is matched against either a concrete [`ENode`] (the
//! rule's own root, handed in by the sweep — [`match_root`]) or an
//! [`EClassId`] (every other position, where the pattern may need to try
//! more than one of that class's representatives — [`match_class`]). Both
//! read metavariables (`ExprNode::Var`) the same way every existing template
//! does: `Var(0)` = A, `Var(1)` = B, etc. A repeated metavariable must bind
//! to the same **canonical** class everywhere it appears — [`Bindings`]
//! enforces that via `egraph.find`.
//!
//! This is deliberately a single first-match search (mirroring `Factor`'s
//! `return` on the first successful candidate), not a full enumeration of
//! every match — the sweep already revisits every `(class, node)` pair every
//! round, so a second match on the same node is found on a later application
//! rather than paid for up front.

use std::collections::BTreeMap;
use std::sync::Arc;

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rewrite::{Rewrite, RewriteAction, TemplateArena};

use pixelflow_ir::expr::ExprData;
use pixelflow_ir::{Node, Rooted};

/// Metavariable → canonical e-class bindings accumulated while matching one
/// pattern.
type Bindings = BTreeMap<u8, EClassId>;

fn bind_var(mv: u8, class: EClassId, egraph: &EGraph, bindings: &mut Bindings) -> bool {
    let class = egraph.find(class);
    match bindings.get(&mv) {
        Some(&existing) => egraph.find(existing) == class,
        None => {
            bindings.insert(mv, class);
            true
        }
    }
}

/// Match `pat` against a concrete node (used only at the pattern root, where
/// the sweep already handed us the specific node to try).
fn match_root(
    egraph: &EGraph,
    pat: Node<'_, ExprData>,
    node: &ENode,
    bindings: &mut Bindings,
) -> bool {
    match *pat {
        // A Var-rooted LHS pattern is degenerate (it would match every node
        // in the graph) and no rule this harness generates ever produces
        // one — `compose_rules` always composes at an Op position. Refuse
        // rather than guess a binding with no class to bind it to.
        ExprData::Var(_) => false,
        ExprData::Const(v) => node.is_const(f32::from_bits(v)),
        ExprData::Param(_)
        | ExprData::Buffer(_)
        | ExprData::Uniform(_)
        | ExprData::Ref(_)
        | ExprData::Reduce(_)
        // No rule this harness generates ever writes a `Guard` into a
        // pattern — the e-graph declines one on `insert` (G1), so nothing
        // could ever have matched here in the first place.
        | ExprData::Guard { .. } => false,
        ExprData::Op(_) => match_op(egraph, pat, node, bindings),
    }
}

/// Match `pat` against every representative of `class`, first-match-wins.
fn match_class(
    egraph: &EGraph,
    pat: Node<'_, ExprData>,
    class: EClassId,
    bindings: &mut Bindings,
) -> bool {
    match *pat {
        ExprData::Var(mv) => bind_var(mv, class, egraph, bindings),
        ExprData::Const(v) => egraph.contains_const(class, f32::from_bits(v)),
        ExprData::Param(_)
        | ExprData::Buffer(_)
        | ExprData::Uniform(_)
        | ExprData::Ref(_)
        | ExprData::Reduce(_)
        | ExprData::Guard { .. } => false,
        ExprData::Op(_) => {
            for node in egraph.nodes(class) {
                let mut trial = bindings.clone();
                if match_op(egraph, pat, node, &mut trial) {
                    *bindings = trial;
                    return true;
                }
            }
            false
        }
    }
}

/// Structural match of an `Op`-shaped pattern against a concrete node:
/// same `OpKind`, same arity, every child matched at its class position.
fn match_op(
    egraph: &EGraph,
    pat: Node<'_, ExprData>,
    node: &ENode,
    bindings: &mut Bindings,
) -> bool {
    let Some(node_op) = node.op() else {
        return false;
    };
    let ExprData::Op(kind) = *pat else {
        return false;
    };
    if node_op.kind() != kind {
        return false;
    }
    let pat_children: Vec<Node<'_, ExprData>> = pat.children().collect();
    let node_children = node.children_slice();
    if pat_children.len() != node_children.len() {
        return false;
    }
    for (pc, nc) in pat_children.iter().zip(node_children.iter()) {
        if !match_class(egraph, *pc, *nc, bindings) {
            return false;
        }
    }
    true
}

/// Distinct metavariable indices used anywhere in the subtree at `node`.
fn collect_metavars(node: Node<'_, ExprData>, out: &mut std::collections::BTreeSet<u8>) {
    match *node {
        ExprData::Var(mv) => {
            out.insert(mv);
        }
        ExprData::Const(_)
        | ExprData::Param(_)
        | ExprData::Buffer(_)
        | ExprData::Uniform(_)
        | ExprData::Ref(_) => {}
        ExprData::Reduce(_) | ExprData::Op(_) | ExprData::Guard { .. } => {
            for c in node.children() {
                collect_metavars(c, out);
            }
        }
    }
}

fn collect_metavars_arena(arena: &ExprArena, id: ExprId, out: &mut std::collections::BTreeSet<u8>) {
    match arena.node(id) {
        pixelflow_ir::arena::ExprNode::Var(mv) => {
            out.insert(*mv);
        }
        pixelflow_ir::arena::ExprNode::Const(_)
        | pixelflow_ir::arena::ExprNode::Param(_)
        | pixelflow_ir::arena::ExprNode::Buffer(_)
        | pixelflow_ir::arena::ExprNode::Uniform(_) => {}
        _ => {
            for c in arena.children(id) {
                collect_metavars_arena(arena, c, out);
            }
        }
    }
}

/// A rewrite rule whose LHS/RHS are runtime data instead of a hand-written
/// combinator. See the module docs for the matching contract.
pub struct TemplateRewrite {
    name: String,
    rooted: Arc<Rooted<ExprData>>,
    /// One past the highest metavariable index used in `lhs`/`rhs` — the
    /// fixed length every produced `bindings` vector has.
    metavar_count: u8,
}

impl TemplateRewrite {
    /// Build a template rule directly from an LHS/RHS pair in `rooted`.
    /// `rooted` must have exactly 2 entries: `[lhs, rhs]`.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` uses a metavariable that never appears in `lhs`.
    #[must_use]
    pub fn new(name: impl Into<String>, rooted: Rooted<ExprData>) -> Self {
        assert_eq!(
            rooted.entries().len(),
            2,
            "TemplateRewrite requires 2 entries: [lhs, rhs]"
        );
        let lhs = rooted.entry_at(0);
        let rhs = rooted.entry_at(1);
        let mut lhs_vars = std::collections::BTreeSet::new();
        collect_metavars(lhs, &mut lhs_vars);
        let mut rhs_vars = std::collections::BTreeSet::new();
        collect_metavars(rhs, &mut rhs_vars);
        assert!(
            rhs_vars.is_subset(&lhs_vars),
            "TemplateRewrite::new: rhs uses metavariable(s) {:?} not bound by lhs {:?}",
            rhs_vars.difference(&lhs_vars).collect::<Vec<_>>(),
            lhs_vars
        );
        let metavar_count = lhs_vars
            .iter()
            .chain(rhs_vars.iter())
            .max()
            .map_or(0, |m| m + 1);
        Self {
            name: name.into(),
            rooted: Arc::new(rooted),
            metavar_count,
        }
    }

    /// Construct from legacy arena + expr IDs.
    #[must_use]
    pub fn from_arena(name: impl Into<String>, arena: ExprArena, lhs: ExprId, rhs: ExprId) -> Self {
        let (rooted, _) = pixelflow_ir::expr::from_arena_roots(&arena, &[lhs, rhs]);
        Self::new(name, rooted)
    }

    #[must_use]
    pub fn rooted(&self) -> &Rooted<ExprData> {
        &self.rooted
    }

    #[must_use]
    pub fn lhs<'a>(&'a self) -> Node<'a, ExprData> {
        self.rooted.entry_at(0)
    }

    #[must_use]
    pub fn rhs<'a>(&'a self) -> Node<'a, ExprData> {
        self.rooted.entry_at(1)
    }
}

impl Rewrite for TemplateRewrite {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let mut bindings = Bindings::new();
        let lhs = self.rooted.entry_at(0);
        if !match_root(egraph, lhs, node, &mut bindings) {
            return None;
        }
        let mut binding_vec = Vec::with_capacity(self.metavar_count as usize);
        for mv in 0..self.metavar_count {
            // Every metavariable in `lhs` is bound by a successful match
            // (matching an Op pattern binds every leaf metavariable it
            // contains); `new`'s invariant guarantees `rhs` uses no others.
            binding_vec.push(*bindings.get(&mv).unwrap_or_else(|| {
                panic!(
                    "TemplateRewrite({}): metavariable {mv} unbound after a successful match \
                     — lhs/rhs invariant violated at construction",
                    self.name
                )
            }));
        }
        Some(RewriteAction::Instantiate {
            template: super::rewrite::TemplatePattern(Arc::clone(&self.rooted)),
            entry: 1,
            bindings: binding_vec,
        })
    }

    fn lhs_template(&self, out: &mut ExprArena) -> Option<ExprId> {
        let env = pixelflow_ir::expr::Environment::default();
        let (arena, roots) =
            pixelflow_ir::expr::to_arena_roots(&self.rooted, &[self.rooted.entry_at(0)], &env);
        Some(out.splice(&arena, roots[0]))
    }

    fn rhs_template(&self, out: &mut ExprArena) -> Option<ExprId> {
        let env = pixelflow_ir::expr::Environment::default();
        let (arena, roots) =
            pixelflow_ir::expr::to_arena_roots(&self.rooted, &[self.rooted.entry_at(1)], &env);
        Some(out.splice(&arena, roots[0]))
    }
}

// ============================================================================
// Composition: A∘B (docs/plans/2026-09-01-phase3-round2-rule-scaling.md §2.2)
// ============================================================================
//
// The validity argument (design doc §2.2) is a theorem about substitution:
// if σ unifies `B.lhs` with the subterm of `A.rhs` at position `p`, then
// `A.lhs σ = A.rhs[p := B.rhs] σ` for every assignment. What follows
// implements exactly that — unify, substitute, replace-at-position — and
// nothing here is trusted on the theorem alone: every composed rule is
// oracle-checked by `math::oracle::cross_form_oracle` before it is used, as
// the design's §2.4 gate requires.

/// Fresh namespace B's metavariables are shifted into before unification, so
/// `Var(0)` from A and `Var(0)` from B are never accidentally the same
/// unification variable. Every rule this harness composes from uses single-
/// digit metavariable indices (the templated 30 of `all_rules()` peak at 3),
/// so 64 is a wide, cheap margin rather than a tight bound.
const B_METAVAR_OFFSET: u8 = 64;

fn push_by_arity(arena: &mut ExprArena, kind: pixelflow_ir::OpKind, children: &[ExprId]) -> ExprId {
    match children.len() {
        1 => arena.push_unary(kind, children[0]),
        2 => arena.push_binary(kind, children[0], children[1]),
        3 => arena.push_ternary(kind, children[0], children[1], children[2]),
        _ => arena.push_nary(kind, children),
    }
}

/// Every position in the subtree at `root`: the root itself (`[]`) and every
/// proper subterm, addressed by the path of child indices to reach it.
pub(crate) fn positions(arena: &ExprArena, root: ExprId) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new()];
    for (i, c) in arena.children(root).enumerate() {
        for mut sub in positions(arena, c) {
            let mut path = vec![i as u8];
            path.append(&mut sub);
            out.push(path);
        }
    }
    out
}

fn walk_position(arena: &ExprArena, root: ExprId, position: &[u8]) -> ExprId {
    let mut cur = root;
    for &i in position {
        cur = arena
            .children(cur)
            .nth(i as usize)
            .expect("walk_position: path index out of bounds for this arena's arity");
    }
    cur
}

/// Rebuild `root`, replacing the subtree at `position` with `replacement`.
fn replace_at(arena: &mut ExprArena, root: ExprId, position: &[u8], replacement: ExprId) -> ExprId {
    let Some((&i, rest)) = position.split_first() else {
        return replacement;
    };
    let children: Vec<ExprId> = arena.children(root).collect();
    let idx = i as usize;
    let mut new_children = children.clone();
    new_children[idx] = replace_at(arena, children[idx], rest, replacement);
    let kind = arena.kind(root);
    push_by_arity(arena, kind, &new_children)
}

/// Shift every metavariable in the subtree at `root` by `offset`, giving B's
/// pattern a namespace disjoint from A's before unification.
fn shift_vars(arena: &mut ExprArena, root: ExprId, offset: u8) -> ExprId {
    let mut used = std::collections::BTreeSet::new();
    collect_metavars_arena(arena, root, &mut used);
    let subs: Vec<(u8, ExprId)> = used
        .into_iter()
        .map(|v| (v, arena.push_var(offset + v)))
        .collect();
    arena.substitute_vars_with(root, &subs)
}

/// Whether `id`'s subtree (resolving through `subst`) reaches metavariable
/// `mv` — the occurs check that keeps unification from building a cyclic
/// substitution (which `apply_subst_deep` would recurse forever on).
fn occurs(arena: &ExprArena, mv: u8, id: ExprId, subst: &BTreeMap<u8, ExprId>, depth: u32) -> bool {
    if depth > 64 {
        // A pattern this deep never arises from the rule library this
        // harness composes; treat it as an occurrence rather than risk an
        // unbounded walk on a construction bug.
        return true;
    }
    match arena.node(id) {
        ExprNode::Var(v) => {
            *v == mv
                || subst
                    .get(v)
                    .is_some_and(|&t| occurs(arena, mv, t, subst, depth + 1))
        }
        ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Buffer(_) | ExprNode::Uniform(_) => {
            false
        }
        _ => arena
            .children(id)
            .any(|c| occurs(arena, mv, c, subst, depth + 1)),
    }
}

fn resolve(arena: &ExprArena, id: ExprId, subst: &BTreeMap<u8, ExprId>) -> ExprId {
    let mut cur = id;
    while let ExprNode::Var(v) = arena.node(cur) {
        match subst.get(v) {
            Some(&t) => cur = t,
            None => break,
        }
    }
    cur
}

/// First-order syntactic unification of `x` and `y`, extending `subst`.
/// Read-only over `arena` — unification only ever records bindings, the
/// substitution is materialized afterward by [`apply_subst_deep`].
fn unify(arena: &ExprArena, x: ExprId, y: ExprId, subst: &mut BTreeMap<u8, ExprId>) -> bool {
    let x = resolve(arena, x, subst);
    let y = resolve(arena, y, subst);
    match (arena.node(x), arena.node(y)) {
        (ExprNode::Var(a), ExprNode::Var(b)) if a == b => true,
        (ExprNode::Var(a), _) => {
            let a = *a;
            if occurs(arena, a, y, subst, 0) {
                return false;
            }
            subst.insert(a, y);
            true
        }
        (_, ExprNode::Var(b)) => {
            let b = *b;
            if occurs(arena, b, x, subst, 0) {
                return false;
            }
            subst.insert(b, x);
            true
        }
        (ExprNode::Const(cx), ExprNode::Const(cy)) => cx.to_bits() == cy.to_bits(),
        (ExprNode::Param(_), _) | (_, ExprNode::Param(_)) => false,
        (ExprNode::Buffer(_), _) | (_, ExprNode::Buffer(_)) => false,
        (ExprNode::Uniform(_), _) | (_, ExprNode::Uniform(_)) => false,
        _ => {
            if arena.kind(x) != arena.kind(y) {
                return false;
            }
            let cx: Vec<ExprId> = arena.children(x).collect();
            let cy: Vec<ExprId> = arena.children(y).collect();
            if cx.len() != cy.len() {
                return false;
            }
            cx.iter()
                .zip(cy.iter())
                .all(|(&a, &b)| unify(arena, a, b, subst))
        }
    }
}

/// Deeply resolve `id` through `subst`, rebuilding whatever changed.
/// Unlike [`resolve`] (which only chases `Var` chains at the root), this
/// walks the whole subtree so a bound metavariable is replaced everywhere it
/// occurs, including inside sibling structure.
fn apply_subst_deep(arena: &mut ExprArena, id: ExprId, subst: &BTreeMap<u8, ExprId>) -> ExprId {
    match arena.node(id).clone() {
        ExprNode::Var(mv) => match subst.get(&mv) {
            Some(&t) => apply_subst_deep(arena, t, subst),
            None => id,
        },
        ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Buffer(_) | ExprNode::Uniform(_) => id,
        _ => {
            let kind = arena.kind(id);
            let children: Vec<ExprId> = arena.children(id).collect();
            let new_children: Vec<ExprId> = children
                .iter()
                .map(|&c| apply_subst_deep(arena, c, subst))
                .collect();
            if new_children == children {
                id
            } else {
                push_by_arity(arena, kind, &new_children)
            }
        }
    }
}

/// Renumber every metavariable reachable from `lhs` (and, transitively via
/// the RHS-uses-only-LHS-vars invariant, every one reachable from `rhs`) to
/// a dense `0..k` range in sorted order. Returns `None` if `rhs` uses a
/// metavariable `lhs` does not — the composition is unsound to expose as a
/// rewrite (nothing on the LHS would bind it) and is dropped rather than
/// handed to [`TemplateRewrite::new`]'s assert.
fn canonicalize_vars(
    arena: &mut ExprArena,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<(ExprId, ExprId, u8)> {
    let mut lhs_vars = std::collections::BTreeSet::new();
    collect_metavars_arena(arena, lhs, &mut lhs_vars);
    let mut rhs_vars = std::collections::BTreeSet::new();
    collect_metavars_arena(arena, rhs, &mut rhs_vars);
    if !rhs_vars.is_subset(&lhs_vars) {
        return None;
    }
    let subs: Vec<(u8, ExprId)> = lhs_vars
        .iter()
        .enumerate()
        .map(|(new_i, &old)| (old, arena.push_var(new_i as u8)))
        .collect();
    let count = lhs_vars.len() as u8;
    let final_lhs = arena.substitute_vars_with(lhs, &subs);
    let final_rhs = arena.substitute_vars_with(rhs, &subs);
    Some((final_lhs, final_rhs, count))
}

impl TemplateRewrite {
    /// Compose `a` then `b`: unify `b`'s LHS against the subterm of `a`'s
    /// RHS at `position`, then build the rule whose single application
    /// creates what `a`-then-`b` would create in two rounds. `position` is a
    /// child-index path into `a.rhs_template()` (`[]` = the whole RHS);
    /// [`positions`] enumerates every valid one for a given `a`.
    ///
    /// Returns `None` when: either side lacks templates, `b.lhs` does not
    /// unify with `a.rhs` at `position`, `b` turns out to be a no-op there,
    /// the composed rule is a literal identity (α-equivalent LHS/RHS), or
    /// the composed RHS would use a metavariable the composed LHS does not
    /// bind. Every `None` is a filter, not a bug — callers report the
    /// surviving count, never treat an empty pool as an error by itself.
    #[must_use]
    pub fn compose(a: &dyn Rewrite, b: &dyn Rewrite, position: &[u8]) -> Option<TemplateRewrite> {
        let mut arena = ExprArena::new();
        let a_lhs = a.lhs_template(&mut arena)?;
        let a_rhs = a.rhs_template(&mut arena)?;

        let mut b_arena = ExprArena::new();
        let b_lhs0 = b.lhs_template(&mut b_arena)?;
        let b_rhs0 = b.rhs_template(&mut b_arena)?;
        let b_lhs_in_a = arena.splice(&b_arena, b_lhs0);
        let b_rhs_in_a = arena.splice(&b_arena, b_rhs0);
        let b_lhs = shift_vars(&mut arena, b_lhs_in_a, B_METAVAR_OFFSET);
        let b_rhs = shift_vars(&mut arena, b_rhs_in_a, B_METAVAR_OFFSET);

        let target = walk_position(&arena, a_rhs, position);
        let mut subst = BTreeMap::new();
        if !unify(&arena, b_lhs, target, &mut subst) {
            return None;
        }

        // Filter: B is a no-op at this position (what it matched already
        // equals what it would rewrite to, once both sides are resolved
        // through the unifier).
        let target_final = apply_subst_deep(&mut arena, target, &subst);
        let b_rhs_final = apply_subst_deep(&mut arena, b_rhs, &subst);
        if arena.subtree_eq(target_final, &arena, b_rhs_final) {
            return None;
        }

        let composed_rhs_pre = replace_at(&mut arena, a_rhs, position, b_rhs);
        let composed_lhs = apply_subst_deep(&mut arena, a_lhs, &subst);
        let composed_rhs = apply_subst_deep(&mut arena, composed_rhs_pre, &subst);

        let (final_lhs, final_rhs, _count) =
            canonicalize_vars(&mut arena, composed_lhs, composed_rhs)?;

        // Filter: identity — the composition changed nothing (e.g.
        // commutative∘commutative).
        if arena.subtree_eq(final_lhs, &arena, final_rhs) {
            return None;
        }

        let name = format!("{}\u{2218}{}@{position:?}", a.name(), b.name());
        Some(TemplateRewrite::from_arena(
            name, arena, final_lhs, final_rhs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::EGraph;
    use pixelflow_ir::OpKind;

    /// `a - b -> a + neg(b)`, hand-built as a template — should behave
    /// exactly like `math::algebra::Canonicalize::<AddNeg>`.
    fn sub_to_add_neg() -> TemplateRewrite {
        let mut a = ExprArena::new();
        let v0 = a.push_var(0);
        let v1 = a.push_var(1);
        let lhs = a.push_binary(OpKind::Sub, v0, v1);
        let v0b = a.push_var(0);
        let v1b = a.push_var(1);
        let neg = a.push_unary(OpKind::Neg, v1b);
        let rhs = a.push_binary(OpKind::Add, v0b, neg);
        TemplateRewrite::from_arena("test_sub_to_add_neg", a, lhs, rhs)
    }

    #[test]
    fn matches_and_instantiates_a_two_level_pattern() {
        let mut eg = EGraph::with_rules(vec![Box::new(sub_to_add_neg())]);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sub = eg.add(ENode::Op {
            op: &crate::egraph::ops::Sub,
            children: vec![x, y],
        });
        let root = eg.find(sub);
        let node = eg.nodes(root)[0].clone();
        let rule = sub_to_add_neg();
        let action = rule
            .apply(&eg, root, &node)
            .expect("pattern should match Sub(x,y)");
        matches!(action, RewriteAction::Instantiate { .. });
    }

    #[test]
    fn compose_matches_a_deeper_pattern_with_a_repeated_metavariable() {
        // (a - a) -> should match a template for Sub(V0, V0) and refuse
        // Sub(V0, V1) style mismatches when the two operands are unrelated.
        let mut a = ExprArena::new();
        let v0 = a.push_var(0);
        let v0b = a.push_var(0);
        let lhs = a.push_binary(OpKind::Sub, v0, v0b);
        let zero = a.push_const(0.0);
        let rule = TemplateRewrite::from_arena("test_self_sub", a, lhs, zero);

        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let self_sub = eg.add(ENode::Op {
            op: &crate::egraph::ops::Sub,
            children: vec![x, x],
        });
        let other_sub = eg.add(ENode::Op {
            op: &crate::egraph::ops::Sub,
            children: vec![x, y],
        });

        let self_root = eg.find(self_sub);
        let self_node = eg.nodes(self_root)[0].clone();
        assert!(rule.apply(&eg, self_root, &self_node).is_some());

        let other_root = eg.find(other_sub);
        let other_node = eg.nodes(other_root)[0].clone();
        assert!(
            rule.apply(&eg, other_root, &other_node).is_none(),
            "repeated metavariable must refuse two different classes"
        );
    }

    #[test]
    fn dag_builder_template_rewrite() {
        // `x - y`, `x + (-y)`, built directly via `dag::Builder` rather than
        // derived from an `ExprArena` — the thing this test exists to
        // exercise. `Builder`/`Id` are `pub(crate)` to `pixelflow_ir`, so
        // this crate can no longer build that shape itself; the fixture
        // lives in `pixelflow_ir::internal_test_support`, the one place
        // still allowed to.
        let rooted = pixelflow_ir::internal_test_support::template_rewrite_sub_fixture();
        let rule = TemplateRewrite::new("test_dag_sub", rooted);
        assert_eq!(rule.lhs().child_count(), 2);
        assert_eq!(rule.rhs().child_count(), 2);

        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sub = eg.add(ENode::Op {
            op: &crate::egraph::ops::Sub,
            children: vec![x, y],
        });
        let sub_root = eg.find(sub);
        let sub_node = eg.nodes(sub_root)[0].clone();
        let action = rule.apply(&eg, sub_root, &sub_node).expect("must match");
        matches!(action, RewriteAction::Instantiate { .. });
    }
}
