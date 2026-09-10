//! Generic template rewrite: a rule whose LHS/RHS are data (an expression
//! graph pattern) rather than a hand-written combinator.
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
//! read metavariables (`ExprData::Var`) the same way every existing template
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

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rewrite::{Rewrite, RewriteAction, TemplateArena};

use pixelflow_ir::expr::{ExprBuilder, ExprData, ExprRef, Term};
use pixelflow_ir::{Node, Rooted};

/// Metavariable → canonical e-class bindings accumulated while matching one
/// pattern.
type Bindings = BTreeMap<u8, EClassId>;

/// The environment a rule pattern indexes: none. A pattern is pure structure —
/// every matcher and every unifier below refuses a `Buffer`/`Uniform` leaf —
/// so the tables a [`Term`] pairs with a root are empty here, once, rather
/// than rebuilt per call.
static EMPTY_ENV: pixelflow_ir::expr::Environment = pixelflow_ir::expr::Environment {
    buffers: Vec::new(),
    uniforms: Vec::new(),
};

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
        ExprData::Param(_) | ExprData::Buffer(_) | ExprData::Uniform(_) => false,
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
        ExprData::Param(_) | ExprData::Buffer(_) | ExprData::Uniform(_) => false,
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
        ExprData::Const(_) | ExprData::Param(_) | ExprData::Buffer(_) | ExprData::Uniform(_) => {}
        ExprData::Op(_) => {
            for c in node.children() {
                collect_metavars(c, out);
            }
        }
    }
}

/// [`collect_metavars`] over a node still under construction in `b`.
fn collect_metavars_in(b: &ExprBuilder, r: ExprRef, out: &mut std::collections::BTreeSet<u8>) {
    collect_metavars(b.node(r), out);
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

    /// Construct from a builder plus the two pattern roots inside it.
    ///
    /// The environment is dropped: a rule pattern names no memory (a `Buffer`
    /// or `Uniform` leaf is refused by every matcher below), so there is
    /// nothing in it to keep.
    #[must_use]
    pub fn from_builder(
        name: impl Into<String>,
        builder: ExprBuilder,
        lhs: ExprRef,
        rhs: ExprRef,
    ) -> Self {
        let (rooted, _env) = builder.finish(&[lhs, rhs]);
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

    fn lhs_template(&self, out: &mut ExprBuilder) -> Option<ExprRef> {
        Some(out.splice(Term::new(self.rooted.entry_at(0), &EMPTY_ENV)))
    }

    fn rhs_template(&self, out: &mut ExprBuilder) -> Option<ExprRef> {
        Some(out.splice(Term::new(self.rooted.entry_at(1), &EMPTY_ENV)))
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

/// Every position in the subtree at `root`: the root itself (`[]`) and every
/// proper subterm, addressed by the path of child indices to reach it.
pub(crate) fn positions(b: &ExprBuilder, root: ExprRef) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new()];
    for (i, c) in b.child_refs(root).iter().enumerate() {
        for mut sub in positions(b, *c) {
            let mut path = vec![i as u8];
            path.append(&mut sub);
            out.push(path);
        }
    }
    out
}

fn walk_position(b: &ExprBuilder, root: ExprRef, position: &[u8]) -> ExprRef {
    let mut cur = root;
    for &i in position {
        cur = *b
            .child_refs(cur)
            .get(i as usize)
            .expect("walk_position: path index out of bounds for this pattern's arity");
    }
    cur
}

/// Rebuild `root`, replacing the subtree at `position` with `replacement`.
fn replace_at(
    b: &mut ExprBuilder,
    root: ExprRef,
    position: &[u8],
    replacement: ExprRef,
) -> ExprRef {
    let Some((&i, rest)) = position.split_first() else {
        return replacement;
    };
    let children: Vec<ExprRef> = b.child_refs(root).to_vec();
    let idx = i as usize;
    let mut new_children = children.clone();
    new_children[idx] = replace_at(b, children[idx], rest, replacement);
    let op = op_of(b, root);
    b.push_nary(op, &new_children)
}

/// The operator at `r`.
///
/// # Panics
///
/// Panics on a leaf — every caller has already matched `ExprData::Op`.
fn op_of(b: &ExprBuilder, r: ExprRef) -> pixelflow_ir::OpKind {
    b.node(r)
        .op()
        .expect("op_of: called on a leaf, but the caller matched an Op")
}

/// Copy the subtree at `root` into `b`, replacing each `Var(v)` for which
/// `subs` has an entry.
///
/// `ExprBuilder` is append-only, so this is a rebuild rather than an in-place
/// edit — which is also what the arena version did, one `push` at a time.
pub(crate) fn substitute_vars(b: &mut ExprBuilder, root: ExprRef, subs: &[(u8, ExprRef)]) -> ExprRef {
    fn go(
        b: &mut ExprBuilder,
        r: ExprRef,
        subs: &[(u8, ExprRef)],
        memo: &mut BTreeMap<ExprRef, ExprRef>,
    ) -> ExprRef {
        if let Some(&hit) = memo.get(&r) {
            return hit;
        }
        let built = match *b.node(r) {
            ExprData::Var(v) => match subs.iter().find(|(x, _)| *x == v) {
                Some(&(_, repl)) => repl,
                None => r,
            },
            ExprData::Const(_)
            | ExprData::Param(_)
            | ExprData::Buffer(_)
            | ExprData::Uniform(_) => r,
            ExprData::Op(op) => {
                let kids: Vec<ExprRef> = b.child_refs(r).to_vec();
                let new_kids: Vec<ExprRef> =
                    kids.iter().map(|&c| go(b, c, subs, memo)).collect();
                if new_kids == kids {
                    r
                } else {
                    b.push_nary(op, &new_kids)
                }
            }
        };
        memo.insert(r, built);
        built
    }
    go(b, root, subs, &mut BTreeMap::new())
}

/// Shift every metavariable in the subtree at `root` by `offset`, giving B's
/// pattern a namespace disjoint from A's before unification.
fn shift_vars(b: &mut ExprBuilder, root: ExprRef, offset: u8) -> ExprRef {
    let mut used = std::collections::BTreeSet::new();
    collect_metavars_in(b, root, &mut used);
    let subs: Vec<(u8, ExprRef)> = used
        .into_iter()
        .map(|v| (v, b.push_var(offset + v)))
        .collect();
    substitute_vars(b, root, &subs)
}

/// Whether `r`'s subtree (resolving through `subst`) reaches metavariable
/// `mv` — the occurs check that keeps unification from building a cyclic
/// substitution (which [`apply_subst_deep`] would recurse forever on).
fn occurs(
    b: &ExprBuilder,
    mv: u8,
    r: ExprRef,
    subst: &BTreeMap<u8, ExprRef>,
    depth: u32,
) -> bool {
    if depth > 64 {
        // A pattern this deep never arises from the rule library this
        // harness composes; treat it as an occurrence rather than risk an
        // unbounded walk on a construction bug.
        return true;
    }
    match *b.node(r) {
        ExprData::Var(v) => {
            v == mv
                || subst
                    .get(&v)
                    .is_some_and(|&t| occurs(b, mv, t, subst, depth + 1))
        }
        ExprData::Const(_)
        | ExprData::Param(_)
        | ExprData::Buffer(_)
        | ExprData::Uniform(_) => false,
        ExprData::Op(_) => b
            .child_refs(r)
            .iter()
            .any(|&c| occurs(b, mv, c, subst, depth + 1)),
    }
}

fn resolve(b: &ExprBuilder, r: ExprRef, subst: &BTreeMap<u8, ExprRef>) -> ExprRef {
    let mut cur = r;
    while let ExprData::Var(v) = *b.node(cur) {
        match subst.get(&v) {
            Some(&t) => cur = t,
            None => break,
        }
    }
    cur
}

/// First-order syntactic unification of `x` and `y`, extending `subst`.
/// Read-only over `b` — unification only ever records bindings, the
/// substitution is materialized afterward by [`apply_subst_deep`].
fn unify(b: &ExprBuilder, x: ExprRef, y: ExprRef, subst: &mut BTreeMap<u8, ExprRef>) -> bool {
    let x = resolve(b, x, subst);
    let y = resolve(b, y, subst);
    match (*b.node(x), *b.node(y)) {
        (ExprData::Var(a), ExprData::Var(c)) if a == c => true,
        (ExprData::Var(a), _) => {
            if occurs(b, a, y, subst, 0) {
                return false;
            }
            subst.insert(a, y);
            true
        }
        (_, ExprData::Var(c)) => {
            if occurs(b, c, x, subst, 0) {
                return false;
            }
            subst.insert(c, x);
            true
        }
        // `ExprData::Const` already holds the bit pattern, so this is the
        // bitwise comparison the arena version spelled `to_bits()`.
        (ExprData::Const(cx), ExprData::Const(cy)) => cx == cy,
        (ExprData::Param(_), _) | (_, ExprData::Param(_)) => false,
        (ExprData::Buffer(_), _) | (_, ExprData::Buffer(_)) => false,
        (ExprData::Uniform(_), _) | (_, ExprData::Uniform(_)) => false,
        // A constant and an operator are different terms, as are any two
        // remaining leaf kinds — only two operators can still unify.
        (ExprData::Const(_), _) | (_, ExprData::Const(_)) => false,
        (ExprData::Op(ox), ExprData::Op(oy)) => {
            if ox != oy {
                return false;
            }
            let cx: Vec<ExprRef> = b.child_refs(x).to_vec();
            let cy: Vec<ExprRef> = b.child_refs(y).to_vec();
            if cx.len() != cy.len() {
                return false;
            }
            cx.iter()
                .zip(cy.iter())
                .all(|(&a, &c)| unify(b, a, c, subst))
        }
    }
}

/// Deeply resolve `r` through `subst`, rebuilding whatever changed.
/// Unlike [`resolve`] (which only chases `Var` chains at the root), this
/// walks the whole subtree so a bound metavariable is replaced everywhere it
/// occurs, including inside sibling structure.
fn apply_subst_deep(b: &mut ExprBuilder, r: ExprRef, subst: &BTreeMap<u8, ExprRef>) -> ExprRef {
    match *b.node(r) {
        ExprData::Var(mv) => match subst.get(&mv) {
            Some(&t) => apply_subst_deep(b, t, subst),
            None => r,
        },
        ExprData::Const(_)
        | ExprData::Param(_)
        | ExprData::Buffer(_)
        | ExprData::Uniform(_) => r,
        ExprData::Op(op) => {
            let children: Vec<ExprRef> = b.child_refs(r).to_vec();
            let new_children: Vec<ExprRef> = children
                .iter()
                .map(|&c| apply_subst_deep(b, c, subst))
                .collect();
            if new_children == children {
                r
            } else {
                b.push_nary(op, &new_children)
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
    b: &mut ExprBuilder,
    lhs: ExprRef,
    rhs: ExprRef,
) -> Option<(ExprRef, ExprRef, u8)> {
    let mut lhs_vars = std::collections::BTreeSet::new();
    collect_metavars_in(b, lhs, &mut lhs_vars);
    let mut rhs_vars = std::collections::BTreeSet::new();
    collect_metavars_in(b, rhs, &mut rhs_vars);
    if !rhs_vars.is_subset(&lhs_vars) {
        return None;
    }
    let subs: Vec<(u8, ExprRef)> = lhs_vars
        .iter()
        .enumerate()
        .map(|(new_i, &old)| (old, b.push_var(new_i as u8)))
        .collect();
    let count = lhs_vars.len() as u8;
    let final_lhs = substitute_vars(b, lhs, &subs);
    let final_rhs = substitute_vars(b, rhs, &subs);
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
        let mut out = ExprBuilder::new();
        let a_lhs = a.lhs_template(&mut out)?;
        let a_rhs = a.rhs_template(&mut out)?;

        // B's sides go into the SAME builder — `lhs_template`/`rhs_template`
        // splice, so there is no second graph to bridge across.
        let b_lhs_raw = b.lhs_template(&mut out)?;
        let b_rhs_raw = b.rhs_template(&mut out)?;
        let b_lhs = shift_vars(&mut out, b_lhs_raw, B_METAVAR_OFFSET);
        let b_rhs = shift_vars(&mut out, b_rhs_raw, B_METAVAR_OFFSET);

        let target = walk_position(&out, a_rhs, position);
        let mut subst = BTreeMap::new();
        if !unify(&out, b_lhs, target, &mut subst) {
            return None;
        }

        // Filter: B is a no-op at this position (what it matched already
        // equals what it would rewrite to, once both sides are resolved
        // through the unifier).
        let target_final = apply_subst_deep(&mut out, target, &subst);
        let b_rhs_final = apply_subst_deep(&mut out, b_rhs, &subst);
        if out.node(target_final).subtree_eq(out.node(b_rhs_final)) {
            return None;
        }

        let composed_rhs_pre = replace_at(&mut out, a_rhs, position, b_rhs);
        let composed_lhs = apply_subst_deep(&mut out, a_lhs, &subst);
        let composed_rhs = apply_subst_deep(&mut out, composed_rhs_pre, &subst);

        let (final_lhs, final_rhs, _count) =
            canonicalize_vars(&mut out, composed_lhs, composed_rhs)?;

        // Filter: identity — the composition changed nothing (e.g.
        // commutative∘commutative).
        if out.node(final_lhs).subtree_eq(out.node(final_rhs)) {
            return None;
        }

        let name = format!("{}\u{2218}{}@{position:?}", a.name(), b.name());
        Some(TemplateRewrite::from_builder(
            name, out, final_lhs, final_rhs,
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
        let mut a = ExprBuilder::new();
        let v0 = a.push_var(0);
        let v1 = a.push_var(1);
        let lhs = a.push_binary(OpKind::Sub, v0, v1);
        let v0b = a.push_var(0);
        let v1b = a.push_var(1);
        let neg = a.push_unary(OpKind::Neg, v1b);
        let rhs = a.push_binary(OpKind::Add, v0b, neg);
        TemplateRewrite::from_builder("test_sub_to_add_neg", a, lhs, rhs)
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
        let mut a = ExprBuilder::new();
        let v0 = a.push_var(0);
        let v0b = a.push_var(0);
        let lhs = a.push_binary(OpKind::Sub, v0, v0b);
        let zero = a.push_const(0.0);
        let rule = TemplateRewrite::from_builder("test_self_sub", a, lhs, zero);

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
        // through `ExprBuilder` — the thing this test exists to
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
