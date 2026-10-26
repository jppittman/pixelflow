//! The guard: what a fold whose bound is a value computes outside the domain
//! of a count.
//!
//! A bound that is a value is read per call, so nothing at compile time knows
//! it names a count. This pass is L5 and L2 of
//! docs/language/2026-10-20-a-bound-is-a-value.md written in the language's own
//! ops: every reachable fold `R` with a value bound comes back as
//!
//! ```text
//! If(dom, If(Lt(lo, hi), R, identity), NaN)
//! ```
//!
//! where `dom` says every value bound is a count, `0 ≤ v ∧ v ≤ 2²⁴ ∧ ⌊v⌋ = v`
//! (a literal bound is one by construction and adds no conjunct). Outside the
//! domain the fold is NaN, and a count that runs backwards is the monoid's
//! identity. The tests use only ordered `Le` and `Lt`, exact `Eq` and `Floor`,
//! none of which is in CLAUDE.md's table of what differs by target, so NaN, ±∞
//! and everything past 2²⁴ fail on every tier the same way.
//!
//! **Why an `If`.** `If(m, a, b)` runs the true arm's own block only if any
//! lane of `m` is set, so the loop that is `R` sits in an arm and does not run
//! at all outside the domain. That is what keeps an out-of-domain count from
//! hanging a counter that never reaches its end: an `f32` counter stalls at
//! 2²⁴, and at `-∞` from the start, and on a target whose latch compare is
//! ordered it never meets a NaN end. The arm skip is dispatch, not a fold, and
//! it is the dispatch wanted here. A bound reads no lattice binder (L3), so
//! every lane of every mask here holds one value: no batch is mixed, and the
//! arm runs on every lane or on none.
//!
//! **Why last.** Nothing after this pass rebuilds the `If`, so lowering
//! asserts the shape this pass produced (`R` read only as an `If`'s true arm).
//! Built as arena nodes, every def of the guard takes its variance from the
//! arena's own transfer, like any other node.
//!
//! **Not idempotent.** Run twice, the pass wraps the already-guarded fold
//! again, which is correct and redundant (the outer guard passes exactly when
//! the inner does). [`super::legalize`] runs it once.

use super::rebuild_arena;
use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::fold::{Bound, Fold};
use crate::kind::OpKind;

/// The last pass of [`super::legalize`]: every reachable fold with a value
/// bound, in its guard (L5, L2). Takes the arena by value, so the identity (no
/// node of the arena is such a fold) moves it back untouched, with no clone and
/// no rebuild to re-order its nodes.
pub(crate) fn guard_value_bounds_owned(mut arena: ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena.nodes().any(|(_, node)| has_a_value_bound(&node)) {
        return (arena, root);
    }
    let root = rebuild_arena(&mut arena, root, |arena, node, m| {
        let &ExprNode::Reduce { fold, lo, hi, body } = node else {
            return None;
        };
        if !has_a_value_bound(node) {
            return None;
        }
        let folded = arena.push_reduce(fold, lo.map(m), hi.map(m), m(body));
        Some(guard(arena, folded))
    });
    (arena, root)
}

/// Whether `node` is a fold with a bound that is a value.
fn has_a_value_bound(node: &ExprNode) -> bool {
    match node {
        ExprNode::Reduce { lo, hi, .. } => lo.value().is_some() || hi.value().is_some(),
        _ => false,
    }
}

/// `folded`, a fold already rebuilt, inside its guard. The nodes are pushed in
/// the order they are read, the fold first and the outer `If` last, so that
/// every value an arm reads was constructed before the arm (the order `Layout`
/// relies on).
fn guard(arena: &mut ExprArena, folded: ExprId) -> ExprId {
    // `push_reduce` discovers constness again, so the bounds are read from the
    // node it kept, and a pair that came out literal needs no guard.
    let ExprNode::Reduce { fold, lo, hi, .. } = arena.node(folded) else {
        unreachable!("push_reduce pushes a Reduce");
    };
    let domain = match (lo.value(), hi.value()) {
        (Some(low), Some(high)) if low != high => {
            let low = in_domain(arena, low);
            let high = in_domain(arena, high);
            arena.push_binary(OpKind::BitAnd, low, high)
        }
        (Some(term), _) | (None, Some(term)) => in_domain(arena, term),
        (None, None) => return folded,
    };
    let start = term(arena, lo);
    let end = term(arena, hi);
    let runs = arena.push_binary(OpKind::Lt, start, end);
    let identity = arena.push_lit(fold.monoid().identity());
    let ran = arena.push_ternary(OpKind::If, runs, folded, identity);
    let nan = arena.push_const(f32::NAN);
    arena.push_ternary(OpKind::If, domain, ran, nan)
}

/// The mask that says `v` names a count: `0 ≤ v ∧ v ≤ 2²⁴ ∧ ⌊v⌋ = v`, with
/// ordered `≤` and exact `=`. NaN fails all three.
fn in_domain(arena: &mut ExprArena, v: ExprId) -> ExprId {
    let zero = arena.push_const(0.0);
    let top = arena.push_const(Fold::EXACT_BOUND as f32);
    let not_below = arena.push_binary(OpKind::Le, zero, v);
    let not_above = arena.push_binary(OpKind::Le, v, top);
    let floor = arena.push_unary(OpKind::Floor, v);
    let whole = arena.push_binary(OpKind::Eq, floor, v);
    let in_range = arena.push_binary(OpKind::BitAnd, not_below, not_above);
    arena.push_binary(OpKind::BitAnd, in_range, whole)
}

/// A bound as a term: its child, or a `Const` of the literal (at most 2²⁴, so
/// exact as an `f32`).
fn term(arena: &mut ExprArena, bound: Bound<ExprId>) -> ExprId {
    match bound {
        Bound::Lit(count) => arena.push_const(count as f32),
        Bound::Value(child) => child,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::{Binder, Monoid};
    use crate::lit::Lit;
    use crate::variance::{Variance, compute_arena_variance};
    use alloc::vec::Vec;

    /// A fresh uniform's leaf in `arena`.
    fn uniform(arena: &mut ExprArena, default: f32) -> ExprId {
        let slot = arena.declare_uniform(crate::Uniform::new(default).decl());
        arena.push_uniform(slot)
    }

    /// `⊕_{k ∈ [lo, hi)} (X + k)` over binder slot 0.
    fn counted(
        monoid: Monoid,
        arena: &mut ExprArena,
        lo: Bound<ExprId>,
        hi: Bound<ExprId>,
    ) -> ExprId {
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let (x, k) = (arena.push_var(0), arena.push_var(binder.var()));
        let body = arena.push_binary(OpKind::Add, x, k);
        arena.push_reduce(Fold::new(monoid, binder), lo, hi, body)
    }

    /// The nodes reachable from `root`.
    fn reachable(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
        let mut seen = alloc::vec![false; arena.len()];
        let mut stack = alloc::vec![root];
        let mut order = Vec::new();
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            order.push(id);
            stack.extend(arena.children(id));
        }
        order
    }

    /// How many reachable nodes read `id` as a child, counted per edge.
    fn readers(arena: &ExprArena, root: ExprId, id: ExprId) -> usize {
        reachable(arena, root)
            .into_iter()
            .map(|parent| arena.children(parent).filter(|child| *child == id).count())
            .sum()
    }

    /// The reachable nodes under `root` that apply `op`.
    fn count_of(arena: &ExprArena, root: ExprId, op: OpKind) -> usize {
        reachable(arena, root)
            .into_iter()
            .filter(|id| match arena.node(*id) {
                ExprNode::Unary(o, ..) | ExprNode::Binary(o, ..) | ExprNode::Ternary(o, ..) => {
                    o == op
                }
                _ => false,
            })
            .count()
    }

    /// A guard taken apart: `If(domain, If(runs, fold, identity), nan)`.
    struct Guard {
        domain: ExprId,
        runs: ExprId,
        fold: ExprId,
        identity: ExprId,
        nan: ExprId,
    }

    impl Guard {
        fn of(arena: &ExprArena, guarded: ExprId) -> Self {
            let ExprNode::Ternary(OpKind::If, domain, ran, nan) = arena.node(guarded) else {
                panic!("the guard is an If");
            };
            let ExprNode::Ternary(OpKind::If, runs, fold, identity) = arena.node(ran) else {
                panic!("the guard's true arm is an If");
            };
            Self {
                domain,
                runs,
                fold,
                identity,
                nan,
            }
        }
    }

    /// With no value bound there is nothing to guard, and the arena moves back
    /// as it came: not cloned, not rebuilt, not re-ordered.
    #[test]
    fn a_kernel_with_no_value_bound_moves_back_untouched() {
        let mut arena = ExprArena::new();
        let sum = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Lit(8));
        let before = arena.len();
        let (arena, root) = guard_value_bounds_owned(arena, sum);
        assert_eq!((arena.len(), root), (before, sum));
    }

    /// The fold is `R`, the original node, and nothing reads it but the
    /// guard's true arm: the property lowering places the loop by.
    #[test]
    fn a_value_bounded_fold_is_read_only_by_its_guard() {
        let mut arena = ExprArena::new();
        let u = uniform(&mut arena, 8.0);
        let fold = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Value(u));
        let (arena, root) = guard_value_bounds_owned(arena, fold);

        let guard = Guard::of(&arena, root);
        assert_eq!(guard.fold, fold, "the fold is the original node");
        assert_eq!(readers(&arena, root, fold), 1, "and has one reader");
        assert_eq!(arena.node(guard.identity), ExprNode::Const(Lit::Real(0.0)));
        assert!(matches!(
            arena.node(guard.nan),
            ExprNode::Const(Lit::Real(v)) if v.is_nan()
        ));
        let ExprNode::Binary(OpKind::Lt, start, end) = arena.node(guard.runs) else {
            panic!("the arm runs while lo < hi");
        };
        assert_eq!(arena.node(start), ExprNode::Const(Lit::Real(0.0)));
        assert_eq!(end, u);
    }

    /// A literal bound is a count already: only a value is tested, so `[0, u)`
    /// has the domain test of `u` alone, two `≤`.
    #[test]
    fn a_literal_bound_adds_no_domain_test() {
        let mut arena = ExprArena::new();
        let u = uniform(&mut arena, 8.0);
        let fold = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Value(u));
        let (arena, root) = guard_value_bounds_owned(arena, fold);
        let domain = Guard::of(&arena, root).domain;
        assert_eq!(count_of(&arena, domain, OpKind::Le), 2);
        assert_eq!(count_of(&arena, domain, OpKind::Floor), 1);
        assert_eq!(count_of(&arena, domain, OpKind::Eq), 1);
        assert_eq!(count_of(&arena, domain, OpKind::BitAnd), 2);
    }

    /// The same for a value `lo` beside a literal `hi`: `[u, 8)` has the domain
    /// test of `u` alone, so that a `u` outside the domain is NaN though the end
    /// is a count already.
    #[test]
    fn a_value_lo_beside_a_literal_hi_has_the_domain_test_of_lo() {
        let mut arena = ExprArena::new();
        let u = uniform(&mut arena, 3.0);
        let fold = counted(Monoid::SUM, &mut arena, Bound::Value(u), Bound::Lit(8));
        let (arena, root) = guard_value_bounds_owned(arena, fold);
        let guard = Guard::of(&arena, root);
        assert!(
            reachable(&arena, guard.domain).contains(&u),
            "the domain reads u"
        );
        assert_eq!(count_of(&arena, guard.domain, OpKind::Le), 2);
        assert_eq!(count_of(&arena, guard.domain, OpKind::Floor), 1);
        assert_eq!(count_of(&arena, guard.domain, OpKind::Eq), 1);
        assert_eq!(count_of(&arena, guard.domain, OpKind::BitAnd), 2);
        let ExprNode::Binary(OpKind::Lt, start, end) = arena.node(guard.runs) else {
            panic!("the arm runs while lo < hi");
        };
        assert_eq!(start, u);
        assert_eq!(arena.node(end), ExprNode::Const(Lit::Real(8.0)));
    }

    /// Each value bound is tested, once however often it is read: `[m, n)` tests
    /// both and joins the tests, `[u, u)` tests the one.
    #[test]
    fn two_value_bounds_are_each_in_the_domain() {
        let mut arena = ExprArena::new();
        let m = uniform(&mut arena, 2.0);
        let n = uniform(&mut arena, 6.0);
        let both = counted(Monoid::SUM, &mut arena, Bound::Value(m), Bound::Value(n));
        let (arena, root) = guard_value_bounds_owned(arena, both);
        let guard = Guard::of(&arena, root);
        let ExprNode::Binary(OpKind::BitAnd, low, high) = arena.node(guard.domain) else {
            panic!("the two tests are joined");
        };
        assert_eq!(count_of(&arena, low, OpKind::Floor), 1);
        assert_eq!(count_of(&arena, high, OpKind::Floor), 1);
        assert_eq!(arena.node(guard.runs), ExprNode::Binary(OpKind::Lt, m, n));
        assert_eq!(count_of(&arena, guard.domain, OpKind::Le), 4);

        let mut arena = ExprArena::new();
        let u = uniform(&mut arena, 3.0);
        let same = counted(Monoid::SUM, &mut arena, Bound::Value(u), Bound::Value(u));
        let (arena, root) = guard_value_bounds_owned(arena, same);
        let domain = Guard::of(&arena, root).domain;
        assert_eq!(count_of(&arena, domain, OpKind::Le), 2, "one test of u");
    }

    /// The guard varies with its bounds and the fold with its body, so that the
    /// tests of a uniform are computed once per call while the loop over `X`
    /// runs per sample: every def takes the variance of the arena's transfer.
    #[test]
    fn the_guard_varies_with_the_bounds_and_the_fold_with_its_body() {
        let mut arena = ExprArena::new();
        let u = uniform(&mut arena, 8.0);
        let fold = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Value(u));
        let (arena, root) = guard_value_bounds_owned(arena, fold);
        let variance = compute_arena_variance(&arena);
        let guard = Guard::of(&arena, root);
        let at = |id: ExprId| variance[id.0 as usize];
        assert_eq!(at(guard.domain), Variance::CONST);
        assert_eq!(at(guard.runs), Variance::CONST);
        assert_eq!(at(guard.fold), Variance::X, "the body reads X");
        assert_eq!(at(root), at(guard.fold), "and so does the guard around it");
    }

    /// A bound on an enclosing fold's index varies with that fold's binder, so
    /// the tests of the inner count are the outer fold's and run once per outer
    /// trip, and the outer fold stays the root.
    #[test]
    fn a_bound_on_an_enclosing_index_is_guarded_inside_that_fold() {
        let mut arena = ExprArena::new();
        let outer = Binder::from_slot(1).expect("slot 1 exists");
        let i = arena.push_var(outer.var());
        let inner = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Value(i));
        let fold = Fold::new(Monoid::SUM, outer);
        let root = arena.push_reduce(fold, Bound::Lit(0), Bound::Lit(8), inner);
        let (arena, root) = guard_value_bounds_owned(arena, root);

        let ExprNode::Reduce { body, .. } = arena.node(root) else {
            panic!("the outer fold stays the root");
        };
        let guard = Guard::of(&arena, body);
        assert_eq!(guard.fold, inner);
        assert_eq!(readers(&arena, root, inner), 1);
        let variance = compute_arena_variance(&arena);
        assert_eq!(
            variance[guard.domain.0 as usize],
            Variance::from_var(outer.var())
        );
    }

    /// A bound that names no count is a value for the guard to make NaN: it is
    /// tested like any term, and nothing here panics.
    #[test]
    fn a_literal_outside_the_domain_is_still_guarded() {
        let mut arena = ExprArena::new();
        let half = arena.push_const(2.5);
        let fold = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Value(half));
        let (arena, root) = guard_value_bounds_owned(arena, fold);
        let guard = Guard::of(&arena, root);
        assert_eq!(count_of(&arena, guard.domain, OpKind::Floor), 1);
        let ExprNode::Binary(OpKind::Lt, _, end) = arena.node(guard.runs) else {
            panic!("the arm runs while lo < hi");
        };
        assert_eq!(end, half);
    }

    /// The arm that is not the loop is the monoid's identity, whatever the
    /// monoid: what an empty range folds to.
    #[test]
    fn the_identity_arm_is_the_monoids() {
        for (monoid, identity) in [
            (Monoid::SUM, 0.0),
            (Monoid::PRODUCT, 1.0),
            (Monoid::MIN, f32::INFINITY),
            (Monoid::MAX, f32::NEG_INFINITY),
        ] {
            let mut arena = ExprArena::new();
            let u = uniform(&mut arena, 8.0);
            let fold = counted(monoid, &mut arena, Bound::Lit(0), Bound::Value(u));
            let (arena, root) = guard_value_bounds_owned(arena, fold);
            let arm = Guard::of(&arena, root).identity;
            assert_eq!(arena.node(arm), ExprNode::Const(Lit::Real(identity)));
        }
    }

    /// Everything above the fold is rebuilt over its guard: a parent that read
    /// the fold reads the guarded fold, and the fold has no reader but the guard.
    #[test]
    fn a_parent_of_the_fold_reads_its_guard() {
        let mut arena = ExprArena::new();
        let u = uniform(&mut arena, 8.0);
        let fold = counted(Monoid::SUM, &mut arena, Bound::Lit(0), Bound::Value(u));
        let two = arena.push_const(2.0);
        let product = arena.push_binary(OpKind::Mul, fold, two);
        let (arena, root) = guard_value_bounds_owned(arena, product);
        let ExprNode::Binary(OpKind::Mul, guarded, _) = arena.node(root) else {
            panic!("the product survives");
        };
        assert_eq!(Guard::of(&arena, guarded).fold, fold);
        assert_eq!(readers(&arena, root, fold), 1);
        assert_eq!(count_of(&arena, root, OpKind::If), 2);
    }
}
