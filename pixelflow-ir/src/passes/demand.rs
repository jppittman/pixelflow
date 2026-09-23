//! **Demand**: the condition under which a value is observed.
//!
//! Every value in a kernel's DAG is observed under some condition. Ask *when*
//! that someone looks at it and you get a predicate over the masks above it:
//!
//! ```text
//! demand(root)                        = true
//! demand(m)   for S = Select(m, a, b) ⊇ demand(S)
//! demand(a)                           ⊇ demand(S) ∧ m
//! demand(b)                           ⊇ demand(S) ∧ ¬m
//! demand(v)   for consumers c₁ … cₙ   = ⋁ᵢ demand_edge(cᵢ → v)
//! ```
//!
//! One backward pass over the DAG in reverse topological order computes it
//! for every value at once. It is control dependence, and it belongs to the
//! graph — not to any select, and not to any scope's slice of a schedule.
//!
//! Executes §1 of `docs/plans/2026-09-07-demand-is-a-dag-property.md`. See
//! [`ordering`](self#the-sort-this-does-not-give-you) below for the part of
//! that plan this module found to be false.
//!
//! # Where this lives, and why it is generic
//!
//! `docs/plans/2026-09-09-exprarena-on-dag.md` moved this pass here from
//! `pixelflow-codegen/src/emit/demand.rs`, which computed it over the
//! *schedule* — `ValueId`-keyed `Def`/`ScheduledOp` sequences, a
//! codegen-only representation. The algebra ([`Demand`], [`Literal`]) and
//! the one backward pass ([`demand_of`]) that propagates it do not care what
//! a "value" is; they care only that values come in a topological order and
//! that one shape (a select-like branch) strengthens the predicate its
//! operands inherit. So the key type and the edge-finding are parameters,
//! not the algorithm: [`demand_of`] is generic over `K: Ord + Copy`, and
//! takes the topological order and the per-node edges as arguments.
//!
//! `pixelflow-codegen`'s `emit::guards` still needs demand keyed by
//! `ValueId` over its own schedule shapes (`ScheduledOp`, at whatever scope
//! a select's telemetry is computed for — a scope's local schedule, not the
//! top-level arena, since that is where `analyze_select_guards` is actually
//! called from, deep inside register allocation's per-scope machinery). It
//! gets that by calling this same generic [`demand_of`] with `ValueId` as
//! the key and a small closure describing a `ScheduledOp`'s edges — not by
//! duplicating the DNF algebra or the backward pass. That is the "smallest
//! public surface" split promised in
//! docs/plans/2026-09-09-exprarena-on-dag.md: the codegen-specific
//! `ScheduledOp`/`ValueId` types never cross into `pixelflow-ir`, and the
//! DNF algebra never has a second implementation.
//!
//! [`demand_of_arena`] is the `ExprId`-keyed instantiation over an
//! [`ExprArena`] directly — what a future `pixelflow-ir` or
//! `pixelflow-search` caller (the extraction-cost use, C2a) reaches for.
//!
//! # The sort this does not give you
//!
//! The plan claims demand also *is* a scheduler: sort values by demand,
//! weakest first, topologically within a demand, and equal-demand values
//! come out contiguous by construction with nothing left to cluster. The
//! claim rests on an invariant it states as, for every producer `u` of a
//! consumer `v`, `demand(u) ⊇ demand(v)`.
//!
//! That invariant is false, and so is its mirror. Both counterexamples are
//! ordinary shapes rather than contrived ones, and both are pinned by tests
//! below:
//!
//! - **A select arm breaks superset.** For `S = Select(m, a, b)` at the
//!   root, `demand(S)` is `true` and `demand(a)` is `m`. `a` produces `S`,
//!   and `m ⊉ true`. Sorting weakest-first puts the select *before* the arm
//!   it consumes.
//! - **A value shared across both arms breaks subset.** Give `p` two
//!   consumers, one in each arm: `demand(p) = m ∨ ¬m`, while its consumer
//!   in the true arm has demand `m`. Now the producer's demand is strictly
//!   weaker, so sorting strongest-first puts the consumer before `p`. This
//!   is what CSE across arms produces, which is to say it is the common
//!   case and not a corner.
//!
//! So demand orders neither way on its own, and a schedule keyed by it is
//! not topological. What survives is demand as a *property* — what a region
//! is guarded on, which is what `pixelflow-codegen::emit::guards` needs.
//! Making regions contiguous remains real work, which is what
//! `cluster_select_arms` is, and this module does not replace it.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::kind::OpKind;

/// How many clauses a predicate may carry before it is widened to
/// [`Demand::always`].
///
/// A **compile-budget knob**, not a correctness bound, and it must not be
/// described as one. Widening can only lose a guard, never skip a demanded
/// value, so any value here is sound; this one is chosen so a pathological
/// graph cannot make the backward pass quadratic in clause count.
const MAX_CLAUSES: usize = 8;

/// One `(mask, polarity)` literal: a mask value, and whether the condition
/// is that mask being **set** or **clear**.
///
/// `pub`: a caller instantiating [`demand_of`] with its own key type builds
/// `Literal`s in its `edges_of` closure (`pixelflow-codegen`'s
/// `demand_of_schedule` does, for `ValueId`), and reads them back out of a
/// [`Demand`] it queries — both need to name the type and its two fields.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Literal<K> {
    /// The value carrying the mask.
    pub mask: K,
    /// `true` when the condition is the mask being *clear* — the false arm.
    pub clear: bool,
}

impl<K: Copy> Literal<K> {
    /// The condition that `mask` is set.
    #[must_use]
    pub fn set(mask: K) -> Self {
        Self { mask, clear: false }
    }

    /// The condition that `mask` is clear.
    #[must_use]
    pub fn clear(mask: K) -> Self {
        Self { mask, clear: true }
    }

    /// The same mask, the other way round.
    fn negated(self) -> Self {
        Self {
            mask: self.mask,
            clear: !self.clear,
        }
    }
}

/// A conjunction of literals. The empty conjunction is `true`.
type Clause<K> = BTreeSet<Literal<K>>;

/// The condition under which a value is observed, in disjunctive normal
/// form: a disjunction of conjunctions.
///
/// Kept subsumption-reduced — no clause is a superset of another — and `==`
/// compares those reduced forms. They are not canonical: `m ∨ ¬m` is true
/// but is not [`Demand::always`], since only consensus would see it (a test
/// below pins that), so two equal predicates can compare unequal. Not `Ord`:
/// nothing orders a `Demand`, and the one use an order had, a schedule
/// sorted by demand, is refuted in the module doc's "The sort this does not
/// give you".
///
/// `pub`: this is [`demand_of`]'s return value's value type, and every
/// caller of `demand_of` — in this crate or across the boundary in
/// `pixelflow-codegen` — has to be able to name and query it. Its own
/// fields stay private; the algebra below (`is_never`, `implies`,
/// `and_literal`, …) is the whole interface, so a caller cannot build one
/// that violates the subsumption-reduced invariant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Demand<K> {
    /// The disjunction. Empty means *never*; containing the empty clause
    /// means *always*.
    clauses: BTreeSet<Clause<K>>,
}

// Not `#[derive(Default)]`: derive would add a `K: Default` bound that
// nothing here needs — an empty `BTreeSet` requires only `Ord` on its
// element, never a default value of `K` itself.
impl<K> Default for Demand<K> {
    fn default() -> Self {
        Self {
            clauses: BTreeSet::new(),
        }
    }
}

// Every method below is `pub` for the same reason `Demand` itself is: a
// caller across the crate boundary builds and combines predicates (a
// `Select`'s mask/arms in `edges_of`) and reads them back (a guard analysis
// asking `is_always`/`implies`), and this algebra is the only legal way to
// do either — see `Demand`'s own doc.
impl<K: Ord + Copy> Demand<K> {
    /// Observed on no lane — the identity for [`Self::or_with`], and what
    /// every value starts as before the backward pass reaches its
    /// consumers.
    #[must_use]
    pub fn never() -> Self {
        Self::default()
    }

    /// Observed on every lane: the disjunction containing the empty
    /// conjunction.
    #[must_use]
    pub fn always() -> Self {
        let mut clauses = BTreeSet::new();
        let _inserted = clauses.insert(Clause::new());
        Self { clauses }
    }

    /// Whether this is [`Self::never`] — no consumer observes the value.
    #[must_use]
    pub fn is_never(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Whether this is [`Self::always`] — no guard can skip the value.
    #[must_use]
    pub fn is_always(&self) -> bool {
        self.clauses.iter().any(BTreeSet::is_empty)
    }

    /// Disjoin `other` into this predicate.
    ///
    /// Clause union, then subsumption: a clause that contains another is
    /// the more specific of the two, so the disjunction already covers it.
    /// Widens to [`Self::always`] past [`MAX_CLAUSES`].
    pub fn or_with(&mut self, other: &Self) {
        if self.is_always() || other.is_never() {
            return;
        }
        if other.is_always() {
            *self = Self::always();
            return;
        }
        for clause in &other.clauses {
            let _inserted = self.clauses.insert(clause.clone());
        }
        self.reduce();
    }

    /// This predicate conjoined with one literal.
    ///
    /// The literal joins every clause; a clause that would then hold both
    /// polarities of one mask is a contradiction and drops out. Dropping
    /// every clause yields [`Self::never`], which is correct: the value is
    /// observed nowhere.
    #[must_use]
    pub fn and_literal(&self, lit: Literal<K>) -> Self {
        let negated = lit.negated();
        let mut out = Self::never();
        for clause in &self.clauses {
            if clause.contains(&negated) {
                continue;
            }
            let mut extended = clause.clone();
            let _inserted = extended.insert(lit);
            let _inserted = out.clauses.insert(extended);
        }
        out.reduce();
        out
    }

    /// Whether this predicate implies `other` — every lane observing this
    /// also observes `other`.
    ///
    /// Conservative: decided by subsumption, so it can answer `false` for a
    /// pair that a full satisfiability check would relate. Sound in the
    /// direction it is used — a `false` costs a guard, never correctness.
    #[must_use]
    pub fn implies(&self, other: &Self) -> bool {
        if other.is_always() || self.is_never() {
            return true;
        }
        self.clauses
            .iter()
            .all(|mine| other.clauses.iter().any(|theirs| theirs.is_subset(mine)))
    }

    /// Drop subsumed clauses, then widen if the budget is spent.
    fn reduce(&mut self) {
        if self.is_always() {
            *self = Self::always();
            return;
        }
        let kept: BTreeSet<Clause<K>> = self
            .clauses
            .iter()
            .filter(|candidate| {
                !self
                    .clauses
                    .iter()
                    .any(|other| other != *candidate && other.is_subset(candidate))
            })
            .cloned()
            .collect();
        self.clauses = kept;
        if self.clauses.len() > MAX_CLAUSES {
            *self = Self::always();
        }
    }
}

/// Every value's demand, by one backward pass over `order`.
///
/// `order` must be topological — producers before consumers — so that
/// visiting it in reverse reaches every consumer of a value before the
/// value itself. `edges_of(k, observed)` names `k`'s operands, each paired
/// with the demand it inherits from `k`'s own `observed` demand: every
/// operand inherits it unchanged except where `k` is a branch, which
/// strengthens what each side inherits by one literal (see [`Literal`]).
/// That one case is the only reason this takes a closure instead of a fixed
/// notion of "children" — a plain DAG walk would do the rest.
///
/// Values no consumer observes come back [`Demand::never`]; that includes
/// anything reachable but dead, which is a fact worth having on its own.
///
/// `pub`: this is the one definition of the pass
/// (docs/plans/2026-09-09-exprarena-on-dag.md), and `pixelflow-codegen`'s
/// `emit::guards` instantiates it directly for its own schedule shapes —
/// see the module doc's "Where this lives, and why it is generic".
pub fn demand_of<K, F>(
    order: impl DoubleEndedIterator<Item = K>,
    root: K,
    mut edges_of: F,
) -> BTreeMap<K, Demand<K>>
where
    K: Ord + Copy,
    F: FnMut(K, &Demand<K>) -> Vec<(K, Demand<K>)>,
{
    let mut demand: BTreeMap<K, Demand<K>> = BTreeMap::new();
    let _previous = demand.insert(root, Demand::always());

    for k in order.rev() {
        let observed = demand.get(&k).cloned().unwrap_or_default();
        // Nothing reads this value, so nothing it reads is read through it.
        if observed.is_never() {
            continue;
        }
        for (operand, edge) in edges_of(k, &observed) {
            demand.entry(operand).or_default().or_with(&edge);
        }
    }
    demand
}

/// [`demand_of`], keyed by [`ExprId`] over an [`ExprArena`] directly.
///
/// The one op that strengthens demand is [`OpKind::Select`]: its mask is
/// observed wherever the select is, but each arm only where the mask
/// agrees. Every other node passes its own demand through unchanged to
/// every child — including a `Gather`, whose index is as demanded as the
/// load, and a `Reduce`/`Guard`/`Write`, whose one child is the body/mask/
/// value [`ExprArena::children`] already yields.
///
/// `pub`: the arena-keyed instantiation belongs beside the generic one for
/// any `pixelflow-ir`/`pixelflow-search` caller that wants demand over a
/// kernel directly rather than over a codegen schedule — the extraction
/// cost use, C2a, is the one currently planned (see the module doc's §
/// "Where this lives, and why it is generic").
#[must_use]
pub fn demand_of_arena(arena: &ExprArena, root: ExprId) -> BTreeMap<ExprId, Demand<ExprId>> {
    demand_of(
        arena.nodes().map(|(id, _)| id),
        root,
        |id, observed| match arena.node(id) {
            ExprNode::Ternary(OpKind::Select, mask, if_true, if_false) => {
                alloc::vec![
                    (mask, observed.clone()),
                    (if_true, observed.and_literal(Literal::set(mask))),
                    (if_false, observed.and_literal(Literal::clear(mask))),
                ]
            }
            _ => arena.children(id).map(|c| (c, observed.clone())).collect(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ExprArena;

    fn v(n: u32) -> ExprId {
        ExprId(n)
    }

    /// `Select(m, a, b)` at the root: the mask is always observed, each arm
    /// only where the mask agrees.
    #[test]
    fn a_select_gives_each_arm_its_own_polarity() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let m = a.push_binary(OpKind::Lt, x, y);
        let t = a.push_unary(OpKind::Sqrt, x);
        let f = a.push_unary(OpKind::Abs, y);
        let root = a.push_ternary(OpKind::Select, m, t, f);

        let demand = demand_of_arena(&a, root);

        assert!(demand[&root].is_always(), "the root is always observed");
        assert!(demand[&m].is_always(), "a mask is observed with its select");
        assert_eq!(demand[&t], Demand::always().and_literal(Literal::set(m)));
        assert_eq!(demand[&f], Demand::always().and_literal(Literal::clear(m)));
    }

    /// The case the per-select shape refuses: one value under the true arm
    /// of one select and the false arm of another is `m₁ ∨ ¬m₂`, which is a
    /// two-clause predicate rather than "not exclusive".
    #[test]
    fn a_value_under_two_selects_is_a_disjunction() {
        // s1 = Select(m1, shared, y); s2 = Select(m2, y, shared); root = s1 + s2
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let m1 = a.push_binary(OpKind::Lt, x, y);
        let m2 = a.push_binary(OpKind::Gt, x, y);
        let shared = a.push_unary(OpKind::Sqrt, x);
        let s1 = a.push_ternary(OpKind::Select, m1, shared, y);
        let s2 = a.push_ternary(OpKind::Select, m2, y, shared);
        let root = a.push_binary(OpKind::Add, s1, s2);

        let demand = demand_of_arena(&a, root);

        let expected = {
            let mut d = Demand::always().and_literal(Literal::set(m1));
            d.or_with(&Demand::always().and_literal(Literal::clear(m2)));
            d
        };
        assert_eq!(demand[&shared], expected);
        assert!(
            !demand[&shared].is_always(),
            "a shared value is still guardable — that is the point"
        );
    }

    /// Nesting conjoins rather than nesting a special case.
    #[test]
    fn a_nested_select_conjoins_its_masks() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let m1 = a.push_binary(OpKind::Lt, x, y);
        let m2 = a.push_binary(OpKind::Gt, x, y);
        let inner_val = a.push_unary(OpKind::Sqrt, x);
        let inner = a.push_ternary(OpKind::Select, m2, inner_val, y);
        let root = a.push_ternary(OpKind::Select, m1, inner, x);

        let demand = demand_of_arena(&a, root);

        let expected = Demand::always()
            .and_literal(Literal::set(m1))
            .and_literal(Literal::set(m2));
        assert_eq!(demand[&inner_val], expected);
    }

    /// A value nothing reads is `never`, which is how the pass reports
    /// unreachable-from-root nodes without being asked.
    #[test]
    fn an_unread_value_is_demanded_nowhere() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_unary(OpKind::Sqrt, x);
        let dead = a.push_unary(OpKind::Abs, x);

        let demand = demand_of_arena(&a, root);

        assert!(demand[&root].is_always());
        assert!(demand.get(&dead).is_none_or(Demand::is_never));
    }

    // ───────────────────── the algebra ─────────────────────

    #[test]
    fn a_contradiction_is_never() {
        let d = Demand::always()
            .and_literal(Literal::set(v(0)))
            .and_literal(Literal::clear(v(0)));
        assert!(d.is_never());
    }

    #[test]
    fn disjoining_a_weaker_clause_subsumes_the_stronger() {
        // (m₀ ∧ m₁) ∨ m₀  =  m₀
        let strong = Demand::always()
            .and_literal(Literal::set(v(0)))
            .and_literal(Literal::set(v(1)));
        let weak = Demand::always().and_literal(Literal::set(v(0)));
        let mut d = strong;
        d.or_with(&weak);
        assert_eq!(d, weak);
    }

    #[test]
    fn always_absorbs_and_never_is_the_identity() {
        let mut d = Demand::always().and_literal(Literal::set(v(0)));
        let before = d.clone();
        d.or_with(&Demand::never());
        assert_eq!(d, before, "never is the identity for or");
        d.or_with(&Demand::always());
        assert!(d.is_always(), "always absorbs");
    }

    #[test]
    fn implication_runs_the_way_the_lattice_does() {
        let m0 = Demand::always().and_literal(Literal::set(v(0)));
        let both = m0.and_literal(Literal::set(v(1)));
        assert!(both.implies(&m0), "stronger implies weaker");
        assert!(!m0.implies(&both), "weaker does not imply stronger");
        assert!(m0.implies(&Demand::always()));
        assert!(Demand::never().implies(&m0));
    }

    /// Past the budget a predicate widens to `always`, which loses a guard
    /// and never a value.
    #[test]
    fn widening_past_the_clause_budget_is_always() {
        let mut d = Demand::never();
        for mask in 0..(MAX_CLAUSES as u32 + 2) {
            // Each clause names a distinct mask, so none subsumes another.
            d.or_with(&Demand::always().and_literal(Literal::set(v(mask))));
        }
        assert!(d.is_always());
    }

    /// **Skippable is not movable**, minimally.
    ///
    /// Two selects share a mask. The inner select `inner` is shared — the
    /// root reads it directly as well as through the outer select `outer`
    /// — so it must stay put. Its true arm `t` is observed only where the
    /// mask is set, so demand calls it exclusive to the outer select's true
    /// arm, and demand is right: skipping it when the mask is all-false is
    /// sound.
    ///
    /// It is still not *movable*: `outer` reads `inner`, and a partition
    /// that moved arm-exclusive values ahead of shared ones would put
    /// `inner` before its own operand. `pixelflow-codegen::emit::guards`'s
    /// stricter rule — every consumer skipped with it — rejects `t`, and
    /// has to.
    #[test]
    fn a_shared_inner_select_makes_its_arms_skippable_but_not_movable() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let m = a.push_binary(OpKind::Lt, x, y);
        let t = a.push_unary(OpKind::Sqrt, x);
        let f = a.push_unary(OpKind::Abs, y);
        let inner = a.push_ternary(OpKind::Select, m, t, f);
        let outer = a.push_ternary(OpKind::Select, m, inner, y);
        let root = a.push_binary(OpKind::Add, outer, inner);

        let demand = demand_of_arena(&a, root);

        // The inner select is read outside the outer one, so nothing may
        // skip it.
        assert!(demand[&inner].is_always());

        // Its arms are still observed only under one polarity each.
        let outer_true = demand[&outer].and_literal(Literal::set(m));
        assert!(
            demand[&t].implies(&outer_true),
            "the inner true arm is skippable when the shared mask is clear"
        );
        assert!(
            !demand[&inner].implies(&outer_true),
            "while the select reading it is not — which is why moving the \
             arm past it would be illegal"
        );
    }

    // ───────────── the sort the plan expected, and does not get ─────────────

    /// **A select arm refutes `demand(producer) ⊇ demand(consumer)`.**
    ///
    /// `docs/plans/2026-09-07-demand-is-a-dag-property.md` §"The invariant
    /// that makes demand a scheduler" claims that invariant, and concludes
    /// a schedule sorted by demand weakest-first is topological with
    /// equal-demand values contiguous "by construction". An arm is a
    /// producer of its select and is demanded strictly less often, so
    /// weakest-first emits the select before the arm it reads.
    #[test]
    fn a_select_arm_is_demanded_less_than_the_select_that_reads_it() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let m = a.push_binary(OpKind::Lt, x, y);
        let t = a.push_unary(OpKind::Sqrt, x);
        let root = a.push_ternary(OpKind::Select, m, t, y);

        let demand = demand_of_arena(&a, root);

        let arm = &demand[&t];
        let select = &demand[&root];
        assert!(
            arm.implies(select),
            "the arm is demanded no more than the select"
        );
        assert!(
            !select.implies(arm),
            "and strictly less — so the producer is not a demand-superset \
             of its consumer, and weakest-first is not topological"
        );
    }

    /// **A value shared across both arms refutes the mirror claim.**
    ///
    /// So strongest-first is not topological either. `shared` feeds one
    /// consumer in each arm, so its demand is `m ∨ ¬m` — strictly weaker
    /// than either consumer's — and `shared` must still be computed first.
    /// This is what CSE across arms produces, so it is the ordinary case.
    #[test]
    fn a_value_shared_across_arms_is_demanded_more_than_its_consumers() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let m = a.push_binary(OpKind::Lt, x, y);
        // shared, read from inside both arms.
        let shared = a.push_binary(OpKind::Add, x, y);
        let t = a.push_unary(OpKind::Sqrt, shared);
        let f = a.push_unary(OpKind::Abs, shared);
        let root = a.push_ternary(OpKind::Select, m, t, f);

        let demand = demand_of_arena(&a, root);

        let shared_demand = &demand[&shared];
        let consumer = &demand[&t];
        assert!(
            consumer.implies(shared_demand),
            "the consumer is demanded no more than what it reads"
        );
        assert!(
            !shared_demand.implies(consumer),
            "and strictly less — so the producer is not a demand-subset of \
             its consumer, and strongest-first is not topological either"
        );
        assert!(
            !shared_demand.is_always(),
            "m ∨ ¬m is semantically true, but subsumption alone does not \
             see it: consensus would, and this module does not implement it"
        );
    }
}
