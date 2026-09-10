//! **Demand**: the condition under which a value is observed.
//!
//! Every value in a schedule is computed for someone. Ask *when* that
//! someone looks at it and you get a predicate over the masks above it:
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
//! graph — not to any select, and not to any scope's slice of the schedule.
//!
//! [`guards`](super::guards) currently answers a special case of this
//! question per `Select`: which schedule entries one arm owns *outright*. A
//! value read by the true arm of `S₁` and the false arm of `S₂` has demand
//! `m₁ ∨ ¬m₂`, which that shape cannot say, so it says "not exclusive" and
//! the value is computed on every batch. Here it is one predicate with two
//! clauses.
//!
//! Executes §1 of `docs/plans/2026-09-07-demand-is-a-dag-property.md`. See
//! [`ordering`](self#the-sort-this-does-not-give-you) below for the part of
//! that plan this module found to be false.
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
//! is guarded on, which is what [`super::guards`] needs. Making regions
//! contiguous remains real work, which is what `cluster_select_arms` is,
//! and this module does not delete it.
//!
//! # Skippable is not movable
//!
//! Demand also finds **more** exclusivity than `guards` does, and the extra
//! is not usable by the partition as it stands. That is not a bug in
//! either; they answer different questions.
//!
//! `guards` calls a value exclusive to an arm when every one of its
//! consumers is skipped along with it. Demand calls it exclusive when it is
//! only ever *observed* under that arm's polarity — which is weaker, and is
//! the right condition for skipping. Two selects sharing a mask separate
//! them: the inner select's arms are observed only when the mask is set, so
//! demand calls them exclusive to the outer select's true arm, and it is
//! right. But the inner select itself is shared, and it reads them. The
//! three-way partition moves shared values ahead of arm-exclusive ones, so
//! it would place a consumer before its own operand.
//!
//! **Skippable is a property of the value; movable is a property of the
//! order.** Acting on the first as though it were the second is caught by
//! `cluster_select_arms`'s own `is_topological` assertion, which is how
//! this was found. The gap between the two counts is recorded per select as
//! `demand_exclusive` against `exclusive` under
//! `PIXELFLOW_GUARD_TELEMETRY`: it is the headroom a partition that ordered
//! *within* its groups would unlock, and it is the measurement C1b should
//! be justified by rather than the scheduling claim above.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::ScheduledOp;
use super::regalloc::{Def, ValueId};

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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub(crate) struct Literal {
    /// The value carrying the mask.
    pub(crate) mask: ValueId,
    /// `true` when the condition is the mask being *clear* — the false arm.
    pub(crate) clear: bool,
}

impl Literal {
    /// The condition that `mask` is set.
    pub(crate) fn set(mask: ValueId) -> Self {
        Self { mask, clear: false }
    }

    /// The condition that `mask` is clear.
    pub(crate) fn clear(mask: ValueId) -> Self {
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
type Clause = BTreeSet<Literal>;

/// The condition under which a value is observed, in disjunctive normal
/// form: a disjunction of conjunctions.
///
/// Kept canonical — the clause set is subsumption-reduced, so no clause is
/// a superset of another — which makes equal predicates structurally equal
/// and lets `Ord` group a schedule by demand.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub(crate) struct Demand {
    /// The disjunction. Empty means *never*; containing the empty clause
    /// means *always*.
    clauses: BTreeSet<Clause>,
}

impl Demand {
    /// Observed on no lane — the identity for [`Self::or_with`], and what
    /// every value starts as before the backward pass reaches its
    /// consumers.
    pub(crate) fn never() -> Self {
        Self {
            clauses: BTreeSet::new(),
        }
    }

    /// Observed on every lane: the disjunction containing the empty
    /// conjunction.
    pub(crate) fn always() -> Self {
        let mut clauses = BTreeSet::new();
        let _inserted = clauses.insert(Clause::new());
        Self { clauses }
    }

    /// Whether this is [`Self::never`] — no consumer observes the value.
    pub(crate) fn is_never(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Whether this is [`Self::always`] — no guard can skip the value.
    pub(crate) fn is_always(&self) -> bool {
        self.clauses.iter().any(BTreeSet::is_empty)
    }

    /// Disjoin `other` into this predicate.
    ///
    /// Clause union, then subsumption: a clause that contains another is
    /// the more specific of the two, so the disjunction already covers it.
    /// Widens to [`Self::always`] past [`MAX_CLAUSES`].
    pub(crate) fn or_with(&mut self, other: &Self) {
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
    pub(crate) fn and_literal(&self, lit: Literal) -> Self {
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
    pub(crate) fn implies(&self, other: &Self) -> bool {
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
        let kept: BTreeSet<Clause> = self
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

/// Every value's demand, by one backward pass over `schedule`.
///
/// `schedule` must be in topological order — producers before consumers,
/// which is what [`Def`] sequences already are — so that visiting it in
/// reverse reaches every consumer of a value before the value itself.
///
/// Values no consumer observes come back [`Demand::never`]; that includes
/// anything scheduled but dead, which is a fact worth having on its own.
#[must_use]
pub(crate) fn demand_of(schedule: &[Def], root: ValueId) -> BTreeMap<ValueId, Demand> {
    let mut demand: BTreeMap<ValueId, Demand> = BTreeMap::new();
    let _previous = demand.insert(root, Demand::always());

    for def in schedule.iter().rev() {
        let observed = demand.get(&def.value).cloned().unwrap_or_default();
        // Nothing reads this value, so nothing it reads is read through it.
        if observed.is_never() {
            continue;
        }
        for (operand, edge) in edges(&def.op, &observed) {
            demand.entry(operand).or_default().or_with(&edge);
        }
    }
    demand
}

/// One value's operands paired with the demand each inherits.
///
/// A `Select` is the only op that strengthens: its mask is observed
/// wherever the select is, but each arm only where the mask agrees. Every
/// other op passes its own demand through unchanged — including `Gather`,
/// whose index is as demanded as the load.
fn edges(op: &ScheduledOp, observed: &Demand) -> Vec<(ValueId, Demand)> {
    match op {
        ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => Vec::new(),
        ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) | ScheduledOp::Gather(a, _) => {
            alloc::vec![(*a, observed.clone())]
        }
        ScheduledOp::Binary(_, a, b) => {
            alloc::vec![(*a, observed.clone()), (*b, observed.clone())]
        }
        ScheduledOp::Ternary(pixelflow_ir::OpKind::Select, mask, if_true, if_false) => {
            alloc::vec![
                (*mask, observed.clone()),
                (*if_true, observed.and_literal(Literal::set(*mask))),
                (*if_false, observed.and_literal(Literal::clear(*mask))),
            ]
        }
        ScheduledOp::Ternary(_, a, b, c) => alloc::vec![
            (*a, observed.clone()),
            (*b, observed.clone()),
            (*c, observed.clone()),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;

    fn v(n: u32) -> ValueId {
        ValueId(n)
    }

    fn def(n: u32, op: ScheduledOp) -> Def {
        Def { value: v(n), op }
    }

    /// `Select(m, a, b)` at the root: the mask is always observed, each arm
    /// only where the mask agrees.
    #[test]
    fn a_select_gives_each_arm_its_own_polarity() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, v(0), v(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, v(0))),
            def(4, ScheduledOp::Unary(OpKind::Abs, v(1))),
            def(5, ScheduledOp::Ternary(OpKind::Select, v(2), v(3), v(4))),
        ];
        let demand = demand_of(&schedule, v(5));

        assert!(demand[&v(5)].is_always(), "the root is always observed");
        assert!(
            demand[&v(2)].is_always(),
            "a mask is observed with its select"
        );
        assert_eq!(
            demand[&v(3)],
            Demand::always().and_literal(Literal::set(v(2)))
        );
        assert_eq!(
            demand[&v(4)],
            Demand::always().and_literal(Literal::clear(v(2)))
        );
    }

    /// The case the per-select shape refuses: one value under the true arm
    /// of one select and the false arm of another is `m₁ ∨ ¬m₂`, which is a
    /// two-clause predicate rather than "not exclusive".
    #[test]
    fn a_value_under_two_selects_is_a_disjunction() {
        // s1 = Select(m1, shared, y);  s2 = Select(m2, y, shared);
        // root = s1 + s2
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, v(0), v(1))),
            def(3, ScheduledOp::Binary(OpKind::Gt, v(0), v(1))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, v(0))),
            def(5, ScheduledOp::Ternary(OpKind::Select, v(2), v(4), v(1))),
            def(6, ScheduledOp::Ternary(OpKind::Select, v(3), v(1), v(4))),
            def(7, ScheduledOp::Binary(OpKind::Add, v(5), v(6))),
        ];
        let demand = demand_of(&schedule, v(7));

        let expected = {
            let mut d = Demand::always().and_literal(Literal::set(v(2)));
            d.or_with(&Demand::always().and_literal(Literal::clear(v(3))));
            d
        };
        assert_eq!(demand[&v(4)], expected);
        assert!(
            !demand[&v(4)].is_always(),
            "a shared value is still guardable — that is the point"
        );
    }

    /// Nesting conjoins rather than nesting a special case.
    #[test]
    fn a_nested_select_conjoins_its_masks() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, v(0), v(1))),
            def(3, ScheduledOp::Binary(OpKind::Gt, v(0), v(1))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, v(0))),
            def(5, ScheduledOp::Ternary(OpKind::Select, v(3), v(4), v(1))),
            def(6, ScheduledOp::Ternary(OpKind::Select, v(2), v(5), v(0))),
        ];
        let demand = demand_of(&schedule, v(6));

        let expected = Demand::always()
            .and_literal(Literal::set(v(2)))
            .and_literal(Literal::set(v(3)));
        assert_eq!(demand[&v(4)], expected);
    }

    /// A value nothing reads is `never`, which is how the pass reports dead
    /// schedule entries without being asked.
    #[test]
    fn an_unread_value_is_demanded_nowhere() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Sqrt, v(0))),
            def(2, ScheduledOp::Unary(OpKind::Abs, v(0))),
        ];
        let demand = demand_of(&schedule, v(1));

        assert!(demand[&v(1)].is_always());
        assert!(demand.get(&v(2)).is_none_or(Demand::is_never));
    }

    // ───────────────── the algebra ─────────────────

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
    /// Two selects share a mask. The inner select `v5` is shared — the root
    /// reads it directly as well as through the outer select `v6` — so it
    /// must stay put. Its true arm `v3` is observed only where the mask is
    /// set, so demand calls it exclusive to the outer select's true arm,
    /// and demand is right: skipping it when the mask is all-false is
    /// sound.
    ///
    /// It is still not *movable*. `v5` reads `v3`, and
    /// `guards::cluster_select_arms` places shared values ahead of
    /// arm-exclusive ones, which would put `v5` before its own operand.
    /// `guards`'s stricter rule — every consumer skipped with it — rejects
    /// `v3`, and has to.
    #[test]
    fn a_shared_inner_select_makes_its_arms_skippable_but_not_movable() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, v(0), v(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, v(0))),
            def(4, ScheduledOp::Unary(OpKind::Abs, v(1))),
            def(5, ScheduledOp::Ternary(OpKind::Select, v(2), v(3), v(4))),
            def(6, ScheduledOp::Ternary(OpKind::Select, v(2), v(5), v(1))),
            def(7, ScheduledOp::Binary(OpKind::Add, v(6), v(5))),
        ];
        let demand = demand_of(&schedule, v(7));

        // The inner select is read outside the outer one, so nothing may
        // skip it.
        assert!(demand[&v(5)].is_always());

        // Its arms are still observed only under one polarity each.
        let outer_true = demand[&v(6)].and_literal(Literal::set(v(2)));
        assert!(
            demand[&v(3)].implies(&outer_true),
            "the inner true arm is skippable when the shared mask is clear"
        );
        assert!(
            !demand[&v(5)].implies(&outer_true),
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
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, v(0), v(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, v(0))),
            def(4, ScheduledOp::Ternary(OpKind::Select, v(2), v(3), v(1))),
        ];
        let demand = demand_of(&schedule, v(4));

        let arm = &demand[&v(3)];
        let select = &demand[&v(4)];
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
    /// So strongest-first is not topological either. `p` feeds one consumer
    /// in each arm, so its demand is `m ∨ ¬m` — strictly weaker than either
    /// consumer's — and `p` must still be computed first. This is what CSE
    /// across arms produces, so it is the ordinary case.
    #[test]
    fn a_value_shared_across_arms_is_demanded_more_than_its_consumers() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, v(0), v(1))),
            // p, read from inside both arms.
            def(3, ScheduledOp::Binary(OpKind::Add, v(0), v(1))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, v(3))),
            def(5, ScheduledOp::Unary(OpKind::Abs, v(3))),
            def(6, ScheduledOp::Ternary(OpKind::Select, v(2), v(4), v(5))),
        ];
        let demand = demand_of(&schedule, v(6));

        let shared = &demand[&v(3)];
        let consumer = &demand[&v(4)];
        assert!(
            consumer.implies(shared),
            "the consumer is demanded no more than what it reads"
        );
        assert!(
            !shared.implies(consumer),
            "and strictly less — so the producer is not a demand-subset of \
             its consumer, and strongest-first is not topological either"
        );
        assert!(
            !shared.is_always(),
            "m ∨ ¬m is semantically true, but subsumption alone does not \
             see it: consensus would, and this module does not implement it"
        );
    }
}
