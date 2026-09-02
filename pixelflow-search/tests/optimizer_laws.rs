//! The laws the optimizer API stands on, as tests rather than as arguments.
//!
//! `docs/plans/2026-09-02-optimizer-api.md` audits five proposed laws against
//! the code. Three of them were previously true but unenforced, which is a
//! weaker position than it sounds: an unenforced law is a comment, and this
//! codebase's own rule is that a convention written in a comment is an
//! invariant something else will eventually break.
//!
//! - **L2 (monotonicity)** — saturation only ever adds equalities, so the
//!   class partition under a larger budget refines the one under a smaller.
//!   This is what makes [`Budget`] a quality/compile-time dial rather than a
//!   correctness dial; without it every budget value would need its own
//!   review.
//! - **L3a (membership)** — re-adding an extracted term lands back in the
//!   class it was extracted from. Pinned here as a unit test, deliberately
//!   *not* as a public membership predicate: a law statement is not a reason
//!   to grow the API.
//! - **L4 (policy neutrality)** — the load-bearing one. Any ordering policy
//!   and any budget yield a term denoting the same function. This is what
//!   lets a future policy PR argue *quality* instead of correctness, and it
//!   is worth more than any per-policy review.
//!
//! L4 is exercised through the ordering lever that exists today — the rule
//! set's order, which is exactly what a `SaturationGuide` generalises from
//! "which rule next" to "which match next". A guide that only orders and
//! truncates is inside the same fence, and its own PR should extend the
//! `POLICIES` table here rather than re-derive the argument.

use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::{OpKind, binding::BindingTable};
use pixelflow_search::egraph::{Budget, EClassId, Optimizer, RuleSet, all_rules};

/// Evaluate an arena through the language's own reference interpreter — the
/// same one the rewrite-soundness tests use, so "denotes the same function"
/// means here what it means there.
fn eval(arena: &ExprArena, root: ExprId, vars: &[f32; 4]) -> f32 {
    pixelflow_ir::eval_scalar(arena, root, vars, &BindingTable::empty())
}

const POINTS: [[f32; 4]; 7] = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0, 4.0],
    [-1.5, 0.25, 2.0, -3.0],
    [0.5, -0.5, 1.25, 0.75],
    [3.0, 1.0, -2.0, 0.5],
    [-0.75, -2.25, 0.125, 1.0],
    [2.5, 2.5, 2.5, 2.5],
];

/// A mid-sized expression with sharing, several rule families in reach, and
/// no transcendental domain hazards at the sample points.
fn fixture() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);
    let one = a.push_const(1.0);
    let two = a.push_const(2.0);

    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let sum = a.push_binary(OpKind::Add, xx, yy);
    let scaled = a.push_binary(OpKind::Mul, sum, two);
    // `sum` is shared: once scaled, once inside the offset.
    let offset = a.push_binary(OpKind::Add, sum, one);
    let prod = a.push_binary(OpKind::Mul, scaled, offset);
    let with_z = a.push_binary(OpKind::Add, prod, z);
    let neg = a.push_unary(OpKind::Neg, with_z);
    let root = a.push_binary(OpKind::Sub, with_z, neg);
    (a, root)
}

/// The ordering policies under test, by name. Each permutes the rule
/// vocabulary, which is the ordering lever the current saturation loop
/// exposes: rules are applied in vector order within a round, so a
/// permutation genuinely changes *which equalities are discovered when*, and
/// under a budget it changes which are discovered at all.
///
/// A `SaturationGuide` is the same lever at finer grain — "which match next"
/// rather than "which rule next" — so a guide PR extends this list instead of
/// re-deriving the argument.
const POLICIES: [&str; 4] = ["declaration-order", "reversed", "rotated-17", "strided-7"];

/// Rules are trait objects and do not clone, so a policy is described by its
/// construction and rebuilt on demand rather than carried around.
fn policy(name: &str) -> RuleSet {
    let mut rules = all_rules();
    match name {
        "declaration-order" => {}
        "reversed" => rules.reverse(),
        "rotated-17" => rules.rotate_left(17),
        "strided-7" => {
            // A fixed stride over a vector whose length is coprime with the
            // stride visits every position exactly once.
            let n = rules.len();
            let mut slots: Vec<Option<_>> = rules.drain(..).map(Some).collect();
            let mut i = 0usize;
            for _ in 0..n {
                while slots[i % n].is_none() {
                    i += 1;
                }
                rules.push(slots[i % n].take().expect("visited slot is occupied"));
                i += 7;
            }
        }
        other => panic!("no such policy: {other}"),
    }
    RuleSet::new(rules)
}

/// Run one configuration end to end, returning the extracted arena.
fn optimize_with(mut optimizer: Optimizer) -> (ExprArena, ExprId) {
    let (arena, root) = fixture();
    let mut eg = optimizer.egraph();
    let root_class = eg.add_arena(&arena, root);
    let node_count = arena.len();
    let optimized = optimizer.run(&mut eg, root_class, node_count);
    optimized.to_arena(&eg, root_class)
}

// ---------------------------------------------------------------------------
// L4 — policy neutrality
// ---------------------------------------------------------------------------

/// **L4.** Every ordering policy, at every budget, extracts a term denoting
/// the same function.
///
/// Note what is asserted and what is not: the *denotation* is asserted, the
/// *cost* is not. Costs are expected to differ — a policy that could not
/// change the cost would not be worth having. That asymmetry is the whole
/// content of the law, and the reason a policy PR owes a quality measurement
/// rather than a correctness suite.
#[test]
fn every_ordering_policy_extracts_the_same_denotation() {
    let (input, input_root) = fixture();
    let expected: Vec<f32> = POINTS.iter().map(|p| eval(&input, input_root, p)).collect();

    let budgets = [
        Budget::Production,
        Budget::Applications(1),
        Budget::Applications(64),
        Budget::Applications(4096),
        Budget::Explicit {
            iterations: 1,
            classes: 5_000,
            allocated_classes: pixelflow_search::egraph::HARD_CLASS_LIMIT,
            applications: None,
        },
        Budget::Explicit {
            iterations: 12,
            classes: 64,
            allocated_classes: pixelflow_search::egraph::HARD_CLASS_LIMIT,
            applications: None,
        },
    ];

    for name in POLICIES {
        for budget in budgets {
            let (out, out_root) =
                optimize_with(Optimizer::production().rules(policy(name)).budget(budget));
            for (point, &want) in POINTS.iter().zip(&expected) {
                let got = eval(&out, out_root, point);
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "policy {name} at {budget:?} changed the denotation at {point:?}: \
                     {got} != {want}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// L2 — monotonicity
// ---------------------------------------------------------------------------

/// One pair of the input's nodes, and whether saturation has merged them.
type Merged = (usize, usize, bool);

/// Every pair of the input's nodes, classified by whether saturation has
/// merged them at this budget.
fn partition(budget: Budget) -> Vec<Merged> {
    let (arena, root) = fixture();
    let mut optimizer = Optimizer::production().budget(budget);
    let mut eg = optimizer.egraph();
    let root_class = eg.add_arena(&arena, root);

    // Re-add every node of the input to recover its class. `add` is
    // idempotent on an already-present node, so this reads the graph rather
    // than growing it.
    let ids: Vec<EClassId> = (0..arena.len())
        .map(|i| eg.add_arena(&arena, ExprId(i as u32)))
        .collect();
    let _ = optimizer.run(&mut eg, root_class, arena.len());

    let mut out = Vec::new();
    for a in 0..ids.len() {
        for b in (a + 1)..ids.len() {
            out.push((a, b, eg.find(ids[a]) == eg.find(ids[b])));
        }
    }
    out
}

/// **L2.** A larger budget refines the partition: everything equal at a
/// smaller budget is still equal at a larger one. Saturation has no
/// operation that removes an equality, and this is the test that says so.
#[test]
fn a_larger_budget_refines_the_partition() {
    let ladder = [
        Budget::Applications(1),
        Budget::Applications(8),
        Budget::Applications(64),
        Budget::Applications(512),
        Budget::Applications(4096),
    ];

    let mut previous: Option<(Budget, Vec<Merged>)> = None;
    for budget in ladder {
        let current = partition(budget);
        if let Some((smaller, prior)) = previous.take() {
            assert_eq!(prior.len(), current.len(), "same node set at every budget");
            for ((a, b, was_equal), (_, _, is_equal)) in prior.iter().zip(&current) {
                assert!(
                    !was_equal || *is_equal,
                    "nodes {a} and {b} were equal at {smaller:?} but not at {budget:?} — \
                     saturation removed an equality, which it has no operation to do"
                );
            }
        }
        previous = Some((budget, current));
    }
}

/// Budget truncation cannot make the graph unsound, which is the other half
/// of why `Budget` is safe: the term extracted at a starved budget still
/// denotes the input.
#[test]
fn a_starved_budget_still_denotes_the_input() {
    let (input, input_root) = fixture();
    for n in [0u64, 1, 2, 3, 5, 13, 100] {
        let (out, out_root) =
            optimize_with(Optimizer::production().budget(Budget::Applications(n)));
        for point in &POINTS {
            let want = eval(&input, input_root, point);
            let got = eval(&out, out_root, point);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "budget of {n} applications changed the denotation at {point:?}: {got} != {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// L3a — membership
// ---------------------------------------------------------------------------

/// **L3a.** Re-adding an extracted term lands back in the class it was
/// extracted from.
///
/// Deliberately a test and not an API: a membership predicate on `Optimizer`
/// would exist only to let a doc quote itself. Re-adding into a clone of the
/// graph and comparing canonical ids is the whole implementation.
#[test]
fn an_extracted_term_re_adds_into_its_own_class() {
    let (arena, root) = fixture();
    let mut optimizer = Optimizer::production();
    let mut eg = optimizer.egraph();
    let root_class = eg.add_arena(&arena, root);
    let optimized = optimizer.run(&mut eg, root_class, arena.len());
    let (out, out_root) = optimized.to_arena(&eg, root_class);

    let mut probe = eg.clone();
    let re_added = probe.add_arena(&out, out_root);
    assert_eq!(
        probe.find(re_added),
        probe.find(root_class),
        "the extracted term must re-add into the class it was extracted from"
    );
}

// ---------------------------------------------------------------------------
// Determinism — what `Budget` buys
// ---------------------------------------------------------------------------

/// The same term under the same budget produces the same extraction, every
/// time.
///
/// This was not true before `Budget`: the production presets carried a
/// 10/50/200 ms `hard_timeout` that `saturate_with_limits` broke on and
/// `saturate_with_full_budget` then reported as `saturated: true`, so the
/// compiler's output was a function of machine load and a truncated run was
/// indistinguishable from a converged one.
#[test]
fn the_same_budget_extracts_the_same_term() {
    let reference = optimize_with(Optimizer::production());
    for _ in 0..8 {
        let again = optimize_with(Optimizer::production());
        assert_eq!(
            arena_shape(&again.0, again.1),
            arena_shape(&reference.0, reference.1),
            "production extraction must not vary run to run"
        );
    }
}

/// The stop reason is typed, so "converged" is distinguishable from "ran out
/// of budget" — which the old boolean `saturated` was not.
#[test]
fn the_stop_reason_names_which_limit_bound() {
    use pixelflow_search::egraph::SaturationStop;

    let (arena, root) = fixture();
    let mut starved = Optimizer::production().budget(Budget::Applications(3));
    let mut eg = starved.egraph();
    let root_class = eg.add_arena(&arena, root);
    let out = starved.run(&mut eg, root_class, arena.len());
    assert_eq!(
        out.stats.stop,
        SaturationStop::ApplicationBudget,
        "an application budget of 3 must report that it is what stopped the run"
    );
    assert!(
        out.stats.applications <= 3,
        "an application budget must actually bound applications, got {}",
        out.stats.applications
    );
}

/// A structural rendering of the extracted DAG, for equality comparisons.
///
/// The arena is append-only and extraction emits children before parents, so
/// the node vector plus the root is already canonical for a given
/// configuration: two runs that agree here produced the same term with the
/// same sharing.
fn arena_shape(arena: &ExprArena, root: ExprId) -> String {
    format!("{root:?}|{:?}", arena.nodes_raw())
}

// ---------------------------------------------------------------------------
// G4 — observation is opt-in, and the budget does not depend on it
// ---------------------------------------------------------------------------

/// A recorder that keeps what it was told, so a test can look at it after
/// the optimizer has dropped its box.
/// What the recorder keeps per application: rule index, round, nodes minted,
/// whether the graph changed.
type Seen = (usize, usize, u64, bool);

#[derive(Clone, Default)]
struct Recorder(std::sync::Arc<std::sync::Mutex<Vec<Seen>>>);

impl pixelflow_search::egraph::Observer for Recorder {
    fn on_application(&mut self, record: &pixelflow_search::egraph::ApplicationRecord) {
        self.0.lock().expect("recorder lock").push((
            record.rule_idx,
            record.step,
            record.minted_count(),
            record.changed(),
        ));
    }
}

/// Production records nothing, an observer records everything, and the
/// application *count* is the same either way.
///
/// The count mattering is the point: the guided loop used to read its budget
/// off the provenance log's length, which made recording a load-bearing part
/// of saturation rather than an observation of it — so observation could not
/// be made optional without changing what the budget meant.
#[test]
fn observation_is_optional_and_does_not_move_the_budget() {
    let (arena, root) = fixture();

    let mut silent = Optimizer::production();
    let mut eg = silent.egraph();
    let root_class = eg.add_arena(&arena, root);
    let quiet = silent.run(&mut eg, root_class, arena.len());
    assert_eq!(
        eg.provenance().application_count(),
        0,
        "production must not build a provenance log"
    );
    assert!(
        quiet.stats.applications > 0,
        "applications are counted whether or not anyone is watching"
    );

    let recorder = Recorder::default();
    let mut watched = Optimizer::production().observe(Some(Box::new(recorder.clone())));
    let mut eg2 = watched.egraph();
    let root_class2 = eg2.add_arena(&arena, root);
    let loud = watched.run(&mut eg2, root_class2, arena.len());

    let seen = recorder.0.lock().expect("recorder lock").len();
    assert_eq!(
        seen as u64, loud.stats.applications,
        "the observer must see every application the run counted"
    );
    assert_eq!(
        quiet.stats.applications, loud.stats.applications,
        "attaching an observer must not change how much work saturation does"
    );

    // Both runs must also agree on the term: observation is observation.
    let (quiet_arena, quiet_root) = quiet.to_arena(&eg, root_class);
    let (loud_arena, loud_root) = loud.to_arena(&eg2, root_class2);
    assert_eq!(
        arena_shape(&quiet_arena, quiet_root),
        arena_shape(&loud_arena, loud_root),
        "attaching an observer must not change what is extracted"
    );
}
