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

use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, Term};
use pixelflow_ir::{OpKind, Rooted, Uniform, binding::BindingTable};
use pixelflow_search::egraph::{Budget, EClassId, Optimizer, RuleSet, all_rules};

/// A graph plus the declarations its leaves index — how every fixture below
/// holds one, and what a [`Term`] is built from.
type Graph = (Rooted<ExprData>, Environment);

fn term(g: &Graph) -> Term<'_> {
    Term::new(g.0.entry(), &g.1)
}

/// Evaluate a term through the language's own reference interpreter — the
/// same one the rewrite-soundness tests use, so "denotes the same function"
/// means here what it means there.
fn eval(g: &Graph, vars: &[f32; 2], arg: (Uniform, f32)) -> f32 {
    let bindings = BindingTable::empty()
        .bind_uniforms(&g.1, &[(arg.0.identity(), arg.1)])
        .expect("the fixture's argument survives every graph built from it");
    pixelflow_ir::eval_scalar(term(g), vars, &bindings)
}

/// One argument value per point in [`POINTS`], so the sweep sweeps the
/// argument too rather than reading one default seven times.
const ARGS: [f32; 7] = [0.75, -1.25, 0.5, 2.0, -0.125, 1.5, -3.0];

const POINTS: [[f32; 2]; 7] = [
    [0.0, 0.0],
    [1.0, 2.0],
    [-1.5, 0.25],
    [0.5, -0.5],
    [3.0, 1.0],
    [-0.75, -2.25],
    [2.5, 2.5],
];

/// A mid-sized expression with sharing, several rule families in reach, and
/// no transcendental domain hazards at the sample points.
///
/// The third leaf is an argument rather than a third axis: a lattice has two,
/// and what makes this fixture worth optimizing is the sharing, not where the
/// leaf comes from.
///
/// The handle comes back with it. An *unbound* argument reads its default at
/// every point, which would quietly turn this file's seven-point sweep into
/// one point evaluated seven times; [`ARGS`] gives it a distinct value per
/// point and [`eval`] binds it by identity.
///
/// **Call this once and thread the result.** `Uniform::new` mints a fresh
/// `UniformIdentity`, so this function is not idempotent: two calls declare
/// two *different* arguments whose handles compare unequal, which is correct
/// for identity-by-instance and a trap for a fixture. Binding one call's
/// handle against another call's arena fails in `bind_uniforms`, and it
/// presents as "the arena lost its declaration" — a very plausible wrong
/// diagnosis, and one this file already paid for once.
fn fixture() -> (Graph, Uniform) {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let arg = Uniform::new(0.75);
    let slot = a.declare_uniform(arg.decl());
    let z = a.push_uniform(slot);
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
    (a.finish(&[root]), arg)
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
///
/// Takes the fixture rather than building its own. `Uniform::new` mints a
/// fresh identity per call, so a second `fixture()` here would declare a
/// *different* argument than the caller holds, and binding the caller's handle
/// against this arena would fail — which is exactly what it did, and looked
/// for a while like extraction dropping declarations. It does not: both this
/// route and `optimize_runtime_term` carry the declaration through.
fn optimize_with(mut optimizer: Optimizer, input: Term<'_>) -> Graph {
    let mut eg = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        input,
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let node_count = pixelflow_search::egraph::reachable_count_term(input);
    let optimized = optimizer.run(&mut eg, root_class, node_count);
    optimized.to_rooted(&eg, root_class)
}

/// **Both optimization routes carry a uniform's declaration through.** Not a
/// law of the e-graph so much as a fact the link step depends on:
/// `optimize_runtime_term` maps uniform *identities* to block offsets
/// downstream, and says so in a comment. A comment is what CLAUDE.md warns
/// something else eventually breaks, so this checks it.
///
/// It also pins the thing that made this look broken: the declaration
/// survives, and a handle from a *different* `fixture()` call is what does
/// not bind.
#[test]
fn extraction_preserves_the_arguments_declaration() {
    let (input, arg) = fixture();
    assert_eq!(
        input.1.uniforms.len(),
        1,
        "the fixture declares one argument"
    );

    let out = optimize_with(Optimizer::production(), term(&input));
    assert_eq!(
        out.1.uniforms.len(),
        1,
        "e-graph extraction dropped the argument's declaration"
    );
    assert!(
        BindingTable::empty()
            .bind_uniforms(&out.1, &[(arg.identity(), 1.0)])
            .is_ok(),
        "the extracted graph must still answer to the handle the fixture minted"
    );

    let shape = pixelflow_ir::variance::LatticeShape::new([64, 64]);
    if let Some(o) = pixelflow_search::runtime::optimize_runtime_term(term(&input), shape) {
        assert_eq!(
            o.1.uniforms.len(),
            1,
            "the production route dropped the argument's declaration"
        );
    }
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
    let (input, arg) = fixture();
    let expected: Vec<f32> = POINTS
        .iter()
        .zip(ARGS)
        .map(|(p, a)| eval(&input, p, (arg, a)))
        .collect();

    let budgets = [
        Budget::Production,
        Budget::Applications(1),
        Budget::Applications(64),
        Budget::Applications(4096),
        Budget::Explicit {
            iterations: 1,
            classes: 5_000,
            applications: None,
        },
        Budget::Explicit {
            iterations: 12,
            classes: 64,
            applications: None,
        },
    ];

    for name in POLICIES {
        for budget in budgets {
            let out = optimize_with(
                Optimizer::production().rules(policy(name)).budget(budget),
                term(&input),
            );
            for ((point, &want), a) in POINTS.iter().zip(&expected).zip(ARGS) {
                let got = eval(&out, point, (arg, a));
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
    let (input, _arg) = fixture();
    let mut optimizer = Optimizer::production().budget(budget);
    let mut eg = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term(&input),
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");

    // Re-add every node of the input to recover its class. `add` is
    // idempotent on an already-present node, so this reads the graph rather
    // than growing it.
    let ids: Vec<EClassId> = input
        .0
        .iter()
        .map(|node| {
            pixelflow_search::egraph::insert_term(
                Term::new(node, &input.1),
                &mut eg,
                pixelflow_search::egraph::Vocabulary::Templates,
            )
            .expect("insert into e-graph")
        })
        .collect();
    let _ = optimizer.run(&mut eg, root_class, input.0.len());

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
    let (input, arg) = fixture();
    for n in [0u64, 1, 2, 3, 5, 13, 100] {
        let out = optimize_with(
            Optimizer::production().budget(Budget::Applications(n)),
            term(&input),
        );
        for (point, a) in POINTS.iter().zip(ARGS) {
            let want = eval(&input, point, (arg, a));
            let got = eval(&out, point, (arg, a));
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
    let (input, _arg) = fixture();
    let mut optimizer = Optimizer::production();
    let mut eg = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term(&input),
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let optimized = optimizer.run(&mut eg, root_class, input.0.len());
    let out = optimized.to_rooted(&eg, root_class);

    let mut probe = eg.clone();
    let re_added = pixelflow_search::egraph::insert_term(
        term(&out),
        &mut probe,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
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
    let (input, _arg) = fixture();
    let reference = optimize_with(Optimizer::production(), term(&input));
    for _ in 0..8 {
        let again = optimize_with(Optimizer::production(), term(&input));
        assert_eq!(
            arena_shape(&again),
            arena_shape(&reference),
            "production extraction must not vary run to run"
        );
    }
}

/// The stop reason is typed, so "converged" is distinguishable from "ran out
/// of budget" — which the old boolean `saturated` was not.
#[test]
fn the_stop_reason_names_which_limit_bound() {
    use pixelflow_search::egraph::SaturationStop;

    let (input, _arg) = fixture();
    let mut starved = Optimizer::production().budget(Budget::Applications(3));
    let mut eg = starved.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term(&input),
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let out = starved.run(&mut eg, root_class, input.0.len());
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
/// The same shape `expr::encode` writes — nodes in the graph's own
/// topological order, children named by dense ordinal, so sharing is part of
/// the rendering — spelled here rather than delegated because `encode`
/// refuses a binding (a `UniformIdentity` is minted per process and cannot be
/// written down) and this fixture declares one. Two runs that agree here
/// produced the same term with the same sharing.
fn arena_shape(g: &Graph) -> String {
    use std::fmt::Write as _;
    let root = g.0.entry();
    let dag = root.dag();
    let mut ordinal = dag.side_table(usize::MAX);
    let mut text = String::new();
    for (i, node) in dag.iter().enumerate() {
        ordinal[node] = i;
        let kids: Vec<usize> = node.children().map(|c| ordinal[c]).collect();
        writeln!(text, "{i} {:?} {kids:?}", *node).expect("fmt");
    }
    write!(text, "root {}", ordinal[root]).expect("fmt");
    text
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
    let (input, _arg) = fixture();

    let mut silent = Optimizer::production();
    let mut eg = silent.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term(&input),
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let quiet = silent.run(&mut eg, root_class, input.0.len());
    assert_eq!(
        eg.provenance().recorded_count(),
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
    let root_class2 = pixelflow_search::egraph::insert_term(
        term(&input),
        &mut eg2,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let loud = watched.run(&mut eg2, root_class2, input.0.len());

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
    let quiet_out = quiet.to_rooted(&eg, root_class);
    let loud_out = loud.to_rooted(&eg2, root_class2);
    assert_eq!(
        arena_shape(&quiet_out),
        arena_shape(&loud_out),
        "attaching an observer must not change what is extracted"
    );
}

// ---------------------------------------------------------------------------
// L5 — every `Optimize` preserves denotation
// ---------------------------------------------------------------------------
//
// L4 above varies policy and budget over one `Optimizer` against one fixture.
// This section varies the *implementation*: `Optimize` is the trait the
// compiler pipeline is actually built from, and `Rewritten::Changed` is
// documented as "A new term, denoting what the input denoted" — a law that
// until now was stated only in that doc comment.
//
// It matters more than it looks, because a pipeline composes these: `Then`
// short-circuits on decline, so a composition can produce a term neither half
// would have produced alone. `Declined` and `Unchanged` pass trivially —
// optimization is never required for correctness — so what is asserted is
// only ever "if you rewrote it, it still means the same thing".
//
// Structure is deliberately not asserted. An optimizer may return a
// completely different term; that is the job. Reassociation and one-versus-two
// roundings move the last bits by design, so the bound is the same relative
// tolerance L4 uses.

use pixelflow_ir::Kernel;
use pixelflow_ir::optimize::{Identity, Optimize, Rewritten, Then};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_search::Tier;
use pixelflow_search::egraph::ops::Vocabulary;
use pixelflow_search::saturate_pass::Saturate;

/// `eval` for terms with no uniform to bind — the corpus below is built
/// through the public `Kernel` API, which mints none.
fn eval_plain(t: Term<'_>, vars: &[f32; 2]) -> f32 {
    pixelflow_ir::eval_scalar(t, vars, &BindingTable::empty())
}

/// L5's own sample points, strictly inside the corpus's domain.
///
/// [`POINTS`] ranges over negatives, and this corpus contains `sqrt` and `ln`,
/// which CLAUDE.md gives a *documented domain* and which return NaN outside
/// it. Measured on `sqrt(x).ln().exp()` at `x = -1.5`: unoptimized evaluates
/// to 8.5e37 (the saturating `exp` ceiling), optimized to NaN. Neither is a
/// miscompile — outside a function's domain the language promises nothing to
/// preserve — but it does mean **L5 holds in-domain and only in-domain**,
/// which is implied by that section of CLAUDE.md and had not been written
/// down. Sampling the positive quadrant is what makes the law statable.
const IN_DOMAIN_POINTS: [[f32; 2]; 6] = [
    [0.5, 0.25],
    [1.0, 2.0],
    [2.75, 0.125],
    [0.0625, 3.5],
    [7.0, 0.75],
    [0.001, 1.5],
];

/// Terms built through the public `Kernel` composition API, so the corpus is
/// what composition really produces rather than what a test author would
/// hand-assemble.
///
/// No `Dwrt` term appears: `eval_scalar` refuses one outright ("no scalar eval
/// for binary Dwrt — lower it first"), so the *input* side of the comparison
/// would not evaluate and the law could not be stated. What a derivative
/// denotes is pinned by `pixelflow-compiler/tests/derivative_under_warp.rs`
/// and by `passes.rs`'s own `dwrt_tests`.
fn denotation_corpus() -> Vec<(&'static str, Kernel)> {
    let x = Kernel::x();
    let y = Kernel::y();
    let k = Kernel::constant;

    vec![
        ("x", x.clone()),
        ("x + y", x.add(&y)),
        // Non-commutative, non-associative, in both operand orders. An
        // optimizer that wrongly commutes `Sub` or `Div` is sign-flipped or
        // reciprocal, and nothing symmetric would notice.
        ("x - y", x.sub(&y)),
        ("y - x", y.sub(&x)),
        ("x / y", x.div(&y)),
        ("y / x", y.div(&x)),
        ("(x - y) - 1", x.sub(&y).sub(&k(1.0))),
        ("1 - (x - y)", k(1.0).sub(&x.sub(&y))),
        // Difference of squares: a reassociation target where a wrong
        // commutation survives as a plausible-looking number.
        ("(x - y) * (x + y)", x.sub(&y).mul(&x.add(&y))),
        ("(x - y).abs()", x.sub(&y).abs()),
        ("x.min(y)", x.min(&y)),
        ("x.max(y)", x.max(&y)),
        // FMA fusion candidates: one rounding after, two before.
        ("x * y + 1", x.mul(&y).add(&k(1.0))),
        ("x * y - 1", x.mul(&y).sub(&k(1.0))),
        // Identity, annihilator, and all-constant folds.
        ("x + 0", x.add(&k(0.0))),
        ("x * 1", x.mul(&k(1.0))),
        ("x * 0", x.mul(&k(0.0))),
        ("2 * 3 + 4", k(2.0).mul(&k(3.0)).add(&k(4.0))),
        ("(x*x + y*y).sqrt()", x.mul(&x).add(&y.mul(&y)).sqrt()),
        // 1/sqrt(v) -> rsqrt is a real rewrite with a real precision delta.
        ("1 / x.sqrt()", k(1.0).div(&x.sqrt())),
        // Sharing: the same subterm reached twice must stay one value.
        ("sin(x) + sin(x)", x.sin().add(&x.sin())),
        ("sqrt(x).ln().exp()", x.sqrt().ln().exp()),
        ("select(x < y, x, y)", x.lt(&y).select(&x, &y)),
        // Composition under a warp — the shape the font path builds.
        ("(x*x).at(2x, y)", x.mul(&x).at(&x.mul(&k(2.0)), &y)),
    ]
}

/// Assert L5 for one implementation. Returns how many corpus terms it
/// actually rewrote, so a caller can tell "preserved denotation" apart from
/// "did nothing at all".
fn assert_preserves_denotation(label: &str, opt: &mut dyn Optimize) -> usize {
    let mut changed = 0;

    for (name, kernel) in denotation_corpus() {
        let input = kernel.term();

        let out = match opt.optimize(input) {
            // A declined term compiles unoptimized; nothing to compare.
            Rewritten::Declined | Rewritten::Unchanged => continue,
            Rewritten::Changed(rooted, env) => {
                changed += 1;
                (rooted, env)
            }
        };

        for point in &IN_DOMAIN_POINTS {
            let want = eval_plain(input, point);
            let got = eval_plain(term(&out), point);

            // NaN counts as agreeing with NaN: an optimizer is not required to
            // invent a value where the input had none.
            let agrees = (want.is_nan() && got.is_nan())
                || got == want
                || (got - want).abs() <= 1e-4 * want.abs().max(1.0);

            assert!(
                agrees,
                "{label} changed what `{name}` denotes at {point:?}: \
                 {got} != {want}"
            );
        }
    }

    changed
}

#[test]
fn the_identity_optimizer_rewrites_nothing() {
    // Vacuous by construction, and that is the point: it pins the control arm
    // every measurement of an optimizer is read against.
    assert_eq!(assert_preserves_denotation("Identity", &mut Identity), 0);
}

/// A pass that *eliminates* a construct cannot be checked against its own
/// input: `eval_scalar` refuses a `Dwrt` or a `Reduce` outright, so the
/// "before" side of `assert_preserves_denotation` does not evaluate and every
/// such term is skipped. Which is why `LowerDwrt` and `ExpandReduce` needed
/// their own harness, and why the two tests they had were worthless before
/// they got one — measured, not supposed: the corpus contains no `Dwrt` and
/// no `Reduce`, so both passes returned `Unchanged` for all 24 terms and the
/// assertions ran **zero** times. They would have passed against a pass that
/// miscompiled every input.
///
/// The way to state a law about a construct you cannot evaluate is an
/// *independent oracle*: write the answer out separately, in ops the
/// evaluator does have, and require the pass to agree with it. The oracle is
/// hand-written calculus and arithmetic here, never something the pass under
/// test produced — an oracle derived from the subject checks nothing, which is
/// the same trap the `sin` range-reduction bug lived in (CLAUDE.md, "Precision
/// is on the table; range is not": the JIT and its oracle shared an expansion,
/// agreed bit-for-bit on garbage, and every same-form test passed).
fn assert_lowers_to_oracle(label: &str, opt: &mut dyn Optimize, subject: &Kernel, oracle: &Kernel) {
    let out = match opt.optimize(subject.term()) {
        Rewritten::Changed(rooted, env) => (rooted, env),
        // Not a skip. A pass whose whole job is to eliminate a construct has
        // failed if it leaves one standing.
        other => panic!("{label} must rewrite a term carrying its construct, got {other:?}"),
    };

    for point in &IN_DOMAIN_POINTS {
        let want = eval_plain(oracle.term(), point);
        let got = eval_plain(term(&out), point);
        let agrees =
            (want.is_nan() && got.is_nan()) || (got - want).abs() <= 1e-4 * want.abs().max(1.0);
        assert!(
            agrees,
            "{label} disagrees with the hand-written oracle at {point:?}: {got} != {want}"
        );
    }
}

/// `LowerDwrt` against derivatives written out by hand.
#[test]
fn lowering_dwrt_agrees_with_hand_written_derivatives() {
    let x = Kernel::x();
    let y = Kernel::y();

    // d/dx (x·x) = 2x
    assert_lowers_to_oracle(
        "LowerDwrt d/dx(x²)",
        &mut LowerDwrt,
        &x.mul(&x).dx(),
        &Kernel::constant(2.0).mul(&x),
    );

    // d/dy (x·y + y) = x + 1
    assert_lowers_to_oracle(
        "LowerDwrt d/dy(xy + y)",
        &mut LowerDwrt,
        &x.mul(&y).add(&y).dy(),
        &x.add(&Kernel::constant(1.0)),
    );

    // d/dx √(x² + y²) = x / √(x² + y²) — the glyph gradient's shape, and the
    // one case here whose derivative is not a polynomial.
    let r = x.mul(&x).add(&y.mul(&y)).sqrt();
    assert_lowers_to_oracle("LowerDwrt d/dx(hypot)", &mut LowerDwrt, &r.dx(), &x.div(&r));
}

/// `ExpandReduce` against sums written out term by term.
#[test]
fn expanding_reduce_agrees_with_hand_written_sums() {
    let x = Kernel::x();
    let k = Kernel::constant;

    // Σ_{i<4} i = 0 + 1 + 2 + 3 = 6, independent of the sample point.
    assert_lowers_to_oracle(
        "ExpandReduce Σi",
        &mut ExpandReduce,
        &Kernel::sum_over(4, |i| i.clone()),
        &k(6.0),
    );

    // Σ_{i<3} (x + i) = 3x + 3
    assert_lowers_to_oracle(
        "ExpandReduce Σ(x+i)",
        &mut ExpandReduce,
        &Kernel::sum_over(3, |i| x.add(i)),
        &k(3.0).mul(&x).add(&k(3.0)),
    );
}

#[test]
fn saturating_preserves_denotation_under_each_vocabulary() {
    // `Templates` is the macro and `Dwrt`-expansion tiers' vocabulary;
    // `Runtime` adds the mask and integer-domain ops.
    // The tier only decides where a telemetry record is written, which is a
    // side channel this law says nothing about; each vocabulary is paired
    // with the tier that actually uses it so the pairing cannot read as
    // arbitrary.
    for (vocab, tier) in [
        (Vocabulary::Templates, Tier::Macro),
        (Vocabulary::Runtime, Tier::Runtime),
    ] {
        let mut sat = Saturate::with(Optimizer::production(), vocab, tier);
        let changed = assert_preserves_denotation(&format!("Saturate<{vocab:?}>"), &mut sat);
        assert!(
            changed > 0,
            "saturation under {vocab:?} rewrote nothing across the whole \
             corpus — either the corpus stopped containing anything \
             optimizable, or the optimizer silently became a no-op"
        );
    }
}

#[test]
fn a_composed_pipeline_preserves_denotation() {
    // `Then` short-circuits on decline, which is the one place a composition
    // can produce a term neither half would have produced alone.
    let mut pipeline = Then(
        LowerDwrt,
        Saturate::with(Optimizer::production(), Vocabulary::Templates, Tier::Macro),
    );
    assert_preserves_denotation("Then(LowerDwrt, Saturate)", &mut pipeline);
}
