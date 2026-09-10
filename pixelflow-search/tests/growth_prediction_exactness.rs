//! `EGraph::predicted_growth` must equal the measured growth
//! (`egraph::growth`) for every rewrite application, exactly — not to a
//! tolerance.
//!
//! This is deliberately *not* a standalone comparison harness: the oracle
//! check lives once, in `EGraph::apply_action_measured` (an `assert_eq!`
//! that fires unconditionally whenever growth telemetry is on, comparing
//! `predicted_growth` — computed immediately before an action commits —
//! against the delta `apply_action_measured` measures around that same
//! commit). A second comparison here would be exactly the "second,
//! unvalidated computation of the same fact" `egraph::growth`'s module doc
//! warns against. This file's job is only to *drive* enough real rewrite
//! activity, on real shader math, for that assertion to be exercised
//! thousands of times over — if `predicted_growth` ever disagrees with what
//! actually got added, this test does not pass quietly; it panics with the
//! rule index and both numbers (see the `assert_eq!` site).
//!
//! Same five named kernels `examples/growth_report.rs` (and
//! `examples/rule_report.rs`, `examples/oracle_filtered_budget_curves.rs`)
//! use — real shader math, not synthetic corpus expressions, chosen because
//! `docs/results/...` measured the finding this test guards against those
//! exact kernels.
//!
//! Whole file gated on `saturation-telemetry`: with the feature off,
//! `EGraph::enable_growth_telemetry` does not exist, and there is nothing
//! here to test.

#![cfg(feature = "saturation-telemetry")]

use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::egraph::{Optimizer, Vocabulary, insert};

/// sin(sqrt(x*x + y*y) * freq) * amp + bias — the swirl shader core.
fn swirl() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let kf = a.push_const(3.0);
    let sf = a.push_binary(OpKind::Mul, s, kf);
    let sn = a.push_unary(OpKind::Sin, sf);
    let ka = a.push_const(0.5);
    let prod = a.push_binary(OpKind::Mul, sn, ka);
    let kb = a.push_const(0.5);
    let out = a.push_binary(OpKind::Add, prod, kb);
    (a, out)
}

/// Circle SDF: sqrt((x-cx)^2 + (y-cy)^2) - r.
fn circle_sdf() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let cx = a.push_const(0.3);
    let cy = a.push_const(-0.2);
    let dx = a.push_binary(OpKind::Sub, x, cx);
    let dy = a.push_binary(OpKind::Sub, y, cy);
    let dx2 = a.push_binary(OpKind::Mul, dx, dx);
    let dy2 = a.push_binary(OpKind::Mul, dy, dy);
    let sum = a.push_binary(OpKind::Add, dx2, dy2);
    let dist = a.push_unary(OpKind::Sqrt, sum);
    let r = a.push_const(0.5);
    let out = a.push_binary(OpKind::Sub, dist, r);
    (a, out)
}

/// FMA-bait polynomial: a*x*x + b*x + c (Horner-able, fusion-able).
fn poly() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let ka = a.push_const(2.0);
    let kb = a.push_const(-3.0);
    let kc = a.push_const(1.0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let ax2 = a.push_binary(OpKind::Mul, ka, xx);
    let bx = a.push_binary(OpKind::Mul, kb, x);
    let s1 = a.push_binary(OpKind::Add, ax2, bx);
    let out = a.push_binary(OpKind::Add, s1, kc);
    (a, out)
}

/// Redundancy bait: (x+y)*(x+y) + 2*(x+y) — CSE + distribution territory.
fn redundant() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_binary(OpKind::Add, x, y);
    let s2 = a.push_binary(OpKind::Mul, s, s);
    let two = a.push_const(2.0);
    let ts = a.push_binary(OpKind::Mul, two, s);
    let out = a.push_binary(OpKind::Add, s2, ts);
    (a, out)
}

/// Division/sqrt bait: x / sqrt(x*x + y*y) (normalize — rsqrt rewrites).
fn normalize() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let out = a.push_binary(OpKind::Div, x, s);
    (a, out)
}

/// A named kernel builder: (label, constructor).
type KernelCase = (&'static str, fn() -> (ExprArena, ExprId));

const CASES: &[KernelCase] = &[
    ("swirl", swirl),
    ("circle_sdf", circle_sdf),
    ("poly", poly),
    ("redundant", redundant),
    ("normalize", normalize),
];

/// Saturate every real-shader case under production budgets with growth
/// telemetry on. `EGraph::apply_action_measured`'s internal `assert_eq!` is
/// what actually checks `predicted_growth` against the measured delta, on
/// every one of the (thousands of) applications this drives — this test
/// passing means every one of them agreed.
#[test]
fn predicted_growth_matches_measured_growth_on_real_shaders() {
    let mut total_applications = 0u64;

    for (name, build) in CASES {
        let (arena, root) = build();
        let mut optimizer = Optimizer::production();
        let mut eg = optimizer.egraph();
        eg.enable_growth_telemetry();
        let root_class =
            insert(&arena, root, &mut eg, Vocabulary::Templates).expect("insert into e-graph");
        let node_count = arena.len();
        let out = optimizer.run(&mut eg, root_class, node_count);

        let growth = eg
            .growth_telemetry()
            .expect("growth telemetry was enabled above");
        let recorded: u64 = growth.by_rule().map(|(_, g)| g.applications()).sum::<u64>()
            + growth.unlabeled().applications();
        assert_eq!(
            recorded, out.stats.applications,
            "{name}: growth telemetry must record exactly the applications the run counted"
        );
        assert!(
            out.stats.applications > 0,
            "{name}: must actually drive rewrite activity for this test to mean anything"
        );
        total_applications += out.stats.applications;
    }

    // A floor, not a target: the finding this test guards against was
    // measured at ~10,800 applications over this exact corpus
    // (`examples/growth_report.rs`) — an order of magnitude below that would
    // mean the corpus stopped exercising real rewrite activity.
    assert!(
        total_applications > 1_000,
        "only {total_applications} applications across the corpus — too few for \
         predicted_growth's exactness to have been meaningfully exercised"
    );
}
