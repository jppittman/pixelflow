//! Per-rule growth distribution over real kernels — the measurement
//! `docs/plans/2026-08-31-guide-design-revision.md` §4.1 calls for before
//! anyone gates a rewrite on "will this grow the graph and give me
//! nothing": how many e-nodes does each rule actually add, per application,
//! under full production saturation?
//!
//! Same five named kernels `examples/rule_report.rs` and
//! `examples/oracle_filtered_budget_curves.rs` use (`swirl`, `circle_sdf`,
//! `poly`, `redundant`, `normalize`) — real shader math, not synthetic
//! corpus expressions, so the numbers here describe what the current rule
//! set actually does to code this repository would plausibly compile.
//!
//! Run: `cargo run -p pixelflow-search --example growth_report --features saturation-telemetry`

use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::egraph::{GrowthTelemetry, Optimizer, RuleId, RuleSet, Vocabulary, insert};

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

fn main() {
    let cases: Vec<KernelCase> = vec![
        ("swirl", swirl),
        ("circle_sdf", circle_sdf),
        ("poly", poly),
        ("redundant", redundant),
        ("normalize", normalize),
    ];

    let rules = RuleSet::production();
    eprintln!("production rule set: {} rules", rules.len());
    let mut total = GrowthTelemetry::default();
    let mut total_applications = 0u64;

    for (name, build) in &cases {
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
        println!(
            "=== {name}: {node_count} nodes in, {} classes, {} applications, stop={:?} ===",
            out.stats.classes, out.stats.applications, out.stats.stop
        );
        total_applications += out.stats.applications;
        total.merge(growth);
    }

    println!(
        "\n=== AGGREGATE over {} kernels, {total_applications} total applications ===",
        cases.len()
    );
    println!(
        "{:<28} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "rule", "apps", "med.add", "max.add", "zero%", "no-op%"
    );

    let mut rows: Vec<(String, RuleId, u64, usize, usize, f64, f64)> = total
        .by_rule()
        .map(|(id, g)| {
            (
                rules.index_of(id).and_then(|idx| rules.label_of(idx)),
                id,
                g.applications(),
                g.median_nodes_added(),
                g.max_nodes_added(),
                g.zero_growth_fraction() * 100.0,
                g.no_op_fraction() * 100.0,
            )
        })
        .map(|(label, id, apps, med, max, zero_pct, noop_pct)| {
            (
                label.unwrap_or_else(|| format!("<unknown {id}>")),
                id,
                apps,
                med,
                max,
                zero_pct,
                noop_pct,
            )
        })
        .collect();
    // Busiest rule first: the report is read top-down for where the budget went.
    rows.sort_by_key(|row| core::cmp::Reverse(row.2));

    for (label, _id, apps, med, max, zero_pct, noop_pct) in &rows {
        println!("{label:<28} {apps:>7} {med:>9} {max:>9} {zero_pct:>8.1}% {noop_pct:>8.1}%");
    }

    let unlabeled = total.unlabeled();
    if unlabeled.applications() > 0 {
        println!(
            "<unlabeled>                  {:>7} {:>9} {:>9} {:>8.1}% {:>8.1}%",
            unlabeled.applications(),
            unlabeled.median_nodes_added(),
            unlabeled.max_nodes_added(),
            unlabeled.zero_growth_fraction() * 100.0,
            unlabeled.no_op_fraction() * 100.0,
        );
    }
}
