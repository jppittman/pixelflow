//! Which rules earn their keep? Run hindsight-labeled episodes over
//! representative kernel expressions and print the per-rule load-bearing
//! report — the first empirical look at the rule library, pre-ML.
//!
//! Run: `cargo run --release -p pixelflow-search --example rule_report`

use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, Term};
use pixelflow_ir::{OpKind, Rooted};
use pixelflow_search::egraph::run_episode;
use pixelflow_search::math::all_rules;
use std::collections::BTreeMap;

/// sin(sqrt(x*x + y*y) * freq) * amp + bias — the swirl shader core.
fn swirl() -> Graph {
    let mut a = ExprBuilder::new();
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
    a.finish(&[out])
}

/// Circle SDF: sqrt((x-cx)^2 + (y-cy)^2) - r.
fn circle_sdf() -> Graph {
    let mut a = ExprBuilder::new();
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
    a.finish(&[out])
}

/// FMA-bait polynomial: a*x*x + b*x + c (Horner-able, fusion-able).
fn poly() -> Graph {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let ka = a.push_const(2.0);
    let kb = a.push_const(-3.0);
    let kc = a.push_const(1.0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let ax2 = a.push_binary(OpKind::Mul, ka, xx);
    let bx = a.push_binary(OpKind::Mul, kb, x);
    let s1 = a.push_binary(OpKind::Add, ax2, bx);
    let out = a.push_binary(OpKind::Add, s1, kc);
    a.finish(&[out])
}

/// Redundancy bait: (x+y)*(x+y) + 2*(x+y) — CSE + distribution territory.
fn redundant() -> Graph {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_binary(OpKind::Add, x, y);
    let s2 = a.push_binary(OpKind::Mul, s, s);
    let two = a.push_const(2.0);
    let ts = a.push_binary(OpKind::Mul, two, s);
    let out = a.push_binary(OpKind::Add, s2, ts);
    a.finish(&[out])
}

/// Division/sqrt bait: x / sqrt(x*x + y*y) (normalize — rsqrt rewrites).
fn normalize() -> Graph {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let out = a.push_binary(OpKind::Div, x, s);
    a.finish(&[out])
}

/// A graph plus the declarations its leaves index.
type Graph = (Rooted<ExprData>, Environment);

/// A named kernel builder: (label, constructor).
type KernelCase = (&'static str, fn() -> Graph);

fn main() {
    let cases: Vec<KernelCase> = vec![
        ("swirl", swirl),
        ("circle_sdf", circle_sdf),
        ("poly", poly),
        ("redundant", redundant),
        ("normalize", normalize),
    ];

    // Aggregate across episodes: rule name -> (fired, load_bearing).
    let mut agg: BTreeMap<String, (usize, usize)> = BTreeMap::new();

    for (name, build) in &cases {
        let (rooted, env) = build();
        let ep = run_episode(Term::new(rooted.entry(), &env), all_rules());
        println!("=== {name} ===");
        println!(
            "  e-graph: {} classes; applications: {}; load-bearing: {}",
            ep.egraph.num_classes(),
            ep.labels.labels.len(),
            ep.labels.load_bearing.len(),
        );
        println!("{}", ep.labels.format_rule_report(&ep.egraph));

        for (rule_idx, stats) in &ep.labels.rule_stats {
            let rname = match ep.egraph.rule(*rule_idx) {
                Some(r) => r.name().to_string(),
                None => format!("<rule {rule_idx}>"),
            };
            let e = agg.entry(rname).or_insert((0, 0));
            e.0 += stats.fired;
            e.1 += stats.load_bearing;
        }
    }

    println!("=== AGGREGATE over {} episodes ===", cases.len());
    let mut rows: Vec<_> = agg.into_iter().collect();
    rows.sort_by(|a, b| {
        (b.1.1 * a.1.0)
            .cmp(&(a.1.1 * b.1.0))
            .then(b.1.0.cmp(&a.1.0))
    });
    println!(
        "{:<40} {:>7} {:>7} {:>7}",
        "rule", "fired", "l-bear", "ratio"
    );
    for (name, (fired, lb)) in rows {
        let ratio = if fired > 0 {
            lb as f64 / fired as f64
        } else {
            0.0
        };
        println!("{name:<40} {fired:>7} {lb:>7} {ratio:>7.3}");
    }
}
