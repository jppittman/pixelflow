//! BwdGenerator quality test: generates expressions and benchmarks them via JIT.
//!
//! Run with: cargo test -p pixelflow-pipeline --test bwdgen_quality -- --ignored --nocapture

use std::collections::HashMap;

use pixelflow_ir::kind::OpKind;
use pixelflow_ir::expr::Expr;
use pixelflow_pipeline::jit_bench::{benchmark_jit, BenchError};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};
use pixelflow_search::nnue::factored::RuleTemplates;

const BATCH_SIZE: usize = 100;

/// Collect all OpKind variants used in an expression.
fn collect_ops(expr: &Expr, ops: &mut HashMap<OpKind, usize>) {
    match expr {
        Expr::Var(_) | Expr::Const(_) | Expr::Param(_) => {}
        Expr::Unary(op, child) => {
            *ops.entry(*op).or_default() += 1;
            collect_ops(child, ops);
        }
        Expr::Binary(op, lhs, rhs) => {
            *ops.entry(*op).or_default() += 1;
            collect_ops(lhs, ops);
            collect_ops(rhs, ops);
        }
        Expr::Ternary(op, a, b, c) => {
            *ops.entry(*op).or_default() += 1;
            collect_ops(a, ops);
            collect_ops(b, ops);
            collect_ops(c, ops);
        }
        Expr::Nary(op, children) => {
            *ops.entry(*op).or_default() += 1;
            for c in children {
                collect_ops(c, ops);
            }
        }
    }
}

fn expr_depth(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) | Expr::Const(_) | Expr::Param(_) => 0,
        Expr::Unary(_, c) => 1 + expr_depth(c),
        Expr::Binary(_, l, r) => 1 + expr_depth(l).max(expr_depth(r)),
        Expr::Ternary(_, a, b, c) => 1 + expr_depth(a).max(expr_depth(b)).max(expr_depth(c)),
        Expr::Nary(_, cs) => 1 + cs.iter().map(expr_depth).max().unwrap_or(0),
    }
}

#[test]
#[ignore]
fn bwdgen_quality_should_succeed_when_called() {
    let config = BwdGenConfig::default();
    let templates = RuleTemplates::new();
    let mut generator = BwdGenerator::new(42, config, templates);

    let mut compiled = 0usize;
    let mut valid_bench = 0usize;
    let mut trivial = 0usize; // < 1ns
    let mut too_high = 0usize; // > 1000ns
    let mut compile_fails: HashMap<String, usize> = HashMap::new();
    let mut bench_times: Vec<f64> = Vec::new();
    let mut all_ops: HashMap<OpKind, usize> = HashMap::new();
    let mut depths: Vec<usize> = Vec::new();

    for i in 0..BATCH_SIZE {
        let pair = generator.generate();
        let expr = &pair.optimized;

        let depth = expr_depth(expr);
        depths.push(depth);
        collect_ops(expr, &mut all_ops);

        // Print every expression for visual inspection
        eprintln!("[{i:3}] depth={depth} nodes={:3} | {expr}", expr.node_count());

        match benchmark_jit(expr) {
            Ok(b) => {
                compiled += 1;
                bench_times.push(b.ns);
                if b.ns < 1.0 {
                    trivial += 1;
                } else if b.ns > 1000.0 {
                    too_high += 1;
                } else {
                    valid_bench += 1;
                }
            }
            Err(BenchError::CompileFailed(msg)) => {
                *compile_fails.entry(msg.to_string()).or_default() += 1;
                eprintln!("[{i}] COMPILE FAIL: {msg}");
            }
            Err(BenchError::UnsupportedArch) => {
                *compile_fails.entry("unsupported_arch".to_string()).or_default() += 1;
                eprintln!("[{i}] UNSUPPORTED ARCH");
            }
            Err(BenchError::InvalidMeasurement(v)) => {
                compiled += 1; // it compiled, just bad measurement
                *compile_fails.entry(format!("invalid_measurement({v})")).or_default() += 1;
                eprintln!("[{i}] INVALID MEASUREMENT: {v}ns");
            }
        }
    }

    // Report
    let total_errors: usize = compile_fails.values().sum();
    eprintln!("\n=== BwdGenerator Quality Report ===");
    eprintln!("Total: {BATCH_SIZE}");
    eprintln!("Compiled (Ok): {compiled} | Errors: {total_errors} | bench_times.len: {}", bench_times.len());
    eprintln!("Valid bench (1-1000ns): {valid_bench} ({:.1}%)", valid_bench as f64 / BATCH_SIZE as f64 * 100.0);
    eprintln!("Trivial (<1ns): {trivial}");
    eprintln!("Too high (>1000ns): {too_high}");

    if !bench_times.is_empty() {
        bench_times.sort_by(|a, b| a.partial_cmp(b).expect("Expected value but got None/Err"));
        let median = bench_times[bench_times.len() / 2];
        let min = bench_times[0];
        let max = bench_times[bench_times.len() - 1];
        eprintln!("Bench range: {min:.1}ns - {max:.1}ns (median {median:.1}ns)");
        // Histogram by bucket
        let mut hist: HashMap<u64, usize> = HashMap::new();
        for &t in &bench_times {
            *hist.entry(t as u64).or_default() += 1;
        }
        let mut hist_sorted: Vec<_> = hist.iter().collect();
        hist_sorted.sort_by_key(|(k, _)| **k);
        eprintln!("Time histogram:");
        for (bucket, count) in &hist_sorted {
            eprintln!("  {bucket}ns: {count}");
        }
    }

    eprintln!("\nCompile/bench failures:");
    for (reason, count) in &compile_fails {
        eprintln!("  {reason}: {count}");
    }

    eprintln!("\nOp variety ({} distinct ops):", all_ops.len());
    let mut ops_sorted: Vec<_> = all_ops.iter().collect();
    ops_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (op, count) in &ops_sorted {
        eprintln!("  {op:?}: {count}");
    }

    eprintln!("\nDepth distribution:");
    let mut depth_hist: HashMap<usize, usize> = HashMap::new();
    for d in &depths {
        *depth_hist.entry(*d).or_default() += 1;
    }
    let mut depth_sorted: Vec<_> = depth_hist.iter().collect();
    depth_sorted.sort_by_key(|(d, _)| **d);
    for (d, c) in &depth_sorted {
        eprintln!("  depth {d}: {c}");
    }

    // Assertions for success criteria
    let compile_rate = compiled as f64 / BATCH_SIZE as f64;
    let valid_rate = valid_bench as f64 / BATCH_SIZE as f64;
    let op_variety = all_ops.len();

    eprintln!("\n=== Criteria Check ===");
    eprintln!("Compile rate: {:.1}% (need ≥80%)", compile_rate * 100.0);
    eprintln!("Valid bench rate: {:.1}% (need ≥70%)", valid_rate * 100.0);
    eprintln!("Op variety: {op_variety} (need ≥5)");

    assert!(compile_rate >= 0.80, "Compile rate {compile_rate:.0}% < 80%");
    assert!(valid_rate >= 0.70, "Valid bench rate {valid_rate:.0}% < 70%");
    assert!(op_variety >= 5, "Op variety {op_variety} < 5");
}
