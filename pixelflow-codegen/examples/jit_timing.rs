//! JIT compilation benchmark for profiling.
//! samply record cargo run --release -p pixelflow-ir --example jit_timing

#[cfg(target_arch = "aarch64")]
use pixelflow_ir::{ExprBuilder, ExprGraph, OpKind};

#[cfg(target_arch = "aarch64")]
fn main() {
    for size in [10, 30, 50, 100, 150, 200] {
        let graph = build_expr(size);
        let actual = graph.dag().len();
        for _ in 0..100 {
            pixelflow_codegen::emit::compile_dag(graph.root(), graph.environment()).unwrap();
        }
        let n = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(
                pixelflow_codegen::emit::compile_dag(graph.root(), graph.environment()).unwrap(),
            );
        }
        let us = start.elapsed().as_micros() as f64 / n as f64;
        eprintln!(
            "nodes={actual:3}  compile={us:7.1}µs ({:.1}µs/node)",
            us / actual as f64
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("aarch64 only");
}

#[cfg(target_arch = "aarch64")]
fn build_expr(target_nodes: usize) -> ExprGraph {
    let mut builder = ExprBuilder::new();
    let x = builder.var(0);
    let y = builder.var(1);
    let mut acc = builder.binary(OpKind::Mul, x, y);
    let c = builder.constant(0.5);
    acc = builder.binary(OpKind::Add, acc, c);
    let ops = [
        OpKind::Add,
        OpKind::Mul,
        OpKind::Sub,
        OpKind::Sin,
        OpKind::Cos,
        OpKind::Sqrt,
    ];
    let vars = [x, y];
    let mut seed = 12345u64;
    let mut built = 5usize;
    while built < target_nodes {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let op = ops[(seed >> 33) as usize % ops.len()];
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let var_idx = (seed >> 33) as usize % vars.len();
        match op.arity() {
            1 => {
                let inner = if op == OpKind::Sqrt {
                    builder.unary(OpKind::Abs, acc)
                } else {
                    acc
                };
                acc = builder.unary(op, inner);
                built += 1;
            }
            _ => {
                acc = builder.binary(op, acc, vars[var_idx]);
                built += 1;
            }
        }
    }
    builder.finish_one(acc)
}
