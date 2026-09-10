//! JIT compilation benchmark for profiling.
//! samply record cargo run --release -p pixelflow-ir --example jit_timing

#[cfg(target_arch = "aarch64")]
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData};
#[cfg(target_arch = "aarch64")]
use pixelflow_ir::kind::OpKind;
#[cfg(target_arch = "aarch64")]
use pixelflow_ir::{Rooted, Term};

#[cfg(target_arch = "aarch64")]
fn main() {
    use pixelflow_codegen::emit::compile;

    for size in [10, 30, 50, 100, 150, 200] {
        let built = build_expr(size);
        let term = Term::new(built.0.entry(), &built.1);
        let actual = built.0.len();
        for _ in 0..100 {
            compile(term).unwrap();
        }
        let n = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(compile(term).unwrap());
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
fn build_expr(target_nodes: usize) -> (Rooted<ExprData>, Environment) {
    let mut arena = ExprBuilder::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let mut acc = arena.push_binary(OpKind::Mul, x, y);
    let c = arena.push_const(0.5);
    acc = arena.push_binary(OpKind::Add, acc, c);
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
    while arena.len() < target_nodes {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let op = ops[(seed >> 33) as usize % ops.len()];
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let var_idx = (seed >> 33) as usize % vars.len();
        match op.arity() {
            1 => {
                let inner = if op == OpKind::Sqrt {
                    arena.push_unary(OpKind::Abs, acc)
                } else {
                    acc
                };
                acc = arena.push_unary(op, inner);
            }
            _ => {
                acc = arena.push_binary(op, acc, vars[var_idx]);
            }
        }
    }
    arena.finish(&[acc])
}
