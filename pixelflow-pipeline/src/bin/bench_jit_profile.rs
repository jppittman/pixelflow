//! Profile JIT compilation with pprof.
//! cargo run --release -p pixelflow-pipeline --features "training profiling" --bin bench_jit_profile

use pixelflow_codegen::emit::compile;
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{Environment, ExprBuilder, ExprData, Rooted, Term};

fn main() {
    let (expr, env) = build_expr(200);
    let term = Term::new(expr.entry(), &env);
    eprintln!("Graph: {} nodes", expr.len());

    // Warmup
    for _ in 0..100 {
        compile(term).unwrap();
    }

    #[cfg(feature = "profiling")]
    let guard = {
        let g = pprof::ProfilerGuardBuilder::default()
            .frequency(997)
            .blocklist(&[
                "libc",
                "libgcc",
                "pthread",
                "vDSP",
                "libsystem_kernel",
                "libsystem_platform",
            ])
            .build()
            .expect("pprof");
        eprintln!("[PPROF] Started");
        g
    };

    let n = 10_000;
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(compile(term).unwrap());
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{n} compiles in {:.1}ms ({:.1}µs/compile)",
        elapsed.as_millis(),
        elapsed.as_micros() as f64 / n as f64
    );

    #[cfg(feature = "profiling")]
    {
        if let Ok(report) = guard.report().build() {
            let path = "/tmp/jit_flamegraph.svg";
            let file = std::fs::File::create(path).unwrap();
            report.flamegraph(file).unwrap();
            eprintln!("[PPROF] Flamegraph: {path}");
        }
    }
}

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
