//! CPU + memory profile of e-graph saturation/extraction in isolation.
//!
//! Scratch measurement tool for the 2026-09-08 e-graph performance profile
//! (docs/plans/2026-09-08-egraph-performance-profile.md). Builds synthetic
//! arenas of increasing node count (same SDF-shaped op mix as
//! `bench_jit_compile_cost`'s `build_kernel_arena`, so results are
//! comparable to the existing G0 numbers) and runs each through
//! `pixelflow_search::runtime::optimize_runtime_arena` — saturate + extract,
//! the same call `Lattice::bake` makes — reporting wall time and bytes
//! allocated per call, plus an optional pprof CPU flamegraph.
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features "training profiling" \
//!     --bin egraph_profile
//! ```

use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::{LatticeShape, OpKind};
use pixelflow_pipeline::alloc_probe::{self, CountingAlloc};

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Same shape as `bench_jit_compile_cost::build_kernel_arena`: an SDF-style
/// core plus a repeating 8-op mix (select, sqrt, mul, add, mul_add, max,
/// sub, square) until `target_nodes` is reached. `salt` keeps two calls
/// canonically distinct so `optimize_runtime_arena`'s cache never hits.
fn build_kernel_arena(target_nodes: usize, salt: f32) -> (ExprArena, ExprId) {
    assert!(target_nodes >= 8);
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let c = arena.push_const(salt);
    let dx = arena.push_binary(OpKind::Sub, x, c);
    let dy = arena.push_binary(OpKind::Sub, y, c);
    let dx2 = arena.push_binary(OpKind::Mul, dx, dx);
    let mut cur = arena.push_ternary(OpKind::MulAdd, dy, dy, dx2);

    let mut step = 0usize;
    while arena.len() < target_nodes {
        let remaining = target_nodes - arena.len();
        cur = match step % 8 {
            0 if remaining >= 2 => {
                let inside = arena.push_binary(OpKind::Lt, cur, c);
                arena.push_ternary(OpKind::Select, inside, dx2, cur)
            }
            1 => arena.push_unary(OpKind::Sqrt, cur),
            2 => arena.push_binary(OpKind::Mul, cur, dx),
            3 => arena.push_binary(OpKind::Add, cur, dy),
            4 => arena.push_ternary(OpKind::MulAdd, cur, c, dx2),
            5 => arena.push_binary(OpKind::Max, cur, dx),
            6 => arena.push_binary(OpKind::Sub, cur, c),
            _ => arena.push_binary(OpKind::Mul, cur, cur),
        };
        step += 1;
    }
    (arena, cur)
}

fn main() {
    let sizes: &[usize] = &[16, 64, 256, 1024, 4096, 16_384];
    println!("=== e-graph saturate+extract profile ===");
    println!(
        "{:>8}  {:>12}  {:>14}  {:>14}  {:>10}",
        "nodes", "wall ns", "bytes/call", "peak bytes", "ns/node"
    );

    let mut salt_counter = 0u32;
    for &size in sizes {
        // A handful of distinct-salt calls per size: median wall time,
        // last call's allocation counters (steady-state, past any
        // one-time lazy-static init).
        const REPS: usize = 5;
        let mut times = Vec::with_capacity(REPS);
        let (mut bytes_per_call, mut peak_bytes) = (0usize, 0usize);
        for _ in 0..REPS {
            salt_counter += 1;
            let salt = 0.25 + (salt_counter as f32) * (1.0 / 65536.0);
            let (arena, root) = build_kernel_arena(size, salt);

            alloc_probe::reset();
            let start = std::time::Instant::now();
            let out = pixelflow_search::runtime::optimize_runtime_arena(
                &arena,
                root,
                LatticeShape::POINT,
            );
            let elapsed = start.elapsed();
            std::hint::black_box(&out);

            times.push(elapsed.as_nanos() as f64);
            bytes_per_call = alloc_probe::allocated_bytes();
            peak_bytes = alloc_probe::peak_bytes();
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[REPS / 2];
        println!(
            "{:>8}  {:>12.0}  {:>14}  {:>14}  {:>10.1}",
            size,
            median,
            bytes_per_call,
            peak_bytes,
            median / size as f64
        );
    }

    #[cfg(feature = "profiling")]
    {
        println!("\n=== CPU flamegraph: fixed 4096-node arena, 2000 distinct calls ===");
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(997)
            .blocklist(&["libc", "libgcc", "pthread"])
            .build()
            .expect("pprof");

        let n = 2_000usize;
        let start = std::time::Instant::now();
        for _ in 0..n {
            salt_counter += 1;
            let salt = 0.25 + (salt_counter as f32) * (1.0 / 65536.0);
            let (arena, root) = build_kernel_arena(4096, salt);
            let out = pixelflow_search::runtime::optimize_runtime_arena(
                &arena,
                root,
                LatticeShape::POINT,
            );
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        println!(
            "{n} calls in {:.1}ms ({:.1}us/call)",
            elapsed.as_millis(),
            elapsed.as_micros() as f64 / n as f64
        );

        if let Ok(report) = guard.report().build() {
            let path = "/tmp/egraph_flamegraph.svg";
            if let Ok(file) = std::fs::File::create(path)
                && report.flamegraph(file).is_ok()
            {
                println!("[PPROF] flamegraph written to {path}");
            }
            let text_path = "/tmp/egraph_profile.txt";
            if let Ok(mut file) = std::fs::File::create(text_path) {
                use std::io::Write;
                let _ = writeln!(file, "{report:?}");
                println!("[PPROF] text report written to {text_path}");
            }
        }
    }
}
