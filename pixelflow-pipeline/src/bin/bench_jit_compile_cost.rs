//! JIT compile-cost benchmark (kernel-unification Phase 0, gate G0).
//!
//! Measures how long `compile_arena_dag` takes for expression arenas of
//! ~8, 32, 128, and 512 nodes, distinguishing:
//!
//! - **fresh**: every compile mmaps a new executable region and munmaps it on
//!   drop (`ExecutableCode` lifecycle — what a naive per-kernel JIT pays).
//! - **reused**: compiles into a persistent `CompileWorkspace` code buffer,
//!   paying `pthread_jit_write_protect_np` toggles + icache invalidation
//!   instead of mmap/munmap (the amortizable steady state).
//!
//! Gate G0: if the amortized (reused) cost exceeds ~1µs/kernel, per-leaf JIT
//! is formally dead.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin bench_jit_compile_cost
//! ```
//!
//! Results are recorded in `docs/results/2026-07-20-jit-compile-cost.md`.

use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_pipeline::jit_bench::benchmark_compile_fresh;
#[cfg(target_arch = "aarch64")]
use pixelflow_pipeline::jit_bench::benchmark_compile_reused;

/// Arena sizes to measure (exact node counts, all reachable from the root).
const ARENA_SIZES: &[usize] = &[8, 32, 128, 512];

/// G0 threshold: amortized compile cost above this kills per-leaf JIT.
const G0_THRESHOLD_NS: f64 = 1_000.0;

/// Build a deterministic arena of exactly `target_nodes` nodes with a
/// font/shader-like op mix: arithmetic, `sqrt`, `mul_add`, `min`/`max`, and a
/// comparison-guarded `select`, seeded from an SDF-style distance core.
///
/// Every appended op consumes the previous root as an operand, so all
/// `target_nodes` nodes are reachable from the returned root. Ops stay within
/// the directly-emittable set (no transcendentals, no gather/reduce) so the
/// fresh path's lowering passes are identity fast-paths and the
/// `CompileWorkspace` path — which skips lowering — compiles identical work.
fn build_kernel_arena(target_nodes: usize) -> (ExprArena, ExprId) {
    // The SDF seed below is 7 nodes; need at least one more op for a root.
    assert!(
        target_nodes >= 8,
        "target_nodes must be >= 8, got {}",
        target_nodes
    );

    let mut arena = ExprArena::new();
    // Circle-SDF-style core: (x-0.5)^2 + (y-0.5)^2 via mul_add. 7 nodes.
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let half = arena.push_const(0.5);
    let dx = arena.push_binary(OpKind::Sub, x, half);
    let dy = arena.push_binary(OpKind::Sub, y, half);
    let dx2 = arena.push_binary(OpKind::Mul, dx, dx);
    let mut cur = arena.push_ternary(OpKind::MulAdd, dy, dy, dx2);

    let mut step = 0usize;
    while arena.len() < target_nodes {
        let remaining = target_nodes - arena.len();
        cur = match step % 8 {
            // Select costs 2 nodes (its Lt guard + the Select itself);
            // when only 1 node of budget remains, fall through to `_`.
            0 if remaining >= 2 => {
                let inside = arena.push_binary(OpKind::Lt, cur, half);
                arena.push_ternary(OpKind::Select, inside, dx2, cur)
            }
            1 => arena.push_unary(OpKind::Sqrt, cur),
            2 => arena.push_binary(OpKind::Mul, cur, dx),
            3 => arena.push_binary(OpKind::Add, cur, dy),
            4 => arena.push_ternary(OpKind::MulAdd, cur, half, dx2),
            5 => arena.push_binary(OpKind::Max, cur, dx),
            6 => arena.push_binary(OpKind::Sub, cur, half),
            _ => arena.push_binary(OpKind::Mul, cur, cur),
        };
        step += 1;
    }

    assert_eq!(
        arena.len(),
        target_nodes,
        "builder produced {} nodes, wanted {}",
        arena.len(),
        target_nodes
    );
    (arena, cur)
}

fn main() {
    println!("=== JIT compile cost (gate G0: amortized <= ~1us/kernel) ===");
    println!(
        "arch: {}, 101 timed compiles per cell, median reported\n",
        std::env::consts::ARCH
    );

    #[cfg(target_arch = "aarch64")]
    println!(
        "{:>6}  {:>10}  {:>12}  {:>12}  {:>10}",
        "nodes", "code B", "fresh ns", "reused ns", "ns/node"
    );
    #[cfg(not(target_arch = "aarch64"))]
    println!(
        "{:>6}  {:>10}  {:>12}  {:>10}",
        "nodes", "code B", "fresh ns", "ns/node"
    );

    let mut worst_amortized_per_node = 0.0f64;
    for &size in ARENA_SIZES {
        let (arena, root) = build_kernel_arena(size);

        let fresh = benchmark_compile_fresh(&arena, root)
            .unwrap_or_else(|e| panic!("fresh compile bench failed at {} nodes: {}", size, e));

        #[cfg(target_arch = "aarch64")]
        {
            let reused = benchmark_compile_reused(&arena, root)
                .unwrap_or_else(|e| panic!("reused compile bench failed at {} nodes: {}", size, e));
            let per_node = reused / size as f64;
            worst_amortized_per_node = worst_amortized_per_node.max(per_node);
            println!(
                "{:>6}  {:>10}  {:>12.0}  {:>12.0}  {:>10.1}",
                size, fresh.code_bytes, fresh.ns, reused, per_node
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let per_node = fresh.ns / size as f64;
            worst_amortized_per_node = worst_amortized_per_node.max(per_node);
            println!(
                "{:>6}  {:>10}  {:>12.0}  {:>10.1}",
                size, fresh.code_bytes, fresh.ns, per_node
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!(
        "\nNOTE: CompileWorkspace (reused code buffer) is aarch64-only; \
         only the fresh mmap-per-compile path was measured on this arch."
    );

    // G0 is stated per-kernel; the smallest arena is the per-leaf shape, so
    // judge the gate on the smallest size's amortized cost.
    let (arena, root) = build_kernel_arena(ARENA_SIZES[0]);
    #[cfg(target_arch = "aarch64")]
    let leaf_ns = benchmark_compile_reused(&arena, root)
        .unwrap_or_else(|e| panic!("G0 leaf measurement failed: {}", e));
    #[cfg(not(target_arch = "aarch64"))]
    let leaf_ns = benchmark_compile_fresh(&arena, root)
        .unwrap_or_else(|e| panic!("G0 leaf measurement failed: {}", e))
        .ns;

    println!(
        "\nG0: amortized cost for a {}-node leaf kernel = {:.0}ns ({}, threshold {:.0}ns) => {}",
        ARENA_SIZES[0],
        leaf_ns,
        if cfg!(target_arch = "aarch64") {
            "reused buffer"
        } else {
            "fresh mmap path"
        },
        G0_THRESHOLD_NS,
        if leaf_ns > G0_THRESHOLD_NS {
            "per-leaf JIT FAILS G0"
        } else {
            "per-leaf JIT passes G0"
        }
    );
}
