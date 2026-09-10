//! JIT compile-cost benchmark (kernel-unification Phase 0, gate G0).
//!
//! Reports two series per arena size (~8, 32, 128, 512 nodes):
//!
//! - **full miss** (headline, `miss ns`): one `jit_cache::compile_cached`
//!   call whose canonical key has never been seen — the production cache-miss
//!   path. On a miss that entry point runs canonical-key construction, the
//!   cache lock, `optimize_runtime_arena` (e-graph saturation + extraction,
//!   itself keyed by the same structure and therefore also a miss for a
//!   distinct kernel), `compile` (mmap + emit + mprotect), and
//!   cache insertion. This is what one *distinct* kernel actually costs at
//!   runtime, and it is the number the G0 verdict reads. A kernel seen before
//!   costs a hash lookup and an `Arc` clone instead — hits are not what this
//!   benchmark measures.
//! - **emit only** (attribution, `emit ns`): `compile` called
//!   directly on one fixed arena — codegen plus the executable-region
//!   lifecycle, with no optimizer, no key, no lock. The gap between the two
//!   series is the optimizer + cache bookkeeping share of a miss.
//!
//! Every timed full-miss call uses a structurally distinct arena: the
//! builder's const leaf takes a unique value per kernel, and that leaf is an
//! operand of var-dependent ops (`X - c`, `mul_add(cur, c, ..)`), so the
//! optimizer cannot fold it away into a shared form. The cache keys on the
//! arena as handed in (pre-optimization), and `benchmark_compile_cached_miss`
//! asserts the cache entry-count delta afterwards, so an accidental hit fails
//! loudly rather than silently deflating the numbers. Both global caches
//! retain every entry (unbounded by design); for this one-shot bench process
//! that retention is fine.
//!
//! Gate G0: if a distinct leaf-sized kernel costs more than ~1µs, per-leaf
//! JIT is formally dead.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin bench_jit_compile_cost
//! ```
//!

use pixelflow_ir::{Environment, ExprBuilder, ExprData, OpKind, Rooted, Term};
use pixelflow_pipeline::jit_bench::{
    COMPILE_MISS_KERNELS, benchmark_compile_cached_miss, benchmark_compile_fresh,
};

/// Arena sizes to measure (exact node counts, all reachable from the root).
const ARENA_SIZES: &[usize] = &[8, 32, 128, 512];

/// G0 threshold: per-kernel compile cost above this kills per-leaf JIT.
const G0_THRESHOLD_NS: f64 = 1_000.0;

/// Const-leaf value for the fixed emit-only arena (the historical builder
/// constant, kept so that series stays comparable across runs).
const EMIT_SERIES_SALT: f32 = 0.5;

/// Build a deterministic arena of exactly `target_nodes` nodes with a
/// font/shader-like op mix: arithmetic, `sqrt`, `mul_add`, `min`/`max`, and a
/// comparison-guarded `select`, seeded from an SDF-style distance core.
///
/// `salt` is the value of the arena's single const leaf. It participates as
/// an operand of var-dependent ops (`X - salt`, `mul_add(cur, salt, ..)`), so
/// it survives constant folding — two arenas built with different salts are
/// canonically distinct all the way through the optimizer and the JIT cache.
/// That is what lets a salted stream guarantee genuine cache misses.
///
/// Every appended op consumes the previous root as an operand, so all
/// `target_nodes` nodes are reachable from the returned root. Ops stay within
/// the directly-emittable set (no transcendentals, no gather/reduce), so the
/// lowering passes are identity fast-paths.
fn build_kernel_term(target_nodes: usize, salt: f32) -> (Rooted<ExprData>, Environment) {
    // The SDF seed below is 7 nodes; need at least one more op for a root.
    assert!(
        target_nodes >= 8,
        "target_nodes must be >= 8, got {}",
        target_nodes
    );

    let mut arena = ExprBuilder::new();
    // Circle-SDF-style core: (x-salt)^2 + (y-salt)^2 via mul_add. 7 nodes.
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
            // Select costs 2 nodes (its Lt guard + the Select itself);
            // when only 1 node of budget remains, fall through to `_`.
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

    assert_eq!(
        arena.len(),
        target_nodes,
        "builder produced {} nodes, wanted {}",
        arena.len(),
        target_nodes
    );
    arena.finish(&[cur])
}

/// A stream of `COMPILE_MISS_KERNELS` canonically distinct kernels of
/// `target_nodes` nodes each. `salt_counter` is global across the whole run
/// so no two kernels anywhere in the process share a salt (and the salt base
/// avoids `EMIT_SERIES_SALT`), keeping every `compile_cached` call a miss.
fn distinct_kernel_stream(
    target_nodes: usize,
    salt_counter: &mut u32,
) -> Vec<(Rooted<ExprData>, Environment)> {
    (0..COMPILE_MISS_KERNELS)
        .map(|_| {
            *salt_counter += 1;
            // 0.25 + n·2⁻¹⁶: exactly representable f32 steps, distinct for
            // every n this run can reach, never equal to EMIT_SERIES_SALT.
            let salt = 0.25 + (*salt_counter as f32) * (1.0 / 65536.0);
            build_kernel_term(target_nodes, salt)
        })
        .collect()
}

fn main() {
    println!("=== JIT compile cost (gate G0: <= ~1us per distinct kernel) ===");
    println!(
        "arch: {}, {} distinct kernels per cell (warmup + timed), median of timed reported",
        std::env::consts::ARCH,
        COMPILE_MISS_KERNELS
    );
    println!(
        "miss ns = full jit_cache::compile_cached miss (key + lock + optimize + emit + insert)"
    );
    println!("emit ns = compile only (attribution)\n");
    println!(
        "{:>6}  {:>10}  {:>12}  {:>12}  {:>10}",
        "nodes", "code B", "miss ns", "emit ns", "miss/node"
    );

    let mut salt_counter = 0u32;
    let mut miss_by_size = Vec::with_capacity(ARENA_SIZES.len());
    for &size in ARENA_SIZES {
        let kernels = distinct_kernel_stream(size, &mut salt_counter);
        let miss_ns = benchmark_compile_cached_miss(kernels)
            .unwrap_or_else(|e| panic!("full-miss bench failed at {} nodes: {}", size, e));
        miss_by_size.push(miss_ns);

        let (expr, env) = build_kernel_term(size, EMIT_SERIES_SALT);
        let fresh = benchmark_compile_fresh(Term::new(expr.entry(), &env))
            .unwrap_or_else(|e| panic!("emit-only bench failed at {} nodes: {}", size, e));

        println!(
            "{:>6}  {:>10}  {:>12.0}  {:>12.0}  {:>10.1}",
            size,
            fresh.code_bytes,
            miss_ns,
            fresh.ns,
            miss_ns / size as f64
        );
    }

    // G0 is stated per-kernel, and the smallest arena is the per-leaf shape.
    // The verdict reads the full production miss path — a per-leaf JIT would
    // pay exactly that per distinct leaf, not just the emit step.
    let leaf_ns = miss_by_size[0];
    println!(
        "\nG0: one distinct {}-node leaf kernel through compile_cached = {:.0}ns \
         (threshold {:.0}ns) => {}",
        ARENA_SIZES[0],
        leaf_ns,
        G0_THRESHOLD_NS,
        if leaf_ns > G0_THRESHOLD_NS {
            "per-leaf JIT FAILS G0"
        } else {
            "per-leaf JIT passes G0"
        }
    );
}
