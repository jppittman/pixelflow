//! Extraction fingerprints for the ShaderToy bench kernels, so a refactor
//! that claims "no behavior change" can be checked rather than asserted.
//!
//! Prints one line per kernel: the input size, the extracted DAG's size, and
//! a structural digest of the extracted arena. Run it on two revisions and
//! diff the output — an unchanged digest is a stronger claim than an
//! unchanged cost, because equal arenas have equal cost under *every* model,
//! while equal costs can hide a different term.
//!
//! ```text
//! cargo run -p pixelflow-pipeline --example optimizer_equivalence
//! ```
//!
//! The digest covers the node vector and the root. The arena is append-only
//! and extraction emits children before parents, so for a fixed
//! configuration that pair is canonical: two runs that agree here produced
//! the same term with the same sharing.

use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::egraph::Optimizer;
use pixelflow_search::runtime::optimize_runtime_arena;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn digest(arena: &ExprArena, root: ExprId) -> u64 {
    let rendered = format!("{root:?}|{:?}", arena.nodes().collect::<Vec<_>>());
    let mut h = FNV_OFFSET;
    for b in rendered.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn main() {
    println!(
        "{:<22} {:>5} {:>5} {:>18} {:>10} {:>9} {:>7} {:>20}",
        "kernel", "in", "out", "digest", "optimize", "applied", "classes", "stop"
    );
    let mut all = FNV_OFFSET;
    for name in SHADERTOY_KERNEL_NAMES {
        let Some((arena, root)) = named_shadertoy_kernel(name) else {
            panic!("shader_bench no longer provides kernel {name:?}");
        };
        let input_nodes = arena.len();
        let started = std::time::Instant::now();
        let optimized = optimize_runtime_arena(&arena, root, LatticeShape::POINT);
        let elapsed = started.elapsed();
        let Some(optimized) = optimized else {
            // A kernel the runtime optimizer declines (a `Dwrt` it cannot
            // lower) is reported rather than skipped: a refactor that turns a
            // successful optimization into a declined one is exactly the kind
            // of change this harness exists to catch.
            println!("{name:<22} {input_nodes:>5} {:>5} {:>18}", "-", "DECLINED");
            continue;
        };
        let (out_arena, out_root) = optimized.as_ref();
        let d = digest(out_arena, *out_root);
        all ^= d;

        // The same production configuration, run directly so the budget it
        // actually spent is visible. This is what calibrates a deterministic
        // `Budget::Applications` against what the presets currently allow.
        let mut optimizer = Optimizer::production();
        let mut eg = optimizer.egraph();
        let root_class = pixelflow_search::egraph::insert(
            &arena,
            root,
            &mut eg,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let stats = optimizer.run(&mut eg, root_class, arena.len()).stats;

        println!(
            "{name:<22} {input_nodes:>5} {:>5} {d:>18x} {:>8.1}ms {:>9} {:>7} {:>20}",
            out_arena.len(),
            elapsed.as_secs_f64() * 1e3,
            stats.applications,
            stats.classes,
            format!("{:?}", stats.stop),
        );
    }
    println!("\ncombined: {all:016x}");
}
