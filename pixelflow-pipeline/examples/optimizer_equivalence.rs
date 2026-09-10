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

use pixelflow_ir::{ExprData, LatticeShape, Node, Term, encode};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::egraph::Optimizer;
use pixelflow_search::runtime::optimize_runtime_term;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The canonical fingerprint of an extracted term: `expr::encode`'s bytes.
/// That encoding is exactly the "reachable nodes, children before parents,
/// dense ordinals" pair this harness's doc describes, so it fingerprints the
/// term and its sharing and nothing else — no Debug rendering, no
/// construction garbage.
fn digest(root: Node<'_, ExprData>) -> u64 {
    let mut h = FNV_OFFSET;
    for b in encode(root) {
        h ^= u64::from(b);
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
        let Some((expr, env)) = named_shadertoy_kernel(name) else {
            panic!("shader_bench no longer provides kernel {name:?}");
        };
        let term = Term::new(expr.entry(), &env);
        let input_nodes = expr.len();
        let started = std::time::Instant::now();
        let optimized = optimize_runtime_term(term, LatticeShape::POINT);
        let elapsed = started.elapsed();
        let Some(optimized) = optimized else {
            // A kernel the runtime optimizer declines (a `Dwrt` it cannot
            // lower) is reported rather than skipped: a refactor that turns a
            // successful optimization into a declined one is exactly the kind
            // of change this harness exists to catch.
            println!("{name:<22} {input_nodes:>5} {:>5} {:>18}", "-", "DECLINED");
            continue;
        };
        let (out_expr, _out_env) = optimized.as_ref();
        let d = digest(out_expr.entry());
        all ^= d;

        // The same production configuration, run directly so the budget it
        // actually spent is visible. This is what calibrates a deterministic
        // `Budget::Applications` against what the presets currently allow.
        let mut optimizer = Optimizer::production();
        let mut eg = optimizer.egraph();
        let root_class = pixelflow_search::egraph::insert_term(
            term,
            &mut eg,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let stats = optimizer.run(&mut eg, root_class, expr.len()).stats;

        println!(
            "{name:<22} {input_nodes:>5} {:>5} {d:>18x} {:>8.1}ms {:>9} {:>7} {:>20}",
            out_expr.len(),
            elapsed.as_secs_f64() * 1e3,
            stats.applications,
            stats.classes,
            format!("{:?}", stats.stop),
        );
    }
    println!("\ncombined: {all:016x}");
}
