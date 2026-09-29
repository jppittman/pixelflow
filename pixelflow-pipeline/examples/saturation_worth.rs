//! What equality saturation is worth to the JIT, measured rather than assumed.
//!
//! Two arms, same backend, same kernels, same harness — the only difference is
//! which [`Optimize`] value the arena goes through before it is compiled:
//!
//! - `identity` — [`Identity`]. The arena as built. This arm did not exist
//!   before `Optimize` did: every JIT entry point saturated unconditionally,
//!   so "the JIT is slower than LLVM" could not be split into "the JIT emits
//!   worse code" and "the e-graph does less for the JIT's input".
//! - `saturated` — the production runtime pipeline, `LowerDwrt` then
//!   `ExpandReduce` then `Saturate`.
//!
//! ```text
//! cargo run --release -p pixelflow-pipeline --example saturation_worth
//! ```
//!
//! Both arms are timed by the same `BenchSession` (median of samples, sentinel
//! drift normalization), so the comparison between them is meaningful even
//! though the absolute numbers are machine-specific.

use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::optimize::{Identity, Optimize};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{LatticeShape, pipeline};
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::Saturate;

/// Run one optimizer arm, returning the term to compile.
///
/// `Unchanged`/`Declined` both mean "compile what you already had", which is
/// exactly the identity arm's whole behavior — the same code path serves both.
fn arm<O: Optimize>(mut opt: O, arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    opt.optimize(arena, root)
        .into_changed()
        .unwrap_or_else(|| (arena.clone(), root))
}

fn main() {
    println!(
        "{:<22} {:>5} {:>5} {:>10} {:>10} {:>9}",
        "kernel", "raw", "opt", "identity", "saturated", "speedup"
    );

    let mut session = BenchSession::new();
    let mut total_identity = 0.0f64;
    let mut total_saturated = 0.0f64;

    for name in SHADERTOY_KERNEL_NAMES {
        let Some((arena, root)) = named_shadertoy_kernel(name) else {
            continue;
        };

        let (raw_arena, raw_root) = arm(Identity, &arena, root);
        let (opt_arena, opt_root) = arm(
            pipeline![
                LowerDwrt,
                ExpandReduce,
                Saturate::runtime(LatticeShape::POINT)
            ],
            &arena,
            root,
        );

        let identity = session.benchmark_arena(&raw_arena, raw_root, BenchMode::Latency);
        let saturated = session.benchmark_arena(&opt_arena, opt_root, BenchMode::Latency);

        match (identity, saturated) {
            (Ok(i), Ok(s)) => {
                total_identity += i.ns;
                total_saturated += s.ns;
                println!(
                    "{:<22} {:>5} {:>5} {:>9.3}ns {:>9.3}ns {:>8.2}x",
                    name,
                    raw_arena.len(),
                    opt_arena.len(),
                    i.ns,
                    s.ns,
                    i.ns / s.ns
                );
            }
            (i, s) => {
                println!(
                    "{:<22} {:>5} {:>5}  identity={:?} saturated={:?}",
                    name,
                    raw_arena.len(),
                    opt_arena.len(),
                    i.err(),
                    s.err()
                );
            }
        }
    }

    if total_saturated > 0.0 {
        println!();
        println!(
            "total: identity {:.3}ns, saturated {:.3}ns — saturation is worth {:.2}x on the JIT",
            total_identity,
            total_saturated,
            total_identity / total_saturated
        );
    }
}
