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

use pixelflow_ir::optimize::{Identity, Optimize};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{Environment, ExprBuilder, ExprData, LatticeShape, Rooted, Term, pipeline};
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::Saturate;

/// Run one optimizer arm, returning the term to compile.
///
/// `Unchanged`/`Declined` both mean "compile what you already had", which is
/// exactly the identity arm's whole behavior — the same code path serves both.
fn arm<O: Optimize>(mut opt: O, term: Term<'_>) -> (Rooted<ExprData>, Environment) {
    opt.optimize(term).into_changed().unwrap_or_else(|| {
        let mut b = ExprBuilder::new();
        let root = b.splice(term);
        b.finish(&[root])
    })
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
        let Some((expr, env)) = named_shadertoy_kernel(name) else {
            continue;
        };
        let term = Term::new(expr.entry(), &env);

        let (raw, raw_env) = arm(Identity, term);
        let (opt, opt_env) = arm(
            pipeline![
                LowerDwrt,
                ExpandReduce,
                Saturate::runtime(LatticeShape::POINT)
            ],
            term,
        );

        let identity = session.benchmark_term(Term::new(raw.entry(), &raw_env), BenchMode::Latency);
        let saturated =
            session.benchmark_term(Term::new(opt.entry(), &opt_env), BenchMode::Latency);

        match (identity, saturated) {
            (Ok(i), Ok(s)) => {
                total_identity += i.ns;
                total_saturated += s.ns;
                println!(
                    "{:<22} {:>5} {:>5} {:>9.3}ns {:>9.3}ns {:>8.2}x",
                    name,
                    raw.len(),
                    opt.len(),
                    i.ns,
                    s.ns,
                    i.ns / s.ns
                );
            }
            (i, s) => {
                println!(
                    "{:<22} {:>5} {:>5}  identity={:?} saturated={:?}",
                    name,
                    raw.len(),
                    opt.len(),
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
