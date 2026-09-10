//! Measure per-op costs through the JIT bench harness, to audit
//! `pixelflow_search::egraph::cost::latency_prior_cycles()` against reality.
//!
//! Two protocols, both on the exact code the JIT emits (so transcendentals are
//! measured in their `expand_transcendentals` lowered form — the form the cost
//! table is actually pricing):
//!
//! 1. **Latency chains** (primary): a kernel with K serial applications of the
//!    op, measured under `BenchMode::Latency` at K=8 and K=32. The per-stage
//!    slope `(ns(32) - ns(8)) / 24` cancels call overhead and chain prelude
//!    exactly. Unary ops carry one Mul per stage to steer values back into the
//!    op's domain; the measured Mul-chain slope is subtracted for those.
//! 2. **Throughput sums** (secondary): a kernel summing K=16 independent
//!    applications, measured under `BenchMode::Throughput`, minus an identical
//!    kernel with the op removed.
//!
//! Output: per-op latency ns, latency "table cycles" (normalized so Add = 4,
//! the existing table's unit), throughput marginal ns, and the current table
//! entry for comparison.
//!
//! Run: `cargo run --release -p pixelflow-pipeline --example measure_latency_prior`

use pixelflow_ir::{Environment, ExprBuilder, ExprData, ExprRef, OpKind, Rooted, Term};
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession};

const K_SHORT: usize = 8;
const K_LONG: usize = 32;
const K_THROUGHPUT: usize = 16;

/// How the op consumes the chain accumulator, and which constants keep the
/// chain inside the op's domain (finite, non-NaN) forever.
#[derive(Clone, Copy)]
enum Stage {
    /// acc = op(mul(acc, c_i)); Mul-slope is subtracted afterwards.
    UnaryScaled(f32, f32),
    /// acc = op(acc, c_i) — const on the right.
    BinaryConstRight(f32, f32),
    /// acc = op(c_i, acc) — const on the left (Div: k/acc oscillates stably).
    BinaryConstLeft(f32, f32),
    /// acc = mul_add(acc, a, b) — fixpoint b/(1-a).
    FusedTernary(f32, f32),
    /// acc = select(ge(acc, 0), mul(acc, c0), mul(acc, c1)); Mul-slope is
    /// subtracted (compare runs in parallel with the Mul on the chain path).
    CondBlend(f32, f32),
}

struct OpSpec {
    op: OpKind,
    stage: Stage,
}

fn specs() -> Vec<OpSpec> {
    use OpKind::*;
    use Stage::*;
    vec![
        // ALU block (also the Add anchor and the Mul stage-correction).
        OpSpec {
            op: Add,
            stage: BinaryConstRight(0.5, -0.5),
        },
        OpSpec {
            op: Sub,
            stage: BinaryConstRight(0.5, -0.5),
        },
        OpSpec {
            op: Mul,
            stage: BinaryConstRight(1.25, 0.8),
        },
        OpSpec {
            op: Div,
            stage: BinaryConstLeft(1.3, 0.7),
        },
        OpSpec {
            op: Neg,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Abs,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Min,
            stage: BinaryConstRight(0.9, 1.1),
        },
        OpSpec {
            op: Max,
            stage: BinaryConstRight(1.1, 0.9),
        },
        OpSpec {
            op: MulAdd,
            stage: FusedTernary(0.5, 0.7),
        },
        OpSpec {
            op: Floor,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Ceil,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Round,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Select,
            stage: CondBlend(0.8, 1.25),
        },
        // Hardware/NR unary math.
        OpSpec {
            op: Sqrt,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Rsqrt,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Recip,
            stage: UnaryScaled(1.25, 0.8),
        },
        // Transcendentals (lowered to polynomial/bit-manip kernels by
        // expand_transcendentals — that lowered cost is what we measure).
        OpSpec {
            op: Sin,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Cos,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Tan,
            stage: UnaryScaled(0.45, 0.45),
        },
        OpSpec {
            op: Asin,
            stage: UnaryScaled(0.5, 0.5),
        },
        OpSpec {
            op: Acos,
            stage: UnaryScaled(0.3, 0.3),
        },
        OpSpec {
            op: Atan,
            stage: UnaryScaled(1.25, 0.8),
        },
        OpSpec {
            op: Atan2,
            stage: BinaryConstRight(1.3, 0.7),
        },
        OpSpec {
            op: Exp,
            stage: UnaryScaled(-0.5, -0.5),
        },
        OpSpec {
            op: Exp2,
            stage: UnaryScaled(-0.5, -0.5),
        },
        OpSpec {
            op: Ln,
            stage: UnaryScaled(3.0, 3.0),
        },
        OpSpec {
            op: Log2,
            stage: UnaryScaled(4.0, 4.0),
        },
        OpSpec {
            op: Log10,
            stage: UnaryScaled(20.0, 20.0),
        },
        OpSpec {
            op: Pow,
            stage: BinaryConstRight(0.9, 0.9),
        },
    ]
}

/// Build the K-stage latency chain kernel for `spec`.
///
/// Prelude `acc0 = abs(x) + 1` keeps the chain seed strictly positive no
/// matter what the latency-mode feedback loop feeds back in.
fn chain_term(spec: &OpSpec, k: usize) -> (Rooted<ExprData>, Environment) {
    let mut arena = ExprBuilder::new();
    let x = arena.push_var(0);
    let ax = arena.push_unary(OpKind::Abs, x);
    let one = arena.push_const(1.0);
    let mut acc = arena.push_binary(OpKind::Add, ax, one);
    for i in 0..k {
        acc = push_stage(&mut arena, spec, acc, i);
    }
    arena.finish(&[acc])
}

fn push_stage(arena: &mut ExprBuilder, spec: &OpSpec, acc: ExprRef, i: usize) -> ExprRef {
    match spec.stage {
        Stage::UnaryScaled(c0, c1) => {
            let c = arena.push_const(if i.is_multiple_of(2) { c0 } else { c1 });
            let m = arena.push_binary(OpKind::Mul, acc, c);
            arena.push_unary(spec.op, m)
        }
        Stage::BinaryConstRight(c0, c1) => {
            let c = arena.push_const(if i.is_multiple_of(2) { c0 } else { c1 });
            arena.push_binary(spec.op, acc, c)
        }
        Stage::BinaryConstLeft(c0, c1) => {
            let c = arena.push_const(if i.is_multiple_of(2) { c0 } else { c1 });
            arena.push_binary(spec.op, c, acc)
        }
        Stage::FusedTernary(a, b) => {
            let ca = arena.push_const(a);
            let cb = arena.push_const(b);
            arena.push_ternary(spec.op, acc, ca, cb)
        }
        Stage::CondBlend(c0, c1) => {
            let zero = arena.push_const(0.0);
            let cond = arena.push_binary(OpKind::Ge, acc, zero);
            let ca = arena.push_const(c0);
            let cb = arena.push_const(c1);
            let a = arena.push_binary(OpKind::Mul, acc, ca);
            let b = arena.push_binary(OpKind::Mul, acc, cb);
            arena.push_ternary(spec.op, cond, a, b)
        }
    }
}

/// Throughput kernel: sum of K independent op applications (or, for the
/// baseline, the same kernel with each op application replaced by its
/// argument). Marginal cost = (op_kernel - baseline) / K.
fn throughput_term(spec: &OpSpec, with_op: bool) -> (Rooted<ExprData>, Environment) {
    let mut arena = ExprBuilder::new();
    let vars: Vec<ExprRef> = (0..4).map(|i| arena.push_var(i)).collect();
    let mut acc: Option<ExprRef> = None;
    for i in 0..K_THROUGHPUT {
        let v = vars[i % 4];
        let c = arena.push_const(0.3 + 0.11 * i as f32);
        let m = arena.push_binary(OpKind::Mul, v, c);
        let term = if with_op {
            match spec.stage {
                Stage::UnaryScaled(..) => arena.push_unary(spec.op, m),
                Stage::BinaryConstRight(c0, _) => {
                    let k = arena.push_const(c0 + 0.013 * i as f32);
                    arena.push_binary(spec.op, m, k)
                }
                Stage::BinaryConstLeft(c0, _) => {
                    let k = arena.push_const(c0 + 0.013 * i as f32);
                    arena.push_binary(spec.op, k, m)
                }
                Stage::FusedTernary(a, b) => {
                    let ca = arena.push_const(a + 0.013 * i as f32);
                    let cb = arena.push_const(b);
                    arena.push_ternary(spec.op, m, ca, cb)
                }
                Stage::CondBlend(c0, c1) => {
                    let zero = arena.push_const(0.0);
                    let cond = arena.push_binary(OpKind::Ge, m, zero);
                    let ca = arena.push_const(c0);
                    let cb = arena.push_const(c1);
                    let a = arena.push_binary(OpKind::Mul, m, ca);
                    let b = arena.push_binary(OpKind::Mul, m, cb);
                    arena.push_ternary(spec.op, cond, a, b)
                }
            }
        } else {
            m
        };
        acc = Some(match acc {
            None => term,
            Some(prev) => arena.push_binary(OpKind::Add, prev, term),
        });
    }
    let root = acc.expect("K_THROUGHPUT > 0");
    arena.finish(&[root])
}

fn median_of(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

/// Measure one term `reps` times, return median ns (raw, not
/// overhead-adjusted — every consumer here works on differences that cancel
/// overhead exactly).
fn measure(session: &mut BenchSession, term: Term<'_>, mode: BenchMode, reps: usize) -> f64 {
    let samples: Vec<f64> = (0..reps)
        .map(|_| {
            session
                .benchmark_term(term, mode)
                .unwrap_or_else(|e| panic!("benchmark failed: {e}"))
                .ns
        })
        .collect();
    median_of(samples)
}

fn main() {
    let reps = 5;
    let mut session = BenchSession::new();
    println!(
        "# session: call_overhead(throughput)={:.3}ns call_overhead(latency)={:.3}ns sentinel={:.3}ns",
        session.call_overhead_ns(BenchMode::Throughput),
        session.call_overhead_ns(BenchMode::Latency),
        session.calibration_ns(),
    );

    let specs = specs();

    // Pass 1: latency chain slopes.
    let mut lat_slope = Vec::new();
    for spec in &specs {
        let (short, short_env) = chain_term(spec, K_SHORT);
        let (long, long_env) = chain_term(spec, K_LONG);
        let ns_short = measure(
            &mut session,
            Term::new(short.entry(), &short_env),
            BenchMode::Latency,
            reps,
        );
        let ns_long = measure(
            &mut session,
            Term::new(long.entry(), &long_env),
            BenchMode::Latency,
            reps,
        );
        let slope = (ns_long - ns_short) / (K_LONG - K_SHORT) as f64;
        lat_slope.push(slope);
    }

    // Pass 2: throughput marginals.
    let mut thr_marginal = Vec::new();
    for spec in &specs {
        let (with_op, with_op_env) = throughput_term(spec, true);
        let (base, base_env) = throughput_term(spec, false);
        let ns_op = measure(
            &mut session,
            Term::new(with_op.entry(), &with_op_env),
            BenchMode::Throughput,
            reps,
        );
        let ns_base = measure(
            &mut session,
            Term::new(base.entry(), &base_env),
            BenchMode::Throughput,
            reps,
        );
        thr_marginal.push((ns_op - ns_base) / K_THROUGHPUT as f64);
    }

    let mul_slope = specs
        .iter()
        .zip(&lat_slope)
        .find(|(s, _)| s.op == OpKind::Mul)
        .map(|(_, &v)| v)
        .expect("Mul is measured");
    let add_slope = specs
        .iter()
        .zip(&lat_slope)
        .find(|(s, _)| s.op == OpKind::Add)
        .map(|(_, &v)| v)
        .expect("Add is measured");

    let prior = pixelflow_search::egraph::CostModel::latency_prior();

    println!(
        "\n{:<10} {:>12} {:>12} {:>14} {:>12} {:>10}",
        "op", "lat_ns", "tbl_cycles", "(Add=4 unit)", "thr_ns", "prior"
    );
    for ((spec, &slope), &thr) in specs.iter().zip(&lat_slope).zip(&thr_marginal) {
        // Unary/select stages carry a Mul on the chain path; subtract its slope.
        let lat_ns = match spec.stage {
            Stage::UnaryScaled(..) | Stage::CondBlend(..) => slope - mul_slope,
            _ => slope,
        };
        // Express in the table's unit: Add is pinned at 4.
        let table_cycles = 4.0 * lat_ns / add_slope;
        println!(
            "{:<10} {:>12.3} {:>12.2} {:>14} {:>12.3} {:>10}",
            format!("{:?}", spec.op),
            lat_ns,
            table_cycles,
            "",
            thr,
            prior.cost(spec.op),
        );
    }
    println!(
        "\n# anchors: Add slope {:.3}ns/stage, Mul slope {:.3}ns/stage; \
         assuming FADD latency 3cyc => est. clock {:.2}GHz",
        add_slope,
        mul_slope,
        3.0 / add_slope,
    );
}
