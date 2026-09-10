//! Horner vs Estrin: does restructuring a polynomial for ILP pay on the code
//! this JIT actually emits?
//!
//! Same coefficients, same value, two schedules (`pixelflow_pipeline::poly`).
//! Horner is a serial chain of `n` FMAs — the form
//! `pixelflow_ir::passes::horner_step` emits for every transcendental
//! expansion. Estrin is a `log₂ n`-deep tree over `log₂ n` extra squarings.
//! The extraction cost model sums op costs, so it prices Estrin as strictly
//! worse and would never choose it; whether the *machine* agrees depends
//! entirely on whether anything else is available to fill the chain's stalls.
//!
//! Three regimes, because the answer differs in each and only one of them is
//! production:
//!
//! - `Latency`: consecutive evaluations serialized. Nothing to overlap, so the
//!   critical path is the whole cost. Estrin's best case.
//! - `Throughput`: independent evaluations, one per call. The out-of-order
//!   window overlaps them across a call boundary production does not pay.
//! - `Scanline`: one call per 64-group row, the kernel's own emitted loop
//!   supplying the independent evaluations over one contiguous, ascending
//!   address stream. **This is the shape `Lattice`'s collapse runs**, and the
//!   only regime whose answer is a shipping decision.
//!
//! Also priced: the four polynomials production actually evaluates
//! (`SIN_CHEB`, `EXP2_POLY`, `LOG2_POLY`, `ATAN_MINIMAX`) at their real degrees
//! with their real coefficients; the accuracy each schedule delivers against an
//! `f64` reference (Estrin reassociates, so it is a different rounding, not
//! just a different order); and the underflow hazard Estrin introduces and
//! Horner does not — see [`HAZARD_SCALE`] — measured both as the code runs
//! today and under `FastMathGuard` (FTZ/DAZ), which is the fix for it and
//! which nothing in the render path currently holds.
//!
//! Run: `cargo run --release -p pixelflow-pipeline --example horner_vs_estrin`
//! At other ISA levels (whether `MulAdd` is one instruction or two is most of
//! the question): `RUSTFLAGS="-C llvm-args=-fp-contract=fast -C
//! target-feature=+avx2,+fma" cargo run --release ...`

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::{JIT_VECTOR_BYTES, Point4, TileSlice};
use pixelflow_core::FastMathGuard;
use pixelflow_ir::passes::{ATAN_MINIMAX, EXP2_POLY, LOG2_POLY, SIN_CHEB};
use pixelflow_ir::{Environment, ExprBuilder, ExprData, Node, OpKind, Rooted, Term};
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession};
use pixelflow_pipeline::poly::{PolyForm, build, critical_path};
use pixelflow_search::egraph::CostModel;

const LANES: usize = JIT_VECTOR_BYTES / 4;

/// Degrees swept. Starts below the production polynomials (`ATAN_MINIMAX` is
/// 4 coefficients) and runs well past them, so a crossover is bracketed rather
/// than assumed.
const DEGREES: [usize; 8] = [4, 6, 8, 9, 12, 16, 24, 32];

/// Timing kernels evaluate at `clamp(X, [ARG_LO, ARG_HI]) · scale`.
///
/// The clamp is not cosmetic. The harness's input buffer spans `10⁻⁴` to `10⁴`
/// and `BenchMode::Latency` feeds the kernel its own output, so an unclamped
/// argument reaches Estrin's `x⁸`/`x¹⁶` at magnitudes that underflow — and an
/// underflowing multiply is a microcode assist, not an instruction. Left in, it
/// moves the throughput column by 6×, which measures the x86 denormal path and
/// not either schedule. Clamped, both forms see identical well-scaled
/// arguments, and the hazard gets measured deliberately at [`HAZARD_SCALE`]
/// instead of leaking into every row.
const ARG_LO: f32 = 0.25;
const ARG_HI: f32 = 1.0;

/// Scale for the arithmetic tables: none.
const NOMINAL_SCALE: f32 = 1.0;

/// Scale for the underflow-hazard table. `arg ≈ 10⁻⁹` is an ordinary normal
/// `f32`, and every Horner step keeps the accumulator near `a₀` whatever the
/// argument — but Estrin forms `x⁸ ≈ 10⁻⁷²` and `x¹⁶` explicitly, and those
/// underflow. Same schedules, same coefficients, one constant different.
///
/// The hazard is a property of the FP *mode*, not of the arithmetic:
/// `FastMathGuard` sets FTZ/DAZ, after which an underflow is a zero rather
/// than a microcode assist. So the table is run twice — with the mode the
/// render path actually runs in (nothing holds the guard today) and with the
/// mode one line of code would give it.
const HAZARD_SCALE: f32 = 1e-9;

/// Degrees the hazard table sweeps: `x⁸` is where the powers start to
/// underflow at [`HAZARD_SCALE`].
const HAZARD_DEGREES: [usize; 3] = [8, 16, 32];

/// Accuracy sweep: `X · ACCURACY_SCALE` over `X ∈ 0..1024` covers `[0, 1)`, the
/// interval the coefficients are conditioned for. Separate from the timing
/// argument because a clamp is the wrong generator for an error sweep and a
/// sweep is the wrong generator for a timing loop.
const ACCURACY_SCALE: f32 = 1.0 / 1024.0;
const ACCURACY_POINTS: usize = 1024;

/// Repeat each timing this many times and take the median. The session's own
/// median-of-20 handles sample noise; this handles run-to-run placement.
const REPS: usize = 5;

/// Well-conditioned on `[0, 1]`: alternating sign, decaying magnitude. A
/// disagreement between the two schedules on these coefficients is a
/// scheduling artifact, not catastrophic cancellation.
fn sweep_coeffs(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
            sign * 0.75f32.powi(i as i32) / (i as f32 + 1.0)
        })
        .collect()
}

/// `p(clamp(X, [ARG_LO, ARG_HI]) · scale)`. The clamp and the scaling `Mul` sit
/// on the dependency path in both arms, so every difference between them
/// belongs to the polynomial.
fn timing_kernel(form: PolyForm, coeffs: &[f32], scale: f32) -> (Rooted<ExprData>, Environment) {
    let mut arena = ExprBuilder::new();
    let x = arena.push_var(0);
    let lo = arena.push_const(ARG_LO);
    let hi = arena.push_const(ARG_HI);
    let clamped = arena.push_binary(OpKind::Max, x, lo);
    let clamped = arena.push_binary(OpKind::Min, clamped, hi);
    let s = arena.push_const(scale);
    let arg = arena.push_binary(OpKind::Mul, clamped, s);
    let root = build(&mut arena, form, coeffs, arg);
    arena.finish(&[root])
}

/// `p(X · ACCURACY_SCALE)` — the argument generator for the error sweep.
fn accuracy_kernel(form: PolyForm, coeffs: &[f32]) -> (Rooted<ExprData>, Environment) {
    let mut arena = ExprBuilder::new();
    let x = arena.push_var(0);
    let scale = arena.push_const(ACCURACY_SCALE);
    let arg = arena.push_binary(OpKind::Mul, x, scale);
    let root = build(&mut arena, form, coeffs, arg);
    arena.finish(&[root])
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

/// Median raw and overhead-adjusted ns per evaluation.
///
/// Both, because they answer different questions and the difference is large
/// here: raw is what a caller pays end to end, adjusted (audit M1) removes the
/// identity kernel's per-eval cost and leaves the arithmetic. At AVX-512 the
/// latency-mode overhead is ~1.4ns against a ~2ns kernel, so a raw ratio
/// understates a 1.6× difference in the polynomial as 1.1×.
fn measure(session: &mut BenchSession, term: Term<'_>, mode: BenchMode) -> (f64, f64) {
    let results: Vec<(f64, f64)> = (0..REPS)
        .map(|_| {
            let r = session
                .benchmark_term(term, mode)
                .unwrap_or_else(|e| panic!("benchmark failed: {e}"));
            (r.ns, r.adjusted_ns)
        })
        .collect();
    (
        median(results.iter().map(|r| r.0).collect()),
        median(results.iter().map(|r| r.1).collect()),
    )
}

/// Op nodes reachable from `root` — what the extraction cost model sums over.
fn nodes(root: Node<'_, ExprData>) -> usize {
    root.descendants()
        .filter(|n| matches!(**n, ExprData::Op(_)))
        .count()
}

/// Sum of the latency prior over reachable nodes: what
/// `CostModel::latency_prior` charges an extraction.
fn prior_sum(root: Node<'_, ExprData>, model: &CostModel) -> f64 {
    root.descendants()
        .filter_map(|n| match *n {
            ExprData::Op(op) => Some(model.cost(op) as f64),
            _ => None,
        })
        .sum()
}

/// What the emitter made of a schedule, alongside its outputs. Spills are the
/// price of Estrin's wider live set, and the first thing to check when a timing
/// moves the wrong way.
struct Emitted {
    outputs: Vec<f32>,
    spills: u32,
    bytes: usize,
}

/// Run a compiled kernel over `n` consecutive integer X values — the JIT's own
/// arithmetic, FMA rounding included, which no scalar reference reproduces and
/// which is why the error check runs here rather than through `eval_scalar`.
fn evaluate(term: Term<'_>, n: usize) -> Emitted {
    let result = compile(term).expect("compile");
    let groups = n.div_ceil(LANES);
    let mut out = vec![0.0f32; groups * LANES];
    let mut x0 = [0.0f32; LANES];
    for (i, lane) in x0.iter_mut().enumerate() {
        *lane = i as f32;
    }
    unsafe {
        result.code.call_collapse(
            core::ptr::null(),
            TileSlice::contiguous(out.as_mut_ptr(), groups, 1),
            Point4::new(x0, [0.0; LANES], [0.0; LANES], [0.0; LANES]),
        );
    }
    out.truncate(n);
    Emitted {
        outputs: out,
        spills: result.spill_count,
        bytes: result.code.len(),
    }
}

/// Max error against an `f64` Horner reference, normalized by the peak `|p|`
/// over the sampled arguments.
///
/// Peak-relative rather than pointwise-relative: `SIN_CHEB` has a root inside
/// the sampled range (it IS `sin` near `t = 1`), and a pointwise relative error
/// there measures the polynomial's conditioning at its own zero rather than the
/// schedule's rounding. Peak-relative is what a rendered pixel sees.
fn max_error(coeffs: &[f32], got: &[f32]) -> f64 {
    let mut worst = 0.0f64;
    let mut peak = f64::MIN_POSITIVE;
    for (i, &g) in got.iter().enumerate() {
        let arg = f64::from(i as f32 * ACCURACY_SCALE);
        let mut want = 0.0f64;
        for &c in coeffs.iter().rev() {
            want = want * arg + f64::from(c);
        }
        peak = peak.max(want.abs());
        worst = worst.max((f64::from(g) - want).abs());
    }
    worst / peak
}

/// One benchmarked polynomial, both schedules. Index 0 is Horner, 1 is Estrin.
struct Row {
    label: String,
    degree: usize,
    nodes: [usize; 2],
    path: [f64; 2],
    sum: [f64; 2],
    ns: [[f64; 2]; 3],
    adj: [[f64; 2]; 3],
    err: [f64; 2],
    spills: [u32; 2],
    bytes: [usize; 2],
}

impl Row {
    /// Scanline-mode ratio on RAW ns: end-to-end cost, the number a shipping
    /// decision is made on.
    fn scanline_ratio(&self) -> f64 {
        self.ns[2][0] / self.ns[2][1]
    }
}

const MODES: [BenchMode; 3] = [
    BenchMode::Latency,
    BenchMode::Throughput,
    BenchMode::Scanline,
];
const FORMS: [PolyForm; 2] = [PolyForm::Horner, PolyForm::Estrin];

fn bench_one(
    session: &mut BenchSession,
    label: &str,
    coeffs: &[f32],
    scale: f32,
    model: &CostModel,
) -> Row {
    let mut row = Row {
        label: label.to_string(),
        degree: coeffs.len(),
        nodes: [0; 2],
        path: [0.0; 2],
        sum: [0.0; 2],
        ns: [[0.0; 2]; 3],
        adj: [[0.0; 2]; 3],
        err: [0.0; 2],
        spills: [0; 2],
        bytes: [0; 2],
    };
    for (f, &form) in FORMS.iter().enumerate() {
        let (expr, env) = timing_kernel(form, coeffs, scale);
        let term = Term::new(expr.entry(), &env);
        row.nodes[f] = nodes(term.root());
        row.sum[f] = prior_sum(term.root(), model);
        row.path[f] = critical_path(term.root(), |k| model.cost(k) as f64);
        // Spills and code size describe the kernel that was TIMED; the
        // accuracy kernel below is a different (unclamped) argument generator
        // and would report a different frame.
        let timed = evaluate(term, LANES);
        row.spills[f] = timed.spills;
        row.bytes[f] = timed.bytes;
        for (m, &mode) in MODES.iter().enumerate() {
            let (raw, adjusted) = measure(session, term, mode);
            row.ns[m][f] = raw;
            row.adj[m][f] = adjusted;
        }

        let (acc, acc_env) = accuracy_kernel(form, coeffs);
        let accurate = evaluate(Term::new(acc.entry(), &acc_env), ACCURACY_POINTS);
        row.err[f] = max_error(coeffs, &accurate.outputs);
    }
    row
}

fn print_header() {
    println!(
        "\n{:<8} {:>3} {:>8} {:>8} {:>8} {:>7} {:>18} {:>18} {:>18} {:>10}",
        "poly",
        "n",
        "nodes",
        "path",
        "sum",
        "spills",
        "latency ns",
        "throughput ns",
        "scanline ns",
        "max err"
    );
    println!(
        "{:<8} {:>3} {:>8} {:>8} {:>8} {:>7} {:>18} {:>18} {:>18} {:>10}",
        "", "", "H/E", "H/E", "H/E", "H/E", "H / E (H÷E)", "H / E (H÷E)", "H / E (H÷E)", "H/E"
    );
}

fn print_row(row: &Row) {
    // Adjusted ns, with the ratio taken on the same quantity: the arithmetic,
    // not the arithmetic plus a constant that is identical in both arms and
    // therefore pulls every ratio toward 1.
    let cell = |m: usize| {
        let [h, e] = row.adj[m];
        format!("{h:.2}/{e:.2} ({:.2}x)", h / e)
    };
    println!(
        "{:<8} {:>3} {:>8} {:>8} {:>8} {:>7} {:>18} {:>18} {:>18} {:>10}",
        row.label,
        row.degree,
        format!("{}/{}", row.nodes[0], row.nodes[1]),
        format!("{:.0}/{:.0}", row.path[0], row.path[1]),
        format!("{:.0}/{:.0}", row.sum[0], row.sum[1]),
        format!("{}/{}", row.spills[0], row.spills[1]),
        cell(0),
        cell(1),
        cell(2),
        format!("{:.0e}/{:.0e}", row.err[0], row.err[1]),
    );
}

/// Marginal cost of one more degree, per call, by differencing two degrees.
///
/// The absolute per-row figures subtract a separately-calibrated overhead;
/// this cancels overhead AND the shared prelude exactly, which is the protocol
/// `measure_latency_prior` uses and the only one that survives a mode whose
/// fixed cost rivals the kernel. `BenchMode::Latency`'s chaining apparatus —
/// a 4×`LANES`-wide `Point4` stored to the stack, a call, and a store→load
/// round trip, all of it serial — costs ~30ns per call at AVX-512, which pins
/// every degree below ~16 to the same reading and makes an absolute ratio
/// there a measurement of the harness. The slope does not care.
///
/// Reported per CALL, not per lane: the quantity being compared is one more
/// instruction in the kernel, and a kernel instruction serves all `LANES`.
fn slope_ns_per_degree(
    session: &mut BenchSession,
    form: PolyForm,
    mode: BenchMode,
    lo: usize,
    hi: usize,
) -> f64 {
    let mut at = |n: usize| {
        let (expr, env) = timing_kernel(form, &sweep_coeffs(n), NOMINAL_SCALE);
        measure(session, Term::new(expr.entry(), &env), mode).0
    };
    let (ns_lo, ns_hi) = (at(lo), at(hi));
    (ns_hi - ns_lo) * LANES as f64 / (hi - lo) as f64
}

/// Degrees the slope is differenced across. `SLOPE_LO` is above the point
/// where AVX-512's latency-mode fixed cost stops dominating.
const SLOPE_LO: usize = 16;
const SLOPE_HI: usize = 32;

/// The polynomials production evaluates, with the coefficient tables
/// `pixelflow_ir::passes` expands — not a restatement of their arithmetic.
fn production_polys() -> [(&'static str, &'static [f32]); 4] {
    [
        ("sin/cos", SIN_CHEB.as_slice()),
        ("exp2", EXP2_POLY.as_slice()),
        ("log2", LOG2_POLY.as_slice()),
        ("atan", ATAN_MINIMAX.as_slice()),
    ]
}

fn main() {
    println!(
        "# horner vs estrin — JIT_VECTOR_BYTES={JIT_VECTOR_BYTES} (LANES={LANES}), \
         arch={}, host fma={}",
        std::env::consts::ARCH,
        cfg!(target_feature = "fma"),
    );

    let model = CostModel::latency_prior();
    let mut session = BenchSession::new();
    println!(
        "# session: overhead latency={:.3}ns throughput={:.3}ns scanline={:.4}ns sentinel={:.3}ns",
        session.call_overhead_ns(BenchMode::Latency),
        session.call_overhead_ns(BenchMode::Throughput),
        session.call_overhead_ns(BenchMode::Scanline),
        session.calibration_ns(),
    );
    println!(
        "# path = critical path through the latency prior; sum = what \
         CostModel::latency_prior charges extraction. Both in table cycles."
    );
    println!(
        "# table ns are per evaluation per lane, MINUS this session's identity-kernel \
         overhead for the mode;\n# (H÷E) above 1 means Estrin is faster. The verdict \
         section quotes raw (unadjusted) scanline ns."
    );

    println!("\n## degree sweep (synthetic, well-conditioned coefficients)");
    print_header();
    let sweep: Vec<Row> = DEGREES
        .iter()
        .map(|&n| {
            let row = bench_one(
                &mut session,
                "sweep",
                &sweep_coeffs(n),
                NOMINAL_SCALE,
                &model,
            );
            print_row(&row);
            row
        })
        .collect();

    println!("\n## production polynomials (real coefficients, real degrees)");
    print_header();
    let production: Vec<Row> = production_polys()
        .iter()
        .map(|(label, coeffs)| {
            let row = bench_one(&mut session, label, coeffs, NOMINAL_SCALE, &model);
            print_row(&row);
            row
        })
        .collect();

    println!(
        "\n## underflow hazard: identical kernels at arg ≈ {HAZARD_SCALE:e}\n\
         # Horner's accumulator stays near a₀ whatever the argument; Estrin \
         forms x⁸ and x¹⁶ explicitly,\n# and an underflowing multiply is a \
         microcode assist rather than an instruction. Compare with the same \
         degrees above."
    );
    println!("# FP mode: default (no FastMathGuard) — what the render path runs today");
    print_header();
    for &n in &HAZARD_DEGREES {
        let row = bench_one(
            &mut session,
            "hazard",
            &sweep_coeffs(n),
            HAZARD_SCALE,
            &model,
        );
        print_row(&row);
    }

    println!("# FP mode: FTZ/DAZ via FastMathGuard — the same kernels, denormals flushed");
    print_header();
    {
        // SAFETY: single-threaded example; the guard restores MXCSR/FPCR on
        // drop, and every kernel timed inside it is JIT code called from this
        // same thread, which is exactly the scope the mode is meant to cover.
        let _guard = unsafe { FastMathGuard::new() };
        for &n in &HAZARD_DEGREES {
            let row = bench_one(
                &mut session,
                "hazard+ftz",
                &sweep_coeffs(n),
                HAZARD_SCALE,
                &model,
            );
            print_row(&row);
        }
    }

    // What one more degree costs. This is the mechanism the per-row ratios are
    // a consequence of, and it is measured rather than inferred: if the two
    // schedules have the SAME marginal cost, the machine is throughput-bound
    // and only op count can separate them — which is the regime where an
    // additive cost model is exactly right.
    println!(
        "\n## marginal cost of one more degree (slope across n={SLOPE_LO}→{SLOPE_HI}, ns per call)"
    );
    println!(
        "# equal slopes = throughput-bound (op count decides, and Estrin has strictly more ops).\n\
         # horner slope above estrin = latency still exposed (depth decides, past some degree)."
    );
    println!("{:<12} {:>10} {:>10}   reading", "mode", "horner", "estrin");
    for &mode in &MODES {
        let h = slope_ns_per_degree(&mut session, PolyForm::Horner, mode, SLOPE_LO, SLOPE_HI);
        let e = slope_ns_per_degree(&mut session, PolyForm::Estrin, mode, SLOPE_LO, SLOPE_HI);
        let reading = if (h - e).abs() < 0.15 * h.abs().max(e.abs()) {
            "throughput-bound: same marginal op, Estrin pays its extra ones"
        } else if h > e {
            "latency exposed: Horner's chain costs more per degree"
        } else {
            "unexpected: Estrin's marginal degree costs more"
        };
        println!(
            "{:<12} {:>10.3} {:>10.3}   {reading}",
            format!("{mode:?}"),
            h,
            e
        );
    }

    // The decision line: scanline mode is production's regime, so its ratio is
    // the one that would justify (or refuse) a change to `passes::horner_step`.
    println!("\n## verdict (scanline = the contiguous row `compile` emits)");
    for row in sweep.iter().chain(production.iter()) {
        let [h, e] = row.ns[2];
        let verdict = if e < h * 0.98 {
            "estrin"
        } else if h < e * 0.98 {
            "horner"
        } else {
            "tie"
        };
        println!(
            "{:<8} n={:<3} scanline {h:.3}ns vs {e:.3}ns → {verdict:<6} ({:.2}x)  \
             [latency {:.2}x adj, {}/{} code bytes, sum-cost prefers {}]",
            row.label,
            row.degree,
            row.scanline_ratio(),
            row.adj[0][0] / row.adj[0][1],
            row.bytes[0],
            row.bytes[1],
            if row.sum[1] > row.sum[0] {
                "horner"
            } else {
                "estrin"
            },
        );
    }
}
