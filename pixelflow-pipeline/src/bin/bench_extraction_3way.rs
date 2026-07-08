//! Phase 2 gate of docs/plans/2026-07-07-guided-saturation-redesign.md: the
//! deferred February 3-way extraction experiment, redone with the freshly
//! retrained Judge (`pixelflow-pipeline/data/expr_nnue_trid.bin`).
//!
//! On the SAME saturated e-graph, compares three extraction policies with
//! real JIT wall-clock:
//!   (a) NNUE      — `IncrementalExtractor` guided by the trained Judge
//!   (b) STATIC    — `extract_dag` + `CostModel::latency_prior()`
//!   (c) NO-SWAP   — the original expression form, un-extracted (the
//!                   peephole/CSE-free baseline: JIT the input as-is)
//!
//! Corpus: the five named production-shaped kernels from
//! `pixelflow-search/examples/rule_report.rs` / `pixelflow-search/tests/prod_kernel_jit.rs`
//! (swirl, circle_sdf, poly, redundant, normalize), reported individually,
//! plus a batch of synthetic expressions from `BwdGenerator` (the same
//! generator `bootstrap_extraction_head` trains on), reported in aggregate.
//!
//! Correctness gate: all three extracted forms must agree numerically on a
//! grid of coordinates before their timings are trusted — see `ABS_TOL`/
//! `REL_TOL`. A policy that returns a wrong kernel invalidates its speed
//! number, so failures are recorded per-kernel and excluded from the
//! geomean rather than silently contaminating it.
//!
//! Run: `cargo run --release -p pixelflow-pipeline --features training --bin bench_extraction_3way`

use std::time::Instant;

use pixelflow_ir::backend::emit::compile_arena_dag;
use pixelflow_ir::backend::emit::executable::{ExecutableCode, KernelFn};
use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::egraph::{
    CostModel, EGraph, IncrementalExtractor, all_rules, choices_to_arena, collect_rule_templates,
    extract_dag,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, ExprNnue};

use pixelflow_pipeline::jit_bench::benchmark_jit_arena;

/// Freshly retrained Judge weights (TRID format), verified loadable by
/// `pixelflow-pipeline/tests/judge_weights_load.rs`.
const NNUE_WEIGHTS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/expr_nnue_trid.bin");

/// E-graph saturation budget — matches `prod_kernel_jit.rs`.
const SATURATE_LIMIT: usize = 40;

/// Alternatives considered per e-class during NNUE refinement — matches
/// `prod_kernel_jit.rs` and `pixelflow-compiler/src/optimize.rs`.
const EXTRACT_TOP_K: usize = 8;

/// Synthetic corpus size (within the plan's requested 30-50 range).
const SYNTHETIC_COUNT: usize = 40;
const SYNTHETIC_SEED: u64 = 0xB0BA_2026;

/// Coordinates for the correctness cross-check grid. Spans zero, small,
/// negative, and larger-magnitude values across all four input registers so
/// a wrong-kernel bug (extraction returning an unrelated expression) shows
/// up, while staying inside the domain every named kernel is well-defined
/// on (circle_sdf/normalize divide/sqrt near-zero are avoided).
const GRID: [(f32, f32, f32, f32); 8] = [
    (0.0, 0.0, 0.0, 0.0),
    (0.3, 0.2, 0.1, -0.1),
    (-0.5, 0.4, 0.2, 0.3),
    (0.8, -0.6, -0.3, 0.5),
    (1.0, 1.0, 0.5, -0.5),
    (-1.2, 0.1, 0.7, 0.2),
    (0.15, -0.95, -0.4, 0.6),
    (0.6, 0.6, -0.2, -0.3),
];

/// Absolute tolerance for the numeric cross-check between extracted forms.
/// Loose enough to absorb differing Chebyshev-polynomial transcendental
/// lowering paths (sin/cos/rsqrt approximate differently depending on which
/// algebraic form survives extraction — see `prod_kernel_jit.rs`'s 1e-1
/// original-vs-optimized cross-check bound), tight enough that a
/// wrong-kernel bug (an unrelated expression) still trips it.
const ABS_TOL: f32 = 2e-1;
/// Relative tolerance, used when the reference magnitude is large.
const REL_TOL: f32 = 0.05;

// ---------------------------------------------------------------------------
// Named kernel corpus — verbatim from pixelflow-search/examples/rule_report.rs
// (which itself reuses the swirl shape from
// pixelflow-search/tests/prod_kernel_jit.rs's build_swirl).
// ---------------------------------------------------------------------------

/// sin(sqrt(x*x + y*y) * freq) * amp + bias — the swirl shader core.
fn swirl() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let kf = a.push_const(3.0);
    let sf = a.push_binary(OpKind::Mul, s, kf);
    let sn = a.push_unary(OpKind::Sin, sf);
    let ka = a.push_const(0.5);
    let prod = a.push_binary(OpKind::Mul, sn, ka);
    let kb = a.push_const(0.5);
    let out = a.push_binary(OpKind::Add, prod, kb);
    (a, out)
}

/// Circle SDF: sqrt((x-cx)^2 + (y-cy)^2) - r.
fn circle_sdf() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let cx = a.push_const(0.3);
    let cy = a.push_const(-0.2);
    let dx = a.push_binary(OpKind::Sub, x, cx);
    let dy = a.push_binary(OpKind::Sub, y, cy);
    let dx2 = a.push_binary(OpKind::Mul, dx, dx);
    let dy2 = a.push_binary(OpKind::Mul, dy, dy);
    let sum = a.push_binary(OpKind::Add, dx2, dy2);
    let dist = a.push_unary(OpKind::Sqrt, sum);
    let r = a.push_const(0.5);
    let out = a.push_binary(OpKind::Sub, dist, r);
    (a, out)
}

/// FMA-bait polynomial: a*x*x + b*x + c (Horner-able, fusion-able).
fn poly() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let ka = a.push_const(2.0);
    let kb = a.push_const(-3.0);
    let kc = a.push_const(1.0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let ax2 = a.push_binary(OpKind::Mul, ka, xx);
    let bx = a.push_binary(OpKind::Mul, kb, x);
    let s1 = a.push_binary(OpKind::Add, ax2, bx);
    let out = a.push_binary(OpKind::Add, s1, kc);
    (a, out)
}

/// Redundancy bait: (x+y)*(x+y) + 2*(x+y) — CSE + distribution territory.
fn redundant() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_binary(OpKind::Add, x, y);
    let s2 = a.push_binary(OpKind::Mul, s, s);
    let two = a.push_const(2.0);
    let ts = a.push_binary(OpKind::Mul, two, s);
    let out = a.push_binary(OpKind::Add, s2, ts);
    (a, out)
}

/// Division/sqrt bait: x / sqrt(x*x + y*y) (normalize — rsqrt rewrites).
fn normalize() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let out = a.push_binary(OpKind::Div, x, s);
    (a, out)
}

fn named_kernels() -> Vec<(&'static str, ExprArena, ExprId)> {
    let (a, ra) = swirl();
    let (b, rb) = circle_sdf();
    let (c, rc) = poly();
    let (d, rd) = redundant();
    let (e, re) = normalize();
    vec![
        ("swirl", a, ra),
        ("circle_sdf", b, rb),
        ("poly", c, rc),
        ("redundant", d, rd),
        ("normalize", e, re),
    ]
}

// ---------------------------------------------------------------------------
// JIT execution helper — verbatim from pixelflow-search/tests/prod_kernel_jit.rs.
// Broadcasts one coordinate to all lanes, reads lane 0.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::x86_64::{_mm_cvtss_f32, _mm_set1_ps};
    // SAFETY: SSE2 is the baseline on x86-64; the JIT emitted `__m128` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            _mm_set1_ps(x),
            _mm_set1_ps(y),
            _mm_set1_ps(z),
            _mm_set1_ps(w),
        );
        _mm_cvtss_f32(r)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::x86_64::{_mm512_cvtss_f32, _mm512_set1_ps};
    // SAFETY: built with +avx512f, so the JIT emitted the `__m512` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            _mm512_set1_ps(x),
            _mm512_set1_ps(y),
            _mm512_set1_ps(z),
            _mm512_set1_ps(w),
        );
        _mm512_cvtss_f32(r)
    }
}

#[cfg(target_arch = "aarch64")]
fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
    // SAFETY: NEON is mandatory on aarch64; the JIT emitted the `float32x4_t` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            vdupq_n_f32(x),
            vdupq_n_f32(y),
            vdupq_n_f32(z),
            vdupq_n_f32(w),
        );
        vgetq_lane_f32(r, 0)
    }
}

// ---------------------------------------------------------------------------
// Per-policy extraction + JIT + benchmark
// ---------------------------------------------------------------------------

/// Outcome of taking one extraction policy from a saturated e-graph (or, for
/// no-swap, the raw arena) through JIT compilation and benchmarking.
enum PolicyOutcome {
    Ok { ns: f64 },
    CorrectnessFail { detail: String },
    CompileFail { detail: String },
}

/// JIT-compile `arena`/`root`, cross-check against `reference_jit` on
/// `GRID`, and benchmark if correct. `reference_jit` is `None` only for the
/// no-swap policy's own compile (which IS the reference — nothing to check
/// itself against).
fn compile_check_and_bench(
    label: &str,
    arena: &ExprArena,
    root: ExprId,
    reference: Option<&ExecutableCode>,
) -> PolicyOutcome {
    let compiled = match compile_arena_dag(arena, root) {
        Ok(c) => c,
        Err(e) => {
            return PolicyOutcome::CompileFail {
                detail: format!("{label}: JIT compile failed: {e}"),
            };
        }
    };

    if let Some(reference) = reference {
        for &(x, y, z, w) in &GRID {
            let want = run_scalar(reference, x, y, z, w);
            let got = run_scalar(&compiled.code, x, y, z, w);
            let diff = (got - want).abs();
            let tol = ABS_TOL.max(want.abs() * REL_TOL);
            if diff > tol {
                return PolicyOutcome::CorrectnessFail {
                    detail: format!(
                        "{label}: mismatch at ({x},{y},{z},{w}): got {got}, want {want} \
                         (diff {diff:.4} > tol {tol:.4})"
                    ),
                };
            }
        }
    }

    match benchmark_jit_arena(arena, root) {
        Ok(bench) => PolicyOutcome::Ok { ns: bench.ns },
        Err(e) => PolicyOutcome::CompileFail {
            detail: format!("{label}: benchmark_jit_arena failed post-compile: {e}"),
        },
    }
}

/// Result of running all three policies on one kernel.
struct KernelResult {
    name: String,
    n_orig: usize,
    ns_noswap: f64,
    ns_static: Option<f64>,
    ns_nnue: Option<f64>,
    extract_us_static: f64,
    extract_us_nnue: f64,
    static_fail: Option<String>,
    nnue_fail: Option<String>,
}

fn evaluate_kernel(name: &str, arena: &ExprArena, root: ExprId, nnue: &ExprNnue) -> KernelResult {
    let n_orig = arena.node_count_subtree(root);

    // NO-SWAP is the ground truth reference for correctness: it's exactly
    // the input expression, unmodified by any extraction policy.
    let noswap_compiled = compile_arena_dag(arena, root).unwrap_or_else(|e| {
        panic!(
            "evaluate_kernel({name}): the *original*, unmodified expression failed to \
             JIT-compile ({e}) — this is a JIT bug, not an extraction bug; failing loudly \
             rather than silently skipping the kernel"
        )
    });
    let noswap_bench = benchmark_jit_arena(arena, root).unwrap_or_else(|e| {
        panic!("evaluate_kernel({name}): benchmarking the original expression failed: {e}")
    });
    let ns_noswap = noswap_bench.ns;

    // Shared saturated e-graph for both learned and static extraction.
    let mut eg = EGraph::with_rules(all_rules());
    let root_class = eg.add_arena(arena, root);
    eg.saturate_with_limit(SATURATE_LIMIT);

    // (a) NNUE extraction.
    let t_nnue = Instant::now();
    let extractor = IncrementalExtractor::new(nnue, EXTRACT_TOP_K);
    let (_cost, nnue_choices) = extractor.extract_choices_only(&eg, root_class);
    let (nnue_arena, nnue_root) = choices_to_arena(&eg, root_class, &nnue_choices);
    let extract_us_nnue = t_nnue.elapsed().as_secs_f64() * 1e6;

    // (b) STATIC extraction — CostModel::latency_prior() via extract_dag.
    let t_static = Instant::now();
    let static_costs = CostModel::latency_prior();
    let static_dag = extract_dag(&eg, root_class, &static_costs);
    let (static_arena, static_root) = choices_to_arena(&eg, root_class, &static_dag.choices);
    let extract_us_static = t_static.elapsed().as_secs_f64() * 1e6;

    let (ns_static, static_fail) = match compile_check_and_bench(
        &format!("{name}/static"),
        &static_arena,
        static_root,
        Some(&noswap_compiled.code),
    ) {
        PolicyOutcome::Ok { ns, .. } => (Some(ns), None),
        PolicyOutcome::CorrectnessFail { detail } | PolicyOutcome::CompileFail { detail } => {
            eprintln!("[FAIL] {detail}");
            (None, Some(detail))
        }
    };

    let (ns_nnue, nnue_fail) = match compile_check_and_bench(
        &format!("{name}/nnue"),
        &nnue_arena,
        nnue_root,
        Some(&noswap_compiled.code),
    ) {
        PolicyOutcome::Ok { ns, .. } => (Some(ns), None),
        PolicyOutcome::CorrectnessFail { detail } | PolicyOutcome::CompileFail { detail } => {
            eprintln!("[FAIL] {detail}");
            (None, Some(detail))
        }
    };

    KernelResult {
        name: name.to_string(),
        n_orig,
        ns_noswap,
        ns_static,
        ns_nnue,
        extract_us_static,
        extract_us_nnue,
        static_fail,
        nnue_fail,
    }
}

// ---------------------------------------------------------------------------
// Synthetic corpus scan for Gather/RawGather/Buffer (should never appear —
// BwdGenerator has no code path that emits them — but the plan asks to
// verify this for comparability rather than assume it).
// ---------------------------------------------------------------------------

fn contains_memory_ops(arena: &ExprArena, root: ExprId) -> bool {
    let n = arena.node_count_subtree(root);
    let _ = n; // node_count_subtree doesn't give us indices; scan the whole arena instead.
    (0..arena.len()).any(|i| {
        matches!(
            arena.kind(ExprId(i as u32)),
            OpKind::Gather | OpKind::RawGather | OpKind::Buffer
        )
    })
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

fn geomean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let sum_ln: f64 = values.iter().map(|v| v.ln()).sum();
    (sum_ln / values.len() as f64).exp()
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v:.3}"),
        None => "FAIL".to_string(),
    }
}

fn main() {
    eprintln!("=== Extraction 3-way bench: NNUE vs STATIC vs NO-SWAP ===");
    eprintln!("Loading Judge weights from {NNUE_WEIGHTS_PATH}");

    let weights_bytes = std::fs::read(NNUE_WEIGHTS_PATH).unwrap_or_else(|e| {
        panic!("failed to read Judge weights at {NNUE_WEIGHTS_PATH}: {e}")
    });
    let nnue = ExprNnue::from_bytes(&weights_bytes).unwrap_or_else(|e| {
        panic!(
            "ExprNnue::from_bytes rejected {NNUE_WEIGHTS_PATH}: {e}. \
             Retrain must produce valid TRID-format weights before this bench can run."
        )
    });
    eprintln!("Judge weights loaded OK ({} bytes).\n", weights_bytes.len());

    // ---------------------------------------------------------------
    // Named kernels
    // ---------------------------------------------------------------
    let mut named_results: Vec<KernelResult> = Vec::new();
    for (name, arena, root) in named_kernels() {
        eprintln!("--- named kernel: {name} ---");
        named_results.push(evaluate_kernel(name, &arena, root, &nnue));
    }

    // ---------------------------------------------------------------
    // Synthetic corpus
    // ---------------------------------------------------------------
    eprintln!("\n--- synthetic corpus (n={SYNTHETIC_COUNT}) ---");
    let templates = collect_rule_templates();
    let mut generator = BwdGenerator::new(SYNTHETIC_SEED, BwdGenConfig::default(), templates);

    let mut synthetic_results: Vec<KernelResult> = Vec::new();
    let mut memory_op_count = 0usize;
    for i in 0..SYNTHETIC_COUNT {
        let pair = generator.generate_arena();
        if contains_memory_ops(&pair.arena, pair.unoptimized) {
            memory_op_count += 1;
        }
        let name = format!("synth_{i:03}");
        synthetic_results.push(evaluate_kernel(&name, &pair.arena, pair.unoptimized, &nnue));
    }
    if memory_op_count > 0 {
        eprintln!(
            "warning: {memory_op_count}/{SYNTHETIC_COUNT} synthetic expressions contained \
             Gather/RawGather/Buffer ops (unexpected for BwdGenerator output)"
        );
    } else {
        eprintln!(
            "confirmed: 0/{SYNTHETIC_COUNT} synthetic expressions contain Gather/RawGather/Buffer \
             (comparable to named kernels, none of which use memory ops either)"
        );
    }

    // ---------------------------------------------------------------
    // Report
    // ---------------------------------------------------------------
    println!("\n=== RESULTS ===\n");
    println!(
        "{:<14} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8} {:>8}",
        "kernel", "nodes", "ns_nnue", "ns_static", "ns_noswap", "nnue/stc", "stc/nosw", "ex_nnue", "ex_stc"
    );
    println!("{}", "-".repeat(14 + 6 + 10 * 3 + 9 * 2 + 8 * 2 + 16));

    let mut all_nnue_static_ratios: Vec<f64> = Vec::new();
    let mut all_static_noswap_ratios: Vec<f64> = Vec::new();

    let print_row = |r: &KernelResult, ratios: &mut Vec<f64>, ratios2: &mut Vec<f64>| {
        let ratio_ns = match r.ns_nnue {
            Some(nnue) => r.ns_static.map(|s| nnue / s),
            None => None,
        };
        let ratio_sn = r.ns_static.map(|s| s / r.ns_noswap);
        if let Some(v) = ratio_ns {
            ratios.push(v);
        }
        if let Some(v) = ratio_sn {
            ratios2.push(v);
        }
        println!(
            "{:<14} {:>6} {:>10} {:>10} {:>10.3} {:>9} {:>9} {:>8.2} {:>8.2}",
            r.name,
            r.n_orig,
            fmt_opt(r.ns_nnue),
            fmt_opt(r.ns_static),
            r.ns_noswap,
            ratio_ns.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
            ratio_sn.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
            r.extract_us_nnue,
            r.extract_us_static,
        );
    };

    for r in &named_results {
        print_row(r, &mut all_nnue_static_ratios, &mut all_static_noswap_ratios);
    }

    println!("{}", "-".repeat(14 + 6 + 10 * 3 + 9 * 2 + 8 * 2 + 16));

    // Synthetic aggregate row (geomean ns per policy across the synthetic set).
    let syn_nnue_ns: Vec<f64> = synthetic_results.iter().filter_map(|r| r.ns_nnue).collect();
    let syn_static_ns: Vec<f64> = synthetic_results.iter().filter_map(|r| r.ns_static).collect();
    let syn_noswap_ns: Vec<f64> = synthetic_results.iter().map(|r| r.ns_noswap).collect();
    let syn_extract_nnue: Vec<f64> = synthetic_results.iter().map(|r| r.extract_us_nnue).collect();
    let syn_extract_static: Vec<f64> = synthetic_results.iter().map(|r| r.extract_us_static).collect();

    let syn_nnue_fail = synthetic_results.iter().filter(|r| r.nnue_fail.is_some()).count();
    let syn_static_fail = synthetic_results.iter().filter(|r| r.static_fail.is_some()).count();

    for r in &synthetic_results {
        let ratio_ns = match r.ns_nnue {
            Some(nnue) => r.ns_static.map(|s| nnue / s),
            None => None,
        };
        let ratio_sn = r.ns_static.map(|s| s / r.ns_noswap);
        if let Some(v) = ratio_ns {
            all_nnue_static_ratios.push(v);
        }
        if let Some(v) = ratio_sn {
            all_static_noswap_ratios.push(v);
        }
    }

    println!(
        "{:<14} {:>6} {:>10} {:>10} {:>10.3} {:>9} {:>9} {:>8.2} {:>8.2}",
        format!("synth(n={SYNTHETIC_COUNT})"),
        syn_noswap_ns.len(),
        format!("{:.3}", geomean(&syn_nnue_ns)),
        format!("{:.3}", geomean(&syn_static_ns)),
        geomean(&syn_noswap_ns),
        format!("{:.3}", geomean(&syn_nnue_ns) / geomean(&syn_static_ns)),
        format!("{:.3}", geomean(&syn_static_ns) / geomean(&syn_noswap_ns)),
        geomean(&syn_extract_nnue),
        geomean(&syn_extract_static),
    );
    println!(
        "  (synthetic failures: nnue={syn_nnue_fail}/{SYNTHETIC_COUNT}, static={syn_static_fail}/{SYNTHETIC_COUNT})"
    );

    println!("{}", "-".repeat(14 + 6 + 10 * 3 + 9 * 2 + 8 * 2 + 16));

    let geomean_nnue_static = geomean(&all_nnue_static_ratios);
    let geomean_static_noswap = geomean(&all_static_noswap_ratios);

    println!(
        "GEOMEAN (n={} nnue/static pairs, n={} static/noswap pairs): nnue/static={:.4}  static/noswap={:.4}",
        all_nnue_static_ratios.len(),
        all_static_noswap_ratios.len(),
        geomean_nnue_static,
        geomean_static_noswap,
    );

    let named_nnue_fail = named_results.iter().filter(|r| r.nnue_fail.is_some()).count();
    let named_static_fail = named_results.iter().filter(|r| r.static_fail.is_some()).count();
    println!(
        "\nNamed-kernel failures: nnue={named_nnue_fail}/{}, static={named_static_fail}/{}",
        named_results.len(),
        named_results.len(),
    );

    let extract_us_nnue_all: Vec<f64> = named_results
        .iter()
        .map(|r| r.extract_us_nnue)
        .chain(synthetic_results.iter().map(|r| r.extract_us_nnue))
        .collect();
    let extract_us_static_all: Vec<f64> = named_results
        .iter()
        .map(|r| r.extract_us_static)
        .chain(synthetic_results.iter().map(|r| r.extract_us_static))
        .collect();
    println!(
        "Mean extraction overhead: nnue={:.2}us static={:.2}us (n={})",
        extract_us_nnue_all.iter().sum::<f64>() / extract_us_nnue_all.len() as f64,
        extract_us_static_all.iter().sum::<f64>() / extract_us_static_all.len() as f64,
        extract_us_nnue_all.len(),
    );

    println!("\n=== GATE VERDICT ===");
    let pct = (geomean_nnue_static - 1.0) * -100.0; // positive = NNUE faster than static
    if geomean_nnue_static.is_nan() {
        println!("INCONCLUSIVE: no valid nnue/static pairs (all failed correctness/compile).");
    } else if pct > 5.0 {
        println!(
            "NNUE > static latency prior by {pct:.1}% geomean — Phase 2 gate PASSES, \
             NNUE extraction earns its keep."
        );
    } else if pct < -5.0 {
        println!(
            "NNUE < static latency prior by {:.1}% geomean — Phase 2 gate FAILS per the plan: \
             port the prior into CostModel as default, keep NNUE opt-in only.",
            -pct
        );
    } else {
        println!(
            "NNUE approx-ties static latency prior ({pct:+.1}% geomean, within the +/-5% band) \
             — Phase 2 gate FAILS per the plan (no meaningful win): \
             port the prior into CostModel as default, keep NNUE opt-in only."
        );
    }
}
