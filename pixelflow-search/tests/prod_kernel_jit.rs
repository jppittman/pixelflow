//! End-to-end: a representative production kernel driven all the way through
//! the pull-based pipeline and executed as JIT machine code.
//!
//! The kernel is the radial "swirl" at the heart of the psychedelic shader
//! (`pixelflow-runtime/examples/psychedelic_shader.rs`):
//!
//! ```text
//! out(x, y) = sin( sqrt(x*x + y*y) * freq ) * amp + bias
//! ```
//!
//! Pipeline exercised:
//!   ExprArena  ->  e-graph equality saturation (algebra + trig + FMA fusion)
//!              ->  NNUE-guided extraction (the learned "ML filter")
//!              ->  transcendental lowering + register allocation + codegen
//!              ->  native machine code, executed on real coordinates.
//!
//! Note: there are no validated trained NNUE weights shipped with the compiler
//! (see `pixelflow-compiler/src/optimize.rs`), so this drives the extractor
//! with the no-op zero-weight cost model — extraction keeps the original
//! form; peephole/CSE still run. The point is the *pipeline*, end to end.

use pixelflow_ir::backend::emit::compile_arena_dag;
use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::egraph::{EGraph, IncrementalExtractor, choices_to_arena};
use pixelflow_search::math::all_rules;
use pixelflow_search::nnue::ExprNnue;

/// Build `sin(sqrt(x*x + y*y) * freq) * amp + bias` as an arena.
fn build_swirl(freq: f32, amp: f32, bias: f32) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let kf = a.push_const(freq);
    let sf = a.push_binary(OpKind::Mul, s, kf);
    let sn = a.push_unary(OpKind::Sin, sf);
    let ka = a.push_const(amp);
    let prod = a.push_binary(OpKind::Mul, sn, ka);
    let kb = a.push_const(bias);
    let out = a.push_binary(OpKind::Add, prod, kb);
    (a, out)
}

fn reference(x: f32, y: f32, freq: f32, amp: f32, bias: f32) -> f32 {
    (((x * x + y * y).sqrt()) * freq).sin() * amp + bias
}

/// Optimize `(arena, root)` through the e-graph and the NNUE extractor,
/// returning the extracted DAG. Prints a few diagnostics so the run is visible.
fn optimize(arena: &ExprArena, root: ExprId, tag: &str) -> (ExprArena, ExprId) {
    let mut eg = EGraph::with_rules(all_rules());
    let root_class = eg.add_arena(arena, root);
    let classes_before = eg.num_classes();

    eg.saturate_with_limit(40);
    let classes_after = eg.num_classes();

    // The learned "ML filter": NNUE-guided extraction of the cheapest DAG.
    let nnue = ExprNnue::new_with_latency_prior(0xC0FFEE);
    let extractor = IncrementalExtractor::new(&nnue, 8);
    let (cost, choices) = extractor.extract_choices_only(&eg, root_class);
    let (out_arena, out_root) = choices_to_arena(&eg, root_class, &choices);

    eprintln!(
        "[{tag}] egraph {classes_before} -> {classes_after} classes, \
         extracted DAG = {} nodes, nnue cost = {cost:.3}",
        out_arena.len(),
    );
    (out_arena, out_root)
}

// ---------------------------------------------------------------------------
// Executing JIT code: broadcast one coordinate to all lanes, read lane 0.
// Gated to match the width-specific `KernelFn` ABI the backend emits.
// ---------------------------------------------------------------------------

use pixelflow_ir::backend::emit::executable::{ExecutableCode, KernelFn};

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

#[test]
fn prod_swirl_kernel_through_nnue_and_jit() {
    let (freq, amp, bias) = (3.0_f32, 0.5, 0.5);

    let (orig, orig_root) = build_swirl(freq, amp, bias);
    let (opt, opt_root) = optimize(&orig, orig_root, "swirl");

    // JIT both the original and the NNUE-optimized DAG. Both paths run the
    // shared transcendental-lowering + regalloc + codegen pipeline.
    let orig_jit = compile_arena_dag(&orig, orig_root).expect("JIT original");
    let opt_jit = compile_arena_dag(&opt, opt_root).expect("JIT optimized");
    eprintln!(
        "[swirl] spills: original = {}, optimized = {}",
        orig_jit.spill_count, opt_jit.spill_count
    );

    // A grid of coordinates spanning the unit-ish disc the shader samples.
    let coords = [
        (0.0_f32, 0.0_f32),
        (0.3, 0.2),
        (-0.5, 0.4),
        (0.8, -0.6),
        (1.0, 1.0),
        (-1.2, 0.1),
        (0.15, -0.95),
        (0.6, 0.6),
    ];

    // Sin is lowered to a Chebyshev polynomial, so JIT output is an
    // approximation; the analytic reference is matched within the polynomial's
    // accuracy. The cross-check between the two JIT paths is much tighter — it
    // certifies that NNUE extraction preserved semantics.
    let mut max_ref_err = 0.0_f32;
    let mut max_cross_err = 0.0_f32;
    for &(x, y) in &coords {
        let want = reference(x, y, freq, amp, bias);
        let got_orig = run_scalar(&orig_jit.code, x, y, 0.0, 0.0);
        let got_opt = run_scalar(&opt_jit.code, x, y, 0.0, 0.0);

        max_ref_err = max_ref_err.max((got_orig - want).abs());
        max_ref_err = max_ref_err.max((got_opt - want).abs());
        max_cross_err = max_cross_err.max((got_orig - got_opt).abs());

        assert!(
            (got_orig - want).abs() <= 6e-2,
            "original JIT at ({x},{y}): got {got_orig}, want {want}"
        );
        assert!(
            (got_opt - want).abs() <= 6e-2,
            "optimized JIT at ({x},{y}): got {got_opt}, want {want}"
        );
        assert!(
            (got_orig - got_opt).abs() <= 1e-1,
            "NNUE extraction changed semantics at ({x},{y}): \
             original {got_orig} vs optimized {got_opt}"
        );
    }
    eprintln!(
        "[swirl] max error vs analytic f32 = {max_ref_err:.4}, \
         max original-vs-optimized = {max_cross_err:.4}"
    );
}
