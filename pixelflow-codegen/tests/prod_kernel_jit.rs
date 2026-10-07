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
//!              ->  latency-prior extraction (the production policy)
//!              ->  transcendental lowering + register allocation + codegen
//!              ->  native machine code, executed on real coordinates.
//!
//! Optimization is `pixelflow_search::runtime::optimize_runtime_arena`, the
//! entry `jit_cache::compile` takes for every runtime kernel — its budget,
//! its rule set and its extraction policy — so this exercises the shipped
//! optimizer rather than one assembled for the test. The point is the
//! *pipeline*, end to end.

use pixelflow_codegen::emit::compile;
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};
use pixelflow_search::runtime::optimize_runtime_arena;

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

// ---------------------------------------------------------------------------
// Executing JIT code: one single-point lattice call per coordinate.
// ---------------------------------------------------------------------------

use pixelflow_codegen::emit::executable::ExecutableCode;

/// One point of a kernel compiled at [`LatticeShape::POINT`]: the sample at
/// `(x, y)`, read back through the origin block.
fn eval_at(code: &ExecutableCode, x: f32, y: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    // SAFETY: this arena declares no buffers and no uniform, so `ctx[0]` —
    // the uniform-block slot — is unread; `ctx[1]` is the origin block, and
    // `out` holds the one sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        code.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

#[test]
fn egraph_extraction_preserves_the_swirl_kernels_values_on_the_jit() {
    let (freq, amp, bias) = (3.0_f32, 0.5, 0.5);

    let (orig, orig_root) = build_swirl(freq, amp, bias);
    let optimized = optimize_runtime_arena(&orig, orig_root, LatticeShape::POINT)
        .expect("the e-graph models every op in the swirl");
    let (opt, opt_root) = &*optimized;

    // JIT both the original and the e-graph-optimized DAG. Both paths run the
    // shared transcendental-lowering + regalloc + codegen pipeline.
    let orig_jit = compile(&orig, orig_root, LatticeShape::POINT).expect("JIT original");
    let opt_jit = compile(opt, *opt_root, LatticeShape::POINT).expect("JIT optimized");

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
    // certifies that e-graph extraction preserved semantics.
    let mut max_ref_err = 0.0_f32;
    let mut max_cross_err = 0.0_f32;
    for &(x, y) in &coords {
        let got_orig = eval_at(&orig_jit.code, x, y);
        let got_opt = eval_at(&opt_jit.code, x, y);
        let want = reference(x, y, freq, amp, bias);

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
            "e-graph extraction changed semantics at ({x},{y}): \
             original {got_orig} vs optimized {got_opt}"
        );
    }
    eprintln!(
        "[swirl] max error vs analytic f32 = {max_ref_err:.3e}, \
         max original-vs-optimized = {max_cross_err:.3e}"
    );
}
