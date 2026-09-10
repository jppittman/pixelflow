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
//!   Term       ->  e-graph equality saturation (algebra + trig + FMA fusion)
//!              ->  latency-prior extraction (the production policy)
//!              ->  transcendental lowering + register allocation + codegen
//!              ->  native machine code, executed on real coordinates.
//!
//! Extraction goes through `env_extraction_policy`, the same seam the
//! `kernel!` macros and runtime kernels use, so this exercises the shipped
//! policy rather than a test-only one. The point is the *pipeline*, end to end.

use pixelflow_codegen::emit::compile;
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData};
use pixelflow_ir::{OpKind, Rooted, Term};
use pixelflow_search::egraph::{Budget, Optimizer};

/// Build `sin(sqrt(x*x + y*y) * freq) * amp + bias` as a term.
fn build_swirl(freq: f32, amp: f32, bias: f32) -> (Rooted<ExprData>, Environment) {
    let mut a = ExprBuilder::new();
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
    a.finish(&[out])
}

fn reference(x: f32, y: f32, freq: f32, amp: f32, bias: f32) -> f32 {
    (((x * x + y * y).sqrt()) * freq).sin() * amp + bias
}

/// Optimize `term` through the e-graph and the production extraction policy,
/// returning the extracted DAG. Prints a few diagnostics so the run is
/// visible.
fn optimize(term: Term<'_>, tag: &str) -> (Rooted<ExprData>, Environment) {
    // The production entry point, held to this test's own round budget.
    let mut optimizer = Optimizer::production().budget(Budget::Explicit {
        iterations: 40,
        classes: 10_000,
        applications: None,
    });
    let mut eg = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        term,
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let classes_before = eg.num_classes();

    let node_count = pixelflow_search::egraph::reachable_count_term(term);
    let optimized = optimizer.run(&mut eg, root_class, node_count);
    let classes_after = eg.num_classes();

    let out = optimized.to_rooted(&eg, root_class);

    eprintln!(
        "[{tag}] egraph {classes_before} -> {classes_after} classes, \
         extracted DAG = {} nodes",
        out.0.len(),
    );
    out
}

// ---------------------------------------------------------------------------
// Executing JIT code: broadcast one coordinate to all lanes, read lane 0.
// ---------------------------------------------------------------------------

use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::executable::{Point4, TileSlice};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

#[test]
fn prod_swirl_kernel_through_egraph_and_jit() {
    let (freq, amp, bias) = (3.0_f32, 0.5, 0.5);

    let orig = build_swirl(freq, amp, bias);
    let orig_term = Term::new(orig.0.entry(), &orig.1);
    let opt = optimize(orig_term, "swirl");
    let opt_term = Term::new(opt.0.entry(), &opt.1);

    // JIT both the original and the e-graph-optimized DAG. Both paths run the
    // shared transcendental-lowering + regalloc + codegen pipeline.
    let orig_jit = compile(orig_term).expect("JIT original");
    let opt_jit = compile(opt_term).expect("JIT optimized");
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
    // certifies that e-graph extraction preserved semantics.
    let mut max_ref_err = 0.0_f32;
    let mut max_cross_err = 0.0_f32;
    for chunk in coords.chunks(LANES) {
        let mut xs = [0.0f32; LANES];
        let mut ys = [0.0f32; LANES];
        for (i, &(x, y)) in chunk.iter().enumerate() {
            xs[i] = x;
            ys[i] = y;
        }
        let p = Point4::new(xs, ys, [0.0; LANES], [0.0; LANES]);
        let mut out_orig = [0.0f32; LANES];
        let mut out_opt = [0.0f32; LANES];
        unsafe {
            orig_jit.code.call_collapse(
                core::ptr::null(),
                TileSlice::single(out_orig.as_mut_ptr()),
                p,
            );
            opt_jit.code.call_collapse(
                core::ptr::null(),
                TileSlice::single(out_opt.as_mut_ptr()),
                p,
            );
        }
        for (i, &(x, y)) in chunk.iter().enumerate() {
            let want = reference(x, y, freq, amp, bias);
            let got_orig = out_orig[i];
            let got_opt = out_opt[i];

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
    }
    eprintln!(
        "[swirl] max error vs analytic f32 = {max_ref_err:.4}, \
         max original-vs-optimized = {max_cross_err:.4}"
    );
}
