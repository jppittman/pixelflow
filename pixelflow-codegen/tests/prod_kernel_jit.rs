//! End-to-end production-shaped DAG through optimization and native emission.
//!
//! This exercises the radial "swirl" kernel used by the psychedelic shader,
//! including the same runtime optimizer and the normal machine-code path.

use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::{
    compile_dag,
    executable::{Point4, TileSlice},
};
use pixelflow_ir::{ExprBuilder, ExprGraph, Kernel, LatticeShape, OpKind};
use pixelflow_search::runtime::optimize_runtime_dag;

fn build_swirl(freq: f32, amp: f32, bias: f32) -> ExprGraph {
    let mut b = ExprBuilder::new();
    let x = b.var(0);
    let y = b.var(1);
    let xx = b.binary(OpKind::Mul, x, x);
    let yy = b.binary(OpKind::Mul, y, y);
    let d = b.binary(OpKind::Add, xx, yy);
    let s = b.unary(OpKind::Sqrt, d);
    let freq_node = b.constant(freq);
    let sf = b.binary(OpKind::Mul, s, freq_node);
    let sn = b.unary(OpKind::Sin, sf);
    let amp_node = b.constant(amp);
    let prod = b.binary(OpKind::Mul, sn, amp_node);
    let bias_node = b.constant(bias);
    let out = b.binary(OpKind::Add, prod, bias_node);
    b.finish_one(out)
}

fn reference(x: f32, y: f32, freq: f32, amp: f32, bias: f32) -> f32 {
    (((x * x + y * y).sqrt()) * freq).sin() * amp + bias
}

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

#[test]
fn prod_swirl_kernel_through_optimizer_and_jit() {
    let (freq, amp, bias) = (3.0_f32, 0.5, 0.5);
    let kernel = Kernel::from_graph(build_swirl(freq, amp, bias));
    let original = compile_dag(kernel.root(), kernel.environment()).expect("JIT original");
    let optimized = optimize_runtime_dag(
        kernel.rooted(),
        kernel.environment(),
        LatticeShape::new([64, 64]),
    );
    let (optimized_root, optimized_env) = optimized
        .as_deref()
        .map(|(rooted, env)| (rooted, env))
        .unwrap_or((kernel.rooted(), kernel.environment()));
    let optimized = compile_dag(optimized_root.entry(), optimized_env).expect("JIT optimized");

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
    let mut max_ref_err = 0.0_f32;
    let mut max_cross_err = 0.0_f32;
    for chunk in coords.chunks(LANES) {
        let mut xs = [0.0f32; LANES];
        let mut ys = [0.0f32; LANES];
        for (i, &(x, y)) in chunk.iter().enumerate() {
            xs[i] = x;
            ys[i] = y;
        }
        let point = Point4::new(xs, ys, [0.0; LANES], [0.0; LANES]);
        let mut out_original = [0.0f32; LANES];
        let mut out_optimized = [0.0f32; LANES];
        unsafe {
            original.code.call_collapse(
                core::ptr::null(),
                TileSlice::single(out_original.as_mut_ptr()),
                point,
            );
            optimized.code.call_collapse(
                core::ptr::null(),
                TileSlice::single(out_optimized.as_mut_ptr()),
                point,
            );
        }
        for (i, &(x, y)) in chunk.iter().enumerate() {
            let want = reference(x, y, freq, amp, bias);
            let got_original = out_original[i];
            let got_optimized = out_optimized[i];
            max_ref_err = max_ref_err.max((got_original - want).abs());
            max_ref_err = max_ref_err.max((got_optimized - want).abs());
            max_cross_err = max_cross_err.max((got_original - got_optimized).abs());
            assert!((got_original - want).abs() <= 6e-2);
            assert!((got_optimized - want).abs() <= 6e-2);
            assert!((got_original - got_optimized).abs() <= 1e-1);
        }
    }
    eprintln!("[swirl] max reference error {max_ref_err:.4}, cross error {max_cross_err:.4}");
}
