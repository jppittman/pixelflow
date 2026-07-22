//! Benchmark: scanline JIT vs single-pixel JIT on the psychedelic shader.
//!
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_scanline_jit

use pixelflow_pipeline::jit_bench::{benchmark_jit_arena, nanos_now};
use pixelflow_pipeline::training::factored::parse_kernel_code_arena;

#[cfg(target_arch = "aarch64")]
use pixelflow_ir::ScanlineJitManifold;
#[cfg(target_arch = "aarch64")]
use pixelflow_ir::backend::emit::compile_arena_dag_scanline;

fn main() {
    // Load psychedelic expressions from file
    let content = std::fs::read_to_string("pixelflow-pipeline/data/psychedelic_full.jsonl")
        .expect("read psychedelic_full.jsonl");
    let first_line = content.lines().next().expect("empty file");
    let d: serde_json::Value = serde_json::from_str(first_line).expect("parse json");
    let code = d["expression"].as_str().expect("no expression field");
    let _name = d["name"].as_str().unwrap_or("?");

    let (arena, root) = parse_kernel_code_arena(code).expect("parse failed");

    println!("=== Psychedelic Red Channel (1920px scanline) ===\n");
    println!("Arena: {} nodes\n", arena.len());

    // Single-pixel JIT benchmark
    let single_result = benchmark_jit_arena(&arena, root).expect("single-pixel JIT failed");
    println!("  Single-pixel JIT:  {:.3}ns/eval", single_result.ns);

    // Scanline JIT benchmark
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;

        let scanline_result =
            compile_arena_dag_scanline(&arena, root).expect("scanline compile failed");
        let scanline_jit = ScanlineJitManifold::new(scanline_result.code);

        let width = 1920usize;
        let parallelism = 4; // NEON = 4 lanes
        let steps = width / parallelism;

        // Build input X values
        let xs: Vec<float32x4_t> = (0..steps)
            .map(|s| unsafe { vdupq_n_f32((s * parallelism) as f32) })
            .collect();
        let mut outputs = vec![unsafe { vdupq_n_f32(0.0) }; steps];

        let y = unsafe { vdupq_n_f32(540.0) };
        let z = unsafe { vdupq_n_f32(0.0) };
        let w = unsafe { vdupq_n_f32(0.0) };

        // Warmup
        for _ in 0..100 {
            unsafe {
                scanline_jit.eval_scanline(&xs, y, z, w, &mut outputs);
            }
        }

        // Benchmark
        let samples = 50;
        let mut times = vec![0u64; samples];
        for t in &mut times {
            let start = nanos_now();
            unsafe {
                scanline_jit.eval_scanline(&xs, y, z, w, &mut outputs);
            }
            *t = nanos_now() - start;
        }
        times.sort();
        let scanline_ns = times[samples / 2] as f64 / width as f64;
        println!("  Scanline JIT:      {:.3}ns/pixel", scanline_ns);
        println!(
            "  Speedup:           {:.2}x vs single-pixel",
            single_result.ns / scanline_ns
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("  (scanline JIT only supported on aarch64)");
}
