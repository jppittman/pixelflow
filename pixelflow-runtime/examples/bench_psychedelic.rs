//! Head-to-head: LLVM vs NNUE+LLVM vs JIT on the psychedelic shader.
//!
//! cargo run --release -p pixelflow-runtime --example bench_psychedelic

use pixelflow_compiler::{kernel, kernel_jit, kernel_raw};
use pixelflow_core::{Field, Manifold, PARALLELISM};

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
}

fn nanos_now() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        mach_absolute_time()
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

// LLVM only — no e-graph optimization
kernel_raw!(struct PsychRaw = || Field -> Field {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = W + 1.3;
    let r_sq = x * x + y * y;
    let radial = (r_sq - 0.7).abs();
    let swirl_scale = (1.0 - radial) * 5.0;
    let vx = x * swirl_scale;
    let vy = y * swirl_scale;
    let phase = time * 0.5;
    let sin_w03 = (time * 0.3).sin();
    let sin_w20 = (time * 2.0).sin();
    let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
    let pulse = 1.0 + sin_w20 * 0.1;
    let radial_factor = (radial * -4.0 * pulse).exp();
    let y_factor_r = (y + sin_w03 * 0.2).exp();
    let raw_r = y_factor_r * radial_factor / swirl;
    let red = (raw_r / (raw_r.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
    let raw_g = y_factor_g * radial_factor / swirl;
    let green = (raw_g / (raw_g.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
    let raw_b = y_factor_b * radial_factor / swirl;
    let blue = (raw_b / (raw_b.abs() + 1.0) + 1.0) * 0.5;
    red + green + blue
});

// NNUE + LLVM — e-graph saturation + neural extraction + DAG CSE + LLVM backend
kernel!(struct PsychOpt = || Field -> Field {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = W + 1.3;
    let r_sq = x * x + y * y;
    let radial = (r_sq - 0.7).abs();
    let swirl_scale = (1.0 - radial) * 5.0;
    let vx = x * swirl_scale;
    let vy = y * swirl_scale;
    let phase = time * 0.5;
    let sin_w03 = (time * 0.3).sin();
    let sin_w20 = (time * 2.0).sin();
    let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
    let pulse = 1.0 + sin_w20 * 0.1;
    let radial_factor = (radial * -4.0 * pulse).exp();
    let y_factor_r = (y + sin_w03 * 0.2).exp();
    let raw_r = y_factor_r * radial_factor / swirl;
    let red = (raw_r / (raw_r.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
    let raw_g = y_factor_g * radial_factor / swirl;
    let green = (raw_g / (raw_g.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
    let raw_b = y_factor_b * radial_factor / swirl;
    let blue = (raw_b / (raw_b.abs() + 1.0) + 1.0) * 0.5;
    red + green + blue
});

#[inline(never)]
fn bench_scanline<M: Manifold<Output = Field>>(shader: &M) -> f64 {
    let width = 1920usize;
    let height = 1080usize;
    let steps_x = width / PARALLELISM;
    let z = Field::from(0.0f32);
    let w = Field::from(0.0f32);

    // Warmup: full frame
    for py in (0..height).step_by(108) {
        let y = Field::from(py as f32);
        for step in 0..steps_x {
            let x = Field::sequential((step * PARALLELISM) as f32);
            std::hint::black_box(shader.eval((x, y, z, w)));
        }
    }

    // Benchmark: 10 scanlines at different Y positions
    // This prevents LLVM from hoisting Y-dependent computations
    let scanlines = 10usize;
    let total_pixels = width * scanlines;
    let samples = 50;
    let mut times = vec![0u64; samples];
    for t in &mut times {
        let start = nanos_now();
        for sy in 0..scanlines {
            let y = Field::from((sy * 108) as f32);
            for step in 0..steps_x {
                let x = Field::sequential((step * PARALLELISM) as f32);
                std::hint::black_box(shader.eval((x, y, z, w)));
            }
        }
        *t = nanos_now() - start;
    }
    times.sort();
    times[samples / 2] as f64 / total_pixels as f64
}

fn main() {
    let raw = PsychRaw {};
    let opt = PsychOpt {};
    let jit = kernel_jit!(|| {
        let scale = 2.0 / 1080.0;
        let x = (X - 960.0) * scale;
        let y = (540.0 - Y) * scale;
        let time = W + 1.3;
        let r_sq = x * x + y * y;
        let radial = (r_sq - 0.7).abs();
        let swirl_scale = (1.0 - radial) * 5.0;
        let vx = x * swirl_scale;
        let vy = y * swirl_scale;
        let phase = time * 0.5;
        let sin_w03 = (time * 0.3).sin();
        let sin_w20 = (time * 2.0).sin();
        let swirl =
            ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
        let pulse = 1.0 + sin_w20 * 0.1;
        let radial_factor = (radial * -4.0 * pulse).exp();
        let y_factor_r = (y + sin_w03 * 0.2).exp();
        let raw_r = y_factor_r * radial_factor / swirl;
        let red = (raw_r / (raw_r.abs() + 1.0) + 1.0) * 0.5;
        let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
        let raw_g = y_factor_g * radial_factor / swirl;
        let green = (raw_g / (raw_g.abs() + 1.0) + 1.0) * 0.5;
        let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
        let raw_b = y_factor_b * radial_factor / swirl;
        let blue = (raw_b / (raw_b.abs() + 1.0) + 1.0) * 0.5;
        red + green + blue
    });

    println!(
        "=== Psychedelic Shader (3ch, 1920px scanline, {} SIMD lanes) ===\n",
        PARALLELISM
    );

    let raw_ns = bench_scanline(&raw);
    let opt_ns = bench_scanline(&opt);
    let jit_ns = bench_scanline(&jit);

    println!("  LLVM only (kernel_raw!):  {:.3}ns/pixel", raw_ns);
    println!("  NNUE + LLVM (kernel!):    {:.3}ns/pixel", opt_ns);
    println!("  JIT (kernel_jit!):        {:.3}ns/pixel", jit_ns);
    println!();
    println!(
        "  NNUE+LLVM vs LLVM: {:.1}%",
        (opt_ns / raw_ns - 1.0) * 100.0
    );
    println!(
        "  JIT vs LLVM:       {:.1}%",
        (jit_ns / raw_ns - 1.0) * 100.0
    );
}
