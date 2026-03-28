//! Cost model calibration from SIMD benchmarks.
//!
//! This tool measures the actual performance of SIMD operations on the current
//! CPU and generates a calibrated cost model for the e-graph optimizer.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-core --bin calibrate_costs
//! ```
//!
//! Generates: `~/.config/pixelflow/cost_model.toml`
//!
//! ## Methodology
//!
//! 1. Measures each operation's latency using simple timing loops
//! 2. Normalizes all costs relative to the fastest operation
//! 3. Detects platform features (FMA, fast reciprocal sqrt)
//! 4. Saves learned costs to TOML for optimizer consumption
//!
//! ## Example Output
//!
//! ```toml
//! # Learned cost model from SIMD benchmarks (AVX-512)
//! # Costs are 100x scaled for sub-nanosecond precision
//! add = 107
//! mul = 117
//! mul_add = 110  # ← FMA detected!
//! rsqrt = 379    # ← 24% faster than recip+sqrt
//! ```

use pixelflow_core::{Field, ManifoldCompat, PARALLELISM};
use std::path::PathBuf;
use std::time::Instant;

/// Number of warmup iterations before measurement.
const WARMUP_ITERS: usize = 100;

/// Number of measurement iterations for stable timing.
const MEASURE_ITERS: usize = 10_000;

/// Benchmark a manifold and return average nanoseconds per call.
fn bench_manifold<M>(name: &str, manifold: M) -> f64
where
    M: pixelflow_core::Manifold<(Field, Field, Field, Field), Output = Field>,
{
    let x = Field::sequential(1.5);
    let y = Field::from(2.5);
    let z = Field::from(0.5);
    let w = Field::from(0.0);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(manifold.eval_raw(std::hint::black_box(x), y, z, w));
    }

    // Measure
    let start = Instant::now();
    for _ in 0..MEASURE_ITERS {
        std::hint::black_box(manifold.eval_raw(std::hint::black_box(x), y, z, w));
    }
    let elapsed = start.elapsed();

    let ns_per_call = elapsed.as_nanos() as f64 / MEASURE_ITERS as f64;
    println!(
        "{:12} {:8.2} ns/call ({:6.2} ns/elem)",
        name,
        ns_per_call,
        ns_per_call / PARALLELISM as f64
    );
    ns_per_call
}

fn main() {
    println!("PixelFlow Cost Model Calibration");
    println!("=================================\n");

    println!("SIMD Backend: {}", backend_name());
    println!("Parallelism:  {} (Field width)", PARALLELISM);
    println!(
        "Iterations:   {} (after {} warmup)",
        MEASURE_ITERS, WARMUP_ITERS
    );
    println!();

    println!("Measuring operation latencies...\n");

    use pixelflow_core::{ManifoldExt, X, Y, Z};

    // Measure binary operations
    let add_ns = bench_manifold("add", X + Y);
    let sub_ns = bench_manifold("sub", X - Y);
    let mul_ns = bench_manifold("mul", X * Y);
    let div_ns = bench_manifold("div", X / Y);

    // Measure unary operations
    let neg_ns = bench_manifold("neg", X.neg());
    let sqrt_ns = bench_manifold("sqrt", X.sqrt());
    let abs_ns = bench_manifold("abs", X.abs());

    // Measure reciprocal (1/x)
    let recip_ns = bench_manifold("recip", 1.0f32 / X);

    // Measure rsqrt (1/sqrt(x)) - may be native instruction
    let rsqrt_ns = bench_manifold("rsqrt", 1.0f32 / X.sqrt());

    // Measure min/max
    let min_ns = bench_manifold("min", X.min(Y));
    let max_ns = bench_manifold("max", X.max(Y));

    // Measure FMA (x*y+z)
    let mul_add_ns = bench_manifold("mul_add", X * Y + Z);

    // Calculate separate mul+add for comparison
    let mul_plus_add_ns = mul_ns + add_ns;

    println!();
    println!("Platform Analysis");
    println!("-----------------");

    // Detect FMA
    let has_fma = mul_add_ns < (mul_plus_add_ns * 0.9);
    if has_fma {
        println!(
            "✓ FMA detected: mul_add ({:.2}ns) < mul+add ({:.2}ns)",
            mul_add_ns, mul_plus_add_ns
        );
    } else {
        println!(
            "✗ No FMA: mul_add ({:.2}ns) ≈ mul+add ({:.2}ns)",
            mul_add_ns, mul_plus_add_ns
        );
    }

    // Detect fast rsqrt
    let rsqrt_expected = recip_ns + sqrt_ns;
    let has_fast_rsqrt = rsqrt_ns < (rsqrt_expected * 0.7);
    if has_fast_rsqrt {
        println!(
            "✓ Fast rsqrt: rsqrt ({:.2}ns) < recip+sqrt ({:.2}ns)",
            rsqrt_ns, rsqrt_expected
        );
    } else {
        println!(
            "✗ No fast rsqrt: rsqrt ({:.2}ns) ≈ recip+sqrt ({:.2}ns)",
            rsqrt_ns, rsqrt_expected
        );
    }

    println!();

    // Normalize to smallest operation (typically add)
    let base = add_ns.min(mul_ns).min(sub_ns);
    println!("Normalizing costs (base = {:.2}ns)...\n", base);

    let model = CostModelData {
        add: normalize(add_ns, base),
        sub: normalize(sub_ns, base),
        mul: normalize(mul_ns, base),
        div: normalize(div_ns, base),
        neg: normalize(neg_ns, base),
        sqrt: normalize(sqrt_ns, base),
        recip: normalize(recip_ns, base),
        rsqrt: normalize(rsqrt_ns, base),
        abs: normalize(abs_ns, base),
        min: normalize(min_ns, base),
        max: normalize(max_ns, base),
        mul_add: normalize(mul_add_ns, base),
        depth_threshold: 32,
        depth_penalty: 100,
    };

    println!("Calibrated Cost Model");
    println!("---------------------");
    println!("add      = {}", model.add);
    println!("sub      = {}", model.sub);
    println!("mul      = {}", model.mul);
    println!("div      = {}", model.div);
    println!("neg      = {}", model.neg);
    println!("sqrt     = {}", model.sqrt);
    println!("recip    = {}", model.recip);
    println!("rsqrt    = {}", model.rsqrt);
    println!("abs      = {}", model.abs);
    println!("min      = {}", model.min);
    println!("max      = {}", model.max);
    println!(
        "mul_add  = {} {}",
        model.mul_add,
        if has_fma { "(FMA)" } else { "" }
    );
    println!();

    // Save to config directory
    let config_dir = get_config_dir();
    std::fs::create_dir_all(&config_dir).expect("Failed to create config directory");

    let output_path = config_dir.join("cost_model.toml");
    save_toml(&model, &output_path).expect("Failed to save cost model");

    println!("✓ Saved to: {}", output_path.display());
    println!();
    println!("The optimizer will automatically use this cost model.");
    println!("To force a specific model, set: PIXELFLOW_COST_MODEL=/path/to/model.toml");
}

/// Cost model data structure (mirrors pixelflow-search's CostModel).
struct CostModelData {
    add: usize,
    sub: usize,
    mul: usize,
    div: usize,
    neg: usize,
    sqrt: usize,
    recip: usize,
    rsqrt: usize,
    abs: usize,
    min: usize,
    max: usize,
    mul_add: usize,
    depth_threshold: usize,
    depth_penalty: usize,
}

/// Save cost model to TOML (simple format, no dependencies).
fn save_toml(model: &CostModelData, path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "# Learned cost model weights")?;
    writeln!(file, "# Generated from SIMD benchmark measurements")?;
    writeln!(file)?;
    writeln!(file, "# Operation costs (100x scaled for precision)")?;
    writeln!(
        file,
        "# Relative to fastest operation with 3-digit accuracy"
    )?;
    writeln!(file, "add = {}", model.add)?;
    writeln!(file, "sub = {}", model.sub)?;
    writeln!(file, "mul = {}", model.mul)?;
    writeln!(file, "div = {}", model.div)?;
    writeln!(file, "neg = {}", model.neg)?;
    writeln!(file, "sqrt = {}", model.sqrt)?;
    writeln!(file, "recip = {}", model.recip)?;
    writeln!(file, "rsqrt = {}", model.rsqrt)?;
    writeln!(file, "abs = {}", model.abs)?;
    writeln!(file, "min = {}", model.min)?;
    writeln!(file, "max = {}", model.max)?;
    writeln!(file, "mul_add = {}", model.mul_add)?;
    writeln!(file)?;
    writeln!(file, "# Depth penalty (compile-time optimization)")?;
    writeln!(file, "depth_threshold = {}", model.depth_threshold)?;
    writeln!(file, "depth_penalty = {}", model.depth_penalty)?;

    Ok(())
}

/// Normalize a measurement to an integer cost relative to base.
/// Uses 100x scaling for 3-digit precision (e.g., 3.78 → 378).
fn normalize(ns: f64, base: f64) -> usize {
    ((ns / base * 100.0).round() as usize).max(1)
}

/// Get the platform-specific config directory.
fn get_config_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config").join("pixelflow")
    } else if let Some(home) = std::env::var_os("USERPROFILE") {
        // Windows fallback
        PathBuf::from(home).join(".pixelflow")
    } else {
        PathBuf::from(".pixelflow")
    }
}

/// Get the SIMD backend name for display.
fn backend_name() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        "AVX-512"
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    {
        "AVX2"
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2",
        not(target_feature = "avx2"),
        not(target_feature = "avx512f")
    ))]
    {
        return "SSE2";
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        return "NEON";
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx512f"),
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        target_arch = "aarch64",
        target_arch = "arm"
    )))]
    {
        return "Scalar (no SIMD)";
    }
}
