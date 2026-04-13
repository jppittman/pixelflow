//! Benchmark to verify ILP features predict actual performance
//!
//! Run with: cargo run -p pixelflow-ml --example ilp_benchmark --release

use std::time::Instant;

/// Wide expression: (a+b)+(c+d) - two parallel adds, then one final add
#[inline(never)]
fn wide_expr(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let left = a + b;
    let right = c + d;
    left + right
}

/// Deep expression: ((a+b)+c)+d - three sequential adds
#[inline(never)]
fn deep_expr(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t1 = a + b;
    let t2 = t1 + c;
    t2 + d
}

/// More complex wide: ((a+b)+(c+d)) + ((e+f)+(g+h))
#[inline(never)]
fn wide_8(vals: &[f32; 8]) -> f32 {
    let l1 = vals[0] + vals[1];
    let l2 = vals[2] + vals[3];
    let r1 = vals[4] + vals[5];
    let r2 = vals[6] + vals[7];
    let left = l1 + l2;
    let right = r1 + r2;
    left + right
}

/// Deep 8: (((((((a+b)+c)+d)+e)+f)+g)+h)
#[inline(never)]
fn deep_8(vals: &[f32; 8]) -> f32 {
    let mut acc = vals[0] + vals[1];
    acc = acc + vals[2];
    acc = acc + vals[3];
    acc = acc + vals[4];
    acc = acc + vals[5];
    acc = acc + vals[6];
    acc + vals[7]
}

/// Wide with expensive ops: (sqrt(a)*sqrt(b)) + (sqrt(c)*sqrt(d))
#[inline(never)]
fn wide_sqrt(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let left = a.sqrt() * b.sqrt();
    let right = c.sqrt() * d.sqrt();
    left + right
}

/// Deep with expensive ops: sqrt(sqrt(sqrt(a)*b)*c)*d
#[inline(never)]
fn deep_sqrt(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t1 = a.sqrt() * b;
    let t2 = t1.sqrt() * c;
    t2.sqrt() * d
}

fn benchmark<F>(name: &str, iterations: u64, mut f: F) -> f64
where
    F: FnMut() -> f32,
{
    // Warmup
    let mut sink = 0.0f32;
    for _ in 0..1000 {
        sink += f();
    }
    std::hint::black_box(sink);

    // Benchmark
    let start = Instant::now();
    let mut result = 0.0f32;
    for _ in 0..iterations {
        result += f();
    }
    std::hint::black_box(result);
    let elapsed = start.elapsed();

    let ns_per_iter = elapsed.as_nanos() as f64 / iterations as f64;
    println!("  {}: {:.2} ns/iter", name, ns_per_iter);
    ns_per_iter
}

fn main() {
    let iterations = 100_000_000u64;

    // Use volatile-ish values to prevent constant folding
    let a = std::hint::black_box(1.1f32);
    let b = std::hint::black_box(2.2f32);
    let c = std::hint::black_box(3.3f32);
    let d = std::hint::black_box(4.4f32);

    let vals8: [f32; 8] = std::hint::black_box([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]);

    println!("=== ILP Performance Benchmark ===\n");
    println!("Iterations: {}\n", iterations);

    // Test 1: Simple 4-value addition
    println!("Test 1: 4-value addition (3 adds)");
    println!("  HCE predicts: wide=24, deep=26 (wide 8% cheaper)");
    let wide_time = benchmark("wide (a+b)+(c+d)", iterations, || wide_expr(a, b, c, d));
    let deep_time = benchmark("deep ((a+b)+c)+d", iterations, || deep_expr(a, b, c, d));
    let speedup = (deep_time - wide_time) / deep_time * 100.0;
    println!(
        "  Actual: wide is {:.1}% {} than deep\n",
        speedup.abs(),
        if speedup > 0.0 { "FASTER" } else { "SLOWER" }
    );

    // Test 2: 8-value addition
    println!("Test 2: 8-value addition (7 adds)");
    println!("  HCE predicts: wide has shorter critical path");
    let wide8_time = benchmark("wide balanced tree", iterations, || wide_8(&vals8));
    let deep8_time = benchmark("deep linear chain", iterations, || deep_8(&vals8));
    let speedup8 = (deep8_time - wide8_time) / deep8_time * 100.0;
    println!(
        "  Actual: wide is {:.1}% {} than deep\n",
        speedup8.abs(),
        if speedup8 > 0.0 { "FASTER" } else { "SLOWER" }
    );

    // Test 3: With expensive operations (sqrt)
    println!("Test 3: With sqrt operations");
    println!("  Wide: sqrt(a)*sqrt(b) + sqrt(c)*sqrt(d) - 4 sqrts in parallel");
    println!("  Deep: nested sqrts in chain - 3 sqrts sequential");
    let wide_sqrt_time = benchmark("wide parallel sqrts", iterations / 10, || {
        wide_sqrt(a, b, c, d)
    });
    let deep_sqrt_time = benchmark("deep sequential sqrts", iterations / 10, || {
        deep_sqrt(a, b, c, d)
    });
    let speedup_sqrt = (deep_sqrt_time - wide_sqrt_time) / deep_sqrt_time * 100.0;
    println!(
        "  Actual: wide is {:.1}% {} than deep\n",
        speedup_sqrt.abs(),
        if speedup_sqrt > 0.0 {
            "FASTER"
        } else {
            "SLOWER"
        }
    );

    // Summary
    println!("=== Summary ===");
    println!(
        "The ILP hypothesis (wide is faster due to parallelism) is {}",
        if speedup > 0.0 && speedup8 > 0.0 {
            "CONFIRMED"
        } else {
            "NOT CONFIRMED"
        }
    );

    if speedup > 0.0 {
        println!("\nHCE predicted 8% improvement, actual was {:.1}%", speedup);
        println!(
            "Prediction accuracy: {:.0}%",
            100.0 - ((speedup - 8.0).abs() / 8.0 * 100.0).min(100.0)
        );
    }
}
