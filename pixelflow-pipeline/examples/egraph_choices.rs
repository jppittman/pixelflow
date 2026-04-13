//! Benchmark realistic e-graph rewrite choices
//!
//! These are the kinds of choices an e-graph optimizer actually makes.
//! Run with: cargo run -p pixelflow-ml --example egraph_choices --release

use std::time::Instant;

// ============================================================================
// Choice 1: x * 2 vs x + x
// ============================================================================

#[inline(never)]
fn mul_by_2(x: f32) -> f32 {
    x * 2.0
}

#[inline(never)]
fn add_self(x: f32) -> f32 {
    x + x
}

// ============================================================================
// Choice 2: a*b + a*c vs a*(b+c) (distributive law)
// ============================================================================

#[inline(never)]
fn distributed(a: f32, b: f32, c: f32) -> f32 {
    a * b + a * c // 2 muls, 1 add
}

#[inline(never)]
fn factored(a: f32, b: f32, c: f32) -> f32 {
    a * (b + c) // 1 mul, 1 add
}

// ============================================================================
// Choice 3: x / y vs x * (1/y) - division strength reduction
// ============================================================================

#[inline(never)]
fn div_direct(x: f32, y: f32) -> f32 {
    x / y
}

#[inline(never)]
fn mul_recip(x: f32, y: f32) -> f32 {
    x * (1.0 / y)
}

// ============================================================================
// Choice 4: sqrt(x) * sqrt(x) vs x (simplification)
// ============================================================================

#[inline(never)]
fn sqrt_squared(x: f32) -> f32 {
    let s = x.sqrt();
    s * s
}

#[inline(never)]
fn identity(x: f32) -> f32 {
    x
}

// ============================================================================
// Choice 5: (a*b) + c vs fma(a, b, c)
// ============================================================================

#[inline(never)]
fn mul_then_add(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

#[inline(never)]
fn fma_op(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

// ============================================================================
// Choice 6: 1/sqrt(x) vs rsqrt approximation (if available)
// ============================================================================

#[inline(never)]
fn div_sqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

// Fast rsqrt approximation (Newton-Raphson style)
#[inline(never)]
fn fast_rsqrt(x: f32) -> f32 {
    // This is what rsqrt instructions do internally
    let half_x = 0.5 * x;
    let mut y = f32::from_bits(0x5f3759df - (x.to_bits() >> 1));
    y = y * (1.5 - half_x * y * y); // One Newton-Raphson iteration
    y
}

fn benchmark<F>(name: &str, iterations: u64, mut f: F) -> (f64, f32)
where
    F: FnMut() -> f32,
{
    // Warmup
    let mut sink = 0.0f32;
    for _ in 0..10000 {
        sink += f();
    }
    std::hint::black_box(sink);

    // Benchmark
    let start = Instant::now();
    let mut result = 0.0f32;
    for _ in 0..iterations {
        result += f();
    }
    let elapsed = start.elapsed();
    std::hint::black_box(result);

    let ns_per_iter = elapsed.as_nanos() as f64 / iterations as f64;
    (ns_per_iter, result)
}

fn compare(name: &str, iterations: u64, opt_a: (&str, f64, f32), opt_b: (&str, f64, f32)) {
    println!("{}", name);
    println!(
        "  {}: {:.2} ns/iter (result: {:.6})",
        opt_a.0, opt_a.1, opt_a.2
    );
    println!(
        "  {}: {:.2} ns/iter (result: {:.6})",
        opt_b.0, opt_b.1, opt_b.2
    );

    let speedup = (opt_a.1 - opt_b.1) / opt_a.1 * 100.0;
    let winner = if opt_b.1 < opt_a.1 { opt_b.0 } else { opt_a.0 };
    println!("  Winner: {} ({:.1}% faster)\n", winner, speedup.abs());
}

fn main() {
    let iterations = 100_000_000u64;

    // Varying inputs to prevent constant folding
    let x = std::hint::black_box(3.14159f32);
    let y = std::hint::black_box(2.71828f32);
    let a = std::hint::black_box(1.41421f32);
    let b = std::hint::black_box(1.73205f32);
    let c = std::hint::black_box(2.23607f32);

    println!("=== E-Graph Rewrite Choice Benchmarks ===\n");
    println!("Testing algebraically equivalent expressions");
    println!("that an e-graph optimizer might choose between.\n");

    // Choice 1
    let r1 = benchmark("x*2", iterations, || mul_by_2(x));
    let r2 = benchmark("x+x", iterations, || add_self(x));
    compare(
        "Choice 1: x*2 vs x+x",
        iterations,
        ("x*2", r1.0, r1.1),
        ("x+x", r2.0, r2.1),
    );

    // Choice 2
    let r1 = benchmark("a*b + a*c", iterations, || distributed(a, b, c));
    let r2 = benchmark("a*(b+c)", iterations, || factored(a, b, c));
    compare(
        "Choice 2: Distributive a*b+a*c vs a*(b+c)",
        iterations,
        ("a*b + a*c", r1.0, r1.1),
        ("a*(b+c)", r2.0, r2.1),
    );

    // Choice 3
    let r1 = benchmark("x/y", iterations, || div_direct(x, y));
    let r2 = benchmark("x*(1/y)", iterations, || mul_recip(x, y));
    compare(
        "Choice 3: x/y vs x*(1/y)",
        iterations,
        ("x/y", r1.0, r1.1),
        ("x*(1/y)", r2.0, r2.1),
    );

    // Choice 4
    let r1 = benchmark("sqrt(x)*sqrt(x)", iterations, || sqrt_squared(x.abs()));
    let r2 = benchmark("x", iterations, || identity(x.abs()));
    compare(
        "Choice 4: sqrt(x)*sqrt(x) vs x",
        iterations,
        ("sqrt*sqrt", r1.0, r1.1),
        ("identity", r2.0, r2.1),
    );

    // Choice 5
    let r1 = benchmark("a*b + c", iterations, || mul_then_add(a, b, c));
    let r2 = benchmark("fma(a,b,c)", iterations, || fma_op(a, b, c));
    compare(
        "Choice 5: a*b+c vs fma(a,b,c)",
        iterations,
        ("mul+add", r1.0, r1.1),
        ("fma", r2.0, r2.1),
    );

    // Choice 6
    let r1 = benchmark("1/sqrt(x)", iterations / 10, || div_sqrt(x.abs()));
    let r2 = benchmark("fast_rsqrt", iterations / 10, || fast_rsqrt(x.abs()));
    compare(
        "Choice 6: 1/sqrt(x) vs fast_rsqrt",
        iterations / 10,
        ("1/sqrt", r1.0, r1.1),
        ("fast_rsqrt", r2.0, r2.1),
    );

    println!("=== Summary ===");
    println!("These benchmarks show what an HCE-guided e-graph should prefer.");
    println!("Key insight: operation count matters less than operation TYPE.");
}
