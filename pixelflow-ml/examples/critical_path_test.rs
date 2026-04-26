//! Test whether critical_path predicts actual performance
//!
//! Uses expensive operations (sqrt, div) where latency matters.
//! Run with: cargo run -p pixelflow-ml --example critical_path_test --release

use std::time::Instant;

const ITERATIONS: u64 = 10_000_000;

// ============================================================================
// Test 1: Wide vs Deep with sqrt (expensive: ~13 cycles latency)
// ============================================================================

/// Wide: sqrt(a) + sqrt(b) + sqrt(c) + sqrt(d)
/// All 4 sqrts can run in parallel, then 3 adds
/// Critical path: sqrt(13) + add(2) + add(2) + add(2) = 19 cycles
#[inline(never)]
fn wide_sqrt_sum(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let sa = a.sqrt();
    let sb = b.sqrt();
    let sc = c.sqrt();
    let sd = d.sqrt();
    sa + sb + sc + sd
}

/// Deep: sqrt(sqrt(sqrt(a) + b) + c) + d
/// Each sqrt depends on previous result
/// Critical path: sqrt(13) + add(2) + sqrt(13) + add(2) + sqrt(13) + add(2) = 45 cycles
#[inline(never)]
fn deep_sqrt_chain(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t1 = a.sqrt() + b;
    let t2 = t1.sqrt() + c;
    t2.sqrt() + d
}

// ============================================================================
// Test 2: Wide vs Deep with div (expensive: ~9 cycles latency)
// ============================================================================

/// Wide: (a/b) + (c/d) - both divisions can run in parallel
/// Critical path: div(9) + add(2) = 11 cycles
#[inline(never)]
fn wide_div(a: f32, b: f32, c: f32, d: f32) -> f32 {
    (a / b) + (c / d)
}

/// Deep: (a / b) / c / d - all divisions sequential
/// Critical path: div(9) + div(9) + div(9) = 27 cycles
#[inline(never)]
fn deep_div(a: f32, b: f32, c: f32, d: f32) -> f32 {
    ((a / b) / c) / d
}

// ============================================================================
// Test 3: Mixed operations - realistic expression
// ============================================================================

/// Wide form: (a*b + c*d) / (e + f)
/// muls parallel, adds, one div at end
#[inline(never)]
fn expr_wide(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> f32 {
    let ab = a * b;
    let cd = c * d;
    let ef = e + f;
    (ab + cd) / ef
}

/// Equivalent but more sequential: a*b/e + c*d/f (same math if e=f)
/// But forces more dependencies
#[inline(never)]
fn expr_sequential(a: f32, b: f32, c: f32, d: f32, e: f32, _f: f32) -> f32 {
    let t1 = a * b;
    let t2 = t1 / e;
    let t3 = c * d;
    let t4 = t3 / e;
    t2 + t4
}

fn benchmark<F>(_name: &str, mut f: F) -> f64
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
    for _ in 0..ITERATIONS {
        result += f();
    }
    let elapsed = start.elapsed();
    std::hint::black_box(result);

    elapsed.as_nanos() as f64 / ITERATIONS as f64
}

fn main() {
    let a = std::hint::black_box(2.5f32);
    let b = std::hint::black_box(1.5f32);
    let c = std::hint::black_box(3.5f32);
    let d = std::hint::black_box(0.5f32);
    let e = std::hint::black_box(1.2f32);
    let f = std::hint::black_box(1.2f32); // Same as e for mathematical equivalence

    println!("=== Critical Path Prediction Test ===\n");
    println!("Testing whether critical_path predicts actual performance.\n");

    // Test 1: sqrt
    println!("Test 1: Wide vs Deep with sqrt");
    println!("  HCE critical_path: wide=19, deep=45 (wide 58% better)");
    let wide_ns = benchmark("wide_sqrt", || wide_sqrt_sum(a, b, c, d));
    let deep_ns = benchmark("deep_sqrt", || deep_sqrt_chain(a, b, c, d));
    let actual_improvement = (deep_ns - wide_ns) / deep_ns * 100.0;
    println!("  wide: {:.2} ns/iter", wide_ns);
    println!("  deep: {:.2} ns/iter", deep_ns);
    println!(
        "  Actual: wide is {:.1}% {}\n",
        actual_improvement.abs(),
        if actual_improvement > 0.0 {
            "faster (CONFIRMED)"
        } else {
            "slower (WRONG!)"
        }
    );

    // Test 2: div
    println!("Test 2: Wide vs Deep with div");
    println!("  HCE critical_path: wide=11, deep=27 (wide 59% better)");
    let wide_ns = benchmark("wide_div", || wide_div(a, b, c, d));
    let deep_ns = benchmark("deep_div", || deep_div(a, b, c, d));
    let actual_improvement = (deep_ns - wide_ns) / deep_ns * 100.0;
    println!("  wide: {:.2} ns/iter", wide_ns);
    println!("  deep: {:.2} ns/iter", deep_ns);
    println!(
        "  Actual: wide is {:.1}% {}\n",
        actual_improvement.abs(),
        if actual_improvement > 0.0 {
            "faster (CONFIRMED)"
        } else {
            "slower (WRONG!)"
        }
    );

    // Test 3: realistic
    println!("Test 3: Realistic expression");
    let wide_ns = benchmark("expr_wide", || expr_wide(a, b, c, d, e, f));
    let seq_ns = benchmark("expr_seq", || expr_sequential(a, b, c, d, e, f));
    let actual_improvement = (seq_ns - wide_ns) / seq_ns * 100.0;
    println!("  wide: {:.2} ns/iter", wide_ns);
    println!("  sequential: {:.2} ns/iter", seq_ns);
    println!(
        "  Actual: wide is {:.1}% {}\n",
        actual_improvement.abs(),
        if actual_improvement > 0.0 {
            "faster (CONFIRMED)"
        } else {
            "slower (WRONG!)"
        }
    );

    println!("=== Summary ===");
    println!("Critical path analysis IS valuable when:");
    println!("  1. Operations have high latency (sqrt ~13cy, div ~9cy)");
    println!("  2. The alternative forms have genuinely different dependency graphs");
    println!("  3. LLVM doesn't reorder operations to hide latency");
}
