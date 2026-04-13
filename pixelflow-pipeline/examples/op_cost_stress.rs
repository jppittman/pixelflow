//! Stress test to measure actual operation costs
//!
//! Uses dependent chains to force serial execution and measure true latency.
//! Run with: cargo run -p pixelflow-ml --example op_cost_stress --release

use std::time::Instant;

const CHAIN_LEN: usize = 1000;
const ITERATIONS: u64 = 100_000;

/// Measure latency of an operation by chaining it
fn measure_chain<F>(name: &str, init: f32, mut op: F) -> f64
where
    F: FnMut(f32) -> f32,
{
    // Warmup
    let mut x = init;
    for _ in 0..CHAIN_LEN {
        x = op(x);
    }
    std::hint::black_box(x);

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut x = std::hint::black_box(init);
        for _ in 0..CHAIN_LEN {
            x = op(x);
        }
        std::hint::black_box(x);
    }
    let elapsed = start.elapsed();

    let total_ops = ITERATIONS * CHAIN_LEN as u64;
    let ns_per_op = elapsed.as_nanos() as f64 / total_ops as f64;

    // Estimate cycles at 3 GHz
    let cycles = ns_per_op * 3.0;
    println!(
        "  {:<15}: {:.2} ns/op (~{:.1} cycles)",
        name, ns_per_op, cycles
    );
    ns_per_op
}

fn main() {
    println!("=== Operation Latency Measurement ===\n");
    println!("Chain length: {}, Iterations: {}", CHAIN_LEN, ITERATIONS);
    println!("Measuring dependent chains to get true latency.\n");

    println!("Basic Operations:");
    let add_ns = measure_chain("add", 1.0, |x| x + 0.001);
    let sub_ns = measure_chain("sub", 1000.0, |x| x - 0.001);
    let mul_ns = measure_chain("mul", 1.0, |x| x * 1.0001);
    let div_ns = measure_chain("div", 1000.0, |x| x / 1.0001);
    let _neg_ns = measure_chain("neg", 1.0, |x| -x + 0.001); // +0.001 to prevent oscillation

    println!("\nExpensive Operations:");
    let sqrt_ns = measure_chain("sqrt", 1000.0, |x| x.sqrt() + 1.0);
    let _rsqrt_ns = measure_chain("1/sqrt", 1000.0, |x| 1.0 / x.sqrt() + 1.0);

    println!("\nFused Operations:");
    let _fma_ns = measure_chain("fma", 1.0, |x| x.mul_add(1.0001, 0.0001));
    let mulsub_ns = measure_chain("mul+add", 1.0, |x| x * 1.0001 + 0.0001);

    println!("\n=== Relative Costs (add = 1.0) ===");
    println!("  add:  1.0");
    println!("  sub:  {:.1}", sub_ns / add_ns);
    println!("  mul:  {:.1}", mul_ns / add_ns);
    println!("  div:  {:.1}", div_ns / add_ns);
    println!("  sqrt: {:.1}", sqrt_ns / add_ns);
    println!("  mul+add: {:.1}", mulsub_ns / add_ns);

    println!("\n=== HCE Weight Comparison ===");
    println!("Current HCE weights assume:");
    println!("  add=4, mul=5, div=15, sqrt=15");
    println!("\nMeasured relative costs:");
    let add_base = 4.0;
    println!("  add:  4 (baseline)");
    println!("  mul:  {:.0}", (mul_ns / add_ns) * add_base);
    println!("  div:  {:.0}", (div_ns / add_ns) * add_base);
    println!("  sqrt: {:.0}", (sqrt_ns / add_ns) * add_base);
}
