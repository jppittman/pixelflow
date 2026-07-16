//! JIT-benchmark the curated expression corpus.
//!
//! Reads `bench_corpus.bin` (binary corpus format), JIT-compiles each
//! expression, runs it a few times to measure cost, and writes results
//! to a JSONL benchmark report (overwriting any prior run).
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_jit_corpus
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_jit_corpus -- \
//!     --corpus pixelflow-pipeline/data/bench_corpus.bin \
//!     --output pixelflow-pipeline/data/corpus_bench.jsonl
//! ```

use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

use clap::Parser;

use pixelflow_pipeline::jit_bench::benchmark_jit_arena;
use pixelflow_pipeline::training::corpus;
use pixelflow_pipeline::training::factored::arena_to_kernel_code;

#[derive(Parser)]
#[command(name = "bench_jit_corpus")]
#[command(about = "JIT-benchmark curated expression corpus")]
struct Args {
    /// Input binary corpus file.
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.bin")]
    corpus: String,

    /// Output JSONL file (overwritten each run).
    #[arg(long, default_value = "pixelflow-pipeline/data/corpus_bench.jsonl")]
    output: String,

    /// Print progress every N expressions.
    #[arg(long, default_value = "100")]
    progress_every: usize,
}

fn main() {
    let args = Args::parse();

    // Load binary corpus.
    let entries = corpus::read_corpus(std::path::Path::new(&args.corpus))
        .unwrap_or_else(|e| panic!("Failed to read corpus {}: {e}", args.corpus));

    println!(
        "Loaded {} corpus entries from {}",
        entries.len(),
        args.corpus
    );

    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("Failed to open output {}: {e}", args.output));

    let mut benchmarked = 0usize;
    let mut jit_failed = 0usize;
    let total_start = Instant::now();

    for (name, arena, root) in &entries {
        let timing_ns = match benchmark_jit_arena(arena, *root) {
            Ok(b) => b.ns,
            Err(e) => {
                eprintln!("JIT failed for '{}': {}", name, e);
                jit_failed += 1;
                continue;
            }
        };

        // Convert to kernel code text for the output JSONL.
        let expression = arena_to_kernel_code(arena, *root);
        writeln!(
            out,
            r#"{{"name":"{}","expression":"{}","timing_ns":{:.2}}}"#,
            name, expression, timing_ns
        )
        .unwrap_or_else(|e| panic!("Write failed: {e}"));

        benchmarked += 1;

        if args.progress_every > 0 && benchmarked % args.progress_every == 0 {
            let elapsed = total_start.elapsed().as_secs_f64();
            let rate = benchmarked as f64 / elapsed;
            println!(
                "  [{:>5}/{}] {:.0}/s  jit_failed={}  last={:.1}ns  {}",
                benchmarked,
                entries.len(),
                rate,
                jit_failed,
                timing_ns,
                name
            );
        }
    }

    let elapsed = total_start.elapsed().as_secs_f64();
    println!(
        "\nDone: {}/{} benchmarked in {:.1}s ({:.0}/s)  jit_failed={}",
        benchmarked,
        entries.len(),
        elapsed,
        benchmarked as f64 / elapsed,
        jit_failed
    );
    println!("Results written to {}", args.output);
}
