#![allow(warnings)]
//! JIT-benchmark the curated expression corpus.
//!
//! Reads `bench_corpus.jsonl` (a JSONL file with `name` and `expression`
//! fields), JIT-compiles each expression using our custom SIMD emitter
//! (`compile_dag`), runs it a few times to measure cost, and writes
//! results to `judge_training.jsonl` (overwriting any prior run).
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --bin bench_jit_corpus
//! cargo run --release -p pixelflow-pipeline --bin bench_jit_corpus -- \
//!     --corpus pixelflow-pipeline/data/bench_corpus.jsonl \
//!     --output pixelflow-pipeline/data/judge_training.jsonl
//! ```

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use pixelflow_pipeline::jit_bench::benchmark_jit;
use pixelflow_pipeline::training::factored::parse_kernel_code;

#[derive(Parser)]
#[command(name = "bench_jit_corpus")]
#[command(about = "JIT-benchmark curated expression corpus")]
struct Args {
    /// Input corpus JSONL (fields: name, expression).
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.jsonl")]
    corpus: String,

    /// Output JSONL file (overwritten each run).
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    output: String,

    /// Print progress every N expressions.
    #[arg(long, default_value = "100")]
    progress_every: usize,
}

#[derive(Deserialize)]
struct CorpusEntry {
    name: String,
    expression: String,
}

fn main() {
    let args = Args::parse();

    // Load corpus.
    let corpus_file = std::fs::File::open(&args.corpus)
        .unwrap_or_else(|e| panic!("Failed to open corpus {}: {}", args.corpus, e));
    let corpus: Vec<CorpusEntry> = BufReader::new(corpus_file)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.unwrap_or_else(|e| panic!("Read error at line {}: {}", i + 1, e));
            serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Parse error at line {}: {}", i + 1, e))
        })
        .collect();

    println!(
        "Loaded {} corpus entries from {}",
        corpus.len(),
        args.corpus
    );

    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("Failed to open output {}: {}", args.output, e));

    let mut benchmarked = 0usize;
    let mut parse_failed = 0usize;
    let mut jit_failed = 0usize;
    let total_start = Instant::now();

    for entry in &corpus {
        let expr = match parse_kernel_code(&entry.expression) {
            Some(e) => e,
            None => {
                eprintln!("parse_kernel_code failed for '{}': {}", entry.name, entry.expression);
                parse_failed += 1;
                continue;
            }
        };

        let timing_ns = match benchmark_jit(&expr) {
            Ok(b) => b.ns,
            Err(e) => {
                eprintln!("JIT failed for '{}': {}", entry.name, e);
                jit_failed += 1;
                continue;
            }
        };

        writeln!(
            out,
            r#"{{"name":"{}","expression":"{}","timing_ns":{:.2}}}"#,
            entry.name, entry.expression, timing_ns
        )
        .unwrap_or_else(|e| panic!("Write failed: {}", e));

        benchmarked += 1;

        if args.progress_every > 0 && benchmarked % args.progress_every == 0 {
            let elapsed = total_start.elapsed().as_secs_f64();
            let rate = benchmarked as f64 / elapsed;
            println!(
                "  [{:>5}/{}] {:.0}/s  jit_failed={}  last={:.1}ns  {}",
                benchmarked,
                corpus.len() - parse_failed,
                rate,
                jit_failed,
                timing_ns,
                entry.name
            );
        }
    }

    let elapsed = total_start.elapsed().as_secs_f64();
    println!(
        "\nDone: {}/{} benchmarked in {:.1}s ({:.0}/s)  parse_failed={}  jit_failed={}",
        benchmarked,
        corpus.len(),
        elapsed,
        benchmarked as f64 / elapsed,
        parse_failed,
        jit_failed
    );
    println!("Results written to {}", args.output);
}

// benchmark_jit is imported from pixelflow_pipeline::jit_bench
