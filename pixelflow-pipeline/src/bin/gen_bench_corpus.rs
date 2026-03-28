#![allow(warnings)]
//! Generate a diverse expression corpus for JIT benchmarking.
//!
//! Generates unique expressions across varied depth/complexity bands,
//! serializes them to kernel code syntax, and writes `bench_corpus.jsonl`.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus -- \
//!     --target 20000 --output pixelflow-pipeline/data/bench_corpus.jsonl
//! ```

use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};

use clap::Parser;
use pixelflow_pipeline::training::factored::{expr_to_kernel_code, parse_kernel_code};
use pixelflow_search::nnue::{ExprGenConfig, ExprGenerator};

#[derive(Parser)]
#[command(name = "gen_bench_corpus")]
#[command(about = "Generate diverse expression corpus for JIT benchmarking")]
struct Args {
    /// Output JSONL file path.
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.jsonl")]
    output: String,

    /// Target number of unique expressions.
    #[arg(long, default_value_t = 360000)]
    target: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum node count (skip overly huge expressions).
    #[arg(long, default_value_t = 500)]
    max_nodes: usize,
}

struct Band {
    max_depth: usize,
    leaf_prob: f32,
    num_vars: usize,
}

const BANDS: &[Band] = &[
    // Tiny expressions — 1-2 vars
    Band { max_depth: 2, leaf_prob: 0.6, num_vars: 1 },
    Band { max_depth: 2, leaf_prob: 0.5, num_vars: 2 },
    Band { max_depth: 3, leaf_prob: 0.6, num_vars: 1 },
    Band { max_depth: 3, leaf_prob: 0.5, num_vars: 2 },
    Band { max_depth: 3, leaf_prob: 0.4, num_vars: 4 },
    // Small
    Band { max_depth: 4, leaf_prob: 0.5, num_vars: 2 },
    Band { max_depth: 4, leaf_prob: 0.4, num_vars: 4 },
    Band { max_depth: 4, leaf_prob: 0.3, num_vars: 4 },
    // Medium
    Band { max_depth: 5, leaf_prob: 0.4, num_vars: 2 },
    Band { max_depth: 5, leaf_prob: 0.3, num_vars: 4 },
    Band { max_depth: 5, leaf_prob: 0.2, num_vars: 4 },
    Band { max_depth: 6, leaf_prob: 0.35, num_vars: 3 },
    Band { max_depth: 6, leaf_prob: 0.3, num_vars: 4 },
    Band { max_depth: 6, leaf_prob: 0.2, num_vars: 4 },
    // Large
    Band { max_depth: 7, leaf_prob: 0.3, num_vars: 4 },
    Band { max_depth: 7, leaf_prob: 0.25, num_vars: 4 },
    Band { max_depth: 7, leaf_prob: 0.2, num_vars: 4 },
    Band { max_depth: 8, leaf_prob: 0.3, num_vars: 4 },
    Band { max_depth: 8, leaf_prob: 0.25, num_vars: 4 },
    Band { max_depth: 8, leaf_prob: 0.2, num_vars: 4 },
    // Deep
    Band { max_depth: 9, leaf_prob: 0.25, num_vars: 4 },
    Band { max_depth: 9, leaf_prob: 0.2, num_vars: 4 },
    Band { max_depth: 10, leaf_prob: 0.2, num_vars: 4 },
    Band { max_depth: 10, leaf_prob: 0.15, num_vars: 4 },
    // Very deep — complex kernels
    Band { max_depth: 11, leaf_prob: 0.2, num_vars: 4 },
    Band { max_depth: 11, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 12, leaf_prob: 0.2, num_vars: 4 },
    Band { max_depth: 12, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 13, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 14, leaf_prob: 0.15, num_vars: 4 },
    // Huge — saturation-scale expressions (100-500+ nodes)
    Band { max_depth: 15, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 16, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 16, leaf_prob: 0.12, num_vars: 4 },
    Band { max_depth: 18, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 18, leaf_prob: 0.12, num_vars: 4 },
    Band { max_depth: 20, leaf_prob: 0.15, num_vars: 4 },
    Band { max_depth: 20, leaf_prob: 0.12, num_vars: 4 },
    Band { max_depth: 20, leaf_prob: 0.10, num_vars: 4 },
];

fn main() {
    let args = Args::parse();

    let per_band = (args.target + BANDS.len() - 1) / BANDS.len();
    let mut seen = HashSet::new();
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut roundtrip_failures = 0usize;
    let mut skipped_too_large = 0usize;
    let mut duplicates = 0usize;

    for (band_idx, band) in BANDS.iter().enumerate() {
        let config = ExprGenConfig {
            max_depth: band.max_depth,
            leaf_prob: band.leaf_prob,
            num_vars: band.num_vars,
            include_fused: false,
        };
        let mut rng = ExprGenerator::new(
            args.seed.wrapping_mul(8).wrapping_add(band_idx as u64),
            config,
        );

        let mut band_count = 0usize;
        let mut attempts = 0usize;
        let max_attempts = per_band * 10;

        while band_count < per_band && attempts < max_attempts {
            attempts += 1;
            let expr = rng.generate();

            if expr.node_count() > args.max_nodes {
                skipped_too_large += 1;
                continue;
            }

            let code = expr_to_kernel_code(&expr);

            if !seen.insert(code.clone()) {
                duplicates += 1;
                continue;
            }

            // Verify round-trip: serialize → parse → re-serialize must match.
            match parse_kernel_code(&code) {
                Some(reparsed) => {
                    let re_emitted = expr_to_kernel_code(&reparsed);
                    if re_emitted != code {
                        eprintln!(
                            "Round-trip MISMATCH:\n  original:  {}\n  re-emitted: {}",
                            code, re_emitted
                        );
                        roundtrip_failures += 1;
                        continue;
                    }
                }
                None => {
                    eprintln!("Round-trip PARSE FAILED: {}", code);
                    roundtrip_failures += 1;
                    continue;
                }
            }

            let name = format!("gen{:05}", entries.len());
            entries.push((name, code));
            band_count += 1;
        }

        println!(
            "Band {:>2} (depth={:>2}, leaf={:.2}, vars={}): {:>5} exprs in {:>6} attempts",
            band_idx, band.max_depth, band.leaf_prob, band.num_vars, band_count, attempts
        );
    }

    // Write output.
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Failed to create output directory: {}", e));
    }

    let file = fs::File::create(&args.output)
        .unwrap_or_else(|e| panic!("Failed to create output file {}: {}", args.output, e));
    let mut out = BufWriter::new(file);

    for (name, expression) in &entries {
        let escaped = expression.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(out, r#"{{"name":"{}","expression":"{}"}}"#, name, escaped)
            .unwrap_or_else(|e| panic!("Write failed: {}", e));
    }

    out.flush().unwrap_or_else(|e| panic!("Flush failed: {}", e));

    println!("\n=== Summary ===");
    println!("Total unique expressions: {}", entries.len());
    println!("Duplicates skipped:       {}", duplicates);
    println!("Too large skipped:        {}", skipped_too_large);
    println!("Round-trip failures:      {}", roundtrip_failures);
    println!("Output: {}", args.output);
}
