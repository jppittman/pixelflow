//! JIT-benchmark a corpus tier file.
//!
//! Reads a binary PXCR corpus (by default the TRAIN tier written by
//! `gen_bench_corpus`), JIT-compiles each expression through a
//! [`BenchSession`] (QoS pinning, DVFS burn-in, drift sentinel, tick-floor
//! autoscaling, overhead subtraction — audit H4/H5/M1/L5), measures BOTH
//! dependency modes (throughput and chain-serialized latency, audit H3), and
//! writes one serde-serialized JSONL record per expression including IQR and
//! mode metadata (audit L3/M5).
//!
//! Failures are never a bare counter (audit M2): every excluded expression is
//! written to a structured failure sidecar, and the run panics if the
//! exclusion rate exceeds [`MAX_EXCLUSION_RATE`].
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_jit_corpus
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_jit_corpus -- \
//!     --corpus pixelflow-pipeline/data/corpus_train.bin \
//!     --output pixelflow-pipeline/data/corpus_bench.jsonl \
//!     --failures pixelflow-pipeline/data/corpus_bench_failures.jsonl
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use clap::Parser;
use pixelflow_ir::ExprGraph;
use serde::Serialize;

use pixelflow_pipeline::jit_bench::{
    BenchError, BenchMode, BenchPosition, BenchResult, BenchSession, CostLabel,
};
use pixelflow_pipeline::training::corpus;
use pixelflow_pipeline::training::factored::graph_to_kernel_code;

/// Censoring alarm threshold (audit M2): a run excluding more than this
/// fraction of its corpus is minting a biased label set and must not be
/// trusted silently.
const MAX_EXCLUSION_RATE: f64 = 0.05;

#[derive(Parser)]
#[command(name = "bench_jit_corpus")]
#[command(about = "JIT-benchmark a corpus tier file")]
struct Args {
    /// Input binary corpus file (a tier file from gen_bench_corpus).
    #[arg(long, default_value = "pixelflow-pipeline/data/corpus_train.bin")]
    corpus: String,

    /// Output JSONL file (overwritten each run).
    #[arg(long, default_value = "pixelflow-pipeline/data/corpus_bench.jsonl")]
    output: String,

    /// Failure sidecar JSONL (overwritten each run): one structured record
    /// per excluded expression (audit M2).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/corpus_bench_failures.jsonl"
    )]
    failures: String,

    /// Print progress every N expressions.
    #[arg(long, default_value = "100")]
    progress_every: usize,
}

/// One benchmark mode's measurement, with the dispersion and scaling metadata
/// the audit found being thrown away (M5) or unrecorded (H5/M1), plus the
/// [`CostLabel`] (docs/plans/2026-08-17-cost-model-domain.md, J3) minted from
/// it: `value_ns` and `position` are the drift-corrected, order-tagged label
/// a training consumer should use, computed once here rather than re-derived
/// (and potentially re-ordered) downstream.
#[derive(Serialize)]
struct ModeRecord {
    mode: &'static str,
    ns: f64,
    adjusted_ns: f64,
    call_overhead_ns: f64,
    iqr_ns: f64,
    repeat_batches: usize,
    /// Most recent sentinel reading when this label was minted (audit H4).
    local_sentinel_ns: f64,
    /// The session's opening sentinel calibration.
    sentinel_calibration_ns: f64,
    /// `sentinel_calibration_ns / local_sentinel_ns` — the factor
    /// [`CostLabel::mint`] applied to reach `value_ns`, persisted so training
    /// can down-weight heavily-corrected samples.
    sentinel_normalization: f64,
    /// [`CostLabel::value`]: `ns` normalized by `sentinel_normalization`,
    /// THEN `call_overhead_ns` subtracted (`CostLabel::mint`'s ordering, not
    /// `adjusted_ns * sentinel_normalization` — that scales a value that
    /// already mixes two clocks and leaves a drift-dependent residue; see
    /// `training::mint::normalized_label_ns`).
    value_ns: f64,
    /// [`CostLabel::position`]: this expression's index in the corpus file
    /// (this binary benchmarks in stored order, unshuffled), the axis timing
    /// drift correlates with.
    position: usize,
}

impl ModeRecord {
    fn mint(b: &BenchResult, position: BenchPosition) -> Self {
        let label = CostLabel::mint(b, position, "bench_jit_corpus");
        let sentinel = label.calibration();
        Self {
            mode: match b.mode {
                BenchMode::Throughput => "throughput",
                BenchMode::Latency => "latency",
                BenchMode::Scanline => "scanline",
            },
            ns: b.ns,
            adjusted_ns: b.adjusted_ns,
            call_overhead_ns: b.call_overhead_ns,
            iqr_ns: b.iqr_ns,
            repeat_batches: b.repeat_batches,
            local_sentinel_ns: sentinel.local_sentinel_ns,
            sentinel_calibration_ns: sentinel.calibration_ns,
            sentinel_normalization: label.drift().get(),
            value_ns: label.value().get(),
            position: position.0,
        }
    }
}

#[derive(Serialize)]
struct BenchRecord<'a> {
    name: &'a str,
    expression: &'a str,
    throughput: ModeRecord,
    latency: ModeRecord,
}

#[derive(Serialize)]
struct FailureRecord<'a> {
    name: &'a str,
    expression: &'a str,
    reason: &'a str,
    error: &'a str,
}

/// Stable category slug for a [`BenchError`], for the sidecar's `reason` field.
fn bench_error_reason(e: &BenchError) -> &'static str {
    match e {
        BenchError::CompileFailed(_) => "compile_failed",
        BenchError::UnsupportedArch => "unsupported_arch",
        BenchError::InvalidMeasurement(_) => "invalid_measurement",
    }
}

fn main() {
    let args = Args::parse();

    let entries = corpus::read_corpus(std::path::Path::new(&args.corpus))
        .unwrap_or_else(|e| panic!("Failed to read corpus {}: {e}", args.corpus));
    assert!(
        !entries.is_empty(),
        "corpus {} is empty — run gen_bench_corpus to build the tiered corpus",
        args.corpus
    );

    println!(
        "Loaded {} corpus entries from {}",
        entries.len(),
        args.corpus
    );

    let mut out = BufWriter::new(
        File::create(&args.output)
            .unwrap_or_else(|e| panic!("Failed to open output {}: {e}", args.output)),
    );
    // The failure sidecar is written through the `File` directly — no
    // `BufWriter`. The workspace compiles with `panic = "abort"`, so a panic
    // (the censoring alarm below, or any harness assertion) skips every
    // destructor and would discard buffered records; the alarm's whole value is
    // pointing the reader at exclusions that must therefore already be on disk.
    // Exclusions are asserted rare (<= 5%), so per-record writes cost nothing.
    let mut failures_out = File::create(&args.failures)
        .unwrap_or_else(|e| panic!("Failed to open failure sidecar {}: {e}", args.failures));

    let mut session = BenchSession::new();

    let mut benchmarked = 0usize;
    let mut excluded = 0usize;
    let total_start = Instant::now();

    for (position, (name, rooted, environment)) in entries.iter().enumerate() {
        let graph = ExprGraph::new(rooted.clone(), environment.clone());
        let expression = graph_to_kernel_code(&graph);
        match session.benchmark_graph_both(&graph) {
            Ok((throughput, latency)) => {
                // Both modes were measured for the same expression at the
                // same point in the (unshuffled, stored-order) corpus walk,
                // so they share one CostLabel position.
                let position = BenchPosition(position);
                let record = BenchRecord {
                    name,
                    expression: &expression,
                    throughput: ModeRecord::mint(&throughput, position),
                    latency: ModeRecord::mint(&latency, position),
                };
                let json = serde_json::to_string(&record).unwrap_or_else(|e| {
                    panic!("Failed to serialize bench record for '{name}': {e}")
                });
                writeln!(out, "{json}")
                    .unwrap_or_else(|e| panic!("Write failed for {}: {e}", args.output));
                benchmarked += 1;

                if args.progress_every > 0 && benchmarked.is_multiple_of(args.progress_every) {
                    let elapsed = total_start.elapsed().as_secs_f64();
                    let rate = benchmarked as f64 / elapsed;
                    println!(
                        "  [{:>5}/{}] {:.0}/s  excluded={}  last_latency={:.1}ns  {}",
                        benchmarked,
                        entries.len(),
                        rate,
                        excluded,
                        latency.adjusted_ns,
                        name
                    );
                }
            }
            Err(e) => {
                let error = e.to_string();
                let record = FailureRecord {
                    name,
                    expression: &expression,
                    reason: bench_error_reason(&e),
                    error: &error,
                };
                let json = serde_json::to_string(&record).unwrap_or_else(|e| {
                    panic!("Failed to serialize failure record for '{name}': {e}")
                });
                writeln!(failures_out, "{json}")
                    .unwrap_or_else(|e| panic!("Write failed for {}: {e}", args.failures));
                eprintln!("EXCLUDED '{name}': {error}");
                excluded += 1;
            }
        }
    }

    out.flush()
        .unwrap_or_else(|e| panic!("Flush failed for {}: {e}", args.output));
    // No flush for the failure sidecar: it is unbuffered by construction.

    let elapsed = total_start.elapsed().as_secs_f64();
    let attempted = entries.len();
    let exclusion_rate = excluded as f64 / attempted as f64;
    println!(
        "\nDone: {}/{} benchmarked in {:.1}s ({:.0}/s)  excluded={} ({:.2}%)  sentinel_samples={}",
        benchmarked,
        attempted,
        elapsed,
        benchmarked as f64 / elapsed,
        excluded,
        exclusion_rate * 100.0,
        session.sentinel_samples().len(),
    );
    println!("Results written to {}", args.output);
    println!("Failure sidecar written to {}", args.failures);

    assert!(
        exclusion_rate <= MAX_EXCLUSION_RATE,
        "censoring alarm (audit M2): {excluded}/{attempted} expressions excluded \
         ({:.2}% > {:.0}%) — the label set is non-randomly censored; \
         see {} for every exclusion",
        exclusion_rate * 100.0,
        MAX_EXCLUSION_RATE * 100.0,
        args.failures
    );
}
