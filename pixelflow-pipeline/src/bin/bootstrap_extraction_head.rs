//! Bootstrap the Judge (value head) on a large diverse corpus.
//!
//! Generates expressions with BwdGenerator, JIT-benchmarks them,
//! and trains the value MLP of ExprNnue on (expression → log-ns) pairs.
//! No policy, no critic, no egraph — just pure value head training.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin bootstrap_judge -- \
//!   --synthetic 50000 --epochs 100 --lr 0.001
//! ```

use std::path::PathBuf;

use clap::Parser;
use pixelflow_search::egraph::all_rules;
use pixelflow_search::nnue::factored::{EMBED_DIM, EdgeAccumulator, ExprNnue, GraphAccumulator, K};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

use pixelflow_pipeline::jit_bench::{benchmark_jit_arena, log_ns};
use pixelflow_pipeline::training::episodes::{build_rule_templates, load_corpus_exprs};
use pixelflow_pipeline::training::unified_backward::{
    UnifiedGradients, apply_unified_sgd, backward_value, forward_cached,
};

#[derive(Parser, Debug)]
#[command(name = "bootstrap_judge")]
#[command(about = "Bootstrap the Judge value head on (expression, JIT timing) pairs")]
struct Args {
    /// Number of synthetic expressions to generate.
    #[arg(long, default_value_t = 50000)]
    synthetic: usize,

    /// Training epochs.
    #[arg(long, default_value_t = 100)]
    epochs: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Momentum.
    #[arg(long, default_value_t = 0.9)]
    momentum: f32,

    /// Weight decay.
    #[arg(long, default_value_t = 1e-5)]
    weight_decay: f32,

    /// Gradient clipping.
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    /// Value loss coefficient.
    #[arg(long, default_value_t = 1.0)]
    value_coeff: f32,

    /// Mini-batch size.
    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    /// Path to model weights (loaded and saved).
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    model: PathBuf,

    /// Path to output model.
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_bootstrapped.bin")]
    output: PathBuf,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Skip corpus expressions from bench_corpus.bin.
    #[arg(long, default_value_t = false)]
    skip_corpus: bool,

    /// Progress print interval.
    #[arg(long, default_value_t = 1000)]
    progress_every: usize,
}

/// A single training sample: accumulator state + ground-truth log-ns cost.
struct ValueSample {
    acc: EdgeAccumulator,
    target_log_ns: f32,
}

fn main() {
    let args = Args::parse();

    // Load or create model
    let mut model = match ExprNnue::load(&args.model) {
        Ok(m) => {
            eprintln!("Loaded model from {}", args.model.display());
            m
        }
        Err(e) => {
            eprintln!("Could not load model from {}: {e}", args.model.display());
            eprintln!("Starting from random initialization");
            ExprNnue::new_random(args.seed)
        }
    };

    let rules = all_rules();
    let templates = build_rule_templates(&rules);

    // ── PHASE 1: Generate + benchmark synthetic expressions ──────────

    eprintln!(
        "\n=== Phase 1: Generating {} synthetic expressions ===",
        args.synthetic
    );

    let config = BwdGenConfig::default();
    let mut generator = BwdGenerator::new(args.seed, config, templates.clone());
    let mut samples: Vec<ValueSample> = Vec::new();
    let mut jit_failed = 0usize;
    let mut generated = 0usize;

    let mut node_counts: Vec<usize> = Vec::new();
    for i in 0..args.synthetic {
        let pair = generator.generate_arena();
        node_counts.push(pair.arena.node_count_subtree(pair.unoptimized));

        match benchmark_jit_arena(&pair.arena, pair.unoptimized) {
            Ok(bench) => {
                if bench.ns <= 0.0 || !bench.ns.is_finite() {
                    jit_failed += 1;
                    continue;
                }

                let acc = EdgeAccumulator::from_arena_dedup(
                    &pair.arena,
                    pair.unoptimized,
                    &model.embeddings,
                );
                samples.push(ValueSample {
                    acc,
                    target_log_ns: log_ns(bench.ns),
                });
                generated += 1;
            }
            Err(_) => {
                jit_failed += 1;
            }
        }

        if args.progress_every > 0 && (i + 1) % args.progress_every == 0 {
            eprintln!(
                "  [{}/{}] generated={} jit_failed={}",
                i + 1,
                args.synthetic,
                generated,
                jit_failed
            );
        }
    }

    node_counts.sort();
    if !node_counts.is_empty() {
        let len = node_counts.len();
        eprintln!(
            "Node counts: min={} p25={} median={} p75={} max={}",
            node_counts[0],
            node_counts[len / 4],
            node_counts[len / 2],
            node_counts[3 * len / 4],
            node_counts[len - 1],
        );
    }

    eprintln!(
        "Synthetic: {}/{} benchmarked ({} JIT failures)",
        generated, args.synthetic, jit_failed
    );

    // ── PHASE 1b: Add corpus expressions ─────────────────────────────

    if !args.skip_corpus {
        let corpus_path = std::path::Path::new("pixelflow-pipeline/data/bench_corpus.bin");
        if !corpus_path.exists() {
            eprintln!(
                "\nNo corpus at {} — skipping (run gen_bench_corpus first)",
                corpus_path.display()
            );
        } else {
            let corpus_exprs = load_corpus_exprs(corpus_path, 100_000, args.seed + 1);
            eprintln!(
                "\nLoading corpus from {} ({} expressions)",
                corpus_path.display(),
                corpus_exprs.len()
            );

            let mut corpus_ok = 0usize;
            let mut corpus_fail = 0usize;

            for (name, arena, root) in &corpus_exprs {
                match benchmark_jit_arena(arena, *root) {
                    Ok(bench) => {
                        if bench.ns <= 0.0 || !bench.ns.is_finite() {
                            corpus_fail += 1;
                            continue;
                        }
                        let acc =
                            EdgeAccumulator::from_arena_dedup(arena, *root, &model.embeddings);
                        samples.push(ValueSample {
                            acc,
                            target_log_ns: log_ns(bench.ns),
                        });
                        corpus_ok += 1;
                    }
                    Err(e) => {
                        if corpus_fail < 5 {
                            eprintln!("  corpus JIT failed for '{name}': {e}");
                        }
                        corpus_fail += 1;
                    }
                }
            }

            eprintln!(
                "Corpus: {}/{} benchmarked ({} failures)",
                corpus_ok,
                corpus_exprs.len(),
                corpus_fail
            );
        } // else corpus exists
    }

    eprintln!("\nTotal training samples: {}", samples.len());
    if samples.is_empty() {
        eprintln!("ERROR: No samples — cannot train. Check JIT and BwdGenerator.");
        std::process::exit(1);
    }

    // Print target distribution
    let mut targets: Vec<f32> = samples.iter().map(|s| s.target_log_ns).collect();
    targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "Target log-ns distribution: min={:.2} p25={:.2} median={:.2} p75={:.2} max={:.2}",
        targets[0],
        targets[targets.len() / 4],
        targets[targets.len() / 2],
        targets[3 * targets.len() / 4],
        targets[targets.len() - 1],
    );

    // ── PHASE 2: Train value head ────────────────────────────────────

    eprintln!(
        "\n=== Phase 2: Training value head ({} epochs, batch={}) ===",
        args.epochs, args.batch_size
    );

    let mut momentum_buf = UnifiedGradients::zero();

    // Simple LCG for shuffling
    let mut rng_state = args.seed;
    let mut rng_next = || -> u64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng_state >> 33
    };

    // We need a dummy rule embedding for forward_cached — value head doesn't use it
    let dummy_rule_embed = [0.0f32; EMBED_DIM];
    let dummy_gacc = GraphAccumulator::new();

    for epoch in 0..args.epochs {
        // Shuffle indices
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng_next() as usize % (i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0f64;
        let mut epoch_count = 0usize;

        for chunk in indices.chunks(args.batch_size) {
            let mut grads = UnifiedGradients::zero();

            for &idx in chunk {
                let sample = &samples[idx];
                let cache = forward_cached(&model, &sample.acc, &dummy_gacc, &dummy_rule_embed);

                backward_value(
                    &model,
                    &cache,
                    sample.target_log_ns,
                    args.value_coeff,
                    &mut grads,
                );

                let err = cache.value_pred - sample.target_log_ns;
                epoch_loss += (err * err) as f64;
                epoch_count += 1;
            }

            let batch_size = chunk.len().max(1) as f32;
            grads.scale(1.0 / batch_size);

            apply_unified_sgd(
                &mut model,
                &grads,
                &mut momentum_buf,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.grad_clip,
            );
        }

        let mse = epoch_loss / epoch_count.max(1) as f64;
        let rmse = mse.sqrt();

        if epoch == 0 || epoch == args.epochs - 1 || (epoch + 1) % 10 == 0 {
            // Compute MAE on last 1000 samples (rough validation)
            let mut mae = 0.0f64;
            let mut mae_count = 0usize;
            let start = if samples.len() > 1000 {
                samples.len() - 1000
            } else {
                0
            };
            for sample in &samples[start..] {
                let cache = forward_cached(&model, &sample.acc, &dummy_gacc, &dummy_rule_embed);
                mae += (cache.value_pred - sample.target_log_ns).abs() as f64;
                mae_count += 1;
            }
            mae /= mae_count.max(1) as f64;

            eprintln!(
                "Epoch {:>3}/{}: MSE={:.6} RMSE={:.4} tail_MAE={:.4}",
                epoch + 1,
                args.epochs,
                mse,
                rmse,
                mae,
            );
        }
    }

    // ── PHASE 3: Save ────────────────────────────────────────────────

    model
        .save(&args.output)
        .unwrap_or_else(|e| panic!("Failed to save model to {}: {e}", args.output.display()));
    eprintln!("\nSaved bootstrapped judge to {}", args.output.display());

    // Quick eval: predict a few samples
    eprintln!("\n=== Sample predictions ===");
    for i in [
        0,
        samples.len() / 4,
        samples.len() / 2,
        3 * samples.len() / 4,
        samples.len() - 1,
    ] {
        let sample = &samples[i];
        let cache = forward_cached(&model, &sample.acc, &dummy_gacc, &dummy_rule_embed);
        let pred_ns = libm::expf(cache.value_pred) as f64;
        let actual_ns = libm::expf(sample.target_log_ns) as f64;
        eprintln!(
            "  sample {i}: pred={:.1}ns actual={:.1}ns (log pred={:.3} actual={:.3})",
            pred_ns, actual_ns, cache.value_pred, sample.target_log_ns,
        );
    }
}
