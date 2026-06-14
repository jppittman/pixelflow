//! Generate a diverse expression corpus for JIT benchmarking.
//!
//! Generates unique expressions across varied depth/complexity bands,
//! converts them to `ExprArena` DAGs, and writes `bench_corpus.bin`
//! (binary corpus format).
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus -- \
//!     --target 20000 --output pixelflow-pipeline/data/bench_corpus.bin
//! ```

use std::collections::HashSet;

use clap::Parser;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};
use pixelflow_pipeline::training::corpus::write_corpus;
use pixelflow_search::egraph::collect_rule_templates;
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

#[derive(Parser)]
#[command(name = "gen_bench_corpus")]
#[command(about = "Generate diverse expression corpus for JIT benchmarking")]
struct Args {
    /// Output binary corpus file path.
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.bin")]
    output: String,

    /// Target number of unique expressions.
    #[arg(long, default_value_t = 360000)]
    target: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum node count (skip overly huge expressions).
    #[arg(long, default_value_t = 1000)]
    max_nodes: usize,
}

struct Band {
    max_depth: usize,
    leaf_prob: f32,
    num_vars: usize,
}

const BANDS: &[Band] = &[
    // Tiny expressions — 1-2 vars
    Band {
        max_depth: 2,
        leaf_prob: 0.6,
        num_vars: 1,
    },
    Band {
        max_depth: 2,
        leaf_prob: 0.5,
        num_vars: 2,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.6,
        num_vars: 1,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.5,
        num_vars: 2,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.4,
        num_vars: 4,
    },
    // Small
    Band {
        max_depth: 4,
        leaf_prob: 0.5,
        num_vars: 2,
    },
    Band {
        max_depth: 4,
        leaf_prob: 0.4,
        num_vars: 4,
    },
    Band {
        max_depth: 4,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    // Medium
    Band {
        max_depth: 5,
        leaf_prob: 0.4,
        num_vars: 2,
    },
    Band {
        max_depth: 5,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 5,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.35,
        num_vars: 3,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    // Large
    Band {
        max_depth: 7,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 7,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 7,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.3,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    // Deep
    Band {
        max_depth: 9,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 9,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 10,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 10,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    // Very deep — complex kernels
    Band {
        max_depth: 11,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 11,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 12,
        leaf_prob: 0.2,
        num_vars: 4,
    },
    Band {
        max_depth: 12,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 13,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 14,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    // Huge — saturation-scale expressions (100-500+ nodes)
    Band {
        max_depth: 15,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 16,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 16,
        leaf_prob: 0.12,
        num_vars: 4,
    },
    Band {
        max_depth: 18,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 18,
        leaf_prob: 0.12,
        num_vars: 4,
    },
    Band {
        max_depth: 20,
        leaf_prob: 0.15,
        num_vars: 4,
    },
    Band {
        max_depth: 20,
        leaf_prob: 0.12,
        num_vars: 4,
    },
    Band {
        max_depth: 20,
        leaf_prob: 0.10,
        num_vars: 4,
    },
];

#[derive(Clone, PartialEq, Eq, Hash)]
enum SigNode {
    Var(u8),
    Const(u32),
    Param(u8),
    Unary(OpKind, u32),
    Binary(OpKind, u32, u32),
    Ternary(OpKind, u32, u32, u32),
    Nary(OpKind, Box<[u32]>),
}

fn arena_structural_key(arena: &ExprArena, root: ExprId) -> Vec<SigNode> {
    enum Task {
        Visit(ExprId),
        Emit(ExprId),
    }

    let mut work = vec![Task::Visit(root)];
    let mut visited = HashSet::new();
    let mut signatures = Vec::new();
    let mut ids = std::collections::HashMap::<ExprId, u32>::new();

    while let Some(task) = work.pop() {
        match task {
            Task::Visit(id) => {
                if !visited.insert(id) {
                    continue;
                }
                work.push(Task::Emit(id));
                let children: Vec<ExprId> = arena.children(id).collect();
                for child in children.into_iter().rev() {
                    work.push(Task::Visit(child));
                }
            }
            Task::Emit(id) => {
                let sig = match arena.node(id) {
                    ExprNode::Var(i) => SigNode::Var(*i),
                    ExprNode::Const(v) => SigNode::Const(v.to_bits()),
                    ExprNode::Param(i) => SigNode::Param(*i),
                    ExprNode::Unary(op, a) => SigNode::Unary(*op, ids[a]),
                    ExprNode::Binary(op, a, b) => SigNode::Binary(*op, ids[a], ids[b]),
                    ExprNode::Ternary(op, a, b, c) => SigNode::Ternary(*op, ids[a], ids[b], ids[c]),
                    ExprNode::Nary(op, start, len) => {
                        let children = arena.nary_children_slice(*start, *len);
                        let child_ids: Vec<u32> = children.iter().map(|child| ids[child]).collect();
                        SigNode::Nary(*op, child_ids.into_boxed_slice())
                    }
                };
                let sig_id = signatures.len() as u32;
                signatures.push(sig);
                ids.insert(id, sig_id);
            }
        }
    }

    signatures
}

fn main() {
    let args = Args::parse();

    let per_band = args.target.div_ceil(BANDS.len());
    let mut seen = HashSet::new();
    let mut entries: Vec<(String, ExprArena, pixelflow_ir::ExprId)> = Vec::new();
    let mut skipped_too_large = 0usize;
    let mut duplicates = 0usize;

    for (band_idx, band) in BANDS.iter().enumerate() {
        let config = BwdGenConfig {
            max_depth: band.max_depth,
            leaf_prob: band.leaf_prob,
            num_vars: band.num_vars,
            fused_op_prob: 0.0,
            ..BwdGenConfig::default()
        };
        let mut rng = BwdGenerator::new(
            args.seed.wrapping_mul(8).wrapping_add(band_idx as u64),
            config,
            collect_rule_templates(),
        );

        let mut band_count = 0usize;
        let mut attempts = 0usize;
        let max_attempts = per_band * 10;

        while band_count < per_band && attempts < max_attempts {
            attempts += 1;
            let pair = rng.generate_arena();
            let arena = pair.arena;
            let root = pair.unoptimized;

            if arena.node_count_subtree(root) > args.max_nodes {
                skipped_too_large += 1;
                continue;
            }

            if !seen.insert(arena_structural_key(&arena, root)) {
                duplicates += 1;
                continue;
            }

            let name = format!("gen{:05}", entries.len());
            entries.push((name, arena, root));
            band_count += 1;
        }

        println!(
            "Band {:>2} (depth={:>2}, leaf={:.2}, vars={}): {:>5} exprs in {:>6} attempts",
            band_idx, band.max_depth, band.leaf_prob, band.num_vars, band_count, attempts
        );
    }

    // Write binary corpus.
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Failed to create output directory: {e}"));
    }

    write_corpus(std::path::Path::new(&args.output), &entries)
        .unwrap_or_else(|e| panic!("Failed to write corpus: {e}"));

    println!("\n=== Summary ===");
    println!("Total unique expressions: {}", entries.len());
    println!("Duplicates skipped:       {duplicates}");
    println!("Too large skipped:        {skipped_too_large}");
    println!("Output: {}", args.output);
}
