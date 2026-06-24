//! Validate and deduplicate raw shader expressions into bench_corpus.bin.
//!
//! Reads `raw_shadertoy.jsonl` (from the Python scraper), validates each expression
//! through `parse_kernel_code_arena` + `arena_to_kernel_code` round-trip, filters by node count,
//! deduplicates on canonical form, and writes to `bench_corpus.bin` (binary corpus format).
//!
//! If an existing `bench_corpus.bin` exists, its entries are loaded for dedup and preserved.
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin validate_corpus
//! ```

use std::collections::HashSet;
use std::io::BufRead;
use std::path::PathBuf;

use pixelflow_ir::ExprArena;
use pixelflow_pipeline::training::corpus::{read_corpus, write_corpus};
use pixelflow_pipeline::training::factored::{arena_to_kernel_code, parse_kernel_code_arena};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("pixelflow-pipeline/data/raw_shadertoy.jsonl"));
    let output_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("pixelflow-pipeline/data/bench_corpus.bin"));

    let min_nodes: usize = 5;
    let max_nodes: usize = 1000;

    eprintln!("Reading:  {}", input_path.display());
    eprintln!("Writing:  {}", output_path.display());
    eprintln!("Node range: [{min_nodes}, {max_nodes}]");

    // Load existing binary corpus for dedup
    let mut seen = HashSet::new();
    let mut existing: Vec<(String, ExprArena, pixelflow_ir::ExprId)> = Vec::new();
    if output_path.exists() {
        match read_corpus(&output_path) {
            Ok(entries) => {
                eprintln!("Existing: {} expressions (dedup base)", entries.len());
                for (_name, arena, root) in &entries {
                    let canonical = arena_to_kernel_code(arena, *root);
                    seen.insert(canonical);
                }
                existing = entries;
            }
            Err(e) => {
                eprintln!("WARNING: Failed to read existing corpus: {e} (starting fresh)");
            }
        }
    }

    // Read input JSONL
    let input_file = std::fs::File::open(&input_path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", input_path.display()));

    let mut total = 0usize;
    let mut parse_failed = 0usize;
    let mut roundtrip_failed = 0usize;
    let mut too_small = 0usize;
    let mut too_large = 0usize;
    let mut duplicates = 0usize;
    let mut validated = 0usize;

    for line in std::io::BufReader::new(input_file).lines() {
        let line = line.unwrap_or_else(|e| panic!("Failed to read line: {e}"));
        total += 1;

        let obj: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                parse_failed += 1;
                continue;
            }
        };

        let expression = match obj.get("expression").and_then(|v| v.as_str()) {
            Some(e) => e,
            None => {
                parse_failed += 1;
                continue;
            }
        };
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Step 1: Parse directly into arena form.
        let (arena, root) = match parse_kernel_code_arena(expression) {
            Some(parsed) => parsed,
            None => {
                if total <= 20 || parse_failed.is_multiple_of(100) {
                    eprintln!(
                        "[PARSE FAIL] {name}: {}",
                        &expression[..expression.len().min(80)]
                    );
                }
                parse_failed += 1;
                continue;
            }
        };

        // Step 2: Node count via arena (structural sharing gives tighter bound)
        let nodes = arena.len();
        if nodes < min_nodes {
            too_small += 1;
            continue;
        }
        if nodes > max_nodes {
            too_large += 1;
            continue;
        }

        // Step 3: Arena round-trip
        let canonical = arena_to_kernel_code(&arena, root);
        match parse_kernel_code_arena(&canonical) {
            Some((reparsed_arena, reparsed_root)) => {
                let re_emitted = arena_to_kernel_code(&reparsed_arena, reparsed_root);
                if re_emitted != canonical {
                    roundtrip_failed += 1;
                    continue;
                }
            }
            None => {
                roundtrip_failed += 1;
                continue;
            }
        }

        // Step 4: Dedup on canonical form
        if !seen.insert(canonical) {
            duplicates += 1;
            continue;
        }

        // Step 5: Collect
        existing.push((name.to_string(), arena, root));
        validated += 1;
    }

    // Write binary corpus
    write_corpus(&output_path, &existing).unwrap_or_else(|e| panic!("Failed to write corpus: {e}"));

    eprintln!("\n=== Validation Results ===");
    eprintln!("  Total input:      {total}");
    eprintln!("  Parse failed:     {parse_failed}");
    eprintln!("  Too small (<{min_nodes}): {too_small}");
    eprintln!("  Too large (>{max_nodes}): {too_large}");
    eprintln!("  Round-trip fail:  {roundtrip_failed}");
    eprintln!("  Duplicates:       {duplicates}");
    eprintln!("  New validated:    {validated}");
    eprintln!("  Total in corpus:  {}", existing.len());
}
