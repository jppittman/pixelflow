//! Cap-break A/B (docs/results/2026-09-02-cap-break-ab.md).
//!
//! PR #1083 introduced `ScanStop` and classified a truncated sweep by it —
//! a real telemetry fix — but the same commit also made a `ClassCap` sweep
//! BREAK the iteration loop, where before the loop continued. Classification
//! and termination are separable. This `#[ignore]`d measurement replays the
//! exact production call on the 204 real kernels of
//! docs/results/2026-09-01-rule-order-real-kernels.md and writes the row the
//! A/B table is built from; the arm is whatever `graph.rs`'s loop was
//! compiled with, named by `PIXELFLOW_CAP_ARM`.
//!
//! Harness helpers (`load_arena`, `arena_cost`, the `run` sequence) are
//! PR #1101's, reused verbatim rather than rebuilt.

use super::*;
use crate::egraph::{Budget, CostModel, Optimizer, SaturationStop};
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DIR_VAR: &str = "PIXELFLOW_TELEMETRY_DIR";
const OUT_VAR: &str = "PIXELFLOW_CAP_OUT";
const ARM_VAR: &str = "PIXELFLOW_CAP_ARM";

/// Deadline for the clock-neutral regime: the production tier's own
/// iteration and class caps, but a wall clock so generous it cannot
/// bind. Production's 10/50/200 ms ceilings are machine-load dependent,
/// so the production regime alone cannot separate the control-flow
/// change from the host's load. This regime is the deterministic one;
/// the production regime is the shipping one. Both are reported.
const GENEROUS_MS_VAR: &str = "PIXELFLOW_CAP_GENEROUS_MS";
const DEFAULT_GENEROUS_MS: u64 = 5_000;

fn generous() -> Duration {
    Duration::from_millis(
        std::env::var(GENEROUS_MS_VAR)
            .ok()
            .map(|v| {
                v.parse()
                    .expect("PIXELFLOW_CAP_GENEROUS_MS must be an integer")
            })
            .unwrap_or(DEFAULT_GENEROUS_MS),
    )
}

fn env_required(var: &str) -> String {
    std::env::var(var).unwrap_or_else(|e| panic!("{var} must be set ({e})"))
}

/// Inverse of the dumpers' `dump_arena` — PR #1101's loader, verbatim.
fn load_arena(path: &Path) -> (String, ExprArena, ExprId) {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("# pixelflow arena dump v1"),
        "{}: bad header",
        path.display()
    );
    let mut name = None;
    let mut arena = ExprArena::new();
    let mut idents: Vec<BufferIdentity> = Vec::new();
    let mut root = None;
    let mut next_id: u32 = 0;
    let mut buf_count: u16 = 0;
    let op = |s: &str| -> OpKind {
        OpKind::all()
            .find(|k| format!("{k:?}") == s)
            .unwrap_or_else(|| panic!("{}: unknown OpKind {s:?}", path.display()))
    };
    let id = |s: &str| -> ExprId {
        ExprId(
            s.parse()
                .unwrap_or_else(|e| panic!("{}: bad id {s:?}: {e}", path.display())),
        )
    };
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        let pushed = match f.as_slice() {
            ["name", n] => {
                name = Some((*n).to_string());
                continue;
            }
            ["buf", ord, w, h] => {
                let ord: usize = ord.parse().expect("buf ordinal");
                while idents.len() <= ord {
                    idents.push(BufferIdentity::mint());
                }
                let slot = arena.declare_buffer(BufferDecl {
                    id: idents[ord],
                    width: w.parse().expect("buf width"),
                    height: h.parse().expect("buf height"),
                });
                assert_eq!(
                    slot.0,
                    buf_count,
                    "{}: buffer slot order drifted",
                    path.display()
                );
                buf_count += 1;
                continue;
            }
            ["root", r] => {
                root = Some(id(r));
                continue;
            }
            ["V", i] => arena.push_var(i.parse().expect("var index")),
            ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
            ["B", slot] => arena.push_buffer(pixelflow_ir::arena::BufferId(
                slot.parse().expect("buffer slot"),
            )),
            ["U", k, a] => arena.push_unary(op(k), id(a)),
            ["Bi", k, a, b] => arena.push_binary(op(k), id(a), id(b)),
            ["T", k, a, b, c] => arena.push_ternary(op(k), id(a), id(b), id(c)),
            other => panic!("{}: unparseable line {other:?}", path.display()),
        };
        assert_eq!(
            pushed,
            ExprId(next_id),
            "{}: replay drifted from dumped ids",
            path.display()
        );
        next_id += 1;
    }
    let name = name.unwrap_or_else(|| panic!("{}: no name line", path.display()));
    let root = root.unwrap_or_else(|| panic!("{}: no root line", path.display()));
    (name, arena, root)
}

/// Latency-prior cost of the arena the JIT would actually execute: the
/// per-op table summed over every reachable operation once. PR #1101's
/// quality metric, and the metric here.
fn arena_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> usize {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut total = 0usize;
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let kind = match arena.node(id) {
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Buffer(_) => None,
            ExprNode::Unary(k, _) | ExprNode::Binary(k, _, _) | ExprNode::Ternary(k, _, _, _) => {
                Some(*k)
            }
            other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                panic!("extracted arena contains {other:?}")
            }
        };
        if let Some(k) = kind {
            assert_ne!(k, OpKind::Dwrt, "Dwrt survived extraction");
            total += costs.cost(k);
        }
        stack.extend(arena.children(id));
    }
    total
}

struct Run {
    stop: SaturationStop,
    iterations: usize,
    total_unions: usize,
    classes_after: usize,
    applications: usize,
    elapsed: Duration,
    cost: usize,
    extracted_nodes: usize,
}

/// The production sequence of `optimize_runtime_arena_uncached` from the
/// e-graph build onward, with the budget as parameters.
///
/// Driven through `Optimizer` — the one entry point production itself now
/// uses (#1108) — rather than through `saturate_with_full_budget` plus a
/// hand-rolled extraction. `Budget::Explicit` is what lets the budget stay a
/// parameter, which this A/B needs: both arms must meet the same caps so the
/// only difference between them is the control flow under test.
fn run(
    arena: &ExprArena,
    root: ExprId,
    max_iterations: usize,
    max_classes: usize,
    timeout: Duration,
) -> Run {
    let mut optimizer = Optimizer::production()
        .budget(Budget::Explicit {
            iterations: max_iterations,
            classes: max_classes,
            applications: None,
        })
        .hard_ceiling(timeout);

    let mut egraph = optimizer.egraph();
    let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(arena, root, &mut egraph, &mut memo)
        .expect("production arena must be e-graph representable (no Param/Nary)");

    let started = Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, reachable_count(arena, root));
    let elapsed = started.elapsed();

    let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);
    let costs = CostModel::latency_prior();

    Run {
        stop: optimized.stats.stop,
        iterations: optimized.stats.iterations,
        total_unions: optimized.stats.unions,
        classes_after: optimized.stats.classes,
        applications: optimized.stats.applications as usize,
        elapsed,
        cost: arena_cost(&extracted, extracted_root, &costs),
        extracted_nodes: reachable_count(&extracted, extracted_root),
    }
}

fn tier_name(config: &crate::egraph::SaturationConfig) -> &'static str {
    match config.max_iterations {
        20 => "blitz",
        50 => "rapid",
        100 => "classical",
        n => panic!("unrecognized production tier: max_iterations {n}"),
    }
}

fn load_averages() -> String {
    std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[test]
#[ignore = "measurement: PIXELFLOW_TELEMETRY_DIR=<dumps> PIXELFLOW_CAP_ARM=A PIXELFLOW_CAP_OUT=<csv> \
            cargo test -p pixelflow-search --release -- --ignored cap_break_ab --nocapture --test-threads=1"]
fn cap_break_ab_real_kernels() {
    let dir = PathBuf::from(env_required(DIR_VAR));
    let arm = env_required(ARM_VAR);
    let out = PathBuf::from(env_required(OUT_VAR));

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "arena"))
        .filter(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            stem.starts_with("shader_")
                || stem == "psychedelic"
                // One representative cell-grid geometry: all three
                // dumped geometries are the same 623 reachable nodes.
                || stem == "cellgrid_80x24_d1"
                || stem.starts_with("glyph16_")
                || stem.starts_with("glyph32_")
        })
        .collect();
    files.sort();
    assert_eq!(
        files.len(),
        204,
        "expected the 204-kernel corpus in {}",
        dir.display()
    );

    let generous_timeout = generous();
    println!(
        "arm {arm}; clock-lifted deadline {:?}; host load at start: {}",
        generous_timeout,
        load_averages()
    );

    let mut csv = String::new();
    writeln!(
        csv,
        "kernel,group,nodes,tier,arm,prod_stop,prod_cost,prod_applications,prod_iterations,\
         prod_unions,prod_classes_after,prod_nodes,prod_elapsed_ms,\
         gen_stop,gen_cost,gen_applications,gen_iterations,gen_unions,gen_classes_after,\
         gen_nodes,gen_elapsed_ms"
    )
    .expect("write");

    for path in &files {
        let (name, raw_arena, raw_root) = load_arena(path);
        let group = name.split(':').next().expect("group prefix").to_string();
        let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(&raw_arena, raw_root)
            .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
        let node_count = reachable_count(&arena, root);
        let config = crate::egraph::saturate::config_for_node_count(node_count);

        // Regime 1: the production budget exactly as it ships.
        let prod = run(
            &arena,
            root,
            config.max_iterations,
            config.max_classes,
            config.hard_timeout,
        );
        // Regime 2: same iteration and class caps, clock lifted, so the
        // only thing that can differ between arms is the control flow.
        let generous = run(
            &arena,
            root,
            config.max_iterations,
            config.max_classes,
            generous_timeout,
        );

        writeln!(
            csv,
            "{},{},{},{},{},{:?},{},{},{},{},{},{},{:.3},{:?},{},{},{},{},{},{},{:.3}",
            csv_escape(&name),
            csv_escape(&group),
            node_count,
            tier_name(&config),
            csv_escape(&arm),
            prod.stop,
            prod.cost,
            prod.applications,
            prod.iterations,
            prod.total_unions,
            prod.classes_after,
            prod.extracted_nodes,
            prod.elapsed.as_secs_f64() * 1e3,
            generous.stop,
            generous.cost,
            generous.applications,
            generous.iterations,
            generous.total_unions,
            generous.classes_after,
            generous.extracted_nodes,
            generous.elapsed.as_secs_f64() * 1e3,
        )
        .expect("write");

        print!(".");
        use std::io::Write as _;
        std::io::stdout().flush().ok();
    }
    println!();
    println!("arm {arm}; host load at end: {}", load_averages());
    std::fs::write(&out, csv).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    println!("wrote {}", out.display());
}
