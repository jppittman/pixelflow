//! The structural-gap inventory: the REAL kernels production compiles beside
//! the SYNTHETIC generators every cost-model and Guide claim was minted on,
//! measured on the same deterministic columns so the two distributions can
//! be laid side by side (`docs/results/2026-09-07-corpus-structural-gaps.md`).
//!
//! Every column is a count, never a clock: arena nodes at construction and
//! after hash-consing (the splice-duplication factor), tree cost against
//! DAG cost of the input and of the extracted term (the sharing ratio), the
//! select/compare/gather/op-kind census, what production saturation did
//! (tier, stop reason, classes, applications, per-rule firing histogram from
//! the provenance journal) and what the emitter made of the result (bytes,
//! spill slots, hoisted values, schedule entries per scope of the collapse
//! nest, trip-weighted memory ops). The guard analysis is the emitter's own
//! `PIXELFLOW_GUARD_TELEMETRY` line on stderr; this binary prints a marker
//! before each compile so a reader can pair the two streams.
//!
//! Usage:
//! ```bash
//! PIXELFLOW_GUARD_TELEMETRY=1 cargo run --release -p pixelflow-pipeline --bin corpus_gaps -- \
//!   --dumps <dir>[,<dir>...] --out rows.csv --synthetic-n 24 2> guards.log
//! ```
//!
//! Rows are appended to `--out` as each kernel finishes, so a run that dies
//! leaves a usable partial table.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use pixelflow_ir::arena::{
    BufferDecl, BufferId, BufferIdentity, ExprNode, UniformDecl, UniformIdentity,
};
use pixelflow_ir::variance::LatticeShape;
use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_pipeline::collapse_bench::{self, LANES, corpus::Trips};
use pixelflow_pipeline::shader_bench::{NAMED_KERNEL_NAMES, SHADERTOY_KERNEL_NAMES, named_kernel};
use pixelflow_pipeline::training::{bezier_family, sh_family};
use pixelflow_search::egraph::{
    Budget, CostModel, InputSize, KeepJournal, Optimizer, RuleSet, Vocabulary,
    collect_rule_templates, insert, reachable_count,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

#[derive(Parser)]
struct Args {
    /// Comma-separated directories of `*.arena` dumps (the REAL population).
    #[arg(long)]
    dumps: String,
    /// CSV to append rows to (header written when the file is created).
    #[arg(long)]
    out: PathBuf,
    /// Kernels drawn per synthetic family configuration.
    #[arg(long, default_value_t = 24)]
    synthetic_n: usize,
    /// Seed for every synthetic draw stream.
    #[arg(long, default_value_t = 20260907)]
    seed: u64,
    /// Skip kernels with exactly this name (for reruns past a known panic).
    #[arg(long)]
    skip: Vec<String>,
}

/// `gen_bench_corpus`'s size bands, restated: (max_depth, leaf_prob, num_vars).
/// They are private to that binary; a drift here changes only which
/// synthetic sizes this inventory samples, never a corpus.
const BWD_BANDS: &[(usize, f32, usize)] = &[
    (2, 0.6, 1),
    (2, 0.5, 2),
    (3, 0.6, 1),
    (3, 0.5, 2),
    (3, 0.4, 4),
    (4, 0.5, 2),
    (4, 0.4, 4),
    (4, 0.3, 4),
    (5, 0.4, 2),
    (5, 0.3, 4),
    (5, 0.2, 4),
    (6, 0.35, 3),
    (6, 0.3, 4),
    (6, 0.2, 4),
    (7, 0.3, 4),
];

/// Extent a synthetic expression is emitted at. The claims were minted with
/// single-batch JIT tiles (no lattice at all); 64×64 is the smallest square
/// that gives every scope of the collapse nest a non-trivial trip count.
const SYNTHETIC_EXTENT: [u32; 2] = [64, 64];
/// `collapse_cost`'s shader extent.
const SHADER_EXTENT: [u32; 2] = [40, 45];
const SCENE_EXTENT: [u32; 2] = [1920, 1080];

struct Kernel {
    name: String,
    group: String,
    population: &'static str,
    arena: ExprArena,
    root: ExprId,
    extent: [u32; 2],
}

fn main() {
    let args = Args::parse();
    assert!(
        std::env::var_os("PIXELFLOW_GUARD_TELEMETRY").is_some(),
        "run with PIXELFLOW_GUARD_TELEMETRY=1 so the emitter's guard analysis lands on stderr"
    );
    let mut out = open_out(&args.out);
    let done: HashSet<String> = existing_names(&args.out);

    let mut kernels: Vec<Kernel> = Vec::new();
    for dir in args.dumps.split(',') {
        kernels.extend(load_dumps(Path::new(dir)));
    }
    kernels.extend(shadertoy_kernels());
    kernels.extend(synthetic_kernels(args.synthetic_n, args.seed));

    let rules = RuleSet::production();
    let total = kernels.len();
    for (i, k) in kernels.into_iter().enumerate() {
        if done.contains(&k.name) || args.skip.contains(&k.name) {
            continue;
        }
        eprintln!("corpus-gaps kernel={} ({}/{})", k.name, i + 1, total);
        let row = measure(&k, &rules);
        writeln!(out, "{row}").expect("write row");
        out.flush().expect("flush");
        println!("{}\t{}\tdone", k.name, k.population);
    }
}

// ============================================================================
// Populations
// ============================================================================

fn extent_for(group: &str, name: &str) -> [u32; 2] {
    match group {
        "glyph16" => [16, 16],
        "glyph32" => [32, 32],
        "shader" => SHADER_EXTENT,
        "psychedelic" | "scene" => SCENE_EXTENT,
        "cellgrid" => match name {
            "cellgrid:80x24_d1" => [800, 384],
            "cellgrid:80x24_d2" => [1600, 768],
            "cellgrid:120x40_d2" => [2400, 1280],
            other => panic!("unknown cell grid geometry {other}"),
        },
        other => panic!("unknown dump group {other} ({name})"),
    }
}

fn load_dumps(dir: &Path) -> Vec<Kernel> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "arena"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no *.arena files in {}", dir.display());
    files
        .iter()
        .filter_map(|p| {
            let (name, arena, root) = load_arena(p);
            let group = name.split(':').next().expect("group prefix").to_string();
            // The `shader:*` dumps predate the retirement of the Z/W axes
            // (they name `Var(2)`, which `emit::compile` refuses); the live
            // definitions in `shader_bench` are taken in-process instead.
            if group == "shader" {
                return None;
            }
            let extent = extent_for(&group, &name);
            Some(Kernel {
                name,
                group,
                population: "real",
                arena,
                root,
                extent,
            })
        })
        .collect()
}

/// The 12 withheld ShaderToy ports, from their live definitions, at
/// `collapse_cost`'s shader extent.
fn shadertoy_kernels() -> Vec<Kernel> {
    SHADERTOY_KERNEL_NAMES
        .iter()
        .map(|name| {
            let (arena, root) = named_kernel(name).expect("shadertoy kernel");
            Kernel {
                name: format!("shader:{name}"),
                group: "shader".to_string(),
                population: "real",
                arena,
                root,
                extent: SHADER_EXTENT,
            }
        })
        .collect()
}

/// Inverse of the dumpers' `dump_arena` (the `production_telemetry` loader
/// in `pixelflow-search/src/runtime.rs`, plus the `uni`/`Un` lines the scene
/// dumper writes for uniform-bearing kernels).
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
    let mut uniform_ids = Vec::new();
    let mut root = None;
    let mut next_id: u32 = 0;
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
                arena.declare_buffer(BufferDecl {
                    id: idents[ord],
                    width: w.parse().expect("buf width"),
                    height: h.parse().expect("buf height"),
                });
                continue;
            }
            ["uni", bits] => {
                let default = f32::from_bits(bits.parse().expect("uniform default bits"));
                uniform_ids.push(arena.declare_uniform(UniformDecl {
                    id: UniformIdentity::mint(),
                    default,
                }));
                continue;
            }
            ["root", r] => {
                root = Some(id(r));
                continue;
            }
            ["V", i] => arena.push_var(i.parse().expect("var index")),
            ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
            ["B", slot] => arena.push_buffer(BufferId(slot.parse().expect("buffer slot"))),
            ["Un", slot] => {
                let slot: usize = slot.parse().expect("uniform slot");
                let uid = *uniform_ids.get(slot).unwrap_or_else(|| {
                    panic!("{}: Un {slot} without a uni declaration", path.display())
                });
                arena.push_uniform(uid)
            }
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

fn synthetic_kernels(n: usize, seed: u64) -> Vec<Kernel> {
    let mut out = Vec::new();
    let synth =
        |name: String, group: &str, arena: ExprArena, root: ExprId, extent: [u32; 2]| Kernel {
            name,
            group: group.to_string(),
            population: "synthetic",
            arena,
            root,
            extent,
        };

    // gen_bench_corpus: BwdGenerator over the size bands, fused ops off.
    for (b, &(max_depth, leaf_prob, num_vars)) in BWD_BANDS.iter().enumerate() {
        let config = BwdGenConfig {
            max_depth,
            leaf_prob,
            num_vars,
            fused_op_prob: 0.0,
            ..BwdGenConfig::default()
        };
        let mut rng = BwdGenerator::new(
            seed ^ ((b as u64 + 1) * 0x9E37_79B9),
            config,
            collect_rule_templates(),
        );
        for i in 0..n {
            let pair = rng.generate_arena();
            out.push(synth(
                format!("bwd_band{b:02}_d{max_depth}:{i}"),
                "bwd_bands",
                pair.arena,
                pair.unoptimized,
                SYNTHETIC_EXTENT,
            ));
        }
    }
    // The extraction-objective probe's 96 BwdGenerator kernels: default
    // config at depths 4..11 (`synth_d{depth}_s{seed}`).
    for depth in [4usize, 6, 8, 11] {
        let config = BwdGenConfig {
            max_depth: depth,
            ..BwdGenConfig::default()
        };
        let mut rng = BwdGenerator::new(
            seed ^ (depth as u64) << 20,
            config,
            collect_rule_templates(),
        );
        for i in 0..n {
            let pair = rng.generate_arena();
            out.push(synth(
                format!("bwd_default_d{depth}:{i}"),
                "bwd_default",
                pair.arena,
                pair.unoptimized,
                SYNTHETIC_EXTENT,
            ));
        }
    }
    // gen_sh_corpus.
    let mut rng = sh_family::Rng::new(seed);
    for i in 0..n * 2 {
        let (arena, root) = sh_family::draw(&mut rng);
        out.push(synth(
            format!("sh:{i}"),
            "sh",
            arena,
            root,
            SYNTHETIC_EXTENT,
        ));
    }
    // gen_bezier_corpus.
    let mut rng = bezier_family::Lcg::new(seed);
    for i in 0..n * 2 {
        let (form, arena, root) = bezier_family::draw(&mut rng);
        out.push(synth(
            format!("bezier_{}:{i}", form.label()),
            "bezier",
            arena,
            root,
            SYNTHETIC_EXTENT,
        ));
    }
    // The five original named production kernels (gen_bench_corpus FINAL tier).
    for name in NAMED_KERNEL_NAMES {
        let (arena, root) = named_kernel(name).expect("named kernel");
        out.push(synth(
            format!("named:{name}"),
            "named",
            arena,
            root,
            SHADER_EXTENT,
        ));
    }
    // collapse_cost's synthetic allocation-pressure corpus, at its own extents.
    for k in collapse_bench::corpus::synthetic() {
        if k.extent[0] < LANES as u32 {
            continue;
        }
        out.push(synth(
            format!("collapse_{}:{}", k.family, k.name),
            "collapse_synth",
            k.arena,
            k.root,
            k.extent,
        ));
    }
    out
}

// ============================================================================
// The measurement
// ============================================================================

const HEADER: &str = "name,group,population,extent_w,extent_h,\
nodes_reachable,nodes_hashcons,splice_factor,tree_nodes,\
selects,compares,compares_feeding_selects,select_masks_shared,arm_true_med,arm_false_med,arm_excl_true_med,arm_excl_false_med,arm_excl_frac,\
gathers,buffers,uniforms,transcendentals,depth,ops,\
nodes_lowered,nodes_lowered_hc,input_dag_cost,input_tree_cost,input_sharing,tier_iters,class_cap,stop,iterations,applications,unions,classes,opt_ms,\
ext_nodes,ext_trip_tree_cost,ext_trip_dag_cost,ext_trip_sharing,ext_tree_cost,ext_dag_cost,ext_sharing,ext_selects,dag_cost_delta_pct,\
emit,bytes,spill_slots,hoisted,carried,frame_instr,row_instr,body_instr,frame_frac,row_frac,body_frac,frame_mem,row_mem,body_mem,body_remats,dyn_memory_ops,dyn_instructions,\
rule_hist";

fn open_out(path: &Path) -> std::fs::File {
    let fresh = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    if fresh {
        writeln!(f, "{HEADER}").expect("write header");
    }
    f
}

fn existing_names(path: &Path) -> HashSet<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HashSet::new();
    };
    text.lines()
        .skip(1)
        .filter_map(|l| l.split(',').next().map(str::to_string))
        .collect()
}

/// A structurally hash-consed copy of the reachable subgraph: the node
/// multiset the e-graph's own interning sees. Buffers and uniforms keep
/// their declarations (identity-equal slots stay one node).
fn hash_cons(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }
    let mut out = ExprArena::new();
    for decl in arena.buffers() {
        let _ = out.declare_buffer(*decl);
    }
    for decl in arena.uniforms() {
        let _ = out.declare_uniform(*decl);
    }
    #[derive(Hash, PartialEq, Eq)]
    enum Key {
        Var(u8),
        Const(u32),
        Param(u8),
        Buffer(u16),
        Uniform(u16),
        Op(OpKind, Vec<u32>),
        /// A fold's identity is its metadata plus its body — the bits are
        /// the metadata, and two folds sharing them fold the same way.
        Reduce(u64, u32),
    }
    type Build = Box<dyn Fn(&mut ExprArena) -> ExprId>;
    let mut interned: HashMap<Key, ExprId> = HashMap::new();
    let mut map: Vec<u32> = vec![u32::MAX; len];
    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        let m = |c: ExprId, map: &[u32]| {
            let d = map[c.0 as usize];
            assert_ne!(d, u32::MAX, "hash_cons: child after parent");
            d
        };
        let (key, build): (Key, Build) = match *arena.node(id) {
            ExprNode::Var(i) => (Key::Var(i), Box::new(move |a| a.push_var(i))),
            ExprNode::Const(v) => (Key::Const(v.to_bits()), Box::new(move |a| a.push_const(v))),
            ExprNode::Param(i) => (Key::Param(i), Box::new(move |a| a.push_param(i))),
            ExprNode::Buffer(b) => (Key::Buffer(b.0), Box::new(move |a| a.push_buffer(b))),
            ExprNode::Uniform(u) => (Key::Uniform(u.0), Box::new(move |a| a.push_uniform(u))),
            ExprNode::Ref(k) => panic!(
                "hash_cons: Ref({k:?}) names a kernel interned in this process; \
                 corpus arenas are self-contained, so expand_refs first"
            ),
            ExprNode::Reduce { fold, body } => {
                let body = ExprId(m(body, &map));
                (
                    Key::Reduce(fold.to_bits(), body.0),
                    Box::new(move |a: &mut ExprArena| a.push_reduce(fold, body)),
                )
            }
            ExprNode::Unary(k, c) => {
                let c = ExprId(m(c, &map));
                (Key::Op(k, vec![c.0]), Box::new(move |a| a.push_unary(k, c)))
            }
            ExprNode::Binary(k, x, y) => {
                let (x, y) = (ExprId(m(x, &map)), ExprId(m(y, &map)));
                (
                    Key::Op(k, vec![x.0, y.0]),
                    Box::new(move |a| a.push_binary(k, x, y)),
                )
            }
            ExprNode::Ternary(k, x, y, z) => {
                let (x, y, z) = (ExprId(m(x, &map)), ExprId(m(y, &map)), ExprId(m(z, &map)));
                (
                    Key::Op(k, vec![x.0, y.0, z.0]),
                    Box::new(move |a| a.push_ternary(k, x, y, z)),
                )
            }
            ExprNode::Nary(k, start, n) => {
                let kids: Vec<ExprId> = arena
                    .nary_children_slice(start, n)
                    .iter()
                    .map(|&c| ExprId(m(c, &map)))
                    .collect();
                let raw: Vec<u32> = kids.iter().map(|c| c.0).collect();
                (Key::Op(k, raw), Box::new(move |a| a.push_nary(k, &kids)))
            }
        };
        let new_id = *interned.entry(key).or_insert_with(|| build(&mut out));
        map[idx] = new_id.0;
    }
    (out, ExprId(map[root.0 as usize]))
}

fn reach_set(arena: &ExprArena, root: ExprId) -> HashSet<u32> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.0) {
            continue;
        }
        stack.extend(arena.children(id));
    }
    seen
}

fn median_usize(v: &mut [usize]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] + v[n / 2]) as f64 / 2.0
    }
}

/// Tree count with multiplicity (a spliced subterm counted once per use)
/// and the latency-prior tree cost, both saturating.
fn tree_figures(arena: &ExprArena, root: ExprId, costs: &CostModel) -> (u128, u128) {
    let len = arena.nodes_raw().len();
    let mut memo: Vec<Option<(u128, u128)>> = vec![None; len];
    let mut order = Vec::new();
    let mut stack = vec![root];
    let mut seen = vec![false; len];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        order.push(id);
        stack.extend(arena.children(id));
    }
    // Children have smaller ids than parents (push order), so ascending id
    // is a valid evaluation order.
    order.sort_unstable();
    for id in order {
        let mut count: u128 = 1;
        let mut cost: u128 = costs.cost(arena.kind(id)) as u128;
        for c in arena.children(id) {
            let (cc, cs) = memo[c.0 as usize].expect("child before parent");
            count = count.saturating_add(cc);
            cost = cost.saturating_add(cs);
        }
        memo[id.0 as usize] = Some((count, cost));
    }
    memo[root.0 as usize].expect("root")
}

fn dag_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> u128 {
    reach_set(arena, root)
        .into_iter()
        .map(|i| costs.cost(arena.kind(ExprId(i))) as u128)
        .sum()
}

fn is_compare(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne
    )
}

fn is_transcendental(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::Sin
            | OpKind::Cos
            | OpKind::Tan
            | OpKind::Asin
            | OpKind::Acos
            | OpKind::Atan
            | OpKind::Atan2
            | OpKind::Exp
            | OpKind::Exp2
            | OpKind::Ln
            | OpKind::Log2
            | OpKind::Log10
            | OpKind::Pow
    )
}

struct Census {
    selects: usize,
    compares: usize,
    compares_feeding_selects: usize,
    select_masks_shared: usize,
    arm_true_med: f64,
    arm_false_med: f64,
    arm_excl_true_med: f64,
    arm_excl_false_med: f64,
    arm_excl_frac: f64,
    gathers: usize,
    transcendentals: usize,
    ops: String,
}

/// The select/compare/op census over the hash-consed graph.
fn census(arena: &ExprArena, root: ExprId) -> Census {
    let nodes = reach_set(arena, root);
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut selects = Vec::new();
    let mut compares = 0;
    let mut gathers = 0;
    let mut transcendentals = 0;
    for &i in &nodes {
        let id = ExprId(i);
        let k = arena.kind(id);
        *hist.entry(format!("{k:?}")).or_default() += 1;
        if is_compare(k) {
            compares += 1;
        }
        if is_transcendental(k) {
            transcendentals += 1;
        }
        if k == OpKind::Gather || k == OpKind::RawGather {
            gathers += 1;
        }
        if let &ExprNode::Ternary(OpKind::Select, m, a, b) = arena.node(id) {
            selects.push((m, a, b));
        }
    }
    let mut mask_uses: HashMap<u32, usize> = HashMap::new();
    let mut cfs = 0;
    let mut arm_t = Vec::new();
    let mut arm_f = Vec::new();
    let mut ex_t = Vec::new();
    let mut ex_f = Vec::new();
    let mut ex_total = 0usize;
    let mut arm_total = 0usize;
    for &(m, a, b) in &selects {
        *mask_uses.entry(m.0).or_default() += 1;
        if is_compare(arena.kind(m)) {
            cfs += 1;
        }
        let ra = reach_set(arena, a);
        let rb = reach_set(arena, b);
        let ea = ra.difference(&rb).count();
        let eb = rb.difference(&ra).count();
        arm_total += ra.len() + rb.len();
        ex_total += ea + eb;
        arm_t.push(ra.len());
        arm_f.push(rb.len());
        ex_t.push(ea);
        ex_f.push(eb);
    }
    let ops = hist
        .iter()
        .map(|(k, n)| format!("{k}:{n}"))
        .collect::<Vec<_>>()
        .join(";");
    Census {
        selects: selects.len(),
        compares,
        compares_feeding_selects: cfs,
        select_masks_shared: mask_uses.values().filter(|&&n| n > 1).count(),
        arm_true_med: median_usize(&mut arm_t),
        arm_false_med: median_usize(&mut arm_f),
        arm_excl_true_med: median_usize(&mut ex_t),
        arm_excl_false_med: median_usize(&mut ex_f),
        arm_excl_frac: if arm_total == 0 {
            0.0
        } else {
            ex_total as f64 / arm_total as f64
        },
        gathers,
        transcendentals,
        ops,
    }
}

fn measure(k: &Kernel, rules: &RuleSet) -> String {
    let costs = CostModel::latency_prior();
    let mut row = String::new();
    write!(
        row,
        "{},{},{},{},{}",
        k.name, k.group, k.population, k.extent[0], k.extent[1]
    )
    .expect("fmt");

    // ---- construction ----
    let nodes_reachable = reachable_count(&k.arena, k.root);
    let (hc, hc_root) = hash_cons(&k.arena, k.root);
    let nodes_hashcons = reachable_count(&hc, hc_root);
    let (tree_nodes, _) = tree_figures(&k.arena, k.root, &costs);
    write!(
        row,
        ",{nodes_reachable},{nodes_hashcons},{:.3},{tree_nodes}",
        nodes_reachable as f64 / nodes_hashcons as f64,
    )
    .expect("fmt");

    let c = census(&hc, hc_root);
    write!(
        row,
        ",{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.3},{},{},{},{},{},{}",
        c.selects,
        c.compares,
        c.compares_feeding_selects,
        c.select_masks_shared,
        c.arm_true_med,
        c.arm_false_med,
        c.arm_excl_true_med,
        c.arm_excl_false_med,
        c.arm_excl_frac,
        c.gathers,
        k.arena.buffers().len(),
        k.arena.uniforms().len(),
        c.transcendentals,
        hc.depth(hc_root),
        c.ops
    )
    .expect("fmt");

    // ---- production saturation (runtime.rs: LowerDwrt, ExpandReduce, Saturate::runtime) ----
    let (lowered, lowered_root) = pixelflow_ir::passes::lower_dwrt_owned(&k.arena, k.root)
        .unwrap_or_else(|e| panic!("{}: lower_dwrt failed: {e}", k.name));
    let (lowered, lowered_root) = pixelflow_ir::passes::expand_reduce_owned(&lowered, lowered_root);
    let node_count = reachable_count(&lowered, lowered_root);
    // Costs are priced on the LOWERED term — what the e-graph is handed —
    // so the input and extracted columns share units (`Dwrt` is expanded
    // by lowering; pricing it as one op would make the pair incomparable).
    let (lhc, lhc_root) = hash_cons(&lowered, lowered_root);
    let nodes_lowered_hc = reachable_count(&lhc, lhc_root);
    let (_, input_tree_cost) = tree_figures(&lowered, lowered_root, &costs);
    let input_dag_cost = dag_cost(&lhc, lhc_root, &costs);
    let mut optimizer = Optimizer::production()
        .for_lattice(LatticeShape::new(k.extent))
        .observe(Some(Box::new(KeepJournal)));
    let mut egraph = optimizer.egraph();
    let root_class = insert(&lowered, lowered_root, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|_| panic!("{}: not e-graph representable", k.name));
    let limits = Budget::Production.limits(InputSize {
        nodes: node_count,
        classes: egraph.num_classes(),
    });
    let started = Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let opt_ms = started.elapsed().as_secs_f64() * 1e3;
    let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);
    let s = &optimized.stats;
    write!(
        row,
        ",{node_count},{nodes_lowered_hc},{input_dag_cost},{input_tree_cost},{:.3},{},{},{:?},{},{},{},{},{:.1}",
        if input_dag_cost == 0 { 1.0 } else { input_tree_cost as f64 / input_dag_cost as f64 },
        limits.iterations, limits.classes, s.stop, s.iterations, s.applications, s.unions, s.classes, opt_ms
    )
    .expect("fmt");

    let ext_nodes = reachable_count(&extracted, extracted_root);
    let ext_census = census(&extracted, extracted_root);
    // `ChoiceCost` is the objective the DP minimized — trip-weighted by the
    // lattice shape since `for_lattice` — so its tree/dag pair is reported as
    // is, and the extracted term is ALSO priced with the flat latency prior
    // (`tree_figures`/`dag_cost`, the input's units) so the two columns
    // "input dag cost" and "extracted dag cost" are comparable.
    let trip_dag = optimized.cost.dag;
    let trip_tree = optimized.cost.tree;
    let (_, ext_tree) = tree_figures(&extracted, extracted_root, &costs);
    let ext_dag = dag_cost(&extracted, extracted_root, &costs);
    write!(
        row,
        ",{ext_nodes},{trip_tree},{trip_dag},{:.3},{ext_tree},{ext_dag},{:.3},{},{:.2}",
        if trip_dag == 0 {
            1.0
        } else {
            trip_tree as f64 / trip_dag as f64
        },
        if ext_dag == 0 {
            1.0
        } else {
            ext_tree as f64 / ext_dag as f64
        },
        ext_census.selects,
        if input_dag_cost == 0 {
            0.0
        } else {
            (ext_dag as f64 - input_dag_cost as f64) / input_dag_cost as f64 * 100.0
        }
    )
    .expect("fmt");

    // Per-rule firing histogram from the provenance journal.
    let mut fires: HashMap<usize, u64> = HashMap::new();
    for (_, rec) in egraph.provenance().applications() {
        *fires.entry(rec.rule_idx).or_default() += 1;
    }
    let mut fires: Vec<(usize, u64)> = fires.into_iter().collect();
    fires.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let rule_hist = fires
        .iter()
        .map(|(idx, n)| {
            format!(
                "{}:{n}",
                rules
                    .label_of(*idx)
                    .unwrap_or_else(|| format!("rule#{idx}"))
            )
        })
        .collect::<Vec<_>>()
        .join(";");

    // ---- emit (Saturate re-splices buffers onto the input's slot order) ----
    let (emit_arena, emit_root) = if lowered.buffers().is_empty() {
        (extracted, extracted_root)
    } else {
        let mut ordered = ExprArena::new();
        for decl in lowered.buffers() {
            let _slot = ordered.declare_buffer(*decl);
        }
        let r = ordered.splice(&extracted, extracted_root);
        (ordered, r)
    };
    eprintln!("corpus-gaps emit={}", k.name);
    // `emit::compile` refuses an arena naming the retired Z/W axes
    // (`Var(2)`/`Var(3)`): production has two coordinate axes and uniforms,
    // and a generator that draws from four "variables" builds kernels the
    // emitter cannot take. Recorded as its own outcome rather than a panic —
    // the fraction of a population that cannot even be emitted is a column.
    let emit_result = if emit_arena.retired_axis(emit_root).is_some() {
        Err(None)
    } else {
        pixelflow_codegen::emit::compile(&emit_arena, emit_root).map_err(Some)
    };
    match emit_result {
        Ok(res) => {
            let t = &res.traffic;
            let trips = Trips::of(k.extent, LANES as u32);
            let feats = collapse_bench::features_of(&res, trips);
            let total =
                f64::from(t.frame.instructions + t.row.instructions + t.body.instructions).max(1.0);
            write!(
                row,
                ",ok,{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{},{},{},{},{},{}",
                feats.bytes_total,
                res.spill_count,
                res.hoisted_values,
                t.carried,
                t.frame.instructions,
                t.row.instructions,
                t.body.instructions,
                f64::from(t.frame.instructions) / total,
                f64::from(t.row.instructions) / total,
                f64::from(t.body.instructions) / total,
                feats.frame.memory_ops(),
                feats.row.memory_ops(),
                feats.body.memory_ops(),
                t.body.remats,
                feats.dyn_memory_ops,
                feats.dyn_instructions,
            )
            .expect("fmt");
        }
        Err(None) => {
            write!(row, ",retired_axis,,,,,,,,,,,,,,,,").expect("fmt");
        }
        Err(Some(e)) => {
            eprintln!("corpus-gaps emit-failed={} err={e:?}", k.name);
            write!(row, ",err,,,,,,,,,,,,,,,,").expect("fmt");
        }
    }
    write!(row, ",{rule_hist}").expect("fmt");
    row
}
