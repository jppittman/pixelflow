//! Mine the extraction witnesses and find the divergence.
//!
//! The e-graph is monotone: a run at a larger budget performs every
//! application the smaller run performed and then more, so the larger graph
//! represents a superset of the terms. When the extractor's own output at a
//! *smaller* budget is cheaper than its output at a larger one — which the
//! 2026-09-08 class-cap sweep found on every family — that cheaper term is
//! provably present in the bigger graph and the extractor walked past it.
//! This harness takes those terms, maps them into the bigger graph by
//! hash-cons lookup, diffs the two choice maps, and reports at which class,
//! and by which extraction stage, the witness was let go.
//!
//! Denotation (written first, not revised after the run):
//! `docs/plans/2026-09-08-extraction-witnesses.md`.
//! Results: `docs/results/2026-09-08-extraction-witnesses.{md,csv,json}`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use pixelflow_core::{Bits, Kernel, Uniform};
use pixelflow_graphics::fonts::Font;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::pixel::Pixel;
use pixelflow_graphics::scene3d::{Hit, Plane, Ray, Rgba, Sphere, checker, sky};
use pixelflow_ir::optimize::{Optimize, Rewritten};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{ExprArena, ExprId, ExprNode, LatticeShape};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::egraph::optimizer::KeepJournal;
use pixelflow_search::egraph::provenance::{ApplicationId, Origin};
use pixelflow_search::egraph::witness::{self, Stage, Ties};
use pixelflow_search::egraph::{
    Budget, CostModel, EClassId, EGraph, ENode, Optimized, Optimizer, Vocabulary,
    config_for_node_count, insert, reachable_count,
};

const SCREEN: [u32; 2] = [1920, 1080];
const SHADER_EXTENT: [u32; 2] = [256, 256];
const CELL_HEIGHT_PT: f32 = 16.0;
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
const APPLICATIONS_PER_CLASS: u64 = 40;

/// The budget ladder, in class caps — the sweep's `cap{b}-app{40b}` arms.
const GLYPH_LADDER: [usize; 3] = [5_000, 10_000, 20_000];
const BIG_LADDER: [usize; 5] = [5_000, 10_000, 20_000, 50_000, 100_000];

#[derive(Parser, Debug)]
#[command(about = "Extraction witnesses: where the extractor walks past a term it holds")]
struct Cli {
    /// The production font, for the glyph families.
    #[arg(
        long,
        default_value = "pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf"
    )]
    font: PathBuf,
    /// Substring filter on kernel name.
    #[arg(long)]
    filter: Option<String>,
    /// Where the three result files land.
    #[arg(long, default_value = "docs/results")]
    out: PathBuf,
    /// Cap on glyphs per density, so a smoke run is minutes not hours.
    #[arg(long, default_value_t = usize::MAX)]
    max_glyphs: usize,
    /// Include the held-out chrome scene. Reported once, separately.
    #[arg(long)]
    chrome: bool,
}

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

struct Case {
    name: String,
    family: String,
    kernel: Kernel,
    extent: [u32; 2],
    ladder: &'static [usize],
}

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

fn rgba8_shifts() -> [u32; 4] {
    <Rgba8 as Pixel>::packed_shifts().expect("Rgba8 packs")
}

/// `egraph_off_on`'s copy of `pixelflow_graphics::render::packed`'s
/// `pub(crate)` byte pack — the same construction, so the two harnesses'
/// chrome rows are the same kernel.
fn packed_kernel(color: &Rgba, shifts: [u32; 4]) -> Kernel {
    color
        .fold(
            &|channels: &[Kernel; 4]| {
                let byte = |c: usize| {
                    channels[c]
                        .mul(&k(255.0))
                        .clamp(&k(0.0), &k(255.0))
                        .trunc_to_int()
                        .shl(shifts[c])
                };
                byte(0).or(&byte(1)).or(&byte(2)).or(&byte(3))
            },
            &|mask, if_true: Bits, if_false: Bits| Bits::select(mask, &if_true, &if_false),
        )
        .into_kernel()
}

fn chrome_color() -> Rgba {
    const CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
    const RADIUS: f32 = 1.0;
    const FLOOR: f32 = -1.0;
    fn world(ray: &Ray) -> Rgba {
        let floor = Plane::at_height(k(FLOOR)).hit(ray);
        floor.select(
            &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
            &sky(ray),
        )
    }
    let ray = Ray::through_screen(SCREEN[0] as f32, SCREEN[1] as f32);
    let sphere: Hit = Sphere::new([k(CENTER.0), k(CENTER.1), k(CENTER.2)], k(RADIUS)).hit(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&world(&mirrored), &world(&ray))
}

fn psych_channel(y_weight: f32, clock: Uniform) -> Kernel {
    let scale = 2.0 / 1080.0;
    let x = Kernel::x().sub(&k(960.0)).mul(&k(scale));
    let y = k(540.0).sub(&Kernel::y()).mul(&k(scale));
    let time = clock.kernel().add(&k(1.3));
    let r_sq = x.mul(&x).add(&y.mul(&y));
    let radial = r_sq.sub(&k(0.7)).abs();
    let swirl_scale = k(1.0).sub(&radial).mul(&k(5.0));
    let vx = x.mul(&swirl_scale);
    let vy = y.mul(&swirl_scale);
    let phase = time.mul(&k(0.5));
    let sin_w03 = time.mul(&k(0.3)).sin();
    let sin_w20 = time.mul(&k(2.0)).sin();
    let vxp = vx.add(&phase);
    let swirl = vxp
        .sin()
        .add(&k(1.0))
        .mul(&vxp.sub(&vy.add(&phase.mul(&k(0.7)))).abs())
        .mul(&k(0.2))
        .add(&k(0.001));
    let pulse = k(1.0).add(&sin_w20.mul(&k(0.1)));
    let radial_factor = radial.mul(&k(-4.0)).mul(&pulse).exp();
    let raw = y
        .mul(&k(y_weight))
        .add(&sin_w03.mul(&k(0.2)))
        .exp()
        .mul(&radial_factor)
        .div(&swirl);
    raw.div(&raw.abs().add(&k(1.0))).add(&k(1.0)).mul(&k(0.5))
}

fn corpus(cli: &Cli) -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();
    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name).expect("registered shader");
        out.push(Case {
            name: format!("shader_{name}"),
            family: "shader".into(),
            kernel: Kernel::from_parts(arena, root),
            extent: SHADER_EXTENT,
            ladder: &BIG_LADDER,
        });
    }
    let clock = Uniform::new(0.0);
    out.push(Case {
        name: "psychedelic_packed".into(),
        family: "psychedelic".into(),
        kernel: packed_kernel(
            &Rgba::from([
                psych_channel(1.0, clock),
                psych_channel(-1.0, clock),
                psych_channel(-2.0, clock),
                k(1.0),
            ]),
            rgba8_shifts(),
        ),
        extent: SCREEN,
        ladder: &BIG_LADDER,
    });
    if cli.chrome {
        out.push(Case {
            name: "chrome_packed".into(),
            family: "chrome".into(),
            kernel: packed_kernel(&chrome_color(), rgba8_shifts()),
            extent: SCREEN,
            ladder: &BIG_LADDER,
        });
    }

    let data =
        std::fs::read(&cli.font).unwrap_or_else(|e| panic!("read {}: {e}", cli.font.display()));
    let parsed = Font::parse(&data).expect("parse the production font");
    for density in [1.0f32, 2.0f32] {
        let tile = (CELL_HEIGHT_PT * density).round().max(1.0) as u32;
        let mut n = 0usize;
        for ch in WARM_RANGE {
            if n >= cli.max_glyphs {
                break;
            }
            // `glyph_kernel_scaled` yields a `Glyph` — its two folds and its
            // support — and `kernel()` is the single exit that applies the
            // coverage ramp once. See fonts/loop_blinn.rs.
            let Some(kernel) = parsed
                .glyph_kernel_scaled(ch, tile as f32)
                .map(|g| g.kernel())
            else {
                continue;
            };
            n += 1;
            out.push(Case {
                name: format!("glyph{tile}_U{:04X}", ch as u32),
                family: format!("glyph{tile}"),
                kernel,
                extent: [tile, tile],
                ladder: &GLYPH_LADDER,
            });
        }
    }

    if let Some(f) = cli.filter.as_deref() {
        out.retain(|c| c.name.contains(f));
    }
    out
}

// ---------------------------------------------------------------------------
// Costs
// ---------------------------------------------------------------------------

fn reachable(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
    let mut seen = vec![false; arena.nodes_raw().len()];
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        out.push(id);
        stack.extend(arena.children(id));
    }
    out
}

/// The sweep's column: the latency prior over the emitted arena's reachable
/// op nodes, each once, **unweighted**. A property of a term, so exact and
/// reproducible.
fn dag_cost(arena: &ExprArena, root: ExprId) -> usize {
    let model = CostModel::latency_prior();
    reachable(arena, root)
        .into_iter()
        .filter_map(|id| match arena.node(id) {
            ExprNode::Unary(k, _) | ExprNode::Binary(k, _, _) | ExprNode::Ternary(k, _, _, _) => {
                Some(model.cost(*k))
            }
            _ => None,
        })
        .sum()
}

fn legalize(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    match pixelflow_ir::pipeline![LowerDwrt, ExpandReduce].optimize(arena, root) {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("legalizing prefix declined a real kernel"),
    }
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct BudgetRow {
    kernel: String,
    family: String,
    class_cap: usize,
    inserted_classes: usize,
    classes_at_stop: usize,
    live_classes: usize,
    applications: u64,
    stop: String,
    objective: usize,
    dag_cost: usize,
    tree_cost: usize,
    extraction_objective: String,
    /// The same extraction with ties broken canonically instead of by
    /// insertion order.
    canonical_objective: usize,
    canonical_dag_cost: usize,
    /// Classes where two or more candidates shared the winning DP cost, so
    /// insertion order decided — counted on the winning arm.
    tied_classes: usize,
    seconds: f64,
}

#[derive(Serialize, Clone, Debug)]
struct WitnessRow {
    kernel: String,
    family: String,
    b_lo: usize,
    b_hi: usize,
    live_lo: usize,
    live_hi: usize,
    objective_witness: usize,
    objective_greedy: usize,
    delta_objective: i64,
    dag_witness: usize,
    dag_greedy: usize,
    delta_dag: i64,
    /// Distinct classes the witness occupies in `G(b_hi)`.
    occupied: usize,
    merges: usize,
    divergences: usize,
    frontier: usize,
    /// `objective` of the best term one accepted frontier swap reaches.
    best_single_swap: usize,
    /// `objective` after greedily accepting every improving frontier swap.
    greedy_swaps: usize,
    swaps_accepted: usize,
    /// `objective` when the witness is cheaper in the extractor's own
    /// weighted objective; `static-only` when it is cheaper only in the
    /// sweep's unweighted `dag_cost` column.
    witness_kind: String,
    realizable: String,
    first_divergence: Option<FirstDivergence>,
    labels: BTreeMap<String, usize>,
}

#[derive(Serialize, Clone, Debug)]
struct FirstDivergence {
    class: u32,
    node_greedy: String,
    node_witness: String,
    dp_cost_greedy: String,
    dp_cost_witness: String,
    own_greedy: usize,
    own_witness: usize,
    tree_greedy: String,
    tree_witness: String,
    shared_greedy: String,
    shared_witness: String,
    rule_greedy: String,
    stage: String,
    swap_delta: i64,
    label: String,
}

#[derive(Serialize, Clone, Debug)]
struct DistractorRow {
    rule: String,
    count: usize,
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

struct Saturated {
    egraph: EGraph,
    optimized: Optimized,
    root: EClassId,
    inserted: usize,
    rule_of: HashMap<ApplicationId, String>,
}

fn saturate_at(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
    class_cap: usize,
) -> Saturated {
    let node_count = reachable_count(arena, root);
    let tier = config_for_node_count(node_count);
    let mut optimizer = Optimizer::production()
        .for_lattice(shape)
        .budget(Budget::Explicit {
            iterations: tier.max_iterations,
            classes: class_cap,
            applications: Some(class_cap as u64 * APPLICATIONS_PER_CLASS),
        })
        // The ladder deliberately runs past production's own caps; the
        // safety ceiling is an assertion about production's budget, and
        // firing it here would say nothing about the extractor.
        .no_ceiling()
        .observe(Some(Box::new(KeepJournal)));
    let mut egraph = optimizer.egraph();
    let root_class = insert(arena, root, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|d| panic!("kernel not representable: {d:?}"));
    let inserted = egraph.num_classes();
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let rules = optimizer.rule_set();
    let rule_of: HashMap<ApplicationId, String> = egraph
        .provenance()
        .applications()
        .map(|(id, rec)| {
            (
                id,
                rules
                    .label_of(rec.rule_idx)
                    .unwrap_or_else(|| format!("#{}", rec.rule_idx)),
            )
        })
        .collect();
    Saturated {
        egraph,
        optimized,
        root: root_class,
        inserted,
        rule_of,
    }
}

fn node_label(egraph: &EGraph, class: EClassId, idx: usize) -> String {
    let nodes = egraph.nodes(class);
    let Some(node) = nodes.get(idx) else {
        return format!("<idx {idx} of {}>", nodes.len());
    };
    match node {
        ENode::Var(i) => format!("Var({i})"),
        ENode::Const(bits) => format!("Const({})", f32::from_bits(*bits)),
        ENode::Buffer(_) => "Buffer".into(),
        ENode::Uniform(_) => "Uniform".into(),
        ENode::Param(i) => format!("Param({i})"),
        ENode::Op { op, children } => {
            let cs: Vec<String> = children
                .iter()
                .map(|&c| egraph.find(c).index().to_string())
                .collect();
            format!("{}({})", op.name(), cs.join(","))
        }
        ENode::Reduce { fold, body } => format!(
            "Reduce[{}..{}]({})",
            fold.range().start,
            fold.range().end,
            egraph.find(*body).index()
        ),
    }
}

fn show(cost: Option<usize>) -> String {
    match cost {
        None => "-".into(),
        Some(c) if c >= witness::cycle_cost() => "CYCLE".into(),
        Some(c) => c.to_string(),
    }
}

/// The rule that minted node `idx` of `class`, or `seed`.
fn minting_rule(sat: &Saturated, class: EClassId, idx: usize) -> String {
    let tags = sat.egraph.tags(class);
    let Some(&tag) = tags.get(idx) else {
        return "unknown".into();
    };
    match sat.egraph.provenance().origin(tag) {
        None => "unknown".into(),
        Some(Origin::Seed) => "seed".into(),
        Some(Origin::Rule(app)) => sat
            .rule_of
            .get(&app)
            .cloned()
            .unwrap_or_else(|| "rule?".into()),
    }
}

/// One `(b_lo, b_hi)` pair to explain: the term the smaller budget produced,
/// the graph and choices the larger one did, and the ladder rows both were
/// measured in.
struct Pair<'a> {
    case: &'a Case,
    shape: LatticeShape,
    lo: &'a BudgetRow,
    hi: &'a BudgetRow,
    sat: &'a Saturated,
    witness_term: &'a (ExprArena, ExprId),
    costs: &'a CostModel,
}

#[allow(clippy::too_many_lines)]
fn analyse(p: &Pair<'_>, distractors: &mut BTreeMap<String, usize>) -> Option<WitnessRow> {
    let Pair {
        case,
        shape,
        lo,
        hi,
        sat,
        witness_term,
        costs,
    } = *p;
    let (b_lo, b_hi) = (lo.class_cap, hi.class_cap);
    let egraph = &sat.egraph;
    let induced =
        match witness::induce(egraph, &witness_term.0, witness_term.1, Vocabulary::Runtime) {
            Ok(i) => i,
            Err(miss) => {
                // Monotonicity says this cannot happen. Loud, and the pair is
                // dropped rather than analysed against a term that is not there.
                eprintln!(
                    "extraction_witnesses: {} b_lo={b_lo} b_hi={b_hi}: witness subterm {} absent \
                 from the bigger graph after {} mapped nodes — monotonicity violated, or the \
                 two runs normalized differently",
                    case.name, miss.node, miss.mapped
                );
                return None;
            }
        };
    assert_eq!(
        egraph.find(induced.root),
        egraph.find(sat.root),
        "{}: the witness's root landed in a different class than the graph's root",
        case.name
    );

    let c_g = &sat.optimized.choices;
    // `C_T` is total only on the classes the witness occupies; every class
    // it reaches is one of those, and everything else falls back to greedy's
    // pick so a swap map is always well defined.
    let mut c_t = induced.choices.clone();
    for (i, g) in c_g.iter().enumerate() {
        if c_t[i].is_none() {
            c_t[i] = *g;
        }
    }
    let cost_t = witness::cost_if_well_founded(egraph, sat.root, &c_t, costs, shape)?;

    let trace = witness::trace(egraph, sat.root, costs, shape);
    assert_eq!(
        trace.cost.dag, sat.optimized.cost.dag,
        "{}: the traced extraction disagrees with the shipped one",
        case.name
    );

    let reach_t = witness::reachable_under(egraph, sat.root, &c_t);
    let divergent: Vec<EClassId> = reach_t
        .iter()
        .copied()
        .filter(|c| c_t[c.index()] != c_g[c.index()])
        .collect();
    // The frontier: divergent classes none of whose `C_T`-descendants
    // diverge. `reach_t` is post-order, so a class's descendants precede it.
    let is_div = {
        let mut v = vec![false; egraph.num_classes()];
        for c in &divergent {
            v[c.index()] = true;
        }
        v
    };
    let mut below_diverges = vec![false; egraph.num_classes()];
    for &c in &reach_t {
        let idx = c_t[c.index()].expect("reachable under C_T");
        let mut any = false;
        for &ch in egraph.nodes(c)[idx].children_slice() {
            let ch = egraph.find(ch);
            any |= is_div[ch.index()] || below_diverges[ch.index()];
        }
        below_diverges[c.index()] = any;
    }
    let frontier: Vec<EClassId> = divergent
        .iter()
        .copied()
        .filter(|c| !below_diverges[c.index()])
        .collect();

    // Swap search from greedy's term.
    let base = sat.optimized.cost.dag;
    let mut best_single = base;
    let mut best_single_class = None;
    let mut swap_delta: HashMap<u32, i64> = HashMap::new();
    for &c in &frontier {
        let mut m = c_g.clone();
        m[c.index()] = c_t[c.index()];
        let Some(cost) = witness::cost_if_well_founded(egraph, sat.root, &m, costs, shape) else {
            continue;
        };
        swap_delta.insert(c.index() as u32, cost.dag as i64 - base as i64);
        if cost.dag < best_single {
            best_single = cost.dag;
            best_single_class = Some(c);
        }
    }
    let _ = best_single_class;
    // Greedy: keep accepting the best improving frontier swap.
    let mut cur = c_g.clone();
    let mut cur_cost = base;
    let mut accepted = 0usize;
    loop {
        let mut best: Option<(usize, EClassId)> = None;
        for &c in &frontier {
            if cur[c.index()] == c_t[c.index()] {
                continue;
            }
            let mut m = cur.clone();
            m[c.index()] = c_t[c.index()];
            let Some(cost) = witness::cost_if_well_founded(egraph, sat.root, &m, costs, shape)
            else {
                continue;
            };
            if cost.dag < cur_cost && best.is_none_or(|(b, _)| cost.dag < b) {
                best = Some((cost.dag, c));
            }
        }
        let Some((cost, c)) = best else { break };
        cur[c.index()] = c_t[c.index()];
        cur_cost = cost;
        accepted += 1;
    }

    // Two witnesses, never confused. An *objective* witness is cheaper in
    // the number the extractor actually minimizes, and is an indictment of
    // the search. A *static-only* witness is cheaper only in the sweep's
    // unweighted `dag_cost` column while being dearer under the shape
    // weighting — that indicts the objective's weighting, not the search,
    // and realizability is not a question one can ask of it.
    let objective_witness = cost_t.dag < base;
    let realizable = if !objective_witness {
        "n/a (static-only)"
    } else if best_single <= cost_t.dag {
        "REALIZABLE-1"
    } else if cur_cost <= cost_t.dag {
        "REALIZABLE-k"
    } else if cur_cost < base {
        "PARTIAL"
    } else {
        "COORDINATED"
    };

    // Label every frontier class.
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut first: Option<FirstDivergence> = None;
    let win = trace.winner();
    for &c in &reach_t {
        if !frontier.contains(&c) {
            continue;
        }
        let w = c_t[c.index()].expect("frontier is under C_T");
        let g = c_g[c.index()].expect("greedy is total on root-reachable classes");
        let dp_w = win.table.cost_of(c, w);
        let dp_g = win.table.cost_of(c, g);
        let delta = swap_delta.get(&(c.index() as u32)).copied();
        let rule_g = minting_rule(sat, c, g);
        let cyc = witness::cycle_cost();
        let label = match (dp_w, dp_g, delta) {
            (Some(a), _, _) if a >= cyc => "CYCLE-PRICED",
            (Some(a), Some(b), _) if a == b => "TIE",
            (Some(a), Some(b), _) if a < b => "LOCAL-MISS",
            (_, _, Some(d)) if d < 0 && rule_g != "seed" => "DISTRACTOR",
            (_, _, Some(d)) if d < 0 => "SHARING",
            (_, _, Some(_)) => "COORDINATED",
            _ => "UNPRICED",
        };
        *labels.entry(label.to_string()).or_default() += 1;
        if label == "DISTRACTOR" {
            *distractors.entry(rule_g.clone()).or_default() += 1;
        }
        if first.is_none() {
            first = Some(FirstDivergence {
                class: c.index() as u32,
                node_greedy: node_label(egraph, c, g),
                node_witness: node_label(egraph, c, w),
                dp_cost_greedy: show(dp_g),
                dp_cost_witness: show(dp_w),
                own_greedy: win
                    .table
                    .own
                    .get(c.index())
                    .and_then(|r| r.get(g))
                    .copied()
                    .unwrap_or(0),
                own_witness: win
                    .table
                    .own
                    .get(c.index())
                    .and_then(|r| r.get(w))
                    .copied()
                    .unwrap_or(0),
                tree_greedy: show(trace.tree.table.cost_of(c, g)),
                tree_witness: show(trace.tree.table.cost_of(c, w)),
                shared_greedy: show(trace.shared.as_ref().and_then(|s| s.table.cost_of(c, g))),
                shared_witness: show(trace.shared.as_ref().and_then(|s| s.table.cost_of(c, w))),
                rule_greedy: rule_g,
                stage: match trace.stage_of(c, w) {
                    Stage::Dp => "dp",
                    Stage::Repair => "repair",
                    Stage::MinOfTwo => "min-of-two",
                }
                .into(),
                swap_delta: delta.unwrap_or(0),
                label: label.into(),
            });
        }
    }

    Some(WitnessRow {
        kernel: case.name.clone(),
        family: case.family.clone(),
        b_lo,
        b_hi,
        live_lo: lo.live_classes,
        live_hi: hi.live_classes,
        objective_witness: cost_t.dag,
        objective_greedy: base,
        delta_objective: cost_t.dag as i64 - base as i64,
        dag_witness: lo.dag_cost,
        dag_greedy: hi.dag_cost,
        delta_dag: lo.dag_cost as i64 - hi.dag_cost as i64,
        occupied: induced.occupied,
        merges: induced.merges,
        divergences: divergent.len(),
        frontier: frontier.len(),
        best_single_swap: best_single,
        greedy_swaps: cur_cost,
        swaps_accepted: accepted,
        witness_kind: if objective_witness {
            "objective".into()
        } else {
            "static-only".into()
        },
        realizable: realizable.into(),
        first_divergence: first,
        labels,
    })
}

fn main() {
    let cli = Cli::parse();
    let costs = CostModel::latency_prior();
    let cases = corpus(&cli);
    eprintln!("extraction_witnesses: {} kernels", cases.len());

    let mut budget_rows: Vec<BudgetRow> = Vec::new();
    let mut witness_rows: Vec<WitnessRow> = Vec::new();
    let mut distractors: BTreeMap<String, usize> = BTreeMap::new();

    for case in &cases {
        let shape = LatticeShape::new(case.extent);
        let held = case.kernel.clone();
        let (arena, root) = held.parts();
        let (la, lr) = legalize(arena, root);

        // Ascending: at each budget the smaller budgets' terms are already
        // in hand, so each graph is built once and analysed against every
        // witness before it is dropped.
        let mut terms: Vec<(usize, (ExprArena, ExprId), BudgetRow)> = Vec::new();
        for &cap in case.ladder {
            let t = Instant::now();
            let sat = saturate_at(&la, lr, shape, cap);
            let secs = t.elapsed().as_secs_f64();
            let (oa, orr) = sat.optimized.to_arena(&sat.egraph, sat.root);
            let live =
                witness::reachable_under(&sat.egraph, sat.root, &sat.optimized.choices).len();
            let (canon_choices, canon) =
                witness::extract_under(&sat.egraph, sat.root, &costs, shape, Ties::Canonical);
            let canon_arena = witness::arena_of(&sat.egraph, sat.root, canon_choices);
            let trace = witness::trace(&sat.egraph, sat.root, &costs, shape);
            let tied = witness::reachable_under(&sat.egraph, sat.root, &sat.optimized.choices)
                .iter()
                .filter(|&&c| trace.winner().table.is_tie(c))
                .count();
            let row = BudgetRow {
                kernel: case.name.clone(),
                family: case.family.clone(),
                class_cap: cap,
                inserted_classes: sat.inserted,
                classes_at_stop: sat.optimized.stats.classes,
                live_classes: live,
                applications: sat.optimized.stats.applications,
                stop: format!("{:?}", sat.optimized.stats.stop),
                objective: sat.optimized.cost.dag,
                dag_cost: dag_cost(&oa, orr),
                tree_cost: sat.optimized.cost.tree,
                extraction_objective: format!("{:?}", sat.optimized.extraction.objective),
                canonical_objective: canon.dag,
                canonical_dag_cost: dag_cost(&canon_arena.0, canon_arena.1),
                tied_classes: tied,
                seconds: secs,
            };
            println!(
                "{:<28} cap={:<7} live={:<6} obj={:<12} dag={:<7} canon_dag={:<7} tied={:<5} {:.1}s",
                row.kernel,
                row.class_cap,
                row.live_classes,
                row.objective,
                row.dag_cost,
                row.canonical_dag_cost,
                row.tied_classes,
                row.seconds
            );

            for (_, term, lo_row) in &terms {
                if lo_row.objective >= row.objective && lo_row.dag_cost >= row.dag_cost {
                    continue;
                }
                if let Some(w) = analyse(
                    &Pair {
                        case,
                        shape,
                        lo: lo_row,
                        hi: &row,
                        sat: &sat,
                        witness_term: term,
                        costs: &costs,
                    },
                    &mut distractors,
                ) {
                    println!(
                        "    WITNESS b_lo={} -> b_hi={} Δobj={} Δdag={} |D|={} |F|={} {}",
                        w.b_lo,
                        w.b_hi,
                        w.delta_objective,
                        w.delta_dag,
                        w.divergences,
                        w.frontier,
                        format_args!("{} {}", w.witness_kind, w.realizable)
                    );
                    witness_rows.push(w);
                }
            }
            terms.push((cap, (oa, orr), row.clone()));
            budget_rows.push(row);
        }
    }

    write_out(&cli.out, &budget_rows, &witness_rows, &distractors);
}

fn write_out(
    out: &Path,
    budgets: &[BudgetRow],
    witnesses: &[WitnessRow],
    distractors: &BTreeMap<String, usize>,
) {
    std::fs::create_dir_all(out).expect("create the results directory");

    let json = serde_json::json!({
        "schema": "extraction-witnesses-v1",
        "budgets": budgets,
        "witnesses": witnesses,
        "distractors": distractors.iter().map(|(rule, count)| DistractorRow { rule: rule.clone(), count: *count }).collect::<Vec<_>>(),
    });
    let jp = out.join("2026-09-08-extraction-witnesses.json");
    std::fs::write(&jp, serde_json::to_string_pretty(&json).expect("serialize"))
        .unwrap_or_else(|e| panic!("write {}: {e}", jp.display()));

    let mut csv = String::from(
        "kernel,family,b_lo,b_hi,live_lo,live_hi,objective_witness,objective_greedy,\
         delta_objective,dag_witness,dag_greedy,delta_dag,occupied,merges,divergences,frontier,\
         best_single_swap,greedy_swaps,swaps_accepted,witness_kind,realizable,first_class,\
         first_label,\
         first_stage,first_rule\n",
    );
    for w in witnesses {
        let (fc, fl, fs, fr) = match &w.first_divergence {
            Some(f) => (
                f.class.to_string(),
                f.label.clone(),
                f.stage.clone(),
                f.rule_greedy.clone(),
            ),
            None => ("-".into(), "-".into(), "-".into(), "-".into()),
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{fc},{fl},{fs},{fr}\n",
            w.kernel,
            w.family,
            w.b_lo,
            w.b_hi,
            w.live_lo,
            w.live_hi,
            w.objective_witness,
            w.objective_greedy,
            w.delta_objective,
            w.dag_witness,
            w.dag_greedy,
            w.delta_dag,
            w.occupied,
            w.merges,
            w.divergences,
            w.frontier,
            w.best_single_swap,
            w.greedy_swaps,
            w.swaps_accepted,
            w.witness_kind,
            w.realizable,
        ));
    }
    let cp = out.join("2026-09-08-extraction-witnesses.csv");
    std::fs::write(&cp, csv).unwrap_or_else(|e| panic!("write {}: {e}", cp.display()));

    // The budget ladder, as its own CSV — the tie-break A/B lives here.
    let mut bcsv = String::from(
        "kernel,family,class_cap,inserted_classes,classes_at_stop,live_classes,applications,\
         stop,objective,dag_cost,tree_cost,extraction_objective,canonical_objective,\
         canonical_dag_cost,tied_classes,seconds\n",
    );
    for b in budgets {
        bcsv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3}\n",
            b.kernel,
            b.family,
            b.class_cap,
            b.inserted_classes,
            b.classes_at_stop,
            b.live_classes,
            b.applications,
            b.stop,
            b.objective,
            b.dag_cost,
            b.tree_cost,
            b.extraction_objective,
            b.canonical_objective,
            b.canonical_dag_cost,
            b.tied_classes,
            b.seconds
        ));
    }
    let bp = out.join("2026-09-08-extraction-witnesses-budgets.csv");
    std::fs::write(&bp, bcsv).unwrap_or_else(|e| panic!("write {}: {e}", bp.display()));

    eprintln!(
        "extraction_witnesses: wrote {} budget rows, {} witnesses to {}",
        budgets.len(),
        witnesses.len(),
        out.display()
    );
}
