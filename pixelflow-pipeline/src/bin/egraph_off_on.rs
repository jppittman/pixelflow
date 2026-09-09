//! What the e-graph buys on the shipped kernels — measured through the real
//! bake path, saturation on against saturation off.
//!
//! docs/plans/2026-09-06-egraph-at-production-scale.md §7 lists "F: no
//! e-graph" as the one column never measured on a shipped kernel. This is
//! that measurement. One process per mode:
//!
//! ```text
//! # built once with the switch: --features pixelflow-search/saturation-switch
//! PIXELFLOW_SATURATION=off egraph_off_on run --out rows/off.jsonl
//! PIXELFLOW_SATURATION=on  egraph_off_on run --out rows/on.jsonl
//!                          egraph_off_on run --out rows/cse.jsonl --variant cse-only
//!                          egraph_off_on run --out rows/sh.jsonl  --variant with-select-hoist
//! egraph_off_on diff --off rows/off.jsonl --on rows/on.jsonl --cse-only rows/cse.jsonl \
//!     --with-select-hoist rows/sh.jsonl \
//!     --out-prefix docs/results/2026-09-07-egraph-off-vs-on-real-shaders
//! ```
//!
//! Every kernel is compiled by the same three calls `jit_cache::compile`
//! makes — `optimize_runtime_arena` (which is where `PIXELFLOW_SATURATION`
//! is honoured), `relink`, `emit::compile` — and, for buffer-free kernels,
//! the bytes are asserted identical to `Manifold::compile`'s, so the
//! instrument *is* the production path rather than a model of it. Rows are
//! appended to the JSONL as each kernel finishes.
//!
//! The corpus is the shipped kernels and nothing else: the chrome sphere
//! (packed, plus its red channel alone), the psychedelic shader (packed, with
//! its clock uniform), the terminal cell grid at a Retina 80×24 geometry, the
//! 190 glyph bakes `GlyphAtlas::warm` performs plus the three `font_rendering`
//! bench glyphs (and `O`@32 at a 640-wide row for the prologue estimate), and
//! the twelve `shader_bench` ShaderToy ports.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use pixelflow_codegen::emit::executable::{ExecutableCode, Point4, TileSlice};
use pixelflow_codegen::emit::{self, CompileResult};
use pixelflow_core::lattice::cell_grid::{CELL_STRIDE, CellGridMetrics};
use pixelflow_core::{Bits, CellGridShape, Kernel, Uniform};
use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::pixel::Pixel;
use pixelflow_graphics::render::scene::{Scene, compile_cell_grid_for};
use pixelflow_graphics::scene3d::{Hit, Plane, Ray, Rgba, Sphere, checker, sky};
use pixelflow_ir::optimize::{Optimize, Rewritten};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{BindingTable, Evaluator, ExprArena, ExprId, ExprNode, LatticeShape, pipeline};
use pixelflow_pipeline::alloc_probe::{self, CountingAlloc};
use pixelflow_pipeline::collapse_bench::corpus::Trips;
use pixelflow_pipeline::collapse_bench::row::StaticFeatures;
use pixelflow_pipeline::collapse_bench::{self, LANES, features_of};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::egraph::{
    Budget, CostModel, EpisodeLabels, KeepJournal, Optimizer, Rewrite, RuleSet, SaturationConfig,
    Vocabulary, all_rules, config_for_node_count, insert, reachable_count,
};
use pixelflow_search::math::round2_rules::experimental_rules;
use pixelflow_search::{Saturate, Tier};

const SCHEMA: &str = "egraph-off-on-v1";
const SCREEN: [u32; 2] = [1920, 1080];
const SHADER_EXTENT: [u32; 2] = [256, 256];
const CELL_HEIGHT_PT: f32 = 16.0;
const CELL_WIDTH_PT: f32 = 10.0;
const ATLAS_CAPACITY: usize = 128;
const DENSITIES: [f32; 2] = [1.0, 2.0];
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
const BENCH_CHARS: [(&str, char); 3] =
    [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')];
const BENCH_PT: f32 = 32.0;
const BENCH_EXTENT: [u32; 2] = [40, 45];
const BENCH_WIDE_EXTENT: [u32; 2] = [640, 45];
const ORACLE_POINTS: usize = 256;
const CLOCK_SAMPLES: usize = 7;
const CLOCK_MIN_SAMPLE_NS: u64 = 2_000_000;
const CLOCK_MAX_CALLS: usize = 20_000;
const SELECT_HOIST_PREFIX: &str = "select-hoist-";
/// The runtime tier's telemetry record, as `pixelflow_search::telemetry`
/// prints it to stderr when `PIXELFLOW_SATURATION_TELEMETRY` is unset.
const SAT_TELEMETRY_PREFIX: &str = "{\"tier\":\"runtime\"";
/// `docs/plans/2026-09-01-production-budget-determinism.md`: the
/// application budget is 40 per e-class of the class cap.
const APPLICATIONS_PER_CLASS: u64 = 40;

/// Peak heap growth per compile (`alloc_probe`), the e-graph's own
/// transient allocation — the number the class-cap sweep buys memory with.
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[derive(Parser)]
#[command(
    name = "egraph_off_on",
    about = "Saturation on vs off on every shipped kernel"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile (and time) every real kernel on this build; one JSONL row each.
    Run {
        #[arg(long)]
        out: PathBuf,
        /// An in-harness optimizer instead of the production one
        /// (`PIXELFLOW_SATURATION` must be unset or `on`).
        #[arg(long)]
        variant: Option<Variant>,
        /// Skip the clock (deterministic columns only).
        #[arg(long)]
        no_clock: bool,
        /// Skip the saturation probe (rule fire counts / load-bearing rules).
        #[arg(long)]
        no_probe: bool,
        /// Only kernels whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
        /// Kernels (exact names) to leave out of this run — the emitter
        /// panics (branch range, frame size) rather than returning an error,
        /// so a kernel one mode cannot emit is excluded here and reported by
        /// `diff` as present in one mode only.
        #[arg(long, num_args = 0..)]
        skip: Vec<String>,
        #[arg(long)]
        font: Option<PathBuf>,
        /// Hold saturation to this e-class cap instead of the production
        /// tier's (the 2026-09-08 class-cap sweep), keeping each kernel's
        /// own tier's round cap; the application cap is `--app-cap`, or the
        /// plan's 40 per class when omitted.
        #[arg(long, conflicts_with_all = ["variant", "classes_per_node"])]
        class_cap: Option<usize>,
        /// A class cap proportional to the kernel: this many classes per
        /// legalized reachable node, clamped to `[--cap-floor, --cap-ceiling]`.
        #[arg(long, conflicts_with = "variant", requires_all = ["cap_floor", "cap_ceiling"])]
        classes_per_node: Option<usize>,
        /// A class cap proportional to what the e-graph holds after
        /// insertion: this many classes per hash-consed input class,
        /// clamped to `[--cap-floor, --cap-ceiling]`.
        #[arg(long, conflicts_with_all = ["variant", "classes_per_node"], requires_all = ["cap_floor", "cap_ceiling"])]
        classes_per_inserted: Option<usize>,
        #[arg(long)]
        cap_floor: Option<usize>,
        #[arg(long)]
        cap_ceiling: Option<usize>,
        /// The application budget beside a cap arm; the plan's 40 per class
        /// of the resolved cap when omitted.
        #[arg(long)]
        app_cap: Option<u64>,
    },
    /// The extraction self-consistency census: what the winning DP arm
    /// claimed for the choice map it selected, against what the term that map
    /// materializes actually costs — same cost model, same shape the
    /// extraction ran at. Deterministic: no clock, no emitter, no probe.
    Consistency {
        /// CSV output (one row per kernel per cap).
        #[arg(long)]
        out: PathBuf,
        /// Class caps to census, one arm each.
        #[arg(long, num_args = 1..)]
        class_cap: Vec<usize>,
        /// Only kernels whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
        #[arg(long)]
        font: Option<PathBuf>,
        /// The application budget beside each cap; the plan's 40 per class
        /// of the cap when omitted.
        #[arg(long)]
        app_cap: Option<u64>,
    },
    /// Aggregate `run --class-cap` rows across caps into the sweep documents.
    CapSweep {
        #[arg(long, num_args = 1..)]
        rows: Vec<PathBuf>,
        #[arg(long)]
        out_prefix: PathBuf,
        /// Free-text lines (load, ceiling overrides) appended verbatim.
        #[arg(long, num_args = 0..)]
        note: Vec<String>,
    },
    /// Diff runs into the results documents.
    Diff {
        #[arg(long, num_args = 1..)]
        off: Vec<PathBuf>,
        #[arg(long, num_args = 1..)]
        on: Vec<PathBuf>,
        /// Rows from `--variant with-select-hoist`.
        #[arg(long, num_args = 0..)]
        with_select_hoist: Vec<PathBuf>,
        /// Rows from `--variant cse-only`.
        #[arg(long, num_args = 0..)]
        cse_only: Vec<PathBuf>,
        #[arg(long)]
        out_prefix: PathBuf,
        /// Free-text lines (load, bench_scene numbers) appended verbatim.
        #[arg(long, num_args = 0..)]
        note: Vec<String>,
    },
}

/// The in-harness optimizers, each `Optimizer::production()` with one
/// lever moved — the same pipeline `optimize_runtime_arena_uncached` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Variant {
    /// Production's 62 rules plus `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`,
    /// which live in `round2_rules::experimental_rules` and are *not* in
    /// `all_rules()`): does the rule the demand plan indicts ever match a
    /// real shader, and what does it do to the guards when it does.
    WithSelectHoist,
    /// Zero rewrite rounds: insert, extract. What the e-graph's hash-consing
    /// alone buys, separated from what the rules buy.
    CseOnly,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::WithSelectHoist => "with-select-hoist",
            Variant::CseOnly => "cse-only",
        }
    }

    fn optimizer(self, shape: LatticeShape) -> Optimizer {
        let base = Optimizer::production().for_lattice(shape);
        match self {
            Variant::WithSelectHoist => base.rules(RuleSet::new(with_select_hoist_rules())),
            Variant::CseOnly => base.budget(Budget::Explicit {
                iterations: 0,
                classes: usize::MAX,
                applications: Some(0),
            }),
        }
    }
}

/// How one arm of the sweep sizes a kernel's class cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CapRule {
    /// One cap for every kernel.
    Flat(usize),
    /// `classes_per_node × node_count`, clamped — the cap the input's own
    /// size asks for, so a kernel the flat cap clips in its first round
    /// gets room and a kernel it never clipped keeps today's budget.
    PerNode {
        classes_per_node: usize,
        floor: usize,
        ceiling: usize,
    },
    /// `classes_per_inserted × inserted_classes`, clamped: the same idea
    /// keyed on the hash-consed input the e-graph actually holds, which is
    /// what separates a glyph (2.3 arena nodes per class) from the chrome
    /// scene (840: a 390k-node tree that hash-conses to 465 classes).
    PerInserted {
        classes_per_inserted: usize,
        floor: usize,
        ceiling: usize,
    },
}

/// The two sizes a cap rule can key on.
#[derive(Clone, Copy, Debug)]
struct InputSizes {
    /// Legalized reachable arena nodes — production's tier key.
    nodes: usize,
    /// E-classes after insertion, before any rewrite.
    inserted: usize,
}

/// One arm of the class-cap sweep: production's optimizer with the
/// kernel's own tier's rounds and these two budget dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CapArm {
    rule: CapRule,
    /// `None`: the plan's ratio of the resolved cap.
    applications: Option<u64>,
}

impl CapArm {
    fn label(self) -> String {
        let apps = match self.applications {
            Some(a) => a.to_string(),
            None => format!("{APPLICATIONS_PER_CLASS}x"),
        };
        match self.rule {
            CapRule::Flat(c) => format!("cap{c}-app{apps}"),
            CapRule::PerNode {
                classes_per_node,
                floor,
                ceiling,
            } => format!("capx{classes_per_node}-{floor}-{ceiling}-app{apps}"),
            CapRule::PerInserted {
                classes_per_inserted,
                floor,
                ceiling,
            } => format!("caph{classes_per_inserted}-{floor}-{ceiling}-app{apps}"),
        }
    }

    /// The (class cap, application cap) this arm holds a kernel of these
    /// sizes to.
    fn resolve(self, sizes: InputSizes) -> (usize, u64) {
        let classes = match self.rule {
            CapRule::Flat(c) => c,
            CapRule::PerNode {
                classes_per_node,
                floor,
                ceiling,
            } => (classes_per_node * sizes.nodes).clamp(floor, ceiling),
            CapRule::PerInserted {
                classes_per_inserted,
                floor,
                ceiling,
            } => (classes_per_inserted * sizes.inserted).clamp(floor, ceiling),
        };
        let applications = self
            .applications
            .unwrap_or(classes as u64 * APPLICATIONS_PER_CLASS);
        (classes, applications)
    }

    /// `node_count` is the legalized reachable count production keys its
    /// tier on: the arm moves the class and application caps and leaves
    /// the tier's round cap where production has it, so a blitz-tier space
    /// glyph is not held to classical's 100 rounds — though at every arm,
    /// the baseline included, its class cap is above its own tier's 500.
    fn optimizer(self, shape: LatticeShape, sizes: InputSizes) -> Optimizer {
        let tier: SaturationConfig = config_for_node_count(sizes.nodes);
        let (classes, applications) = self.resolve(sizes);
        Optimizer::production()
            .for_lattice(shape)
            .budget(Budget::Explicit {
                iterations: tier.max_iterations,
                classes,
                applications: Some(applications),
            })
    }
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// The `saturation-telemetry` record the runtime tier printed for this
/// compile, parsed back — the stop reason, counts and extraction objective
/// of the saturation the production path actually ran.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct SatTelemetry {
    node_count: usize,
    /// `None` on rows written before the optimizer reported it.
    #[serde(default)]
    inserted_classes: Option<usize>,
    max_iterations: usize,
    max_classes: usize,
    max_applications: Option<u64>,
    stop_reason: String,
    iterations: usize,
    classes_at_stop: usize,
    application_count: u64,
    union_count: usize,
    extracted_latency_prior_cost: u64,
    extraction_objective: String,
    live_classes: Option<usize>,
    shared_pass_bytes: Option<usize>,
    wall_clock_us: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GuardTelemetry {
    schedule: u64,
    selects: u64,
    guarded: u64,
    exclusive: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Oracle {
    points: usize,
    /// Same form: `eval_scalar` of the arena that was emitted vs the JIT.
    same_form_max_abs: f64,
    same_form_max_rel: f64,
    same_form_nan_mismatch: usize,
    /// Cross form: `eval_scalar` of the arena as constructed (after the
    /// legalizing prefix, since `eval_scalar` has no `Dwrt`) vs the JIT of
    /// what was compiled from it — what the rewrites moved, at the sample.
    cross_form_max_abs: f64,
    cross_form_max_rel: f64,
    cross_form_nan_mismatch: usize,
    /// Packed kernels: pixels whose 32-bit pattern differs, and the largest
    /// per-byte (per-channel) delta, same form / cross form.
    packed_mismatch_same: usize,
    packed_max_byte_same: u32,
    packed_mismatch_cross: usize,
    packed_max_byte_cross: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Clock {
    ns_per_call_median: f64,
    ns_per_call_min: f64,
    ns_per_call_iqr: f64,
    calls_per_sample: usize,
    ns_per_px: f64,
    /// `Scene::render` at one thread, median of 5 frames — cell grid only.
    scene_ns_per_px: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RuleCount {
    rule: String,
    fired: usize,
    /// `EpisodeLabels::compute_tight` (`derivation_ancestors_tight`).
    load_bearing: usize,
    /// `EpisodeLabels::compute_strict`: the application's own e-node was chosen.
    strict: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SatProbe {
    applications: u64,
    unions: usize,
    classes: usize,
    iterations: usize,
    stop: String,
    wall_ms: f64,
    /// The three `select-hoist-*` rules, always present (zero if never fired).
    select_hoist: Vec<RuleCount>,
    /// Every rule with at least one load-bearing application, most first.
    load_bearing_rules: Vec<RuleCount>,
    /// Probe extraction compiled to the same bytes as the production path.
    bytes_identical_to_production: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct KernelRow {
    schema: String,
    mode: String,
    tier: String,
    lanes: u32,
    git_sha: String,
    name: String,
    class: String,
    extent: [u32; 2],
    packed: bool,
    input_nodes: usize,
    compiled_nodes: usize,
    dag_cost_input: usize,
    dag_cost: usize,
    bytes: u32,
    spill_slots: u32,
    hoisted: u32,
    statics: StaticFeatures,
    guard: Option<GuardTelemetry>,
    optimize_ms: f64,
    emit_ms: f64,
    bytes_identical_to_manifold_compile: Option<bool>,
    picture_hash: u64,
    oracle: Option<Oracle>,
    clock: Option<Clock>,
    probe: Option<SatProbe>,
    /// `None` on rows written before the telemetry column existed, and on
    /// `off` rows (no saturation ran).
    #[serde(default)]
    sat: Option<SatTelemetry>,
    /// Peak net heap growth over optimize + emit (`alloc_probe`); `None` on
    /// rows written before the column existed.
    #[serde(default)]
    peak_alloc_bytes: Option<u64>,
    /// The class and application caps this kernel was held to under a
    /// `--class-cap` / `--classes-per-node` arm (the arm itself is `mode`);
    /// `None` for production's own tier.
    #[serde(default)]
    class_cap: Option<usize>,
    #[serde(default)]
    app_cap: Option<u64>,
    /// E-classes the legalized input inserts to, before any rewrite — the
    /// hash-consed size; `None` on rows written before the column existed.
    #[serde(default)]
    inserted_classes: Option<usize>,
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

struct CellGridCase {
    shape: CellGridShape,
    metrics: CellGridMetrics,
    cells_id: pixelflow_ir::arena::BufferIdentity,
    cells: Arc<Vec<f32>>,
    atlas: Arc<Vec<f32>>,
}

struct RealKernel {
    name: String,
    class: String,
    kernel: Kernel,
    extent: [u32; 2],
    packed: bool,
    cell_grid: Option<CellGridCase>,
}

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `pixelflow_graphics::render::packed::packed_kernel`, which is
/// `pub(crate)`: the byte pack `compile_packed_for` wraps a colour in.
/// `main` asserts the bytes of what this produces equal the production
/// program's, so a drift here is loud.
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

fn rgba8_shifts() -> [u32; 4] {
    <Rgba8 as Pixel>::packed_shifts().expect("Rgba8 packs")
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

fn psychedelic_color() -> Rgba {
    let clock = Uniform::new(0.0);
    Rgba::from([
        psych_channel(1.0, clock),
        psych_channel(-1.0, clock),
        psych_channel(-2.0, clock),
        k(1.0),
    ])
}

fn cell_grid_case() -> (Kernel, CellGridCase) {
    const COLS: u32 = 80;
    const ROWS: u32 = 24;
    const DENSITY: f32 = 2.0;
    const ATLAS_SLOTS_PER_ROW: u32 = 12;
    const ATLAS_SLOT_ROWS: u32 = 11;
    const ATLAS_PAD: u32 = 1;
    let tile_px = (CELL_HEIGHT_PT * DENSITY).round().max(1.0) as u32;
    let slot_px = tile_px + 2 * ATLAS_PAD;
    let cell_w = CELL_WIDTH_PT * DENSITY;
    let cell_h = CELL_HEIGHT_PT * DENSITY;
    let shape = CellGridShape {
        cols: COLS,
        rows: ROWS,
        atlas_width: ATLAS_SLOTS_PER_ROW * slot_px,
        atlas_height: ATLAS_SLOT_ROWS * slot_px,
        frame_w: (COLS as f32 * cell_w).round() as u32,
        frame_h: (ROWS as f32 * cell_h).round() as u32,
    };
    let metrics = CellGridMetrics {
        cell_w,
        cell_h,
        density: DENSITY,
        tile_w: tile_px,
        tile_h: tile_px,
        scale: DENSITY,
    };
    let kernels = shape.channel_kernels();
    let kernel = packed_kernel(&Rgba::from(&kernels.channels), rgba8_shifts());

    let mut cells = Vec::with_capacity((COLS * ROWS) as usize * CELL_STRIDE);
    for i in 0..(COLS * ROWS) as usize {
        let slot = (i % 95) as f32;
        cells.extend_from_slice(&[slot, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }
    let atlas_len = (shape.atlas_width * shape.atlas_height) as usize;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let atlas: Vec<f32> = (0..atlas_len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) & 0xFF) as f32 / 255.0
        })
        .collect();
    (
        kernel,
        CellGridCase {
            shape,
            metrics,
            cells_id: kernels.buffers.cells,
            cells: Arc::new(cells),
            atlas: Arc::new(atlas),
        },
    )
}

fn real_kernels(font: &Path, filter: Option<&str>) -> Vec<RealKernel> {
    let mut out = Vec::new();
    let mut push = |name: String, class: &str, kernel: Kernel, extent: [u32; 2], packed: bool| {
        out.push(RealKernel {
            name,
            class: class.to_string(),
            kernel,
            extent,
            packed,
            cell_grid: None,
        });
    };
    let shifts = rgba8_shifts();
    let chrome = chrome_color();
    push(
        "chrome_packed".into(),
        "chrome",
        packed_kernel(&chrome, shifts),
        SCREEN,
        true,
    );
    let red = chrome.fold(
        &|channels: &[Kernel; 4]| channels[0].clone(),
        &|mask, a: Kernel, b: Kernel| mask.select(&a, &b),
    );
    push("chrome_R".into(), "chrome_channel", red, SCREEN, false);
    push(
        "psychedelic_packed".into(),
        "psychedelic",
        packed_kernel(&psychedelic_color(), shifts),
        SCREEN,
        true,
    );

    let (kernel, case) = cell_grid_case();
    let extent = [case.shape.frame_w, case.shape.frame_h];
    push("cellgrid_80x24_d2".into(), "cellgrid", kernel, extent, true);
    let cell_grid = case;

    let data = std::fs::read(font).unwrap_or_else(|e| panic!("read {}: {e}", font.display()));
    let parsed = Font::parse(&data).expect("parse the production font");
    let mut missing = 0usize;
    for density in DENSITIES {
        let atlas = GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY);
        let tile = atlas.tile_px() as u32;
        for ch in WARM_RANGE {
            // The winding sum reads a bound piece table, so a glyph is a
            // kernel plus its binding. This harness measures the arena the
            // optimizer sees, which the kernel alone carries.
            let Some(glyph) = parsed.glyph_kernel_scaled(ch, tile as f32) else {
                missing += 1;
                continue;
            };
            let kernel = glyph.kernel();
            push(
                format!("glyph{tile}_U{:04X}", ch as u32),
                &format!("glyph{tile}"),
                kernel,
                [tile, tile],
                false,
            );
        }
    }
    eprintln!("egraph_off_on: the font has no glyph for {missing} warmed characters");
    for (label, ch) in BENCH_CHARS {
        let kernel = parsed
            .glyph_kernel_scaled(ch, BENCH_PT)
            .unwrap_or_else(|| panic!("no glyph for {ch:?}"))
            .kernel();
        push(
            format!("bench_{label}"),
            "bench",
            kernel.clone(),
            BENCH_EXTENT,
            false,
        );
        if ch == 'O' {
            push(
                format!("bench_{label}_wide"),
                "bench_wide",
                kernel,
                BENCH_WIDE_EXTENT,
                false,
            );
        }
    }

    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name).expect("registered shader");
        push(
            format!("shader_{name}"),
            "shader",
            Kernel::from_parts(arena, root),
            SHADER_EXTENT,
            false,
        );
    }

    out.iter_mut()
        .find(|r| r.class == "cellgrid")
        .expect("cell grid row")
        .cell_grid = Some(cell_grid);
    if let Some(f) = filter {
        out.retain(|r| r.name.contains(f));
    }
    out
}

// ---------------------------------------------------------------------------
// Compile: the production path's three calls, instrumented
// ---------------------------------------------------------------------------

struct Compiled {
    result: CompileResult,
    linked: ExprArena,
    root: ExprId,
    optimize_ms: f64,
    emit_ms: f64,
    guard: Option<GuardTelemetry>,
    sat: Option<SatTelemetry>,
    peak_alloc_bytes: u64,
}

/// Redirect fd 2 into a file around `f` so `PIXELFLOW_GUARD_TELEMETRY`'s
/// `eprintln!` lands somewhere this process can read it back.
fn capture_stderr<T>(f: impl FnOnce() -> T) -> (T, String) {
    use std::os::unix::io::AsRawFd as _;
    let path = std::env::temp_dir().join(format!("egraph_off_on.{}.stderr", std::process::id()));
    let file = std::fs::File::create(&path).expect("stderr capture file");
    let saved = unsafe { libc::dup(2) };
    assert!(saved >= 0, "dup(2) failed");
    assert!(
        unsafe { libc::dup2(file.as_raw_fd(), 2) } >= 0,
        "dup2 failed"
    );
    let r = f();
    assert!(unsafe { libc::dup2(saved, 2) } >= 0, "dup2 restore failed");
    unsafe { libc::close(saved) };
    drop(file);
    let text = std::fs::read_to_string(&path).expect("read stderr capture");
    std::fs::remove_file(&path).expect("remove stderr capture");
    (r, text)
}

fn parse_guard_telemetry(log: &str) -> Option<GuardTelemetry> {
    let line = log.lines().rfind(|l| l.starts_with("guard-telemetry:"))?;
    let field = |key: &str| -> u64 {
        let needle = format!("{key}=");
        let start = line
            .find(&needle)
            .unwrap_or_else(|| panic!("{key} in {line}"))
            + needle.len();
        line[start..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .expect("digits")
            .parse()
            .expect("u64")
    };
    Some(GuardTelemetry {
        schedule: field("schedule"),
        selects: field("selects"),
        guarded: field("guarded"),
        exclusive: field("exclusive"),
    })
}

/// The runtime tier's telemetry line, if the capture saw one. `on` modes
/// must see exactly one: the harness is built with
/// `pixelflow-search/saturation-telemetry`, and a compile that saturated
/// without leaving its record is a compile whose stop reason is unknown.
fn parse_sat_telemetry(log: &str) -> Option<SatTelemetry> {
    let lines: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with(SAT_TELEMETRY_PREFIX))
        .collect();
    match lines.as_slice() {
        [] => None,
        [line] => {
            Some(serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("saturation-telemetry record does not parse: {e}\n{line}")
            }))
        }
        many => panic!(
            "one compile left {} saturation-telemetry records; expected one:\n{}",
            many.len(),
            many.join("\n")
        ),
    }
}

/// Link `a`/`r` back onto the input's buffer and uniform tables, emit, and
/// gather the emitter's and the optimizer's stderr reports into a row.
fn emit_and_report(
    input: &ExprArena,
    (a, r): (ExprArena, ExprId),
    optimize_ms: f64,
    optimize_log: &str,
) -> Compiled {
    let (linked, root) = if input.buffers().is_empty() && input.uniforms().is_empty() {
        (a, r)
    } else {
        a.relink(r, input.buffers(), input.uniforms())
    };
    let t = Instant::now();
    let (result, log) = capture_stderr(|| emit::compile(&linked, root));
    let emit_ms = t.elapsed().as_secs_f64() * 1e3;
    let result = result.expect("real kernel failed to compile");
    Compiled {
        result,
        linked,
        root,
        optimize_ms,
        emit_ms,
        guard: parse_guard_telemetry(&log),
        sat: parse_sat_telemetry(optimize_log),
        peak_alloc_bytes: alloc_probe::peak_bytes() as u64,
    }
}

fn compile_via_production_path(arena: &ExprArena, root: ExprId, shape: LatticeShape) -> Compiled {
    alloc_probe::reset();
    let t = Instant::now();
    let (optimized, log) =
        capture_stderr(|| pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape));
    let optimize_ms = t.elapsed().as_secs_f64() * 1e3;
    let (a, r) = optimized
        .as_deref()
        .map(|(a, r)| (a.clone(), *r))
        .unwrap_or((arena.clone(), root));
    emit_and_report(arena, (a, r), optimize_ms, &log)
}

/// An in-harness optimizer (a [`Variant`] or a [`CapArm`]): the same
/// pipeline `optimize_runtime_arena_uncached` runs, with that optimizer in
/// the saturation slot.
fn compile_via_optimizer(arena: &ExprArena, root: ExprId, optimizer: Optimizer) -> Compiled {
    alloc_probe::reset();
    let t = Instant::now();
    let (rewritten, log) = capture_stderr(|| {
        pipeline![
            LowerDwrt,
            ExpandReduce,
            Saturate::with(optimizer, Vocabulary::Runtime, Tier::Runtime)
        ]
        .optimize(arena, root)
    });
    let optimize_ms = t.elapsed().as_secs_f64() * 1e3;
    let (a, r) = match rewritten {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("in-harness pipeline declined a real kernel"),
    };
    emit_and_report(arena, (a, r), optimize_ms, &log)
}

fn with_select_hoist_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_rules();
    assert!(
        !rules
            .iter()
            .any(|r| r.name().starts_with(SELECT_HOIST_PREFIX)),
        "select-hoist is in all_rules() now; the with-select-hoist variant is moot"
    );
    let hoist: Vec<Box<dyn Rewrite>> = experimental_rules()
        .into_iter()
        .filter(|r| r.name().starts_with(SELECT_HOIST_PREFIX))
        .collect();
    assert_eq!(hoist.len(), 3, "expected exactly three select-hoist rules");
    rules.extend(hoist);
    rules
}

// ---------------------------------------------------------------------------
// Static columns
// ---------------------------------------------------------------------------

fn reachable(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
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

// ---------------------------------------------------------------------------
// Running the emitted kernel: full-extent output, oracle, clock
// ---------------------------------------------------------------------------

struct Ctx {
    /// Buffer slots in the linked arena's order, then the uniform block.
    slots: Vec<*const f32>,
    _uniforms: Vec<f32>,
    _buffers: Vec<Arc<Vec<f32>>>,
}

/// One kernel's own tabulations, by the [`BufferIdentity`] each was seeded
/// under — `Kernel::buffer_data()` (`bee7813`, "a kernel carries its own
/// tabulations"). A glyph's winding sum reads a piece table this way: the
/// data travels with `RealKernel::kernel` itself, with nothing separate a
/// caller must gather and keep paired with it.
type Carried = [(pixelflow_ir::arena::BufferIdentity, Arc<[f32]>)];

/// Resolve one declared buffer to real memory: `carried` — the kernel's own
/// tabulation — first, then `case`'s cell-grid buffers for the one kernel
/// (the terminal cell grid) that reads externally-owned per-frame memory
/// instead of a self-contained table. Mirrors `Manifold::bind`'s own
/// precedence (`pixelflow-core/src/lattice/manifold.rs`): a slot the kernel
/// carries data for is already spoken for, so a caller-supplied binding is
/// only ever the fallback. This harness cannot call `Manifold::bind` itself
/// — it drives its own instrumented `ExecutableCode`, compiled outside
/// `Manifold::compile` so it can capture guard telemetry and optimize/emit
/// timings — but the resolution a raw context table needs is the same.
///
/// # Panics
/// If `id` names neither the kernel's own tabulation nor (when given) the
/// cell-grid case's buffers — a kernel that declares a buffer nothing here
/// can bind, which is a corpus bug rather than a shape to paper over.
fn resolve_buffer<'a>(
    id: pixelflow_ir::arena::BufferIdentity,
    carried: &'a Carried,
    case: Option<&'a CellGridCase>,
) -> &'a [f32] {
    if let Some((_, data)) = carried.iter().find(|(cid, _)| *cid == id) {
        return data;
    }
    let case = case.unwrap_or_else(|| {
        panic!("buffer {id:?}: not in the kernel's own tabulation and no cell-grid case was given")
    });
    if id == case.cells_id {
        &case.cells
    } else {
        &case.atlas
    }
}

fn context_for(linked: &ExprArena, carried: &Carried, case: Option<&CellGridCase>) -> Ctx {
    let mut buffers: Vec<Arc<Vec<f32>>> = Vec::new();
    for decl in linked.buffers() {
        let data = resolve_buffer(decl.id, carried, case);
        assert_eq!(
            data.len(),
            (decl.width * decl.height) as usize,
            "buffer extent {}x{} does not match its data",
            decl.width,
            decl.height
        );
        buffers.push(Arc::new(data.to_vec()));
    }
    let uniforms: Vec<f32> = linked.uniforms().iter().map(|u| u.default).collect();
    let mut slots: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
    slots.push(uniforms.as_ptr());
    Ctx {
        slots,
        _uniforms: uniforms,
        _buffers: buffers,
    }
}

fn run_once(code: &ExecutableCode, ctx: &Ctx, trips: Trips, out: &mut [f32]) {
    let mut x0 = [0.0f32; LANES];
    for (i, lane) in x0.iter_mut().enumerate() {
        *lane = 0.5 + i as f32;
    }
    let origin = Point4::new(x0, [0.5f32; LANES], [0.0f32; LANES], [0.0f32; LANES]);
    let tile = TileSlice::contiguous(out.as_mut_ptr(), trips.groups as usize, trips.rows as usize);
    unsafe {
        code.call_collapse(ctx.slots.as_ptr(), tile, origin);
    }
}

fn fnv(out: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in out {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

struct OracleForms<'a> {
    /// The arena as constructed, legalized (`legalize`).
    input: (&'a ExprArena, ExprId),
    /// The arena that was emitted.
    linked: (&'a ExprArena, ExprId),
    /// The kernel's own tabulations — see [`Carried`].
    carried: &'a Carried,
    case: Option<&'a CellGridCase>,
    packed: bool,
}

fn oracle(forms: &OracleForms<'_>, out: &[f32], trips: Trips) -> Oracle {
    let OracleForms {
        input,
        linked,
        carried,
        case,
        packed,
    } = *forms;
    let bindings_for = |arena: &ExprArena| -> BindingTable<'_> {
        let decls = arena.buffers();
        let table = if decls.is_empty() {
            BindingTable::empty()
        } else {
            let slices: Vec<&[f32]> = decls
                .iter()
                .map(|d| resolve_buffer(d.id, carried, case))
                .collect();
            BindingTable::bind(arena, &slices).expect("bind oracle buffers")
        };
        table
            .bind_uniforms(arena, &[])
            .expect("bind oracle uniforms")
    };
    let b_in = bindings_for(input.0);
    let b_ln = bindings_for(linked.0);
    let width = trips.groups as usize * LANES;
    let pixels = width * trips.rows as usize;
    let stride = (pixels / ORACLE_POINTS).max(1);
    let mut o = Oracle::default();
    let mut px = 0usize;
    let same_form = Evaluator::new(linked.0, linked.1);
    let cross_form = Evaluator::new(input.0, input.1);
    while px < pixels && o.points < ORACLE_POINTS {
        let (x, y) = ((px % width) as f32 + 0.5, (px / width) as f32 + 0.5);
        let jit = out[px];
        let same = same_form.eval(&[x, y], &b_ln);
        let cross = cross_form.eval(&[x, y], &b_in);
        if packed {
            let byte_delta = |a: f32, b: f32| -> u32 {
                a.to_bits()
                    .to_le_bytes()
                    .iter()
                    .zip(b.to_bits().to_le_bytes())
                    .map(|(&p, q)| u32::from(p.abs_diff(q)))
                    .max()
                    .unwrap_or(0)
            };
            if jit.to_bits() != same.to_bits() {
                o.packed_mismatch_same += 1;
                o.packed_max_byte_same = o.packed_max_byte_same.max(byte_delta(jit, same));
            }
            if jit.to_bits() != cross.to_bits() {
                o.packed_mismatch_cross += 1;
                o.packed_max_byte_cross = o.packed_max_byte_cross.max(byte_delta(jit, cross));
            }
        } else {
            let acc = |reference: f32, max_abs: &mut f64, max_rel: &mut f64, nan: &mut usize| match (
                jit.is_nan(),
                reference.is_nan(),
            ) {
                (true, true) => {}
                (true, false) | (false, true) => *nan += 1,
                (false, false) => {
                    let d = f64::from((jit - reference).abs());
                    *max_abs = max_abs.max(d);
                    if reference.abs() > 1e-3 {
                        *max_rel = max_rel.max(d / f64::from(reference.abs()));
                    }
                }
            };
            acc(
                same,
                &mut o.same_form_max_abs,
                &mut o.same_form_max_rel,
                &mut o.same_form_nan_mismatch,
            );
            acc(
                cross,
                &mut o.cross_form_max_abs,
                &mut o.cross_form_max_rel,
                &mut o.cross_form_nan_mismatch,
            );
        }
        o.points += 1;
        px += stride;
    }
    o
}

fn clock(code: &ExecutableCode, ctx: &Ctx, trips: Trips, out: &mut [f32]) -> Clock {
    let run = |calls: usize, out: &mut [f32]| -> u64 {
        let t = Instant::now();
        for _ in 0..calls {
            run_once(code, ctx, trips, out);
            std::hint::black_box(&out[0]);
        }
        t.elapsed().as_nanos() as u64
    };
    let mut calls = 1usize;
    loop {
        let ns = run(calls, out);
        if ns >= CLOCK_MIN_SAMPLE_NS || calls >= CLOCK_MAX_CALLS {
            break;
        }
        let want = (CLOCK_MIN_SAMPLE_NS as f64 / ns.max(1) as f64).ceil() as usize;
        calls = (calls * want.clamp(2, 16)).min(CLOCK_MAX_CALLS);
    }
    let mut per_call: Vec<f64> = (0..CLOCK_SAMPLES)
        .map(|_| run(calls, out) as f64 / calls as f64)
        .collect();
    per_call.sort_by(f64::total_cmp);
    let pixels = (trips.rows * trips.groups) as f64 * LANES as f64;
    Clock {
        ns_per_call_median: per_call[CLOCK_SAMPLES / 2],
        ns_per_call_min: per_call[0],
        ns_per_call_iqr: per_call[(CLOCK_SAMPLES * 3) / 4] - per_call[CLOCK_SAMPLES / 4],
        calls_per_sample: calls,
        ns_per_px: per_call[CLOCK_SAMPLES / 2] / pixels,
        scene_ns_per_px: None,
    }
}

fn cell_grid_scene_ns_per_px(case: &CellGridCase) -> f64 {
    let program = compile_cell_grid_for::<Rgba8>(case.shape, [0.0, 0.0, 0.0, 1.0]);
    let params = program.params(&case.metrics);
    let scene = Scene::CellGrid(program.frame(&params, case.cells.clone(), case.atlas.clone()));
    let (w, h) = (case.shape.frame_w, case.shape.frame_h);
    let mut frame = Frame::<Rgba8>::new(w, h);
    for _ in 0..2 {
        scene.render(&mut frame, 1);
    }
    let mut samples: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            scene.render(&mut frame, 1);
            std::hint::black_box(&frame.data[0]);
            t.elapsed().as_nanos() as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[2] / f64::from(w * h)
}

// ---------------------------------------------------------------------------
// The saturation probe: what fired, what was load-bearing
// ---------------------------------------------------------------------------

/// The legalizing prefix on its own — what both modes run before the
/// switch (`LowerDwrt`, `ExpandReduce`); `eval_scalar` needs it too.
fn legalize(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    match pipeline![LowerDwrt, ExpandReduce].optimize(arena, root) {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("legalizing prefix declined a real kernel"),
    }
}

fn saturation_probe(
    arena: &ExprArena,
    root: ExprId,
    optimizer: Optimizer,
    production_bytes: &[u8],
) -> SatProbe {
    let (la, lr) = legalize(arena, root);
    let mut optimizer = optimizer.observe(Some(Box::new(KeepJournal)));
    let mut egraph = optimizer.egraph();
    let root_class = insert(&la, lr, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|_| panic!("probe: real kernel not representable"));
    let node_count = reachable_count(&la, lr);
    let t = Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    let labels = EpisodeLabels::compute_tight(&egraph, root_class, &optimized.choices);
    let strict = EpisodeLabels::compute_strict(&egraph, root_class, &optimized.choices);
    let rules = optimizer.rule_set();
    let count = |idx: usize| -> RuleCount {
        let s = labels.rule_stats.get(&idx).copied().unwrap_or_default();
        let t = strict.rule_stats.get(&idx).copied().unwrap_or_default();
        RuleCount {
            rule: rules.label_of(idx).unwrap_or_else(|| format!("#{idx}")),
            fired: s.fired,
            load_bearing: s.load_bearing,
            strict: t.load_bearing,
        }
    };
    let select_hoist: Vec<RuleCount> = (0..rules.len())
        .filter(|&i| {
            rules
                .label_of(i)
                .is_some_and(|l| l.contains(SELECT_HOIST_PREFIX))
        })
        .map(count)
        .collect();
    let mut load_bearing_rules: Vec<RuleCount> = labels
        .rule_stats
        .iter()
        .filter(|(_, s)| s.load_bearing > 0)
        .map(|(&i, _)| count(i))
        .collect();
    load_bearing_rules.sort_by(|a, b| {
        b.load_bearing
            .cmp(&a.load_bearing)
            .then(a.rule.cmp(&b.rule))
    });

    // The probe's own extraction, compiled: it must be the production kernel.
    let (pa, pr) = optimized.to_arena(&egraph, root_class);
    let bytes_identical_to_production = if arena.buffers().is_empty() && arena.uniforms().is_empty()
    {
        let compiled = emit::compile(&pa, pr).expect("probe extraction compiles");
        Some(compiled.code.as_bytes() == production_bytes)
    } else {
        None
    };

    SatProbe {
        applications: optimized.stats.applications,
        unions: optimized.stats.unions,
        classes: optimized.stats.classes,
        iterations: optimized.stats.iterations,
        stop: format!("{:?}", optimized.stats.stop),
        wall_ms,
        select_hoist,
        load_bearing_rules,
        bytes_identical_to_production,
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn mode_label(variant: Option<Variant>, cap: Option<CapArm>) -> String {
    let env = std::env::var("PIXELFLOW_SATURATION").unwrap_or_else(|_| "on".to_string());
    match (variant, cap, env.as_str()) {
        (Some(_), Some(_), _) => panic!("--variant and --class-cap are separate arms"),
        (Some(v), None, "on") => v.label().to_string(),
        (None, Some(c), "on") => c.label(),
        (Some(_), None, other) | (None, Some(_), other) => {
            panic!("an in-harness arm needs PIXELFLOW_SATURATION unset or on, got {other:?}")
        }
        (None, None, "on" | "off") => env,
        (None, None, other) => panic!("PIXELFLOW_SATURATION must be on or off, got {other:?}"),
    }
}

fn head_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

struct RunArgs<'a> {
    out: &'a Path,
    variant: Option<Variant>,
    cap: Option<CapArm>,
    no_clock: bool,
    no_probe: bool,
    filter: Option<&'a str>,
    skip: &'a [String],
    font: Option<&'a Path>,
}

#[allow(clippy::too_many_lines)]
fn run(args: &RunArgs<'_>) {
    let RunArgs {
        out,
        variant,
        cap,
        no_clock,
        no_probe,
        filter,
        skip,
        font,
    } = *args;
    assert!(
        std::env::var_os("PIXELFLOW_GUARD_TELEMETRY").is_some(),
        "set PIXELFLOW_GUARD_TELEMETRY=1: the guard columns come from the emitter's own report"
    );
    assert!(
        std::env::var_os("PIXELFLOW_SATURATION_TELEMETRY").is_none(),
        "unset PIXELFLOW_SATURATION_TELEMETRY: the saturation columns are read back from the \
         record the optimizer prints to stderr, and a file sink would take them away"
    );
    let mode = mode_label(variant, cap);
    let default_font = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf");
    let mut kernels = real_kernels(font.unwrap_or(&default_font), filter);
    for name in skip {
        let before = kernels.len();
        kernels.retain(|k| &k.name != name);
        assert_eq!(
            kernels.len() + 1,
            before,
            "--skip {name}: not a kernel of this corpus"
        );
        eprintln!("egraph_off_on: skipping {name}");
    }
    eprintln!(
        "egraph_off_on: mode={mode} tier={} lanes={} kernels={}",
        collapse_bench::tier(),
        LANES,
        kernels.len()
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create out dir");
    }
    let mut sink = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .unwrap_or_else(|e| panic!("open {}: {e}", out.display()));
    let git_sha = head_sha();

    for (i, rk) in kernels.iter().enumerate() {
        // Linked: a glyph composes its winding sum by reference, and this
        // harness measures the arena the optimizer sees — which is the
        // referent spliced in, not a one-node name for it.
        let (arena, root) = rk.kernel.parts();
        let (arena, root) = pixelflow_ir::passes::expand_refs_owned(arena, root);
        let arena = &arena;
        // A glyph's winding sum reads a bound piece table (S1a,
        // docs/plans/2026-09-09-glyph-as-a-fold-execution.md): its kernel
        // declares a `Buffer` and carries the table's own data
        // (`Kernel::buffer_data`, `bee7813`), with no separate binding for
        // this harness to gather by hand. `context_for`/`oracle` resolve
        // every declared buffer against this before falling back to the
        // cell-grid case's externally-owned memory (`resolve_buffer`).
        let carried: Vec<(pixelflow_ir::arena::BufferIdentity, Arc<[f32]>)> = rk
            .kernel
            .buffer_data()
            .map(|(id, data)| (id, Arc::clone(data)))
            .collect();
        let shape = LatticeShape::new(rk.extent);
        let trips = Trips::of(rk.extent, LANES as u32);
        let started = Instant::now();

        let sizes = {
            let (la, lr) = legalize(arena, root);
            let mut egraph = Optimizer::production().egraph();
            insert(&la, lr, &mut egraph, Vocabulary::Runtime)
                .unwrap_or_else(|_| panic!("{}: not e-graph representable", rk.name));
            InputSizes {
                nodes: reachable_count(&la, lr),
                inserted: egraph.num_classes(),
            }
        };
        let in_harness_optimizer = |shape: LatticeShape| match (variant, cap) {
            (Some(v), None) => Some(v.optimizer(shape)),
            (None, Some(c)) => Some(c.optimizer(shape, sizes)),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("mode_label rejects both arms at once"),
        };
        let compiled = match in_harness_optimizer(shape) {
            Some(optimizer) => compile_via_optimizer(arena, root, optimizer),
            None => compile_via_production_path(arena, root, shape),
        };
        if mode != "off" {
            let sat = compiled.sat.as_ref().unwrap_or_else(|| {
                panic!(
                    "{}: saturation ran but left no telemetry record — build the harness with \
                     --features pixelflow-search/saturation-telemetry",
                    rk.name
                )
            });
            assert_eq!(
                sat.inserted_classes,
                Some(sizes.inserted),
                "{}: the optimizer's inserted class count and the harness's own insertion disagree",
                rk.name
            );
        }
        let bytes_identical_to_manifold_compile = if in_harness_optimizer(shape).is_none()
            && arena.buffers().is_empty()
            && arena.uniforms().is_empty()
        {
            let m = pixelflow_core::lattice::manifold::Manifold::compile(&rk.kernel, rk.extent);
            Some(m.code_bytes() == compiled.result.code.as_bytes())
        } else {
            None
        };

        let ctx = context_for(&compiled.linked, &carried, rk.cell_grid.as_ref());
        let mut buffer = vec![0.0f32; (trips.rows * trips.groups) as usize * LANES];
        run_once(&compiled.result.code, &ctx, trips, &mut buffer);
        let picture_hash = fnv(&buffer);
        let (legal, legal_root) = legalize(arena, root);
        let oracle = Some(oracle(
            &OracleForms {
                input: (&legal, legal_root),
                linked: (&compiled.linked, compiled.root),
                carried: &carried,
                case: rk.cell_grid.as_ref(),
                packed: rk.packed,
            },
            &buffer,
            trips,
        ));
        let clock = (!no_clock).then(|| {
            let mut c = clock(&compiled.result.code, &ctx, trips, &mut buffer);
            if let Some(case) = rk.cell_grid.as_ref() {
                c.scene_ns_per_px = Some(cell_grid_scene_ns_per_px(case));
            }
            c
        });
        let probe = (!no_probe && mode != "off").then(|| {
            let optimizer = in_harness_optimizer(shape)
                .unwrap_or_else(|| Optimizer::production().for_lattice(shape));
            saturation_probe(arena, root, optimizer, compiled.result.code.as_bytes())
        });

        let row = KernelRow {
            schema: SCHEMA.into(),
            mode: mode.clone(),
            tier: collapse_bench::tier().into(),
            lanes: LANES as u32,
            git_sha: git_sha.clone(),
            name: rk.name.clone(),
            class: rk.class.clone(),
            extent: rk.extent,
            packed: rk.packed,
            input_nodes: reachable(arena, root).len(),
            compiled_nodes: reachable(&compiled.linked, compiled.root).len(),
            dag_cost_input: dag_cost(arena, root),
            dag_cost: dag_cost(&compiled.linked, compiled.root),
            bytes: compiled.result.code.len() as u32,
            spill_slots: compiled.result.spill_count,
            hoisted: compiled.result.hoisted_values,
            statics: features_of(&compiled.result, trips),
            guard: compiled.guard,
            optimize_ms: compiled.optimize_ms,
            emit_ms: compiled.emit_ms,
            bytes_identical_to_manifold_compile,
            picture_hash,
            oracle,
            clock,
            probe,
            sat: compiled.sat,
            peak_alloc_bytes: Some(compiled.peak_alloc_bytes),
            class_cap: cap.map(|c| c.resolve(sizes).0),
            app_cap: cap.map(|c| c.resolve(sizes).1),
            inserted_classes: Some(sizes.inserted),
        };
        let line = serde_json::to_string(&row).expect("serialize row");
        writeln!(sink, "{line}").expect("append row");
        sink.flush().expect("flush");
        eprintln!(
            "[{}/{}] {} {}B dag={} opt={:.1}ms emit={:.1}ms {:?} ({:.1}s)",
            i + 1,
            kernels.len(),
            rk.name,
            row.bytes,
            row.dag_cost,
            row.optimize_ms,
            row.emit_ms,
            row.clock.as_ref().map(|c| c.ns_per_px),
            started.elapsed().as_secs_f64()
        );
    }
}

// ---------------------------------------------------------------------------
// consistency: the extractor's claim against the price of what it returned
// ---------------------------------------------------------------------------

const CONSISTENCY_HEADER: &str = "family,kernel,extent,cap,app_cap,input_nodes,inserted_classes,\
classes_after,live_classes,objective,scale,claimed,tree_cost,dag_cost,signed_error,error_frac,\
arena_nodes,arena_dag_unweighted,sharing_ratio,\
optimize_ms\n";

/// One kernel at one cap: saturate, extract, and compare the winning DP arm's
/// own minimized value against the price of the term it returned.
///
/// Nothing here is timed for a claim — `optimize_ms` is context, not a
/// result. Every other column is a deterministic function of the input, so
/// two hosts must produce the same file.
fn consistency(
    out: &Path,
    caps: &[usize],
    filter: Option<&str>,
    font: Option<&Path>,
    app_cap: Option<u64>,
) {
    assert!(!caps.is_empty(), "--class-cap: at least one arm");
    let default_font = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf");
    let kernels = real_kernels(font.unwrap_or(&default_font), filter);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create out dir");
    }
    let mut sink = std::fs::File::create(out).unwrap_or_else(|e| panic!("create {out:?}: {e}"));
    write!(sink, "{CONSISTENCY_HEADER}").expect("header");

    for (i, rk) in kernels.iter().enumerate() {
        let (arena, root) = rk.kernel.parts();
        let shape = LatticeShape::new(rk.extent);
        let (la, lr) = legalize(arena, root);
        let nodes = reachable_count(&la, lr);
        for &cap in caps {
            let arm = CapArm {
                rule: CapRule::Flat(cap),
                applications: app_cap,
            };
            let mut egraph = Optimizer::production().egraph();
            let root_class = insert(&la, lr, &mut egraph, Vocabulary::Runtime)
                .unwrap_or_else(|_| panic!("{}: not e-graph representable", rk.name));
            let inserted = egraph.num_classes();
            let sizes = InputSizes { nodes, inserted };
            let (resolved_cap, resolved_apps) = arm.resolve(sizes);
            let mut optimizer = arm.optimizer(shape, sizes);
            let t = Instant::now();
            let optimized = optimizer.run(&mut egraph, root_class, nodes);
            let optimize_ms = t.elapsed().as_secs_f64() * 1e3;

            let audit = optimized
                .extraction
                .audit
                .expect("production extraction always runs a DP");
            let actual = audit.scale.of(optimized.cost);
            let signed = audit.signed_error(optimized.cost);
            let (a, r) = optimized.to_arena(&egraph, root_class);
            let row = format!(
                "{family},{kernel},{w}x{h},{cap},{apps},{nodes},{inserted},{after},{live},\
{objective},{scale},{claimed},{tree},{dag},{signed},{frac:.6},\
{arena_nodes},{arena_dag},{ratio:.3},{ms:.1}\n",
                family = rk.class,
                kernel = rk.name,
                w = rk.extent[0],
                h = rk.extent[1],
                cap = resolved_cap,
                apps = resolved_apps,
                after = optimized.stats.classes,
                live = optimized
                    .extraction
                    .shared_pass
                    .map_or(0, |p| p.live_classes),
                objective = optimized.extraction.objective.as_str(),
                scale = audit.scale.as_str(),
                claimed = audit.claimed,
                tree = optimized.cost.tree,
                dag = optimized.cost.dag,
                signed = signed,
                frac = if actual == 0 {
                    0.0
                } else {
                    signed as f64 / actual as f64
                },
                arena_nodes = reachable(&a, r).len(),
                arena_dag = dag_cost(&a, r),
                ratio = optimized.cost.tree as f64 / optimized.cost.dag.max(1) as f64,
                ms = optimize_ms,
            );
            write!(sink, "{row}").expect("append row");
            sink.flush().expect("flush");
            eprintln!(
                "[{}/{}] {} cap={resolved_cap} live={} obj={} claimed={} dag={} err={signed} \
({optimize_ms:.0}ms)",
                i + 1,
                kernels.len(),
                rk.name,
                optimized
                    .extraction
                    .shared_pass
                    .map_or(0, |p| p.live_classes),
                optimized.extraction.objective.as_str(),
                audit.claimed,
                optimized.cost.dag,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// diff
// ---------------------------------------------------------------------------

fn read_rows(paths: &[PathBuf]) -> Vec<KernelRow> {
    let mut rows = Vec::new();
    for p in paths {
        let text =
            std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: KernelRow = serde_json::from_str(line).expect("parse row");
            assert_eq!(row.schema, SCHEMA, "{}: schema mismatch", p.display());
            rows.push(row);
        }
    }
    rows
}

/// One row per kernel: deterministic columns from the first file, the
/// clock as the median over every file's measurement.
fn collapse(rows: Vec<KernelRow>) -> BTreeMap<String, KernelRow> {
    let mut by: BTreeMap<String, Vec<KernelRow>> = BTreeMap::new();
    for r in rows {
        by.entry(r.name.clone()).or_default().push(r);
    }
    by.into_iter()
        .map(|(name, mut rs)| {
            for r in &rs[1..] {
                assert_eq!(r.bytes, rs[0].bytes, "{name}: bytes differ between passes");
                assert_eq!(
                    r.picture_hash, rs[0].picture_hash,
                    "{name}: picture differs between passes"
                );
            }
            let mut clocks: Vec<f64> = rs
                .iter()
                .filter_map(|r| r.clock.as_ref().map(|c| c.ns_per_px))
                .collect();
            clocks.sort_by(f64::total_cmp);
            let mut scene: Vec<f64> = rs
                .iter()
                .filter_map(|r| r.clock.as_ref().and_then(|c| c.scene_ns_per_px))
                .collect();
            scene.sort_by(f64::total_cmp);
            let mut opt: Vec<f64> = rs.iter().map(|r| r.optimize_ms).collect();
            opt.sort_by(f64::total_cmp);
            let mut em: Vec<f64> = rs.iter().map(|r| r.emit_ms).collect();
            em.sort_by(f64::total_cmp);
            let mut r = rs.swap_remove(0);
            // The clock may sit in a later file than the deterministic columns
            // (a `--no-clock` census merged with clocked passes): take the
            // first clock any pass carries, then median over all of them.
            let mut clock = r
                .clock
                .take()
                .or_else(|| rs.iter_mut().find_map(|x| x.clock.take()));
            if let Some(c) = clock.as_mut() {
                c.ns_per_px = clocks[clocks.len() / 2];
                c.scene_ns_per_px = (!scene.is_empty()).then(|| scene[scene.len() / 2]);
            }
            r.clock = clock;
            r.optimize_ms = opt[opt.len() / 2];
            r.emit_ms = em[em.len() / 2];
            (name, r)
        })
        .collect()
}

fn pct(on: f64, off: f64) -> f64 {
    if off == 0.0 {
        0.0
    } else {
        (on / off - 1.0) * 100.0
    }
}

fn fmt_pct(p: f64) -> String {
    format!("{p:+.1}%")
}

struct DiffInputs<'a> {
    off: &'a [PathBuf],
    on: &'a [PathBuf],
    with_select_hoist: &'a [PathBuf],
    cse_only: &'a [PathBuf],
}

struct Cmp {
    name: String,
    class: String,
    bytes_pct: f64,
    dag_pct: f64,
    clock_pct: Option<f64>,
    off_ns: Option<f64>,
    on_ns: Option<f64>,
    off_bytes: u32,
    on_bytes: u32,
    off_dag: usize,
    on_dag: usize,
    off_mem: u64,
    on_mem: u64,
    picture_identical: bool,
    rules: Vec<RuleCount>,
    sat_ms: f64,
    total_on_ms: f64,
    on_guard: Option<GuardTelemetry>,
    /// The `with-select-hoist` arm.
    sh_fired: usize,
    sh_lb: usize,
    sh_strict: usize,
    sh_bytes: Option<u32>,
    sh_guard: Option<GuardTelemetry>,
    sh_dag: Option<usize>,
    /// The `cse-only` arm.
    cse_bytes: Option<u32>,
    cse_dag: Option<usize>,
    cse_nodes: Option<usize>,
}

fn rules_cell(rules: &[RuleCount]) -> String {
    let v: Vec<String> = rules
        .iter()
        .take(5)
        .map(|r| format!("`{}` {}/{}/{}", r.rule, r.strict, r.load_bearing, r.fired))
        .collect();
    if v.is_empty() {
        "-".into()
    } else {
        v.join(", ")
    }
}

fn opt_str<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or("-".into(), |v| v.to_string())
}

fn opt_f(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{v:.2}"))
}

fn guard_str(g: &Option<GuardTelemetry>) -> String {
    g.as_ref()
        .map_or("-".to_string(), |g| format!("{}/{}", g.guarded, g.selects))
}

fn sched_str(g: &Option<GuardTelemetry>) -> String {
    g.as_ref()
        .map_or("-".to_string(), |g| g.schedule.to_string())
}

fn probe_str(r: &KernelRow) -> String {
    r.probe.as_ref().map_or("-".into(), |p| {
        format!(
            "{} apps / {} it / {} cls / {}",
            p.applications, p.iterations, p.classes, p.stop
        )
    })
}

#[allow(clippy::too_many_lines)]
fn diff(inputs: &DiffInputs<'_>, out_prefix: &Path, notes: &[String]) {
    let off = collapse(read_rows(inputs.off));
    let on = collapse(read_rows(inputs.on));
    let sh = collapse(read_rows(inputs.with_select_hoist));
    let cse = collapse(read_rows(inputs.cse_only));
    let names: Vec<String> = on
        .keys()
        .filter(|n| off.contains_key(*n))
        .cloned()
        .collect();
    assert!(!names.is_empty(), "no kernel present in both off and on");
    let on_only: Vec<&KernelRow> = on.values().filter(|r| !off.contains_key(&r.name)).collect();

    // ---- CSV -------------------------------------------------------------
    let mut csv = String::from(
        "kernel,class,extent_w,extent_h,packed,input_nodes,off_nodes,on_nodes,cse_nodes,input_dag_cost,off_dag_cost,on_dag_cost,cse_dag_cost,sh_dag_cost,\
         off_bytes,on_bytes,cse_bytes,sh_bytes,off_schedule,on_schedule,off_selects,on_selects,off_guarded,on_guarded,sh_guarded,\
         off_spill,on_spill,off_dyn_mem_ops,on_dyn_mem_ops,off_ns_px,on_ns_px,clock_pct,\
         off_optimize_ms,on_optimize_ms,on_emit_ms,on_applications,on_iterations,on_classes,on_stop,picture_identical,\
         same_form_max_abs_off,same_form_max_abs_on,cross_form_max_abs_on,packed_mismatch_cross_on,\
         select_hoist_fired,select_hoist_load_bearing,select_hoist_strict,top_load_bearing_rule\n",
    );
    let mut json_rows = Vec::new();
    let mut cmps: Vec<Cmp> = Vec::new();
    for name in &names {
        let a = &off[name];
        let b = &on[name];
        let h = sh.get(name);
        let c = cse.get(name);
        let clock_pct = match (a.clock.as_ref(), b.clock.as_ref()) {
            (Some(x), Some(y)) => Some(pct(y.ns_per_px, x.ns_per_px)),
            _ => None,
        };
        let hp = h.and_then(|h| h.probe.as_ref());
        let sh_fired: usize = hp.map_or(0, |p| p.select_hoist.iter().map(|r| r.fired).sum());
        let sh_lb: usize = hp.map_or(0, |p| p.select_hoist.iter().map(|r| r.load_bearing).sum());
        let sh_strict: usize = hp.map_or(0, |p| p.select_hoist.iter().map(|r| r.strict).sum());
        let rules = b
            .probe
            .as_ref()
            .map(|p| p.load_bearing_rules.clone())
            .unwrap_or_default();
        let top_rule = rules
            .first()
            .map(|r| format!("{} ({}/{}/{})", r.rule, r.strict, r.load_bearing, r.fired))
            .unwrap_or_else(|| "-".into());
        let g = |g: &Option<GuardTelemetry>, f: fn(&GuardTelemetry) -> u64| g.as_ref().map_or(0, f);
        let o = |o: &Option<Oracle>| o.clone().unwrap_or_default();
        let pb = b.probe.as_ref();
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{},{},{},{},{},{:e},{:e},{:e},{},{},{},{},{}\n",
            name, b.class, b.extent[0], b.extent[1], b.packed, b.input_nodes, a.compiled_nodes, b.compiled_nodes,
            opt_str(c.map(|c| c.compiled_nodes)),
            b.dag_cost_input, a.dag_cost, b.dag_cost, opt_str(c.map(|c| c.dag_cost)), opt_str(h.map(|h| h.dag_cost)),
            a.bytes, b.bytes, opt_str(c.map(|c| c.bytes)), opt_str(h.map(|h| h.bytes)),
            g(&a.guard, |g| g.schedule), g(&b.guard, |g| g.schedule),
            g(&a.guard, |g| g.selects), g(&b.guard, |g| g.selects),
            g(&a.guard, |g| g.guarded), g(&b.guard, |g| g.guarded), opt_str(h.map(|h| g(&h.guard, |g| g.guarded))),
            a.spill_slots, b.spill_slots, a.statics.dyn_memory_ops, b.statics.dyn_memory_ops,
            a.clock.as_ref().map_or(String::from("-"), |c| format!("{:.3}", c.ns_per_px)),
            b.clock.as_ref().map_or(String::from("-"), |c| format!("{:.3}", c.ns_per_px)),
            clock_pct.map_or(String::from("-"), |p| format!("{p:.1}")),
            a.optimize_ms, b.optimize_ms, b.emit_ms,
            opt_str(pb.map(|p| p.applications)), opt_str(pb.map(|p| p.iterations)), opt_str(pb.map(|p| p.classes)),
            pb.map_or("-".into(), |p| p.stop.clone()),
            a.picture_hash == b.picture_hash,
            o(&a.oracle).same_form_max_abs, o(&b.oracle).same_form_max_abs, o(&b.oracle).cross_form_max_abs,
            o(&b.oracle).packed_mismatch_cross,
            sh_fired, sh_lb, sh_strict,
            top_rule.replace(',', ";"),
        ));
        json_rows.push(serde_json::json!({
            "kernel": name, "class": b.class, "extent": b.extent, "packed": b.packed,
            "off": a, "on": b, "with_select_hoist": h, "cse_only": c,
        }));
        cmps.push(Cmp {
            name: name.clone(),
            class: b.class.clone(),
            bytes_pct: pct(f64::from(b.bytes), f64::from(a.bytes)),
            dag_pct: pct(b.dag_cost as f64, a.dag_cost as f64),
            clock_pct,
            off_ns: a.clock.as_ref().map(|c| c.ns_per_px),
            on_ns: b.clock.as_ref().map(|c| c.ns_per_px),
            off_bytes: a.bytes,
            on_bytes: b.bytes,
            off_dag: a.dag_cost,
            on_dag: b.dag_cost,
            off_mem: a.statics.dyn_memory_ops,
            on_mem: b.statics.dyn_memory_ops,
            picture_identical: a.picture_hash == b.picture_hash,
            rules,
            sat_ms: b.optimize_ms - a.optimize_ms,
            total_on_ms: b.optimize_ms + b.emit_ms,
            on_guard: b.guard.clone(),
            sh_fired,
            sh_lb,
            sh_strict,
            sh_bytes: h.map(|h| h.bytes),
            sh_guard: h.and_then(|h| h.guard.clone()),
            sh_dag: h.map(|h| h.dag_cost),
            cse_bytes: c.map(|c| c.bytes),
            cse_dag: c.map(|c| c.dag_cost),
            cse_nodes: c.map(|c| c.compiled_nodes),
        });
    }
    std::fs::write(out_prefix.with_extension("csv"), &csv).expect("write csv");

    // ---- per-class verdicts ---------------------------------------------
    let mut classes: BTreeMap<String, Vec<&Cmp>> = BTreeMap::new();
    for c in &cmps {
        classes.entry(c.class.clone()).or_default().push(c);
    }
    let median = |mut v: Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    };
    let mut md = String::new();
    md.push_str("# Saturation on vs off, on every shipped kernel\n\n");
    md.push_str(
        "The \"F: no e-graph\" column of [`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §7, \
         measured. Every kernel is compiled through the production path (`optimize_runtime_arena` → `relink` → `emit::compile`, the three calls \
         `jit_cache::compile` makes; buffer-free kernels are asserted byte-identical to `Manifold::compile`) twice: `PIXELFLOW_SATURATION=off` \
         runs the `Identity` path (`LowerDwrt`, `ExpandReduce`, no saturation), `on` is production. Two in-harness arms beside them: `cse-only` \
         (the production optimizer at zero rewrite rounds — insert, extract — so hash-consing's share is separated from the rules') and \
         `with-select-hoist` (production's rules plus the three `SelectHoistUnary` rules, which are **not** in `all_rules()`). \
         Deterministic columns are the claim; the clock, when taken, is a sign. Per-kernel rows: the `.csv`/`.json` beside this file.\n\n",
    );
    for n in notes {
        md.push_str(n);
        md.push('\n');
    }
    md.push('\n');
    md.push_str("## Verdict per shader class\n\n");
    md.push_str("`Δ` is on relative to off; negative is saturation winning. `mem ops` is the trip-weighted dynamic memory-op count (`dyn_memory_ops`). \
                 `cse` is the zero-round arm: the gap between `off` and `cse` is hash-consing, between `cse` and `on` is the rules.\n\n");
    md.push_str("| class | n | Σ bytes off → cse → on | Σ dag_cost off → cse → on | Σ mem ops off → on | median clock Δ | picture identical | verdict |\n|---|---:|---:|---:|---:|---:|---:|---|\n");
    let mut json_classes = Vec::new();
    for (class, cs) in &classes {
        let sb: (u64, u64) = cs.iter().fold((0, 0), |a, c| {
            (a.0 + u64::from(c.off_bytes), a.1 + u64::from(c.on_bytes))
        });
        let sd: (usize, usize) = cs
            .iter()
            .fold((0, 0), |a, c| (a.0 + c.off_dag, a.1 + c.on_dag));
        let sm: (u64, u64) = cs
            .iter()
            .fold((0, 0), |a, c| (a.0 + c.off_mem, a.1 + c.on_mem));
        let all_cse = cs.iter().all(|c| c.cse_bytes.is_some());
        let cse_b: u64 = cs.iter().map(|c| u64::from(c.cse_bytes.unwrap_or(0))).sum();
        let cse_d: usize = cs.iter().map(|c| c.cse_dag.unwrap_or(0)).sum();
        let clocks: Vec<f64> = cs.iter().filter_map(|c| c.clock_pct).collect();
        let mclock = median(clocks);
        let identical = cs.iter().filter(|c| c.picture_identical).count();
        let bytes_pct = pct(sb.1 as f64, sb.0 as f64);
        let dag_pct = pct(sd.1 as f64, sd.0 as f64);
        let verdict = match (bytes_pct, dag_pct, mclock) {
            (b, d, Some(c)) if (b < -2.0 || d < -2.0) && c < -10.0 => "helps (bytes+clock)",
            (b, d, _) if b < -2.0 || d < -2.0 => "helps (static)",
            (b, d, Some(c)) if (b > 2.0 || d > 2.0) && c > 10.0 => "HURTS (bytes+clock)",
            (b, d, _) if b > 2.0 || d > 2.0 => "hurts (static)",
            (_, _, Some(c)) if c < -10.0 => "clock says helps; static flat",
            (_, _, Some(c)) if c > 10.0 => "clock says hurts; static flat",
            _ => "nothing (|Δ| ≤ 2% static)",
        };
        let cse_bs = if all_cse {
            format!("{cse_b}")
        } else {
            "-".into()
        };
        let cse_ds = if all_cse {
            format!("{cse_d}")
        } else {
            "-".into()
        };
        md.push_str(&format!(
            "| {class} | {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} → {} ({}) | {} | {identical}/{} | {verdict} |\n",
            cs.len(), sb.0, cse_bs, sb.1, fmt_pct(bytes_pct), sd.0, cse_ds, sd.1, fmt_pct(dag_pct), sm.0, sm.1, fmt_pct(pct(sm.1 as f64, sm.0 as f64)),
            mclock.map_or("-".into(), fmt_pct), cs.len()
        ));
        json_classes.push(serde_json::json!({
            "class": class, "n": cs.len(), "bytes_off": sb.0, "bytes_on": sb.1, "bytes_pct": bytes_pct,
            "bytes_cse": all_cse.then_some(cse_b), "dag_cse": all_cse.then_some(cse_d),
            "dag_off": sd.0, "dag_on": sd.1, "dag_pct": dag_pct, "mem_off": sm.0, "mem_on": sm.1,
            "median_clock_pct": mclock, "picture_identical": identical, "verdict": verdict,
        }));
    }

    // ---- kernels the off path could not emit ------------------------------
    if !on_only.is_empty() {
        md.push_str("\n## Kernels with no `off` row: the un-saturated arena does not compile\n\n");
        md.push_str("Run with `--skip` in `off` mode because the emitter panics on them (the notes above quote the panic). The columns are the `on` row and, where present, the `cse-only` arm.\n\n");
        md.push_str("| kernel | extent | nodes in → cse → on | dag_cost in → cse → on | bytes cse → on | schedule on | guarded/selects on | spills on | saturation (apps / rounds / classes / stop) | compile on (ms) |\n|---|---|---|---:|---:|---:|---:|---:|---|---:|\n");
        for b in &on_only {
            let c = cse.get(&b.name);
            md.push_str(&format!(
                "| {} | {}×{} | {} → {} → {} | {} → {} → {} | {} → {} | {} | {} | {} | {} | {:.1} |\n",
                b.name, b.extent[0], b.extent[1], b.input_nodes, opt_str(c.map(|c| c.compiled_nodes)), b.compiled_nodes,
                b.dag_cost_input, opt_str(c.map(|c| c.dag_cost)), b.dag_cost, opt_str(c.map(|c| c.bytes)), b.bytes,
                sched_str(&b.guard), guard_str(&b.guard), b.spill_slots, probe_str(b), b.optimize_ms + b.emit_ms
            ));
        }
    }

    // ---- headline kernels ------------------------------------------------
    md.push_str("\n## The headline kernels\n\n| kernel | extent | nodes in → off → cse → on | bytes off → cse → on | dag_cost off → cse → on | schedule off → on | guarded/selects off → on | spills off → on | mem ops off → on | saturation (apps / rounds / classes / stop) | compile off → on (ms; saturation share) |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|\n");
    for c in &cmps {
        if !matches!(
            c.class.as_str(),
            "chrome"
                | "chrome_channel"
                | "psychedelic"
                | "cellgrid"
                | "bench"
                | "bench_wide"
                | "shader"
        ) {
            continue;
        }
        let a = &off[&c.name];
        let b = &on[&c.name];
        let share = if c.total_on_ms > 0.0 {
            c.sat_ms / c.total_on_ms * 100.0
        } else {
            0.0
        };
        md.push_str(&format!(
            "| {} | {}×{} | {} → {} → {} → {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} → {} | {} → {} | {} → {} | {} → {} ({}) | {} | {:.1} → {:.1} ({:.0}%) |\n",
            c.name, b.extent[0], b.extent[1], b.input_nodes, a.compiled_nodes, opt_str(c.cse_nodes), b.compiled_nodes,
            c.off_bytes, opt_str(c.cse_bytes), c.on_bytes, fmt_pct(c.bytes_pct), c.off_dag, opt_str(c.cse_dag), c.on_dag, fmt_pct(c.dag_pct),
            sched_str(&a.guard), sched_str(&b.guard), guard_str(&a.guard), guard_str(&b.guard), a.spill_slots, b.spill_slots,
            c.off_mem, c.on_mem, fmt_pct(pct(c.on_mem as f64, c.off_mem as f64)),
            probe_str(b),
            a.optimize_ms + a.emit_ms, c.total_on_ms, share
        ));
    }

    // ---- prologue on O@32 -------------------------------------------------
    if let (Some(n), Some(w)) = (
        cmps.iter().find(|c| c.name == "bench_O_quadratic"),
        cmps.iter().find(|c| c.name == "bench_O_quadratic_wide"),
    ) && n.off_ns.is_some()
        && w.off_ns.is_some()
    {
        let prologue = |narrow: Option<f64>, wide: Option<f64>| -> Option<f64> {
            let (tn, tw) = (
                narrow? * f64::from(BENCH_EXTENT[0] * BENCH_EXTENT[1]),
                wide? * f64::from(BENCH_WIDE_EXTENT[0] * BENCH_WIDE_EXTENT[1]),
            );
            let rows = f64::from(BENCH_EXTENT[1]);
            let (gn, gw) = (
                f64::from(BENCH_EXTENT[0]) / LANES as f64,
                f64::from(BENCH_WIDE_EXTENT[0]) / LANES as f64,
            );
            let per_row_n = tn / rows;
            let per_row_w = tw / rows;
            let b = (per_row_w - per_row_n) / (gw - gn);
            Some((per_row_n - b * gn) / 1e3)
        };
        md.push_str(&format!(
                "\n## Row prologue on `O`@32 (from the 40- and 640-wide rows, two-point fit)\n\n| | off | on |\n|---|---:|---:|\n| ns/px at 40×45 | {} | {} |\n| ns/px at 640×45 | {} | {} |\n| per-row prologue (µs) | {} | {} |\n",
                opt_f(n.off_ns), opt_f(n.on_ns), opt_f(w.off_ns), opt_f(w.on_ns),
                opt_f(prologue(n.off_ns, w.off_ns)), opt_f(prologue(n.on_ns, w.on_ns)),
            ));
    }

    // ---- regressions and their rules -------------------------------------
    md.push_str("\n## Where saturation makes the kernel worse, and which rule\n\n");
    md.push_str("Sorted by bytes Δ. Rule column: the on-extraction's rules from the provenance journal, `strict/tight/fired` — strict credits an application only when its own e-node was chosen (`EpisodeLabels::compute_strict`), tight is `derivation_ancestors_tight`.\n\n");
    md.push_str("| kernel | bytes off → cse → on | dag_cost off → cse → on | mem ops Δ | rules (strict/tight/fired) |\n|---|---:|---:|---:|---|\n");
    let mut worse: Vec<&Cmp> = cmps
        .iter()
        .filter(|c| c.bytes_pct > 0.0 || c.dag_pct > 0.0)
        .collect();
    worse.sort_by(|a, b| b.bytes_pct.partial_cmp(&a.bytes_pct).unwrap());
    for c in worse.iter().take(15) {
        md.push_str(&format!(
            "| {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} | {} |\n",
            c.name,
            c.off_bytes,
            opt_str(c.cse_bytes),
            c.on_bytes,
            fmt_pct(c.bytes_pct),
            c.off_dag,
            opt_str(c.cse_dag),
            c.on_dag,
            fmt_pct(c.dag_pct),
            fmt_pct(pct(c.on_mem as f64, c.off_mem as f64)),
            rules_cell(&c.rules)
        ));
    }
    if worse.is_empty() {
        md.push_str("| (none: no kernel grew in bytes or dag_cost) | | | | |\n");
    }
    let mut better: Vec<&Cmp> = cmps.iter().filter(|c| c.bytes_pct < 0.0).collect();
    better.sort_by(|a, b| a.bytes_pct.partial_cmp(&b.bytes_pct).unwrap());
    md.push_str("\n### The largest wins, for the same rule attribution\n\n| kernel | bytes off → cse → on | dag_cost off → cse → on | rules (strict/tight/fired) |\n|---|---:|---:|---|\n");
    for c in better.iter().take(10) {
        md.push_str(&format!(
            "| {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} |\n",
            c.name,
            c.off_bytes,
            opt_str(c.cse_bytes),
            c.on_bytes,
            fmt_pct(c.bytes_pct),
            c.off_dag,
            opt_str(c.cse_dag),
            c.on_dag,
            fmt_pct(c.dag_pct),
            rules_cell(&c.rules)
        ));
    }

    // ---- rule census -----------------------------------------------------
    let mut census: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new();
    for c in &cmps {
        for r in &c.rules {
            let e = census.entry(r.rule.clone()).or_default();
            e.0 += r.fired;
            e.1 += r.load_bearing;
            e.2 += r.strict;
            e.3 += 1;
        }
    }
    let mut census: Vec<(String, (usize, usize, usize, usize))> = census.into_iter().collect();
    census.sort_by_key(|(_, (_, _, strict, _))| std::cmp::Reverse(*strict));
    md.push_str("\n## Which rules are load-bearing on real shaders (all kernels, production run)\n\n| rule | kernels where load-bearing | Σ strict | Σ tight | Σ fired |\n|---|---:|---:|---:|---:|\n");
    for (rule, (fired, lb, strict, n)) in census.iter().take(25) {
        md.push_str(&format!("| `{rule}` | {n} | {strict} | {lb} | {fired} |\n"));
    }

    // ---- SelectHoistUnary --------------------------------------------------
    let total_fired: usize = cmps.iter().map(|c| c.sh_fired).sum();
    let total_lb: usize = cmps.iter().map(|c| c.sh_lb).sum();
    let total_strict: usize = cmps.iter().map(|c| c.sh_strict).sum();
    let fired_in: usize = cmps.iter().filter(|c| c.sh_fired > 0).count();
    let lb_in: usize = cmps.iter().filter(|c| c.sh_lb > 0).count();
    let measured = cmps.iter().filter(|c| c.sh_bytes.is_some()).count();
    md.push_str(&format!(
        "\n## `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`)\n\n\
         **Not in production.** The three rules live in `round2_rules::experimental_rules()` and are not part of `all_rules()`, \
         so they fire zero times in every production compile above by construction. The `with-select-hoist` arm adds them to \
         production's rules: fired in **{fired_in} of {measured}** measured kernels ({total_fired} applications); tight-load-bearing in \
         **{lb_in}** ({total_lb}); strict {total_strict}.\n\n",
    ));
    md.push_str("| kernel | fired | tight | strict | bytes on → +hoist | dag_cost on → +hoist | guarded/selects on → +hoist | schedule on → +hoist |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut shv: Vec<&Cmp> = cmps
        .iter()
        .filter(|c| c.sh_fired > 0 || c.sh_bytes.is_some_and(|m| m != c.on_bytes))
        .collect();
    shv.sort_by(|a, b| b.sh_lb.cmp(&a.sh_lb).then(b.sh_fired.cmp(&a.sh_fired)));
    for c in shv.iter().take(30) {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} → {} | {} → {} | {} → {} | {} → {} |\n",
            c.name,
            c.sh_fired,
            c.sh_lb,
            c.sh_strict,
            c.on_bytes,
            opt_str(c.sh_bytes),
            c.on_dag,
            opt_str(c.sh_dag),
            guard_str(&c.on_guard),
            guard_str(&c.sh_guard),
            sched_str(&c.on_guard),
            sched_str(&c.sh_guard),
        ));
    }
    if shv.is_empty() {
        md.push_str(
            "| (never fired on any real shader; bytes identical everywhere) | | | | | | | |\n",
        );
    }
    let sh_changed = cmps
        .iter()
        .filter(|c| c.sh_bytes.is_some_and(|m| m != c.on_bytes))
        .count();
    let guard_delta: i64 = cmps
        .iter()
        .filter_map(|c| {
            Some(
                i64::try_from(c.sh_guard.as_ref()?.guarded).ok()?
                    - i64::try_from(c.on_guard.as_ref()?.guarded).ok()?,
            )
        })
        .sum();
    md.push_str(&format!("\nAdding the rule changed the emitted bytes of **{sh_changed}** kernels; Σ guarded-value delta (+hoist − on) over all measured kernels: **{guard_delta:+}**.\n"));

    // ---- correctness -----------------------------------------------------
    let same_worst = cmps
        .iter()
        .filter_map(|c| {
            on[&c.name].oracle.as_ref().map(|o| {
                (
                    c.name.clone(),
                    o.same_form_max_abs,
                    o.same_form_nan_mismatch,
                )
            })
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let cross_worst = cmps
        .iter()
        .filter_map(|c| {
            on[&c.name].oracle.as_ref().map(|o| {
                (
                    c.name.clone(),
                    o.cross_form_max_abs,
                    o.cross_form_nan_mismatch,
                )
            })
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let nan_same: usize = cmps
        .iter()
        .filter_map(|c| on[&c.name].oracle.as_ref())
        .map(|o| o.same_form_nan_mismatch)
        .sum();
    let nan_same_off: usize = cmps
        .iter()
        .filter_map(|c| off[&c.name].oracle.as_ref())
        .map(|o| o.same_form_nan_mismatch)
        .sum();
    let same_worst_off = cmps
        .iter()
        .filter_map(|c| {
            off[&c.name]
                .oracle
                .as_ref()
                .map(|o| (c.name.clone(), o.same_form_max_abs))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let packed_cross: Vec<String> = on.values().filter(|r| r.packed).map(|r| {
        let o = r.oracle.clone().unwrap_or_default();
        format!("`{}`: same-form {} mismatching of {} sampled pixels (max byte Δ {}), cross-form {} (max byte Δ {})", r.name, o.packed_mismatch_same, o.points, o.packed_max_byte_same, o.packed_mismatch_cross, o.packed_max_byte_cross)
    }).collect();
    let identical = cmps.iter().filter(|c| c.picture_identical).count();
    md.push_str(&format!(
        "\n## Correctness\n\n\
         Same-form: `eval_scalar` of the emitted arena vs the JIT at {ORACLE_POINTS} sampled pixels (a difference here is a JIT bug). Cross-form: `eval_scalar` of the legalized arena as constructed vs the JIT of the on-extraction (what the rewrites moved; divergence at singularities is the algebraic contract, not a defect).\n\n\
         - same-form NaN mismatches over all kernels: on **{nan_same}**, off **{nan_same_off}**; worst same-form |Δ|: on {}, off {}\n\
         - worst cross-form |Δ| (on): {}\n\
         - full-extent output bit-identical off vs on: **{identical} of {}** kernels\n\
         - packed kernels (on): {}\n",
        same_worst.map_or("-".into(), |(n, d, nan)| format!("`{n}` {d:e} ({nan} NaN)")),
        same_worst_off.map_or("-".into(), |(n, d)| format!("`{n}` {d:e}")),
        cross_worst.map_or("-".into(), |(n, d, nan)| format!("`{n}` {d:e} ({nan} NaN)")),
        cmps.len(),
        packed_cross.join("; "),
    ));
    let manifold_check: Vec<&str> = on
        .keys()
        .filter(|n| {
            on[*n].bytes_identical_to_manifold_compile == Some(false)
                || off
                    .get(*n)
                    .is_some_and(|r| r.bytes_identical_to_manifold_compile == Some(false))
        })
        .map(String::as_str)
        .collect();
    let probe_check: Vec<&str> = on
        .keys()
        .filter(|n| {
            on[*n]
                .probe
                .as_ref()
                .is_some_and(|p| p.bytes_identical_to_production == Some(false))
        })
        .map(String::as_str)
        .collect();
    md.push_str(&format!(
        "- instrument = production path: `Manifold::compile` bytes differed for {} kernels {:?}; probe extraction differed from production for {} kernels {:?}\n",
        manifold_check.len(), manifold_check, probe_check.len(), probe_check
    ));

    // ---- compile-time share ---------------------------------------------
    md.push_str("\n## Saturation's share of compile time\n\n`optimize_ms` is `optimize_runtime_arena` (legalize + saturate + extract); `emit_ms` is `emit::compile`. Wall clock at the load stated above — a ratio, not a number.\n\n| class | Σ compile off (ms) | Σ compile on (ms) | Σ saturation (on − off optimize) | share of on |\n|---|---:|---:|---:|---:|\n");
    for (class, cs) in &classes {
        let off_ms: f64 = cs
            .iter()
            .map(|c| off[&c.name].optimize_ms + off[&c.name].emit_ms)
            .sum();
        let on_ms: f64 = cs.iter().map(|c| c.total_on_ms).sum();
        let sat: f64 = cs.iter().map(|c| c.sat_ms).sum();
        md.push_str(&format!(
            "| {class} | {off_ms:.1} | {on_ms:.1} | {sat:.1} | {:.0}% |\n",
            if on_ms > 0.0 {
                sat / on_ms * 100.0
            } else {
                0.0
            }
        ));
    }
    for b in &on_only {
        md.push_str(&format!(
            "| {} (on only) | - | {:.1} | {:.1} | - |\n",
            b.name,
            b.optimize_ms + b.emit_ms,
            b.optimize_ms
        ));
    }

    std::fs::write(out_prefix.with_extension("md"), &md).expect("write md");
    let json = serde_json::json!({
        "schema": SCHEMA,
        "classes": json_classes,
        "on_only": on_only,
        "select_hoist": { "in_production": false, "fired_total": total_fired, "tight_total": total_lb, "strict_total": total_strict, "kernels_fired": fired_in, "kernels_tight": lb_in, "changed_bytes": sh_changed, "guarded_delta_hoist_minus_on": guard_delta },
        "rule_census": census.iter().map(|(r, (f, lb, strict, n))| serde_json::json!({"rule": r, "fired": f, "tight": lb, "strict": strict, "kernels": n})).collect::<Vec<_>>(),
        "kernels": json_rows,
    });
    std::fs::write(
        out_prefix.with_extension("json"),
        serde_json::to_string_pretty(&json).expect("json"),
    )
    .expect("write json");
    print!("{md}");
}

// ---------------------------------------------------------------------------
// cap-sweep: the class-cap sweep across arms
// ---------------------------------------------------------------------------

/// One `(arm, class)` cell of the sweep.
#[derive(Serialize, Default, Clone, Debug)]
struct CapCell {
    /// The arm's label (`mode`): `cap<N>-app<M>` or `capx<R>-<floor>-<ceiling>-app<M>`.
    arm: String,
    /// The class caps the arm resolved to over this cell's kernels.
    class_cap_min: usize,
    class_cap_max: usize,
    app_cap_min: u64,
    app_cap_max: u64,
    class: String,
    n: usize,
    /// Stop reasons, as the telemetry names them.
    stops: BTreeMap<String, usize>,
    /// Rows that stopped on the class cap in their first round — the cap
    /// clipping the input before any rule ran.
    class_cap_in_round_1: usize,
    iterations_median: f64,
    iterations_max: usize,
    applications_median: f64,
    applications_max: u64,
    classes_at_stop_median: f64,
    classes_at_stop_max: usize,
    live_classes_max: usize,
    shared_pass_bytes_max: usize,
    objectives: BTreeMap<String, usize>,
    sum_compiled_nodes: usize,
    sum_bytes: u64,
    sum_dag_cost: usize,
    sum_schedule: u64,
    sum_guarded: u64,
    sum_selects: u64,
    sum_spills: u64,
    sum_optimize_ms: f64,
    sum_emit_ms: f64,
    peak_alloc_mb_max: f64,
    same_form_nan_mismatch: usize,
    same_form_max_abs: f64,
    packed_mismatch_same: usize,
}

fn median_of<T: Copy + Into<f64>>(v: &mut [T]) -> f64 {
    assert!(!v.is_empty(), "median of nothing");
    v.sort_by(|a, b| (*a).into().partial_cmp(&(*b).into()).expect("finite"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2].into()
    } else {
        (v[n / 2 - 1].into() + v[n / 2].into()) / 2.0
    }
}

fn cap_cell(arm: &str, class: &str, rows: &[&KernelRow]) -> CapCell {
    fn sat(r: &KernelRow) -> &SatTelemetry {
        r.sat.as_ref().unwrap_or_else(|| {
            panic!(
                "{}: a cap-sweep row without its saturation telemetry",
                r.name
            )
        })
    }
    let mut cell = CapCell {
        arm: arm.to_string(),
        class_cap_min: usize::MAX,
        app_cap_min: u64::MAX,
        class: class.to_string(),
        n: rows.len(),
        ..CapCell::default()
    };
    let mut iterations = Vec::new();
    let mut applications = Vec::new();
    let mut classes = Vec::new();
    for r in rows {
        let s = sat(r);
        let (c, a) = (r.class_cap.expect("cap row"), r.app_cap.expect("cap row"));
        cell.class_cap_min = cell.class_cap_min.min(c);
        cell.class_cap_max = cell.class_cap_max.max(c);
        cell.app_cap_min = cell.app_cap_min.min(a);
        cell.app_cap_max = cell.app_cap_max.max(a);
        *cell.stops.entry(s.stop_reason.clone()).or_default() += 1;
        if s.stop_reason == "class_cap" && s.iterations <= 1 {
            cell.class_cap_in_round_1 += 1;
        }
        iterations.push(s.iterations as u32);
        applications.push(s.application_count as f64);
        classes.push(s.classes_at_stop as u32);
        cell.iterations_max = cell.iterations_max.max(s.iterations);
        cell.applications_max = cell.applications_max.max(s.application_count);
        cell.classes_at_stop_max = cell.classes_at_stop_max.max(s.classes_at_stop);
        cell.live_classes_max = cell.live_classes_max.max(s.live_classes.unwrap_or(0));
        cell.shared_pass_bytes_max = cell
            .shared_pass_bytes_max
            .max(s.shared_pass_bytes.unwrap_or(0));
        *cell
            .objectives
            .entry(s.extraction_objective.clone())
            .or_default() += 1;
        cell.sum_compiled_nodes += r.compiled_nodes;
        cell.sum_bytes += u64::from(r.bytes);
        cell.sum_dag_cost += r.dag_cost;
        if let Some(g) = &r.guard {
            cell.sum_schedule += g.schedule;
            cell.sum_guarded += g.guarded;
            cell.sum_selects += g.selects;
        }
        cell.sum_spills += u64::from(r.spill_slots);
        cell.sum_optimize_ms += r.optimize_ms;
        cell.sum_emit_ms += r.emit_ms;
        let peak = r
            .peak_alloc_bytes
            .unwrap_or_else(|| panic!("{}: a cap-sweep row without its peak", r.name));
        cell.peak_alloc_mb_max = cell.peak_alloc_mb_max.max(peak as f64 / 1e6);
        if let Some(o) = &r.oracle {
            cell.same_form_nan_mismatch += o.same_form_nan_mismatch;
            cell.same_form_max_abs = cell.same_form_max_abs.max(o.same_form_max_abs);
            cell.packed_mismatch_same += o.packed_mismatch_same;
        }
    }
    cell.iterations_median = median_of(&mut iterations);
    cell.applications_median = median_of(&mut applications);
    cell.classes_at_stop_median = median_of(&mut classes);
    cell
}

fn stops_str(stops: &BTreeMap<String, usize>, n: usize) -> String {
    stops
        .iter()
        .map(|(k, v)| format!("{k} {v}/{n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn range_str<T: PartialEq + std::fmt::Display>(lo: T, hi: T) -> String {
    if lo == hi {
        lo.to_string()
    } else {
        format!("{lo}–{hi}")
    }
}

fn signed_pct(now: f64, base: f64) -> String {
    if base == 0.0 {
        return "-".to_string();
    }
    format!("{:+.1}%", (now - base) / base * 100.0)
}

#[allow(clippy::too_many_lines)]
fn cap_sweep(paths: &[PathBuf], out_prefix: &Path, notes: &[String]) {
    let rows = read_rows(paths);
    assert!(!rows.is_empty(), "no rows");
    // (arm, class) -> rows; a kernel appears once per arm. Arms are ordered
    // by the smallest cap they resolved to, then by label.
    let mut cells: BTreeMap<(usize, String, String), Vec<&KernelRow>> = BTreeMap::new();
    let mut arm_min_cap: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        let Some(c) = r.class_cap else {
            panic!("{}: not a cap-arm row (mode {})", r.name, r.mode);
        };
        let e = arm_min_cap.entry(r.mode.as_str()).or_insert(c);
        *e = (*e).min(c);
    }
    for r in &rows {
        let key = (
            arm_min_cap[r.mode.as_str()],
            r.mode.clone(),
            r.class.clone(),
        );
        let bucket = cells.entry(key).or_default();
        assert!(
            !bucket.iter().any(|x| x.name == r.name),
            "{}: two rows in arm {} — merge or dedupe the inputs",
            r.name,
            r.mode
        );
        bucket.push(r);
    }
    let cells: Vec<CapCell> = cells
        .iter()
        .map(|((_, arm, class), rs)| cap_cell(arm, class, rs))
        .collect();
    let baseline_arm = "cap5000-app200000";
    assert!(
        cells.iter().any(|c| c.arm == baseline_arm),
        "the baseline arm {baseline_arm} (production's classical cap) is not among the rows"
    );
    let baseline = |class: &str| -> Option<&CapCell> {
        cells
            .iter()
            .find(|c| c.arm == baseline_arm && c.class == class)
    };

    // Per-kernel CSV: every deterministic column, one line per (arm, kernel).
    let mut csv = String::from(
        "kernel,class,arm,class_cap,app_cap,tier_nodes,inserted_classes,stop,iterations,applications,classes_at_stop,live_classes,\
         objective,shared_pass_bytes,input_nodes,compiled_nodes,bytes,dag_cost,schedule,selects,guarded,exclusive,\
         spill_slots,hoisted,same_form_nan_mismatch,same_form_max_abs,packed_mismatch_same,optimize_ms,emit_ms,\
         sat_wall_ms,peak_alloc_mb,git_sha\n",
    );
    for r in &rows {
        let s = r.sat.as_ref().expect("checked above");
        let g = |f: fn(&GuardTelemetry) -> u64| r.guard.as_ref().map_or(0, f);
        let o = r.oracle.as_ref();
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.2},{}\n",
            r.name,
            r.class,
            r.mode,
            r.class_cap.expect("cap row"),
            r.app_cap.expect("cap row"),
            s.node_count,
            opt_str(r.inserted_classes),
            s.stop_reason,
            s.iterations,
            s.application_count,
            s.classes_at_stop,
            opt_str(s.live_classes),
            s.extraction_objective,
            opt_str(s.shared_pass_bytes),
            r.input_nodes,
            r.compiled_nodes,
            r.bytes,
            r.dag_cost,
            g(|g| g.schedule),
            g(|g| g.selects),
            g(|g| g.guarded),
            g(|g| g.exclusive),
            r.spill_slots,
            r.hoisted,
            opt_str(o.map(|o| o.same_form_nan_mismatch)),
            opt_str(o.map(|o| o.same_form_max_abs)),
            opt_str(o.map(|o| o.packed_mismatch_same)),
            r.optimize_ms,
            r.emit_ms,
            s.wall_clock_us as f64 / 1e3,
            r.peak_alloc_bytes.expect("cap row") as f64 / 1e6,
            r.git_sha,
        ));
    }
    std::fs::write(out_prefix.with_extension("csv"), csv).expect("write csv");

    let mut md = String::new();
    md.push_str("# The class-cap sweep\n\n");
    md.push_str(
        "Every row is one kernel compiled through the production path's three calls with the \
         production optimizer held to `--class-cap N --app-cap M` (classical's 100 rounds; \
         `Budget::Explicit`). Saturation columns are the optimizer's own `saturation-telemetry` \
         record for that compile, read back from stderr; `objective` is which extraction \
         objective produced the term (`shared` / `tree_cheaper` both ran the sharing-aware pass; \
         `tree_only` means it was abandoned at `SHARED_DAG_PASS_BYTE_BUDGET`). Deterministic \
         columns are the claim; `compile ms` is `optimize_ms + emit_ms` under the counting \
         allocator and is a sign at the stated load; `peak MB` is the peak net heap growth of the \
         compile (`alloc_probe`).\n\n",
    );
    for n in notes {
        md.push_str(n);
        md.push('\n');
    }
    md.push_str("\n## Per family, per arm\n\n");
    md.push_str(
        "| class | arm | cap (min–max) | apps (min–max) | n | stop | cap in round 1 | rounds med / max | apps med / max | classes med / max | live max | objective | Σ nodes | Σ bytes (vs base) | Σ dag_cost (vs base) | Σ guarded/schedule | Σ spills | oracle NaN / max abs | Σ compile ms | peak MB |\n|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for c in &cells {
        let base = baseline(&c.class);
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} / {} | {} | {} | {} | {} ({}) | {} ({}) | {}/{} | {} | {} / {:.3e} | {:.0} | {:.1} |\n",
            c.class,
            c.arm,
            range_str(c.class_cap_min, c.class_cap_max),
            range_str(c.app_cap_min, c.app_cap_max),
            c.n,
            stops_str(&c.stops, c.n),
            c.class_cap_in_round_1,
            c.iterations_median,
            c.iterations_max,
            c.applications_median,
            c.applications_max,
            c.classes_at_stop_median,
            c.classes_at_stop_max,
            c.live_classes_max,
            stops_str(&c.objectives, c.n),
            c.sum_compiled_nodes,
            c.sum_bytes,
            base.map_or("-".into(), |b| signed_pct(c.sum_bytes as f64, b.sum_bytes as f64)),
            c.sum_dag_cost,
            base.map_or("-".into(), |b| signed_pct(
                c.sum_dag_cost as f64,
                b.sum_dag_cost as f64
            )),
            c.sum_guarded,
            c.sum_schedule,
            c.sum_spills,
            c.same_form_nan_mismatch,
            c.same_form_max_abs,
            c.sum_optimize_ms + c.sum_emit_ms,
            c.peak_alloc_mb_max,
        ));
    }

    // Per-kernel movement against the baseline cap, per arm: which kernels
    // got better or worse in bytes / dag_cost, and by how much.
    md.push_str("\n## Kernels that moved against the baseline cap\n\n");
    md.push_str(
        "For each arm, the kernels whose emitted bytes or `dag_cost` differ from the same kernel \
         at the baseline (smallest) cap and the same application arm where one exists, else \
         the baseline's only arm. `+` is worse.\n\n",
    );
    let by_name_at_base: BTreeMap<&str, &KernelRow> = rows
        .iter()
        .filter(|r| r.mode == baseline_arm)
        .map(|r| (r.name.as_str(), r))
        .collect();
    let mut arms: Vec<(usize, &str)> = arm_min_cap.iter().map(|(a, c)| (*c, *a)).collect();
    arms.sort_unstable();
    let mut moved_json = Vec::new();
    for (_, arm) in arms {
        if arm == baseline_arm {
            continue;
        }
        let mut better = 0usize;
        let mut worse = 0usize;
        let mut same = 0usize;
        let mut lines = Vec::new();
        for r in rows.iter().filter(|r| r.mode == arm) {
            let Some(b) = by_name_at_base.get(r.name.as_str()) else {
                continue;
            };
            let db = i64::from(r.bytes) - i64::from(b.bytes);
            let dd = r.dag_cost as i64 - b.dag_cost as i64;
            match (db.signum() + dd.signum()).signum() {
                0 if db == 0 && dd == 0 => same += 1,
                0 => {
                    worse += 1;
                    lines.push((r.name.clone(), db, dd));
                }
                1 => {
                    worse += 1;
                    lines.push((r.name.clone(), db, dd));
                }
                _ => {
                    better += 1;
                    lines.push((r.name.clone(), db, dd));
                }
            }
        }
        lines.sort_by(|a, b| (b.1 + b.2).cmp(&(a.1 + a.2)).then(a.0.cmp(&b.0)));
        md.push_str(&format!(
            "### {arm}: {better} better, {worse} worse, {same} unchanged\n\n"
        ));
        if !lines.is_empty() {
            md.push_str("| kernel | Δ bytes | Δ dag_cost |\n|---|---:|---:|\n");
            for (name, db, dd) in &lines {
                md.push_str(&format!("| {name} | {db:+} | {dd:+} |\n"));
            }
            md.push('\n');
        }
        moved_json.push(serde_json::json!({
            "arm": arm, "better": better, "worse": worse, "same": same,
            "moved": lines.iter().map(|(n, db, dd)| serde_json::json!({"kernel": n, "d_bytes": db, "d_dag_cost": dd})).collect::<Vec<_>>(),
        }));
    }

    let json = serde_json::json!({
        "schema": "class-cap-sweep-v1",
        "baseline_arm": baseline_arm,
        "notes": notes,
        "cells": cells,
        "movement": moved_json,
    });
    std::fs::write(
        out_prefix.with_extension("json"),
        serde_json::to_string_pretty(&json).expect("json"),
    )
    .expect("write json");
    std::fs::write(out_prefix.with_extension("md"), &md).expect("write md");
    print!("{md}");
}

fn main() {
    match Cli::parse().command {
        Command::Run {
            out,
            variant,
            no_clock,
            no_probe,
            filter,
            skip,
            font,
            class_cap,
            classes_per_node,
            classes_per_inserted,
            cap_floor,
            cap_ceiling,
            app_cap,
        } => run(&RunArgs {
            out: &out,
            variant,
            cap: match (class_cap, classes_per_node, classes_per_inserted) {
                (Some(c), None, None) => Some(CapArm {
                    rule: CapRule::Flat(c),
                    applications: app_cap,
                }),
                (None, Some(r), None) => Some(CapArm {
                    rule: CapRule::PerNode {
                        classes_per_node: r,
                        floor: cap_floor.expect("clap: requires cap_floor"),
                        ceiling: cap_ceiling.expect("clap: requires cap_ceiling"),
                    },
                    applications: app_cap,
                }),
                (None, None, Some(r)) => Some(CapArm {
                    rule: CapRule::PerInserted {
                        classes_per_inserted: r,
                        floor: cap_floor.expect("clap: requires cap_floor"),
                        ceiling: cap_ceiling.expect("clap: requires cap_ceiling"),
                    },
                    applications: app_cap,
                }),
                (None, None, None) => {
                    assert!(app_cap.is_none(), "--app-cap needs a cap arm");
                    None
                }
                _ => unreachable!("clap: conflicts_with"),
            },
            no_clock,
            no_probe,
            filter: filter.as_deref(),
            skip: &skip,
            font: font.as_deref(),
        }),
        Command::Consistency {
            out,
            class_cap,
            filter,
            font,
            app_cap,
        } => consistency(
            &out,
            &class_cap,
            filter.as_deref(),
            font.as_deref(),
            app_cap,
        ),
        Command::CapSweep {
            rows,
            out_prefix,
            note,
        } => cap_sweep(&rows, &out_prefix, &note),
        Command::Diff {
            off,
            on,
            with_select_hoist,
            cse_only,
            out_prefix,
            note,
        } => diff(
            &DiffInputs {
                off: &off,
                on: &on,
                with_select_hoist: &with_select_hoist,
                cse_only: &cse_only,
            },
            &out_prefix,
            &note,
        ),
    }
}
