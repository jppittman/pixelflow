//! JP-19 (N2/N3, composition L4/L5): what inlining an identical unit costs
//! per copy, across unit sizes — the number a call has to undercut.
//!
//! ```text
//! cargo run --release -p pixelflow-pipeline --example ref_reuse_cost
//! ```
//!
//! A scene composes `K` copies of one **identical** unit (a fold with a
//! `sqrt` in its body, `trips` long) at `K` distinct offsets and takes their
//! union — the case `composition-is-linking` §3 says L4 is for. Each copy is
//! `unit.by_ref().at(..)`, which since JP-110 survives construction as an
//! `Apply`; but `expand` is the only lowering `Apply` has, so every compile
//! entry still inlines it and this measures the inline side of the choice.
//! There is no call lowering to measure against (L5), so the call side is
//! not a column: it is the parameter `call_ns` in the doc's break-even
//! (`composition-is-linking` §9).
//!
//! One TSV row per `(unit, copies)`:
//! `unit copies linked_nodes code_bytes collapse_us`, then one summary row per
//! `unit` of the marginal cost of one more inlined copy:
//! `unit marginal_bytes marginal_ns_per_batch`, where a batch is one
//! `jit_vector_bytes()` of lanes. Compile milliseconds are not reported: at
//! these sizes they are dominated by structure-keyed cache hits.

use pixelflow_codegen::{jit_cache, jit_vector_bytes};
use pixelflow_core::{Kernel, Lattice, Manifold};
use pixelflow_ir::LatticeShape;
use std::time::Instant;

/// Unit sizes, as fold trips in its body (loop units: bytes stay small, time grows).
const TRIPS: [u32; 4] = [2, 8, 32, 128];
/// Straight-line units, as distinct terms (no fold: bytes grow with the size,
/// the shape of a glyph-scale body).
const TERMS: [u32; 2] = [16, 64];
/// Copies measured at each size: one copy, and enough that the slope is the
/// marginal copy rather than the fixed cost.
const COPIES: [u32; 3] = [1, 4, 16];
/// Frame the scene is collapsed over.
const FRAME: usize = 128;
/// Timed collapses per row; the median is reported.
const SAMPLES: usize = 15;
/// Horizontal / vertical spacing of the copies, in pixels.
const PITCH: [f32; 2] = [7.0, 5.0];
/// Distance between the salts that keep every compile cold.
const SALT_STEP: f32 = 1.0e-3;
/// Bytes in an `f32` lane.
const LANE_BYTES: usize = 4;

/// The identical unit: `Σ_i sqrt((X − 0.37·i)² + Y² + 1)`.
fn unit(trips: u32, salt: f32) -> Kernel {
    Kernel::sum_over(trips, |i| {
        let dx = Kernel::x().sub(&i.mul(&Kernel::constant(0.37)));
        let y = Kernel::y();
        dx.mul(&dx)
            .add(&y.mul(&y))
            .add(&Kernel::constant(1.0 + salt))
            .sqrt()
    })
}

/// The straight-line unit: `Σ_j sqrt((X − 0.37·j)² + (Y − 0.11·j)² + 1)`
/// with `j` a literal, so no two terms are equal and none can share.
fn wide_unit(terms: u32, salt: f32) -> Kernel {
    (0..terms)
        .map(|j| {
            let dx = Kernel::x().sub(&Kernel::constant(0.37 * j as f32));
            let dy = Kernel::y().sub(&Kernel::constant(0.11 * j as f32));
            dx.mul(&dx)
                .add(&dy.mul(&dy))
                .add(&Kernel::constant(1.0 + salt))
                .sqrt()
        })
        .reduce(|a, b| a.add(&b))
        .expect("TERMS are nonzero")
}

/// `copies` instances of `unit` at distinct offsets, joined by `min`.
fn scene(unit: &Kernel, copies: u32) -> Kernel {
    let unit = unit.by_ref();
    (0..copies)
        .map(|c| {
            let cx = Kernel::x().sub(&Kernel::constant(PITCH[0] * c as f32));
            let cy = Kernel::y().sub(&Kernel::constant(PITCH[1] * c as f32));
            unit.at(&cx, &cy)
        })
        .reduce(|a, b| a.min(&b))
        .expect("COPIES are nonzero")
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// `(code bytes, median collapse µs)` of one scene.
fn measure(kernel: &Kernel) -> (usize, u128, f64) {
    let extent = [FRAME as u32, FRAME as u32];
    let start = Instant::now();
    let linked = jit_cache::compile(kernel, LatticeShape::new(extent)).expect("the scene compiles");
    let compile_ms = start.elapsed().as_secs_f64() * 1e3;
    let bytes = linked.kernel.as_bytes().len();

    let bound = Manifold::compile(kernel, extent).bind(&[]);
    let lattice = Lattice::frame(FRAME, FRAME);
    std::hint::black_box(lattice.collapse(&bound)); // warm: compiles the stripe shapes
    let times = (0..SAMPLES)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(lattice.collapse(&bound));
            t.elapsed().as_micros()
        })
        .collect();
    (bytes, median(times), compile_ms)
}

fn main() {
    // One-time start-up (ISA detection, rule tables) belongs to no row.
    std::hint::black_box(measure(&scene(&unit(2, -SALT_STEP), 1)));
    let batches = (FRAME * FRAME) / (jit_vector_bytes() / LANE_BYTES);
    println!("unit\tcopies\tlinked_nodes\tcode_bytes\tcollapse_us\tcold_compile_ms");
    let mut marginals = Vec::new();
    // Each (unit, copies) is built from a fresh salt, so every compile is
    // cold: no saturation-cache or JIT-cache hit from a sibling row.
    let mut salt = 0.0_f32;
    let mut fresh = || {
        salt += SALT_STEP;
        salt
    };
    let mut units: Vec<(String, Box<dyn FnMut() -> Kernel>)> = Vec::new();
    for t in TRIPS {
        let mut s = fresh();
        units.push((
            format!("loop{t}"),
            Box::new(move || {
                s += SALT_STEP;
                unit(t, s)
            }),
        ));
    }
    for n in TERMS {
        let mut s = fresh();
        units.push((
            format!("wide{n}"),
            Box::new(move || {
                s += SALT_STEP;
                wide_unit(n, s)
            }),
        ));
    }
    for (trips, mut make) in units {
        let mut ends = Vec::new();
        for copies in COPIES {
            let kernel = scene(&make(), copies);
            let (arena, root) = kernel.linked_parts();
            let nodes = arena.node_count_subtree(root);
            let (bytes, us, ms) = measure(&kernel);
            println!("{trips}\t{copies}\t{nodes}\t{bytes}\t{us}\t{ms:.1}");
            ends.push((copies, bytes, us, ms));
        }
        let (c0, b0, u0, m0) = ends[0];
        let (c1, b1, u1, m1) = ends[ends.len() - 1];
        let extra = f64::from(c1 - c0);
        let bytes = (b1 as f64 - b0 as f64) / extra;
        let ns_per_batch = (u1 as f64 - u0 as f64) * 1e3 / extra / batches as f64;
        marginals.push((trips, bytes, ns_per_batch, (m1 - m0) / extra, m1 / m0));
    }
    println!(
        "\nunit\tmarginal_bytes\tmarginal_ns_per_batch\tmarginal_compile_ms\tcompile_ratio_16_to_1"
    );
    for (trips, bytes, ns, ms, ratio) in marginals {
        println!("{trips}\t{bytes:.0}\t{ns:.1}\t{ms:.1}\t{ratio:.1}");
    }
}
