//! JP-19 (N2/N3, composition L4/L5): the "inline everything" curve that a
//! preserved reference or a call would have to beat.
//!
//! ```text
//! cargo run --release -p pixelflow-pipeline --example ref_reuse_cost
//! ```
//!
//! A scene composes `K` copies of one **identical** unit (a fold with a
//! `sqrt` in its body) at `K` distinct offsets and takes their union — the
//! case `composition-is-linking` §3 says L4 is for. Today every copy is
//! inlined: `Kernel::at` expands a reference at construction (§8), so the
//! offset copy cannot stay a name until an `Apply` node exists. This prints
//! what that costs, per `K`, as one TSV row:
//!
//! `copies  linked_nodes  compile_ms  code_bytes  collapse_us`
//!
//! `code_bytes / copies` is the marginal bytes one more inlined copy costs;
//! a call is only worth its ABI where that marginal cost times the copies it
//! removes exceeds the call site's own bytes and the per-call spill.
//! `linked_nodes` is the arena the e-graph's units are linked into, so a
//! sub-linear column would mean sharing already happens; a linear one means
//! it does not and a surviving `Ref` has something to save.

use pixelflow_codegen::jit_cache;
use pixelflow_core::{Kernel, Lattice, Manifold};
use pixelflow_ir::LatticeShape;
use std::time::Instant;

/// Fold trips in the unit's body.
const UNIT_TRIPS: u32 = 8;
/// Copies measured. Powers of two, so a linear column is easy to read.
const COPIES: [u32; 5] = [1, 2, 4, 8, 16];
/// Frame the scene is collapsed over.
const FRAME: usize = 128;
/// Timed collapses per row; the median is reported.
const SAMPLES: usize = 15;
/// Horizontal / vertical spacing of the copies, in pixels.
const PITCH: [f32; 2] = [7.0, 5.0];

/// The identical unit: `Σ_i sqrt((X − 0.37·i)² + Y² + 1)`.
fn unit() -> Kernel {
    Kernel::sum_over(UNIT_TRIPS, |i| {
        let dx = Kernel::x().sub(&i.mul(&Kernel::constant(0.37)));
        let y = Kernel::y();
        dx.mul(&dx)
            .add(&y.mul(&y))
            .add(&Kernel::constant(1.0))
            .sqrt()
    })
}

/// `copies` instances of [`unit`] at distinct offsets, joined by `min`.
fn scene(copies: u32) -> Kernel {
    let unit = unit().by_ref();
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

fn main() {
    println!("copies\tlinked_nodes\tcompile_ms\tcode_bytes\tcollapse_us");
    let extent = [FRAME as u32, FRAME as u32];
    for copies in COPIES {
        let kernel = scene(copies);
        let (arena, root) = kernel.linked_parts();
        let linked_nodes = arena.node_count_subtree(root);

        let start = Instant::now();
        let linked =
            jit_cache::compile(&kernel, LatticeShape::new(extent)).expect("the scene compiles");
        let compile_ms = start.elapsed().as_secs_f64() * 1e3;
        let code_bytes = linked.kernel.as_bytes().len();

        let bound = Manifold::compile(&kernel, extent).bind(&[]);
        let lattice = Lattice::frame(FRAME, FRAME);
        std::hint::black_box(lattice.collapse(&bound)); // warm: compiles the stripe shapes
        let times = (0..SAMPLES)
            .map(|_| {
                let t = Instant::now();
                let out = lattice.collapse(&bound);
                std::hint::black_box(&out);
                t.elapsed().as_micros()
            })
            .collect();
        println!(
            "{copies}\t{linked_nodes}\t{compile_ms:.1}\t{code_bytes}\t{}",
            median(times)
        );
    }
}
