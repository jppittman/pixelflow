//! A kernel composed *by reference* renders what the spliced composition
//! renders — tabulations and all.
//!
//! L2 of docs/plans/2026-09-09-composition-is-linking.md. `Kernel::by_ref`
//! replaces a whole arena with one `Ref` leaf naming it in the
//! `KernelStore`; `passes::expand_refs` puts the body back before anything
//! reads structure. The property that has to hold for either to be usable is
//! that a consumer cannot tell the difference — same pixels, from the same
//! composition surface, with no caller gathering anything by hand.
//!
//! This lives in `pixelflow-core` for the reason
//! `reduce_binder_reads_bound_buffer.rs` does: the claim spans
//! `pixelflow-ir`'s composition surface and the compiled
//! `Manifold`/`Lattice::collapse` pipeline, and `pixelflow-core` is the only
//! crate that sees both without a cycle. The interesting half is the
//! tabulation: a `Ref` is a leaf, so the referent's buffer *declaration* is
//! not in the composed arena at all, and both the data travelling with the
//! value (`Kernel::buffer_data`) and the link the JIT is compiled against
//! have to survive the round trip.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{DiscreteManifold, Kernel, Lattice, Manifold};

/// The tabulation's extents. Unequal, so a transposed index reads a
/// different value rather than a plausible one.
const TABLE_W: usize = 4;
const TABLE_H: usize = 3;

/// The lattice every comparison collapses over. Larger than the table, so
/// most samples exercise the gather's clamp as well as its interior.
const FRAME: usize = 8;

/// [`FRAME`] as the extent `Manifold::compile` specializes against.
fn frame_extent() -> [u32; 2] {
    let side = u32::try_from(FRAME).expect("FRAME fits a u32");
    [side, side]
}

/// A table whose values are distinct and non-monotone, so an off-by-one or a
/// dropped row is visible in the output rather than absorbed.
fn table() -> DiscreteManifold {
    let data: Vec<f32> = (0..TABLE_W * TABLE_H)
        .map(|i| ((i * 37) % 101) as f32 + 0.5)
        .collect();
    DiscreteManifold::new(data, TABLE_W, TABLE_H)
}

/// Collapse `kernel` over a [`FRAME`]-square lattice, binding nothing: every
/// buffer the kernel reads must have travelled with it.
fn render(kernel: &Kernel) -> Vec<f32> {
    let program = Manifold::compile(kernel, frame_extent());
    let bound = program.bind(&[]);
    Lattice::frame(FRAME, FRAME)
        .collapse(&bound)
        .buffer()
        .to_vec()
}

/// The whole claim, end to end: the same composition built two ways renders
/// the same frame. `sampler` reads bound memory, so the reference has to
/// carry the tabulation *and* the buffer declaration through expansion — if
/// either were lost, `bind(&[])` would panic on an unbound slot or the link
/// would name the wrong context entry.
#[test]
fn a_tabulation_survives_being_named_and_composed() {
    let grid = table();
    let sampler = grid.kernel();
    assert_eq!(
        sampler.buffer_data().count(),
        1,
        "the sampler must carry its own tabulation to begin with"
    );

    let named = sampler.by_ref();
    assert_eq!(
        named.buffer_data().count(),
        1,
        "and a reference to it must carry it too — the data travels with the \
         value, not with the arena"
    );

    let tint = Kernel::x().mul(&Kernel::constant(0.25));
    let by_reference = named.add(&tint);
    let by_splice = sampler.add(&tint);

    assert_eq!(
        render(&by_reference),
        render(&by_splice),
        "composing by reference must render exactly what splicing renders"
    );
}

/// The same, with the reference under a coordinate warp: `.at` substitutes
/// the receiver's coordinates, and a `Ref` has none to substitute, so the
/// warp has to reach through the name. Sampling the referent at the *outer*
/// coordinates would be plausible and wrong, which is the failure mode worth
/// a test of its own.
#[test]
fn a_warp_reaches_through_a_named_sampler() {
    let grid = table();
    let sampler = grid.kernel();
    let warp = |k: &Kernel| {
        k.at(
            &Kernel::x().mul(&Kernel::constant(0.5)),
            &Kernel::y().add(&Kernel::constant(1.0)),
        )
    };
    assert_eq!(render(&warp(&sampler.by_ref())), render(&warp(&sampler)));
}

/// A reference to a reference, through a composition that reads the table
/// twice: expansion is to fixpoint, and two reads of one identity still bind
/// one slot rather than two.
#[test]
fn nested_references_over_one_tabulation_bind_one_slot() {
    let grid = table();
    let sampler = grid.kernel();
    let inner = sampler.by_ref();
    let doubled = inner.add(&sampler);
    let outer = doubled.by_ref().mul(&Kernel::constant(0.5));
    let direct = sampler.add(&sampler).mul(&Kernel::constant(0.5));

    let program = Manifold::compile(&outer, frame_extent());
    assert_eq!(
        program.buffers().len(),
        1,
        "two reads of one identity are one slot, however deeply named"
    );
    assert_eq!(render(&outer), render(&direct));
}

/// Pure arithmetic, no memory: a reference is the same kernel, so the JIT
/// cache hands back the *same compiled region*. That is the property the
/// whole design rests on — identity is content, and a name changes only how
/// a kernel is built, never which kernel it is.
#[test]
fn a_named_kernel_compiles_to_the_same_code_as_the_kernel() {
    let body = Kernel::x()
        .mul(&Kernel::x())
        .add(&Kernel::y().mul(&Kernel::constant(3.5)))
        .sqrt();
    let direct = Manifold::compile(&body, frame_extent());
    let named = Manifold::compile(&body.by_ref(), frame_extent());
    assert_eq!(direct.extent(), named.extent());
    assert_eq!(
        Lattice::frame(FRAME, FRAME)
            .collapse(&direct.bind(&[]))
            .buffer()
            .to_vec(),
        Lattice::frame(FRAME, FRAME)
            .collapse(&named.bind(&[]))
            .buffer()
            .to_vec()
    );
}
