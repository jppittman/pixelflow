//! The JIT-first bake: a `Kernel` value composed through the language surface
//! (no combinator types, no `Lower`, no arena) tabulates over a lattice via
//! `Lattice::bake` — JIT-compiled once by our own codegen.

use pixelflow_core::Lattice;
use pixelflow_ir::Kernel;

#[test]
fn kernel_bakes_over_lattice() {
    // Circle SDF built entirely as Kernel composition.
    let x = Kernel::x();
    let y = Kernel::y();
    let sdf = x.mul(&x).add(&y.mul(&y)).sqrt().sub(&Kernel::constant(3.0));

    let lattice = Lattice::frame(8, 8);
    let baked = lattice.bake(&sdf.at(
        &Kernel::x().add(&Kernel::constant(0.5)),
        &Kernel::y().add(&Kernel::constant(0.5)),
    ));
    let buf = baked.buffer();
    assert_eq!(buf.len(), 64);

    // Spot-check a few texels against the closed form √(x²+y²) − 3 at centers.
    for &(i, j) in &[(0usize, 0usize), (3, 4), (7, 7)] {
        let (px, py) = (i as f32 + 0.5, j as f32 + 0.5);
        let want = (px * px + py * py).sqrt() - 3.0;
        let got = buf[j * 8 + i];
        assert!(
            (got - want).abs() < 1e-3,
            "texel ({i},{j}): {got} vs {want}"
        );
    }
}

/// The circle SDF from `kernel_bakes_over_lattice`, as a `Kernel`.
fn circle_sdf() -> Kernel {
    let x = Kernel::x();
    let y = Kernel::y();
    x.mul(&x).add(&y.mul(&y)).sqrt().sub(&Kernel::constant(3.0))
}

fn closed_form(px: f32, py: f32) -> f32 {
    (px * px + py * py).sqrt() - 3.0
}

#[test]
fn kernel_bakes_at_every_lattice_shape() {
    let sdf = circle_sdf();

    // A row at a fixed, non-integer Y: X loops via `Lattice::frame`, Y is a
    // contramapped constant. `Lattice::scanline` is the integer-row case
    // (an index range); an arbitrary fixed Y is a contramap instead, the
    // same move `eval_at` makes for a point.
    let fixed_row = sdf.at(&Kernel::x(), &Kernel::constant(2.5));
    let row = Lattice::frame(37, 1).bake(&fixed_row);
    let buf = row.buffer();
    assert_eq!(buf.len(), 37);
    for (i, &got) in buf.iter().enumerate() {
        let want = closed_form(i as f32, 2.5);
        assert!((got - want).abs() < 1e-3, "scanline x={i}: {got} vs {want}");
    }

    // An integer row via `Lattice::scanline`, at row 2.
    let scanned = Lattice::scanline(&sdf, 37, 2);
    assert_eq!(scanned.buffer().len(), 37);
    for (i, &got) in scanned.buffer().iter().enumerate() {
        let want = closed_form(i as f32, 2.0);
        assert!((got - want).abs() < 1e-3, "scanline x={i}: {got} vs {want}");
    }

    // A point: nothing loops.
    let point = Lattice::eval_at(&sdf, 1.5, 4.5);
    let want = closed_form(1.5, 4.5);
    assert!((point - want).abs() < 1e-3);

    // A frame with a scalar tail on every row.
    let frame = Lattice::frame(19, 5).bake(&sdf);
    assert_eq!(frame.buffer().len(), 95);
    for j in 0..5 {
        for i in 0..19 {
            let want = closed_form(i as f32, j as f32);
            let got = frame.buffer()[j * 19 + i];
            assert!(
                (got - want).abs() < 1e-3,
                "frame ({i},{j}): {got} vs {want}"
            );
        }
    }
}

#[test]
fn each_lattice_extent_is_its_own_kernel() {
    use pixelflow_codegen::jit_cache;
    use pixelflow_ir::LatticeShape;

    // The kernel is compiled for its lattice's extents: the same extents
    // share one cache entry, one more column is a different lattice and a
    // different kernel. Resizing is recompilation, by decision.
    let sdf = circle_sdf();
    let shape_of = |l: Lattice| LatticeShape::new(l.extent);
    let compile = |l: Lattice| {
        jit_cache::compile(&sdf, shape_of(l))
            .expect("compile")
            .kernel
    };
    let first = compile(Lattice::frame(8, 8));
    let again = compile(Lattice::frame(8, 8));
    let wider = compile(Lattice::frame(9, 8));
    assert!(std::sync::Arc::ptr_eq(&first, &again));
    assert!(!std::sync::Arc::ptr_eq(&first, &wider));
    assert_eq!(first.shape().extent(), [8, 8]);
    assert_eq!(wider.shape().extent(), [9, 8]);
}
