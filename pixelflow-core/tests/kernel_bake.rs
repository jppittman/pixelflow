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
    let sdf = x
        .mul(&x)
        .add(&y.mul(&y))
        .sqrt()
        .sub(&Kernel::constant(3.0));

    let lattice = Lattice {
        extent: [8, 8, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    let baked = lattice.bake(&sdf);
    let buf = baked.buffer();
    assert_eq!(buf.len(), 64);

    // Spot-check a few texels against the closed form √(x²+y²) − 3 at centers.
    for &(i, j) in &[(0usize, 0usize), (3, 4), (7, 7)] {
        let (px, py) = (i as f32 + 0.5, j as f32 + 0.5);
        let want = (px * px + py * py).sqrt() - 3.0;
        let got = buf[j * 8 + i];
        assert!((got - want).abs() < 1e-3, "texel ({i},{j}): {got} vs {want}");
    }
}
