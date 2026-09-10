use super::*;
use crate::PARALLELISM;
use crate::lattice::manifold::Manifold;
use pixelflow_ir::Kernel;

/// Read one sample of a bound-memory kernel: compile it at a one-sample
/// lattice, bind the buffer it declares, collapse. A test owns its loop; this
/// is the same three steps every consumer takes, at the degenerate shape.
fn sample_bound(kernel: &Kernel, buffer: &DiscreteManifold, x: f32, y: f32) -> f32 {
    let bound = Manifold::compile(kernel, [1, 1]).bind(&[buffer.binding()]);
    bound.eval_at(x, y)
}

// The kernels the tabulation tests bake. Written as `Kernel` arithmetic —
// the arena the compiler consumes — because `Lattice::bake` is the only
// evaluation entry: there is nothing here for a hand-written `Manifold` to
// be handed to.

/// `X + Y`.
fn x_plus_y() -> Kernel {
    Kernel::x().add(&Kernel::y())
}

/// `X`, for the 1D shapes.
fn x_only() -> Kernel {
    Kernel::x()
}

/// `10·Y + X` — each axis readable in one digit of the result, so a
/// transposed or dropped axis in the buffer layout is visible in the value.
fn y_times_10() -> Kernel {
    Kernel::y().mul(&Kernel::constant(10.0)).add(&Kernel::x())
}

// ---- Frame coord generation ----

#[test]
fn frame_coord_generation() {
    let lattice = Lattice { extent: [4, 3] };

    assert_eq!(lattice.len(), 12);
    assert!(!lattice.is_empty());

    // First pixel.
    assert_eq!(lattice.coord(0), (0.0, 0.0));
    // End of first row.
    assert_eq!(lattice.coord(3), (3.0, 0.0));
    // Start of second row.
    assert_eq!(lattice.coord(4), (0.0, 1.0));
    // Last pixel.
    assert_eq!(lattice.coord(11), (3.0, 2.0));

    // Loop axes: X and Y
    assert_eq!(lattice.loop_mask(), 0b0011);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn frame_coord_oob() {
    let lattice = Lattice::frame(4, 3);
    let _c = lattice.coord(12);
}

// ---- Scanline: a row-range collapse ----

#[test]
fn scanline_is_x_only_at_a_fixed_row() {
    // X + Y at row 5: every sample reads its own column plus the fixed row.
    let discrete = Lattice::scanline(&x_plus_y(), 8, 5);
    assert_eq!(discrete.width(), 8);
    assert_eq!(discrete.height(), 1);
    for x in 0..8 {
        let want = x as f32 + 5.0;
        assert!(
            (discrete.buffer()[x] - want).abs() < 1e-5,
            "x={x}: expected {want}, got {}",
            discrete.buffer()[x]
        );
    }
}

#[test]
fn an_empty_scanline_is_a_zero_width_buffer() {
    let discrete = Lattice::scanline(&x_plus_y(), 0, 5);
    assert_eq!(discrete.buffer().len(), 0);
    assert_eq!(discrete.width(), 0);
}

// ---- eval_at = a single value, not a domain ----

#[test]
fn eval_at_is_a_single_value() {
    // X + Y = 3 + 4 = 7
    let got = Lattice::eval_at(&x_plus_y(), 3.0, 4.0);
    assert!((got - 7.0).abs() < 1e-5, "expected 7.0, got {got}");
}

// ---- DiscreteManifold round-trip ----

#[test]
fn discrete_manifold_round_trip() {
    // Bake X + Y over a small grid, then read the buffer back as a manifold.
    let lattice = Lattice::frame(8, 4);
    let discrete = lattice.bake(&x_plus_y());

    assert_eq!(discrete.width(), 8);
    assert_eq!(discrete.height(), 4);
    assert_eq!(discrete.buffer().len(), 32);

    // Check known values: buffer[y * width + x] should equal x + y.
    for y in 0..4 {
        for x in 0..8 {
            let expected = (x + y) as f32;
            let actual = discrete.buffer()[y * 8 + x];
            assert!(
                (actual - expected).abs() < 1e-5,
                "at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual,
            );
        }
    }

    // Now read the DiscreteManifold back by coordinate — the other half of
    // `index(collapse(f)) = f`. Querying at (2.0, 1.0) returns
    // buffer[1*8 + 2] = 3.0.
    let got = sample_bound(&discrete.kernel(), &discrete, 2.0, 1.0);
    assert!((got - 3.0).abs() < 1e-5, "expected 3.0, got {got}");
}

/// **The law, with no side condition.** `index(collapse(f)) = f` at *every*
/// index of a two-axis lattice, for a kernel that reads both axes and is not
/// affine in either.
///
/// It used to need one: a lattice carried an origin, so `collapse(f)(i)` was
/// `f(origin + i)` and the equation only held when `index` knew the origin
/// too. A lattice is an extent, full stop — `Lattice::origin` is gone — so
/// the law is what this asserts directly rather than up to a shift.
///
/// Up to rounding, and not to bits: the two sides are two *compilations* of
/// one expression — a 13x7 frame, and a single-point evaluation whose
/// coordinates fold to constants — so each is extracted against its own set
/// of shared subterms and associates its arithmetic differently. That
/// last-bit freedom is what "Floating point at the edges" reserves, and it
/// is not what this test is for: a real break of the law (an index off by
/// one, a lost origin, a dropped axis) moves a sample by orders of
/// magnitude, not by a few units in the last place.
#[test]
fn index_of_collapse_is_the_kernel_everywhere() {
    /// Units in the last place the two extractions may differ by. Measured
    /// at 3 across this kernel's 91 samples; a real law break is nowhere
    /// near it.
    const LAW_ULPS: i64 = 8;

    // sin(X) · (Y + 2) + X·Y: reads both axes, and no rewrite can turn it
    // into something the buffer could reproduce by accident.
    let k = Kernel::x()
        .sin()
        .mul(&Kernel::y().add(&Kernel::constant(2.0)))
        .add(&Kernel::x().mul(&Kernel::y()));

    let lattice = Lattice::frame(13, 7);
    let collapsed = lattice.bake(&k);
    // `index` is the gather the collapsed buffer composes as a kernel.
    let index = collapsed.kernel();

    for i in 0..lattice.len() {
        let (x, y) = lattice.coord(i);
        // The left-hand side: the buffer read back at the index.
        let indexed = sample_bound(&index, &collapsed, x, y);
        // The right-hand side: the kernel itself, at the same point.
        let direct = Lattice::eval_at(&k, x, y);
        // A sign flip puts the two bit patterns astronomically far apart, so
        // this distance refuses one as loudly as it refuses a wrong value.
        let ulps = (i64::from(indexed.to_bits()) - i64::from(direct.to_bits())).abs();
        assert!(
            ulps <= LAW_ULPS,
            "index(collapse(f)) != f at index {i} = ({x}, {y}): \
             {indexed} vs {direct} ({ulps} ulp)"
        );
    }
}

// ---- Tail handling (non-multiple-of-PARALLELISM width) ----

#[test]
fn frame_bake_non_aligned_width() {
    // Width that's not a multiple of PARALLELISM.
    let width = PARALLELISM + 1;
    let lattice = Lattice::frame(width, 2);
    let discrete = lattice.bake(&x_only());

    assert_eq!(discrete.buffer().len(), width * 2);

    // Each pixel at (x, y) should have value x.
    for y in 0..2 {
        for x in 0..width {
            let expected = x as f32;
            let actual = discrete.buffer()[y * width + x];
            assert!(
                (actual - expected).abs() < 1e-5,
                "at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual,
            );
        }
    }
}

// ---- DiscreteManifold clamp behavior ----

#[test]
fn discrete_manifold_clamp_oob_coords() {
    let buffer = alloc::vec![10.0, 20.0, 30.0, 40.0];
    let dm = DiscreteManifold::new(buffer, 2, 2);
    // Layout: (0,0)=10, (1,0)=20, (0,1)=30, (1,1)=40

    let read = dm.kernel();

    // Negative coords should clamp to 0.
    let got = sample_bound(&read, &dm, -5.0, -5.0);
    assert!(
        (got - 10.0).abs() < 1e-5,
        "expected 10.0 (clamped to 0,0), got {got}"
    );

    // Coords beyond max should clamp.
    let got = sample_bound(&read, &dm, 100.0, 100.0);
    assert!(
        (got - 40.0).abs() < 1e-5,
        "expected 40.0 (clamped to 1,1), got {got}"
    );
}

#[test]
#[should_panic(expected = "does not match dimensions")]
fn discrete_manifold_size_mismatch() {
    let _manifold = DiscreteManifold::new(alloc::vec![1.0, 2.0, 3.0], 2, 2);
}

// ---- Constructor shapes ----

#[test]
fn constructor_shapes() {
    let f = Lattice::frame(1920, 1080);
    assert_eq!(f.extent, [1920, 1080]);

    let i = Lattice::index(132);
    assert_eq!(i.extent, [132, 1]);
    assert_eq!(i.loop_mask(), 0b0001);

    let m = Lattice::index2(64, 32);
    assert_eq!(m.extent, [64, 32]);
    assert_eq!(m.loop_mask(), 0b0011);
}

// ---- Empty lattice ----

#[test]
fn frame_zero_dimensions() {
    let lattice = Lattice::frame(0, 0);
    assert!(lattice.is_empty());
    assert_eq!(lattice.len(), 0);

    // Baking over an empty lattice produces an empty discrete manifold.
    let discrete = lattice.bake(&Kernel::constant(42.0));
    assert_eq!(discrete.buffer().len(), 0);
    assert_eq!(discrete.width(), 0);
    assert_eq!(discrete.height(), 0);
}

/// `bake` binds nothing itself, so a kernel that declares a buffer it does
/// not *carry* data for cannot be baked — and the refusal has to **name the
/// slot it could not fill**, not read a null base pointer and hand back
/// plausible numbers. The rule lives in `Manifold::bind`, which is the only
/// place that can state it once for both callers; this pins that `bake`
/// still reaches it.
///
/// `DiscreteManifold::kernel_for`, not `.kernel()`: the latter seeds the
/// kernel with its own buffer (the data travelling with the value), so
/// `bake` now succeeds on it without a caller binding anything — see
/// [`a_kernel_carrying_its_own_buffer_bakes_with_no_explicit_binding`]. This
/// is the genuinely unbound case: an identity a kernel names but no data was
/// ever seeded under.
#[test]
#[should_panic(expected = "nothing bound to slot")]
fn baking_a_kernel_over_bound_memory_names_the_slot_it_cannot_fill() {
    let id = pixelflow_ir::arena::BufferIdentity::mint();
    let dataless = DiscreteManifold::kernel_for(id, 2, 2);
    let _refused = Lattice::frame(2, 2).bake(&dataless);
}

/// The other half of the pair above: a kernel built from
/// [`DiscreteManifold::kernel`] carries its own buffer, so `Lattice::bake` —
/// which binds nothing itself — still succeeds, reading the data the kernel
/// seeded rather than a caller-supplied binding.
#[test]
fn a_kernel_carrying_its_own_buffer_bakes_with_no_explicit_binding() {
    let texture = DiscreteManifold::new(alloc::vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let baked = Lattice::frame(2, 2).bake(&texture.kernel());
    assert_eq!(baked.buffer(), &[1.0, 2.0, 3.0, 4.0]);
}

// ---- Index-space lattices (feature/tensor indexing) ----

#[test]
fn index_bake_identity() {
    let lattice = Lattice::index(4);
    let result = lattice.bake(&x_only());
    assert_eq!(result.width(), 4);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 0.0).abs() < 1e-6);
    assert!((buf[1] - 1.0).abs() < 1e-6);
    assert!((buf[2] - 2.0).abs() < 1e-6);
    assert!((buf[3] - 3.0).abs() < 1e-6);
}

#[test]
fn sum_over_an_index_domain() {
    // Sum of [0,1,2,3] = 6. The reduction is a binder inside the kernel, so
    // the lattice that tabulates it is a single point.
    let k = Kernel::sum_over(4, |i| i.clone());
    let result = Lattice::eval_at(&k, 0.0, 0.0);
    assert!((result - 6.0).abs() < 1e-5, "expected 6.0, got {result}");
}

#[test]
fn index2_bake_xy_sum() {
    // 3x2 lattice (width=3, height=2). Values = X + Y.
    // Row 0 (Y=0): [0,1,2]. Row 1 (Y=1): [1,2,3].
    let lattice = Lattice::index2(3, 2);
    let result = lattice.bake(&x_plus_y());
    assert_eq!(result.width(), 3);
    assert_eq!(result.height(), 2);
    let buf = result.buffer();
    // Row-major: [y=0,x=0],[y=0,x=1],[y=0,x=2],[y=1,x=0],[y=1,x=1],[y=1,x=2]
    assert!((buf[0] - 0.0).abs() < 1e-6); // x=0, y=0
    assert!((buf[1] - 1.0).abs() < 1e-6); // x=1, y=0
    assert!((buf[2] - 2.0).abs() < 1e-6); // x=2, y=0
    assert!((buf[3] - 1.0).abs() < 1e-6); // x=0, y=1
    assert!((buf[4] - 2.0).abs() < 1e-6); // x=1, y=1
    assert!((buf[5] - 3.0).abs() < 1e-6); // x=2, y=1
}

#[test]
fn nested_sums_over_two_index_domains() {
    // Sum over a 3x2 domain of (i + j) = 0+1+2 + 1+2+3 = 9. Two binders, so
    // the inner index is contracted before the outer fold sees it.
    let k = Kernel::sum_over(2, |j| {
        let j = j.clone();
        Kernel::sum_over(3, move |i| i.add(&j))
    });
    let result = Lattice::eval_at(&k, 0.0, 0.0);
    assert!((result - 9.0).abs() < 1e-5, "expected 9.0, got {result}");
}

// ---- Buffer layout is row-major over the two axes ----

#[test]
fn bake_lays_the_plane_out_row_major() {
    // A lattice is a plane: X innermost, Y outermost, and nothing else.
    let lattice = Lattice::frame(2, 4);
    assert_eq!(lattice.len(), 8);
    assert_eq!(lattice.loop_mask(), 0b0011);

    let discrete = lattice.bake(&y_times_10());
    assert_eq!(discrete.width(), 2);
    assert_eq!(discrete.height(), 4);
    let buf = discrete.buffer();
    let expected = [0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (buf[i] - e).abs() < 1e-5,
            "at {}: expected {}, got {}",
            i,
            e,
            buf[i]
        );
    }
}

// ============================================================================
// BilinearSampler: JIT'd 4-tap read-back semantics
// ============================================================================

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod bilinear_sampler {
    use super::*;
    use crate::BilinearSampler;

    /// One point of the sampler, read the way its production caller composes
    /// it: the blend as a `Kernel`, compiled at a shape and collapsed with
    /// the texture bound. `Lattice::bake` cannot do this, because the
    /// texture is bound memory and `bake` binds none.
    fn sample(s: &BilinearSampler, x: f32, y: f32) -> f32 {
        sample_bound(&s.kernel(), s.texture(), x, y)
    }

    #[test]
    fn constant_field_is_identity() {
        // A 3x3 constant grid: bilinear must return the constant everywhere.
        let s = DiscreteManifold::new(vec![0.75; 9], 3, 3).bilinear();

        for &(x, y) in &[(0.0, 0.0), (0.5, 0.5), (1.25, 0.75), (1.9, 1.1)] {
            let v = sample(&s, x, y);
            assert!(
                (v - 0.75).abs() < 1e-6,
                "constant field not reproduced at ({x}, {y}): got {v}"
            );
        }
    }

    #[test]
    fn linear_gradient_reproduced_exactly() {
        // Buffer holding f(x, y) = x + 2y sampled at integer coords, 4x4.
        let mut buf = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                buf.push(x as f32 + 2.0 * y as f32);
            }
        }
        let s = DiscreteManifold::new(buf, 4, 4).bilinear();

        // Bilinear interpolation of an affine function is exact.
        for &(x, y) in &[(0.5, 0.5), (1.25, 2.75), (0.1, 0.9), (2.5, 1.0)] {
            let expect = x + 2.0 * y;
            let v = sample(&s, x, y);
            assert!(
                (v - expect).abs() < 1e-5,
                "gradient not reproduced at ({x}, {y}): got {v}, want {expect}"
            );
        }
    }

    #[test]
    fn exact_at_grid_points_blended_between() {
        // 2x2 checker: (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=0.
        let s = DiscreteManifold::new(vec![0.0, 1.0, 1.0, 0.0], 2, 2).bilinear();

        // At integer grid points: exact stored values, no smoothing.
        assert!((sample(&s, 0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample(&s, 1.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((sample(&s, 0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((sample(&s, 1.0, 1.0) - 0.0).abs() < 1e-6);

        // Midpoint of an edge: average of its two endpoints.
        assert!((sample(&s, 0.5, 0.0) - 0.5).abs() < 1e-6);
        // Cell center: average of all four corners.
        assert!((sample(&s, 0.5, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn no_nearest_jumps_across_cell_boundaries() {
        // Step buffer: left column 0, right column 1. Nearest-neighbor would
        // jump 0 -> 1 crossing x = 1; bilinear must ramp smoothly.
        let s = DiscreteManifold::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2).bilinear();

        let step = 0.125;
        let mut prev = sample(&s, 0.0, 0.5);
        let mut x = step;
        while x <= 2.0 {
            let v = sample(&s, x, 0.5);
            let jump = (v - prev).abs();
            assert!(
                jump < 0.25,
                "non-smooth jump {jump} at x = {x} (bilinear should ramp, not step)"
            );
            prev = v;
            x += step;
        }
        // And the ramp actually reaches the step's top value.
        assert!((sample(&s, 2.0, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_taps_clamp_to_edge() {
        // Queries outside the grid clamp to the edge texel, matching
        // `DiscreteManifold::kernel`'s gather convention.
        let s = DiscreteManifold::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).bilinear();

        assert!((sample(&s, -5.0, -5.0) - 1.0).abs() < 1e-6);
        assert!((sample(&s, 10.0, -5.0) - 2.0).abs() < 1e-6);
        assert!((sample(&s, -5.0, 10.0) - 3.0).abs() < 1e-6);
        assert!((sample(&s, 10.0, 10.0) - 4.0).abs() < 1e-6);
    }
}

// ============================================================================
// Uniforms: a kernel's arguments, bound per call
// ============================================================================

mod uniforms {
    use super::*;
    use crate::lattice::manifold::{Manifold, PlaneRegion};
    use alloc::sync::Arc;
    use alloc::vec;
    use pixelflow_ir::Uniform;
    use pixelflow_ir::arena::BufferIdentity;

    /// `√((x − cx)² + (y − cy)²) − r` over three handles.
    fn circle() -> (Kernel, [Uniform; 3]) {
        let (cx, cy, r) = (Uniform::new(1.5), Uniform::new(-0.5), Uniform::new(2.0));
        let dx = Kernel::x().sub(&cx.kernel());
        let dy = Kernel::y().sub(&cy.kernel());
        let k = dx.mul(&dx).add(&dy.mul(&dy)).sqrt().sub(&r.kernel());
        (k, [cx, cy, r])
    }

    fn circle_at(x: f32, y: f32, [cx, cy, r]: [f32; 3]) -> f32 {
        ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt() - r
    }

    /// `Lattice::bake` (every argument at its default) and a block holding
    /// those defaults are the same bake, bit for bit.
    #[test]
    fn a_bake_with_defaults_is_bit_for_bit_a_block_of_defaults() {
        let (k, _) = circle();
        let lattice = Lattice::frame(37, 5);
        let baked = lattice.bake(&k);
        let program = Manifold::compile(&k, lattice.extent);
        assert_eq!(program.uniforms().len(), 3);
        let blocked = lattice.collapse(&program.bind(&[]).with_uniforms(&program.block()));
        let (a, b) = (baked.buffer(), blocked.buffer());
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "sample {i}: {x} vs {y}");
        }
    }

    /// Two circles are one compile; the block is what makes them two, and a
    /// set is a write, not a recompile: the JIT agrees with the closed form
    /// across several bindings of one program.
    #[test]
    fn a_block_rebinds_without_recompiling() {
        let (k1, [cx, cy, r]) = circle();
        let (k2, _) = circle();
        let extent = [16, 4];
        let p1 = Manifold::compile(&k1, extent);
        let p2 = Manifold::compile(&k2, extent);
        assert_eq!(
            p1.code_bytes().as_ptr(),
            p2.code_bytes().as_ptr(),
            "two instances of one shape share one compiled region"
        );
        let lattice = Lattice::frame(16, 4);
        let mut block = p1.block();
        for values in [[1.5f32, -0.5, 2.0], [3.0, 1.0, 0.5], [-4.25, 2.5, 7.0]] {
            block.set(cx, values[0]).expect("cx is an argument");
            block.set(cy, values[1]).expect("cy is an argument");
            block.set(r, values[2]).expect("r is an argument");
            assert_eq!(block.get(r), Ok(values[2]));
            let plane = lattice.collapse(&p1.bind(&[]).with_uniforms(&block));
            for (i, got) in plane.buffer().iter().enumerate() {
                let (x, y) = ((i % 16) as f32, (i / 16) as f32);
                let want = circle_at(x, y, values);
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "at ({x},{y}) under {values:?}: {got} vs {want}"
                );
            }
        }
    }

    /// A handle from another composition is a composition mistake, and the
    /// answer is an error — the pixels would be plausible otherwise.
    #[test]
    fn a_handle_that_is_not_an_argument_is_an_error() {
        let (k, [cx, ..]) = circle();
        let program = Manifold::compile(&k, [4, 4]);
        let mut block = program.block();
        let stranger = Uniform::new(0.0);
        assert_eq!(
            block.set(stranger, 1.0),
            Err(crate::UnknownUniform(stranger.identity()))
        );
        assert!(block.get(stranger).is_err());
        assert_eq!(block.set(cx, 1.0), Ok(()));
    }

    /// A block laid out for one program cannot be handed to another.
    #[test]
    #[should_panic(expected = "different program")]
    fn a_block_from_another_program_is_refused() {
        let (k1, _) = circle();
        let (k2, _) = circle();
        let p1 = Manifold::compile(&k1, [4, 4]);
        let p2 = Manifold::compile(&k2, [4, 4]);
        let _refused = p1.bind(&[]).with_uniforms(&p2.block());
    }

    /// A cursor: a column lit where `|x − cx| < ½`, over a background that
    /// reads nothing but the coordinates. Moving the cursor changes the
    /// columns it left and arrived at, and no other pixel.
    #[test]
    fn moving_a_uniform_moves_only_the_pixels_that_read_it() {
        let cx = Uniform::new(2.0);
        let under = Kernel::x()
            .sub(&cx.kernel())
            .abs()
            .lt(&Kernel::constant(0.5));
        let background = Kernel::y().mul(&Kernel::constant(0.25)).add(&Kernel::x());
        let k = under.select(&Kernel::constant(-1.0), &background);
        let (w, h) = (9usize, 3usize);
        let lattice = Lattice::frame(w, h);
        let program = Manifold::compile(&k, lattice.extent);

        let frame_at = |col: f32| {
            let mut block = program.block();
            block.set(cx, col).expect("cx is the argument");
            lattice.collapse(&program.bind(&[]).with_uniforms(&block))
        };
        let (here, there) = (frame_at(2.0), frame_at(5.0));
        for row in 0..h {
            for col in 0..w {
                let (a, b) = (here.buffer()[row * w + col], there.buffer()[row * w + col]);
                let want_a = if col == 2 {
                    -1.0
                } else {
                    col as f32 + 0.25 * row as f32
                };
                let want_b = if col == 5 {
                    -1.0
                } else {
                    col as f32 + 0.25 * row as f32
                };
                assert_eq!(a, want_a, "cursor at 2, ({col},{row})");
                assert_eq!(b, want_b, "cursor at 5, ({col},{row})");
                assert_eq!(a != b, col == 2 || col == 5, "({col},{row}) moved");
            }
        }
    }

    /// The context is filled in the order the code was compiled against —
    /// the link's first-occurrence order — not the arena's declaration
    /// order, and the block pointer follows the buffer slots.
    #[test]
    fn binding_follows_the_link_and_the_block_follows_the_buffers() {
        let (a, b) = (BufferIdentity::mint(), BufferIdentity::mint());
        let read =
            |id| DiscreteManifold::kernel_for(id, 4, 1).at(&Kernel::x(), &Kernel::constant(0.0));
        let scale = Uniform::new(10.0);
        // `b` is declared first (the receiver's arena) and read second.
        let k = read(b).add(&read(a).mul(&scale.kernel()));
        let program = Manifold::compile(&k, [4, 1]);
        assert_eq!(program.buffers()[0].id, b, "slot 0 is the first read");
        assert_eq!(program.buffers()[1].id, a);
        let bound = program.bind(&[(a, Arc::new(vec![1.0; 4])), (b, Arc::new(vec![2.0; 4]))]);
        let mut out = vec![0.0f32; 4];
        bound.collapse_rows(PlaneRegion::rows(4, 0, 1), &mut out, 4);
        assert_eq!(out, [12.0; 4], "b + 10·a at the default");
        let mut block = program.block();
        block.set(scale, 100.0).expect("scale is the argument");
        bound
            .with_uniforms(&block)
            .collapse_rows(PlaneRegion::rows(4, 0, 1), &mut out, 4);
        assert_eq!(out, [102.0; 4], "b + 100·a under the block");
    }
}

mod uniforms_link_and_oracle {
    use super::*;
    use crate::lattice::manifold::Manifold;
    use pixelflow_ir::Uniform;

    /// `Kernel::at` splices every coordinate fragment whether or not the
    /// receiver reads that axis, so a kernel routinely *declares* an
    /// instance it never reads. It must compile, its link must omit the
    /// phantom, and the phantom's handle must be refused as an argument.
    #[test]
    fn a_uniform_spliced_on_an_unread_axis_is_not_an_argument() {
        let (cx, phase) = (Uniform::new(1.0), Uniform::new(0.0));
        // Reads X and cx only; Y is warped by `phase` and never read.
        let circle = Kernel::x().sub(&cx.kernel()).abs();
        let moving = circle.at(&Kernel::x(), &phase.kernel());
        assert_eq!(
            moving.parts().0.uniforms().len(),
            2,
            "the table names the phantom — that is the shape being tested"
        );
        let program = Manifold::compile(&moving, [4, 1]);
        assert_eq!(
            program.uniforms(),
            &[cx.decl()],
            "the link holds only what is read"
        );
        let mut block = program.block();
        assert_eq!(
            block.set(phase, 3.0),
            Err(crate::UnknownUniform(phase.identity()))
        );
        block.set(cx, 2.0).expect("cx is read");
        let plane = Lattice::frame(4, 1).collapse(&program.bind(&[]).with_uniforms(&block));
        assert_eq!(plane.buffer(), &[2.0, 1.0, 0.0, 1.0]);
    }
}
