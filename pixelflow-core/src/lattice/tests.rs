use super::*;
use crate::numeric::Numeric;

// A trivial manifold for testing: returns X + Y.
// This is a struct (not a closure) so it can implement Manifold.
#[derive(Copy, Clone)]
struct XPlusY;

impl Manifold<(Field, Field, Field, Field)> for XPlusY {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: (Field, Field, Field, Field)) -> Field {
        let (x, y, _, _) = p;
        x.raw_add(y)
    }
}

// Constant manifold: returns a fixed value regardless of coordinates.
#[derive(Copy, Clone)]
struct Constant(f32);

impl Manifold<(Field, Field, Field, Field)> for Constant {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, _p: (Field, Field, Field, Field)) -> Field {
        Field::from(self.0)
    }
}

// Returns X only (for simple 1D tests).
#[derive(Copy, Clone)]
struct XOnly;

impl Manifold<(Field, Field, Field, Field)> for XOnly {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: (Field, Field, Field, Field)) -> Field {
        p.0
    }
}

// ---- FrameLattice coord generation ----

#[test]
fn frame_lattice_coord_generation() {
    let lattice = FrameLattice {
        width: 4,
        height: 3,
        z: 1.5,
        w: 2.0,
    };

    assert_eq!(lattice.len(), 12);
    assert!(!lattice.is_empty());

    // First pixel: (0, 0, z, w)
    assert_eq!(lattice.coord(0), (0.0, 0.0, 1.5, 2.0));
    // End of first row: (3, 0, z, w)
    assert_eq!(lattice.coord(3), (3.0, 0.0, 1.5, 2.0));
    // Start of second row: (0, 1, z, w)
    assert_eq!(lattice.coord(4), (0.0, 1.0, 1.5, 2.0));
    // Last pixel: (3, 2, z, w)
    assert_eq!(lattice.coord(11), (3.0, 2.0, 1.5, 2.0));

    // Loop vars: X and Y
    assert_eq!(lattice.loop_vars(), &[0, 1]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn frame_lattice_coord_oob() {
    let lattice = FrameLattice::new(4, 3, 0.0);
    let _ = lattice.coord(12);
}

// ---- ScanlineLattice coord generation ----

#[test]
fn scanline_lattice_coord_generation() {
    let lattice = ScanlineLattice::new(8, 5.0, 1.0, 0.0);
    assert_eq!(lattice.len(), 8);
    assert_eq!(lattice.coord(0), (0.0, 5.0, 1.0, 0.0));
    assert_eq!(lattice.coord(7), (7.0, 5.0, 1.0, 0.0));
    assert_eq!(lattice.loop_vars(), &[0]);
}

// ---- PointLattice collapse = single eval ----

#[test]
fn point_lattice_collapse_single_eval() {
    let lattice = PointLattice::new(3.0, 4.0, 0.0, 0.0);
    assert_eq!(lattice.len(), 1);
    assert_eq!(lattice.loop_vars(), &[]);
    assert_eq!(lattice.coord(0), (3.0, 4.0, 0.0, 0.0));

    let discrete = lattice.collapse(&XPlusY);
    assert_eq!(discrete.width(), 1);
    assert_eq!(discrete.height(), 1);

    // X + Y = 3 + 4 = 7
    let buf = discrete.buffer();
    assert_eq!(buf.len(), 1);
    assert!((buf[0] - 7.0).abs() < 1e-5, "expected 7.0, got {}", buf[0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn point_lattice_coord_oob() {
    let lattice = PointLattice::new(0.0, 0.0, 0.0, 0.0);
    let _ = lattice.coord(1);
}

// ---- DiscreteManifold round-trip ----

#[test]
fn discrete_manifold_round_trip() {
    // Collapse a simple manifold (XPlusY) over a small grid, then read back.
    let lattice = FrameLattice::new(8, 4, 0.0);
    let discrete = lattice.collapse(&XPlusY);

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

    // Now eval the DiscreteManifold at known coordinates.
    // Querying at (2.0, 1.0) should return buffer[1*8 + 2] = 3.0.
    let result = discrete.eval((
        Field::from(2.0),
        Field::from(1.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let mut out = [0.0f32; PARALLELISM];
    result.store(&mut out);
    assert!((out[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", out[0],);
}

// ---- collapse_with Add on constant = value * count ----

#[test]
fn collapse_with_add_constant() {
    let value = 2.5f32;
    let lattice = FrameLattice::new(8, 4, 0.0);
    let result = lattice.collapse_with(ReduceOp::Add, &Constant(value));

    // Each SIMD eval adds `value` to each lane. The number of evals
    // per lane = height * (width / PARALLELISM) since width=8 is aligned.
    let evals_per_lane = 4 * (8 / PARALLELISM);
    let expected_per_lane = evals_per_lane as f32 * value;

    let mut out = [0.0f32; PARALLELISM];
    result.store(&mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected_per_lane).abs() < 1e-3,
            "lane {}: expected {}, got {}",
            i,
            expected_per_lane,
            v,
        );
    }
}

// ---- collapse_with on a non-trivial manifold ----

#[test]
fn collapse_with_mul_constant() {
    // Mul identity is 1.0. For a constant manifold returning 2.0,
    // folding N batches: 2.0^N per lane.
    let lattice = ScanlineLattice::new(PARALLELISM, 0.0, 0.0, 0.0);
    let result = lattice.collapse_with(ReduceOp::Mul, &Constant(2.0));

    // width = PARALLELISM, so exactly 1 batch. Result = 1.0 * 2.0 = 2.0 per lane.
    let mut out = [0.0f32; PARALLELISM];
    result.store(&mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "lane {}: expected 2.0, got {}",
            i,
            v,
        );
    }
}

// ---- Tail handling (non-multiple-of-PARALLELISM width) ----

#[test]
fn frame_lattice_collapse_non_aligned_width() {
    // Width that's not a multiple of PARALLELISM.
    let width = PARALLELISM + 1;
    let lattice = FrameLattice::new(width, 2, 0.0);
    let discrete = lattice.collapse(&XOnly);

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

    // Negative coords should clamp to 0.
    let result = dm.eval((
        Field::from(-5.0),
        Field::from(-5.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let mut out = [0.0f32; PARALLELISM];
    result.store(&mut out);
    assert!(
        (out[0] - 10.0).abs() < 1e-5,
        "expected 10.0 (clamped to 0,0), got {}",
        out[0],
    );

    // Coords beyond max should clamp.
    let result = dm.eval((
        Field::from(100.0),
        Field::from(100.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    result.store(&mut out);
    assert!(
        (out[0] - 40.0).abs() < 1e-5,
        "expected 40.0 (clamped to 1,1), got {}",
        out[0],
    );
}

#[test]
#[should_panic(expected = "does not match dimensions")]
fn discrete_manifold_size_mismatch() {
    let _manifold = DiscreteManifold::new(alloc::vec![1.0, 2.0, 3.0], 2, 2);
}

// ---- ReduceOp identity elements ----

#[test]
fn reduce_op_identities() {
    let mut out = [0.0f32; PARALLELISM];

    ReduceOp::Add.identity().store(&mut out);
    assert_eq!(out[0], 0.0);

    ReduceOp::Mul.identity().store(&mut out);
    assert_eq!(out[0], 1.0);

    ReduceOp::Min.identity().store(&mut out);
    assert_eq!(out[0], f32::INFINITY);

    ReduceOp::Max.identity().store(&mut out);
    assert_eq!(out[0], f32::NEG_INFINITY);
}

// ---- FrameLattice::new convenience ----

#[test]
fn frame_lattice_new_convenience() {
    let l = FrameLattice::new(1920, 1080, 0.5);
    assert_eq!(l.width, 1920);
    assert_eq!(l.height, 1080);
    assert_eq!(l.z, 0.5);
    assert_eq!(l.w, 0.0);
}

// ---- Scanline collapse round-trip ----

#[test]
fn scanline_collapse_round_trip() {
    let lattice = ScanlineLattice::new(16, 3.0, 0.0, 0.0);
    let discrete = lattice.collapse(&XPlusY);

    // Each pixel x should have value x + 3.0.
    for x in 0..16 {
        let expected = x as f32 + 3.0;
        let actual = discrete.buffer()[x];
        assert!(
            (actual - expected).abs() < 1e-5,
            "at x={}: expected {}, got {}",
            x,
            expected,
            actual,
        );
    }
}

// ---- Empty lattice ----

#[test]
fn frame_lattice_zero_dimensions() {
    let lattice = FrameLattice::new(0, 0, 0.0);
    assert!(lattice.is_empty());
    assert_eq!(lattice.len(), 0);

    // Collapsing an empty lattice produces an empty discrete manifold.
    let discrete = lattice.collapse(&Constant(42.0));
    assert_eq!(discrete.buffer().len(), 0);
    assert_eq!(discrete.width(), 0);
    assert_eq!(discrete.height(), 0);
}
