use super::*;
use crate::numeric::Numeric;
use crate::variables::{X, Y};

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

// ---- Frame coord generation ----

#[test]
fn frame_coord_generation() {
    let lattice = Lattice {
        extent: [4, 3, 1, 1],
        origin: [0.0, 0.0, 1.5, 2.0],
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

    // Loop axes: X and Y
    assert_eq!(lattice.loop_mask(), 0b0011);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn frame_coord_oob() {
    let lattice = Lattice::frame(4, 3, 0.0);
    let _c = lattice.coord(12);
}

// ---- Scanline coord generation ----

#[test]
fn scanline_coord_generation() {
    let lattice = Lattice::scanline(8, 5.0, 1.0, 0.0);
    assert_eq!(lattice.len(), 8);
    assert_eq!(lattice.coord(0), (0.0, 5.0, 1.0, 0.0));
    assert_eq!(lattice.coord(7), (7.0, 5.0, 1.0, 0.0));
    assert_eq!(lattice.loop_mask(), 0b0001);
}

// ---- Point collapse = single eval ----

#[test]
fn point_collapse_single_eval() {
    let lattice = Lattice::point(3.0, 4.0, 0.0, 0.0);
    assert_eq!(lattice.len(), 1);
    assert_eq!(lattice.loop_mask(), 0);
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
fn point_coord_oob() {
    let lattice = Lattice::point(0.0, 0.0, 0.0, 0.0);
    let _c = lattice.coord(1);
}

// ---- DiscreteManifold round-trip ----

#[test]
fn discrete_manifold_round_trip() {
    // Collapse a simple manifold (XPlusY) over a small grid, then read back.
    let lattice = Lattice::frame(8, 4, 0.0);
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

// ---- collapse_with Add on constant = value * count (per-lane fold) ----

#[test]
fn collapse_with_add_constant() {
    let value = 2.5f32;
    let lattice = Lattice::frame(8, 4, 0.0);
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
    let lattice = Lattice::scanline(PARALLELISM, 0.0, 0.0, 0.0);
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
fn frame_collapse_non_aligned_width() {
    // Width that's not a multiple of PARALLELISM.
    let width = PARALLELISM + 1;
    let lattice = Lattice::frame(width, 2, 0.0);
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

// ---- Constructor shapes ----

#[test]
fn constructor_shapes() {
    let f = Lattice::frame(1920, 1080, 0.5);
    assert_eq!(f.extent, [1920, 1080, 1, 1]);
    assert_eq!(f.origin, [0.0, 0.0, 0.5, 0.0]);

    let i = Lattice::index(132);
    assert_eq!(i.extent, [132, 1, 1, 1]);
    assert_eq!(i.loop_mask(), 0b0001);

    let m = Lattice::index2(64, 32);
    assert_eq!(m.extent, [64, 32, 1, 1]);
    assert_eq!(m.loop_mask(), 0b0011);
}

// ---- Scanline collapse round-trip ----

#[test]
fn scanline_collapse_round_trip() {
    let lattice = Lattice::scanline(16, 3.0, 0.0, 0.0);
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
fn frame_zero_dimensions() {
    let lattice = Lattice::frame(0, 0, 0.0);
    assert!(lattice.is_empty());
    assert_eq!(lattice.len(), 0);

    // Collapsing an empty lattice produces an empty discrete manifold.
    let discrete = lattice.collapse(&Constant(42.0));
    assert_eq!(discrete.buffer().len(), 0);
    assert_eq!(discrete.width(), 0);
    assert_eq!(discrete.height(), 0);
}

// ---- Index-space lattices (feature/tensor indexing) ----

#[test]
fn index_collapse_identity() {
    let lattice = Lattice::index(4);
    let result = lattice.collapse(&X);
    assert_eq!(result.width(), 4);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 0.0).abs() < 1e-6);
    assert!((buf[1] - 1.0).abs() < 1e-6);
    assert!((buf[2] - 2.0).abs() < 1e-6);
    assert!((buf[3] - 3.0).abs() < 1e-6);
}

#[test]
fn index_collapse_scalar_sum() {
    // Sum of [0,1,2,3] across the whole lattice = 6.
    let lattice = Lattice::index(4);
    let result = lattice.collapse_scalar(ReduceOp::Add, &X);
    assert!((result - 6.0).abs() < 1e-5, "expected 6.0, got {}", result);
}

#[test]
fn index2_collapse_xy_sum() {
    // 3x2 lattice (width=3, height=2). Values = X + Y.
    // Row 0 (Y=0): [0,1,2]. Row 1 (Y=1): [1,2,3].
    let lattice = Lattice::index2(3, 2);
    let result = lattice.collapse(&(X + Y));
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
fn index2_collapse_scalar_sum() {
    // 3x2 lattice. Values = X + Y.
    // Sum = (0+0) + (1+0) + (2+0) + (0+1) + (1+1) + (2+1) = 0+1+2+1+2+3 = 9.
    let lattice = Lattice::index2(3, 2);
    let result = lattice.collapse_scalar(ReduceOp::Add, &(X + Y));
    assert!((result - 9.0).abs() < 1e-5, "expected 9.0, got {}", result);
}

#[test]
fn collapse_axis0_dot_product() {
    // W = column-major layout for matmul:
    // W(input_i=X, output_j=Y): W(0,0)=1, W(1,0)=3, W(0,1)=2, W(1,1)=4
    // Row-major buffer (Y outer, X inner): [W(0,0), W(1,0), W(0,1), W(1,1)] = [1, 3, 2, 4]
    let w_buf = alloc::vec![1.0f32, 3.0, 2.0, 4.0];
    let w = DiscreteManifold::new(w_buf, 2, 2);
    let x_buf = alloc::vec![1.0f32, 2.0];
    let x_vec = DiscreteManifold::new(x_buf, 2, 1);

    struct Product {
        w: DiscreteManifold,
        x: DiscreteManifold,
    }
    impl Manifold<(Field, Field, Field, Field)> for Product {
        type Output = Field;
        fn eval(&self, (xi, yj, _, _): (Field, Field, Field, Field)) -> Field {
            let zero = Field::from(0.0);
            let w_val = self.w.eval((xi, yj, zero, zero));
            let x_val = self.x.eval((xi, zero, zero, zero));
            (w_val * x_val).eval((xi, yj, zero, zero))
        }
    }

    let lattice = Lattice::index2(2, 2); // width=INPUT=2, height=OUTPUT=2
    let result = lattice.collapse_axis(0, ReduceOp::Add, &Product { w, x: x_vec });
    // result: width=2 (= extent[1]), height=1
    // result[0] = W(0,0)*x(0) + W(1,0)*x(1) = 1*1 + 3*2 = 7
    // result[1] = W(0,1)*x(0) + W(1,1)*x(1) = 2*1 + 4*2 = 10
    assert_eq!(result.width(), 2);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 7.0).abs() < 1e-4, "expected 7.0, got {}", buf[0]);
    assert!(
        (buf[1] - 10.0).abs() < 1e-4,
        "expected 10.0, got {}",
        buf[1]
    );
}

#[test]
fn collapse_axis1_row_sum() {
    // 2x3 lattice (width=2, height=3). Values = Y.
    // collapse_axis(1, Add): for each X=i, sum Y over [0,3) = 0+1+2 = 3
    // result: width=2, height=1. result[0]=3, result[1]=3.
    let lattice = Lattice::index2(2, 3);
    let result = lattice.collapse_axis(1, ReduceOp::Add, &Y);
    assert_eq!(result.width(), 2);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", buf[0]);
    assert!((buf[1] - 3.0).abs() < 1e-5, "expected 3.0, got {}", buf[1]);
}

// ---- 4D box collapse (Z/W extents > 1) ----

#[test]
fn box_collapse_4d_layout() {
    // 2 wide, 2 tall, 2 deep: buffer rows are (w, z, y) outer-to-inner.
    let lattice = Lattice {
        extent: [2, 2, 2, 1],
        origin: [0.0; 4],
    };
    assert_eq!(lattice.len(), 8);
    assert_eq!(lattice.loop_mask(), 0b0111);

    struct ZTimes100;
    impl Manifold<(Field, Field, Field, Field)> for ZTimes100 {
        type Output = Field;
        fn eval(&self, (x, y, z, _): (Field, Field, Field, Field)) -> Field {
            z.raw_mul(Field::from(100.0))
                .raw_add(y.raw_mul(Field::from(10.0)))
                .raw_add(x)
        }
    }

    let discrete = lattice.collapse(&ZTimes100);
    assert_eq!(discrete.width(), 2);
    assert_eq!(discrete.height(), 4);
    let buf = discrete.buffer();
    // Rows in order: (z=0,y=0), (z=0,y=1), (z=1,y=0), (z=1,y=1)
    let expected = [0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
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
