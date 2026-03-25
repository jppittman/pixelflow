//! Debug: test if sphere_at returns valid t values
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_compiler::kernel;

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

fn sphere_at(cx: f32, cy: f32, cz: f32, r: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    const EPSILON_SQ: f32 = 0.0001;
    kernel!(|cx: f32, cy: f32, cz: f32, r: f32, eps: f32| -> Jet3 {
        let d_dot_c = X * cx + Y * cy + Z * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = r * r;
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);
        let safe_discriminant = discriminant + eps;
        d_dot_c - safe_discriminant.sqrt()
    })(cx, cy, cz, r, EPSILON_SQ)
}

// Simple test: just return a parameter
fn simple_return_param(val: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    kernel!(|val: f32| -> Jet3 {
        val
    })(val)
}

// Test: X + param
fn x_plus_param(val: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    kernel!(|val: f32| -> Jet3 {
        X + val
    })(val)
}

#[test]
fn test_simple_param() {
    let k = simple_return_param(42.0);
    let rx = Jet3::constant(Field::from(1.0));
    let ry = Jet3::constant(Field::from(2.0));
    let rz = Jet3::constant(Field::from(3.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // Result should be 42.0 - test using range check since Field doesn't have direct subtraction
    let low = Field::from(41.9);
    let high = Field::from(42.1);
    let in_range = result.val.gt(low) & result.val.lt(high);
    assert!(in_range.all(), "simple_return_param(42.0) should return ~42.0");
}

#[test]
fn test_x_plus_param() {
    let k = x_plus_param(10.0);
    // X = 5.0, so result should be 15.0
    let rx = Jet3::constant(Field::from(5.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(0.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // Result should be 15.0
    let low = Field::from(14.9);
    let high = Field::from(15.1);
    let in_range = result.val.gt(low) & result.val.lt(high);
    assert!(in_range.all(), "x_plus_param(10) with X=5 should return ~15.0");
}

// Test d_dot_c = X * cx + Y * cy + Z * cz
// For ray (0, 0, 1) and center (0, 0, 4): d_dot_c = 0*0 + 0*0 + 1*4 = 4
fn test_d_dot_c(cx: f32, cy: f32, cz: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    kernel!(|cx: f32, cy: f32, cz: f32| -> Jet3 {
        X * cx + Y * cy + Z * cz
    })(cx, cy, cz)
}

// Test c_sq = cx*cx + cy*cy + cz*cz (should be 16 for center at (0,0,4))
// Returns Field since this only computes with scalar params
fn test_c_sq(cx: f32, cy: f32, cz: f32) -> impl Manifold<Jet3_4, Output = Field> + Clone {
    kernel!(|cx: f32, cy: f32, cz: f32| -> Field {
        cx * cx + cy * cy + cz * cz
    })(cx, cy, cz)
}

// Test: just r*r (should be 1 for r=1)
// Returns Field since this only computes with scalar params
fn test_r_sq(r: f32) -> impl Manifold<Jet3_4, Output = Field> + Clone {
    kernel!(|r: f32| -> Field {
        r * r
    })(r)
}

// Test: c_sq - r_sq (should be 15 for c_sq=16, r_sq=1)
// Returns Field since this only computes with scalar params
fn test_c_sq_minus_r_sq(cx: f32, cy: f32, cz: f32, r: f32) -> impl Manifold<Jet3_4, Output = Field> + Clone {
    kernel!(|cx: f32, cy: f32, cz: f32, r: f32| -> Field {
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = r * r;
        c_sq - r_sq
    })(cx, cy, cz, r)
}

// Test: d_dot_c * d_dot_c (should be 16)
fn test_d_dot_c_sq(cx: f32, cy: f32, cz: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    kernel!(|cx: f32, cy: f32, cz: f32| -> Jet3 {
        let d_dot_c = X * cx + Y * cy + Z * cz;
        d_dot_c * d_dot_c
    })(cx, cy, cz)
}

// Test discriminant = d_dot_c² - (c_sq - r_sq)
// = 4*4 - (16 - 1) = 16 - 15 = 1
fn test_discriminant(cx: f32, cy: f32, cz: f32, r: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    kernel!(|cx: f32, cy: f32, cz: f32, r: f32| -> Jet3 {
        let d_dot_c = X * cx + Y * cy + Z * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = r * r;
        d_dot_c * d_dot_c - (c_sq - r_sq)
    })(cx, cy, cz, r)
}

#[test]
fn test_step1_d_dot_c() {
    let k = test_d_dot_c(0.0, 0.0, 4.0);
    // Ray direction: (0, 0, 1)
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // d_dot_c should be 4.0
    let low = Field::from(3.9);
    let high = Field::from(4.1);
    let in_range = result.val.gt(low) & result.val.lt(high);
    assert!(in_range.all(), "d_dot_c should be ~4.0");
}

#[test]
fn test_step2_c_sq() {
    let k = test_c_sq(0.0, 0.0, 4.0);
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // c_sq should be 16.0 (Field output)
    let low = Field::from(15.9);
    let high = Field::from(16.1);
    let in_range = result.gt(low) & result.lt(high);
    assert!(in_range.all(), "c_sq should be ~16.0");
}

#[test]
fn test_step3a_r_sq() {
    let k = test_r_sq(1.0);
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // r_sq should be 1.0 (Field output)
    let low = Field::from(0.9);
    let high = Field::from(1.1);
    let in_range = result.gt(low) & result.lt(high);
    assert!(in_range.all(), "r_sq should be ~1.0");
}

#[test]
fn test_step3b_c_sq_minus_r_sq() {
    let k = test_c_sq_minus_r_sq(0.0, 0.0, 4.0, 1.0);
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // c_sq - r_sq = 16 - 1 = 15 (Field output)
    let low = Field::from(14.9);
    let high = Field::from(15.1);
    let in_range = result.gt(low) & result.lt(high);
    assert!(in_range.all(), "c_sq - r_sq should be ~15.0");
}

#[test]
fn test_step3c_d_dot_c_sq() {
    let k = test_d_dot_c_sq(0.0, 0.0, 4.0);
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // d_dot_c_sq = 4*4 = 16
    let low = Field::from(15.9);
    let high = Field::from(16.1);
    let in_range = result.val.gt(low) & result.val.lt(high);
    assert!(in_range.all(), "d_dot_c_sq should be ~16.0");
}

#[test]
fn test_step3_discriminant() {
    let k = test_discriminant(0.0, 0.0, 4.0, 1.0);
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let result = k.eval((rx, ry, rz, rw));

    // discriminant should be 1.0
    let low = Field::from(0.9);
    let high = Field::from(1.1);
    let in_range = result.val.gt(low) & result.val.lt(high);
    assert!(in_range.all(), "discriminant should be ~1.0");
}

#[test]
fn test_sphere_hit() {
    // Sphere at (0, 0, 4) with radius 1
    let sphere = sphere_at(0.0, 0.0, 4.0, 1.0);

    // Ray pointing straight at sphere (0, 0, 1) - normalized
    let rx = Jet3::x(Field::from(0.0));
    let ry = Jet3::y(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(0.0));

    let t = sphere.eval((rx, ry, rz, rw));

    // Use Field comparison operators to test validity
    // Expected: t should be ~3.0 (distance to sphere surface at z=3)
    let zero = Field::from(0.0);
    let two = Field::from(2.0);
    let four = Field::from(4.0);
    let hundred = Field::from(100.0);

    // t > 0 (positive hit)
    let is_positive = t.val.gt(zero);
    assert!(is_positive.all(), "t should be positive");

    // t < 100 (reasonable distance)
    let is_reasonable = t.val.lt(hundred);
    assert!(is_reasonable.all(), "t should be < 100");

    // t > 2 and t < 4 (should be ~3.0)
    let in_range = t.val.gt(two) & t.val.lt(four);
    assert!(in_range.all(), "t should be between 2 and 4 (expected ~3.0)");
}
