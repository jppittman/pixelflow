use pixelflow_compiler::kernel;
use pixelflow_core::{Field, Manifold, ManifoldExt};

type Field4 = (Field, Field, Field, Field);

fn field4(x: f32, y: f32, z: f32, w: f32) -> Field4 {
    (
        Field::from(x),
        Field::from(y),
        Field::from(z),
        Field::from(w),
    )
}

fn main() {
    let p = field4(1.0, 2.0, 3.0, 4.0);

    // 20 different 4-param kernels (no division with Var)
    let k1 = kernel!(|a: f32, b: f32, c: f32, d: f32| X + a + Y * b);
    let k2 = kernel!(|a: f32, b: f32, c: f32, d: f32| X - a + Y * b);
    let k3 = kernel!(|a: f32, b: f32, c: f32, d: f32| X * a + Y * b);
    let k4 = kernel!(|a: f32, b: f32, c: f32, d: f32| X + a - Y * b);
    let k5 = kernel!(|a: f32, b: f32, c: f32, d: f32| (X + a).sqrt() + Y * b);
    let k6 = kernel!(|a: f32, b: f32, c: f32, d: f32| (X + a).abs() + Y * b);
    let k7 = kernel!(|a: f32, b: f32, c: f32, d: f32| X.max(a) + Y * b);
    let k8 = kernel!(|a: f32, b: f32, c: f32, d: f32| X.min(a) + Y * b);
    let k9 = kernel!(|a: f32, b: f32, c: f32, d: f32| (X + a + c) + Y * b);
    let k10 = kernel!(|a: f32, b: f32, c: f32, d: f32| (X - a - c) + Y * b);
    let k11 = kernel!(|a: f32, b: f32, c: f32, d: f32| X + a + Y * b + Z * c);
    let k12 = kernel!(|a: f32, b: f32, c: f32, d: f32| X - a + Y * b + Z * c);
    let k13 = kernel!(|a: f32, b: f32, c: f32, d: f32| X * a + Y * b + Z * c);
    let k14 = kernel!(|a: f32, b: f32, c: f32, d: f32| X + a - Y * b + Z * c);
    let k15 = kernel!(|a: f32, b: f32, c: f32, d: f32| X + a + Y + b + Z + c + W + d);
    let k16 = kernel!(|a: f32, b: f32, c: f32, d: f32| X - a - Y - b - Z - c - W - d);
    let k17 = kernel!(|a: f32, b: f32, c: f32, d: f32| X * a * Y * b);
    let k18 = kernel!(|a: f32, b: f32, c: f32, d: f32| (X * X + Y * Y).sqrt() - a);
    let k19 = kernel!(|a: f32, b: f32, c: f32, d: f32| (X * X + Y * Y + Z * Z).sqrt() - a);
    let k20 = kernel!(|a: f32, b: f32, c: f32, d: f32| {
        let dx = X - a;
        let dy = Y - b;
        (dx * dx + dy * dy).sqrt()
    });

    // Use them all to prevent DCE
    let v = 1.0f32;
    let _ = k1(v, v, v, v).eval(p);
    let _ = k2(v, v, v, v).eval(p);
    let _ = k3(v, v, v, v).eval(p);
    let _ = k4(v, v, v, v).eval(p);
    let _ = k5(v, v, v, v).eval(p);
    let _ = k6(v, v, v, v).eval(p);
    let _ = k7(v, v, v, v).eval(p);
    let _ = k8(v, v, v, v).eval(p);
    let _ = k9(v, v, v, v).eval(p);
    let _ = k10(v, v, v, v).eval(p);
    let _ = k11(v, v, v, v).eval(p);
    let _ = k12(v, v, v, v).eval(p);
    let _ = k13(v, v, v, v).eval(p);
    let _ = k14(v, v, v, v).eval(p);
    let _ = k15(v, v, v, v).eval(p);
    let _ = k16(v, v, v, v).eval(p);
    let _ = k17(v, v, v, v).eval(p);
    let _ = k18(v, v, v, v).eval(p);
    let _ = k19(v, v, v, v).eval(p);
    let _ = k20(v, v, v, v).eval(p);
}
