//! Test that kernel! macro compiles with 5 parameters using WithContext.
//!
//! This is the key test - the old nested Let approach failed with >4 params.

use pixelflow_compiler::kernel;
use pixelflow_core::{Field, Manifold};

type Field4 = (Field, Field, Field, Field);

#[test]
fn test_kernel_5_params_compiles() {
    // THE KEY TEST: This should now compile with 5 params!
    // The old nested Let approach caused trait solver explosion at this point.

    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| { a + b + c + d + e });

    // Construct the kernel - if this compiles, we've proven WithContext works!
    let _kernel_instance = k(1.0, 2.0, 3.0, 4.0, 5.0);

    // Success! The kernel compiles with 5 parameters.
}

#[test]
fn test_kernel_6_params_compiles() {
    // Test 6 parameters
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32, f: f32| { a + b + c + d + e + f });

    let _kernel_instance = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
}

#[test]
fn test_kernel_7_params_compiles() {
    // Test 7 parameters
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32| {
        a + b + c + d + e + f + g
    });

    let _kernel_instance = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
}

#[test]
fn test_kernel_8_params_compiles() {
    // Even more ambitious - 8 parameters!
    // This would definitely fail with nested Let.

    let k = kernel!(
        |a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32| {
            a + b + c + d + e + f + g + h
        }
    );

    let _kernel_instance = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

    // Success! The kernel compiles with 8 parameters.
}

#[test]
fn test_jet_kernel_with_param() {
    use pixelflow_compiler::kernel;
    use pixelflow_core::{Field, Manifold, Y, jet::Jet3};

    type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

    // Test that 1-parameter Jet kernel compiles and evaluates
    let k = kernel!(|h: f32| -> Jet3 { h / Y });
    let f = k(5.0);

    let p: Jet3_4 = (
        Jet3::from(Field::from(1.0)),
        Jet3::from(Field::from(2.0)),
        Jet3::from(Field::from(3.0)),
        Jet3::from(Field::from(4.0)),
    );

    let _result = f.eval(p);
    // If it compiles and evaluates, the test passes
}
