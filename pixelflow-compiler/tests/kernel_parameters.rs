//! A closure takes as many scalar parameters as the kernel declares. Each is
//! declared as a uniform, in declaration order, with the call's value as its
//! default; the uniform table is 64 bits wide and nothing about it is
//! quadratic. This checks both halves: that the expansion compiles at eight,
//! and that every argument reaches the kernel in declaration order.

use pixelflow_compiler::kernel;
use pixelflow_core::Lattice;

/// A sum would pass with the arguments shuffled, so each one is weighted by a
/// distinct power of ten. Every product and partial sum is an integer below
/// 2²⁴, so the value is exact in any association order.
#[test]
fn a_closure_takes_eight_parameters_in_declaration_order() {
    let k = kernel!(
        |a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32| {
            a * 10000000.0
                + b * 1000000.0
                + c * 100000.0
                + d * 10000.0
                + e * 1000.0
                + f * 100.0
                + g * 10.0
                + h
        }
    );
    let k = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert_eq!(Lattice::eval_at(&k, 0.0, 0.0), 12345678.0);
}
