//! A closure takes as many scalar parameters as the kernel declares.
//!
//! The old combinator tier bound parameters with nested `Let`s and fell over
//! the trait solver past four; the arena backend declares each as a uniform,
//! in declaration order, with the call's value as its default — the uniform
//! table is 64 bits wide, and nothing about it is quadratic. These check
//! both halves: that the expansion compiles at 5..8, and that every argument
//! reaches the kernel in declaration order.

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

/// A kernel at one point — compiled at a one-sample lattice, then collapsed.
fn at_origin(k: &Kernel) -> f32 {
    Lattice::eval_at(k, 0.0, 0.0)
}

#[test]
fn kernel_5_params_compiles() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| { a + b + c + d + e });
    assert_eq!(at_origin(&k(1.0, 2.0, 3.0, 4.0, 5.0)), 15.0);
}

#[test]
fn kernel_6_params_compiles() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32, f: f32| { a + b + c + d + e + f });
    assert_eq!(at_origin(&k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)), 21.0);
}

#[test]
fn kernel_7_params_compiles() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32| {
        a + b + c + d + e + f + g
    });
    assert_eq!(at_origin(&k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)), 28.0);
}

#[test]
fn kernel_8_params_compiles() {
    let k = kernel!(
        |a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32| {
            a + b + c + d + e + f + g + h
        }
    );
    assert_eq!(at_origin(&k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)), 36.0);
}

/// Order, not just arity: a sum would pass with the arguments shuffled, so
/// this weights each one by a distinct power of ten.
#[test]
fn builder_arguments_reach_the_arena_in_declaration_order() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| {
        a * 10000.0 + b * 1000.0 + c * 100.0 + d * 10.0 + e
    });
    assert_eq!(at_origin(&k(1.0, 2.0, 3.0, 4.0, 5.0)), 12345.0);
}
