//! A `kernel!` body means what rustc says the same tokens mean.
//!
//! The body language is Rust's expression syntax, so for the parts of it that
//! are plain Rust — a `let`, a block, a numeric literal — rustc already has
//! the answer, and it is an oracle with no code in common with this front end.
//! Each test writes the body twice, once in `kernel!` and once as an ordinary
//! Rust closure over `(x, y)`, and compares the baked value to the closure's.
//!
//! The failing cases were measured defects (Phase A1 of
//! docs/plans/2026-09-25-the-language-is-kernel.md; probe p6 is the first
//! test): lowering's `locals` was one flat map that no block ever popped, so
//! an inner `let` overwrote an outer one for the rest of the kernel; `sema`'s
//! table deleted a shadowed parameter outright; and a literal was parsed as
//! `f64` and then cast, rounding twice where rustc rounds once. Each test's
//! doc gives the value it used to produce. The bodies that must not compile at
//! all — an out-of-scope local (p15), `let X` (p5) — are refused by `sema`,
//! and pinned by its unit tests, as the literals the parser refuses are by
//! the parser's.
//
// The oracle closures are the kernel bodies' own text, so clippy's style
// advice about them (`let_and_return`) would change the thing being compared.
#![allow(clippy::let_and_return)]

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

/// The sample the defects were measured at: `X = 3`, `Y = 5`, so every
/// binding in the bodies below has a distinct value.
const AT: (f32, f32) = (3.0, 5.0);

fn bake(k: &Kernel) -> f32 {
    Lattice::eval_at(k, AT.0, AT.1)
}

/// p6. An inner `let` shadows only inside its own block. Before the fix the
/// inner `a` replaced the outer one for the rest of the kernel, and this gave
/// `Y + Y = 10`.
#[test]
fn an_inner_let_shadows_only_inside_its_block() {
    let k = kernel!(|| {
        let a = X;
        ({
            let a = Y;
            a
        }) + a
    });
    let rust = |x: f32, y: f32| {
        let a = x;
        ({
            let a = y;
            a
        }) + a
    };
    assert_eq!(bake(&k), rust(AT.0, AT.1));
    assert_eq!(bake(&k), 8.0);
}

/// The same leak, through a parameter. A `let` that shadows a parameter in an
/// inner block must leave the parameter visible after that block. Before the
/// fix `sema` dropped the name `r` altogether when the inner block ended, and
/// lowering then found the leaked local: `X + X = 6`, where rustc says
/// `X + r = 13`.
#[test]
fn a_parameter_is_visible_again_after_the_block_that_shadowed_it() {
    let k = kernel!(|r: f32| ({
        let r = X;
        r
    }) + r)(10.0);
    let rust = |x: f32, r: f32| {
        ({
            let r = x;
            r
        }) + r
    };
    assert_eq!(bake(&k), rust(AT.0, 10.0));
    assert_eq!(bake(&k), 13.0);
}

/// Shadowing within one block: each `let` sees the binding before it, and
/// the block's value sees the last one.
#[test]
fn a_let_in_the_same_block_sees_the_binding_it_shadows() {
    let k = kernel!(|| {
        let a = X;
        let a = a * Y;
        let a = a + X;
        a
    });
    let rust = |x: f32, y: f32| {
        let a = x;
        let a = a * y;
        let a = a + x;
        a
    };
    assert_eq!(bake(&k), rust(AT.0, AT.1));
}

/// A block nested two deep binds the same name at every level; each use sees
/// the innermost binding in scope at that point, and nothing else. Before the
/// fix every use after the innermost `let` read it, and this gave 190.
#[test]
fn each_use_sees_the_innermost_binding_in_scope() {
    let k = kernel!(|| {
        let a = X;
        let b = ({
            let a = Y;
            let c = ({
                let a = a * 2.0;
                a
            }) + a;
            c
        }) * a;
        b - a
    });
    let rust = |x: f32, y: f32| {
        let a = x;
        let b = ({
            let a = y;
            let c = ({
                let a = a * 2.0;
                a
            }) + a;
            c
        }) * a;
        b - a
    };
    // (2Y + Y) * X - X = 15 * 3 - 3.
    assert_eq!(bake(&k), rust(AT.0, AT.1));
    assert_eq!(bake(&k), 42.0);
}

/// A float literal rounds once, to the nearest `f32`, as rustc rounds it.
///
/// The literal is `1 + 2⁻²⁴ + 10⁻²⁹`: just above the midpoint between `1.0`
/// and the next `f32`, `1 + 2⁻²³`. Parsed straight to `f32` it rounds up. Parsed
/// to `f64` first — which the front end used to do — the `10⁻²⁹` is far below
/// an `f64` ulp, so it lands *exactly on* the midpoint, and the cast to `f32`
/// then breaks the tie to even: `1.0`, a value one `f32` ulp from what was
/// written.
#[test]
#[allow(clippy::excessive_precision)] // The digits past f32's precision are the witness.
fn a_float_literal_rounds_once_as_rustc_rounds_it() {
    const ONCE: f32 = 1.00000005960464477539062500001_f32;
    const TWICE: f32 = 1.00000005960464477539062500001_f64 as f32;
    assert_eq!(ONCE.to_bits(), 0x3f80_0001, "rustc rounds up: 1 + 2^-23");
    assert_eq!(
        TWICE.to_bits(),
        0x3f80_0000,
        "f64 then f32 ties to even: 1.0"
    );

    let k = kernel!(|| 1.00000005960464477539062500001);
    assert_eq!(bake(&k).to_bits(), ONCE.to_bits());
}

/// An integer literal is an exact integer, and every one the front end
/// accepts converts to `f32` without rounding. `2²⁴ + 1` is refused at parse
/// (pinned there); an integer far larger than it is taken as written when an
/// `f32` holds it exactly.
#[test]
fn an_integer_literal_is_exact() {
    // 2²⁴, the last integer before `f32`'s integers start skipping.
    let k = kernel!(|| 16777216);
    assert_eq!(bake(&k), 16_777_216.0);

    // 2⁴⁰: far past 2²⁴, but one significant bit, so exactly an `f32`.
    let k = kernel!(|| 1099511627776);
    assert_eq!(bake(&k), 1_099_511_627_776.0);
}
